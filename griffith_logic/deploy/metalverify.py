"""Metal(Apple GPU)ランタイムの等価性ガード。

`ir/simulate.py`(numpyのgolden model)を基準に、生成したMetal(.mm)を実際に
clang++でビルド・実行して**整数厳密**に突き合わせる。`deploy/verify.py`
(C版)と同じ設計・同じ理由(教訓=ゴールデンを検証対象自身から
作らない)なので、そちらのdocstringも参照。

Metal特有の事情:
  - `md deploy codegen-metal` はヘッダを生成しない(モデル定義は1本の
    `.mm` に自己完結。関数シグネチャは runtime/metal/mdl_metal.h の
    `mdl_metal_infer` / `mdl_metal_infer_image` と同じ形)ので、
    ドライバ側で `extern "C"` 宣言だけ書けば足りる
    (tests/deploy/test_metal.py の DRIVER と同型)。
  - **画像入力のモデルでは `mdl_metal_infer_image`(GPU前処理カーネル込み)
    を検証対象にする。** 既存の tests/deploy/test_metal.py は
    `mdl_metal_infer`(前処理済みビット列を直接注入)しか通っておらず、
    GPU側の前処理カーネル(`mdl_k_preproc`)自体は未検証だった
    (気づいた欠落。deploy/verify.py の C版verifyは元から
    image_entryを使い分けている)。
"""

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..ir.serialize import ir_from_dict
from ..ir.simulate import simulate
from .metalbackend import generate_metal
from .preproc_desc import SUPPORTED_KINDS, build_preproc_descriptors, encode_bits

RUNTIME_METAL = Path(__file__).resolve().parents[2] / 'runtime' / 'metal'

_DRIVER = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
extern "C" {
int mdl_metal_infer(const unsigned char*, int*);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}
int main(void) {
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  unsigned char* buf = (unsigned char*)malloc(nin);
  int* scores = (int*)malloc(nout * sizeof(int));
  if (fread(buf, 1, nin, stdin) != (size_t)nin) return 1;
  if (mdl_model_init() != 0) return 3;
  if (mdl_metal_infer(buf, scores) != 0) return 2;
  fwrite(scores, sizeof(int), nout, stdout);
  return 0;
}
'''

_IMAGE_DRIVER_TMPL = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
extern "C" {{
int mdl_metal_infer_image(const unsigned char*, int*);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}}
int main(void) {{
  static const int n_img = {n_img};
  int nout = mdl_metal_output_width();
  unsigned char* image = (unsigned char*)malloc((size_t)n_img);
  int* scores = (int*)malloc((size_t)nout * sizeof(int));
  if (fread(image, 1, (size_t)n_img, stdin) != (size_t)n_img) {{
    fprintf(stderr, "image short read\\n");
    return 1;
  }}
  if (mdl_model_init() != 0) return 3;
  if (mdl_metal_infer_image(image, scores) != 0) return 2;
  fwrite(scores, sizeof(int), (size_t)nout, stdout);
  return 0;
}}
'''


def find_clangxx() -> str | None:
    return shutil.which('clang++')


def has_metal() -> bool:
    """macOS + Metal.framework + clang++ が揃っている環境か。"""
    if platform.system() != 'Darwin':
        return False
    return find_clangxx() is not None and RUNTIME_METAL.is_dir()


def build_metal_model(bundle: dict, work_dir, *, clangxx: str | None = None,
                      image_entry: bool = False,
                      name: str = 'model_metal',
                      conv_engine: str = 'split',
                      conv_codegen: bool = False,
                      dense_fuse: bool = False) -> Path:
    """バンドルからMetal(.mm)を生成してビルドし、実行ファイルのパスを返す。

    Args:
        conv_engine: 'split'(既定)または'combined'(旧経路)。
            新旧の出力一致テスト(tests/deploy/test_metal_conv_split.py)
            以外では既定のままでよい
        conv_codegen: 既定False。Trueでモデル特化MSL生成を検証する。
        dense_fuse: 既定False。Trueでdense多層融合カーネル生成
            (`--dense-fuse`)を検証する。
    """
    compiler = clangxx or find_clangxx()
    if compiler is None:
        raise RuntimeError('clang++ が見つからない')
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for filename, text in generate_metal(
            bundle, name=name, conv_engine=conv_engine,
            conv_codegen=conv_codegen, dense_fuse=dense_fuse).items():
        (work / filename).write_text(text, encoding='utf-8')

    if image_entry:
        preproc = bundle.get('preproc') or {}
        plan = build_preproc_descriptors(preproc)
        n_img = plan['image_channels'] * plan['height'] * plan['width']
        driver = _IMAGE_DRIVER_TMPL.format(n_img=n_img)
    else:
        driver = _DRIVER
    (work / 'driver.mm').write_text(driver, encoding='utf-8')

    exe = work / 'app'
    build = subprocess.run(
        [compiler, '-std=c++17', '-O2', '-fobjc-arc',
         f'-I{RUNTIME_METAL}', '-framework', 'Metal', '-framework', 'Foundation',
         str(work / f'{name}.mm'), str(RUNTIME_METAL / 'mdl_metal.mm'),
         str(work / 'driver.mm'), '-o', str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f'生成Metalのビルドに失敗:\n{build.stderr}')
    return exe


def run_metal_model(exe: Path, payload: bytes, n_out: int) -> np.ndarray:
    proc = subprocess.run([str(exe)], input=payload, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f'Metal実行が失敗 (exit={proc.returncode}): '
            f'{proc.stderr.decode(errors="replace")}')
    scores = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
    if scores.size != n_out:
        raise RuntimeError(
            f'Metal出力の要素数が {scores.size} で期待 {n_out} と違う')
    return scores


def verify_metal_matches_simulate(bundle: dict, *, n_samples: int = 3,
                                  seed: int = 0, clangxx: str | None = None,
                                  work_dir=None, verbose: bool = True,
                                  conv_engine: str = 'split',
                                  conv_codegen: bool = False,
                                  dense_fuse: bool = False,
                                  dense_flat: str | None = None,
                                  dense_flat_min_reduction: float | None = None
                                  ) -> dict:
    """Metal実行 ⇔ ir/simulate を整数厳密に突き合わせる。

    画像前処理を持つバンドルでは `mdl_metal_infer_image`(GPU前処理カーネル
    込み)を検証対象にする(deploy/verify.pyのC版と同じ判定基準)。

    Args:
        conv_engine: 'split'(既定)または'combined'(旧経路)。
        conv_codegen: 既定False。Trueでモデル特化MSL生成を検証する。
        dense_fuse: 既定False。Trueでdense多層融合カーネル生成
            (`--dense-fuse`)を検証する。

        dense_flat: ``'auto'|'on'|'off'`` を渡すと `codegen-metal` と同じ
            判定(`resolve_dense_flat(target='gpu')`)で、生成に使うバンドルだけを
            メモリ上で平坦化する。ゴールデンは常に元バンドルIRの simulate。
            None(既定)はバンドルをそのまま生成する。

    Returns:
        {'n_samples', 'n_out', 'path': 'image'|'bits', 'max_abs_diff': 0,
         'dense_flat': decision|None}

    Raises:
        ValueError: 1サンプルでも不一致があった場合
        RuntimeError: ビルド・実行に失敗した場合(GPU無し等はここで例外)
    """
    graph = ir_from_dict(bundle['ir'])      # ゴールデン(元バンドルのIR)
    in_width = int(graph.inputs[0].width)
    n_out = int(graph.outputs[0].width)
    rng = np.random.default_rng(seed)

    gen_bundle, flat_decision = bundle, None
    if dense_flat is not None and not dense_fuse:
        from .flatten_dense import describe_dense_flat, resolve_dense_flat
        gen_bundle, flat_decision = resolve_dense_flat(
            bundle, mode=dense_flat, min_reduction=dense_flat_min_reduction,
            target='gpu')
        if verbose:
            print('  ' + describe_dense_flat(flat_decision))

    # C版verifyと同じ理由(検証対象自身からゴールデンを作らない): preproc宣言
    # から作り直した記述子を基準にする。
    plan = None
    if bundle.get('preproc_desc') is not None:
        preproc = bundle.get('preproc') or {}
        if preproc.get('kind') in SUPPORTED_KINDS:
            plan = build_preproc_descriptors(preproc)
        else:
            raise ValueError(
                f"preproc_desc があるのに preproc.kind={preproc.get('kind')!r} "
                'から記述子を作り直せない(検証の基準を作れない)')

    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix='mdl_verify_metal_')
        work_dir = tmp.name
    try:
        use_image = plan is not None
        exe = build_metal_model(gen_bundle, work_dir, clangxx=clangxx,
                                image_entry=use_image, conv_engine=conv_engine,
                                conv_codegen=conv_codegen,
                                dense_fuse=dense_fuse)
        mode = 'image' if use_image else 'bits'
        for i in range(n_samples):
            if use_image:
                image = rng.integers(
                    0, 256,
                    size=(plan['image_channels'], plan['height'], plan['width']),
                    dtype=np.uint8)
                bits = encode_bits(plan, image[None])
                payload = image.tobytes()
            else:
                bits = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
                payload = bits[0].tobytes()
            want = simulate(graph, bits)[0].astype(np.int64)
            got = run_metal_model(exe, payload, n_out)
            if not np.array_equal(got, want):
                bad = np.nonzero(got != want)[0]
                raise ValueError(
                    f'Metal ⇔ simulate が一致しない(サンプル{i}、経路={mode}): '
                    f'{len(bad)}/{n_out} 要素が不一致。'
                    f'最初の不一致 index={int(bad[0])} '
                    f'Metal={int(got[bad[0]])} simulate={int(want[bad[0]])}')
            if verbose:
                print(f'  サンプル{i + 1}/{n_samples}: 一致({n_out}要素)')
    finally:
        if tmp is not None:
            tmp.cleanup()

    return {'n_samples': n_samples, 'n_out': n_out, 'path': mode,
            'max_abs_diff': 0, 'dense_flat': flat_decision}
