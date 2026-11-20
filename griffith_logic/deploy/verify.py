"""Cランタイムの等価性ガード(K5)。

`ir/simulate.py`(numpyのgolden model)を基準に、生成したCを実際にビルド・
実行して**整数厳密**に突き合わせる。不一致ならその場で失敗させる。

これは教訓を新しい出口にも適用したもの。あのときは
「TS ⇔ simulate(エクスポート済みIRの自己整合性)しか見ておらず、
PyTorch ⇔ simulate を検証していなかった」ため、壊れたIR同士が一致して
全テストが緑のまま壊れたモデルを配信していた。ターゲットが増えるほど
「ゴールデンを検証対象自身から作らない」ことが効くので、Cの検証も必ず
simulate を基準にする。

等価性の鎖:

    PyTorch hard eval ⇔ ir/simulate ⇔ C ランタイム
                              ⇔ TS/WASM/WebGPU(webbench)
                              ⇔ Verilog/VHDL(HDL)

画像入力のモデルでは、前処理も含めて
「生画像 → C前処理 → C推論」 ⇔ 「生画像 → Python記述子実行 → simulate」
を突き合わせる。
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..ir.serialize import ir_from_dict
from ..ir.simulate import simulate
from .cbackend import generate_c
from .preproc_desc import SUPPORTED_KINDS, build_preproc_descriptors, encode_bits

RUNTIME_C = Path(__file__).resolve().parents[2] / 'runtime' / 'c'

_DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
int main(void) {
  MDL_STDIO_BINARY();
  static uint8_t bits[MDL_INPUT_WIDTH];
  static int32_t scores[MDL_OUTPUT_WIDTH];
  int32_t i;
  if (fread(bits, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) {
    fprintf(stderr, "input short read\\n");
    return 1;
  }
  if (mdl_infer(bits, scores) != 0) return 2;
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
    unsigned char b[4];
    uint32_t v = (uint32_t)scores[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }
  return 0;
}
'''

_IMAGE_DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
int main(void) {
  MDL_STDIO_BINARY();
  static uint8_t image[MDL_IMAGE_CHANNELS * MDL_INPUT_HEIGHT
                       * MDL_INPUT_WIDTH_PX];
  static int32_t scores[MDL_OUTPUT_WIDTH];
  int32_t i;
  if (fread(image, 1, sizeof(image), stdin) != sizeof(image)) {
    fprintf(stderr, "image short read\\n");
    return 1;
  }
  if (mdl_infer_image(image, scores) != 0) return 2;
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
    unsigned char b[4];
    uint32_t v = (uint32_t)scores[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }
  return 0;
}
'''

#: `mdl_infer_batch_packed`(dense のバッチ軸パック入口)の照合用ドライバ。
#: stdin = n(int32 LE)+ n*MDL_INPUT_WIDTH バイト、stdout = n*MDL_OUTPUT_WIDTH 個の
#: int32 LE。n が MDL_DENSE_BATCH を超えるときは入口の上限ごとに区切って呼ぶ
#: (区切りと 32 枚/語の語境界の両方を検証に乗せる)。
_PACKED_DRIVER = '''#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include "mdl_stdio.h"
static uint32_t work[MDL_DENSE_WORK_BYTES / 4 + 1];
int main(void) {
  unsigned char hdr[4];
  int32_t n, done, k, i;
  uint8_t *in_;
  int32_t *sc;
  MDL_STDIO_BINARY();
  if (fread(hdr, 1, 4, stdin) != 4) return 1;
  n = (int32_t)((uint32_t)hdr[0] | ((uint32_t)hdr[1] << 8)
                | ((uint32_t)hdr[2] << 16) | ((uint32_t)hdr[3] << 24));
  if (n < 1) return 1;
  in_ = (uint8_t *)malloc((size_t)n * MDL_INPUT_WIDTH);
  sc = (int32_t *)malloc((size_t)n * MDL_OUTPUT_WIDTH * sizeof(int32_t));
  if (in_ == NULL || sc == NULL) return 3;
  if (fread(in_, 1, (size_t)n * MDL_INPUT_WIDTH, stdin)
      != (size_t)n * MDL_INPUT_WIDTH) {
    fprintf(stderr, "input short read\\n");
    return 1;
  }
  for (done = 0; done < n; done += k) {
    k = n - done;
    if (k > MDL_DENSE_BATCH) k = MDL_DENSE_BATCH;
    if (mdl_infer_batch_packed(work, sizeof(work),
                               in_ + (size_t)done * MDL_INPUT_WIDTH,
                               sc + (size_t)done * MDL_OUTPUT_WIDTH, k) != 0)
      return 2;
  }
  for (i = 0; i < n * MDL_OUTPUT_WIDTH; i++) {
    unsigned char b[4];
    uint32_t v = (uint32_t)sc[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }
  free(in_);
  free(sc);
  return 0;
}
'''

#: パック入口の照合に使う最少枚数。33 枚 = 32 枚/語の語境界と、既定の
#: `MDL_DENSE_BATCH`(32)での区切り呼び出しの両方を跨ぐ。
PACKED_VERIFY_MIN_SAMPLES = 33

#: `build_c_model` がパック入口の照合用に並べてビルドする実行ファイル名。
PACKED_EXE_NAME = 'app_packed'


def find_cc() -> str | None:
    for name in ('cc', 'clang', 'gcc'):
        path = shutil.which(name)
        if path:
            return path
    return None


def _compile(compiler: str, work: Path, name: str, driver: str, exe: Path) -> None:
    build = subprocess.run(
        [compiler, '-std=c99', '-O2', f'-I{work}',
         f'-I{RUNTIME_C / "include"}',
         str(work / f'{name}.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
         str(work / driver), '-o', str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f'生成Cのビルドに失敗:\n{build.stderr}')


def build_c_model(bundle: dict, work_dir, *, cc: str | None = None,
                  image_entry: bool = False, name: str = 'model',
                  dense_flat_decision: dict | None = None,
                  dense_batch: int = 0) -> Path:
    """バンドルからCを生成してビルドし、実行ファイルのパスを返す。

    ``dense_flat_decision``: 呼び出し側が `resolve_dense_flat` で作った
    C 生成用バンドルを ``bundle`` に渡すときの判定結果(model.h に記録される)。
    ``dense_batch``: `generate_c` の同名引数(`codegen-c --dense-batch`)。
    生成 C が `mdl_infer_batch_packed` を持つときは、その照合用ドライバでも
    ビルドし、返す実行ファイルと同じディレクトリに ``PACKED_EXE_NAME`` で置く。
    """
    compiler = cc or find_cc()
    if compiler is None:
        raise RuntimeError('Cコンパイラ(cc/clang/gcc)が見つからない')
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    files = generate_c(bundle, name=name, validate=False,
                       dense_flat_decision=dense_flat_decision,
                       dense_batch=dense_batch)
    for filename, text in files.items():
        (work / filename).write_text(text, encoding='utf-8')
    driver = _IMAGE_DRIVER if image_entry else _DRIVER
    (work / 'driver.c').write_text(driver.replace('model.h', f'{name}.h'),
                                   encoding='utf-8')
    # Windows のツールチェーンは -o app でも app.exe を作る。返すパスを実体に合わせる
    exe = work / ('app.exe' if sys.platform == 'win32' else 'app')
    _compile(compiler, work, name, 'driver.c', exe)
    if 'mdl_infer_batch_packed' in files.get(f'{name}.h', ''):
        (work / 'driver_packed.c').write_text(
            _PACKED_DRIVER.replace('model.h', f'{name}.h'), encoding='utf-8')
        _compile(compiler, work, name, 'driver_packed.c',
                 work / (PACKED_EXE_NAME + exe.suffix))
    return exe


def run_c_model_packed(exe: Path, bits: np.ndarray, n_out: int) -> np.ndarray:
    """`_PACKED_DRIVER` でビルドした実行ファイルに n 枚を渡し、[n, n_out] のスコアを返す。"""
    bits = np.ascontiguousarray(bits, dtype=np.uint8)
    n = int(bits.shape[0])
    payload = np.int32(n).astype('<i4').tobytes() + bits.tobytes()
    scores = run_c_model(exe, payload, n * n_out)
    return scores.reshape(n, n_out)


def run_c_model(exe: Path, payload: bytes, n_out: int) -> np.ndarray:
    proc = subprocess.run([str(exe)], input=payload, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f'C実行が失敗 (exit={proc.returncode}): '
            f'{proc.stderr.decode(errors="replace")}')
    scores = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
    if scores.size != n_out:
        raise RuntimeError(
            f'C出力の要素数が {scores.size} で期待 {n_out} と違う')
    return scores


def verify_c_matches_simulate(bundle: dict, *, n_samples: int = 3,
                              seed: int = 0, cc: str | None = None,
                              work_dir=None, verbose: bool = True,
                              dense_flat: str | None = None,
                              dense_flat_min_reduction: float | None = None,
                              dense_batch: int | None = None) -> dict:
    """C実行 ⇔ ir/simulate を整数厳密に突き合わせる。

    画像前処理を持つバンドルでは、生画像から始めて前処理も検証範囲に入れる
    (C側 mdl_infer_image ⇔ Python側 記述子実行 + simulate)。

    ``dense_flat``: ``'auto'|'on'|'off'`` を渡すと `codegen-c` と同じ
    判定(`resolve_dense_flat`)で、C 生成用のバンドルだけをメモリ上で平坦化する。
    **ゴールデンは常に元バンドルの IR の simulate** なので、平坦化の lossless
    性(= 採用した C が元の回路と整数一致すること)がこの関門で実証される。
    None(既定)は従来どおり ``bundle['ir']`` をそのまま生成する。

    ``dense_batch``: `codegen-c --dense-batch` と同じ。None(既定)は codegen-c
    の既定と同じく、平坦化を採用したときだけ ``plan.DENSE_BATCH_AUTO``(32)、
    それ以外は 0。生成 C が `mdl_infer_batch_packed` を持つときは、1 枚ずつの
    `mdl_infer` に加えて**パック入口も** simulate と整数照合する(枚数は
    ``max(n_samples, PACKED_VERIFY_MIN_SAMPLES)``。語境界と入口の上限での
    区切りを跨ぐ)。

    Returns:
        {'n_samples', 'n_out', 'path': 'image'|'bits', 'max_abs_diff': 0,
         'dense_flat': decision|None, 'batch_packed_samples': int}
        (``batch_packed_samples`` はパック入口で照合した枚数。入口が無ければ 0)

    Raises:
        ValueError: 1サンプルでも不一致があった場合
        RuntimeError: ビルド・実行に失敗した場合
    """
    graph = ir_from_dict(bundle['ir'])      # ゴールデン(元バンドルのIR)
    in_width = int(graph.inputs[0].width)
    n_out = int(graph.outputs[0].width)
    rng = np.random.default_rng(seed)

    c_bundle, flat_decision = bundle, None
    if dense_flat is not None:
        from .flatten_dense import describe_dense_flat, resolve_dense_flat
        c_bundle, flat_decision = resolve_dense_flat(
            bundle, mode=dense_flat, min_reduction=dense_flat_min_reduction)
        if verbose:
            print('  ' + describe_dense_flat(flat_decision))
    if dense_batch is None:
        from .plan import DENSE_BATCH_AUTO
        dense_batch = (DENSE_BATCH_AUTO
                       if flat_decision is not None and flat_decision['applied']
                       else 0)

    # Python側の基準は **preproc宣言から作り直した記述子** を使う。
    # バンドルに入っている preproc_desc をそのまま両側で使うと、記述子が
    # 壊れていても両側が同じように壊れて一致してしまう(検証対象自身から
    # ゴールデンを作る、同じ構造の穴)。
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
        tmp = tempfile.TemporaryDirectory(prefix='mdl_verify_')
        work_dir = tmp.name
    try:
        use_image = plan is not None
        exe = build_c_model(c_bundle, work_dir, cc=cc, image_entry=use_image,
                            dense_flat_decision=flat_decision,
                            dense_batch=dense_batch)
        mode = 'image' if use_image else 'bits'
        sample_bits = []
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
            sample_bits.append(bits[0])
            want = simulate(graph, bits)[0].astype(np.int64)
            got = run_c_model(exe, payload, n_out)
            if not np.array_equal(got, want):
                bad = np.nonzero(got != want)[0]
                raise ValueError(
                    f'C ⇔ simulate が一致しない(サンプル{i}、経路={mode}): '
                    f'{len(bad)}/{n_out} 要素が不一致。'
                    f'最初の不一致 index={int(bad[0])} '
                    f'C={int(got[bad[0]])} simulate={int(want[bad[0]])}')
            if verbose:
                print(f'  サンプル{i + 1}/{n_samples}: 一致({n_out}要素)')
        packed_exe = Path(exe).with_name(PACKED_EXE_NAME + Path(exe).suffix)
        n_packed = 0
        if packed_exe.exists():
            n_packed = max(n_samples, PACKED_VERIFY_MIN_SAMPLES)
            extra = rng.integers(0, 2, size=(n_packed - len(sample_bits), in_width),
                                 dtype=np.uint8)
            all_bits = np.concatenate(
                [np.asarray(sample_bits, dtype=np.uint8).reshape(-1, in_width), extra])
            want = simulate(graph, all_bits).astype(np.int64)
            got = run_c_model_packed(packed_exe, all_bits, n_out)
            if not np.array_equal(got, want):
                bad = np.argwhere(got != want)
                s_i, o_i = (int(v) for v in bad[0])
                raise ValueError(
                    'C ⇔ simulate が一致しない(mdl_infer_batch_packed、'
                    f'{n_packed}枚): {len(bad)}/{n_packed * n_out} 要素が不一致。'
                    f'最初の不一致 サンプル{s_i} index={o_i} '
                    f'C={int(got[s_i, o_i])} simulate={int(want[s_i, o_i])}')
            if verbose:
                print(f'  mdl_infer_batch_packed: {n_packed}枚 一致({n_out}要素/枚)')
    finally:
        if tmp is not None:
            tmp.cleanup()

    return {'n_samples': n_samples, 'n_out': n_out, 'path': mode,
            'max_abs_diff': 0, 'dense_flat': flat_decision,
            'batch_packed_samples': n_packed}
