"""モデル1つ(バンドル + samples.json)を iOS デモへ取り込む。

  uv run python examples/31_deploy_ios/gen_model.py <name> \
      --bundle model.bundle.json --samples samples.json

やること(何度実行してもよい。他モデルのファイルには触らない):
  1. バンドル → 生成C(`griffith deploy codegen-c` と同じ generate_c、現在は
     既定で dense-flat auto)を
     Models/<name>/model.c, model.h に書く(ランタイムの複製は作らない。
     runtime/c/ をそのまま参照する)
  2. シンボル衝突対策の「unity ラッパ」Models/<name>/<name>_unit.c を書く
     (model.c が外部公開する mdl_* シンボルを <name>_mdl_* へ #define 置換して
     から #include する。複数モデルを1アプリへリンクできる)
  3. samples.json を Models/<name>/<name>_samples.json へ配置(Resources に入る)
  4. Models/gd_registry.c(全モデルの一覧)を作り直す
  5. GriffithDemo.xcodeproj を作り直す(gen_xcodeproj.py)

追加したモデルは Models/*/ を走査して自動で全部同梱される。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = HERE / 'Models'

# model.c / model.h が **外部公開する** シンボル。unity ラッパで <name>_ を前置して
# 置換する。mdl_kernels.c 側(全モデル共通で1つだけリンクする)のシンボルは含めない。
# 画像モデルでは mdl_infer_image* と mdl_labels が増える。
PREFIXED_SYMBOLS = [
    'mdl_ctx_init', 'mdl_infer_ctx', 'mdl_infer_ctx_v', 'mdl_infer_batch',
    'mdl_infer', 'mdl_infer_image', 'mdl_infer_image_ctx', 'mdl_labels',
    'mdl_infer_image_ctx_v', 'mdl_preproc_range_v', 'mdl_run_ctx',
]
# model.h の include ガードも衝突しうるが、モデルごとに別 TU なので問題ない。


def _count_gates(bundle: dict) -> int:
    """ゲート数 = LUT段の出力幅の合計(lut_flat は gates 数) + conv2d の縮約後命令列ゲート数。"""
    total = 0
    for node in bundle['ir']['nodes']:
        if node['type'] == 'lut_layer':
            total += int(node['params']['out_width'])
        elif node['type'] == 'lut_flat':
            total += len(node['params']['gates'])  # 平坦ネットリスト(--dense-flat)の実行ゲート数
    for p in (bundle.get('programs') or {}).values():
        total += int(p['stats']['gates_after'])
    return total


def _prefixed_symbols_in(src: str) -> list[str]:
    return [s for s in PREFIXED_SYMBOLS if re.search(rf'\b{s}\b', src)]


def _unit_c(name: str, symbols: list[str], gates: int, image_bytes: int | None) -> str:
    defs = '\n'.join(f'#define {s} {name}_{s}' for s in symbols)
    undefs = '\n'.join(f'#undef {s}' for s in symbols)
    # 画像モデルは前処理込みの mdl_infer_image_ctx(u8 NCHW 画像)を呼ぶ。
    infer = f'{name}_mdl_infer_image_ctx' if image_bytes else f'{name}_mdl_infer_ctx'
    kind, in_bytes = (1, image_bytes) if image_bytes else (0, 'MDL_INPUT_WIDTH')
    init = f'{name}_mdl_ctx_init'
    return f'''/* 自動生成(gen_model.py)。手で編集しないこと。
 * モデル "{name}" の unity ラッパ: model.c が公開する mdl_* シンボルを
 * {name}_mdl_* へ置換してから取り込み、複数モデルを1つのアプリへリンクできるようにする。
 * ランタイム(mdl_kernels.c)は全モデル共通の1つをリンクする。 */
#include <stddef.h>
#include <stdint.h>

{defs}
#include "model.c"
{undefs}

#include "GdModel.h"

static int gd_{name}_init(void *state, void *work, size_t bytes) {{
  return {init}((mdl_ctx *)state, work, bytes);
}}
static int gd_{name}_infer(void *state, const uint8_t *bits, int32_t *scores) {{
  return {infer}((mdl_ctx *)state, bits, scores);
}}

const gd_model gd_model_{name} = {{
  "{name}",
  MDL_INPUT_WIDTH, MDL_OUTPUT_WIDTH, MDL_CTX_BYTES, MDL_NUM_LABELS, {gates},
  {kind}, {in_bytes},
  sizeof(mdl_ctx), gd_{name}_init, gd_{name}_infer,
}};
'''


def _registry_c(names: list[str]) -> str:
    decl = '\n'.join(f'extern const gd_model gd_model_{n};' for n in names)
    arr = ',\n'.join(f'  &gd_model_{n}' for n in names)
    return f'''/* 自動生成(gen_model.py)。Models 以下の全モデルの一覧。 */
#include "GdModel.h"

{decl}

static const gd_model *const gd_models[] = {{
{arr}
}};

int gd_model_count(void) {{ return {len(names)}; }}
const gd_model *gd_model_at(int i) {{
  return (i >= 0 && i < {len(names)}) ? gd_models[i] : (const gd_model *)0;
}}
'''


def add_model(name: str, bundle_path: Path, samples_path: Path,
              dense_flat: str = 'auto') -> None:
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        sys.exit(f'name は小文字英数字と _ のみ(Cシンボルに使う): {name!r}')
    from griffith_logic.deploy.bundle import load_bundle
    from griffith_logic.deploy.cbackend import generate_c
    from griffith_logic.deploy.flatten_dense import (
        describe_dense_flat,
        resolve_dense_flat,
    )

    bundle = load_bundle(bundle_path, validate=True)
    # codegen-c と同じ dense-flat 判定(既定 auto)。ゲート数は判定後の
    # 実際に実行する回路で数える。
    bundle, decision = resolve_dense_flat(bundle, mode=dense_flat)
    print(describe_dense_flat(decision))
    files = generate_c(bundle, name='model', validate=False,
                       dense_flat_decision=decision)
    out = MODELS / name
    out.mkdir(parents=True, exist_ok=True)
    for fn in ('model.c', 'model.h'):
        (out / fn).write_text(files[fn], encoding='utf-8')

    # samples を検証して配置
    doc = json.loads(samples_path.read_text(encoding='utf-8'))
    w = int(doc['input_width'])
    hdr = (out / 'model.h').read_text(encoding='utf-8')
    m = re.search(r'#define MDL_INPUT_WIDTH (\d+)', hdr)
    if m and int(m.group(1)) != w:
        sys.exit(f'samples の input_width={w} が model.h の MDL_INPUT_WIDTH={m.group(1)} と不一致')
    image_bytes = None
    if 'mdl_infer_image_ctx' in hdr:
        # 画像モデル: samples は {"image": "<hex of u8 CHW>"}。入力は u8 画像のまま渡す。
        mm = {k: re.search(rf'#define {k} (\d+)', hdr)
              for k in ('MDL_IMAGE_CHANNELS', 'MDL_INPUT_HEIGHT', 'MDL_INPUT_WIDTH_PX')}
        if not all(mm.values()):
            sys.exit('画像モデルの model.h に MDL_IMAGE_CHANNELS/MDL_INPUT_HEIGHT/'
                     'MDL_INPUT_WIDTH_PX が無い')
        image_bytes = 1
        for v in mm.values():
            image_bytes *= int(v.group(1))
        for i, s_ in enumerate(doc['samples']):
            if len(s_.get('image', '')) != 2 * image_bytes:
                sys.exit(f'samples[{i}].image の長さが {2 * image_bytes} hex 桁ではない')
    else:
        nbytes = (w + 7) // 8
        for i, s_ in enumerate(doc['samples']):
            if len(s_['bits']) != 2 * nbytes:
                sys.exit(f'samples[{i}].bits の長さが {2 * nbytes} hex 桁ではない')
    shutil.copyfile(samples_path, out / f'{name}_samples.json')

    src = files['model.c'] + files['model.h']
    symbols = _prefixed_symbols_in(src)
    (out / f'{name}_unit.c').write_text(
        _unit_c(name, symbols, _count_gates(bundle), image_bytes),
        encoding='utf-8')
    print(f'Models/{name}: model.c/.h + {name}_unit.c + {name}_samples.json '
          f'({len(doc["samples"])} samples, gates={_count_gates(bundle)})')


def check_runtime_copy() -> None:
    """runtime/c を直接参照するので、codegen が複製するランタイムとの差は無いはず。"""
    # generate_c 自体はランタイムを複製しない(CLI だけが複製する)。念のため存在確認。
    for rel in ('include/mdl_kernels.h', 'include/mdl_simd.h', 'src/mdl_kernels.c'):
        if not (REPO / 'runtime' / 'c' / rel).exists():
            sys.exit(f'runtime/c/{rel} が見つからない')


def finalize() -> None:
    names = sorted(p.name for p in MODELS.iterdir()
                   if p.is_dir() and (p / f'{p.name}_unit.c').exists())
    if not names:
        sys.exit('Models/ にモデルが無い')
    (MODELS / 'gd_registry.c').write_text(_registry_c(names), encoding='utf-8')
    sys.path.insert(0, str(HERE))
    import gen_xcodeproj
    gen_xcodeproj.write_project(names)
    print(f'registry + GriffithDemo.xcodeproj を更新: {", ".join(names)}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', help='モデル名(小文字英数字と _。Cシンボル接頭辞になる)')
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--samples', required=True, type=Path)
    ap.add_argument('--dense-flat', choices=['auto', 'on', 'off'], default='auto',
                    help='codegen-c の --dense-flat(default: auto)')
    args = ap.parse_args()
    check_runtime_copy()
    add_model(args.name, args.bundle, args.samples, args.dense_flat)
    finalize()


if __name__ == '__main__':
    main()
