"""ESP32デモ用: 配布バンドル → main/model/(model.c/.h + ランタイム + samples.h)。

モデルを差し替えるときはこのスクリプトを1回走らせて `idf.py build` し直すだけ。

    uv run python examples/30_deploy_esp32/gen_model.py                 # XOR(内蔵4パターン)
    uv run python examples/30_deploy_esp32/gen_model.py \\
        --bundle out_examples/har.bundle.json \\
        --samples out_examples/demo_models/samples/har.json --name har

やること:
  (a) `griffith deploy codegen-c <bundle> -o main/model/` を呼ぶ
      (既定で dense-flat auto。削減率が閾値以上なら lut_flat で実行する。
      実行ゲート数・model_info.json の dense_flat はその判定後の値。RAM優先なら --dense-flat off)
      (model.c/model.h + mdl_kernels.c/.h + mdl_simd.h が出る)
  (b) サンプルJSONから main/model/samples.h を書く(無ければXORの4パターンを内蔵生成)
  (c) main/model/model_info.json に名前・サイズ・ゲート数を残す

samplesのJSON形式: {"input_width": int, "samples": [{"bits": "<hex>",
  "label": int|null, "expected_scores": [int,...]|null}]}
`bits` は **バイト列のhex**(`bytes.fromhex`)で、入力 i は バイト i//8 のビット i%8
(LSB-first)。samples.h にもこのパック形式のまま載せ、main.c が0/1バイト列へ
展開してから `mdl_infer_ctx` を呼ぶ(展開済み配列を持つとN×入力幅バイトだけ
flashを食うため)。

画像入力モデル(`"input_kind": "image_u8"`、顔検出など)のサンプルは `bits` の代わりに
`image`(u8 NCHW のhex、`sample_bytes` B)を持つ。samples.h は `SAMPLE_INPUT_IS_IMAGE 1` と
なり、main.c は展開せず `mdl_infer_image_ctx` へそのまま渡す。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_OUT = HERE / 'main' / 'model'

# XOR(2入力) の内蔵サンプル。bit0=a, bit1=b。(a,b)=(0,0),(0,1),(1,0),(1,1)。
XOR_SAMPLES = [
    {'bits': '00', 'label': 0},
    {'bits': '02', 'label': 1},
    {'bits': '01', 'label': 1},
    {'bits': '03', 'label': 0},
]


def run_griffith(*args: str) -> None:
    """`griffith ...` を現在のPython環境で実行する(uv run 経由で呼ばれる想定)。"""
    code = ('import sys; from griffith_logic.cli import main; '
            'sys.exit(main())')
    cmd = [sys.executable, '-c', code, *args]
    print('$ griffith', ' '.join(args))
    subprocess.run(cmd, cwd=REPO, check=True)


def ensure_xor_bundle(path: Path) -> Path:
    """XORバンドルが無ければ examples/06_deploy_c の手順で作る。"""
    if path.exists():
        return path
    model_json = REPO / 'out_examples' / 'xor_model.json'
    if not model_json.exists():
        subprocess.run(
            [sys.executable, str(REPO / 'examples' / '06_deploy_c' / 'export_model.py')],
            cwd=REPO, check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_griffith('deploy', 'bundle', str(model_json), '-o', str(path))
    return path


def load_bundle_summary(bundle_path: Path, dense_flat: str = 'auto') -> dict:
    """バンドルから入出力幅と**実行ゲート数**を取る。

    実行ゲート数は codegen-c が実際に生成する回路のもの: 現在は codegen-c は
    既定(`--dense-flat auto`)で C 生成時に dense を `lut_flat` へ平坦化するので、
    同じ判定(`resolve_dense_flat`、同じ引数)を適用した後の IR で数える。
    """
    from griffith_logic.deploy.bundle import load_bundle
    from griffith_logic.deploy.flatten_dense import resolve_dense_flat

    bundle = load_bundle(str(bundle_path), validate=False)
    c_bundle, decision = resolve_dense_flat(bundle, mode=dense_flat)
    ir = c_bundle['ir']
    gates = 0
    for node in ir['nodes']:
        if node['type'] == 'lut_layer':
            gates += int(node['params']['out_width'])
        elif node['type'] == 'lut_flat':
            # --dense-flatの平坦ネットリスト: 実行ゲート数は gates の三つ組の数
            gates += len(node['params']['gates'])
    for prog in (bundle.get('programs') or {}).values():
        gates += int(prog['stats']['gates_after'])
    return {
        'name': (ir.get('metadata') or {}).get('name') or 'model',
        'input_width': sum(int(i['width']) for i in ir['inputs']),
        'output_width': sum(int(o['width']) for o in ir['outputs']),
        'gates': gates,
        'node_types': sorted({n['type'] for n in ir['nodes']}),
        'dense_flat': {k: decision[k] for k in ('mode', 'applied', 'reduction')},
    }


def load_samples(path: Path | None, input_width: int, output_width: int,
                 max_samples: int | None) -> tuple[list[dict], bool]:
    """(samples, is_image)。is_image=True のとき raw は u8 NCHW 画像バイト列。"""
    is_image = False
    if path is None:
        if input_width != 2:
            raise SystemExit(
                '--samples 省略時の内蔵サンプルは XOR(入力幅2) 専用です。'
                f' このモデルの入力幅は {input_width}。--samples を指定してください')
        samples = XOR_SAMPLES
    else:
        doc = json.loads(path.read_text(encoding='utf-8'))
        if int(doc['input_width']) != input_width:
            raise SystemExit(
                f"samples.input_width={doc['input_width']} が"
                f' バンドルの入力幅 {input_width} と違います')
        samples = doc['samples']
        is_image = doc.get('input_kind') == 'image_u8'
    nbytes = (input_width + 7) // 8
    if is_image:
        nbytes = int(doc['sample_bytes'])
    out = []
    for i, s in enumerate(samples[:max_samples]):
        raw = bytes.fromhex(s['image' if is_image else 'bits'])
        if len(raw) != nbytes:
            raise SystemExit(f'sample {i}: bits は {nbytes}B 必要(実際 {len(raw)}B)')
        scores = s.get('expected_scores')
        if scores is not None and len(scores) != output_width:
            raise SystemExit(
                f'sample {i}: expected_scores の長さが出力幅 {output_width} と違います')
        out.append({'raw': raw,
                    'label': -1 if s.get('label') is None else int(s['label']),
                    'scores': scores})
    if not out:
        raise SystemExit('サンプルが0件です')
    return out, is_image


def render_samples_h(name: str, gates: int, input_width: int, output_width: int,
                     samples: list[dict], is_image: bool = False) -> str:
    n = len(samples)
    nbytes = len(samples[0]['raw']) if is_image else (input_width + 7) // 8
    has_scores = all(s['scores'] is not None for s in samples)
    L = [
        '/* 自動生成 — examples/30_deploy_esp32/gen_model.py。手で編集しない。 */',
        '#ifndef SAMPLES_H',
        '#define SAMPLES_H',
        '#include <stdint.h>',
        '',
        f'#define MODEL_NAME "{name}"',
        f'#define MODEL_GATES {gates}',
        f'#define SAMPLE_COUNT {n}',
        f'#define SAMPLE_INPUT_WIDTH {input_width}',
        f'#define SAMPLE_OUTPUT_WIDTH {output_width}',
        f'#define SAMPLE_INPUT_BYTES {nbytes}',
        f'#define SAMPLE_HAS_SCORES {1 if has_scores else 0}',
        f'#define SAMPLE_INPUT_IS_IMAGE {1 if is_image else 0}',
        '',
        '/* ビット入力: 入力 i = バイト i/8 のビット i%8 (LSB-first)。',
        ' * main.c が0/1バイト列へ展開する。',
        ' * SAMPLE_INPUT_IS_IMAGE=1 のときは u8 NCHW 画像で、そのまま',
        ' * mdl_infer_image_ctx へ渡す */',
        'static const uint8_t SAMPLE_BITS[SAMPLE_COUNT][SAMPLE_INPUT_BYTES] = {',
    ]
    for s in samples:
        L.append('  {' + ', '.join(f'0x{b:02x}' for b in s['raw']) + '},')
    L += ['};', '', '/* 分類ラベル(無ければ -1) */',
          'static const int32_t SAMPLE_EXPECTED[SAMPLE_COUNT] = {']
    L.append('  ' + ', '.join(str(s['label']) for s in samples))
    L.append('};')
    if has_scores:
        L += ['', '/* 参照スコア(IR simulate 由来)。実機との整数一致判定に使う */',
              'static const int32_t SAMPLE_EXPECTED_SCORES[SAMPLE_COUNT][SAMPLE_OUTPUT_WIDTH] = {']
        for s in samples:
            L.append('  {' + ', '.join(str(int(v)) for v in s['scores']) + '},')
        L.append('};')
    L += ['', '#endif /* SAMPLES_H */', '']
    return '\n'.join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bundle', type=Path, default=None,
                    help='配布バンドル(.bundle.json)。省略時はXORを生成して使う')
    ap.add_argument('--samples', type=Path, default=None,
                    help='サンプルJSON(scripts/make_device_samples.py の出力)。省略時はXOR内蔵')
    ap.add_argument('--name', default=None, help='モデル名(省略時はバンドルのmetadata.name)')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help=f'出力先(default: {DEFAULT_OUT.relative_to(REPO)})')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='載せるサンプル数の上限(flash節約用)')
    ap.add_argument('--dense-flat', choices=['auto', 'on', 'off'], default='auto',
                    help='codegen-c の --dense-flat(default: auto)。'
                         '平坦化は信号表ぶん作業RAMが増えるので、RAM優先なら off')
    args = ap.parse_args(argv)

    bundle = args.bundle
    if bundle is None:
        bundle = ensure_xor_bundle(REPO / 'out_examples' / 'xor.bundle.json')
    summary = load_bundle_summary(bundle, args.dense_flat)
    name = args.name or summary['name']

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run_griffith('deploy', 'codegen-c', str(bundle), '-o', str(out),
                 '--dense-flat', args.dense_flat)

    samples, is_image = load_samples(args.samples, summary['input_width'],
                                     summary['output_width'], args.max_samples)
    (out / 'samples.h').write_text(
        render_samples_h(name, summary['gates'], summary['input_width'],
                         summary['output_width'], samples, is_image),
        encoding='utf-8')

    info = {
        'name': name,
        'bundle': str(bundle),
        'samples_file': str(args.samples) if args.samples else 'builtin:xor',
        'n_samples': len(samples),
        'input_is_image': is_image,
        'input_width': summary['input_width'],
        'output_width': summary['output_width'],
        'gates': summary['gates'],
        'dense_flat': summary['dense_flat'],
        'node_types': summary['node_types'],
        'model_c_bytes': (out / 'model.c').stat().st_size,
        'model_h_bytes': (out / 'model.h').stat().st_size,
        'bundle_bytes': bundle.stat().st_size,
    }
    (out / 'model_info.json').write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'生成完了: {out} (model={name}, gates={summary["gates"]:,}, '
          f'samples={len(samples)}, model.c={info["model_c_bytes"]:,}B)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
