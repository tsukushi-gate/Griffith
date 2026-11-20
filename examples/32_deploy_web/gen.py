"""Web デプロイ例: モデル → 配布バンドル → `codegen-web` → 自前ページを重ねる。

ネットワーク不要(XOR を学習する)。成果物は out_examples/deploy_web/ に出る。

    uv run python examples/32_deploy_web/gen.py                # XOR
    uv run python examples/32_deploy_web/gen.py \\
        --bundle out_examples/deploy_mnist/mnist.bundle.json --samples my_samples.json

    python -m http.server 8080 -d out_examples/deploy_web      # → http://localhost:8080/

前提: `make web-runtime` で runtime/web/dist をビルド済みであること
(無いと codegen-web が失敗して指示を出す)。

samples.json の形式(index.html/main.js が読む。bits は 0/1 の配列、長さ = 入力ビット幅):
  {"input_width": 2, "samples": [{"bits": [0, 1], "label": 1}, ...]}
label は無ければ null(正解率は出ない)。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_OUT = REPO / 'out_examples' / 'deploy_web'


def griffith(*args: str) -> None:
    """`uv run griffith ...` をリポジトリ直下で実行する。"""
    cmd = ['uv', 'run', 'griffith', *args]
    print('$', ' '.join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def ensure_xor_bundle(path: Path) -> Path:
    """XOR バンドルを作る(06_deploy_c の手順: 学習 → IR → bundle)。"""
    model_json = REPO / 'out_examples' / 'xor_model.json'
    if not model_json.exists():
        subprocess.run([sys.executable, str(REPO / 'examples' / '06_deploy_c' / 'export_model.py')],
                       cwd=REPO, check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    griffith('deploy', 'bundle', str(model_json), '-o', str(path))
    return path


def xor_samples() -> dict:
    """XOR の全4パターン(入力2ビット、正解 = a XOR b)。"""
    return {'input_width': 2, 'samples': [
        {'bits': [a, b], 'label': a ^ b} for a in (0, 1) for b in (0, 1)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--bundle', type=Path, help='既存のバンドル(省略時は XOR を作る)')
    ap.add_argument('--samples', type=Path, help='samples.json(--bundle 指定時は必須)')
    ap.add_argument('-o', '--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.bundle and not args.samples:
        ap.error('--bundle を指定するときは --samples も必要')

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = args.bundle or ensure_xor_bundle(args.out.parent / 'xor.bundle.json')
    samples = (json.loads(args.samples.read_text(encoding='utf-8'))
               if args.samples else xor_samples())

    # 配布物一式: model.bundle.json + runtime/web/dist(mdl_web.js, *.wasm, チャンク…)
    griffith('deploy', 'codegen-web', str(bundle), '-o', str(args.out))
    # 自前ページを重ねる(dist/index.html の既定デモシェルは上書きされる)
    shutil.copy2(HERE / 'index.html', args.out / 'index.html')
    shutil.copy2(HERE / 'main.js', args.out / 'main.js')
    (args.out / 'samples.json').write_text(json.dumps(samples), encoding='utf-8')

    # 出荷ゲート: Web ランタイムの出力が ir/simulate と整数で完全一致するか(js/wasm 両方)
    for engine in ('js', 'wasm'):
        griffith('deploy', 'verify-web', str(bundle), '--engine', engine)
    print(f'\n完了: python -m http.server 8080 -d {args.out}  →  http://localhost:8080/')


if __name__ == '__main__':
    main()
