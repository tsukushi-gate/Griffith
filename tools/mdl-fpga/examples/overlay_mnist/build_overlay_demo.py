"""オーバーレイの一本道ビルド(生成 → presim → 合成 → 実機手順)。

    オンチップ画像 ROM(N 枚)
      → lgn_overlay(計算型: ゲート記述 op 4bit のみ。配線はアキュムレータ)
        または lgn_overlay_stored(保存型: 記録 {op, a, b} = 自由配線)
      → 期待値(golden = ir.simulate の予測)と比較・計数
      → ISSP で 一致数 / 処理数 / サイクル数 を読む、7セグに一致数

使い方:

    uv run python tools/mdl-fpga/examples/overlay_mnist/build_overlay_demo.py \\
        --ir path/to/mnist_p6_seed0.ir.json --n 1000
    uv run python tools/mdl-fpga/cli.py flash <表示された .sof>
    uv run python tools/mdl-fpga/cli.py issp <表示された projdir>

自由配線(置換・ランダム・学習配線)のモデルは `--wiring stored` で載る。
記録が 32bit/ゲートなので画像 ROM に回せる M10K が減る(DE0-CV の 48K は
N=512 枚で 296/308 = 先行実験 C1)。

オプション: --v(レーン数)/ --wiring / --clock-div / --skip-presim /
--skip-flow。
データ npz は examples/mnist/gen_data.py が作る形式(images_bits, labels)。
実機の 7セグ HEX3-0 は **golden 予測との一致数**(= 全数一致なら N)で、
精度そのものではない(精度は golden.json に出る)。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOL_DIR = HERE.parent.parent
sys.path.insert(0, str(TOOL_DIR))

from mdl_fpga.build import run_build                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ir', required=True)
    ap.add_argument('--data', default=str(
        TOOL_DIR / 'examples' / 'mnist' / 'mnist_test_1000.npz'))
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--v', type=int, default=8)
    ap.add_argument('--wiring', default='auto',
                    choices=['auto', 'computed', 'stored'],
                    help='配線の持ち方(既定 auto = 閉形式なら計算型)')
    ap.add_argument('--clock-div', type=int, default=2, dest='clock_div')
    ap.add_argument('--board', default='de0-cv')
    ap.add_argument('--workdir', default=str(
        TOOL_DIR / 'runs' / 'overlay_mnist'))
    ap.add_argument('--presim-vectors', type=int, default=4,
                    dest='presim_vectors',
                    help='合成前 RTL 検証の枚数(実ネットの vvp は遅い)')
    ap.add_argument('--skip-presim', action='store_true', dest='skip_presim')
    ap.add_argument('--skip-flow', action='store_true', dest='skip_flow',
                    help='生成だけ行い Quartus を回さない')
    args = ap.parse_args()

    data = np.load(args.data)
    images = data['images_bits'][:args.n].astype(np.int64)
    labels = data['labels'][:args.n]

    if args.skip_flow:
        # 合成を回さずに生成物だけ作る(プロジェクトは後から Quartus で開ける)
        from griffith_logic.export.lowering import lower_to_primitives
        from mdl_fpga.check import prepare_overlay_workdir
        from mdl_fpga.ir_io import load_graph
        projdir = Path(args.workdir) / 'quartus'
        graph = lower_to_primitives(load_graph(args.ir))
        plan, _gen, sysgen, golden, used = prepare_overlay_workdir(
            graph, projdir, images=images, labels=labels,
            n_vectors=len(images), v=args.v, clock_div=args.clock_div,
            wiring=args.wiring)
        print(f'生成完了: {projdir}(配線 {plan.wiring}、'
              f'ベクトル {len(used)} 枚、{plan.cycles_per_image} cyc/推論)')
        print(f'  golden 正解 {golden["correct"]}/{golden["n_images"]}')
        print(f'  meta: {json.dumps(sysgen.to_meta()["probe"])}')
        return 0

    res = run_build(args.ir, workdir=args.workdir, board=args.board,
                    mode='overlay', overlay_v=args.v,
                    overlay_vectors=len(images), clock_div=args.clock_div,
                    overlay_wiring=args.wiring,
                    skip_presim=args.skip_presim,
                    presim_vectors=args.presim_vectors,
                    images=images, labels=labels)
    print(res.render())
    if not res.ok:
        return 1
    projdir = Path(res.workdir) / 'quartus'
    print()
    print('次の一手:')
    print(f'  uv run python tools/mdl-fpga/cli.py flash "{res.sof}"')
    print(f'  uv run python tools/mdl-fpga/cli.py issp "{projdir}"')
    print(f'  実機の 7セグ: HEX3-0 = 一致数(期待 {len(images)})、'
          f'HEX5-4 = 処理数の百位({len(images) // 100:02d})')
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
