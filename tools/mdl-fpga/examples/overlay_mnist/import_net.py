"""学習済み IR を取り込み、overlay 化できるか(どちらの方式で載るか)を検査する。

入力はライブラリ側(griffith-logic の recipe)が書き出した
magicdelta-ir v0.2.0 の JSON。ここでやるのは 2 つだけ:

  1) **配線の検査**: 全ゲートの indices を見て方式を決める
     - 計算型(computed): rot / 引き伸ばし rot の閉形式に全数一致する。
       ゲート記述は op 4bit だけになり、配線表・座標表が消える
     - 保存型(stored): 上に外れる自由配線(置換・ランダム・学習配線)。
       1 ゲート = {op 4bit, a, b} の記録(32bit or 40bit)で載せる
     --wiring computed を明示すると、閉形式に外れた IR を言い切って拒否する
  2) **精度ゲート**: ir.simulate の先頭 N 枚の正解数が、学習側の記録
     (--expect-correct)と完全一致するか。取り込み時の取り違え・破損を
     ここで止める(bit-honest)

使い方:

    uv run python tools/mdl-fpga/examples/overlay_mnist/import_net.py \\
        --ir path/to/mnist_p0_seed0.ir.json \\
        --data tools/mdl-fpga/examples/mnist/mnist_test_1000.npz \\
        --expect-correct 940

データ npz は examples/mnist/gen_data.py が作る形式(images_bits, labels)。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOL_DIR = HERE.parent.parent
sys.path.insert(0, str(TOOL_DIR))

from griffith_logic.export.lowering import lower_to_primitives   # noqa: E402
from griffith_logic.ir.simulate import simulate                  # noqa: E402
from mdl_fpga.hdlgen.overlay import (                              # noqa: E402
    OverlayUnsupported,
    estimate_gate_rom_bits,
    estimate_m10k,
    plan_overlay,
    wiring_label,
)
from mdl_fpga.ir_io import convert, load_graph                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ir', required=True, help='学習側が書き出した IR JSON')
    ap.add_argument('--data', default=str(
        TOOL_DIR / 'examples' / 'mnist' / 'mnist_test_1000.npz'))
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--v', type=int, default=8, help='レーン数')
    ap.add_argument('--wiring', default='auto',
                    choices=['auto', 'computed', 'stored'],
                    help='配線の持ち方(既定 auto = 閉形式なら計算型)')
    ap.add_argument('--images', type=int, default=None, dest='images',
                    help='M10K 見積りに使う検証 ROM の枚数(既定 --n)')
    ap.add_argument('--expect-correct', type=int, default=None,
                    dest='expect_correct',
                    help='学習側が記録した先頭 N 枚の正解数(不一致なら失敗)')
    args = ap.parse_args()

    conv = convert(args.ir)
    print(conv.render())
    if not conv.ok:
        return 1

    graph = lower_to_primitives(load_graph(args.ir))
    try:
        plan = plan_overlay(graph, v=args.v, wiring=args.wiring)
    except OverlayUnsupported as e:
        print(f'[FAIL] overlay 化できない: {e}')
        return 1

    print(f'[OK] 配線検査: 全 {plan.total_gates} ゲート → {wiring_label(plan)}')
    for warn in plan.warnings:
        print(f'  注意: {warn}')
    for ly in plan.layers:
        if plan.stored:
            print(f'  層 {ly.index}: {ly.kind} W_in={ly.w_in} '
                  f'W_out={ly.w_out}')
        else:
            print(f'  層 {ly.index}: {ly.kind} W_in={ly.w_in} '
                  f's_a={ly.s_a} s_b={ly.s_b} '
                  f'整数歩幅={ly.int_step} 余り歩幅={ly.rem_step}')
    bits = estimate_gate_rom_bits(plan)
    n_img = args.images if args.images is not None else args.n
    est = estimate_m10k(plan, n_images=n_img)
    print(f'  ゲート記述: {plan.gate_bits}bit/ゲート'
          f'(計算型 4bit なら {bits["computed_bits"] / 8 / 1024:.1f} KiB、'
          f'保存型 {bits["stored_word_bits"]}bit なら '
          f'{bits["stored_bits"] / 8 / 1024:.1f} KiB = {bits["ratio"]:.1f}x)')
    print(f'  ゲート ROM: {plan.nrom} 本 x {plan.gate_rom_depth} 語')
    print(f'  M10K 見積り(近似、検証 ROM {n_img} 枚): ゲート ROM '
          f'{est["gate_rom"]} / 活性 {est["activation"]} / 画像 ROM '
          f'{est["image_rom"]} / 計 {est["total"]}')
    print(f'  予測 {plan.cycles_per_image} cyc/推論')

    data = np.load(args.data)
    images = data['images_bits'][:args.n].astype(np.int64)
    labels = data['labels'][:args.n]
    if images.shape[1] != plan.in_bits:
        print(f'[FAIL] データの幅 {images.shape[1]} != IR の入力幅 '
              f'{plan.in_bits}')
        return 1
    preds = np.argmax(simulate(graph, images), axis=1)
    correct = int((preds == labels).sum())
    print(f'[OK] ir.simulate 先頭 {len(images)} 枚の正解数: {correct} '
          f'({correct / len(images):.2%})')
    if args.expect_correct is not None and correct != args.expect_correct:
        print(f'[FAIL] 精度ゲート不一致: 期待 {args.expect_correct} '
              f'!= 実測 {correct}。IR の取り違えか破損を疑う')
        return 1
    print('次の一手: build_overlay_demo.py で合成 → flash → issp')
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
