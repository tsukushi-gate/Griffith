"""MNIST 分類デモの一本道ビルド。

ROM(1000枚)→ パイプライン推論(latency 7clk)→ argmax → 正解カウント →
7セグ(HEX3-0=正解4桁 / HEX5-4=処理百位)。SW[0]=低速、KEY[0]=リスタート。

    uv run python tools/mdl-fpga/examples/mnist/build_mnist_demo.py
    # → スモークsim → Quartus(fit 十数分)→ flash コマンド提示
    #   実機受入: 高速実行後に HEX3-0 == golden.json['correct'](期待 0938)

オプション: --skip-flow(合成しない)/ --skip-smoke / --n / --smoke-n
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOL_DIR = HERE.parent.parent
sys.path.insert(0, str(TOOL_DIR))

from mdl_fpga.board_db import load_board                      # noqa: E402
from mdl_fpga.check import run_system_check                   # noqa: E402
from mdl_fpga.hdlgen.system import (                          # noqa: E402
    generate_system,
    generate_system_view_wrapper,
    write_system_vectors,
)
from mdl_fpga.hdlgen.wrapper import HDL_DIR                   # noqa: E402
from mdl_fpga.ir_io import load_graph                         # noqa: E402
from mdl_fpga.synth import run_flow, write_project_files      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=str(HERE / 'mnist24k.ir.json'))
    ap.add_argument('--data', default=str(HERE / 'mnist_test_1000.npz'))
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--smoke-n', type=int, default=3)
    ap.add_argument('--workdir',
                    default=str(TOOL_DIR / 'runs' / 'mnist_demo'))
    ap.add_argument('--skip-smoke', action='store_true')
    ap.add_argument('--skip-flow', action='store_true')
    args = ap.parse_args()

    graph = load_graph(args.model)
    data = np.load(args.data)
    images = data['images_bits'][:args.n]
    labels = data['labels'][:args.n]
    wd = Path(args.workdir)

    # 1) 生成
    gen = generate_system(graph, n_images=len(images))
    projdir = wd / 'quartus'
    projdir.mkdir(parents=True, exist_ok=True)
    (projdir / 'top.v').write_text(gen.top_verilog, encoding='utf-8')
    (projdir / 'sys_rom.v').write_text(gen.rom_verilog, encoding='utf-8')
    (projdir / 'mdl_core_pipe.v').write_text(gen.core.verilog,
                                             encoding='utf-8')
    shutil.copyfile(HDL_DIR / 'seg7.v', projdir / 'seg7.v')
    (projdir / 'top_view.v').write_text(generate_system_view_wrapper(),
                                        encoding='utf-8')
    (projdir / 'system_meta.json').write_text(
        json.dumps(gen.system_meta, ensure_ascii=False, indent=1),
        encoding='utf-8')
    golden = write_system_vectors(graph, images, labels, projdir,
                                  gen.label_bits)
    print(f'生成完了: {projdir}')
    print(f'  latency={gen.latency}clk / ROM {len(images)}枚×'
          f'{gen.rom_word_bits}bit / golden 正解 {golden["correct"]}枚')

    # 2) スモーク RTL シム(24Kゲート vvp のため少数枚)
    if not args.skip_smoke:
        print(f'スモーク RTL シム({args.smoke_n}枚、数十秒)...')
        res = run_system_check(graph, images[:args.smoke_n],
                               labels[:args.smoke_n], wd / 'smoke')
        print(res.render())
        if not res.passed:
            return 1

    # 3) Quartus プロジェクト+フロー
    board = load_board('de0-cv')
    write_project_files(projdir, name='top', board=board,
                        sources=gen.sources,
                        groups=gen.groups)
    if args.skip_flow:
        print('（--skip-flow: 合成は未実行。プロジェクトは生成済み）')
        return 0

    print('Quartus フロー実行中(24Kゲート、fit 十数分)...')
    flow = run_flow(projdir, 'top')
    print(flow.render())
    if not flow.ok:
        return 1

    # 4) ゲート(fit summary の妥当性)
    # 注: Quartus は「全画像で常時0の画素列」を ROM から定数化して幅を削り
    # (MNIST の縁)、その定数に依存するゲートの FF も定数伝播で刈る。
    # 実測(1000枚): ram_bits 788K→781K、registers 24K→約18K。機能は保存される
    # (スモークsim+実機 0938 が最終受入)ため、閾値は最適化を許容する値にする。
    s = flow.summary
    expect_ram = gen.rom_word_bits * len(images)
    notes = []
    if s.get('ram_bits', 0) < int(expect_ram * 0.85):
        notes.append(f'⚠ ram_bits={s.get("ram_bits")} が ROM 想定'
                     f'({expect_ram})の85%未満 → M10K 推論失敗の疑い')
    if s.get('registers', 0) < 10000:
        notes.append(f'⚠ registers={s.get("registers")} が極端に少ない'
                     ' → パイプFF欠落の疑い')
    if (s.get('fmax_clk50_mhz') or 0) < 50.0:
        notes.append(f'⚠ Fmax={s.get("fmax_clk50_mhz")}MHz < 50MHz')
    for n in notes:
        print(n)
    if notes:
        return 1

    print()
    print('次の一手:')
    print(f'  uv run python tools/mdl-fpga/cli.py flash "{flow.sof}"')
    print(f'  実機受入: 電源投入/KEY[0]後、HEX3-0 が'
          f' {golden["correct"]:04d}、HEX5-4 が {len(images) // 100:02d} を表示')
    print(f'  可視化: uv run python tools/mdl-fpga/cli.py view {args.model}'
          f' --system "{projdir}"')
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
