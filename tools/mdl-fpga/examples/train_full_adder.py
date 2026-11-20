"""学習済みモデル → IR → mdl-fpga の一本道デモ(全加算器)。

tests/l4_learning/test_toy_problems.py の実証済みレシピ
(全加算器を 4 クラス分類として LutLayer+GroupSum で学習)を使い、
hard acc=100% を確認して IR(full_adder.ir.json)を書き出す。以降は mdl-fpga の担当:

    uv run python tools/mdl-fpga/examples/train_full_adder.py
    uv run python tools/mdl-fpga/cli.py check tools/mdl-fpga/examples/full_adder.ir.json
    uv run python tools/mdl-fpga/cli.py build tools/mdl-fpga/examples/full_adder.ir.json
    uv run python tools/mdl-fpga/cli.py flash <build出力の.sof>
    uv run python tools/mdl-fpga/cli.py hw-check tools/mdl-fpga/examples/full_adder.ir.json

クラス = carry*2 + sum(0..3)。応答ペイロードは 4 クラス × score_bits のスコアで、
argmax の解釈は PC 側(hw-check は bit 列そのものを照合するため解釈不要)。
決定的なツール用サンプル(実験ではない)のため wandb ログはしない。
"""

import itertools
from pathlib import Path

import torch
import torch.nn.functional as F

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export.extract import export_ir
from griffith_logic.ir.serialize import save_ir

OUT = Path(__file__).resolve().parent / 'full_adder.ir.json'


def _dataset():
    xs = torch.tensor(list(itertools.product([0.0, 1.0], repeat=3)))
    ys = torch.tensor([(b[0] + b[1] + b[2] > 1) * 2 + (b[0] + b[1] + b[2]) % 2
                       for b in xs.long().tolist()])
    return xs, ys


def _train(seed: int, lr: float = 0.05, max_steps: int = 2000):
    torch.manual_seed(seed)
    x, y = _dataset()
    d = 64
    model = torch.nn.Sequential(
        LutLayer(3, d, n_inputs=2),
        LutLayer(d, d, n_inputs=2),
        GroupSum(k=4, tau=max(1.0, (d // 4) ** 0.5)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                return model, 1.0
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(-1) == y).float().mean().item()
    return model, acc


def main() -> None:
    model, acc = None, 0.0
    for seed in range(8):
        model, acc = _train(seed)
        print(f'seed {seed}: hard acc = {acc:.3f}')
        if acc == 1.0:
            break
    assert model is not None and acc == 1.0, '収束せず(想定外)'

    graph = export_ir(
        model, input_width=3, name='full_adder',
        metadata={'origin': 'tools/mdl-fpga/examples/train_full_adder.py',
                  'task': 'full_adder as 4-class (carry*2+sum)'})
    save_ir(graph, str(OUT))
    print(f'IR 保存: {OUT}')


if __name__ == '__main__':
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
