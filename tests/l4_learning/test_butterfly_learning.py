"""バタフライQATの収束テスト(L4): XORでhard acc==100%。

forwardが最初から離散(Heaviside)なので、train accがそのままhard acc。
収束後に転写→IR simulateでも100%であることまで確認する(3段の整合)。
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from griffith_logic.convert.butterfly_qat import (
    ButterflyMLP,
    transcribe_butterfly,
)
from griffith_logic.ir.simulate import simulate


def test_xor_hard_100():
    torch.manual_seed(3)
    X = np.array(list(itertools.product([0, 1], repeat=2)), dtype=np.int64)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int64)
    # バッチを増幅(4点×64)して勾配を安定化
    Xb = torch.as_tensor(np.tile(X, (64, 1)), dtype=torch.float64)
    yb = torch.as_tensor(np.tile(y, 64))

    model = ButterflyMLP(in_width=2, num_classes=2, n_blocks=1,
                         min_width=8).double()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    ok = False
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy(model(Xb), yb).backward()
        opt.step()
        with torch.no_grad():
            pred = model(torch.as_tensor(X, dtype=torch.float64)) \
                .argmax(-1).numpy()
        if (pred == y).all():
            ok = True
            break
    assert ok, 'XORがhard acc=100%へ収束しない'

    # 転写後IRでも100%(ロスレスの実地確認)
    graph = transcribe_butterfly(model)
    pred_ir = simulate(graph, X).argmax(-1)
    assert (pred_ir == y).all()
