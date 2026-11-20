"""L4: LTT(線形閾値ゲート)の学習収束。

単層の n=2 閾値ゲートは XOR を表せない(線形分離不能)が、2 層なら表せる。
XOR / 3-parity で **hard acc = 100%** に到達することを固定する(LTT の forward は常に
硬いので soft / hard の離散化ギャップは構造的に無い。合格基準 1.0 は緩めない)。

STE は先行実験の既定 clip と sigmoid を固定対象にする。identity(DWN と同条件の素通し)は
XOR 幅 16×2 で seed 依存に収束しない(実測: lr 0.05 / 0.01、幅 16 / 32、3 層のいずれも
3 seed 中 1〜2 本が 1500 ステップで未到達。plan §5)ので L4 の対象から外し、
層の選択肢としては残す。
"""

import itertools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic import GroupSum
from griffith_logic.layers.ltt import LTTLayer

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning

STE_MODES = ['clip', 'sigmoid']


def _bits_dataset(n_bits, label_fn):
    xs = torch.tensor(list(itertools.product([0.0, 1.0], repeat=n_bits)))
    ys = torch.tensor([label_fn(row) for row in xs.long().tolist()])
    return xs, ys


def _train_until(model, x, y, *, lr=0.05, max_steps=2000, check_every=25):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if (step + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                return 1.0
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


def _ltt_net(in_dim, k_classes, widths, n_inputs, ste, seed=0):
    torch.manual_seed(seed)
    layers = []
    d = in_dim
    for w in widths:
        layers.append(LTTLayer(d, w, n_inputs=min(n_inputs, d), ste=ste, connections='random'))
        d = w
    layers.append(GroupSum(k=k_classes, tau=max(1.0, (d // k_classes) ** 0.5)))
    return nn.Sequential(*layers)


class TestXOR:
    @pytest.mark.parametrize('ste', STE_MODES)
    def test_two_layers_reach_hard_acc_100(self, ste, seed):
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _ltt_net(2, 2, [16, 16], n_inputs=2, ste=ste)
        acc = _train_until(model, x, y, max_steps=2000)
        assert acc == 1.0, f'XOR({ste}): acc={acc}'


class TestParity3:
    def test_hard_acc_100(self, seed):
        x, y = _bits_dataset(3, lambda b: b[0] ^ b[1] ^ b[2])
        model = _ltt_net(3, 2, [24, 24], n_inputs=3, ste='clip')
        acc = _train_until(model, x, y, max_steps=2000)
        assert acc == 1.0, f'parity3: acc={acc}'
