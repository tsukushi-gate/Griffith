"""L4: トイ問題の学習収束スモーク(BP)。

「確実に学習できる」ことを回帰指標として固定する。
合格基準は訓練acc 100% **かつ eval(離散化)でも100%**。
soft学習がhard回路に正しく崩壊することまで含めて
「論文どおり学習できた」と定義する。
"""

import itertools
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


def _bits_dataset(n_bits, label_fn):
    """全2^n_bits入力とラベルの完全データセット"""
    xs = torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=n_bits))
    )
    ys = torch.tensor([label_fn(row) for row in xs.long().tolist()])
    return xs, ys


def _train_until(model, x, y, *, lr=0.05, max_steps=2000, check_every=50):
    """soft訓練し、hard evalで100%になったら早期終了。最終hard accを返す"""
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


def _mlp(in_dim, k_classes, widths, n_inputs=2, param='softmax', selection='soft'):
    torch.manual_seed(0)
    layers = []
    d = in_dim
    for w in widths:
        layers.append(LutLayer(
            d, w, n_inputs=n_inputs,
            parameterization=param, selection=selection,
        ))
        d = w
    layers.append(GroupSum(k=k_classes, tau=max(1.0, (d // k_classes) ** 0.5)))
    return nn.Sequential(*layers)


PARAMS = ['softmax', 'independent', 'iwp', 'warp']


class TestXOR:
    @pytest.mark.parametrize('param', PARAMS)
    def test_xor_soft_and_hard(self, param, seed):
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _mlp(2, 2, [16, 16], param=param)
        acc = _train_until(model, x, y, max_steps=1000)
        assert acc == 1.0, f'{param}: XOR hard acc={acc}'

    def test_xor_hard_st(self, seed):
        """CAGE方式(hard_st)でもXORが学習でき、selection gapがない"""
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _mlp(2, 2, [16, 16], param='softmax', selection='hard_st')
        acc = _train_until(model, x, y, max_steps=1500)
        assert acc == 1.0, f'hard_st: XOR hard acc={acc}'


class TestParity:
    @pytest.mark.parametrize('n_inputs', [2, 4])
    def test_parity3(self, n_inputs, seed):
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        model = _mlp(3, 2, [32, 32], n_inputs=n_inputs, param='softmax')
        acc = _train_until(model, x, y, max_steps=2000)
        assert acc == 1.0, f'parity3 n={n_inputs}: hard acc={acc}'

    def test_parity4_independent(self, seed):
        x, y = _bits_dataset(4, lambda b: sum(b) % 2)
        model = _mlp(4, 2, [64, 64], n_inputs=4, param='independent')
        acc = _train_until(model, x, y, max_steps=2000, lr=0.1)
        assert acc == 1.0, f'parity4: hard acc={acc}'


class TestFullAdder:
    def test_full_adder_as_4class(self, seed):
        """全加算器: (a,b,cin) → (carry,sum) を4クラス分類として学習"""
        def label(b):
            total = b[0] + b[1] + b[2]
            return (total // 2) * 2 + (total % 2)  # carry*2 + sum

        x, y = _bits_dataset(3, label)
        model = _mlp(3, 4, [64, 64], param='softmax')
        acc = _train_until(model, x, y, max_steps=2000)
        assert acc == 1.0, f'full adder: hard acc={acc}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
