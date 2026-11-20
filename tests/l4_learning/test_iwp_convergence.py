"""L4: 論文IWP(independent × sin推定器 × Residual Init)の学習収束。

論文 "Light Differentiable Logic Gate Networks" (arXiv:2510.03250) の設定
(推定器 sin01、Residual Initialization)で XOR / 3-parity が
**hard(離散化後) acc = 100%** に到達することを固定する。

比較として estimator='sigmoid' でも同条件で100%に到達することを要求し、
sin導入が既存の学習性能を落としていないことを担保する。
合格基準は緩めない(閾値は 1.0 固定)。
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

ESTIMATORS = ['sigmoid', 'sin']


def _bits_dataset(n_bits, label_fn):
    """全2^n_bits入力とラベルの完全データセット"""
    xs = torch.tensor(list(itertools.product([0.0, 1.0], repeat=n_bits)))
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


def _iwp_mlp(in_dim, k_classes, widths, n_inputs=2, estimator='sigmoid',
             residual_init=True):
    """論文IWP構成のLutLayerスタック(+GroupSum)"""
    torch.manual_seed(0)
    layers = []
    d = in_dim
    for w in widths:
        layers.append(LutLayer(
            d, w, n_inputs=n_inputs, parameterization='independent',
            estimator=estimator, residual_init=residual_init,
        ))
        d = w
    layers.append(GroupSum(k=k_classes, tau=max(1.0, (d // k_classes) ** 0.5)))
    return nn.Sequential(*layers)


class TestXOR:
    @pytest.mark.parametrize('estimator', ESTIMATORS)
    def test_xor_hard_acc_100(self, estimator, seed):
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _iwp_mlp(2, 2, [16, 16], estimator=estimator)
        acc = _train_until(model, x, y, max_steps=1000)
        assert acc == 1.0, f'IWP({estimator}): XOR hard acc={acc}'


class TestParity3:
    @pytest.mark.parametrize('estimator', ESTIMATORS)
    def test_parity3_hard_acc_100(self, estimator, seed):
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        model = _iwp_mlp(3, 2, [32, 32], estimator=estimator)
        acc = _train_until(model, x, y, max_steps=2000)
        assert acc == 1.0, f'IWP({estimator}): parity3 hard acc={acc}'

    @pytest.mark.parametrize('estimator', ESTIMATORS)
    def test_parity3_n4_hard_acc_100(self, estimator, seed):
        """n_inputs=4(2^4エントリのLUT)でも収束すること"""
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        model = _iwp_mlp(3, 2, [32, 32], n_inputs=4, estimator=estimator)
        acc = _train_until(model, x, y, max_steps=2000, lr=0.1)
        assert acc == 1.0, f'IWP({estimator}) n=4: parity3 hard acc={acc}'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'learning'])
