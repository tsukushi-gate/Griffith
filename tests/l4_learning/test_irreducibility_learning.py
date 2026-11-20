"""L4: 縮約耐性正則化(先行実験 S4)を学習ループへ足したときの挙動。

- 損失に加算して backward が流れる（手本 tests/l4_learning/test_dwn_learning.py）
- toy 課題で自明ゲート（定数・pass-through）の確率質量が下がる
- λ=0 では学習が正則化なしと bitwise 不変
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.group_sum import GroupSum
from griffith_logic.layers.lut import LutLayer
from griffith_logic.regularization.apply import (
    IRREDUCIBLE_HIDDEN_GATE_IDS,
    irreducibility_penalty,
    iter_irreducibility_layers,
)


def _bits_dataset(n_bits, fn):
    xs = torch.tensor(
        [[(i >> b) & 1 for b in range(n_bits)] for i in range(2 ** n_bits)],
        dtype=torch.float32,
    )
    ys = torch.tensor([fn([int(v) for v in row]) for row in xs],
                      dtype=torch.long)
    return xs, ys


def _model(in_dim, classes, widths, seed=0):
    torch.manual_seed(seed)
    layers = [torch.nn.Flatten()]
    d = in_dim
    for w in widths:
        layers.append(LutLayer(d, w, n_inputs=2, selection='hard_st'))
        d = w
    layers.append(GroupSum(classes, tau=1.0))
    return torch.nn.Sequential(*layers)


def _trivial_mass(model):
    """全対象層の平均自明ゲート確率質量（最終層も隠れ層と同じ基準で測る）"""
    total, n = 0.0, 0
    for layer in iter_irreducibility_layers(model):
        p = F.softmax(layer.weights.detach(), dim=-1)
        total += float(p[:, list(IRREDUCIBLE_HIDDEN_GATE_IDS)].sum())
        n += layer.out_dim
    return total / n


def _train(model, x, y, steps=300, lam=0.0, lr=0.05, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        out = model(x)
        loss = F.cross_entropy(out, y)
        if lam != 0.0:
            loss = loss + lam * irreducibility_penalty(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(-1) == y).float().mean().item()
    return acc


class TestBackwardFlow:
    def test_penalty_backward(self):
        """penalty を CE に加算して backward が流れる（勾配が有限・非ゼロ）"""
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _model(2, 2, (8,))
        loss = F.cross_entropy(model(x), y) + 0.1 * irreducibility_penalty(model)
        loss.backward()
        for layer in iter_irreducibility_layers(model):
            assert layer.weights.grad is not None
            assert torch.isfinite(layer.weights.grad).all()
            assert layer.weights.grad.abs().max() > 0


class TestTrivialMassDecreases:
    def test_penalty_reduces_trivial_gate_mass(self):
        """同一 seed・同一データで λ>0 の方が自明ゲート質量が小さくなる"""
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        plain = _model(3, 2, (16, 16), seed=1)
        regged = _model(3, 2, (16, 16), seed=1)
        assert abs(_trivial_mass(plain) - _trivial_mass(regged)) < 1e-12

        _train(plain, x, y, steps=300, lam=0.0)
        _train(regged, x, y, steps=300, lam=1.0)

        m_plain = _trivial_mass(plain)
        m_reg = _trivial_mass(regged)
        assert m_reg < m_plain, (m_plain, m_reg)

    def test_penalty_value_itself_drops(self):
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        model = _model(3, 2, (16, 16), seed=2)
        before = float(irreducibility_penalty(model).detach())
        _train(model, x, y, steps=300, lam=1.0)
        after = float(irreducibility_penalty(model).detach())
        assert after < before, (before, after)


class TestLambdaZeroIsInert:
    def test_lambda_zero_matches_no_regularization_bitwise(self):
        """λ=0 で損失に足さない実装（runner と同じ分岐）なら bitwise 不変"""
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        a = _model(3, 2, (16, 16), seed=3)
        b = _model(3, 2, (16, 16), seed=3)
        acc_a = _train(a, x, y, steps=100, lam=0.0, seed=5)
        acc_b = _train(b, x, y, steps=100, lam=0.0, seed=5)
        assert acc_a == acc_b
        for la, lb in zip(iter_irreducibility_layers(a),
                          iter_irreducibility_layers(b), strict=True):
            assert torch.equal(la.weights, lb.weights)


class TestStillLearns:
    @pytest.mark.parametrize('lam', [0.0, 0.01])
    def test_xor_reaches_hard_acc_100(self, lam):
        """正則化を足しても XOR は hard acc 100% に到達する"""
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _model(2, 2, (16, 16), seed=4)
        acc = _train(model, x, y, steps=600, lam=lam, lr=0.1)
        assert acc == 1.0, f'XOR(λ={lam}): hard acc={acc}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
