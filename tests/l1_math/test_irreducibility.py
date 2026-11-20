"""L1: 縮約耐性正則化 irreducibility_penalty(先行実験 S4)の数学的性質。

検証する数式:
    R = (1/N_h) Σ_n Σ_i p_{n,i} · c_i
    p_{n,i} = softmax(weights + gate_logit_mask)（訓練時 forward と同じ温度 1.0）
    c は16次元の0/1。隠れ層 = {0,15,3,5,10,12}、最終LutLayer = {3,5,10,12}
    N_h = 対象層の全ゲート数（Σ out_dim）

手本: tests/l1_math/test_spectral.py（素朴ループとの厳密照合 + 微分可能性）
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.group_sum import GroupSum
from griffith_logic.layers.lut import LutLayer
from griffith_logic.regularization.apply import (
    IRREDUCIBLE_COST_FINAL,
    IRREDUCIBLE_COST_HIDDEN,
    IRREDUCIBLE_FINAL_GATE_IDS,
    IRREDUCIBLE_HIDDEN_GATE_IDS,
    gate_cost_vector,
    irreducibility_penalty,
    iter_irreducibility_layers,
)

HIDDEN_MASK = (0, 15, 3, 5, 10, 12)
NONTRIVIAL = tuple(g for g in range(16) if g not in HIDDEN_MASK)


def _model(widths=(6, 4), in_dim=8, seed=0, dtype=torch.float64, **kw):
    torch.manual_seed(seed)
    layers = [torch.nn.Flatten()]
    d = in_dim
    for w in widths:
        layers.append(LutLayer(d, w, n_inputs=2, **kw))
        d = w
    layers.append(GroupSum(2, tau=1.0))
    return torch.nn.Sequential(*layers).to(dtype)


def _val(pen: torch.Tensor) -> float:
    return float(pen.detach())


def _naive_penalty(model, cost_hidden, cost_final):
    """素朴な四重ループでの直接計算(ground truth)"""
    luts = [m for m in model.modules() if isinstance(m, LutLayer)]
    total = 0.0
    n_gates = 0
    for li, layer in enumerate(luts):
        cost = cost_final if li == len(luts) - 1 else cost_hidden
        w = layer._masked_weights().detach()
        for n in range(layer.out_dim):
            row = w[n]
            mx = row.max()
            exps = [float(torch.exp(row[i] - mx)) for i in range(16)]
            z = sum(exps)
            for i in range(16):
                total += (exps[i] / z) * float(cost[i])
        n_gates += layer.out_dim
    return total / n_gates


class TestCostVectors:
    def test_hidden_cost_marks_constants_and_passthroughs(self):
        assert IRREDUCIBLE_HIDDEN_GATE_IDS == (0, 15, 3, 5, 10, 12)
        assert IRREDUCIBLE_COST_HIDDEN == gate_cost_vector((0, 15, 3, 5, 10, 12))
        assert sum(IRREDUCIBLE_COST_HIDDEN) == 6.0
        for g in range(16):
            assert IRREDUCIBLE_COST_HIDDEN[g] == float(
                g in IRREDUCIBLE_HIDDEN_GATE_IDS)

    def test_final_cost_keeps_constants(self):
        assert IRREDUCIBLE_FINAL_GATE_IDS == (3, 5, 10, 12)
        assert sum(IRREDUCIBLE_COST_FINAL) == 4.0
        assert IRREDUCIBLE_COST_FINAL[0] == 0.0
        assert IRREDUCIBLE_COST_FINAL[15] == 0.0

    def test_gate_cost_vector_rejects_bad_ids(self):
        with pytest.raises(ValueError, match='0〜15'):
            gate_cost_vector([16])


class TestNaiveAgreement:
    @pytest.mark.parametrize('widths', [(6, 4), (8,), (5, 7, 4)])
    def test_matches_naive_loop(self, widths):
        model = _model(widths=widths, seed=1)
        got = _val(irreducibility_penalty(model))
        expected = _naive_penalty(
            model, IRREDUCIBLE_COST_HIDDEN, IRREDUCIBLE_COST_FINAL)
        assert abs(got - expected) < TOL_F64_EQ * 100, (
            float(got), expected
        )

    def test_matches_hand_computation_on_fixed_weights(self):
        """手計算: 2層(out=2, out=2)、ロジットを陽に与えて閉形式と照合"""
        torch.manual_seed(0)
        l0 = LutLayer(4, 2, n_inputs=2).double()
        l1 = LutLayer(2, 2, n_inputs=2).double()
        with torch.no_grad():
            l0.weights.zero_()
            l0.weights[:, 3] = torch.log(torch.tensor(3.0, dtype=torch.float64))
            l1.weights.zero_()
            l1.weights[:, 0] = torch.log(torch.tensor(2.0, dtype=torch.float64))
        model = torch.nn.Sequential(
            torch.nn.Flatten(), l0, l1, GroupSum(2, tau=1.0))
        # 層0: exp = [1]*15 + [3] at idx3 → Z = 18。自明6種の質量 = (3 + 5)/18
        p0 = (3.0 + 5.0) / 18.0
        # 層1(最終): idx0 に重み2 → Z = 17。最終コストは {3,5,10,12} の4種 = 4/17
        p1 = 4.0 / 17.0
        expected = (2 * p0 + 2 * p1) / 4
        got = _val(irreducibility_penalty(model))
        assert abs(got - expected) < TOL_F64_EQ * 10, (got, expected)


class TestBounds:
    def test_all_mass_on_trivial_gives_one(self):
        model = _model(widths=(4, 4), seed=2)
        for layer in iter_irreducibility_layers(model):
            with torch.no_grad():
                layer.weights.fill_(-50.0)
                layer.weights[:, 3] = 50.0   # op3 は隠れ層・最終層とも c=1
        assert abs(_val(irreducibility_penalty(model)) - 1.0) < 1e-9

    def test_all_mass_on_nontrivial_gives_zero(self):
        model = _model(widths=(4, 4), seed=3)
        for layer in iter_irreducibility_layers(model):
            with torch.no_grad():
                layer.weights.fill_(-50.0)
                layer.weights[:, 6] = 50.0   # XOR はどちらの c も 0
        assert _val(irreducibility_penalty(model)) < 1e-9

    def test_final_layer_constants_are_free(self):
        """最終層で定数(op0/op15)に全質量を置いてもペナルティは 0 になる"""
        torch.manual_seed(4)
        l0 = LutLayer(4, 4, n_inputs=2).double()
        l1 = LutLayer(4, 4, n_inputs=2).double()
        with torch.no_grad():
            for layer in (l0, l1):
                layer.weights.fill_(-50.0)
                layer.weights[:, 0] = 50.0
        model = torch.nn.Sequential(
            torch.nn.Flatten(), l0, l1, GroupSum(2, tau=1.0))
        # 隠れ層(l0)は op0 が c=1、最終層(l1)は c=0 → (4·1 + 4·0)/8 = 0.5
        assert abs(_val(irreducibility_penalty(model)) - 0.5) < 1e-9

    def test_value_is_in_unit_interval(self):
        for seed in range(5):
            model = _model(widths=(6, 6), seed=seed)
            v = _val(irreducibility_penalty(model))
            assert 0.0 <= v <= 1.0, v


class TestGateMaskInteraction:
    def test_masked_positions_contribute_zero(self):
        """gate_mask で全ての自明ゲートを除外すればペナルティは厳密に 0"""
        model = _model(widths=(6, 4), seed=5, gate_mask=HIDDEN_MASK)
        assert _val(irreducibility_penalty(model)) == 0.0

    def test_masked_final_layer_only(self):
        """最終層だけ pass-through を除外 → 最終層の寄与が 0 になる"""
        torch.manual_seed(6)
        l0 = LutLayer(8, 4, n_inputs=2).double()
        l1 = LutLayer(4, 4, n_inputs=2, gate_mask=(3, 5, 10, 12)).double()
        model = torch.nn.Sequential(
            torch.nn.Flatten(), l0, l1, GroupSum(2, tau=1.0))
        p = F.softmax(l0._masked_weights(), dim=-1)
        hidden_mass = float(
            p[:, list(IRREDUCIBLE_HIDDEN_GATE_IDS)].sum().detach())
        assert abs(_val(irreducibility_penalty(model))
                   - hidden_mass / 8) < TOL_F64_EQ * 100

    def test_uses_masked_softmax_not_raw(self):
        """マスク後の再正規化された p を使っている(生の softmax ではない)"""
        torch.manual_seed(7)
        masked = _model(widths=(4, 4), seed=7, gate_mask=(0, 15))
        torch.manual_seed(7)
        plain = _model(widths=(4, 4), seed=7)
        assert _val(irreducibility_penalty(masked)) != _val(
            irreducibility_penalty(plain))


class TestDifferentiability:
    def test_gradient_flows_to_all_layers(self):
        model = _model(widths=(6, 4), seed=8)
        pen = irreducibility_penalty(model)
        pen.backward()
        for layer in iter_irreducibility_layers(model):
            assert layer.weights.grad is not None
            assert torch.isfinite(layer.weights.grad).all()
            assert layer.weights.grad.abs().max() > 0

    def test_masked_gradient_is_exactly_zero(self):
        model = _model(widths=(6, 4), seed=9, gate_mask=HIDDEN_MASK)
        irreducibility_penalty(model).backward()
        for layer in iter_irreducibility_layers(model):
            g = layer.weights.grad[:, list(HIDDEN_MASK)]
            assert torch.equal(g, torch.zeros_like(g))

    def test_gradient_pushes_mass_away_from_trivial_gates(self):
        """1ステップの勾配降下でペナルティが減る"""
        model = _model(widths=(6, 4), seed=10)
        before = _val(irreducibility_penalty(model))
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        opt.zero_grad()
        irreducibility_penalty(model).backward()
        opt.step()
        after = _val(irreducibility_penalty(model))
        assert after < before, (before, after)


class TestCustomCosts:
    def test_custom_cost_vector(self):
        model = _model(widths=(4, 4), seed=11)
        cost = gate_cost_vector([6])
        got = _val(irreducibility_penalty(model, cost, cost))
        expected = _naive_penalty(model, cost, cost)
        assert abs(got - expected) < TOL_F64_EQ * 100

    def test_zero_cost_gives_zero(self):
        model = _model(widths=(4, 4), seed=12)
        zero = tuple([0.0] * 16)
        assert _val(irreducibility_penalty(model, zero, zero)) == 0.0

    def test_wrong_length_cost_raises(self):
        model = _model(widths=(4, 4), seed=13)
        with pytest.raises(ValueError, match='16次元'):
            irreducibility_penalty(model, [1.0, 0.0], IRREDUCIBLE_COST_FINAL)


class TestValidation:
    def test_no_lut_layer_raises(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), GroupSum(2, tau=1.0))
        with pytest.raises(ValueError, match='対象 LutLayer'):
            irreducibility_penalty(model)

    @pytest.mark.parametrize(
        ('n_inputs', 'parameterization'),
        [(4, 'softmax'), (2, 'independent'), (2, 'iwp'), (2, 'warp')],
    )
    def test_non_16gate_raises(self, n_inputs, parameterization):
        layer = LutLayer(8, 4, n_inputs=n_inputs,
                         parameterization=parameterization)
        model = torch.nn.Sequential(torch.nn.Flatten(), layer)
        with pytest.raises(ValueError, match='16ゲートモード'):
            irreducibility_penalty(model)

    def test_layer_order_is_module_order(self):
        model = _model(widths=(3, 5, 7), seed=14)
        luts = iter_irreducibility_layers(model)
        assert [layer.out_dim for layer in luts] == [3, 5, 7]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
