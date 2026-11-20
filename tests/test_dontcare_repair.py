"""don't-care修復パス(analysis/dontcare_repair)のテスト。

    1. 完全な網(hard acc 1.0)にはno-op(精度非減少・変更なしor無害)
    2. SAT witness網に1ゲート誤りを注入 → repair_flipが復元
    3. 全修復で訓練hard精度が非減少
    4. repair_output_layer_sat: 出力層破壊からの復元
"""

import itertools

import pytest
import torch
import torch.nn as nn

pytest.importorskip('pysat')

from griffith_logic.analysis.dontcare_repair import (  # noqa: E402
    repair_flip,
    repair_output_layer_sat,
    repair_refit,
)
from griffith_logic.analysis.sat_capacity import (  # noqa: E402
    LutTopology,
    TargetSpec,
    can_realize,
    witness_to_model,
)
from griffith_logic.group_sum import GroupSum  # noqa: E402
from griffith_logic.layers import LutLayer  # noqa: E402


def _xy(n_in, fn):
    x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=n_in)))
    y = torch.tensor([fn(tuple(int(v) for v in r)) for r in x.tolist()])
    return x, y


def _perfect_model_and_data():
    """SAT witnessからhard acc 1.0のmaj3モデルを作る。"""
    fn = lambda b: int(sum(b) >= 2)  # noqa: E731
    topo = LutTopology(n_in=3, layer_widths=[8], readout='groupsum')
    r = can_realize(topo, TargetSpec(fn=fn), wiring='free')
    assert r.status == 'sat'
    model = witness_to_model(r, topo)
    x, y = _xy(3, fn)
    return model, x, y


def _hard_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x.round()).argmax(-1) == y).float().mean().item()


class TestNoOpOnPerfect:
    @pytest.mark.parametrize('repair', [repair_refit, repair_flip])
    def test_perfect_net_stays_perfect(self, repair):
        model, x, y = _perfect_model_and_data()
        assert _hard_acc(model, x, y) == 1.0
        rep = repair(model, x, y)
        assert rep.acc_before == 1.0
        assert rep.acc_after == 1.0
        assert _hard_acc(model, x, y) == 1.0


class TestPlantedErrorRecovery:
    def test_flip_recovers_planted_wrong_gate(self):
        model, x, y = _perfect_model_and_data()
        layer = [m for m in model if isinstance(m, LutLayer)][0]
        # ニューロン0のゲートを破壊(定数FALSEに)
        with torch.no_grad():
            layer.weights[0].fill_(-10.0)
            layer.weights[0, 0] = 10.0  # gate 0 = FALSE
        broken = _hard_acc(model, x, y)
        rep = repair_flip(model, x, y)
        assert rep.acc_after >= broken
        assert rep.acc_after == 1.0, (
            f'flip修復が復元できない: {broken} -> {rep.acc_after}'
        )

    def test_output_layer_sat_recovers(self):
        model, x, y = _perfect_model_and_data()
        layer = [m for m in model if isinstance(m, LutLayer)][0]
        with torch.no_grad():
            layer.weights[0].fill_(-10.0)
            layer.weights[0, 0] = 10.0
        broken = _hard_acc(model, x, y)
        rep = repair_output_layer_sat(model, x, y)
        assert rep.acc_after == 1.0, (
            f'SAT出力層修復が復元できない: {broken} -> {rep.acc_after}'
        )


class TestNonDecreasing:
    @pytest.mark.parametrize('repair',
                             [repair_refit, repair_flip,
                              repair_output_layer_sat])
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_never_decreases_train_hard_acc(self, repair, seed):
        """未収束のランダム網に対しても訓練hard精度は非減少。"""
        fn = lambda b: b[0] ^ b[1] ^ b[2]  # noqa: E731
        torch.manual_seed(seed)
        model = nn.Sequential(
            LutLayer(3, 8, n_inputs=2, parameterization='softmax'),
            LutLayer(8, 8, n_inputs=2, parameterization='softmax'),
            GroupSum(k=2, tau=2.0),
        )
        x, y = _xy(3, fn)
        before = _hard_acc(model, x, y)
        rep = repair(model, x, y)
        after = _hard_acc(model, x, y)
        assert after >= before - 1e-9, (repair.__name__, before, after)
        assert abs(rep.acc_after - after) < 1e-9
