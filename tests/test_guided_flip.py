"""勾配ガイド付きflip修復(analysis/guided_flip)のテスト。

    1. 非減少: 任意のランダム網で厳密hard損失が増えない
    2. 復元: SAT witness網に1ゲート誤り注入 → guided flipが復元
    3. 網羅flipとの整合: guided flipの採択が網羅flip(repair_flip)と
       同等のhard acc改善に到達する(最小系)
"""

import itertools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip('pysat')

from griffith_logic.analysis.guided_flip import (  # noqa: E402
    guided_flip_repair,
    make_classification_metrics,
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


def _make_fns(model, x, y):
    def loss_on_batch(m):
        return F.cross_entropy(m(x), y)

    return loss_on_batch, make_classification_metrics(x, y)


def _hard_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x.round()).argmax(-1) == y).float().mean().item()


class TestNonDecreasing:
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_random_net_loss_never_increases(self, seed):
        fn = lambda b: b[0] ^ b[1] ^ b[2]  # noqa: E731
        torch.manual_seed(seed)
        model = nn.Sequential(
            LutLayer(3, 8, n_inputs=2, parameterization='softmax'),
            LutLayer(8, 8, n_inputs=2, parameterization='softmax'),
            GroupSum(k=2, tau=2.0),
        )
        x, y = _xy(3, fn)
        loss_fn, eval_fn = _make_fns(model, x, y)
        before = eval_fn(model)
        rep = guided_flip_repair(model, loss_fn, eval_fn,
                                 topk=16, passes=3)
        assert rep.metrics_after[0] <= before[0] + 1e-9
        assert rep.metrics_after[1] <= before[1] + 1e-9 or \
            rep.metrics_after[0] < before[0]
        assert eval_fn(model) == rep.metrics_after

    def test_report_counts_consistent(self):
        fn = lambda b: int(sum(b) >= 2)  # noqa: E731
        torch.manual_seed(5)
        model = nn.Sequential(
            LutLayer(3, 8, n_inputs=2, parameterization='softmax'),
            GroupSum(k=2, tau=2.0),
        )
        x, y = _xy(3, fn)
        loss_fn, eval_fn = _make_fns(model, x, y)
        rep = guided_flip_repair(model, loss_fn, eval_fn, topk=16, passes=2)
        assert rep.n_accepted == len(rep.accepted)
        assert rep.n_accepted <= rep.n_proposed


class TestPlantedRecovery:
    def test_recovers_planted_wrong_gate(self):
        fn = lambda b: int(sum(b) >= 2)  # noqa: E731
        topo = LutTopology(n_in=3, layer_widths=[8], readout='groupsum')
        r = can_realize(topo, TargetSpec(fn=fn), wiring='free')
        assert r.status == 'sat'
        model = witness_to_model(r, topo)
        x, y = _xy(3, fn)
        assert _hard_acc(model, x, y) == 1.0

        layer = [m for m in model if isinstance(m, LutLayer)][0]
        with torch.no_grad():
            layer.weights[0].fill_(-10.0)
            layer.weights[0, 0] = 10.0  # FALSEに破壊
        broken_acc = _hard_acc(model, x, y)

        loss_fn, eval_fn = _make_fns(model, x, y)
        guided_flip_repair(model, loss_fn, eval_fn, topk=32, passes=4)
        assert _hard_acc(model, x, y) >= broken_acc
        assert _hard_acc(model, x, y) == 1.0, (
            f'guided flipが復元できない: {broken_acc} -> '
            f'{_hard_acc(model, x, y)}'
        )


class TestMatchesExhaustive:
    @pytest.mark.parametrize('seed', [0, 1, 2, 3])
    def test_reaches_exhaustive_flip_quality(self, seed):
        """maj3停滞セル: guided flipが網羅flipと同等のhard accへ到達。"""
        from griffith_logic.analysis.dontcare_repair import repair_flip

        fn = lambda b: int(sum(b) >= 2)  # noqa: E731
        x, y = _xy(3, fn)

        def train_model():
            torch.manual_seed(seed)
            model = nn.Sequential(
                LutLayer(3, 4, n_inputs=2, parameterization='softmax'),
                GroupSum(k=2, tau=1.4),
            )
            opt = torch.optim.Adam(model.parameters(), lr=0.02)
            for _ in range(2000):
                model.train()
                opt.zero_grad()
                F.cross_entropy(model(x), y).backward()
                opt.step()
            return model

        m1, m2 = train_model(), train_model()
        loss_fn, eval_fn = _make_fns(m1, x, y)
        guided_flip_repair(m1, loss_fn, eval_fn, topk=64, passes=6)
        repair_flip(m2, x, y, passes=3)
        acc_guided, acc_exh = _hard_acc(m1, x, y), _hard_acc(m2, x, y)
        assert acc_guided >= acc_exh - 1e-9, (
            f'guided({acc_guided}) < exhaustive({acc_exh})'
        )
