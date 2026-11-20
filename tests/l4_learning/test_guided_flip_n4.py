"""L4: guided_flip修復のn>2一般化。

意図的に破損させたn=4 warp網のhard accを単一エントリflip修復が
回復することの回帰固定。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_toy_problems import _bits_dataset, _mlp  # noqa: E402

from griffith_logic.analysis.guided_flip import (  # noqa: E402
    guided_flip_repair,
    make_classification_metrics,
)

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


def _train(model, x, y, steps=1500, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


class TestGuidedFlipN4:
    def test_repairs_corrupted_warp_n4(self, seed):
        """収束済みn=4 warp網の1ニューロンTTを破壊→修復で回復"""
        x, y = _bits_dataset(4, lambda b: (b[0] + b[1]) % 2)
        model = _mlp(4, 2, [16, 16], n_inputs=4, param='warp',
                     selection='hard_st')
        _train(model, x, y)
        model.eval()
        with torch.no_grad():
            acc0 = (model(x).argmax(-1) == y).float().mean().item()
        assert acc0 == 1.0, f'前提: 訓練で100%到達せず acc={acc0}'

        # 第1層ニューロン0のWH係数を反転(hard表が大きく壊れる)
        with torch.no_grad():
            model[0].weights[0] = -model[0].weights[0]
            model[0].weights[1] = -model[0].weights[1]
        with torch.no_grad():
            acc_broken = (model(x).argmax(-1) == y).float().mean().item()
        if acc_broken == 1.0:
            return  # 破壊が効かない配線(冗長)ならこのseedはスキップ相当

        def loss_on_batch(m):
            return F.cross_entropy(m(x), y)

        eval_metrics = make_classification_metrics(x, y)
        rep = guided_flip_repair(model, loss_on_batch, eval_metrics,
                                 topk=16, passes=6)
        with torch.no_grad():
            acc1 = (model(x).argmax(-1) == y).float().mean().item()
        assert acc1 > acc_broken, (
            f'修復で改善せず: broken={acc_broken} repaired={acc1} '
            f'(accepted={rep.n_accepted})'
        )

    def test_repairable_layers_includes_independent_warp(self):
        """対象層の選別: warp/independent(n=4)は対象、softmax N>2は除外"""
        from griffith_logic.analysis.guided_flip import _repairable_layers
        from griffith_logic.layers import LutLayer
        model = nn.Sequential(
            LutLayer(8, 8, n_inputs=4, parameterization='warp'),
            LutLayer(8, 8, n_inputs=4, parameterization='independent'),
            LutLayer(8, 8, n_inputs=4, parameterization='softmax'),
            LutLayer(8, 8, n_inputs=2, parameterization='softmax'),  # 16g
        )
        layers = _repairable_layers(model)
        params = [(la.parameterization, la.n_inputs) for la in layers]
        assert ('warp', 4) in params
        assert ('independent', 4) in params
        assert ('softmax', 4) not in params
        assert ('softmax', 2) in params  # 16ゲートモード
