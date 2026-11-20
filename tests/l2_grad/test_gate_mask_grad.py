"""L2: ゲート種ロジットマスク(先行実験 S1)の勾配契約。

forward が argmax を含む(hard_st / gumbel_st)ため gradcheck は使えない。
代わりに契約テスト方式で以下を検証する:
    - マスク位置の ∂L/∂w が **厳密に 0**(−inf 加算 → softmax確率0 → ヤコビアン0)
    - 出力・grad_x・grad_w がすべて有限(−inf が NaN を生まない)
    - マスク無し層と「マスク位置のロジットを −1e9 に置いた層」の勾配が
      許可位置で一致する(マスクが確率質量の再配分としてのみ効く)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F32_EQ, assert_all_finite

from griffith_logic.layers.lut import LutLayer

HIDDEN_MASK = (0, 15, 3, 5, 10, 12)
SELECTIONS = ['soft', 'hard_st', 'gumbel_st']


def _run(layer, x, seed=0):
    torch.manual_seed(seed)
    x = x.clone().requires_grad_(True)
    y = layer(x)
    y.sum().backward()
    return y, x.grad, layer.weights.grad


class TestMaskedGradientIsZero:
    @pytest.mark.parametrize('selection', SELECTIONS)
    def test_masked_positions_have_exact_zero_grad(self, selection):
        torch.manual_seed(0)
        layer = LutLayer(16, 32, n_inputs=2, selection=selection,
                         gate_mask=HIDDEN_MASK)
        layer.train()
        x = torch.rand(8, 16)
        y, gx, gw = _run(layer, x)
        assert_all_finite(y, f'{selection}: forward')
        assert_all_finite(gx, f'{selection}: grad_x')
        assert_all_finite(gw, f'{selection}: grad_w')
        masked = gw[:, list(HIDDEN_MASK)]
        assert torch.equal(masked, torch.zeros_like(masked)), (
            f'{selection}: マスク位置の勾配が0でない(max|g|='
            f'{masked.abs().max().item()})'
        )

    @pytest.mark.parametrize('selection', SELECTIONS)
    def test_allowed_positions_have_nonzero_grad(self, selection):
        """対照: 許可位置には勾配が流れる(勾配が全部0で通っていない確認)"""
        torch.manual_seed(0)
        layer = LutLayer(16, 32, n_inputs=2, selection=selection,
                         gate_mask=HIDDEN_MASK)
        layer.train()
        _, _, gw = _run(layer, torch.rand(8, 16))
        allowed = gw[:, list(layer.allowed_gate_ids())]
        assert allowed.abs().max() > 0


class TestMaskEquivalentToLargeNegativeLogit:
    @pytest.mark.parametrize('selection', ['soft', 'hard_st'])
    def test_matches_minus_1e9_logits(self, selection):
        """マスクは「除外ロジット −1e9」の極限と数値的に一致する"""
        torch.manual_seed(5)
        masked = LutLayer(16, 24, n_inputs=2, selection=selection,
                          gate_mask=HIDDEN_MASK)
        torch.manual_seed(5)
        manual = LutLayer(16, 24, n_inputs=2, selection=selection)
        with torch.no_grad():
            manual.weights[:, list(HIDDEN_MASK)] = -1e9
        masked.train()
        manual.train()
        x = torch.rand(6, 16)
        y0, gx0, gw0 = _run(masked, x)
        y1, gx1, gw1 = _run(manual, x)
        assert torch.allclose(y0, y1, atol=TOL_F32_EQ, rtol=0)
        assert torch.allclose(gx0, gx1, atol=TOL_F32_EQ, rtol=0)
        allowed = list(masked.allowed_gate_ids())
        assert torch.allclose(gw0[:, allowed], gw1[:, allowed],
                              atol=TOL_F32_EQ, rtol=0)


class TestNoNaNWithSingleAllowedGate:
    def test_one_allowed_gate_is_finite(self):
        """許可が1種だけ(15種除外)でも NaN/inf を出さない"""
        torch.manual_seed(9)
        mask = [g for g in range(16) if g != 6]
        layer = LutLayer(8, 12, n_inputs=2, selection='hard_st', gate_mask=mask)
        layer.train()
        y, gx, gw = _run(layer, torch.rand(4, 8))
        assert_all_finite(y, 'single-gate forward')
        assert_all_finite(gx, 'single-gate grad_x')
        assert_all_finite(gw, 'single-gate grad_w')
        # 全確率が1種に集中 → 勾配は厳密に0(softmaxヤコビアン p(1−p)=0)
        assert torch.equal(gw, torch.zeros_like(gw))
        layer.eval()
        with torch.no_grad():
            assert set(layer._masked_weights().argmax(-1).tolist()) == {6}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
