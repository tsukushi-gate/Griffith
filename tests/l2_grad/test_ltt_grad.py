"""L2: LTT(線形閾値ゲート)の勾配。

STE 学習経路(値 1[z > 0]、勾配 grad_output · surrogate(z) を w / b / x へ)を
core/reference.py の reference_ltt_grads(カスタム Function を使わない素朴実装)と
float64 で照合する(出力 1e-12、勾配 1e-10)。soft 経路 σ(z/s) は gradcheck。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.core.reference import (
    reference_ltt_forward,
    reference_ltt_grads,
    reference_ltt_pre_activation,
)
from griffith_logic.layers.ltt import LTTLayer

SCALE = 0.7


def _case(n, mode, seed=0, boundary=False):
    torch.manual_seed(seed)
    layer = LTTLayer(10, 7, n_inputs=n, ste=mode, ste_scale=SCALE, clamp=False).double()
    x = torch.randint(0, 2, (9, 10)).double()
    if boundary:
        # z = 0.5·(1 の個数) − 0.5 → 入力がちょうど 1 本だけ 1 のとき z = 0(境界)
        with torch.no_grad():
            layer.w.fill_(0.5)
            layer.b.fill_(-0.5)
    return layer, x


class TestSteMatchesReference:
    @pytest.mark.parametrize('boundary', [False, True], ids=['interior', 'boundary'])
    @pytest.mark.parametrize('mode', ['identity', 'clip', 'sigmoid'])
    @pytest.mark.parametrize('n', [2, 3, 6])
    def test_value_and_grads(self, n, mode, boundary):
        layer, x = _case(n, mode, boundary=boundary)
        layer.train()
        xr = x.clone().requires_grad_(True)
        y = layer(xr)
        torch.manual_seed(7)
        g = torch.randn_like(y)
        gx, gw, gb = torch.autograd.grad(y, (xr, layer.w, layer.b), grad_outputs=g)

        y_ref, gx_ref, gw_ref, gb_ref = reference_ltt_grads(
            x, layer.w, layer.b, layer.indices, g, mode, SCALE)
        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)
        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(gb, gb_ref, atol=TOL_F64_GRAD, rtol=0)

    def test_boundary_z_zero_outputs_zero(self):
        """1[z > 0] は厳密な不等号: z = 0 では 0(層と参照で一致)"""
        layer, x = _case(2, 'clip', boundary=True)
        z = reference_ltt_pre_activation(x, layer.w, layer.b, layer.indices)
        assert bool((z == 0).any())
        y = layer.eval()(x)
        assert bool((y[z == 0] == 0).all())
        assert torch.equal(y, reference_ltt_forward(x, layer.w, layer.b, layer.indices))

    def test_clip_blocks_gradient_outside_window(self):
        layer, x = _case(3, 'clip')
        layer.train()
        with torch.no_grad():
            layer.b.fill_(3.0)          # |z| ≥ 3 − 3·1 … 全て窓 |z| < 0.7 の外
            layer.w.fill_(1.0)
        y = layer(x)
        gw, gb = torch.autograd.grad(y.sum(), (layer.w, layer.b))
        assert float(gw.abs().max()) == 0.0
        assert float(gb.abs().max()) == 0.0


class TestSoftPath:
    def test_soft_forward_matches_reference_and_gradcheck(self):
        layer, x = _case(3, 'clip')
        layer.eval()
        soft = layer.soft_forward(x)
        ref = reference_ltt_forward(x, layer.w, layer.b, layer.indices, hard=False, scale=SCALE)
        assert torch.allclose(soft, ref, atol=TOL_F64_EQ, rtol=0)

        idx = layer.indices

        def f(xx, ww, bb):
            return reference_ltt_forward(xx, ww, bb, idx, hard=False, scale=SCALE)

        xr = (x + 0.1 * torch.rand_like(x)).requires_grad_(True)   # 内点(0/1 からずらす)
        wr = layer.w.detach().clone().requires_grad_(True)
        br = layer.b.detach().clone().requires_grad_(True)
        assert torch.autograd.gradcheck(f, (xr, wr, br), **GRADCHECK_KW)
