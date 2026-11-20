"""L2: Gumbel-ST(selection='gumbel_st')の勾配・forward契約。

gumbel_stは「weights + gumbel_scale·G へのHard-ST」と厳密等価という
設計なので、契約は:
    1. 固定ノイズGに対し forward = argmax(w+G)のhard関数
    2. 固定ノイズGに対し backward = softmax((w+G)/τ_b)版soft関数の勾配
    3. gumbel_scale=0 の層出力はhard_stとビット厳密一致
    4. 層のノイズ生成はseed再現可能で、手動合成と一致
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD  # noqa: E402

from griffith_logic.core.reference import (  # noqa: E402
    reference_basis,
    reference_gather,
)
from griffith_logic.lut_layer import LutFunctionHardST, LutLayer  # noqa: E402


def _case(seed=13):
    torch.manual_seed(seed)
    in_dim, out_dim, batch = 8, 4, 3
    layer = LutLayer(in_dim, out_dim, n_inputs=2, parameterization='softmax')
    gate_table = layer.gate_table.double()
    w = torch.randn(out_dim, 16, dtype=torch.float64)
    u = torch.rand(out_dim, 16, dtype=torch.float64)
    g = -torch.log((-torch.log(u + 1e-20)).clamp_min(1e-20))
    x = torch.rand(batch, in_dim, dtype=torch.float64) * 0.8 + 0.1
    return x, w, g, layer.indices, gate_table


def _soft_reference(x, w, indices, gate_table, tau_b):
    gathered = reference_gather(x, indices)
    basis = reference_basis(gathered, 2)
    s = F.softmax(w / tau_b, dim=-1)
    return (basis * (s @ gate_table).unsqueeze(0)).sum(dim=-1)


class TestGumbelSTContracts:
    def test_forward_is_gumbel_argmax_hard(self):
        """forward = argmax(w+G)で選んだhard関数(手計算一致)。"""
        x, w, g, indices, gate_table = _case()
        y = LutFunctionHardST.apply(x, w + g, indices, 2, gate_table, 1.0)

        gathered = reference_gather(x, indices)
        basis = reference_basis(gathered, 2)
        w_hard = gate_table[(w + g).argmax(dim=-1)]
        y_expected = (basis * w_hard.unsqueeze(0)).sum(dim=-1)
        assert torch.allclose(y, y_expected, atol=TOL_F64_EQ, rtol=0)

    @pytest.mark.parametrize('tau_b', [1.0, 0.5, 2.0])
    def test_backward_equals_tempered_soft_at_noisy_logits(self, tau_b):
        """backward = softmax((w+G)/τ_b)版soft関数のautograd勾配と一致。"""
        x, w, g, indices, gate_table = _case()
        torch.manual_seed(19)
        grad_output = torch.randn(x.shape[0], w.shape[0], dtype=torch.float64)

        xr = x.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        y = LutFunctionHardST.apply(xr, wr + g, indices, 2, gate_table, tau_b)
        gx, gw = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)

        xs = x.clone().requires_grad_(True)
        ws = w.clone().requires_grad_(True)
        y_soft = _soft_reference(xs, ws + g, indices, gate_table, tau_b)
        gx_ref, gw_ref = torch.autograd.grad(
            y_soft, (xs, ws), grad_outputs=grad_output,
        )
        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0)

    def test_zero_noise_equals_hard_st(self):
        """gumbel_scale=0 の層はhard_st層とビット厳密に一致。"""
        torch.manual_seed(7)
        layer_g = LutLayer(8, 6, n_inputs=2, parameterization='softmax',
                           selection='gumbel_st')
        layer_g.gumbel_scale = 0.0
        layer_h = LutLayer(8, 6, n_inputs=2, parameterization='softmax',
                           selection='hard_st')
        with torch.no_grad():
            layer_h.weights.copy_(layer_g.weights)
            layer_h.indices.copy_(layer_g.indices)

        x = torch.rand(5, 8)
        layer_g.train()
        layer_h.train()
        assert torch.equal(layer_g(x), layer_h(x))

    def test_layer_noise_is_seed_reproducible(self):
        """層内のノイズ生成が手動合成(同一seed)と一致する。"""
        torch.manual_seed(3)
        layer = LutLayer(8, 4, n_inputs=2, parameterization='softmax',
                         selection='gumbel_st')
        x = torch.rand(5, 8)

        layer.train()
        torch.manual_seed(42)
        y1 = layer(x)

        torch.manual_seed(42)
        u = torch.rand_like(layer.weights)
        g = -torch.log((-torch.log(u + 1e-20)).clamp_min(1e-20))
        y2 = LutFunctionHardST.apply(
            x, layer.weights + layer.gumbel_scale * g,
            layer.indices, 2, layer.gate_table, layer.tau_b,
        )
        assert torch.equal(y1, y2)
