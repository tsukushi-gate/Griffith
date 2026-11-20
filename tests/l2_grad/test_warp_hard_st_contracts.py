"""L2: WARP用Hard-ST/Gumbel-ST(LutFunctionWARPHardST)の契約テスト。

発見したバグの修正: 従来はwarp×hard_stがLutFunctionHardST
(argmax over WH係数=単一minterm選択)へ誤ってdispatchされ、
warpの離散化 1[θ·φ(s)>0] と全く異なる関数を計算していた。

forward契約:  hard真理値表 w=1[θH+noise>0] の多重線形拡張
              (binary入力でeval離散化と厳密一致 = selection gap 0)
backward契約: σ((θH+noise)/τ_b) soft関数の純autograd勾配と厳密一致
degeneracy:   noise=None(hard_st) == noise=0のgumbel_st
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.core import (
    reference_lut_warp_hard_forward,
    reference_lut_warp_hard_st_surrogate,
)
from griffith_logic.core.functions import LutFunctionWARPHardST
from griffith_logic.layers import LutLayer


def _make_case(n, seed=13, batch=3):
    torch.manual_seed(seed)
    in_dim = max(8, 2 * n)
    out_dim = 4
    layer = LutLayer(in_dim, out_dim, n_inputs=n, parameterization='warp')
    indices = layer.indices
    w = torch.randn(out_dim, 2 ** n, dtype=torch.float64)
    x = torch.rand(batch, in_dim, dtype=torch.float64) * 0.8 + 0.1
    return x, w, indices


def _make_noise(w, scale=1.0, seed=29):
    torch.manual_seed(seed)
    u = torch.rand_like(w).clamp(1e-6, 1 - 1e-6)
    return scale * (u.log() - (1 - u).log())


class TestWARPHardSTForwardContract:
    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_forward_equals_reference_hard(self, n):
        """forward = hard真理値表の多重線形拡張(参照実装と厳密一致)"""
        x, w, indices = _make_case(n)
        y = LutFunctionWARPHardST.apply(x, w, indices, n, 1.0, None)
        y_ref = reference_lut_warp_hard_forward(x, w, indices, n)
        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)

    @pytest.mark.parametrize('n', [2, 4])
    def test_forward_with_noise_matches_reference(self, n):
        """gumbel_st(固定ノイズ)でも参照と厳密一致"""
        x, w, indices = _make_case(n)
        noise = _make_noise(w)
        y = LutFunctionWARPHardST.apply(x, w, indices, n, 1.0, noise)
        y_ref = reference_lut_warp_hard_forward(x, w, indices, n, noise=noise)
        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)

    @pytest.mark.parametrize('n', [2, 4])
    def test_selection_gap_zero_on_binary(self, n):
        """binary入力でtrain(hard_st)とevalの出力が厳密一致(gap 0)"""
        torch.manual_seed(17)
        in_dim = max(8, 2 * n)
        layer = LutLayer(in_dim, 6, n_inputs=n, parameterization='warp',
                         selection='hard_st')
        with torch.no_grad():
            layer.weights.normal_()
        x = (torch.rand(16, in_dim) > 0.5).float()
        layer.train()
        y_train = layer(x)
        layer.eval()
        y_eval = layer(x)
        assert torch.allclose(y_train, y_eval, atol=1e-6), (
            f'selection gap = {(y_train - y_eval).abs().max():.3e}'
        )

    def test_hard_st_equals_zero_noise_gumbel(self):
        """gumbel_scale→0の退化: noise=Noneとnoise=0が同一出力"""
        n = 4
        x, w, indices = _make_case(n)
        y_hard = LutFunctionWARPHardST.apply(x, w, indices, n, 1.0, None)
        y_gz = LutFunctionWARPHardST.apply(
            x, w, indices, n, 1.0, torch.zeros_like(w))
        assert torch.allclose(y_hard, y_gz, atol=TOL_F64_EQ, rtol=0)


class TestWARPHardSTBackwardContract:
    @pytest.mark.parametrize('tau_b', [1.0, 0.5, 2.0])
    @pytest.mark.parametrize('n', [2, 4])
    def test_backward_equals_tempered_soft(self, n, tau_b):
        """backward = σ((θH)/τ_b)版soft関数のautograd勾配と厳密一致"""
        x, w, indices = _make_case(n)
        torch.manual_seed(19)
        grad_output = torch.randn(x.shape[0], w.shape[0], dtype=torch.float64)

        xr = x.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        y = LutFunctionWARPHardST.apply(xr, wr, indices, n, tau_b, None)
        gx, gw = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)

        xs = x.clone().requires_grad_(True)
        ws = w.clone().requires_grad_(True)
        y_soft = reference_lut_warp_hard_st_surrogate(
            xs, ws, indices, n, tau_b)
        gx_ref, gw_ref = torch.autograd.grad(
            y_soft, (xs, ws), grad_outputs=grad_output)

        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_x不一致: {(gx - gx_ref).abs().max():.3e}'
        )
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_weights不一致: {(gw - gw_ref).abs().max():.3e}'
        )

    def test_backward_with_noise(self):
        """ノイズ込みのbackwardもσ((θH+noise)/τ_b)の勾配と一致"""
        n, tau_b = 4, 0.7
        x, w, indices = _make_case(n)
        noise = _make_noise(w)
        torch.manual_seed(23)
        grad_output = torch.randn(x.shape[0], w.shape[0], dtype=torch.float64)

        xr = x.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        y = LutFunctionWARPHardST.apply(xr, wr, indices, n, tau_b, noise)
        gx, gw = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)

        xs = x.clone().requires_grad_(True)
        ws = w.clone().requires_grad_(True)
        y_soft = reference_lut_warp_hard_st_surrogate(
            xs, ws, indices, n, tau_b, noise=noise)
        gx_ref, gw_ref = torch.autograd.grad(
            y_soft, (xs, ws), grad_outputs=grad_output)

        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0)

    def test_boundary_points(self):
        """境界点(x∈{0,1}混在)でも勾配契約が成り立つ"""
        n, tau_b = 2, 1.0
        _, w, indices = _make_case(n)
        x = torch.tensor([[0.0, 1.0, 0.5, 0.0, 1.0, 0.3, 0.9, 0.1]],
                         dtype=torch.float64)
        grad_output = torch.ones(1, w.shape[0], dtype=torch.float64)

        xr = x.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        y = LutFunctionWARPHardST.apply(xr, wr, indices, n, tau_b, None)
        gx, gw = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)

        xs = x.clone().requires_grad_(True)
        ws = w.clone().requires_grad_(True)
        y_soft = reference_lut_warp_hard_st_surrogate(
            xs, ws, indices, n, tau_b)
        gx_ref, gw_ref = torch.autograd.grad(
            y_soft, (xs, ws), grad_outputs=grad_output)

        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0)
