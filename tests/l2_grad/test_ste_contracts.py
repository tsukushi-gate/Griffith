"""L2: STE(Straight-Through Estimator)系の契約テスト。

forward不連続のためgradcheckできないFunctionは、
「forward契約」と「backward契約」を解析式との厳密照合で検証する。

HardST(CAGE論文, Kim 2026):
    forward契約:  argmax選択によるhard出力(推論と同一 = selection gap 0)
    backward契約: softmax(weights/τ_b)によるsoft関数のautograd勾配と一致

Binarize(BinaryBrain方式):
    forward契約:  x > threshold の閾値二値化
    backward契約: grad × 1[hardtanh_min < x < hardtanh_max]
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.binarize import BinarizeFunction
from griffith_logic.core import reference_basis, reference_gather
from griffith_logic.lut_layer import LutFunctionHardST, LutLayer


def _make_hard_st_case(n, use_gate_table, seed=13):
    torch.manual_seed(seed)
    in_dim = max(8, 2 * n)
    out_dim, batch = 4, 3
    layer = LutLayer(in_dim, out_dim, n_inputs=n, parameterization='softmax')
    indices = layer.indices
    if use_gate_table:
        assert n == 2
        gate_table = layer.gate_table.double()
        w = torch.randn(out_dim, 16, dtype=torch.float64)
    else:
        gate_table = None
        w = torch.randn(out_dim, 2 ** n, dtype=torch.float64)
    x = torch.rand(batch, in_dim, dtype=torch.float64) * 0.8 + 0.1
    return x, w, indices, gate_table


def _soft_reference(x, w, indices, n, gate_table, tau_b):
    """backward契約の参照: softmax(w/τ_b)によるsoft関数(純autograd)"""
    gathered = reference_gather(x, indices)
    basis = reference_basis(gathered, n)
    s = F.softmax(w / tau_b, dim=-1)
    w_eff = s @ gate_table if gate_table is not None else s
    return (basis * w_eff.unsqueeze(0)).sum(dim=-1)


class TestHardSTForwardContract:
    @pytest.mark.parametrize('use_gate_table', [True, False], ids=['16gate', 'basis'])
    def test_forward_equals_argmax_selection(self, use_gate_table):
        """forward = argmaxで選択したhard関数の出力(手計算と厳密一致)"""
        n = 2 if use_gate_table else 4
        x, w, indices, gate_table = _make_hard_st_case(n, use_gate_table)

        y = LutFunctionHardST.apply(x, w, indices, n, gate_table, 1.0)

        gathered = reference_gather(x, indices)
        basis = reference_basis(gathered, n)
        if gate_table is not None:
            w_hard = gate_table[w.argmax(dim=-1)]
        else:
            w_hard = F.one_hot(w.argmax(dim=-1), w.shape[-1]).to(basis.dtype)
        y_expected = (basis * w_hard.unsqueeze(0)).sum(dim=-1)

        assert torch.allclose(y, y_expected, atol=TOL_F64_EQ, rtol=0)

    def test_selection_gap_zero(self):
        """train(hard_st)とevalの出力が一致(CAGEのselection gap = 0)"""
        torch.manual_seed(17)
        layer = LutLayer(8, 6, n_inputs=2, parameterization='softmax',
                         selection='hard_st')
        x = torch.rand(5, 8)
        layer.train()
        y_train = layer(x)
        layer.eval()
        y_eval = layer(x)
        assert torch.allclose(y_train, y_eval, atol=1e-6), (
            f'selection gap = {(y_train - y_eval).abs().max():.3e}'
        )


class TestHardSTBackwardContract:
    @pytest.mark.parametrize('tau_b', [1.0, 0.5, 2.0])
    @pytest.mark.parametrize('use_gate_table', [True, False], ids=['16gate', 'basis'])
    def test_backward_equals_tempered_soft(self, use_gate_table, tau_b):
        """backward = softmax(w/τ_b)版soft関数のautograd勾配と厳密一致"""
        n = 2 if use_gate_table else 4
        x, w, indices, gate_table = _make_hard_st_case(n, use_gate_table)
        torch.manual_seed(19)
        grad_output = torch.randn(x.shape[0], w.shape[0], dtype=torch.float64)

        xr = x.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        y = LutFunctionHardST.apply(xr, wr, indices, n, gate_table, tau_b)
        gx, gw = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)

        xs = x.clone().requires_grad_(True)
        ws = w.clone().requires_grad_(True)
        y_soft = _soft_reference(xs, ws, indices, n, gate_table, tau_b)
        gx_ref, gw_ref = torch.autograd.grad(
            y_soft, (xs, ws), grad_outputs=grad_output,
        )

        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_x不一致: {(gx - gx_ref).abs().max():.3e}'
        )
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_weights不一致: {(gw - gw_ref).abs().max():.3e}'
        )


class TestBinarizeSTE:
    def test_forward_threshold(self):
        x = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0], dtype=torch.float64)
        y = BinarizeFunction.apply(x, 0.5, 0.0, 1.0)
        assert torch.equal(y, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0],
                                           dtype=torch.float64))

    def test_backward_hardtanh_mask(self):
        """backward = grad × 1[min < x < max] の厳密照合"""
        x = torch.tensor(
            [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
            dtype=torch.float64, requires_grad=True,
        )
        grad_output = torch.arange(1.0, 8.0, dtype=torch.float64)
        y = BinarizeFunction.apply(x, 0.5, 0.0, 1.0)
        (gx,) = torch.autograd.grad(y, (x,), grad_outputs=grad_output)
        # 開区間(0,1)内のみ勾配が通る: x=-0.5,0.0,1.0,1.5は遮断
        expected = grad_output * torch.tensor(
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64,
        )
        assert torch.equal(gx, expected)

    def test_custom_range(self):
        """hardtanh範囲を変えた場合のマスク"""
        x = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float64,
                         requires_grad=True)
        y = BinarizeFunction.apply(x, 0.0, -1.0, 1.0)
        (gx,) = torch.autograd.grad(y, (x,), grad_outputs=torch.ones(4, dtype=torch.float64))
        assert torch.equal(gx, torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
