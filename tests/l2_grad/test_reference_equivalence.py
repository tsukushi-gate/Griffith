"""L2: カスタムFunctionと純autograd参照実装の照合。

各カスタムFunctionの(出力, grad_x, grad_weights)を
core/reference.py(素朴なtorch演算+autograd)と照合する。
gradcheck(数値微分)より厳しい許容誤差(float64で1e-12/1e-10)。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD
from test_gradcheck_matrix import CONFIGS, IDS, make_case

from griffith_logic.core import reference_grads
from griffith_logic.lut_layer import LutLayer


def _custom_grads(x, w, apply_fn, grad_output):
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    y = apply_fn(xr, wr)
    grad_x, grad_w = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)
    return y.detach(), grad_x, grad_w


class TestReferenceEquivalence:
    @pytest.mark.parametrize('boundary', [False, True], ids=['interior', 'boundary'])
    @pytest.mark.parametrize('param,n', CONFIGS, ids=IDS)
    def test_forward_and_grads_match(self, param, n, boundary):
        x, w, apply_fn = make_case(param, n, boundary=boundary)
        torch.manual_seed(5)
        grad_output = torch.randn(x.shape[0], w.shape[0], dtype=torch.float64)

        y, gx, gw = _custom_grads(x, w, apply_fn, grad_output)

        # 参照実装(gate_tableはsoftmax n=2の16ゲートモードのみ)
        layer = _layer_for(param, n)
        gate_table = getattr(layer, 'gate_table', None)
        gate_table = gate_table.double() if gate_table is not None else None
        y_ref, gx_ref, gw_ref = reference_grads(
            x, w, _indices_for(param, n), n, param, grad_output,
            gate_table=gate_table,
        )

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0), (
            f'出力不一致: {(y - y_ref).abs().max():.3e}'
        )
        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_x不一致: {(gx - gx_ref).abs().max():.3e}'
        )
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_weights不一致: {(gw - gw_ref).abs().max():.3e}'
        )


def _layer_for(param, n):
    torch.manual_seed(11)
    return LutLayer(max(8, 2 * n), 4, n_inputs=n, parameterization=param)


def _indices_for(param, n):
    return _layer_for(param, n).indices


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
