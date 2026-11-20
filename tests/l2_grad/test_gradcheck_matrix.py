"""L2: 全パラメータ化 × {x, weights} のgradcheckマトリクス。

torch.autograd.gradcheck(数値微分との照合、float64)で
カスタムautograd Functionのbackwardを機械的に検証する。

対象外(別テストで検証):
    hard_st  — forward不連続のためgradcheck不可 → test_ste_contracts.py
    warp gumbel=True — 確率的forward → 決定的なgumbel=Falseのみ検証
    CUDA版  — test_cuda_parity.py(@cuda)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW

from griffith_logic.lut_layer import (
    LutFunction,
    LutFunctionIndependent,
    LutFunctionIWP,
    LutFunctionRaw,
    LutFunctionWARP,
    LutLayer,
)

# (parameterization, n_inputs)。softmax n=2 は16ゲートモード
CONFIGS = [
    ('softmax', 2),
    ('softmax', 4),
    ('softmax', 6),
    ('independent', 2),
    ('independent', 4),
    ('independent', 6),
    ('raw', 2),
    ('raw', 4),
    ('iwp', 2),
    ('iwp', 4),
    ('iwp', 6),
    ('warp', 2),
    ('warp', 4),
    ('warp', 6),
]

IDS = [f'{p}_n{n}' for p, n in CONFIGS]


def make_case(param, n, boundary=False):
    """(x, w, apply_fn) を生成。boundary=Trueで入力に厳密な0/1を混ぜる"""
    torch.manual_seed(11)
    in_dim = max(8, 2 * n)
    out_dim, batch = 4, 3
    layer = LutLayer(in_dim, out_dim, n_inputs=n, parameterization=param)
    indices = layer.indices
    gate_table = getattr(layer, 'gate_table', None)
    if gate_table is not None:
        gate_table = gate_table.double()

    if param == 'iwp':
        w = torch.randn(out_dim, n, dtype=torch.float64)
    elif param == 'softmax' and n == 2:
        w = torch.randn(out_dim, 16, dtype=torch.float64)
    else:
        w = torch.randn(out_dim, 2 ** n, dtype=torch.float64)

    x = torch.rand(batch, in_dim, dtype=torch.float64) * 0.8 + 0.1
    if boundary:
        x[:, 0] = 0.0
        x[:, 1] = 1.0

    def apply_fn(x, w):
        if param == 'softmax':
            return LutFunction.apply(x, w, indices, n, gate_table)
        if param == 'independent':
            return LutFunctionIndependent.apply(x, w, indices, n)
        if param == 'raw':
            return LutFunctionRaw.apply(x, w, indices, n)
        if param == 'iwp':
            return LutFunctionIWP.apply(x, w, indices, n)
        if param == 'warp':
            return LutFunctionWARP.apply(x, w, indices, n, 1.0, False)
        raise ValueError(param)

    return x, w, apply_fn


class TestGradcheckInterior:
    """内部点(x∈(0.1,0.9))でのgradcheck"""

    @pytest.mark.parametrize('param,n', CONFIGS, ids=IDS)
    def test_grad_x(self, param, n):
        x, w, apply_fn = make_case(param, n)
        x = x.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: apply_fn(x, w), (x,), **GRADCHECK_KW,
        )

    @pytest.mark.parametrize('param,n', CONFIGS, ids=IDS)
    def test_grad_weights(self, param, n):
        x, w, apply_fn = make_case(param, n)
        w = w.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda w: apply_fn(x, w), (w,), **GRADCHECK_KW,
        )

    @pytest.mark.parametrize('param,n', CONFIGS, ids=IDS)
    def test_grad_both(self, param, n):
        x, w, apply_fn = make_case(param, n)
        x = x.requires_grad_(True)
        w = w.requires_grad_(True)
        assert torch.autograd.gradcheck(apply_fn, (x, w), **GRADCHECK_KW)


class TestGradcheckBoundary:
    """境界点(x=0.0/1.0の厳密値を含む)でのgradcheck。

    旧IWP実装はここで破綻していた(clamp方式のゼロ除算回避)。
    数式は[0,1]外でも多項式として滑らかに定義されるためgradcheck可能。
    """

    @pytest.mark.parametrize('param,n', CONFIGS, ids=IDS)
    def test_grad_both_at_boundary(self, param, n):
        x, w, apply_fn = make_case(param, n, boundary=True)
        x = x.requires_grad_(True)
        w = w.requires_grad_(True)
        assert torch.autograd.gradcheck(apply_fn, (x, w), **GRADCHECK_KW)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
