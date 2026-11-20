"""L2: blocks構造部品・Norm部品・LogicDropoutの勾配契約。

    - Add(or/xor/sum): float64 gradcheck + core/reference.py照合
    - Residual(concat/or/xor): float64 gradcheck
    - LogicLayerNorm/LogicBatchNorm: float64 gradcheck
      (BatchNormのSTEクランプは出力が(0,1)内部にある構成で検証。
       クランプ活性点ではforwardが不連続でgradcheck不可のため)
    - LogicDropout: 勾配疎通(マスク位置の勾配契約)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.blocks import (
    Add,
    LogicBatchNorm,
    LogicDropout,
    LogicLayerNorm,
    Residual,
)
from griffith_logic.core.reference import reference_soft_merge

BATCH = 4
WIDTH = 6


def _interior(shape, seed=0):
    """gradcheck用の内部点(0.1, 0.9)入力"""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * 0.8 + 0.1


class _Smooth(nn.Module):
    """Residual gradcheck用の滑らかな[0,1]保存inner(0.2 + 0.6x²)"""

    def forward(self, x):
        return 0.2 + 0.6 * x * x


class TestAddGradcheck:
    @pytest.mark.parametrize('mode', ['or', 'xor', 'sum'])
    def test_gradcheck_two_inputs(self, mode):
        add = Add(mode)
        a = _interior((BATCH, WIDTH), seed=1).requires_grad_(True)
        b = _interior((BATCH, WIDTH), seed=2).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda a_, b_: add([a_, b_]), (a, b), **GRADCHECK_KW,
        )

    @pytest.mark.parametrize('mode', ['or', 'xor', 'sum'])
    def test_gradcheck_three_inputs_with_boundary(self, mode):
        """境界点(0.0/1.0を含む)でも勾配が定義される(多項式なので可)"""
        add = Add(mode)
        a = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float64,
                         requires_grad=True)
        b = torch.tensor([[1.0, 0.0, 0.3]], dtype=torch.float64,
                         requires_grad=True)
        c = torch.tensor([[0.0, 0.0, 0.9]], dtype=torch.float64,
                         requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda *xs: add(list(xs)), (a, b, c), **GRADCHECK_KW,
        )


class TestAddReferenceEquivalence:
    @pytest.mark.parametrize('mode', ['or', 'xor', 'sum'])
    def test_forward_and_grads_match(self, mode):
        add = Add(mode)
        a = _interior((BATCH, WIDTH), seed=3)
        b = _interior((BATCH, WIDTH), seed=4)
        c = _interior((BATCH, WIDTH), seed=5)
        grad_output = torch.randn(BATCH, WIDTH, dtype=torch.float64)

        xs = [t.detach().clone().requires_grad_(True) for t in (a, b, c)]
        y = add(xs)
        grads = torch.autograd.grad(y, xs, grad_outputs=grad_output)

        xs_ref = [t.detach().clone().requires_grad_(True) for t in (a, b, c)]
        y_ref = reference_soft_merge(xs_ref, mode)
        grads_ref = torch.autograd.grad(y_ref, xs_ref, grad_outputs=grad_output)

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)
        for g, g_ref in zip(grads, grads_ref):
            assert torch.allclose(g, g_ref, atol=TOL_F64_GRAD, rtol=0)


class TestResidualGradcheck:
    @pytest.mark.parametrize('merge', ['concat', 'or', 'xor'])
    def test_gradcheck(self, merge):
        res = Residual(_Smooth(), merge=merge).double()
        x = _interior((BATCH, WIDTH), seed=6).requires_grad_(True)
        assert torch.autograd.gradcheck(res, (x,), **GRADCHECK_KW)


class TestNormGradcheck:
    def test_layernorm_gradcheck(self):
        """入力とLayerNormパラメータ両方についてgradcheck"""
        m = LogicLayerNorm(WIDTH).double()
        x = _interior((BATCH, WIDTH), seed=7).requires_grad_(True)
        params = tuple(m.parameters())

        def fn(x_, *ps):
            return m(x_)

        assert torch.autograd.gradcheck(fn, (x, *params), **GRADCHECK_KW)

    def test_batchnorm_gradcheck_interior(self):
        """BN出力が(0,1)内部にある構成でのgradcheck。

        STEクランプはクランプ活性点でforward不連続になるため、
        全出力が(0.02, 0.98)に収まることを前提条件として検証してから
        gradcheckする(train時BNのbatch統計は滑らか)。
        """
        m = LogicBatchNorm(WIDTH).double()
        m.train()
        # 特徴ごとにz-scoreが±1.5以内に収まる決定的入力
        base = torch.tensor([0.3, 0.4, 0.5, 0.6], dtype=torch.float64)
        x = base.unsqueeze(1).expand(4, WIDTH).clone()
        x += 0.01 * torch.arange(WIDTH, dtype=torch.float64)
        with torch.no_grad():
            h = m(x)
            assert ((h > 0.02) & (h < 0.98)).all(), (
                'gradcheck前提(クランプ非活性)が満たされない'
            )
        x = x.requires_grad_(True)
        params = tuple(m.parameters())

        def fn(x_, *ps):
            return m(x_)

        assert torch.autograd.gradcheck(fn, (x, *params), **GRADCHECK_KW)

    def test_batchnorm_ste_clamp_backward_is_identity(self):
        """クランプ活性点でもbackwardはBN勾配をそのまま通す(STE契約)"""
        m = LogicBatchNorm(2).double()
        m.train()
        x = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.9, 0.1]],
            dtype=torch.float64, requires_grad=True,
        )
        y = m(x)
        y.sum().backward()
        grad_ste = x.grad.clone()

        # クランプなしのBN単体の勾配と一致するはず
        x2 = x.detach().clone().requires_grad_(True)
        m.bn(x2).sum().backward()
        assert torch.allclose(grad_ste, x2.grad, atol=TOL_F64_GRAD, rtol=0)


class TestDropoutGradFlow:
    @pytest.mark.parametrize('mode,masked_grad', [
        ('neutral', 0.0),
        ('flip', -1.0),
        ('zero', 0.0),
    ])
    def test_grad_contract(self, mode, masked_grad, seed):
        """非マスク位置の勾配=1、マスク位置の勾配=方式ごとの契約値"""
        drop = LogicDropout(p=0.5, mode=mode)
        drop.train()
        x = _interior((BATCH, WIDTH), seed=8).requires_grad_(True)
        torch.manual_seed(123)
        y = drop(x)
        y.sum().backward()

        # マスク位置の特定: neutralなら0.5固定、flipなら1-x、zeroなら0
        with torch.no_grad():
            changed = ~torch.isclose(y, x)
        assert changed.any(), 'p=0.5でマスクが1つも発生しないのは異常'
        expected = torch.where(
            changed,
            torch.tensor(masked_grad, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        assert torch.allclose(x.grad, expected, atol=TOL_F64_GRAD, rtol=0)
        assert torch.isfinite(x.grad).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
