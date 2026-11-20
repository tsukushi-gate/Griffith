"""L2: 論文IWP(independent)× 推定器(sigmoid/sin)の勾配契約。

論文 "Light Differentiable Logic Gate Networks" (arXiv:2510.03250) の IWP は
当ライブラリの `parameterization='independent'` と数学的に同一で、
論文が実験で採用した推定器は sin01(Ω)=0.5+0.5·sin(Ω)(§C.1.1)。

検証内容:
    1. float64 gradcheck(x / weights / 両方、内部点)
    2. LutFunctionIndependent と core/reference.py の (出力, grad_x, grad_w) 照合
    3. 境界値(x=0.0/1.0 厳密)でも 1. 2. が成立すること

`tests/l2_grad/test_gradcheck_matrix.py` は他テストが CONFIGS を import して
いるため触らない。本ファイルは独立に完結させる。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.core import reference_grads
from griffith_logic.core.estimators import ESTIMATORS
from griffith_logic.lut_layer import LutFunctionIndependent, LutLayer

N_INPUTS = [2, 4, 6]
IDS_N = [f'n{n}' for n in N_INPUTS]


def _make_case(n, boundary=False):
    """(x, w, indices) を生成。boundary=Trueで入力に厳密な0/1を混ぜる。

    `test_gradcheck_matrix.make_case` と同じ規模・同じseedの作法に倣う。
    """
    torch.manual_seed(11)
    in_dim = max(8, 2 * n)
    out_dim, batch = 4, 3
    layer = LutLayer(in_dim, out_dim, n_inputs=n, parameterization='independent')
    indices = layer.indices
    w = torch.randn(out_dim, 2 ** n, dtype=torch.float64)
    x = torch.rand(batch, in_dim, dtype=torch.float64) * 0.8 + 0.1
    if boundary:
        x[:, 0] = 0.0
        x[:, 1] = 1.0
    return x, w, indices


def _apply_fn(indices, n, estimator):
    def fn(x, w):
        return LutFunctionIndependent.apply(x, w, indices, n, estimator)
    return fn


def _custom_grads(x, w, apply_fn, grad_output):
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    y = apply_fn(xr, wr)
    grad_x, grad_w = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)
    return y.detach(), grad_x, grad_w


class TestGradcheck:
    """float64数値微分との照合(内部点/境界点)"""

    @pytest.mark.parametrize('boundary', [False, True],
                             ids=['interior', 'boundary'])
    @pytest.mark.parametrize('estimator', ESTIMATORS)
    @pytest.mark.parametrize('n', N_INPUTS, ids=IDS_N)
    def test_grad_x(self, n, estimator, boundary):
        x, w, indices = _make_case(n, boundary=boundary)
        fn = _apply_fn(indices, n, estimator)
        x = x.requires_grad_(True)
        assert torch.autograd.gradcheck(lambda x: fn(x, w), (x,), **GRADCHECK_KW)

    @pytest.mark.parametrize('boundary', [False, True],
                             ids=['interior', 'boundary'])
    @pytest.mark.parametrize('estimator', ESTIMATORS)
    @pytest.mark.parametrize('n', N_INPUTS, ids=IDS_N)
    def test_grad_weights(self, n, estimator, boundary):
        x, w, indices = _make_case(n, boundary=boundary)
        fn = _apply_fn(indices, n, estimator)
        w = w.requires_grad_(True)
        assert torch.autograd.gradcheck(lambda w: fn(x, w), (w,), **GRADCHECK_KW)

    @pytest.mark.parametrize('boundary', [False, True],
                             ids=['interior', 'boundary'])
    @pytest.mark.parametrize('estimator', ESTIMATORS)
    @pytest.mark.parametrize('n', N_INPUTS, ids=IDS_N)
    def test_grad_both(self, n, estimator, boundary):
        x, w, indices = _make_case(n, boundary=boundary)
        fn = _apply_fn(indices, n, estimator)
        x = x.requires_grad_(True)
        w = w.requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (x, w), **GRADCHECK_KW)


class TestReferenceEquivalence:
    """カスタムFunction ⇔ core/reference.py(素朴実装+autograd)の照合"""

    @pytest.mark.parametrize('boundary', [False, True],
                             ids=['interior', 'boundary'])
    @pytest.mark.parametrize('estimator', ESTIMATORS)
    @pytest.mark.parametrize('n', N_INPUTS, ids=IDS_N)
    def test_forward_and_grads_match(self, n, estimator, boundary):
        x, w, indices = _make_case(n, boundary=boundary)
        fn = _apply_fn(indices, n, estimator)
        torch.manual_seed(5)
        grad_output = torch.randn(x.shape[0], w.shape[0], dtype=torch.float64)

        y, gx, gw = _custom_grads(x, w, fn, grad_output)
        y_ref, gx_ref, gw_ref = reference_grads(
            x, w, indices, n, 'independent', grad_output, estimator=estimator,
        )

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0), (
            f'{estimator} n={n}: 出力不一致 {(y - y_ref).abs().max():.3e}'
        )
        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'{estimator} n={n}: grad_x不一致 {(gx - gx_ref).abs().max():.3e}'
        )
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'{estimator} n={n}: grad_w不一致 {(gw - gw_ref).abs().max():.3e}'
        )


class TestSigmoidBackCompat:
    """既定(sigmoid)の挙動が estimator 引数追加で不変であること"""

    @pytest.mark.parametrize('n', N_INPUTS, ids=IDS_N)
    def test_default_equals_explicit_sigmoid(self, n):
        x, w, indices = _make_case(n)
        y_default = LutFunctionIndependent.apply(x, w, indices, n)
        y_sigmoid = LutFunctionIndependent.apply(x, w, indices, n, 'sigmoid')
        assert torch.equal(y_default, y_sigmoid), (
            f'n={n}: estimator既定値がsigmoidと一致しない'
        )

    @pytest.mark.parametrize('n', N_INPUTS, ids=IDS_N)
    def test_sin_differs_from_sigmoid(self, n):
        """sin推定器が実際に別の関数であること(no-op実装の検出)"""
        x, w, indices = _make_case(n)
        y_sig = LutFunctionIndependent.apply(x, w, indices, n, 'sigmoid')
        y_sin = LutFunctionIndependent.apply(x, w, indices, n, 'sin')
        assert not torch.allclose(y_sig, y_sin), (
            f'n={n}: sinとsigmoidの出力が同一(推定器が効いていない)'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
