"""ButterflyFactor(QAT)のforward/backward契約テスト。

HeavisideSTE(不連続forward)を含むためgradcheckでなく、
new-component-tdd の「forward契約+backward契約」方式:
    forward契約:  ベクトル化実装のhard出力 == reference_butterfly_forward
                  (素朴ループ・純autograd参照)と厳密一致
    backward契約: master重み(a, b, θ)と入力への勾配が参照実装の
                  autograd勾配と TOL_F64_GRAD で一致
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from conftest import TOL_F64_EQ, TOL_F64_GRAD  # noqa: E402

from griffith_logic.convert.butterfly_qat import (  # noqa: E402
    ButterflyFactor,
    ButterflyMLP,
)
from griffith_logic.core.reference import (  # noqa: E402
    reference_butterfly_forward,
)


def _stack(width: int, n_levels: int, seed: int):
    torch.manual_seed(seed)
    log2w = int(np.log2(width))
    factors = [ButterflyFactor(width, lv % log2w).double()
               for lv in range(n_levels)]
    return factors


def _run_both(factors, x):
    """(実装出力, 参照出力) を同一パラメータで計算する。"""
    h = x
    for f in factors:
        h = f(h)
    ref = reference_butterfly_forward(
        x,
        [f.a for f in factors],
        [f.b for f in factors],
        [f.theta for f in factors],
        [torch.stack([f.u, f.v], dim=1) for f in factors],
    )
    return h, ref


class TestButterflyContracts:
    @pytest.mark.parametrize('seed', range(3))
    @pytest.mark.parametrize('width,n_levels', [(4, 2), (8, 3), (8, 6)])
    def test_forward_contract(self, seed, width, n_levels):
        factors = _stack(width, n_levels, seed)
        torch.manual_seed(100 + seed)
        x = (torch.rand(16, width) > 0.5).double()
        got, ref = _run_both(factors, x)
        assert torch.allclose(got, ref, atol=TOL_F64_EQ)
        assert set(got.unique().tolist()).issubset({0.0, 1.0})

    @pytest.mark.parametrize('seed', range(3))
    def test_backward_contract(self, seed):
        width, n_levels = 8, 3
        f_impl = _stack(width, n_levels, seed)
        f_ref = _stack(width, n_levels, seed)  # 同一seed=同一パラメータ
        torch.manual_seed(200 + seed)
        x = (torch.rand(16, width) > 0.5).double()

        out_impl = x
        for f in f_impl:
            out_impl = f(out_impl)
        out_impl.sum().backward()

        out_ref = reference_butterfly_forward(
            x,
            [f.a for f in f_ref],
            [f.b for f in f_ref],
            [f.theta for f in f_ref],
            [torch.stack([f.u, f.v], dim=1) for f in f_ref],
        )
        out_ref.sum().backward()

        for fi, fr in zip(f_impl, f_ref):
            for name in ('a', 'b', 'theta'):
                gi = getattr(fi, name).grad
                gr = getattr(fr, name).grad
                assert torch.allclose(gi, gr, atol=TOL_F64_GRAD), (
                    f'{name}勾配が参照と不一致')

    def test_hard_forward_is_deterministic_function(self):
        """train/evalで出力が変わらない(forwardが最初から離散)。"""
        model = ButterflyMLP(in_width=6, num_classes=2).double()
        x = (torch.rand(32, 6) > 0.5).double()
        model.train()
        with torch.no_grad():
            y_tr = model(x)
        model.eval()
        with torch.no_grad():
            y_ev = model(x)
        assert torch.equal(y_tr, y_ev)
