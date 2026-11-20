"""L1: 基底関数の数学的性質。

検証する数式:
    分割単位性: Σ_k Π_j (x_j if bit(k,j) else 1-x_j) = 1  ∀x∈[0,1]^N
    二値頂点:   x∈{0,1}^N で basis = e_{δ(x)}(one-hot、厳密)
    WH直交性:   Σ_{x∈{0,1}^N} φ_i(x)φ_j(x) = 2^N δ_ij
    実装一致:   butterfly実装 == 素朴な参照実装
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest定数用

from conftest import TOL_F64_EQ

from griffith_logic.core import reference_basis, reference_wh_basis
from griffith_logic.lut_layer import (
    _compute_basis_butterfly,
    _compute_wh_basis_butterfly,
)

NS = [2, 4, 6]


def _all_binary_inputs(n: int) -> torch.Tensor:
    """全2^n通りの二値入力: (2^n, 1, n)。行kのビットjが入力x_j"""
    rows = [
        [float((k >> j) & 1) for j in range(n)]
        for k in range(2 ** n)
    ]
    return torch.tensor(rows, dtype=torch.float64).unsqueeze(1)


class TestPartitionOfUnity:
    @pytest.mark.parametrize('n', NS)
    def test_sum_to_one_float64(self, n, seed):
        """分割単位性(float64, 1e-12)"""
        gathered = torch.rand(16, 5, n, dtype=torch.float64)
        for basis_fn in (reference_basis, _compute_basis_butterfly):
            basis = basis_fn(gathered, n)
            sums = basis.sum(dim=-1)
            assert torch.allclose(
                sums, torch.ones_like(sums), atol=TOL_F64_EQ, rtol=0,
            ), f'{basis_fn.__name__}: max|Σ-1|={((sums - 1).abs().max()):.3e}'

    @pytest.mark.parametrize('n', NS)
    def test_nonnegative(self, n, seed):
        """基底は非負(確率解釈の前提)"""
        gathered = torch.rand(16, 5, n, dtype=torch.float64)
        basis = _compute_basis_butterfly(gathered, n)
        assert (basis >= 0).all()


class TestBinaryVertices:
    @pytest.mark.parametrize('n', NS)
    def test_one_hot_exact(self, n):
        """二値頂点で basis は厳密にone-hot。hot位置kはビット規約 bit(k,j)=x_j"""
        gathered = _all_binary_inputs(n)  # (2^n, 1, n)
        basis = _compute_basis_butterfly(gathered, n)  # (2^n, 1, 2^n)
        expected = torch.eye(2 ** n, dtype=torch.float64).unsqueeze(1)
        assert torch.equal(basis, expected)

    @pytest.mark.parametrize('n', NS)
    def test_reference_agrees(self, n):
        gathered = _all_binary_inputs(n)
        assert torch.equal(
            reference_basis(gathered, n), _compute_basis_butterfly(gathered, n),
        )


class TestWalshHadamard:
    @pytest.mark.parametrize('n', NS)
    def test_orthogonality(self, n):
        """WH基底の直交性: Σ_x φ_i(x)φ_j(x) = 2^n δ_ij(整数演算で厳密)"""
        gathered = _all_binary_inputs(n)
        wh = _compute_wh_basis_butterfly(gathered, n).squeeze(1)  # (2^n点, 2^n基底)
        gram = wh.T @ wh
        expected = (2 ** n) * torch.eye(2 ** n, dtype=torch.float64)
        assert torch.equal(gram, expected)

    @pytest.mark.parametrize('n', NS)
    def test_values_are_pm_one(self, n):
        """二値入力でWH基底値は±1のみ"""
        gathered = _all_binary_inputs(n)
        wh = _compute_wh_basis_butterfly(gathered, n)
        assert torch.equal(wh.abs(), torch.ones_like(wh))

    @pytest.mark.parametrize('n', NS)
    def test_reference_agrees_continuous(self, n, seed):
        """連続入力でもbutterfly実装と参照実装が一致"""
        gathered = torch.rand(8, 3, n, dtype=torch.float64)
        assert torch.allclose(
            reference_wh_basis(gathered, n),
            _compute_wh_basis_butterfly(gathered, n),
            atol=TOL_F64_EQ, rtol=0,
        )


class TestIndexConvention:
    """基底インデックス規約の固定(IR・HDL・エディタが依存する契約)"""

    def test_butterfly_bit_order_n2(self):
        """k=0→(0,0), k=1→(1,0), k=2→(0,1), k=3→(1,1)。bit j of k ⇔ x_j"""
        for k, (x0, x1) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            gathered = torch.tensor([[[float(x0), float(x1)]]], dtype=torch.float64)
            basis = _compute_basis_butterfly(gathered, 2)
            assert basis[0, 0, k] == 1.0, f'k={k} が ({x0},{x1}) に対応していない'

    def test_wh_subset_convention(self):
        """φ_k(s) = Π_{j: bit(k,j)=1} s_j。例: k=3 → s_0·s_1"""
        s0, s1 = -1.0, -1.0  # x=(1,1)
        gathered = torch.tensor([[[1.0, 1.0]]], dtype=torch.float64)
        wh = _compute_wh_basis_butterfly(gathered, 2)
        assert wh[0, 0, 0] == 1.0        # 空集合 = 1
        assert wh[0, 0, 1] == s0         # {0}
        assert wh[0, 0, 2] == s1         # {1}
        assert wh[0, 0, 3] == s0 * s1    # {0,1}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
