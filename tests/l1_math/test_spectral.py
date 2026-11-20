"""L1: Spectral Regularization(DWN論文§3.4)の数学的性質。

検証する数式:
    specnorm(L) = ‖L·C‖₂,  C_ij = (1/2^n) Π_{a∈{b | i_b=1}} (2j_a − 1)
    フーリエ係数(ground truth、ブルートフォースWH変換):
        f̂(S) = (1/2^n) Σ_{x∈{0,1}^n} f(x) Π_{i∈S} (2x_i − 1)
    C行列版とWH変換版は列の符号付き置換の違いを除き同一で、
    ノルム(行ベクトルL2・行列2ノルム)は厳密に一致する。
    パーセバル恒等式: specnorm(L) = ‖L‖₂ / √(2^n)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.regularization.spectral import (
    build_spectral_coefficient_matrix,
    spectral_norm,
)

NS = [1, 2, 4, 6]


def naive_coefficient_matrix(n: int) -> torch.Tensor:
    """C_ij = (1/2^n) Π_{a: bit(i,a)=1} (2j_a − 1) の素朴2重ループ"""
    size = 2 ** n
    c = torch.zeros(size, size, dtype=torch.float64)
    for i in range(size):
        for j in range(size):
            prod = 1.0
            for a in range(n):
                if (i >> a) & 1:
                    prod *= 2.0 * ((j >> a) & 1) - 1.0
            c[i, j] = prod / size
    return c


def brute_force_fourier(tables: torch.Tensor, n: int) -> torch.Tensor:
    """WH変換ブルートフォース: f̂(S) = (1/2^n) Σ_x f(x) Π_{i∈S}(2x_i−1)"""
    size = 2 ** n
    u = tables.shape[0]
    coeffs = torch.zeros(u, size, dtype=torch.float64)
    for r in range(u):
        for s in range(size):
            acc = 0.0
            for x in range(size):
                prod = 1.0
                for i in range(n):
                    if (s >> i) & 1:
                        prod *= 2.0 * ((x >> i) & 1) - 1.0
                acc += tables[r, x].item() * prod
            coeffs[r, s] = acc / size
    return coeffs


class TestCoefficientMatrix:
    @pytest.mark.parametrize('n', NS)
    def test_matches_naive(self, n):
        c = build_spectral_coefficient_matrix(n, dtype=torch.float64)
        assert c.shape == (2 ** n, 2 ** n)
        assert torch.allclose(c, naive_coefficient_matrix(n), atol=TOL_F64_EQ, rtol=0)

    @pytest.mark.parametrize('n', NS)
    def test_scaled_orthogonality(self, n):
        """C·C^T = I/2^n(WH基底の直交性 → ノルム計算の根拠)"""
        c = build_spectral_coefficient_matrix(n, dtype=torch.float64)
        gram = c @ c.T
        expected = torch.eye(2 ** n, dtype=torch.float64) / 2 ** n
        assert torch.allclose(gram, expected, atol=TOL_F64_EQ, rtol=0)


class TestSpectralNorm:
    @pytest.mark.parametrize('n', [1, 2, 4])
    def test_norm_matches_brute_force_fourier(self, n, seed):
        """‖L·C‖₂ = ‖ブルートフォースWH変換のフーリエ係数行列‖₂(1e-12)"""
        tables = torch.rand(5, 2 ** n, dtype=torch.float64) * 2 - 1
        fourier = brute_force_fourier(tables, n)
        expected = torch.linalg.matrix_norm(fourier, ord=2)
        assert torch.allclose(
            spectral_norm(tables, n), expected, atol=TOL_F64_EQ, rtol=0,
        )

    @pytest.mark.parametrize('n', [1, 2, 4])
    def test_row_norms_match_brute_force(self, n, seed):
        """LUTごとのフーリエ係数ベクトルのL2ノルムも厳密一致"""
        tables = torch.rand(5, 2 ** n, dtype=torch.float64) * 2 - 1
        c = build_spectral_coefficient_matrix(n, dtype=torch.float64)
        lc = tables @ c
        fourier = brute_force_fourier(tables, n)
        assert torch.allclose(
            lc.norm(dim=-1), fourier.norm(dim=-1), atol=TOL_F64_EQ, rtol=0,
        )

    def test_constant_function_norm_one(self):
        """定数関数 f≡1 のスペクトルノルムは1(f̂は単一係数のみ非ゼロ)"""
        tables = torch.ones(1, 8, dtype=torch.float64)
        assert torch.allclose(
            spectral_norm(tables, 3),
            torch.tensor(1.0, dtype=torch.float64),
            atol=TOL_F64_EQ, rtol=0,
        )

    @pytest.mark.parametrize('n', NS)
    def test_parseval_identity(self, n, seed):
        """specnorm(L) = ‖L‖₂/√(2^n)(C=直交行列/√(2^n)による帰結)"""
        tables = torch.rand(7, 2 ** n, dtype=torch.float64) * 2 - 1
        expected = torch.linalg.matrix_norm(tables, ord=2) / (2 ** n) ** 0.5
        assert torch.allclose(
            spectral_norm(tables, n), expected, atol=TOL_F64_EQ, rtol=0,
        )

    def test_differentiable(self, seed):
        """正則化項として損失に加算できる(勾配が有限)"""
        tables = (torch.rand(4, 16, dtype=torch.float64) * 2 - 1).requires_grad_(True)
        spectral_norm(tables, 4).backward()
        assert tables.grad is not None and torch.isfinite(tables.grad).all()

    def test_table_width_validated(self):
        with pytest.raises(ValueError):
            spectral_norm(torch.rand(3, 8), 2)  # 8 ≠ 2^2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestSpectralApplyHelpers:
    """apply.py(配管): Gram形式とsoft真理値表抽出の正しさ。"""

    def test_gram_form_matches_svd_form(self):
        """spectral_norm_gram == spectral_norm(full SVD)の厳密一致"""
        from griffith_logic.regularization import (
            spectral_norm,
            spectral_norm_gram,
        )
        torch.manual_seed(31)
        for n in (2, 4, 6):
            tables = torch.rand(37, 2 ** n, dtype=torch.float64) * 2.0 - 1.0
            a = spectral_norm(tables, n)
            b = spectral_norm_gram(tables, n)
            assert torch.allclose(a, b, atol=TOL_F64_EQ * 100, rtol=0), (
                f'n={n}: SVD={a.item():.12f} Gram={b.item():.12f}'
            )

    def test_soft_truth_table_matches_forward_semantics(self):
        """抽出表[o,e] == 各パラメータ化のsoft forwardのbinary頂点評価。

        恒等配線のLutLayerで全2^n頂点を評価した出力と厳密一致すれば、
        正則化が訓練時のforward意味論と同じ表を見ていることが保証される。
        """
        from griffith_logic.layers import LutLayer
        from griffith_logic.regularization import soft_truth_table
        torch.manual_seed(37)
        n, out_dim = 4, 5
        pats = torch.tensor(
            [[(e >> j) & 1 for j in range(n)] for e in range(2 ** n)],
            dtype=torch.float32,
        )
        indices = torch.arange(n).unsqueeze(1).expand(n, out_dim).contiguous()
        for param in ('softmax', 'independent', 'raw', 'iwp', 'warp'):
            layer = LutLayer(n, out_dim, n_inputs=n, parameterization=param,
                             connections=indices)
            with torch.no_grad():
                layer.weights.normal_()
            layer.gumbel = False  # warpのGumbel平滑を切り決定化
            layer.train()
            with torch.no_grad():
                y = layer(pats)  # (2^n, out_dim)
            table = soft_truth_table(
                layer.weights, param, n,
                gate_table=getattr(layer, 'gate_table', None),
                tau_warp=layer.tau_warp,
            )
            assert torch.allclose(y.T, table, atol=1e-5, rtol=0), (
                f'{param}: 表とforward頂点評価の不一致 '
                f'max={(y.T - table).abs().max():.3e}'
            )

    def test_soft_truth_table_16gate(self):
        """16ゲートモード(softmax N=2)も同様に一致"""
        from griffith_logic.layers import LutLayer
        from griffith_logic.regularization import soft_truth_table
        torch.manual_seed(41)
        n, out_dim = 2, 6
        pats = torch.tensor(
            [[(e >> j) & 1 for j in range(n)] for e in range(2 ** n)],
            dtype=torch.float32,
        )
        indices = torch.arange(n).unsqueeze(1).expand(n, out_dim).contiguous()
        layer = LutLayer(n, out_dim, n_inputs=2, parameterization='softmax',
                         connections=indices)
        with torch.no_grad():
            layer.weights.normal_()
        layer.train()
        with torch.no_grad():
            y = layer(pats)
        table = soft_truth_table(layer.weights, 'softmax16g', 2,
                                 gate_table=layer.gate_table)
        assert torch.allclose(y.T, table, atol=1e-5, rtol=0)

    def test_spectral_penalty_walks_model(self):
        """spectral_penalty: LutLayer×2+GroupSumの網で正・微分可能"""
        import torch.nn as nn

        from griffith_logic.group_sum import GroupSum
        from griffith_logic.layers import LutLayer
        from griffith_logic.regularization import spectral_penalty
        torch.manual_seed(43)
        model = nn.Sequential(
            LutLayer(8, 16, n_inputs=4, parameterization='warp'),
            LutLayer(16, 8, n_inputs=4, parameterization='warp'),
            GroupSum(k=2, tau=2.0),
        )
        with torch.no_grad():
            for m in model:
                if isinstance(m, LutLayer):
                    m.weights.normal_()
        pen = spectral_penalty(model)
        assert pen.item() > 0
        pen.backward()
        grads = [m.weights.grad for m in model if isinstance(m, LutLayer)]
        assert all(g is not None and g.abs().sum() > 0 for g in grads)
