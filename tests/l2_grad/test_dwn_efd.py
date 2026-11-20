"""L2: DWN(Bacellar et al., ICML 2024)のEFD係数と勾配式の照合。

検証する数式(論文§3.1、公式実装 torch_dwn の efd_cuda_kernel.cu と照合済み):
    FD(テーブル勾配、one-hot):
        ∂A/∂U_i(U, a) = 1[i = δ(a)]
    EFD(入力勾配):
        ∂A/∂a_j(U, a) = Σ_{k∈{0,1}^n} (−1)^{1−k_j} · c(H(k,a,j)) · A(U, k)
        H(k,a,j) = ビットjを除外した k と a のハミング距離
        係数 c(H): 'paper' = 1/(H+1)、'official' = α·β^H (α=0.5·0.75^{n−1}, β=1/3)
    仕様(公式実装で解決した曖昧点):
        - A(U,k) はテーブルの生の実数値(二値化しない)
        - a は「>0」で二値化したアドレス(ビットj=スロットjがLSB側)
        - forward出力はテーブル実数値。二値化は後段のSTE(閾値0、backward素通し)

DWNFunctionは不連続(整数アドレスgather)のためgradcheckではなく、
素朴ループ実装との式照合(float64, 1e-12)で検証する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.layers.dwn import (
    BinarizeSTEFunction,
    DWNFunction,
    build_efd_coefficients,
)

# ============================================================
# 素朴な参照実装(明らかに正しい3重ループ)
# ============================================================

def paper_coef(h: int) -> float:
    """論文式(3)の係数: 1/(H+1)"""
    return 1.0 / (h + 1)


def official_coef_fn(n: int):
    """公式CUDAカーネルの係数: α·β^H (α=0.5·0.75^{n−1}, β=0.25/0.75)"""
    alpha = 0.5 * 0.75 ** (n - 1)
    beta = 0.25 / 0.75
    return lambda h: alpha * beta ** h


def naive_efd_coefficients(n: int, coef_fn) -> torch.Tensor:
    """C[a, j, k] = (−1)^{1−k_j} · c(H(k,a,j)) の素朴3重ループ実装"""
    size = 2 ** n
    c = torch.zeros(size, n, size, dtype=torch.float64)
    for a in range(size):
        for j in range(n):
            mask = ~(1 << j)
            for k in range(size):
                h = bin((a & mask) ^ (k & mask)).count('1')
                sign = 1.0 if (k >> j) & 1 else -1.0
                c[a, j, k] = sign * coef_fn(h)
    return c


def naive_address(gathered_row) -> int:
    """二値化(>0)してアドレス化。スロットjがビットj(LSB=スロット0)"""
    addr = 0
    for j, v in enumerate(gathered_row.tolist()):
        addr |= int(v > 0) << j
    return addr


def naive_dwn_forward(gathered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """forward: y[b,u] = U_u[δ(a)](テーブル実数値そのまま)"""
    batch, out, n = gathered.shape
    y = torch.zeros(batch, out, dtype=weights.dtype)
    for b in range(batch):
        for u in range(out):
            y[b, u] = weights[u, naive_address(gathered[b, u])]
    return y


def naive_dwn_backward(gathered, weights, grad_out, coef_fn):
    """backward: FD(one-hotテーブル勾配) + EFD(入力勾配)の素朴実装"""
    batch, out, n = gathered.shape
    grad_w = torch.zeros_like(weights)
    grad_g = torch.zeros_like(gathered)
    for b in range(batch):
        for u in range(out):
            a = naive_address(gathered[b, u])
            grad_w[u, a] += grad_out[b, u]  # FD: アクセスエントリのみ
            for j in range(n):
                mask = ~(1 << j)
                dda = 0.0
                for k in range(2 ** n):
                    h = bin((a & mask) ^ (k & mask)).count('1')
                    sign = 1.0 if (k >> j) & 1 else -1.0
                    dda += sign * coef_fn(h) * weights[u, k].item()
                grad_g[b, u, j] = dda * grad_out[b, u]
    return grad_g, grad_w


# ============================================================
# D2: EFD係数テンソル
# ============================================================

class TestEfdCoefficientsHandComputedN2:
    """n=2の全(a,j)を手計算表と照合(paper係数 1/(H+1))。

    j=0: Hはビット1のみ比較 → H = |k_1 − a_1|
    j=1: Hはビット0のみ比較 → H = |k_0 − a_0|
    符号: k_j=1 → +、k_j=0 → −
    """

    # 手計算表: HAND[a][j] = [k=0, k=1, k=2, k=3]
    HAND = [
        # a=0 (00)
        [[-1.0, 1.0, -0.5, 0.5], [-1.0, -0.5, 1.0, 0.5]],
        # a=1 (01)
        [[-1.0, 1.0, -0.5, 0.5], [-0.5, -1.0, 0.5, 1.0]],
        # a=2 (10)
        [[-0.5, 0.5, -1.0, 1.0], [-1.0, -0.5, 1.0, 0.5]],
        # a=3 (11)
        [[-0.5, 0.5, -1.0, 1.0], [-0.5, -1.0, 0.5, 1.0]],
    ]

    def test_full_table_paper(self):
        c = build_efd_coefficients(2, mode='paper', dtype=torch.float64)
        expected = torch.tensor(self.HAND, dtype=torch.float64)
        assert c.shape == (4, 2, 4)
        assert torch.allclose(c, expected, atol=TOL_F64_EQ, rtol=0)

    def test_full_table_official(self):
        """official係数: n=2でα=0.375、c(0)=0.375, c(1)=0.125。
        手計算表の {1 → 0.375, 1/2 → 0.125} 置換と一致する"""
        c = build_efd_coefficients(2, mode='official', dtype=torch.float64)
        hand = torch.tensor(self.HAND, dtype=torch.float64)
        expected = torch.where(hand.abs() == 1.0, hand.sign() * 0.375, hand.sign() * 0.125)
        assert torch.allclose(c, expected, atol=TOL_F64_EQ, rtol=0)


class TestEfdCoefficientsNaiveEquivalence:
    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_paper_matches_naive(self, n):
        c = build_efd_coefficients(n, mode='paper', dtype=torch.float64)
        expected = naive_efd_coefficients(n, paper_coef)
        assert torch.allclose(c, expected, atol=TOL_F64_EQ, rtol=0), (
            f'n={n}: max|Δ|={(c - expected).abs().max():.3e}'
        )

    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_official_matches_naive(self, n):
        c = build_efd_coefficients(n, mode='official', dtype=torch.float64)
        expected = naive_efd_coefficients(n, official_coef_fn(n))
        assert torch.allclose(c, expected, atol=TOL_F64_EQ, rtol=0)

    def test_custom_alpha_beta(self):
        """α, βを明示指定した指数係数"""
        c = build_efd_coefficients(3, mode='official', alpha=1.0, beta=0.5,
                                   dtype=torch.float64)
        expected = naive_efd_coefficients(3, lambda h: 0.5 ** h)
        assert torch.allclose(c, expected, atol=TOL_F64_EQ, rtol=0)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            build_efd_coefficients(2, mode='fd')


# ============================================================
# D3: DWNFunction(forward gather + FD/EFD backward)の式照合
# ============================================================

def _make_case(n, batch=7, out=5, seed_val=3):
    torch.manual_seed(seed_val)
    # {0,1}二値入力(実運用と同じ)+境界値を混ぜる
    gathered = (torch.rand(batch, out, n, dtype=torch.float64) > 0.5).double()
    weights = torch.rand(out, 2 ** n, dtype=torch.float64) * 2 - 1
    grad_out = torch.randn(batch, out, dtype=torch.float64)
    return gathered, weights, grad_out


class TestDWNFunctionForward:
    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_forward_matches_naive(self, n):
        gathered, weights, _ = _make_case(n)
        c = build_efd_coefficients(n, dtype=torch.float64)
        y = DWNFunction.apply(gathered, weights, c)
        expected = naive_dwn_forward(gathered, weights)
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0)

    def test_forward_returns_raw_table_values(self):
        """出力はテーブルの実数値そのまま(二値化はSTEが担当)"""
        weights = torch.tensor([[-0.7, 0.3, 0.9, -0.2]], dtype=torch.float64)
        c = build_efd_coefficients(2, dtype=torch.float64)
        for addr, (x0, x1) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            g = torch.tensor([[[float(x0), float(x1)]]], dtype=torch.float64)
            y = DWNFunction.apply(g, weights, c)
            assert y.item() == weights[0, addr].item()

    def test_address_binarization_threshold_zero(self):
        """アドレス二値化は「>0」(公式カーネル準拠): −0.5→0, 0.0→0, 0.7→1"""
        weights = torch.arange(8, dtype=torch.float64).unsqueeze(0)  # U[a]=a
        c = build_efd_coefficients(3, dtype=torch.float64)
        g = torch.tensor([[[-0.5, 0.0, 0.7]]], dtype=torch.float64)
        y = DWNFunction.apply(g, weights, c)
        assert y.item() == 4.0  # (0,0,1) → addr=4


class TestDWNFunctionBackward:
    @pytest.mark.parametrize('mode', ['paper', 'official'])
    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_grads_match_naive(self, n, mode):
        gathered, weights, grad_out = _make_case(n)
        coef_fn = paper_coef if mode == 'paper' else official_coef_fn(n)
        c = build_efd_coefficients(n, mode=mode, dtype=torch.float64)

        g = gathered.clone().requires_grad_(True)
        w = weights.clone().requires_grad_(True)
        y = DWNFunction.apply(g, w, c)
        grad_g, grad_w = torch.autograd.grad(y, (g, w), grad_outputs=grad_out)

        grad_g_ref, grad_w_ref = naive_dwn_backward(gathered, weights, grad_out, coef_fn)
        assert torch.allclose(grad_w, grad_w_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_weights不一致: {(grad_w - grad_w_ref).abs().max():.3e}'
        )
        assert torch.allclose(grad_g, grad_g_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_gathered不一致: {(grad_g - grad_g_ref).abs().max():.3e}'
        )

    def test_table_grad_is_one_hot(self):
        """FD: アクセスしたエントリのみ勾配1が流れる(EFD的分配はしない)"""
        c = build_efd_coefficients(2, dtype=torch.float64)
        w = (torch.rand(1, 4, dtype=torch.float64) * 2 - 1).requires_grad_(True)
        g = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)  # addr=1
        y = DWNFunction.apply(g, w, c)
        y.backward(torch.ones_like(y))
        assert w.grad.tolist() == [[0.0, 1.0, 0.0, 0.0]]

    def test_input_grad_uses_real_table_values(self):
        """EFD: A(U,k)は生の実数テーブル値(二値化後ではない)を使う。

        n=1: ∂A/∂a_0 = c(0)·(U_1 − U_0)。二値化値を使うと符号情報が失われる"""
        c = build_efd_coefficients(1, dtype=torch.float64)
        w = torch.tensor([[-0.25, 0.75]], dtype=torch.float64)
        g = torch.tensor([[[1.0]]], dtype=torch.float64, requires_grad=True)
        y = DWNFunction.apply(g, w, c)
        y.backward(torch.ones_like(y))
        assert abs(g.grad.item() - (0.75 - (-0.25))) < TOL_F64_GRAD


class TestBinarizeSTE:
    def test_forward_threshold_zero(self):
        x = torch.tensor([-1.0, -0.001, 0.0, 0.001, 1.0], dtype=torch.float64)
        y = BinarizeSTEFunction.apply(x)
        assert y.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]

    def test_backward_identity(self):
        """STE backward: 勾配を素通し(hardtanhクリップなし、公式実装準拠)"""
        x = torch.tensor([-5.0, 0.5, 5.0], dtype=torch.float64, requires_grad=True)
        y = BinarizeSTEFunction.apply(x)
        grad_out = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
        y.backward(grad_out)
        assert torch.equal(x.grad, grad_out)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
