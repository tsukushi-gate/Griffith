"""L1: 論文IWP(Input-Wise Parametrization)の数学的性質。

論文 "Light Differentiable Logic Gate Networks" (arXiv:2510.03250, ICLR 2026)。

    g_ω(x) = Σ_k ω_k · e_k,   e_k = Π_j (x_j if bit(k,j)=1 else 1-x_j)   (Eq.11)
    ω_k = ρ(Ω_k),  ρ = sigmoid または sin01(Ω)=0.5+0.5·sin(Ω)(論文§C.1.1)

**重要な前提(事前監査の結論)**:
    論文IWPは当ライブラリの `parameterization='independent'` と数学的に
    完全一致する。既存の `parameterization='iwp'`(リテラル積=単一ミンターム)
    は論文IWPでは**ない**。両者が別物であることは
    `TestIwpIsNotPaperIwp` が回帰テストとして固定する。

検証内容:
    推定器ρ/ρ'の値域・解析微分一致
    16ゲート全数の表現力等価(表現力の損失がないこと)
    丸め誤差の最小性(§3.3, §E.3)
    Eq.(11) の素朴書き下しとの直接照合
    Residual Initialization のω空間への翻訳(sin推定器)
"""

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.core import reference_lut_forward
from griffith_logic.core.estimators import (
    ESTIMATORS,
    estimator_rho,
    estimator_rho_prime,
)
from griffith_logic.lut_layer import LutLayer, _build_gate_table_n2

# 推定器ごとの「確実にON/OFF」なΩ値
#   sigmoid: ±5.0     → ρ = 0.9933 / 0.0067
#   sin    : ±π/2     → ρ = 1.0    / 0.0(厳密)
ON_OFF = {
    'sigmoid': (5.0, -5.0),
    'sin': (torch.pi / 2, -torch.pi / 2),
}

# Residual Initialization の既定パラメータ(論文§C.1.1 / API契約)
RI_MU = 1.2
RI_SIGMA = 0.25


def _gate_truth(g: int, a: int, b: int) -> int:
    """ゲートIDの定義(独立実装): gの真理値表を(0,0),(0,1),(1,0),(1,1)順で
    読んだ2進数。g = 8·f(0,0) + 4·f(0,1) + 2·f(1,0) + 1·f(1,1)
    """
    pos = {(0, 0): 3, (0, 1): 2, (1, 0): 1, (1, 1): 0}[(a, b)]
    return (g >> pos) & 1


def _naive_eq11(x, weights, indices, n_inputs, estimator):
    """論文 Eq.(11) の素朴な書き下し(ライブラリのbasis関数を使わない独立実装)。

        y[b,o] = Σ_k ρ(Ω[o,k]) · Π_j (x_j if bit(k,j)=1 else 1-x_j)

    Args:
        x: (batch, in_dim) float64
        weights: (out_dim, 2^n_inputs) の実数パラメータ Ω
        indices: (n_inputs, out_dim)
        n_inputs: LUT入力数 N
        estimator: 'sigmoid' | 'sin'

    Returns:
        (batch, out_dim) の出力
    """
    batch = x.shape[0]
    out_dim = weights.shape[0]
    omega = estimator_rho(weights, estimator)
    y = torch.zeros(batch, out_dim, dtype=x.dtype)
    for b in range(batch):
        for o in range(out_dim):
            acc = 0.0
            for k in range(2 ** n_inputs):
                e_k = 1.0
                for j in range(n_inputs):
                    xj = x[b, indices[j, o]].item()
                    e_k *= xj if ((k >> j) & 1) else (1.0 - xj)
                acc += omega[o, k].item() * e_k
            y[b, o] = acc
    return y


class TestEstimatorProperties:
    """推定器 ρ とその導関数 ρ' の基本性質"""

    @pytest.mark.parametrize('estimator', ESTIMATORS)
    def test_range_in_unit_interval(self, estimator):
        """ρ の値域が [0,1] に収まる(ω=ρ(Ω)は確率として解釈される)"""
        w = torch.linspace(-50.0, 50.0, 4001, dtype=torch.float64)
        rho = estimator_rho(w, estimator)
        assert torch.all(rho >= 0.0), f'{estimator}: ρ<0 が存在'
        assert torch.all(rho <= 1.0), f'{estimator}: ρ>1 が存在'

    @pytest.mark.parametrize('estimator', ESTIMATORS)
    def test_derivative_matches_autograd(self, estimator, seed):
        """ρ' の解析式が torch.autograd.grad と float64 で一致"""
        w = (torch.rand(512, dtype=torch.float64) * 20.0 - 10.0)
        wr = w.clone().requires_grad_(True)
        rho = estimator_rho(wr, estimator)
        (grad,) = torch.autograd.grad(rho.sum(), (wr,))
        analytic = estimator_rho_prime(w, estimator)
        assert torch.allclose(grad, analytic, atol=TOL_F64_GRAD, rtol=0), (
            f'{estimator}: ρ\'不一致 {(grad - analytic).abs().max():.3e}'
        )

    def test_known_values(self):
        """代表点での値(仕様の固定): sigmoid(0)=0.5, sin01(0)=0.5, sin01(π/2)=1"""
        zero = torch.zeros(1, dtype=torch.float64)
        assert abs(estimator_rho(zero, 'sigmoid').item() - 0.5) < TOL_F64_EQ
        assert abs(estimator_rho(zero, 'sin').item() - 0.5) < TOL_F64_EQ
        half_pi = torch.tensor([torch.pi / 2], dtype=torch.float64)
        assert abs(estimator_rho(half_pi, 'sin').item() - 1.0) < TOL_F64_EQ


class TestSixteenGateExpressivity:
    """IWPが2入力16ゲートすべてを表現できる(表現力の損失がない)"""

    @pytest.mark.parametrize('estimator', ESTIMATORS)
    def test_all_16_gates_exact(self, estimator):
        """16通りの真理値表 × 4入力パターン = 64ケースで hard eval と厳密一致"""
        on, off = ON_OFF[estimator]
        table = _build_gate_table_n2()  # (16, 4) butterfly基底順
        layer = LutLayer(
            2, 16, n_inputs=2, parameterization='independent',
            estimator=estimator,
        )
        layer.indices = torch.tensor([[0] * 16, [1] * 16], dtype=torch.int64)
        with torch.no_grad():
            layer.weights.copy_(torch.where(table > 0.5, on, off))
        layer.eval()

        for a, b in itertools.product([0, 1], repeat=2):
            y = layer(torch.tensor([[float(a), float(b)]]))
            for g in range(16):
                assert y[0, g].item() == _gate_truth(g, a, b), (
                    f'{estimator}: gate={g}, 入力=({a},{b}): got {y[0, g].item()}'
                )

    @pytest.mark.parametrize('estimator', ESTIMATORS)
    def test_realizable_truth_tables_count_is_16(self, estimator):
        """実現可能な真理値表の集合の濃度が厳密に16(=全Boolean関数)"""
        realized = _realizable_tables_independent(estimator)
        assert len(realized) == 16, (
            f'{estimator}: independentが実現できる真理値表は{len(realized)}種'
        )


class TestRoundingMinimality:
    """論文 §3.3 / §E.3: 0.5閾値の丸めが誤差最小であること"""

    def test_threshold_half_minimizes_error(self, seed):
        """任意の ω∈[0,1] で b*=(ω>0.5) が min_b |ω-b| を達成する"""
        omega = torch.rand(20000, dtype=torch.float64)
        b_star = (omega > 0.5).to(torch.float64)
        err_star = (omega - b_star).abs()
        err_other = (omega - (1.0 - b_star)).abs()
        assert torch.all(err_star <= err_other), (
            f'丸め誤差が最小でないサンプルが存在: '
            f'{int((err_star > err_other).sum())}件'
        )
        # 丸め誤差の上限は 0.5(ω=0.5ちょうどのときのみ達成)
        assert err_star.max().item() <= 0.5

    def test_each_entry_rounds_independently(self, seed):
        """各エントリの丸めは独立(結合制約がないので総誤差も最小)"""
        omega = torch.rand(64, 16, dtype=torch.float64)
        rounded = (omega > 0.5).to(torch.float64)
        # 全2^4通りの代替(先頭4列)と比べて総誤差が最小であることを確認
        base = (omega[:, :4] - rounded[:, :4]).abs().sum(dim=1)
        for bits in itertools.product([0.0, 1.0], repeat=4):
            alt = torch.tensor(bits, dtype=torch.float64)
            cand = (omega[:, :4] - alt).abs().sum(dim=1)
            assert torch.all(base <= cand + TOL_F64_EQ), '総誤差が最小でない'


class TestEq11Equivalence:
    """参照実装が論文 Eq.(11) の素朴書き下しと一致する"""

    @pytest.mark.parametrize('estimator', ESTIMATORS)
    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_reference_matches_naive_formula(self, n, estimator):
        torch.manual_seed(11)
        in_dim, out_dim, batch = max(8, 2 * n), 3, 4
        weights = torch.randn(out_dim, 2 ** n, dtype=torch.float64)
        indices = torch.randint(0, in_dim, (n, out_dim), dtype=torch.int64)
        x = torch.rand(batch, in_dim, dtype=torch.float64)

        y = reference_lut_forward(
            x, weights, indices, n, 'independent', estimator=estimator,
        )
        expected = _naive_eq11(x, weights, indices, n, estimator)
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0), (
            f'{estimator} n={n}: Eq.(11)不一致 {(y - expected).abs().max():.3e}'
        )

    @pytest.mark.parametrize('estimator', ESTIMATORS)
    @pytest.mark.parametrize('n', [2, 4])
    def test_partition_of_unity(self, n, estimator):
        """基底 e_k は分割単位: ω_k を全て1にすると出力は恒等的に1"""
        torch.manual_seed(13)
        in_dim, out_dim, batch = max(8, 2 * n), 3, 4
        on = ON_OFF[estimator][0]
        weights = torch.full((out_dim, 2 ** n), on, dtype=torch.float64)
        indices = torch.randint(0, in_dim, (n, out_dim), dtype=torch.int64)
        x = torch.rand(batch, in_dim, dtype=torch.float64)
        y = reference_lut_forward(
            x, weights, indices, n, 'independent', estimator=estimator,
        )
        omega_on = estimator_rho(
            torch.tensor([on], dtype=torch.float64), estimator,
        ).item()
        assert torch.allclose(
            y, torch.full_like(y, omega_on), atol=TOL_F64_EQ, rtol=0,
        ), f'{estimator} n={n}: 分割単位性が崩れている'


class TestResidualInitOmegaSpace:
    """sin推定器でのResidual Initialization(Ω~N(±μ,σ))のω空間での意味"""

    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_omega_concentrates_on_passthrough(self, n, seed):
        """bit(n-1)==1 のエントリで ω が1側、他で0側へ集中する。

        個々のΩは σ=0.25 のばらつきを持つ(ρ=0.9 に対応するΩ=0.927 は
        μ=1.2 から -1.09σ)ため、全数に >0.9 を課すことはできない。
        よって「全数で離散化契約 ρ>0.5(⇔Ω>0 側)」+「平均で >0.9 / <0.1」
        の二段で集中を検証する。μ=1.2 は 0 から 4.8σ 離れており、
        全数の符号契約は数学的に十分な余裕がある。
        """
        layer = LutLayer(
            max(8, 2 * n), 8, n_inputs=n, parameterization='independent',
            estimator='sin', residual_init=True, ri_mu=RI_MU, ri_sigma=RI_SIGMA,
        )
        omega = estimator_rho(layer.weights.detach().double(), 'sin')
        e = torch.arange(2 ** n)
        mask = ((e >> (n - 1)) & 1).bool()

        on_vals, off_vals = omega[:, mask], omega[:, ~mask]
        assert torch.all(on_vals > 0.5), (
            f'n={n}: pass-through側にρ<=0.5のエントリ '
            f'(最小 {on_vals.min().item():.4f})'
        )
        assert torch.all(off_vals < 0.5), (
            f'n={n}: 非pass-through側にρ>=0.5のエントリ '
            f'(最大 {off_vals.max().item():.4f})'
        )
        assert on_vals.mean().item() > 0.9, (
            f'n={n}: pass-through側の平均ρ={on_vals.mean().item():.4f} <= 0.9'
        )
        assert off_vals.mean().item() < 0.1, (
            f'n={n}: 非pass-through側の平均ρ={off_vals.mean().item():.4f} >= 0.1'
        )

    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_hard_table_is_exact_passthrough(self, n, seed):
        """hard離散化した真理値表が厳密に最終入力のpass-throughになる"""
        out_dim = 5
        layer = LutLayer(
            n, out_dim, n_inputs=n, parameterization='independent',
            estimator='sin', residual_init=True, ri_mu=RI_MU, ri_sigma=RI_SIGMA,
        )
        layer.indices = (
            torch.arange(n, dtype=torch.int64).unsqueeze(1).expand(n, out_dim)
        ).contiguous()
        layer.eval()
        for k in range(2 ** n):
            x = torch.tensor([[float((k >> j) & 1) for j in range(n)]])
            expected = float((k >> (n - 1)) & 1)
            y = layer(x)
            assert torch.all(y == expected), (
                f'n={n} k={k}: pass-throughでない {y.tolist()}'
            )

    def test_sigmoid_residual_init_stays_deterministic(self, seed):
        """estimator='sigmoid' のRIは従来どおり決定的な±5.0(後方互換)"""
        n = 4
        layer = LutLayer(
            8, 6, n_inputs=n, parameterization='independent',
            estimator='sigmoid', residual_init=True,
        )
        e = torch.arange(2 ** n)
        mask = ((e >> (n - 1)) & 1).bool()
        w = layer.weights.detach()
        assert torch.all(w[:, mask] == 5.0), 'sigmoid RI: ON側が+5.0でない'
        assert torch.all(w[:, ~mask] == -5.0), 'sigmoid RI: OFF側が-5.0でない'


class TestEstimatorRestrictedToIndependent:
    """estimator は independent 以外では未対応(契約の明示)"""

    @pytest.mark.parametrize('param', ['softmax', 'iwp', 'warp', 'raw'])
    def test_non_independent_rejects_sin(self, param):
        with pytest.raises(NotImplementedError):
            LutLayer(8, 4, n_inputs=2, parameterization=param, estimator='sin')


def _realizable_tables_independent(estimator):
    """independentが hard eval で実現できる2入力真理値表の集合"""
    on, off = ON_OFF[estimator]
    patterns = [(0, 0), (1, 0), (0, 1), (1, 1)]
    realized = set()
    for bits in itertools.product([0, 1], repeat=4):
        layer = LutLayer(
            2, 1, n_inputs=2, parameterization='independent',
            estimator=estimator,
        )
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)
        with torch.no_grad():
            layer.weights.copy_(torch.tensor(
                [[on if b else off for b in bits]]
            ))
        layer.eval()
        realized.add(tuple(
            layer(torch.tensor([[float(a), float(b)]])).item()
            for a, b in patterns
        ))
    return realized


def _realizable_tables_iwp():
    """既存の parameterization='iwp'(リテラル積)が実現できる真理値表の集合"""
    patterns = [(0, 0), (1, 0), (0, 1), (1, 1)]
    realized = set()
    for signs in itertools.product([5.0, -5.0], repeat=2):
        layer = LutLayer(2, 1, n_inputs=2, parameterization='iwp')
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)
        with torch.no_grad():
            layer.weights.copy_(torch.tensor([list(signs)]))
        layer.eval()
        realized.add(tuple(
            layer(torch.tensor([[float(a), float(b)]])).item()
            for a, b in patterns
        ))
    return realized


class TestIwpIsNotPaperIwp:
    """回帰テスト: 既存 `parameterization='iwp'` は論文IWPではない。

    論文IWP = `parameterization='independent'`(16ゲート全表現)。
    既存 'iwp' はリテラル積(単一ミンターム)で、2入力では4種しか作れない。
    両者を混同した設計変更が入ったらここで落ちる。
    """

    def test_iwp_realizes_only_4_of_16(self):
        realized = _realizable_tables_iwp()
        assert len(realized) == 4, (
            f"既存'iwp'の実現可能真理値表が4種でない: {len(realized)}種"
        )

    def test_iwp_cannot_represent_passthrough_or_xor(self):
        """既存'iwp'では素通し G(A,B)=A / G(A,B)=B と XOR/XNOR が表現不能。

        ゲートIDは `_gate_truth` の規約(g = 8·f(0,0)+4·f(0,1)+2·f(1,0)+f(1,1))。
            3 = 素通しA, 5 = 素通しB, 6 = XOR, 9 = XNOR
        """
        realized = _realizable_tables_iwp()
        patterns = [(0, 0), (1, 0), (0, 1), (1, 1)]
        names = {3: '素通しA', 5: '素通しB', 6: 'XOR', 9: 'XNOR'}
        for g, name in names.items():
            table = tuple(float(_gate_truth(g, a, b)) for a, b in patterns)
            assert table not in realized, (
                f"既存'iwp'が gate={g}({name}) を実現してしまった: {table}"
            )

    def test_independent_strictly_richer_than_iwp(self):
        iwp_tables = _realizable_tables_iwp()
        indep_tables = _realizable_tables_independent('sigmoid')
        assert iwp_tables < indep_tables, (
            "既存'iwp'の表現可能集合がindependentの真部分集合でない"
        )
        assert len(indep_tables - iwp_tables) == 12, (
            'independentのみが実現できる関数が12種でない'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
