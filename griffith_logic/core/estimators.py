"""IWP(Input-Wise Parametrization)の推定器 ρ: ℝ→[0,1]。

論文 "Light Differentiable Logic Gate Networks" (arXiv:2510.03250, ICLR 2026)
の IWP は、真理値表の各エントリ確率 ω_k ∈ [0,1] を実数パラメータ Ω_k から
推定器 ρ で写す:

    ω_k = ρ(Ω_k),   g_ω(p, q) = Σ_k ω_k · e_k(p, q)      (Eq.11)

論文 §C.1.1 が実験で実際に採用したのは sigmoid ではなく正弦推定器

    sin01(Ω) = 0.5 + 0.5·sin(Ω)

である(周期的で飽和せず、勾配が消えない)。本モジュールは ρ とその導関数
dρ/dΩ を提供し、カスタムautograd(core/functions.py)と参照実装
(core/reference.py)の双方から共有する。
"""

import torch

# 対応する推定器の一覧(既定は後方互換のため 'sigmoid')
ESTIMATORS = ('sigmoid', 'sin')


def estimator_rho(w: torch.Tensor, estimator: str = 'sigmoid') -> torch.Tensor:
    """推定器 ρ: ℝ→[0,1] を適用する。

    Args:
        w: 実数パラメータ Ω(任意形状)
        estimator: 'sigmoid' → σ(Ω) / 'sin' → 0.5 + 0.5·sin(Ω)

    Returns:
        ω = ρ(Ω)(wと同形状、値域[0,1])

    Raises:
        ValueError: 未知のestimator
    """
    if estimator == 'sigmoid':
        return torch.sigmoid(w)
    if estimator == 'sin':
        return 0.5 + 0.5 * torch.sin(w)
    raise ValueError(f'未知のestimator: {estimator}')


def estimator_rho_prime(
    w: torch.Tensor, estimator: str = 'sigmoid',
) -> torch.Tensor:
    """推定器の導関数 dρ/dΩ を返す。

    Args:
        w: 実数パラメータ Ω(任意形状)
        estimator: 'sigmoid' → σ(Ω)(1-σ(Ω)) / 'sin' → 0.5·cos(Ω)

    Returns:
        dρ/dΩ(wと同形状)

    Raises:
        ValueError: 未知のestimator
    """
    if estimator == 'sigmoid':
        s = torch.sigmoid(w)
        return s * (1.0 - s)
    if estimator == 'sin':
        return 0.5 * torch.cos(w)
    raise ValueError(f'未知のestimator: {estimator}')
