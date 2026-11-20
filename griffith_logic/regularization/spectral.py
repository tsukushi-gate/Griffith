"""Spectral Regularization(DWN: Bacellar et al., ICML 2024, §3.4)。

n入力擬ブール関数 f: {0,1}^n → ℝ のL2スペクトルノルム:

    specnorm(f) = (1/2^n) ‖{ Σ_{x∈{0,1}^n} f(x) Π_{i∈S}(2x_i−1) | S ⊆ [n] }‖₂

これは f のフーリエ係数(WH変換)ベクトルのL2ノルムに等しい。
u個のLUTをまとめた行列 L ∈ [−1,1]^{u×2^n} に対しては係数行列

    C_ij = (1/2^n) Π_{a∈{b | i_b=1}} (2j_a − 1)

を事前計算して specnorm(L) = ‖L·C‖₂(行列2ノルム=最大特異値)で評価する。
u=1のときは行ベクトルの最大特異値=ベクトルL2ノルムとなり単一関数の定義と一致。

性質(テストで検証):
    - C·C^T = I/2^n(WH基底の直交性)
    - パーセバル恒等式: specnorm(L) = ‖L‖₂/√(2^n)
    - LUTのテーブル値そのものではなくスペクトル構造を通じて滑らかさを
      正則化する(単一入力の反転に対する頑健性を高める)

注: 公式実装(torch_dwn)には本正則化のコードは存在せず、論文の定義のみが
出典。使用例: loss = task_loss + λ * spectral_norm(layer.luts, layer.n_inputs)
"""

import torch


def _popcount(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """整数テンソルの立っているビット数(下位n_bitsのみ)"""
    shifts = torch.arange(n_bits, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).sum(dim=-1)


def build_spectral_coefficient_matrix(
    n_inputs: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = 'cpu',
) -> torch.Tensor:
    """係数行列 C_ij = (1/2^n) Π_{a: bit(i,a)=1} (2j_a − 1) を構築する。

    Π_{a: i_a=1}(2j_a−1) = (−1)^{#{a: i_a=1 ∧ j_a=0}} = (−1)^{popcount(i & ~j)}

    Returns:
        (2^n, 2^n) 係数行列
    """
    size = 2 ** n_inputs
    i = torch.arange(size).unsqueeze(1)
    j = torch.arange(size).unsqueeze(0)
    parity = _popcount(i & ~j, n_inputs) % 2
    signs = torch.where(parity == 1, -1.0, 1.0).to(torch.float64)
    return (signs / size).to(dtype=dtype, device=device)


def spectral_norm(
    tables: torch.Tensor,
    n_inputs: int,
    c: torch.Tensor | None = None,
) -> torch.Tensor:
    """LUTテーブル行列のスペクトルノルム specnorm(L) = ‖L·C‖₂ を計算する。

    Args:
        tables: (u, 2^n) LUTテーブル(実数値、通常は[−1,1])
        n_inputs: LUT入力数 n
        c: 事前計算済み係数行列(Noneなら都度構築)

    Returns:
        スカラーテンソル(微分可能、正則化項として損失に加算できる)
    """
    size = 2 ** n_inputs
    if tables.shape[-1] != size:
        raise ValueError(
            f'tablesの幅({tables.shape[-1]})が2^n_inputs({size})と一致しない'
        )
    if c is None:
        c = build_spectral_coefficient_matrix(
            n_inputs, dtype=tables.dtype, device=tables.device,
        )
    return torch.linalg.matrix_norm(tables @ c.to(tables.dtype), ord=2)
