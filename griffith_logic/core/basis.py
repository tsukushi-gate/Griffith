"""LUT基底関数(butterfly分解)とN=2の16ゲート真理値表。"""

import torch


def _compute_basis_butterfly(gathered: torch.Tensor, n_inputs: int) -> torch.Tensor:
    """
    基底関数をbutterfly分解で再帰的に計算。
    中間状態を保存せず、最終結果のみ返す（eval用）。
    """
    basis = torch.ones(
        gathered.shape[0], gathered.shape[1], 1,
        device=gathered.device, dtype=gathered.dtype,
    )
    for j in range(n_inputs):
        xj = gathered[:, :, j:j+1]
        basis = torch.cat([basis * (1.0 - xj), basis * xj], dim=-1)
    return basis


def _compute_wh_basis_butterfly(gathered: torch.Tensor, n_inputs: int) -> torch.Tensor:
    """
    Walsh-Hadamard基底をbutterfly分解で計算。

    入力を符号化 s = 1 - 2x（{0,1}→{+1,-1}）した上で、
    全部分集合の積 Π_{j∈S} s_j を再帰的に構築する。
    """
    s = 1.0 - 2.0 * gathered  # [0,1] → [+1,-1]
    basis = torch.ones(
        s.shape[0], s.shape[1], 1,
        device=s.device, dtype=s.dtype,
    )
    for j in range(n_inputs):
        sj = s[:, :, j:j+1]
        basis = torch.cat([basis, basis * sj], dim=-1)
    return basis


def _build_wh_entry_matrix(n_inputs: int) -> torch.Tensor:
    """WH係数→エントリロジット変換行列 H[i, e] = (-1)^popcount(i & e)。

    係数インデックスiはbutterfly順(bit j = s_jを含む部分集合)、
    エントリeは入力パターン(bit j = x_j、LSB=slot 0)。
    s_j = 1-2x_j より Π_{j∈i} s_j(e) = (-1)^popcount(i & e)。

    エントリロジット L = θ @ H は、WARPソフト式 θ·φ(s(x)) を
    binary頂点x=eで評価した値(τ=1)と厳密一致する。

    Returns:
        (2^n, 2^n) のfloat32テンソル
    """
    size = 2 ** n_inputs
    i = torch.arange(size).unsqueeze(1)
    e = torch.arange(size).unsqueeze(0)
    both = i & e
    parity = torch.zeros(size, size, dtype=torch.long)
    for j in range(n_inputs):
        parity += (both >> j) & 1
    return torch.where(parity % 2 == 1, -1.0, 1.0)


def _build_gate_table_n2() -> torch.Tensor:
    """
    N=2用の16ゲート真理値表（butterfly基底順）。

    difflogicの16種2入力Boolean関数に対応。
    gate_table[g][k] = ゲートgの、butterfly基底インデックスkに対する出力。

    butterfly基底順: k=0→(0,0), k=1→(1,0), k=2→(0,1), k=3→(1,1)
    ゲートID: 真理値表を(0,0),(0,1),(1,0),(1,1)順で読んだ2進数
      g=0: FALSE, g=1: AND, g=6: XOR, g=7: OR, g=14: NAND, g=15: TRUE

    Returns:
        (16, 4) のfloat32テンソル
    """
    table = torch.zeros(16, 4)
    for g in range(16):
        table[g, 0] = (g >> 3) & 1  # (0,0)
        table[g, 1] = (g >> 1) & 1  # (1,0)
        table[g, 2] = (g >> 2) & 1  # (0,1)
        table[g, 3] = g & 1         # (1,1)
    return table
