"""Spectral正則化のモデル適用ヘルパ。

spectral.pyの定義(DWN §3.4)を訓練ループへ配管するための:
    1. パラメータ化別のsoft真理値表抽出(訓練時のforward意味論と一致)
    2. Gram形式の高効率スペクトルノルム
        C·Cᵀ = I/2^n (WH基底の直交性)より
        specnorm(L) = ‖L·C‖₂ = σ_max(L)/√(2^n) = √(λ_max(LᵀL)/2^n)
        LᵀLは(2^n, 2^n)なのでout_dimが百万級でも数msで計算できる
        (spectral_norm()の full SVD は大規模denseで数秒〜)
    3. モデル走査してLutLayer/LutConv2dのペナルティ和を返すspectral_penalty

使用例:
    loss = task_loss + lam * spectral_penalty(model)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.basis import _build_wh_entry_matrix

_EIG_EPS = 1e-12


def soft_truth_table(
    weights: torch.Tensor,
    parameterization: str,
    n_inputs: int,
    *,
    gate_table: torch.Tensor | None = None,
    tau_warp: float = 1.0,
) -> torch.Tensor:
    """パラメータ化別のsoft真理値表 T ∈ [0,1]^(..., 2^n) を返す。

    訓練時のforwardが実効的に使う表と同一の式(勾配が正則化から
    パラメータへ流れる)。weightsの先頭次元は任意(dense: (out, ·)、
    conv: (nodes, ·) 等)。

    Args:
        weights: (..., n_params) 学習パラメータ
        parameterization: 'softmax' | 'softmax16g' | 'independent' |
            'raw' | 'iwp' | 'warp'
        n_inputs: LUT入力数N
        gate_table: (16, 4)。16ゲートモード(softmax N=2 / softmax16g)で必須
        tau_warp: WARP温度
    """
    n_entries = 2 ** n_inputs

    if parameterization in ('softmax', 'softmax16g'):
        if gate_table is not None:
            return F.softmax(weights, dim=-1) @ gate_table.to(weights.dtype)
        return F.softmax(weights, dim=-1)
    if parameterization == 'independent':
        return torch.sigmoid(weights)
    if parameterization == 'raw':
        return weights
    if parameterization == 'warp':
        h = _build_wh_entry_matrix(n_inputs).to(
            device=weights.device, dtype=weights.dtype)
        return torch.sigmoid((weights @ h) / tau_warp)
    if parameterization == 'iwp':
        # T[..., e] = Π_j (p_j if bit(e,j) else 1-p_j)
        p = torch.sigmoid(weights)  # (..., n_inputs)
        table = torch.ones(*weights.shape[:-1], 1,
                           device=weights.device, dtype=weights.dtype)
        for j in range(n_inputs):
            pj = p[..., j:j + 1]
            table = torch.cat([table * (1.0 - pj), table * pj], dim=-1)
        assert table.shape[-1] == n_entries
        return table
    raise ValueError(f'未知のparameterization: {parameterization}')


def spectral_norm_gram(tables: torch.Tensor, n_inputs: int) -> torch.Tensor:
    """specnorm(L) = √(λ_max(LᵀL)/2^n) のGram形式(spectral_normと等価)。

    tablesは[0,1]のsoft真理値表でも[−1,1]表でも受けるが、
    DWN定義との一致にはL=2T−1を渡すこと(spectral_penaltyが行う)。
    """
    size = 2 ** n_inputs
    if tables.shape[-1] != size:
        raise ValueError(
            f'tablesの幅({tables.shape[-1]})が2^n_inputs({size})と一致しない'
        )
    flat = tables.reshape(-1, size)
    gram = flat.T @ flat  # (2^n, 2^n)
    lam_max = torch.linalg.eigvalsh(gram)[-1].clamp_min(_EIG_EPS)
    return torch.sqrt(lam_max / size)


def iter_spectral_layers(model: torch.nn.Module):
    """spectral正則化の対象層(LutLayer / LutConv2d)を列挙する。"""
    from ..conv import LutConv2d
    from ..layers import LutLayer

    for m in model.modules():
        if isinstance(m, LutLayer | LutConv2d):
            yield m


# ============================================================
# 縮約耐性(irreducibility)正則化 — 先行実験
# ============================================================

# 論理縮約で必ず消えるゲート(16ゲートID規約 core/basis.py _build_gate_table_n2)。
# 隠れ層: 定数2種 + pass-through/否定4種。
# 最終層: 定数はクラス和のバイアスとして意味があるので残し、1入力関数だけを罰する。
IRREDUCIBLE_HIDDEN_GATE_IDS = (0, 15, 3, 5, 10, 12)
IRREDUCIBLE_FINAL_GATE_IDS = (3, 5, 10, 12)


def gate_cost_vector(gate_ids) -> tuple[float, ...]:
    """ゲートIDの集合を16次元の0/1コストベクトルへ変換する。"""
    ids = {int(g) for g in gate_ids}
    if any(not 0 <= g <= 15 for g in ids):
        raise ValueError(f'ゲートIDは0〜15の範囲: {sorted(ids)}')
    return tuple(1.0 if g in ids else 0.0 for g in range(16))


IRREDUCIBLE_COST_HIDDEN = gate_cost_vector(IRREDUCIBLE_HIDDEN_GATE_IDS)
IRREDUCIBLE_COST_FINAL = gate_cost_vector(IRREDUCIBLE_FINAL_GATE_IDS)


def iter_irreducibility_layers(model: torch.nn.Module) -> list:
    """縮約耐性正則化の対象層(16ゲートモードのLutLayer)をモデル順に列挙する。"""
    from ..layers import LutLayer

    return [m for m in model.modules() if isinstance(m, LutLayer)]


def _as_cost_tensor(cost, ref: torch.Tensor) -> torch.Tensor:
    c = torch.as_tensor(cost, device=ref.device, dtype=ref.dtype)
    if c.shape != (16,):
        raise ValueError(f'コストベクトルは16次元: shape={tuple(c.shape)}')
    return c


def irreducibility_penalty(
    model: torch.nn.Module,
    cost_hidden=IRREDUCIBLE_COST_HIDDEN,
    cost_final=IRREDUCIBLE_COST_FINAL,
) -> torch.Tensor:
    """縮約されやすいゲート種へ置かれた確率質量のペナルティ(先行実験)。

        R = (1/N_h) Σ_n Σ_i p_{n,i} · c_i

    - p_{n,i} は訓練時 forward と同じ16ゲートsoftmax(温度1.0)。
      `soft_truth_table(..., 'softmax', ...)` から取るので forward との
      一致が構造的に保証される。`gate_mask` があれば −inf 加算後の
      ロジットを使うのでマスク位置の p は厳密に 0 = 寄与 0。
    - c_i は16次元の0/1コスト。**最終LutLayerだけ** cost_final を使う
      (定数はクラス和のバイアスとして残す。Mommen 方式)。
    - N_h は対象層の全ゲート数(Σ out_dim)なので、R ∈ [0, 1] で層数・幅に
      依らないスケールになる。

    Args:
        model: LutLayer を含む nn.Module。
        cost_hidden: 隠れ層の16次元コスト(既定 = 定数2種 + pass-through4種)。
        cost_final: 最終LutLayerの16次元コスト(既定 = pass-through4種)。

    Returns:
        スカラーテンソル(微分可能)。

    Raises:
        ValueError: 対象 LutLayer が無い / 16ゲートモード
            (n_inputs=2 かつ parameterization='softmax')でない層がある。
    """
    layers = iter_irreducibility_layers(model)
    if not layers:
        raise ValueError(
            'irreducibility_penalty の対象 LutLayer がモデルに存在しません'
        )
    for layer in layers:
        if not getattr(layer, '_use_16gate', False):
            raise ValueError(
                'irreducibility_penalty は16ゲートモード'
                "(n_inputs=2 かつ parameterization='softmax')専用です: "
                f'n_inputs={layer.n_inputs}, '
                f'parameterization={layer.parameterization!r}'
            )

    total = None
    n_gates = 0
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        weights = layer._masked_weights()
        cost = _as_cost_tensor(
            cost_final if i == last else cost_hidden, weights
        )
        p = soft_truth_table(weights, 'softmax', layer.n_inputs)
        contrib = (p * cost).sum()
        total = contrib if total is None else total + contrib
        n_gates += layer.out_dim
    return total / n_gates


def spectral_penalty(model: torch.nn.Module) -> torch.Tensor:
    """モデル内の全LUT層のスペクトルノルム和(微分可能)。

    各層のsoft真理値表T([0,1])を L=2T−1([−1,1]) に写してから
    specnormを取る(DWN §3.4の±1関数定義に一致)。
    層が無ければ0を返す。
    """
    total = None
    for layer in iter_spectral_layers(model):
        param = layer.parameterization
        if param == 'softmax' and getattr(layer, '_use_16gate', False):
            param = 'softmax16g'
        table = soft_truth_table(
            layer.weights, param, layer.n_inputs,
            gate_table=getattr(layer, 'gate_table', None),
            tau_warp=getattr(layer, 'tau_warp', 1.0),
        )
        pen = spectral_norm_gram(2.0 * table - 1.0, layer.n_inputs)
        total = pen if total is None else total + pen
    if total is None:
        return torch.zeros((), device=next(
            model.parameters(), torch.zeros(())).device)
    return total
