"""EGGROLL の Hardware-Efficient form (Sarkar 2026, Section 4.2) の参考実装。

論文の主張:
    u_i (M + σ E_i)^T = u_i M^T + (σ/√r) (u_i B_i) A_i^T

線形層の forward は M^T による「全 worker 共通の matmul」と、
小さい rank-r 補正項に分解できる。N workers ぶんの forward を 1 回の
batched matmul で実行できるため、推論バッチと同等の arithmetic intensity が
出る（vLLM の batched LoRA inference と同形式）。

DLGN（LutLayer）の softmax 非線形は per-worker での materialize が必要なため、
上記の hw_efficient_linear_batched は線形層のデモンストレーション・
テスト用途に留まる。LutLayer 自体のワーカーバッチ化は下記
`batched_lut_forward` / `batched_model_forward` が担う: worker ごとの
摂動後の重み M+σE_i を worker 次元に stack した (N, out_dim, n_entries) の
テンソルとして**実体化**して受け取り(softmax を per-worker に掛けるため
low-rank のままでは扱えない。メモリは N 倍になる)、butterfly基底(入力にのみ
依存し重みに依存しない)を1回だけ計算してworker次元へbroadcastすることで、
N回の直列forwardを1回の batched einsum に畳み込む。

LutLayer.forward と同じ意味論に従う(対応範囲は parameterization='softmax' +
wiring='fixed'):
    - 訓練モード(selection='soft'): gate_mask(除外ゲートの−infロジット)、
      gumbel_noise(Gumbelノイズ注入)を LutLayer と同じ式で適用してから softmax
    - 評価モード: LutLayer の推論時と同じ hard 評価(16ゲートモードは argmax で
      選んだゲートの真理値表、N>2 は softmax>0.5 のエントリのみ1)
Gumbelノイズは worker ごとに独立に生成するが、乱数の消費順は直列 forward の
ループとは異なる(分布は同じだが、直列版とビット単位では一致しない)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .core.basis import _compute_basis_butterfly


def hw_efficient_linear_batched(
    inputs: torch.Tensor,
    M: torch.Tensor,
    A_stack: torch.Tensor,
    B_stack: torch.Tensor,
    sigma: float,
    rank: int,
) -> torch.Tensor:
    """論文 Section 4.2 の hardware-efficient batched form。

    Args:
        inputs:  (batch, in_dim)
        M:       (out_dim, in_dim)  — nn.Linear.weight 互換
        A_stack: (N_workers, out_dim, r)
        B_stack: (N_workers, in_dim, r)
        sigma:   摂動スケール σ
        rank:    r

    Returns:
        (N_workers, batch, out_dim) — 各 worker の forward 出力
    """
    base = inputs @ M.T                                        # (batch, out_dim)
    uB = torch.einsum('bi,nir->nbr', inputs, B_stack)          # (N, batch, r)
    delta = torch.einsum('nbr,nor->nbo', uB, A_stack)          # (N, batch, out_dim)
    coef = sigma / (rank ** 0.5)
    return base.unsqueeze(0) + coef * delta


def naive_linear_batched(
    inputs: torch.Tensor,
    M: torch.Tensor,
    A_stack: torch.Tensor,
    B_stack: torch.Tensor,
    sigma: float,
    rank: int,
) -> torch.Tensor:
    """ナイーブ実装。等価性検証用に各 E_i を materialize して計算する。"""
    N = A_stack.shape[0]
    coef = sigma / (rank ** 0.5)
    outs = []
    for i in range(N):
        E = (A_stack[i] @ B_stack[i].T) * coef                 # (out_dim, in_dim)
        W = M + E
        outs.append(inputs @ W.T)
    return torch.stack(outs, dim=0)


# ----------------------------------------------------------------------
# LutLayer のワーカーバッチ化 forward
# ----------------------------------------------------------------------
#
# `LutFunction.forward`（core/functions.py）の要点:
#   gathered = x[:, indices]                       … 重みに依存しない
#   basis    = butterfly(gathered)                 … 重みに依存しない
#   s        = softmax(weights, dim=-1)
#   w        = s @ gate_table  (16ゲートモードのみ) または s
#   output   = einsum('boe,oe->bo', basis, w)
#
# gathered/basis が重みに依存しないことを利用し、workerごとに異なる
# weights のみを stack して basis へ broadcast すれば、N回の直列forwardが
# 1回の batched einsum になる。第1層のように全workerで入力xが共有される
# 場合は x.dim()==2 のまま basis を1回だけ計算する。第2層以降のように
# 前段の出力がworkerごとに異なる(x.dim()==3, 形状(N,batch,in_dim))場合は
# gather/basis 計算をworker×batchへ展開してから畳み込む。


def _worker_truth_tables(
    weights_stack: torch.Tensor,
    gate_table: torch.Tensor | None,
    hard: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """worker毎のエントリ重み (N, out_dim, 2**n_inputs) を返す。"""
    if hard:
        # LutLayer.forward の推論時(eval)と同じ離散化
        if gate_table is not None:
            return gate_table[weights_stack.argmax(dim=-1)].to(dtype)
        return (F.softmax(weights_stack, dim=-1) > 0.5).to(dtype)
    s = F.softmax(weights_stack, dim=-1)
    return s @ gate_table if gate_table is not None else s


def batched_lut_forward(
    x: torch.Tensor,
    weights_stack: torch.Tensor,
    indices: torch.Tensor,
    n_inputs: int,
    gate_table: torch.Tensor | None = None,
    hard: bool = False,
) -> torch.Tensor:
    """LutLayer(softmax, wiring='fixed')のN worker一括forward。

    hard=False は訓練時の soft 評価(`LutFunction.forward` と数理的に同一)、
    hard=True は LutLayer の推論時(eval)の離散化評価。勾配は計算しない
    (EGGROLLの fitness評価は torch.no_grad() 下で行われるため不要)。
    gate_mask・Gumbelノイズは weights_stack 側に適用済みであること
    (`batched_model_forward` が LutLayer と同じ式で適用する)。

    Args:
        x: (batch, in_dim) — 全worker共有入力（層1相当）、または
           (N, batch, in_dim) — worker毎に異なる入力（層2以降）
        weights_stack: (N, out_dim, n_entries_logits) — worker毎の
            perturbed weights（n_entries_logits は softmax N=2 16ゲート
            モードでは16、それ以外は 2**n_inputs）
        indices: (n_inputs, out_dim) — LutLayer.indices
        n_inputs: LUT入力数
        gate_table: (16, 2**n_inputs) — 16ゲートモードのみ指定
            （LutLayer.gate_table。それ以外は None）
        hard: True なら推論時(eval)の hard 評価

    Returns:
        (N, batch, out_dim)
    """
    out_dim = indices.shape[1]
    flat_indices = indices.T.reshape(-1)

    if x.dim() == 2:
        batch = x.shape[0]
        gathered = x[:, flat_indices].reshape(batch, out_dim, n_inputs)
        basis = _compute_basis_butterfly(gathered, n_inputs)   # (batch,out_dim,e)
        w = _worker_truth_tables(weights_stack, gate_table, hard, x.dtype)  # (N,out_dim,e)
        return torch.einsum('boe,noe->nbo', basis, w)

    if x.dim() == 3:
        N, batch, in_dim = x.shape
        flat = x.reshape(N * batch, in_dim)
        gathered = flat[:, flat_indices].reshape(N * batch, out_dim, n_inputs)
        basis = _compute_basis_butterfly(gathered, n_inputs).reshape(
            N, batch, out_dim, -1
        )
        w = _worker_truth_tables(weights_stack, gate_table, hard, x.dtype)
        return torch.einsum('nboe,noe->nbo', basis, w)

    raise ValueError(f'batched_lut_forward: xの次元数が不正: {x.dim()}')


def batched_model_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    weights_stacks: dict,
) -> torch.Tensor:
    """LutLayerを含む nn.Sequential 相当のモデルをN worker一括forwardする。

    `weights_stacks` に含まれる LutLayer は `batched_lut_forward` へ
    ディスパッチし(各 LutLayer の training フラグに従い、訓練モードは
    gate_mask・gumbel_noise を適用した soft 評価、評価モードは hard 評価)、
    それ以外のモジュール（Flatten, GroupSum等）は通常の
    forward をそのまま適用する（GroupSumはx.shape[:-1]ベースの実装で
    先頭に worker次元(N)が増えても無変更で動作する。Flattenは worker
    次元がまだ現れない最初のLutLayerより前段でのみ通過するため無影響）。

    Args:
        model: LutLayer を含む nn.Sequential（子モジュールを順に適用できる
            コンテナである前提。parameterization='softmax', wiring='fixed' の
            LutLayer のみ対応。訓練モードでは selection='soft' のみ対応）
        x: (batch, in_dim) — 全worker共有の入力
        weights_stacks: {LutLayerモジュール: (N, out_dim, n_entries_logits)}
            EGGROLLOptimizer(batched=True) が渡す perturbed weights

    Returns:
        (N, batch, ...) — モデル最終出力のN worker版
    """
    from .layers.lut import _GUMBEL_EPS, LutLayer

    assert isinstance(model, torch.nn.Sequential), (
        'batched_model_forwardはnn.Sequential(子モジュールを順に適用)のみ対応: '
        f'{type(model)}'
    )

    out = x
    for module in model:
        if isinstance(module, LutLayer):
            if module not in weights_stacks:
                raise KeyError(
                    'batched_model_forward: weights_stacksに未登録のLutLayerが'
                    'モデル中に存在する(全LutLayerを渡すこと。部分バッチ化は未対応)'
                )
            if not (
                module.parameterization == 'softmax'
                and module.wiring == 'fixed'
                and (module.selection == 'soft' or not module.training)
            ):
                raise NotImplementedError(
                    'batched_model_forwardはparameterization=softmax, '
                    "wiring='fixed' のLutLayer(訓練モードでは selection='soft')"
                    f'のみ対応: {module.extra_repr()}'
                )
            gate_table = getattr(module, 'gate_table', None)
            ws = weights_stacks[module]
            # LutLayer._masked_weights と同じ: 除外ゲートのロジットに−inf
            mask = getattr(module, 'gate_logit_mask', None)
            if mask is not None:
                ws = ws + mask
            if module.training and module.gumbel_noise:
                # LutLayer._apply_gumbel と同じ式(worker毎に独立なノイズ)
                u = torch.rand_like(ws).clamp(_GUMBEL_EPS, 1.0 - _GUMBEL_EPS)
                ws = (ws - torch.log(-torch.log(u))) / module.gumbel_tau
            out = batched_lut_forward(
                out, ws, module.indices, module.n_inputs, gate_table,
                hard=not module.training,
            )
        else:
            out = module(out)
    return out
