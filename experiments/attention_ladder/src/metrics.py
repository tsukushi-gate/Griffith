"""attention梯子の指標計算。

分類精度・LM損失/perplexity・attentionエントロピー・離散化ギャップ。
すべて純関数で、学習ループ(harness)から呼ばれる。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# log(0) 回避用の下限
_EPS = 1e-12


def classification_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """分類精度を返す。logits:(B,C) or (...,C), y:同形状のラベル。"""
    if y.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    return correct / y.numel()


def lm_loss_ppl(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """LMのcross entropyとperplexity(=exp(loss))を返す。

    logits:(B,T,V), y:(B,T)。
    """
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    loss_val = float(loss.item())
    # 数値安定のためexpの引数を上限クリップ(初期lossでもoverflowしないが保険)
    ppl = math.exp(min(loss_val, 20.0))
    return loss_val, ppl


def attention_entropy(
    attn_weights_list: list[torch.Tensor | None],
    causal: bool,
) -> float:
    """attention重みのkey方向エントロピー平均を返す(自然対数)。

    各層 (B,H,Sq,Sk) について最終次元(key方向)で -Σ p log p を計算し、
    有効な (B,H,Sq) で平均、さらに層平均を取る。causal時は下三角(query q が
    参照可能な key<=q)のみを有効とする。空/NaNは0除けする。
    """
    if not attn_weights_list:
        return 0.0

    layer_means: list[float] = []
    for w in attn_weights_list:
        if w is None:
            continue
        w = w.detach().float()  # (B,H,Sq,Sk)
        seq_q, seq_k = w.shape[-2], w.shape[-1]

        # -p log p (0 log 0 = 0 として扱う)
        term = w * torch.log(w.clamp_min(_EPS))
        if causal:
            # 下三角のみ有効(query q は key<=q のみ参照)
            mask = torch.tril(
                torch.ones(seq_q, seq_k, device=w.device, dtype=torch.bool)
            )
            term = term.masked_fill(~mask, 0.0)

        ent = -term.sum(dim=-1)  # (B,H,Sq)
        ent = torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)
        layer_means.append(float(ent.mean().item()))

    if not layer_means:
        return 0.0
    return sum(layer_means) / len(layer_means)


def discretization_gap(soft_metric: float, hard_metric: float) -> float:
    """soft評価とhard評価の符号付き差(soft - hard)。L0(全FP)では0。"""
    return float(soft_metric - hard_metric)
