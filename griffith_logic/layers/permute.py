"""恒等配線の並べ替え(IR の permute ノードの実体)。

平坦ベクトルを in_shape に見なして perm の軸順に転置し、再び平坦化する。
GroupSum(assign='stride') のネイティブ抽出(permute [m, k] → [k, m] + group_sum)や
SpatialGroupSum の permute ノードを `ir/build.py` が組み直すときに使う。
パラメータは持たず、{0, 1} 入力でも実数入力でもビット位置の写像だけを行う。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Permute(nn.Module):
    """x を (..., *in_shape) と見なし、軸を perm の順に並べ替えて (..., width) へ戻す。

    Args:
        in_shape: 入力の形状(要素積 = 入力幅)
        perm: 軸の並べ替え(0 .. len(in_shape)−1 の順列)。出力の軸 i = 入力の軸 perm[i]
    """

    def __init__(self, in_shape, perm):
        super().__init__()
        in_shape = tuple(int(v) for v in in_shape)
        perm = tuple(int(v) for v in perm)
        if not in_shape or any(v <= 0 for v in in_shape):
            raise ValueError(f'in_shape は正整数のリストが必要です: {in_shape}')
        if sorted(perm) != list(range(len(in_shape))):
            raise ValueError(
                f'perm は 0..{len(in_shape) - 1} の順列が必要です: {perm}')
        self.in_shape = in_shape
        self.perm = perm
        self.out_shape = tuple(in_shape[p] for p in perm)
        self.width = math.prod(in_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.width, f'入力幅 {x.shape[-1]} != {self.width}'
        lead = x.shape[:-1]
        k = len(lead)
        y = x.reshape(*lead, *self.in_shape)
        y = y.permute(*range(k), *(k + p for p in self.perm))
        return y.reshape(*lead, self.width)

    def extra_repr(self) -> str:
        return f'in_shape={self.in_shape}, perm={self.perm}'
