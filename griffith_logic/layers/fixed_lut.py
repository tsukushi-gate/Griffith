"""凍結 LUT 層。確定済みの真理値表を持ち、学習パラメータを持たない。

IR の lut_layer には LutLayer が受けない幅(ThermometerEncoder の 8 入力、
GroupSum(assign='stride') を展開した 1 入力の恒等)や、学習し直す意味のない
凍結部品(kind='thermometer')がある。`ir/build.py` はそれらを本層で復元し、
抽出器(export/extract.py)は params と train_info をそのまま IR へ戻す
(build → export_ir の往復でノードが変わらない)。

意味論は lut_layer と同じ:
    addr = Σ_j 1[x[idx[j, o]] > 0.5] << j,   y_o = bit_addr(tt_o)
二値入力はこの表引きで厳密に評価する。勾配が要るとき(入力が requires_grad)や
入力が {0, 1} でないときは、積分布の期待値
    y_o = Σ_k tt_o[k] · Π_j (x_j if bit_j(k) else 1 − x_j)
で評価する(LutLayer の soft forward と同じ式。{0, 1} 入力では表引きと一致)。
期待値はエントリ 2^n を順に足すので、n = 8 でも (batch, out) の積 2048 回で済む。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _tables_to_bit_matrix(tables: list[int], n_entries: int) -> np.ndarray:
    """真理値表整数列 → (out, n_entries) の 0/1 行列(ir/simulate.py の n > 6 経路と同式)"""
    out = np.zeros((len(tables), n_entries), dtype=np.uint8)
    n_bytes = (n_entries + 7) // 8
    for o, t in enumerate(tables):
        bits = np.frombuffer(t.to_bytes(n_bytes, 'little'), dtype=np.uint8)
        out[o] = np.unpackbits(bits, bitorder='little')[:n_entries]
    return out


class FixedLutLayer(nn.Module):
    """確定済み真理値表を持つ凍結 LUT 層(任意の入力数、学習パラメータなし)。

    Args:
        in_dim: 入力ビット幅
        out_dim: 出力ビット幅(LUT の個数)
        n_inputs: LUT の入力数 n(1 以上。2 / 4 / 6 に限らない)
        indices: (n_inputs, out_dim) の接続テーブル(int)
        truth_tables: 長さ out_dim の真理値表(整数か hex 文字列。bit k = パターン k)
        train_info: IR へ戻すときの train_info(例 kind='thermometer')
        device: バッファの配置先
    """

    def __init__(
        self, in_dim: int, out_dim: int, n_inputs: int, indices, truth_tables, *,
        train_info: dict | None = None, device: str = 'cpu',
    ):
        super().__init__()
        if n_inputs < 1:
            raise ValueError(f'n_inputs は 1 以上が必要です: {n_inputs}')
        n_entries = 1 << n_inputs
        idx = torch.as_tensor(np.asarray(indices, dtype=np.int64))
        if tuple(idx.shape) != (n_inputs, out_dim):
            raise ValueError(
                f'indices の形状 {tuple(idx.shape)} != ({n_inputs}, {out_dim})')
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= in_dim):
            raise ValueError(f'indices が入力幅 {in_dim} を超えています')
        tables = [int(t, 16) if isinstance(t, str) else int(t) for t in truth_tables]
        if len(tables) != out_dim:
            raise ValueError(f'truth_tables の長さ {len(tables)} != out_dim({out_dim})')
        if any(t < 0 or t >> n_entries for t in tables):
            raise ValueError(f'真理値表が 2^{n_entries} ビットに収まりません')

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_inputs = int(n_inputs)
        self.n_entries = n_entries
        self.truth_tables: list[int] = tables
        self.train_info: dict = dict(train_info or {})
        self.register_buffer('indices', idx.to(device))
        self.register_buffer(
            'table',
            torch.from_numpy(_tables_to_bit_matrix(tables, n_entries).astype(np.float32))
            .to(device),
        )
        self.register_buffer(
            'pow2', (1 << torch.arange(n_inputs, dtype=torch.int64)).to(device))
        self.register_buffer('out_range', torch.arange(out_dim, dtype=torch.int64).to(device))
        # 期待値計算で足す必要のあるエントリ(表の列が全 0 なら飛ばす)
        self._active_entries = [
            k for k in range(n_entries) if any((t >> k) & 1 for t in tables)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_dim, (
            f'入力幅 {x.shape[-1]} != in_dim({self.in_dim})')
        xs = x[..., self.indices]  # (..., n_inputs, out_dim)
        if xs.requires_grad or not self._is_binary(xs):
            return self._expectation(xs)
        shape = (1,) * (xs.dim() - 2) + (self.n_inputs, 1)
        addr = ((xs > 0.5).to(torch.int64) * self.pow2.view(shape)).sum(-2)  # (..., out)
        return self.table[self.out_range, addr]

    @staticmethod
    def _is_binary(xs: torch.Tensor) -> bool:
        return bool(((xs == 0) | (xs == 1)).all())

    def _expectation(self, xs: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(
            xs.shape[:-2] + (self.out_dim,), dtype=xs.dtype, device=xs.device)
        for k in self._active_entries:
            term = None
            for j in range(self.n_inputs):
                v = xs[..., j, :] if (k >> j) & 1 else 1 - xs[..., j, :]
                term = v if term is None else term * v
            acc = acc + self.table[:, k].to(xs.dtype) * term
        return acc

    def extra_repr(self) -> str:
        s = f'in={self.in_dim}, out={self.out_dim}, n_inputs={self.n_inputs}, frozen'
        kind = self.train_info.get('kind')
        if kind:
            s += f', kind={kind}'
        return s
