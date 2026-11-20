"""LTT(線形閾値ゲート)層。

内部調査リポジトリの先行実験の `ltt_lib.LTTLayer`(深さ方向にも検証済み)の移植。
ゲート j は n 本の入力に対する線形閾値関数

    y_j = 1[ Σ_k w[j, k] · x[idx[k, j]] + b[j] > 0 ]        (x ∈ {0, 1})

で、n 入力 LUT(真理値表 2^n bit)としてそのまま FPGA に焼ける。表現できるのは
線形分離可能なブール関数(閾値関数)だけ(n=2 では XOR / XNOR 以外の 14 種)で、
DWN(2^n 個の自由なエントリ)に比べパラメータがゲートあたり n + 1 個と少ない
(TT-Sparse 流の 7 パラメータ、n=6)。

学習: forward は常に硬い閾値(値は {0, 1})で、勾配は straight-through:
    identity : dy/dz = 1                        (torch_dwn の STEFunction と同一。DWN と同条件)
    clip     : dy/dz = 1[|z| < s]               (クリップ線形 / hardtanh 型。先行実験の既定)
    sigmoid  : dy/dz = 4·σ(z/s)·(1 − σ(z/s))    (σ' を z = 0 で 1 に正規化)
s(`ste_scale`)は温度で、どのモードも z = 0 での勾配倍率は 1。w, b は U(−1, 1) で初期化し、
訓練時の forward ごとに [−1, 1] へ clamp する(DWN の表と同じ規約。`clamp=False` で外せる)。
`soft_forward` は σ(z/s)(先行実験と同じ流儀の soft 評価・診断用。学習経路ではない)。

配線は LutLayer と同じ (n_inputs, out_dim) の indices('random' / 'unique' / 明示 Tensor)か、
'learnable'(DWN §3.2 の LearnableMapping。argmax が実効配線)。
IR: export/extract.py の抽出器が全 2^n パターンを列挙して lut_layer にする
(export/discretize.py::truth_tables_from_ltt)。参照実装は core/reference.py の
reference_ltt_* 。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.connections import get_random_connections, get_unique_connections
from .dwn import _DEFAULT_LM_TAU, LearnableMapping


class ThresholdSTE(torch.autograd.Function):
    """y = 1[z > 0] の straight-through 推定(mode / scale で代理勾配を選ぶ)。"""

    @staticmethod
    def forward(ctx, z: torch.Tensor, scale: float, mode: str) -> torch.Tensor:
        ctx.save_for_backward(z)
        ctx.scale = float(scale)
        ctx.mode = mode
        return (z > 0).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        return grad_output * ltt_ste_surrogate(z, ctx.mode, ctx.scale), None, None


def ltt_ste_surrogate(z: torch.Tensor, mode: str, scale: float) -> torch.Tensor:
    """代理勾配 dy/dz(z と同形状)。identity / clip / sigmoid(モジュール docstring の式)。"""
    if mode == 'identity':
        return torch.ones_like(z)
    if mode == 'clip':
        return (z.abs() < scale).to(z.dtype)
    if mode == 'sigmoid':
        s = torch.sigmoid(z / scale)
        return 4.0 * s * (1.0 - s)
    raise ValueError(f"ste は 'identity' / 'clip' / 'sigmoid' のいずれか: {mode!r}")


class LTTLayer(nn.Module):
    """線形閾値ゲート層: 二値入力 → n 入力の閾値関数 × out_dim → 二値出力。

    Args:
        in_dim: 入力ビット数
        out_dim: ゲート数 = 出力ビット数
        n_inputs: ゲートあたりの入力数 n(既定 6 = 先行実験)
        connections: 'random'(均衡乱択)/ 'unique' / 'learnable'(LearnableMapping)/
            (n_inputs, out_dim) の int Tensor
        ste: 代理勾配 'clip'(既定)/ 'identity' / 'sigmoid'
        ste_scale: 代理勾配の温度 s(既定 1.0)
        clamp: 訓練時の forward ごとに w, b を [−1, 1] へ clamp(既定 True)
        lm_tau: connections='learnable' の LearnableMapping の逆伝播 softmax 温度
        device: 配置先
    """

    STE_MODES = ('identity', 'clip', 'sigmoid')
    CONNECTIONS = ('random', 'unique', 'learnable')

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_inputs: int = 6,
        connections: str | torch.Tensor = 'random',
        ste: str = 'clip',
        ste_scale: float = 1.0,
        clamp: bool = True,
        lm_tau: float = _DEFAULT_LM_TAU,
        device: str = 'cpu',
    ):
        super().__init__()
        if ste not in self.STE_MODES:
            raise ValueError(f'ste は {self.STE_MODES} のいずれか: {ste!r}')
        if n_inputs < 1:
            raise ValueError(f'n_inputs は 1 以上: {n_inputs}')
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_inputs = int(n_inputs)
        self.ste = ste
        self.ste_scale = float(ste_scale)
        self.clamp = bool(clamp)

        if isinstance(connections, torch.Tensor):
            idx = connections.to(torch.int64)
            if idx.shape != (self.n_inputs, self.out_dim):
                raise ValueError(
                    f'connections の形は (n_inputs, out_dim) = ({self.n_inputs}, {self.out_dim})'
                    f' が必要です: {tuple(idx.shape)}')
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.in_dim):
                raise ValueError(f'connections の添字が [0, {self.in_dim}) の範囲外です')
            self.mapping_mode = 'explicit'
            self.register_buffer('indices', idx.to(device).contiguous())
        elif connections == 'learnable':
            self.mapping_mode = 'learnable'
            self.learnable_mapping = LearnableMapping(
                self.in_dim, self.out_dim * self.n_inputs, tau=lm_tau).to(device)
        elif connections in ('random', 'unique'):
            self.mapping_mode = connections
            gen = get_random_connections if connections == 'random' else get_unique_connections
            self.register_buffer(
                'indices', gen(self.in_dim, self.out_dim, self.n_inputs, device))
        else:
            raise ValueError(
                f'connections は {self.CONNECTIONS} か Tensor: {connections!r}')

        # 先行実験と同じ U(−1, 1) 初期化(DWN の表の初期化に合わせる)
        self.w = nn.Parameter(torch.rand(self.out_dim, self.n_inputs, device=device) * 2 - 1)
        self.b = nn.Parameter(torch.rand(self.out_dim, device=device) * 2 - 1)

    # ---- 配線 ---------------------------------------------------------
    def effective_indices(self) -> torch.Tensor:
        """実効配線 (n_inputs, out_dim) int64。learnable は argmax(reference_gather と同じ規約)。"""
        if self.mapping_mode == 'learnable':
            mapping = self.learnable_mapping.weights.detach().argmax(dim=0)   # (out·n,)
            return mapping.reshape(self.out_dim, self.n_inputs).T.contiguous()
        return self.indices

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, in_dim) → (batch, out_dim, n_inputs)"""
        if self.mapping_mode == 'learnable':
            mapped = self.learnable_mapping(x)                      # (batch, out·n)
            return mapped.reshape(x.shape[0], self.out_dim, self.n_inputs)
        return x[:, self.indices.T.reshape(-1)].reshape(x.shape[0], self.out_dim, self.n_inputs)

    # ---- forward ------------------------------------------------------
    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """z = w · x + b(batch, out_dim)。訓練時は先に w, b を [−1, 1] へ clamp する。"""
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)
        if self.clamp and self.training:
            with torch.no_grad():
                self.w.clamp_(-1.0, 1.0)
                self.b.clamp_(-1.0, 1.0)
        gathered = self._gather(x)
        return (gathered * self.w.unsqueeze(0)).sum(dim=-1) + self.b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """二値出力 1[z > 0](訓練時は代理勾配付き)。"""
        z = self.pre_activation(x)
        if self.training:
            return ThresholdSTE.apply(z, self.ste_scale, self.ste)
        return (z > 0).to(z.dtype)

    def soft_forward(self, x: torch.Tensor) -> torch.Tensor:
        """σ(z / s)(診断用の soft 評価。学習経路ではない)。"""
        return torch.sigmoid(self.pre_activation(x) / self.ste_scale)

    def extra_repr(self):
        return (f'in_dim={self.in_dim}, out_dim={self.out_dim}, n_inputs={self.n_inputs}, '
                f'connections={self.mapping_mode}, ste={self.ste}/s={self.ste_scale}, '
                f'clamp={self.clamp}')
