"""LogicTransformerBlock。

LogicAttention + Residual(soft論理マージ) + Norm + LogicMLP を組んだ
Transformerブロックの論理ゲート版。全経路が[0,1]論理域で完結する。

IR表現はblackbox+contract(blocks/to_ir.py)。内部にsoftmax/argmaxを
要するLogicAttentionを含むためHDL直接生成は不可(理由と
=hard-argmaxマクロ展開がP2バックログである旨はblocks/attention.py
のdocstring参照)。
"""

import torch
import torch.nn as nn

from ..hybrid.policy import LogicSpec
from ..layers.logic_linear import LogicMLP
from .attention import LogicAttention
from .norm import LogicBatchNorm, LogicLayerNorm
from .structural import Add

NORM_MODES = ('layernorm', 'batchnorm', 'none')
BLOCK_MERGE_MODES = ('or', 'xor')  # 幅不変が必要なためconcatは非対応


def _make_norm(norm: str, width: int) -> nn.Module:
    if norm == 'layernorm':
        return LogicLayerNorm(width)
    if norm == 'batchnorm':
        return LogicBatchNorm(width)
    return nn.Identity()


class LogicTransformerBlock(nn.Module):
    """Transformerブロックの論理ゲート版。forward(x: (B, S, D)) → (B, S, D)。

    構成(残差はsoft論理マージ、幅不変):
        h = norm1(merge(x, attn(x, x, x)))    — self-attention + 残差
        y = norm2(merge(h, mlp(h)))           — LogicMLP + 残差

    mlpはLogicMLP(layers/logic_linear.py)を再利用:
        encoder='identity'(入力は既に[0,1])、中間幅 d_model×mlp_ratio、
        decoder='groupsum'+τ=mlp_ratio でグループ平均として[0,1]へ復帰

    Args:
        d_model: 位置あたりのビット幅
        n_heads: attentionヘッド数
        mlp_ratio: MLP中間幅の倍率
        norm: 'layernorm' | 'batchnorm' | 'none'
        merge: 残差マージ 'or' | 'xor'(soft論理。IR addノードと同一式)
        attn_mode: LogicAttentionのmode('soft'=訓練向け既定)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        mlp_ratio: int = 2,
        norm: str = 'layernorm',
        merge: str = 'xor',
        *,
        attn_mode: str = 'soft',
        causal: bool = False,
        n_inputs: int = 2,
        parameterization: str = 'softmax',
        connections: str = 'random',
        selection: str = 'soft',
    ):
        super().__init__()
        assert norm in NORM_MODES, f'norm は {NORM_MODES} のいずれか: {norm}'
        assert merge in BLOCK_MERGE_MODES, (
            f'merge は {BLOCK_MERGE_MODES} のいずれか: {merge}'
            '(concatは幅が拡大するためTransformerブロックでは非対応)'
        )
        self.d_model = d_model
        self.norm_mode = norm
        self.merge_mode = merge

        self.attn = LogicAttention(
            d_model, n_heads, mode=attn_mode,
            causal=causal,
            n_inputs=n_inputs, parameterization=parameterization,
            connections=connections, selection=selection,
        )
        self.merge = Add(merge)
        self.norm1 = _make_norm(norm, d_model)
        self.norm2 = _make_norm(norm, d_model)
        self.mlp = LogicMLP(
            d_model,
            spec=LogicSpec(
                encoder='identity',
                decoder='groupsum',
                width=d_model * mlp_ratio,
                decoder_tau=float(mlp_ratio),
                n_inputs=n_inputs,
                parameterization=parameterization,
                connections=connections,
                selection=selection,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, d_model) の[0,1]ビット列

        Returns:
            (B, S, d_model)
        """
        h = self.norm1(self.merge([x, self.attn(x, x, x)]))
        return self.norm2(self.merge([h, self.mlp(h)]))

    def extra_repr(self):
        return (f'd_model={self.d_model}, norm={self.norm_mode}, '
                f'merge={self.merge_mode}')
