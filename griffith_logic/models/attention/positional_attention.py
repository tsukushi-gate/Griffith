"""
DLGN Positional Attention の実装。

Task 3: 位置指定ルックアップ (Positional Lookup)
  入力系列 [V0, V1, ..., V(N-1)] と位置インデックス pos を受け取り、
  V_pos を出力する。キーはデータ内容ではなく「位置」で決まる。

Task 1 (DLGNBooleanAttention) との違い:
  - キーはデータから生成せず、固定の位置インデックス（0, 1, 2, ..., N-1）
  - 位置キーは register_buffer として保持（学習不要）
  - クエリは pos（pos_bits ビット）を query_encoder に通す
  - キーは 0..N-1 を二進数化して key_encoder に通す（全スロット共通 LutStack）
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...group_sum import GroupSum
from ..rddlgn import LutStack

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class PositionalAttentionConfig:
    """DLGNPositionalAttention のハイパーパラメータ。"""

    # タスク設定
    pos_bits: int = 3  # 位置インデックスのビット数。num_slots = 2^pos_bits
    value_bits: int = 4

    # LutStack サイズ
    encoder_sizes: list = field(default_factory=lambda: [32, 16])
    value_encoder_sizes: list = field(default_factory=lambda: [32, 16])
    decoder_sizes: list = field(default_factory=lambda: [64, 32])

    # LutLayer 設定
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    grad_factor: float = 1.0

    # Attention 設定
    tau_attn: float = 1.0
    share_key_query: bool = True

    # 出力設定
    tau_out: float = 1.0

    # WARP 設定
    tau_warp: float = 1.0
    gumbel: bool = True

    @property
    def num_slots(self) -> int:
        return 2 ** self.pos_bits

    @property
    def input_dim(self) -> int:
        return self.num_slots * self.value_bits + self.pos_bits

    @property
    def num_classes(self) -> int:
        return 2 ** self.value_bits

    @property
    def d_attn(self) -> int:
        return self.encoder_sizes[-1]

    @property
    def d_val(self) -> int:
        return self.value_encoder_sizes[-1]

    @classmethod
    def tiny(cls) -> 'PositionalAttentionConfig':
        return cls()


# ---------------------------------------------------------------------------
# DLGNPositionalAttention
# ---------------------------------------------------------------------------

def _build_position_indices(num_slots: int, pos_bits: int) -> torch.Tensor:
    """
    0..num_slots-1 を MSB ファーストで二進数化した (num_slots, pos_bits) テンソル。
    """
    idx = torch.zeros(num_slots, pos_bits, dtype=torch.float32)
    for i in range(num_slots):
        for b in range(pos_bits):
            idx[i, b] = (i >> (pos_bits - 1 - b)) & 1
    return idx


class DLGNPositionalAttention(nn.Module):
    """
    位置ベース DLGN Attention。

    アーキテクチャ:
        入力: [V0, V1, ..., V(N-1), pos]  （フラット二値）
          ↓ parse
        values (batch, num_slots, value_bits), pos (batch, pos_bits)
          ↓ key_encoder: 固定の位置インデックス [0..N-1] → (num_slots, d_attn)
          ↓ query_encoder(pos) → (batch, d_attn)
          ↓ value_encoder(values) → (batch, num_slots, d_val)
          ↓ XNOR 類似度
        scores (batch, num_slots)
          ↓ softmax/argmax → selected (batch, d_val)
          ↓ output_decoder + GroupSum
        logits (batch, num_classes)
    """

    def __init__(self, config: PositionalAttentionConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config

        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            tau_warp=config.tau_warp,
            gumbel=config.gumbel,
        )

        # 位置インデックステーブル（非学習、register_buffer）
        pos_table = _build_position_indices(config.num_slots, config.pos_bits).to(device)
        self.register_buffer('position_indices', pos_table)

        # Key encoder: 位置インデックス (pos_bits) → d_attn
        self.key_encoder = LutStack(
            config.pos_bits, config.encoder_sizes, **lut_kwargs,
        )
        # Query encoder: pos (pos_bits) → d_attn
        if config.share_key_query:
            self.query_encoder = self.key_encoder
        else:
            self.query_encoder = LutStack(
                config.pos_bits, config.encoder_sizes, **lut_kwargs,
            )

        # Value encoder
        self.value_encoder = LutStack(
            config.value_bits, config.value_encoder_sizes, **lut_kwargs,
        )

        # Output decoder
        self.output_decoder = LutStack(
            config.d_val, config.decoder_sizes, **lut_kwargs,
        )

        assert config.decoder_sizes[-1] % config.num_classes == 0, (
            f'decoder_sizes[-1]={config.decoder_sizes[-1]} は '
            f'num_classes={config.num_classes} で割り切れる必要があります'
        )
        self.group_sum = GroupSum(k=config.num_classes, tau=config.tau_out)

        self._last_attn: torch.Tensor | None = None

    def parse_input(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            values: (batch, num_slots, value_bits)
            pos:    (batch, pos_bits)
        """
        cfg = self.config
        batch = x.shape[0]
        values_region = cfg.num_slots * cfg.value_bits

        values = x[:, :values_region].reshape(
            batch, cfg.num_slots, cfg.value_bits,
        )
        pos = x[:, values_region:]
        return values, pos

    def forward(
        self, x: torch.Tensor, hard_attention: bool | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        batch = x.shape[0]

        if hard_attention is None:
            hard_attention = not self.training

        values, pos = self.parse_input(x)

        # Key: 位置インデックスを key_encoder に通す（非バッチ、1回だけでOK）
        # position_indices: (num_slots, pos_bits) → (num_slots, d_attn)
        k = self.key_encoder(self.position_indices)  # (num_slots, d_attn)

        # Query: pos (batch, pos_bits) → (batch, d_attn)
        q = self.query_encoder(pos)

        # Value: (batch, num_slots, value_bits) → (batch, num_slots, d_val)
        v_flat = values.reshape(batch * cfg.num_slots, cfg.value_bits)
        v = self.value_encoder(v_flat).reshape(batch, cfg.num_slots, -1)

        # XNOR 類似度: q (B, D) vs k (N, D) → (B, N)
        # k: (N, D) → (1, N, D) broadcast with q: (B, 1, D)
        q_exp = q.unsqueeze(1)  # (B, 1, D)
        k_exp = k.unsqueeze(0)  # (1, N, D)
        xnor = q_exp * k_exp + (1.0 - q_exp) * (1.0 - k_exp)  # (B, N, D)
        scores = xnor.sum(dim=-1)  # (B, N)

        # Attention
        if hard_attention:
            idx = scores.argmax(dim=-1)  # (B,)
            selected = v[torch.arange(batch, device=v.device), idx]
            attn = F.one_hot(idx, num_classes=cfg.num_slots).to(v.dtype)
        else:
            attn = F.softmax(scores / cfg.tau_attn, dim=-1)  # (B, N)
            selected = (attn.unsqueeze(-1) * v).sum(dim=1)  # (B, d_val)

        self._last_attn = attn.detach()

        h = self.output_decoder(selected)
        logits = self.group_sum(h)
        return logits

    @property
    def last_attention(self) -> torch.Tensor | None:
        return self._last_attn

    def count_parameters(self) -> dict[str, int]:
        counts = {}
        counts['key_encoder'] = sum(p.numel() for p in self.key_encoder.parameters())
        if not self.config.share_key_query:
            counts['query_encoder'] = sum(
                p.numel() for p in self.query_encoder.parameters()
            )
        counts['value_encoder'] = sum(
            p.numel() for p in self.value_encoder.parameters()
        )
        counts['output_decoder'] = sum(
            p.numel() for p in self.output_decoder.parameters()
        )
        counts['total'] = sum(counts.values())
        return counts
