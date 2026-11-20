"""
DLGN Multi-Query Boolean Attention の実装。

Task 2: マルチクエリ連想記憶
  1回のforwardで M個のクエリを同時に処理。
  同じKVメモリに対して複数回のattentionを並列適用する。

Task 1 (DLGNBooleanAttention) の直接的な拡張:
  - Query次元に num_queries を追加
  - 各クエリが独立に XNOR 類似度を計算
  - LutStack は (batch * num_queries) 軸でブロードキャスト
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
class MultiQueryAttentionConfig:
    """DLGNMultiQueryAttention のハイパーパラメータ。"""

    # タスク設定
    num_pairs: int = 4
    num_queries: int = 3
    key_bits: int = 4
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
    def input_dim(self) -> int:
        """入力ビット数。"""
        return (
            self.num_pairs * (self.key_bits + self.value_bits)
            + self.num_queries * self.key_bits
        )

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
    def tiny(cls) -> 'MultiQueryAttentionConfig':
        return cls()


# ---------------------------------------------------------------------------
# DLGNMultiQueryAttention
# ---------------------------------------------------------------------------

class DLGNMultiQueryAttention(nn.Module):
    """
    マルチクエリ版 DLGN Boolean Attention。

    入力フォーマット:
        x = [K1, V1, ..., K_np, V_np, Q1, Q2, ..., Q_nq]

    出力: logits (batch, num_queries, num_classes)
    """

    def __init__(self, config: MultiQueryAttentionConfig, device: str = 'cpu'):
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

        # Key encoder
        self.key_encoder = LutStack(
            config.key_bits, config.encoder_sizes, **lut_kwargs,
        )
        if config.share_key_query:
            self.query_encoder = self.key_encoder
        else:
            self.query_encoder = LutStack(
                config.key_bits, config.encoder_sizes, **lut_kwargs,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            queries: (batch, num_queries, key_bits)
            keys:    (batch, num_pairs, key_bits)
            values:  (batch, num_pairs, value_bits)
        """
        cfg = self.config
        batch = x.shape[0]
        pair_size = cfg.key_bits + cfg.value_bits
        pair_region_size = cfg.num_pairs * pair_size

        pairs = x[:, :pair_region_size].reshape(batch, cfg.num_pairs, pair_size)
        keys = pairs[:, :, :cfg.key_bits]
        values = pairs[:, :, cfg.key_bits:]

        queries = x[:, pair_region_size:].reshape(
            batch, cfg.num_queries, cfg.key_bits,
        )
        return queries, keys, values

    def forward(
        self, x: torch.Tensor, hard_attention: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) float [0,1]
            hard_attention: True=argmax選択、False=softmax加重、None=auto

        Returns:
            logits: (batch, num_queries, num_classes)
        """
        cfg = self.config
        batch = x.shape[0]

        if hard_attention is None:
            hard_attention = not self.training

        queries, keys, values = self.parse_input(x)

        # Query encoding: (batch*num_queries, key_bits) → (batch, num_queries, d_attn)
        q_flat = queries.reshape(batch * cfg.num_queries, cfg.key_bits)
        q = self.query_encoder(q_flat).reshape(batch, cfg.num_queries, -1)

        # Key encoding: (batch*num_pairs, key_bits) → (batch, num_pairs, d_attn)
        k_flat = keys.reshape(batch * cfg.num_pairs, cfg.key_bits)
        k = self.key_encoder(k_flat).reshape(batch, cfg.num_pairs, -1)

        # Value encoding: (batch*num_pairs, value_bits) → (batch, num_pairs, d_val)
        v_flat = values.reshape(batch * cfg.num_pairs, cfg.value_bits)
        v = self.value_encoder(v_flat).reshape(batch, cfg.num_pairs, -1)

        # XNOR 類似度: (batch, num_queries, num_pairs)
        # q: (B, Q, D), k: (B, P, D) → scores: (B, Q, P)
        q_exp = q.unsqueeze(2)  # (B, Q, 1, D)
        k_exp = k.unsqueeze(1)  # (B, 1, P, D)
        xnor = q_exp * k_exp + (1.0 - q_exp) * (1.0 - k_exp)  # (B, Q, P, D)
        scores = xnor.sum(dim=-1)  # (B, Q, P)

        # Attention: 各クエリが独立にキーをattend
        if hard_attention:
            idx = scores.argmax(dim=-1)  # (B, Q)
            # gather v with idx: (B, Q, d_val)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, cfg.d_val)  # (B, Q, d_val)
            selected = torch.gather(
                v.unsqueeze(1).expand(-1, cfg.num_queries, -1, -1),  # (B, Q, P, d_val)
                dim=2,
                index=idx_exp.unsqueeze(2),  # (B, Q, 1, d_val)
            ).squeeze(2)  # (B, Q, d_val)
            attn = F.one_hot(idx, num_classes=cfg.num_pairs).to(v.dtype)
        else:
            attn = F.softmax(scores / cfg.tau_attn, dim=-1)  # (B, Q, P)
            # weighted sum: (B, Q, P, 1) * (B, 1, P, d_val) → (B, Q, d_val)
            selected = torch.einsum('bqp,bpd->bqd', attn, v)

        self._last_attn = attn.detach()

        # Decode: (batch * num_queries, d_val) → logits
        dec_flat = self.output_decoder(
            selected.reshape(batch * cfg.num_queries, cfg.d_val),
        )
        logits = self.group_sum(dec_flat).reshape(
            batch, cfg.num_queries, cfg.num_classes,
        )

        return logits

    @property
    def last_attention(self) -> torch.Tensor | None:
        """直前の attention 重み (batch, num_queries, num_pairs)。"""
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
