"""
DLGN Boolean Attention の実装。

Gate-Level Boolean Evolutionary Geometric Attention (arXiv:2511.17550) に基づく
最もシンプルなAttention機構のDLGN実装。

Query/Key/Value を LutStack で二値ベクトルに射影し、XNOR類似度で Attention スコアを計算。
訓練時は softmax で重み付き平均、推論時は argmax で Hard Attention。

タスク: 連想記憶 (Associative Recall)
  入力 = 複数のキー・バリューペアとクエリキー（全て二値）
  出力 = クエリキーに一致するキーのバリュー
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
class BooleanAttentionConfig:
    """DLGNBooleanAttention のハイパーパラメータ。"""

    # タスク設定
    num_pairs: int = 4
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
    tau_attn: float = 1.0     # attention softmax 温度
    share_key_query: bool = True

    # 出力設定
    tau_out: float = 1.0      # GroupSum 温度

    # WARP 設定
    tau_warp: float = 1.0
    gumbel: bool = True

    @property
    def input_dim(self) -> int:
        """入力ビット数 (num_pairs × (key_bits + value_bits) + key_bits)。"""
        return self.num_pairs * (self.key_bits + self.value_bits) + self.key_bits

    @property
    def num_classes(self) -> int:
        """出力クラス数 (2^value_bits)。"""
        return 2 ** self.value_bits

    @property
    def d_attn(self) -> int:
        """Attention 射影の次元。"""
        return self.encoder_sizes[-1]

    @property
    def d_val(self) -> int:
        """Value 射影の次元。"""
        return self.value_encoder_sizes[-1]

    @classmethod
    def tiny(cls) -> 'BooleanAttentionConfig':
        """デバッグ用の小構成。"""
        return cls(
            num_pairs=4,
            key_bits=4,
            value_bits=4,
            encoder_sizes=[32, 16],
            value_encoder_sizes=[32, 16],
            decoder_sizes=[64, 32],
        )


# ---------------------------------------------------------------------------
# DLGNBooleanAttention
# ---------------------------------------------------------------------------

class DLGNBooleanAttention(nn.Module):
    """
    論理ゲート版 Attention 機構。

    アーキテクチャ:
        入力 (batch, input_dim) 二値ベクトル [0,1]
          ↓ parse_input
        Q (batch, key_bits), Ks (batch, num_pairs, key_bits), Vs (batch, num_pairs, value_bits)
          ↓ key_encoder / query_encoder / value_encoder (LutStack)
        q (batch, d_attn), k (batch, num_pairs, d_attn), v (batch, num_pairs, d_val)
          ↓ XNOR 類似度
        scores (batch, num_pairs)
          ↓ softmax or argmax で Value 選択
        selected (batch, d_val)
          ↓ output_decoder (LutStack + GroupSum)
        logits (batch, num_classes)
    """

    def __init__(self, config: BooleanAttentionConfig, device: str = 'cpu'):
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

        # Key encoder: key_bits → d_attn
        self.key_encoder = LutStack(
            config.key_bits, config.encoder_sizes, **lut_kwargs,
        )
        # Query encoder: 共有 or 独立
        if config.share_key_query:
            self.query_encoder = self.key_encoder
        else:
            self.query_encoder = LutStack(
                config.key_bits, config.encoder_sizes, **lut_kwargs,
            )

        # Value encoder: value_bits → d_val
        self.value_encoder = LutStack(
            config.value_bits, config.value_encoder_sizes, **lut_kwargs,
        )

        # Output decoder: d_val → decoder_sizes[-1]
        self.output_decoder = LutStack(
            config.d_val, config.decoder_sizes, **lut_kwargs,
        )

        # 最終層サイズが num_classes で割り切れるか確認
        assert config.decoder_sizes[-1] % config.num_classes == 0, (
            f'decoder_sizes[-1]={config.decoder_sizes[-1]} は '
            f'num_classes={config.num_classes} で割り切れる必要があります'
        )

        # GroupSum: 最終層出力 → num_classes のロジット
        self.group_sum = GroupSum(k=config.num_classes, tau=config.tau_out)

        # 最後の attention 重み（診断用）
        self._last_attn: torch.Tensor | None = None

    def parse_input(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        フラット二値ベクトルを Q, Ks, Vs に分解。

        Args:
            x: (batch, input_dim) float [0,1]

        Returns:
            query: (batch, key_bits)
            keys:  (batch, num_pairs, key_bits)
            values:(batch, num_pairs, value_bits)
        """
        cfg = self.config
        pair_size = cfg.key_bits + cfg.value_bits

        # キー・バリュー部分: (batch, num_pairs * pair_size)
        pair_region = x[:, : cfg.num_pairs * pair_size]
        # (batch, num_pairs, pair_size)
        pairs = pair_region.reshape(-1, cfg.num_pairs, pair_size)
        keys = pairs[:, :, : cfg.key_bits]
        values = pairs[:, :, cfg.key_bits:]

        # クエリ: (batch, key_bits)
        query = x[:, cfg.num_pairs * pair_size:]

        return query, keys, values

    def forward(
        self, x: torch.Tensor, hard_attention: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) 二値入力 [0,1]
            hard_attention: True=argmax選択（推論専用）、False=softmax加重、
                            None=self.training で自動切替

        Returns:
            logits: (batch, num_classes)
        """
        cfg = self.config
        batch = x.shape[0]

        # hard_attention の自動切替: 訓練時は soft、推論時は hard
        if hard_attention is None:
            hard_attention = not self.training

        # 1. 入力を Q, K, V に分解
        query, keys, values = self.parse_input(x)

        # 2. 各要素を LutStack で射影
        q = self.query_encoder(query)  # (batch, d_attn)

        # K: (batch * num_pairs, key_bits) → (batch, num_pairs, d_attn)
        k_flat = keys.reshape(batch * cfg.num_pairs, cfg.key_bits)
        k = self.key_encoder(k_flat).reshape(batch, cfg.num_pairs, -1)

        # V: (batch * num_pairs, value_bits) → (batch, num_pairs, d_val)
        v_flat = values.reshape(batch * cfg.num_pairs, cfg.value_bits)
        v = self.value_encoder(v_flat).reshape(batch, cfg.num_pairs, -1)

        # 3. XNOR 類似度
        #    XNOR(a, b) = a*b + (1-a)*(1-b)
        #    score_i = sum_d XNOR(q[d], k_i[d])   ← popcount 相当
        q_exp = q.unsqueeze(1)  # (batch, 1, d_attn)
        xnor = q_exp * k + (1.0 - q_exp) * (1.0 - k)  # (batch, num_pairs, d_attn)
        scores = xnor.sum(dim=-1)  # (batch, num_pairs)

        # 4. Attention による Value 選択
        if hard_attention:
            # 推論時: argmax で 1 つだけ選択
            idx = scores.argmax(dim=-1)  # (batch,)
            selected = v[torch.arange(batch, device=v.device), idx]  # (batch, d_val)
            # 診断用に one-hot attention を記録
            attn = F.one_hot(idx, num_classes=cfg.num_pairs).to(v.dtype)
        else:
            # 訓練時: softmax で加重平均
            attn = F.softmax(scores / cfg.tau_attn, dim=-1)  # (batch, num_pairs)
            selected = (attn.unsqueeze(-1) * v).sum(dim=1)   # (batch, d_val)

        self._last_attn = attn.detach()

        # 5. Decoder → logits
        h = self.output_decoder(selected)  # (batch, decoder_sizes[-1])
        logits = self.group_sum(h)         # (batch, num_classes)

        return logits

    @property
    def last_attention(self) -> torch.Tensor | None:
        """直前の forward で計算された attention 重み (batch, num_pairs)。"""
        return self._last_attn

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す。"""
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
