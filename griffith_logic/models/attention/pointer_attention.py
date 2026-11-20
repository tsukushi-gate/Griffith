"""
DLGN Pointer Attention の実装。

Task 5: seq2seqソート (Pointer Network 版)
  内容ベースのキー、自己回帰デコード、マスキング付き Attention で sort を解く。

Vinyals et al. (2015) の Pointer Networks を DLGN フレームワークで実装:
  - Encoder: 各入力値を value_encoder（memory） / key_encoder（content key）で射影
  - Decoder: リカレント状態 + 前ステップ出力から query を生成
  - Attention: XNOR類似度 - mask_scale × 累積マスク
  - 出力: 選ばれた memory を output_decoder でクラス分類

既存の `LutStack`, `GroupSum`, `LutLayer` のみを再利用する。
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...group_sum import GroupSum
from ..rddlgn import LutStack


def _build_position_indices(num_slots: int, pos_bits: int) -> torch.Tensor:
    """0..num_slots-1 を MSB ファーストで二進数化した (num_slots, pos_bits) テンソル。"""
    idx = torch.zeros(num_slots, pos_bits, dtype=torch.float32)
    for i in range(num_slots):
        for b in range(pos_bits):
            idx[i, b] = (i >> (pos_bits - 1 - b)) & 1
    return idx


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class PointerAttentionConfig:
    """DLGNPointerAttention のハイパーパラメータ。"""

    seq_len: int = 4
    value_bits: int = 4

    # LutStack サイズ
    value_encoder_sizes: list = field(default_factory=lambda: [32, 16])  # → d_val
    key_encoder_sizes: list = field(default_factory=lambda: [32, 16])    # → d_attn
    decoder_cell_sizes: list = field(default_factory=lambda: [64, 16])   # → d_state
    query_producer_sizes: list = field(default_factory=lambda: [32, 16]) # → d_attn
    output_decoder_sizes: list = field(default_factory=lambda: [64, 32]) # → num_classes

    # LutLayer 設定
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    grad_factor: float = 1.0

    # Attention 設定
    tau_attn: float = 1.0
    mask_scale: float = 10.0      # 累積マスクで score を抑圧するスケール

    # 出力設定
    tau_out: float = 1.0

    # WARP 設定
    tau_warp: float = 1.0
    gumbel: bool = True

    @property
    def d_val(self) -> int:
        return self.value_encoder_sizes[-1]

    @property
    def d_attn(self) -> int:
        return self.key_encoder_sizes[-1]

    @property
    def d_state(self) -> int:
        return self.decoder_cell_sizes[-1]

    @property
    def input_dim(self) -> int:
        return self.seq_len * self.value_bits

    @property
    def num_classes(self) -> int:
        return 2 ** self.value_bits

    @property
    def pos_bits(self) -> int:
        """seq_len を表すのに必要なビット数（key_encoder の位置入力用）。"""
        return max(1, math.ceil(math.log2(self.seq_len)))

    @property
    def key_input_bits(self) -> int:
        """key_encoder の入力ビット数 = value_bits + pos_bits。"""
        return self.value_bits + self.pos_bits

    @classmethod
    def tiny(cls) -> 'PointerAttentionConfig':
        return cls()


# ---------------------------------------------------------------------------
# DLGNPointerAttention
# ---------------------------------------------------------------------------

class DLGNPointerAttention(nn.Module):
    """
    Pointer Network 風の DLGN seq2seq モデル。

    アーキテクチャ:
        Encoder (並列):
          values (B, N, value_bits)
            → value_encoder → v_enc (B, N, d_val)    memory
            → key_encoder   → k_enc (B, N, d_attn)   content key

        Decoder (t = 0..N-1, リカレント):
          state_t = decoder_cell([state_{t-1}, prev_val])
          query_t = query_producer(state_t)
          scores_t = XNOR(query_t, k_enc) - mask_scale × used_mask
          attn_t = softmax(scores_t / τ) (訓練) / one_hot(argmax) (推論)
          selected_t = attn_t @ v_enc
          logits_t = group_sum(output_decoder(selected_t))

          teacher forcing: prev_val = v_enc[target_pointer[t]]
          free running:    prev_val = selected_t

          used_mask += attn_t
    """

    def __init__(self, config: PointerAttentionConfig, device: str = 'cpu'):
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

        # Encoder
        self.value_encoder = LutStack(
            config.value_bits, config.value_encoder_sizes, **lut_kwargs,
        )
        # key_encoder: (value_bits + pos_bits) → d_attn
        # 位置情報を混ぜることで copy/reverse でも attend 先を識別できる
        self.key_encoder = LutStack(
            config.key_input_bits, config.key_encoder_sizes, **lut_kwargs,
        )
        # 位置インデックステーブル（非学習）
        pos_table = _build_position_indices(config.seq_len, config.pos_bits).to(device)
        self.register_buffer('position_indices', pos_table)

        # Decoder cell: [state_prev, prev_val] → state_new
        decoder_cell_in = config.d_state + config.d_val
        self.decoder_cell = LutStack(
            decoder_cell_in, config.decoder_cell_sizes, **lut_kwargs,
        )

        # Query producer: state → query
        self.query_producer = LutStack(
            config.d_state, config.query_producer_sizes, **lut_kwargs,
        )

        # 学習可能な初期状態と BOS 値（パラメータではなく sigmoid を通して [0,1] に保つ）
        self.state_init_logits = nn.Parameter(
            torch.zeros(config.d_state)
        )
        self.bos_val_logits = nn.Parameter(
            torch.zeros(config.d_val)
        )

        # Output decoder
        self.output_decoder = LutStack(
            config.d_val, config.output_decoder_sizes, **lut_kwargs,
        )

        assert config.output_decoder_sizes[-1] % config.num_classes == 0, (
            f'output_decoder_sizes[-1]={config.output_decoder_sizes[-1]} は '
            f'num_classes={config.num_classes} で割り切れる必要があります'
        )
        self.group_sum = GroupSum(k=config.num_classes, tau=config.tau_out)

        self._last_attn: torch.Tensor | None = None  # (B, seq_len, N)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            v_enc: (batch, N, d_val)
            k_enc: (batch, N, d_attn)
        """
        cfg = self.config
        batch = x.shape[0]
        values = x.reshape(batch, cfg.seq_len, cfg.value_bits)

        v_flat = values.reshape(batch * cfg.seq_len, cfg.value_bits)
        v_enc = self.value_encoder(v_flat).reshape(batch, cfg.seq_len, -1)

        # key_encoder の入力: 各位置で (value_bits + pos_bits)
        # position_indices: (seq_len, pos_bits) → (batch, seq_len, pos_bits)
        pos_broadcast = self.position_indices.unsqueeze(0).expand(
            batch, -1, -1,
        )
        key_inputs = torch.cat([values, pos_broadcast], dim=-1)  # (B, N, VB+PB)
        k_flat = key_inputs.reshape(batch * cfg.seq_len, cfg.key_input_bits)
        k_enc = self.key_encoder(k_flat).reshape(batch, cfg.seq_len, -1)
        return v_enc, k_enc

    def forward(
        self,
        x: torch.Tensor,
        target_pointers: torch.Tensor | None = None,
        hard_attention: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            target_pointers: (batch, seq_len) 正解ポインタ。None なら free-running
            hard_attention: 推論時の argmax 切替。None なら self.training で自動

        Returns:
            logits: (batch, seq_len, num_classes)
            attn_all: (batch, seq_len, N) attention 重みも返す（self._last_attn に保存）
        """
        cfg = self.config
        batch = x.shape[0]
        device = x.device

        if hard_attention is None:
            hard_attention = not self.training

        v_enc, k_enc = self.encode(x)  # (B, N, d_val), (B, N, d_attn)

        # 初期状態と BOS 値（sigmoid で [0,1] に）
        state = torch.sigmoid(self.state_init_logits).unsqueeze(0).expand(batch, -1)
        prev_val = torch.sigmoid(self.bos_val_logits).unsqueeze(0).expand(batch, -1)

        used_mask = torch.zeros(batch, cfg.seq_len, device=device, dtype=v_enc.dtype)

        logits_list = []
        attn_list = []

        for t in range(cfg.seq_len):
            # Decoder cell: 状態更新
            cell_in = torch.cat([state, prev_val], dim=-1)
            state = self.decoder_cell(cell_in)

            # Query 生成
            query = self.query_producer(state)  # (B, d_attn)

            # XNOR 類似度
            q_exp = query.unsqueeze(1)   # (B, 1, d_attn)
            xnor = q_exp * k_enc + (1.0 - q_exp) * (1.0 - k_enc)  # (B, N, d_attn)
            raw_scores = xnor.sum(dim=-1)  # (B, N)

            # マスク適用
            scores = raw_scores - cfg.mask_scale * used_mask

            # Attention
            if hard_attention:
                idx = scores.argmax(dim=-1)  # (B,)
                attn = F.one_hot(idx, num_classes=cfg.seq_len).to(v_enc.dtype)
                selected = v_enc[torch.arange(batch, device=device), idx]  # (B, d_val)
            else:
                attn = F.softmax(scores / cfg.tau_attn, dim=-1)  # (B, N)
                selected = torch.einsum('bn,bnd->bd', attn, v_enc)

            # Output
            h = self.output_decoder(selected)
            logits = self.group_sum(h)
            logits_list.append(logits)
            attn_list.append(attn)

            # Teacher forcing or free-running: 次ステップの prev_val を決定
            if target_pointers is not None:
                # Teacher forcing: 正解ポインタで v_enc をルックアップ
                tgt_idx = target_pointers[:, t]  # (B,)
                prev_val = v_enc[torch.arange(batch, device=device), tgt_idx]
            else:
                prev_val = selected

            # マスク累積
            used_mask = used_mask + attn

        logits_all = torch.stack(logits_list, dim=1)  # (B, seq_len, C)
        attn_all = torch.stack(attn_list, dim=1)      # (B, seq_len, N)

        self._last_attn = attn_all.detach()
        return logits_all, attn_all

    @property
    def last_attention(self) -> torch.Tensor | None:
        return self._last_attn

    def count_parameters(self) -> dict[str, int]:
        counts = {}
        counts['value_encoder'] = sum(p.numel() for p in self.value_encoder.parameters())
        counts['key_encoder'] = sum(p.numel() for p in self.key_encoder.parameters())
        counts['decoder_cell'] = sum(p.numel() for p in self.decoder_cell.parameters())
        counts['query_producer'] = sum(p.numel() for p in self.query_producer.parameters())
        counts['output_decoder'] = sum(p.numel() for p in self.output_decoder.parameters())
        counts['state_init'] = self.state_init_logits.numel()
        counts['bos_val'] = self.bos_val_logits.numel()
        counts['total'] = sum(counts.values())
        return counts
