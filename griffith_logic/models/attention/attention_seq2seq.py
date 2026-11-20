"""
DLGN Attention-based seq2seq モデル。

Task 4: seq2seqソート
  入力系列 [V0, V1, ..., V(N-1)] を受け取り、ソートされた系列を出力。
  固定長 seq2seq（入力長=出力長=N）。

アーキテクチャ:
  - Encoder: 各入力トークンを value_encoder で射影 → memory (B, N, d_val)
             同時に位置インデックスを key_encoder で射影 → keys (N, d_attn)
  - Decoder (各ステップ t):
      直前の出力トークン prev_t-1 を value_encoder で射影
      + ステップ位置インデックス t を step_encoder で射影 → query q_t (d_attn)
      XNOR類似度(q_t, keys) → attn (N)
      selected = Σ attn_i * memory_i (d_val)
      output_decoder(concat(selected, prev_t-1_projected)) → logits_t

RDDLGN より単純化:
  - BOS/EOS 不要（固定長）
  - リカレント状態なし（Attentionで十分）
  - Encoderは feed-forward（K-layersなし）
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...group_sum import GroupSum
from ..rddlgn import LutStack


@dataclass
class AttentionSeq2SeqConfig:
    """DLGNAttentionSeq2Seq のハイパーパラメータ。"""

    seq_len: int = 4
    value_bits: int = 4

    # LutStack サイズ
    encoder_sizes: list = field(default_factory=lambda: [32, 16])  # key/query射影
    value_encoder_sizes: list = field(default_factory=lambda: [32, 16])
    decoder_sizes: list = field(default_factory=lambda: [64, 32])

    # LutLayer 設定
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    grad_factor: float = 1.0
    selection: str = 'soft'  # 'soft' | 'hard_st' | 'gumbel_st'

    # Attention 設定
    tau_attn: float = 1.0
    # seq2seqでは出力位置→入力位置の異なるマッピングを学ぶため、
    # key/query encoder は別々にするのがデフォルト
    share_key_query: bool = False

    # 出力設定
    tau_out: float = 1.0

    # WARP 設定
    tau_warp: float = 1.0
    gumbel: bool = True

    @property
    def pos_bits(self) -> int:
        """seq_len を表すのに必要なビット数。"""
        import math
        return max(1, math.ceil(math.log2(self.seq_len)))

    @property
    def input_dim(self) -> int:
        return self.seq_len * self.value_bits

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
    def tiny(cls) -> 'AttentionSeq2SeqConfig':
        return cls()


def _build_position_indices(num_slots: int, pos_bits: int, device: str = 'cpu') -> torch.Tensor:
    """0..num_slots-1 を MSB ファーストで二進数化。"""
    idx = torch.zeros(num_slots, pos_bits, dtype=torch.float32, device=device)
    for i in range(num_slots):
        for b in range(pos_bits):
            idx[i, b] = (i >> (pos_bits - 1 - b)) & 1
    return idx


class DLGNAttentionSeq2Seq(nn.Module):
    """
    DLGN Boolean Attention を使った固定長 seq2seq モデル。

    クエリは「出力位置 t」の位置エンコーディングに基づく。
    これにより「t番目に欲しい情報はどこにあるか」をAttentionで学習する。
    シンプルな「1ヘッド、cross-attention のみ」の Transformer 相当。
    """

    def __init__(self, config: AttentionSeq2SeqConfig, device: str = 'cpu'):
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
            selection=config.selection,
        )

        # 入力位置 / 出力位置 共通のインデックステーブル
        pos_table = _build_position_indices(config.seq_len, config.pos_bits, device)
        self.register_buffer('position_indices', pos_table)

        # key_encoder: 入力位置 pos_bits → d_attn （入力スロット用）
        self.key_encoder = LutStack(
            config.pos_bits, config.encoder_sizes, **lut_kwargs,
        )
        # query_encoder: 出力ステップ pos_bits → d_attn
        if config.share_key_query:
            self.query_encoder = self.key_encoder
        else:
            self.query_encoder = LutStack(
                config.pos_bits, config.encoder_sizes, **lut_kwargs,
            )

        # value_encoder: 入力トークン value_bits → d_val
        self.value_encoder = LutStack(
            config.value_bits, config.value_encoder_sizes, **lut_kwargs,
        )

        # output_decoder: d_val → decoder_sizes[-1]
        self.output_decoder = LutStack(
            config.d_val, config.decoder_sizes, **lut_kwargs,
        )

        assert config.decoder_sizes[-1] % config.num_classes == 0, (
            f'decoder_sizes[-1]={config.decoder_sizes[-1]} は '
            f'num_classes={config.num_classes} で割り切れる必要があります'
        )
        self.group_sum = GroupSum(k=config.num_classes, tau=config.tau_out)

        self._last_attn: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len * value_bits)

        Returns:
            memory: (batch, seq_len, d_val)
            keys:   (seq_len, d_attn)  ← 位置キー、全サンプル共通
        """
        cfg = self.config
        batch = x.shape[0]

        # values: (batch, seq_len, value_bits)
        values = x.reshape(batch, cfg.seq_len, cfg.value_bits)

        # memory: value_encoder を各位置に並列適用
        v_flat = values.reshape(batch * cfg.seq_len, cfg.value_bits)
        memory = self.value_encoder(v_flat).reshape(batch, cfg.seq_len, -1)

        # keys: 位置インデックスを key_encoder に通す
        keys = self.key_encoder(self.position_indices)  # (seq_len, d_attn)

        return memory, keys

    def forward(
        self, x: torch.Tensor, hard_attention: bool | None = None,
    ) -> torch.Tensor:
        """
        全ての出力ステップを並列に計算（Teacher Forcingと同等だが、
        本モデルは出力をデコーダ入力に使わないため完全並列可）。

        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch, seq_len, num_classes)
        """
        cfg = self.config
        batch = x.shape[0]

        if hard_attention is None:
            hard_attention = not self.training

        memory, keys = self.encode(x)  # (B, N, d_val), (N, d_attn)

        # 出力側クエリ: 全ての出力位置を並列に処理
        # query_encoder に位置インデックスを通す → (seq_len, d_attn)
        q_all = self.query_encoder(self.position_indices)  # (N, d_attn)

        # XNOR 類似度: (N_out, D) vs (N_in, D) → (N_out, N_in)
        # batch次元はあとで broadcast
        q_exp = q_all.unsqueeze(1)  # (N_out, 1, D)
        k_exp = keys.unsqueeze(0)   # (1, N_in, D)
        xnor = q_exp * k_exp + (1.0 - q_exp) * (1.0 - k_exp)
        scores = xnor.sum(dim=-1)  # (N_out, N_in)

        # scores は (N_out, N_in) で batch依存しない（Attention全体が出力位置のみの関数）
        # 各バッチに同じ attention が適用される
        if hard_attention:
            idx = scores.argmax(dim=-1)  # (N_out,)
            # selected: memory[:, idx, :] で各出力位置が単一入力位置を選択
            selected = memory[:, idx, :]  # (B, N_out, d_val)
            attn = F.one_hot(idx, num_classes=cfg.seq_len).to(memory.dtype)
            attn = attn.unsqueeze(0).expand(batch, -1, -1)  # (B, N_out, N_in)
        else:
            attn = F.softmax(scores / cfg.tau_attn, dim=-1)  # (N_out, N_in)
            # selected: (B, N_out, d_val) = einsum over N_in
            selected = torch.einsum('ji,bid->bjd', attn, memory)
            attn = attn.unsqueeze(0).expand(batch, -1, -1)

        self._last_attn = attn.detach()

        # Decode: (B * N_out, d_val) → logits
        dec_flat = self.output_decoder(
            selected.reshape(batch * cfg.seq_len, cfg.d_val),
        )
        logits = self.group_sum(dec_flat).reshape(
            batch, cfg.seq_len, cfg.num_classes,
        )
        return logits

    @property
    def last_attention(self) -> torch.Tensor | None:
        """attention: (batch, seq_len_out, seq_len_in)。"""
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
