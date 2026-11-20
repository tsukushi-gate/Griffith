"""
DLGNAttentionTranslator - Bahdanau-style Attention seq2seq

RDDLGN (Bührer et al., 2025) の構造を踏襲しつつ、デコーダに Cross-Attention を追加。
エンコーダは並列（リカレント削除）、デコーダは RNN + XNOR類似度ベース Cross-Attention。

アーキテクチャ:
  Encoder (parallel):
    src → src_embedding(σ) → N-layers(LutStack) → memory (B, src_len, d_mem)
    src → src_embedding(σ) + position_indices → key_encoder → src_keys (B, src_len, d_attn)

  Decoder (recurrent, for each t):
    tgt_t → tgt_embedding(σ) → L-layers → l_t
    p_t = P-layers([p_{t-1}, l_t, context_{t-1}])
    query_t = query_producer(p_t)
    scores_t = XNOR(query_t, src_keys)
    attn_t = softmax(scores_t / τ)
    context_t = attn_t @ memory
    m_input = concat(p_t, l_t, context_t)
    logits_t = group_sum(M-layers(m_input))

RDDLGNとの違い:
  - K-layers (リカレントエンコーダ) を削除 → 並列化
  - Cross-Attention が memory 全体を動的に参照 → 系列長制限突破
  - Decoder P-layers の入力に前ステップ context も追加 → 情報統合
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...group_sum import GroupSum
from ...training.spec import LearningSpec, with_learning
from ..rddlgn import LogicEmbedding, LutStack

# dfa_targetsに指定できる論理名。
#   'encoder' — AttentionEncoderのn_layers/key_encoder(並列、単発呼び出し)
#   'l_layers' — AttentionDecoderのl_layers(並列、単発呼び出し)
# cross-attention(query_producer)とP-layers/M-layers(RNNステップ、
# タイムステップごとに再帰状態p_prev/context_prevを介して結合)はBPのまま
# (現行DFAフック実装は「単発呼び出し」を前提とするため、リカレントに
# 状態を跨ぐP/M-layersへの適用はスコープ外。文献知見(Launay 2020)通り
# attention経路がBPを要する点とも整合する)。
DFA_TARGET_CHOICES = ('encoder', 'l_layers')


def _build_position_indices(num_slots: int, pos_bits: int) -> torch.Tensor:
    """0..num_slots-1 を MSB ファーストで二進数化した (num_slots, pos_bits)。"""
    idx = torch.zeros(num_slots, pos_bits, dtype=torch.float32)
    for i in range(num_slots):
        for b in range(pos_bits):
            idx[i, b] = (i >> (pos_bits - 1 - b)) & 1
    return idx


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class TranslatorConfig:
    """DLGNAttentionTranslator のハイパーパラメータ。"""

    vocab_size: int = 50
    d_emb: int = 32
    src_len: int = 16
    tgt_len: int = 16

    # LutStack サイズ
    n_layer_sizes: list = field(default_factory=lambda: [64, 64])       # Encoder N-layers
    key_encoder_sizes: list = field(default_factory=lambda: [32, 16])   # key projection
    l_layer_sizes: list = field(default_factory=lambda: [64, 64])       # Decoder L-layers
    p_layer_sizes: list = field(default_factory=lambda: [96, 64])       # Decoder P-layers (RNN)
    query_producer_sizes: list = field(default_factory=lambda: [32, 16])
    m_layer_sizes: list = field(default_factory=lambda: [128, 128, 300])

    # LutLayer 設定
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    grad_factor: float = 1.0
    # ゲート選択方式(訓練時): 'soft'=連続ブレンド / 'hard_st'=argmax forward+STE backward
    # (CAGE方式)。hard_st は訓練から離散hard経路を通し soft-hard gap を縮める。
    # 'gumbel_st'=argmax(weights+Gumbel) forward+STE backward(Mind the Gap方式)。
    selection: str = 'soft'
    # 入力配線(訓練時): 'fixed'=random固定 / 'learnable'=DWN配線学習(各LutStackの第1層のみ)。
    # 「停滞は容量でなく配線」仮説の検証レバー。BP専用。lm_tauはlearnable時の温度。
    wiring: str = 'fixed'
    lm_tau: float = 0.001

    # Attention 設定
    tau_attn: float = 1.0

    # 出力設定
    tau: float = 1.0  # GroupSum 温度

    # 隠れ状態初期化
    init_mode: str = 'zero'
    init_scale: float = 0.5

    # 語彙共有
    shared_vocab: bool = True

    # WARP
    tau_warp: float = 1.0
    gumbel: bool = True

    @property
    def pos_bits(self) -> int:
        max_len = max(self.src_len, self.tgt_len)
        return max(1, math.ceil(math.log2(max_len)))

    @property
    def d_mem(self) -> int:
        return self.n_layer_sizes[-1]

    @property
    def d_attn(self) -> int:
        return self.key_encoder_sizes[-1]

    @property
    def d_l(self) -> int:
        return self.l_layer_sizes[-1]

    @property
    def d_p(self) -> int:
        return self.p_layer_sizes[-1]

    @classmethod
    def tiny(cls) -> 'TranslatorConfig':
        """合成タスク用デバッグ構成。"""
        vocab_size = 50
        group_factor = 20
        return cls(
            vocab_size=vocab_size,
            d_emb=128,
            src_len=16,
            tgt_len=16,
            n_layer_sizes=[512, 512],
            key_encoder_sizes=[256, 128],
            l_layer_sizes=[512, 512],
            p_layer_sizes=[1024, 512],
            query_producer_sizes=[256, 128],
            m_layer_sizes=[2000, 2000, vocab_size * group_factor],
            tau=1.0,
        )

    @classmethod
    def from_paper(cls, scale: float = 1.0) -> 'TranslatorConfig':
        """RDDLGN 論文Table 1 ベースでスケーリング（K-layersは削除、他はそのまま）。"""

        def _scale(sizes: list[int]) -> list[int]:
            return [max(4, int(s * scale)) for s in sizes]

        vocab_size = max(4, int(16000 * scale)) if scale < 1.0 else 16000
        d_emb = max(4, int(1024 * scale)) if scale < 1.0 else 1024

        m_sizes = _scale([400000, 400000, 480000])
        group_factor = max(1, m_sizes[-1] // vocab_size)
        m_sizes[-1] = vocab_size * group_factor

        return cls(
            vocab_size=vocab_size,
            d_emb=d_emb,
            src_len=16,
            tgt_len=16,
            n_layer_sizes=_scale([12000, 12000]),
            key_encoder_sizes=_scale([2000, 1000]),
            l_layer_sizes=_scale([12000, 12000]),
            p_layer_sizes=_scale([64000, 48000]),
            query_producer_sizes=_scale([2000, 1000]),
            m_layer_sizes=m_sizes,
        )


# ---------------------------------------------------------------------------
# Encoder (並列)
# ---------------------------------------------------------------------------

class AttentionEncoder(nn.Module):
    """
    並列エンコーダ。全トークンを同時に N-layers で処理。
    加えて、key_encoder で memory + 位置情報から attention key を生成。
    """

    def __init__(self, config: TranslatorConfig, device: str = 'cpu'):
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
            wiring=config.wiring,
            lm_tau=config.lm_tau,
        )

        # N-layers: d_emb → d_mem （全トークン並列）
        self.n_layers = LutStack(config.d_emb, config.n_layer_sizes, **lut_kwargs)

        # key_encoder: (d_mem + pos_bits) → d_attn
        self.key_encoder = LutStack(
            config.d_mem + config.pos_bits, config.key_encoder_sizes, **lut_kwargs,
        )

        # 位置インデックス（非学習）
        pos_table = _build_position_indices(config.src_len, config.pos_bits).to(device)
        self.register_buffer('position_indices', pos_table)

    def forward(self, src_embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_embedded: (B, src_len, d_emb) sigmoid後の埋め込み

        Returns:
            memory: (B, src_len, d_mem)
            src_keys: (B, src_len, d_attn)
        """
        batch, src_len, _ = src_embedded.shape

        # N-layers: 全トークンをバッチ化
        h = src_embedded.reshape(batch * src_len, -1)
        memory = self.n_layers(h).reshape(batch, src_len, -1)

        # key_encoder: memory + position
        pos_broadcast = self.position_indices.unsqueeze(0).expand(batch, -1, -1)
        key_inputs = torch.cat([memory, pos_broadcast], dim=-1)
        key_flat = key_inputs.reshape(batch * src_len, -1)
        src_keys = self.key_encoder(key_flat).reshape(batch, src_len, -1)

        return memory, src_keys


# ---------------------------------------------------------------------------
# Decoder (リカレント)
# ---------------------------------------------------------------------------

def _init_hidden(batch: int, dim: int, device: torch.device,
                 mode: str = 'zero', scale: float = 0.5) -> torch.Tensor:
    if mode == 'gaussian':
        return torch.sigmoid(torch.randn(batch, dim, device=device) * scale)
    if mode == 'one':
        return torch.ones(batch, dim, device=device)
    return torch.full((batch, dim), 0.5, device=device)


class AttentionDecoder(nn.Module):
    """
    リカレントデコーダ + Cross-Attention。

    各タイムステップ:
      l_t = L-layers(tgt_emb_t)
      p_t = P-layers([p_prev, l_t, context_prev])
      query_t = query_producer(p_t)
      attn_t = softmax(XNOR(query_t, src_keys) / τ)
      context_t = attn_t @ memory
      logits_t = GroupSum(M-layers([p_t, l_t, context_t]))
    """

    def __init__(self, config: TranslatorConfig, device: str = 'cpu'):
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
            wiring=config.wiring,
            lm_tau=config.lm_tau,
        )

        # L-layers: d_emb → d_l (並列)
        self.l_layers = LutStack(config.d_emb, config.l_layer_sizes, **lut_kwargs)

        # P-layers: (d_p + d_l + d_mem) → d_p （リカレント）
        p_in = config.d_p + config.d_l + config.d_mem
        self.p_layers = LutStack(p_in, config.p_layer_sizes, **lut_kwargs)

        # Query producer: d_p → d_attn
        self.query_producer = LutStack(
            config.d_p, config.query_producer_sizes, **lut_kwargs,
        )

        # M-layers: (d_p + d_l + d_mem) → m_last
        m_in = config.d_p + config.d_l + config.d_mem
        self.m_layers = LutStack(m_in, config.m_layer_sizes, **lut_kwargs)

        assert config.m_layer_sizes[-1] % config.vocab_size == 0, (
            f'm_layer_sizes[-1]={config.m_layer_sizes[-1]} は '
            f'vocab_size={config.vocab_size} で割り切れる必要があります'
        )
        self.group_sum = GroupSum(k=config.vocab_size, tau=config.tau)

        self.d_p = config.d_p
        self.d_mem = config.d_mem
        self.init_mode = config.init_mode
        self.init_scale = config.init_scale

        self._last_attn: torch.Tensor | None = None  # (B, tgt_len, src_len)

    def forward(
        self,
        target_embedded: torch.Tensor,
        memory: torch.Tensor,
        src_keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_embedded: (B, tgt_len, d_emb)
            memory: (B, src_len, d_mem)
            src_keys: (B, src_len, d_attn)

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        cfg = self.config
        batch, tgt_len, _ = target_embedded.shape
        device = target_embedded.device

        # L-layers: 並列
        l = target_embedded.reshape(batch * tgt_len, -1)
        l = self.l_layers(l).reshape(batch, tgt_len, -1)

        # 初期状態
        p_prev = _init_hidden(batch, self.d_p, device, self.init_mode, self.init_scale)
        context_prev = _init_hidden(
            batch, self.d_mem, device, self.init_mode, self.init_scale,
        )

        logits_list = []
        attn_list = []

        for t in range(tgt_len):
            l_t = l[:, t, :]

            # P-layers (RNN step)
            p_input = torch.cat([p_prev, l_t, context_prev], dim=-1)
            p_t = self.p_layers(p_input)

            # Cross-Attention
            query_t = self.query_producer(p_t)  # (B, d_attn)
            q_exp = query_t.unsqueeze(1)         # (B, 1, d_attn)
            xnor = q_exp * src_keys + (1.0 - q_exp) * (1.0 - src_keys)
            scores = xnor.sum(dim=-1)             # (B, src_len)
            attn = F.softmax(scores / cfg.tau_attn, dim=-1)  # (B, src_len)
            context_t = torch.einsum('bs,bsd->bd', attn, memory)  # (B, d_mem)

            # M-layers
            m_input = torch.cat([p_t, l_t, context_t], dim=-1)
            m_out = self.m_layers(m_input)
            logits_t = self.group_sum(m_out)

            logits_list.append(logits_t)
            attn_list.append(attn)

            # 状態更新
            p_prev = p_t
            context_prev = context_t

        self._last_attn = torch.stack(attn_list, dim=1).detach()
        return torch.stack(logits_list, dim=1)

    @property
    def last_attention(self) -> torch.Tensor | None:
        return self._last_attn


# ---------------------------------------------------------------------------
# DLGNAttentionTranslator
# ---------------------------------------------------------------------------

class DLGNAttentionTranslator(nn.Module):
    """
    Bahdanau-style Attention seq2seq。

    Encoder (並列) + Decoder (リカレント) + Cross-Attention。
    RDDLGN (Bührer et al., 2025) と同じインターフェース:
        forward(src_tokens, tgt_tokens) -> (logits, emb_reg_loss)
    """

    def __init__(
        self,
        config: TranslatorConfig,
        device: str = 'cpu',
        dfa_targets: list[str] | None = None,
        dfa_lr: float | None = None,
    ):
        """
        Args:
            config: TranslatorConfig
            device: パラメータ配置デバイス
            dfa_targets: with_learning(rule='dfa')で注釈するモジュール群。
                None(既定)= 注釈なし = forward/学習ともに従来のBP完全維持
                (非回帰)。'encoder'/'l_layers'を指定可能
                (DFA_TARGET_CHOICES参照)。cross-attention(query_producer)
                とリカレントなP-layers/M-layers(タイムステップを跨いで
                p_prev/context_prevを共有する)はスコープ外(常にBP)。
            dfa_lr: DFA注釈モジュールのLearningSpec.lr(Noneならトレーナ既定)。
        """
        super().__init__()
        self.config = config
        self.dfa_targets = list(dfa_targets) if dfa_targets else []
        for t in self.dfa_targets:
            assert t in DFA_TARGET_CHOICES, (
                f'dfa_targets の要素は {DFA_TARGET_CHOICES} のいずれか: {t}'
            )

        # Embedding
        self.src_embedding = LogicEmbedding(config.vocab_size, config.d_emb)
        if config.shared_vocab:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = LogicEmbedding(config.vocab_size, config.d_emb)

        # Encoder / Decoder
        self.encoder = AttentionEncoder(config, device=device)
        self.decoder = AttentionDecoder(config, device=device)

        self._annotate_dfa(dfa_lr)

    def _annotate_dfa(self, dfa_lr: float | None) -> None:
        """dfa_targetsに従いwith_learning(rule='dfa')を注釈する。

        dfa_targets=[](既定)なら何もしない(全モジュール未注釈=完全BP)。
        """
        if not self.dfa_targets:
            return
        spec_kwargs = {} if dfa_lr is None else {'lr': dfa_lr}
        if 'encoder' in self.dfa_targets:
            with_learning(self.encoder.n_layers, LearningSpec(rule='dfa', **spec_kwargs))
            with_learning(self.encoder.key_encoder, LearningSpec(rule='dfa', **spec_kwargs))
        if 'l_layers' in self.dfa_targets:
            with_learning(self.decoder.l_layers, LearningSpec(rule='dfa', **spec_kwargs))

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_tokens: (B, src_len) 整数トークンID
            tgt_tokens: (B, tgt_len) teacher forcing 用

        Returns:
            logits: (B, tgt_len, vocab_size)
            emb_reg_loss: スカラー（二値化正則化）
        """
        src_emb = self.src_embedding(src_tokens)
        tgt_emb = self.tgt_embedding(tgt_tokens)

        memory, src_keys = self.encoder(src_emb)
        logits = self.decoder(tgt_emb, memory, src_keys)

        emb_reg_loss = self.src_embedding.binary_reg_loss()
        if not self.config.shared_vocab:
            emb_reg_loss = (emb_reg_loss + self.tgt_embedding.binary_reg_loss()) / 2.0

        return logits, emb_reg_loss

    def count_parameters(self) -> dict[str, int]:
        counts = {}
        counts['embedding'] = sum(p.numel() for p in self.src_embedding.parameters())
        if not self.config.shared_vocab:
            counts['embedding'] += sum(p.numel() for p in self.tgt_embedding.parameters())
        counts['n_layers'] = sum(p.numel() for p in self.encoder.n_layers.parameters())
        counts['key_encoder'] = sum(p.numel() for p in self.encoder.key_encoder.parameters())
        counts['l_layers'] = sum(p.numel() for p in self.decoder.l_layers.parameters())
        counts['p_layers'] = sum(p.numel() for p in self.decoder.p_layers.parameters())
        counts['query_producer'] = sum(p.numel() for p in self.decoder.query_producer.parameters())
        counts['m_layers'] = sum(p.numel() for p in self.decoder.m_layers.parameters())
        counts['total'] = sum(counts.values())
        return counts
