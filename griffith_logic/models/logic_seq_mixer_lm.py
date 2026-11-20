"""LogicSeqMixerLM — decoder-only の Mixer 文字レベル言語モデル(最小実証)。

目的は **「Mixer(attention不使用の静的Toeplitz混合)で LLM 的な自己回帰生成が
成立するか」を最小コストで確かめること**であり、規模を追わない。よって

    - encoder / context / cross-attention をすべて外す(decoder-only)
    - :mod:`griffith_logic.models.logic_seq_mixer` の
      :class:`ToeplitzTokenMix` (causal=True) と :class:`LogicSeqMixerBlock` を
      **import して再利用**する(再実装しない。元モジュールは読み取りのみ)
    - ゲート数を既存の char-LM ベースライン
      (``experiments/train_rddlgn_charlm.py`` の既定 = 1,954 ゲート)へ揃える

構成::

    tokens (B,T)
      → LogicEmbedding(V, d_emb)            [eval時 Heaviside で {0,1}]
      → 位置ビット連結(LearnedPositionBits)
      → embed_proj: LutStack(d_emb+pos → d_channels)
      → LogicSeqMixerBlock(causal=True) × n_blocks
      → head: LutStack(d_channels → m_layer_sizes)
      → GroupSum(k=vocab_size)
      → logits (B,T,V)

設計判断:
    - **因果性は構造で保証する**。トークン混合は :class:`ToeplitzTokenMix` の
      因果窓 ``[t-K+1, t]`` のみを見るため、上三角マスクを一切使わずに
      「位置 t の出力は入力 x[0..t] にのみ依存する」が成り立つ。他の全経路
      (埋め込み・位置ビット・channel mix・XOR残差・head・GroupSum)は
      位置ごとに独立に適用されるので時間方向の漏れ口が無い。
      ``tests/integration/test_logic_seq_mixer_lm.py`` で全 t を掃引して
      ビット厳密に検証している。
    - **位置符号化は必須**。Toeplitz混合はシフト不変なので、それ自体では絶対位置を
      持たない(logic_seq_mixer.py の同名設計判断を参照)。char-LM では文頭からの
      距離が効くため既定で ``positional='learned_bits'`` を入れる。位置ビットは
      LUTニューロンを持たない定数なのでゲート数には現れない。
    - **ブロック数の上限は元モデルと同じ4**。MIXER-2 の実測で soft-XOR 残差は
      1ブロックあたり約1e-2倍の勾配減衰を起こし、n_blocks>=6 で学習が始まらない。
      本モジュールは
      encoder を持たないため元の ``n_enc_blocks+n_dec_blocks<=4`` を
      ``n_blocks<=4`` として引き継ぐ。
    - **soft/hard は ``self.training`` に一元化**する(ライブラリ全体の規約)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..group_sum import GroupSum
from .logic_seq_mixer import (
    _MAX_BLOCKS,
    LearnedPositionBits,
    LogicSeqMixerBlock,
    LogicSeqMixerConfig,
)
from .rddlgn import LogicEmbedding, LutStack


@dataclass
class LogicSeqMixerLMConfig:
    """decoder-only Mixer 文字LM のハイパーパラメータ。

    既定値は ``experiments/train_rddlgn_charlm.py`` の既定構成(1,954ゲート)へ
    ゲート数を揃えた設計である(本モデル 1,967ゲート、差 +0.67%):

        embed_proj  96
        blocks      2 × (1×(192+96) + 30) = 636
        head        650 + 585            = 1,235
    """

    vocab_size: int = 65
    d_emb: int = 32
    context_len: int = 64
    d_channels: int = 96
    channel_hidden: int | None = 192
    n_blocks: int = 2
    channel_depth: int = 1
    token_kernels: tuple[int, ...] = (3, 8, 16)
    token_depth: int = 2
    gate: str = 'and'                # 'and' | 'none'
    residual_mode: str = 'xor'       # 'xor' | 'none'
    residual_init: bool = False
    m_layer_sizes: list[int] = field(default_factory=lambda: [650, 585])
    positional: str = 'learned_bits'  # 'learned_bits' | 'none'(アブレーション)
    n_pos_bits: int = 16
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    selection: str = 'soft'
    tau: float = 1.0                 # GroupSum温度(logits = 群和 / tau)
    gumbel: bool = True
    grad_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.n_blocks < 1:
            raise ValueError(f'n_blocksは1以上: {self.n_blocks}')
        if self.n_blocks > _MAX_BLOCKS:
            raise ValueError(
                f'n_blocksは{_MAX_BLOCKS}以下: {self.n_blocks}。'
                'MIXER-2でnb>=6はsoft-XOR勾配死により学習不能と実証済み'
            )
        # 残りの値域検証は LogicSeqMixerConfig.__post_init__ に委ねる
        # (単一情報源。to_block_config() を呼ぶことで発火させる)
        self.to_block_config()

    @property
    def pos_width(self) -> int:
        """埋め込みへ連結する位置ビット幅(positional='none' なら0)。"""
        return self.n_pos_bits if self.positional == 'learned_bits' else 0

    def to_block_config(self) -> LogicSeqMixerConfig:
        """:class:`LogicSeqMixerBlock` へ渡す設定を組み立てる。

        ブロックは ``n_enc_blocks`` / ``n_dec_blocks`` を参照しない(段数は本体側で
        ループする)ため、元設定の必須項目を満たす最小値 1/1 を入れてある。
        """
        return LogicSeqMixerConfig(
            vocab_size=self.vocab_size,
            d_emb=self.d_emb,
            seq_len=self.context_len,
            d_channels=self.d_channels,
            channel_hidden=self.channel_hidden,
            n_enc_blocks=1,
            n_dec_blocks=1,
            channel_depth=self.channel_depth,
            token_kernels=tuple(self.token_kernels),
            token_depth=self.token_depth,
            gate=self.gate,
            residual_mode=self.residual_mode,
            residual_init=self.residual_init,
            m_layer_sizes=tuple(self.m_layer_sizes),
            positional=self.positional,
            n_pos_bits=self.n_pos_bits,
            n_inputs=self.n_inputs,
            parameterization=self.parameterization,
            connections=self.connections,
            selection=self.selection,
            tau=self.tau,
            gumbel=self.gumbel,
            grad_factor=self.grad_factor,
        )


def estimate_gates_lm(config: LogicSeqMixerLMConfig) -> dict[str, int]:
    """モデルを実体化せずにゲート数(LUTニューロン数)を解析式で見積もる。

    :meth:`LogicSeqMixerLM.count_gates` と厳密に一致する(テストで照合)。
    """
    block_cfg = config.to_block_config()
    hidden = block_cfg.hidden_width

    embed_proj = config.d_channels
    per_block_channel = config.channel_depth * (hidden + config.d_channels)
    per_block_token = sum(
        (config.token_depth - 1) * k + 1 for k in config.token_kernels
    )
    blocks = config.n_blocks * (per_block_channel + per_block_token)
    head = sum(config.m_layer_sizes)

    counts = {
        'embed_proj': embed_proj,
        'blocks': blocks,
        'token_mix': config.n_blocks * per_block_token,
        'head': head,
    }
    # 'token_mix' は 'blocks' の内訳なので total からは除く
    counts['total'] = embed_proj + blocks + head
    return counts


class LogicSeqMixerLM(nn.Module):
    """decoder-only Mixer による char-level 言語モデル(attention不使用)。

    forward: (B, T) 整数トークンID -> (B, T, vocab_size) logits
    """

    def __init__(self, config: LogicSeqMixerLMConfig, device: str = 'cpu') -> None:
        super().__init__()
        self.config = config
        block_cfg = config.to_block_config()
        self.block_config = block_cfg

        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            gumbel=config.gumbel,
            selection=config.selection,
        )

        self.embedding = LogicEmbedding(config.vocab_size, config.d_emb)
        self.pos = (
            LearnedPositionBits(
                config.context_len + 2, config.n_pos_bits, device=device)
            if config.positional == 'learned_bits' else None
        )
        self.embed_proj = LutStack(
            config.d_emb + config.pos_width, [config.d_channels], **lut_kwargs)
        self.blocks = nn.ModuleList([
            LogicSeqMixerBlock(block_cfg, causal=True, device=device)
            for _ in range(config.n_blocks)
        ])
        self.head = LutStack(
            config.d_channels, list(config.m_layer_sizes), **lut_kwargs)
        self.group_sum = GroupSum(k=config.vocab_size, tau=config.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) 整数トークンID

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        if x.dim() != 2:
            raise ValueError(f'xは(B, T)の2次元テンソル: {tuple(x.shape)}')
        batch, seq_len = x.shape

        emb = self.embedding(x)                            # (B, T, d_emb)
        if self.pos is not None:
            bits = self.pos(seq_len).unsqueeze(0).expand(batch, -1, -1)
            emb = torch.cat([emb, bits], dim=-1)

        h = self.embed_proj(emb.reshape(batch * seq_len, -1))
        h = h.reshape(batch, seq_len, -1)                   # (B, T, d_channels)

        for block in self.blocks:
            h = block(h)

        m = self.head(h.reshape(batch * seq_len, -1))
        logits = self.group_sum(m).reshape(batch, seq_len, self.config.vocab_size)
        return logits

    def pop_aux_loss(self) -> torch.Tensor:
        """埋め込みの二値化正則化損失 mean(x*(1-x))。

        ``RDDLGNCharLM.pop_aux_loss`` と同一定義(同一プロトコル比較のため)。
        """
        return self.embedding.binary_reg_loss()

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, greedy: bool = True,
    ) -> torch.Tensor:
        """自己回帰生成。系列は context_len で左から切り詰める。

        ``RDDLGNCharLM.generate`` と同じ流儀(毎ステップ全系列を再評価)。
        再帰状態を持たない本モデルでは context_len が有効文脈の上限になる。
        """
        context_len = self.config.context_len
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]
            logits = self(idx_cond)          # (B, T, V)
            next_logits = logits[:, -1, :]
            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す('total'キー必須)。"""
        counts = {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'pos': sum(p.numel() for p in self.pos.parameters())
            if self.pos is not None else 0,
            'embed_proj': sum(p.numel() for p in self.embed_proj.parameters()),
            'blocks': sum(p.numel() for p in self.blocks.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
        }
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts

    @staticmethod
    def _gates_of(root: nn.Module | None) -> int:
        """rootの部分木に含まれるLUTニューロン数。"""
        if root is None:
            return 0
        return sum(
            int(m.num_neurons) for m in root.modules() if hasattr(m, 'num_neurons')
        )

    def count_gates(self) -> dict[str, int]:
        """ゲート数(LUTニューロン数)の内訳。'total'は全モジュール走査と一致する。"""
        token_mix = sum(self._gates_of(b.token_mix) for b in self.blocks)
        counts = {
            'embed_proj': self._gates_of(self.embed_proj),
            'blocks': sum(self._gates_of(b) for b in self.blocks),
            'token_mix': token_mix,
            'head': self._gates_of(self.head),
        }
        counts['total'] = (
            counts['embed_proj'] + counts['blocks'] + counts['head']
        )
        return counts

    def extra_repr(self) -> str:
        c = self.config
        return (
            f'vocab={c.vocab_size}, d_emb={c.d_emb}, context_len={c.context_len}, '
            f'd_channels={c.d_channels}, hidden={self.block_config.hidden_width}, '
            f'n_blocks={c.n_blocks}, channel_depth={c.channel_depth}, '
            f'kernels={tuple(c.token_kernels)}, gate={c.gate}, '
            f'residual={c.residual_mode}, '
            f'positional={c.positional}({c.pos_width}bit)'
        )


def bits_per_character(loss_nats: float) -> float:
    """交差エントロピー(nats/文字)を bpc(bits per character)へ変換する。

    char-LM の標準的な定義 ``bpc = loss_nats / ln(2)`` そのもの。既存の char-LM
    スクリプト(``train_rddlgn_charlm.py`` / ``train_logic_gpt.py``)は
    ``F.cross_entropy(..., reduction='sum') / 総トークン数`` を val_loss として
    出しており、これは1文字あたりの nats なので、この式で bpc に直せる。
    社内比較点(val_loss 4.7266)にも同じ式を適用する。
    """
    return loss_nats / math.log(2.0)


__all__ = [
    'LogicSeqMixerLM',
    'LogicSeqMixerLMConfig',
    'bits_per_character',
    'estimate_gates_lm',
]
