"""FPSeqMixer — LogicSeqMixer の float(通常NN)対照モデル。

:mod:`griffith_logic.models.logic_seq_mixer` と **トポロジ・評価プロトコルを
同一に保ったまま**、論理ゲート(LutLayer / LogicEmbedding / GroupSum)を通常の
浮動小数点部品(`nn.Embedding` / `nn.Linear` + GELU / 出力 Linear)へ置き換えた
対照モデルである。目的は次の2つの差を単離すること:

    (a) FP BLEU − 論理版 soft BLEU  … 構造の表現力差(連続 vs 緩和論理)
    (b) FP BLEU − 論理版 hard BLEU  … 「回路化の総コスト」(構造差 + 離散化損)

設計判断(重要):
    - **「同じ次元」で揃えることは不可能なので「同一パラメータ予算」で揃える**。
      LutLayer は fan-in=2 固定なので out_dim=400,000 の層でもパラメータは
      400,000×16=6.4M で済むが、FP で幅400,000 の全結合を作ると 1.6×10^11
      パラメータになり実行不能かつ無意味である。よって本モデルは
      **トポロジ(ブロック段数・Toeplitz 静的因果混合・カーネル幅 (3,8,16)・
      チャネル混合・context 経路・出力ヘッド)を LogicSeqMixer と同型に保ち、
      幅 `d_model` だけを調整して総パラメータ数を論理版の 40,847,232 へ
      合わせる**。既定 :meth:`FPSeqMixerConfig.from_paper` の d_model=480 が
      その値で、実測 40,660,648(論理版比 -0.46%、要求の ±30% 以内)である。
      cross アーム(`cross_attention='single_head'`)は 41,585,128(+1.81%)。
    - **ゲート数とパラメータ数は単位が違う**。LUT ニューロン数(ゲート数)と
      float パラメータ数を割り算・倍率比較してはならない。レポートでも併記のみ
      とすること。
    - **因果性は窓の構造で保証する**。論理版と同じく上三角マスクではなく、
      位置 t の窓を ``[t-K+1, t]`` に張ることで構造的に未来を見えなくする
      (`ToeplitzTokenMixFP`)。混合重みは全時刻・グループ内全チャネルで共有され、
      これが Toeplitz(シフト不変)構造の実体である点も論理版と同一。
    - **FP 側でのみ導入した差分は2つだけ**で、いずれも「float で学習を成立させる
      ための最小限の置換」である: (1) 活性を GELU にする(LUT の代替)、
      (2) 各ブロックに pre-LayerNorm を置く(論理版の出力域 [0,1] による
      暗黙の正規化に相当するものが float には無いため)。`norm='none'` で
      アブレーションできる。
    - **残差は加算**にする。論理版の soft-XOR は x,h∈{0,1} 上の XOR の緩和で
      あり、float 表現では通常の加算残差が対応物である。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 設定値域(ValueErrorメッセージと検証の単一情報源)
_FP_GATES = ('mul', 'none')
_FP_RESIDUAL_MODES = ('add', 'none')
_FP_CONTEXT_MODES = ('broadcast', 'gather', 'concat_all')
_FP_CROSS_ATTENTION = ('none', 'single_head')
_FP_POSITIONAL = ('learned', 'none')
_FP_NORMS = ('layer', 'none')

# ブロック段数の上限(論理版 _MAX_BLOCKS と同一。トポロジを揃えるため)
_FP_MAX_BLOCKS = 4


@dataclass
class FPSeqMixerConfig:
    """FPSeqMixer の全ハイパーパラメータ。

    既定値は :meth:`from_paper`(論理版とパラメータ予算を揃えた構成)と同値。
    スモークは :meth:`tiny` を使うこと。
    """

    vocab_size: int = 16000
    d_emb: int = 1024
    seq_len: int = 16
    d_model: int = 480
    channel_hidden: int | None = None   # 既定 None -> d_model*2
    n_enc_blocks: int = 2
    n_dec_blocks: int = 2
    channel_depth: int = 2              # チャネル混合MLPの段数(論理版と同数)
    token_kernels: tuple[int, ...] = (3, 8, 16)
    token_depth: int = 2                # トークン混合MLPの段数(論理版と同数)
    gate: str = 'mul'                   # 'mul'(SGUの u1⊙v) | 'none'
    residual_mode: str = 'add'          # 'add' | 'none'
    norm: str = 'layer'                 # 'layer'(pre-LN) | 'none'
    positional: str = 'learned'         # 'learned' | 'none'
    n_pos: int = 64                     # 位置ベクトルの次元(論理版 n_pos_bits と同数)
    context_mode: str = 'gather'        # 'broadcast' | 'gather' | 'concat_all'
    cross_attention: str = 'none'       # 'none' | 'single_head'
    cross_dim: int | None = None        # 既定 None -> d_model
    dropout: float = 0.0
    shared_vocab: bool = True

    def __post_init__(self) -> None:
        n_blocks = self.n_enc_blocks + self.n_dec_blocks
        if self.n_enc_blocks < 1 or self.n_dec_blocks < 1:
            raise ValueError(
                f'n_enc_blocks/n_dec_blocksは1以上: '
                f'{self.n_enc_blocks}/{self.n_dec_blocks}'
            )
        if n_blocks > _FP_MAX_BLOCKS:
            raise ValueError(
                f'n_enc_blocks+n_dec_blocks は{_FP_MAX_BLOCKS}以下: '
                f'{self.n_enc_blocks}+{self.n_dec_blocks}={n_blocks}'
                '(論理版 LogicSeqMixer とトポロジを揃えるための制約)'
            )
        if self.gate not in _FP_GATES:
            raise ValueError(f'gateは{_FP_GATES}のいずれか: {self.gate}')
        if self.residual_mode not in _FP_RESIDUAL_MODES:
            raise ValueError(
                f'residual_modeは{_FP_RESIDUAL_MODES}のいずれか: {self.residual_mode}')
        if self.norm not in _FP_NORMS:
            raise ValueError(f'normは{_FP_NORMS}のいずれか: {self.norm}')
        if self.context_mode not in _FP_CONTEXT_MODES:
            raise ValueError(
                f'context_modeは{_FP_CONTEXT_MODES}のいずれか: {self.context_mode}')
        if self.cross_attention not in _FP_CROSS_ATTENTION:
            raise ValueError(
                f'cross_attentionは{_FP_CROSS_ATTENTION}のいずれか: '
                f'{self.cross_attention}')
        if self.positional not in _FP_POSITIONAL:
            raise ValueError(
                f'positionalは{_FP_POSITIONAL}のいずれか: {self.positional}')
        if self.positional == 'learned' and self.n_pos < 1:
            raise ValueError(f'n_posは1以上: {self.n_pos}')
        if len(self.token_kernels) < 1:
            raise ValueError('token_kernelsは1要素以上必要')
        if any(k < 1 for k in self.token_kernels):
            raise ValueError(f'token_kernelsの各要素は1以上: {self.token_kernels}')
        if self.token_depth < 1:
            raise ValueError(f'token_depthは1以上: {self.token_depth}')
        if self.channel_depth < 1:
            raise ValueError(f'channel_depthは1以上: {self.channel_depth}')
        hidden = self.hidden_width
        if hidden % 2 != 0:
            raise ValueError(
                f'channel_hidden({hidden})はchunk(2)のため偶数である必要がある')
        n_groups = len(self.token_kernels)
        if self.token_width % n_groups != 0:
            raise ValueError(
                f'channel_hidden//2({self.token_width})はtoken_kernelsの本数'
                f'({n_groups})で割り切れる必要がある。'
                f'余り={self.token_width % n_groups}'
            )
        if self.vocab_size < 1:
            raise ValueError(f'vocab_sizeは1以上: {self.vocab_size}')

    @property
    def hidden_width(self) -> int:
        """チャネル混合の中間幅(既定は d_model*2)。"""
        return self.channel_hidden if self.channel_hidden is not None \
            else self.d_model * 2

    @property
    def token_width(self) -> int:
        """トークン混合が担当する幅(SGUのsplit後 = channel_hidden//2)。"""
        return self.hidden_width // 2

    @property
    def pos_width(self) -> int:
        """埋め込みへ連結する位置ベクトル幅(positional='none' なら0)。"""
        return self.n_pos if self.positional == 'learned' else 0

    @property
    def ctx_width(self) -> int:
        """decoderへ渡す文脈ベクトルの幅(context_modeで決まる)。"""
        if self.context_mode == 'concat_all':
            return self.seq_len * self.d_model
        return self.d_model

    @property
    def pos_table_len(self) -> int:
        """位置テーブルの行数(= 受け付ける最大系列長)。

        論理版と同一の理由で seq_len+2 を確保する(teacher forcing のデコーダ
        入力が ``[BOS, tgt, EOS][:-1]`` で seq_len+1 になるため)。
        """
        return self.seq_len + 2

    @classmethod
    def from_paper(cls) -> FPSeqMixerConfig:
        """論理版 `LogicSeqMixerConfig.from_paper()` とパラメータ予算を揃えた構成。

        vocab / d_emb / seq_len / ブロック段数 / カーネル幅は論理版と同一で、
        `d_model=480` だけが FP 側の自由度である。総パラメータ実測 40,660,648 は
        論理版 40,847,232 の **-0.46%**(要求 ±30% 以内)。
        """
        return cls(
            vocab_size=16000,
            d_emb=1024,
            seq_len=16,
            d_model=480,
            channel_hidden=None,      # -> 960
            n_enc_blocks=2,
            n_dec_blocks=2,
            channel_depth=2,
            token_kernels=(3, 8, 16),
            token_depth=2,
            n_pos=64,
            context_mode='gather',
        )

    @classmethod
    def tiny(cls) -> FPSeqMixerConfig:
        """スモーク用の最小構成(論理版 tiny と同じ vocab/seq_len)。"""
        return cls(
            vocab_size=50,
            d_emb=64,
            seq_len=8,
            d_model=24,
            channel_hidden=48,
            n_pos=16,
            n_enc_blocks=1,
            n_dec_blocks=1,
            channel_depth=2,
            token_kernels=(3, 4, 8),
            token_depth=2,
            context_mode='gather',
        )


def estimate_parameters(config: FPSeqMixerConfig) -> int:
    """モデルを実体化せずに総パラメータ数を解析式で見積もる。

    :meth:`FPSeqMixer.count_parameters` の 'total' と厳密に一致する
    (テストで照合済み)。構成検討を実体化なしで回すために使う。
    """
    c = config
    d = c.d_model
    hidden = c.hidden_width
    tw = c.token_width
    ctx = c.ctx_width

    def linear(i: int, o: int) -> int:
        return i * o + o

    def norm(width: int) -> int:
        return 2 * width if c.norm == 'layer' else 0

    # 埋め込み
    total = c.vocab_size * c.d_emb
    if not c.shared_vocab:
        total += c.vocab_size * c.d_emb

    # 位置テーブル(encoder / decoder それぞれ)
    if c.positional == 'learned':
        total += 2 * c.pos_table_len * c.n_pos

    # 埋め込み射影(encoder / decoder)
    total += 2 * linear(c.d_emb + c.pos_width, d)
    # context 射影(decoder のみ)
    total += linear(d + ctx, d)

    # 1ブロックあたり
    per_token = 0
    for k in c.token_kernels:
        widths = [k] * c.token_depth + [1]
        per_token += sum(linear(widths[i], widths[i + 1]) for i in range(c.token_depth))
    per_block = norm(d)
    per_block += linear(d, hidden) + (c.channel_depth - 1) * linear(hidden, hidden)
    per_block += linear(tw, d) + (c.channel_depth - 1) * linear(d, d)
    per_block += per_token
    total += (c.n_enc_blocks + c.n_dec_blocks) * per_block

    # 最終正規化(encoder / decoder)
    total += 2 * norm(d)

    # cross attention(単一ヘッド)
    if c.cross_attention == 'single_head':
        cd = c.cross_dim if c.cross_dim is not None else d
        total += norm(d) + 3 * linear(d, cd) + linear(cd, d)

    # 出力ヘッド
    total += linear(d + ctx, c.vocab_size)
    return total


class LearnedPositionVectors(nn.Module):
    """位置ごとの学習可能ベクトル(絶対位置符号化)。

    論理版 :class:`LearnedPositionBits` の float 対応物。sigmoid/Heaviside を
    通さず素の実数ベクトルを埋め込みへ連結する。
    """

    def __init__(self, seq_len: int, n_pos: int):
        super().__init__()
        if seq_len < 1:
            raise ValueError(f'seq_lenは1以上: {seq_len}')
        if n_pos < 1:
            raise ValueError(f'n_posは1以上: {n_pos}')
        self.seq_len = seq_len
        self.n_pos = n_pos
        self.table = nn.Parameter(torch.randn(seq_len, n_pos) * 0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.seq_len:
            raise ValueError(
                f'系列長({seq_len})が位置符号化の最大長({self.seq_len})を超えている')
        return self.table[:seq_len]

    def extra_repr(self) -> str:
        return f'seq_len={self.seq_len}, n_pos={self.n_pos}'


def _concat_position(
    emb: torch.Tensor, pos: LearnedPositionVectors | None
) -> torch.Tensor:
    """埋め込み (B, T, d_emb) の末尾に位置ベクトルを連結する(posがNoneなら素通し)。"""
    if pos is None:
        return emb
    batch, seq_len, _ = emb.shape
    vecs = pos(seq_len).unsqueeze(0).expand(batch, -1, -1)
    return torch.cat([emb, vecs], dim=-1)


def _mlp(widths: list[int], dropout: float, last_activation: bool) -> nn.Sequential:
    """widths=[in, h..., out] の全結合MLPを作る(活性はGELU)。"""
    layers: list[nn.Module] = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        is_last = i == len(widths) - 2
        if not is_last or last_activation:
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ToeplitzTokenMixFP(nn.Module):
    """Toeplitz型の静的トークン混合(float版)。

    論理版 :class:`ToeplitzTokenMix` と**同じ窓の張り方・同じ重み共有**を持つ。
    チャネルを ``len(kernels)`` グループへ等分し、グループごとに「窓幅K → 1」を
    写像する MLP を1つだけ持ち、全時刻・グループ内全チャネルで共有する。
    因果性は上三角マスクではなく窓の構造(``[t-K+1, t]``)で保証される。

    forward: (B, T, C) -> (B, T, C)
    """

    def __init__(
        self,
        seq_len: int,
        d_channels: int,
        kernels: tuple[int, ...],
        depth: int,
        causal: bool,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_channels % len(kernels) != 0:
            raise ValueError(
                f'd_channels({d_channels})はkernels本数({len(kernels)})で'
                f'割り切れる必要がある'
            )
        if any(k < 1 for k in kernels):
            raise ValueError(f'kernelsの各要素は1以上: {kernels}')
        if depth < 1:
            raise ValueError(f'depthは1以上: {depth}')

        self.seq_len = seq_len
        self.d_channels = d_channels
        self.kernels = tuple(kernels)
        self.causal = causal
        self.group_width = d_channels // len(kernels)

        self.mixers = nn.ModuleList([
            _mlp([k] * depth + [1], dropout, last_activation=False)
            for k in self.kernels
        ])

    def _window(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """(B, T, Cg) から (B, T, Cg, K) の時間窓を作る(pad値0.0)。"""
        if self.causal:
            left, right = kernel_size - 1, 0
        else:
            left, right = (kernel_size - 1) // 2, kernel_size // 2
        padded = F.pad(x, (0, 0, left, right), mode='constant', value=0.0)
        return padded.unfold(1, kernel_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'xは(B, T, C)の3次元テンソル: {tuple(x.shape)}')
        if x.shape[-1] != self.d_channels:
            raise ValueError(
                f'チャネル幅が不一致: {x.shape[-1]} != {self.d_channels}')

        batch, seq_len, _ = x.shape
        outputs = []
        for g, (kernel_size, mixer) in enumerate(
                zip(self.kernels, self.mixers, strict=True)):
            start = g * self.group_width
            xg = x[:, :, start:start + self.group_width]
            win = self._window(xg, kernel_size)
            flat = win.reshape(-1, kernel_size)
            mixed = mixer(flat)
            outputs.append(mixed.reshape(batch, seq_len, self.group_width))
        return torch.cat(outputs, dim=-1)

    def extra_repr(self) -> str:
        return (
            f'seq_len={self.seq_len}, d_channels={self.d_channels}, '
            f'kernels={self.kernels}, causal={self.causal}'
        )


class FPSeqMixerBlock(nn.Module):
    """SGU様式の1ブロック(float版): pre-LN → channel_in → split → token_mix →
    乗算ゲート → channel_out → 加算残差。

    forward: (B, T, D) -> (B, T, D)
    """

    def __init__(self, config: FPSeqMixerConfig, causal: bool):
        super().__init__()
        self.config = config
        self.causal = causal
        d = config.d_model
        hidden = config.hidden_width
        tw = config.token_width

        self.norm = nn.LayerNorm(d) if config.norm == 'layer' else nn.Identity()
        self.channel_mix_in = _mlp(
            [d] + [hidden] * config.channel_depth, config.dropout,
            last_activation=True,
        )
        self.token_mix = ToeplitzTokenMixFP(
            config.seq_len, tw, config.token_kernels, config.token_depth,
            causal, dropout=config.dropout,
        )
        # 出口の最終層は活性なし(加算残差へ素直に流す)
        self.channel_mix_out = _mlp(
            [tw] + [d] * config.channel_depth, config.dropout,
            last_activation=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        u = self.channel_mix_in(z)
        u1, u2 = u.chunk(2, dim=-1)
        v = self.token_mix(u2)
        y = u1 * v if self.config.gate == 'mul' else v
        h = self.channel_mix_out(y)
        if self.config.residual_mode == 'none':
            return h
        return x + h


class FPSeqMixerEncoder(nn.Module):
    """双方向(非因果)ブロックのencoder。トークン系列をそのまま返す。"""

    def __init__(self, config: FPSeqMixerConfig):
        super().__init__()
        self.config = config
        self.pos = (
            LearnedPositionVectors(config.pos_table_len, config.n_pos)
            if config.positional == 'learned' else None
        )
        self.embed_proj = nn.Linear(config.d_emb + config.pos_width, config.d_model)
        self.blocks = nn.ModuleList([
            FPSeqMixerBlock(config, causal=False)
            for _ in range(config.n_enc_blocks)
        ])
        self.out_norm = (
            nn.LayerNorm(config.d_model) if config.norm == 'layer' else nn.Identity()
        )
        self.out_dim = config.d_model

    def forward(self, src_emb: torch.Tensor) -> torch.Tensor:
        """src_emb: (B, S, d_emb) -> (B, S, d_model)"""
        h = self.embed_proj(_concat_position(src_emb, self.pos))
        for block in self.blocks:
            h = block(h)
        return self.out_norm(h)


class SingleHeadCrossAttention(nn.Module):
    """通常の単一ヘッド dot-product cross-attention(論理版アーム(ii)の対応物)。"""

    def __init__(self, d_model: int, d_head: int, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head)
        self.k_proj = nn.Linear(d_model, d_head)
        self.v_proj = nn.Linear(d_model, d_head)
        self.out_proj = nn.Linear(d_head, d_model)
        self.scale = 1.0 / math.sqrt(d_head)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)                       # (B, T, H)
        k = self.k_proj(memory)                      # (B, S, H)
        v = self.v_proj(memory)                      # (B, S, H)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            weights = F.dropout(weights, p=self.dropout)
        return self.out_proj(torch.matmul(weights, v))


class FPSeqMixerDecoder(nn.Module):
    """因果ブロックのdecoder + 出力ヘッド(Linear -> vocabロジット)。

    位置tの出力は tgt の t 以降のトークンに依存しない(トークン混合が因果窓のみを
    見るため、上三角マスクなしで構造的に保証される)。
    """

    def __init__(self, config: FPSeqMixerConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        ctx = config.ctx_width

        self.pos = (
            LearnedPositionVectors(config.pos_table_len, config.n_pos)
            if config.positional == 'learned' else None
        )
        self.embed_proj = nn.Linear(config.d_emb + config.pos_width, d)
        self.context_proj = nn.Linear(d + ctx, d)
        self.blocks = nn.ModuleList([
            FPSeqMixerBlock(config, causal=True)
            for _ in range(config.n_dec_blocks)
        ])

        self.cross_norm: nn.Module | None = None
        self.cross_attn: SingleHeadCrossAttention | None = None
        if config.cross_attention == 'single_head':
            cross_dim = config.cross_dim if config.cross_dim is not None else d
            self.cross_norm = (
                nn.LayerNorm(d) if config.norm == 'layer' else nn.Identity()
            )
            self.cross_attn = SingleHeadCrossAttention(
                d, cross_dim, dropout=config.dropout)

        self.out_norm = nn.LayerNorm(d) if config.norm == 'layer' else nn.Identity()
        self.head = nn.Linear(d + ctx, config.vocab_size)
        self.d_enc = d

    def _context(self, enc_states: torch.Tensor, tgt_len: int) -> torch.Tensor:
        """文脈テンソル (B, T, ctx_width) を作る(tgtには一切依存しない=因果安全)。

        論理版 :meth:`LogicSeqMixerDecoder._context` と**厳密に同じ意味論**
        (gather が S を超える位置を全0で埋める挙動まで含む)。
        """
        mode = self.config.context_mode
        batch, s_len, _ = enc_states.shape
        if mode == 'broadcast':
            return enc_states[:, -1].unsqueeze(1).expand(-1, tgt_len, -1)
        if mode == 'gather':
            pos = torch.arange(tgt_len, device=enc_states.device)
            ctx = enc_states[:, pos.clamp(max=s_len - 1)]
            if tgt_len > s_len:
                ctx = ctx * (pos < s_len).to(ctx.dtype).view(1, -1, 1)
            return ctx
        target_len = self.config.seq_len
        states = enc_states[:, :target_len]
        if states.shape[1] < target_len:
            pad = states.new_zeros(
                batch, target_len - states.shape[1], self.d_enc)
            states = torch.cat([states, pad], dim=1)
        flat = states.reshape(batch, target_len * self.d_enc)
        return flat.unsqueeze(1).expand(-1, tgt_len, -1)

    def forward(
        self,
        tgt_emb: torch.Tensor,
        enc_states: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """tgt_emb: (B, T, d_emb), enc_states: (B, S, d_model) -> (B, T, vocab_size)"""
        tgt_len = tgt_emb.shape[1]

        h = self.embed_proj(_concat_position(tgt_emb, self.pos))
        ctx = self._context(enc_states, tgt_len)
        h = self.context_proj(torch.cat([h, ctx], dim=-1))

        for block in self.blocks:
            h = block(h)

        if self.cross_attn is not None:
            h = h + self.cross_attn(self.cross_norm(h), enc_states)

        h = self.out_norm(h)
        logits = self.head(torch.cat([h, ctx], dim=-1))
        if return_hidden:
            return logits, h
        return logits


class FPSeqMixer(nn.Module):
    """LogicSeqMixer の float 対照モデル。

    forward は **logits のみ**を返す(論理版の第2要素 emb_reg_loss は、
    埋め込みの二値化正則化という論理版固有の項なので float 版には存在しない)。
    """

    def __init__(self, config: FPSeqMixerConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config

        self.src_embedding = nn.Embedding(config.vocab_size, config.d_emb)
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=0.02)
        if config.shared_vocab:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_emb)
            nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=0.02)

        self.encoder = FPSeqMixerEncoder(config)
        self.decoder = FPSeqMixerDecoder(config)
        self.to(device)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            src_tokens: (B, S) ソーストークンID
            tgt_tokens: (B, T) ターゲットトークンID(teacher forcing、シフト済み)
            return_hidden: Trueで中間活性も返す(診断用)

        Returns:
            既定: logits (B, T, vocab_size)。
            return_hidden=True: (logits, [encoder出力, decoder最終表現])。
        """
        src_emb = self.src_embedding(src_tokens)
        tgt_emb = self.tgt_embedding(tgt_tokens)
        enc_states = self.encoder(src_emb)

        if return_hidden:
            logits, dec_hidden = self.decoder(tgt_emb, enc_states, return_hidden=True)
            return logits, [enc_states, dec_hidden]
        return self.decoder(tgt_emb, enc_states)

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す('total'キー必須)。

        論理版 :meth:`LogicSeqMixer.count_parameters` と同じキー体系
        ('embedding' / 'encoder' / 'decoder' / 'head' / 'total')にそろえてある。
        """
        counts: dict[str, int] = {}
        counts['embedding'] = sum(p.numel() for p in self.src_embedding.parameters())
        if not self.config.shared_vocab:
            counts['embedding'] += sum(
                p.numel() for p in self.tgt_embedding.parameters())
        counts['encoder'] = sum(p.numel() for p in self.encoder.parameters())
        head = sum(p.numel() for p in self.decoder.head.parameters())
        counts['decoder'] = sum(
            p.numel() for p in self.decoder.parameters()) - head
        counts['head'] = head
        counts['total'] = sum(counts.values())
        return counts

    def extra_repr(self) -> str:
        c = self.config
        return (
            f'vocab={c.vocab_size}, d_emb={c.d_emb}, seq_len={c.seq_len}, '
            f'd_model={c.d_model}, hidden={c.hidden_width}, '
            f'enc/dec_blocks={c.n_enc_blocks}/{c.n_dec_blocks}, '
            f'channel_depth={c.channel_depth}, kernels={c.token_kernels}, '
            f'gate={c.gate}, residual={c.residual_mode}, norm={c.norm}, '
            f'context={c.context_mode}({c.ctx_width}), '
            f'cross={c.cross_attention}, '
            f'positional={c.positional}({c.pos_width})'
        )


__all__ = [
    'FPSeqMixer',
    'FPSeqMixerBlock',
    'FPSeqMixerConfig',
    'FPSeqMixerDecoder',
    'FPSeqMixerEncoder',
    'LearnedPositionVectors',
    'SingleHeadCrossAttention',
    'ToeplitzTokenMixFP',
    'estimate_parameters',
]
