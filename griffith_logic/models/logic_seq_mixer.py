"""LogicSeqMixer — 翻訳向け Mixer-DLGN(attention不使用のencoder-decoder)。

token混合を「Toeplitz型の静的因果混合」(gMLP-SGU の Toeplitz 制限様式)で、
channel混合を LutStack(論理ゲートMLP)で行う seq2seq モデル。RDDLGN
(arXiv:2508.06097)と同じ土俵(WMT'14 En-De、系列長16、語彙16k)で比較できるよう、
出力ヘッド(M層相当)とゲート数オーダーを揃えてある。

設計判断:
    - **TSMixerをそのまま使わない**。TSMixerは「固定長の多変量時系列 → 固定長の
      実数予測」を前提とした構造で、語彙上のトークン生成・自己回帰デコード・
      因果マスクのいずれも持たない。翻訳に転用するには実質的に別アーキテクチャへ
      作り直す必要があるため、本実装は事前サーベイの推奨第1位である
      「gMLP(SGU)+ Toeplitz/circulant 制限 + 因果窓」を採用する。
    - **なぜToeplitz窓か**。混合重みが入力非依存(静的)なので学習後に配線を固定でき、
      LUT化・二値化と直結する。さらに密な T×T 混合では各出力が T トークン分の
      ビットに依存して fan-in が爆発するのに対し、カーネル幅 K の Toeplitz 窓は
      fan-in が K に固定され、2入力LUT木の深さが log(K) で収まる(survey §3.5)。
      重み共有(全位置・グループ内全チャネルで1つのLutStack)がシフト不変性の実体で、
      LightConv(arXiv:1901.10430、WMT14 En-De 28.9 BLEU)と数学的に同族である。
    - **要素ごとの乗算ゲート `u1 ⊙ v` は二値領域でANDゲート1個に落ちる**。softmaxの
      ような大域非線形(除算を含む)を避けられるのが、SGU様式を選ぶ最大の理由。
    - **n_enc_blocks + n_dec_blocks <= 4 に制限する**。MIXER-2 の実測で、soft-XOR残差を重ねると
      初期勾配が1ブロックあたり約1e-2倍で指数的に減衰し、n_blocks=6 で 3.72e-14、
      n_blocks=8 で 2.60e-18 となって学習が開始しない。深さの壁を破る対処
      (residual_mode='none' × residual_init=True)は見つかったが、同ゲート数では
      浅い構成に負ける(同§7.4b)ため、容量は深さでなく幅で増やす方針を採る。
    - **residual_mode='xor' と residual_init=True の併用を禁止する**。RIは混合出力を
      h≈x へ寄せるが、``soft_xor(x, x) = 2x(1-x)`` は x∈{0,1} で恒等的に0であり、
      勾配が厳密に0になる(同§7.3の実測: 0.00e+00)。設定段階でValueErrorにする。
    - **cross混合だけ動的を残す選択肢(cross_attention='logic_single_head')を用意する**。
      hard-coded Gaussian attention(arXiv:2005.00742)の実測では、自己注意を完全固定化
      しても BLEU はほぼ落ちない一方、cross まで固定化すると -5〜-9 BLEU 落ち、
      学習ヘッドを1個足すだけで大きく回復する(hc-all 21.1 → sh-x 28.2、IWSLT16 En-De)。
      gMLP が複数文アライメントで aMLP(単一ヘッドtiny attention)を必要とした事実も
      独立の証拠である(survey §3.3)。既定のアーム(i)は RDDLGN と同じ
      「context をブロードキャスト連結するだけ」の完全静的構成にしておき、
      アーム(ii)との差分が cross の動的性の価値を測る実験になる。なお cross の
      attention重みも eval では argmax の one-hot へ離散化する(`hard=not self.training`)。
      soft のままだと重みが連続の実数で残り、collapsed(完全ブール回路)を主張できない
      ためで、モデル全体の「soft/hardは self.training に一元化」規約とも一致する。
    - **soft/hard は `self.training` に一元化**する。LutLayer と LogicEmbedding は
      eval() で自動的に離散化される既存規約に乗り、`hard=` 引数をAPIへ足さない。
    - **絶対位置符号化(`positional='learned_bits'`)は必須である**。Toeplitz混合は
      shift-invariant(全位置で同一の関数を適用する)なので、それ自体では絶対位置の
      情報を持たない。全デコード時刻へ同じ context をブロードキャストする構成では、
      デコーダが「今が何番目か」を区別できず ``src[t]`` を選び出せない。RDDLGNが
      同じブロードキャスト構造で機能するのは**再帰的なP層が暗黙の位置カウンタを
      持つ**ためで、静的混合の本モデルには等価物が無い。copyタスク実測
      (vocab=32, seq_len=8, 2000step)でこれが確認された:

          d_channels  48 / 252 / 1020 / 2040 → arm(i) hard acc
                    0.2279 / 0.2487 / 0.2392 / 0.1831

      **幅を42倍にしても0.18〜0.25の床から動かない**(XLではむしろ悪化)ため、
      容量ではなく構造の欠落である。よって位置ごとの学習可能ビットパターンを
      encoder/decoder双方の埋め込み直後へ連結する。アブレーション用に
      `positional='none'`(従来挙動)を残してある。位置ビットは定数なので
      ゲート数には現れない(LUTニューロンを持たない)。
    - **文脈の渡し方に3段階(`context_mode` × `cross_attention`)を用意する**。
      既定の `'broadcast'`(RDDLGNと同じ、encoder最終位置だけを全時刻へ)は、調査md
      §3.3 の hard-coded Gaussian attention でいう **hc-all 相当のハンディクラス**で、
      公平比較の基準線として絶対に変えない。ただしcopy掃引で、この構成は
      位置符号化を入れてもゲート数を揃えた比較で床から動かないことが実測された:

          ≈4,995ゲート  arm(i)+位置符号化 0.2476 / arm(ii) cross 0.7292
          ≈11,975ゲート arm(i)+位置符号化 0.2448 / arm(ii) cross 0.5594

      診断は「copyに要るのは位置tに応じてcontextのt番目のフィールドを選ぶ
      **位置条件付きマルチプレクサ**で、`connections='random'` のLUT2スタックには
      その配線が確率的にほぼ存在しない」。そこで **`'gather'`(位置対応の静的
      アライメント。tはenc_states[min(t,S-1)]を見る)** を追加した。入力非依存で
      完全に静的=LUT化可能であり、§3.3 の「fixed encoder patterns」に対応する。
      broadcast(hc-all) / gather(fixed pattern) / logic_single_head(sh-x=動的1ヘッド)
      の3点比較で、**翻訳に必要なのが「動的性」なのか「単なる位置対応」なのか**を
      単離できる。`'concat_all'`(全系列を平坦化連結)は上限の目安用。
    - **ゲート予算をRDDLGN論文(Table 1、計1,526,000ゲート)へ揃える**: 本体248,952
      ≒論文の N+K+L+P=246,000、出力ヘッド1,280,000=論文のM、合計1,528,952
      (実測値。:meth:`LogicSeqMixer.count_gates`)。同一ゲート数帯での直接比較を
      成立させるための設計制約である。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.attention import LogicAttention
from ..group_sum import GroupSum
from .rddlgn import LogicEmbedding, LutStack

# 設定値域(ValueErrorメッセージと検証の単一情報源)
_GATES = ('and', 'none')
_RESIDUAL_MODES = ('xor', 'none')
_CONTEXT_MODES = ('broadcast', 'gather', 'concat_all')
_CROSS_ATTENTION = ('none', 'logic_single_head')
_POSITIONAL = ('learned_bits', 'none')

# ブロック段数の上限(MIXER-2 §7.2の勾配実測に基づく)
_MAX_BLOCKS = 4


def _soft_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """微分可能なXOR残差結合。a,b∈{0,1} のとき厳密にbitwise XORと一致する。"""
    return a + b - 2.0 * a * b


def _soft_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """微分可能なANDゲート。a,b∈{0,1} のとき厳密にbitwise ANDと一致する。"""
    return a * b


@dataclass
class LogicSeqMixerConfig:
    """LogicSeqMixerの全ハイパーパラメータ。

    既定値は「型の説明」であり、実験に使う構成は :meth:`from_paper`(RDDLGN論文と
    同オーダー)か :meth:`tiny`(スモーク)から作ること。
    """

    vocab_size: int = 16000
    d_emb: int = 1024
    seq_len: int = 16
    # ブロック内のチャネル幅(ビット数)。既定はfrom_paper()と同値(token_kernels
    # 3本で等分でき、本体ゲート数27×d_channelsが論文の246kに揃う)
    d_channels: int = 9216
    channel_hidden: int | None = None  # 既定 None -> d_channels*2
    n_enc_blocks: int = 2
    n_dec_blocks: int = 2
    channel_depth: int = 2             # チャネル混合LutStackの段数
    token_kernels: tuple[int, ...] = (3, 8, 16)   # ヘッドごとのカーネル幅
    token_depth: int = 2               # トークン混合LutStackの段数
    gate: str = 'and'                  # 'and' | 'none'
    residual_mode: str = 'xor'         # 'xor' | 'none'
    residual_init: bool = False
    m_layer_sizes: tuple[int, ...] = (400000, 400000, 480000)  # 出力ヘッド(RDDLGN M相当)
    positional: str = 'learned_bits'   # 'learned_bits' | 'none'(従来挙動=アブレーション)
    n_pos_bits: int = 64               # 位置ビット幅(positional='learned_bits'時)
    # encoder表現をdecoderへ渡す方式:
    #   'broadcast'  — 最終位置を全時刻へ(RDDLGN同等。既定・比較の基準線)
    #   'gather'     — 位置対応の静的アライメント(t番目はenc_states[min(t,S-1)])
    #   'concat_all' — enc_states全系列を平坦化して全時刻へ連結
    context_mode: str = 'broadcast'
    cross_attention: str = 'none'      # 'none'(アームi) | 'logic_single_head'(アームii)
    cross_qk_bits: int = 256
    cross_v_bits: int = 1024
    n_inputs: int = 2
    parameterization: str = 'softmax'
    connections: str = 'random'
    selection: str = 'soft'
    tau: float = 2.0
    gumbel: bool = True
    grad_factor: float = 1.0
    shared_vocab: bool = True

    def __post_init__(self) -> None:
        n_blocks = self.n_enc_blocks + self.n_dec_blocks
        if self.n_enc_blocks < 1 or self.n_dec_blocks < 1:
            raise ValueError(
                f'n_enc_blocks/n_dec_blocksは1以上: '
                f'{self.n_enc_blocks}/{self.n_dec_blocks}'
            )
        if n_blocks > _MAX_BLOCKS:
            raise ValueError(
                f'n_enc_blocks+n_dec_blocks は{_MAX_BLOCKS}以下: '
                f'{self.n_enc_blocks}+{self.n_dec_blocks}={n_blocks}。'
                'MIXER-2でnb>=6はsoft-XOR勾配死(初期勾配3.7e-14)により'
                '学習不能と実証済み'
            )
        if self.gate not in _GATES:
            raise ValueError(f'gateは{_GATES}のいずれか: {self.gate}')
        if self.residual_mode not in _RESIDUAL_MODES:
            raise ValueError(
                f'residual_modeは{_RESIDUAL_MODES}のいずれか: {self.residual_mode}')
        if self.context_mode not in _CONTEXT_MODES:
            raise ValueError(
                f'context_modeは{_CONTEXT_MODES}のいずれか: {self.context_mode}')
        if self.cross_attention not in _CROSS_ATTENTION:
            raise ValueError(
                f'cross_attentionは{_CROSS_ATTENTION}のいずれか: {self.cross_attention}')
        if self.positional not in _POSITIONAL:
            raise ValueError(
                f'positionalは{_POSITIONAL}のいずれか: {self.positional}')
        if self.positional == 'learned_bits' and self.n_pos_bits < 1:
            raise ValueError(f'n_pos_bitsは1以上: {self.n_pos_bits}')
        if self.residual_mode == 'xor' and self.residual_init:
            raise ValueError(
                "residual_mode='xor' と residual_init=True は併用禁止: "
                'soft_xor(x, x)=2x(1-x) は x∈{0,1} で恒等的に0となり勾配が'
                '厳密に0になる(MIXER-2 §7.3で実測 0.00e+00)。'
                "residual_mode='none' と組み合わせること"
            )
        if len(self.m_layer_sizes) < 1:
            raise ValueError('m_layer_sizesは1要素以上必要')
        if any(s < 1 for s in self.m_layer_sizes):
            raise ValueError(f'm_layer_sizesの各要素は1以上: {self.m_layer_sizes}')
        if self.m_layer_sizes[-1] % self.vocab_size != 0:
            raise ValueError(
                f'm_layer_sizesの最終層({self.m_layer_sizes[-1]})は'
                f'vocab_size({self.vocab_size})で割り切れる必要がある'
                f'(GroupSumのgroupsize)。余り='
                f'{self.m_layer_sizes[-1] % self.vocab_size}'
            )
        if len(self.token_kernels) < 1:
            raise ValueError('token_kernelsは1要素以上必要')
        if any(k < 1 for k in self.token_kernels):
            raise ValueError(f'token_kernelsの各要素は1以上: {self.token_kernels}')
        if self.token_depth < 1:
            raise ValueError(f'token_depthは1以上: {self.token_depth}')
        if self.channel_depth < 1:
            raise ValueError(f'channel_depthは1以上: {self.channel_depth}')
        n_groups = len(self.token_kernels)
        if self.d_channels % n_groups != 0:
            raise ValueError(
                f'd_channels({self.d_channels})はtoken_kernelsの本数'
                f'({n_groups})で割り切れる必要がある(チャネルをヘッドへ等分するため)。'
                f'余り={self.d_channels % n_groups}'
            )
        hidden = self.hidden_width
        if hidden % 2 != 0:
            raise ValueError(
                f'channel_hidden({hidden})はchunk(2)のため偶数である必要がある')
        if (hidden // 2) % n_groups != 0:
            raise ValueError(
                f'channel_hidden//2({hidden // 2})はtoken_kernelsの本数'
                f'({n_groups})で割り切れる必要がある。余り={(hidden // 2) % n_groups}'
            )

    @property
    def hidden_width(self) -> int:
        """チャネル混合の中間幅(channel_hidden、既定は d_channels*2)。"""
        return self.channel_hidden if self.channel_hidden is not None \
            else self.d_channels * 2

    @property
    def pos_width(self) -> int:
        """埋め込みへ連結する位置ビット幅(positional='none' なら0)。"""
        return self.n_pos_bits if self.positional == 'learned_bits' else 0

    @property
    def ctx_width(self) -> int:
        """decoderへ渡す文脈ベクトルのビット幅(context_modeで決まる)。

        'broadcast'/'gather' は encoder 1位置分(d_channels)、'concat_all' は
        全系列分(seq_len×d_channels)。ゲート数はLUTの出力数で決まるため、
        この幅が変わっても :func:`estimate_gates` の値は変化しない
        (変わるのは配線のfan-inとindicesのパラメータ数だけ)。
        """
        if self.context_mode == 'concat_all':
            return self.seq_len * self.d_channels
        return self.d_channels

    @property
    def pos_table_len(self) -> int:
        """位置符号化テーブルの行数(= 受け付ける最大系列長)。

        teacher forcingのデコーダ入力は ``[BOS, tgt(seq_len個), EOS][:-1]`` で
        長さ seq_len+1 になる(RDDLGNと同一の評価プロトコル)。オフバイワンで
        溢れないよう **seq_len+2** を確保する。greedy decodeでprefixが
        1,2,...と伸びる場合も先頭からの ``[:T]`` スライスで賄える
        (lucidrains/g-mlp-pytorch の ``weight[:, :n, :n]`` と同じ流儀)。
        """
        return self.seq_len + 2

    @property
    def token_width(self) -> int:
        """トークン混合が担当する幅(SGUのsplit後 = channel_hidden//2)。"""
        return self.hidden_width // 2

    @classmethod
    def from_paper(cls) -> LogicSeqMixerConfig:
        """RDDLGN論文構成(1.526Mゲート)と同オーダーへ揃えた既定。

        RDDLGNの N+K+L+P = 24k+86k+24k+112k ≈ 246k(本体)/ M = 1,280,000(出力ヘッド)に
        対し、本モデルも本体≈249k・ヘッド1,280,000で合計約1.53Mゲートになる
        (実測値は :meth:`LogicSeqMixer.count_gates` / :func:`estimate_gates` で確認)。
        d_channels=9216 は token_kernels 3本で等分でき、かつ本体ゲート数
        27×d_channels が 246k 近傍に落ちる値として選んだ。dataclassの既定値も
        同じ構成に揃えてあるため、``LogicSeqMixerConfig() == from_paper()`` である。
        """
        return cls(
            vocab_size=16000,
            d_emb=1024,
            seq_len=16,
            d_channels=9216,
            channel_hidden=None,   # -> 18432
            n_enc_blocks=2,
            n_dec_blocks=2,
            channel_depth=2,
            token_kernels=(3, 8, 16),
            token_depth=2,
            m_layer_sizes=(400000, 400000, 480000),
        )

    @classmethod
    def tiny(cls) -> LogicSeqMixerConfig:
        """スモーク・テスト用の最小構成(GroupSum整除性を満たす)。

        位置符号化(既定 'learned_bits')を入れると埋め込み射影の入力が
        d_emb + n_pos_bits へ増えるため、**射影のタップ数 2×d_channels が
        入力ビット数を下回らない**ようにサイズを決めてある
        (2×72=144 >= 64+16=80)。タップが足りないと入力ビットの一部が構造的に
        捨てられ、コピータスクのhard精度が落ちる(L4実測: d_channels=48 /
        n_pos_bits=64 で 0.9896、n_pos_bits=16 で 0.9479 → d_channels=72 /
        n_pos_bits=16 で 1.0000)。
        """
        vocab_size = 50
        return cls(
            vocab_size=vocab_size,
            d_emb=64,
            seq_len=8,
            d_channels=72,
            channel_hidden=144,
            n_pos_bits=16,
            n_enc_blocks=1,
            n_dec_blocks=1,
            channel_depth=2,
            token_kernels=(3, 4, 8),
            token_depth=2,
            m_layer_sizes=(128, vocab_size * 3),
            tau=1.0,
        )


def estimate_gates(config: LogicSeqMixerConfig) -> dict[str, int]:
    """モデルを実体化せずにゲート数(LUTニューロン数)を解析式で見積もる。

    from_paper()構成は数十万ゲート規模でCPU実体化が重いため、構成検討時は
    こちらを使う。:meth:`LogicSeqMixer.count_gates` と厳密に一致する
    (tiny構成で照合済み)。

    Returns:
        'embed_proj' / 'enc_blocks' / 'dec_blocks' / 'token_mix' / 'cross' /
        'head' / 'total' の内訳。
    """
    c = config
    hidden = c.hidden_width
    token_width = c.token_width

    # 埋め込み射影: src射影 + tgt射影 + context連結の射影。
    # context_projの出力幅はd_channels固定なので、context_modeを変えても
    # ゲート数は変わらない(変わるのは入力fan-in=ctx_widthのみ)。
    embed_proj = 3 * c.d_channels

    # 1ブロックのチャネル混合(in: d_channels->hidden, out: token_width->d_channels)
    per_block_channel = c.channel_depth * (hidden + c.d_channels)
    enc_blocks = c.n_enc_blocks * per_block_channel
    dec_blocks = c.n_dec_blocks * per_block_channel

    # トークン混合: グループごとに K -> ... -> 1 のLutStack(全位置・全チャネル共有)
    per_block_token = sum(
        (c.token_depth - 1) * k + 1 for k in c.token_kernels
    )
    token_mix = (c.n_enc_blocks + c.n_dec_blocks) * per_block_token

    # cross attention(アームii): q/k/v射影 + 出力射影
    cross = 0
    if c.cross_attention == 'logic_single_head':
        cross = 2 * c.cross_qk_bits + c.cross_v_bits + c.d_channels

    head = sum(c.m_layer_sizes)

    counts = {
        'embed_proj': embed_proj,
        'enc_blocks': enc_blocks,
        'dec_blocks': dec_blocks,
        'token_mix': token_mix,
        'cross': cross,
        'head': head,
    }
    counts['total'] = sum(counts.values())
    # token_width は上の式に現れないが、ブロック構成の前提として明示しておく
    assert token_width * 2 == hidden
    return counts


class LearnedPositionBits(nn.Module):
    """位置ごとの学習可能な[0,1]ビットパターン(絶対位置符号化)。

    訓練時は sigmoid(連続[0,1])、推論時は Heaviside(閾値0.5、離散{0,1})という
    :class:`LogicEmbedding` と完全に同じ流儀にそろえる。乗算器を持たない定数
    ビットパターンなので、ハードウェア展開時のゲート数への寄与はゼロである
    (LUTニューロンを持たないため ``count_gates`` にも現れない)。
    """

    def __init__(self, seq_len: int, n_bits: int, device: str = 'cpu'):
        super().__init__()
        if seq_len < 1:
            raise ValueError(f'seq_lenは1以上: {seq_len}')
        if n_bits < 1:
            raise ValueError(f'n_bitsは1以上: {n_bits}')
        self.seq_len = seq_len
        self.n_bits = n_bits
        self.logits = nn.Parameter(torch.randn(seq_len, n_bits, device=device))

    def forward(self, seq_len: int) -> torch.Tensor:
        """先頭 seq_len 位置分のビットパターン (seq_len, n_bits) を返す。"""
        if seq_len > self.seq_len:
            raise ValueError(
                f'系列長({seq_len})が位置符号化の最大長({self.seq_len})を超えている')
        bits = torch.sigmoid(self.logits[:seq_len])
        if self.training:
            return bits
        return (bits > 0.5).to(bits.dtype)

    def extra_repr(self) -> str:
        return f'seq_len={self.seq_len}, n_bits={self.n_bits}'


def _concat_position(
    emb: torch.Tensor, pos: LearnedPositionBits | None
) -> torch.Tensor:
    """埋め込み (B, T, d_emb) の末尾に位置ビットを連結する(posがNoneなら素通し)。"""
    if pos is None:
        return emb
    batch, seq_len, _ = emb.shape
    bits = pos(seq_len).unsqueeze(0).expand(batch, -1, -1)  # (B, T, n_bits)
    return torch.cat([emb, bits], dim=-1)


class ToeplitzTokenMix(nn.Module):
    """Toeplitz型の静的トークン混合(gMLP-SGU の Toeplitz 制限 / LightConv 相当)。

    チャネルを ``len(kernels)`` グループへ等分し、グループごとに
    「窓幅K → ... → 1」を写像する LutStack を**1つだけ**持つ。この LutStack は
    全時刻・グループ内全チャネルで共有され、これがシフト不変性(Toeplitz構造)の
    実体である。因果性は上三角マスクではなく**窓の張り方**で保証する
    (causal=True なら位置tの窓は [t-K+1, t] のみ)。

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
        n_inputs: int = 2,
        device: str = 'cpu',
        grad_factor: float = 1.0,
        parameterization: str = 'softmax',
        connections: str = 'random',
        selection: str = 'soft',
        gumbel: bool = True,
        residual_init: bool = False,
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
            LutStack(
                k,
                [k] * (depth - 1) + [1],
                n_inputs=n_inputs,
                device=device,
                grad_factor=grad_factor,
                parameterization=parameterization,
                connections=connections,
                gumbel=gumbel,
                selection=selection,
            )
            for k in self.kernels
        ])
        # residual_initはLutStack経由で渡せないため、必要なら層へ直接適用する
        # (幅1の出力層に恒等初期化を入れても意味がないので既定では何もしない)
        self.residual_init = residual_init

    def _window(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """(B, T, Cg) から (B, T, Cg, K) の時間窓を作る。

        ``core.reference.reference_causal_window`` と数値的に厳密一致する
        高速版(pad + unfold)。pad値は0.0。
        """
        if self.causal:
            left, right = kernel_size - 1, 0
        else:
            left, right = (kernel_size - 1) // 2, kernel_size // 2
        # 時間軸(dim=1)を左右にゼロパディング
        padded = F.pad(x, (0, 0, left, right), mode='constant', value=0.0)
        # (B, T, Cg, K): 最後の要素が causal のとき現在時刻
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
            xg = x[:, :, start:start + self.group_width]      # (B, T, Cg)
            win = self._window(xg, kernel_size)               # (B, T, Cg, K)
            flat = win.reshape(-1, kernel_size)               # (B*T*Cg, K)
            mixed = mixer(flat)                               # (B*T*Cg, 1)
            outputs.append(mixed.reshape(batch, seq_len, self.group_width))
        return torch.cat(outputs, dim=-1)

    def extra_repr(self) -> str:
        return (
            f'seq_len={self.seq_len}, d_channels={self.d_channels}, '
            f'kernels={self.kernels}, causal={self.causal}'
        )


class LogicSeqMixerBlock(nn.Module):
    """SGU様式の1ブロック: channel_mix_in → split → token_mix → ANDゲート → 残差。

    forward: (B, T, C) -> (B, T, C)
    """

    def __init__(self, config: LogicSeqMixerConfig, causal: bool, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.causal = causal
        hidden = config.hidden_width
        token_width = config.token_width

        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            gumbel=config.gumbel,
            selection=config.selection,
        )

        # チャネル混合(入口): d_channels -> hidden(channel_depth段)
        self.channel_mix_in = LutStack(
            config.d_channels,
            [hidden] * config.channel_depth,
            **lut_kwargs,
        )
        # トークン混合: split後の片側(token_width)にだけ掛ける
        self.token_mix = ToeplitzTokenMix(
            config.seq_len,
            token_width,
            config.token_kernels,
            config.token_depth,
            causal,
            residual_init=config.residual_init,
            **lut_kwargs,
        )
        # チャネル混合(出口): token_width -> d_channels(channel_depth段)
        self.channel_mix_out = LutStack(
            token_width,
            [config.d_channels] * config.channel_depth,
            **lut_kwargs,
        )

    def _combine(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """残差の合流。'none' なら混合出力をそのまま返す(素の直列)。"""
        if self.config.residual_mode == 'none':
            return h
        return _soft_xor(x, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # 位置をバッチ側へ畳んでチャネル混合(位置間で重み共有)
        u = self.channel_mix_in(x.reshape(batch * seq_len, -1))
        u = u.reshape(batch, seq_len, -1)
        u1, u2 = u.chunk(2, dim=-1)

        v = self.token_mix(u2)
        y = _soft_and(u1, v) if self.config.gate == 'and' else v

        h = self.channel_mix_out(y.reshape(batch * seq_len, -1))
        h = h.reshape(batch, seq_len, -1)
        return self._combine(x, h)


class LogicSeqMixerEncoder(nn.Module):
    """双方向(非因果)ブロックのencoder。**トークン系列をそのまま返す**(poolingしない)。"""

    def __init__(self, config: LogicSeqMixerConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config
        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            gumbel=config.gumbel,
            selection=config.selection,
        )
        self.pos = (
            LearnedPositionBits(
                config.pos_table_len, config.n_pos_bits, device=device)
            if config.positional == 'learned_bits' else None
        )
        self.embed_proj = LutStack(
            config.d_emb + config.pos_width, [config.d_channels], **lut_kwargs)
        self.blocks = nn.ModuleList([
            LogicSeqMixerBlock(config, causal=False, device=device)
            for _ in range(config.n_enc_blocks)
        ])
        self.out_dim = config.d_channels

    def forward(self, src_emb: torch.Tensor) -> torch.Tensor:
        """src_emb: (B, S, d_emb) -> (B, S, d_enc)"""
        batch, seq_len, _ = src_emb.shape
        src_emb = _concat_position(src_emb, self.pos)
        h = self.embed_proj(src_emb.reshape(batch * seq_len, -1))
        h = h.reshape(batch, seq_len, -1)
        for block in self.blocks:
            h = block(h)
        return h


class LogicSeqMixerDecoder(nn.Module):
    """因果ブロックのdecoder + 出力ヘッド(LutStack → GroupSum)。

    位置tの出力は tgt の t 以降のトークンに依存しない(トークン混合が因果窓のみを
    見るため、上三角マスクなしで構造的に保証される)。
    """

    def __init__(self, config: LogicSeqMixerConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config
        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            device=device,
            grad_factor=config.grad_factor,
            parameterization=config.parameterization,
            connections=config.connections,
            gumbel=config.gumbel,
            selection=config.selection,
        )
        d_enc = config.d_channels

        self.pos = (
            LearnedPositionBits(
                config.pos_table_len, config.n_pos_bits, device=device)
            if config.positional == 'learned_bits' else None
        )
        self.embed_proj = LutStack(
            config.d_emb + config.pos_width, [config.d_channels], **lut_kwargs)
        # context連結: [decoder表現; 文脈(context_modeで決まる幅)] -> d_channels
        ctx_width = config.ctx_width
        self.context_proj = LutStack(
            config.d_channels + ctx_width, [config.d_channels], **lut_kwargs)

        self.blocks = nn.ModuleList([
            LogicSeqMixerBlock(config, causal=True, device=device)
            for _ in range(config.n_dec_blocks)
        ])

        # アーム(ii)のときだけ生成する(アーム(i)ではゲート数に混入させない)
        self.cross_attn: LogicAttention | None = None
        self.cross_out: LutStack | None = None
        if config.cross_attention == 'logic_single_head':
            self.cross_attn = LogicAttention(
                d_model=config.d_channels,
                n_heads=1,
                mode='soft',
                qk_sizes=[config.cross_qk_bits],
                v_sizes=[config.cross_v_bits],
                # queryはdecoder表現・key/valueはencoder系列と出自が異なるため共有しない
                share_qk=False,
                tau=config.tau,
                causal=False,
                n_inputs=config.n_inputs,
                parameterization=config.parameterization,
                connections=config.connections,
                grad_factor=config.grad_factor,
                selection=config.selection,
            )
            self.cross_out = LutStack(
                config.cross_v_bits, [config.d_channels], **lut_kwargs)

        # 出力ヘッド: [decoder表現; context] -> m_layer_sizes -> GroupSum
        self.head = LutStack(
            config.d_channels + ctx_width, list(config.m_layer_sizes), **lut_kwargs)
        self.group_sum = GroupSum(k=config.vocab_size, tau=config.tau)
        self.d_enc = d_enc

    def _context(self, enc_states: torch.Tensor, tgt_len: int) -> torch.Tensor:
        """文脈テンソル (B, T, ctx_width) を作る(tgtには一切依存しない=因果安全)。

        'broadcast' — enc_states[:, -1] を全時刻へ。RDDLGNと同じハンディクラスで、
            hard-coded Gaussian attention(arXiv:2005.00742)の hc-all 相当。
        'gather'    — 位置 t が enc_states[:, min(t, S-1)] を見る静的アライメント。
            入力非依存(=完全に静的でLUT化可能)で、調査md §3.3 の
            「fixed encoder patterns / 固定ガウス注意」に対応する。broadcast(hc-all)と
            cross_attention='logic_single_head'(sh-x=動的1ヘッド)の**中間項**であり、
            3点比較で「翻訳に必要なのは動的性か、単なる位置対応か」を単離できる。
            デコード長Tが S を超える場合は最終位置に張り付く(min(t, S-1))。
        'concat_all' — enc_states 全系列を平坦化して全時刻へ連結。S<seq_len は0詰め、
            S>seq_len は先頭seq_len位置に切り詰める(射影の入力幅が固定のため)。
        """
        mode = self.config.context_mode
        batch, s_len, _ = enc_states.shape
        if mode == 'broadcast':
            return enc_states[:, -1].unsqueeze(1).expand(-1, tgt_len, -1)
        if mode == 'gather':
            pos = torch.arange(tgt_len, device=enc_states.device)
            ctx = enc_states[:, pos.clamp(max=s_len - 1)]
            if tgt_len > s_len:
                # S を超える位置(teacher forcingのEOS位置など)は**全0ビット**にする。
                # ここを enc_states[S-1] で埋めると、最後の内容位置(t=S-1)と
                # EOS位置(t=S)が同一の文脈を共有してしまい、定数ターゲットである
                # EOS側に引っ張られて**最後の内容位置だけが壊れる**
                # (実測 L=5: pos4のhard acc=0.00、L=4: pos3=0.31)。
                # 全0は「対応するencoder位置が無い」ことを表す明確な符号で、
                # 位置ごとの文脈の一意性(位置tはenc_states[t]だけを見る)を保つ。
                ctx = ctx * (pos < s_len).to(ctx.dtype).view(1, -1, 1)
            return ctx
        # 'concat_all': (B, seq_len*d_enc) へ整形してから全時刻へブロードキャスト
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
        """tgt_emb: (B, T, d_emb), enc_states: (B, S, d_enc) -> (B, T, vocab_size)"""
        batch, tgt_len, _ = tgt_emb.shape

        tgt_emb = _concat_position(tgt_emb, self.pos)
        h = self.embed_proj(tgt_emb.reshape(batch * tgt_len, -1))
        h = h.reshape(batch, tgt_len, -1)

        # context: encoder表現をdecoder各時刻へ渡す(方式はcontext_mode)
        ctx = self._context(enc_states, tgt_len)                 # (B, T, ctx_width)
        h = torch.cat([h, ctx], dim=-1)
        h = self.context_proj(h.reshape(batch * tgt_len, -1))
        h = h.reshape(batch, tgt_len, -1)

        for block in self.blocks:
            h = block(h)

        if self.cross_attn is not None:
            # soft/hardは self.training に一元化する(モデル全体の規約)。eval時は
            # attention重みもargmaxのone-hotへ離散化し、cross経路まで含めて
            # 完全ブール回路(collapsed)を主張できる状態にする。
            attended = self.cross_attn(
                h, enc_states, enc_states, hard=not self.training)  # (B, T, d_v)
            projected = self.cross_out(attended.reshape(batch * tgt_len, -1))
            projected = projected.reshape(batch, tgt_len, -1)
            h = _soft_xor(h, projected)

        hidden = h
        m = self.head(torch.cat([h, ctx], dim=-1).reshape(batch * tgt_len, -1))
        logits = self.group_sum(m).reshape(batch, tgt_len, self.config.vocab_size)
        if return_hidden:
            return logits, hidden
        return logits


class LogicSeqMixer(nn.Module):
    """翻訳向け Mixer-DLGN。attention不使用のencoder-decoder。"""

    def __init__(self, config: LogicSeqMixerConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config

        self.src_embedding = LogicEmbedding(config.vocab_size, config.d_emb)
        if config.shared_vocab:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = LogicEmbedding(config.vocab_size, config.d_emb)

        self.encoder = LogicSeqMixerEncoder(config, device=device)
        self.decoder = LogicSeqMixerDecoder(config, device=device)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        return_hidden: bool = False,
    ) -> (tuple[torch.Tensor, torch.Tensor]
          | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]):
        """
        Args:
            src_tokens: (B, S) ソーストークンID
            tgt_tokens: (B, T) ターゲットトークンID(teacher forcing、シフト済み)
            return_hidden: Trueで中間活性も返す(蒸留・診断用)

        Returns:
            既定: (logits (B,T,vocab_size), emb_reg_loss 0次元Tensor) の2-tuple。
            return_hidden=True: (logits, emb_reg_loss, hidden_states) の3-tuple。
                hidden_states は [encoder出力 (B,S,d_enc), decoder最終表現 (B,T,d_channels)]。
        """
        src_emb = self.src_embedding(src_tokens)
        tgt_emb = self.tgt_embedding(tgt_tokens)

        enc_states = self.encoder(src_emb)

        if return_hidden:
            logits, dec_hidden = self.decoder(tgt_emb, enc_states, return_hidden=True)
        else:
            logits = self.decoder(tgt_emb, enc_states)

        emb_reg_loss = self.src_embedding.binary_reg_loss()
        if not self.config.shared_vocab:
            emb_reg_loss = (emb_reg_loss + self.tgt_embedding.binary_reg_loss()) / 2.0

        if return_hidden:
            return logits, emb_reg_loss, [enc_states, dec_hidden]
        return logits, emb_reg_loss

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す('total'キー必須)。"""
        counts: dict[str, int] = {}
        counts['embedding'] = sum(p.numel() for p in self.src_embedding.parameters())
        if not self.config.shared_vocab:
            counts['embedding'] += sum(
                p.numel() for p in self.tgt_embedding.parameters())
        counts['encoder'] = sum(p.numel() for p in self.encoder.parameters())
        decoder_head = sum(p.numel() for p in self.decoder.head.parameters())
        counts['decoder'] = sum(
            p.numel() for p in self.decoder.parameters()) - decoder_head
        counts['head'] = decoder_head
        counts['total'] = sum(counts.values())
        return counts

    @staticmethod
    def _gates_of(root: nn.Module | None) -> int:
        """rootの部分木に含まれるLUTニューロン数(共有モジュールは1回だけ数える)。"""
        if root is None:
            return 0
        return sum(
            int(m.num_neurons) for m in root.modules() if hasattr(m, 'num_neurons')
        )

    def count_gates(self) -> dict[str, int]:
        """ゲート数(LUTニューロン数)の内訳。'total'は全モジュール走査と一致する。"""
        dec = self.decoder
        token_mix = sum(
            self._gates_of(b.token_mix)
            for b in list(self.encoder.blocks) + list(dec.blocks)
        )
        enc_blocks = sum(self._gates_of(b) for b in self.encoder.blocks) \
            - sum(self._gates_of(b.token_mix) for b in self.encoder.blocks)
        dec_blocks = sum(self._gates_of(b) for b in dec.blocks) \
            - sum(self._gates_of(b.token_mix) for b in dec.blocks)
        cross = self._gates_of(dec.cross_attn) + self._gates_of(dec.cross_out)
        head = self._gates_of(dec.head)
        embed_proj = (self._gates_of(self.encoder.embed_proj)
                      + self._gates_of(dec.embed_proj)
                      + self._gates_of(dec.context_proj))

        counts = {
            'embed_proj': embed_proj,
            'enc_blocks': enc_blocks,
            'dec_blocks': dec_blocks,
            'token_mix': token_mix,
            'cross': cross,
            'head': head,
        }
        counts['total'] = sum(counts.values())
        return counts

    def extra_repr(self) -> str:
        c = self.config
        return (
            f'vocab={c.vocab_size}, d_emb={c.d_emb}, seq_len={c.seq_len}, '
            f'd_channels={c.d_channels}, hidden={c.hidden_width}, '
            f'enc/dec_blocks={c.n_enc_blocks}/{c.n_dec_blocks}, '
            f'channel_depth={c.channel_depth}, kernels={c.token_kernels}, '
            f'gate={c.gate}, residual={c.residual_mode}, '
            f'context={c.context_mode}({c.ctx_width}bit), '
            f'cross={c.cross_attention}, '
            f'positional={c.positional}({c.pos_width}bit)'
        )


__all__ = [
    'LearnedPositionBits',
    'LogicSeqMixer',
    'LogicSeqMixerBlock',
    'LogicSeqMixerConfig',
    'LogicSeqMixerDecoder',
    'LogicSeqMixerEncoder',
    'ToeplitzTokenMix',
    'estimate_gates',
]
