"""合成可能なLogicAttention部品。

既存attention 6モデル(models/attention/)に散在していた
Q/K/V射影(LutStack)・XNOR類似度スコア・attention選択
(soft=softmax / hard=argmax STE)・value集約を、再利用可能な単一部品
として抽出した。既存6モデルは無改造(boolean_attention.pyとの数値等価は
tests/l2_grad/test_blocks_attention.py で恒久照合)。

HDL生成不可の理由:
    soft attentionはsoftmax(実数の指数・除算)、hard attentionはargmax
    (整数スコアの比較器木+one-hot MUX)を要し、現行IRの純論理
    プリミティブ(lut_layer/concat/add等)では表現できない。
    そのためIR上は blackbox+contract(in/out幅付きblackboxノード。
    blocks/to_ir.py)として表現する。hard-argmaxの比較器木+MUXによる
    マクロ展開HDL化は(P2バックログ、起票のみ)で扱う。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: LutStackはコンストラクタ内で遅延importする。モジュール先頭でimportすると
# blocks.attention → models/__init__ → logic_gpt → blocks.transformer → blocks.attention
# の循環importになる(mainマージで顕在化)。

ATTENTION_MODES = ('soft', 'hard')
SIMILARITIES = ('xnor',)


class LogicAttention(nn.Module):
    """XNOR類似度によるattention部品。

    アーキテクチャ(boolean_attention.py と同一の数式):
        q/k/v(各 (B, S, d_model) の[0,1]ビット列)
          ↓ q_proj / k_proj / v_proj (LutStack)
        (B, S, d_qk), (B, S, d_qk), (B, S, d_v)
          ↓ XNOR類似度: score = Σ_d [q·k + (1-q)(1-k)]  (ヘッドごと)
        scores (B, H, S_q, S_k)
          ↓ soft: softmax(score/τ) / hard: argmaxのone-hot(STE)
        value集約 (B, S_q, d_v)

    mode:
        'soft' — softmax加重平均(訓練用、微分可能)
        'hard' — argmaxのone-hot選択。forwardは厳密にone-hot、
                 attention重み経路のbackwardはsoftmaxの勾配を通すSTE
                 (value集約の∂out/∂vはforward値=one-hotを使う。
                 forward契約+backward契約はSTE契約テスト方式で検証)

    Args:
        d_model: 入力ビット幅(q/k/v共通)
        n_heads: ヘッド数(射影次元 d_qk, d_v を等分割)
        mode: 'soft' | 'hard'(forwardのhard引数で呼び出しごとに上書き可)
        similarity: 'xnor' のみ対応
        qk_sizes: q/k射影のLutStackサイズ列(既定 [d_model])
        v_sizes: v射影のLutStackサイズ列(既定 [d_model])
        share_qk: q射影とk射影を共有(boolean_attentionのshare_key_query相当)
        tau: attention softmax温度
        causal: Trueで自己回帰マスク(位置iはj<=iのみ参照)。decoder-only
            LM(LogicGPT)用。S_q==S_k前提。既定Falseで従来の全結合self/cross-attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        mode: str = 'soft',
        similarity: str = 'xnor',
        *,
        qk_sizes: list | None = None,
        v_sizes: list | None = None,
        share_qk: bool = True,
        tau: float = 1.0,
        causal: bool = False,
        n_inputs: int = 2,
        parameterization: str = 'softmax',
        connections: str = 'random',
        grad_factor: float = 1.0,
        selection: str = 'soft',
    ):
        super().__init__()
        assert mode in ATTENTION_MODES, f'mode は {ATTENTION_MODES} のいずれか: {mode}'
        assert similarity in SIMILARITIES, (
            f'similarity は {SIMILARITIES} のいずれか: {similarity}'
        )
        qk_sizes = list(qk_sizes) if qk_sizes is not None else [d_model]
        v_sizes = list(v_sizes) if v_sizes is not None else [d_model]
        assert qk_sizes[-1] % n_heads == 0 and v_sizes[-1] % n_heads == 0, (
            f'射影次元(qk={qk_sizes[-1]}, v={v_sizes[-1]})は'
            f'n_heads({n_heads})で割り切れる必要がある'
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.mode = mode
        self.similarity = similarity
        self.tau = tau
        self.causal = causal
        self.d_qk = qk_sizes[-1]
        self.d_v = v_sizes[-1]

        lut_kwargs = dict(
            n_inputs=n_inputs,
            parameterization=parameterization,
            connections=connections,
            grad_factor=grad_factor,
            selection=selection,
        )
        from ..models.rddlgn import LutStack  # 循環import回避の遅延import

        self.k_proj = LutStack(d_model, qk_sizes, **lut_kwargs)
        self.q_proj = self.k_proj if share_qk \
            else LutStack(d_model, qk_sizes, **lut_kwargs)
        self.v_proj = LutStack(d_model, v_sizes, **lut_kwargs)

        # 直前のattention重み(診断用): (B, H, S_q, S_k)
        self._last_attn: torch.Tensor | None = None

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hard: bool,
    ) -> torch.Tensor:
        """射影済みq/k/vからXNOR類似度attention集約を計算する。

        Args:
            q: (B, S_q, d_qk)、k: (B, S_k, d_qk)、v: (B, S_k, d_v)。値は[0,1]

        Returns:
            (B, S_q, d_v)
        """
        batch, s_q, _ = q.shape
        s_k = k.shape[1]
        h = self.n_heads
        qh = q.reshape(batch, s_q, h, -1).transpose(1, 2)  # (B,H,Sq,dh)
        kh = k.reshape(batch, s_k, h, -1).transpose(1, 2)
        vh = v.reshape(batch, s_k, h, -1).transpose(1, 2)

        # XNOR類似度: score = Σ_d [q·k + (1-q)(1-k)](popcount相当)
        scores = qh @ kh.transpose(-1, -2) \
            + (1.0 - qh) @ (1.0 - kh).transpose(-1, -2)  # (B,H,Sq,Sk)

        if self.causal:
            # 自己回帰: 位置iはj<=iのみ参照。未来位置(j>i)のスコアを最小値へ
            # 潰し、softmax重み=0・argmax対象外にする(参照実装と同じfinfo.min)。
            # S_q==S_k(自己注意)前提。
            assert s_q == s_k, (
                f'causal=True は S_q==S_k の自己注意を想定: {s_q} vs {s_k}'
            )
            future = torch.triu(
                torch.ones(s_q, s_k, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores / self.tau, dim=-1)
        if hard:
            onehot = F.one_hot(scores.argmax(dim=-1), s_k).to(scores.dtype)
            # STE: forwardは厳密にone-hot、backwardはsoftmaxの勾配
            attn = onehot + (attn - attn.detach())
        self._last_attn = attn.detach()

        out = attn @ vh  # (B,H,Sq,dvh)
        return out.transpose(1, 2).reshape(batch, s_q, -1)

    def forward(
        self,
        q_bits: torch.Tensor,
        k_bits: torch.Tensor,
        v_bits: torch.Tensor,
        hard: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q_bits: (B, S_q, d_model) [0,1]ビット列
            k_bits: (B, S_k, d_model)
            v_bits: (B, S_k, d_model)
            hard: None=self.modeに従う / True/False=呼び出しごとの上書き

        Returns:
            (B, S_q, d_v)
        """
        if hard is None:
            hard = self.mode == 'hard'
        batch, s_q, _ = q_bits.shape
        s_k = k_bits.shape[1]
        q = self.q_proj(q_bits.reshape(batch * s_q, self.d_model)) \
            .reshape(batch, s_q, -1)
        k = self.k_proj(k_bits.reshape(batch * s_k, self.d_model)) \
            .reshape(batch, s_k, -1)
        v = self.v_proj(v_bits.reshape(batch * s_k, self.d_model)) \
            .reshape(batch, s_k, -1)
        return self.attend(q, k, v, hard)

    @property
    def last_attention(self) -> torch.Tensor | None:
        """直前のforwardのattention重み (B, H, S_q, S_k)"""
        return self._last_attn

    def extra_repr(self):
        return (f'd_model={self.d_model}, n_heads={self.n_heads}, '
                f'mode={self.mode}, similarity={self.similarity}')
