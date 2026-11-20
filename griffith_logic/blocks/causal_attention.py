"""causal対応の射影ゲート化attentionとTransformerブロック(D1 LogicGPT用)。

本モジュールは `experiments/attention_ladder/` の実験実装を、再利用可能なコア部品へ
忠実に昇格(コピー+改名)したもの。実験と数値等価であることを `tests/test_logic_gpt.py`
の等価テストで担保する。experiments 配下への実行時依存は持たない(自己完結コピー)。

「梯子」は attention の(射影/スコア/正規化)を1段ずつゲート化する:
  - attention ∈ {'fp','logic_proj'}  … 'fp'=既存MHA(L0) / 'logic_proj'=射影ゲート化(L1〜)
  - ffn ∈ {'fp','logic'}             … 'fp'=既存Sequential(L0-FP) / 'logic'=LogicMLP(梯子共通)

`LadderAttention` は q/k/v を独立の論理射影スタックで{0,1}ビット列に射影し、per-headの
実内積(fp_dot)または XNOR-popcount でスコアを取り、softmax/離散選択器で正規化する。
`_FPAttentionAdapter` は既存 nn.MultiheadAttention を薄くラップするL0基準。
`LadderTransformerBlock` は attn/ffn を受け取り pre-norm残差を実行する汎用ブロック。

コア実装(griffith_logic/core, layers 等)は素朴なground truthとして流用する。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..binarize import Binarize
from ..hybrid.policy import LogicSpec
from ..layers.logic_linear import LogicMLP
from ..layers.lut import LutLayer

# ============================================================
# attention梯子のバリアント語彙(実験の _ATTENTIONS/_FFNS を継承)
# ============================================================
# attentionの実装バリアント
_ATTENTION_FP = 'fp'
_ATTENTION_LOGIC_PROJ = 'logic_proj'
_ATTENTIONS = (_ATTENTION_FP, _ATTENTION_LOGIC_PROJ)

# ffnの実装バリアント
_FFN_FP = 'fp'
_FFN_LOGIC = 'logic'
_FFNS = (_FFN_FP, _FFN_LOGIC)

# LadderAttention 既定値(model_cfg 未指定時)
_DEFAULT_QK_BITS = 128
_DEFAULT_V_BITS = 128
_DEFAULT_N_INPUTS = 6
_DEFAULT_PARAMETERIZATION = 'warp'
_DEFAULT_PROJ_DEPTH = 1
_DEFAULT_TAU = 1.0
# 射影LutLayerへのGumbelノイズ注入（Mind the Gap）。既定Off=既存挙動bitwise不変
_DEFAULT_GUMBEL_NOISE = False
_DEFAULT_GUMBEL_TAU = 1.0
_DEFAULT_SCORE = 'fp_dot'
_DEFAULT_NORM = 'softmax'
_DEFAULT_TOPK = 4

# 論理FFN(LogicMLP)の固定レシピ(動作検証済み構成)
# n_inputs=2: DLGN標準の2入力ゲート。2^2=4基底でMPS pure-torch経路が高速
# (n_inputs=6は2^6=64基底で~16倍遅く、ローカルMPSでは非現実的)
_LOGIC_FFN_N_INPUTS = 2
_LOGIC_FFN_DEPTH = 2

# スコア方式(L1=fp_dot / L2=xnor_popcount)
SCORES = ('fp_dot', 'xnor_popcount')
# XNOR-popcount の二値化閾値(soft[0,1]を hard{0,1} に据える)
_XNOR_THRESHOLD = 0.5
# 正規化方式(L1/L2=softmax / L3で離散選択器4案を追加)
NORMS = ('softmax', 'mux_gumbel', 'softmax_int', 'topk_ste', 'threshold')
# 温度アニーリング(tau)を使う選択器norm(softmax/softmax_intは非対象=既存挙動を保護)
_SELECTOR_NORMS = ('mux_gumbel', 'topk_ste', 'threshold')

# Binarize の閾値(実数の pre-norm 出力を符号で{0,1}化)
_BINARIZE_THRESHOLD = 0.0

# 決定的タイブレーク用の微小バイアス係数(argmax同点はクエリに近い位置優先)
_TIEBREAK_EPS = 1e-4
# Gumbelノイズの数値安定用(U~Uniform を [eps,1-eps] にクランプ)
_GUMBEL_EPS = 1e-9
# 正規化(加重和)の0除け
_NORM_EPS = 1e-9
# A案(threshold)の選択率補助損失の下限/上限(5〜50%に収める)
_RATE_MIN = 0.05
_RATE_MAX = 0.5
# softmax_int(S案)の温度クランプ下限
_SOFTMAX_INT_TEMP_MIN = 1e-3


class _LogicProjection(nn.Module):
    """実数入力(B,S,d_model)を{0,1}ビット列(B,S,bits)に射影する論理スタック。

    構成: `Binarize(threshold=0.0)` → `LutLayer(d_model→bits)`
          → `LutLayer(bits→bits)` を(depth-1)段。
    訓練時は soft[0,1]、eval時は hard{0,1}(LutLayer が `self.training` で自動分岐。
    Binarize は forward 常に{0,1}でbackwardのみSTE)。
    """

    def __init__(
        self,
        d_model: int,
        bits: int,
        *,
        n_inputs: int,
        parameterization: str,
        depth: int,
        gumbel_noise: bool = _DEFAULT_GUMBEL_NOISE,
        gumbel_tau: float = _DEFAULT_GUMBEL_TAU,
    ) -> None:
        super().__init__()
        assert depth >= 1, f'proj_depth は1以上: {depth}'
        self.d_model = d_model
        self.bits = bits
        # Binarize は閾値0で実数→{0,1}(backwardはHardTanh STE)
        self.binarize = Binarize(threshold=_BINARIZE_THRESHOLD)
        luts: list[nn.Module] = []
        prev = d_model
        for _ in range(depth):
            luts.append(LutLayer(
                in_dim=prev,
                out_dim=bits,
                n_inputs=n_inputs,
                parameterization=parameterization,
                gumbel_noise=gumbel_noise,
                gumbel_tau=gumbel_tau,
            ))
            prev = bits
        self.luts = nn.Sequential(*luts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S,d_model) 実数(ブロックで pre-norm 済み前提)
        batch, seq, _ = x.shape
        h = self.binarize(x.reshape(batch * seq, self.d_model))
        h = self.luts(h)
        return h.reshape(batch, seq, self.bits)


class LadderAttention(nn.Module):
    """射影をゲート化したself-attention(L1)。

    q/k/v は独立の論理射影スタックで{0,1}ビット列に射影し、per-headの実内積で
    スコアを取り、FP softmax で正規化して value を集約する。出力は(B,S,v_bits)実数
    (v_bits=d_model なので残差に直接加算できる)。

    `hard` 引数は取らない: soft/hard は `model.train()`/`model.eval()` に従い
    LutLayer/Binarize が自動分岐する。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        causal: bool = False,
        score: str = 'fp_dot',
        norm: str = 'softmax',
        qk_bits: int = 128,
        v_bits: int = 128,
        n_inputs: int = 6,
        parameterization: str = 'warp',
        proj_depth: int = 1,
        tau: float = 1.0,
        topk: int = 4,
        gumbel_noise: bool = _DEFAULT_GUMBEL_NOISE,
        gumbel_tau: float = _DEFAULT_GUMBEL_TAU,
    ) -> None:
        super().__init__()
        if score not in SCORES:
            raise NotImplementedError(
                f"score='{score}' は未実装です(対応は {SCORES})。"
            )
        if norm not in NORMS:
            raise NotImplementedError(
                f"norm='{norm}' は未実装です(対応は {NORMS})。"
            )
        assert qk_bits % n_heads == 0 and v_bits % n_heads == 0, (
            f'qk_bits({qk_bits})/v_bits({v_bits})は n_heads({n_heads})で'
            f'割り切れる必要がある'
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal
        self.score = score
        self.norm = norm
        self.qk_bits = qk_bits
        self.v_bits = v_bits
        self.tau = tau
        self.topk = topk
        self.dk = qk_bits // n_heads  # ヘッドあたりのQ/Kビット幅

        proj_kwargs = dict(
            n_inputs=n_inputs,
            parameterization=parameterization,
            depth=proj_depth,
            gumbel_noise=gumbel_noise,
            gumbel_tau=gumbel_tau,
        )
        # q/k/v は独立スタック(共有しない)
        self.q_proj = _LogicProjection(d_model, qk_bits, **proj_kwargs)
        self.k_proj = _LogicProjection(d_model, qk_bits, **proj_kwargs)
        self.v_proj = _LogicProjection(d_model, v_bits, **proj_kwargs)

        # L2(xnor_popcount)用の学習可能スカラ(module当たり1個, FP)。
        # fp_dotの /sqrt(dk) のような固定スケールは付けず、このスカラが吸収する。
        # 定数tensorのため乱数を消費せず、fp_dot段のRNG消費順には影響しない。
        if score == 'xnor_popcount':
            self.score_scale = nn.Parameter(torch.tensor(1.0))

        # L3の正規化パラメータ(該当normのみ生成。torch.zeros/tensorはRNG非消費なので
        # softmax段(L1/L2)のRNG消費順には一切影響しない)。
        if norm == 'softmax_int':
            # S案: 整数スコアにそのままsoftmax。温度は学習パラメータ
            self.softmax_temp = nn.Parameter(torch.tensor(1.0))
        if norm == 'threshold':
            # A案: a_i=[s_i≥θ]。θはヘッドごと学習(初期0)
            self.threshold = nn.Parameter(torch.zeros(n_heads))

        # エントロピー計測が読む: (B,H,Sq,Sk)。属性名は既存MHAブロックと合わせる。
        # 選択器では選択重み(one-hot/mask/正規化a/softmax)を保持する。
        self.last_attn_weights: torch.Tensor | None = None
        # 検証用: 生の離散選択マスク(one-hot/mask/{0,1}a)。softmax系ではNone
        self.last_selection: torch.Tensor | None = None
        # 補助損失の蓄積(前回forward分)。pop_aux_loss で回収・0クリアする
        self.aux_loss: torch.Tensor | float = 0.0

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 上三角(strictly upper, diagonal=1)を -inf でマスク。対角は常に有効
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        return torch.triu(mask, diagonal=1)

    def _xnor_popcount_scores(
        self, qh: torch.Tensor, kh: torch.Tensor,
    ) -> torch.Tensor:
        """XNOR-popcount類似度(1が一致するビット数)。(B,H,Sq,Sk)。

        qh/kh は per-head の射影出力(訓練時 soft[0,1] / eval時 hard{0,1})。
        soft値(連続緩和)を勾配経路に、hard値(整数popcount)を順伝播値に据える
        STE(forward=hard / backward=∂soft)。学習可能スカラ score_scale で
        スケールする(fp_dotの /sqrt(dk) 固定スケールは付けない)。
        eval時は qh/kh が既に{0,1}なので s_soft==s_hard(=整数popcount)になる。
        """
        kt = kh.transpose(-1, -2)
        # soft値: 連続緩和。s_soft = qh·kh^T + (1-qh)·(1-kh)^T
        s_soft = qh @ kt + (1.0 - qh) @ (1.0 - kt)
        # hard値: 0.5閾値で二値化した整数popcount
        qb = (qh > _XNOR_THRESHOLD).float()
        kbt = (kh > _XNOR_THRESHOLD).float().transpose(-1, -2)
        s_hard = qb @ kbt + (1.0 - qb) @ (1.0 - kbt)
        # STE: forward=s_hard, backward=∂s_soft
        s = s_soft + (s_hard - s_soft).detach()
        return self.score_scale * s

    def _tiebreak_bias(
        self, seq_q: int, seq_k: int, device: torch.device,
    ) -> torch.Tensor:
        """決定的タイブレーク用の微小バイアス (Sq,Sk)。

        argmax同点は「クエリに近い位置優先」。非causalは `-eps*|i-j|`、
        causalは `-eps*(i-j)`(近い過去優先, future位置は元々 -inf マスク済み)。
        """
        i = torch.arange(seq_q, device=device).unsqueeze(1)  # (Sq,1)
        j = torch.arange(seq_k, device=device).unsqueeze(0)  # (1,Sk)
        diff = (i - j) if self.causal else (i - j).abs()
        return (-_TIEBREAK_EPS) * diff.float()

    # -- 正規化(L3の4案 + L1/L2 softmax) ---------------------------------

    def _norm_softmax(
        self, scores: torch.Tensor, vh: torch.Tensor,
    ) -> torch.Tensor:
        """L1/L2既定: FP softmax(挙動はL1から一切変更しない)。"""
        attn = torch.softmax(scores / self.tau, dim=-1)  # (B,H,Sq,Sk)
        self.last_attn_weights = attn
        self.last_selection = None
        return attn @ vh

    def _norm_softmax_int(
        self, scores: torch.Tensor, vh: torch.Tensor,
    ) -> torch.Tensor:
        """S案(対照): 整数スコアにそのまま温度学習softmax。平坦化定量化の基準。"""
        temp = self.softmax_temp.clamp_min(_SOFTMAX_INT_TEMP_MIN)
        attn = torch.softmax(scores / temp, dim=-1)
        self.last_attn_weights = attn
        self.last_selection = None
        return attn @ vh

    def _norm_mux_gumbel(
        self, scores: torch.Tensor, vh: torch.Tensor,
    ) -> torch.Tensor:
        """C案(本命): ヘッドごとargmaxの1トークンを選択(Gumbel-softmax ST)。"""
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        # 決定的タイブレーク(選択前=因果マスク後に適用)
        biased = scores + self._tiebreak_bias(seq_q, seq_k, scores.device)

        if self.training:
            # Gumbel-softmax straight-through
            u = torch.rand_like(biased).clamp_(_GUMBEL_EPS, 1.0 - _GUMBEL_EPS)
            g = -torch.log(-torch.log(u))
            y_soft = torch.softmax((biased + g) / self.tau, dim=-1)
            idx = y_soft.argmax(dim=-1)  # (B,H,Sq)
            y_hard = F.one_hot(idx, seq_k).to(scores.dtype)
            # forward=one-hot / backward=soft
            attn = y_hard - y_soft.detach() + y_soft
            # ヘッド多様性正則化(soft版で微分可能): 異なるヘッド間の同一トークン
            # 選択の二乗和。P=Σ_h y_soft(トークン別ヘッド占有)
            p = y_soft.sum(dim=1)  # (B,Sq,Sk)
            sum_p2 = (p ** 2).sum(dim=-1)  # (B,Sq): Σ_k P_k^2
            sum_self = (y_soft ** 2).sum(dim=(1, -1))  # (B,Sq): Σ_hΣ_k y_soft^2
            penalty = (sum_p2 - sum_self).mean()  # mean over (b,q)
            self.aux_loss = self.aux_loss + penalty
            self.last_selection = y_hard
        else:
            # eval: Gumbelノイズ無しの hard argmax one-hot
            idx = biased.argmax(dim=-1)
            attn = F.one_hot(idx, seq_k).to(scores.dtype)
            self.last_selection = attn

        self.last_attn_weights = attn
        return attn @ vh  # one-hotなので v_argmax

    def _norm_topk_ste(
        self, scores: torch.Tensor, vh: torch.Tensor,
    ) -> torch.Tensor:
        """B案: hard top-k + 選択vのビット多数決(STE)。"""
        k = min(self.topk, scores.shape[-1])
        # backward用: softmax(scores/tau)。因果マスク(-inf)は softmax で0になる
        soft = torch.softmax(scores / self.tau, dim=-1)  # (B,H,Sq,Sk)
        # 上位kの hard mask。future位置(-inf)は選ばない
        _, topk_idx = scores.topk(k, dim=-1)  # (B,H,Sq,k)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        mask = mask * torch.isfinite(scores).to(scores.dtype)  # 無効位置を除外

        # 集約: 選択k個のvのビット多数決(forward=hard)。backward=soft加重和
        count = mask.sum(dim=-1, keepdim=True)  # 実選択数(causal端では<k)
        selected_sum = mask @ vh  # (B,H,Sq,dvh): 選択vのビット和
        hard_out = (selected_sum > (count / 2.0)).to(scores.dtype)  # 多数決
        soft_out = soft @ vh  # soft加重和
        out = soft_out + (hard_out - soft_out).detach()  # STE

        # last_attn_weights=mask(正規化)。エントロピー計測用
        norm_mask = mask / (count + _NORM_EPS)
        self.last_attn_weights = norm_mask
        self.last_selection = mask
        return out

    def _norm_threshold(
        self, scores: torch.Tensor, vh: torch.Tensor,
    ) -> torch.Tensor:
        """A案: a_i=[s_i≥θ]、θヘッドごと学習(STE)。"""
        theta = self.threshold.view(1, self.n_heads, 1, 1)
        a_hard = (scores >= theta).to(scores.dtype)  # future(-inf)は0
        if self.training:
            # 温度は tau を共有。sigmoid((s-θ)/temp) の連続緩和で勾配を通す
            a_soft = torch.sigmoid((scores - theta) / self.tau)
            a = a_hard - a_soft.detach() + a_soft  # STE
            # 選択率補助損失: rateを5〜50%に収める
            rate = a.mean()
            aux = (
                torch.relu(torch.tensor(_RATE_MIN, device=rate.device) - rate)
                + torch.relu(rate - _RATE_MAX)
            )
            self.aux_loss = self.aux_loss + aux
        else:
            a = a_hard

        # 集約: 加重和を正規化
        denom = a.sum(dim=-1, keepdim=True) + _NORM_EPS
        out = (a @ vh) / denom
        self.last_attn_weights = a / denom  # 正規化a
        self.last_selection = a_hard  # 生の{0,1}選択(検証用)
        return out

    def _normalize(
        self, scores: torch.Tensor, vh: torch.Tensor,
    ) -> torch.Tensor:
        """norm に応じて正規化(集約)を行い (B,H,Sq,dvh) を返す。"""
        if self.norm == 'softmax':
            return self._norm_softmax(scores, vh)
        if self.norm == 'softmax_int':
            return self._norm_softmax_int(scores, vh)
        if self.norm == 'mux_gumbel':
            return self._norm_mux_gumbel(scores, vh)
        if self.norm == 'topk_ste':
            return self._norm_topk_ste(scores, vh)
        if self.norm == 'threshold':
            return self._norm_threshold(scores, vh)
        raise NotImplementedError(f"norm='{self.norm}' は未実装です")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S,d_model) 実数(pre-norm済み前提)。self-attention(q=k=v=x)
        batch, seq, _ = x.shape
        heads = self.n_heads
        # 補助損失は毎forwardでリセット(選択器の学習時のみ加算される)
        self.aux_loss = 0.0

        q = self.q_proj(x)  # (B,S,qk_bits)
        k = self.k_proj(x)  # (B,S,qk_bits)
        v = self.v_proj(x)  # (B,S,v_bits)

        # ヘッド分割: (B,S,bits) → (B,H,S,bits//H)
        qh = q.reshape(batch, seq, heads, -1).transpose(1, 2)
        kh = k.reshape(batch, seq, heads, -1).transpose(1, 2)
        vh = v.reshape(batch, seq, heads, -1).transpose(1, 2)

        # スコア: fp_dot=per-head実内積/sqrt(dk) / xnor_popcount=STE popcount。
        # (B,H,Sq,Sk)。'fp_dot' 経路はL1から一切変更しない。
        if self.score == 'fp_dot':
            scores = qh @ kh.transpose(-1, -2) / math.sqrt(self.dk)
        else:  # xnor_popcount
            scores = self._xnor_popcount_scores(qh, kh)
        # 因果マスクは正規化(選択)前に適用: future位置を -inf(選択不可)にする
        if self.causal:
            scores = scores + self._causal_mask(seq, x.device)

        # 正規化(softmax=L1/L2 / L3は離散選択器)。value集約まで担う
        out = self._normalize(scores, vh)  # (B,H,Sq,dvh)

        # ヘッド結合: (B,H,Sq,dvh) → (B,Sq,v_bits)
        return out.transpose(1, 2).reshape(batch, seq, self.v_bits)

    def set_temperature(self, tau: float) -> None:
        """温度アニーリング。選択器norm(mux_gumbel/topk_ste/threshold)のみ更新する。

        softmax/softmax_int は tau に依存しない(L1/L2既存挙動を保護)ため無視する。
        """
        if self.norm in _SELECTOR_NORMS:
            self.tau = float(tau)

    def pop_aux_loss(self) -> torch.Tensor | float:
        """前回forwardで蓄積した aux_loss を返し、0クリアする。"""
        aux = self.aux_loss
        self.aux_loss = 0.0
        return aux

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, n_heads={self.n_heads}, '
            f'causal={self.causal}, score={self.score}, norm={self.norm}, '
            f'qk_bits={self.qk_bits}, v_bits={self.v_bits}, tau={self.tau}, '
            f'topk={self.topk}'
        )


class _FPAttentionAdapter(nn.Module):
    """既存 nn.MultiheadAttention を薄くラップするアダプタ(L0基準)。

    入力 h=(B,S,d_model)(ブロックで pre-norm 済み)に self-attention を適用し、
    `last_attn_weights` に (B,H,Sq,Sk) を保持する。残差手前のdropoutも内部で適用し、
    L0(fp)の既存出力と一致させる。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self._causal = causal
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights: torch.Tensor | None = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 上三角(strictly upper)を -inf でマスク。対角は常に有効
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_mask = self._causal_mask(h.size(1), h.device) if self._causal else None
        # need_weights=True, average_attn_weights=False で (B,H,Sq,Sk) を取得
        attn_out, attn_weights = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn_weights = attn_weights
        return self.dropout(attn_out)


class LadderTransformerBlock(nn.Module):
    """汎用pre-norm残差ブロック。

    `attn`(callable module, `last_attn_weights` を持つ)と `ffn`(module)を受け取り、
    pre-norm残差 `x = x + attn(norm1(x)); x = x + ffn(norm2(x))` を実行する。
    attention重み (B,H,Sq,Sk) は `last_attn_weights` に転記する。
    """

    def __init__(self, d_model: int, attn: nn.Module, ffn: nn.Module) -> None:
        super().__init__()
        # 注: RNG消費順(L0互換)のため attn/ffn は呼び出し側で先に構築して渡す。
        # norm1/norm2(LayerNorm)は乱数を消費しないため構築順に依存しない。
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ffn
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        self.last_attn_weights = self.attn.last_attn_weights
        x = x + self.ffn(self.norm2(x))
        return x


def set_temperature(model: nn.Module, tau: float) -> None:
    """モデル配下の全 LadderAttention に温度アニーリングを反映する。

    LadderAttention 側で選択器normのみ反映される(softmax/fp段は no-op)。
    """
    for module in model.modules():
        if isinstance(module, LadderAttention):
            module.set_temperature(tau)


def pop_aux_loss(model: nn.Module) -> torch.Tensor | float:
    """モデル配下の全 LadderAttention の aux_loss を合計し0クリアして返す。"""
    total: torch.Tensor | float = 0.0
    for module in model.modules():
        if isinstance(module, LadderAttention):
            total = total + module.pop_aux_loss()
    return total


def build_attention(
    model_cfg: dict,
    d_model: int,
    n_heads: int,
    dropout: float,
    causal: bool,
) -> nn.Module:
    """model_cfg['attention'] に応じて attention モジュールを構築する。"""
    attention = model_cfg.get('attention', _ATTENTION_FP)
    if attention == _ATTENTION_FP:
        return _FPAttentionAdapter(d_model, n_heads, dropout, causal)
    if attention == _ATTENTION_LOGIC_PROJ:
        return LadderAttention(
            d_model,
            n_heads,
            causal=causal,
            score=str(model_cfg.get('score', _DEFAULT_SCORE)),
            norm=str(model_cfg.get('norm', _DEFAULT_NORM)),
            qk_bits=int(model_cfg.get('qk_bits', _DEFAULT_QK_BITS)),
            v_bits=int(model_cfg.get('v_bits', _DEFAULT_V_BITS)),
            n_inputs=int(model_cfg.get('n_inputs', _DEFAULT_N_INPUTS)),
            parameterization=str(
                model_cfg.get('parameterization', _DEFAULT_PARAMETERIZATION)
            ),
            proj_depth=int(model_cfg.get('proj_depth', _DEFAULT_PROJ_DEPTH)),
            tau=float(model_cfg.get('tau', _DEFAULT_TAU)),
            topk=int(model_cfg.get('topk', _DEFAULT_TOPK)),
            gumbel_noise=bool(
                model_cfg.get('gumbel_noise', _DEFAULT_GUMBEL_NOISE)
            ),
            gumbel_tau=float(model_cfg.get('gumbel_tau', _DEFAULT_GUMBEL_TAU)),
        )
    raise NotImplementedError(f"attention='{attention}' は未実装です")


def build_ffn(
    model_cfg: dict,
    d_model: int,
    mlp_ratio: int,
    dropout: float,
) -> nn.Module:
    """model_cfg['ffn'] に応じて FFN モジュールを構築する。"""
    ffn = model_cfg.get('ffn', _FFN_FP)
    if ffn == _FFN_FP:
        hidden = d_model * mlp_ratio
        return nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
    if ffn == _FFN_LOGIC:
        # 全段共通の論理ゲート層(動作検証済み構成)。実数in/実数out。
        spec = LogicSpec(
            parameterization='warp',
            encoder='sigmoid',
            decoder='groupsum_centered',
            depth=_LOGIC_FFN_DEPTH,
            width=d_model * mlp_ratio,
            n_inputs=_LOGIC_FFN_N_INPUTS,
        )
        return LogicMLP(d_model, d_model, spec=spec)
    raise NotImplementedError(f"ffn='{ffn}' は未実装です")


def build_block(
    model_cfg: dict,
    d_model: int,
    n_heads: int,
    mlp_ratio: int,
    dropout: float,
    causal: bool,
) -> LadderTransformerBlock:
    """1ブロックを構築する。RNG消費順(L0互換)のため attn → ffn の順に構築する。"""
    attn = build_attention(model_cfg, d_model, n_heads, dropout, causal)
    ffn = build_ffn(model_cfg, d_model, mlp_ratio, dropout)
    return LadderTransformerBlock(d_model, attn, ffn)
