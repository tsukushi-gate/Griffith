"""attention梯子のモデル(L0基準 + L1射影ゲート化)。

「梯子」は attention の(射影/スコア/正規化)を1段ずつゲート化する。FFNは全段共通で
論理ゲート層(LogicMLP)に統一でき、埋め込み・ヘッドはFP固定。段は model_cfg の
2キーで表現する:
  - attention ∈ {'fp','logic_proj'}  … 'fp'=既存MHA(L0) / 'logic_proj'=射影ゲート化(L1)
  - ffn ∈ {'fp','logic'}             … 'fp'=既存Sequential(L0-FP) / 'logic'=LogicMLP(梯子共通)

汎用 `LadderTransformerBlock` が attn(callable, `last_attn_weights` を持つ)と ffn を
受け取り pre-norm残差を実行する。attention='fp' は既存MHAを薄くラップするアダプタ、
'logic_proj' は `LadderAttention`。ffn='logic' はコアの `LogicMLP` を流用する
(コア実装 griffith_logic/ は読み取りのみ)。
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

# ------------------------------------------------------------
# import規約: src/ をパス追加(harness/train と同じ流儀)。
# 通常は harness 経由で既にパスが通っているが、直接importにも耐えるよう保険を置く。
# ------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ladder_attention import LadderAttention  # noqa: E402

from griffith_logic.hybrid.policy import LogicSpec  # noqa: E402
from griffith_logic.layers.logic_linear import LogicMLP  # noqa: E402

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
_DEFAULT_SCORE = 'fp_dot'
_DEFAULT_NORM = 'softmax'
_DEFAULT_TOPK = 4

# 論理FFN(LogicMLP)の固定レシピ(動作検証済み構成)
# n_inputs=2: DLGN標準の2入力ゲート。2^2=4基底でMPS pure-torch経路が高速
# (n_inputs=6は2^6=64基底で~16倍遅く、ローカルMPSでは非現実的)
_LOGIC_FFN_N_INPUTS = 2
_LOGIC_FFN_DEPTH = 2


class _FPAttentionAdapter(nn.Module):
    """既存 nn.MultiheadAttention を薄くラップするアダプタ。

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


def _set_temperature(model: nn.Module, tau: float) -> None:
    """モデル配下の全 LadderAttention に温度アニーリングを反映する。

    LadderAttention 側で選択器normのみ反映される(softmax/fp段は no-op)。
    """
    for module in model.modules():
        if isinstance(module, LadderAttention):
            module.set_temperature(tau)


def _pop_aux_loss(model: nn.Module) -> torch.Tensor | float:
    """モデル配下の全 LadderAttention の aux_loss を合計し0クリアして返す。"""
    total: torch.Tensor | float = 0.0
    for module in model.modules():
        if isinstance(module, LadderAttention):
            total = total + module.pop_aux_loss()
    return total


def _build_attention(
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
        )
    raise NotImplementedError(f"attention='{attention}' は未実装です")


def _build_ffn(
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


def _build_block(
    model_cfg: dict,
    d_model: int,
    n_heads: int,
    mlp_ratio: int,
    dropout: float,
    causal: bool,
) -> LadderTransformerBlock:
    """1ブロックを構築する。RNG消費順(L0互換)のため attn → ffn の順に構築する。"""
    attn = _build_attention(model_cfg, d_model, n_heads, dropout, causal)
    ffn = _build_ffn(model_cfg, d_model, mlp_ratio, dropout)
    return LadderTransformerBlock(d_model, attn, ffn)


class LadderClassifier(nn.Module):
    """分類用梯子Transformer(sMNIST等)。

    入力射影 + 学習位置埋め込み → 非causalブロック → 時間mean-pool → 分類ヘッド。
    forward(x:(B,T,input_dim)) -> (B,num_classes)。埋め込み・ヘッドはFP固定。
    """

    def __init__(self, model_cfg: dict, meta: dict) -> None:
        super().__init__()
        d_model = int(model_cfg['d_model'])
        n_heads = int(model_cfg['n_heads'])
        n_layers = int(model_cfg['n_layers'])
        mlp_ratio = int(model_cfg['mlp_ratio'])
        dropout = float(model_cfg.get('dropout', 0.0))

        seq_len = int(meta['seq_len'])
        input_dim = int(meta['input_dim'])
        num_classes = int(meta['num_classes'])

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            _build_block(model_cfg, d_model, n_heads, mlp_ratio, dropout, causal=False)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        # 各層のattention重み (B,H,Sq,Sk) を保持
        self.last_attn_weights: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        h = self.input_proj(x) + self.pos_embed(pos).unsqueeze(0)

        self.last_attn_weights = []
        for block in self.blocks:
            h = block(h)
            self.last_attn_weights.append(block.last_attn_weights)

        h = self.norm(h)
        h = h.mean(dim=1)  # 時間方向mean-pool
        return self.head(h)

    def set_temperature(self, tau: float) -> None:
        """全 LadderAttention の温度を更新(選択器normのみ有効)。"""
        _set_temperature(self, tau)

    def pop_aux_loss(self) -> torch.Tensor | float:
        """前回forwardの補助損失合計を返し0クリアする。"""
        return _pop_aux_loss(self)


class LadderLanguageModel(nn.Module):
    """言語モデル用梯子Transformer(Tiny Shakespeare等)。

    トークン埋め込み + 学習位置埋め込み → causalブロック → LMヘッド。
    forward(x:(B,T) long) -> (B,T,vocab)。埋め込み・ヘッドはFP固定。
    """

    def __init__(self, model_cfg: dict, meta: dict) -> None:
        super().__init__()
        d_model = int(model_cfg['d_model'])
        n_heads = int(model_cfg['n_heads'])
        n_layers = int(model_cfg['n_layers'])
        mlp_ratio = int(model_cfg['mlp_ratio'])
        dropout = float(model_cfg.get('dropout', 0.0))

        seq_len = int(meta['seq_len'])
        vocab_size = int(meta['vocab_size'])

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            _build_block(model_cfg, d_model, n_heads, mlp_ratio, dropout, causal=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.last_attn_weights: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos).unsqueeze(0)

        self.last_attn_weights = []
        for block in self.blocks:
            h = block(h)
            self.last_attn_weights.append(block.last_attn_weights)

        h = self.norm(h)
        return self.head(h)  # (B,T,vocab)

    def set_temperature(self, tau: float) -> None:
        """全 LadderAttention の温度を更新(選択器normのみ有効)。"""
        _set_temperature(self, tau)

    def pop_aux_loss(self) -> torch.Tensor | float:
        """前回forwardの補助損失合計を返し0クリアする。"""
        return _pop_aux_loss(self)


def build_model(model_cfg: dict, meta: dict) -> nn.Module:
    """設定とメタ情報からモデルを構築する。

    `model_cfg['attention'] ∈ {'fp','logic_proj'}` × `model_cfg['ffn'] ∈ {'fp','logic'}`
    を許可する(既定は fp/fp = L0-FP)。未知の値は NotImplementedError を投げる
    (1段=1変数の原則)。分類/LMともに対応(LMはcausal=True)。
    """
    attention = model_cfg.get('attention', _ATTENTION_FP)
    if attention not in _ATTENTIONS:
        raise NotImplementedError(
            f"attention='{attention}' は未実装です(対応: {_ATTENTIONS})。"
        )
    ffn = model_cfg.get('ffn', _FFN_FP)
    if ffn not in _FFNS:
        raise NotImplementedError(
            f"ffn='{ffn}' は未実装です(対応: {_FFNS})。"
        )

    task = meta['task']
    if task == 'classification':
        return LadderClassifier(model_cfg, meta)
    if task == 'lm':
        return LadderLanguageModel(model_cfg, meta)
    raise ValueError(f'未知のtask: {task}')
