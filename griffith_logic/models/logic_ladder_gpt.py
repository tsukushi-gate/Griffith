"""LogicLadderGPT: 自己回帰言語モデルの論理ゲート化(D1、旧名LogicGPT)。

PR #6のKD生徒モデル LogicGPT と名前が衝突したため、mainマージ時に
ladder系(本モデル)を LogicLadderGPT へ改名した。

`experiments/attention_ladder/` の `LadderLanguageModel`(causal decoder-only
Transformer、attention段階論理化)を、再利用可能なコアモデルへ昇格したもの。
`LogicLadderGPTConfig` で L0(fp/fp)から L3-C(射影ゲート化+選択器+論理FFN)までを表現する。

構成: token埋め込み + 学習位置埋め込み → causalブロック → LMヘッド。
forward(idx:(B,T) long) -> (B,T,vocab)。埋め込み・ヘッドはFP固定。ブロック内の
attention/ffn を `attention`/`ffn` フラグと `attention_kwargs` で段階的にゲート化する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..blocks.causal_attention import (
    build_block,
    pop_aux_loss,
    set_temperature,
)

# LogicLadderGPTConfig → model_cfg 変換時に、attention_kwargs へ回さない上位キー
_MODEL_TOP_KEYS = frozenset({
    'd_model', 'n_heads', 'n_layers', 'mlp_ratio', 'dropout', 'attention', 'ffn',
})


@dataclass
class LogicLadderGPTConfig:
    """LogicLadderGPT の構成。ladder の model_cfg 語彙をそのまま継承する。

    L0 は attention='fp', ffn='fp'。L3-C は attention='logic_proj', ffn='logic' に
    加え、選択器種別・温度等を `attention_kwargs`(ladder model_cfg の残りキー)へ透過する。
    """

    vocab_size: int
    seq_len: int                    # 最大コンテキスト長
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0
    attention: str = 'fp'           # ladderの_ATTENTIONS語彙をそのまま継承
    ffn: str = 'fp'                 # ladderの_FFNS語彙をそのまま継承
    attention_kwargs: dict = field(default_factory=dict)
    weight_tying: bool = False      # token_embed と head の重み共有

    def to_model_cfg(self) -> dict:
        """ladder互換の model_cfg 辞書へ変換する(build_block へ渡す形)。"""
        model_cfg = {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout,
            'attention': self.attention,
            'ffn': self.ffn,
        }
        # 選択器種別・温度等(score/norm/qk_bits/...)を透過する
        model_cfg.update(self.attention_kwargs)
        return model_cfg

    @classmethod
    def from_model_cfg(
        cls,
        model_cfg: dict,
        *,
        vocab_size: int,
        seq_len: int,
    ) -> LogicLadderGPTConfig:
        """yamlの model_cfg 辞書から Config を構築する。

        上位キー(d_model等)以外は `attention_kwargs` へ透過する。L0/L3-C の
        どちらの model_cfg でも構築できる。
        """
        attention_kwargs = {
            key: value
            for key, value in model_cfg.items()
            if key not in _MODEL_TOP_KEYS
        }
        return cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=int(model_cfg.get('d_model', 128)),
            n_heads=int(model_cfg.get('n_heads', 4)),
            n_layers=int(model_cfg.get('n_layers', 2)),
            mlp_ratio=int(model_cfg.get('mlp_ratio', 4)),
            dropout=float(model_cfg.get('dropout', 0.0)),
            attention=str(model_cfg.get('attention', 'fp')),
            ffn=str(model_cfg.get('ffn', 'fp')),
            attention_kwargs=attention_kwargs,
            weight_tying=bool(model_cfg.get('weight_tying', False)),
        )


class LogicLadderGPT(nn.Module):
    """自己回帰言語モデル(decoder-only causal Transformer)の論理ゲート化。

    サブモジュール名は `LadderLanguageModel` と一致させており、同一 state_dict を
    直接転写して数値等価にできる(tests/test_logic_gpt.py の等価テストで担保)。
    """

    def __init__(self, config: LogicLadderGPTConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        model_cfg = config.to_model_cfg()

        # RNG消費順(ladder互換)のため token_embed → pos_embed → blocks →
        # norm → head の順に構築する。埋め込み・ヘッドはFP固定。
        self.token_embed = nn.Embedding(config.vocab_size, d_model)
        self.pos_embed = nn.Embedding(config.seq_len, d_model)
        self.blocks = nn.ModuleList([
            build_block(
                model_cfg, d_model, config.n_heads, config.mlp_ratio,
                config.dropout, causal=True,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, config.vocab_size)
        if config.weight_tying:
            # token埋め込みとLMヘッドの重みを共有(head.weight is token_embed.weight)
            self.head.weight = self.token_embed.weight
        # 各層のattention重み (B,H,Sq,Sk) を保持
        self.last_attn_weights: list[torch.Tensor] = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """(B,T) long → (B,T,vocab) logits。T <= config.seq_len。"""
        seq_len = idx.size(1)
        if seq_len > self.config.seq_len:
            raise ValueError(
                f'系列長 T={seq_len} が seq_len={self.config.seq_len} を超えています'
            )
        pos = torch.arange(seq_len, device=idx.device)
        h = self.token_embed(idx) + self.pos_embed(pos).unsqueeze(0)

        self.last_attn_weights = []
        for block in self.blocks:
            h = block(h)
            self.last_attn_weights.append(block.last_attn_weights)

        h = self.norm(h)
        return self.head(h)  # (B,T,vocab)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """自己回帰生成。(B,T)→(B,T+max_new_tokens)。

        コンテキストは seq_len 末尾でクロップする。greedy=True または
        temperature<=0 で argmax(決定的)。それ以外は temperature スケール後に
        top_k フィルタ→softmax→multinomial でサンプルする(generatorで再現可能)。

        soft/hard(選択器の乱数含む)は呼び出し側の train()/eval() 状態に従う。
        決定的な生成には eval() を推奨する。
        """
        for _ in range(max_new_tokens):
            # コンテキストを seq_len 末尾でクロップ
            idx_cond = idx[:, -self.config.seq_len:]
            logits = self(idx_cond)          # (B,T,vocab)
            logits = logits[:, -1, :]        # 末尾位置のみ (B,vocab)

            if greedy or temperature <= 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # (B,1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < thresh, float('-inf'))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(
                    probs, num_samples=1, generator=generator,
                )  # (B,1)

            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def set_temperature(self, tau: float) -> None:
        """全 LadderAttention の温度を更新(選択器normのみ有効, ladder互換)。"""
        set_temperature(self, tau)

    def pop_aux_loss(self) -> torch.Tensor | float:
        """前回forwardの補助損失合計を返し0クリアする(ladder互換)。"""
        return pop_aux_loss(self)

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す(wandbログ用)。"""
        counts = {
            'token_embed': sum(p.numel() for p in self.token_embed.parameters()),
            'pos_embed': sum(p.numel() for p in self.pos_embed.parameters()),
            'blocks': sum(p.numel() for p in self.blocks.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
        }
        # weight_tying時は head と token_embed が同一tensorで二重計上になるため、
        # total は実体ベース(重複排除)で数える
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        counts['total'] = total
        return counts


__all__ = ['LogicLadderGPT', 'LogicLadderGPTConfig']
