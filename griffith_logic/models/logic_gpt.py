"""LogicGPT: decoder-only 因果self-attention 論理ゲートTransformer。

教師nanoGPT(decoder-only causal Transformer)と**同じアーキ族**のchar-LM生徒。
RDDLGN(再帰)やDLGNAttentionTranslator(再帰デコーダ+cross-attention)と異なり、
教師と同じ「因果self-attention + FFNブロックの積み重ね」構造で蒸留する。

    LogicEmbedding → 位置符号(soft-XOR注入) → N×LogicTransformerBlock(causal=True)
    → LutStackヘッド → GroupSum で各位置のvocabロジットを出す。全経路[0,1]論理域。

forward(tokens[B,S]) → (logits[B,S,vocab], emb_reg_loss) はRDDLGNと同一契約
(訓練ループの二値化正則化ランプをそのまま再利用できる)。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..blocks.positional import position_codes
from ..blocks.transformer import LogicTransformerBlock
from ..group_sum import GroupSum
from ..training.spec import LearningSpec, with_learning
from .rddlgn import LogicEmbedding, LutStack

# dfa_targetsに指定できる論理名。
#   'mlp'  — 各LogicTransformerBlockのFFN(LogicMLP)
#   'head' — 出力ヘッド(LutStack、GroupSum直前)
# q/k/v射影とattention softmax(LogicAttention)は文献知見(Launay 2020)通り
# 常にBPのまま(注釈対象に含めない)。
DFA_TARGET_CHOICES = ('mlp', 'head')
DFA_MODE_CHOICES = ('micro', 'macro')


@dataclass
class LogicGPTConfig:
    """LogicGPTのハイパーパラメータ。"""

    vocab_size: int = 67
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 1
    block_size: int = 64
    mlp_ratio: int = 2
    # 位置符号: block_size <= 2^pos_width かつ d_model >= pos_width が必要
    pos_width: int = 8
    pos_code: str = 'binary'
    # ヘッド幅 = vocab_size * head_group_factor(GroupSumのグループ幅)
    head_group_factor: int = 8
    tau: float = 1.0
    n_inputs: int = 2
    parameterization: str = 'softmax'
    selection: str = 'soft'  # 'soft' | 'hard_st' | 'gumbel_st'(配管)
    connections: str = 'random'
    merge: str = 'xor'
    norm: str = 'layernorm'

    @classmethod
    def tiny(cls) -> 'LogicGPTConfig':
        """デバッグ・CPU smoke用の小構成。"""
        return cls(
            vocab_size=67, d_model=64, n_layers=2, n_heads=1,
            block_size=64, mlp_ratio=2, pos_width=6, head_group_factor=4,
            tau=1.0,
        )

    @classmethod
    def from_paper(cls, scale: float = 0.05) -> 'LogicGPTConfig':
        """RDDLGN論文スケール(既定scale=0.05)に相当する規模の構成。

        LutLayerのパラメータ数はout_dim(≒2^n_inputs×出力幅)支配で
        in_dimに依存しないため、d_model/n_layers/mlp_ratioを実測しながら
        手動較正した(count_parameters()で ~1.2M に着地することを確認済み、
        docs参照:実装レポート)。
        """
        if abs(scale - 0.05) > 1e-9:
            raise NotImplementedError(
                'from_paper は現状 scale=0.05 の較正済み構成のみ対応'
            )
        return cls(
            vocab_size=67, d_model=1152, n_layers=6, n_heads=2,
            block_size=64, mlp_ratio=4, pos_width=8, head_group_factor=32,
            tau=1.0,
        )


class LogicGPT(nn.Module):
    """decoder-only 因果Transformerの論理ゲート版。forward(tokens[B,S])→(logits, emb_reg)。

    因果性: logits[:, i] は tokens[:, :i+1] のみに依存する。
    - LogicEmbedding/位置符号/ヘッド/GroupSum はすべて位置独立
    - LogicTransformerBlockは causal=True(未来位置を参照しない)
    - LogicLayerNormは特徴軸(d_model)正規化で系列軸を跨がない
    """

    def __init__(
        self,
        config: LogicGPTConfig,
        dfa_targets: list[str] | None = None,
        dfa_mode: str = 'micro',
        dfa_lr: float | None = None,
    ):
        """
        Args:
            config: LogicGPTConfig
            dfa_targets: with_learning(rule='dfa')で注釈するモジュール群。
                None(既定)= 注釈なし = forward/学習ともに従来のBP完全維持
                (非回帰)。['mlp']/['head']/['mlp','head']を指定可能。
                q/k/v射影とattention softmax(LogicAttention)は文献知見
                (Launay 2020: attentionの完全BP-free化は未達)通り常にBPのまま
                注釈対象に含めない。'attn'を渡した場合は将来の完全BP-free
                探索用としてLogicAttentionのq/k/v射影(k_proj/q_proj/v_proj)
                も注釈する(現状は実験的、既定では使わない)。
            dfa_mode: 'micro'(既定)= LogicMLP/head単位で個別に注釈。
                'macro' = LogicTransformerBlock単位でまとめて注釈する
                (block全体の出力[B,S,d_model]をDFAの局所グラフ境界にする。
                Launay 2020のblock単位フィードバックに対応)。
                dfa_targets=Noneの場合は無視される。'macro'時、
                dfa_targets内の'head'指定は引き続きheadを個別に注釈する
                (head自体はblockの外にあるため)。'mlp'指定は無視される
                (mlp単体でなくblock全体を注釈するため)。
        """
        super().__init__()
        self.config = config
        self.dfa_targets = list(dfa_targets) if dfa_targets else []
        self.dfa_mode = dfa_mode
        for t in self.dfa_targets:
            assert t in DFA_TARGET_CHOICES + ('attn',), (
                f'dfa_targets の要素は {DFA_TARGET_CHOICES + ("attn",)} '
                f'のいずれか: {t}'
            )
        assert dfa_mode in DFA_MODE_CHOICES, (
            f'dfa_mode は {DFA_MODE_CHOICES} のいずれか: {dfa_mode}'
        )
        assert config.block_size <= (1 << config.pos_width), (
            f'block_size({config.block_size}) <= 2^pos_width'
            f'({1 << config.pos_width}) が必要'
        )
        assert config.d_model >= config.pos_width, (
            f'd_model({config.d_model}) >= pos_width({config.pos_width}) が必要'
        )

        self.embedding = LogicEmbedding(config.vocab_size, config.d_model)

        # 位置符号 (block_size, d_model): 先頭pos_widthビットに符号、残りは0。
        # soft-XORで各位置の埋め込みに注入する(幅不変を保つ)。
        table = position_codes(config.block_size, config.pos_width, config.pos_code)
        pos = torch.zeros(config.block_size, config.d_model)
        pos[:, :config.pos_width] = table
        self.register_buffer('pos', pos)

        lut_kwargs = dict(
            n_inputs=config.n_inputs,
            parameterization=config.parameterization,
            connections=config.connections,
            selection=config.selection,
        )
        self.blocks = nn.ModuleList([
            LogicTransformerBlock(
                config.d_model, n_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio, norm=config.norm,
                merge=config.merge, causal=True, **lut_kwargs,
            )
            for _ in range(config.n_layers)
        ])

        head_width = config.vocab_size * config.head_group_factor
        self.head = LutStack(config.d_model, [head_width], **lut_kwargs)
        self.group_sum = GroupSum(config.vocab_size, tau=config.tau)

        self._annotate_dfa(dfa_lr)

    def _annotate_dfa(self, dfa_lr: float | None) -> None:
        """dfa_targets/dfa_modeに従いwith_learning(rule='dfa')を注釈する。

        dfa_targets=[](既定)なら何もしない(全モジュール未注釈=完全BP)。
        """
        if not self.dfa_targets:
            return
        spec_kwargs = {} if dfa_lr is None else {'lr': dfa_lr}

        if self.dfa_mode == 'macro':
            # block単位(macro): LogicTransformerBlock自体を1つのDFA境界にする。
            # 'mlp'単体注釈は意味を持たない(blockごと包含されるため無視)。
            if 'mlp' in self.dfa_targets:
                for blk in self.blocks:
                    with_learning(blk.mlp, LearningSpec(rule='bp'))
            for blk in self.blocks:
                with_learning(blk, LearningSpec(rule='dfa', **spec_kwargs))
        else:
            # micro: LogicMLP(FFN)単体を注釈する。
            if 'mlp' in self.dfa_targets:
                for blk in self.blocks:
                    with_learning(blk.mlp, LearningSpec(rule='dfa', **spec_kwargs))

        if 'head' in self.dfa_targets:
            with_learning(self.head, LearningSpec(rule='dfa', **spec_kwargs))

        if 'attn' in self.dfa_targets:
            # 実験的: 完全BP-free探索用にq/k/v射影も注釈する(既定では使わない)。
            for blk in self.blocks:
                attn = blk.attn
                with_learning(attn.k_proj, LearningSpec(rule='dfa', **spec_kwargs))
                if attn.q_proj is not attn.k_proj:
                    with_learning(attn.q_proj, LearningSpec(rule='dfa', **spec_kwargs))
                with_learning(attn.v_proj, LearningSpec(rule='dfa', **spec_kwargs))

    def forward(
        self, tokens: torch.Tensor, return_hidden: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Args:
            tokens: (B, S) 整数トークンID(S <= block_size)
            return_hidden: True の場合、中間活性リストも追加で返す
                (exit-head蒸留用。既定Falseでは既存の2-tuple契約を完全維持する)。

        Returns:
            return_hidden=False(既定): (logits[B, S, vocab_size], emb_reg_loss)
                の2-tuple(既存契約と同一)。
            return_hidden=True: (logits, emb_reg_loss, hidden_states) の3-tuple。
                hidden_states は長さ n_layers のリスト([0,1]論理域の中間活性)。
                各要素は各 LogicTransformerBlock の出力: (B, S, d_model)
                (attention+residual+norm+MLP+residual+normを経た後のブロック
                出力そのもの。全ブロックが同じ形状 [0,1] 論理域を保つため、
                exit head蒸留の中間表現として自然な粒度として選択した)。
        """
        b, s = tokens.shape
        assert s <= self.config.block_size, (
            f'S({s}) <= block_size({self.config.block_size}) が必要'
        )
        x = self.embedding(tokens)                      # (B,S,d_model) [0,1]
        pos = self.pos[:s].unsqueeze(0).to(x.dtype)     # (1,S,d_model)
        x = x + pos - 2.0 * x * pos                     # soft-XOR 位置注入
        hidden_states: list[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)                                  # (B,S,d_model) 因果
            if return_hidden:
                hidden_states.append(x)
        h = self.head(x.reshape(b * s, self.config.d_model))  # (B*S, vocab*gf)
        logits = self.group_sum(h).reshape(b, s, self.config.vocab_size)
        emb_reg_loss = self.embedding.binary_reg_loss()
        if return_hidden:
            return logits, emb_reg_loss, hidden_states
        return logits, emb_reg_loss

    def count_parameters(self) -> dict[str, int]:
        """パラメータ数の内訳を返す(RDDLGNのcount_parameters()と同じ構成)。"""
        counts = {}
        counts['embedding'] = sum(p.numel() for p in self.embedding.parameters())
        counts['blocks'] = sum(p.numel() for p in self.blocks.parameters())
        counts['head'] = sum(p.numel() for p in self.head.parameters())
        counts['total'] = sum(counts.values())
        return counts
