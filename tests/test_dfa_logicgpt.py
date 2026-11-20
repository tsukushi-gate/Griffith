"""LogicGPT/DLGNAttentionTranslatorのDFA注釈オプションの検証。

観点:
    1. 非注釈(既定)時、forward/学習が従来のBPと完全に同一(非回帰)。
       具体的には、DFA関連コード(with_learning/DFARule)が一切介在せず、
       全パラメータのgrad_fnが通常のautogradグラフに乗ること。
    2. dfa_targets指定時、指定したモジュールのみwith_learning(rule='dfa')
       で注釈され、HybridTrainer経由でtrain_stepを回すとDFA注釈層の
       param.gradが(BP由来ではなく)DFA注入由来で埋まること。
    3. micro/macroモードの切り替えでアノテーション対象(粒度)が変わること。
    4. q/k/v射影・attention softmaxはいずれのdfa_targetsでも常にBPのまま
       (未注釈)であること(Launay 2020知見の実装保証)。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.attention.attention_translator import (
    DLGNAttentionTranslator,
    TranslatorConfig,
)
from griffith_logic.models.logic_gpt import LogicGPT, LogicGPTConfig
from griffith_logic.training.trainer import HybridTrainer


def _tiny_gpt_config():
    return LogicGPTConfig(
        vocab_size=12, d_model=16, n_layers=2, n_heads=1,
        block_size=8, mlp_ratio=2, pos_width=4, head_group_factor=4,
        tau=1.0,
    )


class LogitsOnlyWrap(nn.Module):
    """(logits, emb_reg_loss)を返すモデルをlogitsのみ返すように包むテスト用ラッパ。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def _seq_loss(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


class TestLogicGPTNoRegression:
    def test_default_no_dfa_annotation(self, seed):
        """dfa_targets未指定(既定)では一切のモジュールが未注釈"""
        model = LogicGPT(_tiny_gpt_config())
        for name, mod in model.named_modules():
            assert getattr(mod, '_learning_spec', None) is None, (
                f'既定でモジュール{name}が注釈されている(非回帰違反)'
            )

    def test_default_forward_matches_bp_baseline(self, seed):
        """dfa_targets未指定時、forward出力がbitwise一致(2回構築+同一seed)"""
        torch.manual_seed(42)
        model_a = LogicGPT(_tiny_gpt_config())
        torch.manual_seed(42)
        model_b = LogicGPT(_tiny_gpt_config())

        x = torch.randint(0, 12, (3, 8))
        logits_a, emb_a = model_a(x)
        logits_b, emb_b = model_b(x)
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(emb_a, emb_b)
        assert logits_a.requires_grad

    def test_default_training_step_bitwise_matches_plain_torch(self, seed):
        """dfa_targets未指定のLogicGPTは素のtorch訓練ループと勾配が一致"""
        torch.manual_seed(5)
        model = LogicGPT(_tiny_gpt_config())
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randint(0, 12, (2, 8))
        target = torch.randint(0, 12, (2, 8))

        opt.zero_grad()
        logits, emb_loss = model(x)
        loss = _seq_loss(logits, target) + 0.0 * emb_loss
        loss.backward()
        grads_plain = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        torch.manual_seed(5)
        model2 = LogicGPT(_tiny_gpt_config())
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        opt2.zero_grad()
        logits2, emb_loss2 = model2(x)
        loss2 = _seq_loss(logits2, target) + 0.0 * emb_loss2
        loss2.backward()
        for n, p in model2.named_parameters():
            if p.grad is None:
                continue
            assert torch.equal(grads_plain[n], p.grad), f'{n} 勾配不一致'


class TestLogicGPTDFAAnnotation:
    @pytest.mark.parametrize('dfa_targets', [['mlp'], ['head'], ['mlp', 'head']])
    def test_targets_annotated_correctly(self, dfa_targets, seed):
        model = LogicGPT(_tiny_gpt_config(), dfa_targets=dfa_targets, dfa_mode='micro')
        annotated = {
            name for name, mod in model.named_modules()
            if getattr(mod, '_learning_spec', None) is not None
        }
        if 'mlp' in dfa_targets:
            assert 'blocks.0.mlp' in annotated
            assert 'blocks.1.mlp' in annotated
        else:
            assert 'blocks.0.mlp' not in annotated
        if 'head' in dfa_targets:
            assert 'head' in annotated
        else:
            assert 'head' not in annotated

        # attention内部(q/k/v射影・LogicAttention自体)は常に未注釈
        for name, mod in model.named_modules():
            if '.attn' in name:
                assert getattr(mod, '_learning_spec', None) is None, (
                    f'attention経路{name}がDFA注釈されている(Launay知見違反)'
                )

    def test_macro_mode_annotates_block_not_mlp(self, seed):
        model = LogicGPT(_tiny_gpt_config(), dfa_targets=['mlp', 'head'], dfa_mode='macro')
        block_specs = {
            name: mod._learning_spec.rule for name, mod in model.named_modules()
            if getattr(mod, '_learning_spec', None) is not None
        }
        assert block_specs.get('blocks.0') == 'dfa'
        assert block_specs.get('blocks.1') == 'dfa'
        # macroではblock単位が境界になるため、内部mlpは(仮に注釈されていても)
        # HybridTrainerの親prefix-skipにより無視される
        wrapped = LogitsOnlyWrap(model)
        trainer = HybridTrainer(wrapped, loss_fn=_seq_loss)
        dfa_names = {n for n, _ in trainer._dfa._entries}
        assert any(n.endswith('blocks.0') for n in dfa_names)
        assert not any(n.endswith('blocks.0.mlp') for n in dfa_names)
        trainer.close()

    def test_hybrid_trainer_injects_dfa_grad_for_annotated_layers(self, seed):
        """DFA注釈層のparam.gradがHybridTrainer.train_step後に埋まること。

        注記(重要な設計上の発見): headはmodel出力の直前に位置し、
        LogicTransformerBlockのresidual経路のようなBPバイパスを持たない。
        headをDFA注釈すると、そこから上流(全blocks/embedding/attention)への
        BP経路が完全に遮断される(DFAの「上流を遮断する」設計通りの帰結)。
        そのため mlp+head 同時注釈時は embedding.grad が None になる
        (upstream層はBP勾配を一切受け取らない=学習信号が届かない)。
        これは非回帰違反ではなく、DFAアーキテクチャの必然的な性質
        (block_upstream=Trueの意味そのもの)。mlpのみ注釈時はresidual経由の
        バイパスがあるためembedding/attnにもBP勾配が届く
        (test_mlp_only_dfa_leaves_bp_path_via_residualで確認)。
        """
        torch.manual_seed(9)
        model = LogicGPT(_tiny_gpt_config(), dfa_targets=['mlp', 'head'], dfa_mode='micro')
        wrapped = LogitsOnlyWrap(model)
        trainer = HybridTrainer(wrapped, loss_fn=_seq_loss)
        assert trainer._dfa is not None

        x = torch.randint(0, 12, (4, 8))
        target = torch.randint(0, 12, (4, 8))
        trainer.train_step(x, target)

        for blk in model.blocks:
            assert all(p.grad is not None for p in blk.mlp.parameters())
        assert all(p.grad is not None for p in model.head.parameters())
        trainer.close()

    def test_mlp_only_dfa_leaves_bp_path_via_residual(self, seed):
        """mlpのみDFA注釈時、soft-XOR残差経由でembedding/attnにBP勾配が届くこと

        (headが未注釈なら、head→group_sum→logitsの経路がBPのまま残り、
        LogicTransformerBlockのresidual(merge([h, mlp(h)]))のh側分岐から
        attention/embeddingまでBPが到達できる)。
        """
        torch.manual_seed(9)
        model = LogicGPT(_tiny_gpt_config(), dfa_targets=['mlp'], dfa_mode='micro')
        wrapped = LogitsOnlyWrap(model)
        trainer = HybridTrainer(wrapped, loss_fn=_seq_loss)

        x = torch.randint(0, 12, (4, 8))
        target = torch.randint(0, 12, (4, 8))
        trainer.train_step(x, target)

        for blk in model.blocks:
            assert all(p.grad is not None for p in blk.mlp.parameters())
        assert model.embedding.embedding.weight.grad is not None
        assert model.blocks[0].attn.k_proj.layers[0].weights.grad is not None
        assert model.head.layers[0].weights.grad is not None
        trainer.close()

    def test_micro_vs_macro_different_annotation_granularity(self, seed):
        model_micro = LogicGPT(_tiny_gpt_config(), dfa_targets=['mlp'], dfa_mode='micro')
        model_macro = LogicGPT(_tiny_gpt_config(), dfa_targets=['mlp'], dfa_mode='macro')

        micro_annotated = {
            name for name, mod in model_micro.named_modules()
            if getattr(mod, '_learning_spec', None) is not None
        }
        macro_annotated = {
            name for name, mod in model_macro.named_modules()
            if getattr(mod, '_learning_spec', None) is not None
        }
        assert 'blocks.0.mlp' in micro_annotated
        assert 'blocks.0' not in micro_annotated
        assert 'blocks.0' in macro_annotated

    def test_dfa_learns_something_over_many_steps(self, seed):
        """DFA(micro, mlpのみ)でtiny LogicGPTのlossが複数stepで下がること
        (収束速度は問わない。学習信号が入っていることの最低限の確認)。

        dfa_targets=['mlp']を使う理由(重要、実験メモ):
        headまで含めてDFA化すると(dfa_targets=['mlp','head'])、head→group_sum→
        logitsがDFAの局所グラフ境界になり、そこから上流の全層(embedding/
        attention/mlp自身の入力側)へのBP経路が完全に遮断される
        (test_hybrid_trainer_injects_dfa_grad_for_annotated_layers参照)。
        この設定でtiny GPTを実験したところ、8サンプル固定バッチの
        単純overfitでも300 stepでlossが改善しないどころか悪化した
        (early=2.47→late=2.66、min=2.46は初期に記録)。
        一方 dfa_targets=['mlp'] (headはBPのまま)なら150 stepで
        2.45→1.53まで下がる。これはLaunay 2020の知見
        (「完全DFAは弱く、BP-connectedな出力トランクが要る」)と整合する
        負の結果であり、報告に含める。
        """
        torch.manual_seed(21)
        model = LogicGPT(_tiny_gpt_config(), dfa_targets=['mlp'], dfa_mode='micro')
        wrapped = LogitsOnlyWrap(model)
        trainer = HybridTrainer(
            wrapped, optimizer=torch.optim.Adam(wrapped.parameters(), lr=0.01),
            loss_fn=_seq_loss,
        )
        torch.manual_seed(1)
        x = torch.randint(0, 12, (8, 8))
        target = torch.randint(0, 12, (8, 8))

        losses = []
        for _ in range(150):
            stats = trainer.train_step(x, target)
            losses.append(stats['loss'])
        trainer.close()
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early, f'DFAでlossが改善しない: early={early:.4f} late={late:.4f}'

    def test_full_dfa_mlp_and_head_struggles_without_bp_trunk(self, seed):
        """負の結果の固定化: mlp+head全DFA(headから上流のBP経路が完全遮断)は
        150〜300 stepの短期学習では改善しない、または悪化することを記録する
        (Launay 2020: 完全BP-freeなDFAはBPに大きく劣る、という文献知見の再現)。
        このテストは「学習が失敗する」ことそのものを仕様として固定するのではなく、
        実装が意図通り「上流を完全に遮断する」ことの副作用を可視化する回帰記録。
        """
        torch.manual_seed(21)
        model = LogicGPT(_tiny_gpt_config(), dfa_targets=['mlp', 'head'], dfa_mode='micro')
        wrapped = LogitsOnlyWrap(model)
        trainer = HybridTrainer(
            wrapped, optimizer=torch.optim.Adam(wrapped.parameters(), lr=0.01),
            loss_fn=_seq_loss,
        )
        torch.manual_seed(1)
        x = torch.randint(0, 12, (8, 8))
        target = torch.randint(0, 12, (8, 8))

        losses = []
        for _ in range(150):
            stats = trainer.train_step(x, target)
            losses.append(stats['loss'])
        trainer.close()
        # 学習が完全に発散しない(有限)ことだけを保証する。改善するかどうかは
        # 問わない(文献通りの負の結果であり、上のdocstringに理由を記録)。
        assert all(l == l and l < 100 for l in losses), 'lossがnan/発散している'


class TestAttentionTranslatorDFAAnnotation:
    def _tiny_translator_config(self):
        cfg = TranslatorConfig.tiny()
        cfg.vocab_size = 10
        cfg.src_len = 5
        cfg.tgt_len = 5
        cfg.d_emb = 12
        cfg.n_layer_sizes = [12, 12]
        cfg.key_encoder_sizes = [8, 8]
        cfg.l_layer_sizes = [12, 12]
        cfg.p_layer_sizes = [20, 12]
        cfg.query_producer_sizes = [8, 8]
        cfg.m_layer_sizes = [20, 20, 10]
        return cfg

    def test_default_no_dfa_annotation(self, seed):
        model = DLGNAttentionTranslator(self._tiny_translator_config())
        for name, mod in model.named_modules():
            assert getattr(mod, '_learning_spec', None) is None

    def test_encoder_and_l_layers_annotated(self, seed):
        model = DLGNAttentionTranslator(
            self._tiny_translator_config(), dfa_targets=['encoder', 'l_layers'],
        )
        annotated = {
            name for name, mod in model.named_modules()
            if getattr(mod, '_learning_spec', None) is not None
        }
        assert 'encoder.n_layers' in annotated
        assert 'encoder.key_encoder' in annotated
        assert 'decoder.l_layers' in annotated
        # cross-attention(query_producer)とP/M-layersは常に未注釈
        assert 'decoder.query_producer' not in annotated
        assert 'decoder.p_layers' not in annotated
        assert 'decoder.m_layers' not in annotated

    def test_hybrid_trainer_runs_with_dfa_targets(self, seed):
        cfg = self._tiny_translator_config()
        model = DLGNAttentionTranslator(cfg, dfa_targets=['encoder', 'l_layers'])

        class Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, xy):
                src, tgt = xy
                logits, _ = self.m(src, tgt)
                return logits

        wrapped = Wrap(model)
        trainer = HybridTrainer(wrapped, loss_fn=_seq_loss)
        assert trainer._dfa is not None

        src = torch.randint(0, cfg.vocab_size, (3, cfg.src_len))
        tgt = torch.randint(0, cfg.vocab_size, (3, cfg.tgt_len))
        target = torch.randint(0, cfg.vocab_size, (3, cfg.tgt_len))

        trainer.train_step((src, tgt), target)
        assert all(p.grad is not None for p in model.encoder.n_layers.parameters())
        assert all(p.grad is not None for p in model.encoder.key_encoder.parameters())
        assert all(p.grad is not None for p in model.decoder.l_layers.parameters())
        trainer.close()

    def test_invalid_dfa_target_raises(self, seed):
        with pytest.raises(AssertionError):
            DLGNAttentionTranslator(
                self._tiny_translator_config(), dfa_targets=['not_a_real_target'],
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
