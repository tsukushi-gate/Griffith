"""L2: Attention系6モデルの最小挙動テスト。

対象(griffith_logic/models/attention/):
    DLGNBooleanAttention / DLGNMultiQueryAttention / DLGNPositionalAttention
    DLGNAttentionSeq2Seq / DLGNPointerAttention / DLGNAttentionTranslator

各モデルについて:
    (a) 最小構成Configでのforward形状
    (b) 勾配疎通: backward後に全学習パラメータのgradが存在し全有限
    (c) boolean_attentionのみ: hard(eval)モードのattention重みが
        XNORスコアargmaxのone-hotになる意味論の検証
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F32_EQ

from griffith_logic.models.attention import (
    AttentionSeq2SeqConfig,
    BooleanAttentionConfig,
    DLGNAttentionSeq2Seq,
    DLGNAttentionTranslator,
    DLGNBooleanAttention,
    DLGNMultiQueryAttention,
    DLGNPointerAttention,
    DLGNPositionalAttention,
    MultiQueryAttentionConfig,
    PointerAttentionConfig,
    PositionalAttentionConfig,
    TranslatorConfig,
)

BATCH = 2


def _binary(batch, dim, seed):
    torch.manual_seed(seed)
    return torch.randint(0, 2, (batch, dim)).float()


def _assert_grads_finite(model):
    """全学習パラメータについてgradの存在と有限性を検証"""
    for name, p in model.named_parameters():
        assert p.grad is not None, f'{name} のgradがNone'
        assert torch.isfinite(p.grad).all(), f'{name} のgradに非有限値'


# ============================================================
# Task 1: Boolean Attention(連想記憶)
# ============================================================

class TestBooleanAttention:
    def _make(self, seed=41):
        torch.manual_seed(seed)
        cfg = BooleanAttentionConfig(
            num_pairs=2, key_bits=2, value_bits=2,
            encoder_sizes=[8, 4], value_encoder_sizes=[8, 4],
            decoder_sizes=[16, 8],
        )
        return DLGNBooleanAttention(cfg), cfg

    def test_forward_shape(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=1)
        logits = model(x)
        assert logits.shape == (BATCH, cfg.num_classes)

    def test_grad_flow(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=2)
        target = torch.randint(0, cfg.num_classes, (BATCH,))
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        _assert_grads_finite(model)

    def test_hard_attention_is_argmax_onehot(self):
        """(c) hard(eval)モード: attention重み == XNORスコアargmaxのone-hot"""
        model, cfg = self._make()
        model.eval()
        batch = 4
        x = _binary(batch, cfg.input_dim, seed=3)

        with torch.no_grad():
            logits = model(x)
            attn = model.last_attention

            # one-hot性: 値は{0,1}かつ各行の和が1
            assert attn.shape == (batch, cfg.num_pairs)
            assert torch.all((attn == 0.0) | (attn == 1.0))
            assert torch.equal(attn.sum(dim=-1), torch.ones(batch))

            # 選択インデックス == XNOR類似度のargmax(独立に再計算)
            query, keys, values = model.parse_input(x)
            q = model.query_encoder(query)
            k = model.key_encoder(
                keys.reshape(batch * cfg.num_pairs, cfg.key_bits),
            ).reshape(batch, cfg.num_pairs, -1)
            q_exp = q.unsqueeze(1)
            xnor = q_exp * k + (1.0 - q_exp) * (1.0 - k)
            expected_idx = xnor.sum(dim=-1).argmax(dim=-1)
            assert torch.equal(attn.argmax(dim=-1), expected_idx)

            # argmaxで選択したvalue経由のlogitsと一致(hard選択の意味論)
            v = model.value_encoder(
                values.reshape(batch * cfg.num_pairs, cfg.value_bits),
            ).reshape(batch, cfg.num_pairs, -1)
            selected = v[torch.arange(batch), expected_idx]
            logits_manual = model.group_sum(model.output_decoder(selected))
            assert torch.allclose(logits, logits_manual, atol=TOL_F32_EQ)


# ============================================================
# Task 2: Multi-Query Attention
# ============================================================

class TestMultiQueryAttention:
    def _make(self, seed=43):
        torch.manual_seed(seed)
        cfg = MultiQueryAttentionConfig(
            num_pairs=2, num_queries=2, key_bits=2, value_bits=2,
            encoder_sizes=[8, 4], value_encoder_sizes=[8, 4],
            decoder_sizes=[16, 8],
        )
        return DLGNMultiQueryAttention(cfg), cfg

    def test_forward_shape(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=4)
        logits = model(x)
        assert logits.shape == (BATCH, cfg.num_queries, cfg.num_classes)

    def test_grad_flow(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=5)
        target = torch.randint(0, cfg.num_classes, (BATCH, cfg.num_queries))
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.num_classes), target.reshape(-1),
        )
        loss.backward()
        _assert_grads_finite(model)

    def test_hard_attention_onehot(self):
        model, cfg = self._make()
        model.eval()
        x = _binary(BATCH, cfg.input_dim, seed=6)
        with torch.no_grad():
            model(x)
        attn = model.last_attention  # (B, Q, P)
        assert attn.shape == (BATCH, cfg.num_queries, cfg.num_pairs)
        assert torch.all((attn == 0.0) | (attn == 1.0))
        assert torch.equal(
            attn.sum(dim=-1), torch.ones(BATCH, cfg.num_queries),
        )


# ============================================================
# Task 3: Positional Attention(位置指定ルックアップ)
# ============================================================

class TestPositionalAttention:
    def _make(self, seed=47):
        torch.manual_seed(seed)
        cfg = PositionalAttentionConfig(
            pos_bits=2, value_bits=2,
            encoder_sizes=[8, 4], value_encoder_sizes=[8, 4],
            decoder_sizes=[16, 8],
        )
        return DLGNPositionalAttention(cfg), cfg

    def test_forward_shape(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=7)
        logits = model(x)
        assert logits.shape == (BATCH, cfg.num_classes)

    def test_grad_flow(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=8)
        target = torch.randint(0, cfg.num_classes, (BATCH,))
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        _assert_grads_finite(model)


# ============================================================
# Task 4: Attention seq2seq(固定長ソート)
# ============================================================

class TestAttentionSeq2Seq:
    def _make(self, seed=53):
        torch.manual_seed(seed)
        cfg = AttentionSeq2SeqConfig(
            seq_len=4, value_bits=2,
            encoder_sizes=[8, 4], value_encoder_sizes=[8, 4],
            decoder_sizes=[16, 8],
        )
        return DLGNAttentionSeq2Seq(cfg), cfg

    def test_forward_shape(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=9)
        logits = model(x)
        assert logits.shape == (BATCH, cfg.seq_len, cfg.num_classes)

    def test_grad_flow(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=10)
        target = torch.randint(0, cfg.num_classes, (BATCH, cfg.seq_len))
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.num_classes), target.reshape(-1),
        )
        loss.backward()
        _assert_grads_finite(model)


# ============================================================
# Task 5: Pointer Attention(Pointer Network風ソート)
# ============================================================

class TestPointerAttention:
    def _make(self, seed=59):
        torch.manual_seed(seed)
        cfg = PointerAttentionConfig(
            seq_len=4, value_bits=2,
            value_encoder_sizes=[8, 4], key_encoder_sizes=[8, 4],
            decoder_cell_sizes=[8, 4], query_producer_sizes=[8, 4],
            output_decoder_sizes=[16, 8],
        )
        return DLGNPointerAttention(cfg), cfg

    def test_forward_shape(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=11)
        logits, attn = model(x)
        assert logits.shape == (BATCH, cfg.seq_len, cfg.num_classes)
        assert attn.shape == (BATCH, cfg.seq_len, cfg.seq_len)

    def test_grad_flow_teacher_forcing(self):
        model, cfg = self._make()
        model.train()
        x = _binary(BATCH, cfg.input_dim, seed=12)
        target_ptr = torch.stack([
            torch.randperm(cfg.seq_len) for _ in range(BATCH)
        ])
        target = torch.randint(0, cfg.num_classes, (BATCH, cfg.seq_len))
        logits, _ = model(x, target_pointers=target_ptr)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.num_classes), target.reshape(-1),
        )
        loss.backward()
        _assert_grads_finite(model)


# ============================================================
# Task 6: Attention Translator(Bahdanau-style seq2seq)
# ============================================================

class TestAttentionTranslator:
    def _make(self, seed=61):
        torch.manual_seed(seed)
        cfg = TranslatorConfig(
            vocab_size=8, d_emb=8, src_len=4, tgt_len=4,
            n_layer_sizes=[8, 8], key_encoder_sizes=[8, 4],
            l_layer_sizes=[8, 8], p_layer_sizes=[8, 8],
            query_producer_sizes=[8, 4], m_layer_sizes=[16, 16],
        )
        return DLGNAttentionTranslator(cfg), cfg

    def test_forward_shape(self):
        model, cfg = self._make()
        model.train()
        torch.manual_seed(13)
        src = torch.randint(0, cfg.vocab_size, (BATCH, cfg.src_len))
        tgt = torch.randint(0, cfg.vocab_size, (BATCH, cfg.tgt_len))
        logits, emb_reg = model(src, tgt)
        assert logits.shape == (BATCH, cfg.tgt_len, cfg.vocab_size)
        assert emb_reg.ndim == 0
        assert torch.isfinite(emb_reg)

    def test_grad_flow(self):
        model, cfg = self._make()
        model.train()
        torch.manual_seed(14)
        src = torch.randint(0, cfg.vocab_size, (BATCH, cfg.src_len))
        tgt = torch.randint(0, cfg.vocab_size, (BATCH, cfg.tgt_len))
        logits, emb_reg = model(src, tgt)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1),
        ) + emb_reg
        loss.backward()
        _assert_grads_finite(model)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
