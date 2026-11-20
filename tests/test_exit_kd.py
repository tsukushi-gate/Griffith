"""exit-head蒸留(OFA-KD式ロジット空間投影)のテスト。

WS0調査(クロスアーキテクチャKDのサーベイ §7)の結論を実装した
LogitExitHead/make_exit_kd_lossと、RDDLGN/LogicGPTのreturn_hidden対応を検証する。

- return_hidden=False(既定)の非回帰: 返り値の型・形状が既存契約と完全に同一。
- return_hidden=Trueでの中間活性の形状。
- LogitExitHeadの形状。
- make_exit_kd_lossのマスク規約遵守 + s==t投影完全一致時の損失最小性。
- exit有無でreturn_hidden=False経路の出力(数値)が同一(モデルにexit headを
  外付けするだけでは既存の推論経路に影響しないことの確認)。

CPU数秒で完結する高速スイート対象。
"""

import torch

from griffith_logic import RDDLGN, LogicGPT, LogicGPTConfig, RDDLGNConfig
from griffith_logic.convert.distill import LogitExitHead, kd_kl_loss_seq, make_exit_kd_loss


def _rddlgn_model(seed: int = 0) -> RDDLGN:
    torch.manual_seed(seed)
    return RDDLGN(RDDLGNConfig.tiny())


def _logicgpt_model(seed: int = 0) -> LogicGPT:
    torch.manual_seed(seed)
    return LogicGPT(LogicGPTConfig.tiny())


# ----------------------------------------------------------------------
# 1. return_hidden=False(既定)の非回帰
# ----------------------------------------------------------------------

class TestReturnHiddenDefaultNonRegression:

    def test_rddlgn_forward_default_returns_2tuple(self):
        m = _rddlgn_model()
        m.train()
        src = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))
        tgt = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))
        out = m(src, tgt)
        assert isinstance(out, tuple)
        assert len(out) == 2
        logits, emb_reg_loss = out
        assert logits.shape == (2, m.config.seq_len, m.config.vocab_size)
        assert emb_reg_loss.shape == ()

    def test_rddlgn_forward_explicit_false_matches_omitted(self):
        """return_hidden=False明示指定と省略時が完全一致(bitwise)。"""
        m = _rddlgn_model(seed=1)
        m.train()
        torch.manual_seed(123)
        src = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))
        tgt = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))

        m.eval()
        logits_a, reg_a = m(src, tgt)
        logits_b, reg_b = m(src, tgt, return_hidden=False)
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(reg_a, reg_b)

    def test_logicgpt_forward_default_returns_2tuple(self):
        m = _logicgpt_model()
        m.train()
        tok = torch.randint(0, m.config.vocab_size, (3, 16))
        out = m(tok)
        assert isinstance(out, tuple)
        assert len(out) == 2
        logits, reg = out
        assert logits.shape == (3, 16, m.config.vocab_size)
        assert reg.shape == ()

    def test_logicgpt_forward_explicit_false_matches_omitted(self):
        m = _logicgpt_model(seed=2)
        m.eval()
        tok = torch.randint(0, m.config.vocab_size, (2, 10))
        logits_a, reg_a = m(tok)
        logits_b, reg_b = m(tok, return_hidden=False)
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(reg_a, reg_b)


# ----------------------------------------------------------------------
# 2. return_hidden=True の形状
# ----------------------------------------------------------------------

class TestReturnHiddenShapes:

    def test_rddlgn_return_hidden_true_shapes(self):
        m = _rddlgn_model()
        m.train()
        batch = 2
        src = torch.randint(0, m.config.vocab_size, (batch, m.config.seq_len))
        tgt = torch.randint(0, m.config.vocab_size, (batch, m.config.seq_len))

        logits, emb_reg_loss, hidden_states = m(src, tgt, return_hidden=True)

        assert logits.shape == (batch, m.config.seq_len, m.config.vocab_size)
        assert emb_reg_loss.shape == ()
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == 2

        context, p_stack = hidden_states
        context_dim = m.config.k_layer_sizes[-1]
        p_out_dim = m.config.p_layer_sizes[-1]
        assert context.shape == (batch, context_dim)
        assert p_stack.shape == (batch, m.config.seq_len, p_out_dim)

    def test_rddlgn_return_hidden_true_logits_match_default(self):
        """return_hidden=Trueでもlogits/emb_reg_lossの値はFalse時と完全一致する。"""
        m = _rddlgn_model(seed=3)
        m.eval()
        src = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))
        tgt = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))

        logits_a, reg_a = m(src, tgt)
        logits_b, reg_b, _hidden = m(src, tgt, return_hidden=True)
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(reg_a, reg_b)

    def test_logicgpt_return_hidden_true_shapes(self):
        m = _logicgpt_model()
        m.train()
        batch, seqlen = 3, 12
        tok = torch.randint(0, m.config.vocab_size, (batch, seqlen))

        logits, reg, hidden_states = m(tok, return_hidden=True)

        assert logits.shape == (batch, seqlen, m.config.vocab_size)
        assert reg.shape == ()
        assert isinstance(hidden_states, list)
        assert len(hidden_states) == m.config.n_layers
        for h in hidden_states:
            assert h.shape == (batch, seqlen, m.config.d_model)

    def test_logicgpt_return_hidden_true_logits_match_default(self):
        m = _logicgpt_model(seed=4)
        m.eval()
        tok = torch.randint(0, m.config.vocab_size, (2, 10))

        logits_a, reg_a = m(tok)
        logits_b, reg_b, _hidden = m(tok, return_hidden=True)
        assert torch.equal(logits_a, logits_b)
        assert torch.equal(reg_a, reg_b)


# ----------------------------------------------------------------------
# 3. LogitExitHead
# ----------------------------------------------------------------------

class TestLogitExitHead:

    def test_output_shape_2d(self):
        head = LogitExitHead(hidden_dim=16, vocab_size=67)
        hidden = torch.rand(4, 16)
        out = head(hidden)
        assert out.shape == (4, 67)

    def test_output_shape_3d_sequence(self):
        head = LogitExitHead(hidden_dim=32, vocab_size=67)
        hidden = torch.rand(2, 8, 32)
        out = head(hidden)
        assert out.shape == (2, 8, 67)

    def test_is_plain_linear_module(self):
        """訓練時のみのfloat足場: 中身はnn.Linearであり論理ゲート化を要さない。"""
        head = LogitExitHead(hidden_dim=8, vocab_size=10)
        assert isinstance(head.proj, torch.nn.Linear)
        assert head.proj.in_features == 8
        assert head.proj.out_features == 10


# ----------------------------------------------------------------------
# 4. make_exit_kd_loss
# ----------------------------------------------------------------------

class TestMakeExitKdLoss:

    def test_mask_convention_all_invalid_raises(self):
        """kd_kl_loss_seqと同じマスク規約: 全て無効ならValueError。"""
        head = LogitExitHead(hidden_dim=8, vocab_size=6)
        loss_fn = make_exit_kd_loss([head], weights=[1.0], temperature=1.0)
        hidden = torch.rand(2, 4, 8)
        teacher_logits = torch.rand(2, 4, 6)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        try:
            loss_fn([hidden], teacher_logits, mask=mask)
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_small_when_projection_matches_teacher_exactly(self):
        """exit headの投影がteacherロジットに完全一致するよう手動設定すれば
        KLはほぼ0に近づく(温度で割ってもsoftmax(t)=softmax(t)なので厳密に0)。
        """
        hidden_dim = 5
        vocab = 6
        head = LogitExitHead(hidden_dim=hidden_dim, vocab_size=vocab)

        teacher_logits = torch.randn(3, 4, vocab)

        # projが「hiddenの最初vocab列をそのまま複製して返す」恒等写像になるよう
        # 重みを手動設定する(hidden=teacher_logits[..., :hidden_dim]と一致させ、
        # weightをidentity風に組んで teacher_logits を再現させる)。
        # 簡単のためhidden_dim==vocabの場合を使う。
        hidden_dim = vocab
        head = LogitExitHead(hidden_dim=hidden_dim, vocab_size=vocab)
        with torch.no_grad():
            head.proj.weight.copy_(torch.eye(vocab))
            head.proj.bias.zero_()

        hidden = teacher_logits.clone()
        mask = torch.ones(3, 4, dtype=torch.bool)

        loss_fn = make_exit_kd_loss([head], weights=[1.0], temperature=1.0)
        loss = loss_fn([hidden], teacher_logits, mask=mask)
        assert loss.item() < 1e-5

    def test_multi_head_weighted_sum_matches_manual_computation(self):
        """複数exit headの加重和がΣ w_l·KL_lと一致する(手動計算との突合)。"""
        vocab = 5
        head_a = LogitExitHead(hidden_dim=vocab, vocab_size=vocab)
        head_b = LogitExitHead(hidden_dim=vocab, vocab_size=vocab)
        torch.manual_seed(0)

        hidden_a = torch.randn(2, 3, vocab)
        hidden_b = torch.randn(2, 3, vocab)
        teacher_logits = torch.randn(2, 3, vocab)
        mask = torch.ones(2, 3, dtype=torch.bool)

        weights = [0.4, 0.7]
        loss_fn = make_exit_kd_loss([head_a, head_b], weights=weights, temperature=2.0)
        loss = loss_fn([hidden_a, hidden_b], teacher_logits, mask=mask)

        expected_a = kd_kl_loss_seq(head_a(hidden_a), teacher_logits, temperature=2.0, mask=mask)
        expected_b = kd_kl_loss_seq(head_b(hidden_b), teacher_logits, temperature=2.0, mask=mask)
        expected = weights[0] * expected_a + weights[1] * expected_b

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_mismatched_lengths_raise(self):
        head = LogitExitHead(hidden_dim=4, vocab_size=4)
        loss_fn = make_exit_kd_loss([head, head], weights=[1.0, 1.0], temperature=1.0)
        hidden = torch.rand(2, 3, 4)
        teacher_logits = torch.rand(2, 3, 4)
        try:
            loss_fn([hidden], teacher_logits)  # hidden_statesが1個しかない
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_constructor_rejects_mismatched_weights(self):
        head = LogitExitHead(hidden_dim=4, vocab_size=4)
        try:
            make_exit_kd_loss([head, head], weights=[1.0])
            raised = False
        except ValueError:
            raised = True
        assert raised


# ----------------------------------------------------------------------
# 5. exit有無でreturn_hidden=False経路が同一(モデルへの外付けが既存経路に影響しない)
# ----------------------------------------------------------------------

class TestExitHeadDoesNotAffectDefaultPath:

    def test_rddlgn_default_forward_unaffected_by_exit_head_presence(self):
        """exit headをモデルの外に構築しても(モデル自体は無改造なので)
        return_hidden=False経路の出力は変化しない。"""
        m = _rddlgn_model(seed=5)
        m.eval()
        src = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))
        tgt = torch.randint(0, m.config.vocab_size, (2, m.config.seq_len))

        logits_before, reg_before = m(src, tgt)

        # exit head構築(モデルには一切登録しない、独立モジュール)
        context_dim = m.config.k_layer_sizes[-1]
        _head = LogitExitHead(hidden_dim=context_dim, vocab_size=m.config.vocab_size)

        logits_after, reg_after = m(src, tgt)
        assert torch.equal(logits_before, logits_after)
        assert torch.equal(reg_before, reg_after)

    def test_logicgpt_default_forward_unaffected_by_exit_head_presence(self):
        m = _logicgpt_model(seed=6)
        m.eval()
        tok = torch.randint(0, m.config.vocab_size, (2, 10))

        logits_before, reg_before = m(tok)
        _head = LogitExitHead(hidden_dim=m.config.d_model, vocab_size=m.config.vocab_size)
        logits_after, reg_after = m(tok)

        assert torch.equal(logits_before, logits_after)
        assert torch.equal(reg_before, reg_after)
