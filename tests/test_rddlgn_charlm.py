"""RDDLGN 文字LM(attention有無比較の対照群)のテスト。

experiments/train_rddlgn_charlm.py の RDDLGNCharLM を検証する:
  - shape契約: (B,T) → (B,T,V)
  - causality: 未来トークンの改変が過去時刻の logits に影響しない(再帰=causal)
  - 収束スモーク: 単一バッチを10ステップ過学習して loss が下がる
  - パラメータ規模: LogicGPT論理部(33,349)と同程度
"""

import os
import sys

import torch
import torch.nn.functional as F

# experiments/ をパスに追加(パッケージ外スクリプト)
_EXP_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments')
sys.path.insert(0, os.path.abspath(_EXP_DIR))

from train_rddlgn_charlm import CharLMConfig, RDDLGNCharLM  # noqa: E402

DEVICE = 'cpu'
VOCAB = 65


def _tiny_config(**overrides) -> CharLMConfig:
    """テスト用の小構成(高速)。M最終層は vocab_size の倍数に保つ。"""
    base = dict(
        vocab_size=VOCAB,
        d_emb=16,
        n_layer_sizes=[32, 32],
        k_layer_sizes=[48, 48],
        m_layer_sizes=[64, VOCAB],  # 群化係数1
        tau=1.0,
    )
    base.update(overrides)
    return CharLMConfig(**base)


# ---------------------------------------------------------------------------
# shape契約
# ---------------------------------------------------------------------------

class TestShape:

    def test_forward_shape(self):
        model = RDDLGNCharLM(_tiny_config(), device=DEVICE)
        x = torch.randint(0, VOCAB, (4, 8))
        logits = model(x)
        assert logits.shape == (4, 8, VOCAB)

    def test_forward_different_seq_len(self):
        """系列長が変わっても動作する(再帰なので任意長)。"""
        model = RDDLGNCharLM(_tiny_config(), device=DEVICE)
        for t in (1, 3, 16):
            logits = model(torch.randint(0, VOCAB, (2, t)))
            assert logits.shape == (2, t, VOCAB)

    def test_gradient_flow(self):
        """全学習パラメータに勾配が伝播する。"""
        model = RDDLGNCharLM(_tiny_config(), device=DEVICE)
        model.train()
        x = torch.randint(0, VOCAB, (2, 6))
        y = torch.randint(0, VOCAB, (2, 6))
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f'{name} に勾配がない'

    def test_m_last_must_divide_vocab(self):
        """M最終層が vocab_size で割り切れないと構築時に弾く。"""
        import pytest
        with pytest.raises(ValueError):
            _tiny_config(m_layer_sizes=[64, 63])  # 63 は 65 で割り切れない


# ---------------------------------------------------------------------------
# causality(再帰構造から自動的に保証される因果性)
# ---------------------------------------------------------------------------

class TestCausality:

    def _assert_future_does_not_affect_past(self, model):
        model.eval()
        torch.manual_seed(1)
        b, t = 3, 12
        t0 = t // 2  # この時刻までの logits は未来改変に不変であるべき
        x = torch.randint(0, VOCAB, (b, t))

        with torch.no_grad():
            logits1 = model(x)
            # 未来位置(t0より後)を別トークンへ改変
            x2 = x.clone()
            while True:
                x2[:, t0 + 1:] = torch.randint(0, VOCAB, (b, t - t0 - 1))
                if not torch.equal(x2[:, t0 + 1:], x[:, t0 + 1:]):
                    break
            logits2 = model(x2)

        # 過去(0..t0)の logits は完全一致する(未来を一切参照しない)
        assert torch.equal(logits1[:, :t0 + 1], logits2[:, :t0 + 1]), \
            '未来トークンの改変が過去時刻の logits に影響した(causal性違反)'
        # 逆に、改変した未来位置の logits は変化しているはず(モデルが入力に依存)
        assert not torch.equal(logits1[:, t0 + 1:], logits2[:, t0 + 1:]), \
            '未来位置の logits が入力改変で全く変わらない(依存性の消失)'

    def test_causality_eval_mode(self):
        """eval(hard)モードで causal性を検証。"""
        model = RDDLGNCharLM(_tiny_config(), device=DEVICE)
        self._assert_future_does_not_affect_past(model)

    def test_causality_train_mode(self):
        """train(soft)モードでも causal性が保たれる。"""
        model = RDDLGNCharLM(_tiny_config(), device=DEVICE)
        model.train()
        torch.manual_seed(2)
        b, t = 2, 10
        t0 = t // 2
        x = torch.randint(0, VOCAB, (b, t))
        with torch.no_grad():
            logits1 = model(x)
            x2 = x.clone()
            x2[:, t0 + 1:] = (x2[:, t0 + 1:] + 7) % VOCAB
            logits2 = model(x2)
        assert torch.allclose(logits1[:, :t0 + 1], logits2[:, :t0 + 1], atol=1e-6)


# ---------------------------------------------------------------------------
# 収束スモーク
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_overfits_single_batch(self):
        """単一バッチを10ステップ過学習して loss が明確に下がる。"""
        torch.manual_seed(0)
        model = RDDLGNCharLM(_tiny_config(), device=DEVICE)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

        x = torch.randint(0, VOCAB, (8, 8))
        y = torch.randint(0, VOCAB, (8, 8))

        losses = []
        for _ in range(10):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # 10ステップで loss が下がっていること(過学習方向)
        assert losses[-1] < losses[0], \
            f'loss が下がっていない: {losses[0]:.4f} -> {losses[-1]:.4f}'


# ---------------------------------------------------------------------------
# パラメータ規模(LogicGPT論理部 33,349 との対比)
# ---------------------------------------------------------------------------

class TestParamBudget:

    def test_default_config_matches_logic_gpt_scale(self):
        """既定構成の総パラメータ数が LogicGPT論理部(33,349)と同程度(±5%)。"""
        model = RDDLGNCharLM(CharLMConfig(), device=DEVICE)
        total = model.count_parameters()['total']
        assert abs(total - 33349) / 33349 < 0.05, \
            f'パラメータ数 {total} が LogicGPT 33,349 と乖離しすぎ'

    def test_count_parameters_consistency(self):
        """count_parameters の total が実パラメータ総数と一致する。"""
        model = RDDLGNCharLM(_tiny_config(), device=DEVICE)
        counts = model.count_parameters()
        actual = sum(p.numel() for p in model.parameters())
        assert counts['total'] == actual
