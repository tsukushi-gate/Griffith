"""LogicGPT: decoder-only 因果Transformer char-LM生徒のモデルレベルテスト。

因果性(位置pのトークン変更が位置<pのlogitsに影響しない)がこのモデルの
存在意義なので、それを最重要ゲートとして検証する。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import LogicGPT, LogicGPTConfig


def _model(seed=0):
    torch.manual_seed(seed)
    return LogicGPT(LogicGPTConfig.tiny())


def test_forward_shape_and_reg():
    m = _model()
    m.train()
    tok = torch.randint(0, m.config.vocab_size, (3, 16))
    logits, reg = m(tok)
    assert logits.shape == (3, 16, m.config.vocab_size)
    assert torch.isfinite(reg)


def test_gradient_flows_to_embedding():
    m = _model()
    m.train()
    tok = torch.randint(0, m.config.vocab_size, (2, 12))
    logits, reg = m(tok)
    loss = logits.pow(2).mean() + reg
    loss.backward()
    grad = m.embedding.embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_causality_late_token_does_not_change_earlier_logits():
    """位置pのトークンを変えても位置<pのlogitsは厳密に不変(因果性の定義)。

    因果マスキングはscoresへの厳密なmasked_fill(構造的マスク)であり、
    数値的な近似要素を含まないため、位置<pのlogitsはbit-exactに一致するはず
    (実測でtorch.equalが成立することを確認済み。ゆえにallcloseへ緩めない)。
    """
    m = _model()
    m.eval()  # hard(Heaviside/argmax)経路でも因果性が成り立つこと
    tok = torch.randint(0, m.config.vocab_size, (2, 16))
    p = 9
    tok2 = tok.clone()
    tok2[:, p] = (tok[:, p] + 5) % m.config.vocab_size
    l1, _ = m(tok)
    l2, _ = m(tok2)
    # 位置<pは完全一致、位置pは変化する(トークンが実際に効いている確認)
    assert torch.equal(l1[:, :p], l2[:, :p])
    assert (l1[:, p] - l2[:, p]).abs().max() > 1e-4


def test_causality_soft_mode():
    """soft(訓練)モードでも因果性を保つ(bit-exact、hardモードと同じ根拠)。"""
    m = _model()
    m.train()
    tok = torch.randint(0, m.config.vocab_size, (2, 14))
    p = 7
    tok2 = tok.clone()
    tok2[:, p] = (tok[:, p] + 1) % m.config.vocab_size
    with torch.no_grad():
        l1, _ = m(tok)
        l2, _ = m(tok2)
    assert torch.equal(l1[:, :p], l2[:, :p])


def test_count_parameters_breakdown_and_total():
    m = _model()
    counts = m.count_parameters()
    assert set(counts) == {'embedding', 'blocks', 'head', 'total'}
    assert counts['total'] == sum(p.numel() for p in m.parameters())
    assert counts['total'] > 0


def test_from_paper_config_lands_near_1_2m_params():
    """RDDLGN論文×0.05スケール相当(~1.2M)に較正済みであることの回帰チェック。"""
    cfg = LogicGPTConfig.from_paper()
    m = LogicGPT(cfg)
    total = m.count_parameters()['total']
    assert 1_000_000 <= total <= 1_500_000, total
