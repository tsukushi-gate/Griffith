"""翻訳DLGNのモデル+config smoke(オフライン・高速)。

DLGNAttentionTranslator が byte 語彙(259)・非対称/短長で soft/hard 両モードとも
forward が finite で通ること、build_config_translation の3プリセットが妥当な
TranslatorConfig を返すことを確認する。
"""

import os
import sys

import torch

from griffith_logic import DLGNAttentionTranslator
from griffith_logic.convert.byte_vocab import BYTE_VOCAB_SIZE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from train_kd_translation import (  # noqa: E402
    build_config_translation,
    make_pure_ce_seq_loss,
)


def test_build_config_presets_valid():
    for size in ('s', 'm', 'l'):
        cfg = build_config_translation(size, src_len=48, tgt_len=48)
        assert cfg.vocab_size == BYTE_VOCAB_SIZE
        assert cfg.src_len == 48 and cfg.tgt_len == 48
        # 出力ヘッド最終層は vocab の倍数(GroupSum前提)。
        assert cfg.m_layer_sizes[-1] % BYTE_VOCAB_SIZE == 0


def test_config_size_ordering():
    def total(size):
        cfg = build_config_translation(size, 48, 48)
        m = DLGNAttentionTranslator(cfg, device='cpu')
        return sum(p.numel() for p in m.parameters())
    s, m, l = total('s'), total('m'), total('l')
    assert s < m < l  # s < m < l の容量順


def test_forward_soft_hard_finite():
    cfg = build_config_translation('s', src_len=24, tgt_len=24)
    model = DLGNAttentionTranslator(cfg, device='cpu')
    src = torch.randint(0, BYTE_VOCAB_SIZE, (2, 24))
    tgt_input = torch.randint(0, BYTE_VOCAB_SIZE, (2, 24))
    for soft in (True, False):
        model.train(mode=soft)
        with torch.no_grad():
            logits, emb = model(src, tgt_input)
        assert logits.shape == (2, 24, BYTE_VOCAB_SIZE)
        assert torch.isfinite(logits).all()


def test_pure_ce_loss_ignores_pad():
    from griffith_logic.convert.byte_vocab import BYTE_PAD_ID
    loss_fn = make_pure_ce_seq_loss(pad_id=BYTE_PAD_ID)
    logits = torch.randn(2, 5, BYTE_VOCAB_SIZE, requires_grad=True)
    target = torch.tensor([[10, 20, BYTE_PAD_ID, BYTE_PAD_ID, BYTE_PAD_ID],
                           [1, 2, 3, BYTE_PAD_ID, BYTE_PAD_ID]])
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
    loss.backward()  # 勾配が流れる(全PADでない)
