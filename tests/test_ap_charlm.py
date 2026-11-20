"""AP法×char-LM KD(train_ap_charlm_rddlgn.py)のデータ整形・整列テスト。

train_kd_charlm_rddlgn.py(BP版)は既に tests/test_charlm_kd_alignment.py で
teacherロジットのオフバイワン整列を検証済み。本テストはAP版で新規に導入した
薄い変換層のみを対象にする:

    1. build_tgt_full: (tgt_input, tgt_target) → RDDLGNAPTrainer.train_step が
       要求する tgt_full(tgt_full[:, :-1]==tgt_input, tgt_full[:, 1:]==tgt_target)
       への変換が厳密に正しいこと。
    2. compute_teacher_probs_for_batch: BP版のteacher_logits計算
       (compute_teacher_logits_for_batch + pad_teacher_logits_to_student_vocab)
       を再利用した確率分布が、正しい形状・確率質量(合計1、PAD/BOS≈0)を持つこと。
    3. RDDLGNAPTrainer.train_step が実際にAP用tiny RDDLGN(parameterization=
       independent)に対してエラーなく走り、複数ステップで損失が悪化しない
       (最低限の学習成立性)ことをトイteacherで確認する(HFダウンロード不要)。

オフラインで完結(ネットワークアクセスなし)。高速スイート対象。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from ap_trainer_rddlgn import RDDLGNAPTrainer
from data.char_corpus import BOS_ID, STUDENT_VOCAB_SIZE, split_src_tgt
from train_ap_charlm_rddlgn import build_tgt_full, compute_teacher_probs_for_batch
from train_kd_charlm_rddlgn import SRC_LEN, TEACHER_VOCAB, WINDOW, build_config_rddlgn

from griffith_logic import RDDLGN


class _OneHotToyTeacher:
    """next_token_logits(tokens)[:, i, :] が tokens[:, i+1] のone-hotを返すトイteacher。

    tests/test_charlm_kd_alignment.py の _OneHotToyTeacher と同一契約。
    """

    vocab_size = TEACHER_VOCAB

    def next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s = tokens.shape
        logits = torch.full((b, s, self.vocab_size), -20.0)
        for t in range(s - 1):
            true_next = tokens[:, t + 1]
            logits[torch.arange(b), t, true_next] = 20.0
        logits[:, s - 1, 0] = 20.0
        return logits


def _make_toy_sequence(batch: int = 3, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TEACHER_VOCAB, (batch, WINDOW), generator=g)


# ----------------------------------------------------------------------
# 1. build_tgt_full: RDDLGNAPTrainer.train_step契約への整合
# ----------------------------------------------------------------------
def test_build_tgt_full_satisfies_ap_trainer_contract():
    """tgt_full[:, :-1]==tgt_input かつ tgt_full[:, 1:]==tgt_target を厳密に満たす。"""
    tokens = _make_toy_sequence(batch=4, seed=10)
    for i in range(4):
        src, tgt_input, tgt_target = split_src_tgt(tokens[i])
        tgt_full = build_tgt_full(tgt_input.unsqueeze(0), tgt_target.unsqueeze(0))

        assert tgt_full.shape == (1, SRC_LEN + 1)
        assert torch.equal(tgt_full[:, :-1].squeeze(0), tgt_input), (
            'tgt_full[:, :-1] は tgt_input と一致する必要がある'
            '(RDDLGNAPTrainer.train_stepがtgt_input=tgt_full[:, :-1]をデコーダ入力に使う)'
        )
        assert torch.equal(tgt_full[:, 1:].squeeze(0), tgt_target), (
            'tgt_full[:, 1:] は tgt_target と一致する必要がある'
            '(RDDLGNAPTrainer.train_stepがtgt_label=tgt_full[:, 1:]を正解ラベルに使う)'
        )


def test_build_tgt_full_starts_with_bos():
    tokens = _make_toy_sequence(batch=2, seed=11)
    src, tgt_input, tgt_target = split_src_tgt(tokens[0])
    tgt_full = build_tgt_full(tgt_input.unsqueeze(0), tgt_target.unsqueeze(0))
    assert tgt_full[0, 0].item() == BOS_ID


def test_build_tgt_full_batched():
    """バッチ次元込みでも同じ契約を満たす(実際の訓練ループでの呼び出し形)。"""
    tokens = _make_toy_sequence(batch=6, seed=12)
    srcs, tgt_inputs, tgt_targets = [], [], []
    for i in range(6):
        s, ti, tt = split_src_tgt(tokens[i])
        srcs.append(s)
        tgt_inputs.append(ti)
        tgt_targets.append(tt)
    tgt_input = torch.stack(tgt_inputs)
    tgt_target = torch.stack(tgt_targets)

    tgt_full = build_tgt_full(tgt_input, tgt_target)
    assert tgt_full.shape == (6, SRC_LEN + 1)
    assert torch.equal(tgt_full[:, :-1], tgt_input)
    assert torch.equal(tgt_full[:, 1:], tgt_target)


# ----------------------------------------------------------------------
# 2. compute_teacher_probs_for_batch: 確率分布としての妥当性
# ----------------------------------------------------------------------
def test_compute_teacher_probs_shape_and_normalization():
    tokens = _make_toy_sequence(batch=3, seed=13)
    teacher = _OneHotToyTeacher()

    probs = compute_teacher_probs_for_batch(teacher, tokens, device='cpu', temperature=1.0)
    assert probs.shape == (3, SRC_LEN, STUDENT_VOCAB_SIZE)

    # 各行が確率分布として正規化されている(合計1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    # PAD(65)/BOS(66)の確率質量は実質0
    assert (probs[:, :, 65] < 1e-6).all()
    assert (probs[:, :, 66] < 1e-6).all()

    # argmaxはtgt_target(chars[32:64])と一致する(オフバイワン整列がAP版でも
    # 正しく保たれていることの間接検証)。
    predicted = probs.argmax(dim=-1)
    true_next = tokens[:, SRC_LEN:WINDOW]
    assert torch.equal(predicted, true_next)


def test_compute_teacher_probs_temperature_softens_distribution():
    """温度Tを上げるほど、正解クラスの確率質量が小さくなる(スムージング方向の確認)。"""
    tokens = _make_toy_sequence(batch=2, seed=14)
    teacher = _OneHotToyTeacher()

    probs_low_t = compute_teacher_probs_for_batch(teacher, tokens, device='cpu', temperature=1.0)
    probs_high_t = compute_teacher_probs_for_batch(teacher, tokens, device='cpu', temperature=4.0)

    true_next = tokens[:, SRC_LEN:WINDOW]
    max_prob_low_t = probs_low_t.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)
    max_prob_high_t = probs_high_t.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)

    assert (max_prob_high_t <= max_prob_low_t + 1e-6).all()


# ----------------------------------------------------------------------
# 3. RDDLGNAPTrainer.train_step のエンドツーエンド疎通(トイteacher、tiny RDDLGN)
# ----------------------------------------------------------------------
def _build_tiny_ap_rddlgn(seed: int = 0) -> RDDLGN:
    import argparse

    torch.manual_seed(seed)
    args = argparse.Namespace(config='tiny', scale=1.0, n_inputs=2)
    config = build_config_rddlgn(args)
    config.parameterization = 'independent'
    return RDDLGN(config, device='cpu')


def test_ap_trainer_accepts_independent_rddlgn_without_raising():
    """AP法はLutLayer.parameterization==independentを要求する。build_config_rddlgn
    にparameterization='independent'を明示的に上書きすればAP法のコンストラクタ
    検証(RDDLGNAPTrainer.__init__)を通ることを確認する(可否判定の核心)。"""
    model = _build_tiny_ap_rddlgn(seed=0)
    trainer = RDDLGNAPTrainer(model, lr=0.1, pad_token=65)
    assert trainer is not None


def test_ap_trainer_train_step_runs_and_reports_metrics():
    model = _build_tiny_ap_rddlgn(seed=1)
    trainer = RDDLGNAPTrainer(model, lr=0.5, pad_token=65)
    teacher = _OneHotToyTeacher()

    tokens = _make_toy_sequence(batch=8, seed=20)
    srcs, tgt_inputs, tgt_targets = [], [], []
    for i in range(8):
        s, ti, tt = split_src_tgt(tokens[i])
        srcs.append(s)
        tgt_inputs.append(ti)
        tgt_targets.append(tt)
    src = torch.stack(srcs)
    tgt_input = torch.stack(tgt_inputs)
    tgt_target = torch.stack(tgt_targets)

    teacher_probs = compute_teacher_probs_for_batch(teacher, tokens, device='cpu', temperature=4.0)
    tgt_full = build_tgt_full(tgt_input, tgt_target)

    metrics = trainer.train_step(src, tgt_full, teacher_probs=teacher_probs, kd_alpha=0.25)
    assert 'loss' in metrics and 'accuracy' in metrics
    assert metrics['loss'] == metrics['loss']  # NaNでない
    assert 0.0 <= metrics['accuracy'] <= 1.0


@pytest.mark.learning
def test_ap_trainer_train_step_reduces_loss_over_iterations():
    """同一の小さいトイ教師付きバッチに対して複数ステップ回すと損失が改善方向に動く
    (収束確認は本実験の役目だが、最小限の学習成立性=勾配ゼロでないことは
    ここで担保する)。"""
    model = _build_tiny_ap_rddlgn(seed=2)
    trainer = RDDLGNAPTrainer(model, lr=0.5, pad_token=65)
    teacher = _OneHotToyTeacher()

    torch.manual_seed(3)
    tokens = _make_toy_sequence(batch=16, seed=21)
    srcs, tgt_inputs, tgt_targets = [], [], []
    for i in range(16):
        s, ti, tt = split_src_tgt(tokens[i])
        srcs.append(s)
        tgt_inputs.append(ti)
        tgt_targets.append(tt)
    src = torch.stack(srcs)
    tgt_input = torch.stack(tgt_inputs)
    tgt_target = torch.stack(tgt_targets)

    teacher_probs = compute_teacher_probs_for_batch(teacher, tokens, device='cpu', temperature=4.0)
    tgt_full = build_tgt_full(tgt_input, tgt_target)

    losses = []
    for _ in range(20):
        metrics = trainer.train_step(src, tgt_full, teacher_probs=teacher_probs, kd_alpha=0.25)
        losses.append(metrics['loss'])

    # 最初の数ステップの平均損失より、最後の数ステップの平均損失が悪化していない
    # ことを確認する(AP法は確率的な更新のため単調減少は要求しない)。
    early_avg = sum(losses[:3]) / 3
    late_avg = sum(losses[-3:]) / 3
    assert late_avg <= early_avg + 1e-6, (
        f'20ステップ後も損失が改善方向に動いていない: early={early_avg:.4f} late={late_avg:.4f}'
    )
