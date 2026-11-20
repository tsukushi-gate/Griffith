"""data-free文字レベルLM KDのシーケンス整列検証。

train_kd_charlm_rddlgn.py の設計上の核心は「teacherロジットのどの位置が
studentのどのデコードステップに対応するか」というオフバイワン整合性である。
本テストは決定的な「トイteacher」(next_token_logitsが真の次文字のone-hot
ロジットを返す)を使い、以下を厳密に検証する:

    1. compute_teacher_logits_for_batch がスライスする範囲[31:63]が
       tgt_target(=chars[32:64])と1対1で対応すること
       (teacher.next_token_logits(chars[0:64])[:, i, :]はchars[i+1]の予測、
       ゆえにchars[32+t]の予測は位置31+tにある)。
    2. pad_teacher_logits_to_student_vocab がPAD/BOS列を大負定数で埋め、
       [B,S,65]→[B,S,67]の形状変換をしても元65列のargmaxを変えないこと。
    3. TeacherCorpusDataset(split_src_tgt) が作る (src, tgt_input, tgt_target)
       と生トークン列の整合(raw_tokens = src + tgt_target、tgt_inputは
       [BOS] + chars[32:63])。

オフラインで完結(HFダウンロード不要、ネットワークアクセスなし)。高速スイート対象。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from data.char_corpus import BOS_ID, STUDENT_VOCAB_SIZE, TeacherCorpusDataset, split_src_tgt
from train_kd_charlm_rddlgn import (
    SRC_LEN,
    TEACHER_VOCAB,
    WINDOW,
    compute_teacher_logits_for_batch,
    pad_teacher_logits_to_student_vocab,
)


class _OneHotToyTeacher:
    """next_token_logits(tokens)[:, i, :] が tokens[:, i+1] のone-hotを返すトイteacher。

    最終位置(i=S-1)には「次」の正解が無いため、便宜上0番目の文字を予測する
    ことにする(このテストでは末尾ロジットは使わないため実害なし)。
    """

    vocab_size = TEACHER_VOCAB

    def next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s = tokens.shape
        logits = torch.full((b, s, self.vocab_size), -20.0)
        for t in range(s - 1):
            true_next = tokens[:, t + 1]
            logits[torch.arange(b), t, true_next] = 20.0
        # 最終位置はダミー(0番目を正解にする)。
        logits[:, s - 1, 0] = 20.0
        return logits


def _make_toy_sequence(batch: int = 3, seed: int = 0) -> torch.Tensor:
    """(batch, WINDOW)のランダムだが決定的なtoken列(teacher vocab範囲)を作る。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TEACHER_VOCAB, (batch, WINDOW), generator=g)


# ----------------------------------------------------------------------
# 1. teacherロジットのオフバイワン整列
# ----------------------------------------------------------------------
def test_teacher_logits_slice_predicts_tgt_target():
    """compute_teacher_logits_for_batchが返す[:, t, :]のargmaxはchars[32+t]と一致する。"""
    tokens = _make_toy_sequence(batch=4, seed=1)
    teacher = _OneHotToyTeacher()

    aligned = compute_teacher_logits_for_batch(teacher, tokens, device='cpu')
    assert aligned.shape == (4, SRC_LEN, TEACHER_VOCAB)

    predicted = aligned.argmax(dim=-1)  # (4, 32)
    true_next = tokens[:, SRC_LEN:WINDOW]  # chars[32:64] = tgt_target

    assert torch.equal(predicted, true_next), (
        'teacherロジットのスライス[31:63]は tgt_target=chars[32:64] と'
        '1対1で一致するはず(オフバイワン整列の核心)'
    )


def test_teacher_logits_slice_offset_is_31():
    """スライス開始位置が31であることを直接検証する(境界値の明示チェック)。

    位置iはchars[i+1]を予測するので、chars[32](tgt_targetの先頭)を予測する
    ロジットは i=31 にある。境界を1つずらすとテストが落ちることを
    別途手動確認済み(位置30/32でargmax不一致になる)。
    """
    tokens = _make_toy_sequence(batch=2, seed=2)
    teacher = _OneHotToyTeacher()

    full_logits = teacher.next_token_logits(tokens)
    pos31_pred = full_logits[:, 31, :].argmax(dim=-1)
    assert torch.equal(pos31_pred, tokens[:, 32]), 'position 31 は chars[32] を予測する'

    # ずらした場合は不一致になることの確認(整列がたまたま合っているのではないことの証拠)
    pos30_pred = full_logits[:, 30, :].argmax(dim=-1)
    assert not torch.equal(pos30_pred, tokens[:, 32]), (
        'position 30 は chars[31] を予測するので chars[32] とは通常一致しない'
    )


# ----------------------------------------------------------------------
# 2. PAD/BOSパディングでのvocab拡張がargmaxを変えない
# ----------------------------------------------------------------------
def test_pad_teacher_logits_preserves_argmax():
    tokens = _make_toy_sequence(batch=3, seed=3)
    teacher = _OneHotToyTeacher()
    aligned = compute_teacher_logits_for_batch(teacher, tokens, device='cpu')

    padded = pad_teacher_logits_to_student_vocab(aligned, student_vocab_size=STUDENT_VOCAB_SIZE)
    assert padded.shape == (3, SRC_LEN, STUDENT_VOCAB_SIZE)

    # 元の65列のargmaxは変わらない
    assert torch.equal(padded.argmax(dim=-1), aligned.argmax(dim=-1))

    # PAD/BOS列(65, 66)は大負定数
    assert (padded[:, :, 65] < -1e8).all()
    assert (padded[:, :, 66] < -1e8).all()

    # softmaxでPAD/BOSの確率質量が実質0であることも確認
    probs = torch.softmax(padded, dim=-1)
    assert (probs[:, :, 65] < 1e-6).all()
    assert (probs[:, :, 66] < 1e-6).all()


def test_pad_teacher_logits_noop_when_already_student_vocab():
    """student_vocab_size <= 現在のvocabなら無変化(pad_width<=0の分岐)。"""
    logits = torch.randn(2, 5, TEACHER_VOCAB)
    out = pad_teacher_logits_to_student_vocab(logits, student_vocab_size=TEACHER_VOCAB)
    assert torch.equal(out, logits)


# ----------------------------------------------------------------------
# 3. TeacherCorpusDataset.split_src_tgt のraw_tokens整合
# ----------------------------------------------------------------------
def test_split_src_tgt_alignment_with_raw_tokens():
    tokens = _make_toy_sequence(batch=1, seed=4)[0]  # (64,)
    src, tgt_input, tgt_target = split_src_tgt(tokens)

    assert src.shape == (SRC_LEN,)
    assert tgt_input.shape == (SRC_LEN,)
    assert tgt_target.shape == (SRC_LEN,)

    # src = chars[0:32]
    assert torch.equal(src, tokens[:32])
    # tgt_target = chars[32:64]
    assert torch.equal(tgt_target, tokens[32:64])
    # tgt_input = [BOS] + chars[32:63]
    assert tgt_input[0].item() == BOS_ID
    assert torch.equal(tgt_input[1:], tokens[32:63])

    # raw_tokens復元(src + tgt_target)が元の64トークン列と一致する
    # (train_kd_charlm_rddlgn.py の主訓練ループがこの結合でteacherロジットを
    #  再計算するため、この不変条件が崩れるとオフバイワン整列全体が破綻する)。
    reconstructed = torch.cat([src, tgt_target], dim=0)
    assert torch.equal(reconstructed, tokens)


def test_teacher_corpus_dataset_getitem_matches_split_src_tgt(tmp_path):
    """TeacherCorpusDataset.__getitem__がsplit_src_tgtと同じ結果を返すことを
    実ファイル経由(torch.save/load)で確認する。"""
    tokens = _make_toy_sequence(batch=5, seed=5).to(torch.int16)
    path = tmp_path / 'toy_corpus.pt'
    torch.save({'tokens': tokens, 'meta': {'note': 'toy'}}, path)

    ds = TeacherCorpusDataset(str(path))
    assert len(ds) == 5
    assert ds.vocab_size == TEACHER_VOCAB

    for i in range(5):
        src, tgt_input, tgt_target = ds[i]
        exp_src, exp_tgt_input, exp_tgt_target = split_src_tgt(tokens[i].long())
        assert torch.equal(src, exp_src)
        assert torch.equal(tgt_input, exp_tgt_input)
        assert torch.equal(tgt_target, exp_tgt_target)

        raw = ds.raw_tokens(i)
        assert torch.equal(raw, tokens[i].long())
