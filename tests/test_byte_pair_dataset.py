"""BytePairDataset(翻訳DLGN)の整形契約を検証(オフライン)。"""

import os
import sys

import torch

from griffith_logic.convert.byte_vocab import BYTE_BOS_ID, BYTE_EOS_ID, BYTE_PAD_ID

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
from data.byte_pair_corpus import BytePairDataset, build_pair, collate_pairs  # noqa: E402


def test_build_pair_shapes_and_shift():
    src, tgt_input, tgt_target = build_pair([1, 2, 3], [72, 73], src_len=5, tgt_len=4)
    assert src.tolist() == [1, 2, 3, BYTE_PAD_ID, BYTE_PAD_ID]
    # tgt_target = [72,73,EOS,PAD]
    assert tgt_target.tolist() == [72, 73, BYTE_EOS_ID, BYTE_PAD_ID]
    # tgt_input = [BOS] + tgt_target[:-1]
    assert tgt_input.tolist() == [BYTE_BOS_ID, 72, 73, BYTE_EOS_ID]
    # 位置iのtgt_inputはtgt_target[i-1]を、次にtgt_target[i]を予測する整列
    assert tgt_input[1:].tolist() == tgt_target[:-1].tolist()


def test_build_pair_truncation_keeps_room_for_eos():
    # tgt_len=3 なので body は 2 まで、最後は必ず EOS。
    _, tgt_input, tgt_target = build_pair([1], [10, 11, 12, 13], src_len=2, tgt_len=3)
    assert tgt_target.tolist() == [10, 11, BYTE_EOS_ID]
    assert tgt_input.tolist() == [BYTE_BOS_ID, 10, 11]


def test_dataset_roundtrip(tmp_path):
    p = tmp_path / 'toy.pt'
    torch.save({'src': [[1, 2], [3]], 'tgt': [[65], [66, 67]], 'meta': {'k': 1}}, p)
    ds = BytePairDataset(str(p), src_len=3, tgt_len=3)
    assert len(ds) == 2
    src, tgt_input, tgt_target = ds[1]
    assert src.tolist() == [3, BYTE_PAD_ID, BYTE_PAD_ID]
    assert tgt_target.tolist() == [66, 67, BYTE_EOS_ID]
    batch = collate_pairs([ds[0], ds[1]])
    assert batch[0].shape == (2, 3) and batch[2].shape == (2, 3)
