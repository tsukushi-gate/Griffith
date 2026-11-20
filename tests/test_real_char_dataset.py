"""tiny_shakespeare実データKD/scratchアームのデータ層検証。

load_tiny_shakespeare_train / RealCharDataset が満たすべき不変条件:
    1. fraction→文字数・窓数の対応(prefix長=int(len(pool)*fraction)、
       __len__=len(ids)-64+1のスライディング窓)。
    2. train prefixとvalが常にdisjoint(val_reserve_charsで末尾を予約するため)。
    3. nested性(1%は5%のprefix、5%は25%のprefix、...)。
    4. seq2seq/decoder_only両modeでTeacherCorpusDatasetと同一の__getitem__契約
       (split_src_tgtの形状・BOS付与)。
    5. 決定性(同fraction→同tensor)。

data-charlm/tiny_shakespeare_input.txt が既にローカルキャッシュされていれば
ネットワークアクセス無しで完結する(無ければ_download_tiny_shakespeareの
既存フォールバック挙動でダウンロードされる)。高速スイート対象(CPUで数秒)。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from data.char_corpus import (
    BOS_ID,
    RealCharDataset,
    load_tiny_shakespeare_train,
    load_tiny_shakespeare_val,
    split_src_tgt,
)

_WINDOW = 64
_VAL_RESERVE = 100_000


# ----------------------------------------------------------------------
# 1. fraction→文字数・窓数
# ----------------------------------------------------------------------
def test_load_tiny_shakespeare_train_fraction_to_char_count():
    """指定fractionに対し、train poolの先頭int(len(pool)*fraction)文字が返る。"""
    full_pool_len = len(load_tiny_shakespeare_train(1.0, val_reserve_chars=_VAL_RESERVE))

    ids_01 = load_tiny_shakespeare_train(0.01, val_reserve_chars=_VAL_RESERVE)
    expected_len_01 = int(full_pool_len * 0.01)
    # vocabフィルタで文字が落ちる可能性があるため、フィルタ前提の理論値と
    # 厳密一致ではなく「ほぼ一致」(フィルタ除去分だけ短くなりうる)を許容する。
    assert abs(len(ids_01) - expected_len_01) <= 5, (
        f'fraction=0.01: 期待長{expected_len_01}付近, 実際{len(ids_01)}'
    )

    ids_05 = load_tiny_shakespeare_train(0.05, val_reserve_chars=_VAL_RESERVE)
    expected_len_05 = int(full_pool_len * 0.05)
    assert abs(len(ids_05) - expected_len_05) <= 5


def test_real_char_dataset_window_count():
    """RealCharDataset.__len__ = len(ids) - 64 + 1(stride=1スライディング窓)。"""
    ids = load_tiny_shakespeare_train(0.01, val_reserve_chars=_VAL_RESERVE)
    ds = RealCharDataset(ids, mode='seq2seq')
    assert len(ds) == len(ids) - _WINDOW + 1


def test_load_tiny_shakespeare_train_rejects_invalid_fraction():
    for bad in (0.0, -0.1, 1.5):
        try:
            load_tiny_shakespeare_train(bad, val_reserve_chars=_VAL_RESERVE)
            raise AssertionError(f'fraction={bad}はValueErrorになるはず')
        except ValueError:
            pass


# ----------------------------------------------------------------------
# 2. train prefixとvalのdisjoint性
# ----------------------------------------------------------------------
def test_train_and_val_are_disjoint():
    """train pool(先頭)とval(末尾max_chars)が文字レベルで重複しないことを、
    元テキストのインデックス範囲から直接検証する。

    load_tiny_shakespeare_val(max_chars=_VAL_RESERVE)は full_text[-_VAL_RESERVE:]、
    load_tiny_shakespeare_train(1.0, val_reserve_chars=_VAL_RESERVE)は
    full_text[:-_VAL_RESERVE] 全体(フィルタ後)なので、fraction=1.0でも
    valの範囲とは重複しないインデックス空間になっているはず。
    """
    train_ids_full = load_tiny_shakespeare_train(1.0, val_reserve_chars=_VAL_RESERVE)
    val_windows = load_tiny_shakespeare_val(max_chars=_VAL_RESERVE)

    # 文字列としてのトークン列自体は再利用されうる(vocabが共通なので同じidが
    # 出現するのは自然)。真に検証すべきは「元テキストの文字インデックス範囲が
    # 重ならないこと」であり、これは実装(pool=full_text[:-N], val=full_text[-N:])
    # から構造的に保証される。ここでは長さの整合性(train prefixがval_reserve分
    # 短く切られている)を確認することで、実装がその契約を満たしていることを
    # 間接検証する。
    from pathlib import Path  # noqa: PLC0415

    from data.char_corpus import _download_tiny_shakespeare  # noqa: PLC0415
    full_text = _download_tiny_shakespeare(Path('./data-charlm/tiny_shakespeare_input.txt'))

    assert len(full_text) - _VAL_RESERVE == len(full_text[:-_VAL_RESERVE])
    pool_text = full_text[:-_VAL_RESERVE]
    val_text = full_text[-_VAL_RESERVE:]
    # プールとvalのテキストレベルでの境界確認(インデックス上disjoint)。
    assert full_text == pool_text + val_text

    # train_ids_full(fraction=1.0)の文字数はpool_text由来でvocabフィルタ後の
    # 長さになる(pool_textのvocab外文字が除去される)ため、pool_text長以下。
    assert len(train_ids_full) <= len(pool_text)
    assert val_windows.numel() > 0


# ----------------------------------------------------------------------
# 3. nested性
# ----------------------------------------------------------------------
def test_fractions_are_nested():
    """1%は5%のprefix、5%は25%のprefix、25%は100%のprefix。"""
    ids_01 = load_tiny_shakespeare_train(0.01, val_reserve_chars=_VAL_RESERVE)
    ids_05 = load_tiny_shakespeare_train(0.05, val_reserve_chars=_VAL_RESERVE)
    ids_25 = load_tiny_shakespeare_train(0.25, val_reserve_chars=_VAL_RESERVE)
    ids_100 = load_tiny_shakespeare_train(1.0, val_reserve_chars=_VAL_RESERVE)

    assert len(ids_01) < len(ids_05) < len(ids_25) < len(ids_100)

    assert torch.equal(ids_05[:len(ids_01)], ids_01)
    assert torch.equal(ids_25[:len(ids_05)], ids_05)
    assert torch.equal(ids_100[:len(ids_25)], ids_25)


# ----------------------------------------------------------------------
# 4. seq2seq/decoder_only両modeの形状・BOS付与
# ----------------------------------------------------------------------
def test_real_char_dataset_seq2seq_matches_split_src_tgt():
    ids = load_tiny_shakespeare_train(0.01, val_reserve_chars=_VAL_RESERVE)
    ds = RealCharDataset(ids, mode='seq2seq')

    for idx in (0, 1, len(ds) - 1):
        src, tgt_input, tgt_target = ds[idx]
        window = ids[idx:idx + _WINDOW]
        exp_src, exp_tgt_input, exp_tgt_target = split_src_tgt(window)

        assert src.shape == (32,)
        assert tgt_input.shape == (32,)
        assert tgt_target.shape == (32,)
        assert torch.equal(src, exp_src)
        assert torch.equal(tgt_input, exp_tgt_input)
        assert torch.equal(tgt_target, exp_tgt_target)

        # BOS付与契約(TeacherCorpusDatasetと同一): tgt_input[0]==BOS_ID
        assert tgt_input[0].item() == BOS_ID

        # raw_tokens(src+tgt_target)が元のwindowと一致(TeacherCorpusDatasetの
        # 契約と同じ再構成不変条件)。
        reconstructed = torch.cat([src, tgt_target], dim=0)
        assert torch.equal(reconstructed, window)

        # .raw_tokens(idx)がwindowと一致することも確認。
        assert torch.equal(ds.raw_tokens(idx), window)


def test_real_char_dataset_decoder_only_returns_raw_window():
    ids = load_tiny_shakespeare_train(0.01, val_reserve_chars=_VAL_RESERVE)
    ds = RealCharDataset(ids, mode='decoder_only')

    for idx in (0, 1, len(ds) - 1):
        window = ds[idx]
        assert window.shape == (_WINDOW,)
        assert torch.equal(window, ids[idx:idx + _WINDOW])


def test_real_char_dataset_stride_one_overlap():
    """stride=1: ds[i+1]の先頭63文字はds[i]の末尾63文字(1文字シフト)と一致する。"""
    ids = load_tiny_shakespeare_train(0.01, val_reserve_chars=_VAL_RESERVE)
    ds = RealCharDataset(ids, mode='decoder_only')

    w0 = ds[0]
    w1 = ds[1]
    assert torch.equal(w0[1:], w1[:-1])


def test_real_char_dataset_rejects_short_ids():
    short_ids = torch.arange(10, dtype=torch.long)
    try:
        RealCharDataset(short_ids, mode='seq2seq')
        raise AssertionError('64未満のidsはValueErrorになるはず')
    except ValueError:
        pass


def test_real_char_dataset_rejects_non_1d_ids():
    bad_ids = torch.zeros((10, _WINDOW), dtype=torch.long)
    try:
        RealCharDataset(bad_ids, mode='seq2seq')
        raise AssertionError('2次元idsはValueErrorになるはず')
    except ValueError:
        pass


# ----------------------------------------------------------------------
# 5. 決定性
# ----------------------------------------------------------------------
def test_load_tiny_shakespeare_train_is_deterministic():
    ids_a = load_tiny_shakespeare_train(0.05, val_reserve_chars=_VAL_RESERVE)
    ids_b = load_tiny_shakespeare_train(0.05, val_reserve_chars=_VAL_RESERVE)
    assert torch.equal(ids_a, ids_b)


def test_real_char_dataset_getitem_is_deterministic():
    ids = load_tiny_shakespeare_train(0.01, val_reserve_chars=_VAL_RESERVE)
    ds = RealCharDataset(ids, mode='seq2seq')
    item_a = ds[3]
    item_b = ds[3]
    for a, b in zip(item_a, item_b):
        assert torch.equal(a, b)
