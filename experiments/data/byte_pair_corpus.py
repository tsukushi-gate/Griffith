"""byte-level 対訳コーパス(翻訳DLGN)の Dataset。

gen_translation_corpus.py が保存する `{'src': list[list[int]], 'tgt': list[list[int]],
'meta': {...}}`(いずれも生UTF-8バイト列、特殊トークン無し)を読み、seq2seq 訓練契約
`(src[src_len], tgt_input[tgt_len], tgt_target[tgt_len])` へ整形する。

    tgt_target = tgt bytes + EOS を tgt_len に pad/truncate
    tgt_input  = [BOS] + tgt_target[:-1]           (teacher forcing、位置iはtgt_target[i]を予測)
    src        = src bytes を src_len に pad/truncate

vocab は byte_vocab の単一情報源(BYTE_VOCAB_SIZE=259)。PAD 位置は損失の ignore_index。
char_corpus.py の split_src_tgt を「src/tgt が別系列・非対称長」へ一般化した版。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from griffith_logic.convert.byte_vocab import (
    BYTE_BOS_ID,
    BYTE_EOS_ID,
    BYTE_PAD_ID,
)


def _pad_trunc(seq: list[int], length: int, pad_id: int) -> list[int]:
    seq = list(seq[:length])
    return seq + [pad_id] * (length - len(seq))


def build_pair(
    src_bytes: list[int], tgt_bytes: list[int], src_len: int, tgt_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1対の byte列 -> (src, tgt_input, tgt_target)。テスト・生成側からも再利用可能。"""
    src = _pad_trunc(list(src_bytes), src_len, BYTE_PAD_ID)
    # EOS 用に1つ空けて truncate してから EOS を付す。
    tgt_body = list(tgt_bytes[:tgt_len - 1])
    tgt_target = _pad_trunc(tgt_body + [BYTE_EOS_ID], tgt_len, BYTE_PAD_ID)
    tgt_input = [BYTE_BOS_ID] + tgt_target[:-1]
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(tgt_input, dtype=torch.long),
        torch.tensor(tgt_target, dtype=torch.long),
    )


class BytePairDataset(Dataset):
    """byte対訳 .pt を読み seq2seq 3-tuple を返す Dataset。"""

    def __init__(self, path: str, src_len: int, tgt_len: int):
        data = torch.load(path, weights_only=False)
        self.src: list[list[int]] = [list(map(int, s)) for s in data['src']]
        self.tgt: list[list[int]] = [list(map(int, t)) for t in data['tgt']]
        if len(self.src) != len(self.tgt):
            raise ValueError(f'src/tgt 件数不一致: {len(self.src)} vs {len(self.tgt)}')
        self.meta = data.get('meta', {})
        self.src_len = src_len
        self.tgt_len = tgt_len

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return build_pair(self.src[idx], self.tgt[idx], self.src_len, self.tgt_len)


def collate_pairs(batch):
    """(src, tgt_input, tgt_target) のバッチ化(char_corpus の collate_with_raw と同契約)。"""
    src = torch.stack([b[0] for b in batch])
    tgt_input = torch.stack([b[1] for b in batch])
    tgt_target = torch.stack([b[2] for b in batch])
    return src, tgt_input, tgt_target
