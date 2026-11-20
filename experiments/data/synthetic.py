"""合成タスク用データセット（copy / reverse / sort）。"""

import torch
from torch.utils.data import Dataset

# 特殊トークンID
TOKEN_PAD = 0
TOKEN_BOS = 1
TOKEN_EOS = 2
SPECIAL_TOKENS = 3  # PAD, BOS, EOS


class CopyDataset(Dataset):
    """入力をそのまま出力するタスク。seq2seqの基本動作確認。"""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(
            SPECIAL_TOKENS, vocab_size, (num_samples, seq_len)
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src = self.data[idx]
        tgt = self.data[idx].clone()
        return src, tgt


class ReverseDataset(Dataset):
    """入力を反転して出力するタスク。"""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(
            SPECIAL_TOKENS, vocab_size, (num_samples, seq_len)
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src = self.data[idx]
        tgt = src.flip(0)
        return src, tgt


class SortDataset(Dataset):
    """入力をソートして出力するタスク。"""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(
            SPECIAL_TOKENS, vocab_size, (num_samples, seq_len)
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src = self.data[idx]
        tgt, _ = src.sort()
        return src, tgt


TASK_DATASETS = {
    'copy': CopyDataset,
    'reverse': ReverseDataset,
    'sort': SortDataset,
}


def _register_temporal_tasks():
    """時間的信用割当タスク(delayed_recall / temporal_xor)を登録。

    循環importを避けるため関数内でimportして遅延登録する。
    """
    from .temporal import DelayedRecallDataset, TemporalXORDataset
    TASK_DATASETS.setdefault('delayed_recall', DelayedRecallDataset)
    TASK_DATASETS.setdefault('temporal_xor', TemporalXORDataset)


_register_temporal_tasks()
