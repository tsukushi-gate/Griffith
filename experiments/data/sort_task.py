"""seq2seq タスクのデータセット（copy / reverse / sort）。

共通仕様:
    入力: [V0, V1, ..., V(N-1)] をフラット化した二値テンソル (seq_len * value_bits,)
    ターゲット: 整数クラスID の (seq_len,) テンソル

サポートするタスク:
    - 'copy':    ターゲット = 入力そのまま（サニティチェック）
    - 'reverse': ターゲット = 入力を逆順（位置ベースAttentionで解ける）
    - 'sort':    ターゲット = 入力を昇順ソート（内容ベース、難易度高）
"""

import random

import torch
from torch.utils.data import Dataset


def _int_to_bits(value: int, num_bits: int) -> list[int]:
    return [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]


class Seq2SeqBinaryDataset(Dataset):
    """
    seq2seq タスク汎用データセット。

    Args:
        task: 'copy' | 'reverse' | 'sort'
    """

    TASKS = ('copy', 'reverse', 'sort')

    def __init__(
        self,
        task: str = 'reverse',
        seq_len: int = 4,
        value_bits: int = 4,
        num_samples: int = 50000,
        seed: int | None = None,
    ):
        assert task in self.TASKS, f'task は {self.TASKS} のいずれか: {task}'
        self.task = task
        self.seq_len = seq_len
        self.value_bits = value_bits
        self.num_samples = num_samples
        self.num_classes = 2 ** value_bits
        self.input_dim = seq_len * value_bits

        rng = random.Random(seed)

        inputs = torch.zeros(num_samples, self.input_dim, dtype=torch.float32)
        targets = torch.zeros(num_samples, seq_len, dtype=torch.long)

        for n in range(num_samples):
            values = [rng.randint(0, self.num_classes - 1) for _ in range(seq_len)]

            if task == 'copy':
                target = list(values)
            elif task == 'reverse':
                target = list(reversed(values))
            elif task == 'sort':
                target = sorted(values)

            flat: list[int] = []
            for v in values:
                flat.extend(_int_to_bits(v, value_bits))
            inputs[n] = torch.tensor(flat, dtype=torch.float32)
            targets[n] = torch.tensor(target)

        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# 互換性のためのエイリアス
SortBinaryDataset = Seq2SeqBinaryDataset


class Seq2SeqWithPointersDataset(Dataset):
    """
    seq2seq タスク + ターゲットポインタ付き（Pointer Network 用）。

    ポインタ `target_pointer[t] = i` は "sorted[t] == values[i]" を満たす。
    `Seq2SeqBinaryDataset` と独立に提供し、既存スクリプトの互換性を壊さない。

    ポインタ計算ルール:
        - copy:    pointer[t] = t
        - reverse: pointer[t] = seq_len - 1 - t
        - sort:    stable sort（同値は元の順序を保つ）
    """

    TASKS = ('copy', 'reverse', 'sort')

    def __init__(
        self,
        task: str = 'sort',
        seq_len: int = 4,
        value_bits: int = 4,
        num_samples: int = 50000,
        seed: int | None = None,
    ):
        assert task in self.TASKS, f'task は {self.TASKS} のいずれか: {task}'
        self.task = task
        self.seq_len = seq_len
        self.value_bits = value_bits
        self.num_samples = num_samples
        self.num_classes = 2 ** value_bits
        self.input_dim = seq_len * value_bits

        rng = random.Random(seed)

        inputs = torch.zeros(num_samples, self.input_dim, dtype=torch.float32)
        targets = torch.zeros(num_samples, seq_len, dtype=torch.long)
        pointers = torch.zeros(num_samples, seq_len, dtype=torch.long)

        for n in range(num_samples):
            values = [rng.randint(0, self.num_classes - 1) for _ in range(seq_len)]

            if task == 'copy':
                target = list(values)
                ptr = list(range(seq_len))
            elif task == 'reverse':
                target = list(reversed(values))
                ptr = list(reversed(range(seq_len)))
            elif task == 'sort':
                # stable sort: 同値は元の位置順を保つ
                # (value, original_idx) でソートして index 列を得る
                indexed = sorted(enumerate(values), key=lambda p: (p[1], p[0]))
                ptr = [i for (i, _) in indexed]
                target = [values[i] for i in ptr]

            flat: list[int] = []
            for v in values:
                flat.extend(_int_to_bits(v, value_bits))
            inputs[n] = torch.tensor(flat, dtype=torch.float32)
            targets[n] = torch.tensor(target)
            pointers[n] = torch.tensor(ptr)

        self.inputs = inputs
        self.targets = targets
        self.pointers = pointers

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self.pointers[idx]
