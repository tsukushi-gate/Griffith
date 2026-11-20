"""位置指定ルックアップタスクのデータセット。

num_slots 個の値 V0..V(N-1) と位置インデックス pos が与えられ、V_pos を予測する。
num_slots = 2^pos_bits。
"""

import random

import torch
from torch.utils.data import Dataset


def _int_to_bits(value: int, num_bits: int) -> list[int]:
    return [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]


class PositionalLookupDataset(Dataset):
    """
    各サンプル:
        入力: [V0, V1, ..., V(N-1), pos]  （全てフラット二値）
        ターゲット: V_pos の整数値
    """

    def __init__(
        self,
        pos_bits: int = 3,
        value_bits: int = 4,
        num_samples: int = 50000,
        seed: int | None = None,
    ):
        self.pos_bits = pos_bits
        self.value_bits = value_bits
        self.num_slots = 2 ** pos_bits
        self.num_classes = 2 ** value_bits
        self.num_samples = num_samples
        self.input_dim = self.num_slots * value_bits + pos_bits

        rng = random.Random(seed)

        inputs = torch.zeros(num_samples, self.input_dim, dtype=torch.float32)
        targets = torch.zeros(num_samples, dtype=torch.long)
        positions = torch.zeros(num_samples, dtype=torch.long)

        for n in range(num_samples):
            values = [rng.randint(0, self.num_classes - 1) for _ in range(self.num_slots)]
            pos = rng.randint(0, self.num_slots - 1)

            flat: list[int] = []
            for v in values:
                flat.extend(_int_to_bits(v, value_bits))
            flat.extend(_int_to_bits(pos, pos_bits))

            inputs[n] = torch.tensor(flat, dtype=torch.float32)
            targets[n] = values[pos]
            positions[n] = pos

        self.inputs = inputs
        self.targets = targets
        # attention 診断用（正解位置）
        self.positions = positions

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self.positions[idx]
