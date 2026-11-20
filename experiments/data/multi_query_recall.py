"""マルチクエリ連想記憶タスクのデータセット。

num_pairs 個のキー・バリューペアと num_queries 個のクエリを入力とし、
各クエリに対応するバリューを予測する。

入力は全てフラット二値ベクトル、ターゲットは (num_queries,) の整数ラベル。
"""

import random

import torch
from torch.utils.data import Dataset


def _int_to_bits(value: int, num_bits: int) -> list[int]:
    return [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]


class MultiQueryRecallDataset(Dataset):
    """
    マルチクエリ版連想記憶タスク。

    各サンプル:
        入力: [K1, V1, ..., K_np, V_np, Q1, ..., Q_nq] をフラット化
        ターゲット: (num_queries,) の整数テンソル。各要素は対応クエリの答え
    """

    def __init__(
        self,
        num_pairs: int = 4,
        num_queries: int = 3,
        key_bits: int = 4,
        value_bits: int = 4,
        num_samples: int = 50000,
        allow_duplicate_queries: bool = True,
        seed: int | None = None,
    ):
        assert 2 ** key_bits >= num_pairs, (
            f'key_bits={key_bits} では 2^{key_bits}={2**key_bits} 個しか作れず '
            f'num_pairs={num_pairs} に足りない'
        )
        if not allow_duplicate_queries:
            assert num_queries <= num_pairs, (
                'allow_duplicate_queries=False の場合 num_queries <= num_pairs'
            )

        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.num_samples = num_samples
        self.num_classes = 2 ** value_bits

        pair_size = key_bits + value_bits
        self.input_dim = num_pairs * pair_size + num_queries * key_bits

        rng = random.Random(seed)
        all_possible_keys = list(range(2 ** key_bits))

        inputs = torch.zeros(num_samples, self.input_dim, dtype=torch.float32)
        targets = torch.zeros(num_samples, num_queries, dtype=torch.long)
        query_indices = torch.zeros(num_samples, num_queries, dtype=torch.long)

        for n in range(num_samples):
            selected_keys = rng.sample(all_possible_keys, num_pairs)
            values = [rng.randint(0, self.num_classes - 1) for _ in range(num_pairs)]

            if allow_duplicate_queries:
                q_idx_list = [rng.randint(0, num_pairs - 1) for _ in range(num_queries)]
            else:
                q_idx_list = rng.sample(range(num_pairs), num_queries)

            flat: list[int] = []
            for i in range(num_pairs):
                flat.extend(_int_to_bits(selected_keys[i], key_bits))
                flat.extend(_int_to_bits(values[i], value_bits))
            for qi in q_idx_list:
                flat.extend(_int_to_bits(selected_keys[qi], key_bits))

            inputs[n] = torch.tensor(flat, dtype=torch.float32)
            targets[n] = torch.tensor([values[qi] for qi in q_idx_list])
            query_indices[n] = torch.tensor(q_idx_list)

        self.inputs = inputs
        self.targets = targets
        self._query_indices = query_indices

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self._query_indices[idx]
