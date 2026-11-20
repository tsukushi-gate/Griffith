"""連想記憶タスク（Associative Recall）のデータセット。

num_pairs 個のキー・バリューペアと 1 つのクエリキーを入力とし、
クエリに一致するキーのバリューを予測する。

入力は全てフラット二値ベクトル、ターゲットはバリューを整数化したクラスID。
"""

import random

import torch
from torch.utils.data import Dataset


def _int_to_bits(value: int, num_bits: int) -> list[int]:
    """整数を MSB ファーストの二値リストに変換。"""
    return [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]


class AssociativeRecallDataset(Dataset):
    """
    連想記憶タスク。

    各サンプル:
        入力: [K1, V1, K2, V2, ..., K_n, V_n, Q] をフラット化した二値ベクトル
              長さ = num_pairs * (key_bits + value_bits) + key_bits
        ターゲット: Q に一致する K_i の V_i を整数化した値 (0 ～ 2^value_bits - 1)

    各サンプルで num_pairs 個のユニークなキーをランダムに選ぶため、
    key_bits は 2^key_bits >= num_pairs を満たす必要がある。
    """

    def __init__(
        self,
        num_pairs: int = 4,
        key_bits: int = 4,
        value_bits: int = 4,
        num_samples: int = 50000,
        seed: int | None = None,
    ):
        assert 2 ** key_bits >= num_pairs, (
            f'key_bits={key_bits} では 2^{key_bits}={2**key_bits} 個のユニークキーしか作れず、'
            f'num_pairs={num_pairs} に足りない'
        )

        self.num_pairs = num_pairs
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.num_samples = num_samples
        self.input_dim = num_pairs * (key_bits + value_bits) + key_bits
        self.num_classes = 2 ** value_bits

        rng = random.Random(seed)

        all_possible_keys = list(range(2 ** key_bits))
        inputs = torch.zeros(num_samples, self.input_dim, dtype=torch.float32)
        targets = torch.zeros(num_samples, dtype=torch.long)

        pair_size = key_bits + value_bits
        query_offset = num_pairs * pair_size

        for n in range(num_samples):
            selected_keys = rng.sample(all_possible_keys, num_pairs)
            values = [rng.randint(0, self.num_classes - 1) for _ in range(num_pairs)]
            query_idx = rng.randint(0, num_pairs - 1)

            flat: list[int] = []
            for i in range(num_pairs):
                flat.extend(_int_to_bits(selected_keys[i], key_bits))
                flat.extend(_int_to_bits(values[i], value_bits))
            flat.extend(_int_to_bits(selected_keys[query_idx], key_bits))

            inputs[n] = torch.tensor(flat, dtype=torch.float32)
            targets[n] = values[query_idx]

        self.inputs = inputs
        self.targets = targets
        # 正解 pair index も保存（attention 診断用）
        self._query_indices = torch.tensor(
            [0] * num_samples, dtype=torch.long,
        )
        # 上のダミーを再計算 (実装簡略化のためサンプル毎に再生成してから収集)
        # → 上ループで正しい値を保存するようにリファクタ
        self._compute_query_indices(seed)

    def _compute_query_indices(self, seed: int | None) -> None:
        """正解ペアインデックスを再計算して self._query_indices に格納。

        __init__ 内で生成したサンプルは乱数ステートを消費済みのため、
        ここでは inputs/targets をそのまま解析して正解を復元する。
        """
        cfg = self
        pair_size = cfg.key_bits + cfg.value_bits
        query_offset = cfg.num_pairs * pair_size

        # クエリの二値を整数化
        query_bits = self.inputs[:, query_offset:query_offset + cfg.key_bits]
        # keys: (num_samples, num_pairs, key_bits)
        keys = self.inputs[:, :query_offset].reshape(
            -1, cfg.num_pairs, pair_size,
        )[:, :, :cfg.key_bits]

        # 一致するペアを探す: (num_samples, num_pairs)
        matches = (keys == query_bits.unsqueeze(1)).all(dim=-1)
        # 最初の一致インデックス
        self._query_indices = matches.float().argmax(dim=-1).to(torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self._query_indices[idx]
