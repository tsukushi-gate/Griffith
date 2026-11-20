"""検証ベクトルの生成と hex パッキング。

パッキング規約はライブラリ export/verify.py と同一(bit i を整数の i 桁目へ、
group_sum スコアは score_bits 刻みで連結)。小さい入力空間は全数、それ以外は
シード付き乱数で生成する。
"""

from __future__ import annotations

import numpy as np

EXHAUSTIVE_LIMIT = 4096  # 2^in_width がこれ以下なら全数検証


def gen_inputs(in_width: int, n_vectors: int, seed: int = 0) -> np.ndarray:
    """(batch, in_width) の 0/1 入力を生成する。"""
    if (1 << in_width) <= min(EXHAUSTIVE_LIMIT, max(n_vectors, 1) * 16):
        space = 1 << in_width
        vals = np.arange(space, dtype=np.int64)
        bits = ((vals[:, None] >> np.arange(in_width)) & 1).astype(np.int64)
        return bits
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_vectors, in_width), dtype=np.int64)


def pack_bits(rows: np.ndarray) -> list[int]:
    """(batch, width) の 0/1 配列 → bit i を i 桁目に詰めた整数列。"""
    return [sum(int(b) << i for i, b in enumerate(row)) for row in rows]


def pack_scores(scores: np.ndarray, score_bits: int) -> list[int]:
    """(batch, k) の整数スコア → score_bits ごとに連結した整数列。"""
    return [
        sum(int(s) << (c * score_bits) for c, s in enumerate(row))
        for row in scores
    ]


def hex_lines(values: list[int], width_bits: int) -> str:
    digits = (width_bits + 3) // 4
    return '\n'.join(f'{v:0{digits}x}' for v in values) + '\n'


def unpack_bits(value: int, width: int) -> np.ndarray:
    """整数 → (width,) の 0/1 配列(pack_bits の逆)。"""
    return np.array([(value >> i) & 1 for i in range(width)], dtype=np.int64)


def unpack_scores(value: int, k: int, score_bits: int) -> np.ndarray:
    mask = (1 << score_bits) - 1
    return np.array([(value >> (c * score_bits)) & mask for c in range(k)],
                    dtype=np.int64)
