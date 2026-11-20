"""配線の受容野・被覆率(学習を走らせずに配線テーブルだけから決まる量)。

FFN の各層の配線テーブル(`(n_inputs, out_dim)` の添字)から、

- `layer_stats`                  — 層ごとの被覆率・使用回数分布(`core.connections.connection_stats`
                                   を層ごとに呼ぶ)と、rot 系の `(b - a) mod in_dim` が定数かどうか
- `class_receptive_field`        — **クラス受容野**。GroupSum のクラス割当を考慮し、
                                   1 クラスが読める入力ビットの割合
- `receptive_field_upper_bound`  — 幅 W の n 入力層が原理的に読める入力の割合の上限
                                   (2 入力なら `2W / 入力次元`)
- `wiring_summary`               — 上 3 つをまとめた 1 ネット分の辞書

を計算する。すべて**純粋関数**(乱数もデバイス依存も無い)で、同じ添字を与えれば同じ値を返す。

用途: 配線の設計則(受容野が入力を覆うようシフト量を選ぶ)を、学習を走らせる前に
配線だけで検証する。社内先行実験の配線検証と設計計算がこの計算に当たる。

`class_receptive_field` は最終 LutLayer の出力をクラスへ割り当てる規約として
GroupSum の**連続ブロック割当**(クラス c はゲート `c*group … (c+1)*group-1`)を仮定する。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..core.connections import connection_stats

__all__ = [
    'CIFAR10_3T',
    'ImageLayout',
    'as_index_array',
    'class_receptive_field',
    'layer_stats',
    'receptive_field_upper_bound',
    'wiring_summary',
]


@dataclass(frozen=True)
class ImageLayout:
    """入力ビットの並び(しきい値 minor)。

    平坦化順は `idx = ((channel * rows + row) * cols + col) * thresholds + threshold`
    (`datasets.vision.ThresholdTransform(interleave=True)` と同じ順)。
    本モジュールでは画素占有率の分母(1 画素あたりのしきい値本数)にだけ使う。
    """

    channels: int = 3
    rows: int = 32
    cols: int = 32
    thresholds: int = 3

    @property
    def n_bits(self) -> int:
        """入力ビット数(= channels·rows·cols·thresholds)"""
        return self.channels * self.rows * self.cols * self.thresholds

    @property
    def n_pixels(self) -> int:
        """画素数(しきい値本数を潰した数)"""
        return self.channels * self.rows * self.cols


#: CIFAR-10 32x32 RGB の 3 値サーモメータ(9,216 ビット)
CIFAR10_3T = ImageLayout()


def as_index_array(indices) -> np.ndarray:
    """添字テーブルを int64 の numpy 配列にする(torch.Tensor も受ける)。

    Args:
        indices: `(n_inputs, out_dim)` の添字テーブル(torch.Tensor / numpy / 入れ子リスト)

    Returns:
        int64 の numpy 配列(コピー可否は入力依存。呼び出し側は書き換えないこと)
    """
    if isinstance(indices, torch.Tensor):
        return indices.detach().to('cpu').to(torch.int64).numpy()
    return np.asarray(indices, dtype=np.int64)


def _as_layer_arrays(indices: Sequence) -> list[np.ndarray]:
    """層順の添字テーブル列を int64 numpy 配列のリストにする(形状も検証)。"""
    arrays = [as_index_array(t) for t in indices]
    if not arrays:
        raise ValueError('indices が空です(層が 1 つ以上必要)')
    for i, arr in enumerate(arrays):
        if arr.ndim != 2:
            raise ValueError(
                f'層 {i} の添字は (n_inputs, out_dim) の 2 次元が必要です: {arr.shape}'
            )
    return arrays


def layer_stats(indices: Sequence, in_dim0: int) -> list[dict]:
    """層順の添字テーブルに `core.connections.connection_stats` を掛ける。

    層 i > 0 の入力次元は前層のゲート数(= `indices[i-1].shape[1]`)とみなす。
    `n_inputs = 2` のときは `(b - a) mod in_dim` の種類数も返す。rot / stretched_rot は
    対の差が一定なので 1 種類(= `s_b - s_a`)になり、設計どおりのシフト量が
    入っているかの確認に使える。

    Args:
        indices: 層順の `(n_inputs, out_dim)` 添字テーブルの列
        in_dim0: 層 0 の入力次元

    Returns:
        層ごとの辞書のリスト。`connection_stats` のキー
        (`coverage` / `usage_hist` / `same_input` / `n_gates`)に加えて
        `layer` / `in_dim` / `unreferenced` / `unreferenced_ratio` /
        `raw_diff_unique` / `raw_diff_value` を持つ。

    Raises:
        ValueError: 添字が 2 次元でない / 範囲外(`connection_stats` が送出)
    """
    arrays = _as_layer_arrays(indices)
    out: list[dict] = []
    for i, idx in enumerate(arrays):
        in_dim = int(in_dim0) if i == 0 else int(arrays[i - 1].shape[1])
        stats = connection_stats(torch.from_numpy(idx), in_dim)
        stats['layer'] = i
        stats['in_dim'] = in_dim
        stats['unreferenced'] = stats['usage_hist']['0']
        stats['unreferenced_ratio'] = stats['usage_hist']['0'] / in_dim
        if idx.shape[0] == 2:
            unique = sorted({int(v) for v in ((idx[1] - idx[0]) % in_dim)})
            stats['raw_diff_unique'] = len(unique)
            stats['raw_diff_value'] = unique[0] if len(unique) == 1 else None
        else:
            stats['raw_diff_unique'] = None
            stats['raw_diff_value'] = None
        out.append(stats)
    return out


def class_receptive_field(
    indices: Sequence,
    in_dim0: int,
    classes: int = 10,
    thresholds: int = CIFAR10_3T.thresholds,
) -> dict:
    """クラス受容野: 1 クラスが読める入力ビットの割合。

    最終層の幅を `classes` で割った `group` 個のゲート(GroupSum の連続ブロック割当で
    クラス c に加算されるゲート)から配線を**後ろ向きに**辿り、到達する入力ビットの
    割合を数える。層ごとに bool マスクを逆写像するだけなので、幅 8,000 × 6 層でも
    1 秒未満で終わる。

    層 0 の被覆率(`layer_stats` の `coverage`)は「全クラス合計で読む入力」なので、
    100% でもクラスごとには一部しか読めていないことがある。設計則で問うのは本関数の
    `class_cov_mean` / `class_cov_min` のほう。

    Args:
        indices: 層順の `(n_inputs, out_dim)` 添字テーブルの列
        in_dim0: 層 0 の入力次元
        classes: クラス数(最終層の幅がこれで割り切れること)
        thresholds: 1 画素あたりのしきい値本数(画素占有率の分母に使う)

    Returns:
        `{'classes', 'group', 'class_cov_mean', 'class_cov_min', 'class_cov_max',
          'class_cov'(クラスごとの割合), 'pixel_occ_min'}`

    Raises:
        ValueError: 最終層の幅がクラス数で割り切れない / 添字が 2 次元でない
    """
    arrays = _as_layer_arrays(indices)
    n_layers = len(arrays)
    widths = [int(a.shape[1]) for a in arrays]
    out_width = widths[-1]
    if classes <= 0 or out_width % classes:
        raise ValueError(
            f'最終層の幅 {out_width} がクラス数 {classes} で割り切れません'
        )
    group = out_width // classes
    covs: list[float] = []
    pixels: list[float] = []
    for c in range(classes):
        current = np.zeros(out_width, bool)
        current[c * group:(c + 1) * group] = True
        for i in range(n_layers - 1, -1, -1):
            in_dim = int(in_dim0) if i == 0 else widths[i - 1]
            nxt = np.zeros(in_dim, bool)
            selected = np.nonzero(current)[0]
            for row in range(arrays[i].shape[0]):
                nxt[arrays[i][row][selected]] = True
            current = nxt
        covs.append(float(current.mean()))
        reached = np.nonzero(current)[0]
        pixels.append(
            len(set((reached // int(thresholds)).tolist()))
            / (int(in_dim0) // int(thresholds))
        )
    return {
        'classes': int(classes),
        'group': int(group),
        'class_cov_mean': float(np.mean(covs)),
        'class_cov_min': float(min(covs)),
        'class_cov_max': float(max(covs)),
        'class_cov': covs,
        'pixel_occ_min': float(min(pixels)),
    }


def receptive_field_upper_bound(
    width: int, in_dim: int, n_inputs: int = 2
) -> float:
    """幅 `width` の層が読める入力の割合の上限(2 入力なら `2W / 入力次元`)。

    層 0 のゲートは 1 個あたり `n_inputs` 本しか入力を読まないので、幅 W の層 0 が
    触れる入力は高々 `n_inputs · W` 種類しかない。後続の層はその集合を増やせないから、
    **クラス受容野も層 0 の被覆率も** この値を超えられない(配線の種類によらない構造上限)。

    設計への使い方: 目標のクラス受容野が 1.0 なら、幅は
    `width >= in_dim / n_inputs` が必要になる(CIFAR-10 の 9,216 入力・2 入力ゲートなら
    幅 4,608 以上)。それより狭い予算では、シフト量をどう設計してもこの上限で頭打ちになる。

    Args:
        width: 層の出力ゲート数(層 0 の幅)
        in_dim: 層 0 の入力次元
        n_inputs: 1 ゲートあたりの入力数

    Returns:
        `min(1.0, n_inputs · width / in_dim)`

    Raises:
        ValueError: width / in_dim / n_inputs が正でない
    """
    if width <= 0 or in_dim <= 0 or n_inputs <= 0:
        raise ValueError(
            f'width / in_dim / n_inputs は正の整数が必要です: '
            f'width={width}, in_dim={in_dim}, n_inputs={n_inputs}'
        )
    return min(1.0, int(n_inputs) * int(width) / int(in_dim))


def wiring_summary(
    indices: Sequence,
    in_dim0: int,
    classes: int = 10,
    layout: ImageLayout = CIFAR10_3T,
) -> dict:
    """1 ネット分の配線統計(層ごと + クラス受容野 + 受容野の構造上限)。

    Args:
        indices: 層順の `(n_inputs, out_dim)` 添字テーブルの列
        in_dim0: 層 0 の入力次元
        classes: クラス数
        layout: 入力ビットの並び(画素占有率のしきい値本数に使う)

    Returns:
        `{'layers': layer_stats(...), 'class_rf': class_receptive_field(...),
          'rf_upper_bound': receptive_field_upper_bound(...)}`
    """
    arrays = _as_layer_arrays(indices)
    return {
        'layers': layer_stats(arrays, in_dim0),
        'class_rf': class_receptive_field(
            arrays, in_dim0, classes, layout.thresholds
        ),
        'rf_upper_bound': receptive_field_upper_bound(
            int(arrays[0].shape[1]), in_dim0, int(arrays[0].shape[0])
        ),
    }
