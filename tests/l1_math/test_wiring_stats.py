"""L1: 配線の受容野・被覆率(analysis/wiring_stats.py)の解析値。

検証する仕様:
    1. クラス受容野 = 最終層のクラス割当ブロックから配線を後ろ向きに辿って到達する
       入力ビットの割合。社内先行実験(CIFAR-10 FFN 48K、9,216 入力 ×
       幅 8,000 × 6 層)の 6 腕の解析値と一致すること。
    2. 受容野の構造上限 = min(1, n_inputs·W / in_dim)。2 入力ゲートなら 2W/入力で、
       CIFAR-10 9,216 入力では 24K(幅 4,000)が 86.8%、12K(幅 2,000)が 43.4%。
    3. layer_stats は core.connections.connection_stats と同じ被覆率を層ごとに返し、
       rot / stretched_rot では (b − a) mod in_dim が 1 種類(= s_b − s_a)になる。
    4. 決定性: 同じ添字なら同じ値(乱数もデバイス依存も無い純粋関数)。

期待値の出所(先行実験の metrics.json headline。`class_rf_*`):
    R 0.30735677083333335 / P 0.9898980034722222 / R2 0.878146701388889 /
    SR2 1.0 / SR2-NG 0.7417534722222222 / RAND 0.7028157552083334
いずれも seed 0/1 の平均。P / RAND は配線が乱数依存なので、runner と同じ
generator 規約(`torch.Generator().manual_seed(seed + 1_000_003)` を層順に消費)で
再生成して比べる(= torch の CPU 乱数ストリームに依存する唯一のテスト)。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.analysis.wiring_stats import (
    class_receptive_field,
    layer_stats,
    receptive_field_upper_bound,
    wiring_summary,
)
from griffith_logic.core.connections import (
    connection_stats,
    get_permutation_connections,
    get_rot_connections,
    get_stretched_rot_connections,
    get_uniform_random_connections,
)

# 先行実験の構成(CIFAR-10 3値サーモメータ 9,216 ビット → 幅 8,000 × 6 層 → 10 クラス)
IN_DIM = 9216
WIDTH = 8000
LAYERS = 6
CLASSES = 10

# training/runner.py の _WIRING_SEED_OFFSET(配線用 generator を初期化用と分ける)
WIRING_SEED_OFFSET = 1_000_003

# 隠れ層のシフト量。H_MNIST は先行実験の MNIST 用スケジュール、
# H_SR2 は先行実験 phase 2 で受容野が入力を覆うよう設計し直したもの
H_MNIST = [4, 20, 90, 400, 1800]
H_SR2 = [64, 640, 1024, 1800, 4096]
H_SR2_NG = [1800, 1800, 1800, 1800, 424]  # H_SR2 と同じ和 7,624 を非幾何に配る

# 腕 → 層ごとの配線指定(先行実験 workspace/arms.py の ARM_OVERRIDES と同じ配線)
ARM_SPECS = {
    'R': [('rot', 3, 96)] + [('rot', 0, s) for s in H_MNIST],
    'P': [('perm',)] * LAYERS,
    'R2': [('rot', 3, 96)] + [('rot', 0, s) for s in H_SR2],
    'SR2': [('stretched_rot', 3, 96)] + [('rot', 0, s) for s in H_SR2],
    'SR2-NG': [('stretched_rot', 3, 96)] + [('rot', 0, s) for s in H_SR2_NG],
    'RAND': [('random_uniform',)] * LAYERS,
}

# 先行実験 metrics.json headline の class_rf_*(seed 0/1 の平均)
EXP100_CLASS_RF = {
    'R': 0.30735677083333335,
    'P': 0.9898980034722222,
    'R2': 0.878146701388889,
    'SR2': 1.0,
    'SR2-NG': 0.7417534722222222,
    'RAND': 0.7028157552083334,
}

# 先行実験 metrics.json headline の coverage_L0_*(層 0 の被覆率。seed 非依存の腕のみ)
EXP100_COVERAGE_L0 = {
    'R': 0.8781467013888888,
    'R2': 0.8781467013888888,
    'SR2': 1.0,
    'SR2-NG': 1.0,
}


def build_arm(arm: str, seed: int) -> list[torch.Tensor]:
    """腕 × seed の層ごとの配線テーブルを作る(runner._build_model と同じ順序で消費)。"""
    generator = torch.Generator().manual_seed(seed + WIRING_SEED_OFFSET)
    tables = []
    for i, spec in enumerate(ARM_SPECS[arm]):
        in_dim = IN_DIM if i == 0 else WIDTH
        kind = spec[0]
        if kind == 'rot':
            tables.append(get_rot_connections(in_dim, WIDTH, spec[1], spec[2]))
        elif kind == 'stretched_rot':
            tables.append(
                get_stretched_rot_connections(in_dim, WIDTH, spec[1], spec[2])
            )
        elif kind == 'perm':
            tables.append(
                get_permutation_connections(in_dim, WIDTH, generator)
            )
        else:  # random_uniform
            tables.append(
                get_uniform_random_connections(
                    in_dim, WIDTH, 2, 'cpu', generator
                )
            )
    return tables


class TestClassReceptiveFieldExp100:
    """先行実験の 6 腕のクラス受容野を解析値と照合する"""

    @pytest.mark.parametrize('arm', sorted(EXP100_CLASS_RF))
    def test_matches_exp100_headline(self, arm):
        values = [
            class_receptive_field(
                build_arm(arm, seed), IN_DIM, CLASSES
            )['class_cov_mean']
            for seed in (0, 1)
        ]
        mean = sum(values) / len(values)
        assert mean == pytest.approx(
            EXP100_CLASS_RF[arm], abs=TOL_F64_EQ, rel=0
        ), f'{arm}: クラス受容野が先行実験の解析値と違う'

    def test_sr2_covers_every_input_for_every_class(self):
        """SR2(受容野 100%)は全クラスが全入力ビットへ到達する"""
        rf = class_receptive_field(build_arm('SR2', 0), IN_DIM, CLASSES)
        assert rf['class_cov_min'] == 1.0
        assert rf['class_cov_max'] == 1.0
        assert rf['pixel_occ_min'] == 1.0
        assert rf['classes'] == CLASSES
        assert rf['group'] == WIDTH // CLASSES
        assert len(rf['class_cov']) == CLASSES

    def test_old_rot_is_far_below_layer0_coverage(self):
        """層 0 の被覆率(87.8%)だけでは足りず、旧 rot のクラス受容野は 30.7%"""
        tables = build_arm('R', 0)
        assert layer_stats(tables, IN_DIM)[0]['coverage'] == pytest.approx(
            EXP100_COVERAGE_L0['R'], abs=TOL_F64_EQ, rel=0
        )
        rf = class_receptive_field(tables, IN_DIM, CLASSES)
        assert rf['class_cov_mean'] < 0.35

    def test_rejects_width_not_divisible_by_classes(self):
        tables = [get_rot_connections(16, 15, 0, 1)]
        with pytest.raises(ValueError, match='割り切れません'):
            class_receptive_field(tables, 16, CLASSES)

    def test_rejects_non_2d_indices(self):
        with pytest.raises(ValueError, match='2 次元'):
            class_receptive_field([torch.zeros(8, dtype=torch.int64)], 8, 2)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match='空'):
            class_receptive_field([], IN_DIM, CLASSES)


class TestReceptiveFieldUpperBound:
    """幅 W の n 入力層が読める入力の割合の上限 = min(1, n·W/in_dim)"""

    @pytest.mark.parametrize(
        ('budget', 'expected'),
        [
            (24000, 2 * (24000 // LAYERS) / IN_DIM),   # 幅 4,000 → 86.8%
            (12000, 2 * (12000 // LAYERS) / IN_DIM),   # 幅 2,000 → 43.4%
        ],
    )
    def test_cifar_budget_bounds(self, budget, expected):
        width = budget // LAYERS
        bound = receptive_field_upper_bound(width, IN_DIM)
        assert bound == pytest.approx(expected, abs=TOL_F64_EQ, rel=0)

    def test_cifar_budget_bounds_percent(self):
        """先行実験 phase 3 の設計計算(24K 86.8% / 12K 43.4%)と小数第1位まで一致"""
        assert round(receptive_field_upper_bound(4000, IN_DIM) * 100, 1) == 86.8
        assert round(receptive_field_upper_bound(2000, IN_DIM) * 100, 1) == 43.4

    def test_saturates_at_one(self):
        # 幅 4,608 で 2W = 9,216 = 入力次元。以降は 1.0 で頭打ち
        assert receptive_field_upper_bound(4608, IN_DIM) == 1.0
        assert receptive_field_upper_bound(WIDTH, IN_DIM) == 1.0

    def test_actual_class_rf_never_exceeds_bound(self):
        """狭い幅では、どう設計してもクラス受容野は上限を超えられない"""
        width = 2000
        bound = receptive_field_upper_bound(width, IN_DIM)
        tables = [get_stretched_rot_connections(IN_DIM, width, 3, 96)]
        tables += [
            get_rot_connections(width, width, 0, round(s * width / WIDTH))
            for s in H_SR2
        ]
        rf = class_receptive_field(tables, IN_DIM, CLASSES)
        assert rf['class_cov_max'] <= bound + TOL_F64_EQ

    @pytest.mark.parametrize('n_inputs', [2, 4, 6])
    def test_scales_with_n_inputs(self, n_inputs):
        assert receptive_field_upper_bound(1000, 100000, n_inputs) == (
            pytest.approx(n_inputs * 1000 / 100000, abs=TOL_F64_EQ, rel=0)
        )

    @pytest.mark.parametrize(
        ('width', 'in_dim', 'n_inputs'), [(0, 10, 2), (10, 0, 2), (10, 10, 0)]
    )
    def test_rejects_non_positive(self, width, in_dim, n_inputs):
        with pytest.raises(ValueError, match='正の整数'):
            receptive_field_upper_bound(width, in_dim, n_inputs)


class TestLayerStats:
    """層ごとの被覆率は core.connections.connection_stats と同じ値"""

    @pytest.mark.parametrize('arm', sorted(EXP100_COVERAGE_L0))
    def test_layer0_coverage_matches_exp100(self, arm):
        stats = layer_stats(build_arm(arm, 0), IN_DIM)
        assert stats[0]['coverage'] == pytest.approx(
            EXP100_COVERAGE_L0[arm], abs=TOL_F64_EQ, rel=0
        )

    def test_agrees_with_core_connection_stats(self):
        tables = build_arm('R2', 0)
        stats = layer_stats(tables, IN_DIM)
        for i, table in enumerate(tables):
            in_dim = IN_DIM if i == 0 else WIDTH
            core = connection_stats(table, in_dim)
            assert stats[i]['coverage'] == core['coverage']
            assert stats[i]['usage_hist'] == core['usage_hist']
            assert stats[i]['same_input'] == core['same_input']
            assert stats[i]['n_gates'] == core['n_gates']
            assert stats[i]['layer'] == i
            assert stats[i]['in_dim'] == in_dim
            assert stats[i]['unreferenced'] == core['usage_hist']['0']
            assert stats[i]['unreferenced_ratio'] == (
                core['usage_hist']['0'] / in_dim
            )

    def test_rot_has_constant_pair_difference(self):
        """rot / stretched_rot は (b − a) mod in_dim が 1 種類(= s_b − s_a)"""
        stats = layer_stats(build_arm('SR2', 0), IN_DIM)
        assert stats[0]['raw_diff_unique'] == 1
        assert stats[0]['raw_diff_value'] == 96 - 3
        for i, s_b in enumerate(H_SR2, start=1):
            assert stats[i]['raw_diff_unique'] == 1
            assert stats[i]['raw_diff_value'] == s_b

    def test_permutation_has_many_pair_differences(self):
        stats = layer_stats(build_arm('P', 0), IN_DIM)
        assert stats[0]['raw_diff_unique'] > 1
        assert stats[0]['raw_diff_value'] is None

    def test_n_inputs_other_than_two_has_no_pair_difference(self):
        idx = torch.arange(3 * 5, dtype=torch.int64).reshape(3, 5) % 5
        stats = layer_stats([idx], 5)
        assert stats[0]['raw_diff_unique'] is None
        assert stats[0]['raw_diff_value'] is None


class TestDeterminism:
    """乱数もデバイス依存も無い純粋関数であること"""

    def test_repeated_calls_are_identical(self):
        tables = build_arm('SR2-NG', 0)
        first = class_receptive_field(tables, IN_DIM, CLASSES)
        second = class_receptive_field(tables, IN_DIM, CLASSES)
        assert first == second
        assert layer_stats(tables, IN_DIM) == layer_stats(tables, IN_DIM)

    def test_accepts_numpy_and_torch_alike(self):
        tables = build_arm('R', 0)
        as_numpy = [t.numpy() for t in tables]
        assert (
            class_receptive_field(as_numpy, IN_DIM, CLASSES)
            == class_receptive_field(tables, IN_DIM, CLASSES)
        )

    def test_wiring_summary_bundles_the_three(self):
        tables = build_arm('SR2', 0)
        summary = wiring_summary(tables, IN_DIM, CLASSES)
        assert summary['layers'] == layer_stats(tables, IN_DIM)
        assert summary['class_rf'] == class_receptive_field(
            tables, IN_DIM, CLASSES
        )
        assert summary['rf_upper_bound'] == receptive_field_upper_bound(
            WIDTH, IN_DIM, 2
        )
