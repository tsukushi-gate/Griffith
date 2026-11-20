"""L1: rot / 置換の配線生成器(先行実験 S3)の定義。

検証する仕様:
    rot  : idx_a[g] = (g + s_a) mod in_dim, idx_b[g] = (g + s_b) mod in_dim
           (社内先行実験 wiring.py rot_layer)
    perm : 2·out_dim 個の入力スロットを in_dim 個のソースが
           floor/ceil(2·out_dim / in_dim) 回ずつ埋める均衡全射
           (先行実験 perm_index / perm_layer と同じ意味。a == b は排除しない)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.core.connections import (
    connection_stats,
    get_permutation_connections,
    get_random_connections,
    get_rot_connections,
    get_stretched_rot_connections,
    get_uniform_random_connections,
)
from griffith_logic.layers.lut import LutLayer

# 先行実験 B0 の rot スケジュール（先行実験 sched/cifar_d6.json P6c_d6）
P6C_D6_SA = [3, 0, 0, 0, 0, 0]
P6C_D6_SB = [96, 4, 20, 90, 400, 1800]
P6C_D6_DIMS = [(9216, 8000)] + [(8000, 8000)] * 5


class TestRotConnections:
    @pytest.mark.parametrize(
        ('in_dim', 'out_dim', 's_a', 's_b'),
        [
            (9216, 8000, 3, 96),
            (8000, 8000, 0, 1800),
            (16, 8, 0, 0),
            (7, 20, 5, 13),      # out_dim > in_dim（巻き戻る）
            (10, 10, -3, 27),    # 負値・in_dim超
        ],
    )
    def test_matches_formula(self, in_dim, out_dim, s_a, s_b):
        idx = get_rot_connections(in_dim, out_dim, s_a, s_b)
        assert idx.shape == (2, out_dim)
        assert idx.dtype == torch.int64
        expected_a = torch.tensor(
            [(g + s_a) % in_dim for g in range(out_dim)], dtype=torch.int64)
        expected_b = torch.tensor(
            [(g + s_b) % in_dim for g in range(out_dim)], dtype=torch.int64)
        assert torch.equal(idx[0], expected_a)
        assert torch.equal(idx[1], expected_b)
        assert idx.min() >= 0 and idx.max() < in_dim

    def test_deterministic(self):
        a = get_rot_connections(9216, 8000, 3, 96)
        torch.manual_seed(123)
        b = get_rot_connections(9216, 8000, 3, 96)
        assert torch.equal(a, b)

    def test_p6c_d6_layer0_window(self):
        """P6c_d6 層0 は入力 [3, 8096) しか触らない（先行実験の既知の注意点）"""
        idx = get_rot_connections(9216, 8000, 3, 96)
        used = torch.unique(idx)
        assert int(used.min()) == 3
        assert int(used.max()) == 8095


class TestPermutationConnections:
    @pytest.mark.parametrize(
        ('in_dim', 'out_dim'), P6C_D6_DIMS[:1] + [(8000, 8000), (784, 8000), (37, 8)]
    )
    def test_usage_counts_are_floor_or_ceil(self, in_dim, out_dim):
        gen = torch.Generator().manual_seed(0)
        idx = get_permutation_connections(in_dim, out_dim, gen)
        assert idx.shape == (2, out_dim)
        assert idx.dtype == torch.int64
        counts = torch.bincount(idx.reshape(-1), minlength=in_dim)
        total = 2 * out_dim
        lo, hi = total // in_dim, -(-total // in_dim)  # floor / ceil
        assert int(counts.min()) >= lo, (int(counts.min()), lo)
        assert int(counts.max()) <= hi, (int(counts.max()), hi)
        assert int(counts.sum()) == total

    def test_equal_dims_use_every_source_exactly_twice(self):
        gen = torch.Generator().manual_seed(1)
        idx = get_permutation_connections(8000, 8000, gen)
        counts = torch.bincount(idx.reshape(-1), minlength=8000)
        assert torch.equal(counts, torch.full((8000,), 2))

    def test_layer0_9216_to_8000_uses_every_source(self):
        """先行実験層0: 9216 -> 16000 スロットで各ソース 1〜2 回（未参照ゼロ）"""
        gen = torch.Generator().manual_seed(2)
        idx = get_permutation_connections(9216, 8000, gen)
        counts = torch.bincount(idx.reshape(-1), minlength=9216)
        assert int(counts.min()) == 1
        assert int(counts.max()) == 2
        assert int((counts == 0).sum()) == 0

    def test_counts_are_exactly_floor_or_ceil(self):
        """使用回数が q または q+1 のみ（a==b を避ける等の細工が無い）"""
        gen = torch.Generator().manual_seed(3)
        in_dim, out_dim = 37, 40
        idx = get_permutation_connections(in_dim, out_dim, gen)
        total = 2 * out_dim
        q, r = divmod(total, in_dim)
        counts = torch.bincount(idx.reshape(-1), minlength=in_dim)
        assert set(counts.tolist()) <= {q, q + 1}
        assert int((counts == q + 1).sum()) == r
        assert int(counts.sum()) == total

    def test_remainder_subset_is_randomized(self):
        """余り（+1 回使われるソース）の選び方が seed で変わる = 偏らない。

        先行実験の原実装はタイルの切り詰めなので余りが常に先頭のソースに寄る。
        本実装は乱択するので、構造的未参照率の測定に位置バイアスが入らない。
        """
        in_dim, out_dim = 100, 60   # total=120 → q=1, r=20
        sets = []
        for seed in range(4):
            idx = get_permutation_connections(
                in_dim, out_dim, torch.Generator().manual_seed(seed))
            counts = torch.bincount(idx.reshape(-1), minlength=in_dim)
            sets.append(frozenset((counts == 2).nonzero().flatten().tolist()))
            assert len(sets[-1]) == 20
        assert len(set(sets)) > 1, '余りの選び方が seed で変わらない'

    def test_pigeonhole_case_uses_random_subset(self):
        """2·out_dim < in_dim では全ソースは使えない（q=0）。乱択の 1 回ずつ"""
        in_dim, out_dim = 9216, 500   # total=1000 < 9216
        idx = get_permutation_connections(
            in_dim, out_dim, torch.Generator().manual_seed(0))
        counts = torch.bincount(idx.reshape(-1), minlength=in_dim)
        assert set(counts.tolist()) == {0, 1}
        assert int((counts == 1).sum()) == 1000
        # 先頭 1000 個への切り詰めではない（乱択されている）
        assert int(counts[:1000].sum()) < 1000

    def test_same_seed_same_wiring(self):
        a = get_permutation_connections(
            9216, 8000, torch.Generator().manual_seed(7))
        b = get_permutation_connections(
            9216, 8000, torch.Generator().manual_seed(7))
        assert torch.equal(a, b)

    def test_different_seed_different_wiring(self):
        a = get_permutation_connections(
            9216, 8000, torch.Generator().manual_seed(7))
        b = get_permutation_connections(
            9216, 8000, torch.Generator().manual_seed(8))
        assert not torch.equal(a, b)

    def test_independent_of_global_rng(self):
        torch.manual_seed(999)
        a = get_permutation_connections(
            256, 128, torch.Generator().manual_seed(5))
        torch.manual_seed(1)
        b = get_permutation_connections(
            256, 128, torch.Generator().manual_seed(5))
        assert torch.equal(a, b)

    def test_same_input_gates_are_not_excluded(self):
        """a == b のゲートは仕様上ありうる（原実装と同じく排除しない）"""
        gen = torch.Generator().manual_seed(0)
        idx = get_permutation_connections(4, 64, gen)
        assert int((idx[0] == idx[1]).sum()) > 0


class TestStretchedRotConnections:
    """引き伸ばし rot(先行実験 T2)の定義: base[g] = round(g·in/out)、半整数は切り上げ"""

    @staticmethod
    def _base(g, in_dim, out_dim):
        """素朴な参照: floor(g·in/out + 1/2)(有理数で厳密に、半整数は切り上げ)"""
        return (2 * g * in_dim + out_dim) // (2 * out_dim)

    @pytest.mark.parametrize(
        ('in_dim', 'out_dim', 's_a', 's_b'),
        [
            (9216, 8000, 3, 96),   # 先行実験層0
            (8000, 8000, 0, 1800),  # in == out(rot と一致するはず)
            (16, 8, 0, 0),
            (7, 20, 5, 13),        # out_dim > in_dim(引き伸ばしではなく縮み)
            (10, 10, -3, 27),      # 負値・in_dim超
        ],
    )
    def test_matches_formula(self, in_dim, out_dim, s_a, s_b):
        idx = get_stretched_rot_connections(in_dim, out_dim, s_a, s_b)
        assert idx.shape == (2, out_dim)
        assert idx.dtype == torch.int64
        expected_a = torch.tensor(
            [(self._base(g, in_dim, out_dim) + s_a) % in_dim
             for g in range(out_dim)], dtype=torch.int64)
        expected_b = torch.tensor(
            [(self._base(g, in_dim, out_dim) + s_b) % in_dim
             for g in range(out_dim)], dtype=torch.int64)
        assert torch.equal(idx[0], expected_a)
        assert torch.equal(idx[1], expected_b)
        assert idx.min() >= 0 and idx.max() < in_dim

    @pytest.mark.parametrize(
        ('in_dim', 'out_dim', 's_a', 's_b'),
        [(8000, 8000, 0, 1800), (9216, 9216, 3, 96), (10, 10, -3, 27)],
    )
    def test_equals_rot_when_dims_equal(self, in_dim, out_dim, s_a, s_b):
        """in_dim == out_dim では base[g] = g なので rot と bitwise 一致"""
        assert torch.equal(
            get_stretched_rot_connections(in_dim, out_dim, s_a, s_b),
            get_rot_connections(in_dim, out_dim, s_a, s_b),
        )

    def test_rounding_is_half_up_not_banker(self):
        """半整数の丸めは切り上げ(Python の round は偶数丸めなので不一致)"""
        idx = get_stretched_rot_connections(16, 8, 0, 0)   # base = round(2g) 整数
        assert torch.equal(idx[0], torch.arange(0, 16, 2))
        # in=4, out=8 では g が奇数のとき base が半整数
        idx2 = get_stretched_rot_connections(4, 8, 0, 0)
        assert idx2[0].tolist() == [0, 1, 1, 2, 2, 3, 3, 0]  # 4 → mod 4 = 0
        banker = [round(g * 4 / 8) % 4 for g in range(8)]
        assert idx2[0].tolist() != banker, '偶数丸めと同じになっている'

    def test_locality_is_preserved(self):
        """対の差が一定 = rot と同じ局所性(先行実験順では s_b - s_a = 93)"""
        in_dim = 9216
        idx = get_stretched_rot_connections(in_dim, 8000, 3, 96)
        diff = (idx[1] - idx[0]) % in_dim
        assert set(diff.tolist()) == {(96 - 3) % in_dim}

    def test_exp100_layer0_covers_every_input(self):
        """先行実験層0(9216→8000, s_a=3, s_b=96)の被覆率は 100.0%(実測)。

        plan.md の見積もりは「約 98%」だったが、基準位置 8000 個は増分 1 か 2 で
        進むため、+3 と +96 の和集合が 9216 本すべてを覆う。
        同条件の rot は 8093/9216 = 87.81%(TestRotConnections 参照)。
        """
        in_dim, out_dim = 9216, 8000
        idx = get_stretched_rot_connections(in_dim, out_dim, 3, 96)
        used = torch.unique(idx)
        assert int(used.numel()) == 9216
        assert int(used.numel()) / in_dim == 1.0
        counts = torch.bincount(idx.reshape(-1), minlength=in_dim)
        assert int((counts == 0).sum()) == 0
        assert int((counts == 1).sum()) == 2432
        assert int((counts == 2).sum()) == 6784
        assert int((counts >= 3).sum()) == 0

    def test_coverage_is_not_always_full(self):
        """整数比では基準位置が飛び、s_b - s_a の剰余次第で穴が残る。

        in=16, out=8 は base が偶数のみ。s_a=3 / s_b=5(差が偶数)では
        奇数の入力しか読まず被覆 50%、s_a=3 / s_b=6(差が奇数)なら 100%。
        先行実験の 9216/8000 = 1.152 は整数比でないのでこの穴は出ない。
        """
        half = get_stretched_rot_connections(16, 8, 3, 5)
        assert int(torch.unique(half).numel()) == 8
        assert set(half.reshape(-1).tolist()) <= set(range(1, 16, 2))
        full = get_stretched_rot_connections(16, 8, 3, 6)
        assert int(torch.unique(full).numel()) == 16

    def test_base_positions_are_distinct_and_in_range(self):
        """基準位置は 8000 個すべて相異なる(引き伸ばしの前提)"""
        idx = get_stretched_rot_connections(9216, 8000, 0, 0)
        assert int(torch.unique(idx[0]).numel()) == 8000
        assert int(idx[0].max()) == (2 * 7999 * 9216 + 8000) // 16000

    def test_deterministic_and_independent_of_global_rng(self):
        torch.manual_seed(0)
        a = get_stretched_rot_connections(9216, 8000, 3, 96)
        torch.manual_seed(123)
        b = get_stretched_rot_connections(9216, 8000, 3, 96)
        assert torch.equal(a, b)


class TestSeededRandomConnections:
    """get_random_connections の generator 化(先行実験 T3)と既定の不変性"""

    def test_none_generator_matches_global_rng(self):
        """generator=None は従来どおりグローバル RNG(bitwise 不変)"""
        torch.manual_seed(4321)
        legacy = get_random_connections(256, 128, 2)
        torch.manual_seed(4321)
        explicit_none = get_random_connections(256, 128, 2, 'cpu', None)
        assert torch.equal(legacy, explicit_none)

    def test_same_seed_same_wiring(self):
        a = get_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(7))
        b = get_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(7))
        assert torch.equal(a, b)

    def test_different_seed_different_wiring(self):
        a = get_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(7))
        b = get_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(8))
        assert not torch.equal(a, b)

    def test_independent_of_global_rng(self):
        torch.manual_seed(999)
        a = get_random_connections(
            256, 128, 2, 'cpu', torch.Generator().manual_seed(5))
        torch.manual_seed(1)
        b = get_random_connections(
            256, 128, 2, 'cpu', torch.Generator().manual_seed(5))
        assert torch.equal(a, b)

    def test_usage_counts_are_balanced(self):
        """**実装の性質**: 'random' は均衡している(未参照が出ない)。

        randperm(total) % in_dim を randperm(in_dim) で貼り替える実装なので、
        使用回数は floor / ceil(total / in_dim) の 2 値だけ。
        先行実験層0(9216→8000, n=2)では被覆率が 100% になり、
        置換配線(perm)と同じ分布になる。無制約の乱択は
        get_uniform_random_connections が必要。
        """
        idx = get_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(1_000_003))
        counts = torch.bincount(idx.reshape(-1), minlength=9216)
        assert set(counts.tolist()) == {1, 2}
        assert int((counts == 1).sum()) == 2432
        assert int((counts == 2).sum()) == 6784
        assert int(torch.unique(idx).numel()) == 9216


class TestUniformRandomConnections:
    """無制約の一様乱択(先行実験 RAND 腕の意図した配線)"""

    def test_shape_dtype_range(self):
        idx = get_uniform_random_connections(
            64, 32, 2, 'cpu', torch.Generator().manual_seed(0))
        assert idx.shape == (2, 32)
        assert idx.dtype == torch.int64
        assert idx.min() >= 0 and idx.max() < 64

    def test_n_inputs_generalizes(self):
        idx = get_uniform_random_connections(
            64, 32, 4, 'cpu', torch.Generator().manual_seed(0))
        assert idx.shape == (4, 32)

    def test_same_seed_same_wiring(self):
        a = get_uniform_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(3))
        b = get_uniform_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(3))
        assert torch.equal(a, b)
        c = get_uniform_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(4))
        assert not torch.equal(a, c)

    def test_independent_of_global_rng(self):
        torch.manual_seed(11)
        a = get_uniform_random_connections(
            256, 128, 2, 'cpu', torch.Generator().manual_seed(5))
        torch.manual_seed(12)
        b = get_uniform_random_connections(
            256, 128, 2, 'cpu', torch.Generator().manual_seed(5))
        assert torch.equal(a, b)

    def test_is_not_balanced(self):
        """均衡させないので使用回数が 3 回以上の入力と未参照の入力が出る"""
        idx = get_uniform_random_connections(
            9216, 8000, 2, 'cpu', torch.Generator().manual_seed(0))
        counts = torch.bincount(idx.reshape(-1), minlength=9216)
        assert int((counts == 0).sum()) > 0
        assert int((counts >= 3).sum()) > 0

    def test_coverage_matches_analytic_expectation(self):
        """被覆率の期待値 1 - (1 - 1/in)^(n·out) = 0.8244(in=9216, out=8000)"""
        in_dim, out_dim = 9216, 8000
        expected = 1.0 - (1.0 - 1.0 / in_dim) ** (2 * out_dim)
        for seed in range(3):
            idx = get_uniform_random_connections(
                in_dim, out_dim, 2, 'cpu',
                torch.Generator().manual_seed(seed))
            coverage = int(torch.unique(idx).numel()) / in_dim
            assert abs(coverage - expected) < 0.01, (seed, coverage, expected)
        assert abs(expected - 0.8244) < 0.001


class TestConnectionStats:
    """connection_stats(先行実験の配線検証。純粋関数)"""

    def test_known_small_case(self):
        # in_dim=5、4 ゲート: 入力 0 は 3 回、1 は 2 回、2 は 1 回、3/4 は未参照
        conn = torch.tensor([[0, 0, 1, 2],
                             [0, 1, 3, 3]], dtype=torch.int64)
        stats = connection_stats(conn, 5)
        assert set(stats) == {'coverage', 'usage_hist', 'same_input', 'n_gates'}
        assert stats['n_gates'] == 4
        assert stats['usage_hist'] == {'0': 1, '1': 1, '2': 2, '3+': 1}
        assert sum(stats['usage_hist'].values()) == 5
        assert stats['coverage'] == 4 / 5
        assert stats['same_input'] == 1     # ゲート0 が (0, 0)

    def test_rot_layer0_matches_known_coverage(self):
        """R 腕 層0(9216→8000, s_a=3, s_b=96)= 8093/9216 = 87.81%"""
        stats = connection_stats(get_rot_connections(9216, 8000, 3, 96), 9216)
        assert stats['usage_hist'] == {
            '0': 1123, '1': 186, '2': 7907, '3+': 0}
        assert stats['coverage'] == 8093 / 9216
        assert abs(stats['coverage'] - 0.8781) < 0.0001
        assert stats['same_input'] == 0
        assert stats['n_gates'] == 8000

    def test_perm_layer0_is_full_coverage(self):
        stats = connection_stats(
            get_permutation_connections(
                9216, 8000, torch.Generator().manual_seed(0)), 9216)
        assert stats['coverage'] == 1.0
        assert stats['usage_hist']['0'] == 0
        assert stats['usage_hist']['3+'] == 0

    def test_n_inputs_four_counts_duplicate_slots(self):
        conn = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.int64)
        stats = connection_stats(conn, 5)
        assert stats['n_gates'] == 2
        assert stats['same_input'] == 0
        conn2 = torch.tensor([[0, 1], [1, 1], [2, 3], [3, 4]], dtype=torch.int64)
        assert connection_stats(conn2, 5)['same_input'] == 1

    def test_rejects_out_of_range_and_bad_shape(self):
        with pytest.raises(ValueError):
            connection_stats(torch.tensor([[0, 5]], dtype=torch.int64), 5)
        with pytest.raises(ValueError):
            connection_stats(torch.tensor([[0, -1]], dtype=torch.int64), 5)
        with pytest.raises(ValueError):
            connection_stats(torch.tensor([0, 1], dtype=torch.int64), 5)

    def test_is_pure(self):
        """入力テンソルを書き換えない・グローバル RNG を消費しない"""
        conn = get_rot_connections(64, 32, 3, 9)
        before = conn.clone()
        torch.manual_seed(0)
        first = torch.rand(1)
        connection_stats(conn, 64)
        torch.manual_seed(0)
        assert torch.equal(conn, before)
        assert torch.equal(first, torch.rand(1))


class TestUsableAsLutLayerConnections:
    @pytest.mark.parametrize('kind', ['rot', 'perm'])
    def test_lut_layer_accepts(self, kind):
        if kind == 'rot':
            idx = get_rot_connections(64, 32, 3, 9)
        else:
            idx = get_permutation_connections(
                64, 32, torch.Generator().manual_seed(0))
        layer = LutLayer(64, 32, n_inputs=2, connections=idx)
        assert torch.equal(layer.indices, idx)
        layer.eval()
        with torch.no_grad():
            y = layer(torch.rand(4, 64).round())
        assert y.shape == (4, 32)

    def test_full_exp099_rot_schedule_builds(self):
        """B0 の 6 層 rot スケジュールが実寸で通る"""
        for i, (in_dim, out_dim) in enumerate(P6C_D6_DIMS):
            idx = get_rot_connections(
                in_dim, out_dim, P6C_D6_SA[i], P6C_D6_SB[i])
            assert idx.shape == (2, out_dim)
            assert idx.min() >= 0 and idx.max() < in_dim


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
