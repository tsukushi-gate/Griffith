"""L1: 構造化固定配線の生成器の定義。

検証する仕様(移植元は内部調査リポジトリ先行実験。
数式を写し、小さなケースは手計算の期待値と照合する):
    rot(step)  : idx = (g·step + s) mod in_dim。step=1 は従来の rot と bitwise 同一
    rot_n      : idx[m] = (g·step + shifts[m]) mod in_dim
    butterfly  : idx[m] = (g + m·stride) mod in_dim(循環 radix-n 段)。gather(step=n)は
                 n·W ≥ in_dim で全入力を被覆、k 段 stride n^(s−1) は n^k ≥ W で全到達
    local1d    : idx[m] = (c_g + m·stride) mod in_dim、c_g = round(g·(in−1)/(out−1))
    local2d    : トーラス格子のオフセット読み(層 0 は画像格子 × セル当たりビット)
    dedupe     : 重複ピンを +1 ずらしで潰す(先行実験 _fix_degenerate / 先行実験 dedupe_ins)
"""

import numpy as np
import pytest
import torch

from griffith_logic.core.connections import (
    butterfly_strides,
    connection_stats,
    dedupe_connections,
    get_butterfly_connections,
    get_local1d_connections,
    get_local2d_connections,
    get_rot_connections,
    get_rot_n_connections,
    local1d_strides,
    local2d_offset_schedule,
    prop_map,
    ring_offsets,
)

# 先行実験 C2D_OFFSETS(隠れ層の 2D オフセット巡回)。ring 列の先頭 6 個と一致するべき値
EXP075_C2D_OFFSETS = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)]


def _reach_sets(layers):
    """層ごとの配線 (n, W_l) を前向きに辿り、最終層の各出力が到達する層 0 入力の集合"""
    first = layers[0]
    cur = [set(first[:, g].tolist()) for g in range(first.shape[1])]
    for idx in layers[1:]:
        cur = [set().union(*(cur[int(src)] for src in idx[:, g].tolist()))
               for g in range(idx.shape[1])]
    return cur


def _columns_distinct(idx: torch.Tensor) -> bool:
    srt = idx.sort(dim=0).values
    return bool((srt[1:] != srt[:-1]).all())


class TestRotStep:
    @pytest.mark.parametrize(
        ('in_dim', 'out_dim', 's_a', 's_b'),
        [(9216, 8000, 3, 96), (7, 20, 5, 13), (10, 10, -3, 27)],
    )
    def test_step_one_is_bitwise_identical_to_rot(self, in_dim, out_dim, s_a, s_b):
        assert torch.equal(get_rot_connections(in_dim, out_dim, s_a, s_b, step=1),
                           get_rot_connections(in_dim, out_dim, s_a, s_b))

    @pytest.mark.parametrize(
        ('in_dim', 'out_dim', 's_a', 's_b', 'step'),
        [(96, 24, 0, 1, 4), (50, 20, 0, 7, 3), (12, 30, 2, 5, 5)],
    )
    def test_step_formula(self, in_dim, out_dim, s_a, s_b, step):
        """先行実験 rot2: idx = (g·r + s) mod W_in"""
        idx = get_rot_connections(in_dim, out_dim, s_a, s_b, step=step)
        assert idx.shape == (2, out_dim)
        assert idx.dtype == torch.int64
        for g in range(out_dim):
            assert int(idx[0, g]) == (g * step + s_a) % in_dim
            assert int(idx[1, g]) == (g * step + s_b) % in_dim

    def test_step_spreads_gates_over_wide_input(self):
        """前層の幅 96 > 出力幅 24(先行実験の conv 頭部): step=4 なら読む範囲が入力全体
        [0, 93] に広がる(rot は先頭 [0, 24] だけ)。2 入力 × 24 ゲートなので被覆は 48 本"""
        wide = get_rot_connections(96, 24, 0, 1, step=4)
        assert int(wide.max()) == 93
        assert connection_stats(wide, 96)['coverage'] == 48 / 96
        narrow = get_rot_connections(96, 24, 0, 1)
        assert int(narrow.max()) == 24
        assert connection_stats(narrow, 96)['coverage'] == 25 / 96


class TestRotN:
    def test_two_shifts_match_rot(self):
        assert torch.equal(get_rot_n_connections(20, 8, [3, 5], step=2),
                           get_rot_connections(20, 8, 3, 5, step=2))

    def test_formula_n4(self):
        idx = get_rot_n_connections(11, 6, [0, 2, 7, -1])
        assert idx.shape == (4, 6)
        for g in range(6):
            assert idx[:, g].tolist() == [g % 11, (g + 2) % 11, (g + 7) % 11, (g - 1) % 11]

    def test_gather_covers_every_input_three_times(self):
        """先行実験 bf6_gather の小ケース: in=24, W=12, n=6 → 各入力 3 回ずつ"""
        idx = get_rot_n_connections(24, 12, range(6), step=6)
        for i in range(12):
            assert idx[:, i].tolist() == [(6 * i + m) % 24 for m in range(6)]
        stats = connection_stats(idx, 24)
        assert stats['coverage'] == 1.0
        assert stats['usage_hist'] == {'0': 0, '1': 0, '2': 0, '3+': 24}
        assert stats['same_input'] == 0

    def test_empty_shifts_rejected(self):
        with pytest.raises(ValueError):
            get_rot_n_connections(8, 4, [])


class TestPropMap:
    def test_identity_when_dims_equal(self):
        assert torch.equal(prop_map(10, 10), torch.arange(10))

    def test_endpoints_and_monotone(self):
        c = prop_map(8, 20)
        assert int(c[0]) == 0
        assert int(c[-1]) == 19
        assert bool((c[1:] >= c[:-1]).all())

    def test_single_gate(self):
        assert prop_map(1, 20).tolist() == [0]

    def test_matches_numpy_formula_exp088_layer0(self):
        """先行実験 L2D6 層 0 の実寸(W=330 → 784 画素)で numpy の式と bitwise 一致"""
        i = np.arange(330, dtype=np.int64)
        expected = np.round(i.astype(np.float64) * (784 - 1) / (330 - 1)).astype(np.int64)
        assert prop_map(330, 784).numpy().tolist() == expected.tolist()


class TestButterfly:
    def test_formula(self):
        idx = get_butterfly_connections(12, 12, 3, 4)
        for g in range(12):
            assert idx[:, g].tolist() == [g, (g + 4) % 12, (g + 8) % 12]

    @pytest.mark.parametrize(('w', 'n', 'stride'),
                             [(12, 6, 1), (24, 6, 6), (16, 4, 4), (8, 2, 1)])
    def test_square_stage_is_fanout_balanced(self, w, n, stride):
        idx = get_butterfly_connections(w, w, n, stride)
        counts = torch.bincount(idx.reshape(-1), minlength=w)
        assert torch.equal(counts, torch.full((w,), n))

    @pytest.mark.parametrize(('w', 'n'), [(8, 2), (9, 3), (16, 4), (12, 6), (36, 6)])
    def test_reachability_closed_form(self, w, n):
        """k 段(stride n^(s−1))で各出力 g の到達集合は {g + t mod W : 0 ≤ t < n^k}"""
        layers = []
        k = 0
        while n ** k < w:
            layers.append(get_butterfly_connections(w, w, n, n ** k))
            k += 1
            reach = _reach_sets(layers)
            for g in range(w):
                expected = {(g + t) % w for t in range(min(n ** k, w))}
                assert reach[g] == expected, (w, n, k, g)
        assert all(len(r) == w for r in _reach_sets(layers))

    def test_strides_schedule(self):
        assert butterfly_strides(4, 6) == [1, 6, 36, 216]
        assert butterfly_strides(3, 4, 'desc') == [16, 4, 1]
        assert butterfly_strides(0, 2) == []
        with pytest.raises(ValueError):
            butterfly_strides(2, 2, 'random')

    def test_exp088_w12_stride6_degenerate_and_dedupe(self):
        """先行実験の小ケース W=12, n=6, stride 6: 6m mod 12 は 2 値しか取らず重複する。
        dedupe で各ゲートが相異なる 6 入力になる(先行実験 build() の挙動)"""
        raw = get_butterfly_connections(12, 12, 6, 6)
        assert raw[:, 0].tolist() == [0, 6, 0, 6, 0, 6]
        assert connection_stats(raw, 12)['same_input'] == 12
        fixed = dedupe_connections(raw, 12)
        assert fixed[:, 0].tolist() == [0, 6, 1, 7, 2, 8]
        assert connection_stats(fixed, 12)['same_input'] == 0
        assert _columns_distinct(fixed)


class TestLocal1d:
    def test_formula_with_prop_map_centers(self):
        in_dim, out_dim, stride = 20, 8, 3
        idx = get_local1d_connections(in_dim, out_dim, 2, stride)
        c = prop_map(out_dim, in_dim)
        for g in range(out_dim):
            assert idx[:, g].tolist() == [int(c[g]), (int(c[g]) + stride) % in_dim]

    def test_square_matches_rot(self):
        assert torch.equal(get_local1d_connections(16, 16, 2, 5),
                           get_rot_connections(16, 16, 0, 5))

    def test_n_inputs_general(self):
        idx = get_local1d_connections(9, 9, 4, 2)
        assert idx[:, 7].tolist() == [7, 0, 2, 4]

    def test_geometric_stride_schedule(self):
        assert local1d_strides(5, 100, frac=0.16) == [1, 2, 4, 8, 16]
        assert local1d_strides(1, 100) == [1]
        s = local1d_strides(6, 2000)        # 先行実験の既定(幅 2000、13% = S_max 260)
        assert len(s) == 6
        assert s[0] == 1
        assert s[-1] == 260
        assert s == sorted(s)


class TestLocal2d:
    def test_ring_offsets(self):
        assert ring_offsets(8) == [(1, 0), (0, 1), (1, 1), (1, -1),
                                   (2, 0), (0, 2), (2, 2), (2, -2)]
        assert ring_offsets(6) == EXP075_C2D_OFFSETS
        assert ring_offsets(0) == []

    def test_contig_schedule_n4_is_one_ring_per_layer(self):
        """先行実験 L2D4: contig で n=4 なら層 l が半径 l+1 のリング 1 周"""
        sched = local2d_offset_schedule(3, 4, 'contig')
        for layer, offs in enumerate(sched):
            s = layer + 1
            assert offs == [(s, 0), (0, s), (s, s), (s, -s)]

    def test_stride_schedule(self):
        seq = ring_offsets(6)
        assert local2d_offset_schedule(3, 2, 'stride') == [
            [seq[0], seq[3]], [seq[1], seq[4]], [seq[2], seq[5]]]
        with pytest.raises(ValueError):
            local2d_offset_schedule(2, 2, 'spiral')

    def test_hidden_square_grid_torus(self):
        """3×3 格子(in=out=9): 右隣・下隣。端はトーラスで巻く"""
        idx = get_local2d_connections(9, 9, [(1, 0), (0, 1)])
        assert idx.shape == (2, 9)
        assert idx[:, 4].tolist() == [5, 7]
        assert idx[:, 2].tolist() == [0, 5]
        assert idx[:, 8].tolist() == [6, 2]

    def test_ragged_grid_wraps_modulo_cells(self):
        """W=7 は G=3 の格子に 2 セル余る。はみ出したセルは mod 7 で巻き戻す(先行実験の % W)"""
        idx = get_local2d_connections(7, 7, [(1, 0), (0, 1)])
        assert idx[:, 5].tolist() == [3, 1]
        assert idx[:, 6].tolist() == [0, 0]       # 重複(dedupe で潰せる)
        assert dedupe_connections(idx, 7)[:, 6].tolist() == [0, 1]

    def test_image_layer0_bits_feature_major(self):
        """格子 (2,3) × 2 bit(in=12)、出力 6(比例写像は恒等): 右隣・下隣の 2 セル × 2 bit"""
        idx = get_local2d_connections(12, 6, [(1, 0), (0, 1)], grid=(2, 3), bits_per_cell=2)
        assert idx.shape == (4, 6)
        assert idx[:, 0].tolist() == [2, 3, 6, 7]
        assert idx[:, 2].tolist() == [0, 1, 10, 11]
        assert idx[:, 5].tolist() == [6, 7, 4, 5]

    def test_l2d4_layer0_self_cell(self):
        """先行実験 L2D4 層 0: 自画素 × 4 bit(in=16 = 2×2 格子 × 4 bit)"""
        idx = get_local2d_connections(16, 4, [(0, 0)], grid=(2, 2), bits_per_cell=4)
        for g in range(4):
            assert idx[:, g].tolist() == [4 * g + t for t in range(4)]

    def test_exp088_l2d6_layer0_pixel_coverage(self):
        """先行実験 L2D6 層 0 の実寸(W=330、28×28×3 = 2352 bit、右隣 + 下隣 × 3 bit):
        被覆画素は 586 / 784 = 74.74%(先行実験の記録値)"""
        idx = get_local2d_connections(
            2352, 330, [(1, 0), (0, 1)], grid=(28, 28), bits_per_cell=3)
        assert idx.shape == (6, 330)
        assert int(torch.unique(idx // 3).numel()) == 586
        for g in (0, 100, 329):
            col = idx[:, g].tolist()
            assert col[0] % 3 == 0
            assert col[0:3] == [col[0], col[0] + 1, col[0] + 2]
            assert col[3] % 3 == 0
            assert col[3:6] == [col[3], col[3] + 1, col[3] + 2]

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            get_local2d_connections(10, 4, [(1, 0)], bits_per_cell=3)   # 10 は 3 の倍数でない
        with pytest.raises(ValueError):
            get_local2d_connections(9, 4, [(1, 0)], grid=(2, 2))         # 格子が小さい
        with pytest.raises(ValueError):
            get_local2d_connections(9, 4, [])


class TestDedupe:
    def test_two_input_rule(self):
        idx = torch.tensor([[0, 3, 4], [0, 3, 1]])
        assert dedupe_connections(idx, 5).tolist() == [[0, 3, 4], [1, 4, 1]]

    def test_n_input_walks_forward_and_wraps(self):
        idx = torch.tensor([[4], [4], [4]])
        assert dedupe_connections(idx, 5).tolist() == [[4], [0], [1]]

    def test_unchanged_when_distinct_and_does_not_mutate(self):
        idx = torch.tensor([[0, 1], [2, 3]])
        out = dedupe_connections(idx, 4)
        assert torch.equal(out, idx)
        assert out is not idx

    def test_rejects(self):
        with pytest.raises(ValueError):
            dedupe_connections(torch.zeros(3, 2, dtype=torch.int64), 2)   # n_inputs > in_dim
        with pytest.raises(ValueError):
            dedupe_connections(torch.zeros(4, dtype=torch.int64), 4)


class TestConnectionStatsOnStructured:
    @pytest.mark.parametrize(('idx', 'in_dim'), [
        (get_butterfly_connections(24, 24, 6, 6), 24),
        (get_local1d_connections(30, 12, 4, 2), 30),
        (get_local2d_connections(48, 10, [(1, 0), (0, 1)], grid=(4, 4), bits_per_cell=3), 48),
        (get_rot_n_connections(40, 10, range(4), step=4), 40),
    ])
    def test_stats_run_on_every_kind(self, idx, in_dim):
        stats = connection_stats(idx, in_dim)
        assert stats['n_gates'] == idx.shape[1]
        assert sum(stats['usage_hist'].values()) == in_dim
        assert 0.0 < stats['coverage'] <= 1.0


class TestMakeConnections:
    """make_connections: 配線種の単一の入口が各生成器と一致し、引数を検査する"""

    def test_structured_kinds_match_generators(self):
        from griffith_logic.core.connections import make_connections
        cases = [
            ('rot', dict(s_a=0, s_b=1, step=3), 32, 64, 2,
             get_rot_connections(32, 64, 0, 1, 'cpu', step=3)),
            ('rot_n', dict(shifts=[0, 1, 2], step=3), 30, 10, 3,
             get_rot_n_connections(30, 10, [0, 1, 2], 'cpu', step=3)),
            ('butterfly', dict(stride=4), 64, 64, 2, get_butterfly_connections(64, 64, 2, 4)),
            ('local1d', dict(stride=5), 64, 40, 2, get_local1d_connections(64, 40, 2, 5)),
            ('local2d', dict(offsets=[[0, 0], [0, 1]], grid=[8, 8]), 64, 40, 2,
             get_local2d_connections(64, 40, [(0, 0), (0, 1)], grid=(8, 8))),
        ]
        for kind, spec, in_dim, out_dim, n, want in cases:
            got = make_connections(kind, in_dim, out_dim, n, **spec)
            assert torch.equal(got, want), kind

    def test_random_kinds_are_seeded(self):
        from griffith_logic.core.connections import get_random_connections, make_connections
        a = make_connections('random', 32, 64, 2, seed=5)
        b = make_connections('random', 32, 64, 2, seed=5)
        want = get_random_connections(32, 64, 2, 'cpu', torch.Generator().manual_seed(5))
        assert torch.equal(a, b) and torch.equal(a, want)
        assert make_connections('unique', 8, 4, 2).shape == (2, 4)

    def test_dedupe_flag(self):
        from griffith_logic.core.connections import make_connections
        # stride ≡ 0 (mod in_dim) なら全スロットが同じ入力を読む(重複の極端な例)
        raw = make_connections('butterfly', 12, 12, 2, stride=12)
        assert (raw[0] == raw[1]).all()
        fixed = make_connections('butterfly', 12, 12, 2, stride=12, dedupe=True)
        assert not (fixed[0] == fixed[1]).any()
        assert torch.equal(fixed, dedupe_connections(raw, 12))

    def test_argument_checks(self):
        from griffith_logic.core.connections import make_connections
        with pytest.raises(ValueError, match='未知の配線種'):
            make_connections('spiral', 8, 8, 2)
        with pytest.raises(ValueError, match='shifts が必要'):
            make_connections('rot_n', 8, 8, 2, step=2)
        with pytest.raises(ValueError, match='受け付けません'):
            make_connections('butterfly', 8, 8, 2, stride=1, grid=[2, 2])
        with pytest.raises(ValueError, match='一致しません'):
            make_connections('rot_n', 9, 3, 2, shifts=[0, 1, 2])
        # None の引数は未指定扱い(LayerConnectionsConfig の全欄をそのまま渡せる)
        assert make_connections('rot', 8, 8, 2, s_a=0, s_b=1, step=None).shape == (2, 8)
