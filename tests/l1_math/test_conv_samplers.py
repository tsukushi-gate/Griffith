"""L1: conv-DLGN の葉サンプラ(conv_samplers.py)の定義。

移植元は内部調査リポジトリ先行実験 `samplers.py`。検証する性質:
    - 座標は窓内 [0, K)、チャネルは [0, C_in)
    - 計算可能腕の被覆: dilated8(位相 4 × drop 9)/ cross5(位相 25)/ checker13(位相 2)は
      チャネル群で 5×5 窓の全 25 画素を読む
    - 半割ペアリングの対(葉 j と j + L/2)は同一読みにならない(退化ゲート無し)
    - random8 の衝突回避と generator 決定性
    - ビット反転置換で先行実験の葉順(半割)を LutConv2d の葉順(隣接)へ写す
"""

import pytest
import torch

from griffith_logic.conv_samplers import (
    SAMPLER_SPECS,
    bit_reversal_permutation,
    build_leaf_selection,
    channel_rule_computable,
    checker13_points,
    cross5_points,
    dilated8_points,
    leaf_indices_flat,
    sampler_random8,
)

PATTERN_ARMS = ['cross5', 'dilated8', 'checker13']


def _window_points(sel_dy, sel_dx):
    return set(zip(sel_dy.reshape(-1).tolist(), sel_dx.reshape(-1).tolist(), strict=True))


class TestSpecs:
    def test_table(self):
        assert SAMPLER_SPECS['random8'] == {
            'kernel': 3, 'depth': 3, 'padding': 1, 'computable': False, 'n_points': None}
        assert SAMPLER_SPECS['cross5']['n_points'] == 5
        assert SAMPLER_SPECS['dilated8']['n_points'] == 8
        assert SAMPLER_SPECS['checker13'] == {
            'kernel': 5, 'depth': 4, 'padding': 2, 'computable': True, 'n_points': 13}
        for spec in SAMPLER_SPECS.values():
            assert spec['padding'] == (spec['kernel'] - 1) // 2

    def test_unknown_name(self):
        with pytest.raises(ValueError):
            build_leaf_selection('grid9', 4, 2)


class TestPoints:
    def test_dilated8_points_drop_varies(self):
        pts0 = dilated8_points(0, 0, 0)
        assert len(pts0) == 8
        assert len(set(pts0)) == 8
        assert all(0 <= dy < 5 and 0 <= dx < 5 for dy, dx in pts0)
        assert (0, 0) not in pts0                       # drop=0 は raster 先頭 (0,0)
        assert (2, 2) not in dilated8_points(0, 0, 4)   # drop=4 は中心
        # 全 drop の和集合は 2 マス空け格子 9 点
        union = set().union(*(set(dilated8_points(0, 0, d)) for d in range(9)))
        assert union == {(2 * i, 2 * j) for i in range(3) for j in range(3)}

    def test_cross5_points(self):
        assert set(cross5_points(0, 0)) == {(0, 2), (2, 0), (2, 2), (2, 4), (4, 2)}
        shifted = set(cross5_points(1, 3))
        assert shifted == {((dy + 1) % 5, (dx + 3) % 5) for dy, dx in cross5_points(0, 0)}

    def test_checker13_points(self):
        even, odd = set(checker13_points(0)), set(checker13_points(1))
        assert len(even) == 13
        assert len(odd) == 13                  # 奇数 12 点 + 中心
        assert (2, 2) in even and (2, 2) in odd
        assert even | odd == {(dy, dx) for dy in range(5) for dx in range(5)}


class TestChannelRule:
    def test_computable_reads_pair_alternately(self):
        c_in, num_chn = 8, 2
        ch = channel_rule_computable(c_out=16, c_in=c_in, n_leaves=8, layer_idx=1, num_chn=num_chn)
        assert ch.shape == (16, 8)
        for c in range(16):
            base = (c + 1) % c_in
            pair = {base, (base + c_in // 2) % c_in}
            assert set(ch[c].tolist()) <= pair
            # 葉ごとに交互
            picks = [(j + c) % num_chn for j in range(8)]
            expected = [[base, (base + c_in // 2) % c_in][p] for p in picks]
            assert ch[c].tolist() == expected


class TestPatternCoverage:
    def test_dilated8_covers_window_over_phases_and_drops(self):
        """位相 4 × drop 9 = 36 チャネルで 5×5 の全 25 画素を読む(drop 固定だと 21/25)"""
        _, dy, dx = build_leaf_selection('dilated8', c_out=36, c_in=4)
        assert len(_window_points(dy, dx)) == 25

    def test_cross5_covers_window_over_25_phases(self):
        _, dy, dx = build_leaf_selection('cross5', c_out=25, c_in=4)
        assert len(_window_points(dy, dx)) == 25
        # 1 チャネルは 5 点を 8 葉で読む(3 葉は重複読み)
        assert len(set(zip(dy[0].tolist(), dx[0].tolist(), strict=True))) == 5

    def test_checker13_two_phases_cover_window(self):
        _, dy, dx = build_leaf_selection('checker13', c_out=2, c_in=4)
        assert dy.shape == (2, 16)
        assert len(_window_points(dy, dx)) == 25

    @pytest.mark.parametrize('arm', PATTERN_ARMS)
    @pytest.mark.parametrize('layer_idx', [0, 1, 2])
    def test_half_split_pairs_never_read_same_point(self, arm, layer_idx):
        """半割ペアリングの対(葉 j, j + L/2)は (ch, dy, dx) が常に異なる = 退化ゲート無し"""
        ch, dy, dx = build_leaf_selection(arm, c_out=40, c_in=6, layer_idx=layer_idx)
        half = ch.shape[1] // 2
        same = ((ch[:, :half] == ch[:, half:]) & (dy[:, :half] == dy[:, half:])
                & (dx[:, :half] == dx[:, half:]))
        assert not bool(same.any())

    @pytest.mark.parametrize('arm', PATTERN_ARMS)
    def test_ranges_and_determinism(self, arm):
        a = build_leaf_selection(arm, c_out=12, c_in=5, layer_idx=2)
        b = build_leaf_selection(arm, c_out=12, c_in=5, layer_idx=2)
        k = SAMPLER_SPECS[arm]['kernel']
        for t in a:
            assert int(t.min()) >= 0
        assert int(a[0].max()) < 5
        assert int(a[1].max()) < k and int(a[2].max()) < k
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        # 層が変わると位相が変わる(全く同じにはならない)
        c = build_leaf_selection(arm, c_out=12, c_in=5, layer_idx=3)
        assert not all(torch.equal(x, y) for x, y in zip(a, c, strict=True))


class TestRandom8:
    def test_shapes_ranges_and_collision_free_pairs(self):
        g = torch.Generator().manual_seed(0)
        ch, dy, dx = sampler_random8(c_out=64, c_in=9, generator=g)
        assert ch.shape == dy.shape == dx.shape == (64, 8)
        assert int(ch.max()) < 9 and int(dy.max()) < 3 and int(dx.max()) < 3
        same = ((ch[:, :4] == ch[:, 4:]) & (dy[:, :4] == dy[:, 4:]) & (dx[:, :4] == dx[:, 4:]))
        assert not bool(same.any())

    def test_at_most_num_chn_channels_per_tree(self):
        g = torch.Generator().manual_seed(1)
        ch, _, _ = sampler_random8(c_out=32, c_in=9, num_chn=2, generator=g)
        assert all(len(set(row)) <= 2 for row in ch.tolist())

    def test_generator_determinism(self):
        a = sampler_random8(16, 4, generator=torch.Generator().manual_seed(7))
        b = sampler_random8(16, 4, generator=torch.Generator().manual_seed(7))
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


class TestLeafOrderMapping:
    def test_bit_reversal(self):
        assert bit_reversal_permutation(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]
        assert bit_reversal_permutation(16).tolist()[:4] == [0, 8, 4, 12]
        assert bit_reversal_permutation(2).tolist() == [0, 1]
        with pytest.raises(ValueError):
            bit_reversal_permutation(6)

    def test_leaf_indices_flat_places_half_split_pairs_adjacently(self):
        """先行実験の葉 j は LutConv2d の葉 bitrev(j) へ。対 (j, j+4) が隣接対 (2i, 2i+1) になる"""
        ch = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0]])
        dy = torch.tensor([[0, 0, 1, 1, 2, 2, 0, 1]])
        dx = torch.tensor([[0, 1, 2, 0, 1, 2, 2, 1]])
        flat = leaf_indices_flat(ch, dy, dx, 3, 3)
        assert flat.shape == (1, 8)
        src = (ch * 9 + dy * 3 + dx)[0].tolist()
        perm = bit_reversal_permutation(8).tolist()
        for j in range(8):
            assert int(flat[0, perm[j]]) == src[j]
        # 半割の対 {(j, j+4)} の集合 = 写した後の隣接対 {(2i, 2i+1)} の集合
        half_split = {(src[j], src[j + 4]) for j in range(4)}
        adjacent = {(int(flat[0, 2 * i]), int(flat[0, 2 * i + 1])) for i in range(4)}
        assert adjacent == half_split

    @pytest.mark.parametrize('arm', PATTERN_ARMS + ['random8'])
    def test_adjacent_leaves_differ_after_mapping(self, arm):
        """写した後の隣接対(LutConv2d の対)が同一添字を読まない"""
        g = torch.Generator().manual_seed(3)
        ch, dy, dx = build_leaf_selection(arm, c_out=30, c_in=6, layer_idx=1, generator=g)
        k = SAMPLER_SPECS[arm]['kernel']
        flat = leaf_indices_flat(ch, dy, dx, k, k)
        assert not bool((flat[:, 0::2] == flat[:, 1::2]).any())
