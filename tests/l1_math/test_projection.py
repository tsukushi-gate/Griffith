"""with_input_projection の性質テスト(L1、純numpy検証)。

射影の定義: グループg(温度計しきい値昇順)の出力ビットi =
1[popcount(グループ入力) >= i+1]。性質:
    - 有効符号語(単調列)上では恒等
    - 無効パターンは同popcountの有効符号語へ射影(単調列化)
    - グループ外ビットは恒等通過
    - 純関数(元グラフは不変)
"""

import numpy as np
import pytest

from griffith_logic.export.abc_opt import valid_codewords, with_input_projection
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import tt_to_hex
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

WIDTH = 6
GROUPS = [[0, 1, 2], [4, 5]]  # bit 3はグループ外(恒等通過)


def _identity_graph(width: int) -> IRGraph:
    """x → 恒等lut_layer → y(射影層の出力をsimulateで観測するための土台)"""
    node = IRNode(
        id='ident',
        type='lut_layer',
        params={
            'in_width': width,
            'out_width': width,
            'n_inputs': 1,
            'indices': np.arange(width, dtype=np.int64)[None, :],
            'truth_tables': tt_to_hex([0x2] * width),
        },
    )
    graph = IRGraph(
        inputs=[IRPort(id='x', width=width)],
        outputs=[IRPort(id='y', width=width)],
        nodes=[node],
        edges=[IREdge(src='x', dst='ident'), IREdge(src='ident', dst='y')],
    )
    validate(graph)
    return graph


def _all_patterns(width: int) -> np.ndarray:
    n = 1 << width
    k = np.arange(n)[:, None]
    return ((k >> np.arange(width)[None, :]) & 1).astype(np.int64)


def _expected_projection(x: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """素朴照合: 各グループをpopcount保存の温度計符号語へ射影する。"""
    out = x.copy()
    for g in groups:
        pc = x[:, g].sum(axis=1)
        for i, bit in enumerate(g):
            out[:, bit] = (pc >= i + 1).astype(np.int64)
    return out


class TestWithInputProjection:
    def test_identity_on_valid_codewords(self):
        graph = with_input_projection(_identity_graph(WIDTH), GROUPS)
        rng = np.random.default_rng(0)
        x = valid_codewords(rng, 256, WIDTH, GROUPS)
        assert np.array_equal(simulate(graph, x), x), (
            '有効符号語上で恒等でない'
        )

    def test_projects_all_patterns_exhaustive(self):
        graph = with_input_projection(_identity_graph(WIDTH), GROUPS)
        x = _all_patterns(WIDTH)  # 無効パターン含む全64通り
        assert np.array_equal(simulate(graph, x), _expected_projection(x, GROUPS))

    def test_output_is_always_valid_codeword(self):
        """射影出力は必ず単調列(冪等: 射影の射影=射影)"""
        graph = with_input_projection(_identity_graph(WIDTH), GROUPS)
        y = simulate(graph, _all_patterns(WIDTH))
        for g in GROUPS:
            bits = y[:, g]
            # 単調非増加(しきい値昇順なので後ろのビットほど立ちにくい)
            assert (np.diff(bits, axis=1) <= 0).all(), f'グループ{g}が非単調'
        assert np.array_equal(simulate(graph, y), y), '射影が冪等でない'

    def test_pure_function(self):
        base = _identity_graph(WIDTH)
        n_nodes, n_edges = len(base.nodes), len(base.edges)
        x = _all_patterns(WIDTH)
        before = simulate(base, x)
        projected = with_input_projection(base, GROUPS)
        assert len(base.nodes) == n_nodes and len(base.edges) == n_edges
        assert np.array_equal(simulate(base, x), before), '元グラフが変化した'
        assert projected is not base
        validate(base)

    def test_downstream_indices_unchanged(self):
        """射影層の出力ビット順=入力ビット順(後続indicesの変更不要)"""
        base = _identity_graph(WIDTH)
        projected = with_input_projection(base, GROUPS)
        ident = projected.node_by_id('ident')
        assert np.array_equal(
            np.asarray(ident.params['indices']),
            np.asarray(base.node_by_id('ident').params['indices']),
        )

    def test_rejects_overlapping_groups(self):
        with pytest.raises(AssertionError):
            with_input_projection(_identity_graph(WIDTH), [[0, 1], [1, 2]])

    def test_rejects_out_of_range_bits(self):
        with pytest.raises(AssertionError):
            with_input_projection(_identity_graph(WIDTH), [[4, 5, 6]])


class TestValidCodewords:
    def test_shapes_and_monotonicity(self):
        rng = np.random.default_rng(1)
        x = valid_codewords(rng, 512, WIDTH, GROUPS)
        assert x.shape == (512, WIDTH)
        assert set(np.unique(x)).issubset({0, 1})
        for g in GROUPS:
            assert (np.diff(x[:, g], axis=1) <= 0).all()

    def test_all_levels_reachable(self):
        rng = np.random.default_rng(2)
        x = valid_codewords(rng, 2048, WIDTH, GROUPS)
        for g in GROUPS:
            levels = set(x[:, g].sum(axis=1).tolist())
            assert levels == set(range(len(g) + 1)), (
                f'グループ{g}のレベルが網羅されていない: {levels}'
            )
