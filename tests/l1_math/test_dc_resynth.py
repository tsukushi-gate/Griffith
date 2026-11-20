"""don't care 再合成の健全性テスト。

守るべき性質:
    1. care set 上では元の木と**必ず**一致する(全数照合)
    2. 差異は don't care パターンでのみ起こりうる
    3. care=None(全パターン)なら厳密等価 = lossless
    4. ゲート数は決して増えない(単調性)
    5. numpy高速フォワード ``forward_graph`` は ``ir/simulate`` と整数一致
"""
import numpy as np
import pytest

from griffith_logic.export.dc_resynth import (
    embed_balanced,
    essential_vars,
    eval_tree,
    forward_graph,
    gate2_mask,
    resynth_node_params,
    resynth_tree,
    synth_tree,
    tree_cost,
    tree_depth,
    tree_function,
)
from griffith_logic.export.optimize import (
    ConvTreeSpec,
    build_conv_program,
    spec_from_node_params,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import tt_to_hex
from griffith_logic.ir.simulate import simulate

DEPTH = 3
N_LEAVES = 1 << DEPTH
N_INTERNAL = N_LEAVES - 1


def _random_tree(rng, n_taps=8):
    """ランダムな (indices_row, tts_row)。タップIDは 0..(n_taps-1)。"""
    row = [int(v) for v in rng.integers(0, n_taps, size=N_LEAVES)]
    tts = [int(v) for v in rng.integers(0, 16, size=N_INTERNAL)]
    return row, tts


def _mask_of(taps, u):
    """taps の各IDへ 2^u パターン上の値マスクを割り当てる。"""
    n_pat = 1 << u
    out = {}
    for v, pp in enumerate(taps):
        mm = 0
        for p in range(n_pat):
            if (p >> v) & 1:
                mm |= 1 << p
        out[pp] = mm
    return out, n_pat, (1 << n_pat) - 1


def _solo_gates(row, tts):
    spec = ConvTreeSpec(out_channels=1, tree_depth=DEPTH,
                        indices_flat=[list(row)],
                        truth_tables=[t & 0xf for t in tts])
    return build_conv_program(spec).stats.gates_after


class TestSynthCore:
    def test_gate2_mask_matches_scalar_definition(self):
        """gate2_mask のビット規約が bit index=(b<<1)|a と一致する。"""
        for tt in range(16):
            a = 0b1010
            b = 0b1100
            got = gate2_mask(tt, a, b, 0b1111)
            for i in range(4):
                av = (a >> i) & 1
                bv = (b >> i) & 1
                want = (tt >> (((bv & 1) << 1) | (av & 1))) & 1
                assert ((got >> i) & 1) == want, (tt, i)

    def test_synth_finds_exact_minimum_for_known_functions(self):
        """既知の最小ゲート数を持つ関数で cost が一致する。"""
        u = 3
        n_pat = 1 << u
        full = (1 << n_pat) - 1
        cols = [sum(1 << p for p in range(n_pat) if (p >> v) & 1)
                for v in range(u)]
        # a  -> 0ゲート
        assert synth_tree(cols[0], cols, full).cost == 0
        # a & b -> 1ゲート
        assert synth_tree(cols[0] & cols[1], cols, full).cost == 1
        # (a&b)|(a&c) = a&(b|c) -> 2ゲート(素朴には3)
        target = (cols[0] & cols[1]) | (cols[0] & cols[2])
        assert synth_tree(target, cols, full).cost == 2

    def test_synth_result_is_correct_on_care(self):
        """探索が返す木は care 上で target と一致する(ランダム全数)。"""
        rng = np.random.default_rng(0)
        for _ in range(200):
            u = int(rng.integers(2, 5))
            n_pat = 1 << u
            full = (1 << n_pat) - 1
            cols = [sum(1 << p for p in range(n_pat) if (p >> v) & 1)
                    for v in range(u)]
            target = int(rng.integers(0, n_pat and (1 << n_pat)))
            res = synth_tree(target, cols, full)
            if res is None:
                continue
            got = _eval_solution(res.tree, cols, full)
            assert got == target, (u, target, res.cost)


def _eval_solution(node, cols, full):
    if node[0] == 'c':
        return full if node[1] else 0
    if node[0] == 't':
        return cols[node[1]]
    return gate2_mask(node[1], _eval_solution(node[2], cols, full),
                      _eval_solution(node[3], cols, full), full)


class TestEmbedding:
    def test_embedded_tree_reproduces_solution(self):
        """embed_balanced が解の木と同じ関数のバランス木を返す。"""
        rng = np.random.default_rng(1)
        for _ in range(200):
            row, tts = _random_tree(rng)
            taps, onset = tree_function(row, tts, DEPTH)
            u = len(taps)
            mask_of, n_pat, full = _mask_of(taps, u)
            cols = [mask_of[pp] for pp in taps]
            res = synth_tree(onset, cols, full)
            if res is None:
                continue
            assert tree_depth(res.tree) <= DEPTH
            leaves, new_tts = embed_balanced(res.tree, taps, DEPTH)
            assert len(leaves) == N_LEAVES and len(new_tts) == N_INTERNAL
            assert eval_tree(leaves, new_tts, DEPTH, mask_of, full) == onset

    def test_embedded_gate_count_equals_tree_cost(self):
        """埋め込み後にA1縮約したゲート数が tree_cost と一致する(パススルーが畳まれる)。"""
        rng = np.random.default_rng(2)
        checked = 0
        for _ in range(200):
            row, tts = _random_tree(rng)
            taps, onset = tree_function(row, tts, DEPTH)
            u = len(taps)
            mask_of, _n, full = _mask_of(taps, u)
            cols = [mask_of[pp] for pp in taps]
            res = synth_tree(onset, cols, full)
            if res is None:
                continue
            leaves, new_tts = embed_balanced(res.tree, taps, DEPTH)
            assert _solo_gates(leaves, new_tts) == tree_cost(res.tree)
            checked += 1
        assert checked > 50


class TestResynthTree:
    def test_lossless_is_exactly_equivalent(self):
        """care=None なら全 2^u パターンで元の木と一致する。"""
        rng = np.random.default_rng(3)
        for _ in range(300):
            row, tts = _random_tree(rng)
            leaves, new_tts, before, after = resynth_tree(row, tts, DEPTH, None)
            taps, _ = tree_function(row, tts, DEPTH)
            mask_of, _n, full = _mask_of(taps, len(taps))
            want = eval_tree(row, tts, DEPTH, mask_of, full)
            assert set(leaves) <= set(taps), '新しい木が元に無いタップを使った'
            assert eval_tree(leaves, new_tts, DEPTH, mask_of, full) == want
            assert after <= before, 'ゲート数が増えた'

    def test_care_set_is_matched_and_only_dc_may_differ(self):
        """care 上は必ず一致し、差異は care 外パターンに限られる。"""
        rng = np.random.default_rng(4)
        n_differ = 0
        for _ in range(300):
            row, tts = _random_tree(rng)
            taps, _ = tree_function(row, tts, DEPTH)
            u = len(taps)
            n_pat = 1 << u
            care = {int(v) for v in rng.choice(n_pat, size=max(2, n_pat // 3),
                                               replace=False)}
            leaves, new_tts, before, after = resynth_tree(row, tts, DEPTH, care)
            mask_of, _n, full = _mask_of(taps, u)
            want = eval_tree(row, tts, DEPTH, mask_of, full)
            got = eval_tree(leaves, new_tts, DEPTH, mask_of, full)
            diff = want ^ got
            for p in care:
                assert not ((diff >> p) & 1), f'careパターン{p}で不一致'
            if diff:
                n_differ += 1
            assert after <= before
        assert n_differ > 0, 'DC を使った変更が一度も起きていない(テストが無意味)'

    def test_gate_count_monotonic_in_care(self):
        """care を狭めるほどゲート数は増えない(単調性)。"""
        rng = np.random.default_rng(5)
        for _ in range(120):
            row, tts = _random_tree(rng)
            taps, _ = tree_function(row, tts, DEPTH)
            n_pat = 1 << len(taps)
            _l, _t, before, full_after = resynth_tree(row, tts, DEPTH, None)
            care = set(range(n_pat // 2))
            _l2, _t2, before2, dc_after = resynth_tree(row, tts, DEPTH, care)
            assert before == before2
            assert dc_after <= full_after <= before

    def test_empty_care_collapses_to_constant(self):
        """一度も観測されない木は定数(0ゲート)へ潰れる。"""
        rng = np.random.default_rng(6)
        row, tts = _random_tree(rng)
        leaves, new_tts, _before, after = resynth_tree(row, tts, DEPTH, set())
        assert after == 0
        assert _solo_gates(leaves, new_tts) == 0


class TestPruneIdentical:
    """厳密な下界スキップ+本質変数への制限は、旧経路と結果(ゲート数)が同一。

    ``resynth_tree(prune=False)`` が旧経路(全タップ全探索)で、
    ``prune=True``(既定)が新経路。探索空間の枝刈りが探索結果を
    変えないことを、ランダム木・退化木の両方で突き合わせる。
    """

    @pytest.mark.parametrize('n_taps,seed', [(3, 21), (4, 22), (5, 23), (6, 24)])
    def test_gate_counts_identical(self, n_taps, seed):
        rng = np.random.default_rng(seed)
        n_improved = 0
        for _ in range(60 if n_taps <= 5 else 25):
            row, tts = _random_tree(rng, n_taps=n_taps)
            _l0, _t0, b0, a0 = resynth_tree(row, tts, DEPTH, None, prune=False)
            leaves, new_tts, b1, a1 = resynth_tree(row, tts, DEPTH, None)
            assert (b0, a0) == (b1, a1)
            n_improved += a1 < b1
            taps, _ = tree_function(row, tts, DEPTH)
            mask_of, _n, full = _mask_of(taps, len(taps))
            assert (eval_tree(leaves, new_tts, DEPTH, mask_of, full)
                    == eval_tree(row, tts, DEPTH, mask_of, full))
            assert _solo_gates(leaves, new_tts) == a1
        if n_taps >= 4:
            assert n_improved > 0, '改善例が無くテストが無意味'

    def test_constant_one_tree_stays_constant_one(self):
        """定数1に潰れる木を定数0にしない(本質変数0個の境界)。"""
        row = [0, 1, 0, 1, 0, 1, 0, 1]
        tts = [0x6] * 4 + [0x9, 0x9] + [0xf]   # 根=定数1
        leaves, new_tts, _b, after = resynth_tree(row, tts, DEPTH, None)
        taps, onset = tree_function(row, tts, DEPTH)
        mask_of, _n, full = _mask_of(taps, len(taps))
        assert eval_tree(leaves, new_tts, DEPTH, mask_of, full) == onset
        assert after == 0

    def test_essential_vars(self):
        # f(x0,x1,x2) = x0 & x2  → 本質変数は 0 と 2
        u = 3
        onset = 0
        for p in range(1 << u):
            if (p & 1) and (p >> 2) & 1:
                onset |= 1 << p
        assert essential_vars(onset, u) == [0, 2]
        assert essential_vars(0, u) == []
        assert essential_vars((1 << (1 << u)) - 1, u) == []


class TestNodeParams:
    def test_resynth_node_params_roundtrip(self):
        """hex truth_tables を受けて hex で返し、全ocが lossless 等価。"""
        rng = np.random.default_rng(7)
        oc = 12
        idx = [[int(v) for v in rng.integers(0, 9, size=N_LEAVES)]
               for _ in range(oc)]
        tts = [int(v) for v in rng.integers(0, 16, size=oc * N_INTERNAL)]
        params = {
            'out_channels': oc, 'tree_depth': DEPTH, 'n_inputs': 2,
            'in_shape': [1, 4, 4], 'kernel_size': [3, 3],
            'stride': [1, 1], 'padding': [1, 1],
            'indices_flat': idx, 'truth_tables': tt_to_hex(tts),
        }
        new_params, st = resynth_node_params(params, None)
        assert isinstance(new_params['truth_tables'][0], str)
        assert st.n_trees == oc
        assert st.cost_after <= st.cost_before
        before = build_conv_program(spec_from_node_params(params)).stats.gates_after
        after = build_conv_program(
            spec_from_node_params(new_params)).stats.gates_after
        assert after <= before


class TestForwardGraph:
    def _graph(self):
        """conv2d → or_pool2d → upsample2d → permute → group_sum の小IR。"""
        rng = np.random.default_rng(11)
        c, h, w = 3, 8, 8
        oc = 4
        idx = [[int(v) for v in rng.integers(0, c * 9, size=N_LEAVES)]
               for _ in range(oc)]
        tts = [int(v) for v in rng.integers(0, 16, size=oc * N_INTERNAL)]
        nodes = [
            IRNode(id='cv', type='conv2d', params={
                'in_shape': [c, h, w], 'out_channels': oc, 'kernel_size': [3, 3],
                'stride': [1, 1], 'padding': [1, 1], 'tree_depth': DEPTH,
                'n_inputs': 2, 'indices_flat': idx,
                'truth_tables': tt_to_hex(tts)}),
            IRNode(id='pool', type='or_pool2d', params={
                'in_shape': [oc, h, w], 'kernel_size': 2, 'stride': 2,
                'padding': 0}),
            IRNode(id='up', type='upsample2d', params={
                'in_shape': [oc, h // 2, w // 2], 'scale': 2}),
            IRNode(id='pm', type='permute', params={
                'in_shape': [2, 2, h, w], 'perm': [0, 2, 3, 1]}),
            IRNode(id='gs', type='group_sum', params={
                'in_width': oc * h * w, 'k': 2 * h * w, 'tau': 1.0}),
        ]
        edges = [IREdge(src='x', dst='cv'), IREdge(src='cv', dst='pool'),
                 IREdge(src='pool', dst='up'), IREdge(src='up', dst='pm'),
                 IREdge(src='pm', dst='gs'), IREdge(src='gs', dst='y')]
        return IRGraph(
            metadata={}, inputs=[IRPort(id='x', width=c * h * w)],
            outputs=[IRPort(id='y', width=2 * h * w)], nodes=nodes, edges=edges)

    def test_matches_ir_simulate(self):
        """高速フォワードが ir/simulate と整数完全一致する。"""
        g = self._graph()
        rng = np.random.default_rng(12)
        x = rng.integers(0, 2, size=(4, g.inputs[0].width), dtype=np.int64)
        want = simulate(g, x)
        got, _ = forward_graph(g, x)
        assert np.array_equal(np.asarray(want), np.asarray(got))

    def test_concat_matches_ir_simulate(self):
        """concat を含む場合も ir/simulate と一致する。"""
        g = self._graph()
        nodes = list(g.nodes)
        nodes.insert(3, IRNode(id='cat', type='concat',
                               params={'in_widths': [64, 64]}))
        edges = [IREdge(src='x', dst='cv'), IREdge(src='cv', dst='pool'),
                 IREdge(src='pool', dst='up'),
                 IREdge(src='up', dst='cat', dst_slot=0),
                 IREdge(src='up', dst='cat', dst_slot=1),
                 IREdge(src='cat', dst='pm'), IREdge(src='pm', dst='gs'),
                 IREdge(src='gs', dst='y')]
        for n in nodes:
            if n.id == 'pm':
                n.params = {'in_shape': [4, 2, 8, 8], 'perm': [0, 2, 3, 1]}
            if n.id == 'gs':
                n.params = {'in_width': 256, 'k': 128, 'tau': 1.0}
        g2 = IRGraph(metadata={}, inputs=g.inputs,
                     outputs=[IRPort(id='y', width=128)],
                     nodes=nodes, edges=edges)
        rng = np.random.default_rng(13)
        x = rng.integers(0, 2, size=(3, g2.inputs[0].width), dtype=np.int64)
        assert np.array_equal(np.asarray(simulate(g2, x)),
                              np.asarray(forward_graph(g2, x)[0]))

    def test_care_sets_are_observed_patterns(self):
        """採取した care set が、実際に観測されたパターン集合と一致する。"""
        g = self._graph()
        rng = np.random.default_rng(14)
        x = rng.integers(0, 2, size=(6, g.inputs[0].width), dtype=np.uint8)
        _out, care = forward_graph(g, x, care_nodes={'cv'})
        cv = next(n for n in g.nodes if n.id == 'cv')
        assert len(care['cv']) == cv.params['out_channels']
        for oc, pats in enumerate(care['cv']):
            u = len({int(v) for v in cv.params['indices_flat'][oc]})
            assert pats, 'care set が空'
            assert max(pats) < (1 << u)

    @pytest.mark.parametrize('n_taps', [2, 4, 8])
    def test_resynth_preserves_forward_output_when_lossless(self, n_taps):
        """lossless 再合成後のIRは元IRと出力整数が完全一致する。"""
        g = self._graph()
        cv = next(n for n in g.nodes if n.id == 'cv')
        rng = np.random.default_rng(15 + n_taps)
        cv.params['indices_flat'] = [
            [int(v) for v in rng.integers(0, n_taps, size=N_LEAVES)]
            for _ in range(cv.params['out_channels'])]
        x = rng.integers(0, 2, size=(5, g.inputs[0].width), dtype=np.uint8)
        want, _ = forward_graph(g, x)
        cv.params, _st = resynth_node_params(cv.params, None)
        got, _ = forward_graph(g, x)
        assert np.array_equal(np.asarray(want), np.asarray(got))
