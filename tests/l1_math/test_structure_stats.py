"""L1: 構造統計ライブラリ(analysis/structure.py、SM1)の数学的性質。

検証方針:
    support_sizes: 素朴な全パターン照合(依存 ⇔ ∃k: tt[k] != tt[k^(1<<j)])と一致
    fanout_counts: 素朴な全数カウントと一致 + build_reverse_index(CSR逆引き)と一致
    reach_cone:    到達不能ゲートの真理値表を書き換えてもsimulate出力がbit一致
すべてビット/整数の厳密比較(許容誤差なし)。
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.analysis.structure import (
    degenerate_input_stats,
    fanout_counts,
    fanout_stats,
    gate_class_rates,
    gate_probs_from_weights,
    neuron_uncertainty,
    reach_cone,
    reachable_masks,
    support_sizes,
    wiring_locality,
)
from griffith_logic.core.connections import build_reverse_index
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.simulate import simulate


def naive_support(tt: int, n: int) -> int:
    """素朴なサポートサイズ: 入力jに依存 ⇔ ∃k: tt[k] != tt[k^(1<<j)]"""
    deps = 0
    for j in range(n):
        for k in range(1 << n):
            if ((tt >> k) & 1) != ((tt >> (k ^ (1 << j))) & 1):
                deps += 1
                break
    return deps


class TestSupportSizes:
    def test_all_16_gates_n2(self):
        tts = list(range(16))
        got = support_sizes(tts, 2)
        want = [naive_support(t, 2) for t in tts]
        np.testing.assert_array_equal(got, want)

    def test_known_values_n2(self):
        # FALSE=0x0, TRUE=0xF: 定数(support 0)
        # A=0xA(k∈{1,3}), B=0xC, !A=0x5, !B=0x3: パススルー(support 1)
        # AND=0x8, XOR=0x6: support 2
        got = support_sizes([0x0, 0xF, 0xA, 0xC, 0x5, 0x3, 0x8, 0x6], 2)
        np.testing.assert_array_equal(got, [0, 0, 1, 1, 1, 1, 2, 2])

    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_random_matches_naive(self, n):
        rng = np.random.default_rng(42)
        size = 1 << n
        tts = [int(v) for v in rng.integers(0, 2**size, size=50, dtype=np.uint64)]
        got = support_sizes(tts, n)
        want = [naive_support(t, n) for t in tts]
        np.testing.assert_array_equal(got, want)

    def test_hex_string_input(self):
        # IRのparams['truth_tables'](hex文字列)をそのまま渡せる
        got = support_sizes(['0x0', '0x8', '0xa'], 2)
        np.testing.assert_array_equal(got, [0, 2, 1])

    def test_gate_class_rates(self):
        rates = gate_class_rates([0x0, 0xF, 0xA, 0x8], 2)
        assert rates['const_rate'] == pytest.approx(0.5)
        assert rates['passthrough_rate'] == pytest.approx(0.25)
        assert rates['support_hist'] == {0: 2, 1: 1, 2: 1}
        assert rates['num_gates'] == 4


class TestFanout:
    def test_matches_naive_count(self):
        rng = np.random.default_rng(0)
        in_width, out_dim, n_inputs = 17, 40, 3
        idx = rng.integers(0, in_width, size=(n_inputs, out_dim))
        counts = fanout_counts(idx, in_width)
        naive = np.zeros(in_width, dtype=np.int64)
        for slot in range(n_inputs):
            for o in range(out_dim):
                naive[idx[slot, o]] += 1
        np.testing.assert_array_equal(counts, naive)

    def test_matches_build_reverse_index(self):
        rng = np.random.default_rng(1)
        in_width, out_dim, n_inputs = 12, 30, 2
        idx = torch.tensor(
            rng.integers(0, in_width, size=(n_inputs, out_dim)),
            dtype=torch.int64,
        )
        counts = fanout_counts(idx, in_width)
        starts, _, _ = build_reverse_index(idx, in_width)
        np.testing.assert_array_equal(counts, np.diff(starts.numpy()))

    def test_stats_consistency(self):
        idx = [[0, 0, 1], [1, 0, 2]]  # wire0: 3回, wire1: 2回, wire2: 1回, wire3: 0回
        stats = fanout_stats(idx, 4)
        assert stats['max'] == 3
        assert stats['zero_frac'] == pytest.approx(0.25)
        assert stats['hist'] == {0: 1, 1: 1, 2: 1, 3: 1}
        assert stats['mean'] == pytest.approx(6 / 4)
        assert stats['top'][0] == [0, 3]

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            fanout_counts([[0, 5]], 4)


class TestDegenerateInputs:
    def test_hand_case(self):
        # ゲート0: (0,1)正常 / ゲート1: (2,2)縮退 / ゲート2: (3,1)正常
        stats = degenerate_input_stats([[0, 2, 3], [1, 2, 1]])
        assert stats['count'] == 1
        assert stats['rate'] == pytest.approx(1 / 3)

    def test_n1_never_degenerate(self):
        stats = degenerate_input_stats([[0, 1, 2]])
        assert stats['count'] == 0


class TestWiringLocality:
    def test_spread_hand_case(self):
        # in_width=11: ゲート0=(0,10)→spread 1.0 / ゲート1=(5,5)→0.0
        stats = wiring_locality([[0, 5], [10, 5]], 11)
        assert stats['spread_mean'] == pytest.approx(0.5)

    def test_pair_share_matches_bruteforce(self):
        rng = np.random.default_rng(7)
        in_width, out_dim, n_inputs = 8, 12, 2
        idx = rng.integers(0, in_width, size=(n_inputs, out_dim))
        stats = wiring_locality(idx, in_width)
        # 素朴照合: 全ゲートペアの共有ワイヤ数(異なり集合の共通部分)の平均
        sets = [set(idx[:, o].tolist()) for o in range(out_dim)]
        total = 0
        pairs = 0
        for a in range(out_dim):
            for b in range(a + 1, out_dim):
                total += len(sets[a] & sets[b])
                pairs += 1
        assert stats['pair_share_expected'] == pytest.approx(total / pairs)


def _two_layer_graph() -> IRGraph:
    """in(4) → n0 lut(4→4, AND) → n1 lut(4→2, XOR) → out(2)。

    n1はn0のワイヤ{0,1}しか参照しない → n0のゲート2,3は到達不能。
    """
    n0 = IRNode(id='n0', type='lut_layer', params={
        'in_width': 4, 'out_width': 4, 'n_inputs': 2,
        'indices': [[0, 1, 2, 3], [1, 2, 3, 0]],
        'truth_tables': ['0x8', '0x8', '0x8', '0x8'],
    })
    n1 = IRNode(id='n1', type='lut_layer', params={
        'in_width': 4, 'out_width': 2, 'n_inputs': 2,
        'indices': [[0, 1], [1, 0]],
        'truth_tables': ['0x6', '0x6'],
    })
    return IRGraph(
        inputs=[IRPort(id='in', width=4)],
        outputs=[IRPort(id='out', width=2)],
        nodes=[n0, n1],
        edges=[
            IREdge(src='in', dst='n0'),
            IREdge(src='n0', dst='n1'),
            IREdge(src='n1', dst='out'),
        ],
    )


def _all_inputs(width: int) -> np.ndarray:
    return np.array(
        [[(k >> j) & 1 for j in range(width)] for k in range(1 << width)],
        dtype=np.int64,
    )


class TestReachCone:
    def test_unreachable_detection(self):
        graph = _two_layer_graph()
        masks = reachable_masks(graph)
        np.testing.assert_array_equal(masks['n0'], [True, True, False, False])
        np.testing.assert_array_equal(masks['n1'], [True, True])

        cone = reach_cone(graph)
        assert cone['total_lut_gates'] == 6
        assert cone['unreachable_gates'] == 2
        assert cone['unreachable_frac'] == pytest.approx(2 / 6)
        assert cone['per_node']['n0']['unreachable'] == 2
        assert cone['per_node']['n1']['unreachable'] == 0

    def test_unreachable_gates_dont_affect_output(self):
        """到達不能ゲートの真理値表を全書き換えしてもsimulate出力はbit一致。"""
        graph = _two_layer_graph()
        x = _all_inputs(4)
        base = simulate(graph, x)

        mutated = _two_layer_graph()
        tts = mutated.node_by_id('n0').params['truth_tables']
        tts[2] = '0xf'  # AND → TRUE
        tts[3] = '0x6'  # AND → XOR
        np.testing.assert_array_equal(simulate(mutated, x), base)

    def test_reachable_gate_does_affect_output(self):
        """対照実験: 到達可能ゲートの書き換えは出力を変える(検出力の確認)。"""
        graph = _two_layer_graph()
        x = _all_inputs(4)
        base = simulate(graph, x)

        mutated = _two_layer_graph()
        mutated.node_by_id('n0').params['truth_tables'][0] = '0xf'
        assert not np.array_equal(simulate(mutated, x), base)

    def test_output_depth(self):
        cone = reach_cone(_two_layer_graph())
        assert cone['output_depth']['max'] == 2
        assert cone['output_depth']['hist'] == {2: 2}

    def test_group_sum_tail(self):
        """in(4) → n0 lut(4→4) → gs group_sum(k=2) → out: 全ゲート到達可能。"""
        n0 = IRNode(id='n0', type='lut_layer', params={
            'in_width': 4, 'out_width': 4, 'n_inputs': 2,
            'indices': [[0, 1, 2, 3], [1, 2, 3, 0]],
            'truth_tables': ['0x8'] * 4,
        })
        gs = IRNode(id='gs', type='group_sum', params={
            'in_width': 4, 'k': 2, 'tau': 1.0,
        })
        graph = IRGraph(
            inputs=[IRPort(id='in', width=4)],
            outputs=[IRPort(id='out', width=2)],
            nodes=[n0, gs],
            edges=[
                IREdge(src='in', dst='n0'),
                IREdge(src='n0', dst='gs'),
                IREdge(src='gs', dst='out'),
            ],
        )
        cone = reach_cone(graph)
        assert cone['unreachable_gates'] == 0
        assert cone['output_depth']['max'] == 1


class TestPromotedFunctions:
    """昇格したゲート確率系関数のsanity(詳細はeditor既存テストが担保)。"""

    def test_softmax_probs_sum_to_one(self):
        w = torch.randn(8, 16)
        probs = gate_probs_from_weights(w, 'softmax', 2)
        np.testing.assert_allclose(probs.sum(dim=-1).numpy(), 1.0, rtol=1e-5)

    def test_uncertainty_normalized(self):
        for param, shape in [('softmax', (8, 16)), ('independent', (8, 4)),
                             ('raw', (8, 4)), ('iwp', (8, 2))]:
            u = neuron_uncertainty(torch.randn(*shape), param, 2)
            assert u is not None
            assert float(u.min()) >= 0.0
            assert float(u.max()) <= 1.0 + 1e-6

    def test_warp_returns_none(self):
        assert gate_probs_from_weights(torch.randn(4, 4), 'warp', 2) is None
        assert neuron_uncertainty(torch.randn(4, 4), 'warp', 2) is None

    def test_editor_backend_reexport(self):
        """editor backendからの再exportが同一オブジェクトであること。"""
        from tools.editor.backend import extract
        assert extract.gate_probs_from_weights is gate_probs_from_weights
        assert extract.neuron_uncertainty is neuron_uncertainty
