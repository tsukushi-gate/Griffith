"""活性・トグル・電力proxy(analysis/activity.py、SM5)の手計算一致テスト。

極小IR(既知真理値表+既知入力分布)で活性率・トグルproxy・fanout重み付け
電力proxyを厳密に照合する。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.analysis.activity import activity_stats, ir_bit_fanout
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort


def _graph_two_layer() -> IRGraph:
    """in(2) → l0[AND(0,1), XOR(0,1), FALSE] → l1[OR(0,1), NOT2] → out(2)。

    l1のindices: gate0=(0,1)、gate1=(2,2)。
    l0のfanout: wire0=2回, wire1=2回, wire2=2回(l1のスロット重複込み)。
    """
    l0 = IRNode(id='l0', type='lut_layer', params={
        'in_width': 2, 'out_width': 3, 'n_inputs': 2,
        'indices': [[0, 0, 0], [1, 1, 1]],
        'truth_tables': ['0x8', '0x6', '0x0'],
    })
    l1 = IRNode(id='l1', type='lut_layer', params={
        'in_width': 3, 'out_width': 2, 'n_inputs': 2,
        'indices': [[0, 2], [1, 2]],
        'truth_tables': ['0xe', '0x3'],  # OR / NOT(slot0)
    })
    return IRGraph(
        inputs=[IRPort(id='in', width=2)],
        outputs=[IRPort(id='out', width=2)],
        nodes=[l0, l1],
        edges=[
            IREdge(src='in', dst='l0'),
            IREdge(src='l0', dst='l1'),
            IREdge(src='l1', dst='out'),
        ],
    )


def _all_patterns() -> np.ndarray:
    # 全4パターンを1回ずつ → 一様入力分布
    return np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)


class TestBitFanout:
    def test_lut_and_output_port_counting(self):
        fanout = ir_bit_fanout(_graph_two_layer())
        # in: l0が各ワイヤを3回ずつ参照(3ゲート×該当スロット)
        np.testing.assert_array_equal(fanout['in'], [3, 3])
        # l0: l1のindicesでwire0=1回, wire1=1回, wire2=2回
        np.testing.assert_array_equal(fanout['l0'], [1, 1, 2])
        # l1: 出力ポートで各ビット1回
        np.testing.assert_array_equal(fanout['l1'], [1, 1])

    def test_positional_consumer(self):
        l0 = IRNode(id='l0', type='lut_layer', params={
            'in_width': 4, 'out_width': 4, 'n_inputs': 2,
            'indices': [[0, 1, 2, 3], [1, 2, 3, 0]],
            'truth_tables': ['0x8'] * 4,
        })
        gs = IRNode(id='gs', type='group_sum',
                    params={'in_width': 4, 'k': 2, 'tau': 1.0})
        graph = IRGraph(
            inputs=[IRPort(id='in', width=4)],
            outputs=[IRPort(id='out', width=2)],
            nodes=[l0, gs],
            edges=[
                IREdge(src='in', dst='l0'),
                IREdge(src='l0', dst='gs'),
                IREdge(src='gs', dst='out'),
            ],
        )
        fanout = ir_bit_fanout(graph)
        np.testing.assert_array_equal(fanout['l0'], [1, 1, 1, 1])


class TestActivityStats:
    def test_hand_computed_uniform_input(self):
        stats = activity_stats(_graph_two_layer(), _all_patterns())
        l0 = stats['per_node']['l0']
        # p: AND=1/4, XOR=1/2, FALSE=0
        assert l0['act_mean'] == pytest.approx((0.25 + 0.5 + 0.0) / 3)
        # toggle: 2p(1-p): AND=0.375, XOR=0.5, FALSE=0
        assert l0['toggle_mean'] == pytest.approx((0.375 + 0.5 + 0.0) / 3)
        # データ上定数: FALSEゲートのみ
        assert l0['const_frac'] == pytest.approx(1 / 3)
        # 電力proxy = Σ toggle×fanout = 0.375*1 + 0.5*1 + 0*2
        assert l0['power_proxy'] == pytest.approx(0.875)

    def test_totals_aggregate(self):
        stats = activity_stats(_graph_two_layer(), _all_patterns())
        per = stats['per_node']
        assert stats['totals']['power_proxy'] == pytest.approx(
            sum(v['power_proxy'] for v in per.values())
        )
        # l1: OR(AND,XOR)=p 3/4... 単に全ビット平均の整合のみ確認
        bits = 3 + 2
        toggle_total = per['l0']['toggle_mean'] * 3 + per['l1']['toggle_mean'] * 2
        assert stats['totals']['toggle_mean'] == pytest.approx(toggle_total / bits)

    def test_batch_split_invariant(self):
        x = np.tile(_all_patterns(), (8, 1))
        full = activity_stats(_graph_two_layer(), x, batch_size=1000)
        split = activity_stats(_graph_two_layer(), x, batch_size=3)
        for nid in full['per_node']:
            assert split['per_node'][nid]['act_mean'] == pytest.approx(
                full['per_node'][nid]['act_mean']
            )
            assert split['per_node'][nid]['power_proxy'] == pytest.approx(
                full['per_node'][nid]['power_proxy']
            )

    def test_sequential_toggle(self):
        # in幅1のパススルー: 入力系列 0,1,1,0 → フリップ2回/3遷移
        l0 = IRNode(id='l0', type='lut_layer', params={
            'in_width': 1, 'out_width': 1, 'n_inputs': 1,
            'indices': [[0]],
            'truth_tables': ['0x2'],  # 恒等(N=1: bit1=1)
        })
        graph = IRGraph(
            inputs=[IRPort(id='in', width=1)],
            outputs=[IRPort(id='out', width=1)],
            nodes=[l0],
            edges=[IREdge(src='in', dst='l0'), IREdge(src='l0', dst='out')],
        )
        x = np.array([[0], [1], [1], [0]], dtype=np.int64)
        stats = activity_stats(graph, x, sequential=True, batch_size=2)
        assert stats['per_node']['l0']['seq_toggle_mean'] == pytest.approx(2 / 3)

    def test_group_sum_excluded(self):
        l0 = IRNode(id='l0', type='lut_layer', params={
            'in_width': 2, 'out_width': 2, 'n_inputs': 2,
            'indices': [[0, 1], [1, 0]],
            'truth_tables': ['0x8', '0xe'],
        })
        gs = IRNode(id='gs', type='group_sum',
                    params={'in_width': 2, 'k': 1, 'tau': 1.0})
        graph = IRGraph(
            inputs=[IRPort(id='in', width=2)],
            outputs=[IRPort(id='out', width=1)],
            nodes=[l0, gs],
            edges=[
                IREdge(src='in', dst='l0'),
                IREdge(src='l0', dst='gs'),
                IREdge(src='gs', dst='out'),
            ],
        )
        stats = activity_stats(graph, _all_patterns())
        assert 'gs' not in stats['per_node']
        assert 'l0' in stats['per_node']
