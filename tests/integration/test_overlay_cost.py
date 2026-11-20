"""overlayコスト指標(analysis/overlay_cost.py、SM8)の手計算一致テスト。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.analysis.overlay_cost import OverlayEngineSpec, overlay_cost
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort


def _graph() -> IRGraph:
    """in(4) → l0(4→3) → l1(3→2) → out。

    スケジュール位置: l0=[0,1,2], l1=[3,4]。
    l1の参照: g0=(wire0,wire1)→距離(3,2) / g1=(wire2,wire2)→距離(2,2)。
    """
    l0 = IRNode(id='l0', type='lut_layer', params={
        'in_width': 4, 'out_width': 3, 'n_inputs': 2,
        'indices': [[0, 1, 2], [1, 2, 3]],
        'truth_tables': ['0x8', '0x6', '0xe'],
    })
    l1 = IRNode(id='l1', type='lut_layer', params={
        'in_width': 3, 'out_width': 2, 'n_inputs': 2,
        'indices': [[0, 2], [1, 2]],
        'truth_tables': ['0xe', '0x6'],
    })
    return IRGraph(
        inputs=[IRPort(id='in', width=4)],
        outputs=[IRPort(id='out', width=2)],
        nodes=[l0, l1],
        edges=[
            IREdge(src='in', dst='l0'),
            IREdge(src='l0', dst='l1'),
            IREdge(src='l1', dst='out'),
        ],
    )


class TestMemoryFootprint:
    def test_hand_computed_bits(self):
        cost = overlay_cost(_graph())
        # l0: in_width4→index2bit、ゲートあたり 2^2 + 2*2 = 8bit × 3
        # l1: in_width3→index2bit、8bit × 2
        assert cost['memory']['per_node'] == {'l0': 24, 'l1': 16}
        assert cost['memory']['total_bits'] == 40
        assert cost['memory']['tt_bits'] == 20
        assert cost['memory']['index_bits'] == 20

    def test_fixed_index_bits(self):
        cost = overlay_cost(_graph(), OverlayEngineSpec(index_bits=16))
        assert cost['memory']['per_node']['l0'] == 3 * (4 + 2 * 16)


class TestCycles:
    def test_layer_barrier_ceil(self):
        cost = overlay_cost(_graph(), OverlayEngineSpec(gates_per_cycle=2))
        assert cost['cycles']['per_node'] == {'l0': 2, 'l1': 1}
        assert cost['cycles']['total'] == 3

    def test_default_one_gate_per_cycle(self):
        cost = overlay_cost(_graph())
        assert cost['cycles']['total'] == 5


class TestReuseDistance:
    def test_hand_computed_distances(self):
        cost = overlay_cost(_graph())
        # 距離 = [3,2,2,2](l1のみ。l0は一次入力参照6件)
        assert cost['reuse']['reads'] == 4
        assert cost['reuse']['primary_reads'] == 6
        assert cost['reuse']['mean'] == pytest.approx(2.25)
        assert cost['reuse']['max'] == 3


class TestBanking:
    def test_two_banks_group_conflict(self):
        cost = overlay_cost(_graph(), OverlayEngineSpec(gates_per_cycle=2, banks=2))
        # l1のサイクル群: アドレス[0,2,1,2] → bank0=3件, bank1=1件
        # 競合=Σmax(0,c-1)=2 / 非一次読出し4件
        assert cost['banking']['conflict_rate'] == pytest.approx(0.5)
        assert cost['banking']['mean_serialization'] == pytest.approx(3.0)

    def test_single_bank_serializes_per_gate(self):
        cost = overlay_cost(_graph(), OverlayEngineSpec(gates_per_cycle=1, banks=1))
        # l1 g0: addr[0,1]→bank0×2 / g1: addr[2,2]→bank0×2 → 各サイクル直列化2
        assert cost['banking']['conflict_rate'] == pytest.approx(0.5)
        assert cost['banking']['mean_serialization'] == pytest.approx(2.0)


class TestNonLutNodes:
    def test_group_sum_positions_dont_break(self):
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
        cost = overlay_cost(graph)
        assert cost['cycles']['total'] == 4
        assert cost['reuse']['reads'] == 0  # lut層は1段のみ(全て一次入力)
