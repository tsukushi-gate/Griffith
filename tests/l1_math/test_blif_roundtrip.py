"""IR⇔BLIF往復のビット厳密性テスト(compile-pipeline)。

すべて整数信号の厳密一致で検証する(許容誤差を使わない)。
    1. ビット順回帰(左右反転バグ検出)
    2. ランダムLUT鎖(fan-in混在・定数/恒等tt・concat含む)の往復
    3. 可変fan-in DAGの層化(飛び越し配線・PO driver共有・定数ノード)
    4. decompose_mlp toy往復(split_lut_core/reattach_group_sum込み)
    5. pad_ttの単体テスト
"""

import itertools
import random as pyrandom

import numpy as np
import torch
from torch import nn

from griffith_logic.convert.decompose_thermo import DecomposeSpec, decompose_mlp
from griffith_logic.export.blif import (
    BlifNetwork,
    blif_to_ir,
    pad_tt,
    read_blif,
    reattach_group_sum,
    split_lut_core,
    write_blif,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import tt_from_hex, tt_to_hex
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _all_patterns(width: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=width)), dtype=np.int64)


def _rand_patterns(width: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, width)).astype(np.int64)


def _roundtrip(graph: IRGraph, path: str) -> IRGraph:
    write_blif(graph, path)
    net = read_blif(path)
    return blif_to_ir(net, in_width=graph.inputs[0].width)


# ============================================================
# 1. ビット順回帰
# ============================================================

def test_bit_order_regression(tmp_path):
    """x0∧¬x1(非対称)で往復が全4パターン一致することを確認する。

    tt=0x2: bit1のみ立つ = アドレス1(スロット0=1,スロット1=0)で出力1。
    左右反転バグがあれば bit2(スロット0=0,スロット1=1)側で不一致になる。
    """
    node = IRNode(
        id='n0', type='lut_layer',
        params={
            'in_width': 2, 'out_width': 1, 'n_inputs': 2,
            'indices': [[0], [1]],
            'truth_tables': tt_to_hex([0x2]),
        },
    )
    graph = IRGraph(
        inputs=[IRPort(id='x', width=2)],
        outputs=[IRPort(id='y', width=1)],
        nodes=[node],
        edges=[IREdge(src='x', dst='n0'), IREdge(src='n0', dst='y')],
    )
    validate(graph)

    graph2 = _roundtrip(graph, str(tmp_path / 'bitorder.blif'))
    validate(graph2)

    x = _all_patterns(2)
    expected = simulate(graph, x)
    actual = simulate(graph2, x)
    assert (expected == actual).all(), (expected.ravel(), actual.ravel())
    # 非対称性そのものの確認(反転していれば的外れなテストになるため)
    assert expected.ravel().tolist() == [0, 0, 1, 0]


# ============================================================
# 2. ランダムLUT鎖(fan-in混在・定数/恒等tt・concat含む)
# ============================================================

def _rand_lut_node(node_id, in_width, out_width, n_inputs, rng) -> IRNode:
    indices = [[int(rng.integers(0, in_width)) for _ in range(out_width)]
               for _ in range(n_inputs)]
    # tt本体は最大2^64値(n_inputs=6)になりnumpy int64境界を超えるため
    # pythonのrandom(任意精度)で生成する(numpy rngからシードして決定的に)
    pyrng = pyrandom.Random(int(rng.integers(0, 2**31)))
    tts = [pyrng.getrandbits(1 << n_inputs) for _ in range(out_width)]
    return IRNode(
        id=node_id, type='lut_layer',
        params={'in_width': in_width, 'out_width': out_width,
                'n_inputs': n_inputs, 'indices': indices,
                'truth_tables': tt_to_hex(tts)},
    )


def _build_random_chain(seed: int) -> IRGraph:
    """fan-in1..6混在・定数/恒等tt・concatを含むランダム多層IR。"""
    rng = np.random.default_rng(seed)
    in_width = 6

    branch_a = _rand_lut_node('a1', in_width, 5, 3, rng)
    branch_b = _rand_lut_node('b1', in_width, 4, 2, rng)

    # 定数tt(全0/全1)を混入
    a_tts = tt_from_hex(branch_a.params['truth_tables'])
    a_tts[0] = 0                    # 定数0
    a_tts[1] = (1 << (1 << 3)) - 1  # 定数1
    branch_a.params['truth_tables'] = tt_to_hex(a_tts)

    cat = IRNode(id='cat', type='concat', params={'in_widths': [5, 4]})

    mid = _rand_lut_node('mid', 9, 5, 6, rng)  # n_inputs=6(上限)も含める

    tail = _rand_lut_node('tail', 5, 4, 1, rng)  # fan-in1
    tail_tts = tt_from_hex(tail.params['truth_tables'])
    tail_tts[0] = 0x2   # 恒等(パススルー)
    tail.params['truth_tables'] = tt_to_hex(tail_tts)

    nodes = [branch_a, branch_b, cat, mid, tail]
    edges = [
        IREdge(src='x', dst='a1'),
        IREdge(src='x', dst='b1'),
        IREdge(src='a1', dst='cat', dst_slot=0),
        IREdge(src='b1', dst='cat', dst_slot=1),
        IREdge(src='cat', dst='mid'),
        IREdge(src='mid', dst='tail'),
        IREdge(src='tail', dst='y'),
    ]
    return IRGraph(
        inputs=[IRPort(id='x', width=in_width)],
        outputs=[IRPort(id='y', width=4)],
        nodes=nodes, edges=edges,
    )


def test_random_lut_chain_roundtrip(tmp_path):
    for seed in (0, 1, 2):
        graph = _build_random_chain(seed)
        validate(graph)
        stats = write_blif(graph, str(tmp_path / f'chain_{seed}.blif'))
        assert stats['n_inputs'] == 6
        assert stats['n_outputs'] == 4

        graph2 = _roundtrip(graph, str(tmp_path / f'chain_{seed}.blif'))
        validate(graph2)

        x = _all_patterns(6)  # in_width=6は全数(64パターン)で厳密照合
        expected = simulate(graph, x)
        actual = simulate(graph2, x)
        assert (expected == actual).all(), f'seed={seed}で不一致'


# ============================================================
# 3. 可変fan-in DAGの層化(飛び越し配線・PO driver共有・定数ノード)
# ============================================================

def test_variable_fanin_dag_layerization(tmp_path):
    """手書きBlifNetworkの層化 → BLIF書き戻し → 再読込 → simulate一致。

    構造(inputs=[a,b,c]):
        g1 = a & b                (level1)
        g2 = a | c                (level1)
        g3 = g1 & c               (level2、level0のcを直接参照)
        g5 = 定数1                 (level1、fan-in0)
        g4 = a | g3                (level3、level0のaへ飛び越し配線)
        outputs = [g4, g3, g5]     (g3はg4の内部fanoutとPO driverを兼ねる)
    """
    net = BlifNetwork(
        inputs=['a', 'b', 'c'],
        outputs=['g4', 'g3', 'g5'],
        names=[
            (['a', 'b'], 'g1', [('11', '1')]),
            (['a', 'c'], 'g2', [('1-', '1'), ('-1', '1')]),
            (['g1', 'c'], 'g3', [('11', '1')]),
            ([], 'g5', [('', '1')]),
            (['a', 'g3'], 'g4', [('1-', '1'), ('-1', '1')]),
        ],
    )
    graph1 = blif_to_ir(net, in_width=3)
    validate(graph1)

    x = _all_patterns(3)
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    expected = np.stack([
        a,                    # g4 = a | (a&b&c) = a(吸収則)
        a & b & c,            # g3
        np.ones_like(a),      # g5(定数1)
    ], axis=1)
    actual1 = simulate(graph1, x)
    assert (actual1 == expected).all(), 'blif_to_irの層化結果が期待値と不一致'

    path = str(tmp_path / 'dag.blif')
    write_blif(graph1, path)
    net2 = read_blif(path)
    graph2 = blif_to_ir(net2, in_width=3)
    validate(graph2)

    actual2 = simulate(graph2, x)
    assert (actual1 == actual2).all(), '書き戻し後の再構成が一致しない'


# ============================================================
# 4. decompose_mlp toy往復(split_lut_core/reattach_group_sum込み)
# ============================================================

def test_decompose_mlp_toy_roundtrip(tmp_path):
    """decompose_mlp(hidden_arm='adder')のtoy IRでcore分離→BLIF往復→再結合。"""
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    in_w, hidden, classes = 6, 4, 3

    model = nn.Sequential(nn.Linear(in_w, hidden), nn.Linear(hidden, classes))
    with torch.no_grad():
        q = rng.choice([-1.0, 1.0], size=(hidden, in_w))
        model[0].weight.copy_(torch.tensor(q, dtype=torch.float32))
        model[0].bias.zero_()

    calib_x = rng.integers(0, 2, size=(64, in_w)).astype(np.int64)
    spec = DecomposeSpec()
    graph = decompose_mlp(model, calib_x, spec, hidden_arm='adder')
    validate(graph)

    core_graph, gs_node = split_lut_core(graph)
    validate(core_graph)

    path = str(tmp_path / 'toy.blif')
    write_blif(core_graph, path)
    net = read_blif(path)
    core2 = blif_to_ir(net, in_width=core_graph.inputs[0].width)
    graph2 = reattach_group_sum(core2, gs_node)
    validate(graph2)

    x = _rand_patterns(in_w, 256, seed=7)
    expected = simulate(graph, x)
    actual = simulate(graph2, x)
    assert (expected == actual).all(), '再結合後のsimulateが元IRと不一致'
    assert (expected.argmax(-1) == actual.argmax(-1)).all(), 'argmaxが不一致'


# ============================================================
# 5. pad_ttの単体テスト
# ============================================================

def test_pad_tt_ignores_padding_slots():
    rng = np.random.default_rng(11)
    for n_real in range(1, 4):
        for extra in range(0, 3):
            n_target = n_real + extra
            tt = int(rng.integers(0, 1 << (1 << n_real)))
            padded = pad_tt(tt, n_real, n_target)
            mask = (1 << n_real) - 1
            for addr in range(1 << n_target):
                expected_bit = (tt >> (addr & mask)) & 1
                actual_bit = (padded >> addr) & 1
                assert actual_bit == expected_bit, (n_real, n_target, addr)


def test_pad_tt_identity_when_equal():
    assert pad_tt(0b1011, 2, 2) == 0b1011
    assert pad_tt(0, 3, 3) == 0


def test_split_reattach_without_group_sum():
    """group_sum無し(bits系)グラフはsplit/reattachがパススルーになる。"""
    node = IRNode(
        id='L1', type='lut_layer',
        params={'in_width': 4, 'out_width': 3, 'n_inputs': 2,
                'indices': [[0, 1, 2], [3, 2, 0]],
                'truth_tables': ['0x8', '0x6', '0x2']},
    )
    g = IRGraph(
        metadata={},
        inputs=[IRPort(id='x', width=4)],
        outputs=[IRPort(id='y', width=3)],
        nodes=[node],
        edges=[IREdge(src='x', dst='L1'), IREdge(src='L1', dst='y')],
    )
    core, gs = split_lut_core(g)
    assert gs is None
    assert core is g
    assert reattach_group_sum(core, None) is core
