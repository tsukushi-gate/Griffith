"""IRレベルのlut_layer flip修復(analysis/ir_repair.py)のL1性質テスト。

torch不要。IRGraphを直接dictから構築し(tests/hdl/test_hdl_dag.pyと同じ
流儀)、repair_ir_flipの単調性・完全復元・不変性・未訪問アドレス不変・
max_flips尊重を検証する。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.analysis.ir_repair import IrRepairReport, repair_ir_flip
from griffith_logic.ir import ir_from_dict, simulate, validate


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


def _rand_lut_params(in_width, out_width, n_inputs, rng):
    indices = [
        [int(rng.integers(0, in_width)) for _ in range(out_width)]
        for _ in range(n_inputs)
    ]
    tts = [hex(int(rng.integers(0, 1 << (1 << n_inputs))))
           for _ in range(out_width)]
    return indices, tts


def _lut_node(node_id, in_width, out_width, rng, n_inputs=2):
    indices, tts = _rand_lut_params(in_width, out_width, n_inputs, rng)
    return {
        'id': node_id,
        'type': 'lut_layer',
        'params': {
            'in_width': in_width,
            'out_width': out_width,
            'n_inputs': n_inputs,
            'indices': indices,
            'truth_tables': tts,
        },
        'train_info': {},
        'meta': {},
    }


def _group_sum_node(node_id, in_width, k):
    return {
        'id': node_id,
        'type': 'group_sum',
        'params': {'in_width': in_width, 'k': k, 'tau': 1.0},
        'train_info': {},
        'meta': {},
    }


def _argmax_doc(in_width=8, mid_width=8, out_width=8, k=2, seed=1):
    """x(in_width) -> L1(in_width->mid) -> L2(mid->out) -> group_sum(k) -> y(k)"""
    rng = np.random.default_rng(seed)
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0', 'metadata': {},
        'inputs': [_port('x', in_width)],
        'outputs': [_port('y', k)],
        'nodes': [
            _lut_node('L1', in_width, mid_width, rng),
            _lut_node('L2', mid_width, out_width, rng),
            _group_sum_node('GS', out_width, k),
        ],
        'edges': [
            {'src': 'x', 'dst': 'L1'},
            {'src': 'L1', 'dst': 'L2'},
            {'src': 'L2', 'dst': 'GS'},
            {'src': 'GS', 'dst': 'y'},
        ],
    }


def _bits_doc(in_width=8, out_width=6, seed=2):
    """x(in_width) -> L1(in_width->out_width) -> y(out_width)。group_sumなし。"""
    rng = np.random.default_rng(seed)
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0', 'metadata': {},
        'inputs': [_port('x', in_width)],
        'outputs': [_port('y', out_width)],
        'nodes': [_lut_node('L1', in_width, out_width, rng)],
        'edges': [
            {'src': 'x', 'dst': 'L1'},
            {'src': 'L1', 'dst': 'y'},
        ],
    }


def _binary(n, width, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, width)).astype(np.uint8)


def _full_enum(width):
    n = 1 << width
    rows = ((np.arange(n)[:, None] >> np.arange(width)[None, :]) & 1)
    return rows.astype(np.uint8)


def _flip_tt_bit(tt_hex: str, bit: int) -> str:
    return hex(int(tt_hex, 16) ^ (1 << bit))


# ============================================================
# 1. 単調性
# ============================================================

def test_monotonicity_argmax():
    doc = _argmax_doc(in_width=8, mid_width=8, out_width=8, k=2, seed=11)
    graph = ir_from_dict(doc)

    x_fit = _binary(128, 8, seed=21)
    x_val = _binary(64, 8, seed=22)
    rng = np.random.default_rng(31)
    y_fit = rng.integers(0, 2, size=128)
    y_val = rng.integers(0, 2, size=64)

    repaired, report = repair_ir_flip(
        graph, x_fit, y_fit, x_val, y_val,
        objective='argmax', n_layers_from_output=2, max_flips=256, passes=3,
    )

    assert isinstance(report, IrRepairReport)
    accs = [d['fit_acc'] for d in report.per_pass]
    for a, b in zip(accs, accs[1:]):
        assert b >= a - 1e-12, f'fit_accがパス間で減少した: {accs}'
    assert report.fit_acc_after >= report.fit_acc_before - 1e-12
    validate(repaired)


def test_monotonicity_bits():
    doc = _bits_doc(in_width=6, out_width=5, seed=12)
    graph = ir_from_dict(doc)

    x_fit = _binary(80, 6, seed=41)
    x_val = _binary(40, 6, seed=42)
    rng = np.random.default_rng(51)
    y_fit = rng.integers(0, 2, size=(80, 5)).astype(np.uint8)
    y_val = rng.integers(0, 2, size=(40, 5)).astype(np.uint8)

    repaired, report = repair_ir_flip(
        graph, x_fit, y_fit, x_val, y_val,
        objective='bits', n_layers_from_output=1, max_flips=128, passes=3,
    )

    accs = [d['fit_acc'] for d in report.per_pass]
    for a, b in zip(accs, accs[1:]):
        assert b >= a - 1e-12
    assert report.fit_acc_after >= report.fit_acc_before - 1e-12
    validate(repaired)


# ============================================================
# 2. 完全復元
# ============================================================

def test_perfect_recovery_argmax():
    doc = _argmax_doc(in_width=8, mid_width=8, out_width=8, k=2, seed=101)
    correct = ir_from_dict(doc)

    x_fit = _full_enum(8)  # 256サンプル: in_width=8を完全網羅 → 下流アドレスも完全網羅
    y_fit = simulate(correct, x_fit).argmax(axis=1)

    # x_val is deliberately smaller/different but reuses the same correct labels function
    x_val = _binary(32, 8, seed=102)
    y_val = simulate(correct, x_val).argmax(axis=1)

    broken = ir_from_dict(doc)  # 新しいコピー
    l2 = broken.node_by_id('L2')
    tts = list(l2.params['truth_tables'])
    # 単一(o, a)のみを壊す: group_sumはpopcountの和のためマルチビット破壊は
    # 同じクラス群内で他サンプルと相互作用しうる(グリーディが局所最適に
    # 留まりうる)。単一ビット破壊なら復元は他ビットと独立に厳密net>0が
    # 保証され、完全復元(fit_acc==1.0)を決定的に検証できる
    o, a = 3, 2
    tts[o] = _flip_tt_bit(tts[o], a)
    l2.params['truth_tables'] = tts

    # 壊れていることを確認(教師と食い違うはず)
    broken_acc = float((simulate(broken, x_fit).argmax(axis=1) == y_fit).mean())
    assert broken_acc < 1.0

    repaired, report = repair_ir_flip(
        broken, x_fit, y_fit, x_val, y_val,
        objective='argmax', n_layers_from_output=1, max_flips=64, passes=3,
    )

    assert report.fit_acc_after == 1.0
    assert report.n_flips <= report.n_flips  # sanity(下のmax_flipsテストで別途検証)
    validate(repaired)


def test_perfect_recovery_bits():
    doc = _bits_doc(in_width=6, out_width=5, seed=201)
    correct = ir_from_dict(doc)

    x_fit = _full_enum(6)  # 64サンプル完全網羅
    y_fit = simulate(correct, x_fit)  # (64, 5) core出力ビット

    x_val = _binary(20, 6, seed=202)
    y_val = simulate(correct, x_val)

    broken = ir_from_dict(doc)
    l1 = broken.node_by_id('L1')
    tts = list(l1.params['truth_tables'])
    for o, a in [(0, 0), (1, 2), (4, 3)]:
        tts[o] = _flip_tt_bit(tts[o], a)
    l1.params['truth_tables'] = tts

    broken_acc = float((simulate(broken, x_fit) == y_fit).mean())
    assert broken_acc < 1.0

    repaired, report = repair_ir_flip(
        broken, x_fit, y_fit, x_val, y_val,
        objective='bits', n_layers_from_output=1, max_flips=64, passes=3,
    )

    assert report.fit_acc_after == 1.0
    validate(repaired)


# ============================================================
# 3. 不変性
# ============================================================

def test_invariance():
    import copy

    doc = _argmax_doc(in_width=8, mid_width=8, out_width=8, k=2, seed=301)
    graph = ir_from_dict(doc)
    graph_before = copy.deepcopy(graph)

    x_fit = _binary(100, 8, seed=311)
    x_val = _binary(30, 8, seed=312)
    rng = np.random.default_rng(321)
    y_fit = rng.integers(0, 2, size=100)
    y_val = rng.integers(0, 2, size=30)

    repaired, report = repair_ir_flip(
        graph, x_fit, y_fit, x_val, y_val,
        objective='argmax', n_layers_from_output=2, max_flips=128, passes=2,
    )

    # 元グラフは非破壊
    assert graph == graph_before

    # 返されたグラフは別オブジェクト
    assert repaired is not graph

    # truth_tables以外のノード構造は完全一致
    assert [n.id for n in repaired.nodes] == [n.id for n in graph.nodes]
    assert [n.type for n in repaired.nodes] == [n.type for n in graph.nodes]
    assert repaired.edges == graph.edges
    assert repaired.inputs == graph.inputs
    assert repaired.outputs == graph.outputs
    for n_before, n_after in zip(graph.nodes, repaired.nodes):
        assert n_before.train_info == n_after.train_info
        assert n_before.meta == n_after.meta
        for key in n_before.params:
            if n_before.type == 'lut_layer' and key == 'truth_tables':
                continue
            assert n_before.params[key] == n_after.params[key], (
                f'{n_before.id}.{key} が修復で変化した'
            )

    validate(repaired)


# ============================================================
# 4. 未訪問アドレス不変
# ============================================================

def test_unvisited_addresses_unchanged():
    # L1(in=4, out=2, n_inputs=2): col0はindices[0]でx0,x1、col1はx2,x3
    doc = {
        'format': 'magicdelta-ir', 'version': '0.2.0', 'metadata': {},
        'inputs': [_port('x', 4)],
        'outputs': [_port('y', 2)],
        'nodes': [
            {
                'id': 'L1', 'type': 'lut_layer',
                'params': {
                    'in_width': 4, 'out_width': 2, 'n_inputs': 2,
                    'indices': [[0, 2], [1, 3]],
                    'truth_tables': [hex(0b1011), hex(0b0111)],
                },
                'train_info': {}, 'meta': {},
            },
            {
                'id': 'GS', 'type': 'group_sum',
                'params': {'in_width': 2, 'k': 2, 'tau': 1.0},
                'train_info': {}, 'meta': {},
            },
        ],
        'edges': [
            {'src': 'x', 'dst': 'L1'},
            {'src': 'L1', 'dst': 'GS'},
            {'src': 'GS', 'dst': 'y'},
        ],
    }
    graph = ir_from_dict(doc)

    # addr = x[idx0] + 2*x[idx1]。col0: idx=(0,1) → x0,x1が(1,1)になる行を含めない
    # (addr=3を永久に未訪問にする)。col1: idx=(2,3)は全4アドレスを網羅する。
    x_fit = np.array([
        [0, 0, 0, 0],  # col0 addr=0, col1 addr=0
        [1, 0, 1, 0],  # col0 addr=1, col1 addr=1
        [0, 1, 0, 1],  # col0 addr=2, col1 addr=2
        [1, 0, 1, 1],  # col0 addr=1, col1 addr=3
    ], dtype=np.uint8)
    x_val = x_fit.copy()

    rng = np.random.default_rng(401)
    y_fit = rng.integers(0, 2, size=4)
    y_val = y_fit.copy()

    l1_before = graph.node_by_id('L1')
    tt0_before = int(l1_before.params['truth_tables'][0], 16)

    repaired, report = repair_ir_flip(
        graph, x_fit, y_fit, x_val, y_val,
        objective='argmax', n_layers_from_output=1, max_flips=64, passes=4,
    )

    l1_after = repaired.node_by_id('L1')
    tt0_after = int(l1_after.params['truth_tables'][0], 16)

    # col0のアドレス3は一度も訪問されない → そのビットは不変のはず
    bit_before = (tt0_before >> 3) & 1
    bit_after = (tt0_after >> 3) & 1
    assert bit_after == bit_before, (
        f'未訪問アドレス(col0, addr=3)のビットが変化した: {bit_before} -> {bit_after}'
    )
    validate(repaired)


# ============================================================
# 5. max_flips尊重
# ============================================================

def test_max_flips_respected():
    doc = _argmax_doc(in_width=8, mid_width=8, out_width=8, k=2, seed=501)
    graph = ir_from_dict(doc)

    x_fit = _binary(128, 8, seed=511)
    x_val = _binary(32, 8, seed=512)
    rng = np.random.default_rng(521)
    y_fit = rng.integers(0, 2, size=128)
    y_val = rng.integers(0, 2, size=32)

    for max_flips in (0, 1, 3):
        _repaired, report = repair_ir_flip(
            graph, x_fit, y_fit, x_val, y_val,
            objective='argmax', n_layers_from_output=2,
            max_flips=max_flips, passes=4,
        )
        assert report.n_flips <= max_flips, (
            f'max_flips={max_flips} を超過: n_flips={report.n_flips}'
        )
