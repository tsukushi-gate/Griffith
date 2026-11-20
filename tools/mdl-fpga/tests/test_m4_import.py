"""E1: 社内先行実験ネットリスト取込の変換規約テスト(外部資産不要)。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

# examples/mnist はパッケージ外スクリプトのため直接ロード
_SPEC = importlib.util.spec_from_file_location(
    'mnist_import',
    Path(__file__).resolve().parent.parent / 'examples' / 'mnist'
    / 'import_netlist.py')
mnist_import = importlib.util.module_from_spec(_SPEC)
sys.modules['mnist_import'] = mnist_import
_SPEC.loader.exec_module(mnist_import)


def _demo_eval(layers, out_cfg, x):
    """demo 規約の素朴評価器(先行実験 verify_netlist.py と同式)。"""
    cur = x
    for layer in layers:
        nxt = np.zeros((x.shape[0], len(layer)), dtype=np.int64)
        for gi, g in enumerate(layer):
            a, b = cur[:, g['a']], cur[:, g['b']]
            idx = 2 * a + b
            nxt[:, gi] = (g['op'] >> (3 - idx)) & 1
        cur = nxt
    k, gs = out_cfg['classes'], out_cfg['group_size']
    return cur.reshape(x.shape[0], k, gs).sum(axis=2)


def test_op_conversion_self_test():
    mnist_import.op_conversion_self_test()
    # 代表値の目視固定: AND(op1)→0x8, XOR(op6)→0x6, A(op3)→0xa, TRUE(op15)→0xf
    assert mnist_import.op_to_tt(1) == 0x8
    assert mnist_import.op_to_tt(6) == 0x6
    assert mnist_import.op_to_tt(3) == 0xA
    assert mnist_import.op_to_tt(15) == 0xF
    assert mnist_import.op_to_tt(0) == 0x0


def test_mini_netlist_equivalence():
    """ミニネットリスト(全16op網羅)の IR simulate = demo 評価器 全数一致。"""
    layers = [
        # layer0: 16ゲート(全opを1回ずつ、入力4bitから)
        [{'op': op, 'a': op % 4, 'b': (op + 1) % 4} for op in range(16)],
        # layer1: 4ゲート(前層参照、group_sum用に k=2×group=2)
        [{'op': 6, 'a': 0, 'b': 5}, {'op': 1, 'a': 2, 'b': 7},
         {'op': 7, 'a': 8, 'b': 12}, {'op': 9, 'a': 3, 'b': 15}],
    ]
    out_cfg = {'classes': 2, 'group_size': 2, 'tau': 1.0}
    doc = {'input_bits': 4, 'layers': layers, 'output': out_cfg}

    graph = ir_from_dict(mnist_import.netlist_to_ir_dict(doc, name='mini'))
    x = np.array([[(v >> i) & 1 for i in range(4)] for v in range(16)],
                 dtype=np.int64)
    got = simulate(graph, x)                    # (16, 2) スコア
    exp = _demo_eval(layers, out_cfg, x)
    assert np.array_equal(got, exp), (got, exp)
