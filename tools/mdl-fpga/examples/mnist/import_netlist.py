"""社内先行実験 demo ネットリスト(op/a/b 形式)→ magicdelta-ir v0.2.0 変換。

変換元(社内先行実験 demo/model/netlist.json):
    {input_bits: 784,
     binarization: "MNIST ToTensor([0,1]) > 0.5, row-major 28x28 flatten",
     layers: [[{op,a,b}, ...] × 6],       # 各層4000ゲート、layer0はx参照
     output: {classes: 10, group_size: 400, tau: 15.0}}

op→tt の規約差(ここが本質):
    demo: idx = 2*A + B、out = (op >> (3-idx)) & 1   (先行実験 netlist2verilog.py)
    IR:   addr = A + 2*B(slot0=A が LSB)、out = (tt >> addr) & 1
全 16op×4 パターンの自己テストを変換前に必ず実行する。

精度 gate(bit-honest): 変換後 IR の ir.simulate による先頭 N 枚の正解数が
netlist_meta.json の記録(first1000_acc=0.9380 → 938 枚)と完全一致しなければ
失敗させる。使い方:

    uv run python tools/mdl-fpga/examples/mnist/gen_data.py
    uv run python tools/mdl-fpga/examples/mnist/import_netlist.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from griffith_logic.ir.serialize import ir_from_dict, save_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

HERE = Path(__file__).resolve().parent
# 社内先行実験の demo モデル(netlist.json)は本リポジトリに同梱しない。
# 既定はこのディレクトリ直下を見て、無ければ --netlist で明示指定する。
DEFAULT_NETLIST = Path(__file__).resolve().parent / 'netlist.json'
DEFAULT_DATA = HERE / 'mnist_test_1000.npz'
DEFAULT_OUT = HERE / 'mnist24k.ir.json'


def op_to_tt(op: int) -> int:
    """demo op コード → IR 真理値表(N=2)。"""
    tt = 0
    for a in (0, 1):
        for b in (0, 1):
            idx = 2 * a + b                    # demo 規約
            bit = (op >> (3 - idx)) & 1
            addr = a + 2 * b                   # IR 規約(slot0=a=LSB)
            tt |= bit << addr
    return tt


def op_conversion_self_test() -> None:
    """全 16op × 4 パターンで demo 規約と IR 規約の一致を検証する。"""
    for op in range(16):
        tt = op_to_tt(op)
        for a in (0, 1):
            for b in (0, 1):
                demo_out = (op >> (3 - (2 * a + b))) & 1
                ir_out = (tt >> (a + 2 * b)) & 1
                assert ir_out == demo_out, (op, a, b, tt)


def netlist_to_ir_dict(doc: dict, name: str = 'mnist24k') -> dict:
    """demo ネットリスト → IR dict(ir_from_dict 互換)。"""
    in_bits = int(doc['input_bits'])
    out = doc['output']
    k, tau = int(out['classes']), float(out['tau'])

    nodes, edges = [], []
    prev_id, prev_w = 'x', in_bits
    for li, layer in enumerate(doc['layers']):
        w = len(layer)
        node_id = f'l{li}'
        nodes.append({
            'id': node_id, 'type': 'lut_layer',
            'params': {
                'in_width': prev_w, 'out_width': w, 'n_inputs': 2,
                'indices': [[int(g['a']) for g in layer],
                            [int(g['b']) for g in layer]],
                'truth_tables': [hex(op_to_tt(int(g['op']))) for g in layer],
            },
        })
        edges.append({'src': prev_id, 'dst': node_id})
        prev_id, prev_w = node_id, w

    assert prev_w == k * int(out['group_size']), (prev_w, out)
    nodes.append({'id': 'gs', 'type': 'group_sum',
                  'params': {'k': k, 'in_width': prev_w, 'tau': tau}})
    edges.append({'src': prev_id, 'dst': 'gs'})
    edges.append({'src': 'gs', 'dst': 'y'})

    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {
            'name': name,
            'origin': '社内先行実験 demo/model/netlist.json (先行実験, 24K gates)',
            'binarization': doc.get('binarization'),
        },
        'inputs': [{'id': 'x', 'width': in_bits, 'shape': [28, 28],
                    'semantics': 'image_28x28'}],
        'outputs': [{'id': 'y', 'width': k, 'semantics': 'class_scores'}],
        'nodes': nodes, 'edges': edges,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--netlist', default=str(DEFAULT_NETLIST))
    ap.add_argument('--data', default=str(DEFAULT_DATA),
                    help='gen_data.py が生成した npz(精度gate用)')
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--expect-correct', type=int, default=938,
                    help='先頭1000枚の期待正解数(netlist_meta.json: 0.9380)')
    args = ap.parse_args()

    op_conversion_self_test()
    print('op→tt 自己テスト: OK(16op×4パターン)')

    with open(args.netlist, encoding='utf-8') as f:
        doc = json.load(f)
    graph = ir_from_dict(netlist_to_ir_dict(doc))
    validate(graph, target='hdl')
    n_gates = sum(n.params['out_width'] for n in graph.nodes
                  if n.type == 'lut_layer')
    print(f'IR 構築: {len(graph.nodes)} ノード / {n_gates} ゲート')

    data = np.load(args.data)
    images, labels = data['images_bits'], data['labels']
    print(f'精度 gate: {len(images)} 枚を ir.simulate で推論中...')
    scores = simulate(graph, images.astype(np.int64))
    preds = np.argmax(scores, axis=1)
    correct = int((preds == labels).sum())
    acc = correct / len(images)
    print(f'  正解 {correct}/{len(images)}(acc={acc:.4f})')
    if correct != args.expect_correct:
        print(f'NG: 期待 {args.expect_correct} と不一致 → 変換ミスの疑い')
        return 1

    save_ir(graph, args.out)
    print(f'保存: {args.out}(精度gate PASS: 記録値と完全一致)')
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
