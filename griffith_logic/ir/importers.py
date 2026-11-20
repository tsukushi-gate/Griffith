"""他形式のネットリスト → magicdelta-ir。

内部調査リポジトリの学習済み資産(op/a/b 形式の平坦ネットリスト、RDDLGN 文字 LM の
`rddlgn-netlist-v1`)を IR に写し、check / simulate / HDL の経路へ通す。資産そのものは同梱せず、
テストは小さな合成フィクスチャで「変換後 IR の simulate = 元形式の参照実装」を全数照合する。

op/a/b 形式(2 入力ゲートの平坦ネットリスト。`tools/mdl-fpga/examples/mnist/import_netlist.py` の
例示スクリプトをライブラリへ昇格):

    {"input_bits": 784,
     "layers": [[{"op": 0..15, "a": int, "b": int}, ...] × L],   # 層 0 の a / b は入力ビット
     "output": {"classes": 10, "group_size": 400, "tau": 15.0, "assign": "contig"|"stride"(任意)}}

op ニブルの規約(初期の先行実験から使っている社内形式)と IR の真理値表の規約は違う:

    op:  idx = 2·A + B,  out = (op >> (3 − idx)) & 1
    IR:  addr = A + 2·B(slot 0 = A が LSB), out = (tt >> addr) & 1

`op_to_tt` がこれを写す(全 16 op × 4 パターンの自己テスト `op_conversion_self_test`)。

rddlgn-netlist-v1(lab の文字 LM。`models/rddlgn_netlist.py` の dict):

    {"format": "rddlgn-netlist-v1", "W", "E", "vocab", "group_size",
     "layers": [{"name": "L0", "group": "L"|"P"|"M", "in_dim", "out_dim", "s_a", "s_b",
                 "ops": "<16 進 1 桁 × out_dim>"}, ...],      # L*, P*, M* の順
     "emb": ["<'0'/'1' × E>", ...]}                           # vocab 行
    ゲート k の入力 a = x[(k + s_a) mod in_dim]、b = x[(k + s_b) mod in_dim]、出力は op ニブル。
    1 時刻: l = L(emb[tok])、p = P([p_prev ; l])、m = M(p)。状態 p の初期値は 0。
    クラス c のスコア = m[c·group_size : (c+1)·group_size] の 1 の数、greedy は同点で小さい ID。

`rddlgn_netlist_to_step_ir` は 1 時刻分を 1 ステップ IR(docs/reference/step-ir-contract.md:
x = [emb_t | p_prev]、y = [p_t | m])にする。埋め込み表・argmax・サンプラは IR の外(付帯情報)。
"""

from __future__ import annotations

import numpy as np

from .schema import IREdge, IRGraph, IRNode, IRPort
from .serialize import tt_to_hex

__all__ = [
    'op_conversion_self_test',
    'op_to_tt',
    'opab_netlist_to_ir',
    'rddlgn_netlist_to_step_ir',
]

_IDENTITY_TT_1IN = 0x2


def op_to_tt(op: int) -> int:
    """社内 op ニブル(idx = 2A + B、bit = (op >> (3 − idx)) & 1)→ IR の 2 入力真理値表。"""
    op = int(op)
    if not 0 <= op <= 15:
        raise ValueError(f'op は 0..15: {op}')
    tt = 0
    for a in (0, 1):
        for b in (0, 1):
            bit = (op >> (3 - (2 * a + b))) & 1
            tt |= bit << (a + 2 * b)
    return tt


def op_conversion_self_test() -> None:
    """全 16 op × 4 パターンで op 規約と IR 規約の一致を確認する(変換前に呼ぶ)。"""
    for op in range(16):
        tt = op_to_tt(op)
        for a in (0, 1):
            for b in (0, 1):
                assert ((tt >> (a + 2 * b)) & 1) == ((op >> (3 - (2 * a + b))) & 1), (op, a, b)


def _lut_layer_node(node_id: str, in_width: int, idx_a, idx_b, ops) -> IRNode:
    indices = np.stack([np.asarray(idx_a, dtype=np.int64), np.asarray(idx_b, dtype=np.int64)])
    if indices.size and (indices.min() < 0 or indices.max() >= in_width):
        raise ValueError(f'{node_id}: 添字が [0, {in_width}) の範囲外')
    return IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': int(in_width),
            'out_width': int(indices.shape[1]),
            'n_inputs': 2,
            'indices': indices,
            'truth_tables': tt_to_hex([op_to_tt(o) for o in ops]),
        },
        train_info={'imported_from': 'op/a/b'},
    )


def _stride_permutation_node(node_id: str, width: int, k: int) -> IRNode:
    """stride 割当(ゲート i → クラス i mod k)を contig の group_sum に写す転置
    (export/extract.py の GroupSum(assign='stride') 抽出と同じ形)。"""
    m = width // k
    src = np.empty(width, dtype=np.int64)
    for c in range(k):
        for j in range(m):
            src[c * m + j] = j * k + c
    return IRNode(
        id=node_id, type='lut_layer',
        params={'in_width': width, 'out_width': width, 'n_inputs': 1,
                'indices': src.reshape(1, width),
                'truth_tables': tt_to_hex([_IDENTITY_TT_1IN] * width)},
        train_info={'lowered_from': 'GroupSum(assign=stride)'},
    )


def opab_netlist_to_ir(doc: dict, *, name: str = 'netlist') -> IRGraph:
    """op/a/b 形式の平坦ネットリスト → IR(lut_layer 列 + group_sum)。

    `output` が無ければ最終層のビットをそのまま出力する(GroupSum 無し)。
    `output.assign == 'stride'` は転置を挟む(GroupSum(assign='stride') の抽出と同じ形)。
    """
    op_conversion_self_test()
    in_bits = int(doc['input_bits'])
    layers = doc.get('layers') or []
    if not layers:
        raise ValueError('layers が空')
    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    prev_id, prev_w = 'x', in_bits
    for li, layer in enumerate(layers):
        node = _lut_layer_node(
            f'l{li}', prev_w,
            [int(g['a']) for g in layer], [int(g['b']) for g in layer],
            [int(g['op']) for g in layer])
        nodes.append(node)
        edges.append(IREdge(src=prev_id, dst=node.id))
        prev_id, prev_w = node.id, node.params['out_width']

    out = doc.get('output')
    semantics = None
    if out is not None:
        k = int(out['classes'])
        if prev_w % k != 0:
            raise ValueError(f'最終層幅 {prev_w} がクラス数 {k} で割り切れない')
        if 'group_size' in out and int(out['group_size']) * k != prev_w:
            raise ValueError(
                f"group_size({out['group_size']}) × classes({k}) != 最終層幅 {prev_w}")
        if out.get('assign', 'contig') == 'stride':
            perm = _stride_permutation_node('gs_permute', prev_w, k)
            nodes.append(perm)
            edges.append(IREdge(src=prev_id, dst=perm.id))
            prev_id = perm.id
        gs = IRNode(id='gs', type='group_sum',
                    params={'in_width': prev_w, 'k': k, 'tau': float(out.get('tau', 1.0))})
        nodes.append(gs)
        edges.append(IREdge(src=prev_id, dst=gs.id))
        prev_id, prev_w = gs.id, k
        semantics = 'class_scores'
    edges.append(IREdge(src=prev_id, dst='y'))
    metadata = {'name': name, 'imported_from': 'op/a/b netlist'}
    if doc.get('binarization'):
        metadata['binarization'] = doc['binarization']
    return IRGraph(
        metadata=metadata,
        inputs=[IRPort(id='x', width=in_bits, encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=prev_w, semantics=semantics)],
        nodes=nodes, edges=edges,
    )


def _rot_indices(out_dim: int, in_dim: int, s_a: int, s_b: int):
    k = np.arange(out_dim, dtype=np.int64)
    return (k + int(s_a)) % in_dim, (k + int(s_b)) % in_dim


def _slice_node(node_id: str, in_width: int, start: int, width: int) -> IRNode:
    return IRNode(
        id=node_id, type='lut_layer',
        params={'in_width': in_width, 'out_width': width, 'n_inputs': 1,
                'indices': np.arange(start, start + width, dtype=np.int64).reshape(1, width),
                'truth_tables': tt_to_hex([_IDENTITY_TT_1IN] * width)},
        train_info={'lowered_from': 'InputSlice'},
    )


def rddlgn_netlist_to_step_ir(doc: dict, *, name: str = 'rddlgn_charlm_step'
                              ) -> tuple[IRGraph, dict]:
    """rddlgn-netlist-v1 → 1 ステップ IR(x = [emb_t | p_prev] → y = [p_t | m])と付帯情報。

    Returns:
        (graph, spec)。spec: state_width / ext_width / readout_width / state_in / state_out /
        init_bits(全 0)/ layout / group_sum({k, group_size, tau: 1.0, assign: 'contig'})/
        emb(vocab × E の 0/1 の list。ラッパ側の埋め込み ROM 用)
    """
    if doc.get('format') != 'rddlgn-netlist-v1':
        raise ValueError(f"format は 'rddlgn-netlist-v1' が必要: {doc.get('format')!r}")
    op_conversion_self_test()
    e_dim, vocab, gsz = int(doc['E']), int(doc['vocab']), int(doc['group_size'])
    groups: dict[str, list[dict]] = {'L': [], 'P': [], 'M': []}
    for layer in doc['layers']:
        g = layer['group']
        if g not in groups:
            raise ValueError(f"layer.group は L / P / M: {g!r}")
        groups[g].append(layer)
    if not groups['L'] or not groups['P'] or not groups['M']:
        raise ValueError('L / P / M の各グループに 1 層以上が必要')
    emb = doc['emb']
    if len(emb) != vocab or any(len(row) != e_dim for row in emb):
        raise ValueError('emb は vocab 行 × E 文字')

    p_width = int(groups['P'][-1]['out_dim'])
    in_width = e_dim + p_width
    x_emb = _slice_node('x_emb', in_width, 0, e_dim)
    x_p = _slice_node('x_p', in_width, e_dim, p_width)
    nodes: list[IRNode] = [x_emb, x_p]
    edges: list[IREdge] = [IREdge(src='x', dst='x_emb'), IREdge(src='x', dst='x_p')]

    def _stack(prefix: str, layers: list[dict], entry_id: str, entry_w: int) -> tuple[str, int]:
        prev_id, prev_w = entry_id, entry_w
        for i, layer in enumerate(layers):
            in_dim, out_dim = int(layer['in_dim']), int(layer['out_dim'])
            if in_dim != prev_w:
                raise ValueError(f"{layer['name']}: in_dim({in_dim}) != 前段幅({prev_w})")
            ops = [int(ch, 16) for ch in layer['ops']]
            if len(ops) != out_dim:
                raise ValueError(f"{layer['name']}: ops の長さ({len(ops)}) != out_dim({out_dim})")
            idx_a, idx_b = _rot_indices(out_dim, in_dim, layer['s_a'], layer['s_b'])
            node = _lut_layer_node(f'{prefix}{i}', in_dim, idx_a, idx_b, ops)
            node.train_info = {'imported_from': 'rddlgn-netlist-v1', 'name': layer['name'],
                               'rot': [int(layer['s_a']), int(layer['s_b'])]}
            nodes.append(node)
            edges.append(IREdge(src=prev_id, dst=node.id))
            prev_id, prev_w = node.id, out_dim
        return prev_id, prev_w

    l_last, l_w = _stack('l', groups['L'], 'x_emb', e_dim)
    p_concat = IRNode(id='p_concat', type='concat', params={'in_widths': [p_width, l_w]})
    nodes.append(p_concat)
    edges += [IREdge(src='x_p', dst='p_concat', dst_slot=0),
              IREdge(src=l_last, dst='p_concat', dst_slot=1)]
    p_last, p_w = _stack('p', groups['P'], 'p_concat', p_width + l_w)
    if p_w != p_width:
        raise ValueError(f'P 最終層幅({p_w})が状態幅({p_width})と不一致')
    m_last, m_w = _stack('m', groups['M'], p_last, p_w)
    if m_w != vocab * gsz:
        raise ValueError(f'M 最終層幅({m_w}) != vocab({vocab}) × group_size({gsz})')
    out = IRNode(id='out', type='concat', params={'in_widths': [p_width, m_w]})
    nodes.append(out)
    edges += [IREdge(src=p_last, dst='out', dst_slot=0),
              IREdge(src=m_last, dst='out', dst_slot=1),
              IREdge(src='out', dst='y')]
    graph = IRGraph(
        metadata={'name': name, 'imported_from': 'rddlgn-netlist-v1',
                  'step_contract': 'docs/reference/step-ir-contract.md'},
        inputs=[IRPort(id='x', width=in_width, encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=p_width + m_w)],
        nodes=nodes, edges=edges,
    )
    spec = {
        'ext_width': e_dim, 'state_width': p_width, 'readout_width': m_w,
        'state_in': [e_dim, in_width], 'state_out': [0, p_width],
        'init_bits': [0] * p_width,
        'layout': {'emb': [0, e_dim], 'state': [e_dim, in_width]},
        'group_sum': {'k': vocab, 'group_size': gsz, 'tau': 1.0, 'assign': 'contig'},
        'emb': [[int(ch) for ch in row] for row in emb],
    }
    return graph, spec
