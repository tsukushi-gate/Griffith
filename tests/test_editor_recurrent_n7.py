"""エディタのRecurrent時間展開(LSTM/GRU unroll)の等価性検証。

フロント(tools/editor/frontend/src/blocks/recurrentUnit.ts)が生成する unroll IR と
**同一構成**のIRを Python 側で再構成し、
    - validate() 通過
    - simulate(hard) が blocks/recurrent.py の scan_cell(hard/eval) と **ビット一致**
を保証する(北極星: GUI生成物がバックエンドの validate/simulate を通り、参照セルと厳密一致)。

構成(recurrentUnit.ts と同一代数):
    ゲート/候補 = LutStack → lut_layer列(セルの真理値表を truth_tables_from_lut_layer で充填)
    状態更新 = ブール演算: and=lut(tt=0x8) / not=lut(tt=0x1) / passthrough=lut(tt=0xa) / or=addノード
    mux(sel,a,b) = (sel∧a) ∨ (¬sel∧b)。unrollは各step独立層だが同一重み充填でtied scan_cellと一致。

本テストは griffith_logic 本体を一切変更せず、読み取り(extract/validate/simulate)のみで確認する。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.blocks import (  # noqa: E402
    LogicGRUCell,
    LogicLSTMCell,
    scan_cell,
)
from griffith_logic.export.discretize import truth_tables_from_lut_layer  # noqa: E402
from griffith_logic.ir import ir_from_dict, simulate, validate  # noqa: E402
from griffith_logic.ir.serialize import tt_to_hex  # noqa: E402

# recurrentUnit.ts と同一の状態代数 tt 定数(n_inputs=2, addr=in0*1+in1*2)。
AND_TT = '0x8'  # out=1 は addr=3 のみ → bit3
NOT_TT = '0x1'  # 両slot同一入力、¬in0 → addr0
PASS_TT = '0xa'  # slot0素通し → addr1,addr3


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


def _lut(node_id, in_w, out_w, n_inputs, indices, tt_hex):
    return {
        'id': node_id, 'type': 'lut_layer',
        'params': {'in_width': in_w, 'out_width': out_w, 'n_inputs': n_inputs,
                   'indices': indices, 'truth_tables': tt_hex},
        'train_info': {}, 'meta': {},
    }


def _fixed_bin(node_id, width, tt):
    """固定ブールlut(2W→W): 出力oは [busA[o], busB[o]] を読む。"""
    a = list(range(width))
    b = list(range(width, 2 * width))
    return _lut(node_id, 2 * width, width, 2, [a, b], [tt] * width)


def _not(node_id, width):
    idx = list(range(width))
    return _lut(node_id, width, width, 2, [idx, list(idx)], [NOT_TT] * width)


def _slice(node_id, total, offset, out_w):
    idx = [offset + o for o in range(out_w)]
    return _lut(node_id, total, out_w, 2, [idx, list(idx)], [PASS_TT] * out_w)


def _concat(node_id, in_widths):
    return {'id': node_id, 'type': 'concat',
            'params': {'in_widths': list(in_widths)}, 'train_info': {}, 'meta': {}}


def _or_add(node_id, width):
    return {'id': node_id, 'type': 'add',
            'params': {'width': width, 'mode': 'or'}, 'train_info': {}, 'meta': {}}


def _const0(node_id, width):
    return {'id': node_id, 'type': 'constant',
            'params': {'width': width, 'bits': [0] * width}, 'train_info': {}, 'meta': {}}


def _gate_nodes(stack, prefix, z_src, nodes, edges):
    """LutStack(セル)を lut_layer列へ抽出(hard真理値表を充填)。out_id を返す。"""
    prev = z_src
    out_id = None
    for k, layer in enumerate(stack.layers):
        nid = f'{prefix}{k}'
        idx = layer.indices.cpu().numpy().astype(np.int64).tolist()
        tt = tt_to_hex(truth_tables_from_lut_layer(layer).tolist())
        nodes.append(_lut(nid, layer.in_dim, layer.out_dim, layer.n_inputs, idx, tt))
        edges.append({'src': prev, 'dst': nid})
        prev = nid
        out_id = nid
    return out_id


def _build_lstm_ir(cell, unroll, in_width, hidden):
    """recurrentUnit.ts の LSTM 展開と同一構成のIR dictを組む。"""
    total = unroll * in_width
    nodes, edges = [], []
    nodes.append(_slice('rnn_in', total, 0, total))  # 入力分配(恒等)
    nodes.append(_const0('init_h', hidden))
    nodes.append(_const0('init_c', hidden))
    h_prev, c_prev = 'init_h', 'init_c'

    for t in range(unroll):
        xt = f'x{t}'
        nodes.append(_slice(xt, total, t * in_width, in_width))
        edges.append({'src': 'rnn_in', 'dst': xt})
        z = f'z{t}'
        nodes.append(_concat(z, [in_width, hidden]))
        edges.append({'src': xt, 'dst': z, 'dst_slot': 0})
        edges.append({'src': h_prev, 'dst': z, 'dst_slot': 1})

        f_out = _gate_nodes(cell.f_gate, f'f{t}_', z, nodes, edges)
        i_out = _gate_nodes(cell.i_gate, f'i{t}_', z, nodes, edges)
        o_out = _gate_nodes(cell.o_gate, f'o{t}_', z, nodes, edges)
        cand_out = _gate_nodes(cell.c_gate, f'g{t}_', z, nodes, edges)

        # ic = and(i, cand)
        nodes.append(_concat(f'ic_cat{t}', [hidden, hidden]))
        edges.append({'src': i_out, 'dst': f'ic_cat{t}', 'dst_slot': 0})
        edges.append({'src': cand_out, 'dst': f'ic_cat{t}', 'dst_slot': 1})
        nodes.append(_fixed_bin(f'ic{t}', hidden, AND_TT))
        edges.append({'src': f'ic_cat{t}', 'dst': f'ic{t}'})

        # notf = not(f)
        nodes.append(_not(f'nf{t}', hidden))
        edges.append({'src': f_out, 'dst': f'nf{t}'})

        # t1 = and(f, c_prev)
        nodes.append(_concat(f't1cat{t}', [hidden, hidden]))
        edges.append({'src': f_out, 'dst': f't1cat{t}', 'dst_slot': 0})
        edges.append({'src': c_prev, 'dst': f't1cat{t}', 'dst_slot': 1})
        nodes.append(_fixed_bin(f't1_{t}', hidden, AND_TT))
        edges.append({'src': f't1cat{t}', 'dst': f't1_{t}'})

        # t2 = and(notf, ic)
        nodes.append(_concat(f't2cat{t}', [hidden, hidden]))
        edges.append({'src': f'nf{t}', 'dst': f't2cat{t}', 'dst_slot': 0})
        edges.append({'src': f'ic{t}', 'dst': f't2cat{t}', 'dst_slot': 1})
        nodes.append(_fixed_bin(f't2_{t}', hidden, AND_TT))
        edges.append({'src': f't2cat{t}', 'dst': f't2_{t}'})

        # c_t = or(t1, t2)
        c_new = f'c{t}'
        nodes.append(_or_add(c_new, hidden))
        edges.append({'src': f't1_{t}', 'dst': c_new, 'dst_slot': 0})
        edges.append({'src': f't2_{t}', 'dst': c_new, 'dst_slot': 1})

        # h_t = and(o, c_t)
        nodes.append(_concat(f'hcat{t}', [hidden, hidden]))
        edges.append({'src': o_out, 'dst': f'hcat{t}', 'dst_slot': 0})
        edges.append({'src': c_new, 'dst': f'hcat{t}', 'dst_slot': 1})
        h_new = f'h{t}'
        nodes.append(_fixed_bin(h_new, hidden, AND_TT))
        edges.append({'src': f'hcat{t}', 'dst': h_new})

        h_prev, c_prev = h_new, c_new

    return _wrap(nodes, edges, total, hidden, h_prev)


def _build_gru_ir(cell, unroll, in_width, hidden):
    """recurrentUnit.ts の GRU 展開と同一構成のIR dictを組む。"""
    total = unroll * in_width
    nodes, edges = [], []
    nodes.append(_slice('rnn_in', total, 0, total))
    nodes.append(_const0('init_h', hidden))
    h_prev = 'init_h'

    for t in range(unroll):
        xt = f'x{t}'
        nodes.append(_slice(xt, total, t * in_width, in_width))
        edges.append({'src': 'rnn_in', 'dst': xt})
        z = f'z{t}'
        nodes.append(_concat(z, [in_width, hidden]))
        edges.append({'src': xt, 'dst': z, 'dst_slot': 0})
        edges.append({'src': h_prev, 'dst': z, 'dst_slot': 1})

        zg_out = _gate_nodes(cell.z_gate, f'z{t}_', z, nodes, edges)
        rg_out = _gate_nodes(cell.r_gate, f'r{t}_', z, nodes, edges)

        # rh = and(r, h_prev)
        nodes.append(_concat(f'rhcat{t}', [hidden, hidden]))
        edges.append({'src': rg_out, 'dst': f'rhcat{t}', 'dst_slot': 0})
        edges.append({'src': h_prev, 'dst': f'rhcat{t}', 'dst_slot': 1})
        nodes.append(_fixed_bin(f'rh{t}', hidden, AND_TT))
        edges.append({'src': f'rhcat{t}', 'dst': f'rh{t}'})

        # cand = h_cand(concat([x, rh]))
        nodes.append(_concat(f'gin{t}', [in_width, hidden]))
        edges.append({'src': xt, 'dst': f'gin{t}', 'dst_slot': 0})
        edges.append({'src': f'rh{t}', 'dst': f'gin{t}', 'dst_slot': 1})
        cand_out = _gate_nodes(cell.h_cand, f'g{t}_', f'gin{t}', nodes, edges)

        # h_t = mux(z, cand, h_prev) = (z∧cand) ∨ (¬z∧h_prev)
        nodes.append(_not(f'nz{t}', hidden))
        edges.append({'src': zg_out, 'dst': f'nz{t}'})

        nodes.append(_concat(f't1cat{t}', [hidden, hidden]))
        edges.append({'src': zg_out, 'dst': f't1cat{t}', 'dst_slot': 0})
        edges.append({'src': cand_out, 'dst': f't1cat{t}', 'dst_slot': 1})
        nodes.append(_fixed_bin(f't1_{t}', hidden, AND_TT))
        edges.append({'src': f't1cat{t}', 'dst': f't1_{t}'})

        nodes.append(_concat(f't2cat{t}', [hidden, hidden]))
        edges.append({'src': f'nz{t}', 'dst': f't2cat{t}', 'dst_slot': 0})
        edges.append({'src': h_prev, 'dst': f't2cat{t}', 'dst_slot': 1})
        nodes.append(_fixed_bin(f't2_{t}', hidden, AND_TT))
        edges.append({'src': f't2cat{t}', 'dst': f't2_{t}'})

        h_new = f'h{t}'
        nodes.append(_or_add(h_new, hidden))
        edges.append({'src': f't1_{t}', 'dst': h_new, 'dst_slot': 0})
        edges.append({'src': f't2_{t}', 'dst': h_new, 'dst_slot': 1})
        h_prev = h_new

    return _wrap(nodes, edges, total, hidden, h_prev)


def _wrap(nodes, edges, in_width, out_width, exit_id):
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {'name': 'n7_recurrent'},
        'inputs': [_port('x', in_width)],
        'outputs': [_port('y', out_width)],
        'nodes': nodes,
        'edges': [{'src': 'x', 'dst': 'rnn_in'}, *edges, {'src': exit_id, 'dst': 'y'}],
    }


def _scan_final_h(cell, xs, is_lstm):
    """scan_cell(hard/eval)を{0,1}初期状態で回し最終隠れ状態hを返す(np int)。"""
    cell.eval()
    b = xs.shape[0]
    hidden = cell.hidden_size
    with torch.no_grad():
        if is_lstm:
            state = (torch.zeros(b, hidden), torch.zeros(b, hidden))
            _outs, (h, _c) = scan_cell(cell, xs, state)
        else:
            state = torch.zeros(b, hidden)
            _outs, h = scan_cell(cell, xs, state)
    return h.numpy().astype(np.int64)


def _run_case(cell, builder, is_lstm, unroll, in_width, hidden, batch=6, seed=7):
    doc = builder(cell, unroll, in_width, hidden)
    graph = ir_from_dict(doc)
    assert validate(graph) == [], f'validate失敗: {validate(graph)}'

    g = torch.Generator().manual_seed(seed)
    xs = (torch.rand(batch, unroll, in_width, generator=g) > 0.5).float()
    x_flat = xs.reshape(batch, unroll * in_width).numpy().astype(np.uint8)

    sim = simulate(graph, x_flat)  # (batch, hidden)
    ref = _scan_final_h(cell, xs, is_lstm)
    assert sim.shape == ref.shape == (batch, hidden)
    assert np.array_equal(sim, ref), (
        f'ビット不一致(不一致数={int((sim != ref).sum())}/{sim.size})'
    )


# ---------------------------------------------------------------------------
# LSTM 等価性(幅・unroll・depthを変えて hard scan_cell とビット一致)
# ---------------------------------------------------------------------------

def test_lstm_equivalence_w4_t4_depth1():
    cell = LogicLSTMCell(4, 4, depth=1, seed=0)
    _run_case(cell, _build_lstm_ir, True, unroll=4, in_width=4, hidden=4)


def test_lstm_equivalence_w4_t8_depth1():
    cell = LogicLSTMCell(4, 4, depth=1, seed=1)
    _run_case(cell, _build_lstm_ir, True, unroll=8, in_width=4, hidden=4)


def test_lstm_equivalence_depth2():
    """ゲートLutStack depth=2(多段)でも一致する。"""
    cell = LogicLSTMCell(3, 4, depth=2, seed=2)
    _run_case(cell, _build_lstm_ir, True, unroll=5, in_width=3, hidden=4)


def test_lstm_equivalence_asym_widths():
    """入力幅≠隠れ幅でスライス/concatの境界が正しい。"""
    cell = LogicLSTMCell(2, 6, depth=1, seed=3)
    _run_case(cell, _build_lstm_ir, True, unroll=4, in_width=2, hidden=6)


# ---------------------------------------------------------------------------
# GRU 等価性
# ---------------------------------------------------------------------------

def test_gru_equivalence_w4_t4_depth1():
    cell = LogicGRUCell(4, 4, depth=1, seed=0)
    _run_case(cell, _build_gru_ir, False, unroll=4, in_width=4, hidden=4)


def test_gru_equivalence_w4_t8_depth1():
    cell = LogicGRUCell(4, 4, depth=1, seed=1)
    _run_case(cell, _build_gru_ir, False, unroll=8, in_width=4, hidden=4)


def test_gru_equivalence_depth2_asym():
    cell = LogicGRUCell(2, 5, depth=2, seed=2)
    _run_case(cell, _build_gru_ir, False, unroll=6, in_width=2, hidden=5)
