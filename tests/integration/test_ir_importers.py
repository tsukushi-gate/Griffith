"""統合: 他形式のネットリスト → IR 変換器(ir/importers.py)。

小さな合成フィクスチャ(乱数の op / 配線)で、変換後 IR の simulate が元形式の規則を
そのまま書いた numpy 参照と全数一致することを確認する。内部資産そのものは同梱しない。
"""

import numpy as np
import pytest

from griffith_logic.export.step_contract import check_step_ir
from griffith_logic.ir.importers import (
    op_conversion_self_test,
    op_to_tt,
    opab_netlist_to_ir,
    rddlgn_netlist_to_step_ir,
)
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _op_eval(op, a, b):
    """社内 op ニブルの規則: idx = 2A + B、out = (op >> (3 − idx)) & 1"""
    return (op >> (3 - (2 * a + b))) & 1


class TestOpConvention:
    def test_self_test_and_known_ops(self):
        op_conversion_self_test()
        # op=8 → 1000b: A=B=0 のときだけ 1 = NOR。IR の tt は addr 0 のみ 1 = 0b0001
        assert op_to_tt(0b1000) == 0b0001
        # op=1 → 0001b: A=B=1 のときだけ 1 = AND。IR では addr 3 = 0b1000
        assert op_to_tt(0b0001) == 0b1000
        # op=6 → 0110b: XOR。IR は addr 1, 2 → 0b0110
        assert op_to_tt(0b0110) == 0b0110
        with pytest.raises(ValueError):
            op_to_tt(16)


def _opab_fixture(rng, in_bits=12, widths=(20, 20, 8), classes=2, assign='contig'):
    layers, prev = [], in_bits
    for w in widths:
        layers.append([{'op': int(rng.integers(0, 16)), 'a': int(rng.integers(0, prev)),
                        'b': int(rng.integers(0, prev))} for _ in range(w)])
        prev = w
    return {'input_bits': in_bits, 'binarization': 'synthetic', 'layers': layers,
            'output': {'classes': classes, 'group_size': prev // classes, 'tau': 3.0,
                       'assign': assign}}


def _opab_reference(doc, x):
    h = x.astype(np.int64)
    for layer in doc['layers']:
        a = np.array([g['a'] for g in layer])
        b = np.array([g['b'] for g in layer])
        op = np.array([g['op'] for g in layer])
        h = (op[None, :] >> (3 - (2 * h[:, a] + h[:, b]))) & 1
    out = doc['output']
    k, gs = out['classes'], out['group_size']
    if out.get('assign') == 'stride':
        return h.reshape(h.shape[0], gs, k).sum(axis=1)
    return h.reshape(h.shape[0], k, gs).sum(axis=2)


class TestOpAbNetlist:
    @pytest.mark.parametrize('assign', ['contig', 'stride'])
    def test_simulate_matches_reference(self, assign):
        rng = np.random.default_rng(0)
        doc = _opab_fixture(rng, assign=assign)
        ir = opab_netlist_to_ir(doc, name='fixture')
        validate(ir, target='hdl')
        x = rng.integers(0, 2, size=(200, doc['input_bits'])).astype(np.int64)
        assert np.array_equal(simulate(ir, x), _opab_reference(doc, x))
        assert ir.nodes[-1].type == 'group_sum'
        assert ir.nodes[-1].params['tau'] == 3.0
        assert ir.outputs[0].semantics == 'class_scores'
        types = [n.type for n in ir.nodes]
        assert types.count('lut_layer') == (4 if assign == 'stride' else 3)
        # JSON 往復
        ir2 = ir_from_dict(ir_to_dict(ir))
        assert np.array_equal(simulate(ir2, x), simulate(ir, x))

    def test_without_output_returns_raw_bits(self):
        rng = np.random.default_rng(1)
        doc = _opab_fixture(rng)
        del doc['output']
        ir = opab_netlist_to_ir(doc)
        x = rng.integers(0, 2, size=(50, doc['input_bits'])).astype(np.int64)
        h = x
        for layer in doc['layers']:
            a = np.array([g['a'] for g in layer])
            b = np.array([g['b'] for g in layer])
            op = np.array([g['op'] for g in layer])
            h = (op[None, :] >> (3 - (2 * h[:, a] + h[:, b]))) & 1
        assert np.array_equal(simulate(ir, x), h)

    def test_errors(self):
        rng = np.random.default_rng(2)
        doc = _opab_fixture(rng, widths=(10, 9))        # 9 は classes 2 で割り切れない
        doc['output']['group_size'] = 4
        with pytest.raises(ValueError):
            opab_netlist_to_ir(doc)
        bad = _opab_fixture(rng)
        bad['layers'][0][0]['a'] = 99
        with pytest.raises(ValueError):
            opab_netlist_to_ir(bad)
        with pytest.raises(ValueError):
            opab_netlist_to_ir({'input_bits': 4, 'layers': []})


def _rddlgn_fixture(rng, e_dim=8, w=16, vocab=4, gsz=4, n_m=2):
    def layer(name, group, in_dim, out_dim):
        return {'name': name, 'group': group, 'in_dim': in_dim, 'out_dim': out_dim,
                's_a': int(rng.integers(0, in_dim)), 's_b': int(rng.integers(0, in_dim)),
                'ops': ''.join(f'{int(v):x}' for v in rng.integers(0, 16, size=out_dim))}
    layers = [layer('L0', 'L', e_dim, w), layer('L1', 'L', w, w),
              layer('P0', 'P', 2 * w, w), layer('P1', 'P', w, w)]
    prev = w
    for i in range(n_m):
        out = vocab * gsz if i == n_m - 1 else w
        layers.append(layer(f'M{i}', 'M', prev, out))
        prev = out
    emb = [''.join(str(int(v)) for v in rng.integers(0, 2, size=e_dim)) for _ in range(vocab)]
    return {'format': 'rddlgn-netlist-v1', 'W': w, 'E': e_dim, 'vocab': vocab,
            'group_size': gsz, 'layers': layers, 'emb': emb}


def _rot_layer_np(layer, x):
    k = np.arange(layer['out_dim'])
    a = x[(k + layer['s_a']) % layer['in_dim']]
    b = x[(k + layer['s_b']) % layer['in_dim']]
    ops = np.array([int(ch, 16) for ch in layer['ops']])
    return (ops >> (3 - (2 * a + b))) & 1


def _greedy_reference(doc, prompt, n_gen):
    """netlist の規則どおりの numpy 参照(状態 0 から、prompt を読んでから n_gen 生成)"""
    groups = {g: [lay for lay in doc['layers'] if lay['group'] == g] for g in 'LPM'}
    emb = np.array([[int(ch) for ch in row] for row in doc['emb']])
    p = np.zeros(groups['P'][-1]['out_dim'], dtype=np.int64)
    out_tokens = []
    tokens = list(prompt)
    for t in range(len(prompt) + n_gen):
        tok = tokens[t]
        h = emb[tok]
        for lay in groups['L']:
            h = _rot_layer_np(lay, h)
        h = np.concatenate([p, h])
        for lay in groups['P']:
            h = _rot_layer_np(lay, h)
        p = h
        for lay in groups['M']:
            h = _rot_layer_np(lay, h)
        scores = h.reshape(doc['vocab'], doc['group_size']).sum(axis=1)
        nxt = int(np.argmax(scores))        # 同点は小さい ID
        if t >= len(prompt) - 1:
            out_tokens.append(nxt)
            if t + 1 < len(prompt) + n_gen and t + 1 >= len(prompt):
                tokens.append(nxt)
    return out_tokens[:n_gen]


class TestRddlgnNetlist:
    def test_step_ir_matches_reference_generation(self):
        rng = np.random.default_rng(3)
        doc = _rddlgn_fixture(rng)
        graph, spec = rddlgn_netlist_to_step_ir(doc)
        validate(graph, target='hdl')
        cs = check_step_ir(graph, spec['state_width'])
        assert cs.state_in == tuple(spec['state_in'])
        emb = np.array(spec['emb'])
        prompt = [1, 3, 0]
        n_gen = 5
        want = _greedy_reference(doc, prompt, n_gen)

        p = np.array(spec['init_bits'])
        got = []
        tokens = list(prompt)
        k, gsz = spec['group_sum']['k'], spec['group_sum']['group_size']
        for t in range(len(prompt) + n_gen):
            x = np.concatenate([emb[tokens[t]], p])[None, :]
            y = simulate(graph, x)[0]
            p = y[:spec['state_width']]
            scores = y[spec['state_width']:].reshape(k, gsz).sum(axis=1)
            nxt = int(np.argmax(scores))
            if t >= len(prompt) - 1:
                got.append(nxt)
                if t + 1 >= len(prompt) and t + 1 < len(prompt) + n_gen:
                    tokens.append(nxt)
        assert got[:n_gen] == want

    def test_errors(self):
        rng = np.random.default_rng(4)
        doc = _rddlgn_fixture(rng)
        with pytest.raises(ValueError):
            rddlgn_netlist_to_step_ir({**doc, 'format': 'other'})
        bad = _rddlgn_fixture(rng)
        bad['layers'][-1]['out_dim'] = 15
        bad['layers'][-1]['ops'] = bad['layers'][-1]['ops'][:15]
        with pytest.raises(ValueError):
            rddlgn_netlist_to_step_ir(bad)
