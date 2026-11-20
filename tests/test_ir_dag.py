"""IR v0.2 多入力DAGテスト。

ダイヤモンド型グラフ(fanout→concat)とadd(or/xor/sum)について:
    - validate通過・topological_order妥当・異常系の日本語エラー
    - build_torch: DAGはIRGraphModule、直列は従来どおりnn.Sequential
    - simulate_trace全ノードキー存在、simulate ⇔ hard forward ビット厳密一致
    - concatビット順規約(slot 0が出力ビット列の先頭=下位ビット側)
    - codegen DAG対応(生成コードexec→forwardがbuild_torchと一致)
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.ir import (
    IRGraphModule,
    IRValidationError,
    build_torch,
    ir_from_dict,
    ir_to_dict,
    simulate,
    simulate_trace,
    validate,
)

BATCH = 64


def _rand_lut_params(in_width, out_width, n_inputs, rng):
    """決定的な乱数LUTパラメータ(independentは任意ttを表現可能)"""
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
        'train_info': {'parameterization': 'independent'},
        'meta': {},
    }


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


def _diamond_doc(seed=11) -> dict:
    """x → n1 → fanoutで n2/n3 → concat → n5 → y のダイヤモンドIR dict"""
    rng = np.random.default_rng(seed)
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': 'diamond'},
        'inputs': [_port('x', 8)],
        'outputs': [_port('y', 8)],
        'nodes': [
            _lut_node('n1_lutlayer', 8, 8, rng),
            _lut_node('n2_lutlayer', 8, 6, rng),
            _lut_node('n3_lutlayer', 8, 10, rng),
            {'id': 'n4_concat', 'type': 'concat',
             'params': {'in_widths': [6, 10]}, 'train_info': {}, 'meta': {}},
            _lut_node('n5_lutlayer', 16, 8, rng),
        ],
        'edges': [
            {'src': 'x', 'dst': 'n1_lutlayer'},
            {'src': 'n1_lutlayer', 'dst': 'n2_lutlayer'},
            {'src': 'n1_lutlayer', 'dst': 'n3_lutlayer'},
            {'src': 'n2_lutlayer', 'dst': 'n4_concat'},  # dst_slot省略=0
            {'src': 'n3_lutlayer', 'dst': 'n4_concat', 'dst_slot': 1},
            {'src': 'n4_concat', 'dst': 'n5_lutlayer'},
            {'src': 'n5_lutlayer', 'dst': 'y'},
        ],
    }


def _add_doc(mode, n_branches=2, seed=13) -> dict:
    """x → fanoutで n1..nk(各4→4) → add(mode) → y のIR dict"""
    rng = np.random.default_rng(seed)
    nodes = [_lut_node(f'n{i + 1}_lutlayer', 4, 4, rng)
             for i in range(n_branches)]
    nodes.append({'id': 'na_add', 'type': 'add',
                  'params': {'width': 4, 'mode': mode},
                  'train_info': {}, 'meta': {}})
    edges = [{'src': 'x', 'dst': f'n{i + 1}_lutlayer'}
             for i in range(n_branches)]
    edges += [{'src': f'n{i + 1}_lutlayer', 'dst': 'na_add', 'dst_slot': i}
              for i in range(n_branches)]
    edges.append({'src': 'na_add', 'dst': 'y'})
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': f'add_{mode}'},
        'inputs': [_port('x', 4)],
        'outputs': [_port('y', 4)],
        'nodes': nodes,
        'edges': edges,
    }


def _binary(batch, width, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, width)).astype(np.int64)


def _hard_forward(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32))
    return np.round(y.numpy()).astype(np.int64)


class TestDiamond:
    def test_validate_and_topo(self):
        graph = ir_from_dict(_diamond_doc())
        assert validate(graph) == []
        order = [n.id for n in graph.topological_order()]
        assert len(order) == 5
        assert order.index('n1_lutlayer') < order.index('n2_lutlayer')
        assert order.index('n1_lutlayer') < order.index('n3_lutlayer')
        assert order.index('n2_lutlayer') < order.index('n4_concat')
        assert order.index('n3_lutlayer') < order.index('n4_concat')
        assert order.index('n4_concat') < order.index('n5_lutlayer')

    def test_build_torch_returns_graph_module(self):
        graph = ir_from_dict(_diamond_doc())
        model = build_torch(graph)
        assert isinstance(model, IRGraphModule)
        x = torch.randint(0, 2, (BATCH, 8)).float()
        y = model(x)
        assert y.shape == (BATCH, 8)

    def test_simulate_trace_has_all_nodes(self):
        graph = ir_from_dict(_diamond_doc())
        x = _binary(BATCH, 8)
        trace = simulate_trace(graph, x)
        for n in graph.nodes:
            assert n.id in trace, f'{n.id} がtraceにない'
        assert trace['n2_lutlayer'].shape == (BATCH, 6)
        assert trace['n4_concat'].shape == (BATCH, 16)

    def test_simulate_matches_hard_forward(self):
        graph = ir_from_dict(_diamond_doc())
        x = _binary(BATCH, 8)
        expected = simulate(graph, x)
        actual = _hard_forward(build_torch(graph), x)
        assert (expected == actual).all(), 'simulateとhard evalが不一致'

    def test_dst_slot_roundtrip(self):
        """dst_slot=0は省略、≠0のみキーを含む(表現の安定性)"""
        doc = _diamond_doc()
        graph = ir_from_dict(doc)
        edges = ir_to_dict(graph)['edges']
        assert edges == doc['edges']
        assert 'dst_slot' not in edges[3]
        assert edges[4]['dst_slot'] == 1


class TestConcatBitOrder:
    def test_slot0_is_head(self):
        """ビット順規約: slot 0の信号が出力ビット列の先頭(下位ビット側)"""
        graph = ir_from_dict(_diamond_doc())
        x = _binary(BATCH, 8)
        trace = simulate_trace(graph, x)
        cat = trace['n4_concat']
        assert (cat[:, :6] == trace['n2_lutlayer']).all(), \
            'slot 0が出力ビット列の先頭にない'
        assert (cat[:, 6:] == trace['n3_lutlayer']).all(), \
            'slot 1が出力ビット列の後続にない'

    def test_passthrough_columns(self):
        """pass-through LUTでconcat出力の列順を明示検証"""
        # na = x[0]のpass-through(tt=0xa)、nb = x[1]のpass-through(tt=0xc)
        doc = {
            'format': 'magicdelta-ir',
            'version': '0.2.0',
            'metadata': {'name': 'concat_order'},
            'inputs': [_port('x', 2)],
            'outputs': [_port('y', 2)],
            'nodes': [
                {'id': 'na', 'type': 'lut_layer',
                 'params': {'in_width': 2, 'out_width': 1, 'n_inputs': 2,
                            'indices': [[0], [1]], 'truth_tables': ['0xa']},
                 'train_info': {'parameterization': 'independent'}, 'meta': {}},
                {'id': 'nb', 'type': 'lut_layer',
                 'params': {'in_width': 2, 'out_width': 1, 'n_inputs': 2,
                            'indices': [[0], [1]], 'truth_tables': ['0xc']},
                 'train_info': {'parameterization': 'independent'}, 'meta': {}},
                {'id': 'nc', 'type': 'concat',
                 'params': {'in_widths': [1, 1]}, 'train_info': {}, 'meta': {}},
            ],
            'edges': [
                {'src': 'x', 'dst': 'na'},
                {'src': 'x', 'dst': 'nb'},
                {'src': 'na', 'dst': 'nc', 'dst_slot': 0},
                {'src': 'nb', 'dst': 'nc', 'dst_slot': 1},
                {'src': 'nc', 'dst': 'y'},
            ],
        }
        graph = ir_from_dict(doc)
        assert validate(graph) == []
        x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
        y = simulate(graph, x)
        assert (y[:, 0] == x[:, 0]).all(), 'slot 0(=x0)が列0に来ていない'
        assert (y[:, 1] == x[:, 1]).all(), 'slot 1(=x1)が列1に来ていない'
        # torch側も同一規約
        assert (_hard_forward(build_torch(graph), x) == y).all()


class TestAddSemantics:
    @pytest.mark.parametrize('mode,op', [
        ('or', lambda a, b: a | b),
        ('xor', lambda a, b: a ^ b),
        ('sum', lambda a, b: a + b),
    ])
    def test_two_input_matches_numpy(self, mode, op):
        graph = ir_from_dict(_add_doc(mode))
        x = _binary(BATCH, 4)
        trace = simulate_trace(graph, x)
        expected = op(trace['n1_lutlayer'], trace['n2_lutlayer'])
        assert (trace['na_add'] == expected).all(), f'add({mode})の意味論不一致'
        assert (simulate(graph, x) == expected).all()

    @pytest.mark.parametrize('mode', ['or', 'xor', 'sum'])
    def test_hard_forward_matches_simulate(self, mode):
        graph = ir_from_dict(_add_doc(mode))
        x = _binary(BATCH, 4)
        assert (_hard_forward(build_torch(graph), x) == simulate(graph, x)).all()

    def test_three_input_or(self):
        graph = ir_from_dict(_add_doc('or', n_branches=3))
        assert validate(graph) == []
        x = _binary(BATCH, 4)
        trace = simulate_trace(graph, x)
        expected = (trace['n1_lutlayer'] | trace['n2_lutlayer']
                    | trace['n3_lutlayer'])
        assert (trace['na_add'] == expected).all()
        assert (_hard_forward(build_torch(graph), x) == expected).all()

    def test_sum_rejected_for_hdl(self):
        graph = ir_from_dict(_add_doc('sum'))
        validate(graph)  # genericは通る
        with pytest.raises(IRValidationError, match='sum'):
            validate(graph, target='hdl')


class TestSerialFastPath:
    def test_serial_returns_sequential(self):
        """直列グラフは従来どおりnn.Sequentialを返す(fast-path維持)"""
        rng = np.random.default_rng(3)
        doc = {
            'format': 'magicdelta-ir',
            'version': '0.2.0',
            'metadata': {'name': 'serial'},
            'inputs': [_port('x', 8)],
            'outputs': [_port('y', 8)],
            'nodes': [
                _lut_node('n1_lutlayer', 8, 8, rng),
                _lut_node('n2_lutlayer', 8, 8, rng),
            ],
            'edges': [
                {'src': 'x', 'dst': 'n1_lutlayer'},
                {'src': 'n1_lutlayer', 'dst': 'n2_lutlayer'},
                {'src': 'n2_lutlayer', 'dst': 'y'},
            ],
        }
        graph = ir_from_dict(doc)
        model = build_torch(graph)
        assert isinstance(model, nn.Sequential)
        x = _binary(BATCH, 8)
        assert (_hard_forward(model, x) == simulate(graph, x)).all()


class TestValidateErrors:
    def test_slot_missing(self):
        doc = _diamond_doc()
        # slot1のエッジをslot2へ(in_widthsは2要素なので範囲外)
        doc['edges'][4]['dst_slot'] = 2
        with pytest.raises(IRValidationError, match='範囲外'):
            validate(ir_from_dict(doc))

    def test_slot_gap_in_concat(self):
        doc = _diamond_doc()
        doc['nodes'][3]['params']['in_widths'] = [6, 10, 4]
        doc['edges'][4]['dst_slot'] = 2  # slot1が欠番
        with pytest.raises(IRValidationError, match='欠番'):
            validate(ir_from_dict(doc))

    def test_slot_duplicate(self):
        doc = _diamond_doc()
        doc['edges'][4]['dst_slot'] = 0  # slot0に2本
        with pytest.raises(IRValidationError, match='重複'):
            validate(ir_from_dict(doc))

    def test_width_mismatch_concat(self):
        doc = _diamond_doc()
        doc['nodes'][3]['params']['in_widths'] = [6, 8]  # n3は10bit出力
        with pytest.raises(IRValidationError, match='幅不整合'):
            validate(ir_from_dict(doc))

    def test_width_mismatch_add(self):
        doc = _add_doc('or')
        doc['nodes'][2]['params']['width'] = 5  # 各枝は4bit出力
        with pytest.raises(IRValidationError, match='幅不整合'):
            validate(ir_from_dict(doc))

    def test_cycle_detected(self):
        """slot整合は正しいが循環しているグラフ"""
        rng = np.random.default_rng(5)
        doc = {
            'format': 'magicdelta-ir',
            'version': '0.2.0',
            'metadata': {'name': 'cyclic'},
            'inputs': [_port('x', 4)],
            'outputs': [_port('y', 4)],
            'nodes': [
                _lut_node('n1_lutlayer', 4, 4, rng),
                {'id': 'n2_add', 'type': 'add',
                 'params': {'width': 4, 'mode': 'or'},
                 'train_info': {}, 'meta': {}},
                _lut_node('n3_lutlayer', 4, 4, rng),
            ],
            'edges': [
                {'src': 'x', 'dst': 'n1_lutlayer'},
                {'src': 'n1_lutlayer', 'dst': 'n2_add', 'dst_slot': 0},
                {'src': 'n3_lutlayer', 'dst': 'n2_add', 'dst_slot': 1},
                {'src': 'n2_add', 'dst': 'n3_lutlayer'},  # n2→n3→n2 の循環
                {'src': 'n2_add', 'dst': 'y'},
            ],
        }
        with pytest.raises(IRValidationError, match='循環'):
            validate(ir_from_dict(doc))

    def test_add_arity_too_few(self):
        doc = _add_doc('or')
        doc['edges'] = [e for e in doc['edges']
                        if not (e['dst'] == 'na_add'
                                and e.get('dst_slot', 0) == 1)]
        with pytest.raises(IRValidationError, match='2本以上'):
            validate(ir_from_dict(doc))

    def test_single_input_node_rejects_fanin(self):
        doc = _diamond_doc()
        doc['edges'].append({'src': 'n2_lutlayer', 'dst': 'n5_lutlayer',
                             'dst_slot': 1})
        with pytest.raises(IRValidationError, match='入力エッジは1本のみ'):
            validate(ir_from_dict(doc))

    def test_unknown_add_mode(self):
        doc = _add_doc('or')
        doc['nodes'][2]['params']['mode'] = 'and'
        with pytest.raises(IRValidationError, match='未知のaddモード'):
            validate(ir_from_dict(doc))


class TestCodegenDag:
    def _exec_build_model(self, code: str):
        ns = {}
        exec(compile(code, '<codegen>', 'exec'), ns)
        model = ns['build_model']()
        model.eval()
        return model

    def test_diamond_codegen_matches_build(self):
        from tools.editor.backend.codegen import generate_code
        graph = ir_from_dict(_diamond_doc())
        code = generate_code(graph)
        assert 'GraphModel' in code
        assert 'torch.cat' in code
        model = self._exec_build_model(code)
        x = _binary(BATCH, 8)
        assert (_hard_forward(model, x) == simulate(graph, x)).all(), \
            'codegen生成モデルとsimulateが不一致'

    @pytest.mark.parametrize('mode', ['or', 'xor', 'sum'])
    def test_add_codegen_matches_simulate(self, mode):
        from tools.editor.backend.codegen import generate_code
        graph = ir_from_dict(_add_doc(mode, n_branches=3))
        code = generate_code(graph)
        model = self._exec_build_model(code)
        x = _binary(BATCH, 4)
        assert (_hard_forward(model, x) == simulate(graph, x)).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
