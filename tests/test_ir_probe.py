"""推論プローブのテスト。

二値入力+確定的重み(真理値表から構築)で、probe_soft(eval mode)の
ビット化(>0.5)が probe(=simulate_traceビット厳密)と全ノードで一致する
ことを、直列(nn.Sequential)とダイヤモンドDAG(IRGraphModule)の両方で検証する。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import (
    IRGraphModule,
    build_torch,
    ir_from_dict,
    probe,
    probe_soft,
    validate,
)

BATCH = 32


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


def _serial_doc(seed=11) -> dict:
    """x → lut → lut → group_sum → y の直列IR(idはn{i}規約と非一致)"""
    rng = np.random.default_rng(seed)
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': 'serial_probe'},
        'inputs': [_port('x', 8)],
        'outputs': [_port('y', 4)],
        'nodes': [
            _lut_node('first_layer', 8, 12, rng),
            _lut_node('second_layer', 12, 8, rng),
            {'id': 'scores', 'type': 'group_sum',
             'params': {'in_width': 8, 'k': 4, 'tau': 1.0},
             'train_info': {}, 'meta': {}},
        ],
        'edges': [
            {'src': 'x', 'dst': 'first_layer'},
            {'src': 'first_layer', 'dst': 'second_layer'},
            {'src': 'second_layer', 'dst': 'scores'},
            {'src': 'scores', 'dst': 'y'},
        ],
    }


def _diamond_doc(seed=13) -> dict:
    """x → n1 → fanoutで n2/n3 → concat → n5 → y のダイヤモンドIR"""
    rng = np.random.default_rng(seed)
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': 'diamond_probe'},
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
            {'src': 'n2_lutlayer', 'dst': 'n4_concat'},
            {'src': 'n3_lutlayer', 'dst': 'n4_concat', 'dst_slot': 1},
            {'src': 'n4_concat', 'dst': 'n5_lutlayer'},
            {'src': 'n5_lutlayer', 'dst': 'y'},
        ],
    }


def _bits(width, batch=BATCH, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, width))


def _assert_soft_bits_match_hard(soft_trace, hard_trace, graph):
    """二値ノードはビット化一致、group_sumは整数スコア一致を検証する"""
    gs_ids = {n.id for n in graph.nodes if n.type == 'group_sum'}
    assert set(soft_trace) <= set(hard_trace)
    for nid, t in soft_trace.items():
        v = t.detach().numpy()
        if nid in gs_ids:
            # tau=1.0なら整数スコアがそのまま一致する
            assert np.array_equal(v.astype(np.int64), hard_trace[nid]), nid
        else:
            assert np.array_equal(
                (v > 0.5).astype(np.int64), hard_trace[nid],
            ), f'{nid}: soft⇔hard不一致'


class TestProbeSerial:
    def test_positional_fallback_all_nodes_match(self, seed):
        """build_torch直組み(id非一致)→位置対応フォールバックで全ノード一致"""
        graph = ir_from_dict(_serial_doc())
        validate(graph)
        model = build_torch(graph)
        assert isinstance(model, nn.Sequential)
        model.eval()

        x = _bits(8)
        hard = probe(graph, x)
        soft = probe_soft(model, graph, torch.tensor(x, dtype=torch.float32))
        assert set(soft) == {'x', 'first_layer', 'second_layer', 'scores'}
        _assert_soft_bits_match_hard(soft, hard, graph)

    def test_numbering_convention_with_flatten(self, seed):
        """エディタ構成(Flatten前置Sequential)のn{i}規約で全ノード一致"""
        torch.manual_seed(3)
        model = nn.Sequential(
            nn.Flatten(),
            LutLayer(8, 16, n_inputs=2, parameterization='independent'),
            LutLayer(16, 8, n_inputs=2, parameterization='independent'),
            GroupSum(k=4, tau=1.0),
        )
        model.eval()
        graph = export_ir(model, input_width=8)
        assert [n.id for n in graph.nodes] == \
            ['n1_lutlayer', 'n2_lutlayer', 'n3_groupsum']

        x = _bits(8)
        hard = probe(graph, x)
        soft = probe_soft(model, graph, torch.tensor(x, dtype=torch.float32))
        assert set(soft) == {'x', 'n1_lutlayer', 'n2_lutlayer', 'n3_groupsum'}
        _assert_soft_bits_match_hard(soft, hard, graph)

    def test_mismatched_module_count_raises(self, seed):
        """モジュール数とノード数が食い違えば日本語エラー"""
        graph = ir_from_dict(_serial_doc())
        model = nn.Sequential(
            LutLayer(8, 12, n_inputs=2, parameterization='independent'),
        )
        model.eval()
        with pytest.raises(ValueError, match='対応付けに失敗'):
            probe_soft(model, graph, torch.zeros(2, 8))

    def test_unsupported_model_type_raises(self, seed):
        graph = ir_from_dict(_serial_doc())
        with pytest.raises(TypeError, match='IRGraphModule'):
            probe_soft(LutLayer(8, 4, n_inputs=2), graph, torch.zeros(2, 8))


class TestProbeDiamond:
    def test_dag_all_nodes_match(self, seed):
        """ダイヤモンドDAG: forward_trace経由で全ノード(concat含む)一致"""
        graph = ir_from_dict(_diamond_doc())
        validate(graph)
        model = build_torch(graph)
        assert isinstance(model, IRGraphModule)
        model.eval()

        x = _bits(8)
        hard = probe(graph, x)
        soft = probe_soft(model, graph, torch.tensor(x, dtype=torch.float32))
        assert {'x', 'n1_lutlayer', 'n2_lutlayer', 'n3_lutlayer',
                'n4_concat', 'n5_lutlayer'} <= set(soft)
        _assert_soft_bits_match_hard(soft, hard, graph)

    def test_probe_includes_input_port(self, seed):
        graph = ir_from_dict(_diamond_doc())
        x = _bits(8, batch=4)
        hard = probe(graph, x)
        assert np.array_equal(hard['x'], x)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
