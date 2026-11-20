"""IR constantノードのテスト。

x(4) → concat([x, c0]) → lut(8→4) → y のグラフで:
    - validate通過・arity 0(入力エッジなし)の扱い・異常系
    - 3段等価性: PyTorch hard eval ⇔ IR simulate ⇔ Verilog
      (文字列妥当性は無条件、実機シミュレーションは@hdl/@hdl_vhdl)
    - serializeラウンドトリップ
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.export.verify import (
    ghdl_available,
    iverilog_available,
    run_ghdl_equivalence,
    run_iverilog_equivalence,
)
from griffith_logic.export.verilog import to_verilog
from griffith_logic.export.vhdl import to_vhdl
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

BATCH = 32
BITS = [1, 0, 1, 1]


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


def _lut_node(node_id, in_width, out_width, seed=3, n_inputs=2):
    rng = np.random.default_rng(seed)
    indices = [
        [int(rng.integers(0, in_width)) for _ in range(out_width)]
        for _ in range(n_inputs)
    ]
    tts = [hex(int(rng.integers(0, 1 << (1 << n_inputs))))
           for _ in range(out_width)]
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


def _constant_doc(bits=None) -> dict:
    """x(4) + constant(4) → concat → lut(8→4) → y"""
    bits = BITS if bits is None else bits
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': 'constant_demo'},
        'inputs': [_port('x', 4)],
        'outputs': [_port('y', 4)],
        'nodes': [
            {'id': 'c0', 'type': 'constant',
             'params': {'width': len(bits), 'bits': list(bits)},
             'train_info': {}, 'meta': {}},
            {'id': 'n1_concat', 'type': 'concat',
             'params': {'in_widths': [4, len(bits)]},
             'train_info': {}, 'meta': {}},
            _lut_node('n2_lutlayer', 4 + len(bits), 4),
        ],
        'edges': [
            {'src': 'x', 'dst': 'n1_concat'},
            {'src': 'c0', 'dst': 'n1_concat', 'dst_slot': 1},
            {'src': 'n1_concat', 'dst': 'n2_lutlayer'},
            {'src': 'n2_lutlayer', 'dst': 'y'},
        ],
    }


def _binary(batch, width, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, width)).astype(np.int64)


def _hard_forward(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32))
    return np.round(y.numpy()).astype(np.int64)


class TestConstantValidate:
    def test_valid_graph(self):
        graph = ir_from_dict(_constant_doc())
        assert validate(graph) == []
        assert validate(graph, target='hdl') == []

    def test_topological_order_includes_constant(self):
        graph = ir_from_dict(_constant_doc())
        order = [n.id for n in graph.topological_order()]
        assert 'c0' in order
        assert order.index('c0') < order.index('n1_concat')

    def test_bits_length_mismatch(self):
        doc = _constant_doc()
        doc['nodes'][0]['params']['bits'] = [1, 0]  # width=4に対し2要素
        with pytest.raises(IRValidationError, match='width'):
            validate(ir_from_dict(doc))

    def test_bits_value_out_of_range(self):
        doc = _constant_doc()
        doc['nodes'][0]['params']['bits'] = [1, 0, 2, 1]
        with pytest.raises(IRValidationError, match='0/1'):
            validate(ir_from_dict(doc))

    def test_constant_rejects_input_edge(self):
        doc = _constant_doc()
        doc['edges'].append({'src': 'x', 'dst': 'c0'})
        with pytest.raises(IRValidationError, match='入力エッジを持てない'):
            validate(ir_from_dict(doc))

    def test_width_mismatch_into_concat(self):
        doc = _constant_doc()
        doc['nodes'][0]['params']['width'] = 3
        doc['nodes'][0]['params']['bits'] = [1, 0, 1]
        with pytest.raises(IRValidationError, match='幅不整合'):
            validate(ir_from_dict(doc))


class TestConstantSimulate:
    def test_trace_constant_rows(self):
        graph = ir_from_dict(_constant_doc())
        x = _binary(BATCH, 4)
        trace = simulate_trace(graph, x)
        assert (trace['c0'] == np.array(BITS)).all()
        assert trace['c0'].shape == (BATCH, 4)
        # concat: slot 0=x、slot 1=定数
        assert (trace['n1_concat'][:, :4] == x).all()
        assert (trace['n1_concat'][:, 4:] == np.array(BITS)).all()

    def test_hard_forward_matches_simulate(self):
        """3段等価性の第1段: PyTorch hard eval ⇔ simulate"""
        graph = ir_from_dict(_constant_doc())
        model = build_torch(graph)
        assert isinstance(model, IRGraphModule)
        x = _binary(BATCH, 4)
        assert (_hard_forward(model, x) == simulate(graph, x)).all()

    def test_serialize_roundtrip(self):
        doc = _constant_doc()
        graph = ir_from_dict(doc)
        graph2 = ir_from_dict(ir_to_dict(graph))
        x = _binary(BATCH, 4)
        assert (simulate(graph, x) == simulate(graph2, x)).all()


class TestConstantHdlStrings:
    """3段等価性の第2段(文字列妥当性、シミュレータ不要)"""

    def test_verilog_hardwired_literal(self):
        graph = ir_from_dict(_constant_doc())
        src = to_verilog(graph)
        # bits=[1,0,1,1] → リテラルは左がMSBなので 1101
        assert 'wire [3:0] w_c0;' in src
        assert "assign w_c0 = 4'b1101;" in src
        assert 'assign w_n1_concat = {w_c0, x};' in src

    def test_vhdl_hardwired_literal(self):
        graph = ir_from_dict(_constant_doc())
        src = to_vhdl(graph)
        assert 'signal w_c0 : std_logic_vector(3 downto 0);' in src
        assert 'w_c0 <= "1101";' in src


@pytest.mark.hdl
@pytest.mark.skipif(not iverilog_available(), reason='iverilog未導入')
class TestConstantVerilogEquivalence:
    def test_three_way_equivalence(self, tmp_path, seed):
        """3段等価性: PyTorch hard eval ⇔ simulate ⇔ iverilog"""
        graph = ir_from_dict(_constant_doc())
        x = _binary(256, 4, seed=9)
        assert (_hard_forward(build_torch(graph), x)
                == simulate(graph, x)).all()
        result = run_iverilog_equivalence(graph, x, str(tmp_path))
        assert result['passed'], (
            f'constant: Verilogとgolden modelが不一致\n{result["stdout"]}'
        )


@pytest.mark.hdl_vhdl
@pytest.mark.skipif(not ghdl_available(), reason='ghdl未導入')
class TestConstantVhdlEquivalence:
    def test_three_way_equivalence(self, tmp_path, seed):
        graph = ir_from_dict(_constant_doc())
        x = _binary(256, 4, seed=9)
        result = run_ghdl_equivalence(graph, x, str(tmp_path))
        assert result['passed'], (
            f'constant: VHDLとgolden modelが不一致\n{result["stdout"]}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
