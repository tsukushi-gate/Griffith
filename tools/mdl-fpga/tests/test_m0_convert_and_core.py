"""M0: convert(IR 検証)とコア生成器の単体テスト(シミュレータ不要)。"""

import json

import numpy as np
from conftest import build_groupsum_model, build_parity4

from mdl_fpga.hdlgen.core import generate_core
from mdl_fpga.ir_io import convert, load_graph
from mdl_fpga.vectors import gen_inputs, pack_bits, unpack_bits


def test_convert_ok(parity4_path):
    res = convert(parity4_path)
    assert res.ok, res.render()
    assert any('lowering 後の LUT ゲート総数: 3' in t for _, t in res.messages)


def test_convert_missing_file(tmp_path):
    res = convert(tmp_path / 'nope.ir.json')
    assert not res.ok
    assert res.messages[0][0] == '環境'


def test_convert_rejects_blackbox(tmp_path):
    doc = build_parity4()
    doc['nodes'].append({
        'id': 'attn', 'type': 'blackbox',
        'params': {'torch_class': 'DLGNBooleanAttention',
                   'in_width': 1, 'out_width': 1},
    })
    doc['edges'] = [
        {'src': 'x', 'dst': 'xor_l1'},
        {'src': 'xor_l1', 'dst': 'xor_l2'},
        {'src': 'xor_l2', 'dst': 'attn'},
        {'src': 'attn', 'dst': 'y'},
    ]
    p = tmp_path / 'bb.ir.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    res = convert(p)
    assert not res.ok
    assert any('blackbox' in t for _, t in res.messages)
    assert any('hls4ml' in t for _, t in res.messages)


def test_core_gen_structure(parity4_path):
    gen = generate_core(load_graph(parity4_path))
    assert gen.in_width == 4 and gen.out_width == 1
    assert gen.module_name == 'mdl_core'
    assert 'module mdl_core' in gen.verilog
    # 安定命名: node id 由来の wire 名
    assert gen.name_map['xor_l1']['wire'] == 'w_xor_l1'
    assert gen.name_map['xor_l2']['wire'] == 'w_xor_l2'
    assert "assign w_xor_l2[0] = 4'h6 >> {w_xor_l1[1], w_xor_l1[0]};" in gen.verilog


def test_core_gen_observe_keep(parity4_path):
    gen = generate_core(load_graph(parity4_path), observe=('xor_l1',))
    assert '(* keep *) wire [1:0] w_xor_l1;' in gen.verilog
    assert gen.name_map['xor_l1']['observed'] is True
    assert gen.name_map['xor_l2']['observed'] is False


def test_core_gen_groupsum_widths(groupsum_path):
    gen = generate_core(load_graph(groupsum_path))
    # group=2 → score_bits=2、k=2 → 出力4bit
    assert gen.score_bits == 2
    assert gen.out_width == 4


def test_vectors_exhaustive_and_roundtrip():
    x = gen_inputs(4, n_vectors=256, seed=0)
    assert x.shape == (16, 4)  # 2^4 全数
    packed = pack_bits(x)
    assert packed[0] == 0 and packed[-1] == 15
    row = unpack_bits(packed[5], 4)
    assert np.array_equal(row, x[5])
