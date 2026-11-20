"""E2: パイプラインコア(FF毎層)の生成とストリーミング等価検証。"""

from conftest import build_groupsum_model, requires_iverilog

from griffith_logic.ir.serialize import ir_from_dict

from mdl_fpga.check import run_check_pipelined
from mdl_fpga.hdlgen.core import generate_core
from mdl_fpga.ir_io import load_graph


def test_comb_output_has_no_clk(parity4_path):
    """既定(comb)経路は clk レスのまま(byte 同一性は HEAD 比較で確認済み)。"""
    gen = generate_core(load_graph(parity4_path))
    assert not gen.pipelined and gen.latency == 0
    assert 'clk' not in gen.verilog
    assert 'reg ' not in gen.verilog


def test_pipelined_structure(parity4_path):
    gen = generate_core(load_graph(parity4_path), pipelined=True)
    assert gen.pipelined and gen.latency == 2   # xor_l1 → xor_l2
    v = gen.verilog
    assert 'input  wire clk,' in v
    # comb 部は _c、FF が本名を持つ(name_map の wire 名は不変)
    assert 'wire [1:0] w_xor_l1_c;' in v
    assert "reg [1:0] w_xor_l1 = {2{1'b0}};" in v
    assert 'always @(posedge clk) w_xor_l1 <= w_xor_l1_c;' in v
    assert gen.name_map['xor_l1']['wire'] == 'w_xor_l1'
    assert gen.name_map['xor_l1']['stage'] == 1
    assert gen.name_map['xor_l2']['stage'] == 2
    assert 'assign y = w_xor_l2;' in v   # 最終段レジスタ出力


def test_pipelined_groupsum_latency():
    gen = generate_core(ir_from_dict(build_groupsum_model()), pipelined=True)
    assert gen.latency == 2   # gates(1) → gs(2)
    assert gen.name_map['gs']['stage'] == 2
    assert 'reg [3:0] w_gs' in gen.verilog


def _skew_model() -> dict:
    """スキュー均衡が必要なグラフ: x→(深さ1)a、x→b→c(深さ2)、concat(a,c)。"""
    lut = {'in_width': 4, 'out_width': 2, 'n_inputs': 2,
           'indices': [[0, 2], [1, 3]], 'truth_tables': ['0x6', '0x6']}
    lut2 = {'in_width': 2, 'out_width': 2, 'n_inputs': 2,
            'indices': [[0, 1], [1, 0]], 'truth_tables': ['0x8', '0xe']}
    return {
        'format': 'magicdelta-ir', 'version': '0.2.0',
        'metadata': {'name': 'skew'},
        'inputs': [{'id': 'x', 'width': 4}],
        'outputs': [{'id': 'y', 'width': 4}],
        'nodes': [
            {'id': 'a', 'type': 'lut_layer', 'params': dict(lut)},
            {'id': 'b', 'type': 'lut_layer', 'params': dict(lut)},
            {'id': 'c', 'type': 'lut_layer', 'params': dict(lut2)},
            {'id': 'cat', 'type': 'concat', 'params': {'in_widths': [2, 2]}},
        ],
        'edges': [
            {'src': 'x', 'dst': 'a'},
            {'src': 'x', 'dst': 'b'},
            {'src': 'b', 'dst': 'c'},
            {'src': 'a', 'dst': 'cat', 'dst_slot': 0},
            {'src': 'c', 'dst': 'cat', 'dst_slot': 1},
            {'src': 'cat', 'dst': 'y'},
        ],
    }


def test_skew_balancing_delay_regs(tmp_path):
    import json
    p = tmp_path / 'skew.ir.json'
    p.write_text(json.dumps(_skew_model()), encoding='utf-8')
    gen = generate_core(load_graph(p), pipelined=True)
    # a(段1)と c(段2)を concat(段3)で合流 → a に +1 の遅延が入る
    assert gen.latency == 3
    assert 'w_a_d1' in gen.verilog
    assert 'スキュー均衡ディレイ' in gen.verilog


@requires_iverilog
def test_streaming_equivalence_parity(parity4_path, tmp_path):
    res = run_check_pipelined(parity4_path, workdir=tmp_path / 'run',
                              n_vectors=16)
    assert res.passed, res.render() + '\n' + res.stdout
    assert res.edges['pipe_vs_golden(a)'] == 'PASS'
    assert res.edges['pipe_vs_comb(x)'] == 'PASS'


@requires_iverilog
def test_streaming_equivalence_groupsum(groupsum_path, tmp_path):
    res = run_check_pipelined(groupsum_path, workdir=tmp_path / 'run',
                              n_vectors=16)
    assert res.passed, res.render() + '\n' + res.stdout


@requires_iverilog
def test_streaming_equivalence_skew(tmp_path):
    import json
    p = tmp_path / 'skew.ir.json'
    p.write_text(json.dumps(_skew_model()), encoding='utf-8')
    res = run_check_pipelined(p, workdir=tmp_path / 'run', n_vectors=16)
    assert res.passed, res.render() + '\n' + res.stdout
