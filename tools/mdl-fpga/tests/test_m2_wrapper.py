"""M2: UART ラッパ(フルチェーン)の bit 一致検証。

top(実機書込みと同一構造、CLKS_PER_BIT 短縮)+ frame_rx/tx + uart_rx/tx を
iverilog で回し、UART フレーム経由の入出力が golden と一致することを確認する。
"""

from conftest import requires_iverilog

from mdl_fpga.check import run_wrapper_check
from mdl_fpga.hdlgen.wrapper import generate_wrapper


def test_wrapper_gen_widths():
    w = generate_wrapper(4, 1, clks_per_bit=16)
    assert w.req_payload_bytes == 1 and w.rsp_payload_bytes == 1
    assert 'module top (' in w.top_verilog
    assert '.CLKS_PER_BIT(16)' in w.top_verilog

    w2 = generate_wrapper(784, 20, top_name='mnist_top')
    assert w2.req_payload_bytes == 98 and w2.rsp_payload_bytes == 3
    assert w2.clks_per_bit == 434  # 50MHz / 115200
    assert 'module mnist_top (' in w2.top_verilog


@requires_iverilog
def test_wrapper_e2e_parity(parity4_path, tmp_path):
    res = run_wrapper_check(parity4_path, workdir=tmp_path / 'run', n_vectors=16)
    assert res.passed, res.render() + '\n' + res.stdout


@requires_iverilog
def test_wrapper_e2e_groupsum(groupsum_path, tmp_path):
    res = run_wrapper_check(groupsum_path, workdir=tmp_path / 'run', n_vectors=16)
    assert res.passed, res.render() + '\n' + res.stdout
