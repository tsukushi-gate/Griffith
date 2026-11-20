"""M6: DE10-Lite ボード定義(Tier2 実機未検証)+ synth MAX 10 対応の検証。

ピン値のアサートは転記元(litex-boards + 実プロジェクト qsf、boards/NOTICE)
と機械照合済みの値。実機テストはなし(unverified のため requires_hw 対象外)。
"""

import json

from conftest import build_parity4
from mdl_fpga.board_db import load_board, resolve_header_io
from mdl_fpga.hdlgen.ports import TopPorts
from mdl_fpga.io_link import resolve_uart_link
from mdl_fpga.synth import (
    _STA_CORNER_DEFAULT, _STA_CORNERS, _STA_TCL, parse_fit_summary,
    parse_fmax, write_project_files)

# Quartus fit.summary の LE 系 family 行(MAX 10)。行形式は既存 Cyclone V
# サンプルと同じ fit.summary 形式
FIT_SAMPLE_MAX10 = """
Fitter Status : Successful - ...
Total logic elements : 1,234 / 49,760 ( 2 % )
Total registers : 456
Total pins : 23 / 360 ( 6 % )
Total block memory bits : 0 / 1,677,312 ( 0 % )
"""

STA_SAMPLE_1200 = """
; Slow 1200mV 85C Model Fmax Summary ;
+------------+-----------------+------------+------+
; 88.11 MHz ; 88.11 MHz ; clk50 ; ;
"""


def test_de10lite_board_loads_and_key_facts():
    b = load_board('de10-lite')
    assert b['device'] == '10M50DAF484C7G'
    assert b['family'] == 'MAX 10'
    assert b['tier'] == 2 and b['unverified'] is True
    assert b['resources']['unit'] == 'le'
    assert b['resources']['le'] == 49760
    assert b['limits']['calibrated'] is False
    assert b['detection']['jtag_idcodes'] == ['0x031050dd']
    # 転記元と機械照合済みの代表ピン
    assert b['clock'] == {'port': 'MAX10_CLK1_50', 'pin': 'PIN_P11',
                          'hz': 50000000}
    assert b['led']['pins'][0] == 'PIN_A8'
    assert b['switch']['pins'][9] == 'PIN_F15'
    assert b['button']['pins'] == ['PIN_B8', 'PIN_A7']
    assert b['hex']['digits'][0]['pins'][0] == 'PIN_C14'
    assert b['hex']['digits'][5]['pins'][6] == 'PIN_N20'
    # GPIO ヘッダ(JP1): 論理0=V10(物理1)、非IO位置はスキップされている
    assert resolve_header_io(b, 'GPIO', 0) == {'pin': 'PIN_V10', 'physical': 1}
    assert resolve_header_io(b, 'GPIO', 35)['pin'] == 'PIN_AA2'
    assert resolve_header_io(b, 'GPIO', 10)['physical'] == 13  # 11/12 は電源系


def test_default_uart_link_uses_first_header():
    """既定リンクはボード最初のヘッダの index 0/1 に読み替わる。"""
    b = load_board('de10-lite')
    r = resolve_uart_link(b, None)
    assert (r.rx_pin, r.tx_pin) == ('PIN_V10', 'PIN_W10')
    assert (r.rx_physical, r.tx_physical) == (1, 2)
    # de0-cv は従来どおり(GPIO_0[0]/[1])
    d = resolve_uart_link(load_board('de0-cv'), None)
    assert (d.rx_pin, d.tx_pin) == ('PIN_N16', 'PIN_B16')


def test_qsf_for_max10(tmp_path):
    b = load_board('de10-lite')
    write_project_files(tmp_path, name='x', board=b, sources=['x.v'],
                        groups=('clock', 'led', 'switch', 'button', 'hex'))
    qsf = (tmp_path / 'x.qsf').read_text(encoding='utf-8')
    assert 'set_global_assignment -name FAMILY "MAX 10"' in qsf
    assert 'DEVICE 10M50DAF484C7G' in qsf
    assert 'PIN_P11 -to MAX10_CLK1_50' in qsf
    assert 'PIN_C14 -to HEX0[0]' in qsf
    assert 'PIN_B8 -to KEY[0]' in qsf
    sdc = (tmp_path / 'x.sdc').read_text(encoding='utf-8')
    assert 'create_clock -name clk50 -period 20.000 ' \
           '[get_ports MAX10_CLK1_50]' in sdc


def test_topports_from_de10lite(tmp_path):
    b = load_board('de10-lite')
    p = TopPorts.from_board(b, resolve_uart_link(b, None))
    assert p.clock == 'MAX10_CLK1_50'
    assert p.button_width == 2
    assert p.led_width == 10 and p.switch_width == 10
    # system 生成が通る(hex 6桁 / led 10 の前提を満たす)
    from conftest import build_groupsum_model
    from mdl_fpga.hdlgen.system import generate_system
    from mdl_fpga.ir_io import load_graph
    ir = tmp_path / 'gs.ir.json'
    ir.write_text(json.dumps(build_groupsum_model()), encoding='utf-8')
    gen = generate_system(load_graph(ir), n_images=6, ports=p)
    assert 'MAX10_CLK1_50' in gen.top_verilog
    assert 'input  wire [1:0] KEY,' in gen.top_verilog


def test_parse_fit_summary_le():
    s = parse_fit_summary(FIT_SAMPLE_MAX10)
    assert s['le'] == 1234 and s['le_avail'] == 49760
    assert 'alm' not in s


def test_parse_fmax_generalized_corners():
    s = parse_fmax(STA_SAMPLE_1200)
    assert s['fmax_clk50_mhz'] == 88.11


def test_sta_corner_selection():
    assert '-voltage 1100' in _STA_CORNERS['Cyclone V']
    tcl_cv = _STA_TCL.format(name='x', corner=_STA_CORNERS['Cyclone V'])
    assert 'create_timing_netlist -model slow -temperature 85 -voltage 1100' \
        in tcl_cv
    # 未収載 family はコーナー明示なし(ツール既定 slow に任せる=値の創作をしない)
    tcl_other = _STA_TCL.format(name='x', corner=_STA_CORNER_DEFAULT)
    assert 'create_timing_netlist -model slow\n' in tcl_other


def test_build_generates_max10_project_without_quartus(tmp_path):
    """Quartus 無し Toolchain で de10-lite ビルド → Tier2 警告+生成物のみで停止。"""
    from mdl_fpga.build import run_build
    from mdl_fpga.toolchain import Toolchain
    ir = tmp_path / 'p4.ir.json'
    ir.write_text(json.dumps(build_parity4()), encoding='utf-8')
    res = run_build(ir, workdir=tmp_path / 'wd', board='de10-lite',
                    tc=Toolchain())          # 空 Toolchain=presim skip+合成せず
    assert res.ok is False
    assert any('Tier2' in m for m in res.messages)
    assert any('未較正' in m for m in res.messages)
    qsf = (tmp_path / 'wd' / 'quartus' / 'top.qsf').read_text(encoding='utf-8')
    assert 'FAMILY "MAX 10"' in qsf
    assert 'PIN_V10 -to UART_RXD' in qsf     # 既定リンク=GPIO[0]/[1]
    assert 'PIN_W10 -to UART_TXD' in qsf
