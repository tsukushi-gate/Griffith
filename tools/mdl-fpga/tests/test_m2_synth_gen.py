"""M2: 合成プロジェクト生成(qsf/sdc)とレポートパーサの単体テスト(Quartus 不要)。

board JSON v2(F2)で uart は board から除去され、write_project_files は
io.link から解決した uart 辞書を引数で受ける(更新は進捗記録記録済み)。
"""

from mdl_fpga.board_db import list_boards, load_board, resolve_header_io
from mdl_fpga.io_link import resolve_uart_link
from mdl_fpga.synth import parse_fit_summary, parse_fmax, write_project_files

DE0CV_UART = resolve_uart_link(load_board('de0-cv'), None).to_uart_dict()

FIT_SAMPLE = """
Fitter Status : Successful - ...
Logic utilization (in ALMs) : 6,964 / 18,480 ( 38 % )
Total registers : 6096
Total pins : 23 / 224 ( 10 % )
Total block memory bits : 0 / 3,153,920 ( 0 % )
Total RAM Blocks : 0 / 308 ( 0 % )
Total DSP Blocks : 0 / 66 ( 0 % )
"""

STA_SAMPLE = """
; Slow 1100mV 85C Model Fmax Summary ;
+------------+-----------------+------------+------+
; 67.59 MHz ; 67.59 MHz ; clk50 ; ;
"""


def test_board_db():
    assert 'de0-cv' in list_boards()
    b = load_board('de0-cv')
    assert b['device'] == '5CEBA4F23C7'
    assert b['clock']['pin'] == 'PIN_M9'
    # uart は v2 で board から除去(headers の確認済みピンから io.link が解決)
    assert 'uart' not in b
    assert resolve_header_io(b, 'GPIO_0', 0)['pin'] == 'PIN_N16'
    assert b['detection']['jtag_idcodes'] == ['0x02b050dd']   # 実機採取値
    assert len(b['led']['pins']) == 10 and len(b['switch']['pins']) == 10


def test_write_project_files_parameterized_name(tmp_path):
    """demo の落とし穴(プロジェクト名 top 固定)が直っていることの回帰。"""
    b = load_board('de0-cv')
    write_project_files(tmp_path, name='parity_demo', board=b,
                        sources=['parity_demo.v', 'mdl_core.v'],
                        uart=DE0CV_UART)
    qsf = (tmp_path / 'parity_demo.qsf').read_text(encoding='utf-8')
    assert 'TOP_LEVEL_ENTITY parity_demo' in qsf
    assert 'DEVICE 5CEBA4F23C7' in qsf
    assert 'PIN_M9 -to CLOCK_50' in qsf
    assert 'WEAK_PULL_UP_RESISTOR ON -to UART_RXD' in qsf
    assert 'VERILOG_FILE mdl_core.v' in qsf
    assert 'AUTO_SHIFT_REGISTER_RECOGNITION OFF' in qsf
    sdc = (tmp_path / 'parity_demo.sdc').read_text(encoding='utf-8')
    assert 'create_clock -name clk50 -period 20.000' in sdc
    assert 'set_false_path -from [get_ports UART_RXD]' in sdc
    qpf = (tmp_path / 'parity_demo.qpf').read_text(encoding='utf-8')
    assert 'PROJECT_REVISION = "parity_demo"' in qpf


def test_parse_fit_summary():
    s = parse_fit_summary(FIT_SAMPLE)
    assert s['alm'] == 6964 and s['alm_avail'] == 18480
    assert s['alm_pct'] == 38.0
    assert s['registers'] == 6096
    assert s['ram_blocks'] == 0 and s['ram_blocks_avail'] == 308


def test_parse_fmax():
    s = parse_fmax(STA_SAMPLE)
    assert s['fmax_clk50_mhz'] == 67.59
    assert s['fmax_restricted_clk50_mhz'] == 67.59


def test_groups_hex_button(tmp_path):
    """システム用グループ(hex/button、UARTなし)の出力。"""
    b = load_board('de0-cv')
    write_project_files(tmp_path, name='mnist', board=b,
                        sources=['top.v'],
                        groups=('clock', 'led', 'switch', 'button', 'hex'))
    qsf = (tmp_path / 'mnist.qsf').read_text(encoding='utf-8')
    assert 'PIN_U21 -to HEX0[0]' in qsf
    assert 'PIN_W19 -to HEX5[6]' in qsf
    assert 'PIN_U7 -to KEY[0]' in qsf
    assert 'IO_STANDARD "3.3-V LVTTL" -to HEX3[*]' in qsf
    assert 'UART' not in qsf                      # 未使用群は出力しない
    sdc = (tmp_path / 'mnist.sdc').read_text(encoding='utf-8')
    assert 'set_false_path -from [get_ports {KEY[*]}]' in sdc
    assert 'set_false_path -to   [get_ports {HEX5[*]}]' in sdc
    assert 'UART' not in sdc


def test_groups_unknown_or_missing(tmp_path):
    import pytest as _pytest
    b = load_board('de0-cv')
    with _pytest.raises(KeyError):
        write_project_files(tmp_path, name='x', board=b, sources=[],
                            groups=('clock', 'nonsense'))
    b2 = {k: v for k, v in b.items() if k != 'hex'}
    with _pytest.raises(KeyError):
        write_project_files(tmp_path, name='x', board=b2, sources=[],
                            groups=('clock', 'hex'))
    # uart グループ指定なのに uart 引数なし → 明示エラー
    with _pytest.raises(KeyError, match='uart 引数'):
        write_project_files(tmp_path, name='x', board=b, sources=[],
                            groups=('clock', 'uart'))
