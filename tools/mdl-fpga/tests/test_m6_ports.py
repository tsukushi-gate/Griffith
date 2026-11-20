"""M6: TopPorts(生成トップのポート名/幅のデータ駆動化)の検証。

既定 TopPorts は DE0-CV の従来リテラルと同値(byte 一致はゴールデンが担保)。
ここでは from_board の導出と、非既定ポート名/幅への追従を検証する。
"""

import json

import pytest

from conftest import build_groupsum_model
from mdl_fpga.board_db import load_board
from mdl_fpga.hdlgen.ports import TopPorts
from mdl_fpga.hdlgen.system import generate_system
from mdl_fpga.hdlgen.wrapper import generate_wrapper, generate_wrapper_tb
from mdl_fpga.io_link import resolve_uart_link
from mdl_fpga.ir_io import load_graph


def test_from_board_de0cv_equals_defaults():
    """board JSON 由来の TopPorts が既定値と一致(将来の board 改変の検知)。"""
    b = load_board('de0-cv')
    assert TopPorts.from_board(b) == TopPorts()
    r = resolve_uart_link(b, None)
    assert TopPorts.from_board(b, r) == TopPorts()


def _alt_board() -> dict:
    return {
        'key': 'alt', 'clock': {'port': 'CLK25', 'pin': 'PIN_X1',
                                'hz': 25_000_000},
        'switch': {'port': 'SW_ALT', 'pins': [f'PIN_S{i}' for i in range(8)]},
        'led': {'port': 'LED_ALT', 'pins': [f'PIN_L{i}' for i in range(10)]},
        'button': {'port': 'BTN', 'pins': ['PIN_B1', 'PIN_B2']},
        'hex': {'digits': [{'port': f'D{i}', 'pins': [f'PIN_H{i}{j}'
                                                      for j in range(7)]}
                           for i in range(6)]},
    }


def test_from_board_alt_names_and_widths():
    p = TopPorts.from_board(_alt_board())
    assert p.clock == 'CLK25'
    assert (p.switch, p.switch_width) == ('SW_ALT', 8)
    assert (p.led, p.led_width) == ('LED_ALT', 10)
    assert (p.button, p.button_width) == ('BTN', 2)
    assert p.hex_digits == ('D0', 'D1', 'D2', 'D3', 'D4', 'D5')
    assert p.reset_switch_index == 7


def test_wrapper_follows_ports():
    p = TopPorts.from_board(_alt_board())
    gen = generate_wrapper(4, 1, clks_per_bit=16, ports=p)
    v = gen.top_verilog
    assert 'input  wire       CLK25,' in v
    assert 'input  wire [7:0] SW_ALT,' in v
    assert 'output wire [9:0] LED_ALT,' in v
    assert 'wire clk = CLK25;' in v
    assert 'rst_d1 <= SW_ALT[7];' in v
    assert 'assign LED_ALT = ' in v
    assert 'CLOCK_50' not in v and 'LEDR' not in v
    tb = generate_wrapper_tb(gen, 4)
    assert '.CLK25(clk), .SW_ALT(sw), .LED_ALT(ledr),' in tb
    assert "sw[7] = 1'b1;" in tb


def test_wrapper_led_width_minimums():
    # M7 で幅一般化: 最小本数(通常4 / selftest6)未満のみ拒否する
    with pytest.raises(NotImplementedError, match='led_width'):
        generate_wrapper(4, 1, ports=TopPorts(led_width=3))
    with pytest.raises(NotImplementedError, match='led_width'):
        generate_wrapper(4, 1, with_selftest=True, ports=TopPorts(led_width=5))
    # 8本(旧仕様では拒否)は生成できる
    gen = generate_wrapper(4, 1, ports=TopPorts(led_width=8))
    assert 'output wire [7:0] LEDR,' in gen.top_verilog
    assert 'wire [3:0] y_leds' in gen.top_verilog   # 8-4=4bit の y 表示


def test_system_follows_ports(tmp_path):
    ir = tmp_path / 'gs.ir.json'
    ir.write_text(json.dumps(build_groupsum_model()), encoding='utf-8')
    graph = load_graph(ir)
    p = TopPorts.from_board(_alt_board())
    gen = generate_system(graph, n_images=6, ports=p)
    v = gen.top_verilog
    assert 'input  wire [1:0] BTN,' in v
    assert 'output wire [6:0] D5' in v
    assert 'BTN[0]}};' in v.replace(' ', '') or 'BTN[0]' in v
    assert '.seg(D0));' in v
    assert 'CLOCK_50' not in v and 'HEX0' not in v
    assert gen.groups == ('clock', 'led', 'switch', 'button', 'hex')


def test_system_hex_digits_not_6_rejected(tmp_path):
    ir = tmp_path / 'gs.ir.json'
    ir.write_text(json.dumps(build_groupsum_model()), encoding='utf-8')
    graph = load_graph(ir)
    p = TopPorts(hex_digits=('D0', 'D1', 'D2', 'D3'))
    with pytest.raises(NotImplementedError, match='hex'):
        generate_system(graph, n_images=6, ports=p)


def test_wrapper_gen_carries_groups():
    gen = generate_wrapper(4, 1)
    assert gen.groups == ('clock', 'uart', 'led', 'switch')
    assert gen.ports == TopPorts()
