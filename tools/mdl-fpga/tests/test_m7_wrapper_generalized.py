"""M7: ラッパの LED 幅/リセット汎用化(byte 一致契約の両面検証)。

- 既定 TopPorts(DE0-CV)の出力が tests/goldens/de0cv/ と厳密一致し続けること
  (test_golden_hdl.py と二重の帯紐)
- Tang Nano 20K ports(6LED 負論理+BTN[1] 負論理リセット)の出力が
  tests/goldens/tangnano20k/ と厳密一致すること
- 最小幅チェック(通常4 / selftest6)
"""

from pathlib import Path

import pytest

from mdl_fpga.board_db import load_board
from mdl_fpga.hdlgen.ports import TopPorts
from mdl_fpga.hdlgen.wrapper import generate_wrapper, generate_wrapper_tb
from mdl_fpga.io_link import resolve_uart_link

GOLD_DE0 = Path(__file__).parent / 'goldens' / 'de0cv'
GOLD_TN = Path(__file__).parent / 'goldens' / 'tangnano20k'


def _tn_ports() -> TopPorts:
    b = load_board('tang-nano-20k')
    return TopPorts.from_board(b, resolve_uart_link(b, None))


def test_default_ports_still_match_de0cv_golden():
    # 厳密一致は test_golden_hdl.py が正。ここでは汎用化の切替点だけを
    # 直接断言する(帯紐=幅10・switchリセット・正論理が従来文字列のまま)
    v = generate_wrapper(4, 1).top_verilog
    assert 'input  wire [9:0] SW,' in v
    assert 'rst_d1 <= SW[9];' in v
    assert 'wire [5:0] y_leds' in v
    assert ('assign LEDR = {y_leds, tx_busy, err_sticky, '
            '(ok_stretch != 0), (rx_stretch != 0)};') in v
    assert '~(' not in v            # 既定は正論理のまま
    st = generate_wrapper(4, 1, with_selftest=True).top_verilog
    # out_width=1 < 4 なのでゼロパディング形(de0cv golden と同一文字列)
    assert ('assign LEDR = {{{3{1\'b0}}, y_reg}, err_sticky, '
            '(ok_stretch != 0), tx_busy, hb[25], st_fail, st_pass};') in st
    assert '[9:6]=y 下位ビット' in st


def test_tn20k_wrapper_golden():
    p = _tn_ports()
    gen = generate_wrapper(4, 1, ports=p, clock_hz=27_000_000)
    assert gen.top_verilog == \
        (GOLD_TN / 'top_wrapper_4x1_tn20k.v').read_text(encoding='utf-8')
    st = generate_wrapper(4, 1, ports=p, clock_hz=27_000_000,
                          with_selftest=True)
    assert st.top_verilog == \
        (GOLD_TN / 'top_wrapper_4x1_selftest_tn20k.v').read_text(
            encoding='utf-8')
    tb = generate_wrapper_tb(gen, 8)
    assert tb == (GOLD_TN / 'tb_wrapper_4x1_tn20k.v').read_text(
        encoding='utf-8')


def test_tn20k_wrapper_key_properties():
    p = _tn_ports()
    # ボタンは正論理(押下=High/idle=Low)。実機 A/B 実験で確定
    # (~BTN[1] 版は常時リセットで hw-check 全滅 →
    #  リセット無効化版 8/8 PASS。boards/NOTICE 参照)
    assert (p.reset_group, p.reset_bit, p.reset_active_low) == \
        ('button', 1, False)
    assert p.led_active_low is True
    gen = generate_wrapper(4, 1, ports=p, clock_hz=27_000_000)
    v = gen.top_verilog
    assert 'input  wire [1:0] BTN,' in v
    assert 'output wire [5:0] LED,' in v
    assert 'rst_d1 <= BTN[1];' in v             # 正論理(押下=High=リセット)
    assert 'assign LED = ~(' in v               # 負論理 LED
    assert gen.clks_per_bit == 234              # 27MHz / 115200
    assert gen.groups == ('clock', 'uart', 'led', 'button')
    # selftest(6LED = 状態6bit のみ、y ビット無し)
    st = generate_wrapper(4, 1, ports=p, clock_hz=27_000_000,
                          with_selftest=True).top_verilog
    assert ('assign LED = ~({err_sticky, (ok_stretch != 0), tx_busy, '
            'hb[25], st_fail, st_pass});') in st
    # selftest の LED コメント行(checksumエラーで終わり y 表示は無い)
    assert '// [3]=TX中 [4]=フレーム受理 [5]=checksumエラー\n' in st


def test_tn20k_tb_reset_polarity():
    p = _tn_ports()
    gen = generate_wrapper(4, 1, ports=p, clock_hz=27_000_000)
    tb = generate_wrapper_tb(gen, 4)
    assert "reg  [1:0] sw = 2'd0;" in tb        # 休止=全0(押していない=Low)
    assert "sw[1] = 1'b1;" in tb                # アサート=High(押下)
    assert '.BTN(sw)' in tb


def test_active_low_reset_variant_still_supported():
    # 負論理ボタンのボード(将来)向けの経路も生成できることを固定
    p = TopPorts(reset_group='button', button='KEYN', button_width=2,
                 reset_active_low=True, led_width=6)
    gen = generate_wrapper(4, 1, ports=p)
    assert 'rst_d1 <= ~KEYN[1];' in gen.top_verilog
    tb = generate_wrapper_tb(gen, 4)
    assert "reg  [1:0] sw = 2'd3;" in tb        # 休止=全1
    assert "sw[1] = 1'b0;" in tb                # アサート=Low


def test_led_width_minimums():
    with pytest.raises(NotImplementedError, match='led_width'):
        generate_wrapper(4, 1, ports=TopPorts(led_width=3))
    with pytest.raises(NotImplementedError, match='led_width'):
        generate_wrapper(4, 1, with_selftest=True,
                         ports=TopPorts(led_width=5))
    # 4本(y ビット無し)は通常モードで生成できる
    v = generate_wrapper(4, 1, ports=TopPorts(led_width=4)).top_verilog
    assert 'y_leds' not in v
    assert ('assign LEDR = {tx_busy, err_sticky, '
            '(ok_stretch != 0), (rx_stretch != 0)};') in v
