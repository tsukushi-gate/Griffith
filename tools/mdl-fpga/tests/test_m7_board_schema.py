"""M7: ボードスキーマ多ベンダ化(pin_format/toolchain)+ Tang Nano 20K 定義。

- tang-nano-20k.json が読めて転記値(2ソース機械突合、boards/NOTICE)を保持すること
- pin_format の強制: gowin ボードは素番号を受理、intel ボードは素番号を拒否
- RawAttach の共用体緩和と io_link 側のボード別厳密チェック
- 既存2ボード(de0-cv / de10-lite)の回帰
"""

import json

import pytest

from mdl_fpga.board_db import (
    PIN_FORMATS, load_board, pin_format_of, resolve_header_io)
from mdl_fpga.io_link import resolve_uart_link
from mdl_fpga.manifest import IoLink, RawAttach


def test_tang_nano_20k_loads_with_transcribed_pins():
    b = load_board('tang-nano-20k')
    assert b['toolchain'] == 'gowin-oss'
    assert b['pin_format'] == 'gowin'
    # Tier1 = 実機受入完了(up全段グリーン+hw-check bit一致)
    assert b['tier'] == 1 and 'unverified' not in b
    # 転記値(litex-boards 4f4caad + Sipeed example e23949a の機械突合)
    assert b['device'] == 'GW2AR-LV18QN88C8/I7'
    assert b['family'] == 'GW2A-18C'
    assert b['clock'] == {'port': 'CLK27', 'pin': '4', 'hz': 27_000_000}
    assert b['led']['pins'] == ['15', '16', '17', '18', '19', '20']
    assert b['led']['active_low'] is True
    assert b['button']['pins'] == ['88', '87']
    # 正論理(押下=High)。実機 A/B 実験で確定(NOTICE 参照)
    assert b['button']['active_low'] is False
    assert 'switch' not in b            # スライドスイッチ非搭載(物理事実)
    assert 'hex' not in b               # 7セグ非搭載
    assert b['resources']['unit'] == 'lut4'
    assert b['resources']['lut4'] == 20736
    # 実測較正(mnist24k 24000ゲート=LUT4 15966で76%ф配置失敗)
    assert b['limits']['calibrated'] is True
    assert b['limits']['gates_per_lut4'] == 1.5
    assert b['limits']['spatial_gates_max'] == 15000
    det = b['detection']
    assert det['ofl_board'] == 'tangnano20k'
    assert det['cable'] == 'ft2232'
    assert det['jtag_part_regex'] == 'GW2A'
    # 実機採取(openFPGALoader -b tangnano20k --detect 生出力)
    assert det['jtag_idcodes'] == ['0x0000081b']
    assert det['usb'][0]['vid'] == '0x0403'
    assert b['serial']['usb'][0]['pid'] == '0x6010'


def test_tang_nano_20k_bl616_uart_header():
    b = load_board('tang-nano-20k')
    # BL616 内蔵 UART: io[0]=FPGA 側 RX(litex serial.rx=70)/ io[1]=TX(=69)
    assert resolve_header_io(b, 'BL616_UART', 0)['pin'] == '70'
    assert resolve_header_io(b, 'BL616_UART', 1)['pin'] == '69'
    # 既定リンク(link=None)が最初のヘッダ=BL616_UART に解決される
    r = resolve_uart_link(b, None)
    assert (r.rx_pin, r.tx_pin) == ('70', '69')
    assert r.source == 'default'


def test_pin_format_of_resolution():
    assert pin_format_of({'pin_format': 'gowin'}) == 'gowin'
    assert pin_format_of({'toolchain': 'gowin-oss'}) == 'gowin'
    assert pin_format_of({'toolchain': 'quartus'}) == 'intel'
    assert pin_format_of({}) == 'intel'   # 既定(既存ボード互換)


def _board_stub(**over) -> dict:
    base = {
        'schema_version': 2, 'key': 'x', 'name': 'X', 'tier': 3,
        'device': 'D', 'family': 'F',
        'clock': {'port': 'CLK', 'pin': 'PIN_M9', 'hz': 50_000_000},
        'resources': {'unit': 'le', 'le': 1}, 'limits': {},
    }
    base.update(over)
    return base


def _load_stub(tmp_path, monkeypatch, board: dict):
    import mdl_fpga.board_db as bdb
    d = tmp_path / 'boards'
    d.mkdir(exist_ok=True)
    (d / 'x.json').write_text(json.dumps(board), encoding='utf-8')
    monkeypatch.setattr(bdb, 'BOARDS_DIR', d)
    return bdb.load_board('x')


def test_pin_format_enforced_gowin_accepts_bare_number(tmp_path, monkeypatch):
    b = _load_stub(tmp_path, monkeypatch, _board_stub(
        toolchain='gowin-oss',
        clock={'port': 'CLK27', 'pin': '4', 'hz': 27_000_000},
        led={'port': 'LED', 'pins': ['15', '16']}))
    assert b['led']['pins'] == ['15', '16']


def test_pin_format_enforced_intel_rejects_bare_number(tmp_path, monkeypatch):
    with pytest.raises(Exception, match='pin_format'):
        _load_stub(tmp_path, monkeypatch, _board_stub(
            toolchain='quartus',
            led={'port': 'LEDR', 'pins': ['76']}))


def test_pin_format_enforced_gowin_rejects_intel_pin(tmp_path, monkeypatch):
    with pytest.raises(Exception, match='pin_format'):
        _load_stub(tmp_path, monkeypatch, _board_stub(
            pin_format='gowin',
            clock={'port': 'CLK27', 'pin': 'PIN_M9', 'hz': 27_000_000}))


def test_unknown_pin_format_rejected(tmp_path, monkeypatch):
    with pytest.raises(Exception, match='pin_format'):
        _load_stub(tmp_path, monkeypatch, _board_stub(pin_format='nope'))


def test_raw_attach_union_and_board_strictness():
    # 共用体: Gowin 素番号も Intel PIN_xx も RawAttach 自体は受理
    RawAttach(rx_pin='70', tx_pin='69')
    RawAttach(rx_pin='PIN_N16', tx_pin='PIN_B16')
    with pytest.raises(ValueError):     # pydantic ValidationError は ValueError 系
        RawAttach(rx_pin='not a pin', tx_pin='69')
    # ボード別厳密化: gowin ボードに PIN_xx を渡すと io_link が拒否
    tn = load_board('tang-nano-20k')
    link = IoLink(attach=RawAttach(rx_pin='PIN_N16', tx_pin='PIN_B16'))
    with pytest.raises(ValueError, match='ピン形式'):
        resolve_uart_link(tn, link)
    # 正しい形式は通る
    ok = resolve_uart_link(tn, IoLink(attach=RawAttach(rx_pin='70',
                                                       tx_pin='69')))
    assert ok.rx_pin == '70'
    # intel ボードに素番号を渡しても拒否
    de = load_board('de0-cv')
    with pytest.raises(ValueError, match='ピン形式'):
        resolve_uart_link(de, IoLink(attach=RawAttach(rx_pin='70',
                                                      tx_pin='69')))


def test_existing_boards_regression():
    for key, fmt in (('de0-cv', 'intel'), ('de10-lite', 'intel')):
        b = load_board(key)
        assert b['toolchain'] == 'quartus'
        assert pin_format_of(b) == fmt
    # 従来の既定リンクが従来配線のまま
    de = load_board('de0-cv')
    r = resolve_uart_link(de, None)
    assert (r.rx_pin, r.tx_pin) == ('PIN_N16', 'PIN_B16')


def test_pin_formats_table_sane():
    import re
    assert set(PIN_FORMATS) >= {'intel', 'gowin', 'alnum'}
    assert re.match(PIN_FORMATS['gowin'], '76')
    assert not re.match(PIN_FORMATS['gowin'], 'PIN_N16')
    assert re.match(PIN_FORMATS['intel'], 'PIN_N16')
    assert re.match(PIN_FORMATS['alnum'], 'E3')
