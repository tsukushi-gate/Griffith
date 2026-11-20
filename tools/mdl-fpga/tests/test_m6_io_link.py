"""M6: io.link(manifest 0.2)の解決と後方互換の検証。

uart 配線は board(物理事実)ではなく manifest io.link(プロジェクト側の
配線判断)が持つ。既定リンクは従来の DE0-CV 実配線と同値で、qsf/sdc は
Stage0 ゴールデンと一致する(出力不変の契約)。
"""

import json
from pathlib import Path

import pytest

from mdl_fpga.board_db import load_board
from mdl_fpga.io_link import (
    DEFAULT_UART_LINK, UART_RX_PORT, UART_TX_PORT, resolve_uart_link)
from mdl_fpga.manifest import (
    HeaderAttach, IoLink, RawAttach, load_manifest)
from mdl_fpga.synth import write_project_files

GOLD = Path(__file__).resolve().parent / 'goldens' / 'de0cv'
EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'


def test_default_link_resolves_to_legacy_pins():
    b = load_board('de0-cv')
    r = resolve_uart_link(b, None)
    assert (r.rx_port, r.rx_pin) == (UART_RX_PORT, 'PIN_N16')
    assert (r.tx_port, r.tx_pin) == (UART_TX_PORT, 'PIN_B16')
    assert r.baud == 115200
    assert r.source == 'default'
    assert r.rx_physical == 1 and r.tx_physical == 2
    d = r.to_uart_dict()
    assert d['rx'] == {'port': 'UART_RXD', 'pin': 'PIN_N16'}
    assert d['tx'] == {'port': 'UART_TXD', 'pin': 'PIN_B16'}


def test_raw_pin_attach():
    b = load_board('de0-cv')
    link = IoLink(baud=9600,
                  attach=RawAttach(rx_pin='PIN_A1', tx_pin='PIN_B2'))
    r = resolve_uart_link(b, link)
    assert r.rx_pin == 'PIN_A1' and r.tx_pin == 'PIN_B2'
    assert r.baud == 9600 and r.source == 'manifest'
    assert r.rx_physical is None


def test_unknown_header_and_unrecorded_index():
    b = load_board('de0-cv')
    with pytest.raises(KeyError, match='未定義'):
        resolve_uart_link(
            b, IoLink(attach=HeaderAttach(header='NOPE', rx_index=0,
                                          tx_index=1)))
    with pytest.raises(KeyError, match='未収録'):
        resolve_uart_link(
            b, IoLink(attach=HeaderAttach(header='GPIO_0', rx_index=0,
                                          tx_index=39)))


def test_unsupported_link_type():
    b = load_board('de0-cv')
    with pytest.raises(ValueError, match='未対応'):
        resolve_uart_link(b, IoLink(type='spi'))


def test_manifest_01_backward_compat(tmp_path):
    doc = {'manifest_version': '0.1', 'ir': 'x.ir.json'}
    p = tmp_path / 'm.manifest.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    mf = load_manifest(p)
    assert mf.io.link is None
    assert mf.effective_baud() == 115200
    # link 無し → 既定リンク(=従来配線)で解決される
    r = resolve_uart_link(load_board('de0-cv'), mf.io.link)
    assert r.rx_pin == 'PIN_N16'


def test_baud_mismatch_rejected(tmp_path):
    doc = {'manifest_version': '0.2', 'ir': 'x.ir.json',
           'io': {'protocol': 'uart-115200-8n1',
                  'link': {'type': 'uart', 'baud': 9600}}}
    p = tmp_path / 'm.manifest.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='不一致'):
        load_manifest(p)


def test_full_adder_manifest_e2e_matches_golden(tmp_path):
    """examples の manifest 実例 → io.link 解決 → qsf/sdc が Stage0 ゴールデンと一致。"""
    mf = load_manifest(EXAMPLES / 'full_adder.manifest.json')
    assert mf.board == 'de0-cv'
    assert mf.effective_baud() == 115200
    b = load_board(mf.board)
    uart = resolve_uart_link(b, mf.io.link).to_uart_dict()
    write_project_files(tmp_path, name='top', board=b,
                        sources=['top.v', 'mdl_core.v', 'uart_rx.v',
                                 'uart_tx.v', 'frame_rx.v', 'frame_tx.v'],
                        uart=uart)
    for ext in ('qsf', 'sdc'):
        actual = (tmp_path / f'top.{ext}').read_text(encoding='utf-8')
        expected = (GOLD / f'top.{ext}').read_text(encoding='utf-8')
        assert actual == expected, f'manifest 経由の top.{ext} がゴールデン不一致'


def test_default_link_constant_is_legacy_wiring():
    at = DEFAULT_UART_LINK.attach
    assert isinstance(at, HeaderAttach)
    assert (at.header, at.rx_index, at.tx_index) == ('GPIO_0', 0, 1)
    assert DEFAULT_UART_LINK.baud == 115200
