"""M6: board JSON v2 スキーマの総なめ検証。

boards/*.json は全て schema_version=2 で pydantic 検証を通ること。
ピン未収録の解決要求は「創作禁止」の明示エラーになること。
"""

import json

import pytest

from mdl_fpga.board_db import (
    BOARD_SCHEMA_VERSION, get_header, list_boards, load_board,
    resolve_header_io)


def test_all_boards_validate():
    keys = list_boards()
    assert 'de0-cv' in keys
    for key in keys:
        b = load_board(key)
        assert b['schema_version'] == BOARD_SCHEMA_VERSION
        unit = b['resources']['unit']
        assert isinstance(b['resources'][unit], int)
        for h in b.get('headers', []):
            for idx in h.get('io', {}):
                assert idx.isdigit()


def test_de0cv_detection_and_headers():
    b = load_board('de0-cv')
    det = b['detection']
    assert det['cable'] == 'usb-blaster'
    # 実機採取値(boards/NOTICE 参照)
    assert det['jtag_idcodes'] == ['0x02b050dd']
    assert det['jtag_part_regex'] == '5CEBA4|5CE\\(BA4\\|FA4\\)'
    assert det['silk_hint']
    assert resolve_header_io(b, 'GPIO_0', 0)['pin'] == 'PIN_N16'
    assert resolve_header_io(b, 'GPIO_0', 0)['physical'] == 1
    assert resolve_header_io(b, 'GPIO_0', 1)['pin'] == 'PIN_B16'


def test_unknown_header_and_unrecorded_index():
    b = load_board('de0-cv')
    with pytest.raises(KeyError, match='未定義'):
        get_header(b, 'NOPE')
    with pytest.raises(KeyError, match='未収録'):
        resolve_header_io(b, 'GPIO_0', 39)


def test_schema_version_required(tmp_path, monkeypatch):
    import mdl_fpga.board_db as bdb
    bad = {'key': 'x', 'name': 'X', 'tier': 3, 'device': 'D', 'family': 'F',
           'clock': {}, 'resources': {'unit': 'le', 'le': 1}, 'limits': {}}
    d = tmp_path / 'boards'
    d.mkdir()
    (d / 'x.json').write_text(json.dumps(bad), encoding='utf-8')
    monkeypatch.setattr(bdb, 'BOARDS_DIR', d)
    with pytest.raises(ValueError, match='schema_version'):
        bdb.load_board('x')


def test_resources_unit_required(tmp_path, monkeypatch):
    import mdl_fpga.board_db as bdb
    bad = {'schema_version': 2, 'key': 'x', 'name': 'X', 'tier': 3,
           'device': 'D', 'family': 'F', 'clock': {},
           'resources': {'alm': 100}, 'limits': {}}
    d = tmp_path / 'boards'
    d.mkdir()
    (d / 'x.json').write_text(json.dumps(bad), encoding='utf-8')
    monkeypatch.setattr(bdb, 'BOARDS_DIR', d)
    with pytest.raises(Exception, match='resources.unit'):
        bdb.load_board('x')
