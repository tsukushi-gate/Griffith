"""board-def fetch(AUTOFLOW 工程3)の検証。

- プロバイダ: litex AST / cst / qsf / part.hpp の事実抽出(実行なし)
- 突合エンジン: 2ソース一致・不一致停止・at(位置)参照・単一ソース検出
- 組み立て: BoardDef スキーマ検証を通る draft の生成、tier/unverified 規約
- 再生成一致: レシピ+references/ のローカルコピーから、実機受入済みの
  boards/tang-nano-20k.json と同一の物理事実(ピン・デバイス)を再導出できる
"""

import json
from pathlib import Path

import pytest

from mdl_fpga.board_db import load_board
from mdl_fpga.boarddef import providers
from mdl_fpga.boarddef.fetch import (
    REPO_ROOT, CrossCheckError, SourceData, build_board, crosscheck_role,
    crosscheck_scalar, run_fetch)

GOLDEN_QSF = Path(__file__).resolve().parent / 'goldens' / 'de0cv' / 'top.qsf'
TN20K_REFS = REPO_ROOT / 'references' / 'tang-nano-20k'

requires_tn20k_refs = pytest.mark.skipif(
    not TN20K_REFS.exists(),
    reason='references/tang-nano-20k のローカルコピーが無い')


# ---------------------------------------------------------- providers ---


@requires_tn20k_refs
def test_parse_litex_real_platform():
    text = (TN20K_REFS / 'litex-boards' / 'sipeed_tang_nano_20k.py'
            ).read_text(encoding='utf-8')
    p = providers.parse_litex_platform(text)
    assert [e.pins for e in p.entries('clk27')] == [['4']]
    assert p.entries('clk27')[0].iostandard == 'LVCMOS33'
    leds = p.entries('led_n')
    assert [e.index for e in leds] == list(range(6))
    assert [e.pins[0] for e in leds] == ['15', '16', '17', '18', '19', '20']
    serial = p.entries('serial')[0]
    assert serial.subsignals['rx'] == ['70']
    assert serial.subsignals['tx'] == ['69']
    assert p.device == 'GW2AR-LV18QN88C8/I7'
    assert p.devicename == 'GW2AR-18C'
    assert p.cable == 'ft2232'
    assert set(p.as_gpio) == {'mspi', 'sspi', 'ready', 'done'}
    # Pins(幅) 形(SIP 内蔵 SDRAM)はピン無し・幅のみで抽出される
    addr = p.entries('O_sdram_addr')[0]
    assert addr.pins == [] and addr.width == 11
    # コネクタ('-' プレースホルダ込みの素の列)
    assert p.connectors['J5'][3] == '75'


def test_parse_litex_synthetic_dict_connector():
    text = '''
_io = [
    ("clk", 0, Pins("9"), IOStandard("LVCMOS33")),
    ("pair", 0, Subsignal("p", Pins("1 2")), Subsignal("n", Pins("3 4"))),
]
_connectors = [("J1", {7: "A7", 8: "A8"})]

class Platform(XilinxPlatform):
    def __init__(self):
        XilinxPlatform.__init__(self, "xc7a35t", _io, _connectors)
'''
    p = providers.parse_litex_platform(text)
    assert p.entries('pair')[0].subsignals['p'] == ['1', '2']
    assert p.connectors['J1'] == {'7': 'A7', '8': 'A8'}
    assert p.device == 'xc7a35t'


@requires_tn20k_refs
def test_parse_cst_real_files():
    hdmi = providers.parse_cst(
        (TN20K_REFS / 'sipeed-examples' / 'hdmi_src_hdmi.cst'
         ).read_text(encoding='utf-8'))
    assert hdmi.locs['I_key'] == '87'
    assert hdmi.locs['O_tmds_clk_p'] == '33,34'      # 差動ペアはそのまま保持
    assert hdmi.part_number == 'GW2AR-LV18QN88C8/I7'
    assert hdmi.ports['I_key']['PULL_MODE'] == 'DOWN'
    top = providers.parse_cst(
        (TN20K_REFS / 'sipeed-examples' / 'top.cst').read_text(encoding='utf-8'))
    assert top.locs == {'clk': '4', 'rst': '88',
                        'uart_rx': '70', 'uart_tx': '69'}
    assert top.ports['uart_rx']['IO_TYPE'] == 'LVCMOS33'


def test_parse_qsf_golden():
    m = providers.parse_qsf(GOLDEN_QSF.read_text(encoding='utf-8'))
    assert m['CLOCK_50'] == 'PIN_M9'
    assert m['LEDR[0]'] == 'PIN_AA2'
    assert m['SW[9]'] == 'PIN_AB12'


def test_parse_part_hpp():
    text = '''
static std::map<uint32_t, fpga_model> fpga_list = {
    {0x0000081b, {"Gowin", "GW2A(R)-18(C)", "", 8}},
    {0x02b050dd, {"Intel", "Cyclone V 5CEBA4", "", 10}},
};'''
    parts = providers.parse_part_hpp(text)
    assert {'idcode': '0x0000081b', 'vendor': 'Gowin',
            'model': 'GW2A(R)-18(C)'} in parts
    assert len(parts) == 2


# ----------------------------------------------------------- 突合層 ---


def _fake_sources() -> dict:
    lx = providers.parse_litex_platform('''
_io = [
    ("clk", 0, Pins("4"), IOStandard("LVCMOS33")),
    ("led", 0, Pins("15"), IOStandard("LVCMOS33")),
    ("led", 1, Pins("16"), IOStandard("LVCMOS33")),
]
class Platform(GowinPlatform):
    def __init__(self):
        GowinPlatform.__init__(self, "GW-TEST", _io, [])
''')
    cst = providers.parse_cst('''
//Part Number: GW-TEST
IO_LOC "clk" 4;
IO_LOC "led[0]" 15;
IO_LOC "led[1]" 16;
''')
    return {'a': SourceData('litex', 'test', platform=lx),
            'b': SourceData('cst', 'test', files={'x.cst': cst})}


def test_crosscheck_role_two_sources_agree():
    rr = crosscheck_role('led', [
        {'source': 'a', 'signal': 'led'},
        {'source': 'b', 'file': 'x.cst', 'net': 'led[{i}]', 'count': 2},
    ], _fake_sources())
    assert rr.pins == ['15', '16']
    assert rr.verified and rr.single_positions() == []


def test_crosscheck_role_mismatch_refuses():
    src = _fake_sources()
    src['b'].files['x.cst'].locs['led[1]'] = '99'
    with pytest.raises(CrossCheckError, match='不一致'):
        crosscheck_role('led', [
            {'source': 'a', 'signal': 'led'},
            {'source': 'b', 'file': 'x.cst', 'net': 'led[{i}]', 'count': 2},
        ], src)


def test_crosscheck_role_at_position_witness():
    rr = crosscheck_role('led', [
        {'source': 'a', 'signal': 'led'},
        {'source': 'b', 'file': 'x.cst', 'net': 'led[0]', 'at': 0},
    ], _fake_sources())
    assert rr.pins == ['15', '16']
    assert not rr.verified                     # 位置1 は単一ソース
    assert rr.single_positions() == [1]


def test_crosscheck_role_at_mismatch_refuses():
    with pytest.raises(CrossCheckError, match=r'led\[0\]'):
        crosscheck_role('led', [
            {'source': 'a', 'signal': 'led'},
            {'source': 'b', 'file': 'x.cst', 'net': 'led[1]', 'at': 0},
        ], _fake_sources())


def test_crosscheck_role_missing_net_refuses():
    with pytest.raises(CrossCheckError, match='存在しない'):
        crosscheck_role('x', [{'source': 'b', 'file': 'x.cst', 'net': 'nope'}],
                        _fake_sources())


def test_crosscheck_scalar():
    val, ok, _w = crosscheck_scalar('device', [
        {'source': 'a', 'field': 'device'},
        {'source': 'b', 'file': 'x.cst', 'field': 'part_number'},
    ], _fake_sources())
    assert val == 'GW-TEST' and ok


# ------------------------------------------------------- 組み立て/CLI ---


def _minimal_recipe(tmp_path) -> Path:
    doc = {
        'recipe_version': 1, 'key': 'testboard', 'name': 'Test Board',
        'toolchain': 'gowin-oss', 'pin_format': 'gowin', 'family': 'GW-F',
        'sources': {
            'a': {'provider': 'litex',
                  'url': 'https://example.invalid/platform.py'},
        },
        'device_refs': [{'source': 'a', 'field': 'device'}],
        'clock': {'port': 'CLK', 'hz': 1000000,
                  'refs': [{'source': 'a', 'signal': 'clk'}]},
        'led': {'port': 'LED', 'refs': [{'source': 'a', 'signal': 'led'}]},
        'resources': {'unit': 'lut4', 'lut4': 100, 'note': 'テスト値'},
    }
    p = tmp_path / 'testboard.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    return p


_FAKE_PLATFORM = '''
_io = [
    ("clk", 0, Pins("4")),
    ("led", 0, Pins("15")),
]
class Platform(GowinPlatform):
    def __init__(self):
        GowinPlatform.__init__(self, "GW-TEST", _io, [])
'''


def test_run_fetch_url_source_single_source_tier3(tmp_path):
    calls = []

    def fetcher(url):
        calls.append(url)
        return _FAKE_PLATFORM

    res = run_fetch(_minimal_recipe(tmp_path), out_dir=tmp_path / 'out',
                    fetcher=fetcher)
    assert res.ok, res.error
    assert calls == ['https://example.invalid/platform.py']
    board = json.loads(Path(res.board_path).read_text(encoding='utf-8'))
    assert board['tier'] == 3 and board['unverified'] is True
    assert board['clock']['pin'] == '4'
    assert board['led']['pins'] == ['15']
    assert 'clock[0]' in res.single_source     # 単一ソース位置が記録される
    notice = Path(res.notice_path).read_text(encoding='utf-8')
    assert '単一ソース位置' in notice


def test_run_fetch_offline_without_local_refuses(tmp_path):
    res = run_fetch(_minimal_recipe(tmp_path), out_dir=tmp_path / 'out',
                    offline=True)
    assert not res.ok and 'offline' in res.error


def test_run_fetch_refuses_overwrite(tmp_path):
    def fetcher(url):
        return _FAKE_PLATFORM

    out = tmp_path / 'out'
    r1 = run_fetch(_minimal_recipe(tmp_path), out_dir=out, fetcher=fetcher)
    assert r1.ok
    r2 = run_fetch(_minimal_recipe(tmp_path), out_dir=out, fetcher=fetcher)
    assert not r2.ok and '上書き' in r2.error


# --------------------------------------------- Tang Nano 20K 再生成一致 ---


@requires_tn20k_refs
def test_tang_nano_20k_regeneration_matches_committed(tmp_path):
    """レシピ+ローカルコピー(オフライン)で、実機受入済みの committed
    board JSON と同一の物理事実が再導出されること(fetch の受入基準)。"""
    res = run_fetch('tang-nano-20k', out_dir=tmp_path / 'out', offline=True)
    assert res.ok, res.error
    gen = json.loads(Path(res.board_path).read_text(encoding='utf-8'))
    ref = load_board('tang-nano-20k')

    assert gen['device'] == ref['device']
    assert gen['family'] == ref['family']
    assert gen['clock'] == ref['clock']
    assert gen['led']['pins'] == ref['led']['pins']
    assert gen['led']['active_low'] == ref['led']['active_low']
    assert gen['button']['pins'] == ref['button']['pins']
    assert gen['button']['active_low'] == ref['button']['active_low']
    gh = {h['name']: h for h in gen['headers']}
    rh = {h['name']: h for h in ref['headers']}
    assert gh.keys() == rh.keys()
    assert {k: v['pin'] for k, v in gh['BL616_UART']['io'].items()} == \
           {k: v['pin'] for k, v in rh['BL616_UART']['io'].items()}
    assert gen['io_standard'] == ref['io_standard']
    assert gen['synth_options']['nextpnr_vopts'] == \
           ref['synth_options']['nextpnr_vopts']
    assert set(gen['synth_options']['gowin_pack_flags']) == \
           set(ref['synth_options']['gowin_pack_flags'])
    # draft 規約: 実機由来の値(tier1 / IDCODE / USB 署名 / 較正値)は持たない
    assert gen['tier'] == 2 and gen['unverified'] is True
    assert 'jtag_idcodes' not in gen.get('detection', {})
    assert 'usb' not in gen.get('detection', {})
    assert gen['limits']['calibrated'] is False
    # 全役割が2ソース検証済み(単一ソース位置なし)
    assert res.single_source == []


@requires_tn20k_refs
def test_cli_board_def_fetch_smoke(tmp_path, capsys):
    from mdl_fpga.cli import main
    rc = main(['board-def', 'fetch', 'tang-nano-20k', '--offline',
               '--out', str(tmp_path / 'cli-out')])
    out = capsys.readouterr().out
    assert rc == 0
    assert '[OK]' in out and 'tang-nano-20k.json' in out
