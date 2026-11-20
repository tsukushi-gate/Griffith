"""M7: gowin-oss バックエンドのプロジェクト生成+レポートパーサ(ツール不要)。

test_m2_synth_gen(qsf/sdc 生成+Quartus レポートパーサ)の Gowin 版。
cst/ys/gowin_build.json は tests/goldens/tangnano20k/ と厳密一致。
nextpnr レポートは缶詰 JSON(Phase4 実機フロー後に実出力へ更新予定)。
"""

import json
from pathlib import Path

import pytest

from mdl_fpga.backends.gowin_oss import (
    GowinOssBackend, parse_nextpnr_log_fmax, parse_nextpnr_report,
    write_project_files)
from mdl_fpga.board_db import load_board
from mdl_fpga.io_link import resolve_uart_link

GOLD = Path(__file__).parent / 'goldens' / 'tangnano20k'


def _gen(tmp_path, groups=('clock', 'uart', 'led', 'button'), **kw):
    board = load_board('tang-nano-20k')
    uart = resolve_uart_link(board, None).to_uart_dict()
    sources = ['top.v', 'mdl_core.v', 'uart_rx.v', 'uart_tx.v',
               'frame_rx.v', 'frame_tx.v']
    write_project_files(tmp_path, name='top', board=board, sources=sources,
                        clock_hz=27_000_000, groups=groups, uart=uart, **kw)
    return tmp_path


def test_cst_ys_build_json_golden(tmp_path):
    d = _gen(tmp_path)
    for f in ('top.cst', 'top.ys', 'gowin_build.json'):
        assert (d / f).read_text(encoding='utf-8') == \
               (GOLD / f).read_text(encoding='utf-8'), f'{f} が golden と不一致'


def test_build_json_contents(tmp_path):
    d = _gen(tmp_path)
    doc = json.loads((d / 'gowin_build.json').read_text(encoding='utf-8'))
    assert doc['backend'] == 'gowin-oss'
    assert doc['device'] == 'GW2AR-LV18QN88C8/I7'
    assert doc['family'] == 'GW2A-18C'
    assert doc['clock_hz'] == 27_000_000
    # litex toolchain.options からの機械転記(dual-purpose ピン開放)
    assert doc['gowin_pack_flags'] == ['--mspi_as_gpio', '--sspi_as_gpio',
                                       '--ready_as_gpio', '--done_as_gpio']
    assert doc['nextpnr_vopts'] == ['sspi_as_gpio']


def test_groups_validation(tmp_path):
    board = load_board('tang-nano-20k')
    with pytest.raises(KeyError, match='uart 引数が必要'):
        write_project_files(tmp_path, name='top', board=board,
                            sources=['top.v'], clock_hz=27_000_000,
                            groups=('clock', 'uart'), uart=None)
    with pytest.raises(KeyError, match='未知のポート群'):
        write_project_files(tmp_path, name='top', board=board,
                            sources=['top.v'], clock_hz=27_000_000,
                            groups=('clock', 'nope'), uart=None)
    # Tang Nano 20K に switch(スライドスイッチ)は無い → 明示エラー
    with pytest.raises(KeyError, match='switch'):
        write_project_files(tmp_path, name='top', board=board,
                            sources=['top.v'], clock_hz=27_000_000,
                            groups=('clock', 'switch'), uart=None)


def test_parse_nextpnr_report_canned():
    # GW2A-18C の実レポートから凍結(full_adder+selftest 実走。
    # runs/tn20k_full_adder_build/gowin/top_report.json の抜粋)
    doc = {
        'utilization': {
            'ALU': {'available': 15552, 'used': 332},
            'DFF': {'available': 15552, 'used': 336},
            'IOB': {'available': 384, 'used': 11},
            'LUT4': {'available': 20736, 'used': 576},
            # LUT 合成木(素 LUT ではない)→ lut に合算しないこと
            'MUX2_LUT5': {'available': 10368, 'used': 66},
            'MUX2_LUT6': {'available': 5184, 'used': 21},
        },
        'fmax': {
            'clk': {'achieved': 209.6436004638672, 'constraint': 27},
        },
    }
    s = parse_nextpnr_report(doc)
    assert s['lut'] == 576 and s['lut_avail'] == 20736   # MUX2_LUT* 非合算
    assert s['lut_pct'] == pytest.approx(2.78, abs=0.01)
    assert s['ff'] == 336 and s['ff_avail'] == 15552
    assert s['alu'] == 332 and s['alu_avail'] == 15552   # Gowin キャリー系
    assert s['fmax_mhz'] == pytest.approx(209.64, abs=0.01)
    assert s['fmax_clk_mhz'] == pytest.approx(209.64, abs=0.01)
    assert s['freq_target_mhz'] == pytest.approx(27.0)
    assert s['timing_ok'] is True


def test_parse_nextpnr_report_timing_fail_and_empty():
    doc = {'utilization': {'LUT4': {'used': 1, 'available': 100}},
           'fmax': {'clk': {'achieved': 20.0, 'constraint': 27.0}}}
    s = parse_nextpnr_report(doc)
    assert s['timing_ok'] is False
    assert parse_nextpnr_report({}) == {}


def test_parse_nextpnr_log_fmax_fallback():
    log = ("Info: Max frequency for clock 'CLK27$glb_clk': 63.41 MHz "
           "(PASS at 27.00 MHz)\n")
    s = parse_nextpnr_log_fmax(log)
    assert s['fmax_mhz'] == pytest.approx(63.41)
    assert parse_nextpnr_log_fmax('no match') == {}


def test_backend_properties_and_convert_refusal():
    be = GowinOssBackend()
    assert be.name == 'gowin-oss'
    assert be.bitstream_ext == '.fs'
    assert be.project_subdir == 'gowin'
    assert be.convert_formats == ()
    assert be.bitstream_globs('top') == ('gowin/top.fs', 'top.fs')
    with pytest.raises(ValueError, match='変換は不要'):
        be.convert_bitstream('top.fs', 'svf')


def test_available_with_stub():
    class NoneTc:
        yosys = None
        nextpnr_himbaechel = None
        gowin_pack = None

    class AllTc:
        yosys = 'y'
        nextpnr_himbaechel = 'n'
        gowin_pack = 'g'

    ok, msg = GowinOssBackend().available(NoneTc())
    assert not ok and 'yosys' in msg and 'nextpnr-himbaechel' in msg
    ok, _ = GowinOssBackend().available(AllTc())
    assert ok
