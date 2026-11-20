"""pin-validate(AUTOFLOW 工程4)の検証。

- stub 生成(verilog / cst / qsf)と方向・重複の扱い(純粋関数、ツール不要)
- ツール不在時は skipped(OK と言わない)
- 統合(要 oss-cad-suite): 実 nextpnr が正しい board を受理し、
  実在しないピンに改竄した board を拒否すること
"""

import copy

import pytest
from conftest import _tc

from mdl_fpga.board_db import load_board
from mdl_fpga.pin_validate import (
    collect_pins, run_pin_validate, stub_cst, stub_qsf, stub_verilog)

requires_gowin_oss = pytest.mark.skipif(
    _tc.yosys is None or _tc.nextpnr_himbaechel is None,
    reason='yosys / nextpnr-himbaechel が見つからない(doctor 参照)')


def test_collect_pins_tang_nano_20k():
    b = load_board('tang-nano-20k')
    entries, dups = collect_pins(b)
    assert dups == []
    by_name = {n: (p, d) for n, p, d in entries}
    assert by_name['pv_clock'] == ('4', 'input')
    assert by_name['pv_led_0_'] == ('15', 'output')
    assert by_name['pv_button_1_'] == ('87', 'input')
    assert by_name['pv_BL616_UART_0_'] == ('70', 'input')
    # clock1 + led6 + btn2 + uart2 = 11 ピン
    assert len(entries) == 11


def test_collect_pins_dedup():
    b = copy.deepcopy(load_board('tang-nano-20k'))
    b['headers'].append({'name': 'X', 'io': {'0': {'pin': '15'}},
                         'power': [], 'gnd': []})   # led[0] と同一ピン
    entries, dups = collect_pins(b)
    assert len(dups) == 1 and 'X[0]' in dups[0]
    assert len(entries) == 11                        # 重複は数に入らない


def test_stub_verilog_directions():
    v = stub_verilog([('pv_clock', '4', 'input'), ('pv_led_0_', '15', 'output'),
                      ('pv_btn', '88', 'input')])
    assert 'input pv_clock' in v and 'output pv_led_0_' in v
    assert 'pv_clock ^ pv_btn' in v
    assert 'assign pv_led_0_ = pv_agg;' in v


def test_stub_cst_and_qsf_content():
    b = load_board('tang-nano-20k')
    entries, _ = collect_pins(b)
    cst = stub_cst(entries, b)
    assert 'IO_LOC "pv_clock" 4;' in cst
    assert 'IO_PORT "pv_clock" IO_TYPE=LVCMOS33;' in cst
    de = load_board('de0-cv')
    d_entries, _ = collect_pins(de)
    qsf = stub_qsf(d_entries, de)
    assert 'set_global_assignment -name DEVICE 5CEBA4F23C7' in qsf
    assert 'set_location_assignment PIN_M9 -to pv_clock' in qsf
    assert 'IO_STANDARD "3.3-V LVTTL"' in qsf


def test_skipped_when_tools_missing(tmp_path, monkeypatch):
    # 先行テストが PATH に oss-cad-suite を前置していても不在状態を再現する
    monkeypatch.setattr('shutil.which', lambda n, **kw: None)
    from mdl_fpga.toolchain import Toolchain
    empty = Toolchain(oss_root=tmp_path / 'nothing')
    res = run_pin_validate('tang-nano-20k', tc=empty,
                           workdir=tmp_path / 'wd')
    assert res.skipped and not res.ok and res.exit_code == 2
    assert '見つからない' in res.detail


def test_unknown_board_reports_failure():
    res = run_pin_validate('no-such-board')
    assert not res.ok and not res.skipped
    assert '読込失敗' in res.detail


# ------------------------------------------------------------- 統合 ---


@requires_gowin_oss
def test_gowin_validate_accepts_committed_board(tmp_path):
    res = run_pin_validate('tang-nano-20k', workdir=tmp_path / 'ok')
    assert res.skipped is False
    assert res.ok, f'{res.detail}\n' + '\n'.join(res.errors)
    assert res.n_pins == 11 and res.backend == 'gowin-oss'


@requires_gowin_oss
def test_gowin_validate_rejects_nonexistent_pin(tmp_path):
    b = copy.deepcopy(load_board('tang-nano-20k'))
    b['clock']['pin'] = '999'          # QFN88 に存在しないピン
    res = run_pin_validate(b, workdir=tmp_path / 'bad')
    assert res.skipped is False
    assert not res.ok, 'ありえないピンを受理してはならない'


@requires_gowin_oss
def test_cli_board_def_validate_smoke(tmp_path, capsys):
    from mdl_fpga.cli import main
    rc = main(['board-def', 'validate', 'tang-nano-20k',
               '--workdir', str(tmp_path / 'cli')])
    out = capsys.readouterr().out
    assert rc == 0
    assert '[OK]' in out
