"""M7: 合成バックエンド層(backends/)のレジストリ・ディスパッチ・シム互換。

- board JSON の toolchain 欄 → バックエンド解決(family 推定フォールバック込み)
- 未知 toolchain の明示エラー
- 偽バックエンドの monkeypatch 登録 = build.py 無改修の拡張性証明
- mdl_fpga.synth シムの再エクスポート互換(既存 import が無改修で動く)
"""

import pytest

from mdl_fpga import backends
from mdl_fpga.backends import get_backend
from mdl_fpga.backends.base import FlowResult, SynthBackend
from mdl_fpga.backends.quartus import QuartusBackend


def test_registry_dispatch_by_toolchain_field():
    b = get_backend({'toolchain': 'quartus', 'key': 'x'})
    assert isinstance(b, QuartusBackend)
    assert b.name == 'quartus'
    assert b.bitstream_ext == '.sof'
    assert b.project_subdir == 'quartus'
    assert b.convert_formats == ('svf', 'rbf')


def test_registry_dispatch_gowin():
    from mdl_fpga.backends.gowin_oss import GowinOssBackend
    b = get_backend({'toolchain': 'gowin-oss', 'key': 'tang-nano-20k'})
    assert isinstance(b, GowinOssBackend)
    assert b.bitstream_ext == '.fs'
    assert b.project_subdir == 'gowin'
    # 実ボード定義からの解決
    from mdl_fpga.board_db import load_board
    assert isinstance(get_backend(load_board('tang-nano-20k')),
                      GowinOssBackend)
    assert isinstance(get_backend(load_board('de0-cv')), QuartusBackend)


def test_family_fallback_for_legacy_boards():
    from mdl_fpga.backends.gowin_oss import GowinOssBackend
    # toolchain 欄が無い旧 board JSON は family から推定
    assert isinstance(get_backend({'family': 'Cyclone V'}), QuartusBackend)
    assert isinstance(get_backend({'family': 'MAX 10'}), QuartusBackend)
    assert isinstance(get_backend({'family': 'GW2A-18C'}), GowinOssBackend)


def test_unknown_toolchain_and_undecidable_family():
    with pytest.raises(KeyError, match='未知の toolchain'):
        get_backend({'toolchain': 'nope', 'key': 'x'})
    with pytest.raises(KeyError, match='決定できない'):
        get_backend({'key': 'x', 'family': 'Mystery9000'})


def test_fake_backend_registration(monkeypatch):
    """レジストリに1行追加するだけで build 側のディスパッチに載ることの証明。"""
    class FakeBackend:
        name = 'fake'
        bitstream_ext = '.bit'
        project_subdir = 'fake'
        convert_formats = ()

        def available(self, tc):
            return True, ''

    monkeypatch.setitem(backends._REGISTRY, 'fake', FakeBackend)
    b = get_backend({'toolchain': 'fake'})
    assert b.project_subdir == 'fake'
    assert 'fake' in backends.backend_names()


def test_quartus_backend_delegates(tmp_path):
    """QuartusBackend.write_project_files が旧 synth と同一出力を出す。"""
    from mdl_fpga.board_db import load_board
    from mdl_fpga.io_link import resolve_uart_link
    board = load_board('de0-cv')
    uart = resolve_uart_link(board, None).to_uart_dict()
    d1, d2 = tmp_path / 'a', tmp_path / 'b'
    QuartusBackend().write_project_files(
        d1, name='top', board=board, sources=['top.v'],
        clock_hz=50_000_000, groups=('clock', 'uart', 'led', 'switch'),
        uart=uart)
    import mdl_fpga.synth as synth
    synth.write_project_files(
        d2, name='top', board=board, sources=['top.v'],
        clock_hz=50_000_000, groups=('clock', 'uart', 'led', 'switch'),
        uart=uart)
    for f in ('top.qpf', 'top.qsf', 'top.sdc'):
        assert (d1 / f).read_text(encoding='utf-8') == \
               (d2 / f).read_text(encoding='utf-8')


def test_available_with_stub_toolchain():
    class NoQuartusTc:
        def quartus(self, exe):
            return None

    class YesQuartusTc:
        def quartus(self, exe):
            return 'C:/fake/quartus_map.exe'

    ok, msg = QuartusBackend().available(NoQuartusTc())
    assert not ok and 'Quartus' in msg
    ok, msg = QuartusBackend().available(YesQuartusTc())
    assert ok


def test_synth_shim_reexports():
    """既存コードの import 面が全て残っていること(互換シム)。"""
    import mdl_fpga.synth as synth
    for sym in ('FlowResult', 'write_project_files', 'run_flow',
                'convert_sof', 'parse_fit_summary', 'parse_fmax',
                'diagnose_failure', 'STAGES', '_cpf_args', '_write_summary'):
        assert hasattr(synth, sym), f'synth.{sym} が失われた(シム欠落)'
    assert synth.FlowResult is FlowResult


def test_flow_result_render_branches():
    r = FlowResult(ok=True, projdir='p', name='top',
                   summary={'alm': 10, 'alm_avail': 100,
                            'fmax_clk50_mhz': 150.0,
                            'setup_slack_clk50_ns': 1.2})
    assert 'ALM 10/100' in r.render()
    r2 = FlowResult(ok=True, projdir='p', name='top',
                    summary={'lut': 500, 'lut_avail': 20736,
                             'fmax_mhz': 60.0})
    assert 'LUT 500/20736' in r2.render()
    assert r2.bitstream is None


def test_protocol_conformance():
    assert isinstance(QuartusBackend(), SynthBackend)
