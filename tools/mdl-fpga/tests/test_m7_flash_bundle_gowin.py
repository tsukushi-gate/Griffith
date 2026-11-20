"""M7: flash/bundle/hw-check/detect の Gowin/BL616 対応(ツール・実機不要)。"""

import json
import zipfile
from pathlib import Path

import pytest

from mdl_fpga.board_db import load_board
from mdl_fpga.bundle import run_bundle
from mdl_fpga.detect import _probe_argsets
from mdl_fpga.flash import _choose_via, _ofl_args
from mdl_fpga.hw_check import serial_signatures
from mdl_fpga.toolchain import Toolchain


@pytest.fixture(autouse=True)
def _isolate_flasher_default(monkeypatch):
    """開発機の flasher 既定(env/local.toml)に依存せず非同梱 README を検証
    する(同梱時の README は test_standalone.py が担当)。"""
    monkeypatch.delenv('MDL_FPGA_FLASHER_PACK', raising=False)
    monkeypatch.setattr('mdl_fpga.bundle._local_flasher_pack', lambda: None)


class _TcWithOfl(Toolchain):
    @property
    def openfpgaloader(self):  # type: ignore[override]
        return 'C:/fake/openFPGALoader.exe'


def test_choose_via_fs_routes_to_openfpgaloader():
    via, err = _choose_via(Path('top.fs'), 'auto', _TcWithOfl())
    assert (via, err) == ('openfpgaloader', None)
    # 明示指定でも同様
    via, err = _choose_via(Path('top.fs'), 'openfpgaloader', _TcWithOfl())
    assert (via, err) == ('openfpgaloader', None)
    # .fs を quartus 経路には流せない
    via, err = _choose_via(Path('top.fs'), 'quartus', Toolchain())
    assert via is None and 'quartus_pgm' in err or '.fs' in err


def test_ofl_args_board_derivation():
    tn = load_board('tang-nano-20k')
    p = Path('top.fs')
    # ofl_board があれば -b(BL616 内蔵プログラマ)
    assert _ofl_args(tn, 'usb-blaster', p) == ['-b', 'tangnano20k', 'top.fs']
    # ofl_board 無し・cable ありのボードは -c <board cable>
    de = load_board('de0-cv')
    assert _ofl_args(de, 'ignored', p) == ['-c', 'usb-blaster', 'top.fs']
    # board 無指定は引数の ofl_cable
    assert _ofl_args(None, 'my-cable', p) == ['-c', 'my-cable', 'top.fs']


def test_detect_probe_argsets_include_tangnano():
    sets = dict(_probe_argsets())
    assert sets['-b tangnano20k'] == ['-b', 'tangnano20k', '--detect']
    assert sets['-c usb-blaster'] == ['-c', 'usb-blaster', '--detect']
    assert sets['direct'] == ['--detect']
    # -b 候補が -c 候補より先(内蔵プログラマ優先)に並ぶこと
    labels = [lb for lb, _ in _probe_argsets()]
    assert labels.index('-b tangnano20k') < labels.index('-c ft2232')
    assert labels[-1] == 'direct'


def test_serial_signatures_order():
    tn = load_board('tang-nano-20k')
    sigs = serial_signatures(tn)
    # 現状 tang の serial.usb は未採取=空 → 既定 CP210x のみ
    assert sigs[-1]['vid'] == 0x10C4 and sigs[-1]['pid'] == 0xEA60
    # 採取後の形をシミュレート(board 署名が先頭に来る)
    tn2 = dict(tn)
    tn2['serial'] = {'usb': [{'vid': '0x1234', 'pid': '0x5678',
                              'desc_regex': 'JTAG.*Serial'}]}
    sigs2 = serial_signatures(tn2)
    assert sigs2[0]['vid'] == '0x1234'
    assert len(sigs2) == 2


def _fake_gowin_workdir(tmp_path: Path) -> Path:
    wd = tmp_path / 'wd'
    (wd / 'gowin').mkdir(parents=True)
    (wd / 'gowin' / 'top.fs').write_bytes(b'\x00fake-fs')
    (wd / 'gowin' / 'flow_summary.json').write_text(
        json.dumps({'name': 'top', 'ok': True, 'lut': 576}), encoding='utf-8')
    (wd / 'presim').mkdir()
    (wd / 'presim' / 'inputs_padded.hex').write_text('00\n', encoding='utf-8')
    return wd


def test_run_bundle_gowin(tmp_path):
    wd = _fake_gowin_workdir(tmp_path)
    res = run_bundle(wd, board='tang-nano-20k', tc=Toolchain())
    assert res.ok, res.messages
    assert 'bitstream/top.fs' in res.contents
    assert 'flow_summary.json' in res.contents
    assert 'board/tang-nano-20k.json' in res.contents
    assert 'golden/inputs_padded.hex' in res.contents
    # svf/rbf は生成されない(変換不要)
    assert not any(c.endswith('.svf') or c.endswith('.rbf')
                   for c in res.contents)
    assert any('変換は gowin-oss では不要' in m for m in res.messages)
    with zipfile.ZipFile(res.zip_path) as zf:
        readme = zf.read('README.md').decode('utf-8')
    assert 'openFPGALoader.exe -b tangnano20k bitstream/top.fs' in readme
    assert '配線不要' in readme          # BL616 内蔵 UART → 結線不要の案内
    assert 'ブリッジの TXD' not in readme
    assert 'quartus_pgm' not in readme   # Quartus への言及なし


def test_run_bundle_gowin_missing_fs(tmp_path):
    res = run_bundle(tmp_path, board='tang-nano-20k', tc=Toolchain())
    assert res.ok is False
    assert any('.fs が見つからない' in m for m in res.messages)


def test_run_bundle_quartus_message_unchanged(tmp_path):
    # 既存挙動の回帰(de0-cv 既定): .sof が見つからない文言
    res = run_bundle(tmp_path, tc=Toolchain())
    assert res.ok is False
    assert any('.sof が見つからない' in m for m in res.messages)
