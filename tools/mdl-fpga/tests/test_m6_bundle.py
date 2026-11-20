"""M6: bundle 動詞+flash --via 拡張+convert_sof の検証(実機/Quartus 不要分)。

実変換(quartus_cpf)と実書込は実機セッションで検証する(requires_hw 相当は
検証手順に記載)。ここでは引数組立の純関数と zip 梱包/README 生成を固定する。
"""

import zipfile
from pathlib import Path

import pytest

from mdl_fpga.bundle import extract_wiring_note, run_bundle
from mdl_fpga.flash import _choose_via, run_flash
from mdl_fpga.synth import _cpf_args, convert_sof
from mdl_fpga.toolchain import Toolchain


@pytest.fixture(autouse=True)
def _isolate_flasher_default(monkeypatch):
    """開発機の flasher 既定(env/local.toml)に依存せず bundle 内容を検証する。"""
    monkeypatch.delenv('MDL_FPGA_FLASHER_PACK', raising=False)
    monkeypatch.setattr('mdl_fpga.bundle._local_flasher_pack', lambda: None)

EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'


def test_cpf_args():
    args = _cpf_args('rbf', Path('a.sof'), Path('a.rbf'))
    assert '--option=bitstream_compression=off' in args   # JTAG 展開不可のため必須
    args = _cpf_args('svf', Path('a.sof'), Path('a.svf'))
    assert args[:6] == ['-c', '-q', '12.0MHz', '-g', '3.3', '-n']
    with pytest.raises(ValueError, match='未対応'):
        _cpf_args('pof', Path('a.sof'), Path('a.pof'))


def test_convert_sof_without_cpf(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    sof = tmp_path / 'x.sof'
    sof.write_bytes(b'\x00fake')
    with pytest.raises(FileNotFoundError, match='quartus_cpf'):
        convert_sof(sof, 'svf', tc=Toolchain())


def test_choose_via_matrix(tmp_path, monkeypatch):
    # 他テスト(viewer 系)がプロセス PATH に oss-cad を前置するため、
    # PATH を空にして「ツール無し」を確定させる(shutil.which 汚染対策)
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    tc = Toolchain()   # ツール無し環境を模擬
    sof, svf = Path('x.sof'), Path('x.svf')
    # auto は拡張子で経路を選ぶ(ツール不在はエラー文で表面化)
    via, err = _choose_via(sof, 'auto', tc)
    assert via is None and 'quartus_pgm' in err
    via, err = _choose_via(svf, 'auto', tc)
    assert via is None and 'openFPGALoader' in err
    # 経路と拡張子の矛盾は言い切る
    via, err = _choose_via(svf, 'quartus', tc)
    assert via is None and '.svf' in err and 'openfpgaloader' in err
    via, err = _choose_via(sof, 'openfpgaloader', tc)
    assert via is None and 'bundle' in err
    via, err = _choose_via(sof, 'nonsense', tc)
    assert via is None and '未知' in err
    # ツールがあれば通る(偽 exe を置いて解決させる)
    qbin = tmp_path / 'qbin'
    qbin.mkdir()
    (qbin / 'quartus_pgm.exe').write_bytes(b'')
    tc2 = Toolchain(quartus_bin=qbin)
    assert _choose_via(sof, 'auto', tc2) == ('quartus', None)


def test_run_flash_missing_file():
    res = run_flash('no_such.sof', tc=Toolchain())
    assert res.ok is False and '存在しない' in res.detail
    assert 'ok' in res.to_dict() and res.to_dict()['bitstream'].endswith('no_such.sof')


def _fake_build_workdir(tmp_path: Path) -> Path:
    wd = tmp_path / 'wd'
    out = wd / 'quartus' / 'output_files'
    out.mkdir(parents=True)
    (out / 'top.sof').write_bytes(b'\x00fake-sof')
    (wd / 'quartus' / 'flow_summary.json').write_text(
        '{"ok": true}', encoding='utf-8')
    presim = wd / 'presim'
    presim.mkdir()
    (presim / 'inputs_padded.hex').write_text('00\n', encoding='utf-8')
    (presim / 'expected_padded.hex').write_text('00\n', encoding='utf-8')
    return wd


def test_run_bundle_without_cpf(tmp_path, monkeypatch):
    """cpf 無し環境: sof のみ+WARN で ok=True。README に配線表と両OS手順。"""
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    wd = _fake_build_workdir(tmp_path)
    res = run_bundle(wd, manifest=EXAMPLES / 'full_adder.manifest.json',
                     tc=Toolchain())
    assert res.ok is True
    assert any('WARN' in m for m in res.messages)
    assert any('Programmer' in m for m in res.messages)   # svf/rbf 無し注意
    names = set(res.contents)
    assert {'bitstream/top.sof', 'board/de0-cv.json', 'manifest.json',
            'flow_summary.json', 'golden/inputs_padded.hex',
            'golden/expected_padded.hex', 'README.md'} <= names
    with zipfile.ZipFile(res.zip_path) as zf:
        assert set(zf.namelist()) == names
        readme = zf.read('README.md').decode('utf-8')
    # 配線表(既定リンク=GPIO_0[0]/[1] = JP1 物理 1/2)
    assert 'PIN_N16(コネクタ物理ピン 1)' in readme
    assert 'PIN_B16(コネクタ物理ピン 2)' in readme
    assert 'baud: 115200' in readme
    # svf/rbf 無し時: 書込(macOS)節は brew 手順ではなく「書込不可」の言い切り
    # (app 節の flasher 導入誘導に brew が載るのは正当=STANDALONE.md)
    assert 'macOS では書込不可' in readme
    assert 'SRAM 揮発' in readme


def test_run_bundle_no_sof(tmp_path):
    res = run_bundle(tmp_path, tc=Toolchain())
    assert res.ok is False
    assert any('.sof が見つからない' in m for m in res.messages)


def test_run_bundle_formats_empty(tmp_path):
    """formats=() は変換なし(cpf 不要)で WARN も出ない。"""
    wd = _fake_build_workdir(tmp_path)
    res = run_bundle(wd, formats=(), tc=Toolchain())
    assert res.ok is True
    assert not any('WARN' in m for m in res.messages)


def test_extract_wiring_note():
    note = extract_wiring_note('de0-cv')
    assert 'PIN_N16' in note and 'PIN_B16' in note and '115200' in note
    note10 = extract_wiring_note('de10-lite')
    assert 'PIN_V10' in note10
