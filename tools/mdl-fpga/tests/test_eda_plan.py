"""eda plan / eda install(AUTOFLOW 工程6)の検証。

- family→ベンダ/バックエンド判定と導入計画(gowin=自動導入可、
  quartus=本体は人+qdz 版一致の提示、vivado=Modules 行の提示)
- oss-cad-suite 自動導入: 資産 URL 規約・プラットフォーム解決・
  展開/検証/local.toml 更新(ネットワークはテスト注入で遮断)
"""

import sys
import tarfile
from pathlib import Path

import pytest

from mdl_fpga.board_db import load_board
from mdl_fpga.eda_plan import (
    asset_url, install_oss_cad_suite, latest_tag, plan_for_board,
    resolve_platform, update_local_toml)
from mdl_fpga.toolchain import Toolchain


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('', encoding='utf-8')
    return p


# ------------------------------------------------------------- plan ---


def test_plan_gowin_missing_tools_is_auto_installable(tmp_path, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda n, **kw: None)
    tc = Toolchain(oss_root=tmp_path / 'nothing')
    plan = plan_for_board(load_board('tang-nano-20k'), tc)
    assert plan.backend == 'gowin-oss' and plan.vendor == 'gowin'
    assert not plan.ready
    assert plan.auto_installable
    assert any('eda install oss-cad-suite' in a for a in plan.actions)


def test_plan_gowin_ready(tmp_path, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda n, **kw: None)
    root = tmp_path / 'oss-cad-suite'
    for n in ('yosys', 'nextpnr-himbaechel', 'gowin_pack', 'openFPGALoader'):
        _touch(root / 'bin' / f'{n}.exe')
    plan = plan_for_board(load_board('tang-nano-20k'),
                          Toolchain(oss_root=root))
    assert plan.ready and plan.missing == [] and plan.actions == []


def test_plan_quartus_absent_shows_qdz_and_unattended(monkeypatch):
    monkeypatch.setattr('shutil.which', lambda n, **kw: None)
    plan = plan_for_board(load_board('de0-cv'), Toolchain())
    assert plan.backend == 'quartus' and plan.vendor == 'intel'
    assert not plan.ready and not plan.auto_installable
    joined = ' '.join(plan.actions)
    assert 'cyclonev' in joined            # family→qdz コンポーネント名
    assert '--mode unattended' in joined and '--accept_eula 1' in joined


def test_plan_quartus_installed_version_matched_qdz(tmp_path, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda n, **kw: None)
    qbin = tmp_path / 'intelFPGA_lite' / '23.1std' / 'quartus' / 'bin64'
    _touch(qbin / 'quartus_map.exe')       # fit 等は不足の状態
    plan = plan_for_board(load_board('de10-lite'), Toolchain(quartus_bin=qbin))
    assert 'quartus_fit' in plan.missing
    joined = ' '.join(plan.actions)
    assert 'max10-23.1std.qdz' in joined   # 版一致の qdz 名を提示
    assert 'Install Devices' in joined


def test_plan_vivado_modules_line():
    plan = plan_for_board({'key': 'x', 'family': 'Artix-7'}, Toolchain())
    assert plan.vendor == 'xilinx' and plan.backend is None
    joined = ' '.join(plan.actions)
    assert 'Artix-7:1' in joined and 'xsetup' in joined


def test_plan_unknown_family():
    plan = plan_for_board({'key': 'x', 'family': 'Unknown9000'}, Toolchain())
    assert not plan.ready
    assert any('toolchain' in a for a in plan.actions)


# ---------------------------------------------------------- install ---


def test_resolve_platform_shapes(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr('platform.machine', lambda: 'AMD64')
    assert resolve_platform() == ('windows', 'x64', '.exe')
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr('platform.machine', lambda: 'aarch64')
    assert resolve_platform() == ('linux', 'arm64', '.tgz')
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr('platform.machine', lambda: 'arm64')
    assert resolve_platform() == ('darwin', 'arm64', '.tgz')


def test_asset_url_convention():
    url = asset_url('2026-07-20', 'linux', 'x64', '.tgz')
    assert url == ('https://github.com/YosysHQ/oss-cad-suite-build/releases/'
                   'download/2026-07-20/oss-cad-suite-linux-x64-20260720.tgz')


def test_latest_tag_injected():
    assert latest_tag(json_fetch=lambda url: {'tag_name': '2026-01-02'}) \
        == '2026-01-02'


def _make_fake_tgz(path: Path) -> None:
    src = path.parent / 'stage' / 'oss-cad-suite' / 'bin'
    _touch(src / 'yosys')
    with tarfile.open(path, 'w:gz') as tf:
        tf.add(src.parent, arcname='oss-cad-suite')


def test_install_oss_cad_suite_tgz(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr('platform.machine', lambda: 'x86_64')
    fake = tmp_path / 'payload.tgz'
    _make_fake_tgz(fake)

    def fetch_bytes(url, dst):
        assert 'oss-cad-suite-linux-x64-20260102.tgz' in url
        dst.write_bytes(fake.read_bytes())

    dest = tmp_path / 'home' / 'oss-cad-suite'
    msgs = []
    got = install_oss_cad_suite(
        dest, json_fetch=lambda url: {'tag_name': '2026-01-02'},
        fetch_bytes=fetch_bytes, progress=msgs.append)
    assert got == dest and (dest / 'bin' / 'yosys').exists()
    # アーカイブは削除せず残す
    assert list(dest.parent.glob('oss-cad-suite-linux-*.tgz'))
    with pytest.raises(FileExistsError, match='上書きしない'):
        install_oss_cad_suite(dest, tag='2026-01-02',
                              fetch_bytes=fetch_bytes, progress=msgs.append)


def test_install_rejects_wrong_dirname(tmp_path):
    with pytest.raises(ValueError, match='oss-cad-suite'):
        install_oss_cad_suite(tmp_path / 'tools', tag='2026-01-02')


def test_update_local_toml_preserves_other_lines(tmp_path):
    toml = tmp_path / 'local.toml'
    toml.write_text('quartus_bin = "C:/q/bin64"\noss_cad_root = "C:/old"\n',
                    encoding='utf-8')
    update_local_toml(Path('C:/new/oss-cad-suite'), toml)
    text = toml.read_text(encoding='utf-8')
    assert 'quartus_bin = "C:/q/bin64"' in text
    assert 'oss_cad_root = "C:/new/oss-cad-suite"' in text
    assert 'C:/old' not in text


def test_update_local_toml_creates_file(tmp_path):
    toml = tmp_path / 'local.toml'
    update_local_toml(Path('/opt/oss-cad-suite'), toml)
    assert 'oss_cad_root = "/opt/oss-cad-suite"' in \
        toml.read_text(encoding='utf-8')


def test_cli_eda_plan_smoke(capsys):
    from mdl_fpga.cli import main
    rc = main(['eda', 'plan', '--board', 'tang-nano-20k'])
    out = capsys.readouterr().out
    assert 'EDA 導入計画' in out
    assert rc in (0, 1)     # 導入状況に依存(表示されること自体を確認)
