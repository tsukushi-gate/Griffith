"""PORTABILITY.md 1-4: ツールチェーン探索の非 Windows 対応。

- oss-cad-suite / Quartus の実行体を拡張子なし(Linux/macOS 形)でも解決する
- Windows 形({name}.exe)が同居する場合はそちらを優先(既存挙動の回帰)
- verilator は verilator_bin(Windows 形)→ verilator(ラッパ)の順に探す
- 既知パス候補に非 Windows の場所が含まれる
"""

import shutil

import pytest

from mdl_fpga.toolchain import (
    _OSS_CAD_CANDIDATES, _QUARTUS_CANDIDATES, Toolchain)


@pytest.fixture
def no_path_fallback(monkeypatch):
    """PATH 由来の解決を無効化(oss_bin / quartus_bin 直下の解決だけを試験)。"""
    monkeypatch.setattr(shutil, 'which', lambda name, **kw: None)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')
    return path


def test_oss_exe_resolves_extensionless(tmp_path, no_path_fallback):
    root = tmp_path / 'oss-cad-suite'
    exe = _touch(root / 'bin' / 'iverilog')      # Linux/macOS 形
    tc = Toolchain(oss_root=root)
    assert tc.iverilog == str(exe)


def test_oss_exe_prefers_windows_exe(tmp_path, no_path_fallback):
    root = tmp_path / 'oss-cad-suite'
    win = _touch(root / 'bin' / 'iverilog.exe')
    _touch(root / 'bin' / 'iverilog')
    tc = Toolchain(oss_root=root)
    assert tc.iverilog == str(win)


def test_oss_exe_none_when_missing(tmp_path, no_path_fallback):
    tc = Toolchain(oss_root=tmp_path / 'oss-cad-suite')
    assert tc.iverilog is None


def test_verilator_wrapper_fallback(tmp_path, no_path_fallback):
    root = tmp_path / 'oss-cad-suite'
    wrapper = _touch(root / 'bin' / 'verilator')   # Linux/macOS はラッパのみ
    tc = Toolchain(oss_root=root)
    assert tc.verilator == str(wrapper)


def test_verilator_bin_preferred(tmp_path, no_path_fallback):
    root = tmp_path / 'oss-cad-suite'
    vbin = _touch(root / 'bin' / 'verilator_bin.exe')
    _touch(root / 'bin' / 'verilator')
    tc = Toolchain(oss_root=root)
    assert tc.verilator == str(vbin)


def test_quartus_resolves_extensionless(tmp_path, no_path_fallback):
    qbin = tmp_path / 'quartus' / 'bin'
    exe = _touch(qbin / 'quartus_map')             # Linux 形
    tc = Toolchain(quartus_bin=qbin)
    assert tc.quartus('quartus_map') == str(exe)
    assert tc.quartus('quartus_fit') is None


def test_quartus_prefers_windows_exe(tmp_path, no_path_fallback):
    qbin = tmp_path / 'quartus' / 'bin64'
    win = _touch(qbin / 'quartus_map.exe')
    _touch(qbin / 'quartus_map')
    tc = Toolchain(quartus_bin=qbin)
    assert tc.quartus('quartus_map') == str(win)


def test_known_candidates_cover_non_windows():
    assert '/opt/oss-cad-suite' in _OSS_CAD_CANDIDATES
    assert any(c.replace('\\', '/').endswith('quartus/bin')
               for c in _QUARTUS_CANDIDATES), 'Linux 用 quartus/bin 候補が必要'
    # 既存 Windows 候補の回帰
    assert r'C:\oss-cad-suite' in _OSS_CAD_CANDIDATES
    assert any('bin64' in c for c in _QUARTUS_CANDIDATES)
