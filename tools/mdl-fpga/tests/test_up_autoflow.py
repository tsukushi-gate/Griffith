"""AUTOFLOW 工程5(up への統合)の検証。

- board_db.resolve_board: key / board JSON パス / dict の共通解決
  (検証水準は load_board と同一=信頼性第一)
- advise/up が draft(boards/ 昇格前の board JSON)をパスで受ける
- up の draft 安全ゲート: unverified ボードは build 前に pin-validate、
  FAIL なら合成へ進めない
- up の未知ボード誘導(チップ検出済み→ board-def fetch 案内)
- up の build 前段ツール充足チェック(不足なら eda plan を言い切る)
"""

import json
from dataclasses import dataclass, field

import pytest
from conftest import build_parity4
from test_m6_up import (
    FakeAdvice, FakeBackend, FakeBest, FakeBuild, FakeDetect, FakeFlash)

from mdl_fpga.board_db import load_board, resolve_board
from mdl_fpga.up import run_up


def _draft_board_path(tmp_path, **over):
    doc = {
        'schema_version': 2, 'key': 'draftboard', 'name': 'Draft Board',
        'tier': 3, 'unverified': True,
        'toolchain': 'gowin-oss', 'pin_format': 'gowin',
        'device': 'GW-TEST', 'family': 'GW-F',
        'clock': {'port': 'CLK', 'pin': '4', 'hz': 27_000_000},
        'led': {'port': 'LED', 'pins': ['15', '16']},
        'resources': {'unit': 'lut4', 'lut4': 100, 'note': 'テスト値'},
        'limits': {},
    }
    doc.update(over)
    p = tmp_path / 'draftboard.json'
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    return p


def _ir(tmp_path):
    p = tmp_path / 'p4.ir.json'
    p.write_text(json.dumps(build_parity4()), encoding='utf-8')
    return p


def _stage(res, name):
    return next(s for s in res.stages if s.name == name)


# ------------------------------------------------------ resolve_board ---


def test_resolve_board_by_key_matches_load_board():
    assert resolve_board('tang-nano-20k') == load_board('tang-nano-20k')


def test_resolve_board_by_path_and_dict(tmp_path):
    p = _draft_board_path(tmp_path)
    doc = resolve_board(str(p))
    assert doc['key'] == 'draftboard' and doc['unverified'] is True
    assert resolve_board(doc) is doc          # dict は検証して素通し


def test_resolve_board_missing_path():
    with pytest.raises(FileNotFoundError, match='key 指定なら拡張子を外す'):
        resolve_board('no/such/file.json')


def test_resolve_board_rejects_wrong_schema(tmp_path):
    p = _draft_board_path(tmp_path, schema_version=1)
    with pytest.raises(ValueError, match='schema_version'):
        resolve_board(str(p))


def test_resolve_board_rejects_invalid_pins(tmp_path):
    # gowin フォーマットに intel ピン → 検証水準が落ちていないことの証拠
    p = _draft_board_path(tmp_path,
                          clock={'port': 'CLK', 'pin': 'PIN_M9',
                                 'hz': 27_000_000})
    with pytest.raises(Exception, match='pin_format'):
        resolve_board(str(p))


def test_advise_accepts_draft_path(tmp_path):
    from mdl_fpga.advisor import run_advise
    adv = run_advise(_ir(tmp_path), board=str(_draft_board_path(tmp_path)))
    assert adv.unit == 'lut4'
    assert adv.calibrated is False            # draft は未較正=保守判定
    assert adv.spatial_ok                     # parity4(3ゲート)は載る


# ------------------------------------------------------- up 統合 ---


@dataclass
class FakePinValidate:
    ok: bool = True
    skipped: bool = False
    backend: str = 'gowin-oss'
    n_pins: int = 3
    detail: str = ''

    def to_dict(self):
        return {'ok': self.ok, 'skipped': self.skipped}


def _patch_core(monkeypatch, *, pv=None, build_calls=None, backend=None):
    monkeypatch.setattr('mdl_fpga.detect.run_detect',
                        lambda tc=None, probes=None: FakeDetect(
                            best_value=FakeBest()))
    monkeypatch.setattr('mdl_fpga.advisor.run_advise',
                        lambda model, board=None: FakeAdvice())
    monkeypatch.setattr('mdl_fpga.backends.get_backend',
                        lambda board: backend or FakeBackend())
    monkeypatch.setattr('mdl_fpga.pin_validate.run_pin_validate',
                        lambda board, tc=None, **kw: pv or FakePinValidate())

    def fake_build(model, **kw):
        if build_calls is not None:
            build_calls.append(kw.get('board'))
        return FakeBuild()
    monkeypatch.setattr('mdl_fpga.build.run_build', fake_build)
    monkeypatch.setattr('mdl_fpga.flash.run_flash',
                        lambda sof, **kw: FakeFlash())


def test_up_draft_board_runs_pin_validate_gate(tmp_path, monkeypatch):
    _patch_core(monkeypatch)
    draft = _draft_board_path(tmp_path)
    res = run_up(_ir(tmp_path), board=str(draft))
    assert res.ok is True
    names = [s.name for s in res.stages]
    assert names == ['detect', 'advise', 'pin-validate', 'build', 'flash',
                     'hw-check']
    assert '3 ピン受理' in _stage(res, 'pin-validate').detail


def test_up_draft_pin_validate_fail_stops_before_build(tmp_path, monkeypatch):
    calls = []
    _patch_core(monkeypatch, build_calls=calls,
                pv=FakePinValidate(ok=False, detail='nextpnr がピン割当を拒否'))
    res = run_up(_ir(tmp_path), board=str(_draft_board_path(tmp_path)))
    assert res.ok is False
    assert _stage(res, 'pin-validate').ok is False
    assert '合成へは進めない' in _stage(res, 'pin-validate').detail
    assert _stage(res, 'build').skipped is True
    assert calls == []                        # build は呼ばれない


def test_up_draft_pin_validate_skipped_tools_missing(tmp_path, monkeypatch):
    _patch_core(monkeypatch, pv=FakePinValidate(ok=False, skipped=True,
                                                detail='yosys が見つからない'))
    res = run_up(_ir(tmp_path), board=str(_draft_board_path(tmp_path)))
    pv = _stage(res, 'pin-validate')
    assert pv.skipped is True and 'スキップ' in pv.detail
    assert res.ok is True                     # 既知ボード経路と同じ扱いで続行


def test_up_verified_board_has_no_pin_validate_stage(tmp_path, monkeypatch):
    _patch_core(monkeypatch)
    res = run_up(_ir(tmp_path), board='tang-nano-20k')   # Tier1(unverified 無し)
    assert res.ok is True
    assert 'pin-validate' not in [s.name for s in res.stages]


def test_up_unknown_board_guides_to_board_def_fetch(tmp_path, monkeypatch):
    @dataclass
    class Jtag:
        idcode: str = '0x0000081b'
        part_text: str = 'GW2A'

    @dataclass
    class Obs:
        jtag: list = field(default_factory=lambda: [Jtag()])

    det = FakeDetect(best_value=None)
    det.observation = Obs()
    _patch_core(monkeypatch)
    monkeypatch.setattr('mdl_fpga.detect.run_detect',
                        lambda tc=None, probes=None: det)
    res = run_up(_ir(tmp_path))
    d = _stage(res, 'detect')
    assert d.ok is False
    assert 'board-def fetch' in d.detail
    assert '0x0000081b' in d.detail           # チップ検出済みの情報を言い切る


def test_up_board_resolution_failure_is_explicit(tmp_path, monkeypatch):
    _patch_core(monkeypatch)
    res = run_up(_ir(tmp_path), board='no-such-board')
    a = _stage(res, 'advise')
    assert a.ok is False and 'ボード解決失敗' in a.detail
    assert _stage(res, 'build').skipped is True


def test_up_missing_tools_shows_eda_plan(tmp_path, monkeypatch):
    class Unavailable:
        name = 'gowin-oss'

        def available(self, tc):
            return False, 'gowin-oss 合成に必要なツールが不足: yosys'

    calls = []
    _patch_core(monkeypatch, build_calls=calls, backend=Unavailable())
    res = run_up(_ir(tmp_path), board='tang-nano-20k')
    b = _stage(res, 'build')
    assert b.ok is False
    assert 'EDA 導入計画' in b.detail         # eda plan の内容を言い切る
    assert calls == []                        # 合成は実行されない
