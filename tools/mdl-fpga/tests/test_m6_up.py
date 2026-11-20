"""M6: up 動詞(一気通貫オーケストレーション)の検証。

各段の run_* は monkeypatch で差し替え(ハード/Quartus 不要)。
up.py は遅延 import(from .detect import run_detect 等)なので、
モジュール属性の差し替えがそのまま効く。
"""

import json
from dataclasses import dataclass, field

from conftest import build_parity4
from mdl_fpga.up import run_up


@dataclass
class FakeBest:
    key: str = 'de0-cv'
    confidence: float = 0.6


@dataclass
class FakeDetect:
    best_value: FakeBest | None = None
    candidates: list = field(default_factory=list)

    def best(self):
        return self.best_value

    def to_dict(self):
        return {'candidates': self.candidates}


@dataclass
class FakeAdvice:
    spatial_ok: bool = True
    overlay_ok: bool = True
    gates: int = 3
    calibrated: bool = True

    def to_dict(self):
        return {'gates': self.gates, 'spatial_ok': self.spatial_ok}

    def render(self):
        return 'advisor: 搭載不可' if not self.spatial_ok else 'ok'


@dataclass
class FakeBuild:
    ok: bool = True
    sof: str | None = 'top.sof'
    workdir: str = 'wd'
    messages: list = field(default_factory=list)

    def to_dict(self):
        return {'ok': self.ok, 'sof': self.sof}

    def render(self):
        return 'build: FAIL(理由X)'


@dataclass
class FakeFlash:
    ok: bool = True
    detail: str = '書込完了'

    def to_dict(self):
        return {'ok': self.ok}


class FakeBackend:
    """build 前段のツール充足チェックを常に満たす(ローカル EDA 有無に非依存)。"""

    name = 'fake'

    def available(self, tc):
        return True, ''


def _patch_all(monkeypatch, *, detect=None, advice=None, build=None,
               flash=None, advise_calls=None):
    monkeypatch.setattr('mdl_fpga.detect.run_detect',
                        lambda tc=None, probes=None: detect or FakeDetect(
                            best_value=FakeBest()))
    monkeypatch.setattr('mdl_fpga.backends.get_backend',
                        lambda board: FakeBackend())

    def fake_advise(model, board=None):
        if advise_calls is not None:
            advise_calls.append(board)
        return advice or FakeAdvice()
    monkeypatch.setattr('mdl_fpga.advisor.run_advise', fake_advise)
    monkeypatch.setattr('mdl_fpga.build.run_build',
                        lambda model, **kw: build or FakeBuild())
    monkeypatch.setattr('mdl_fpga.flash.run_flash',
                        lambda sof, **kw: flash or FakeFlash())


def _ir(tmp_path):
    p = tmp_path / 'p4.ir.json'
    p.write_text(json.dumps(build_parity4()), encoding='utf-8')
    return p


def _stage(res, name):
    return next(s for s in res.stages if s.name == name)


def test_all_success_detect_autoselect(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    res = run_up(_ir(tmp_path))
    assert res.ok is True
    assert [s.name for s in res.stages] == \
        ['detect', 'advise', 'build', 'flash', 'hw-check']
    assert 'de0-cv' in _stage(res, 'detect').detail
    assert _stage(res, 'hw-check').skipped is True   # --hw-check 未指定
    json.dumps(res.to_dict(), ensure_ascii=False)    # JSON 化可能


def test_board_explicit_skips_detect(tmp_path, monkeypatch):
    calls = []
    _patch_all(monkeypatch, advise_calls=calls)
    res = run_up(_ir(tmp_path), board='de10-lite')
    assert _stage(res, 'detect').skipped is True
    assert calls == ['de10-lite']


def test_manifest_board_wins_over_detect(tmp_path, monkeypatch):
    calls = []
    _patch_all(monkeypatch,
               detect=FakeDetect(best_value=FakeBest(key='de10-lite')),
               advise_calls=calls)
    ir = _ir(tmp_path)
    mf = tmp_path / 'p4.manifest.json'
    mf.write_text(json.dumps({'manifest_version': '0.2',
                              'ir': ir.name, 'board': 'de0-cv'}),
                  encoding='utf-8')
    res = run_up(mf)
    assert '優先' in _stage(res, 'detect').detail
    assert calls == ['de0-cv']          # manifest の board で判定される


def test_detect_failure_stops_pipeline(tmp_path, monkeypatch):
    _patch_all(monkeypatch, detect=FakeDetect(best_value=None))
    res = run_up(_ir(tmp_path))
    assert res.ok is False
    assert _stage(res, 'detect').ok is False
    assert '--board' in _stage(res, 'detect').detail
    for name in ('advise', 'build', 'flash', 'hw-check'):
        assert _stage(res, name).skipped is True


def test_advise_ng_stops_before_build(tmp_path, monkeypatch):
    _patch_all(monkeypatch, advice=FakeAdvice(spatial_ok=False))
    res = run_up(_ir(tmp_path), board='de0-cv')
    assert res.ok is False
    assert _stage(res, 'advise').ok is False
    assert _stage(res, 'build').skipped is True


def test_build_failure_stops(tmp_path, monkeypatch):
    _patch_all(monkeypatch, build=FakeBuild(ok=False, sof=None))
    res = run_up(_ir(tmp_path), board='de0-cv')
    assert res.ok is False
    assert _stage(res, 'build').ok is False
    assert _stage(res, 'flash').skipped is True


def test_flash_failure_skips_hwcheck(tmp_path, monkeypatch):
    _patch_all(monkeypatch, flash=FakeFlash(ok=False, detail='書込失敗'))
    res = run_up(_ir(tmp_path), board='de0-cv', hw_check=True)
    assert res.ok is False
    assert _stage(res, 'flash').ok is False
    assert _stage(res, 'hw-check').skipped is True


def test_no_flash_option(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    res = run_up(_ir(tmp_path), board='de0-cv', flash=False)
    assert res.ok is True
    assert _stage(res, 'flash').skipped is True
