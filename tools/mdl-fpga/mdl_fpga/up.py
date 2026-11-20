"""一気通貫実行(mdl-fpga up): detect → advise → build → flash →(hw-check)。

各段は独立動詞の run_* をそのまま呼ぶ薄いオーケストレーション。
失敗した段で止め、「原因+次の一手」を段別に言い切り、以降は skipped。
board 未指定なら detect で自動決定するが、manifest の board は「意図の宣言」
として detect 結果より優先する(矛盾時は警告)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .toolchain import Toolchain, discover

STAGE_NAMES = ('detect', 'advise', 'build', 'flash', 'hw-check')


@dataclass
class StageOutcome:
    name: str
    ok: bool
    skipped: bool = False
    detail: str = ''
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'name': self.name, 'ok': self.ok, 'skipped': self.skipped,
                'detail': self.detail, 'data': self.data}


@dataclass
class UpResult:
    ok: bool
    stages: list[StageOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'ok': self.ok, 'stages': [s.to_dict() for s in self.stages]}

    def render(self) -> str:
        lines = [f'up: {"OK" if self.ok else "FAIL"}']
        for s in self.stages:
            mark = 'skip' if s.skipped else ('ok' if s.ok else 'FAIL')
            lines.append(f'  [{mark}] {s.name}'
                         + (f': {s.detail}' if s.detail else ''))
        return '\n'.join(lines)


def _skip_rest(res: UpResult, done: set) -> None:
    for name in STAGE_NAMES:
        if name not in done:
            res.stages.append(StageOutcome(name, ok=False, skipped=True,
                                           detail='前段の失敗によりスキップ'))


def run_up(model: str | Path, *, board: str | None = None,
           workdir: str | Path | None = None, flash: bool = True,
           hw_check: bool = False, tc: Toolchain | None = None,
           progress: Callable[[str], None] | None = None) -> UpResult:
    tc = tc or discover()
    res = UpResult(ok=False)
    say = progress or (lambda _s: None)
    done: set = set()

    from .manifest import is_manifest_file, load_manifest
    mf = load_manifest(model) if is_manifest_file(model) else None
    ir_model = mf.ir_path() if mf else Path(model)

    # 1) detect(board 未指定時のみ。JTAG を触るのはユーザーが up を打った時)
    chosen = board
    if chosen is not None:
        res.stages.append(StageOutcome('detect', ok=True, skipped=True,
                                       detail=f'--board {board} 指定のためスキップ'))
    else:
        say('detect: ボード検出中…')
        from .detect import run_detect
        det = run_detect(tc)
        best = det.best()
        if mf is not None:
            # manifest は意図の宣言 → detect 結果より優先(矛盾は警告)
            note = (f'検出={best.key} だが manifest の board={mf.board} を優先'
                    if best and best.key != mf.board
                    else f'manifest の board={mf.board} を使用')
            res.stages.append(StageOutcome('detect', ok=True, detail=note,
                                           data=det.to_dict()))
            chosen = mf.board
        elif best is not None:
            chosen = best.key
            res.stages.append(StageOutcome(
                'detect', ok=True,
                detail=f'{best.key}(確信度 {best.confidence:.1f})',
                data=det.to_dict()))
        else:
            reason = '候補が同率複数で絞れない' if det.candidates else '候補なし'
            # 未知ボード誘導(AUTOFLOW 工程2→3): チップが見えているなら
            # board-def fetch での定義生成へ案内する
            jtag = getattr(getattr(det, 'observation', None), 'jtag', None) or []
            hint = ''
            if jtag and not det.candidates:
                d = jtag[0]
                part = f' / {d.part_text}' if getattr(d, 'part_text', '') else ''
                hint = (f'。チップ自体は検出済み(IDCODE {d.idcode}{part})='
                        '未登録ボードの可能性。`board-def fetch <ボード名>` で'
                        '定義 draft を生成し、--board <draft の JSON パス> で'
                        'そのまま使える(レシピが無いボードは '
                        'boarddef/recipes/ に追加する)')
            res.stages.append(StageOutcome(
                'detect', ok=False,
                detail=f'ボードを特定できない({reason})。次の一手: '
                       '--board で明示指定するか、電源/USB(UB2)/doctor を確認'
                       + hint,
                data=det.to_dict()))
            done.add('detect')
            _skip_rest(res, done)
            return res
    done.add('detect')

    # ボード解決(key / board JSON パス / draft。AUTOFLOW 工程5)。
    # 以降の段はこの検証済み dict を前提に判断する(信頼性第一)
    from .board_db import resolve_board
    try:
        board_doc = resolve_board(chosen)
    except Exception as e:  # noqa: BLE001 - 段別の言い切りへ変換する
        res.stages.append(StageOutcome(
            'advise', ok=False,
            detail=f'ボード解決失敗({chosen}): {e}'))
        done.add('advise')
        _skip_rest(res, done)
        return res

    # 2) advise(EDA 不要の事前判定。載らないなら合成の時間を浪費しない)
    say('advise: 搭載可能性を事前判定…')
    from .advisor import run_advise
    adv = run_advise(model, board=chosen)
    if not adv.spatial_ok:
        res.stages.append(StageOutcome('advise', ok=False, detail=adv.render(),
                                       data=adv.to_dict()))
        done.add('advise')
        _skip_rest(res, done)
        return res
    res.stages.append(StageOutcome(
        'advise', ok=True,
        detail=f'搭載可能(gates={adv.gates}, board={chosen}'
               + ('' if adv.calibrated else '、未較正=保守判定') + ')',
        data=adv.to_dict()))
    done.add('advise')

    # 2.5) pin-validate(draft ボードのみ。実機未検証定義を合成前に機械検証。
    # 信頼性第一: FAIL は build へ進めない。ツール不在は skipped と言い切る)
    if board_doc.get('unverified'):
        say('pin-validate: draft ボードのピン実在を検証中…')
        from .pin_validate import run_pin_validate
        pv = run_pin_validate(board_doc, tc=tc)
        if pv.skipped:
            res.stages.append(StageOutcome(
                'pin-validate', ok=True, skipped=True,
                detail=f'検証ツール不在のためスキップ: {pv.detail}',
                data=pv.to_dict()))
        elif pv.ok:
            res.stages.append(StageOutcome(
                'pin-validate', ok=True,
                detail=f'{pv.n_pins} ピン受理({pv.backend}。draft={chosen})',
                data=pv.to_dict()))
        else:
            res.stages.append(StageOutcome(
                'pin-validate', ok=False,
                detail=f'draft のピン検証に失敗: {pv.detail}。'
                       'board-def fetch の生成元ソース/レシピを確認'
                       '(実在しないピンのまま合成へは進めない)',
                data=pv.to_dict()))
            _skip_rest(res, done)
            return res

    # 3) build(前段: ツールチェーン充足の事前確認。不足なら導入計画を言い切る)
    from .backends import get_backend
    from .eda_plan import plan_for_board
    try:
        be_ok, be_why = get_backend(board_doc).available(tc)
    except KeyError as e:
        be_ok, be_why = False, str(e)
    if not be_ok:
        plan = plan_for_board(board_doc, tc)
        res.stages.append(StageOutcome(
            'build', ok=False,
            detail=f'{be_why}\n{plan.render()}',
            data={'eda_plan': plan.to_dict()}))
        done.add('build')
        _skip_rest(res, done)
        return res
    say('build: HDL 生成+合成中…(fit は数分かかることがある)')
    from .build import run_build
    b = run_build(model, workdir=workdir, board=chosen, tc=tc)
    res.stages.append(StageOutcome('build', ok=b.ok,
                                   detail='' if b.ok else b.render(),
                                   data=b.to_dict()))
    done.add('build')
    if not b.ok or not b.sof:
        _skip_rest(res, done)
        return res

    # 4) flash
    if not flash:
        res.stages.append(StageOutcome('flash', ok=True, skipped=True,
                                       detail='--no-flash 指定'))
        done.add('flash')
    else:
        say('flash: JTAG 書込中…')
        from .flash import run_flash
        f = run_flash(b.sof, board=chosen, tc=tc)
        res.stages.append(StageOutcome('flash', ok=f.ok, detail=f.detail,
                                       data=f.to_dict()))
        done.add('flash')
        if not f.ok:
            _skip_rest(res, done)
            return res

    # 5) hw-check(オプション。UART ブリッジ未接続は skipped で言い切る)
    if not hw_check:
        res.stages.append(StageOutcome('hw-check', ok=True, skipped=True,
                                       detail='--hw-check 未指定'))
    else:
        say('hw-check: 実機 bit 一致検証中…')
        from .hw_check import run_hw_check
        baud = mf.effective_baud() if mf else 115200
        h = run_hw_check(ir_model, baud=baud, board=chosen)
        if not h.passed and h.port == '':
            res.stages.append(StageOutcome(
                'hw-check', ok=True, skipped=True,
                detail='UART ブリッジ未接続のためスキップ: '
                       + '; '.join(h.messages)))
        else:
            res.stages.append(StageOutcome(
                'hw-check', ok=h.passed,
                detail=f'{h.n_ok}/{h.n_vectors} bit一致(port={h.port})',
                data={'messages': h.messages}))

    res.ok = all(s.ok for s in res.stages if not s.skipped)
    return res
