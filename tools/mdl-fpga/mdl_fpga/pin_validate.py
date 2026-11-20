"""pin-validate: board JSON のピン実在・適法性検証(AUTOFLOW 工程4)。

EDA/nextpnr は「ボードの配線」を知らないが「チップ」は完全に知っている。
これを利用し、board-def fetch の draft や手転記の board JSON に対して
stub トップ+全ピン割当を実ツールに食わせ、実在しないピン・I/O 不可ピン・
規格違反を機械検出する(情報源ではなく検証者としての EDA)。

- gowin-oss: yosys(synth_gowin)→ nextpnr-himbaechel(cst を食わせて配置)
- quartus:   quartus_map → quartus_fit(--quick で map のみ)

ツール不在時は skipped として報告する(検証できていないことを OK と言わない)。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backends import get_backend
from .board_db import resolve_board
from .toolchain import TOOL_DIR, Toolchain, discover

_ERROR_RE = re.compile(r'^.*\bERROR\b.*$', re.M | re.I)


# ------------------------------------------------------------ stub 生成 ---


def collect_pins(board: dict) -> tuple[list[tuple[str, str, str]], list[str]]:
    """board JSON の全ピンを (ポート名, ピン, 方向) に列挙する。

    戻り値: (エントリ列, 重複で除外したラベル列)。同一ピンの二重割当は
    ツール側の偽エラーになるため先勝ちで除外し、報告に残す。
    """
    entries: list[tuple[str, str, str]] = []
    dups: list[str] = []
    seen: dict[str, str] = {}

    def _add(label: str, pin: str, direction: str) -> None:
        name = 'pv_' + re.sub(r'\W', '_', label)
        if pin in seen:
            dups.append(f'{label}(pin {pin} は {seen[pin]} と重複)')
            return
        seen[pin] = label
        entries.append((name, pin, direction))

    _add('clock', board['clock']['pin'], 'input')
    for role, direction in (('led', 'output'), ('switch', 'input'),
                            ('button', 'input')):
        for i, pin in enumerate((board.get(role) or {}).get('pins', [])):
            _add(f'{role}[{i}]', pin, direction)
    for digit in (board.get('hex') or {}).get('digits', []):
        for i, pin in enumerate(digit['pins']):
            _add(f'{digit["port"]}[{i}]', pin, 'output')
    for h in board.get('headers', []):
        for k in sorted((h.get('io') or {}), key=int):
            _add(f'{h["name"]}[{k}]', h['io'][k]['pin'], 'input')
    return entries, dups


def stub_verilog(entries: list[tuple[str, str, str]], name: str = 'pinval') -> str:
    ins = [n for n, _p, d in entries if d == 'input']
    outs = [n for n, _p, d in entries if d == 'output']
    ports = [f'    input {n}' for n in ins] + [f'    output {n}' for n in outs]
    agg = ' ^ '.join(ins) if ins else "1'b0"
    lines = ['// mdl-fpga pin-validate stub(全ピンを生かす XOR 束)',
             f'module {name}(',
             ',\n'.join(ports),
             ');',
             f'    wire pv_agg = {agg};']
    lines += [f'    assign {n} = pv_agg;' for n in outs]
    lines.append('endmodule')
    return '\n'.join(lines) + '\n'


def stub_cst(entries: list[tuple[str, str, str]], board: dict) -> str:
    io_type = board.get('io_standard')
    lines = ['// mdl-fpga pin-validate 自動生成 CST']
    for name, pin, _d in entries:
        lines.append(f'IO_LOC "{name}" {pin};')
        if io_type:
            lines.append(f'IO_PORT "{name}" IO_TYPE={io_type};')
    return '\n'.join(lines) + '\n'


def stub_qsf(entries: list[tuple[str, str, str]], board: dict,
             name: str = 'pinval') -> str:
    lines = [
        '# mdl-fpga pin-validate 自動生成 QSF',
        f'set_global_assignment -name FAMILY "{board["family"]}"',
        f'set_global_assignment -name DEVICE {board["device"]}',
        f'set_global_assignment -name TOP_LEVEL_ENTITY {name}',
        'set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files',
        f'set_global_assignment -name VERILOG_FILE {name}.v',
    ]
    for pname, pin, _d in entries:
        lines.append(f'set_location_assignment {pin} -to {pname}')
    io_std = board.get('io_standard')
    if io_std:
        for pname, _pin, _d in entries:
            lines.append(
                f'set_instance_assignment -name IO_STANDARD "{io_std}" '
                f'-to {pname}')
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------- 実行層 ---


@dataclass
class PinValidateResult:
    ok: bool
    skipped: bool = False
    backend: str = ''
    board_key: str = ''
    workdir: str = ''
    n_pins: int = 0
    duplicates: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    detail: str = ''

    @property
    def exit_code(self) -> int:
        if self.skipped:
            return 2
        return 0 if self.ok else 1

    def to_dict(self) -> dict:
        return {'ok': self.ok, 'skipped': self.skipped,
                'backend': self.backend, 'board_key': self.board_key,
                'workdir': self.workdir, 'n_pins': self.n_pins,
                'duplicates': self.duplicates, 'errors': self.errors,
                'detail': self.detail}

    def render(self) -> str:
        if self.skipped:
            return (f'[SKIP] pin-validate({self.board_key}): {self.detail}\n'
                    '       検証は行われていない(OK ではない)')
        head = ('[OK]  ' if self.ok else '[FAIL]')
        lines = [f'{head} pin-validate({self.board_key}, {self.backend}): '
                 f'{self.n_pins} ピンを実ツールで確認']
        if self.duplicates:
            lines.append(f'       重複割当(検証から除外): '
                         f'{"; ".join(self.duplicates)}')
        if not self.ok:
            lines.append(f'       {self.detail}')
            lines += [f'       {e}' for e in self.errors[:10]]
        elif self.detail:
            lines.append(f'       {self.detail}')
        return '\n'.join(lines)


def _load_board_arg(board: str | dict) -> dict:
    return resolve_board(board)   # key / JSON パス / dict の共通解決(board_db)


def _run_stage(tc: Toolchain, argv: list[str], cwd: Path, log_name: str,
               timeout: int, quartus_exe: str | None = None):
    if quartus_exe is not None:
        proc = tc.run_quartus(quartus_exe, argv, cwd=cwd, timeout=timeout)
    else:
        proc = tc.run_sim_tool(argv, cwd=cwd, timeout=timeout)
    log = (proc.stdout or '') + (proc.stderr or '')
    (cwd / log_name).write_text(log, encoding='utf-8', errors='replace')
    return proc.returncode, log


def _gowin_validate(board: dict, entries, wd: Path, tc: Toolchain,
                    timeout: int, res: PinValidateResult) -> PinValidateResult:
    missing = [n for n, p in (('yosys', tc.yosys),
                              ('nextpnr-himbaechel', tc.nextpnr_himbaechel))
               if p is None]
    if missing:
        res.skipped = True
        res.detail = (f'{", ".join(missing)} が見つからない'
                      '(oss-cad-suite 同梱。doctor で確認)')
        return res
    (wd / 'pinval.v').write_text(stub_verilog(entries), encoding='utf-8')
    (wd / 'pinval.cst').write_text(stub_cst(entries, board), encoding='utf-8')
    (wd / 'pinval.ys').write_text(
        'read_verilog pinval.v\n'
        'synth_gowin -top pinval -nowidelut -json pinval.json\n',
        encoding='utf-8')
    rc, log = _run_stage(tc, [tc.yosys, 'pinval.ys'], wd, 'yosys.log', timeout)
    if rc != 0:
        res.detail = 'yosys(stub 合成)が失敗'
        res.errors = _ERROR_RE.findall(log)[:20]
        return res
    args = [tc.nextpnr_himbaechel, '--json', 'pinval.json',
            '--write', 'pinval_pnr.json', '--device', board['device'],
            '--vopt', f'family={board["family"]}', '--vopt', 'cst=pinval.cst']
    for v in (board.get('synth_options') or {}).get('nextpnr_vopts', []):
        args += ['--vopt', v]
    rc, log = _run_stage(tc, args, wd, 'nextpnr.log', timeout)
    if rc != 0:
        res.detail = ('nextpnr がピン割当を拒否(実在しないピン/使用不可'
                      'ピンの可能性。nextpnr.log 参照)')
        res.errors = _ERROR_RE.findall(log)[:20]
        return res
    res.ok = True
    res.detail = 'nextpnr(chipdb)が全ピン割当を受理'
    return res


def _quartus_validate(board: dict, entries, wd: Path, tc: Toolchain,
                      timeout: int, quick: bool,
                      res: PinValidateResult) -> PinValidateResult:
    if tc.quartus('quartus_map') is None:
        res.skipped = True
        res.detail = 'quartus_map が見つからない(doctor で確認)'
        return res
    (wd / 'pinval.v').write_text(stub_verilog(entries), encoding='utf-8')
    (wd / 'pinval.qsf').write_text(stub_qsf(entries, board), encoding='utf-8')
    (wd / 'pinval.qpf').write_text('PROJECT_REVISION = "pinval"\n',
                                   encoding='utf-8')
    stages = [('quartus_map', 'map.log')]
    if not quick:
        stages.append(('quartus_fit', 'fit.log'))
    for exe, log_name in stages:
        if tc.quartus(exe) is None:
            res.skipped = True
            res.detail = f'{exe} が見つからない(doctor で確認)'
            return res
        rc, log = _run_stage(tc, ['pinval', '-c', 'pinval'], wd, log_name,
                             timeout, quartus_exe=exe)
        if rc != 0:
            res.detail = (f'{exe} がピン割当を拒否(実在しないピン/規格違反'
                          f'の可能性。{log_name} 参照)')
            res.errors = _ERROR_RE.findall(log)[:20]
            return res
    res.ok = True
    res.detail = ('Quartus(map+fit)が全ピン割当を受理' if not quick
                  else 'Quartus(map のみ、--quick)が受理。fit まで回すと'
                       '配置・規格の検証が強くなる')
    return res


def run_pin_validate(board: str | dict, via: str = 'auto', *,
                     quick: bool = False, workdir: str | Path | None = None,
                     tc: Toolchain | None = None,
                     timeout: int = 1800) -> PinValidateResult:
    try:
        doc = _load_board_arg(board)
    except Exception as e:  # noqa: BLE001 - CLI 向けに読込失敗も結果へ変換
        return PinValidateResult(ok=False, board_key=str(board),
                                 detail=f'board 読込失敗: {e}')
    key = doc.get('key', '?')
    res = PinValidateResult(ok=False, board_key=key)
    if via == 'auto':
        try:
            via = get_backend(doc).name
        except KeyError as e:
            res.detail = str(e)
            return res
    res.backend = via
    entries, dups = collect_pins(doc)
    res.n_pins = len(entries)
    res.duplicates = dups
    tc = tc or discover()
    wd = Path(workdir) if workdir else (
        TOOL_DIR / 'runs' / f'pinval-{key}-{time.strftime("%Y%m%d-%H%M%S")}')
    wd.mkdir(parents=True, exist_ok=True)
    res.workdir = str(wd)
    if via == 'gowin-oss':
        return _gowin_validate(doc, entries, wd, tc, timeout, res)
    if via == 'quartus':
        return _quartus_validate(doc, entries, wd, tc, timeout, quick, res)
    res.skipped = True
    res.detail = f'backend {via!r} の pin-validate は未対応'
    return res
