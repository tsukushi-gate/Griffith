"""JTAG 書込(mdl-fpga flash)。demo program.py の実証手順の仕様化。

経路は2つ:
- quartus: quartus_pgm -c <cable> -m jtag -o "p;<sof>"(.sof、SRAM 揮発)
- openfpgaloader: openFPGALoader <svf|rbf|fs>(EDA 無し PC 用。Gowin .fs は
  直接、.sof は事前に bundle / synth.convert_sof で svf/rbf へ変換しておく)
via='auto' は拡張子で自動選択(.sof→quartus / それ以外→openfpgaloader)。

openFPGALoader の指定は board 定義から導出する(_ofl_args):
detection.ofl_board があれば `-b <name>`(Tang Nano 20K 等のボード内蔵
プログラマ)、無ければ `-c <cable>`(USB-Blaster 等の外付けケーブル)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .toolchain import Toolchain, discover


@dataclass
class FlashResult:
    ok: bool
    sof: str               # ビットストリームパス(歴史的に sof 名。svf/rbf も入る)
    cable: str
    detail: str = ''
    via: str = ''

    def to_dict(self) -> dict:
        return {'ok': self.ok, 'bitstream': self.sof, 'cable': self.cable,
                'via': self.via, 'detail': self.detail}

    def render(self) -> str:
        lines = [f'flash: {"OK" if self.ok else "FAIL"}({self.sof})']
        if self.via:
            lines.append(f'  via: {self.via}')
        if self.detail:
            lines.append(f'  {self.detail}')
        return '\n'.join(lines)


def list_cables(tc: Toolchain | None = None) -> str:
    tc = tc or discover()
    proc = tc.run_quartus('quartus_pgm', ['-l'], cwd='.', timeout=60)
    return (proc.stdout or '') + (proc.stderr or '')


def _choose_via(path: Path, via: str,
                tc: Toolchain) -> tuple[str | None, str | None]:
    """(実効via, エラーメッセージ) を返す純関数コア(テストで固定)。"""
    ext = path.suffix.lower()
    if via == 'auto':
        via = 'quartus' if ext == '.sof' else 'openfpgaloader'
    if via == 'quartus':
        if ext != '.sof':
            return None, (f'{ext} は quartus_pgm で書けない(.sof のみ)。'
                          '--via openfpgaloader を使う')
        if tc.quartus('quartus_pgm') is None:
            return None, ('quartus_pgm が見つからない。Quartus 本体か '
                          'Programmer 単体版を導入し doctor で確認')
        return 'quartus', None
    if via == 'openfpgaloader':
        if ext == '.sof':
            return None, ('.sof は openFPGALoader で書けない。bundle(または '
                          'synth.convert_sof)で svf/rbf に変換するか '
                          '--via quartus を使う')
        if tc.openfpgaloader is None:
            return None, ('openFPGALoader が見つからない(oss-cad-suite '
                          '同梱。doctor 参照)')
        return 'openfpgaloader', None
    return None, f'未知の --via: {via}(auto | quartus | openfpgaloader)'


def _ofl_args(board_def: dict | None, ofl_cable: str, path: Path) -> list[str]:
    """openFPGALoader の引数組立(純関数、テストで固定)。

    board の detection.ofl_board があれば `-b <name>`(内蔵プログラマ、
    Tang Nano 20K の BL616 等)。無ければ `-c <cable>`(board の
    detection.cable → 引数 ofl_cable の順で解決)。
    """
    det = (board_def or {}).get('detection') or {}
    if det.get('ofl_board'):
        return ['-b', det['ofl_board'], str(path)]
    return ['-c', det.get('cable') or ofl_cable, str(path)]


def run_flash(bitstream: str | Path, cable: str = '1',
              via: str = 'auto', ofl_cable: str = 'usb-blaster',
              board: str | None = None,
              tc: Toolchain | None = None) -> FlashResult:
    """JTAG 書込。cable=quartus_pgm のケーブル番号 / ofl_cable=openFPGALoader
    のケーブル名(既定 usb-blaster)。board を渡すとボード定義から
    openFPGALoader の -b/-c を導出する(build の「次の一手」が付けてくる)。"""
    tc = tc or discover()
    board_def: dict | None = None
    if board:
        from .board_db import resolve_board
        board_def = resolve_board(board)
    path = Path(bitstream).resolve()
    if not path.exists():
        return FlashResult(ok=False, sof=str(path), cable=cable,
                           detail='ビットストリームが存在しない。先に build を実行')
    eff_via, err = _choose_via(path, via, tc)
    if err:
        return FlashResult(ok=False, sof=str(path), cable=cable, detail=err)

    if eff_via == 'quartus':
        proc = tc.run_quartus('quartus_pgm',
                              ['-c', cable, '-m', 'jtag', '-o', f'p;{path}'],
                              cwd=path.parent, timeout=300)
        out = (proc.stdout or '') + (proc.stderr or '')
        if proc.returncode == 0:
            return FlashResult(ok=True, sof=str(path), cable=cable,
                               via=eff_via,
                               detail='JTAG 書込完了(SRAM 揮発。電源断で消える)')
        cables = list_cables(tc)
        return FlashResult(
            ok=False, sof=str(path), cable=cable, via=eff_via,
            detail=f'書込失敗:\n{out}\n検出ケーブル一覧:\n{cables}\n'
                   '次の一手: ボード電源/USB(UB2 ポート)/ドライバ'
                   '(usb_blaster)を確認')

    # openfpgaloader(DLL 解決のため run_sim_tool=bin+lib PATH 前置で実行)
    ofl = _ofl_args(board_def, ofl_cable, path)
    cable_label = ' '.join(ofl[:2])
    proc = tc.run_sim_tool([tc.openfpgaloader, *ofl],
                           cwd=str(path.parent), timeout=300)
    out = (proc.stdout or '') + (proc.stderr or '')
    if proc.returncode == 0:
        return FlashResult(ok=True, sof=str(path), cable=cable_label,
                           via=eff_via,
                           detail='JTAG 書込完了(SRAM 揮発。電源断で消える)')
    return FlashResult(
        ok=False, sof=str(path), cable=cable_label, via=eff_via,
        detail=f'openFPGALoader 書込失敗:\n{out.strip()[:800]}\n'
               '次の一手: ①ボード電源/USB ②ドライバ(外付け USB-Blaster は '
               'Quartus ドライバと WinUSB が排他。BL616 等の内蔵プログラマは '
               'WinUSB/CDC ドライバの認識を Device Manager で確認)'
               '③--board でボード定義を渡すと -b/-c を自動導出する'
               '(Intel 系の代替は Quartus Programmer --via quartus + .sof)')
