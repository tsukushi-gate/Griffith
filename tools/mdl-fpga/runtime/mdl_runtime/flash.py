"""書込(openFPGALoader 解決+実行)。

解決順: 同梱 flasher/<os>/ → PATH → 誘導メッセージ。
ベンダ EDA は同梱しない(EULA)ため、必要なボードでは誘導のみ。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

GUIDANCE = (
    'openFPGALoader が見つからない。導入方法:\n'
    '  - macOS : brew install openfpgaloader\n'
    '  - Linux : ディストリのパッケージ or oss-cad-suite\n'
    '  - Windows: oss-cad-suite'
    '(https://github.com/YosysHQ/oss-cad-suite-build)を展開し '
    'bin と lib を PATH に追加\n'
    '  ※ Windows はドライバ注意(FTDI 系は Zadig で該当 Interface のみ '
    'WinUSB 化。詳細は bundle README)')


def _os_dir() -> str:
    if sys.platform == 'win32':
        return 'windows'
    if sys.platform == 'darwin':
        return 'darwin'
    return 'linux'


def resolve_flasher(app_dir: str | Path) -> tuple[str | None, str]:
    """(実行体パス, 由来説明)。見つからなければ (None, 誘導文)。"""
    root = Path(app_dir).resolve().parent      # app/ の親= zip 展開ルート
    exe_name = 'openFPGALoader.exe' if sys.platform == 'win32' \
        else 'openFPGALoader'
    for base in (root / 'flasher' / _os_dir(),
                 root / 'flasher' / _os_dir() / 'bin'):
        cand = base / exe_name
        if cand.is_file():
            return str(cand), f'同梱 flasher({cand.parent})'
    w = shutil.which('openFPGALoader')
    if w:
        return w, 'PATH'
    return None, GUIDANCE


def flash_argv(contract: dict, app_dir: str | Path, exe: str,
               persist: bool = False) -> tuple[list[str], Path]:
    """(argv, bitstream パス)。bitstream は zip ルート相対。

    persist=True で SPI フラッシュへ永続書込(-f。電源断でも保持、
    対応ボードのみ)。既定は SRAM 書込=揮発(電源断で消える)。
    """
    root = Path(app_dir).resolve().parent
    bit = root / contract['flash']['bitstream']
    args = list(contract['flash']['args'])
    if persist:
        args.append('-f')
    return [exe, *args, str(bit)], bit


def run_flash(contract: dict, app_dir: str | Path, *,
              exe: str | None = None, persist: bool = False,
              dry_run: bool = False, timeout: int = 300) -> dict:
    if exe is None:
        exe, source = resolve_flasher(app_dir)
        if exe is None:
            return {'ok': False, 'detail': source}
    else:
        source = '明示指定'
    argv, bit = flash_argv(contract, app_dir, exe, persist=persist)
    if not bit.exists():
        return {'ok': False, 'detail': f'ビットストリームが無い: {bit}'}
    if dry_run:
        return {'ok': True, 'argv': argv, 'source': source, 'dry_run': True}
    # 同梱 flasher は lib/ 同階層が必要な場合がある → 実行ディレクトリを寄せる
    proc = subprocess.run(argv, capture_output=True, text=True,
                          errors='replace', timeout=timeout,
                          cwd=str(Path(exe).parent))
    out = (proc.stdout or '') + (proc.stderr or '')
    return {'ok': proc.returncode == 0, 'argv': argv, 'source': source,
            'detail': out.strip()[-2000:]}
