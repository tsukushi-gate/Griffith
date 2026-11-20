"""flasher パック生成(eda flasher-pack。STANDALONE.md 実装順序5)。

開発機の oss-cad-suite から openFPGALoader+依存 DLL+**ライセンス全文**を
抽出し、bundle が既定同梱するパック(runs/flasher-pack 等)を組み立てる。
再配布義務(Apache-2.0/LGPL の全文同梱、LGPL のソース入手先明示)を
NOTICE.md と LICENSES/ で満たす。現状は Windows 分のみ
(linux/darwin は各 OS の oss-cad-suite から同様に抽出して追加する)。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2

from .toolchain import Toolchain, discover

# bin/ から(実行体)
WINDOWS_EXES = ('openFPGALoader.exe',)
# lib/ から(依存 DLL。exe と同じディレクトリに置けば PATH 不要で動く)
WINDOWS_DLLS = (
    'libftdi1.dll', 'libftdipp1.dll', 'libgcc_s_seh-1.dll', 'libhidapi.dll',
    'libstdc++-6.dll', 'libusb-1.0.dll', 'libwinpthread-1.dll', 'zlib1.dll',
)
# oss-cad-suite の license/ にある全文をそのまま同梱する
LICENSE_FILES = ('LICENSE.openfpgaloader', 'LICENSE.libftdi', 'LICENSE.libusb')
# license/ に全文が無い同梱物(SPDX+上流 URL を NOTICE に記載)
EXTRA_COMPONENTS = (
    ('libhidapi.dll', 'HIDAPI(BSD-3-Clause を選択)',
     'https://github.com/libusb/hidapi'),
    ('zlib1.dll', 'zlib License', 'https://zlib.net/'),
    ('libgcc_s_seh-1.dll / libstdc++-6.dll',
     'GPL-3.0-or-later WITH GCC-exception-3.1(ランタイム例外により'
     'バイナリ同梱可)', 'https://gcc.gnu.org/'),
    ('libwinpthread-1.dll', 'MIT/BSD(mingw-w64)',
     'https://www.mingw-w64.org/'),
)


@dataclass
class PackResult:
    out: str
    version: str = 'unknown'
    files: list[str] = field(default_factory=list)
    license_missing: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f'flasher パック生成: {self.out}',
                 f'  openFPGALoader: {self.version}',
                 f'  ファイル {len(self.files)} 件(windows/ + LICENSES/ + '
                 'NOTICE.md)']
        if self.license_missing:
            lines.append('  WARN: ライセンス全文が oss-cad-suite に無い: '
                         f'{", ".join(self.license_missing)}'
                         '(NOTICE の SPDX+URL 記載で代替)')
        lines.append('  bundle の既定同梱にするには local.toml の '
                     f'flasher_pack = "{Path(self.out).as_posix()}"')
        return '\n'.join(lines)


def _notice(version: str, oss_root: Path, license_missing: list[str]) -> str:
    lines = [
        '# flasher パック NOTICE(再配布物の構成とライセンス)',
        '',
        f'- openFPGALoader {version}(SPDX-License-Identifier: Apache-2.0)',
        '  https://github.com/trabucayre/openFPGALoader',
        '  ライセンス全文: LICENSES/LICENSE.openfpgaloader',
        '- libftdi1 / libftdipp1(LGPL-2.1-only)',
        '  https://www.intra2net.com/en/developer/libftdi/',
        '  ライセンス全文: LICENSES/LICENSE.libftdi',
        '- libusb-1.0(LGPL-2.1-or-later)https://libusb.info/',
        '  ライセンス全文: LICENSES/LICENSE.libusb',
    ]
    for name, lic, url in EXTRA_COMPONENTS:
        lines.append(f'- {name}: {lic} — {url}')
    lines += [
        '',
        '## 取得元とソースコード入手先(LGPL 対応)',
        '',
        f'- 本パックのバイナリは oss-cad-suite の Windows ビルド'
        f'({oss_root})から抽出した',
        '- 各バイナリのソースは上記の各上流リポジトリ、およびビルド構成は',
        '  https://github.com/YosysHQ/oss-cad-suite-build から入手できる',
    ]
    if license_missing:
        lines += ['', f'(注: {", ".join(license_missing)} の全文は'
                      ' oss-cad-suite 配布に含まれず未同梱。上記 URL を参照)']
    return '\n'.join(lines) + '\n'


def build_pack(out_dir: str | Path, tc: Toolchain | None = None,
               runner=None) -> PackResult:
    """パックを組み立てる(既存ファイルは上書き=更新。削除はしない)。"""
    tc = tc or discover()
    if tc.oss_root is None:
        raise FileNotFoundError(
            'oss-cad-suite が見つからない(doctor / local.toml の '
            'oss_cad_root を確認)')
    out = Path(out_dir)
    win = out / 'windows'
    lic_dir = out / 'LICENSES'
    win.mkdir(parents=True, exist_ok=True)
    lic_dir.mkdir(parents=True, exist_ok=True)
    res = PackResult(out=str(out))

    for name, sub in [(n, 'bin') for n in WINDOWS_EXES] + \
                     [(n, 'lib') for n in WINDOWS_DLLS]:
        src = tc.oss_root / sub / name
        if not src.exists():
            raise FileNotFoundError(f'{src} が無い(oss-cad-suite の構成変更?'
                                    ' WINDOWS_DLLS の見直しが必要)')
        copy2(src, win / name)
        res.files.append(f'windows/{name}')
    for lf in LICENSE_FILES:
        src = tc.oss_root / 'license' / lf
        if src.exists():
            copy2(src, lic_dir / lf)
            res.files.append(f'LICENSES/{lf}')
        else:
            res.license_missing.append(lf)

    exe = win / WINDOWS_EXES[0]
    proc = (runner or subprocess.run)(
        [str(exe), '--Version'], capture_output=True, text=True,
        errors='replace', cwd=str(win), timeout=60)
    if getattr(proc, 'returncode', 1) == 0 and (proc.stdout or '').strip():
        res.version = proc.stdout.strip().splitlines()[0]

    (out / 'NOTICE.md').write_text(
        _notice(res.version, tc.oss_root, res.license_missing),
        encoding='utf-8')
    res.files.append('NOTICE.md')
    return res
