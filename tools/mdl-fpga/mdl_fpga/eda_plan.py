"""eda plan / eda install: ツールチェーン導入計画と自動導入(AUTOFLOW 工程6)。

doctor(診断)を「導入計画の提示(plan)と実行(install)」へ一段進める。
バックエンド判定は backends レジストリ(board の toolchain 欄→family 推定)。

自動化水準(AUTOFLOW の表のとおり):
- oss-cad-suite(gowin-oss 等): **完全自動**(GitHub Releases から取得・展開。
  ログイン・EULA クリック不要)
- Quartus: 本体はユーザ導入(plan が版・URL・無人フラグ・qdz 名を提示)
- Vivado: install_config.txt の Modules 行と batch 手順の提示のみ

ベンダ EDA の EULA 同意はユーザの意思表示が前提のため、本モジュールは
同意フラグ付きコマンドを「提示」するだけで勝手に実行しない。
"""

from __future__ import annotations

import json
import platform as _platform
import re
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .backends import get_backend
from .toolchain import LOCAL_TOML, Toolchain, discover

OSS_CAD_REPO = 'YosysHQ/oss-cad-suite-build'

# family 名 → ベンダ(バックエンド未実装でも plan は出す)
_FAMILY_VENDOR: tuple[tuple[str, str], ...] = (
    (r'^gw|gowin', 'gowin'),
    (r'cyclone|max ?10|arria|stratix', 'intel'),
    (r'artix|kintex|virtex|zynq|spartan|^xc7', 'xilinx'),
    (r'ice40|ecp5|machxo|crosslink', 'lattice'),
)

# Intel family → Quartus デバイスサポート(.qdz)のコンポーネント名
_QDZ_COMPONENTS = {
    'Cyclone V': 'cyclonev',
    'MAX 10': 'max10',
    'Cyclone 10 LP': 'cyclone10lp',
    'Cyclone IV': 'cyclone',
    'Arria II': 'arria_lite',
}

# Xilinx family → Vivado install_config.txt の Modules 名
_VIVADO_MODULES = {
    'Artix-7': 'Artix-7',
    'Kintex-7': 'Kintex-7',
    'Zynq-7000': 'Zynq-7000',
    'Spartan-7': 'Spartan-7',
}


def _vendor_of(family: str) -> str | None:
    for pat, vendor in _FAMILY_VENDOR:
        if re.search(pat, family, re.I):
            return vendor
    return None


@dataclass
class EdaPlan:
    board_key: str
    family: str
    vendor: str | None
    backend: str | None                    # backends レジストリ名(未実装は None)
    tools: list[tuple[str, str | None]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    auto_installable: bool = False
    actions: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.tools) and not self.missing

    def to_dict(self) -> dict:
        return {'board_key': self.board_key, 'family': self.family,
                'vendor': self.vendor, 'backend': self.backend,
                'tools': [{'name': n, 'path': p} for n, p in self.tools],
                'missing': self.missing,
                'auto_installable': self.auto_installable,
                'ready': self.ready, 'actions': self.actions}

    def render(self) -> str:
        lines = [f'EDA 導入計画: {self.board_key}(family={self.family} / '
                 f'backend={self.backend or "未実装"})']
        for n, p in self.tools:
            mark = '[OK]  ' if p else '[FAIL]'
            lines.append(f'  {mark} {n}: {p or "見つからない"}')
        if self.ready:
            lines.append('  → 必要なツールは全て導入済み(build/flash 可)')
        else:
            lines.append(f'  → 不足: {", ".join(self.missing) or "(判定不能)"}')
            lines += [f'  手順: {a}' for a in self.actions]
        return '\n'.join(lines)


def _plan_gowin_oss(board: dict, tc: Toolchain, plan: EdaPlan) -> None:
    plan.tools = [('yosys', tc.yosys),
                  ('nextpnr-himbaechel', tc.nextpnr_himbaechel),
                  ('gowin_pack', tc.gowin_pack),
                  ('openFPGALoader', tc.openfpgaloader)]
    plan.missing = [n for n, p in plan.tools if p is None]
    if plan.missing:
        plan.auto_installable = True
        plan.actions = [
            '`mdl-fpga eda install oss-cad-suite` で自動導入できる'
            '(GitHub Releases から取得・展開。ログイン・EULA クリック不要)',
            '導入後 `doctor` で gowin-oss: 可 を確認する',
        ]


def _quartus_version(tc: Toolchain) -> str | None:
    if tc.quartus_bin is None:
        return None
    m = re.search(r'(\d+\.\d+\w*)', str(tc.quartus_bin))
    return m.group(1) if m else None


def _plan_quartus(board: dict, tc: Toolchain, plan: EdaPlan) -> None:
    exes = ['quartus_map', 'quartus_fit', 'quartus_asm', 'quartus_sta',
            'quartus_pgm']
    plan.tools = [(e, tc.quartus(e)) for e in exes]
    plan.missing = [n for n, p in plan.tools if p is None]
    if not plan.missing:
        return
    ver = _quartus_version(tc)
    comp = _QDZ_COMPONENTS.get(board.get('family', ''))
    qdz = (f'{comp}-{ver}.qdz' if comp and ver
           else f'{comp or "<family>"}-<版>.qdz')
    if tc.quartus_bin is None:
        plan.actions = [
            'Quartus Prime Lite 本体をユーザが導入する(EULA 同意が必要な'
            'ため自動化しない): https://www.intel.com/content/www/us/en/'
            'software-kit/download.html から本体インストーラと'
            f'デバイスファイル {qdz} を取得',
            f'無人インストール例: QuartusLiteSetup-<版>.exe --mode unattended '
            f'--unattendedmodeui minimal --accept_eula 1 --installdir <dir>'
            '(qdz を同じフォルダに置くと該当 family だけ入る)',
            '導入後 MDL_FPGA_QUARTUS_BIN か local.toml の quartus_bin を設定し '
            '`doctor` で確認',
        ]
    else:
        plan.actions = [
            f'導入済み Quartus({ver or "版不明"})に不足コンポーネントが'
            f'ある。デバイスファイル {qdz}(本体と同一版のみ可)を取得し、'
            'Quartus GUI の Tools → Install Devices で投入する',
        ]


def _plan_vivado(board: dict, plan: EdaPlan) -> None:
    mod = _VIVADO_MODULES.get(board.get('family', ''), board.get('family', ''))
    plan.missing = ['vivado(バックエンド未実装)']
    plan.actions = [
        'Vivado バックエンドは未実装(合成は不可)。導入自体は batch モードで'
        '無人化できる:',
        '`xsetup -b ConfigGen` で install_config.txt を生成し、'
        f'Modules 行を `{mod}:1`(他は :0)に編集',
        '`xsetup --agree XilinxEULA,3rdPartyEULA --batch Install '
        '--config install_config.txt`(EULA 同意はユーザの意思表示として実行)',
    ]


def plan_for_board(board: dict, tc: Toolchain | None = None) -> EdaPlan:
    tc = tc or discover()
    family = board.get('family', '')
    try:
        backend = get_backend(board).name
    except KeyError:
        backend = None
    plan = EdaPlan(board_key=board.get('key', '?'), family=family,
                   vendor=_vendor_of(family), backend=backend)
    if backend == 'gowin-oss':
        _plan_gowin_oss(board, tc, plan)
    elif backend == 'quartus':
        _plan_quartus(board, tc, plan)
    elif plan.vendor == 'xilinx':
        _plan_vivado(board, plan)
    else:
        plan.missing = [f'family {family!r} のツールチェーン判定が未対応']
        plan.actions = ['boards/*.json の toolchain 欄を明示するか、'
                        'backends/ に SynthBackend を追加する']
    return plan


# ------------------------------------------- oss-cad-suite 自動導入 ---


def resolve_platform() -> tuple[str, str, str]:
    """(os名, arch, 拡張子)。oss-cad-suite のリリース資産名の規約に合わせる。"""
    machine = _platform.machine().lower()
    arch = 'arm64' if machine in ('arm64', 'aarch64') else 'x64'
    if sys.platform == 'win32':
        return 'windows', arch, '.exe'
    if sys.platform == 'darwin':
        return 'darwin', arch, '.tgz'
    return 'linux', arch, '.tgz'


def asset_url(tag: str, os_name: str, arch: str, ext: str) -> str:
    stamp = tag.replace('-', '')
    return (f'https://github.com/{OSS_CAD_REPO}/releases/download/{tag}/'
            f'oss-cad-suite-{os_name}-{arch}-{stamp}{ext}')


def latest_tag(json_fetch=None) -> str:
    url = f'https://api.github.com/repos/{OSS_CAD_REPO}/releases/latest'
    if json_fetch is None:
        req = urllib.request.Request(url, headers={'User-Agent': 'mdl-fpga'})
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            doc = json.loads(r.read().decode('utf-8'))
    else:
        doc = json_fetch(url)
    return doc['tag_name']


def _download(url: str, dst: Path, progress=print) -> None:
    req = urllib.request.Request(url, headers={'User-Agent': 'mdl-fpga'})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        total = int(r.headers.get('Content-Length') or 0)
        done = last = 0
        with open(dst, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and done - last > (64 << 20):   # 64MB ごとに進捗
                    progress(f'  ダウンロード {done / total:6.1%}')
                    last = done


def update_local_toml(oss_root: Path, toml_path: Path | None = None) -> str:
    """local.toml の oss_cad_root を更新する(他行は温存。行が無ければ追記)。"""
    toml_path = toml_path or LOCAL_TOML
    value = str(oss_root).replace('\\', '/')
    line = f'oss_cad_root = "{value}"'
    if toml_path.exists():
        text = toml_path.read_text(encoding='utf-8')
        if re.search(r'^\s*oss_cad_root\s*=', text, re.M):
            text = re.sub(r'^\s*oss_cad_root\s*=.*$', line, text, count=1,
                          flags=re.M)
        else:
            text = text.rstrip('\n') + f'\n{line}\n'
    else:
        text = f'# mdl-fpga ローカル設定(eda install が生成)\n{line}\n'
    toml_path.write_text(text, encoding='utf-8')
    return line


def install_oss_cad_suite(dest_root: str | Path | None = None,
                          tag: str | None = None, *,
                          fetch_bytes=None, runner=None, json_fetch=None,
                          write_config: bool = False,
                          progress=print) -> Path:
    """oss-cad-suite を取得・展開する(完全自動。既存導入は上書きしない)。

    fetch_bytes(url, dst) / runner(argv) / json_fetch(url) はテスト注入点。
    アーカイブは削除せず残す(再展開・検証用)。
    """
    dest_root = Path(dest_root) if dest_root else Path.home() / 'oss-cad-suite'
    if dest_root.name != 'oss-cad-suite':
        raise ValueError(
            f'導入先は …/oss-cad-suite である必要がある({dest_root})。'
            'アーカイブが oss-cad-suite/ を展開するため')
    if dest_root.exists():
        raise FileExistsError(
            f'{dest_root} は既に存在する(上書きしない)。更新する場合は'
            '別ディレクトリへ導入して local.toml を切り替える')
    os_name, arch, ext = resolve_platform()
    tag = tag or latest_tag(json_fetch=json_fetch)
    url = asset_url(tag, os_name, arch, ext)
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    archive = dest_root.parent / Path(url).name
    if not archive.exists():
        progress(f'取得: {url}')
        (fetch_bytes or _download)(url, archive)
    else:
        progress(f'取得済みアーカイブを使用: {archive}')
    progress(f'展開: {archive.name} → {dest_root.parent}')
    if ext == '.tgz':
        with tarfile.open(archive) as tf:
            tf.extractall(dest_root.parent, filter='data')
    else:
        # Windows 版は自己展開アーカイブ(7z SFX)。-y -o<dir> で無人展開
        argv = [str(archive), '-y', f'-o{dest_root.parent}']
        proc = (runner or subprocess.run)(argv)
        rc = getattr(proc, 'returncode', 0)
        if rc != 0:
            raise RuntimeError(
                f'自己展開が失敗(rc={rc})。手動で {archive} を実行して'
                f'{dest_root.parent} に展開する')
    if not (dest_root / 'bin').exists():
        raise RuntimeError(
            f'{dest_root}/bin が見つからない(展開失敗の可能性)。'
            f'アーカイブは {archive} に残っている')
    if write_config:
        line = update_local_toml(dest_root)
        progress(f'local.toml 更新: {line}')
    else:
        progress('local.toml に設定する場合: '
                 f'oss_cad_root = "{str(dest_root).replace(chr(92), "/")}"'
                 '(または --write-config)')
    progress(f'導入完了: {dest_root}(doctor で確認を推奨)')
    return dest_root
