"""外部ツールチェーン(oss-cad-suite / Quartus)の探索と実行環境。

探索順序(doctor の診断もこの順): 環境変数 → local.toml → PATH → 既知パス。
oss-cad-suite は bin と lib の両方を PATH 先頭に置かないと DLL 解決に失敗する
(0xC0000139。社内先行実験の教訓)ため sim_env() が必ず前置する。
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent.parent          # tools/mdl-fpga/
LOCAL_TOML = TOOL_DIR / 'local.toml'

# 既知のインストール候補(グロブ)。local.toml か環境変数があればそちらが優先。
# 非該当 OS のパターンは glob が空を返すだけなので OS 分岐は不要(PORTABILITY.md 3)
_QUARTUS_CANDIDATES = [
    r'C:\altera_lite\*\quartus\bin64',
    r'C:\intelFPGA_lite\*\quartus\bin64',
    r'C:\intelFPGA\*\quartus\bin64',
    # Linux(bin64 ではなく bin)
    str(Path.home() / 'intelFPGA_lite' / '*' / 'quartus' / 'bin'),
    str(Path.home() / 'intelFPGA' / '*' / 'quartus' / 'bin'),
    '/opt/intelFPGA_lite/*/quartus/bin',
    '/opt/intelFPGA/*/quartus/bin',
]
_OSS_CAD_CANDIDATES = [
    r'C:\oss-cad-suite',
    r'D:\oss-cad-suite',
    str(Path.home() / 'oss-cad-suite'),
    '/opt/oss-cad-suite',
]


def _load_local_config() -> dict:
    if LOCAL_TOML.exists():
        with open(LOCAL_TOML, 'rb') as f:
            return tomllib.load(f)
    return {}


def _first_glob(patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(glob.glob(pat), reverse=True)  # 新しいバージョン優先
        for h in hits:
            if Path(h).is_dir():
                return Path(h)
    return None


@dataclass
class Toolchain:
    """解決済みツールパス。見つからないものは None(doctor が診断する)。"""

    oss_root: Path | None = None      # oss-cad-suite ルート(bin/ lib/ を持つ)
    quartus_bin: Path | None = None   # quartus/bin64
    # ベンダ EDA 探索枠(現状は探索+doctor 表示のみ。バックエンド実装は将来):
    vivado_bin: Path | None = None     # Xilinx Vivado の bin
    gowin_eda_bin: Path | None = None  # Gowin EDA(gw_sh)の bin
    radiant_bin: Path | None = None    # Lattice Radiant の bin
    sources: dict = field(default_factory=dict)  # 何経由で見つけたか(診断用)

    # --- oss-cad-suite ---

    @property
    def oss_bin(self) -> Path | None:
        return self.oss_root / 'bin' if self.oss_root else None

    def sim_env(self) -> dict:
        """iverilog/vvp/verilator 実行用の環境変数(bin+lib を PATH 前置)。"""
        env = os.environ.copy()
        if self.oss_root:
            bin_ = str(self.oss_root / 'bin')
            lib_ = str(self.oss_root / 'lib')
            env['PATH'] = os.pathsep.join([bin_, lib_, env.get('PATH', '')])
        return env

    def _oss_exe(self, name: str) -> str | None:
        # Windows 版は {name}.exe、Linux/macOS 版は拡張子なし(PORTABILITY.md 1)
        if self.oss_bin:
            for cand in (self.oss_bin / f'{name}.exe', self.oss_bin / name):
                if cand.is_file():
                    return str(cand)
        return shutil.which(name)

    @property
    def iverilog(self) -> str | None:
        return self._oss_exe('iverilog')

    @property
    def vvp(self) -> str | None:
        return self._oss_exe('vvp')

    @property
    def verilator(self) -> str | None:
        # Windows 版 oss-cad-suite は verilator_bin(.exe)が実行体。
        # Linux/macOS 版はラッパ verilator が正(PORTABILITY.md 4)
        return self._oss_exe('verilator_bin') or self._oss_exe('verilator')

    @property
    def openfpgaloader(self) -> str | None:
        """openFPGALoader(oss-cad-suite 同梱。EDA 無し検出/書込に使う)。

        実行は必ず run_sim_tool 経由(bin+lib の PATH 前置が無いと
        0xC0000135 で DLL 解決に失敗する。iverilog と同じ罠を実測確認済み)。
        """
        return self._oss_exe('openFPGALoader')

    # --- gowin-oss 合成(yosys → nextpnr-himbaechel → gowin_pack) ---

    @property
    def yosys(self) -> str | None:
        return self._oss_exe('yosys')

    @property
    def nextpnr_himbaechel(self) -> str | None:
        return self._oss_exe('nextpnr-himbaechel')

    @property
    def gowin_pack(self) -> str | None:
        """apycula の gowin_pack(oss-cad-suite 同梱の Python エントリポイント)。"""
        return self._oss_exe('gowin_pack')

    def oss_share(self) -> Path | None:
        """oss-cad-suite の share/(nextpnr chipdb 存在確認等に使う)。"""
        return self.oss_root / 'share' if self.oss_root else None

    # --- Quartus ---

    def quartus(self, exe: str) -> str | None:
        """quartus_map / quartus_fit / quartus_asm / quartus_sta / quartus_pgm 等。"""
        # Windows は {exe}.exe、Linux は拡張子なし(PORTABILITY.md 2)
        if self.quartus_bin:
            for cand in (self.quartus_bin / f'{exe}.exe', self.quartus_bin / exe):
                if cand.is_file():
                    return str(cand)
        return shutil.which(exe)

    def run_quartus(self, exe: str, args: list[str], cwd: str | Path,
                    timeout: int = 3600) -> subprocess.CompletedProcess:
        path = self.quartus(exe)
        if path is None:
            raise FileNotFoundError(f'{exe} が見つからない(doctor を実行して確認)')
        return subprocess.run(
            [path, *args], cwd=str(cwd), timeout=timeout,
            capture_output=True, text=True, errors='replace',
        )

    def run_sim_tool(self, argv: list[str], cwd: str | Path,
                     timeout: int = 600) -> subprocess.CompletedProcess:
        """iverilog/vvp を sim_env で実行する(argv[0] は絶対パス)。"""
        return subprocess.run(
            argv, cwd=str(cwd), timeout=timeout, env=self.sim_env(),
            capture_output=True, text=True, errors='replace',
        )


def discover() -> Toolchain:
    """環境変数 → local.toml → PATH → 既知パス の順でツールチェーンを解決する。"""
    tc = Toolchain()
    cfg = _load_local_config()

    # oss-cad-suite
    env_oss = os.environ.get('MDL_FPGA_OSS_CAD')
    cfg_oss = cfg.get('oss_cad_root')
    for src, cand in (('env:MDL_FPGA_OSS_CAD', env_oss), ('local.toml', cfg_oss)):
        if cand and Path(cand).is_dir():
            tc.oss_root = Path(cand)
            tc.sources['oss_cad'] = src
            break
    if tc.oss_root is None:
        w = shutil.which('iverilog')
        if w:  # PATH に居るなら root は親の親(bin の上)
            tc.oss_root = Path(w).resolve().parent.parent
            tc.sources['oss_cad'] = 'PATH'
    if tc.oss_root is None:
        hit = _first_glob(_OSS_CAD_CANDIDATES)
        if hit:
            tc.oss_root = hit
            tc.sources['oss_cad'] = 'known-location'

    # Quartus
    env_q = os.environ.get('MDL_FPGA_QUARTUS_BIN')
    cfg_q = cfg.get('quartus_bin')
    for src, cand in (('env:MDL_FPGA_QUARTUS_BIN', env_q), ('local.toml', cfg_q)):
        if cand and Path(cand).is_dir():
            tc.quartus_bin = Path(cand)
            tc.sources['quartus'] = src
            break
    if tc.quartus_bin is None:
        w = shutil.which('quartus_sh')
        if w:
            tc.quartus_bin = Path(w).resolve().parent
            tc.sources['quartus'] = 'PATH'
    if tc.quartus_bin is None:
        hit = _first_glob(_QUARTUS_CANDIDATES)
        if hit:
            tc.quartus_bin = hit
            tc.sources['quartus'] = 'known-location'

    # ベンダ EDA 探索枠(env → local.toml のみ。既知グロブは導入時に追加)。
    # 現状は「見つかったら doctor に表示」まで。バックエンド実装は将来コミット
    for attr, env_key, cfg_key in (
            ('vivado_bin', 'MDL_FPGA_VIVADO_BIN', 'vivado_bin'),
            ('gowin_eda_bin', 'MDL_FPGA_GOWIN_EDA_BIN', 'gowin_eda_bin'),
            ('radiant_bin', 'MDL_FPGA_RADIANT_BIN', 'radiant_bin')):
        for src, cand in ((f'env:{env_key}', os.environ.get(env_key)),
                          ('local.toml', cfg.get(cfg_key))):
            if cand and Path(cand).is_dir():
                setattr(tc, attr, Path(cand))
                tc.sources[attr] = src
                break

    return tc
