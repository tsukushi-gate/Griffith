"""ボード検出(mdl-fpga detect)。

EDA 無しを優先: openFPGALoader(oss-cad-suite 同梱)→ jtagconfig(Quartus)
の順で JTAG を叩き、openFPGALoader --scan-usb と pyserial で USB/COM を列挙し、
boards/*.json の detection 署名と照合して候補ボードを確信度付きで返す。

JTAG ケーブルは排他資源のため、このモジュールの実行は必ずユーザー操作起点
(CLI 実行かビューアの認識ボタン)で行う。自動ポーリング禁止。

原理的な限界(設計判断の根拠):
- IDCODE はデバイス(FPGA)しか教えない。パッケージ/スピードグレードは
  IDCODE に含まれず、ボード(ピン割当)は JTAG からは分からない
- 同一 FPGA を積んだ別ボードは原理的に区別できない → 候補が同率複数の
  ときは自動選択せず、ユーザーに確認する(silk_hint で案内)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .board_db import list_boards, load_board
from .toolchain import Toolchain, discover

# 照合の重み。part_text は jtagconfig が IDCODE から復号した部品名なので
# IDCODE 一致に準じる強さを持たせる(IDCODE 未採取ボードの検出に効く)
SCORE_IDCODE = 0.7
SCORE_PART = 0.5
SCORE_USB = 0.2
SCORE_CABLE = 0.1
BEST_THRESHOLD = 0.5


def normalize_idcode(text: str) -> str | None:
    """'02B040DD' / '0x2b040dd' → '0x02b040dd'(int 正規化で表記ゆれ吸収)。"""
    t = text.strip().lower()
    t = t[2:] if t.startswith('0x') else t
    try:
        v = int(t, 16)
    except ValueError:
        return None
    return f'0x{v:08x}'


@dataclass
class JtagDevice:
    idcode: str            # normalize_idcode 済み
    part_text: str = ''    # ツール表示の部品名テキスト(無ければ空)
    via: str = ''          # 'openfpgaloader' | 'jtagconfig'


@dataclass
class SerialPortSig:
    device: str
    vid: int | None = None
    pid: int | None = None
    description: str = ''


@dataclass
class UsbSig:
    vid: int
    pid: int
    text: str = ''         # 行の残り(manufacturer/serial/product の自由文)


@dataclass
class Observation:
    cables: list[str] = field(default_factory=list)
    jtag: list[JtagDevice] = field(default_factory=list)
    serial: list[SerialPortSig] = field(default_factory=list)
    usb: list[UsbSig] = field(default_factory=list)
    probes_run: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class BoardCandidate:
    key: str
    name: str
    tier: int
    confidence: float
    reasons: list[str] = field(default_factory=list)
    silk_hint: str | None = None
    unverified: bool = False


@dataclass
class DetectResult:
    observation: Observation
    candidates: list[BoardCandidate] = field(default_factory=list)

    def best(self) -> BoardCandidate | None:
        """一意に絞れた候補(同率トップが複数なら None=ユーザー確認へ)。"""
        if not self.candidates:
            return None
        top = self.candidates[0]
        if top.confidence < BEST_THRESHOLD:
            return None
        if (len(self.candidates) > 1
                and self.candidates[1].confidence >= top.confidence):
            return None
        return top

    def to_dict(self) -> dict:
        return {'observation': asdict(self.observation),
                'candidates': [asdict(c) for c in self.candidates],
                'best': asdict(self.best()) if self.best() else None}

    def render(self) -> str:
        obs = self.observation
        lines = [f'detect: プローブ={", ".join(obs.probes_run) or "(なし)"}']
        for c in obs.cables:
            lines.append(f'  ケーブル: {c}')
        for d in obs.jtag:
            part = f'({d.part_text})' if d.part_text else ''
            lines.append(f'  JTAG: {d.idcode} {part} via {d.via}')
        for s in obs.serial:
            lines.append(f'  シリアル: {s.device}({s.description})')
        for n in obs.notes:
            lines.append(f'  note: {n}')
        if self.candidates:
            lines.append('  候補:')
            for i, c in enumerate(self.candidates, 1):
                mark = '(実機未検証)' if c.unverified else ''
                lines.append(f'    {i}. {c.name}({c.key}){mark} '
                             f'確信度 {c.confidence:.1f} — '
                             f'{", ".join(c.reasons)}')
            b = self.best()
            if b is None:
                lines.append('  候補が一意に絞れない。基板のシルク印字で確認し '
                             '--board で明示指定する')
        else:
            lines.append('  候補なし。次の一手: ①ボードの電源と USB(UB2 ポート)'
                         '②doctor でケーブル/ドライバ確認 ③既知ボードなら '
                         '--board で明示指定')
        return '\n'.join(lines)


# ---------------------------------------------------------------- パーサ

# jtagconfig 出力例(https://www.01signal.com/vendor-specific/intel/
# fpga-command-line-jtag/ 掲載の実出力):
#   1) USB-BlasterII [1-5.1.2]
#     02B040DD   5CGTFD9(A5|C5|D5|E5)/..
_JTAGCFG_CABLE_RE = re.compile(
    r'^\s*\d+\)\s+(?P<cable>.+?)\s*$')
_JTAGCFG_DEV_RE = re.compile(
    r'^\s+(?P<idcode>[0-9A-Fa-f]{8})\s+(?P<part>\S+)')


def parse_jtagconfig(text: str) -> tuple[list[str], list[JtagDevice]]:
    cables: list[str] = []
    devices: list[JtagDevice] = []
    for line in text.splitlines():
        m = _JTAGCFG_DEV_RE.match(line)
        if m:
            idc = normalize_idcode(m.group('idcode'))
            if idc:
                devices.append(JtagDevice(idcode=idc,
                                          part_text=m.group('part'),
                                          via='jtagconfig'))
            continue
        m = _JTAGCFG_CABLE_RE.match(line)
        if m:
            cables.append(m.group('cable'))
    return cables, devices


# openFPGALoader --detect の出力(README 掲載例。実機採取で更新予定):
#   index 0:
#           idcode 0x218a043
#           manufacturer lattice
#           family MachXO2
#           model  LCMXO2-7000HC
# キー行のみを防御的に拾う(形式差に強くする)
_OFL_KV_RE = re.compile(r'^\s*(idcode|manufacturer|family|model)\s+(.+?)\s*$')


def parse_openfpgaloader_detect(text: str) -> list[JtagDevice]:
    devices: list[JtagDevice] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current and current.get('idcode'):
            part = ' '.join(x for x in (current.get('family'),
                                        current.get('model')) if x)
            devices.append(JtagDevice(idcode=current['idcode'],
                                      part_text=part.strip(),
                                      via='openfpgaloader'))
        current = None

    for line in text.splitlines():
        if re.match(r'^\s*index\s+\d+', line):
            flush()
            current = {}
            continue
        m = _OFL_KV_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if current is None:
            current = {}
        if key == 'idcode':
            current['idcode'] = normalize_idcode(val)
        else:
            current[key] = val
    flush()
    return devices


# openFPGALoader --scan-usb のヘッダ(v1.1.1 実出力):
#   Bus device vid:pid       probe_type manufacturer serial product
# 行の列構成には依存せず vid:pid と残りテキストのみ拾う
_SCAN_USB_RE = re.compile(
    r'(0x[0-9A-Fa-f]{1,4}):(0x[0-9A-Fa-f]{1,4})\s*(.*)$')


def parse_scan_usb(text: str) -> list[UsbSig]:
    sigs: list[UsbSig] = []
    for line in text.splitlines():
        if line.lower().startswith('bus '):
            continue
        m = _SCAN_USB_RE.search(line)
        if m:
            sigs.append(UsbSig(vid=int(m.group(1), 16),
                               pid=int(m.group(2), 16),
                               text=m.group(3).strip()))
    return sigs


# ---------------------------------------------------------------- プローブ

def probe_serial(obs: Observation) -> None:
    try:
        from serial.tools import list_ports
    except ImportError:
        obs.notes.append('pyserial 未導入のためシリアル列挙をスキップ'
                         '(`uv sync --extra fpga`)')
        return
    obs.probes_run.append('serial')
    for p in list_ports.comports():
        obs.serial.append(SerialPortSig(
            device=p.device, vid=p.vid, pid=p.pid,
            description=f'{p.description} {p.manufacturer or ""}'.strip()))


def _run_ofl(tc: Toolchain, args: list[str],
             timeout: int = 30) -> subprocess.CompletedProcess | None:
    exe = tc.openfpgaloader
    if exe is None:
        return None
    # bin+lib PATH 前置が必須(無いと 0xC0000135。実測)→ run_sim_tool 経由
    return tc.run_sim_tool([exe, *args], cwd=str(Path.cwd()), timeout=timeout)


def _board_cables() -> list[str]:
    cables: list[str] = []
    for key in list_boards():
        cable = (load_board(key).get('detection') or {}).get('cable')
        if cable and cable not in cables:
            cables.append(cable)
    return cables


def _probe_argsets() -> list[tuple[str, list[str]]]:
    """JTAG プローブの試行順: (ラベル, openFPGALoader 引数) の一覧。

    ボード DB の detection.ofl_board(`-b`、内蔵プログラマ系)と
    detection.cable(`-c`、外付けケーブル系)の両方から重複排除で生成し、
    最後に素の --detect を試す。
    """
    seen: set[str] = set()
    out: list[tuple[str, list[str]]] = []
    for key in list_boards():
        det = load_board(key).get('detection') or {}
        ofl_board = det.get('ofl_board')
        if ofl_board and f'b:{ofl_board}' not in seen:
            seen.add(f'b:{ofl_board}')
            out.append((f'-b {ofl_board}', ['-b', ofl_board, '--detect']))
        cable = det.get('cable')
        if cable and f'c:{cable}' not in seen:
            seen.add(f'c:{cable}')
            out.append((f'-c {cable}', ['-c', cable, '--detect']))
    out.append(('direct', ['--detect']))
    return out


def probe_usb(obs: Observation, tc: Toolchain) -> None:
    r = _run_ofl(tc, ['--scan-usb'])
    if r is None:
        obs.notes.append('openFPGALoader が見つからない(oss-cad-suite 同梱。'
                         'doctor 参照)→ USB 列挙をスキップ')
        return
    obs.probes_run.append('usb')
    if r.returncode == 0:
        obs.usb.extend(parse_scan_usb(r.stdout + r.stderr))
    else:
        obs.notes.append(f'--scan-usb 失敗: {(r.stderr or r.stdout).strip()[:120]}')


def probe_jtag_openfpgaloader(obs: Observation, tc: Toolchain) -> bool:
    """ボード DB 由来の -b/-c 候補を順に試す。1つでも成功したら True。"""
    if tc.openfpgaloader is None:
        obs.notes.append('openFPGALoader 不在 → jtagconfig にフォールバック')
        return False
    obs.probes_run.append('openfpgaloader')
    for label, args in _probe_argsets():
        r = _run_ofl(tc, args)
        if r is None:
            return False
        devices = parse_openfpgaloader_detect(r.stdout + r.stderr)
        if r.returncode == 0 and devices:
            obs.jtag.extend(devices)
            if label != 'direct':
                obs.cables.append(label.split(' ', 1)[1])
            return True
        first = (r.stderr or r.stdout).strip().splitlines()
        obs.notes.append(
            f'openFPGALoader({label}): '
            f'{first[0] if first else f"rc={r.returncode}"}')
    return False


def probe_jtag_jtagconfig(obs: Observation, tc: Toolchain) -> bool:
    exe = tc.quartus('jtagconfig')
    if exe is None:
        obs.notes.append('jtagconfig 不在(Quartus 未導入でも openFPGALoader が'
                         'あれば検出可能)')
        return False
    obs.probes_run.append('jtagconfig')
    try:
        r = subprocess.run([exe], capture_output=True, text=True,
                           errors='replace', timeout=30)
    except Exception as e:  # noqa: BLE001
        obs.notes.append(f'jtagconfig 実行失敗: {e}')
        return False
    cables, devices = parse_jtagconfig(r.stdout + r.stderr)
    if devices:
        obs.cables.extend(c for c in cables if c not in obs.cables)
        obs.jtag.extend(devices)
        return True
    msg = (r.stdout + r.stderr).strip()
    obs.notes.append(f'jtagconfig: {msg.splitlines()[0] if msg else "検出なし"}')
    return False


def run_probes(tc: Toolchain | None = None,
               probes: tuple = ('serial', 'usb', 'jtag')) -> Observation:
    tc = tc or discover()
    obs = Observation()
    if 'serial' in probes:
        probe_serial(obs)
    if 'usb' in probes:
        probe_usb(obs, tc)
    if 'jtag' in probes:
        # EDA 無し優先: openFPGALoader → 失敗時のみ jtagconfig
        if not probe_jtag_openfpgaloader(obs, tc):
            probe_jtag_jtagconfig(obs, tc)
    return obs


# ---------------------------------------------------------------- 照合

def _to_int(v) -> int | None:
    if isinstance(v, int):
        return v
    try:
        return int(str(v), 0)
    except (TypeError, ValueError):
        return None


def match_boards(obs: Observation, boards: list[dict]) -> list[BoardCandidate]:
    out: list[BoardCandidate] = []
    obs_idcodes = {d.idcode for d in obs.jtag}
    cable_texts = [c.lower().replace(' ', '-') for c in obs.cables]
    for board in boards:
        det = board.get('detection') or {}
        score = 0.0
        reasons: list[str] = []

        known = {normalize_idcode(x) for x in det.get('jtag_idcodes', [])}
        hit = (known - {None}) & obs_idcodes
        if hit:
            score += SCORE_IDCODE
            reasons.append(f'JTAG IDCODE 一致({", ".join(sorted(hit))})')

        part_re = det.get('jtag_part_regex')
        if part_re and any(re.search(part_re, d.part_text or '', re.I)
                           for d in obs.jtag):
            score += SCORE_PART
            reasons.append(f'部品名一致({part_re})')

        for sig in det.get('usb', []):
            vid, pid = _to_int(sig.get('vid')), _to_int(sig.get('pid'))
            desc_re = sig.get('desc_regex')
            for u in obs.usb:
                if vid is not None and u.vid != vid:
                    continue
                if pid is not None and u.pid != pid:
                    continue
                if desc_re and not re.search(desc_re, u.text, re.I):
                    continue
                score += SCORE_USB
                reasons.append(f'USB 署名一致({vid:#06x}:{pid:#06x})'
                               if vid is not None and pid is not None
                               else 'USB 署名一致')
                break
            else:
                continue
            break

        cable = det.get('cable')
        ofl_board = det.get('ofl_board')
        if ofl_board and any(ofl_board.lower() in c for c in cable_texts):
            # -b <ofl_board> で応答した=そのボード名指定で JTAG が通った
            score += SCORE_CABLE
            reasons.append(f'openFPGALoader ボード名一致({ofl_board})')
        elif cable and any(cable in c for c in cable_texts):
            score += SCORE_CABLE
            reasons.append(f'ケーブル種一致({cable})')

        if score > 0:
            out.append(BoardCandidate(
                key=board['key'], name=board['name'],
                tier=board.get('tier', 3),
                confidence=min(1.0, round(score, 2)),
                reasons=reasons,
                silk_hint=det.get('silk_hint'),
                unverified=bool(board.get('unverified'))
                or board.get('tier', 3) != 1))
    out.sort(key=lambda c: (-c.confidence, c.tier, c.key))
    return out


def run_detect(tc: Toolchain | None = None,
               probes: tuple = ('serial', 'usb', 'jtag')) -> DetectResult:
    """検出の一括実行(CLI とビューア API の共通入口。to_dict() が API 契約)。"""
    obs = run_probes(tc, probes)
    boards = [load_board(k) for k in list_boards()]
    return DetectResult(observation=obs, candidates=match_boards(obs, boards))
