"""デプロイバンドル生成(mdl-fpga bundle)。

build の成果物を「EDA の無い PC(Windows/macOS)で書込・検証できる」形に
梱包する: bitstream + board 定義 + golden ベクトル + flow_summary +
配線・書込手順 README の zip。

変換方針はバックエンド依存(backend.convert_formats):
- quartus: sof → svf/rbf 変換(quartus_cpf)をビルド PC 側で済ませる
  (書込 PC は openFPGALoader だけで済む)。cpf 不在時は sof のみ+WARN
- gowin-oss: .fs は openFPGALoader が直接書けるため変換なし
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .backends import get_backend
from .board_db import resolve_board
from .io_link import ResolvedUart, resolve_uart_link
from .manifest import load_manifest
from .toolchain import TOOL_DIR, Toolchain, discover

# vendor 同梱するランタイムのソース(STANDALONE.md。mdl 非依存を lint で保証)
RUNTIME_SRC = TOOL_DIR / 'runtime' / 'mdl_runtime'

_REQUIREMENTS = 'pyserial>=3.5\npygame>=2.5\n'

_RUN_PY = '''\
"""bundle 同梱アプリの入口(mdl_runtime CLI の薄い皮)。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mdl_runtime.cli import main  # noqa: E402

sys.exit(main(['--app', str(HERE), *sys.argv[1:]]))
'''


@dataclass
class BundleResult:
    ok: bool
    zip_path: str = ''
    contents: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'ok': self.ok, 'zip_path': self.zip_path,
                'contents': self.contents, 'messages': self.messages}

    def render(self) -> str:
        lines = [f'bundle: {"OK" if self.ok else "FAIL"}']
        if self.zip_path:
            lines.append(f'  zip: {self.zip_path}')
        for c in self.contents:
            lines.append(f'    - {c}')
        lines.extend(f'  {m}' for m in self.messages)
        return '\n'.join(lines)


def _find_bitstream(workdir: Path, backend) -> Path | None:
    """backend の探索パターンで主ビットストリームを見つける。"""
    for pat in backend.bitstream_globs():
        hits = sorted(workdir.glob(pat))
        if hits:
            return hits[0]
    return None


def _find_sof(workdir: Path) -> Path | None:
    """旧名互換(Quartus 前提の探索)。新規コードは _find_bitstream を使う。"""
    for pat in ('quartus/output_files/*.sof', 'output_files/*.sof', '*.sof'):
        hits = sorted(workdir.glob(pat))
        if hits:
            return hits[0]
    return None


def _is_builtin_uart(board: dict, uart: ResolvedUart) -> bool:
    """UART がボード内蔵ブリッジ(builtin ヘッダ)経由か(README の配線節切替)。"""
    for h in board.get('headers', []):
        if not h.get('builtin'):
            continue
        pins = {io.get('pin') for io in h.get('io', {}).values()}
        if uart.rx_pin in pins and uart.tx_pin in pins:
            return True
    return False


def _phys(v: int | None) -> str:
    return f'(コネクタ物理ピン {v})' if v is not None else ''


def _flasher_win_lines(ofl_cmd: str, has_app: bool,
                       driver_note: str) -> list[str]:
    """書き込み(Windows)節: 同梱 flasher を主線にした手順。"""
    lines = ['同梱の flasher を使う(ツール導入不要):', '']
    if has_app:
        lines += ['    python app\\run.py flash',
                  '',
                  '(`--persist` で SPI フラッシュへ永続書込=電源断でも保持。'
                  '対応ボードのみ)',
                  f'   直接実行する場合: `flasher\\windows\\{ofl_cmd}`']
    else:
        lines += [f'    flasher\\windows\\{ofl_cmd}', '']
    lines.append(driver_note)
    return lines


def _flasher_mac_lines(ofl_cmd: str, has_app: bool) -> list[str]:
    lines = ['同梱の flasher を使う(ツール導入不要):', '']
    if has_app:
        lines += ['    python3 app/run.py flash',
                  '',
                  f'   直接実行する場合: `flasher/darwin/{ofl_cmd}`']
    else:
        lines += [f'    flasher/darwin/{ofl_cmd}', '']
    return lines


def _readme(name: str, board: dict, uart: ResolvedUart,
            has_svf: bool, has_rbf: bool, has_golden: bool,
            has_app: bool = False,
            flasher_oses: frozenset = frozenset()) -> str:
    ofl_file = f'bitstream/{name}.svf' if has_svf else (
        f'bitstream/{name}.rbf' if has_rbf else None)
    lines = [
        f'# {name} 書込バンドル({board["name"]})',
        '',
        'mdl-fpga bundle が生成。この PC に EDA(Quartus)が無くても書き込める。',
        '',
        '## 内容',
        f'- bitstream/{name}.sof … Quartus Programmer 用',
    ]
    if has_svf:
        lines.append(f'- bitstream/{name}.svf … openFPGALoader 用(EDA 不要)')
    if has_rbf:
        lines.append(f'- bitstream/{name}.rbf … openFPGALoader 用(EDA 不要)')
    lines += [
        f'- board/{board["key"]}.json … ボード定義(配線の根拠)',
        '- flow_summary.json … 合成結果(ある場合)',
    ]
    if has_golden:
        lines.append('- golden/ … 実機 hw-check 用の入力/期待値ベクトル')
    driver_note = ('   ※ ドライバ注意: Quartus の USB-Blaster ドライバと '
                   'WinUSB(Zadig)は排他。openFPGALoader が掴めない場合は'
                   'ドライバを確認')
    lines += ['', '## 書き込み(Windows)', '']
    if ofl_file and 'windows' in flasher_oses:
        cmd = f'openFPGALoader.exe -c usb-blaster {ofl_file}'
        lines += _flasher_win_lines(cmd, has_app, driver_note)
    elif ofl_file:
        lines += [
            '1. oss-cad-suite(https://github.com/YosysHQ/oss-cad-suite-build)'
            'を展開',
            '2. コマンドプロンプトで:',
            '',
            f'       openFPGALoader.exe -c usb-blaster {ofl_file}',
            '',
            '   ※ DLL エラーが出る場合は oss-cad-suite の bin と lib の両方を'
            ' PATH に追加',
            driver_note,
        ]
    lines += [
        f'- 代替(Quartus Programmer 単体版): '
        f'`quartus_pgm -c 1 -m jtag -o "p;bitstream/{name}.sof"`',
        '',
        '## 書き込み(macOS)',
        '',
    ]
    if ofl_file and 'darwin' in flasher_oses:
        lines += _flasher_mac_lines(f'openFPGALoader -c usb-blaster {ofl_file}',
                                    has_app)
    elif ofl_file:
        lines += [
            '1. `brew install openfpgaloader`',
            f'2. `openFPGALoader -c usb-blaster {ofl_file}`',
            '',
            '   ※ Quartus は macOS 非対応だが、この経路なら EDA 完全不要',
        ]
    else:
        lines += ['(svf/rbf が未同梱のため macOS では書込不可。ビルド PC で '
                  'quartus_cpf による変換を行って bundle し直す)']
    rx = f'{uart.rx_pin}{_phys(uart.rx_physical)}'
    tx = f'{uart.tx_pin}{_phys(uart.tx_physical)}'
    lines += [
        '',
        '## UART 配線(hw-check する場合)',
        '',
        f'- ボード側 RX({uart.rx_port}): {rx} ← ブリッジの TXD',
        f'- ボード側 TX({uart.tx_port}): {tx} → ブリッジの RXD',
        f'- baud: {uart.baud}(8N1)',
        '- GND: USB-UART ブリッジの GND とボードの GND を必ず接続',
        '',
        '## 注意',
        '',
        '- SRAM 揮発コンフィグ: 電源を切ると消える(再書込が必要)',
        '- 書込後はボードのリセットスイッチ位置を確認(リセット中は動かない)',
    ]
    return '\n'.join(lines) + '\n'


def _readme_gowin(name: str, board: dict, uart: ResolvedUart,
                  has_golden: bool, has_app: bool = False,
                  flasher_oses: frozenset = frozenset()) -> str:
    det = board.get('detection') or {}
    ofl = (f'-b {det["ofl_board"]}' if det.get('ofl_board')
           else f'-c {det.get("cable", "ft2232")}')
    ext = 'fs'
    lines = [
        f'# {name} 書込バンドル({board["name"]})',
        '',
        'mdl-fpga bundle が生成。書込はオープンツールの openFPGALoader のみで'
        '完結する(ベンダ EDA 不要)。',
        '',
        '## 内容',
        f'- bitstream/{name}.{ext} … openFPGALoader 用(そのまま書込可)',
        f'- board/{board["key"]}.json … ボード定義(配線の根拠)',
        '- flow_summary.json … 合成結果(ある場合)',
    ]
    if has_golden:
        lines.append('- golden/ … 実機 hw-check 用の入力/期待値ベクトル')
    driver_note = ('   ※ ドライバ注意: FTDI 系は Zadig で該当 Interface のみ'
                   ' WinUSB 化(内蔵 UART 側の COM は残る)')
    ofl_cmd = f'openFPGALoader.exe {ofl} bitstream/{name}.{ext}'
    lines += ['', '## 書き込み(Windows)', '']
    if 'windows' in flasher_oses:
        lines += _flasher_win_lines(ofl_cmd.replace('/', '\\'), has_app,
                                    driver_note)
    else:
        lines += [
            '1. oss-cad-suite(https://github.com/YosysHQ/oss-cad-suite-build)'
            'を展開',
            '2. コマンドプロンプトで:',
            '',
            f'       {ofl_cmd}',
            '',
            '   ※ DLL エラーが出る場合は oss-cad-suite の bin と lib の両方を'
            ' PATH に追加',
            driver_note,
        ]
    lines += ['', '## 書き込み(macOS)', '']
    if 'darwin' in flasher_oses:
        lines += _flasher_mac_lines(
            f'openFPGALoader {ofl} bitstream/{name}.{ext}', has_app)
    else:
        lines += [
            '1. `brew install openfpgaloader`',
            f'2. `openFPGALoader {ofl} bitstream/{name}.{ext}`',
        ]
    lines += [
        '',
        '## UART(hw-check する場合)',
        '',
    ]
    if _is_builtin_uart(board, uart):
        lines += [
            '- 配線不要: ボード内蔵 USB-UART(BL616 等)に基板上で直結済み。'
            'USB ケーブル 1 本で書込と UART の両方が通る',
            f'- baud: {uart.baud}(8N1)',
        ]
    else:
        rx = f'{uart.rx_pin}{_phys(uart.rx_physical)}'
        tx = f'{uart.tx_pin}{_phys(uart.tx_physical)}'
        lines += [
            f'- ボード側 RX({uart.rx_port}): {rx} ← ブリッジの TXD',
            f'- ボード側 TX({uart.tx_port}): {tx} → ブリッジの RXD',
            f'- baud: {uart.baud}(8N1)',
            '- GND: USB-UART ブリッジの GND とボードの GND を必ず接続',
        ]
    lines += [
        '',
        '## 注意',
        '',
        '- SRAM 揮発コンフィグ: 電源を切ると消える(再書込が必要)',
        '- 書込後はボードのリセットボタン/スイッチ位置を確認',
    ]
    return '\n'.join(lines) + '\n'


def _app_texts(manifest: str | Path, board_def: dict, bitstream_arc: str,
               n_goldens: int) -> list[tuple[str, str]]:
    """app/ 同梱物(io_contract+goldens+runtime vendor+入口)。"""
    from .contract import build_io_contract
    contract, goldens = build_io_contract(
        manifest, board_def, bitstream_arc=bitstream_arc, n_goldens=n_goldens)
    texts = [
        (json.dumps(contract, ensure_ascii=False, indent=2) + '\n',
         'app/io_contract.json'),
        (json.dumps(goldens, ensure_ascii=False) + '\n', 'app/goldens.json'),
        (_REQUIREMENTS, 'app/requirements.txt'),
        (_RUN_PY, 'app/run.py'),
    ]
    for p in sorted(RUNTIME_SRC.rglob('*.py')):
        arc = 'app/mdl_runtime/' + p.relative_to(RUNTIME_SRC).as_posix()
        texts.append((p.read_text(encoding='utf-8'), arc))
    return texts


def _flasher_blobs(pack: Path) -> list[tuple[bytes, str]]:
    """flasher パック(dir か zip)→ flasher/ 配下のバイナリ列。"""
    if pack.is_dir():
        return [(p.read_bytes(), 'flasher/' + p.relative_to(pack).as_posix())
                for p in sorted(pack.rglob('*')) if p.is_file()]
    if pack.suffix.lower() == '.zip':
        out = []
        with zipfile.ZipFile(pack) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                out.append((zf.read(info),
                            'flasher/' + info.filename.lstrip('/')))
        return out
    raise ValueError(f'flasher パックは dir か .zip: {pack}')


def _readme_app_section(has_app: bool, has_flasher: bool) -> str:
    if not has_app:
        return ''
    lines = [
        '',
        '## スタンドアローンアプリ(app/)',
        '',
        '書込済みボードと PC を USB で繋げば、以下だけで動く'
        '(griffith-logic も EDA も不要):',
        '',
        '1. Python 3.11 以降を導入: https://www.python.org/downloads/',
        '   (Windows はインストーラの「Add python.exe to PATH」にチェック)',
        '2. 依存を導入: `pip install -r app/requirements.txt`',
        '3. 実行例:',
        '',
        '       python app/run.py flash        # 書込'
        '(同梱 flasher → PATH の順に自動解決)',
        '       python app/run.py selftest     # 実機 bit 一致セルフテスト',
        '       python app/run.py gui          # GUI(手描き/トグル+出力表示)',
        '       python app/run.py send 0x1f    # 1ベクトル送信(CUI)',
        '       python app/run.py serve-stdio  # 他フレームワーク連携'
        '(JSON 行 API)',
        '',
        '- シリアルポートは契約(app/io_contract.json)の USB 署名で自動検出。'
        '`--port COM5` 等で明示も可',
    ]
    if has_flasher:
        lines.append('- flasher/ に openFPGALoader を同梱済み'
                     '(ライセンスは flasher/ 内の LICENSE を参照)')
    else:
        lines.append('- flasher は非同梱: macOS `brew install openfpgaloader`'
                     ' / Windows は oss-cad-suite を展開して PATH へ')
    return '\n'.join(lines) + '\n'


def _local_flasher_pack() -> str | None:
    """local.toml の flasher_pack(開発機の既定同梱。テストでは差し替える)。"""
    from .toolchain import _load_local_config
    return _load_local_config().get('flasher_pack') or None


def run_bundle(workdir: str | Path, *, manifest: str | Path | None = None,
               board: str | None = None, out: str | Path | None = None,
               formats: tuple = ('svf', 'rbf'),
               app: bool = True, flasher: bool = True,
               flasher_pack: str | Path | None = None,
               n_goldens: int = 32,
               tc: Toolchain | None = None) -> BundleResult:
    tc = tc or discover()
    workdir = Path(workdir)
    res = BundleResult(ok=False)

    mf = None
    if manifest is not None:
        mf = load_manifest(manifest)
    board_key = board or (mf.board if mf else None) or 'de0-cv'
    board_def = resolve_board(board_key)
    backend = get_backend(board_def)
    uart = resolve_uart_link(board_def, mf.io.link if mf else None)

    bit = _find_bitstream(workdir, backend)
    if bit is None:
        pats = ' / '.join(backend.bitstream_globs())
        res.messages.append(
            f'{workdir} に {backend.bitstream_ext} が見つからない。'
            f'build 完了後の workdir({pats})を指定する。'
            'ボードが違う場合は --board で明示する')
        return res
    name = bit.stem

    # 変換(バックエンドの対応形式のみ。quartus_cpf 不在は WARN で続行)
    converted: dict[str, Path] = {}
    for fmt in formats:
        if fmt not in backend.convert_formats:
            res.messages.append(
                f'{fmt} 変換は {backend.name} では不要/非対応'
                f'({backend.bitstream_ext} を直接書き込む)')
            continue
        try:
            converted[fmt] = backend.convert_bitstream(bit, fmt, tc=tc)
        except FileNotFoundError as e:
            res.messages.append(f'WARN: {fmt} 変換をスキップ({e})')
            break                    # cpf 不在なら以降のフォーマットも不可
        except RuntimeError as e:
            res.messages.append(f'WARN: {fmt} 変換失敗({e})')

    zip_path = Path(out) if out else workdir / f'{name}_bundle.zip'
    ext = bit.suffix.lstrip('.')
    entries: list[tuple[Path, str]] = [(bit, f'bitstream/{name}.{ext}')]
    entries += [(p, f'bitstream/{p.name}') for p in converted.values()]

    # board は解決済み dict を書き出す(draft=パス指定でも成立)
    texts: list[tuple[str, str]] = [
        (json.dumps(board_def, ensure_ascii=False, indent=2) + '\n',
         f'board/{board_def["key"]}.json'),
    ]
    if manifest is not None:
        entries.append((Path(manifest), 'manifest.json'))
    for cand in (bit.parent.parent / 'flow_summary.json',
                 bit.parent / 'flow_summary.json'):
        if cand.exists():
            entries.append((cand, 'flow_summary.json'))
            break
    presim = workdir / 'presim'
    golden = sorted(presim.glob('*.hex')) if presim.exists() else []
    entries += [(p, f'golden/{p.name}') for p in golden]

    # --- app/(スタンドアローンランタイム同梱。STANDALONE.md)---
    # 契約のビットストリームは openFPGALoader で書ける形式を優先する
    # (quartus は svf/rbf 変換があればそれ、無ければ sof=Programmer 用)
    contract_arc = f'bitstream/{name}.{ext}'
    for fmt in ('svf', 'rbf'):
        if fmt in converted:
            contract_arc = f'bitstream/{converted[fmt].name}'
            break
    has_app = False
    if app:
        if manifest is None:
            res.messages.append(
                'app 同梱をスキップ: io_contract 生成に manifest が必要'
                '(--manifest を指定するか --no-app)')
        else:
            texts += _app_texts(manifest, board_def, contract_arc, n_goldens)
            has_app = True

    # --- flasher/(openFPGALoader 同梱。既定同梱・オプトアウト=--no-flasher)---
    # 解決順: 引数 → 環境変数 → local.toml の flasher_pack(開発機の既定)
    blobs: list[tuple[bytes, str]] = []
    pack = None
    if flasher:
        pack = (flasher_pack or os.environ.get('MDL_FPGA_FLASHER_PACK')
                or _local_flasher_pack())
    if pack:
        try:
            blobs = _flasher_blobs(Path(pack))
            res.messages.append(f'flasher 同梱: {pack}({len(blobs)} ファイル)')
        except (OSError, ValueError) as e:
            res.messages.append(f'WARN: flasher パック読込失敗({e})。'
                                'README の導入誘導に降格')
    elif has_app and flasher:
        res.messages.append(
            'flasher 非同梱(パック未指定)。--flasher-pack <dir|zip> / '
            '環境変数 MDL_FPGA_FLASHER_PACK / local.toml の flasher_pack で'
            '同梱できる(README に導入誘導あり)')

    flasher_oses = frozenset(
        {'windows', 'linux', 'darwin'}
        & {a.split('/')[1] for _, a in blobs if a.count('/') >= 2})
    if backend.name == 'gowin-oss':
        readme = _readme_gowin(name, board_def, uart,
                               has_golden=bool(golden), has_app=has_app,
                               flasher_oses=flasher_oses)
    else:
        readme = _readme(name, board_def, uart,
                         has_svf='svf' in converted,
                         has_rbf='rbf' in converted,
                         has_golden=bool(golden), has_app=has_app,
                         flasher_oses=flasher_oses)
    readme += _readme_app_section(has_app, bool(blobs))

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            zf.write(src, arc)
        for content, arc in texts:
            zf.writestr(arc, content)
        for data, arc in blobs:
            zf.writestr(arc, data)
        zf.writestr('README.md', readme)
    res.zip_path = str(zip_path)
    res.contents = ([arc for _, arc in entries]
                    + [arc for _, arc in texts]
                    + [arc for _, arc in blobs] + ['README.md'])
    res.ok = True
    res.messages.append(
        '書込 PC 側の使い方は zip 内 README.md(Windows/macOS 両対応)')
    if backend.name == 'quartus' and \
            'svf' not in converted and 'rbf' not in converted and formats:
        res.messages.append(
            '注意: svf/rbf 未同梱のため書込 PC に Quartus Programmer が必要')
    return res


def extract_wiring_note(board_key: str,
                        manifest: str | Path | None = None) -> str:
    """hw_check などが使う配線案内(README と同一ソース)。"""
    mf = load_manifest(manifest) if manifest else None
    board_def = resolve_board(board_key)
    uart = resolve_uart_link(board_def, mf.io.link if mf else None)
    return (f'RX({uart.rx_port})={uart.rx_pin}{_phys(uart.rx_physical)} / '
            f'TX({uart.tx_port})={uart.tx_pin}{_phys(uart.tx_physical)} / '
            f'baud {uart.baud}')
