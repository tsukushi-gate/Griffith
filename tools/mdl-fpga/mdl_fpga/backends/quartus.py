"""Quartus 合成バックエンド(.qsf/.sdc 生成 → map/fit/asm/sta → レポートパース)。

旧 mdl_fpga/synth.py からの機械的移設(挙動不変。tests/goldens/de0cv/ の
qsf/sdc ゴールデンと byte 一致を維持する)。SDC のクロック網名 clk50 は
本バックエンド内部のラベル(ゴールデン依存のため周波数に関わらず固定)。

demo1 実証フローの仕様化。設計注記(構想書 §4): demo の run_flow.py は
プロジェクト名固定で流用不可だった実証済みの落とし穴のため、本実装は
プロジェクト名・ディレクトリを最初から可変にする。

パーサはベンダ版でフォーマットが動く前提の防御的実装(取れない値は None)。
生ログは run ディレクトリに全保存する(§9)。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from ..toolchain import Toolchain, discover
from .base import FlowResult, write_summary

STAGES = ('quartus_map', 'quartus_fit', 'quartus_asm', 'quartus_sta')

# STA タイミングコーナー(create_timing_netlist の引数)。Cyclone V は従来の
# 実証値を維持。未収載 family は電圧/温度を推測記入せず、コーナー指定なしで
# ツール既定の slow コーナーに任せる(値の創作をしない)
_STA_CORNERS = {
    'Cyclone V': '-model slow -temperature 85 -voltage 1100',
}
_STA_CORNER_DEFAULT = '-model slow'

_STA_TCL = """project_open {name} -revision {name}
create_timing_netlist {corner}
read_sdc {name}.sdc
update_timing_netlist
set paths [get_timing_paths -setup -npaths 1 -to_clock clk50]
foreach_in_collection pth $paths {{ puts "SLACK clk50 [get_path_info $pth -slack]" }}
delete_timing_netlist
project_close
"""

# 派生クロック(clk_core)がある構成用。既存の _STA_TCL は clk50 固定のまま
# 残す(ゴールデン相当の既存テストが直接 format しているため)。
_STA_TCL_MULTI = """project_open {name} -revision {name}
create_timing_netlist {corner}
read_sdc {name}.sdc
update_timing_netlist
foreach clk {{{clocks}}} {{
    set paths [get_timing_paths -setup -npaths 1 -to_clock $clk]
    foreach_in_collection pth $paths {{ puts "SLACK $clk [get_path_info $pth -slack]" }}
}}
delete_timing_netlist
project_close
"""

# 既知の失敗パターン → 初心者向け診断(原因・次の一手)
_KNOWN_ERRORS = (
    ('276003', '合成が論理を実装できない(demo2 実証の Error 276003 系)。'
               'モデル規模がボードに対して大きすぎる可能性。advisor の判定を確認し、'
               'モデル縮小か overlay 方式を検討'),
    ('Error (171000)', 'フィッタが配置配線に失敗。配線天井(ALM 31-48%で破綻する実測則)の'
                       '可能性。モデル縮小を検討'),
    ('No valid license', 'Quartus のライセンス問題。Lite Edition はライセンス不要のため、'
                         'エディション/インストールを doctor で確認'),
)


def write_project_files(projdir: str | Path, *, name: str, board: dict,
                        sources: list[str], clock_hz: int = 50_000_000,
                        groups: tuple = ('clock', 'uart', 'led', 'switch'),
                        uart: dict | None = None,
                        clock_div: int = 1) -> None:
    """qpf / qsf / sdc を生成する(ピン・デバイスはボード DB から)。

    groups で使用ポート群を選ぶ(トップの実ポートに合わせる):
        'clock' 'uart' 'led' 'switch' 'button'(KEY) 'hex'(HEX0-5)
    uart のピンは board JSON(v2 は物理事実のみ)ではなく manifest の io.link
    から解決した値を引数で受ける(io_link.resolve_uart_link(...).to_uart_dict())。
    既定タプル+従来相当 uart の出力は従来と byte 一致(ゴールデンで固定)。

    clock_div > 1 のときは内部で /N 分周した clk_core を派生クロックとして
    SDC に宣言する(オーバーレイの 25MHz 動作。PLL を使わないトグル FF 分周)。
    clock_div=1(既定)では SDC は従来と byte 一致する。
    """
    projdir = Path(projdir)
    projdir.mkdir(parents=True, exist_ok=True)
    for g in groups:
        if g not in ('clock', 'uart', 'led', 'switch', 'button', 'hex'):
            raise KeyError(f'未知のポート群: {g}')
        if g == 'uart':
            if uart is None:
                raise KeyError(
                    'uart グループには uart 引数が必要'
                    '(io_link.resolve_uart_link(board, link).to_uart_dict())。'
                    'board JSON v2 は uart ピンを持たない')
        elif g != 'clock' and g not in board:
            raise KeyError(
                f'ボードDBに {g} が未定義({board.get("key")})。'
                'boards/*.json を拡張してください')
    assert 'clock' in groups, 'clock は必須'

    (projdir / f'{name}.qpf').write_text(
        f'PROJECT_REVISION = "{name}"\n', encoding='utf-8')

    io_std = board['io_standard']
    q = [
        '# mdl-fpga 自動生成 QSF(ピン表はボードDB=2ソース突合済み実測値)',
        f'set_global_assignment -name FAMILY "{board["family"]}"',
        f'set_global_assignment -name DEVICE {board["device"]}',
        f'set_global_assignment -name TOP_LEVEL_ENTITY {name}',
        'set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files',
        'set_global_assignment -name AUTO_SHIFT_REGISTER_RECOGNITION OFF',
        'set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL',
        'set_global_assignment -name RESERVE_ALL_UNUSED_PINS_WEAK_PULLUP '
        '"AS INPUT TRI-STATED WITH WEAK PULL-UP"',
    ]
    q.append(f'set_location_assignment {board["clock"]["pin"]} '
             f'-to {board["clock"]["port"]}')
    if 'uart' in groups:
        q.append(f'set_location_assignment {uart["rx"]["pin"]} '
                 f'-to {uart["rx"]["port"]}')
        q.append(f'set_location_assignment {uart["tx"]["pin"]} '
                 f'-to {uart["tx"]["port"]}')
    if 'led' in groups:
        for i, pin in enumerate(board['led']['pins']):
            q.append(f'set_location_assignment {pin} '
                     f'-to {board["led"]["port"]}[{i}]')
    if 'switch' in groups:
        for i, pin in enumerate(board['switch']['pins']):
            q.append(f'set_location_assignment {pin} '
                     f'-to {board["switch"]["port"]}[{i}]')
    if 'button' in groups:
        for i, pin in enumerate(board['button']['pins']):
            q.append(f'set_location_assignment {pin} '
                     f'-to {board["button"]["port"]}[{i}]')
    if 'hex' in groups:
        for digit in board['hex']['digits']:
            for i, pin in enumerate(digit['pins']):
                q.append(f'set_location_assignment {pin} '
                         f'-to {digit["port"]}[{i}]')
    if 'uart' in groups:
        # UART_RXD 弱プルアップ(ケーブル未接続時の誤スタートビット防止、demo 実証)
        q.append(f'set_instance_assignment -name WEAK_PULL_UP_RESISTOR ON '
                 f'-to {uart["rx"]["port"]}')
    io_ports: list[str] = [board['clock']['port']]
    if 'uart' in groups:
        io_ports += [uart['rx']['port'], uart['tx']['port']]
    if 'led' in groups:
        io_ports.append(f'{board["led"]["port"]}[*]')
    if 'switch' in groups:
        io_ports.append(f'{board["switch"]["port"]}[*]')
    if 'button' in groups:
        io_ports.append(f'{board["button"]["port"]}[*]')
    if 'hex' in groups:
        io_ports += [f'{d["port"]}[*]' for d in board['hex']['digits']]
    for port in io_ports:
        q.append(f'set_instance_assignment -name IO_STANDARD "{io_std}" -to {port}')
    for src in sources:
        q.append(f'set_global_assignment -name VERILOG_FILE {src}')
    q.append(f'set_global_assignment -name SDC_FILE {name}.sdc')
    (projdir / f'{name}.qsf').write_text('\n'.join(q) + '\n', encoding='utf-8')

    period_ns = 1e9 / clock_hz
    clk_port = board['clock']['port']
    sdc = [
        f'create_clock -name clk50 -period {period_ns:.3f} [get_ports {clk_port}]',
    ]
    if clock_div > 1:
        sdc.append(
            'create_generated_clock -name clk_core -source '
            f'[get_ports {clk_port}] -divide_by {clock_div} '
            '[get_registers {clk_core}]')
    sdc.append('derive_clock_uncertainty')
    if 'switch' in groups:
        sdc.append(
            f'set_false_path -from [get_ports {{{board["switch"]["port"]}[*]}}]')
    if 'led' in groups:
        sdc.append(
            f'set_false_path -to   [get_ports {{{board["led"]["port"]}[*]}}]')
    if 'uart' in groups:
        sdc.append(
            f'set_false_path -from [get_ports {uart["rx"]["port"]}]')
        sdc.append(
            f'set_false_path -to   [get_ports {uart["tx"]["port"]}]')
    if 'button' in groups:
        sdc.append(
            f'set_false_path -from [get_ports {{{board["button"]["port"]}[*]}}]')
    if 'hex' in groups:
        for d in board['hex']['digits']:
            sdc.append(f'set_false_path -to   [get_ports {{{d["port"]}[*]}}]')
    (projdir / f'{name}.sdc').write_text('\n'.join(sdc) + '\n', encoding='utf-8')


def _cpf_args(fmt: str, sof: Path, out: Path) -> list[str]:
    """quartus_cpf の引数組立(純関数、テストで固定)。

    rbf: JTAG はリアルタイム展開不可のため圧縮を必ず無効化(openFPGALoader
    文書の要件)。svf: openFPGALoader の JTAG 再生用(12MHz / 3.3V / p=SRAM)。
    """
    if fmt == 'rbf':
        return ['-c', '--option=bitstream_compression=off',
                str(sof), str(out)]
    if fmt == 'svf':
        return ['-c', '-q', '12.0MHz', '-g', '3.3', '-n', 'p',
                str(sof), str(out)]
    raise ValueError(f'未対応フォーマット: {fmt}(svf | rbf)')


def convert_sof(sof: str | Path, fmt: str, out: str | Path | None = None,
                tc: Toolchain | None = None) -> Path:
    """.sof → .svf/.rbf(quartus_cpf)。EDA 無し PC で書き込むための変換。"""
    tc = tc or discover()
    # cwd=sof.parent で実行するため相対パスのままだと 213009 で読めない(実測)
    sof = Path(sof).resolve()
    out_path = (Path(out) if out else sof.with_suffix(f'.{fmt}')).resolve()
    if tc.quartus('quartus_cpf') is None:
        raise FileNotFoundError(
            'quartus_cpf が見つからない(doctor 参照)。変換なしでも '
            'bundle は sof のみで継続できる')
    proc = tc.run_quartus('quartus_cpf', _cpf_args(fmt, sof, out_path),
                          cwd=sof.parent, timeout=600)
    if proc.returncode != 0 or not out_path.exists():
        tail = ((proc.stdout or '') + (proc.stderr or '')).strip()[-500:]
        raise RuntimeError(f'quartus_cpf 失敗(fmt={fmt}): {tail}')
    return out_path


def parse_fit_summary(text: str) -> dict:
    """output_files/<name>.fit.summary → 資源使用量(demo run_flow.py と同regex)。"""
    fields = {
        'alm': r'Logic utilization \(in ALMs\)',
        'le': r'Total logic elements',       # MAX 10 等 LE 系 family の行
        'registers': r'Total registers',
        'pins': r'Total pins',
        'ram_bits': r'Total block memory bits',
        'ram_blocks': r'Total RAM Blocks',
        'dsp': r'Total DSP Blocks',
    }
    out: dict = {}
    for key, label in fields.items():
        m = re.search(
            label + r'\s*:\s*([\d,]+)(?:\s*/\s*([\d,]+))?(?:\s*\(\s*([\d.]+)\s*%)?',
            text, re.I)
        if not m:
            continue
        out[key] = int(m.group(1).replace(',', ''))
        if m.group(2):
            out[f'{key}_avail'] = int(m.group(2).replace(',', ''))
        if m.group(3):
            out[f'{key}_pct'] = float(m.group(3))
    return out


def parse_fmax(text: str) -> dict:
    # コーナー表記は family で変わる(Cyclone V=1100mV / MAX 10 等)ため
    # 電圧/温度部分を一般化(既存の Cyclone V サンプルはそのまま一致)
    m = re.search(
        r'Slow (?:\d+mV )?(?:\d+C )?Model Fmax Summary.*?;\s*([\d.]+)\s*MHz\s*;'
        r'\s*([\d.]+)\s*MHz\s*;\s*clk50\s*;',
        text, re.S)
    if not m:
        return {}
    return {'fmax_clk50_mhz': float(m.group(1)),
            'fmax_restricted_clk50_mhz': float(m.group(2))}


def parse_fmax_all(text: str) -> dict:
    """Fmax Summary の表から全クロックの Fmax を拾う(clk_core 等)。

    レポート冒頭の目次にも同じ見出しが出るため、**行が取れた最初の節**を使う
    (= Slow モデルの最初のコーナー)。clk50 の 2 キーは parse_fmax が返すものと
    同じ名前になる(従来互換)。
    """
    out: dict = {}
    pattern = (r';\s*([\d.]+)\s*MHz\s*;\s*([\d.]+)\s*MHz\s*;'
               r'\s*([\w.$:|\[\]-]+)\s*;')
    for m in re.finditer(r'Model Fmax Summary', text):
        rows = re.findall(pattern, text[m.end():m.end() + 4000])
        if not rows:
            continue
        for fmax, restricted, clk in rows:
            out[f'fmax_{clk}_mhz'] = float(fmax)
            out[f'fmax_restricted_{clk}_mhz'] = float(restricted)
        return out
    return out


def parse_fit_entities(text: str) -> list[dict]:
    """fit.rpt の "Fitter Resource Utilization by Entity" 表を読む。

    エンティティ別の M10K / ALM / Block Memory Bits を返す(取れない値は None)。
    ゲート ROM 単体のブロック数(計算型配線の効率の証拠)をここから出す。

    装置検証: 社内先行実験の実 fit.rpt 3 本に対して
    summarize_entity_m10k と合わせ、報告値(48K V8: ROM 32 / 活性 14 /
    画像 78 / 計 125、96K V8: 48 / 20 / 78 / 147、48K V32: 26 / 64 / 78 / 169)
    を再現することを確認済み。
    """
    i = text.find('; Fitter Resource Utilization by Entity')
    if i < 0:
        return []
    lines = text[i:].splitlines()
    hdr = next((k for k, ln in enumerate(lines)
                if 'Compilation Hierarchy Node' in ln), None)
    if hdr is None:
        return []
    cols = [c.strip() for c in lines[hdr].split(';')]
    want: dict = {}
    for label, key in (('ALMs needed [=A-B+C]', 'alm'),
                       ('Block Memory Bits', 'mem_bits'),
                       ('M10Ks', 'm10k'),
                       ('Dedicated Logic Registers', 'registers')):
        for j, c in enumerate(cols):
            if c == label:
                want[key] = j
                break
    if not want:
        return []
    rows: list[dict] = []
    for ln in lines[hdr + 1:]:
        if ln.startswith('+'):
            continue
        if not ln.startswith(';'):
            break
        f = [c.strip() for c in ln.split(';')]
        if len(f) <= max(want.values()):
            continue
        if not f[1].strip():
            continue
        rec: dict = {'node': f[1].strip()}
        for key, j in want.items():
            mm = re.match(r'^([\d.]+)', f[j])
            rec[key] = float(mm.group(1)) if mm else None
        rows.append(rec)
    return rows


def summarize_entity_m10k(rows: list[dict]) -> dict:
    """エンティティ別 M10K を用途別に束ねる(子ノードの二重計上を避ける)。

    node 名(先頭の | を除いた局所名)の接頭辞で分類する: rom32 / rom40 =
    ゲート ROM(保存型はレーンごとに 1 本。殻の gate_rom: は数えないので
    二重計上にならない)、tdp_ram = 活性バッファ、altsyncram:img_rom =
    画像 ROM、altsyncram:exp_rom = 期待値 ROM。total は階層の根(表の先頭行)。
    """
    buckets = {'gate_rom': ('rom32:', 'rom40:'), 'activation': ('tdp_ram:',),
               'image_rom': ('altsyncram:img_rom',),
               'label_rom': ('altsyncram:exp_rom',)}
    out = {k: 0.0 for k in buckets}
    for r in rows:
        name = r['node'].lstrip('|')
        for key, prefixes in buckets.items():
            if name.startswith(prefixes):
                out[key] += r.get('m10k') or 0.0
    result = {k: int(v) for k, v in out.items()}
    if rows:
        result['total'] = int(rows[0].get('m10k') or 0)
    return result


def diagnose_failure(log: str) -> str | None:
    for needle, advice in _KNOWN_ERRORS:
        if needle in log:
            return advice
    return None


def run_flow(projdir: str | Path, name: str, tc: Toolchain | None = None,
             timeout_per_stage: int = 3600,
             family: str | None = None,
             clock_div: int = 1) -> FlowResult:
    """map → fit → asm → sta → slack tcl を実行し flow_summary.json を書く。

    family は STA コーナー選択に使う(省略時は Cyclone V の従来動作)。
    clock_div > 1 のときは派生クロック clk_core のスラックも取る。
    """
    tc = tc or discover()
    projdir = Path(projdir)
    res = FlowResult(ok=False, projdir=str(projdir), name=name)

    for exe in STAGES:
        t0 = time.time()
        proc = tc.run_quartus(exe, [name], cwd=projdir,
                              timeout=timeout_per_stage)
        dt = time.time() - t0
        stage_log = (proc.stdout or '') + (proc.stderr or '')
        (projdir / f'{exe}.log').write_text(stage_log, encoding='utf-8',
                                            errors='replace')
        res.stages.append({'stage': exe, 'rc': proc.returncode, 'seconds': dt})
        if proc.returncode != 0:
            res.diagnosis = diagnose_failure(stage_log) or (
                f'{exe} が失敗。生ログ: {projdir / (exe + ".log")}。'
                'GitHub issue には flow_summary.json とこのログを添付')
            write_summary(projdir, res)
            return res

    # スラック抽出(2回目の STA を tcl スクリプトモードで)
    corner = _STA_CORNERS.get(family or 'Cyclone V', _STA_CORNER_DEFAULT)
    if clock_div > 1:
        tcl = _STA_TCL_MULTI.format(name=name, corner=corner,
                                    clocks='clk50 clk_core')
    else:
        tcl = _STA_TCL.format(name=name, corner=corner)
    (projdir / 'sta_slack.tcl').write_text(tcl, encoding='utf-8')
    proc = tc.run_quartus('quartus_sta', ['-t', 'sta_slack.tcl'], cwd=projdir,
                          timeout=timeout_per_stage)
    sta_out = (proc.stdout or '') + (proc.stderr or '')
    (projdir / 'sta_slack_out.txt').write_text(sta_out, encoding='utf-8',
                                               errors='replace')
    for clk, slack in re.findall(r'SLACK\s+(\S+)\s+([-\d.]+)', sta_out):
        res.summary[f'setup_slack_{clk}_ns'] = float(slack)

    fit_summary = projdir / 'output_files' / f'{name}.fit.summary'
    if fit_summary.exists():
        res.summary.update(parse_fit_summary(
            fit_summary.read_text(encoding='utf-8', errors='replace')))
    fit_rpt = projdir / 'output_files' / f'{name}.fit.rpt'
    if fit_rpt.exists():
        rows = parse_fit_entities(
            fit_rpt.read_text(encoding='utf-8', errors='replace'))
        if rows:
            res.summary['entities'] = [r for r in rows
                                       if (r.get('m10k') or 0) > 0]
            res.summary['m10k_by_entity'] = summarize_entity_m10k(rows)
    sta_rpt = projdir / 'output_files' / f'{name}.sta.rpt'
    if sta_rpt.exists():
        sta_text = sta_rpt.read_text(encoding='utf-8', errors='replace')
        res.summary.update(parse_fmax(sta_text))
        res.summary.update(parse_fmax_all(sta_text))

    sof = projdir / 'output_files' / f'{name}.sof'
    if sof.exists():
        res.sof = str(sof)
        res.ok = True
    else:
        res.diagnosis = 'asm 完了したが .sof が見つからない(ツール内部エラー)'
    write_summary(projdir, res)
    return res


class QuartusBackend:
    """SynthBackend 実装(Intel Quartus Lite/Standard)。

    実体は上のモジュール関数群(旧 synth.py)への薄い委譲。
    """
    name = 'quartus'
    bitstream_ext = '.sof'
    project_subdir = 'quartus'
    convert_formats = ('svf', 'rbf')

    def available(self, tc: Toolchain) -> tuple[bool, str]:
        if tc.quartus('quartus_map') is None:
            return False, ('Quartus が見つからないため合成をスキップ。'
                           'doctor で導入状況を確認(プロジェクトファイルは'
                           '生成済みなので Quartus 導入後にそのまま合成可能)')
        return True, ''

    def write_project_files(self, projdir: str | Path, *, name: str,
                            board: dict, sources: list[str],
                            clock_hz: int, groups: tuple,
                            uart: dict | None, clock_div: int = 1) -> None:
        write_project_files(projdir, name=name, board=board, sources=sources,
                            clock_hz=clock_hz, groups=groups, uart=uart,
                            clock_div=clock_div)

    def run_flow(self, projdir: str | Path, name: str,
                 tc: Toolchain | None = None, *, board: dict,
                 clock_hz: int, clock_div: int = 1) -> FlowResult:
        return run_flow(projdir, name, tc=tc, family=board.get('family'),
                        clock_div=clock_div)

    def convert_bitstream(self, bitstream: str | Path, fmt: str,
                          tc: Toolchain | None = None) -> Path:
        return convert_sof(bitstream, fmt, tc=tc)

    def bitstream_globs(self, name_hint: str = '*') -> tuple[str, ...]:
        return (f'quartus/output_files/{name_hint}.sof',
                f'output_files/{name_hint}.sof',
                f'{name_hint}.sof')
