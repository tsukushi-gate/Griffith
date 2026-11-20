"""gowin-oss 合成バックエンド(yosys → nextpnr-himbaechel → gowin_pack → .fs)。

オープンツールチェーンのみで完結する(ベンダ EDA 不要。全ツール oss-cad-suite
同梱)。Tang Nano 20K(GW2AR-18C)が最初のターゲット。

- 制約は .cst(IO_LOC/IO_PORT)。qsf と同じく board dict+resolve 済み uart から生成
- タイミング目標は nextpnr --freq(SDC は使わない)
- .fs は openFPGALoader が直接書けるため形式変換なし(convert_formats=())
- 引数の実在確認: nextpnr-himbaechel --vopt help / gowin_pack --help の
  実出力に基づく(--vopt family/cst/sspi_as_gpio、gowin_pack -d/-o/--*_as_gpio)

**既知の罠(実機実測、Tang Nano 20K full_adder)**:
wide-LUT(MUX2_LUT5-8)パッキング使用時、nextpnr-himbaechel(gowin)が
**配置依存で機能不良ビットストリームを生成することがある**。実測 A/B:
widelut+既定seed+selftest → y[1]=正解⊕x[0] の決定論的化け(Fmax 188MHz
PASS のまま)/ widelut+seed7+selftest → 正常 / widelut+seed7+selftest無し
→ 化け再発 / **nowidelut+既定seed → 正常**。yosys ネットリスト自体は
シム一致(presim PASS)のため、MUX2 系スライスパッキング起因とみられる。
対策: 本バックエンドは **synth_gowin -nowidelut を既定**にする(LUT4 数は
1割ほど増えるが正しさ優先)。実験用に環境変数 MDL_FPGA_GOWIN_WIDELUT=1 で
wide-LUT を再有効化、MDL_FPGA_NEXTPNR_SEED で配置 seed を変更できる。
使用値は flow_summary.json(nextpnr_seed)/ gowin_build.json(nowidelut)に
常に記録される。いずれにせよ実機受入は hw-check(bit 一致)まで必須。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from ..toolchain import Toolchain, discover
from .base import FlowResult, write_summary

STAGES = ('yosys', 'nextpnr', 'gowin_pack')

# 既知の失敗パターン → 初心者向け診断(原因・次の一手)
_KNOWN_ERRORS = (
    ('failed to route', 'nextpnr が配線に失敗。モデル規模がボードに対して大きい'
                        '可能性。advisor の判定を確認しモデル縮小を検討'),
    ('Unable to place', 'nextpnr が配置に失敗。リソース超過の可能性。'
                        'advisor の判定を確認しモデル縮小を検討'),
    ('failed to place', 'nextpnr が配置に失敗。リソース超過の可能性。'
                        'advisor の判定を確認しモデル縮小を検討'),
    ('chipdb', 'nextpnr の Gowin chipdb が見つからない/読めない。'
               'oss-cad-suite を最新版に更新する(doctor 参照)'),
    ('ERROR:', 'yosys/nextpnr がエラーで停止。生ログを確認'
               '(段別 .log を run ディレクトリに保存済み)'),
)


def _io_type(board: dict) -> str:
    """CST の IO_TYPE 値。gowin ボードは io_standard に Gowin 表記
    (LVCMOS33 等)をそのまま記録する契約。"""
    return board['io_standard']


def write_project_files(projdir: str | Path, *, name: str, board: dict,
                        sources: list[str], clock_hz: int = 27_000_000,
                        groups: tuple = ('clock', 'uart', 'led', 'button'),
                        uart: dict | None = None) -> None:
    """{name}.cst / {name}.ys / gowin_build.json を生成する。

    groups の意味は quartus バックエンドと同一('clock' 'uart' 'led'
    'switch' 'button' 'hex')。ピン・デバイスはボード DB(2ソース突合済み
    転記値)から。
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

    io_type = _io_type(board)
    cst: list[str] = [
        '// mdl-fpga 自動生成 CST(ピン表はボードDB=2ソース突合済み転記値)',
    ]

    def loc(port: str, pin: str, *extra: str) -> None:
        cst.append(f'IO_LOC "{port}" {pin};')
        attrs = ' '.join([f'IO_TYPE={io_type}', *extra])
        cst.append(f'IO_PORT "{port}" {attrs};')

    loc(board['clock']['port'], board['clock']['pin'])
    if 'uart' in groups:
        # RX 弱プルアップ(未接続時の誤スタートビット防止。qsf と同方針)
        loc(uart['rx']['port'], uart['rx']['pin'], 'PULL_MODE=UP')
        loc(uart['tx']['port'], uart['tx']['pin'])
    if 'led' in groups:
        for i, pin in enumerate(board['led']['pins']):
            loc(f'{board["led"]["port"]}[{i}]', pin)
    if 'switch' in groups:
        for i, pin in enumerate(board['switch']['pins']):
            loc(f'{board["switch"]["port"]}[{i}]', pin)
    if 'button' in groups:
        # プルは idle 側へ: 負論理(idle=High)→UP / 正論理(idle=Low)→DOWN
        # (Tang Nano 20K は正論理を実機 A/B 実験で確定。board JSON の note 参照)
        pull = 'PULL_MODE=UP' if board['button'].get('active_low') \
            else 'PULL_MODE=DOWN'
        for i, pin in enumerate(board['button']['pins']):
            loc(f'{board["button"]["port"]}[{i}]', pin, pull)
    if 'hex' in groups:
        for digit in board['hex']['digits']:
            for i, pin in enumerate(digit['pins']):
                loc(f'{digit["port"]}[{i}]', pin)
    (projdir / f'{name}.cst').write_text('\n'.join(cst) + '\n',
                                         encoding='utf-8')

    # wide-LUT は配置依存の機能不良の実測があるため既定で無効
    # (モジュール docstring の既知の罠を参照。実験用に env で再有効化可)
    widelut = os.environ.get('MDL_FPGA_GOWIN_WIDELUT') == '1'
    synth_opts = '' if widelut else ' -nowidelut'
    ys = ['# mdl-fpga 自動生成 yosys スクリプト(gowin-oss バックエンド)']
    ys += [f'read_verilog {src}' for src in sources]
    ys.append(f'synth_gowin -top {name}{synth_opts} -json {name}.json')
    (projdir / f'{name}.ys').write_text('\n'.join(ys) + '\n', encoding='utf-8')

    opts = board.get('synth_options', {})
    build_doc = {
        'backend': 'gowin-oss',
        'top': name,
        'sources': sources,
        'device': board['device'],
        'family': board['family'],
        'cst': f'{name}.cst',
        'clock_hz': clock_hz,
        'nowidelut': not widelut,
        'nextpnr_vopts': list(opts.get('nextpnr_vopts', [])),
        'gowin_pack_flags': list(opts.get('gowin_pack_flags', [])),
    }
    (projdir / 'gowin_build.json').write_text(
        json.dumps(build_doc, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_nextpnr_report(doc: dict) -> dict:
    """nextpnr --report の JSON → summary 断片(防御的。取れない値は落とす)。

    bel 名は GW2A-18C の実レポート(full_adder 実走)に基づく:
    素 LUT は `LUT4`(MUX2_LUT5/6/7/8 は LUT 合成木なので合算しない)、
    FF は `DFF`、キャリー系は `ALU`(Gowin はここにも論理が載るため
    advisor 較正の参考として別掲する)。
    """
    out: dict = {}
    util = doc.get('utilization', {})
    lut = lut_avail = ff = ff_avail = alu = alu_avail = 0
    for bel, v in util.items():
        used = v.get('used', 0) if isinstance(v, dict) else 0
        avail = v.get('available', 0) if isinstance(v, dict) else 0
        if re.fullmatch(r'LUT\d+', bel):          # LUT4 等(MUX2_LUT* は除外)
            lut += used
            lut_avail += avail
        elif re.fullmatch(r'DFF\w*', bel):
            ff += used
            ff_avail += avail
        elif bel == 'ALU':
            alu += used
            alu_avail += avail
    if lut_avail:
        out['lut'] = lut
        out['lut_avail'] = lut_avail
        out['lut_pct'] = round(100.0 * lut / lut_avail, 2)
    if ff_avail:
        out['ff'] = ff
        out['ff_avail'] = ff_avail
    if alu_avail:
        out['alu'] = alu
        out['alu_avail'] = alu_avail
    fmax = doc.get('fmax', {})
    achieved: list[float] = []
    ok = True
    for clk, v in fmax.items():
        if not isinstance(v, dict) or 'achieved' not in v:
            continue
        a = float(v['achieved'])
        achieved.append(a)
        key = re.sub(r'\W+', '_', clk).strip('_')
        out[f'fmax_{key}_mhz'] = a
        c = v.get('constraint')
        if c is not None:
            out.setdefault('freq_target_mhz', float(c))
            if a < float(c):
                ok = False
    if achieved:
        out['fmax_mhz'] = min(achieved)
        out['timing_ok'] = ok
    return out


_FMAX_LOG_RE = re.compile(
    r"Max frequency for clock\s+'([^']+)':\s*([\d.]+)\s*MHz")


def parse_nextpnr_log_fmax(log: str) -> dict:
    """--report が読めない場合のフォールバック(ログの Max frequency 行)。"""
    out: dict = {}
    vals = []
    for clk, mhz in _FMAX_LOG_RE.findall(log):
        key = re.sub(r'\W+', '_', clk).strip('_')
        out[f'fmax_{key}_mhz'] = float(mhz)
        vals.append(float(mhz))
    if vals:
        out['fmax_mhz'] = min(vals)
    return out


def diagnose_failure(log: str) -> str | None:
    for needle, advice in _KNOWN_ERRORS:
        if needle in log:
            return advice
    return None


def run_flow(projdir: str | Path, name: str, tc: Toolchain | None = None,
             timeout_per_stage: int = 3600, *, board: dict,
             clock_hz: int) -> FlowResult:
    """yosys → nextpnr-himbaechel → gowin_pack を実行し flow_summary.json を書く。"""
    tc = tc or discover()
    projdir = Path(projdir)
    res = FlowResult(ok=False, projdir=str(projdir), name=name)
    opts = board.get('synth_options', {})

    nextpnr_args = [
        '--json', f'{name}.json', '--write', f'{name}_pnr.json',
        '--device', board['device'],
        '--vopt', f'family={board["family"]}',
        '--vopt', f'cst={name}.cst',
        '--report', f'{name}_report.json',
        '--freq', f'{clock_hz / 1e6:g}',
    ]
    for v in opts.get('nextpnr_vopts', []):
        nextpnr_args += ['--vopt', v]
    # 配置依存の機能不良(モジュール docstring 参照)への回避口。
    # 既定は nextpnr のデフォルト seed(=1、決定的)
    seed = os.environ.get('MDL_FPGA_NEXTPNR_SEED')
    if seed:
        nextpnr_args += ['--seed', seed]
    res.summary['nextpnr_seed'] = int(seed) if seed else 1
    pack_args = ['-d', board['family'],
                 *opts.get('gowin_pack_flags', []),
                 '-o', f'{name}.fs', f'{name}_pnr.json']

    stage_argv = {
        'yosys': [tc.yosys, '-l', 'yosys_run.log', f'{name}.ys'],
        'nextpnr': [tc.nextpnr_himbaechel, *nextpnr_args],
        'gowin_pack': [tc.gowin_pack, *pack_args],
    }
    for stage in STAGES:
        argv = stage_argv[stage]
        if argv[0] is None:
            res.diagnosis = (f'{stage} が見つからない(oss-cad-suite 同梱。'
                             'doctor で導入状況を確認)')
            write_summary(projdir, res)
            return res
        t0 = time.time()
        proc = tc.run_sim_tool([str(a) for a in argv], cwd=projdir,
                               timeout=timeout_per_stage)
        dt = time.time() - t0
        stage_log = (proc.stdout or '') + (proc.stderr or '')
        (projdir / f'{stage}.log').write_text(stage_log, encoding='utf-8',
                                              errors='replace')
        res.stages.append({'stage': stage, 'rc': proc.returncode,
                           'seconds': dt})
        if proc.returncode != 0:
            res.diagnosis = diagnose_failure(stage_log) or (
                f'{stage} が失敗。生ログ: {projdir / (stage + ".log")}。'
                'GitHub issue には flow_summary.json とこのログを添付')
            write_summary(projdir, res)
            return res

    report = projdir / f'{name}_report.json'
    if report.exists():
        try:
            res.summary.update(parse_nextpnr_report(
                json.loads(report.read_text(encoding='utf-8',
                                            errors='replace'))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if 'fmax_mhz' not in res.summary:
        log_path = projdir / 'nextpnr.log'
        if log_path.exists():
            res.summary.update(parse_nextpnr_log_fmax(
                log_path.read_text(encoding='utf-8', errors='replace')))
    res.summary.setdefault('freq_target_mhz', clock_hz / 1e6)

    fs = projdir / f'{name}.fs'
    if fs.exists():
        res.sof = str(fs)
        res.ok = True
    else:
        res.diagnosis = 'gowin_pack 完了したが .fs が見つからない(ツール内部エラー)'
    write_summary(projdir, res)
    return res


class GowinOssBackend:
    """SynthBackend 実装(Gowin、オープンツールチェーン)。"""
    name = 'gowin-oss'
    bitstream_ext = '.fs'
    project_subdir = 'gowin'
    convert_formats: tuple[str, ...] = ()

    def available(self, tc: Toolchain) -> tuple[bool, str]:
        missing = [n for n, p in (('yosys', tc.yosys),
                                  ('nextpnr-himbaechel', tc.nextpnr_himbaechel),
                                  ('gowin_pack', tc.gowin_pack)) if p is None]
        if missing:
            return False, (f'gowin-oss 合成に必要なツールが不足: '
                           f'{", ".join(missing)}(oss-cad-suite 同梱。'
                           'doctor で導入状況を確認)')
        return True, ''

    def write_project_files(self, projdir: str | Path, *, name: str,
                            board: dict, sources: list[str],
                            clock_hz: int, groups: tuple,
                            uart: dict | None, clock_div: int = 1) -> None:
        if clock_div != 1:
            raise NotImplementedError(
                f'gowin-oss は clock_div={clock_div} に未対応'
                '(派生クロック制約を .ys/.cst に書く実装が無い)。'
                'clock_div=1 で生成するか quartus バックエンドを使う')
        write_project_files(projdir, name=name, board=board, sources=sources,
                            clock_hz=clock_hz, groups=groups, uart=uart)

    def run_flow(self, projdir: str | Path, name: str,
                 tc: Toolchain | None = None, *, board: dict,
                 clock_hz: int, clock_div: int = 1) -> FlowResult:
        return run_flow(projdir, name, tc=tc, board=board, clock_hz=clock_hz)

    def convert_bitstream(self, bitstream: str | Path, fmt: str,
                          tc: Toolchain | None = None) -> Path:
        raise ValueError(
            f'.fs の {fmt} 変換は不要(openFPGALoader が .fs を直接書き込む。'
            'flash --via openfpgaloader / bundle をそのまま使う)')

    def bitstream_globs(self, name_hint: str = '*') -> tuple[str, ...]:
        return (f'gowin/{name_hint}.fs', f'{name_hint}.fs')
