"""実機の ISSP(In-System Sources and Probes)経由でオーバーレイを走らせる。

UART の hw-check は「bit 一致」しか測れない(1,000 枚で約 10 秒かかり、
サイクル数もスループットも取れない)。オーバーレイの検証システムは画像 ROM と
サイクルカウンタをチップ内に持つので、JTAG の ISSP で start を叩き、probe を
読むだけで **一致数・処理数・サイクル数** が同時に取れる。

    source[0] = start, source[1] = rst
    probe     = {cycle(32bit), match, proc, done}   ← 割付は生成側の契約
                (projdir/overlay_system.json の probe に記録される)

判定は Python 側で行う(Tcl は生の probe 値を出すだけ)。生値は 32bit カウンタの
剰余なので、設計上の総サイクルとの合同(mod 2^32)で厳密照合する。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .toolchain import Toolchain, discover

_TCL = """\
# mdl-fpga 自動生成: ISSP で overlay を走らせて probe を読む
set hw  [lindex [get_hardware_names] 0]
set dev [lindex [get_device_names -hardware_name $hw] 0]
puts "HW=$hw DEV=$dev"
start_insystem_source_probe -device_name $dev -hardware_name $hw
proc rd {{}} {{
    set v [read_probe_data -instance_index {index} -value_in_hex]
    regsub {{^0x}} $v {{}} v
    return [expr wide(0x$v)]
}}
proc wr {{val}} {{
    write_source_data -instance_index {index} -value $val -value_in_hex
}}
wr {rst_val} ; after 50 ; wr 0 ; after 50 ; wr {start_val}
set done 0 ; set t 0
while {{$t < {timeout_ms}}} {{
    set v [rd] ; set done [expr $v & 1]
    if {{$done}} break
    after {poll_ms} ; incr t {poll_ms}
}}
set v [rd]
puts "PROBE_RAW=$v"
puts "PROBE_WAIT_MS=$t"
wr 0
end_insystem_source_probe
"""


@dataclass
class IsspResult:
    """実機実行の結果(生値と判定)。"""
    ok: bool = False
    done: int = 0
    proc: int = 0
    match: int = 0
    cycle_raw: int = 0
    wraps: int = 0
    true_cycles: int = 0
    cyc_per_inference: float = 0.0
    inf_per_s: float = 0.0
    expected_images: int = 0
    expected_cycles_per_image: int = 0
    core_clock_hz: float = 0.0
    messages: list[str] = field(default_factory=list)
    stdout: str = ''

    def render(self) -> str:
        lines = [f'issp: {"PASS" if self.ok else "FAIL"}']
        lines.append(f'  一致 {self.match} / 処理 {self.proc} '
                     f'/ 期待 {self.expected_images}')
        lines.append(f'  サイクル {self.true_cycles:,}'
                     f'(生値 {self.cycle_raw:,}, ラップ {self.wraps})'
                     f' → {self.cyc_per_inference:.1f} cyc/推論'
                     f'(式 {self.expected_cycles_per_image})')
        lines.append(f'  スループット {self.inf_per_s:.1f} inf/s '
                     f'@ {self.core_clock_hz / 1e6:.3f} MHz')
        lines.extend(f'  {m}' for m in self.messages)
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def load_meta(projdir: str | Path) -> dict:
    """projdir/overlay_system.json(生成時の契約)を読む。"""
    path = Path(projdir) / 'overlay_system.json'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} が無い。overlay の build 出力ディレクトリを指定する'
            '(build --mode overlay が生成する)')
    return json.loads(path.read_text(encoding='utf-8'))


def build_issp_tcl(meta: dict, *, index: int = 0, timeout_ms: int = 60000,
                   poll_ms: int = 20) -> str:
    """quartus_stp に渡す Tcl を組み立てる。"""
    src = meta['probe']['source']
    return _TCL.format(index=index, timeout_ms=timeout_ms, poll_ms=poll_ms,
                       rst_val=format(1 << src['rst'], 'x'),
                       start_val=format(1 << src['start'], 'x'))


def decode_probe(value: int, layout: dict) -> dict:
    """probe の生値を {done, proc, match, cycle} に分解する。"""
    out = {}
    for key in ('done', 'proc', 'match', 'cycle'):
        lsb, width = layout[key]
        out[key] = (value >> lsb) & ((1 << width) - 1)
    return out


def evaluate(meta: dict, fields: dict) -> IsspResult:
    """生値から判定を組み立てる(実機なしでもテストできる純関数)。

    サイクルカウンタは 32bit の自由走行カウンタなので、設計上の総サイクル
    N * cyc/画像 との **合同(mod 2^32)** で照合する(ラップ回数は総サイクルの
    設計値から一意に決まる)。
    """
    n = int(meta['n_images'])
    cyc = int(meta['cycles_per_image'])
    fcore = float(meta['core_clock_hz'])
    width = int(meta['probe']['cycle'][1])
    mod = 1 << width
    total_model = n * cyc
    wraps = total_model // mod
    expect_raw = total_model % mod

    res = IsspResult(
        done=fields['done'], proc=fields['proc'], match=fields['match'],
        cycle_raw=fields['cycle'], wraps=wraps,
        expected_images=n, expected_cycles_per_image=cyc, core_clock_hz=fcore)
    cyc_exact = (fields['cycle'] == expect_raw)
    counts_ok = (fields['done'] == 1 and fields['proc'] == n
                 and fields['match'] == n)
    res.true_cycles = fields['cycle'] + wraps * mod
    if res.proc:
        res.cyc_per_inference = res.true_cycles / res.proc
        if fcore:
            res.inf_per_s = res.proc / (res.true_cycles / fcore)
    res.ok = cyc_exact and counts_ok
    if not counts_ok:
        res.messages.append(
            f'一致数/処理数が期待と違う(done={res.done} proc={res.proc} '
            f'match={res.match}、期待 {n}/{n})')
    if not cyc_exact:
        res.messages.append(
            f'サイクルが設計式と一致しない(生値 {res.cycle_raw} != '
            f'期待 {expect_raw} = {n} x {cyc} mod 2^{width})')
    return res


def run_issp(projdir: str | Path, *, index: int = 0, timeout_ms: int = 60000,
             tc: Toolchain | None = None) -> IsspResult:
    """書き込み済みの実機を ISSP で走らせて結果を返す(要 JTAG 接続)。

    .sof の書き込みは行わない(flash 動詞の担当)。
    """
    tc = tc or discover()
    projdir = Path(projdir)
    meta = load_meta(projdir)
    if tc.quartus('quartus_stp') is None:
        res = IsspResult()
        res.messages.append(
            'quartus_stp が見つからない(doctor 参照)。ISSP は Quartus 同梱の '
            'quartus_stp が必要')
        return res

    tcl = build_issp_tcl(meta, index=index, timeout_ms=timeout_ms)
    (projdir / 'issp_run.tcl').write_text(tcl, encoding='utf-8')
    proc = tc.run_quartus('quartus_stp', ['-t', 'issp_run.tcl'], cwd=projdir,
                          timeout=max(60, timeout_ms // 1000 + 60))
    out = (proc.stdout or '') + (proc.stderr or '')
    (projdir / 'issp_run_out.txt').write_text(out, encoding='utf-8',
                                              errors='replace')
    m = re.search(r'PROBE_RAW=(\d+)', out)
    if not m:
        res = IsspResult(stdout=out)
        res.messages.append(
            'probe を読めなかった(PROBE_RAW 行が無い)。JTAG 接続と '
            '.sof の書き込みを確認する。生ログ: issp_run_out.txt')
        return res
    fields = decode_probe(int(m.group(1)), meta['probe'])
    res = evaluate(meta, fields)
    res.stdout = out
    return res
