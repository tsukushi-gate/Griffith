"""bit 一致検証(mdl-fpga check / sim)。

検証の辺(構想書 §6):
  (a) ライブラリ ir.simulate(正解オラクル) ⇔ ツール自製コアの RTL シミュレーション
  (c) ツール生成器 ⇔ ライブラリ export/verilog.py の相互 bit 一致(ドリフト検出)
どちらも同一ベクトルを 1 つのテストベンチで評価する。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.export.verilog import write_verilog
from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.simulate import simulate

from .hdlgen.core import CoreGen, generate_core
from .ir_io import load_graph
from .toolchain import TOOL_DIR, Toolchain, discover
from .vectors import gen_inputs, hex_lines, pack_bits, pack_scores

_TB_TEMPLATE = """`timescale 1ns/1ps
// mdl-fpga 自動生成テストベンチ(check)
module tb;
    reg  [{in_msb}:0] x;
    wire [{out_msb}:0] y_tool;
{lib_decl}
    mdl_core dut_tool(.x(x), .y(y_tool));
{lib_inst}
    reg [{in_msb}:0]  in_vecs  [0:{last}];
    reg [{out_msb}:0] exp_vecs [0:{last}];
    integer i, err_tool, err_cross;

    initial begin
        $readmemh("inputs.hex", in_vecs);
        $readmemh("expected.hex", exp_vecs);
        err_tool = 0; err_cross = 0;
        for (i = 0; i < {n_vecs}; i = i + 1) begin
            x = in_vecs[i]; #1;
            if (y_tool !== exp_vecs[i]) begin
                err_tool = err_tool + 1;
                $display("MISMATCH_TOOL %0d: got=%h exp=%h", i, y_tool, exp_vecs[i]);
            end
{cross_body}
        end
        if (err_tool == 0 && err_cross == 0) $display("ALLPASS");
        else $display("FAILED tool=%0d cross=%0d of {n_vecs}", err_tool, err_cross);
        $finish;
    end
endmodule
"""

_CROSS_BODY = """            if (y_tool !== y_lib) begin
                err_cross = err_cross + 1;
                $display("MISMATCH_CROSS %0d: tool=%h lib=%h", i, y_tool, y_lib);
            end"""


@dataclass
class CheckResult:
    passed: bool
    n_vectors: int = 0
    workdir: str = ''
    edges: dict = field(default_factory=dict)   # 検証辺 → 'PASS'/'FAIL'/'SKIP'
    messages: list[str] = field(default_factory=list)
    stdout: str = ''

    def render(self) -> str:
        lines = [f'check: {"PASS" if self.passed else "FAIL"}'
                 f'({self.n_vectors} ベクトル)']
        for k, v in self.edges.items():
            lines.append(f'  {k}: {v}')
        lines.extend(f'  {m}' for m in self.messages)
        lines.append(f'  生成物: {self.workdir}')
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {'passed': self.passed, 'n_vectors': self.n_vectors,
                'edges': self.edges, 'messages': self.messages,
                'workdir': self.workdir}


def default_workdir(model: str | Path, verb: str) -> Path:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return TOOL_DIR / 'runs' / f'{Path(model).stem}_{verb}_{stamp}'


def expected_packed(graph: IRGraph, x: np.ndarray) -> tuple[list[int], int]:
    """golden(ir.simulate)を出力バス表現へパックする(verify.py と同規約)。"""
    expected = simulate(graph, x)
    out_src = graph.output_source()
    nodes = {n.id: n for n in graph.nodes}
    last = nodes[out_src] if out_src in nodes else graph.topological_order()[-1]
    if last.type == 'group_sum':
        k = last.params['k']
        group = last.params['in_width'] // k
        score_bits = max(1, math.ceil(math.log2(group + 1)))
        return pack_scores(expected, score_bits), k * score_bits
    return pack_bits(expected), expected.shape[1]


def _write_check_files(workdir: Path, gen: CoreGen, graph: IRGraph,
                       x: np.ndarray, cross_check: bool) -> int:
    """コア/ベクトル/TB を書き出し、ベクトル数を返す。"""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'mdl_core.v').write_text(gen.verilog, encoding='utf-8')
    (workdir / 'name_map.json').write_text(
        json.dumps(gen.name_map, ensure_ascii=False, indent=2), encoding='utf-8')

    exp_packed, out_width = expected_packed(graph, x)
    assert out_width == gen.out_width, (out_width, gen.out_width)

    (workdir / 'inputs.hex').write_text(
        hex_lines(pack_bits(x), gen.in_width), encoding='utf-8')
    (workdir / 'expected.hex').write_text(
        hex_lines(exp_packed, out_width), encoding='utf-8')

    if cross_check:
        write_verilog(graph, str(workdir / 'lib_core.v'))
        lib_decl = f'    wire [{out_width - 1}:0] y_lib;'
        lib_inst = '    logic_core dut_lib(.x(x), .y(y_lib));'
        cross_body = _CROSS_BODY
    else:
        lib_decl = ''
        lib_inst = ''
        cross_body = ''

    tb = _TB_TEMPLATE.format(
        in_msb=gen.in_width - 1, out_msb=out_width - 1,
        last=len(x) - 1, n_vecs=len(x),
        lib_decl=lib_decl, lib_inst=lib_inst, cross_body=cross_body,
    )
    (workdir / 'tb.v').write_text(tb, encoding='utf-8')
    return len(x)


def run_check(model: str | Path, workdir: str | Path | None = None,
              n_vectors: int = 256, seed: int = 0, cross_check: bool = True,
              tc: Toolchain | None = None) -> CheckResult:
    tc = tc or discover()
    res = CheckResult(passed=False)

    if tc.iverilog is None or tc.vvp is None:
        res.messages.append('iverilog/vvp が見つからない(doctor で確認)')
        res.edges['ref_vs_rtl'] = 'SKIP'
        return res

    graph = lower_to_primitives(load_graph(model))
    gen = generate_core(graph)
    x = gen_inputs(gen.in_width, n_vectors, seed)

    wd = Path(workdir) if workdir else default_workdir(model, 'check')
    res.workdir = str(wd)
    n = _write_check_files(wd, gen, graph, x, cross_check)
    res.n_vectors = n

    sources = ['tb.v', 'mdl_core.v'] + (['lib_core.v'] if cross_check else [])
    comp = tc.run_sim_tool([tc.iverilog, '-g2012', '-o', 'sim.vvp', *sources],
                           cwd=wd)
    if comp.returncode != 0:
        res.messages.append(f'iverilog コンパイル失敗:\n{comp.stderr}')
        res.edges['ref_vs_rtl'] = 'FAIL'
        return res
    run = tc.run_sim_tool([tc.vvp, 'sim.vvp'], cwd=wd)
    res.stdout = run.stdout

    ok = 'ALLPASS' in run.stdout
    tool_fail = 'MISMATCH_TOOL' in run.stdout
    cross_fail = 'MISMATCH_CROSS' in run.stdout
    res.edges['ref_vs_rtl(a)'] = 'FAIL' if tool_fail else 'PASS'
    if cross_check:
        res.edges['generator_cross(c)'] = 'FAIL' if cross_fail else 'PASS'
    res.passed = ok
    if not ok:
        head = '\n'.join(run.stdout.splitlines()[:10])
        res.messages.append(f'不一致詳細(先頭):\n{head}')

    (wd / 'check_report.json').write_text(
        json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    return res


_PIPE_TB_TEMPLATE = """`timescale 1ns/1ps
// mdl-fpga 自動生成テストベンチ(パイプラインコア等価性)
// 毎クロック新ベクトルを投入(II=1 の実証)し、
//   (a) golden(ir.simulate)照合  (x) comb コアの L サイクル遅延照合
// の2辺を同時に検証する。
module tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg  [{in_msb}:0] x = {{{in_width}{{1'b0}}}};
    wire [{out_msb}:0] y_pipe;
    wire [{out_msb}:0] y_comb;
    mdl_core_pipe dut_pipe (.clk(clk), .x(x), .y(y_pipe));
    mdl_core      dut_comb (.x(x), .y(y_comb));

    reg [{out_msb}:0] dly [0:{lat_minus1}];
    reg [{in_msb}:0]  in_vecs  [0:{last}];
    reg [{out_msb}:0] exp_vecs [0:{last}];
    integer t, k, err_g, err_x;

    initial begin
        $readmemh("inputs.hex", in_vecs);
        $readmemh("expected.hex", exp_vecs);
        err_g = 0; err_x = 0;
        for (t = 0; t < {n_vecs} + {latency}; t = t + 1) begin
            x = (t < {n_vecs}) ? in_vecs[t] : {{{in_width}{{1'b0}}}};
            #1;
            for (k = {lat_minus1}; k > 0; k = k - 1) dly[k] = dly[k-1];
            dly[0] = y_comb;
            @(posedge clk); #1;
            if (t >= {lat_minus1} && (t - {lat_minus1}) < {n_vecs}) begin
                if (y_pipe !== exp_vecs[t - {lat_minus1}]) begin
                    err_g = err_g + 1;
                    $display("MISMATCH_GOLD %0d: got=%h exp=%h",
                             t - {lat_minus1}, y_pipe, exp_vecs[t - {lat_minus1}]);
                end
                if (y_pipe !== dly[{lat_minus1}]) begin
                    err_x = err_x + 1;
                    $display("MISMATCH_CROSS %0d: pipe=%h comb_dly=%h",
                             t - {lat_minus1}, y_pipe, dly[{lat_minus1}]);
                end
            end
        end
        if (err_g == 0 && err_x == 0) $display("ALLPASS");
        else $display("FAILED gold=%0d cross=%0d", err_g, err_x);
        $finish;
    end
endmodule
"""


def run_check_pipelined(model: str | Path, workdir: str | Path | None = None,
                        n_vectors: int = 256, seed: int = 0,
                        tc: Toolchain | None = None) -> CheckResult:
    """パイプラインコアの等価性検証(golden + comb 遅延クロスの2辺)。"""
    tc = tc or discover()
    res = CheckResult(passed=False)
    if tc.iverilog is None or tc.vvp is None:
        res.messages.append('iverilog/vvp が見つからない(doctor で確認)')
        res.edges['pipe_vs_golden(a)'] = 'SKIP'
        return res

    graph = lower_to_primitives(load_graph(model))
    comb = generate_core(graph, module_name='mdl_core')
    pipe = generate_core(graph, module_name='mdl_core_pipe', pipelined=True)
    x = gen_inputs(pipe.in_width, n_vectors, seed)

    wd = Path(workdir) if workdir else default_workdir(model, 'pipecheck')
    wd.mkdir(parents=True, exist_ok=True)
    res.workdir = str(wd)
    res.n_vectors = len(x)
    res.messages.append(f'latency = {pipe.latency}clk(II=1 ストリーミング投入)')

    (wd / 'mdl_core.v').write_text(comb.verilog, encoding='utf-8')
    (wd / 'mdl_core_pipe.v').write_text(pipe.verilog, encoding='utf-8')
    exp_packed, out_width = expected_packed(graph, x)
    (wd / 'inputs.hex').write_text(
        hex_lines(pack_bits(x), pipe.in_width), encoding='utf-8')
    (wd / 'expected.hex').write_text(
        hex_lines(exp_packed, out_width), encoding='utf-8')
    (wd / 'tb.v').write_text(_PIPE_TB_TEMPLATE.format(
        in_msb=pipe.in_width - 1, in_width=pipe.in_width,
        out_msb=out_width - 1, last=len(x) - 1, n_vecs=len(x),
        latency=pipe.latency, lat_minus1=pipe.latency - 1,
    ), encoding='utf-8')

    comp = tc.run_sim_tool(
        [tc.iverilog, '-g2012', '-o', 'sim.vvp',
         'tb.v', 'mdl_core.v', 'mdl_core_pipe.v'], cwd=wd)
    if comp.returncode != 0:
        res.messages.append(f'iverilog コンパイル失敗:\n{comp.stderr}')
        res.edges['pipe_vs_golden(a)'] = 'FAIL'
        return res
    run = tc.run_sim_tool([tc.vvp, 'sim.vvp'], cwd=wd, timeout=1800)
    res.stdout = run.stdout
    res.passed = 'ALLPASS' in run.stdout
    res.edges['pipe_vs_golden(a)'] = \
        'FAIL' if 'MISMATCH_GOLD' in run.stdout else 'PASS'
    res.edges['pipe_vs_comb(x)'] = \
        'FAIL' if 'MISMATCH_CROSS' in run.stdout else 'PASS'
    if not res.passed:
        head = '\n'.join(run.stdout.splitlines()[:10])
        res.messages.append(f'不一致詳細(先頭):\n{head}')
    return res


def run_wrapper_check(model: str | Path, workdir: str | Path | None = None,
                      n_vectors: int = 16, seed: int = 0,
                      sim_clks_per_bit: int = 16,
                      tc: Toolchain | None = None) -> CheckResult:
    """フルチェーン検証: UART フレーム送受信込みで golden と bit 一致を確認する。

    実機に書き込む top と同一構造(CLKS_PER_BIT のみ短縮)を iverilog で回す。
    """
    from .hdlgen.wrapper import (
        copy_static_hdl,
        generate_wrapper,
        generate_wrapper_tb,
    )

    tc = tc or discover()
    res = CheckResult(passed=False)
    if tc.iverilog is None or tc.vvp is None:
        res.messages.append('iverilog/vvp が見つからない(doctor で確認)')
        res.edges['wrapper_e2e'] = 'SKIP'
        return res

    graph = lower_to_primitives(load_graph(model))
    gen = generate_core(graph)
    x = gen_inputs(gen.in_width, n_vectors, seed)

    wd = Path(workdir) if workdir else default_workdir(model, 'wrapper')
    wd.mkdir(parents=True, exist_ok=True)
    res.workdir = str(wd)
    res.n_vectors = len(x)

    (wd / 'mdl_core.v').write_text(gen.verilog, encoding='utf-8')
    wrap = generate_wrapper(gen.in_width, gen.out_width, top_name='top_sim',
                            clks_per_bit=sim_clks_per_bit)
    (wd / 'top_sim.v').write_text(wrap.top_verilog, encoding='utf-8')
    statics = copy_static_hdl(wd)

    exp_packed, out_width = expected_packed(graph, x)
    (wd / 'inputs_padded.hex').write_text(
        hex_lines(pack_bits(x), 8 * wrap.req_payload_bytes), encoding='utf-8')
    (wd / 'expected_padded.hex').write_text(
        hex_lines(exp_packed, 8 * wrap.rsp_payload_bytes), encoding='utf-8')

    tb = generate_wrapper_tb(wrap, n_vectors=len(x),
                             sim_clks_per_bit=sim_clks_per_bit)
    (wd / 'tb_wrapper.v').write_text(tb, encoding='utf-8')

    sources = ['tb_wrapper.v', 'top_sim.v', 'mdl_core.v', *statics]
    comp = tc.run_sim_tool([tc.iverilog, '-g2012', '-o', 'sim.vvp', *sources],
                           cwd=wd)
    if comp.returncode != 0:
        res.messages.append(f'iverilog コンパイル失敗:\n{comp.stderr}')
        res.edges['wrapper_e2e'] = 'FAIL'
        return res
    run = tc.run_sim_tool([tc.vvp, 'sim.vvp'], cwd=wd, timeout=1800)
    res.stdout = run.stdout
    res.passed = 'ALLPASS' in run.stdout
    res.edges['wrapper_e2e'] = 'PASS' if res.passed else 'FAIL'
    if not res.passed:
        head = '\n'.join(run.stdout.splitlines()[:10])
        res.messages.append(f'不一致詳細(先頭):\n{head}')
    return res


def run_system_check(graph_or_model, images, labels,
                     workdir: str | Path, slow_div_tb: int = 4,
                     tc: Toolchain | None = None) -> CheckResult:
    """分類デモシステム(ROM+FSM+argmax+BCD)の自己検証 RTL シム。

    2パス実行: 高速→(KEY0リスタート)→低速(throttle 経路)。各画像の pred を
    golden(ir.simulate)と逐次照合し、完走時の BCD カウンタも照合する。
    """
    from .hdlgen.system import generate_system, generate_system_tb, \
        write_system_vectors
    from .hdlgen.wrapper import HDL_DIR
    import shutil

    tc = tc or discover()
    res = CheckResult(passed=False)
    if tc.iverilog is None or tc.vvp is None:
        res.messages.append('iverilog/vvp が見つからない(doctor で確認)')
        res.edges['system_rtl'] = 'SKIP'
        return res

    if isinstance(graph_or_model, str | Path):
        graph = lower_to_primitives(load_graph(graph_or_model))
    else:
        graph = lower_to_primitives(graph_or_model)

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    res.workdir = str(wd)
    res.n_vectors = len(images)

    gen = generate_system(graph, n_images=len(images))
    golden = write_system_vectors(graph, images, labels, wd, gen.label_bits)
    res.messages.append(
        f'golden: 正解 {golden["correct"]}/{golden["n_images"]}'
        f'(latency={gen.latency}clk)')

    (wd / 'top.v').write_text(gen.top_verilog, encoding='utf-8')
    (wd / 'sys_rom.v').write_text(gen.rom_verilog, encoding='utf-8')
    (wd / 'mdl_core_pipe.v').write_text(gen.core.verilog, encoding='utf-8')
    shutil.copyfile(HDL_DIR / 'seg7.v', wd / 'seg7.v')
    (wd / 'tb_system.v').write_text(
        generate_system_tb(gen, golden, slow_div_tb=slow_div_tb),
        encoding='utf-8')

    comp = tc.run_sim_tool(
        [tc.iverilog, '-g2012', '-o', 'sim.vvp', 'tb_system.v',
         'top.v', 'sys_rom.v', 'mdl_core_pipe.v', 'seg7.v'], cwd=wd)
    if comp.returncode != 0:
        res.messages.append(f'iverilog コンパイル失敗:\n{comp.stderr}')
        res.edges['system_rtl'] = 'FAIL'
        return res
    run = tc.run_sim_tool([tc.vvp, 'sim.vvp'], cwd=wd, timeout=3600)
    res.stdout = run.stdout
    res.passed = 'ALLPASS' in run.stdout
    res.edges['system_rtl'] = 'PASS' if res.passed else 'FAIL'
    if not res.passed:
        head = '\n'.join(run.stdout.splitlines()[:12])
        res.messages.append(f'不一致詳細(先頭):\n{head}')
    return res


def run_sim_only(model: str | Path, workdir: str | Path | None = None,
                 n_vectors: int = 256, seed: int = 0,
                 tc: Toolchain | None = None) -> CheckResult:
    """golden 照合なしの RTL 実行(sim verb)。出力を y_out.hex に残す。"""
    tc = tc or discover()
    res = CheckResult(passed=False)
    if tc.iverilog is None or tc.vvp is None:
        res.messages.append('iverilog/vvp が見つからない(doctor で確認)')
        return res

    graph = lower_to_primitives(load_graph(model))
    gen = generate_core(graph)
    x = gen_inputs(gen.in_width, n_vectors, seed)

    wd = Path(workdir) if workdir else default_workdir(model, 'sim')
    wd.mkdir(parents=True, exist_ok=True)
    res.workdir = str(wd)
    (wd / 'mdl_core.v').write_text(gen.verilog, encoding='utf-8')
    (wd / 'inputs.hex').write_text(
        hex_lines(pack_bits(x), gen.in_width), encoding='utf-8')

    digits = (gen.out_width + 3) // 4
    tb = f"""`timescale 1ns/1ps
module tb;
    reg  [{gen.in_width - 1}:0] x;
    wire [{gen.out_width - 1}:0] y;
    mdl_core dut(.x(x), .y(y));
    reg [{gen.in_width - 1}:0] in_vecs [0:{len(x) - 1}];
    integer i, fd;
    initial begin
        $readmemh("inputs.hex", in_vecs);
        fd = $fopen("y_out.hex", "w");
        for (i = 0; i < {len(x)}; i = i + 1) begin
            x = in_vecs[i]; #1;
            $fdisplay(fd, "%0{digits}h", y);
        end
        $fclose(fd);
        $display("SIM_DONE {len(x)}");
        $finish;
    end
endmodule
"""
    (wd / 'tb.v').write_text(tb, encoding='utf-8')
    comp = tc.run_sim_tool([tc.iverilog, '-g2012', '-o', 'sim.vvp',
                            'tb.v', 'mdl_core.v'], cwd=wd)
    if comp.returncode != 0:
        res.messages.append(f'iverilog コンパイル失敗:\n{comp.stderr}')
        return res
    run = tc.run_sim_tool([tc.vvp, 'sim.vvp'], cwd=wd)
    res.stdout = run.stdout
    res.n_vectors = len(x)
    res.passed = 'SIM_DONE' in run.stdout
    res.messages.append(f'出力: {wd / "y_out.hex"}')
    return res


# ---------------------------------------------------------------- overlay


def prepare_overlay_workdir(graph_or_model, workdir: str | Path, *,
                            images: np.ndarray | None = None,
                            labels: np.ndarray | None = None,
                            n_vectors: int = 64, v: int = 8, seed: int = 0,
                            clock_div: int = 2, clock_hz: int = 50_000_000,
                            with_addrref: bool = True,
                            top_name: str = 'top',
                            wiring: str = 'auto',
                            ports=None):
    """オーバーレイの生成物一式(RTL・.mem・ベクトル・TB)を workdir に用意する。

    wiring は 'auto'(既定)/ 'computed' / 'stored'。保存型では配線が記録に
    入るのでアドレス生成 TB は作らない(代わりに run_record_check が .mem を
    復号して IR と全数照合する)。

    Returns:
        (plan, gen, sysgen, golden, images)
    """
    import shutil

    from .hdlgen.overlay import (
        generate_overlay,
        plan_overlay,
        write_overlay_files,
    )
    from .hdlgen.overlay_system import (
        generate_addrgen_tb,
        generate_overlay_system,
        generate_overlay_tb,
        generate_overlay_top_tb,
        write_overlay_vectors,
    )
    from .hdlgen.wrapper import HDL_DIR

    if isinstance(graph_or_model, str | Path):
        graph = lower_to_primitives(load_graph(graph_or_model))
    else:
        graph = lower_to_primitives(graph_or_model)

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    plan = plan_overlay(graph, v=v, wiring=wiring)
    gen = generate_overlay(plan,
                           with_addrref=with_addrref and not plan.stored)
    write_overlay_files(gen, wd, include_sim_only=True)

    if images is None:
        images = gen_inputs(plan.in_bits, n_vectors, seed)
        if len(images) > n_vectors:
            images = images[:n_vectors]
    images = np.asarray(images, dtype=np.int64)
    golden = write_overlay_vectors(graph, images, wd, plan, labels=labels)

    sysgen = generate_overlay_system(plan, n_images=len(images),
                                     top_name=top_name, clock_div=clock_div,
                                     clock_hz=clock_hz, ports=ports)
    (wd / f'{top_name}.v').write_text(sysgen.top_verilog, encoding='utf-8')
    shutil.copyfile(HDL_DIR / 'seg7.v', wd / 'seg7.v')
    (wd / 'tb_overlay.v').write_text(
        generate_overlay_tb(plan, len(images)), encoding='utf-8')
    (wd / 'tb_overlay_top.v').write_text(
        generate_overlay_top_tb(sysgen), encoding='utf-8')
    if not plan.stored:
        (wd / 'tb_addrgen.v').write_text(generate_addrgen_tb(plan),
                                         encoding='utf-8')
    (wd / 'overlay_system.json').write_text(
        json.dumps(sysgen.to_meta(), ensure_ascii=False, indent=2),
        encoding='utf-8')
    return plan, gen, sysgen, golden, images


_OVERLAY_CORE_SOURCES = ('lgn_overlay.v', 'rom32.v', 'tdp_ram.v',
                         'gate_eval.v', 'overlay_argmax.v')
_OVERLAY_STORED_SOURCES = ('lgn_overlay_stored.v', 'gate_rom.v', 'rom32.v',
                           'rom40.v', 'tdp_ram.v', 'gate_eval.v',
                           'overlay_argmax.v')


def overlay_sim_sources(workdir: str | Path) -> tuple[tuple[str, ...], str]:
    """workdir の生成物から (コアのソース, ハーネス名) を決める。

    保存型のテンプレートが置いてあれば保存型として組む(生成時の
    plan.wiring が唯一の真実源で、その痕跡がファイルの有無)。
    """
    if (Path(workdir) / 'lgn_overlay_stored.v').exists():
        return _OVERLAY_STORED_SOURCES, 'overlay_harness_stored.v'
    return _OVERLAY_CORE_SOURCES, 'overlay_harness.v'


def run_overlay_sim(workdir: str | Path, *, use_top: bool = False,
                    top_name: str = 'top', timeout: int = 3600,
                    tc: Toolchain | None = None) -> tuple[bool, str]:
    """用意済みの workdir でオーバーレイ全系シムを走らせる。"""
    tc = tc or discover()
    wd = Path(workdir)
    core, harness = overlay_sim_sources(wd)
    if use_top:
        tb = 'tb_overlay_top.v'
        sources = [tb, *core, harness, 'seg7.v',
                   'altsource_probe_stub.v', f'{top_name}.v']
        vvp = 'sim_overlay_top.vvp'
    else:
        tb = 'tb_overlay.v'
        sources = [tb, *core, harness]
        vvp = 'sim_overlay.vvp'
    comp = tc.run_sim_tool([tc.iverilog, '-g2012', '-o', vvp, *sources], cwd=wd)
    if comp.returncode != 0:
        return False, f'iverilog コンパイル失敗:\n{comp.stderr}'
    run = tc.run_sim_tool([tc.vvp, vvp], cwd=wd, timeout=timeout)
    return 'ALLPASS' in run.stdout, run.stdout


def run_overlay_check(graph_or_model, workdir: str | Path | None = None, *,
                      images: np.ndarray | None = None,
                      labels: np.ndarray | None = None,
                      n_vectors: int = 64, v: int = 8, seed: int = 0,
                      clock_div: int = 2, use_top: bool = False,
                      timeout: int = 3600, wiring: str = 'auto',
                      tc: Toolchain | None = None) -> CheckResult:
    """オーバーレイの bit 一致検証: ライブラリ ir.simulate ⇔ overlay RTL。

    期待値 ROM は golden(ir.simulate の予測)なので、一致数 == ベクトル数が
    「参照シムと bit 一致」の判定になる。1 画像あたりのサイクル数が設計式と
    厳密一致することも同じ TB で見る。計算型・保存型のどちらの配線でも同じ
    辺を通る(wiring で選ぶ。既定 auto)。
    """
    tc = tc or discover()
    res = CheckResult(passed=False)
    if tc.iverilog is None or tc.vvp is None:
        res.messages.append('iverilog/vvp が見つからない(doctor で確認)')
        res.edges['overlay_vs_golden(a)'] = 'SKIP'
        return res

    if workdir is not None:
        wd = Path(workdir)
    else:
        stem = (graph_or_model if isinstance(graph_or_model, str | Path)
                else 'overlay')
        wd = default_workdir(stem, 'overlaycheck')
    plan, _gen, sysgen, _golden, images = prepare_overlay_workdir(
        graph_or_model, wd, images=images, labels=labels,
        n_vectors=n_vectors, v=v, seed=seed, clock_div=clock_div,
        wiring=wiring)
    from .hdlgen.overlay import wiring_label
    res.workdir = str(wd)
    res.n_vectors = len(images)
    res.messages.append(
        f'overlay: V={plan.v} W={plan.w} DEPTH={plan.depth} '
        f'gates={plan.total_gates} cyc/画像={plan.cycles_per_image} '
        f'配線={wiring_label(plan)}')
    for warn in plan.warnings:
        res.messages.append(f'注意: {warn}')

    ok, out = run_overlay_sim(wd, use_top=use_top, top_name=sysgen.top_name,
                              timeout=timeout, tc=tc)
    res.stdout = out
    res.passed = ok
    edge = 'overlay_top_vs_golden(a)' if use_top else 'overlay_vs_golden(a)'
    res.edges[edge] = 'PASS' if ok else 'FAIL'
    if not ok:
        head = '\n'.join(out.splitlines()[:12])
        res.messages.append(f'不一致詳細(先頭):\n{head}')
    (wd / 'overlay_check_report.json').write_text(
        json.dumps(res.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8')
    return res


def run_record_check(graph_or_model, workdir: str | Path | None = None, *,
                     v: int = 8, regenerate: bool = True) -> CheckResult:
    """保存型の記録照合: 書き出した .mem を復号 ⇔ IR の (op, a, b)。

    計算型の run_addrgen_check(RTL の発行アドレス ⇔ Python 参照列)に対応する
    保存型の辺。配線は記録そのものなので RTL を回さず、**ファイルに落ちた
    語を 1 語ずつ復号して** IR と全数照合する(EDA 不要)。1 語でも壊れて
    いれば FAIL になる。

    regenerate=False なら既存 workdir の .mem をそのまま読む(破損検出の
    テストや、合成に渡した現物を後から検証する用途)。
    """
    from .hdlgen.overlay import OverlayUnsupported, check_gatemem, plan_overlay

    res = CheckResult(passed=False)
    if isinstance(graph_or_model, str | Path):
        graph = lower_to_primitives(load_graph(graph_or_model))
    else:
        graph = lower_to_primitives(graph_or_model)

    if workdir is not None:
        wd = Path(workdir)
    else:
        stem = (graph_or_model if isinstance(graph_or_model, str | Path)
                else 'overlay')
        wd = default_workdir(stem, 'recordcheck')
    if regenerate:
        prepare_overlay_workdir(graph, wd, n_vectors=1, v=v, wiring='stored')
    res.workdir = str(wd)

    plan = plan_overlay(graph, v=v, wiring='stored')
    res.n_vectors = plan.total_gates
    roms = []
    for k in range(plan.v):
        path = wd / f'gatemem_lane{k}.mem'
        if not path.exists():
            res.messages.append(f'{path.name} が無い(生成されていない)')
            res.edges['record_vs_ir'] = 'FAIL'
            return res
        roms.append(path.read_text(encoding='utf-8').split())
    try:
        checked = check_gatemem(plan, roms)
    except OverlayUnsupported as e:
        res.messages.append(f'記録の照合で不一致: {e}')
        res.edges['record_vs_ir'] = 'FAIL'
        return res
    res.passed = True
    res.edges['record_vs_ir'] = 'PASS'
    res.messages.append(
        f'照合件数: {checked} ゲート x (op, a, b)'
        f'(記録 {plan.record_bits}bit = op 4 + a/b 各 {plan.addr_field_bits})')
    return res


def run_addrgen_check(graph_or_model, workdir: str | Path | None = None, *,
                      v: int = 8, timeout: int = 3600,
                      tc: Toolchain | None = None) -> CheckResult:
    """アドレス生成器の全数照合: RTL の発行アドレス ⇔ Python 参照列。

    計算型配線の要(配線表を持たないこと)を直接検証する辺。IR の indices は
    生成時に plan_overlay が全数照合済みなので、この辺が通れば
    「IR の配線 = Python 参照 = RTL の発行アドレス」が閉じる。
    """
    tc = tc or discover()
    res = CheckResult(passed=False)
    if tc.iverilog is None or tc.vvp is None:
        res.messages.append('iverilog/vvp が見つからない(doctor で確認)')
        res.edges['addrgen_vs_python'] = 'SKIP'
        return res

    wd = Path(workdir) if workdir else default_workdir('overlay', 'addrgen')
    plan, _gen, _sys, _golden, _img = prepare_overlay_workdir(
        graph_or_model, wd, n_vectors=1, v=v, with_addrref=True,
        wiring='computed')
    res.workdir = str(wd)
    res.n_vectors = plan.total_gates

    comp = tc.run_sim_tool(
        [tc.iverilog, '-g2012', '-o', 'sim_addrgen.vvp', 'tb_addrgen.v',
         *_OVERLAY_CORE_SOURCES], cwd=wd)
    if comp.returncode != 0:
        res.messages.append(f'iverilog コンパイル失敗:\n{comp.stderr}')
        res.edges['addrgen_vs_python'] = 'FAIL'
        return res
    run = tc.run_sim_tool([tc.vvp, 'sim_addrgen.vvp'], cwd=wd, timeout=timeout)
    res.stdout = run.stdout
    res.passed = 'ALLPASS' in run.stdout
    res.edges['addrgen_vs_python'] = 'PASS' if res.passed else 'FAIL'
    n_checks = 4 * plan.v * plan.depth * plan.layer_words
    res.messages.append(f'照合件数: {n_checks}(ゲート x a/b x p1/RAM 提示)')
    if not res.passed:
        head = '\n'.join(run.stdout.splitlines()[:12])
        res.messages.append(f'不一致詳細(先頭):\n{head}')
    return res
