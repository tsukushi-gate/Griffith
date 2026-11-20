"""オーバーレイの検証システム生成(画像 ROM → 推論 → 一致計数 → ISSP/7セグ)。

hdlgen.system(空間焼込みの分類デモ)のオーバーレイ版。UART の hw-check では
サイクル数とスループットを測れないため、社内先行実験の
「オンチップ画像 ROM + サイクルカウンタ + ISSP」を移植する:

    top(CLOCK_50 → clock_div → clk_core)
      ├ altsource_probe(ISSP): probe = {cycle(32), match, proc, done}
      │                        source = {rst, start}
      ├ overlay_harness: 画像 ROM N 枚 → lgn_overlay → 期待値と比較・計数
      └ 十進 BCD カウンタ → 7セグ(HEX3-0 = 一致数、HEX5-4 = 処理数の百位)

**期待値 ROM はラベルではなく golden(ライブラリ ir.simulate)の予測**なので、
match == N は「実シリコンが参照シムと bit 一致した」ことを意味する。精度は
golden 側の値(golden.json に記録する)。

$readmemh はファイル名のみで参照する(cwd 罠対策: シム作業 dir / Quartus
プロジェクト dir に .mem をコロケートする)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.simulate import simulate

from ..vectors import pack_bits
from .overlay import OverlayPlan, bits_for
from .ports import TopPorts
from .system import _emit_bcd_inc, _digit_range, to_bcd

# ISSP の source ビット割付(先行実験と同一)
SRC_START = 0
SRC_RST = 1


@dataclass
class OverlaySystemGen:
    """生成した検証システム(トップ RTL とその契約)。"""
    top_verilog: str
    top_name: str
    plan: OverlayPlan
    n_images: int
    clock_div: int
    clock_hz: int
    count_bits: int
    cycle_bits: int
    probe_width: int
    sources: list[str] = field(default_factory=list)
    groups: tuple = ('clock', 'led', 'hex')
    ports: TopPorts = field(default_factory=TopPorts)

    @property
    def core_clock_hz(self) -> float:
        return self.clock_hz / self.clock_div

    def probe_layout(self) -> dict:
        """ISSP probe のビット割付(issp.py が読む契約)。"""
        c = self.count_bits
        return {
            'done': [0, 1],
            'proc': [1, c],
            'match': [1 + c, c],
            'cycle': [1 + 2 * c, self.cycle_bits],
            'width': self.probe_width,
            'source': {'start': SRC_START, 'rst': SRC_RST},
        }

    def to_meta(self) -> dict:
        return {
            'kind': 'overlay',
            'top': self.top_name,
            'n_images': self.n_images,
            'clock_hz': self.clock_hz,
            'clock_div': self.clock_div,
            'core_clock_hz': self.core_clock_hz,
            'cycles_per_image': self.plan.cycles_per_image,
            'expected_total_cycles': self.n_images * self.plan.cycles_per_image,
            'probe': self.probe_layout(),
            'overlay': self.plan.to_dict(),
        }


def clock_block(div: int, clk_port: str) -> tuple[str, list[str]]:
    """clk_core 生成部と SDC 行(create_generated_clock)を返す。

    PLL を使わずトグル FF で分周する(先行実験の実機実証構成)。
    """
    if div == 1:
        return f'    wire clk_core = {clk_port};\n', []
    if div == 2:
        return ("    reg clk_core = 1'b0;\n"
                f'    always @(posedge {clk_port}) clk_core <= ~clk_core;\n',
                ['create_generated_clock -name clk_core -source '
                 f'[get_ports {clk_port}] -divide_by 2 '
                 '[get_registers {clk_core}]'])
    if div == 4:
        return ("    reg divcnt = 1'b0;\n"
                "    reg clk_core = 1'b0;\n"
                f'    always @(posedge {clk_port}) begin divcnt <= ~divcnt;\n'
                '        if (divcnt) clk_core <= ~clk_core; end\n',
                ['create_generated_clock -name clk_core -source '
                 f'[get_ports {clk_port}] -divide_by 4 '
                 '[get_registers {clk_core}]'])
    raise ValueError(f'clock_div={div} は非対応(1 / 2 / 4)')


def harness_module(plan: OverlayPlan) -> str:
    """検証ハーネスのモジュール名(配線の持ち方で変わる唯一のもの)。"""
    return ('overlay_harness_stored' if plan.stored else 'overlay_harness')


def _harness_params(plan: OverlayPlan, n_images: int, count_bits: int,
                    idx_bits: int, cycle_bits: int, indent: str = ' ' * 8) -> str:
    l0 = plan.layers[0]
    items = [
        ('N', n_images), ('IN', plan.in_bits), ('IDX_W', idx_bits),
        ('CW', cycle_bits), ('CNT_W', count_bits),
        ('V', plan.v), ('W', plan.w), ('DEPTH', plan.depth),
        ('CLASSES', plan.classes), ('GS', plan.group),
    ]
    if plan.stored:
        items += [('REC_BITS', plan.record_bits),
                  ('ADDR_F', plan.addr_field_bits)]
    else:
        items += [('L0_INT_STEP', l0.int_step), ('L0_REM_STEP', l0.rem_step)]
    items.append(('PRED_W', plan.pred_w))
    body = ', '.join(f'.{k}({v})' for k, v in items)
    return f'{indent}{body},\n{indent}.MEM_IMG("images.mem"), .MEM_EXP("expected.mem")'


def generate_overlay_system(plan: OverlayPlan, *, n_images: int,
                            top_name: str = 'top', clock_div: int = 2,
                            clock_hz: int = 50_000_000,
                            ports: TopPorts | None = None) -> OverlaySystemGen:
    """オーバーレイ検証システムのトップ RTL を生成する。"""
    ports = ports if ports is not None else TopPorts()
    if len(ports.hex_digits) != 6:
        raise NotImplementedError(
            f'hex {len(ports.hex_digits)} 桁は未対応'
            '(一致数 4 桁 + 処理数 2 桁の 6 桁表示前提)')
    if ports.led_width < 10:
        raise NotImplementedError(
            f'led_width={ports.led_width} は未対応(10 本前提)')

    count_bits = bits_for(n_images)
    cycle_bits = 32
    idx_bits = max(1, (n_images - 1).bit_length())
    probe_width = 1 + 2 * count_bits + cycle_bits
    clk_top, _sdc = clock_block(clock_div, ports.clock)

    bcd_match = _emit_bcd_inc('match_bcd', 4, indent=' ' * 16)
    bcd_proc = _emit_bcd_inc('proc_bcd', 2, indent=' ' * 20)
    hex_ports = '\n'.join(
        f'    output wire [6:0] {h}{"," if i < 5 else ""}'
        for i, h in enumerate(ports.hex_digits))
    led_hi = (f'{ports.led}[9:1] = match_count[8:0];' if count_bits >= 9 else
              f'{ports.led}[9:1] = {{{{{9 - count_bits}{{1\'b0}}}}, match_count}};')

    core_name = 'lgn_overlay_stored' if plan.stored else 'lgn_overlay'
    if plan.stored:
        gate_desc = (
            f'// ゲート記述 = {{op 4bit, a, b}} の {plan.record_bits}bit'
            f'(gatemem_lane0..{plan.nrom - 1}.mem)、\n'
            f'//   配線 = 保存型(記録の a/b フィールド 各 '
            f'{plan.addr_field_bits}bit = 自由配線)')
    else:
        gate_desc = (
            f'// ゲート記述 = op {plan.gate_bits}bit のみ'
            f'(opmem0..{plan.nrom - 1}.mem)、\n'
            '//   配線 = 計算型(addrinit_a/b.mem + reminit.mem の'
            'アキュムレータ初期値)')

    top_v = f"""// mdl-fpga 自動生成 オーバーレイ検証システム({top_name})
// 画像 ROM {n_images} 枚 → {core_name}(V={plan.v} W={plan.w} DEPTH={plan.depth}
//   IN={plan.in_bits} CLASSES={plan.classes} GS={plan.group}) → 期待値照合
{gate_desc}
// 予測 cyc/画像 = {plan.cycles_per_image}(= ceil(IN/V) + DEPTH*(W/V) + 4*DEPTH + 1 + 6)
// clk_core = {ports.clock} / {clock_div}
// probe[0]=done [{count_bits}:1]=proc [{2 * count_bits}:{1 + count_bits}]=match
//   [{probe_width - 1}:{1 + 2 * count_bits}]=cycle(32b) / source[0]=start [1]=rst
// {_digit_range(ports.hex_digits[3], ports.hex_digits[0])}=一致数(10進4桁) / {_digit_range(ports.hex_digits[5], ports.hex_digits[4])}=処理数の百位
`timescale 1ns/1ps
module {top_name} (
    input  wire       {ports.clock},
    output wire [{ports.led_width - 1}:0] {ports.led},
{hex_ports}
);
{clk_top}
    // ---- ISSP(In-System Sources and Probes)----
    wire [1:0]  src;
    wire        done;
    wire [{count_bits - 1}:0] proc_count;
    wire [{count_bits - 1}:0] match_count;
    wire [{cycle_bits - 1}:0] cycle_count;
    wire        proc_pulse, match_pulse;
    wire [{probe_width - 1}:0] prb = {{cycle_count, match_count, proc_count, done}};

    altsource_probe #(
        .sld_auto_instance_index ("YES"),
        .sld_instance_index      (0),
        .instance_id             ("ONB"),
        .probe_width             ({probe_width}),
        .source_width            (2),
        .source_initial_value    ("0"),
        .enable_metastability    ("NO")
    ) issp (.probe(prb), .source(src));

    // ---- source を clk_core 域へ 2FF 同期 ----
    reg [1:0] start_sync = 2'b0, rst_sync = 2'b0;
    always @(posedge clk_core) begin
        start_sync <= {{start_sync[0], src[{SRC_START}]}};
        rst_sync   <= {{rst_sync[0],   src[{SRC_RST}]}};
    end

    {harness_module(plan)} #(
{_harness_params(plan, n_images, count_bits, idx_bits, cycle_bits)}
    ) u_sys (
        .clk(clk_core), .start(start_sync[1]), .rst(rst_sync[1]),
        .done(done), .proc_count(proc_count), .match_count(match_count),
        .cycle_count(cycle_count),
        .proc_pulse(proc_pulse), .match_pulse(match_pulse)
    );

    // ---- 十進カウンタ(7セグ表示用。パック BCD)----
    reg [15:0] match_bcd = 16'd0;
    reg [7:0]  proc_bcd  = 8'd0;
    reg [6:0]  proc100   = 7'd0;
    always @(posedge clk_core) begin
        if (rst_sync[1]) begin
            match_bcd <= 16'd0; proc_bcd <= 8'd0; proc100 <= 7'd0;
        end else begin
            if (match_pulse) begin
{bcd_match}
            end
            if (proc_pulse) begin
                if (proc100 == 7'd99) begin
                    proc100 <= 7'd0;
{bcd_proc}
                end else begin
                    proc100 <= proc100 + 7'd1;
                end
            end
        end
    end

    seg7 u_h0 (.d(match_bcd[ 3: 0]), .seg({ports.hex_digits[0]}));
    seg7 u_h1 (.d(match_bcd[ 7: 4]), .seg({ports.hex_digits[1]}));
    seg7 u_h2 (.d(match_bcd[11: 8]), .seg({ports.hex_digits[2]}));
    seg7 u_h3 (.d(match_bcd[15:12]), .seg({ports.hex_digits[3]}));
    seg7 u_h4 (.d(proc_bcd [ 3: 0]), .seg({ports.hex_digits[4]}));
    seg7 u_h5 (.d(proc_bcd [ 7: 4]), .seg({ports.hex_digits[5]}));

    assign {ports.led}[0]   = done;
    assign {led_hi}
endmodule
"""
    from .overlay import static_sources
    sources = [f'{top_name}.v', 'overlay_argmax.v', 'seg7.v',
               *static_sources(plan.wiring)]
    return OverlaySystemGen(
        top_verilog=top_v, top_name=top_name, plan=plan, n_images=n_images,
        clock_div=clock_div, clock_hz=clock_hz, count_bits=count_bits,
        cycle_bits=cycle_bits, probe_width=probe_width, sources=sources,
        groups=('clock', 'led', 'hex'), ports=ports)


# ---------------------------------------------------------------- ベクトル


def write_overlay_vectors(graph: IRGraph, images: np.ndarray,
                          workdir: str | Path, plan: OverlayPlan,
                          labels: np.ndarray | None = None) -> dict:
    """images.mem / expected.mem / golden.json を書き出し golden を返す。

    expected.mem は **golden の予測クラス**(ir.simulate の argmax)。
    labels を渡すと精度も golden.json に記録する(表示は一致数のまま)。
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    graph = lower_to_primitives(graph)
    images = np.asarray(images, dtype=np.int64)
    in_w = plan.in_bits
    if images.shape[1] != in_w:
        raise ValueError(f'images の幅 {images.shape[1]} != IN_BITS {in_w}')

    img_digits = (in_w + 3) // 4
    (workdir / 'images.mem').write_text(
        '\n'.join(f'{v:0{img_digits}x}' for v in pack_bits(images)) + '\n',
        encoding='utf-8')

    # 大きいネット x 多数枚では simulate の中間層(batch x W の int64)が
    # 数百 MB になるため、枚数をチャンクに割って評価する
    chunk = max(1, min(len(images), max(1, 8_000_000 // max(1, plan.w))))
    preds = np.concatenate([
        np.argmax(simulate(graph, images[i:i + chunk]), axis=1)
        for i in range(0, len(images), chunk)])
    pred_digits = (plan.pred_w + 3) // 4
    (workdir / 'expected.mem').write_text(
        '\n'.join(f'{int(p):0{pred_digits}x}' for p in preds) + '\n',
        encoding='utf-8')

    golden = {'n_images': int(len(images)),
              'preds': [int(p) for p in preds]}
    if labels is not None:
        labels = np.asarray(labels)
        correct = int((preds == labels).sum())
        golden['correct'] = correct
        golden['accuracy'] = correct / len(images)
        golden['labels'] = [int(v) for v in labels]
    (workdir / 'golden.json').write_text(
        json.dumps(golden, ensure_ascii=False, indent=1), encoding='utf-8')
    return golden


# ---------------------------------------------------------------- TB


def generate_overlay_tb(plan: OverlayPlan, n_images: int) -> str:
    """全系シム TB(overlay_harness を直接駆動。一致数とサイクル式を照合)。"""
    count_bits = bits_for(n_images)
    idx_bits = max(1, (n_images - 1).bit_length())
    cyc = plan.cycles_per_image
    timeout = n_images * cyc * 2 + 10000
    return f"""`timescale 1ns/1ps
// mdl-fpga 自動生成 オーバーレイ全系 TB
// golden(ir.simulate の予測)と RTL の一致数、および 1 画像あたりサイクル数が
// 式 {cyc} = ceil(IN/V) + DEPTH*(W/V) + 4*DEPTH + 1 + 6 と厳密一致するかを見る。
module tb;
    localparam integer N         = {n_images};
    localparam integer CYC_MODEL = {cyc};

    reg clk = 0, rst = 1, start = 0;
    wire done;
    wire [{count_bits - 1}:0] proc_count, match_count;
    wire [31:0] cycle_count;
    wire proc_pulse, match_pulse;

    always #5 clk = ~clk;

    {harness_module(plan)} #(
{_harness_params(plan, n_images, count_bits, idx_bits, 32)}
    ) dut (
        .clk(clk), .start(start), .rst(rst), .done(done),
        .proc_count(proc_count), .match_count(match_count),
        .cycle_count(cycle_count),
        .proc_pulse(proc_pulse), .match_pulse(match_pulse)
    );

    integer timeout, percyc;
    initial begin
        rst = 1; start = 0;
        repeat (5) @(posedge clk);
        rst = 0;
        @(posedge clk);
        start = 1;
        timeout = 0;
        while (!done && timeout < {timeout}) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        @(posedge clk);
        $display("done=%b proc=%0d match=%0d cycle=%0d",
                 done, proc_count, match_count, cycle_count);
        percyc = (proc_count > 0) ? (cycle_count / proc_count) : 0;
        $display("cyc/image = %0d (model %0d)", percyc, CYC_MODEL);
        if (done && match_count == N && proc_count == N
            && percyc == CYC_MODEL && cycle_count == N * CYC_MODEL)
            $display("OVERLAY RESULT: ALLPASS (%0d/%0d match, cyc/img=%0d)",
                     match_count, N, CYC_MODEL);
        else
            $display("OVERLAY RESULT: FAILED done=%b proc=%0d match=%0d expected %0d cyc/img=%0d expected %0d",
                     done, proc_count, match_count, N, percyc, CYC_MODEL);
        start = 0;
        $finish;
    end
endmodule
"""


def generate_overlay_top_tb(gen: OverlaySystemGen) -> str:
    """トップ(ISSP スタブ + 分周 + 7セグ)まで含めた TB。

    source は ISSP スタブへ階層参照で書く(実機の quartus_stp と同じ手順)。
    """
    n = gen.n_images
    plan = gen.plan
    cyc = plan.cycles_per_image
    timeout = n * cyc * gen.clock_div * 2 + 20000
    match_bcd = to_bcd(n, 4)
    proc_bcd = to_bcd(n // 100, 2)
    p = gen.ports
    hex_conn = ', '.join(f'.{h}(h{i})' for i, h in enumerate(p.hex_digits))
    return f"""`timescale 1ns/1ps
// mdl-fpga 自動生成 オーバーレイ トップ TB(ISSP スタブ使用)
// 実機手順(rst パルス → start)をそのまま踏み、probe の done/proc/match/cycle
// と 7セグの BCD を照合する。**全ベクトル一致が前提の TB**(golden 予測 ROM)。
module tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    wire [{p.led_width - 1}:0] ledr;
    wire [6:0] h0, h1, h2, h3, h4, h5;

    {gen.top_name} dut (.{p.clock}(clk), .{p.led}(ledr), {hex_conn});

    integer guard;
    initial begin
        // rst=1 → 0 → start=1(先行実験の ISSP 手順と同じ)
        dut.issp.source = 2'b10;
        repeat (10) @(posedge clk);
        dut.issp.source = 2'b00;
        repeat (10) @(posedge clk);
        dut.issp.source = 2'b01;
        guard = 0;
        while (!ledr[0] && guard < {timeout}) begin
            @(posedge clk);
            guard = guard + 1;
        end
        repeat (4) @(posedge clk);
        $display("done=%b proc=%0d match=%0d cycle=%0d",
                 ledr[0], dut.u_sys.proc_count, dut.u_sys.match_count,
                 dut.u_sys.cycle_count);
        $display("bcd match=%h proc=%h", dut.match_bcd, dut.proc_bcd);
        if (ledr[0] && dut.u_sys.match_count == {n}
            && dut.u_sys.proc_count == {n}
            && dut.u_sys.cycle_count == {n} * {cyc}
            && dut.match_bcd == 16'h{match_bcd:04x}
            && dut.proc_bcd == 8'h{proc_bcd:02x})
            $display("OVERLAY TOP RESULT: ALLPASS");
        else
            $display("OVERLAY TOP RESULT: FAILED");
        $finish;
    end
endmodule
"""


def generate_addrgen_tb(plan: OverlayPlan) -> str:
    """アドレス生成器の全数照合 TB(RTL の発行アドレス ⇔ Python 参照列)。

    lgn_overlay を 1 画像ぶん走らせ、発行された全ゲート語のレーン別アドレス
    a_p1/b_p1 と、活性 RAM へ提示される {rd_x_word, rd_x_sel} の再結合を、
    addrref_{a,b}.mem(hdlgen.overlay.emit_addrref)と 1 件ずつ突き合わせる。
    """
    v, w, depth = plan.v, plan.w, plan.depth
    l0 = plan.layers[0]
    total_words = depth * plan.layer_words
    timeout = 20 * total_words + 100000
    return f"""`timescale 1ns/1ps
// mdl-fpga 自動生成 アドレス生成 TB(全数照合)
module tb;
    localparam integer V           = {v};
    localparam integer W           = {w};
    localparam integer DEPTH       = {depth};
    localparam integer IN_BITS     = {plan.in_bits};
    localparam integer CLASSES     = {plan.classes};
    localparam integer GS          = {plan.group};
    localparam integer LOGV        = {plan.logv};
    localparam integer ACT_ADDR    = {plan.act_addr};
    localparam integer LAYER_WORDS = {plan.layer_words};
    localparam integer TOTAL_WORDS = {total_words};
    localparam integer TOTAL_GATES = {plan.total_gates};
    localparam integer PA          = ACT_ADDR + LOGV;
    localparam integer AWL         = {plan.aw};
    localparam integer MAXPRINT    = 20;

    reg [AWL-1:0] ref_a [0:TOTAL_GATES-1];
    reg [AWL-1:0] ref_b [0:TOTAL_GATES-1];

    reg clk = 0, rst = 1, start = 0;
    reg [IN_BITS-1:0] img;
    wire done;
    wire [{plan.pred_w - 1}:0] pred;

    always #5 clk = ~clk;

    lgn_overlay #(
        .V(V), .W(W), .DEPTH(DEPTH), .IN_BITS(IN_BITS), .CLASSES(CLASSES),
        .GS(GS), .L0_INT_STEP({l0.int_step}), .L0_REM_STEP({l0.rem_step}),
        .PRED_W({plan.pred_w})
    ) dut (
        .clk(clk), .rst(rst), .start(start), .img(img),
        .done(done), .pred(pred)
    );

    integer widx, k, i, errors, checks, timeout, printed;
    reg [AWL-1:0] ea, eb;
    reg [PA-1:0]  ga, gb;

    task chk;
        input [63:0] got;
        input [63:0] exp_v;
        input [8*12-1:0] tag;
        begin
            checks = checks + 1;
            if (got !== exp_v) begin
                errors = errors + 1;
                if (printed < MAXPRINT) begin
                    printed = printed + 1;
                    $display("MISMATCH_ADDR %0s word=%0d lane=%0d got=%0d exp=%0d",
                             tag, widx, k, got, exp_v);
                end
            end
        end
    endtask

    initial begin
        $readmemh("addrref_a.mem", ref_a);
        $readmemh("addrref_b.mem", ref_b);
        for (i = 0; i < IN_BITS; i = i + 1) img[i] = ((i * 7 + 3) % 5) == 0;

        rst = 1; start = 0;
        widx = 0; errors = 0; checks = 0; printed = 0;
        repeat (5) @(posedge clk);
        rst = 0;
        @(posedge clk);
        start = 1;

        timeout = 0;
        while (!done && timeout < {timeout}) begin
            @(posedge clk);
            #1;
            if (dut.v1 === 1'b1) begin
                if (widx >= TOTAL_WORDS) begin
                    errors = errors + 1;
                    if (printed < MAXPRINT) begin
                        printed = printed + 1;
                        $display("OVER-ISSUE: widx=%0d >= %0d", widx, TOTAL_WORDS);
                    end
                end else begin
                    for (k = 0; k < V; k = k + 1) begin
                        ea = ref_a[widx * V + k];
                        eb = ref_b[widx * V + k];
                        ga = {{dut.rd_a_word[k], dut.rd_a_sel[k]}};
                        gb = {{dut.rd_b_word[k], dut.rd_b_sel[k]}};
                        chk({{{{(64-AWL){{1'b0}}}}, dut.a_p1[k]}},
                            {{{{(64-AWL){{1'b0}}}}, ea}}, "a_p1");
                        chk({{{{(64-AWL){{1'b0}}}}, dut.b_p1[k]}},
                            {{{{(64-AWL){{1'b0}}}}, eb}}, "b_p1");
                        chk({{{{(64-PA){{1'b0}}}},  ga}},
                            {{{{(64-PA){{1'b0}}}}, ea[PA-1:0]}}, "rd_a");
                        chk({{{{(64-PA){{1'b0}}}},  gb}},
                            {{{{(64-PA){{1'b0}}}}, eb[PA-1:0]}}, "rd_b");
                    end
                end
                widx = widx + 1;
            end
            timeout = timeout + 1;
        end

        $display("issued words=%0d (expect %0d) checks=%0d (expect %0d) errors=%0d done=%b",
                 widx, TOTAL_WORDS, checks, 4 * V * TOTAL_WORDS, errors, done);
        if (done && errors == 0 && widx == TOTAL_WORDS
            && checks == 4 * V * TOTAL_WORDS)
            $display("ADDRGEN RESULT: ALLPASS (%0d address checks)", checks);
        else
            $display("ADDRGEN RESULT: FAILED");
        start = 0;
        $finish;
    end
endmodule
"""
