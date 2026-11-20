"""分類デモシステム生成(ROM → パイプライン推論 → 正解カウント → 7セグ)。

group_sum 終端の任意分類 IR に対して汎用:
    sys_rom(画像+ラベル、同期1clk読み、M10K 推論)
      → FSM(FETCH/LOAD/CORE/CHECK/THROTTLE/DONE、KEY[0]=リスタート)
      → mdl_core_pipe(全ノード直後 FF、latency=L)
      → argmax 比較木(社内先行実験 gen_core.py の実証パターン移植。
         同点=小さい index 勝ち=np.argmax。emit 前に乱数自己テスト)
      → BCD デケードカウンタ(double-dabble 不採用: 変換雲が不要で、
         ビューアの hex 表示がそのまま 10 進に読める)
      → seg7 × 6(HEX3-0=正解4桁 / HEX5-4=処理数の百位)

$readmemh はファイル名のみ(cwd 罠対策: sim workdir / sim_build / Quartus
プロジェクトへ rom.hex をコロケートする)。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.simulate import simulate

from ..vectors import pack_bits
from .core import CoreGen, generate_core
from .ports import TopPorts

# ---------------------------------------------------------------- argmax 木


def tree_order(k: int) -> list[tuple]:
    """ペア縮約の比較木。[(node_idx, left_ref, right_ref)]、ref=('s',c)|('n',i)。

    奇数個の余りは次レベルへ持ち上げる(社内先行実験 gen_core.py と同型)。
    """
    items: list[tuple] = [('s', c) for c in range(k)]
    nodes: list[tuple] = []
    idx = 0
    while len(items) > 1:
        nxt: list[tuple] = []
        for i in range(0, len(items) - 1, 2):
            nodes.append((idx, items[i], items[i + 1]))
            nxt.append(('n', idx))
            idx += 1
        if len(items) % 2 == 1:
            nxt.append(items[-1])
        items = nxt
    return nodes


def eval_tree(scores) -> int:
    """tree_order の Python 評価(自己テスト用)。"""
    k = len(scores)
    vals: dict[tuple, tuple] = {('s', c): (scores[c], c) for c in range(k)}
    for idx, left, right in tree_order(k):
        ls, li = vals[left]
        rs, ri = vals[right]
        vals[('n', idx)] = (rs, ri) if rs > ls else (ls, li)  # 同点=左(小index)
    return vals[('n', len(tree_order(k)) - 1)][1] if tree_order(k) else 0


def argmax_tree_self_test(k: int, max_score: int, n_cases: int = 10000,
                          seed: int = 0) -> None:
    """比較木 = np.argmax(同点は小 index)を乱数+同点多発ケースで検証。"""
    rng = np.random.default_rng(seed)
    for hi in (max_score, 3, 1):
        for _ in range(n_cases // 3):
            scores = rng.integers(0, hi + 1, size=k)
            assert eval_tree(scores.tolist()) == int(np.argmax(scores)), scores


def _emit_argmax_tree(k: int, sw: int, pw: int, src: str) -> list[str]:
    """比較木 Verilog(src = パック済みスコアバス)。pred 信号を定義する。"""
    lines = ['    // argmax 比較木(同点=小さい index 勝ち = np.argmax)']
    for c in range(k):
        lines.append(
            f'    wire [{sw - 1}:0] s{c} = '
            f'{src}[{(c + 1) * sw - 1}:{c * sw}];')
    nodes = tree_order(k)

    def ref_s(ref):
        return f's{ref[1]}' if ref[0] == 's' else f'n{ref[1]}_s'

    def ref_i(ref):
        return f"{pw}'d{ref[1]}" if ref[0] == 's' else f'n{ref[1]}_i'

    for idx, left, right in nodes:
        lines.append(f'    wire [{sw - 1}:0] n{idx}_s;')
        lines.append(f'    wire [{pw - 1}:0] n{idx}_i;')
        lines.append(
            f'    wire n{idx}_gt = ({ref_s(right)} > {ref_s(left)});')
        lines.append(
            f'    assign n{idx}_s = n{idx}_gt ? {ref_s(right)} : {ref_s(left)};')
        lines.append(
            f'    assign n{idx}_i = n{idx}_gt ? {ref_i(right)} : {ref_i(left)};')
    root = len(nodes) - 1
    lines.append(f'    wire [{pw - 1}:0] pred = n{root}_i;')
    return lines


# ---------------------------------------------------------------- BCD


def _digit_range(hi: str, lo: str) -> str:
    """'HEX3','HEX0' → 'HEX3-0'(共通プレフィクス時の慣用表記)。それ以外は 'D3-D0'。"""
    if hi[:-1] == lo[:-1] and hi[-1].isdigit() and lo[-1].isdigit():
        return f'{hi}-{lo[-1]}'
    return f'{hi}-{lo}'


def _emit_bcd_inc(reg: str, digits: int, indent: str = ' ' * 24) -> str:
    """パック BCD レジスタの +1(デケード桁上げ)を if 連鎖で生成する。"""
    lines: list[str] = []

    def rec(d: int, pad: str) -> None:
        lo, hi = d * 4, d * 4 + 3
        if d == digits - 1:
            lines.append(f"{pad}if ({reg}[{hi}:{lo}] != 4'd9)")
            lines.append(f"{pad}    {reg}[{hi}:{lo}] <= {reg}[{hi}:{lo}] + 4'd1;")
            lines.append(f"{pad}else {reg}[{hi}:{lo}] <= 4'd0;")
            return
        lines.append(f"{pad}if ({reg}[{hi}:{lo}] != 4'd9)")
        lines.append(f"{pad}    {reg}[{hi}:{lo}] <= {reg}[{hi}:{lo}] + 4'd1;")
        lines.append(f"{pad}else begin")
        lines.append(f"{pad}    {reg}[{hi}:{lo}] <= 4'd0;")
        rec(d + 1, pad + '    ')
        lines.append(f"{pad}end")

    rec(0, indent)
    return '\n'.join(lines)


def to_bcd(value: int, digits: int) -> int:
    """整数 → パック BCD(テスト/golden 照合用)。"""
    out = 0
    for d in range(digits):
        out |= (value % 10) << (4 * d)
        value //= 10
    return out


# ---------------------------------------------------------------- 生成本体


@dataclass
class SystemGen:
    top_verilog: str
    rom_verilog: str
    core: CoreGen
    sources: list[str] = field(default_factory=list)
    n_images: int = 0
    label_bits: int = 0
    rom_word_bits: int = 0
    addr_bits: int = 0
    latency: int = 0
    k: int = 0
    score_bits: int = 0
    pred_bits: int = 0
    slow_div: int = 0
    system_meta: dict = field(default_factory=dict)
    # 使用資源セットの単一情報源(synth write_project_files の groups へ渡す)
    groups: tuple = ('clock', 'led', 'switch', 'button', 'hex')
    ports: TopPorts = field(default_factory=TopPorts)


def generate_system(graph: IRGraph, *, n_images: int,
                    top_name: str = 'top',
                    clock_hz: int = 50_000_000,
                    slow_images_per_s: float = 10.0,
                    ports: TopPorts | None = None) -> SystemGen:
    ports = ports if ports is not None else TopPorts()
    if len(ports.hex_digits) != 6:
        raise NotImplementedError(
            f'hex {len(ports.hex_digits)}桁は未対応(正解4桁+処理数2桁の '
            '6桁表示前提。桁割付の設計を追加してから対応する)')
    if ports.led_width != 10:
        raise NotImplementedError(
            f'led_width={ports.led_width} は未対応(state/heartbeat 等の '
            'ビット割付が 10 本前提)')
    graph = lower_to_primitives(graph)
    gs_nodes = [n for n in graph.nodes if n.type == 'group_sum']
    assert len(gs_nodes) == 1, 'group_sum 終端の分類 IR が対象'
    gs = gs_nodes[0]
    k = int(gs.params['k'])
    group = int(gs.params['in_width']) // k
    sw = max(1, math.ceil(math.log2(group + 1)))
    pw = max(1, (k - 1).bit_length())
    max_score = group

    argmax_tree_self_test(k, max_score)

    core = generate_core(graph, module_name='mdl_core_pipe', pipelined=True)
    in_w = core.in_width
    label_bits = max(1, (k - 1).bit_length())
    word = in_w + label_bits
    aw = max(1, (n_images - 1).bit_length())
    lat = core.latency
    slow_div = max(4, int(clock_hz / slow_images_per_s))

    rom_v = f"""// mdl-fpga 自動生成 画像ROM(N={n_images}, word={word}bit:
//   [{in_w - 1}:0]=画像(pack_bits規約=bit i が画素 i)
//   [{word - 1}:{in_w}]=ラベル)
// 同期1クロック読み → M10K 推論。rom.hex はプロジェクト/シム作業dirへコロケート。
module sys_rom (
    input  wire            clk,
    input  wire [{aw - 1}:0]  addr,
    output reg  [{word - 1}:0] q
);
    (* ramstyle = "M10K" *) reg [{word - 1}:0] mem [0:{n_images - 1}];
    initial $readmemh("rom.hex", mem);
    always @(posedge clk) q <= mem[addr];
endmodule
"""

    argmax_lines = '\n'.join(_emit_argmax_tree(k, sw, pw, 'scores'))
    bcd_correct = _emit_bcd_inc('correct_bcd', 4)
    bcd_proc = _emit_bcd_inc('proc_bcd', 2, indent=' ' * 28)

    hex_ports = '\n'.join(
        f'    output wire [6:0] {h}{"," if i < 5 else ""}'
        for i, h in enumerate(ports.hex_digits))
    top_v = f"""// mdl-fpga 自動生成 分類デモシステム({top_name})
// ROM({n_images}枚) → mdl_core_pipe(latency={lat}clk) → argmax → 正解カウント → 7セグ
// {ports.switch}[0]=低速モード(SLOW_DIVで分周) / {ports.button}[0]=リスタート(負論理) / 完了後ホールド
// {_digit_range(ports.hex_digits[3], ports.hex_digits[0])}=正解数(10進4桁) / {_digit_range(ports.hex_digits[5], ports.hex_digits[4])}=処理数の百位(10進2桁)
// {ports.led}[0]=heartbeat [3:1]=state [4]={ports.switch}[0] [8]=running [9]=done
module {top_name} #(
    parameter integer SLOW_DIV = {slow_div}
) (
    input  wire       {ports.clock},
    input  wire [{ports.switch_width - 1}:0] {ports.switch},
    input  wire [{ports.button_width - 1}:0] {ports.button},
    output wire [{ports.led_width - 1}:0] {ports.led},
{hex_ports}
);
    localparam integer N     = {n_images};
    localparam integer LAT   = {lat};
    localparam integer IN_W  = {in_w};

    wire clk = {ports.clock};

    // ---- リスタート({ports.button}[0] 負論理、2FF 同期) ----
    reg [1:0] key0_sync = 2'b11;
    always @(posedge clk) key0_sync <= {{key0_sync[0], {ports.button}[0]}};
    wire restart = ~key0_sync[1];

    // ---- ROM ----
    reg  [{aw - 1}:0] img_idx = {aw}'d0;
    wire [{word - 1}:0] rom_q;
    sys_rom u_rom (.clk(clk), .addr(img_idx), .q(rom_q));

    // ---- コア(パイプライン) ----
    reg  [IN_W-1:0] xin = {{IN_W{{1'b0}}}};
    reg  [{label_bits - 1}:0] label_r = {label_bits}'d0;
    wire [{k * sw - 1}:0] scores;
    mdl_core_pipe u_core (.clk(clk), .x(xin), .y(scores));

{argmax_lines}

    // ---- FSM ----
    localparam [2:0] S_RESTART = 3'd0, S_FETCH = 3'd1, S_LOAD = 3'd2,
                     S_CORE = 3'd3, S_CHECK = 3'd4, S_THROTTLE = 3'd5,
                     S_DONE = 3'd6;
    reg [2:0]  state = S_RESTART;
    reg [15:0] lat_cnt = 16'd0;
    reg [31:0] slow_cnt = 32'd0;
    reg [15:0] correct_bcd = 16'd0;   // 10進4桁(パックBCD)
    reg [7:0]  proc_bcd = 8'd0;       // 処理数の百位(10進2桁)
    reg [6:0]  proc100_cnt = 7'd0;    // 0..99

    always @(posedge clk) begin
        if (restart) begin
            state <= S_RESTART;
        end else begin
            case (state)
                S_RESTART: begin
                    img_idx     <= {aw}'d0;
                    correct_bcd <= 16'd0;
                    proc_bcd    <= 8'd0;
                    proc100_cnt <= 7'd0;
                    slow_cnt    <= 32'd0;
                    state       <= S_FETCH;
                end
                S_FETCH: begin
                    // この posedge で rom_q <= mem[img_idx]
                    state <= S_LOAD;
                end
                S_LOAD: begin
                    xin     <= rom_q[IN_W-1:0];
                    label_r <= rom_q[{word - 1}:IN_W];
                    lat_cnt <= 16'd0;
                    state   <= S_CORE;
                end
                S_CORE: begin
                    lat_cnt <= lat_cnt + 16'd1;
                    if (lat_cnt == LAT - 1) state <= S_CHECK;
                end
                S_CHECK: begin
                    if (pred == label_r) begin
{bcd_correct}
                    end
                    if (proc100_cnt == 7'd99) begin
                        proc100_cnt <= 7'd0;
{bcd_proc}
                    end else begin
                        proc100_cnt <= proc100_cnt + 7'd1;
                    end
                    if (img_idx == N - 1) begin
                        state <= S_DONE;
                    end else begin
                        img_idx <= img_idx + {aw}'d1;
                        slow_cnt <= 32'd0;
                        state <= {ports.switch}[0] ? S_THROTTLE : S_FETCH;
                    end
                end
                S_THROTTLE: begin
                    slow_cnt <= slow_cnt + 32'd1;
                    if (slow_cnt >= SLOW_DIV - 1) state <= S_FETCH;
                end
                S_DONE: state <= S_DONE;   // ホールド(restart 待ち)
                default: state <= S_RESTART;
            endcase
        end
    end

    // ---- 7セグ表示 ----
    seg7 u_h0 (.d(correct_bcd[ 3: 0]), .seg({ports.hex_digits[0]}));
    seg7 u_h1 (.d(correct_bcd[ 7: 4]), .seg({ports.hex_digits[1]}));
    seg7 u_h2 (.d(correct_bcd[11: 8]), .seg({ports.hex_digits[2]}));
    seg7 u_h3 (.d(correct_bcd[15:12]), .seg({ports.hex_digits[3]}));
    seg7 u_h4 (.d(proc_bcd  [ 3: 0]), .seg({ports.hex_digits[4]}));
    seg7 u_h5 (.d(proc_bcd  [ 7: 4]), .seg({ports.hex_digits[5]}));

    // ---- LED ----
    reg [25:0] hb = 26'd0;
    always @(posedge clk) hb <= hb + 26'd1;
    assign {ports.led}[0]   = hb[25];              // heartbeat
    assign {ports.led}[3:1] = state;
    assign {ports.led}[4]   = {ports.switch}[0];
    assign {ports.led}[7:5] = 3'b000;
    assign {ports.led}[8]   = (state != S_DONE);   // running
    assign {ports.led}[9]   = (state == S_DONE);   // done

endmodule
"""

    system_meta = _build_system_meta(core, gs.id, n_images=n_images,
                                     label_bits=label_bits, aw=aw, lat=lat,
                                     pred_bits=pw)

    return SystemGen(
        top_verilog=top_v,
        rom_verilog=rom_v,
        core=core,
        sources=[f'{top_name}.v', 'mdl_core_pipe.v', 'sys_rom.v', 'seg7.v'],
        n_images=n_images,
        label_bits=label_bits,
        rom_word_bits=word,
        addr_bits=aw,
        latency=lat,
        k=k,
        score_bits=sw,
        pred_bits=pw,
        slow_div=slow_div,
        system_meta=system_meta,
        groups=('clock', 'led', 'switch', 'button', 'hex'),
        ports=ports,
    )


def _build_system_meta(core: CoreGen, gs_id: str, *, n_images: int,
                       label_bits: int, aw: int, lat: int,
                       pred_bits: int) -> dict:
    """ビューア「システムモード」用メタ(nodes に path、edges、set_x 禁止)。"""
    return {
        'mode': 'system',
        'latency': lat,
        'allow_set_x': False,
        'n_images': n_images,
        'extra_signals': {
            'x': 'u_sys.xin',
            'y_comb': f'u_sys.u_core.w_{gs_id}',
            'y': f'u_sys.u_core.w_{gs_id}',
        },
        'nodes': [
            {'id': 'img_idx', 'type': 'system', 'width': aw,
             'path': 'u_sys.img_idx', 'label': 'ROM index'},
            {'id': 'label', 'type': 'system', 'width': label_bits,
             'path': 'u_sys.label_r', 'label': '正解ラベル'},
            {'id': 'pred', 'type': 'system', 'width': pred_bits,
             'path': 'u_sys.pred', 'label': 'argmax(pred)'},
            {'id': 'correct', 'type': 'system', 'width': 16,
             'path': 'u_sys.correct_bcd', 'label': '正解数(BCD)'},
            {'id': 'proc100', 'type': 'system', 'width': 8,
             'path': 'u_sys.proc_bcd', 'label': '処理数x100(BCD)'},
            {'id': 'fsm', 'type': 'system', 'width': 3,
             'path': 'u_sys.state', 'label': 'FSM state'},
        ],
        'edges': [
            {'src': 'img_idx', 'dst': 'label', 'slot': 0},
            {'src': 'img_idx', 'dst': 'x', 'slot': 0},
            {'src': gs_id, 'dst': 'pred', 'slot': 0},
            {'src': 'pred', 'dst': 'correct', 'slot': 0},
            {'src': 'label', 'dst': 'correct', 'slot': 1},
        ],
    }


def generate_system_view_wrapper(top_name: str = 'top',
                                 ports: TopPorts | None = None) -> str:
    """ビューア(システムモード)用の薄ラッパ。

    rst → button[0](負論理リスタート)、switch=0(高速)、HEX/LED は開放。
    ブリッジの clk/rst 駆動契約(top_view)をそのまま満たす。
    """
    p = ports if ports is not None else TopPorts()
    key_expr = f"{{{p.button_width - 1}'b{'1' * (p.button_width - 1)}, ~rst}}"
    hex_conn = ', '.join(
        f'.{h}(h{i})' for i, h in enumerate(p.hex_digits))
    return f"""`timescale 1ns/1ps
// mdl-fpga 自動生成 ビューア用薄ラッパ(システムモード)
module top_view (
    input wire clk,
    input wire rst
);
    wire [{p.led_width - 1}:0] ledr;
    wire [6:0] h0, h1, h2, h3, h4, h5;
    {top_name} u_sys (
        .{p.clock}(clk), .{p.switch}({p.switch_width}'b0), .{p.button}({key_expr}), .{p.led}(ledr),
        {hex_conn});
endmodule
"""


# ---------------------------------------------------------------- ベクトル/golden


def write_system_vectors(graph: IRGraph, images: np.ndarray,
                         labels: np.ndarray, workdir: str | Path,
                         label_bits: int) -> dict:
    """rom.hex / golden_pred.hex / golden.json を書き出し golden を返す。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    graph = lower_to_primitives(graph)
    in_w = graph.inputs[0].width
    word_bits = in_w + label_bits
    digits = (word_bits + 3) // 4

    x_vals = pack_bits(images.astype(np.int64))
    lines = []
    for xv, lb in zip(x_vals, labels):
        word = xv | (int(lb) << in_w)
        lines.append(f'{word:0{digits}x}')
    (workdir / 'rom.hex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    scores = simulate(graph, images.astype(np.int64))
    preds = np.argmax(scores, axis=1)
    correct = int((preds == labels).sum())
    (workdir / 'golden_pred.hex').write_text(
        '\n'.join(f'{int(p):x}' for p in preds) + '\n', encoding='utf-8')
    golden = {
        'n_images': int(len(images)),
        'correct': correct,
        'accuracy': correct / len(images),
        'per_image': [
            {'label': int(lb), 'pred': int(p), 'ok': bool(p == lb)}
            for lb, p in zip(labels, preds)
        ],
    }
    (workdir / 'golden.json').write_text(
        json.dumps(golden, ensure_ascii=False, indent=1), encoding='utf-8')
    return golden


# ---------------------------------------------------------------- システム TB


def generate_system_tb(gen: SystemGen, golden: dict,
                       slow_div_tb: int = 4) -> str:
    """小型システムの自己検証 TB(pred逐次照合+BCD不変条件+完走+restart+throttle)。"""
    n = gen.n_images
    p = gen.ports
    correct_bcd = to_bcd(golden['correct'], 4)
    proc_bcd = to_bcd(golden['n_images'] // 100, 2)
    per_image_clks = gen.latency + 6
    timeout = (n * per_image_clks + 200) * 4   # throttle 走行含む余裕
    hex_conn = ', '.join(f'.{h}(h{i})' for i, h in enumerate(p.hex_digits))
    return f"""`timescale 1ns/1ps
// mdl-fpga 自動生成 システム自己検証TB
module tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg [{p.switch_width - 1}:0] sw = {p.switch_width}'d0;
    reg [{p.button_width - 1}:0] key = {p.button_width}'b{'1' * p.button_width};
    wire [{p.led_width - 1}:0] ledr;
    wire [6:0] h0, h1, h2, h3, h4, h5;

    top #(.SLOW_DIV({slow_div_tb})) dut (
        .{p.clock}(clk), .{p.switch}(sw), .{p.button}(key), .{p.led}(ledr),
        {hex_conn});

    reg [{gen.pred_bits - 1}:0] golden_pred [0:{n - 1}];
    integer errors, checks, i, guard;

    task check_bcd_digits;
        integer d;
        begin
            for (d = 0; d < 4; d = d + 1)
                if (dut.correct_bcd[d*4 +: 4] > 4'd9) begin
                    errors = errors + 1;
                    $display("BCD_INVALID correct digit %0d", d);
                end
        end
    endtask

    task run_pass(input integer expect_checks);
        begin
            checks = 0;
            guard = 0;
            while (dut.state != 3'd6 && guard < {timeout}) begin
                @(posedge clk); #1;
                guard = guard + 1;
                if (dut.state == 3'd4) begin   // S_CHECK
                    if (dut.pred !== golden_pred[dut.img_idx]) begin
                        errors = errors + 1;
                        $display("MISMATCH_PRED img=%0d got=%0d exp=%0d",
                                 dut.img_idx, dut.pred,
                                 golden_pred[dut.img_idx]);
                    end
                    check_bcd_digits;
                    checks = checks + 1;
                    // S_CHECK は1枚につき1クロック → 次状態まで進める
                    @(posedge clk); #1;
                    guard = guard + 1;
                end
            end
            if (dut.state != 3'd6) begin
                errors = errors + 1;
                $display("TIMEOUT guard=%0d state=%0d", guard, dut.state);
            end
            if (checks != expect_checks) begin
                errors = errors + 1;
                $display("BAD_CHECKS got=%0d exp=%0d", checks, expect_checks);
            end
            if (dut.correct_bcd !== 16'h{correct_bcd:04x}) begin
                errors = errors + 1;
                $display("BAD_CORRECT got=%h exp=%h",
                         dut.correct_bcd, 16'h{correct_bcd:04x});
            end
            if (dut.proc_bcd !== 8'h{proc_bcd:02x}) begin
                errors = errors + 1;
                $display("BAD_PROC got=%h exp=%h", dut.proc_bcd, 8'h{proc_bcd:02x});
            end
        end
    endtask

    initial begin
        $readmemh("golden_pred.hex", golden_pred);
        errors = 0;

        // --- pass1: 高速モード ---
        repeat (4) @(posedge clk);
        run_pass({n});

        // --- restart(KEY[0] パルス)→ pass2: 低速モード(throttle 経路)---
        key[0] = 1'b0;
        repeat (4) @(posedge clk);
        key[0] = 1'b1;
        // 同期解除後、S_RESTART 本体がカウンタをクリアするまで数クロック待つ
        repeat (4) @(posedge clk); #1;
        if (dut.correct_bcd !== 16'd0) begin
            errors = errors + 1;
            $display("RESTART_NOT_CLEARED %h", dut.correct_bcd);
        end
        sw[0] = 1'b1;
        run_pass({n});

        if (errors == 0) $display("ALLPASS");
        else $display("FAILED %0d", errors);
        $finish;
    end
endmodule
"""
