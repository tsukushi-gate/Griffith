"""オンチップ自己検証回路の生成(UART 不要の実機 bit 一致確認)。

参照シム(golden)のベクトルと期待値を case 文 ROM として焼き込み、
実機の電源投入直後に全ベクトルを照合する。結果は LED で目視できる:
    pass=1(全一致)/ fail=1(1件でも不一致、sticky)/ done=1(照合完了)
demo2 の ISSP セルフテストの思想を、専用ツール無しで読める形に単純化したもの。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core import CoreGen


@dataclass
class SelftestGen:
    verilog: str
    n_vectors: int


def generate_selftest(gen: CoreGen, x_vals: list[int], exp_vals: list[int],
                      settle_cycles: int = 3) -> SelftestGen:
    """ベクトル焼き込み型の selftest モジュールを生成する。

    Args:
        gen: コア生成結果(幅とモジュール名を使う)
        x_vals: 入力ベクトル(pack_bits 済み整数)
        exp_vals: 期待出力(出力バス表現へ pack 済み整数)
    """
    assert len(x_vals) == len(exp_vals) and len(x_vals) >= 1
    n = len(x_vals)
    iw, ow = gen.in_width, gen.out_width
    idx_w = max(1, math.ceil(math.log2(n + 1)))

    rom_lines = []
    for i, (xv, ev) in enumerate(zip(x_vals, exp_vals)):
        rom_lines.append(
            f"            {idx_w}'d{i}: begin "
            f"vrom = {iw}'h{xv:x}; erom = {ow}'h{ev:x}; end")
    rom = '\n'.join(rom_lines)

    src = f"""// mdl-fpga 自動生成オンチップ自己検証(selftest)
// {n} ベクトルを ROM(case 文)として焼き込み、mdl_core の実機出力と照合する。
// pass=全一致 / fail=不一致あり(sticky)/ done=照合完了。
module selftest (
    input  wire clk,
    input  wire rst,
    output reg  done = 1'b0,
    output reg  fail = 1'b0,
    output wire pass
);
    localparam integer N = {n};

    reg [{idx_w - 1}:0] vec_idx = {idx_w}'d0;
    reg [{iw - 1}:0]    x_cur   = {iw}'d0;
    reg [{ow - 1}:0]    exp_cur = {ow}'d0;

    wire [{ow - 1}:0] y;
    {gen.module_name} u_core (.x(x_cur), .y(y));

    // ベクトル ROM(組み合わせ case)
    reg [{iw - 1}:0] vrom;
    reg [{ow - 1}:0] erom;
    always @* begin
        case (vec_idx)
{rom}
            default: begin vrom = {iw}'d0; erom = {ow}'d0; end
        endcase
    end

    localparam [1:0] S_APPLY = 2'd0, S_WAIT = 2'd1, S_CHECK = 2'd2, S_DONE = 2'd3;
    reg [1:0] st   = S_APPLY;
    reg [3:0] wcnt = 4'd0;

    always @(posedge clk) begin
        if (rst) begin
            st      <= S_APPLY;
            vec_idx <= {idx_w}'d0;
            done    <= 1'b0;
            fail    <= 1'b0;
            wcnt    <= 4'd0;
        end else begin
            case (st)
                S_APPLY: begin
                    x_cur   <= vrom;
                    exp_cur <= erom;
                    wcnt    <= 4'd0;
                    st      <= S_WAIT;
                end
                S_WAIT: begin   // 組み合わせ雲の伝搬待ち(余裕をみて {settle_cycles} 周期)
                    wcnt <= wcnt + 4'd1;
                    if (wcnt >= 4'd{settle_cycles}) st <= S_CHECK;
                end
                S_CHECK: begin
                    if (y != exp_cur) fail <= 1'b1;
                    if (vec_idx == N - 1) begin
                        st <= S_DONE;
                    end else begin
                        vec_idx <= vec_idx + {idx_w}'d1;
                        st      <= S_APPLY;
                    end
                end
                S_DONE: done <= 1'b1;
            endcase
        end
    end

    assign pass = done & ~fail;
endmodule
"""
    return SelftestGen(verilog=src, n_vectors=n)


def generate_selftest_tb(n_vectors: int, settle_cycles: int = 3) -> str:
    """selftest 単体の TB(done まで回して pass/fail を判定)。"""
    max_cycles = (settle_cycles + 4) * n_vectors + 100
    return f"""`timescale 1ns/1ps
module tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;
    wire done, fail, pass;
    selftest dut (.clk(clk), .rst(rst), .done(done), .fail(fail), .pass(pass));
    integer i;
    initial begin
        repeat (5) @(posedge clk);
        rst = 1'b0;
        for (i = 0; i < {max_cycles} && !done; i = i + 1) @(posedge clk);
        repeat (2) @(posedge clk);
        if (!done)      $display("TIMEOUT");
        else if (pass)  $display("ALLPASS");
        else            $display("FAILED");
        $finish;
    end
endmodule
"""
