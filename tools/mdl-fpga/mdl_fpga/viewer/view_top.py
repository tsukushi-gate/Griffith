"""ビューア用トップ(top_view)の生成。

UART を介さず内部境界へ直接値を注入する「境界注入」モード(構想書 §3.4)の
最小形。xin レジスタは毎クロック x_in を取り込む(load 信号なし)。
    x_in → [xin FF] → mdl_core(組み合わせ) → [y_reg FF] → y
クロックは cocotb 側が手動で駆動する(フリーランしない=クリック=1clk)。
"""

from __future__ import annotations


def generate_view_top(in_width: int, out_width: int,
                      core_module: str = 'mdl_core',
                      pipelined: bool = False) -> str:
    core_ports = '.clk(clk), .x(xin), .y(y_comb)' if pipelined \
        else '.x(xin), .y(y_comb)'
    return f"""`timescale 1ns/1ps
// mdl-fpga 自動生成ビューア用トップ(境界注入モード{'、パイプラインコア' if pipelined else ''})
module top_view (
    input  wire clk,
    input  wire rst,
    input  wire [{in_width - 1}:0] x_in,
    output wire [{out_width - 1}:0] y
);
    reg [{in_width - 1}:0] xin = {{{in_width}{{1'b0}}}};
    always @(posedge clk) begin
        if (rst) xin <= {{{in_width}{{1'b0}}}};
        else     xin <= x_in;
    end

    wire [{out_width - 1}:0] y_comb;
    {core_module} u_core ({core_ports});

    reg [{out_width - 1}:0] y_reg = {{{out_width}{{1'b0}}}};
    always @(posedge clk) begin
        if (rst) y_reg <= {{{out_width}{{1'b0}}}};
        else     y_reg <= y_comb;
    end
    assign y = y_reg;
endmodule
"""
