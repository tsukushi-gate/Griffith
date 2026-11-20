`timescale 1ns/1ps
// mdl-fpga 自動生成 ビューア用薄ラッパ(システムモード)
module top_view (
    input wire clk,
    input wire rst
);
    wire [9:0] ledr;
    wire [6:0] h0, h1, h2, h3, h4, h5;
    top u_sys (
        .CLOCK_50(clk), .SW(10'b0), .KEY({3'b111, ~rst}), .LEDR(ledr),
        .HEX0(h0), .HEX1(h1), .HEX2(h2), .HEX3(h3), .HEX4(h4), .HEX5(h5));
endmodule
