// tdp_ram.v -- True Dual-Port RAM(Intel 推奨 HDL テンプレート、M10K 単一ブロック推論用)。
//
// 出典: 社内先行実験 hdl/tdp_ram.v(先行実験が無変更で使用)を移植。
// **宣言形は変えない**(両ポート RW 宣言のこの形で Quartus が M10K TDP に推論する)。
//
// 活性バッファ複製 1 個 = 本モジュール 1 個。ポートA = operand a 読み or 書込
// (cur 時)、ポートB = operand b 読み。複製内でポートを層ごとに時分割する
// (prev 層 = 2 read / cur 層 = 1 write)ので 3 ポートは要らない。
`timescale 1ns/1ps
module tdp_ram #(
    parameter integer DW    = 8,
    parameter integer DEPTH = 1024,
    parameter integer AW    = 10
) (
    input  wire          clk,
    input  wire          we_a,
    input  wire [AW-1:0] addr_a,
    input  wire [DW-1:0] data_a,
    output reg  [DW-1:0] q_a,
    input  wire          we_b,
    input  wire [AW-1:0] addr_b,
    input  wire [DW-1:0] data_b,
    output reg  [DW-1:0] q_b
);
    (* ramstyle = "M10K" *) reg [DW-1:0] ram [0:DEPTH-1];
    always @(posedge clk) begin           // port A
        if (we_a) begin ram[addr_a] <= data_a; q_a <= data_a; end
        else q_a <= ram[addr_a];
    end
    always @(posedge clk) begin           // port B
        if (we_b) begin ram[addr_b] <= data_b; q_b <= data_b; end
        else q_b <= ram[addr_b];
    end
endmodule
