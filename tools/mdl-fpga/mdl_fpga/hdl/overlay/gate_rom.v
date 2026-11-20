// gate_rom.v -- 保存型オーバーレイのゲート記録 ROM(記録幅で実体を選ぶ薄い殻)。
//
// 記録幅 DW は 32bit(a/b 各 14bit = 社内先行実験と同形式)か
// 40bit(a/b 各 18bit。層幅が 16,384 を超える場合)の 2 通りしかない。
// **rom32 / rom40 の宣言形は変えずに**、どちらを実体化するかだけを
// generate で選ぶ(レーンごとの unroll を 2 倍に書かないための殻であり、
// Quartus から見える ROM の記述は rom32.v / rom40.v のまま)。
`timescale 1ns/1ps
module gate_rom #(
    parameter          FILE  = "",
    parameter integer  DEPTH = 6000,
    parameter integer  AW    = 13,
    parameter integer  DW    = 32
) (
    input  wire          clk,
    input  wire          en,
    input  wire [AW-1:0] addr,
    output wire [DW-1:0] q
);
    generate
        if (DW <= 32) begin: g32
            wire [31:0] q32;
            rom32 #(.FILE(FILE), .DEPTH(DEPTH), .AW(AW)) r (
                .clk(clk), .en(en), .addr(addr), .q(q32));
            assign q = q32[DW-1:0];
        end else begin: g40
            wire [39:0] q40;
            rom40 #(.FILE(FILE), .DEPTH(DEPTH), .AW(AW)) r (
                .clk(clk), .en(en), .addr(addr), .q(q40));
            assign q = q40[DW-1:0];
        end
    endgenerate
endmodule
