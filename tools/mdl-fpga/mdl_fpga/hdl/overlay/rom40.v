// rom40.v -- 40bit 同期読み ROM(保存型オーバーレイの広い記録用)。
//
// 出典: rom32.v(社内先行実験 hdl/rom32.v)と**同型の宣言**で幅だけを
// 40bit にしたもの。M10K の最も広いモード(256 語 x 40bit)の自然幅なので、
// 2*AW + 4 が 32bit に収まらない(層幅 > 16,384)ときの記録幅に使う。
// **宣言形(ramstyle 属性 / $readmemh / 同期読み)は rom32 と同じに保つ**:
// Quartus の M10K 推論はこの書き方に依存する。
// $sformatf は合成非対応のため、ファイル名は呼び出し側がリテラルで与える。
`timescale 1ns/1ps
module rom40 #(
    parameter          FILE  = "",
    parameter integer  DEPTH = 6000,
    parameter integer  AW    = 13
) (
    input  wire          clk,
    input  wire          en,
    input  wire [AW-1:0] addr,
    output reg  [39:0]   q
);
    (* ramstyle = "M10K" *) reg [39:0] mem [0:DEPTH-1];
    initial $readmemh(FILE, mem);
    always @(posedge clk) if (en) q <= mem[addr];
endmodule
