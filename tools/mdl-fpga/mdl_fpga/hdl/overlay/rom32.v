// rom32.v -- 32bit 同期読み ROM(ゲート op ROM)。
//
// 出典: 社内先行実験 hdl/rom32.v(先行実験が無変更で使用)を移植。
// **宣言形(ramstyle 属性 / $readmemh / 同期読み)は変えない**: Quartus の
// M10K 推論結果(先行実験実測のゲート ROM 32 blk)はこの書き方に依存する。
// $sformatf は合成非対応のため、ファイル名は呼び出し側がリテラルで与える。
`timescale 1ns/1ps
module rom32 #(
    parameter          FILE  = "",
    parameter integer  DEPTH = 6000,
    parameter integer  AW    = 13
) (
    input  wire          clk,
    input  wire          en,
    input  wire [AW-1:0] addr,
    output reg  [31:0]   q
);
    (* ramstyle = "M10K" *) reg [31:0] mem [0:DEPTH-1];
    initial $readmemh(FILE, mem);
    always @(posedge clk) if (en) q <= mem[addr];
endmodule
