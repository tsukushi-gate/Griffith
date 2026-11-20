// gate_eval.v -- 2 入力論理ゲート評価器(op デコード)。
//
// 出典: 社内先行実験 hdl/gate_eval.v(先行実験が無変更で使用)を移植。
//
// op[3:0] は真理値表で、列 (A,B)=(0,0) が MSB。入力 (A,B) に対し idx = 2A+B、
// out = (op >> (3-idx)) & 1。ライブラリの
// griffith_logic.ir.weights.gate_id_from_tt_n2 が返す 16 ゲート ID と
// 同一規約である(op[3]=(0,0) / op[2]=(0,1) / op[1]=(1,0) / op[0]=(1,1)。
// IR の真理値表は bit k のアドレスが k = a + 2b、a = indices[0] = A、
// b = indices[1] = B)。
`timescale 1ns/1ps
module gate_eval (
    input  wire [3:0] op,
    input  wire       A,
    input  wire       B,
    output wire       out
);
    assign out = op[3 - {A, B}];
endmodule
