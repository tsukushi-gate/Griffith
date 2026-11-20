// overlay_harness.v -- オーバーレイのオンチップ自己検証ハーネス。
//
// 出典: 社内先行実験 hdl/harness.v を移植。差分は
//   [差分1] サイクルカウンタを 24bit から 32bit へ(実機の合同検算が不要になる)
//   [差分2] 画像 ROM を $readmemb から $readmemh へ(mdl-fpga の他の生成物と
//           .mem の書式を揃える)
//   [差分3] 7 セグ表示用に proc_pulse / match_pulse を出力(BCD 十進カウンタは
//           生成側 top が持つ)
//   [差分4] 予測クラス幅 PRED_W とカウンタ幅 CNT_W をパラメータ化
//
// オンチップ ROM のテストベクタ N 枚を 1 枚ずつ overlay へ渡し、start/done
// ハンドシェイクで推論させ、pred と期待値(ライブラリ ir.simulate の予測)を
// 比較して match / proc / cycle を数える。**期待値はラベルではなく golden の
// 予測**なので、match == N は「実シリコンが参照シムと bit 一致した」ことを
// 意味する(精度そのものは golden 側の値)。
`timescale 1ns/1ps
module overlay_harness #(
    parameter integer N           = 1000,   // テストベクタ枚数
    parameter integer IN          = 784,    // 入力ビット幅
    parameter integer IDX_W       = 10,     // 画像アドレス幅(>= clog2(N))
    parameter integer CW          = 32,     // サイクルカウンタ幅
    parameter integer CNT_W       = 10,     // proc/match カウンタ幅
    // ---- overlay 構成(素通し)----
    parameter integer V           = 8,
    parameter integer W           = 8000,
    parameter integer DEPTH       = 6,
    parameter integer CLASSES     = 10,
    parameter integer GS          = 800,
    parameter integer L0_INT_STEP = 8,
    parameter integer L0_REM_STEP = 0,
    parameter integer PRED_W      = (CLASSES < 2) ? 1 : $clog2(CLASSES),
    parameter         MEM_IMG     = "images.mem",
    parameter         MEM_EXP     = "expected.mem"
) (
    input  wire             clk,
    input  wire             start,       // clk 同期済みレベル
    input  wire             rst,         // clk 同期済みレベル(1 で IDLE)
    output reg              done,
    output reg [CNT_W-1:0]  proc_count,
    output reg [CNT_W-1:0]  match_count,
    output reg [CW-1:0]     cycle_count,
    output reg              proc_pulse,  // 1 枚処理するたび 1clk
    output reg              match_pulse  // うち一致したとき 1clk
);
    // ---- オンチップ ROM ----
    (* ramstyle = "M10K" *) reg [IN-1:0]     img_rom [0:N-1];
    (* ramstyle = "M10K" *) reg [PRED_W-1:0] exp_rom [0:N-1];
    initial begin
        $readmemh(MEM_IMG, img_rom);
        $readmemh(MEM_EXP, exp_rom);
    end

    // ---- overlay 接続 ----
    reg  [IN-1:0]     cur_img;
    reg  [PRED_W-1:0] cur_exp;
    reg               ov_start;
    wire              ov_done;
    wire [PRED_W-1:0] ov_pred;

    lgn_overlay #(
        .V(V), .W(W), .DEPTH(DEPTH), .IN_BITS(IN), .CLASSES(CLASSES), .GS(GS),
        .L0_INT_STEP(L0_INT_STEP), .L0_REM_STEP(L0_REM_STEP), .PRED_W(PRED_W)
    ) u_ov (
        .clk(clk), .rst(rst), .start(ov_start), .img(cur_img),
        .done(ov_done), .pred(ov_pred)
    );

    localparam [2:0] S_IDLE=3'd0, S_LD=3'd1, S_GO=3'd2, S_WDONE=3'd3,
                     S_WIDLE=3'd4, S_FIN=3'd5;
    reg [2:0]       state;
    reg [IDX_W-1:0] k;
    reg             running;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 0; ov_start <= 0; running <= 0;
            proc_count <= 0; match_count <= 0; cycle_count <= 0; k <= 0;
            proc_pulse <= 0; match_pulse <= 0;
        end else begin
            proc_pulse  <= 0;
            match_pulse <= 0;
            if (running) cycle_count <= cycle_count + 1'b1;
            case (state)
                S_IDLE: begin
                    done <= 0; ov_start <= 0;
                    if (start) begin
                        k <= 0; proc_count <= 0; match_count <= 0;
                        cycle_count <= 0; running <= 1; state <= S_LD;
                    end
                end
                S_LD: begin                       // ROM 同期読み → 次段で安定
                    cur_img <= img_rom[k];
                    cur_exp <= exp_rom[k];
                    state   <= S_GO;
                end
                S_GO: begin                       // overlay 起動
                    ov_start <= 1;
                    state    <= S_WDONE;
                end
                S_WDONE: begin                    // 推論完了待ち → 比較・計数
                    if (ov_done) begin
                        proc_count <= proc_count + 1'b1;
                        proc_pulse <= 1'b1;
                        if (ov_pred == cur_exp) begin
                            match_count <= match_count + 1'b1;
                            match_pulse <= 1'b1;
                        end
                        ov_start <= 0;
                        state    <= S_WIDLE;
                    end
                end
                S_WIDLE: begin                    // overlay の IDLE 復帰待ち
                    if (!ov_done) begin
                        if (k == N-1) begin running <= 0; state <= S_FIN; end
                        else begin k <= k + 1'b1; state <= S_LD; end
                    end
                end
                S_FIN: begin
                    done <= 1;
                    if (!start) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
