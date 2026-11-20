// mdl-fpga 自動生成 オーバーレイ検証システム(top)
// 画像 ROM 8 枚 → lgn_overlay(V=8 W=32 DEPTH=3
//   IN=24 CLASSES=2 GS=16) → 期待値照合
// ゲート記述 = op 4bit のみ(opmem0..0.mem)、
//   配線 = 計算型(addrinit_a/b.mem + reminit.mem のアキュムレータ初期値)
// 予測 cyc/画像 = 34(= ceil(IN/V) + DEPTH*(W/V) + 4*DEPTH + 1 + 6)
// clk_core = CLOCK_50 / 2
// probe[0]=done [4:1]=proc [8:5]=match
//   [40:9]=cycle(32b) / source[0]=start [1]=rst
// HEX3-0=一致数(10進4桁) / HEX5-4=処理数の百位
`timescale 1ns/1ps
module top (
    input  wire       CLOCK_50,
    output wire [9:0] LEDR,
    output wire [6:0] HEX0,
    output wire [6:0] HEX1,
    output wire [6:0] HEX2,
    output wire [6:0] HEX3,
    output wire [6:0] HEX4,
    output wire [6:0] HEX5
);
    reg clk_core = 1'b0;
    always @(posedge CLOCK_50) clk_core <= ~clk_core;

    // ---- ISSP(In-System Sources and Probes)----
    wire [1:0]  src;
    wire        done;
    wire [3:0] proc_count;
    wire [3:0] match_count;
    wire [31:0] cycle_count;
    wire        proc_pulse, match_pulse;
    wire [40:0] prb = {cycle_count, match_count, proc_count, done};

    altsource_probe #(
        .sld_auto_instance_index ("YES"),
        .sld_instance_index      (0),
        .instance_id             ("ONB"),
        .probe_width             (41),
        .source_width            (2),
        .source_initial_value    ("0"),
        .enable_metastability    ("NO")
    ) issp (.probe(prb), .source(src));

    // ---- source を clk_core 域へ 2FF 同期 ----
    reg [1:0] start_sync = 2'b0, rst_sync = 2'b0;
    always @(posedge clk_core) begin
        start_sync <= {start_sync[0], src[0]};
        rst_sync   <= {rst_sync[0],   src[1]};
    end

    overlay_harness #(
        .N(8), .IN(24), .IDX_W(3), .CW(32), .CNT_W(4), .V(8), .W(32), .DEPTH(3), .CLASSES(2), .GS(16), .L0_INT_STEP(8), .L0_REM_STEP(0), .PRED_W(1),
        .MEM_IMG("images.mem"), .MEM_EXP("expected.mem")
    ) u_sys (
        .clk(clk_core), .start(start_sync[1]), .rst(rst_sync[1]),
        .done(done), .proc_count(proc_count), .match_count(match_count),
        .cycle_count(cycle_count),
        .proc_pulse(proc_pulse), .match_pulse(match_pulse)
    );

    // ---- 十進カウンタ(7セグ表示用。パック BCD)----
    reg [15:0] match_bcd = 16'd0;
    reg [7:0]  proc_bcd  = 8'd0;
    reg [6:0]  proc100   = 7'd0;
    always @(posedge clk_core) begin
        if (rst_sync[1]) begin
            match_bcd <= 16'd0; proc_bcd <= 8'd0; proc100 <= 7'd0;
        end else begin
            if (match_pulse) begin
                if (match_bcd[3:0] != 4'd9)
                    match_bcd[3:0] <= match_bcd[3:0] + 4'd1;
                else begin
                    match_bcd[3:0] <= 4'd0;
                    if (match_bcd[7:4] != 4'd9)
                        match_bcd[7:4] <= match_bcd[7:4] + 4'd1;
                    else begin
                        match_bcd[7:4] <= 4'd0;
                        if (match_bcd[11:8] != 4'd9)
                            match_bcd[11:8] <= match_bcd[11:8] + 4'd1;
                        else begin
                            match_bcd[11:8] <= 4'd0;
                            if (match_bcd[15:12] != 4'd9)
                                match_bcd[15:12] <= match_bcd[15:12] + 4'd1;
                            else match_bcd[15:12] <= 4'd0;
                        end
                    end
                end
            end
            if (proc_pulse) begin
                if (proc100 == 7'd99) begin
                    proc100 <= 7'd0;
                    if (proc_bcd[3:0] != 4'd9)
                        proc_bcd[3:0] <= proc_bcd[3:0] + 4'd1;
                    else begin
                        proc_bcd[3:0] <= 4'd0;
                        if (proc_bcd[7:4] != 4'd9)
                            proc_bcd[7:4] <= proc_bcd[7:4] + 4'd1;
                        else proc_bcd[7:4] <= 4'd0;
                    end
                end else begin
                    proc100 <= proc100 + 7'd1;
                end
            end
        end
    end

    seg7 u_h0 (.d(match_bcd[ 3: 0]), .seg(HEX0));
    seg7 u_h1 (.d(match_bcd[ 7: 4]), .seg(HEX1));
    seg7 u_h2 (.d(match_bcd[11: 8]), .seg(HEX2));
    seg7 u_h3 (.d(match_bcd[15:12]), .seg(HEX3));
    seg7 u_h4 (.d(proc_bcd [ 3: 0]), .seg(HEX4));
    seg7 u_h5 (.d(proc_bcd [ 7: 4]), .seg(HEX5));

    assign LEDR[0]   = done;
    assign LEDR[9:1] = {{5{1'b0}}, match_count};
endmodule
