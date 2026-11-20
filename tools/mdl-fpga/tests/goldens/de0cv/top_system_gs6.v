// mdl-fpga 自動生成 分類デモシステム(top)
// ROM(6枚) → mdl_core_pipe(latency=2clk) → argmax → 正解カウント → 7セグ
// SW[0]=低速モード(SLOW_DIVで分周) / KEY[0]=リスタート(負論理) / 完了後ホールド
// HEX3-0=正解数(10進4桁) / HEX5-4=処理数の百位(10進2桁)
// LEDR[0]=heartbeat [3:1]=state [4]=SW[0] [8]=running [9]=done
module top #(
    parameter integer SLOW_DIV = 5000000
) (
    input  wire       CLOCK_50,
    input  wire [9:0] SW,
    input  wire [3:0] KEY,
    output wire [9:0] LEDR,
    output wire [6:0] HEX0,
    output wire [6:0] HEX1,
    output wire [6:0] HEX2,
    output wire [6:0] HEX3,
    output wire [6:0] HEX4,
    output wire [6:0] HEX5
);
    localparam integer N     = 6;
    localparam integer LAT   = 2;
    localparam integer IN_W  = 4;

    wire clk = CLOCK_50;

    // ---- リスタート(KEY[0] 負論理、2FF 同期) ----
    reg [1:0] key0_sync = 2'b11;
    always @(posedge clk) key0_sync <= {key0_sync[0], KEY[0]};
    wire restart = ~key0_sync[1];

    // ---- ROM ----
    reg  [2:0] img_idx = 3'd0;
    wire [4:0] rom_q;
    sys_rom u_rom (.clk(clk), .addr(img_idx), .q(rom_q));

    // ---- コア(パイプライン) ----
    reg  [IN_W-1:0] xin = {IN_W{1'b0}};
    reg  [0:0] label_r = 1'd0;
    wire [3:0] scores;
    mdl_core_pipe u_core (.clk(clk), .x(xin), .y(scores));

    // argmax 比較木(同点=小さい index 勝ち = np.argmax)
    wire [1:0] s0 = scores[1:0];
    wire [1:0] s1 = scores[3:2];
    wire [1:0] n0_s;
    wire [0:0] n0_i;
    wire n0_gt = (s1 > s0);
    assign n0_s = n0_gt ? s1 : s0;
    assign n0_i = n0_gt ? 1'd1 : 1'd0;
    wire [0:0] pred = n0_i;

    // ---- FSM ----
    localparam [2:0] S_RESTART = 3'd0, S_FETCH = 3'd1, S_LOAD = 3'd2,
                     S_CORE = 3'd3, S_CHECK = 3'd4, S_THROTTLE = 3'd5,
                     S_DONE = 3'd6;
    reg [2:0]  state = S_RESTART;
    reg [15:0] lat_cnt = 16'd0;
    reg [31:0] slow_cnt = 32'd0;
    reg [15:0] correct_bcd = 16'd0;   // 10進4桁(パックBCD)
    reg [7:0]  proc_bcd = 8'd0;       // 処理数の百位(10進2桁)
    reg [6:0]  proc100_cnt = 7'd0;    // 0..99

    always @(posedge clk) begin
        if (restart) begin
            state <= S_RESTART;
        end else begin
            case (state)
                S_RESTART: begin
                    img_idx     <= 3'd0;
                    correct_bcd <= 16'd0;
                    proc_bcd    <= 8'd0;
                    proc100_cnt <= 7'd0;
                    slow_cnt    <= 32'd0;
                    state       <= S_FETCH;
                end
                S_FETCH: begin
                    // この posedge で rom_q <= mem[img_idx]
                    state <= S_LOAD;
                end
                S_LOAD: begin
                    xin     <= rom_q[IN_W-1:0];
                    label_r <= rom_q[4:IN_W];
                    lat_cnt <= 16'd0;
                    state   <= S_CORE;
                end
                S_CORE: begin
                    lat_cnt <= lat_cnt + 16'd1;
                    if (lat_cnt == LAT - 1) state <= S_CHECK;
                end
                S_CHECK: begin
                    if (pred == label_r) begin
                        if (correct_bcd[3:0] != 4'd9)
                            correct_bcd[3:0] <= correct_bcd[3:0] + 4'd1;
                        else begin
                            correct_bcd[3:0] <= 4'd0;
                            if (correct_bcd[7:4] != 4'd9)
                                correct_bcd[7:4] <= correct_bcd[7:4] + 4'd1;
                            else begin
                                correct_bcd[7:4] <= 4'd0;
                                if (correct_bcd[11:8] != 4'd9)
                                    correct_bcd[11:8] <= correct_bcd[11:8] + 4'd1;
                                else begin
                                    correct_bcd[11:8] <= 4'd0;
                                    if (correct_bcd[15:12] != 4'd9)
                                        correct_bcd[15:12] <= correct_bcd[15:12] + 4'd1;
                                    else correct_bcd[15:12] <= 4'd0;
                                end
                            end
                        end
                    end
                    if (proc100_cnt == 7'd99) begin
                        proc100_cnt <= 7'd0;
                            if (proc_bcd[3:0] != 4'd9)
                                proc_bcd[3:0] <= proc_bcd[3:0] + 4'd1;
                            else begin
                                proc_bcd[3:0] <= 4'd0;
                                if (proc_bcd[7:4] != 4'd9)
                                    proc_bcd[7:4] <= proc_bcd[7:4] + 4'd1;
                                else proc_bcd[7:4] <= 4'd0;
                            end
                    end else begin
                        proc100_cnt <= proc100_cnt + 7'd1;
                    end
                    if (img_idx == N - 1) begin
                        state <= S_DONE;
                    end else begin
                        img_idx <= img_idx + 3'd1;
                        slow_cnt <= 32'd0;
                        state <= SW[0] ? S_THROTTLE : S_FETCH;
                    end
                end
                S_THROTTLE: begin
                    slow_cnt <= slow_cnt + 32'd1;
                    if (slow_cnt >= SLOW_DIV - 1) state <= S_FETCH;
                end
                S_DONE: state <= S_DONE;   // ホールド(restart 待ち)
                default: state <= S_RESTART;
            endcase
        end
    end

    // ---- 7セグ表示 ----
    seg7 u_h0 (.d(correct_bcd[ 3: 0]), .seg(HEX0));
    seg7 u_h1 (.d(correct_bcd[ 7: 4]), .seg(HEX1));
    seg7 u_h2 (.d(correct_bcd[11: 8]), .seg(HEX2));
    seg7 u_h3 (.d(correct_bcd[15:12]), .seg(HEX3));
    seg7 u_h4 (.d(proc_bcd  [ 3: 0]), .seg(HEX4));
    seg7 u_h5 (.d(proc_bcd  [ 7: 4]), .seg(HEX5));

    // ---- LED ----
    reg [25:0] hb = 26'd0;
    always @(posedge clk) hb <= hb + 26'd1;
    assign LEDR[0]   = hb[25];              // heartbeat
    assign LEDR[3:1] = state;
    assign LEDR[4]   = SW[0];
    assign LEDR[7:5] = 3'b000;
    assign LEDR[8]   = (state != S_DONE);   // running
    assign LEDR[9]   = (state == S_DONE);   // done

endmodule
