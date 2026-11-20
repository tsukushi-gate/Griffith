// mdl-fpga 自動生成トップ(top)
// core: mdl_core(in=3bit, out=2bit)
// UART: 115200 baud(CLKS_PER_BIT=434)/ 要求 4B / 応答 4B
// リセット: SW[9](2FF 同期)。LEDR[0]=RX活動 [1]=フレーム受理 [2]=checksumエラー(sticky)
//           [3]=TX中 [9:4]=y 下位ビット
// TODO(M1): CE/step デバッグ機構(クリック=1clk)はビューアと同時に導入する

module top (
    input  wire       CLOCK_50,
    input  wire [9:0] SW,
    output wire [9:0] LEDR,
    input  wire       UART_RXD,
    output wire       UART_TXD
);
    localparam integer IN_W  = 3;
    localparam integer OUT_W = 2;
    localparam integer REQ_P = 1;
    localparam integer RSP_Q = 1;

    wire clk = CLOCK_50;

    // ---- リセット(SW[9] を 2FF 同期) ----
    reg rst_d1 = 1'b0, rst_d2 = 1'b0;
    always @(posedge clk) begin
        rst_d1 <= SW[9];
        rst_d2 <= rst_d1;
    end
    wire rst = rst_d2;

    // ---- 要求フレーム受信 ----
    wire [8*REQ_P-1:0] rx_x;
    wire [7:0]         rx_seq;
    wire               frame_valid, frame_err;
    frame_rx #(
        .PAYLOAD_BYTES(REQ_P),
        .CLKS_PER_BIT(434),
        .TIMEOUT_CYCLES(100000)
    ) u_frx (
        .clk(clk), .rst(rst), .rx(UART_RXD),
        .x(rx_x), .seq(rx_seq), .frame_valid(frame_valid), .err_pulse(frame_err)
    );

    // ---- コア(組み合わせ)+ 入出力 FF ----
    reg  [IN_W-1:0]  xin = {IN_W{1'b0}};
    reg  [7:0]       seq_hold = 8'd0;
    wire [OUT_W-1:0] y_comb;
    mdl_core u_core (.x(xin), .y(y_comb));
    reg  [OUT_W-1:0] y_reg = {OUT_W{1'b0}};

    wire [8*RSP_Q-1:0] resp_payload = {{6{1'b0}}, y_reg};

    // ---- 応答フレーム送信 ----
    reg  tx_go = 1'b0;
    wire tx_busy;
    frame_tx #(
        .PAYLOAD_BYTES(RSP_Q),
        .CLKS_PER_BIT(434)
    ) u_ftx (
        .clk(clk), .rst(rst),
        .payload(resp_payload), .seq(seq_hold), .start(tx_go),
        .tx(UART_TXD), .busy(tx_busy)
    );

    // ---- メイン FSM(最新フレームのみ処理、demo1 実証構造) ----
    localparam [1:0] S_IDLE = 2'd0, S_CORE = 2'd1, S_TXW = 2'd2;
    reg [1:0]          state = S_IDLE;
    reg                pending = 1'b0;
    reg [8*REQ_P-1:0]  pend_x = {8*REQ_P{1'b0}};
    reg [7:0]          pend_seq = 8'd0;
    reg [1:0]          corecnt = 2'd0;
    reg                saw_busy = 1'b0;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; pending <= 1'b0; tx_go <= 1'b0;
            corecnt <= 2'd0; saw_busy <= 1'b0;
        end else begin
            tx_go <= 1'b0;
            if (frame_valid) begin
                pend_x   <= rx_x;
                pend_seq <= rx_seq;
                pending  <= 1'b1;
            end
            case (state)
                S_IDLE: if (pending) begin
                    xin      <= pend_x[IN_W-1:0];
                    seq_hold <= pend_seq;
                    pending  <= 1'b0;
                    corecnt  <= 2'd0;
                    state    <= S_CORE;
                end
                S_CORE: begin
                    corecnt <= corecnt + 2'd1;
                    if (corecnt == 2'd1) begin   // xin 確定から 2 サイクル
                        y_reg    <= y_comb;
                        tx_go    <= 1'b1;
                        saw_busy <= 1'b0;
                        state    <= S_TXW;
                    end
                end
                S_TXW: begin
                    if (tx_busy) saw_busy <= 1'b1;
                    if (saw_busy && !tx_busy) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // ---- LED(活動表示。demo1 のストレッチ手法を簡略化) ----
    localparam integer LW = 22;
    reg [LW-1:0] rx_stretch = {LW{1'b0}};
    reg [LW-1:0] ok_stretch = {LW{1'b0}};
    reg          err_sticky = 1'b0;
    always @(posedge clk) begin
        if (rst) begin
            rx_stretch <= {LW{1'b0}};
            ok_stretch <= {LW{1'b0}};
            err_sticky <= 1'b0;
        end else begin
            if (!UART_RXD)            rx_stretch <= {LW{1'b1}};
            else if (rx_stretch != 0) rx_stretch <= rx_stretch - 1'b1;
            if (frame_valid) begin
                ok_stretch <= {LW{1'b1}};
                err_sticky <= 1'b0;
            end else if (ok_stretch != 0) begin
                ok_stretch <= ok_stretch - 1'b1;
            end
            if (frame_err) err_sticky <= 1'b1;
        end
    end

    // LEDR[0]=RX活動 [1]=フレーム受理 [2]=checksumエラー(sticky) [3]=TX中 [9:4]=y 下位ビット
    wire [5:0] y_leds = {{4{1'b0}}, y_reg};
    assign LEDR = {y_leds, tx_busy, err_sticky, (ok_stretch != 0), (rx_stretch != 0)};

endmodule
