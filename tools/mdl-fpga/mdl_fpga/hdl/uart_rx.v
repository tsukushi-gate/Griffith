// uart_rx.v -- UART 8N1 受信器
// 出典: 社内先行実験 demo/hdl/uart_rx.v(DE0-CV 実機で MNIST 1000/1000 bit一致を
// 実証済みの物理層)を mdl-fpga へ取り込み。ロジックは同一、ヘッダのみ更新。
//
// 8N1 受信。rx を 2FF 同期し、スタートビット(0)検出後、スタートビット中央で
// 再確認 -> 以降各データビット中央 (LSB first) をサンプル。
// 出力 data[7:0] と valid (受信完了サイクルに 1 パルス)。rst は同期リセット。

module uart_rx #(
    parameter integer CLKS_PER_BIT = 434
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,        // シリアル入力 (非同期)
    output reg  [7:0] data,
    output reg        valid      // 1 サイクルパルス
);
    // ---- 入力 2FF 同期 (アイドルは High) ----
    reg rx_d1  = 1'b1;
    reg rx_sync = 1'b1;
    always @(posedge clk) begin
        rx_d1   <= rx;
        rx_sync <= rx_d1;
    end

    localparam [2:0] S_IDLE  = 3'd0,
                     S_START = 3'd1,
                     S_DATA  = 3'd2,
                     S_STOP  = 3'd3;

    reg [2:0]  state    = S_IDLE;
    reg [15:0] clk_cnt  = 16'd0;   // ビット内サイクルカウンタ
    reg [2:0]  bit_idx  = 3'd0;
    reg [7:0]  shreg    = 8'd0;

    // スタートビット中央 = (CLKS_PER_BIT-1)/2
    wire [15:0] HALF = (CLKS_PER_BIT - 1) >> 1;

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            clk_cnt <= 16'd0;
            bit_idx <= 3'd0;
            valid   <= 1'b0;
            data    <= 8'd0;
        end else begin
            valid <= 1'b0;  // 既定 (1 サイクルパルス化)
            case (state)
                S_IDLE: begin
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (rx_sync == 1'b0) begin   // スタートビット候補
                        state <= S_START;
                    end
                end
                S_START: begin
                    if (clk_cnt == HALF) begin
                        clk_cnt <= 16'd0;
                        if (rx_sync == 1'b0) begin  // 中央でも Low -> 真のスタート
                            state <= S_DATA;
                        end else begin
                            state <= S_IDLE;        // グリッチ
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end
                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt          <= 16'd0;
                        shreg[bit_idx]   <= rx_sync;  // ビット中央でサンプル (LSB first)
                        if (bit_idx == 3'd7) begin
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end
                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
                        data    <= shreg;
                        valid   <= 1'b1;   // 受信完了パルス
                        state   <= S_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
