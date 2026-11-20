// uart_tx.v -- UART 8N1 送信器
// 出典: 社内先行実験 demo/hdl/uart_tx.v(DE0-CV 実機実証済み)を mdl-fpga へ
// 取り込み。ロジックは同一、ヘッダのみ更新。
//
// 8N1 送信。start パルスで data[7:0] をラッチし、
// スタート(0) -> データ8bit(LSB first) -> ストップ(1) を CLKS_PER_BIT サイクルずつ送出。
// busy は送信中 High。rst は同期リセット。

module uart_tx #(
    parameter integer CLKS_PER_BIT = 434
) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       start,   // 1 サイクルパルスで送信開始 (busy 中は無視)
    output reg        tx,
    output reg        busy
);
    localparam [2:0] S_IDLE  = 3'd0,
                     S_START = 3'd1,
                     S_DATA  = 3'd2,
                     S_STOP  = 3'd3;

    reg [2:0]  state   = S_IDLE;
    reg [15:0] clk_cnt = 16'd0;
    reg [2:0]  bit_idx = 3'd0;
    reg [7:0]  shreg   = 8'd0;

    initial begin
        tx   = 1'b1;   // アイドルは High
        busy = 1'b0;
    end

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            clk_cnt <= 16'd0;
            bit_idx <= 3'd0;
            tx      <= 1'b1;
            busy    <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx   <= 1'b1;
                    busy <= 1'b0;
                    if (start) begin
                        shreg   <= data;
                        clk_cnt <= 16'd0;
                        bit_idx <= 3'd0;
                        busy    <= 1'b1;
                        state   <= S_START;
                    end
                end
                S_START: begin
                    tx <= 1'b0;   // スタートビット
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
                        state   <= S_DATA;
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end
                S_DATA: begin
                    tx <= shreg[bit_idx];   // LSB first
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
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
                    tx <= 1'b1;   // ストップビット
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
                        busy    <= 1'b0;
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
