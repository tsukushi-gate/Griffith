// frame_tx.v -- 汎用応答フレーム送信器(mdl-fpga)
// 原型: 社内先行実験 demo/hdl/frame_tx.v(24B 固定・pred バイト付き)を
// PAYLOAD_BYTES で汎用化。pred は持たず、y ビット列をそのまま返す
// (argmax 等の解釈は PC 側レンダラの責務)。
//
// ===== mdl-fpga フレーム仕様(応答 FPGA->PC、PAYLOAD_BYTES+3 バイト固定長) =====
//   [0]              = SOF 0x5A
//   [1]              = seq エコー
//   [2..Q+1]         = ペイロード Q バイト(バイト k の bit j LSB-first = y[8k+j]、上位パディング0)
//   [Q+2]            = checksum = (byte1..byteQ+1 の総和) mod 256
// ==============================================================================

module frame_tx #(
    parameter integer PAYLOAD_BYTES = 22,
    parameter integer CLKS_PER_BIT  = 434
) (
    input  wire                       clk,
    input  wire                       rst,
    input  wire [8*PAYLOAD_BYTES-1:0] payload,
    input  wire [7:0]                 seq,
    input  wire                       start,   // 1 サイクルパルス(busy 中は無視)
    output wire                       tx,
    output reg                        busy
);
    reg  [7:0] tx_data;
    reg        tx_start;
    wire       tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_tx (
        .clk(clk), .rst(rst), .data(tx_data), .start(tx_start),
        .tx(tx), .busy(tx_busy)
    );

    localparam [1:0] S_IDLE = 2'd0, S_LOAD = 2'd1, S_WBUSY = 2'd2, S_WDONE = 2'd3;
    reg [1:0]  state = S_IDLE;
    reg [8*PAYLOAD_BYTES-1:0] hold;
    reg [7:0]  seq_hold;
    reg [7:0]  chk;
    reg [15:0] idx;   // 0=SOF, 1=seq, 2..Q+1=payload, Q+2=checksum

    // 現在バイトの選択(組み合わせ)
    reg [7:0] cur;
    always @* begin
        if (idx == 16'd0)               cur = 8'h5A;
        else if (idx == 16'd1)          cur = seq_hold;
        else if (idx <= PAYLOAD_BYTES + 1) cur = hold[(idx - 2) * 8 +: 8];
        else                            cur = chk;
    end

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            busy     <= 1'b0;
            tx_start <= 1'b0;
            idx      <= 16'd0;
            chk      <= 8'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    busy     <= 1'b0;
                    tx_start <= 1'b0;
                    if (start) begin
                        hold     <= payload;
                        seq_hold <= seq;
                        chk      <= 8'd0;
                        idx      <= 16'd0;
                        busy     <= 1'b1;
                        state    <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    tx_data  <= cur;
                    tx_start <= 1'b1;
                    state    <= S_WBUSY;
                end
                S_WBUSY: begin
                    tx_start <= 1'b0;
                    if (tx_busy) state <= S_WDONE;
                end
                S_WDONE: begin
                    if (!tx_busy) begin
                        // checksum は byte1..byteQ+1(seq+payload)を送出後に累積
                        if (idx >= 16'd1 && idx <= PAYLOAD_BYTES + 1)
                            chk <= chk + cur;
                        if (idx == PAYLOAD_BYTES + 2) begin
                            state <= S_IDLE;   // 全 Q+3 バイト送出完了
                        end else begin
                            idx   <= idx + 16'd1;
                            state <= S_LOAD;
                        end
                    end
                end
            endcase
        end
    end
endmodule
