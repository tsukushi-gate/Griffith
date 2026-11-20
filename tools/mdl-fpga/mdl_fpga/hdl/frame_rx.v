// frame_rx.v -- 汎用要求フレーム受信器(mdl-fpga)
// 原型: 社内先行実験 demo/hdl/frame_rx.v(98B 固定)を PAYLOAD_BYTES で
// 汎用化(任意の入力幅のモデルに対応)。FSM 構造・再同期規約は実証済み原型と同一。
//
// ===== mdl-fpga フレーム仕様(要求 PC->FPGA、PAYLOAD_BYTES+3 バイト固定長) =====
//   [0]              = SOF 0xA5
//   [1]              = seq(8bit、送信ごとに +1 mod 256)
//   [2..P+1]         = ペイロード P バイト(バイト k の bit j LSB-first = x[8k+j])
//   [P+2]            = checksum = (byte1..byteP+1 の総和) mod 256
// 再同期: SOF 受信後、バイト間無受信が TIMEOUT_CYCLES を超えたら HUNT へ。
// checksum 不一致はフレーム破棄 + err_pulse(応答なし)。
// ==============================================================================

module frame_rx #(
    parameter integer PAYLOAD_BYTES  = 98,
    parameter integer CLKS_PER_BIT   = 434,
    parameter integer TIMEOUT_CYCLES = 100000   // 50MHz で 2ms
) (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         rx,
    output reg [8*PAYLOAD_BYTES-1:0]    x,
    output reg [7:0]                    seq,
    output reg                          frame_valid,  // 1 サイクルパルス
    output reg                          err_pulse     // checksum 不一致
);
    wire [7:0] rx_data;
    wire       rx_valid;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_rx (
        .clk(clk), .rst(rst), .rx(rx), .data(rx_data), .valid(rx_valid)
    );

    localparam S_HUNT = 1'b0, S_RECV = 1'b1;
    reg        state = S_HUNT;
    reg [15:0] idx   = 16'd0;   // 1=seq, 2..P+1=payload, P+2=checksum
    reg [7:0]  chk   = 8'd0;
    reg [31:0] tocnt = 32'd0;

    always @(posedge clk) begin
        if (rst) begin
            state       <= S_HUNT;
            idx         <= 16'd0;
            chk         <= 8'd0;
            tocnt       <= 32'd0;
            frame_valid <= 1'b0;
            err_pulse   <= 1'b0;
        end else begin
            frame_valid <= 1'b0;
            err_pulse   <= 1'b0;
            case (state)
                S_HUNT: begin
                    tocnt <= 32'd0;
                    if (rx_valid && rx_data == 8'hA5) begin
                        idx   <= 16'd1;
                        chk   <= 8'd0;
                        state <= S_RECV;
                    end
                end
                S_RECV: begin
                    if (rx_valid) begin
                        tocnt <= 32'd0;
                        if (idx == 16'd1) begin
                            seq <= rx_data;
                            chk <= chk + rx_data;
                            idx <= idx + 16'd1;
                        end else if (idx <= PAYLOAD_BYTES + 1) begin
                            x[(idx - 2) * 8 +: 8] <= rx_data;
                            chk <= chk + rx_data;
                            idx <= idx + 16'd1;
                        end else begin
                            // idx == PAYLOAD_BYTES+2: checksum バイト
                            if (rx_data == chk) frame_valid <= 1'b1;
                            else                err_pulse   <= 1'b1;
                            state <= S_HUNT;
                        end
                    end else if (tocnt >= TIMEOUT_CYCLES - 1) begin
                        state <= S_HUNT;   // バイト間タイムアウト → 再同期
                    end else begin
                        tocnt <= tocnt + 32'd1;
                    end
                end
            endcase
        end
    end
endmodule
