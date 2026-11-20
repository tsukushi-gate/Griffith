`timescale 1ns/1ps
// mdl-fpga 自動生成フルチェーン TB(UART ビット駆動、CLKS_PER_BIT=16 に短縮)
module tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;   // 50MHz

    reg  [9:0] sw = 10'd0;
    wire [9:0] ledr;
    reg  rxd = 1'b1;
    wire txd;

    top_sim dut (.CLOCK_50(clk), .SW(sw), .LEDR(ledr),
                 .UART_RXD(rxd), .UART_TXD(txd));

    localparam integer REQ_P = 1;
    localparam integer RSP_Q = 1;
    localparam integer BIT_NS = 160;

    reg [8*REQ_P-1:0] in_vecs  [0:3];
    reg [8*RSP_Q-1:0] exp_vecs [0:3];

    task send_byte(input [7:0] b);
        integer j;
        begin
            rxd = 1'b0; #(BIT_NS);              // start
            for (j = 0; j < 8; j = j + 1) begin
                rxd = b[j]; #(BIT_NS);          // LSB first
            end
            rxd = 1'b1; #(BIT_NS);              // stop
        end
    endtask

    task recv_byte(output [7:0] b);
        integer j;
        begin
            @(negedge txd);                     // start bit
            #(BIT_NS + BIT_NS / 2);             // bit0 中央へ
            for (j = 0; j < 8; j = j + 1) begin
                b[j] = txd;
                #(BIT_NS);
            end
        end
    endtask

    integer i, k, errors;
    reg [7:0] chk, rb;
    reg [7:0] resp [0:RSP_Q+2];
    reg [8*RSP_Q-1:0] got;

    initial begin
        $readmemh("inputs_padded.hex", in_vecs);
        $readmemh("expected_padded.hex", exp_vecs);
        errors = 0;

        // リセット(SW[9])
        sw[9] = 1'b1; repeat (10) @(posedge clk);
        sw[9] = 1'b0; repeat (10) @(posedge clk);

        for (i = 0; i < 4; i = i + 1) begin
            // ---- 要求フレーム送信 ----
            send_byte(8'hA5);
            chk = 8'd0;
            send_byte(i[7:0]); chk = chk + i[7:0];
            for (k = 0; k < REQ_P; k = k + 1) begin
                send_byte(in_vecs[i][k*8 +: 8]);
                chk = chk + in_vecs[i][k*8 +: 8];
            end
            send_byte(chk);

            // ---- 応答フレーム受信 ----
            for (k = 0; k < RSP_Q + 3; k = k + 1) begin
                recv_byte(rb);
                resp[k] = rb;
            end
            // 検証: SOF / seq / checksum / payload
            if (resp[0] !== 8'h5A) begin
                errors = errors + 1; $display("BAD_SOF %0d: %h", i, resp[0]);
            end
            if (resp[1] !== i[7:0]) begin
                errors = errors + 1; $display("BAD_SEQ %0d: %h", i, resp[1]);
            end
            chk = 8'd0;
            for (k = 1; k <= RSP_Q + 1; k = k + 1) chk = chk + resp[k];
            if (resp[RSP_Q+2] !== chk) begin
                errors = errors + 1; $display("BAD_CHK %0d", i);
            end
            for (k = 0; k < RSP_Q; k = k + 1) got[k*8 +: 8] = resp[k+2];
            if (got !== exp_vecs[i]) begin
                errors = errors + 1;
                $display("MISMATCH %0d: got=%h exp=%h", i, got, exp_vecs[i]);
            end
        end

        if (errors == 0) $display("ALLPASS");
        else $display("FAILED %0d", errors);
        $finish;
    end
endmodule
