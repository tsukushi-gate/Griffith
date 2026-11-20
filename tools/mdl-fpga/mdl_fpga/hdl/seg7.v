// seg7.v -- 4bit → 7セグメントデコーダ(負論理、DE0-CV系)
// seg[0..6] = セグメント a..g。0=点灯。10進0-9のみ点灯、それ以外は消灯。
module seg7 (
    input  wire [3:0] d,
    output reg  [6:0] seg
);
    always @* begin
        case (d)                 //  gfedcba(0=点灯)
            4'd0: seg = 7'b1000000;
            4'd1: seg = 7'b1111001;
            4'd2: seg = 7'b0100100;
            4'd3: seg = 7'b0110000;
            4'd4: seg = 7'b0011001;
            4'd5: seg = 7'b0010010;
            4'd6: seg = 7'b0000010;
            4'd7: seg = 7'b1111000;
            4'd8: seg = 7'b0000000;
            4'd9: seg = 7'b0010000;
            default: seg = 7'b1111111;   // 消灯
        endcase
    end
endmodule
