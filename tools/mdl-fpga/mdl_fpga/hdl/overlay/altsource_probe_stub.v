// altsource_probe_stub.v -- **シミュレーション専用**の ISSP ダミー。
//
// Quartus の altsource_probe(In-System Sources and Probes)は実機 IP なので
// iverilog では解決できない。生成トップ(top.v)をシムで丸ごと動かすために、
// ポートとパラメータだけ同形のスタブを置く。source はテストベンチから
// 階層参照で駆動する(dut.issp.source = 2'b01 のように)。
// **Quartus のソースリストには入れない**(qsf には含めないこと)。
`timescale 1ns/1ps
module altsource_probe #(
    parameter         sld_auto_instance_index = "YES",
    parameter integer sld_instance_index      = 0,
    parameter         instance_id             = "ONB",
    parameter integer probe_width             = 1,
    parameter integer source_width            = 1,
    parameter         source_initial_value    = "0",
    parameter         enable_metastability    = "NO"
) (
    input  wire [probe_width-1:0]  probe,
    output reg  [source_width-1:0] source
);
    initial source = {source_width{1'b0}};
    wire _unused = |probe;
endmodule
