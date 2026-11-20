"""トップレベルラッパ生成(UART フレーミング + 入出力 FF + FSM)。

構成(社内先行実験 demo1 top.v の実証構造を汎用化):
    UART_RXD → frame_rx → xin FF → mdl_core(組み合わせ) → y FF
             → frame_tx → UART_TXD
- 要求ペイロード P = ceil(in_width/8) バイト、応答 Q = ceil(out_width/8) バイト。
- 処理中に届いたフレームは最新 1 件だけ保持(ライブデモ向けドロップ許容)。
- リセットは SW[9](2FF 同期)。
- CE/step デバッグ機構は未実装(M1 ビューアと同時に導入する。TODO を明示)。
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .ports import TopPorts

HDL_DIR = Path(__file__).resolve().parent.parent / 'hdl'
STATIC_SOURCES = ('uart_rx.v', 'uart_tx.v', 'frame_rx.v', 'frame_tx.v')


@dataclass
class WrapperGen:
    top_verilog: str
    top_name: str
    in_width: int
    out_width: int
    req_payload_bytes: int
    rsp_payload_bytes: int
    clks_per_bit: int
    sources: list[str]   # top を含む全 Verilog ファイル名(合成順)
    # 使用資源セットの単一情報源(synth write_project_files の groups へ渡す)
    groups: tuple = ('clock', 'uart', 'led', 'switch')
    ports: TopPorts = field(default_factory=TopPorts)


def copy_static_hdl(dest_dir: str | Path) -> list[str]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in STATIC_SOURCES:
        shutil.copyfile(HDL_DIR / name, dest / name)
    return list(STATIC_SOURCES)


def generate_wrapper(in_width: int, out_width: int, *,
                     top_name: str = 'top',
                     core_module: str = 'mdl_core',
                     clock_hz: int = 50_000_000,
                     baud: int = 115200,
                     timeout_cycles: int | None = None,
                     clks_per_bit: int | None = None,
                     with_selftest: bool = False,
                     ports: TopPorts | None = None) -> WrapperGen:
    ports = ports if ports is not None else TopPorts()
    min_led = 6 if with_selftest else 4
    if ports.led_width < min_led:
        raise NotImplementedError(
            f'led_width={ports.led_width} は未対応(状態表示に最低 {min_led} 本'
            f'必要: selftest=6 / 通常=4。これ未満のボードは割付設計を追加してから'
            '対応する)')
    req_p = math.ceil(in_width / 8)
    rsp_q = math.ceil(out_width / 8)
    if clks_per_bit is None:
        clks_per_bit = round(clock_hz / baud)
    if timeout_cycles is None:
        timeout_cycles = max(1000, clock_hz // 500)  # 2ms 相当

    pad = 8 * rsp_q - out_width
    if pad > 0:
        resp_expr = f"{{{{{pad}{{1'b0}}}}, y_reg}}"
    else:
        resp_expr = 'y_reg'

    # リセット元(switch=SW[9] 従来 / button=スイッチ非搭載ボード)と極性
    rst_port = ports.reset_port
    rst_bit = ports.reset_bit
    rst_read = f'{rst_port}[{rst_bit}]'
    rst_expr = f'~{rst_read}' if ports.reset_active_low else rst_read
    rst_note = '(押下=Low)' if ports.reset_active_low else ''

    def _led_assign(concat: str) -> str:
        rhs = f'~({concat})' if ports.led_active_low else concat
        return f'    assign {ports.led} = {rhs};'

    if with_selftest:
        selftest_section = """
    // ---- オンチップ自己検証(selftest.v、UART 不要の実機 bit 一致確認) ----
    wire st_done, st_fail, st_pass;
    selftest u_selftest (.clk(clk), .rst(rst),
                         .done(st_done), .fail(st_fail), .pass(st_pass));

    // ---- ハートビート(生存確認、約 0.7Hz) ----
    reg [25:0] hb = 26'd0;
    always @(posedge clk) hb <= hb + 26'd1;
"""
        n_y = ports.led_width - 6
        if n_y > 0:
            if out_width >= n_y:
                y4 = f'y_reg[{n_y - 1}:0]'
            else:
                y4 = f"{{{{{n_y - out_width}{{1'b0}}}}, y_reg}}"
            st_tail = f' [{ports.led_width - 1}:6]=y 下位ビット'
            st_concat = (f'{{{y4}, err_sticky, (ok_stretch != 0), '
                         'tx_busy, hb[25], st_fail, st_pass}')
        else:
            st_tail = ''
            st_concat = ('{err_sticky, (ok_stretch != 0), '
                         'tx_busy, hb[25], st_fail, st_pass}')
        led_assign = (
            f'    // {ports.led}[0]=selftest PASS(点灯=全ベクトル bit 一致) '
            '[1]=FAIL [2]=heartbeat\n'
            f'    // [3]=TX中 [4]=フレーム受理 [5]=checksumエラー{st_tail}\n'
            + _led_assign(st_concat))
    else:
        selftest_section = ''
        n_y = ports.led_width - 4
        if n_y > 0:
            if out_width >= n_y:
                y_leds_expr = f'y_reg[{n_y - 1}:0]'
            else:
                y_leds_expr = f"{{{{{n_y - out_width}{{1'b0}}}}, y_reg}}"
            nl_tail = f' [{ports.led_width - 1}:4]=y 下位ビット'
            wire_line = f'    wire [{n_y - 1}:0] y_leds = {y_leds_expr};\n'
            nl_concat = ('{y_leds, tx_busy, err_sticky, '
                         '(ok_stretch != 0), (rx_stretch != 0)}')
        else:
            nl_tail = ''
            wire_line = ''
            nl_concat = ('{tx_busy, err_sticky, '
                         '(ok_stretch != 0), (rx_stretch != 0)}')
        led_assign = (
            f'    // {ports.led}[0]=RX活動 [1]=フレーム受理 [2]=checksumエラー(sticky) '
            f'[3]=TX中{nl_tail}\n'
            + wire_line
            + _led_assign(nl_concat))

    hdr_tail = (f' [{ports.led_width - 1}:4]=y 下位ビット'
                if ports.led_width > 4 else '')
    top = f"""// mdl-fpga 自動生成トップ({top_name})
// core: {core_module}(in={in_width}bit, out={out_width}bit)
// UART: {baud} baud(CLKS_PER_BIT={clks_per_bit})/ 要求 {req_p + 3}B / 応答 {rsp_q + 3}B
// リセット: {rst_read}{rst_note}(2FF 同期)。{ports.led}[0]=RX活動 [1]=フレーム受理 [2]=checksumエラー(sticky)
//           [3]=TX中{hdr_tail}
// TODO(M1): CE/step デバッグ機構(クリック=1clk)はビューアと同時に導入する

module {top_name} (
    input  wire       {ports.clock},
    input  wire [{ports.reset_width - 1}:0] {rst_port},
    output wire [{ports.led_width - 1}:0] {ports.led},
    input  wire       {ports.uart_rx},
    output wire       {ports.uart_tx}
);
    localparam integer IN_W  = {in_width};
    localparam integer OUT_W = {out_width};
    localparam integer REQ_P = {req_p};
    localparam integer RSP_Q = {rsp_q};

    wire clk = {ports.clock};

    // ---- リセット({rst_read} を 2FF 同期) ----
    reg rst_d1 = 1'b0, rst_d2 = 1'b0;
    always @(posedge clk) begin
        rst_d1 <= {rst_expr};
        rst_d2 <= rst_d1;
    end
    wire rst = rst_d2;

    // ---- 要求フレーム受信 ----
    wire [8*REQ_P-1:0] rx_x;
    wire [7:0]         rx_seq;
    wire               frame_valid, frame_err;
    frame_rx #(
        .PAYLOAD_BYTES(REQ_P),
        .CLKS_PER_BIT({clks_per_bit}),
        .TIMEOUT_CYCLES({timeout_cycles})
    ) u_frx (
        .clk(clk), .rst(rst), .rx({ports.uart_rx}),
        .x(rx_x), .seq(rx_seq), .frame_valid(frame_valid), .err_pulse(frame_err)
    );

    // ---- コア(組み合わせ)+ 入出力 FF ----
    reg  [IN_W-1:0]  xin = {{IN_W{{1'b0}}}};
    reg  [7:0]       seq_hold = 8'd0;
    wire [OUT_W-1:0] y_comb;
    {core_module} u_core (.x(xin), .y(y_comb));
    reg  [OUT_W-1:0] y_reg = {{OUT_W{{1'b0}}}};

    wire [8*RSP_Q-1:0] resp_payload = {resp_expr};

    // ---- 応答フレーム送信 ----
    reg  tx_go = 1'b0;
    wire tx_busy;
    frame_tx #(
        .PAYLOAD_BYTES(RSP_Q),
        .CLKS_PER_BIT({clks_per_bit})
    ) u_ftx (
        .clk(clk), .rst(rst),
        .payload(resp_payload), .seq(seq_hold), .start(tx_go),
        .tx({ports.uart_tx}), .busy(tx_busy)
    );

    // ---- メイン FSM(最新フレームのみ処理、demo1 実証構造) ----
    localparam [1:0] S_IDLE = 2'd0, S_CORE = 2'd1, S_TXW = 2'd2;
    reg [1:0]          state = S_IDLE;
    reg                pending = 1'b0;
    reg [8*REQ_P-1:0]  pend_x = {{8*REQ_P{{1'b0}}}};
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
    reg [LW-1:0] rx_stretch = {{LW{{1'b0}}}};
    reg [LW-1:0] ok_stretch = {{LW{{1'b0}}}};
    reg          err_sticky = 1'b0;
    always @(posedge clk) begin
        if (rst) begin
            rx_stretch <= {{LW{{1'b0}}}};
            ok_stretch <= {{LW{{1'b0}}}};
            err_sticky <= 1'b0;
        end else begin
            if (!{ports.uart_rx})            rx_stretch <= {{LW{{1'b1}}}};
            else if (rx_stretch != 0) rx_stretch <= rx_stretch - 1'b1;
            if (frame_valid) begin
                ok_stretch <= {{LW{{1'b1}}}};
                err_sticky <= 1'b0;
            end else if (ok_stretch != 0) begin
                ok_stretch <= ok_stretch - 1'b1;
            end
            if (frame_err) err_sticky <= 1'b1;
        end
    end
{selftest_section}
{led_assign}

endmodule
"""
    return WrapperGen(
        top_verilog=top,
        top_name=top_name,
        in_width=in_width,
        out_width=out_width,
        req_payload_bytes=req_p,
        rsp_payload_bytes=rsp_q,
        clks_per_bit=clks_per_bit,
        sources=[f'{top_name}.v', 'mdl_core.v', *STATIC_SOURCES]
        + (['selftest.v'] if with_selftest else []),
        groups=('clock', 'uart', 'led', ports.reset_group),
        ports=ports,
    )


def generate_wrapper_tb(gen: WrapperGen, n_vectors: int,
                        sim_clks_per_bit: int = 16) -> str:
    """フルチェーン TB(UART ビット駆動)。CLKS_PER_BIT を小さくして高速化。

    inputs_padded.hex(幅 8*REQ_P)/ expected_padded.hex(幅 8*RSP_Q)を読む。
    """
    req_p = gen.req_payload_bytes
    rsp_q = gen.rsp_payload_bytes
    p = gen.ports
    bit_ns = sim_clks_per_bit * 10  # 50MHz = 10ns/clk
    rst_port = p.reset_port
    rst_bit = p.reset_bit
    rst_w = p.reset_width
    # リセット非アサート値/アサート値(active-low ボタンは休止=全1)
    if p.reset_active_low:
        sw_init = f"{rst_w}'d{(1 << rst_w) - 1}"
        rst_on, rst_off = "1'b0", "1'b1"
    else:
        sw_init = f"{rst_w}'d0"
        rst_on, rst_off = "1'b1", "1'b0"
    return f"""`timescale 1ns/1ps
// mdl-fpga 自動生成フルチェーン TB(UART ビット駆動、CLKS_PER_BIT={sim_clks_per_bit} に短縮)
module tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;   // 50MHz

    reg  [{rst_w - 1}:0] sw = {sw_init};
    wire [{p.led_width - 1}:0] ledr;
    reg  rxd = 1'b1;
    wire txd;

    top_sim dut (.{p.clock}(clk), .{rst_port}(sw), .{p.led}(ledr),
                 .{p.uart_rx}(rxd), .{p.uart_tx}(txd));

    localparam integer REQ_P = {req_p};
    localparam integer RSP_Q = {rsp_q};
    localparam integer BIT_NS = {bit_ns};

    reg [8*REQ_P-1:0] in_vecs  [0:{n_vectors - 1}];
    reg [8*RSP_Q-1:0] exp_vecs [0:{n_vectors - 1}];

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

        // リセット({rst_port}[{rst_bit}])
        sw[{rst_bit}] = {rst_on}; repeat (10) @(posedge clk);
        sw[{rst_bit}] = {rst_off}; repeat (10) @(posedge clk);

        for (i = 0; i < {n_vectors}; i = i + 1) begin
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
"""
