// lgn_overlay_stored.v -- 論理ゲート NN の逐次評価オーバーレイ(保存型配線版)。
//
// 出典と差分: 社内先行実験 hdl/lgn_overlay.v(32bit/ゲートの
// アドレス指定型)を mdl-fpga へ移植した。**自由配線**(置換・ランダム・学習
// 配線など任意の添字)を載せられる代わりに、1 ゲートの記録に前層アドレスを
// 2 本持つ。計算型配線版 lgn_overlay.v とはゲート記録の読み方だけが違い、
// パイプライン位相・層境界条件・drain・popcount・FSM は同一なので
// **サイクル数は計算型と厳密に同じ**(先行実験 C1 と先行実験 D1 が同じ 6,129)。
//
// 先行実験からの差分は次の 5 点:
//   [差分1] 記録幅をパラメータ化(REC_BITS = 4 + 2*ADDR_F)。32bit のとき
//           {op[31:28], a[27:14], b[13:0]} = 先行実験と同一形式。層幅が
//           16,384 を超える場合だけ 40bit(a/b 各 18bit)にする。
//   [差分2] レーン ROM を gate_rom(rom32 / rom40 を記録幅で選ぶ殻)経由に
//           した。rom32.v / rom40.v の宣言形は変えない(M10K 推論の条件)。
//   [差分3] 入力幅を V 境界へゼロ拡張(IN_PAD)。先行実験の IN_BITS % V == 0
//           制約が外れる。
//   [差分4] popcount カウンタ幅を $clog2(GS+1)、予測幅を PRED_W にした
//           (先行実験は 12bit / 4bit 固定)。argmax は比較木モジュール
//           overlay_argmax(hdlgen/overlay.py が生成、同点は小 index 勝ち)。
//   [差分5] 活性語アドレスの取り出しを a_i[LOGV +: ACT_ADDR] にした
//           (先行実験は a_i[13:LOGV] の暗黙切詰め。W が 2 の冪でも可)。
//
// ゲート記述をメモリから毎サイクル V ゲート分読み、V 並列に 2 入力ゲートを
// 評価して活性バッファを層ごとに更新する。配線 (a, b) は**記録から読む**
// = 保存型(自由配線)。
//
// インタフェース(1 画像 = 1 推論、start/done ハンドシェイク):
//   start(level)で img の推論開始 → done(level)で pred 確定。
//
// パイプライン(層内 II=1、レイテンシ 3。層境界で drain):
//   p0: gatemem[gp] 提示(gp = グローバルゲート語ポインタ、毎 issue +1)
//   p1: 記録語確定 → 各レーンの {op,a,b} 抽出、前層バッファへ read 提示
//   p2: act 語確定 → bit mux → op 評価 → V bit 結果語
//   p3: 結果語を現層バッファ(全 V 複製へ同報)書込。最終層なら popcount 累積
//
// 活性バッファ: bufA/bufB を層ごとに prev(読)/cur(書)交替。各 V 複製 =
// true dual-port。複製 k がレーン k 専属(ポートA = operand a、ポートB =
// operand b)なので 2V read/cycle の競合がゼロになる。
//
// 参照する .mem(hdlgen/overlay.py が生成し、シム作業 dir または Quartus
// プロジェクト dir に置く。ファイル名はリテラル = 合成可能な形):
//   gatemem_lane0..{V-1}.mem  レーン k のゲート記録(1 行 = 1 語 hex)。
//                             レーン k の語 gp = ゲート gp*V + k
`timescale 1ns/1ps
module lgn_overlay_stored #(
    parameter integer V        = 8,     // レーン数(2 の冪、V % 8 == 0)
    parameter integer W        = 8000,  // 層出力幅(全層一様)。W % V == 0
    parameter integer DEPTH    = 6,     // 層数
    parameter integer IN_BITS  = 784,   // 入力ビット幅
    parameter integer CLASSES  = 10,
    parameter integer GS       = 800,   // group_size = W/CLASSES。GS % V == 0
    parameter integer REC_BITS = 32,    // 1 ゲートの記録幅(32 or 40)
    parameter integer ADDR_F   = 14,    // a / b のフィールド幅 = (REC_BITS-4)/2
    // 予測クラスのビット幅(ポート宣言で使うため parameter)
    parameter integer PRED_W   = (CLASSES < 2) ? 1 : $clog2(CLASSES)
) (
    input  wire               clk,
    input  wire               rst,       // 同期リセット(1 で IDLE)
    input  wire               start,     // 推論開始
    input  wire [IN_BITS-1:0] img,       // 二値画像(img[j] = 入力ビット j)
    output reg                done,      // 推論完了(level)
    output reg  [PRED_W-1:0]  pred       // 予測クラス(argmax、同点は小 index)
);
    // ---- 導出定数 ----
    localparam integer LOGV        = $clog2(V);
    localparam integer ACT_WORDS   = W / V;              // 活性バッファ深さ(語)
    localparam integer ACT_ADDR    = $clog2(ACT_WORDS);
    localparam integer LAYER_WORDS = W / V;              // 1 層の語数
    localparam integer GATE_WORDS  = (DEPTH * W) / V;    // 全ゲート語数
    localparam integer GATE_ADDR   = $clog2(GATE_WORDS);
    localparam integer IN_PAD      = ((IN_BITS + V - 1) / V) * V;
    localparam integer IN_WORDS    = IN_PAD / V;         // 画像ロード語数
    localparam integer WPC         = GS / V;             // words per class
    localparam integer CNT_W       = $clog2(GS + 1);     // popcount カウンタ幅
    localparam integer LWW         = (LAYER_WORDS <= 1) ? 1 : $clog2(LAYER_WORDS);
    localparam integer GWORD       = V * REC_BITS;       // ゲート語ビット幅

    // ================= 制御 FSM =================
    localparam [2:0] S_IDLE=3'd0, S_LOAD=3'd1, S_EVAL=3'd2, S_ARGMAX=3'd3, S_DONE=3'd4;
    reg [2:0]              state;
    reg [IN_PAD-1:0]       img_reg;            // V 境界へゼロ拡張した画像
    reg [GATE_ADDR-1:0]    gp;                 // グローバルゲート語ポインタ
    reg [LWW:0]            lw_cnt;             // 層内発行カウンタ 0..LAYER_WORDS
    reg [$clog2(DEPTH+1)-1:0] ly;              // 現層 index
    reg                    A_is_cur;           // A が cur(書)buffer か
    reg [ACT_ADDR-1:0]     load_w;             // 画像ロード語カウンタ

    wire is_last_layer = (ly == DEPTH-1);
    wire issue = (state == S_EVAL) && (lw_cnt < LAYER_WORDS);

    // ---- パイプライン valid ----
    reg v1, v2, v3;
    wire pipe_empty = ~(v1 | v2 | v3);

    // ---- p1 レジスタ ----
    wire [GWORD-1:0] gword_p1;
    reg  [LWW-1:0]   lw_p1;
    reg              last_p1;

    // ---- ゲート ROM: レーン別に 1 本(記録幅 REC_BITS)。$sformatf は合成
    //      非対応につきリテラル名で unroll。レーン k の語 gp = ゲート gp*V+k。
    generate
      if (V > 0) begin: gl0 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane0.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*0 +: REC_BITS] = q; end
      if (V > 1) begin: gl1 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane1.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*1 +: REC_BITS] = q; end
      if (V > 2) begin: gl2 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane2.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*2 +: REC_BITS] = q; end
      if (V > 3) begin: gl3 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane3.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*3 +: REC_BITS] = q; end
      if (V > 4) begin: gl4 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane4.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*4 +: REC_BITS] = q; end
      if (V > 5) begin: gl5 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane5.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*5 +: REC_BITS] = q; end
      if (V > 6) begin: gl6 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane6.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*6 +: REC_BITS] = q; end
      if (V > 7) begin: gl7 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane7.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*7 +: REC_BITS] = q; end
      if (V > 8) begin: gl8 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane8.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*8 +: REC_BITS] = q; end
      if (V > 9) begin: gl9 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane9.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*9 +: REC_BITS] = q; end
      if (V > 10) begin: gl10 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane10.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*10 +: REC_BITS] = q; end
      if (V > 11) begin: gl11 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane11.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*11 +: REC_BITS] = q; end
      if (V > 12) begin: gl12 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane12.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*12 +: REC_BITS] = q; end
      if (V > 13) begin: gl13 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane13.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*13 +: REC_BITS] = q; end
      if (V > 14) begin: gl14 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane14.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*14 +: REC_BITS] = q; end
      if (V > 15) begin: gl15 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane15.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*15 +: REC_BITS] = q; end
      if (V > 16) begin: gl16 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane16.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*16 +: REC_BITS] = q; end
      if (V > 17) begin: gl17 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane17.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*17 +: REC_BITS] = q; end
      if (V > 18) begin: gl18 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane18.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*18 +: REC_BITS] = q; end
      if (V > 19) begin: gl19 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane19.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*19 +: REC_BITS] = q; end
      if (V > 20) begin: gl20 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane20.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*20 +: REC_BITS] = q; end
      if (V > 21) begin: gl21 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane21.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*21 +: REC_BITS] = q; end
      if (V > 22) begin: gl22 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane22.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*22 +: REC_BITS] = q; end
      if (V > 23) begin: gl23 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane23.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*23 +: REC_BITS] = q; end
      if (V > 24) begin: gl24 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane24.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*24 +: REC_BITS] = q; end
      if (V > 25) begin: gl25 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane25.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*25 +: REC_BITS] = q; end
      if (V > 26) begin: gl26 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane26.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*26 +: REC_BITS] = q; end
      if (V > 27) begin: gl27 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane27.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*27 +: REC_BITS] = q; end
      if (V > 28) begin: gl28 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane28.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*28 +: REC_BITS] = q; end
      if (V > 29) begin: gl29 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane29.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*29 +: REC_BITS] = q; end
      if (V > 30) begin: gl30 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane30.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*30 +: REC_BITS] = q; end
      if (V > 31) begin: gl31 wire [REC_BITS-1:0] q; gate_rom #(.FILE("gatemem_lane31.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR),.DW(REC_BITS)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign gword_p1[REC_BITS*31 +: REC_BITS] = q; end
    endgenerate

    // ---- p2 レジスタ ----
    reg [4*V-1:0]      op_p2;
    reg [LOGV*V-1:0]   selA_p2, selB_p2;
    reg [LWW-1:0]      lw_p2;
    reg                last_p2;
    // ---- p3 レジスタ ----
    reg [V-1:0]        res_p3;
    reg [LWW-1:0]      lw_p3;
    reg                last_p3;

    // ============ 保存型配線: 記録のフィールドがそのままアドレス ============
    // レーン k の記録 = {op[REC_BITS-1 -: 4], a[2*ADDR_F-1:ADDR_F], b[ADDR_F-1:0]}
    // (REC_BITS = 32 なら {op[31:28], a[27:14], b[13:0]} = 先行実験と同一)。
    // 活性 RAM は V 語ずつ詰めてあるので、アドレスの上位 ACT_ADDR bit が語、
    // 下位 LOGV bit が語内ビット位置になる(LOGV + ACT_ADDR = ceil(log2(W))
    // <= ADDR_F は生成器が検査する)。
    wire [ACT_ADDR-1:0] rd_a_word [0:V-1];
    wire [ACT_ADDR-1:0] rd_b_word [0:V-1];
    wire [LOGV-1:0]     rd_a_sel  [0:V-1];
    wire [LOGV-1:0]     rd_b_sel  [0:V-1];
    genvar gi;
    generate for (gi=0; gi<V; gi=gi+1) begin: g_addr
        wire [ADDR_F-1:0] a_i = gword_p1[REC_BITS*gi + ADDR_F +: ADDR_F];
        wire [ADDR_F-1:0] b_i = gword_p1[REC_BITS*gi +: ADDR_F];
        assign rd_a_word[gi] = a_i[LOGV +: ACT_ADDR];
        assign rd_b_word[gi] = b_i[LOGV +: ACT_ADDR];
        assign rd_a_sel[gi]  = a_i[LOGV-1:0];
        assign rd_b_sel[gi]  = b_i[LOGV-1:0];
    end endgenerate

    // ---- 現層 cur バッファへの書込(p3 or LOAD)----
    wire                cur_we    = (state==S_LOAD) ? 1'b1 : (v3 & (state==S_EVAL));
    wire [ACT_ADDR-1:0] cur_waddr = (state==S_LOAD) ? load_w : lw_p3[ACT_ADDR-1:0];
    wire [V-1:0]        img_slice = img_reg[load_w*V +: V];
    wire [V-1:0]        cur_wdata = (state==S_LOAD) ? img_slice : res_p3;
    wire                weA = cur_we &  A_is_cur;
    wire                weB = cur_we & ~A_is_cur;

    // ========== 活性バッファ(bufA/bufB、各 V 複製 = 単一 M10K TDP)==========
    wire [V-1:0] pq_a [0:V-1];   // operand a 語(prev)
    wire [V-1:0] pq_b [0:V-1];   // operand b 語(prev)
    generate for (gi=0; gi<V; gi=gi+1) begin: g_buf
        wire [V-1:0] qAa, qAb, qBa, qBb;
        tdp_ram #(.DW(V), .DEPTH(ACT_WORDS), .AW(ACT_ADDR)) bufA (
            .clk(clk),
            .we_a(weA), .addr_a(A_is_cur ? cur_waddr : rd_a_word[gi]), .data_a(cur_wdata), .q_a(qAa),
            .we_b(1'b0), .addr_b(rd_b_word[gi]), .data_b({V{1'b0}}), .q_b(qAb)
        );
        tdp_ram #(.DW(V), .DEPTH(ACT_WORDS), .AW(ACT_ADDR)) bufB (
            .clk(clk),
            .we_a(weB), .addr_a(~A_is_cur ? cur_waddr : rd_a_word[gi]), .data_a(cur_wdata), .q_a(qBa),
            .we_b(1'b0), .addr_b(rd_b_word[gi]), .data_b({V{1'b0}}), .q_b(qBb)
        );
        assign pq_a[gi] = A_is_cur ? qBa : qAa;   // prev = A_is_cur ? B : A
        assign pq_b[gi] = A_is_cur ? qBb : qAb;
    end endgenerate

    // ================= p2: bit mux + op 評価 =================
    wire [V-1:0] res_w;
    generate for (gi=0; gi<V; gi=gi+1) begin: g_eval
        wire Abit = pq_a[gi][ selA_p2[LOGV*gi +: LOGV] ];
        wire Bbit = pq_b[gi][ selB_p2[LOGV*gi +: LOGV] ];
        gate_eval u_eval (.op(op_p2[4*gi +: 4]), .A(Abit), .B(Bbit), .out(res_w[gi]));
    end endgenerate

    // ---- popcount(最終層融合)----
    reg  [CNT_W-1:0] cnt [0:CLASSES-1];
    reg  [$clog2(CLASSES+1)-1:0] pc_cls;
    reg  [LWW-1:0]   pc_sub;
    integer pi;
    reg [$clog2(V+1)-1:0] popc;
    always @* begin
        popc = 0;
        for (pi=0; pi<V; pi=pi+1) popc = popc + res_p3[pi];
    end

    // ---- argmax(比較木。同点は小 index 勝ち = np.argmax)----
    wire [CLASSES*CNT_W-1:0] scores_packed;
    generate for (gi=0; gi<CLASSES; gi=gi+1) begin: g_pack
        assign scores_packed[CNT_W*gi +: CNT_W] = cnt[gi];
    end endgenerate
    wire [PRED_W-1:0] amax;
    overlay_argmax u_argmax (.scores(scores_packed), .pred_index(amax));

    // ================= 逐次論理 =================
    integer c;
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 1'b0;
            v1<=0; v2<=0; v3<=0; gp<=0; lw_cnt<=0; ly<=0; A_is_cur<=1'b1; load_w<=0;
        end else begin
            v1 <= 1'b0;

            case (state)
            S_IDLE: begin
                done <= 1'b0;
                if (start) begin
                    img_reg <= img;            // IN_PAD > IN_BITS のとき上位は 0
                    state <= S_LOAD;
                    load_w <= 0;
                    A_is_cur <= 1'b1;          // LOAD 中は cur=A
                    gp <= 0; lw_cnt <= 0; ly <= 0;
                    v1<=0; v2<=0; v3<=0;
                    for (c=0;c<CLASSES;c=c+1) cnt[c] <= 0;
                    pc_cls <= 0; pc_sub <= 0;
                end
            end

            S_LOAD: begin
                // img_reg を bufA へ IN_WORDS 語ストリーム書込
                if (load_w == IN_WORDS-1) begin
                    state <= S_EVAL;
                    A_is_cur <= 1'b0;      // layer0: prev=A(読), cur=B(書)
                    lw_cnt <= 0;
                    ly <= 0;
                end else begin
                    load_w <= load_w + 1'b1;
                end
            end

            S_EVAL: begin
                // --- p0/p1 発行 ---
                if (issue) begin
                    lw_p1    <= lw_cnt[LWW-1:0];
                    last_p1  <= is_last_layer;
                    v1       <= 1'b1;
                    gp       <= gp + 1'b1;
                    lw_cnt   <= lw_cnt + 1'b1;
                end
                // --- p1 -> p2 --- op と bit-select(a,b 下位 LOGV)を登録
                v2 <= v1; lw_p2 <= lw_p1; last_p2 <= last_p1;
                for (c=0;c<V;c=c+1) begin
                    op_p2[4*c +: 4]         <= gword_p1[REC_BITS*c + REC_BITS-1 -: 4];
                    selA_p2[LOGV*c +: LOGV] <= rd_a_sel[c];
                    selB_p2[LOGV*c +: LOGV] <= rd_b_sel[c];
                end
                // --- p2 -> p3 ---
                v3 <= v2; lw_p3 <= lw_p2; last_p3 <= last_p2;
                res_p3 <= res_w;

                // --- p3 popcount 累積(最終層)---
                if (v3 && last_p3) begin
                    cnt[pc_cls] <= cnt[pc_cls] + popc;
                    if (pc_sub == WPC-1) begin pc_sub <= 0; pc_cls <= pc_cls + 1'b1; end
                    else pc_sub <= pc_sub + 1'b1;
                end

                // --- 層完了判定(発行終了 + パイプ drain)---
                if ((lw_cnt == LAYER_WORDS) && pipe_empty) begin
                    if (is_last_layer) begin
                        state <= S_ARGMAX;
                    end else begin
                        ly       <= ly + 1'b1;
                        A_is_cur <= ~A_is_cur;    // prev/cur swap
                        lw_cnt   <= 0;
                    end
                end
            end

            S_ARGMAX: begin
                pred  <= amax;
                done  <= 1'b1;
                state <= S_DONE;
            end

            S_DONE: begin
                done <= 1'b1;
                if (!start) begin done <= 1'b0; state <= S_IDLE; end
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
