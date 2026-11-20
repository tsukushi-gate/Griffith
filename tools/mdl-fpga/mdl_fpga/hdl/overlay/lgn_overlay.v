// lgn_overlay.v -- 論理ゲート NN の逐次評価オーバーレイ(計算型配線版)。
//
// 出典と差分: 社内先行実験 hdl/lgn_overlay.v(32bit/ゲート)→
// 先行実験 hdl/lgn_overlay4.v(op 4bit + rot アキュムレータ)を mdl-fpga へ移植。
// 先行実験からの設計差分は次の 3 点だけで、パイプライン位相・層境界条件・
// drain・popcount・FSM は無変更(= サイクル数は先行実験と厳密に同一)。
//   [差分1] 引き伸ばし rot 対応: レーンごとに余りレジスタ rem を 1 本足し、
//           基準位置の歩進を「整数歩幅 + 余りの桁上げ」にした(Bresenham 型)。
//           rot 層は 整数歩幅 = V / 余り歩幅 = 0 で同じ回路を通る(乗算器なし)。
//   [差分2] 配線アドレス幅 AW を $clog2(W) から $clog2(W+1) に 1bit 広げ、
// 「W が 2 の冪ちょうどではないこと」という先行実験の制約を外した。
//   [差分3] argmax を組合せループから比較木モジュール overlay_argmax
//           (hdlgen/overlay.py が生成、同点は小 index 勝ち = np.argmax)へ分離。
//
// ゲート記述をメモリから毎サイクル V ゲート分(op 4bit x V)読み、V 並列に
// 2 入力ゲートを評価して活性バッファを層ごとに更新する。配線(a,b)はメモリに
// 持たず、層ごとのシフト量から算術で生成する = **計算型配線**。
//
// インタフェース(1 画像 = 1 推論、start/done ハンドシェイク):
//   start(level)で img の推論開始 → done(level)で pred 確定。
//
// パイプライン(層内 II=1、レイテンシ 3。層境界で drain):
//   p0: opmem[gp] 提示(gp = グローバルゲート語ポインタ、毎 issue +1)
//   p1: op 語確定 → 各レーンの (a,b) を acc から確定、前層バッファへ read 提示
//   p2: act 語確定 → bit mux → op 評価 → V bit 結果語
//   p3: 結果語を現層バッファ(全 V 複製へ同報)書込。最終層なら popcount 累積
//
// 活性バッファ: bufA/bufB を層ごとに prev(読)/cur(書)交替。各 V 複製 =
// true dual-port。複製 k がレーン k 専属(ポートA = operand a、ポートB =
// operand b)なので 2V read/cycle の競合がゼロになる。
//
// 参照する .mem(いずれも hdlgen/overlay.py が生成し、シム作業 dir または
// Quartus プロジェクト dir に置く。ファイル名はリテラル = 合成可能な形):
//   opmem0..3.mem     op 4bit x V(32bit 語)。ROM 本数 NROM = ceil(4V/32)
//   addrinit_a/b.mem  層頭でロードするレーン別アドレス初期値(DEPTH*V 行)
//   reminit.mem       同、余りレジスタ初期値(DEPTH*V 行。rot 層は 0)
`timescale 1ns/1ps
module lgn_overlay #(
    parameter integer V           = 8,     // レーン数(2 の冪、V % 8 == 0)
    parameter integer W           = 8000,  // 層出力幅(全層一様)。W % V == 0
    parameter integer DEPTH       = 6,     // 層数
    parameter integer IN_BITS     = 784,   // 入力ビット幅
    parameter integer CLASSES     = 10,
    parameter integer GS          = 800,   // group_size = W/CLASSES。GS % V == 0
    // 層 0 の基準位置の歩幅(生成器が計算。rot 層は (V, 0))
    parameter integer L0_INT_STEP = 8,
    parameter integer L0_REM_STEP = 0,
    // 予測クラスのビット幅(ポート宣言で使うため parameter)
    parameter integer PRED_W      = (CLASSES < 2) ? 1 : $clog2(CLASSES)
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
    // [差分2] W をちょうど表現できる幅。acc/a_p1/W_in が保持する値は常に
    //   < W_in <= max(IN_BITS, W) = W なので AW bit に必ず収まる。
    localparam integer AW          = $clog2(W + 1);
    localparam integer REM_DIV     = 2 * W;              // 余りの除数(= 2*W_out)
    localparam integer REM_W       = $clog2(REM_DIV + 1);
    localparam integer NROM        = (4*V + 31) / 32;    // op ROM 本数

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
    wire [32*NROM-1:0] opword_p1;
    reg  [LWW-1:0]     lw_p1;
    reg                last_p1;

    // ---- ゲート ROM: op 4bit x V = 4V bit を 1 issue 分の語とし、32bit の
    //      rom32 を NROM = ceil(4V/32) 本並べて連結する($sformatf は合成
    //      非対応につきリテラル名で unroll)。語 gp = ゲート
    //      {gp*V .. gp*V+V-1} の op。lane k = opword_p1[4k+3:4k](lane0 = LSB)。
    generate
      if (NROM > 0) begin: gr0 wire [31:0] q; rom32 #(.FILE("opmem0.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign opword_p1[32*0 +: 32] = q; end
      if (NROM > 1) begin: gr1 wire [31:0] q; rom32 #(.FILE("opmem1.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign opword_p1[32*1 +: 32] = q; end
      if (NROM > 2) begin: gr2 wire [31:0] q; rom32 #(.FILE("opmem2.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign opword_p1[32*2 +: 32] = q; end
      if (NROM > 3) begin: gr3 wire [31:0] q; rom32 #(.FILE("opmem3.mem"),.DEPTH(GATE_WORDS),.AW(GATE_ADDR)) r (.clk(clk),.en(issue),.addr(gp),.q(q)); assign opword_p1[32*3 +: 32] = q; end
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

    // ================= 計算型配線のアドレス生成 =================
    // レーン k は層内語 w のゲート g = w*V + k を担当する。基準位置の分子を
    // N = 2*g*W_in + W_out と置くと g -> g+V で N は 2*V*W_in 増え、これは
    //   2*V*W_in = int_step * REM_DIV + rem_step   (0 <= rem_step < REM_DIV)
    // と分解できる。よって余りレジスタに rem_step を足して REM_DIV 以上なら
    // 引く(rem + rem_step < 2*REM_DIV なので条件減算 1 段で閉じる)ことで
    // 基準位置の歩進 int_step + carry が乗算器なしに得られる。
    // アドレスは a <- a + (int_step + carry) を W_in で条件減算 1 段
    // (int_step + 1 <= W_in を生成器が assert する)。
    // 全層一様幅なので W_out = W は定数、W_in と歩幅だけが層 0 で変わる。
    wire [AW:0]    w_in     = (ly == 0) ? IN_BITS : W;
    wire [AW:0]    step_int = (ly == 0) ? L0_INT_STEP : V;
    wire [REM_W:0] step_rem = (ly == 0) ? L0_REM_STEP : 0;

    reg  [AW-1:0]    acc_a [0:V-1];
    reg  [AW-1:0]    acc_b [0:V-1];
    reg  [REM_W-1:0] rem   [0:V-1];
    reg  [AW-1:0]    a_p1  [0:V-1];   // いま p1 にある語のアドレス
    reg  [AW-1:0]    b_p1  [0:V-1];
    // 層頭でロードする初期表(DEPTH*V 語なので M10K は消費しない)
    reg  [AW-1:0]    init_a [0:DEPTH*V-1];
    reg  [AW-1:0]    init_b [0:DEPTH*V-1];
    reg  [REM_W-1:0] init_r [0:DEPTH*V-1];
    initial begin
        $readmemh("addrinit_a.mem", init_a);
        $readmemh("addrinit_b.mem", init_b);
        $readmemh("reminit.mem",    init_r);
    end

    wire [ACT_ADDR-1:0] rd_a_word [0:V-1];
    wire [ACT_ADDR-1:0] rd_b_word [0:V-1];
    wire [LOGV-1:0]     rd_a_sel  [0:V-1];
    wire [LOGV-1:0]     rd_b_sel  [0:V-1];
    wire [REM_W:0]      rem_sum   [0:V-1];
    wire                rem_carry [0:V-1];
    wire [REM_W-1:0]    nxt_r     [0:V-1];
    wire [AW:0]         inc       [0:V-1];
    wire [AW:0]         sum_a     [0:V-1];
    wire [AW:0]         sum_b     [0:V-1];
    wire [AW-1:0]       nxt_a     [0:V-1];
    wire [AW-1:0]       nxt_b     [0:V-1];

    genvar gi;
    generate for (gi=0; gi<V; gi=gi+1) begin: g_addr
        assign rd_a_word[gi] = a_p1[gi][LOGV+ACT_ADDR-1:LOGV];
        assign rd_b_word[gi] = b_p1[gi][LOGV+ACT_ADDR-1:LOGV];
        assign rd_a_sel[gi]  = a_p1[gi][LOGV-1:0];
        assign rd_b_sel[gi]  = b_p1[gi][LOGV-1:0];
        assign rem_sum[gi]   = rem[gi] + step_rem;
        assign rem_carry[gi] = (rem_sum[gi] >= REM_DIV);
        assign nxt_r[gi]     = rem_carry[gi] ? (rem_sum[gi] - REM_DIV) : rem_sum[gi];
        assign inc[gi]       = step_int + rem_carry[gi];
        assign sum_a[gi]     = acc_a[gi] + inc[gi];
        assign sum_b[gi]     = acc_b[gi] + inc[gi];
        assign nxt_a[gi]     = (sum_a[gi] >= w_in) ? (sum_a[gi] - w_in) : sum_a[gi];
        assign nxt_b[gi]     = (sum_b[gi] >= w_in) ? (sum_b[gi] - w_in) : sum_b[gi];
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
                    for (c=0;c<V;c=c+1) begin   // 層 0 の配線初期化
                        acc_a[c] <= init_a[c];
                        acc_b[c] <= init_b[c];
                        rem[c]   <= init_r[c];
                    end
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
                    // 発行した語のアドレスを p1 へ、acc/rem は次語へ歩進
                    for (c=0;c<V;c=c+1) begin
                        a_p1[c]  <= acc_a[c];
                        b_p1[c]  <= acc_b[c];
                        acc_a[c] <= nxt_a[c];
                        acc_b[c] <= nxt_b[c];
                        rem[c]   <= nxt_r[c];
                    end
                end
                // --- p1 -> p2 ---
                v2 <= v1; lw_p2 <= lw_p1; last_p2 <= last_p1;
                for (c=0;c<V;c=c+1) begin
                    op_p2[4*c +: 4]         <= opword_p1[4*c+3 -: 4];
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
                        for (c=0;c<V;c=c+1) begin // 次層の配線初期化
                            acc_a[c] <= init_a[(ly+1)*V + c];
                            acc_b[c] <= init_b[(ly+1)*V + c];
                            rem[c]   <= init_r[(ly+1)*V + c];
                        end
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
