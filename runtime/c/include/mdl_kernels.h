/* griffith-logic — 論理ゲートNNの移植可能Cランタイム(カーネル層)
 *
 * C99。依存は <stdint.h> / <string.h> のみ。実行時のヒープ確保はゼロ
 * (作業領域は呼び出し側が渡す)。フリースタンディング環境でも使えるよう
 * libc は memcpy/memset しか使わない。
 *
 * ============================================================
 * ビットプレーンのレイアウト契約(全ターゲット共通)
 * ============================================================
 * 値は uint32_t の配列で [batch][C][H][wpr] の row-major。
 *   wpr = words_per_row(W) = ceil(W/32)
 *   列 x はワード x>>5 のビット x&31(LSB-first)
 *   行頭は必ずワード境界(水平シフトが行内で完結するための必須条件)
 *   端数ビット(x >= W)は **ノード値では常に0** が不変条件。NOT系ゲートは
 *   ここに1を立てるので、行を書き終える箇所で必ず tail_mask を AND する
 *
 * 例外: スクラッチ内のタップ平面は端数ビットが0とは限らない。左シフト
 * (s<0)は元の bit W-1 を位置 W へ運ぶためで、TS側 shiftRowInto も同じ
 * (マスクしない)。ゲートの評価はビットごとに独立なので端数ビットが何で
 * あっても x<W の結果には影響せず、不変条件は出力書き出しの tail_mask で
 * 回復する。
 *
 * この契約は tools/webbench/src/engines/bitplane.ts と同一であり、
 * JS / WASM / WebGPU / この C 実装は同じ入力に対して同じワード列を返す。
 *
 * ============================================================
 * conv2d の実行モデル
 * ============================================================
 * ゲート木の縮約(命令列化)と、タップ平面・作業平面のメモリ割り付けは
 * **エクスポータ側(Python)で完了している**。ここに来る命令列は
 * オペランドが既に「スクラッチ内の絶対ワードオフセット」に解決済みで、
 * C 側は種別分岐も生存区間解析も持たない。
 *
 * スクラッチの構成(単位はワード):
 *   [0,            tap_rows_words)   定数0平面
 *   [tap_rows_words, 2*tap_rows_words) 定数1平面(全ビット1)
 *   [pool_base,    reg_base)          タップ平面プール
 *   [reg_base,     scratch_words)     命令の作業平面
 */
#ifndef MDL_KERNELS_H
#define MDL_KERNELS_H

/* stdint.h / stddef.h はフリースタンディング処理系でも必ず提供される
 * (C99 §4)。ホスト専用のヘッダはここでも実装側でも使わない。 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* レイアウト補助                                                      */
/* ------------------------------------------------------------------ */

/** 1行に必要なワード数(= ceil(W/32))。 */
static inline int32_t mdl_words_per_row(int32_t width) {
  return (width + 31) >> 5;
}

/** 行末尾ワードの有効ビットマスク(W%32==0 なら全ビット有効)。 */
static inline uint32_t mdl_tail_mask(int32_t width) {
  int32_t r = width & 31;
  return r == 0 ? 0xffffffffu : ((uint32_t)1 << r) - 1u;
}

/* ------------------------------------------------------------------ */
/* 2入力ゲート                                                         */
/* ------------------------------------------------------------------ */

/** 真理値表 tt(bit index = (b<<1)|a、値 = f(a,b))で1ワードぶん評価する。 */
uint32_t mdl_gate2(uint32_t tt, uint32_t a, uint32_t b);

/** buf 内の区間 [dst_off, dst_off+n) へ gate2 を適用する。
 *  dst はスロット割当によりオペランドと必ず別領域なのでエイリアスしない。 */
void mdl_gate2_range(uint32_t *buf, int32_t dst_off, int32_t a_off,
                     int32_t b_off, uint32_t tt, int32_t n);

/* ------------------------------------------------------------------ */
/* pack / unpack                                                       */
/* ------------------------------------------------------------------ */

/** 0/1 のバイト列 [C][H][W] を [C][H][wpr] のビットプレーンへ詰める。
 *  非ゼロを1とみなす。端数ビットは0のまま(不変条件)。 */
void mdl_pack_planes(const uint8_t *src, uint32_t *dst,
                     int32_t c, int32_t h, int32_t w);

/** ビットプレーンを 0/1 のバイト列 [C][H][W] へ戻す(検証・readout用)。 */
void mdl_unpack_planes(const uint32_t *src, uint8_t *dst,
                       int32_t c, int32_t h, int32_t w);

/* ------------------------------------------------------------------ */
/* conv2d                                                              */
/* ------------------------------------------------------------------ */

/** conv2d 1ノードの実行レイアウト。全フィールドはエクスポータが決める。 */
typedef struct {
  int32_t h, w, h_out, w_out;
  int32_t s_h, s_w, p_h, p_w;
  int32_t out_channels;
  int32_t wpr_in, wpr_out, tap_wpr, tap_rows_words;
  uint32_t mask_out;
  int32_t use_fast_path;      /* s==1 かつ w_out==w */
  const int32_t *plane_c;     /* [n_planes] */
  const int32_t *plane_dy;    /* [n_planes] */
  const int32_t *plane_dx;    /* [n_planes] */
  int32_t n_planes, plane_stride, plane_rows;
  int32_t in_plane_words, out_plane_words;
  const int32_t *instr_flat;  /* [n_instrs*4] = {tt, a_off, b_off, dst_off} */
  int32_t n_instrs;
  const int32_t *oc_src_off;  /* [out_channels] */
  const int32_t *write_start; /* [n_instrs+2] */
  const int32_t *write_oc;    /* [out_channels] */
  int32_t pool_base, reg_base, scratch_words;
} mdl_conv_layout;

/** 「共有シフト平面」を1本作る(高速路: stride=1 かつ w_out==w)。
 *
 *  水平シフト量 s = dx - p_w は dy に依存しないので、(channel, dx) ごとに
 *  上下 p_h 行のゼロ padding を含む h+2*p_h 行の平面を1回だけ作れば、
 *  dy 違いの全タップは同じ平面を行オフセット dy から読むだけで得られる
 *  (XNNPACK の indirection buffer と同型)。
 *
 *  dst と src は別バッファであること(エイリアス不可)。 */
void mdl_shared_shift_plane(const uint32_t *src, int32_t src_base,
                            int32_t h, int32_t wpr, int32_t p_h, int32_t s,
                            uint32_t *dst, int32_t dst_base);

/** 汎用(stride任意・pad任意・w_out!=w 可)のタップ平面構築。
 *  1ビットずつの gather で遅いが常に正しい。 */
void mdl_tap_plane_strided(const uint32_t *src, int32_t src_base,
                           int32_t h, int32_t w, int32_t wpr_in,
                           int32_t dy, int32_t dx,
                           int32_t h_out, int32_t w_out, int32_t wpr_out,
                           int32_t s_h, int32_t s_w, int32_t p_h, int32_t p_w,
                           uint32_t *dst, int32_t dst_base);

/** conv2d を1バッチ要素ぶん実行する。
 *
 *  scratch は少なくとも layout->scratch_words ワード。呼び出し前に
 *  mdl_conv_init_scratch で定数平面を用意しておくこと(1回でよい)。
 *  out は out_plane_words ワード。 */
/** タップ平面プールだけを作る(命令列の実行は行わない)。
 *  融合コード(生成側で命令列を直線展開したもの)から呼ぶ。 */
void mdl_conv_build_planes(const mdl_conv_layout *layout, const uint32_t *in,
                           uint32_t *scratch);

/** 出力の行末語に tail_mask をかける。融合コードは内側ループを
 *  分岐無しに保つため、マスクを最後にまとめてかける。 */
void mdl_conv_mask_tails(const mdl_conv_layout *layout, uint32_t *out);

void mdl_conv_exec(const mdl_conv_layout *layout, const uint32_t *in,
                   uint32_t *scratch, uint32_t *out);

/** スクラッチの定数平面(0 と 全1)を用意する。レイアウトごとに1回。 */
void mdl_conv_init_scratch(const mdl_conv_layout *layout, uint32_t *scratch);

/* ------------------------------------------------------------------ */
/* 配線・プーリング(ビットプレーンのまま)                            */
/* ------------------------------------------------------------------ */

/** or_pool2d(kernel=2, stride=2, padding=0)。
 *  縦OR → 水平隣接OR → 偶数ビット圧縮。出力は [C][h_out][wpr_out]。
 *  h_out = (h-2)/2+1、w_out = (w-2)/2+1。 */
void mdl_or_pool2x2(const uint32_t *in, uint32_t *out,
                    int32_t c, int32_t h, int32_t w);

/** upsample2d(scale=2, nearest)。1入力ビットを出力の (2i, 2i+1) と
 *  2行へ複製する。出力は [C][2h][wpr(2w)]。 */
void mdl_upsample2d_x2(const uint32_t *in, uint32_t *out,
                       int32_t c, int32_t h, int32_t w);

/** チャネル軸だけの並べ替え(H, W, wpr は不変)。
 *  出力チャネル o の内容は入力チャネル order[o] のコピー。 */
void mdl_permute_channels(const uint32_t *in, uint32_t *out,
                          const int32_t *order, int32_t c_out,
                          int32_t h, int32_t wpr);

/** チャネル軸concat。入力は空間形状(h, wpr)が揃っていること。
 *  srcs[i] は [c_in[i]][h][wpr]。 */
void mdl_concat_planes(const uint32_t *const *srcs, const int32_t *c_in,
                       int32_t n_in, uint32_t *out, int32_t h, int32_t wpr);

/* ------------------------------------------------------------------ */
/* 読み出し                                                            */
/* ------------------------------------------------------------------ */

/** SpatialGroupSum(permute[0,2,3,1] + group_sum の融合形)。
 *
 *  score[o*h*w + y*w + x] = Σ_{g<gs} bit(チャネル o*gs+g, y, x)
 *
 *  ビットスライスの carry-save 加算(垂直カウンタ)で1ワード = 32空間位置を
 *  並列に足す。planes は呼び出し側が渡す作業領域で
 *  mdl_spatial_group_sum_planes(gs) * wpr ワード必要。
 *  出力 scores は [n_out * h * w] 要素。 */
void mdl_spatial_group_sum(const uint32_t *in, int32_t *scores,
                           int32_t n_out, int32_t gs, int32_t h, int32_t w,
                           uint32_t *planes);

/** 上の planes に必要な平面数 = ceil(log2(gs+1))。 */
int32_t mdl_spatial_group_sum_planes(int32_t gs);

/* ------------------------------------------------------------------ */
/* 非パック(0/1 の int32 列)で動く層                                  */
/* ------------------------------------------------------------------ */

/* 論理値(0/1)の非パック表現は uint8_t。整数スコアになるのは group_sum 系の
 * 出力だけなので、そこだけ int32_t で受ける。 */

/** lut_layer(n_inputs <= 5、真理値表は 2^n <= 32 ビットで uint32 に収まる)。
 *  addr = Σ_j x[indices[j][o]] << j、out[o] = (tt[o] >> addr) & 1。
 *  indices は [n_inputs][out_width] の row-major。 */
void mdl_lut_layer(const uint8_t *x, uint8_t *out, int32_t n_inputs,
                   int32_t out_width, const int32_t *indices,
                   const uint32_t *truth_tables);

/* ---- 真理値表のビットスライス表現 ----
 *
 * `mdl_lut_layer` は「1ゲートぶんの真理値表を1語で持ち、addr で可変シフト」
 * する。ゲートごとに 4 バイト読むので、k6000 級では表だけで 144KB を舐める。
 *
 * ビットスライス表現では真理値表を**最小項ごとの平面**へ転置する:
 *
 *   tt_slices[m][j] のビット l = ゲート (j*32+l) の真理値表のビット m
 *
 * 並びは `[2^n_inputs][ceil(out_width/32)]`。こうすると 32 ゲートぶんの
 * 真理値表が 2^n 語に収まり(2入力なら 4 語 = 0.5 バイト/ゲート)、
 * **種類の違う32ゲートを1組のビット演算でまとめて評価できる**:
 *
 *   sel(c, u, v) = u ^ (c & (u ^ v))              … 32レーン一括の三項演算
 *   out = sel(a1, sel(a0, t0, t1), sel(a0, t2, t3))   … 2入力なら9演算
 *
 * 入力側は添字が不規則なので 32 バイト集めてから `mdl_mask32` で詰める。
 * 値は `mdl_lut_layer` と完全に一致する。
 */

/** ビットスライス真理値表版の lut_layer。n_inputs <= 5。 */
void mdl_lut_layer_slice(const uint8_t *x, uint8_t *out, int32_t n_inputs,
                         int32_t out_width, const int32_t *indices,
                         const uint32_t *tt_slices);

/* ---- 添字の16bit化 ----
 *
 * 添字(indices)は「1つ前の層の出力内の位置」を指すので、値域は
 * 0..in_width-1。in_width <= 65536 の層は uint16_t で表現でき、int32_t版
 * (4byte/添字)の半分になる。真理値表に次いで添字が重みの支配項
 * だったので(実測: デプロイキット設計メモ節、添字が
 * 添字+真理値表の94%を占める)、ここが効く。
 *
 * 2経路(int32_t版 / uint16_t版)はコンパイル時に別関数として両方生成され、
 * cbackend が層ごとに in_width を見てどちらを呼ぶか選ぶ(実行時分岐は
 * 無い)。値は1ビットも変わらない(同じマクロ本体を展開しているだけ)。
 */

/** ビットスライス真理値表版の lut_layer、添字uint16_t版。
 *  in_width(前段の出力幅)<= 65536 のときだけ選ばれる。 */
void mdl_lut_layer_slice_u16(const uint8_t *x, uint8_t *out, int32_t n_inputs,
                             int32_t out_width, const uint16_t *indices,
                             const uint32_t *tt_slices);

/* ---- 縮約済み dense 平坦ネットリスト(IR の lut_flat ノード) ----
 *
 * 作業領域 `sig` の信号表を先頭から1ゲートずつ埋める**表駆動**の評価器:
 *
 *     sig[0]                 = 0(定数0)
 *     sig[1 .. in_width]     = 入力ビット(x[k] → sig[1+k])
 *     sig[in_width+1+g]      = (tt[g] >> ((sig[b[g]] << 1) | sig[a[g]])) & 1
 *     out[i]                 = sig[o[i] >> 1] ^ (o[i] & 1)   (o = 2*信号+極性)
 *
 * `tt` は lut_layer と同じ規約の 4bit 真理値表(bit index = (b<<1)|a)。
 * a / b は自分より前の信号のみを指す(トポロジカル順、生成側が保証)。
 * 添字表(a, b, o)は信号数が 32768 以下なら uint16_t、超えたら uint32_t
 * (cbackend が選ぶ。値は1ビットも変わらない)。`sig` は
 * 1 + in_width + n_gates バイト以上を呼び出し側が用意する(生成コードは
 * コンテキストのスクラッチ領域を渡す)。 */
void mdl_lut_flat_u16(const uint8_t *x, uint8_t *out, uint8_t *sig,
                      int32_t in_width, int32_t n_gates, int32_t out_width,
                      const uint8_t *tt, const uint16_t *a, const uint16_t *b,
                      const uint16_t *o);
void mdl_lut_flat_u32(const uint8_t *x, uint8_t *out, uint8_t *sig,
                      int32_t in_width, int32_t n_gates, int32_t out_width,
                      const uint8_t *tt, const uint32_t *a, const uint32_t *b,
                      const uint32_t *o);

/* ---- lut_flat のバッチ軸パック版 ----
 *
 * 上の `mdl_lut_flat_*` と同じ信号表を、**1信号 = bw 語(1語 = 32 枚)**で持つ:
 *
 *     sig[0 .. bw)                       = 0(定数0、全枚)
 *     sig[(1+k)*bw .. )                  = 入力ビット k(x の行 k)
 *     sig[(in_width+1+g)*bw .. )         = ゲート g の出力(32枚/語)
 *     out[i*bw .. )                      = sig[(o[i]>>1)*bw ..] ^ (極性 ? ~0 : 0)
 *
 * 1 ゲート = bw 語の butterfly-select(と同じ 3 演算、tt は 0/全1 に
 * broadcast)。ゲートの添字・表は単枚版と同じ配列を共有する。
 * `x` が `sig + bw` を指していれば入力のコピーを省く(生成コードはそう置く)。
 * 端数の枚(>= n)のビットは値が不定でよい(`mdl_group_sum_packed` が無視する)。
 * `sig` は (1 + in_width + n_gates) * bw 語以上を呼び出し側が用意する。 */
void mdl_lut_flat_packed_u16(const uint32_t *x, uint32_t *out, uint32_t *sig,
                             int32_t in_width, int32_t n_gates,
                             int32_t out_width, const uint8_t *tt,
                             const uint16_t *a, const uint16_t *b,
                             const uint16_t *o, int32_t bw);
void mdl_lut_flat_packed_u32(const uint32_t *x, uint32_t *out, uint32_t *sig,
                             int32_t in_width, int32_t n_gates,
                             int32_t out_width, const uint8_t *tt,
                             const uint32_t *a, const uint32_t *b,
                             const uint32_t *o, int32_t bw);

/** group_sum。in_width は k で割り切れること。 */
void mdl_group_sum(const uint8_t *x, int32_t *out, int32_t in_width, int32_t k);

/** libc memcpy相当。generated model.c がアキュムレータのシード
 * コピー(退化ゲート除去のentryコピー)に使う。実体は mdl_kernels.c 側で
 * MDL_NO_LIBC のあり/なしに応じて切り替える(通常ビルドはlibc memcpyへ
 * 委譲、フリースタンディングはスカラーループ)。generated model.c 自体は
 * <string.h> に依存しない(ベアメタル向けの契約を保つため)。 */
void *mdl_memcpy(void *dst, const void *src, size_t n);

/* ------------------------------------------------------------------ */
/* dense のバッチ軸ビットパック */
/* ------------------------------------------------------------------ */
/*
 * conv は「1ワード = 空間32画素」でパックしている。dense には空間が無いので
 * 同じ手が使えず、1ビットを1バイトで持っていた(difflogic 本家が推論時に
 * ビットを**バッチ軸**へ詰めるのと同じ問題)。
 *
 * バッチ軸パックの並びは `[width][ceil(batch/32)]`:
 *   語 (i, b) のビット k = 画像 b*32+k の特徴 i
 *
 * これで
 *   ・LUT が32画像を1命令で評価できる(真理値表の最小項の論理和)
 *   ・**重みと添字表が32画像で1回しか読まれない**(k6000 は表が422KBあり、
 *     1枚ずつ回すと毎回L1から溢れる。ここが効く)
 */

/** 1枚あたり width ビットの 0/1 バイト列 batch 枚 → [width][ceil(batch/32)]。 */
void mdl_pack_batch(const uint8_t *inputs, uint32_t *x, int32_t width,
                    int32_t batch);

/** バッチ軸パックのまま LUT 層を評価する。n_inputs <= 5。
 * 真理値表はゲートあたり1語の並び(ビットスライス表現ではない。
 *  こちらは「1ゲートの表 × 語ループ」なので組み直すと7%遅い)。 */
void mdl_lut_layer_packed(const uint32_t *x, uint32_t *out, int32_t n_inputs,
                          int32_t out_width, const int32_t *indices,
                          const uint32_t *truth_tables, int32_t bwords);

/** mdl_lut_layer_packed の添字uint16_t版。in_width<=65536のときだけ
 * 選ばれる。packed teacher forcing(RDDLGNのM-layers)の主要経路。*/
void mdl_lut_layer_packed_u16(const uint32_t *x, uint32_t *out,
                              int32_t n_inputs, int32_t out_width,
                              const uint16_t *indices,
                              const uint32_t *truth_tables, int32_t bwords);

/** バッチ軸パックのまま group_sum する。scores は [batch][k]。 */
void mdl_group_sum_packed(const uint32_t *x, int32_t *scores, int32_t in_width,
                          int32_t k, int32_t batch, int32_t bwords);


/* ------------------------------------------------------------------ */
/* 前処理(画像 u8 → ビットプレーン)                                   */
/* ------------------------------------------------------------------ */

/* 記述子1チャネルぶんのワード数。deploy/preproc_desc.py と同一。 */
#define MDL_PREPROC_STRIDE 16

#define MDL_PREPROC_OP_THERMO 0
#define MDL_PREPROC_OP_EDGE 1
#define MDL_PREPROC_OP_CURV 2
#define MDL_PREPROC_OP_CHROMA 3
#define MDL_PREPROC_OP_SAT 4
#define MDL_PREPROC_OP_THERMO_STEP 5

/** 記述子に従って画像を二値チャネルへ符号化し、直接ビットプレーンへ書く。
 *
 *  **浮動小数点演算は一切現れない**。入力が u8 なので画素値は f32(k/255) の
 *  256通りしかなく、I(k) = f32(k/255)*2^31 が厳密な整数になることを使って、
 *  edge/curv/chroma/彩度の比較を int64 の整数比較へ落としてある
 *  (閾値側の cut は記述子に焼かれている)。thermometer は256ビットLUT、
 *  単調な場合は「k >= k0」の u8 比較1個に畳まれている。
 *
 *  image は [c_img][h][w] の u8(NCHW)。out は [channels][h][wpr]。
 *  境界は replicate。 */
void mdl_preproc_exec(const uint8_t *image, int32_t c_img, int32_t h, int32_t w,
                      const int32_t *desc, int32_t channels,
                      const uint32_t *qval, uint32_t *out);

/** ch_start から ch_stride 飛ばしのチャネルだけを符号化する。
 *
 *  チャネル同士は完全に独立(同じ画像を読み、別々の平面へ書く)なので、
 *  **飛び飛びの担当を別スレッドから同時に呼んでよい**(出力平面が
 *  重ならない)。実測で detface の推論時間の大半は前処理なので、
 *  1枚のレイテンシを詰めるならここを割るのが一番効く。
 *
 *  連続範囲ではなく**ストライド**にしてあるのは負荷分散のため。チャネルは
 *  種別ごとに並んでいて費用が18倍も違う(thermometer 0.14 に対し CURV は
 *  2.29 ms/100ch)ので、連続に割ると最後のスレッドに高価な種別が集中する
 *  (実測でストライドに変えて8スレッドが 1.81倍 → 2.4倍 になった)。 */
void mdl_preproc_exec_range(const uint8_t *image, int32_t c_img, int32_t h,
                            int32_t w, const int32_t *desc, int32_t ch_start,
                            int32_t ch_stride, int32_t channels,
                            const uint32_t *qval, uint32_t *out);

/* ------------------------------------------------------------------ */
/* 読み出し                                                            */
/* ------------------------------------------------------------------ */

/** スコア列の argmax(同点は先勝ち。TS側 argmax と同じ規約)。 */
int32_t mdl_argmax(const int32_t *scores, int32_t n);

/** 画素ごとの argmax(セグメンテーション)。
 *  scores は [n_out][h*w]、labels は [h*w] に書く。 */
void mdl_argmax_spatial(const int32_t *scores, int32_t n_out, int32_t n_pixels,
                        uint8_t *labels);

#ifdef __cplusplus
}
#endif

#endif /* MDL_KERNELS_H */
