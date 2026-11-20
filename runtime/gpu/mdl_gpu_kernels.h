/* griffith-logic — GPUカーネル本体(CUDA / Metal 共通のソース)
 *
 * CUDA と Metal で**同一の式**を使うため、修飾子だけをマクロで吸収して
 * 1本のソースを両方から include する。ここを分けると片方だけ直して
 * ビット不一致を作り込むので、意図的に1ファイルに閉じている。
 *
 * 呼び出し側が定義しておくマクロ:
 *   MDL_DEV      デバイス関数の修飾子(CUDA: __device__ / Metal: なし)
 *   MDL_GLOBAL   グローバル(バッファ)ポインタ修飾子(Metal: device / CUDA: なし)
 *   MDL_CONST    読み取り専用ポインタ修飾子(Metal: constant / CUDA: const)
 *   MDL_LOCAL    スレッドローカル(レジスタ)配列へのポインタ修飾子
 *                (Metal: thread / CUDA: なし)。追加。MSLはポインタ引数に
 *                アドレス空間修飾子を必須で要求する(CUDAは generic pointer
 *                なので省略できる)ため、呼び出し側のローカル配列を指す
 *                引数だけこの修飾子が要る。
 *
 * 並列化の考え方(WGSL版 webgpuBitplane.ts と同型):
 *   ビットプレーンの1ワード = 32空間位置。ほぼ全てのopが「出力ワード1個」
 *   単位で独立なので、1スレッド=1出力ワードで素直に並列化できる。
 *   conv の命令列だけは「同じワード位置の列を命令順に舐める」形になり、
 *   スレッド間の依存が無い(スレッド i はスクラッチの列 i しか触らない)。
 */
#ifndef MDL_GPU_KERNELS_H
#define MDL_GPU_KERNELS_H

/* ocごとの命令列分割(conv2dのGPU並列度を(batch,i)から(batch,oc,i)へ
 * 拡張する)のスレッドローカルレジスタ配列サイズ。conv_exec_split の
 * 各スレッド(=1つのoc)は自分専用の `unsigned int slots[MDL_CONV_SPLIT_MAX_SLOTS]`
 * を持ち、命令列縮約後の中間値をそこへ置く(全oc結合のレジスタ平面を
 * 共有スクラッチに置く旧経路とは違い、スレッド間の往復が無い)。
 *
 * Python側 `griffith_logic/deploy/layout.py::CONV_SPLIT_MAX_SLOTS` と
 * **手で**同じ値に同期させること(コンパイル時定数なのでここから直接
 * 読めない。ズレるとPython側が「収まる」と判定したのにC/CUDA/Metalの
 * 配列があふれる)。実測(seg/det系4モデルの全conv2d
 * ノード): oc分割後のmax_slots_per_ocは2〜5(全oc結合のn_slots1〜6より
 * 小さいか同じ — 各ocの依存閉包は全体より小さい部分グラフなので当然)。
 * 余裕を見て8。これを超えるモデルは `build_conv_layout_split` が生成時に
 * ValueError で落とす(未検証のレジスタ数を黙って使わない)。 */
#define MDL_CONV_SPLIT_MAX_SLOTS 8

/* ---- 2入力ゲート(CPU版 mdl_gate2 と同一の16式) ----
 *実測(scripts/gpu_calib/gate2_ab.mm、Metalネイティブ)で無分岐マスク式が
 * switch比1.22〜1.72倍速(32/1Mスレッド両方、低並列で効果大)と確認できたため
 * この形に置換。標準的な2入力LUTの最小項展開:
 *   m_i = 0u - ((tt>>i)&1u)  (ttのbit iが1なら全ビット1、0なら全ビット0)
 *   bit0=(a=0,b=0) bit1=(a=1,b=0) bit2=(a=0,b=1) bit3=(a=1,b=1)
 * switch版との出力完全一致は gate2_ab.mm 側のA/Bハーネスで検査済み。 */
MDL_DEV inline unsigned int mdl_g_gate2(unsigned int tt, unsigned int a,
                                        unsigned int b) {
  const unsigned int m0 = 0u - ((tt >> 0) & 1u);
  const unsigned int m1 = 0u - ((tt >> 1) & 1u);
  const unsigned int m2 = 0u - ((tt >> 2) & 1u);
  const unsigned int m3 = 0u - ((tt >> 3) & 1u);
  const unsigned int na = ~a, nb = ~b;
  return (m0 & na & nb) | (m1 & a & nb) | (m2 & na & b) | (m3 & a & b);
}


/* ---- 64bit符号付き整数のエミュレーション(二の補数、lo/hiの2語) ----
 * Metal Shading Language には64bit整数が無い。CURV の D は値域が
 * [-2^32, 2^32] で34bit必要なので、WGSL版(gpuPreproc.ts の I64_WGSL)と
 * 同じく2語で持つ。使うのは加算・符号反転・比較だけ。
 * CUDA/CPU でも同じコードを通すので、経路による差が出ない。 */
struct mdl_i64 { unsigned int lo; unsigned int hi; };

MDL_DEV inline mdl_i64 mdl_i64_make(unsigned int v) {
  mdl_i64 r; r.lo = v; r.hi = 0u; return r;
}

MDL_DEV inline mdl_i64 mdl_i64_add(mdl_i64 a, mdl_i64 b) {
  mdl_i64 r;
  r.lo = a.lo + b.lo;
  r.hi = a.hi + b.hi + ((r.lo < a.lo) ? 1u : 0u);
  return r;
}

MDL_DEV inline mdl_i64 mdl_i64_neg(mdl_i64 a) {
  mdl_i64 r;
  r.lo = ~a.lo + 1u;
  r.hi = ~a.hi + ((r.lo == 0u) ? 1u : 0u);
  return r;
}

MDL_DEV inline mdl_i64 mdl_i64_sub(mdl_i64 a, mdl_i64 b) {
  return mdl_i64_add(a, mdl_i64_neg(b));
}

/* a >= b(符号付き) */
MDL_DEV inline bool mdl_i64_ge(mdl_i64 a, mdl_i64 b) {
  const int ah = (int)a.hi;
  const int bh = (int)b.hi;
  if (ah != bh) return ah > bh;
  return a.lo >= b.lo;
}

MDL_DEV inline int mdl_g_clamp(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

MDL_DEV inline unsigned int mdl_g_tail_mask(int width) {
  int r = width & 31;
  return r == 0 ? 0xffffffffu : ((unsigned int)1 << r) - 1u;
}

/* ---- 共有シフト平面(高速路: stride=1 かつ w_out==w) ----
 * スレッド1個 = 出力平面のワード1個。CPU版の直線ループを「自分のワードが
 * どの入力ワードから来るか」の式に読み替えたもの(値は同一)。 */
MDL_DEV inline unsigned int mdl_g_shift_word(
    MDL_GLOBAL const unsigned int *src, int src_base,
    int h, int wpr, int p_h, int s, int idx) {
  const int row = idx / wpr;            /* 共有平面の行(0..h+2*p_h-1) */
  const int j = idx - row * wpr;
  const int y = row - p_h;              /* 対応する入力行 */
  if (y < 0 || y >= h) return 0u;       /* 上下のゼロパディング */
  {
    const int base = src_base + y * wpr;
    const int word_shift = s >> 5;      /* 算術シフト(負でも床) */
    const int bit_shift = s - word_shift * 32;
    const int j0 = j + word_shift;
    const unsigned int lo = (j0 >= 0 && j0 < wpr) ? src[base + j0] : 0u;
    if (bit_shift == 0) return lo;
    {
      const int j1 = j0 + 1;
      const unsigned int hi = (j1 >= 0 && j1 < wpr) ? src[base + j1] : 0u;
      return (lo >> bit_shift) | (hi << (32 - bit_shift));
    }
  }
}

/* ---- 汎用タップ平面(stride>1 / w_out!=w)。1スレッド=出力ワード1個 ---- */
MDL_DEV inline unsigned int mdl_g_tap_word_strided(
    MDL_GLOBAL const unsigned int *src, int src_base,
    int h, int w, int wpr_in, int dy, int dx,
    int w_out, int wpr_out, int s_h, int s_w, int p_h, int p_w, int idx) {
  const int oy = idx / wpr_out;
  const int j = idx - oy * wpr_out;
  const int y = oy * s_h - p_h + dy;
  unsigned int acc = 0u;
  int i;
  if (y < 0 || y >= h) return 0u;
  for (i = 0; i < 32; i++) {
    const int ox = j * 32 + i;
    int xx;
    if (ox >= w_out) break;
    xx = ox * s_w - p_w + dx;
    if (xx < 0 || xx >= w) continue;
    if ((src[src_base + y * wpr_in + (xx >> 5)] >> (xx & 31)) & 1u) {
      acc |= (unsigned int)1 << i;
    }
  }
  return acc;
}

/* ---- or_pool2d(2x2, stride2, pad0) ---- */
MDL_DEV inline unsigned int mdl_g_compress_even32(unsigned int v) {
  unsigned int x = v & 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

MDL_DEV inline unsigned int mdl_g_or_pool_word(
    MDL_GLOBAL const unsigned int *in, int h, int w, int wpr,
    int h_out, int w_out, int wpr_out, int idx) {
  const int plane_out = h_out * wpr_out;
  const int ch = idx / plane_out;
  const int rem = idx - ch * plane_out;
  const int oy = rem / wpr_out;
  const int wo = rem - oy * wpr_out;
  MDL_GLOBAL const unsigned int *ra = in + (ch * h + 2 * oy) * wpr;
  MDL_GLOBAL const unsigned int *rb = ra + wpr;
  unsigned int lo, hi = 0u;
  int j = 2 * wo;
  {
    const unsigned int t0 = (j < wpr) ? (ra[j] | rb[j]) : 0u;
    const unsigned int t1 = ((j + 1) < wpr) ? (ra[j + 1] | rb[j + 1]) : 0u;
    lo = t0 | (t0 >> 1) | (t1 << 31);
  }
  j = 2 * wo + 1;
  if (j < wpr) {
    const unsigned int t0 = ra[j] | rb[j];
    const unsigned int t1 = ((j + 1) < wpr) ? (ra[j + 1] | rb[j + 1]) : 0u;
    hi = t0 | (t0 >> 1) | (t1 << 31);
  }
  {
    unsigned int v = mdl_g_compress_even32(lo)
                   | (mdl_g_compress_even32(hi) << 16);
    if (wo == wpr_out - 1) v &= mdl_g_tail_mask(w_out);
    return v;
  }
}

/* ---- upsample2d(scale=2, nearest) ---- */
MDL_DEV inline unsigned int mdl_g_spread_even16(unsigned int v) {
  unsigned int x = v & 0xffffu;
  x = (x | (x << 8)) & 0x00ff00ffu;
  x = (x | (x << 4)) & 0x0f0f0f0fu;
  x = (x | (x << 2)) & 0x33333333u;
  x = (x | (x << 1)) & 0x55555555u;
  return x;
}

MDL_DEV inline unsigned int mdl_g_upsample_word(
    MDL_GLOBAL const unsigned int *in, int h, int w, int wpr,
    int wpr_out, int idx) {
  const int h_out = h * 2;
  const int w_out = w * 2;
  const int plane_out = h_out * wpr_out;
  const int ch = idx / plane_out;
  const int rem = idx - ch * plane_out;
  const int oy = rem / wpr_out;
  const int ow = rem - oy * wpr_out;
  const int si = ow >> 1;
  const unsigned int sw = (si < wpr) ? in[(ch * h + (oy >> 1)) * wpr + si] : 0u;
  const unsigned int part = ((ow & 1) == 0) ? (sw & 0xffffu)
                                            : ((sw >> 16) & 0xffffu);
  const unsigned int even = mdl_g_spread_even16(part);
  unsigned int v = even | (even << 1);
  if (ow == wpr_out - 1) v &= mdl_g_tail_mask(w_out);
  return v;
}

/* ---- SpatialGroupSum(permute[0,2,3,1] + group_sum の融合形) ----
 * 1スレッド = 出力の1ワード分の空間位置(32画素)。ビットスライスの
 * carry-save 加算で縦方向に足し、最後に32個のスコアへ展開する。 */
MDL_DEV inline void mdl_g_spatial_group_sum_word(
    MDL_GLOBAL const unsigned int *in, MDL_GLOBAL int *scores,
    int n_out, int gs, int h, int w, int wpr, int n_planes, int idx) {
  const int rows = h * wpr;
  const int o = idx / rows;
  const int rem = idx - o * rows;
  const int y = rem / wpr;
  const int j = rem - y * wpr;
  const int ch_words = h * wpr;
  unsigned int planes[32];              /* n_planes <= 32 */
  const unsigned int mask = (j == wpr - 1) ? mdl_g_tail_mask(w) : 0xffffffffu;
  int g, p, i;

  for (p = 0; p < n_planes; p++) planes[p] = 0u;
  for (g = 0; g < gs; g++) {
    unsigned int carry = in[(o * gs + g) * ch_words + y * wpr + j];
    for (p = 0; carry != 0u && p < n_planes; p++) {
      const unsigned int t = planes[p] & carry;
      planes[p] ^= carry;
      carry = t;
    }
  }
  for (i = 0; i < 32; i++) {
    const int x = j * 32 + i;
    int v = 0;
    if (((mask >> i) & 1u) == 0u) break;
    for (p = 0; p < n_planes; p++) {
      v |= (int)((planes[p] >> i) & 1u) << p;
    }
    scores[(o * h + y) * w + x] = v;
  }
}

/* ---- SpatialGroupSum: 並列縮約版 ----
 * 元の mdl_g_spatial_group_sum_word は「1スレッド=出力ワード1個、gs方向は
 * 直列」で、n_out*h*wpr が小さいモデルではGPUがほぼ空回りする(実測例:
 * det_face_lut_k256 先頭段は n_out=1,gs=256,h=32,w=32 → スレッド32本のみ、
 * Metal GPU時間の39%を占めていた)。ここでは gs 方向を R レーンに分割し、
 * 各レーンが自分のチャンク(g_lo..g_hi)だけをCSAで縮約して32画素ぶんの
 * 部分和(整数。ビットプレーンではなく既にデコード済みの値)を返す。
 * 整数加算は結合則・交換則を持つので、呼び出し側がスレッドグループ共有
 * メモリでRレーンぶん単純に総和すれば元(1レーン版)と同じ整数値になる
 * (浮動小数点のような加算順序依存が無いので、これはビット厳密に保証できる)。
 * R=1(g_lo=0,g_hi=gs)なら mdl_g_spatial_group_sum_word と完全に同じ計算。
 * 呼び出し側(mdl_k_sgs_r、CUDA/Metal共通の外殻はバリア構文が違うので
 * 各backendのエントリファイルに書く)が縮約と書き出しを担う。 */
MDL_DEV inline void mdl_g_spatial_group_sum_partial(
    MDL_GLOBAL const unsigned int *in, int gs, int h, int w, int wpr,
    int n_planes, int o, int y, int j, int g_lo, int g_hi,
    MDL_LOCAL int *out32) {
  const int ch_words = h * wpr;
  unsigned int planes[32];              /* n_planes <= 32 */
  const unsigned int mask = (j == wpr - 1) ? mdl_g_tail_mask(w) : 0xffffffffu;
  int p, g, i;

  for (p = 0; p < n_planes; p++) planes[p] = 0u;
  for (g = g_lo; g < g_hi; g++) {
    unsigned int carry = in[(o * gs + g) * ch_words + y * wpr + j];
    for (p = 0; carry != 0u && p < n_planes; p++) {
      const unsigned int t = planes[p] & carry;
      planes[p] ^= carry;
      carry = t;
    }
  }
  for (i = 0; i < 32; i++) {
    int v = 0;
    if (((mask >> i) & 1u) == 0u) { out32[i] = 0; continue; }
    for (p = 0; p < n_planes; p++) v |= (int)((planes[p] >> i) & 1u) << p;
    out32[i] = v;
  }
}

/* Rレーンぶんの部分和(スレッドグループ共有メモリで総和済み)を書き出す。
 * 境界チェックは mdl_g_spatial_group_sum_word の末尾ループと同一
 * (w が32の倍数でない最終ワードだけ打ち切る)。sum32 は呼び出し側
 * スレッドのローカル配列(縮約済みの32整数)。 */
MDL_DEV inline void mdl_g_spatial_group_sum_finish(
    MDL_GLOBAL int *scores, int h, int w, int wpr, int o, int y, int j,
    MDL_LOCAL const int *sum32) {
  const unsigned int mask = (j == wpr - 1) ? mdl_g_tail_mask(w) : 0xffffffffu;
  int i;
  for (i = 0; i < 32; i++) {
    const int x = j * 32 + i;
    if (((mask >> i) & 1u) == 0u) break;
    scores[(o * h + y) * w + x] = sum32[i];
  }
}

/* ---- 前処理(画像u8 → ビットプレーン) ----
 * 1スレッド = 出力ワード1個(32画素)。しきい値表は生成時に焼いてあるので
 * デバイス側に64bit演算も浮動小数点も出てこない。 */
#define MDL_G_OP_THERMO 0
#define MDL_G_OP_EDGE 1
#define MDL_G_OP_CURV 2
#define MDL_G_OP_CHROMA 3
#define MDL_G_OP_SAT 4
#define MDL_G_OP_THERMO_STEP 5
#define MDL_G_DESC_STRIDE 16

MDL_DEV inline unsigned int mdl_g_preproc_word(
    MDL_GLOBAL const unsigned char *image, int c_img, int h, int w, int wpr,
    MDL_CONST int *desc, MDL_CONST short *thr, MDL_CONST unsigned int *qval,
    int idx) {
  const int rows = h * wpr;
  const int ch = idx / rows;
  const int rem = idx - ch * rows;
  const int y = rem / wpr;
  const int j = rem - y * wpr;
  MDL_CONST int *d = desc + ch * MDL_G_DESC_STRIDE;
  const int op = d[0];
  const int ca = d[1], cb = d[2];
  const int dy = d[3], dx = d[4];
  const int dir_le = d[7];
  MDL_CONST short *t = thr + ch * 256;
  const int plane_px = h * w;
  const int x0 = j * 32;
  const int n = (w - x0) < 32 ? (w - x0) : 32;
  unsigned int acc = 0u;
  int i;

  if (op == MDL_G_OP_THERMO_STEP) {
    for (i = 0; i < n; i++) {
      acc |= (unsigned int)(image[ca * plane_px + y * w + x0 + i]
                            >= (unsigned char)cb) << i;
    }
    return acc;
  }
  if (op == MDL_G_OP_THERMO) {
    for (i = 0; i < n; i++) {
      const unsigned int k = image[ca * plane_px + y * w + x0 + i];
      acc |= (((unsigned int)d[8 + (k >> 5)] >> (k & 31u)) & 1u) << i;
    }
    return acc;
  }
  if (op == MDL_G_OP_CURV) {
    const int yp = mdl_g_clamp(y + dy, 0, h - 1);
    const int yn = mdl_g_clamp(y - dy, 0, h - 1);
    for (i = 0; i < n; i++) {
      const int x = x0 + i;
      const int xp = mdl_g_clamp(x + dx, 0, w - 1);
      const int xn = mdl_g_clamp(x - dx, 0, w - 1);
      /* CURV は3項で34bit必要。Metalに64bit整数が無いので2語で持つ。 */
      mdl_i64 cut; mdl_i64 m2; mdl_i64 diff;
      cut.lo = (unsigned int)d[5]; cut.hi = (unsigned int)d[6];
      m2 = mdl_i64_make(qval[image[ca * plane_px + y * w + x]]);
      m2 = mdl_i64_add(m2, m2);
      diff = mdl_i64_sub(
          mdl_i64_add(mdl_i64_make(qval[image[ca * plane_px + yn * w + xn]]),
                      mdl_i64_make(qval[image[ca * plane_px + yp * w + xp]])),
          m2);
      acc |= (unsigned int)(dir_le ? mdl_i64_ge(cut, diff)
                                   : mdl_i64_ge(diff, cut)) << i;
    }
    return acc;
  }
  if (op == MDL_G_OP_SAT) {
    for (i = 0; i < n; i++) {
      unsigned char mx = 0u, mn = 255u;
      int c;
      for (c = 0; c < c_img; c++) {
        const unsigned char v = image[c * plane_px + y * w + x0 + i];
        if (v > mx) mx = v;
        if (v < mn) mn = v;
      }
      acc |= (unsigned int)(dir_le ? ((int)mx < (int)t[mn])
                                   : ((int)mx >= (int)t[mn])) << i;
    }
    return acc;
  }
  /* EDGE / CHROMA: しきい値表でu8比較1回に還元済み。 */
  {
    const int ya = (op == MDL_G_OP_EDGE) ? mdl_g_clamp(y + dy, 0, h - 1) : y;
    const int cha = (op == MDL_G_OP_EDGE) ? ca : ca;
    const int chb = (op == MDL_G_OP_EDGE) ? ca : cb;
    for (i = 0; i < n; i++) {
      const int x = x0 + i;
      const int xa = (op == MDL_G_OP_EDGE) ? mdl_g_clamp(x + dx, 0, w - 1) : x;
      const int a = image[cha * plane_px + ya * w + xa];
      const int tv = t[image[chb * plane_px + y * w + x]];
      acc |= (unsigned int)(dir_le ? (a < tv) : (a >= tv)) << i;
    }
    return acc;
  }
}

/* ---- 非パック層(dense系) ---- */
MDL_DEV inline unsigned char mdl_g_lut_out(
    MDL_GLOBAL const unsigned char *x, int n_inputs, int out_width,
    MDL_CONST int *indices, MDL_CONST unsigned int *tt, int o) {
  unsigned int addr = 0u;
  int j;
  for (j = 0; j < n_inputs; j++) {
    addr |= (unsigned int)x[indices[j * out_width + o]] << j;
  }
  return (unsigned char)((tt[o] >> addr) & 1u);
}

/* 添字uint16_t版。呼び出し側(cbackend/gpubackend)が
 * in_width(この層の入力幅)<=65536のときだけ選ぶ。式は mdl_g_lut_out と
 * 同一(indicesの型だけが違う)。`unsigned short` はCUDA/MSL両方で
 * 16bit符号無し整数として解釈が一致する。 */
MDL_DEV inline unsigned char mdl_g_lut_out_u16(
    MDL_GLOBAL const unsigned char *x, int n_inputs, int out_width,
    MDL_CONST unsigned short *indices, MDL_CONST unsigned int *tt, int o) {
  unsigned int addr = 0u;
  int j;
  for (j = 0; j < n_inputs; j++) {
    addr |= (unsigned int)x[indices[j * out_width + o]] << j;
  }
  return (unsigned char)((tt[o] >> addr) & 1u);
}

/* ---- 縮約済み dense 平坦ネットリスト(lut_flat) ----
 * CPU版 runtime/c/src/mdl_kernels.c::MDL_LUT_FLAT_BODY と同一の式。
 * GPU 向けには生成時に `deploy/flatlevels.py::levelize_lut_flat` が
 * ゲートを ASAP の段ごとに並べ替え、参照される入力だけを信号表へ詰め直して
 * ある(信号 0=定数0、1..n_in=入力 in_idx[u]、n_in+1..=段順のゲート)。
 * 同じ段のゲートは互いに依存しないので、段ごとに1回の起動(融合版では
 * 1回のバリア区間)で全ゲートを並列に評価できる。信号表は1枚 sig_stride
 * バイトの0/1バイト列。 */
MDL_DEV inline unsigned char mdl_g_lut_flat_gate(unsigned int tt,
                                                 unsigned int av,
                                                 unsigned int bv) {
  return (unsigned char)((tt >> ((bv << 1) | av)) & 1u);
}

/* 信号表の先頭(定数0 + 詰め直した入力)。u=0 は定数0。 */
MDL_DEV inline unsigned char mdl_g_lut_flat_init_elem(
    MDL_GLOBAL const unsigned char *xb, MDL_CONST int *in_idx, int u) {
  return u == 0 ? (unsigned char)0 : xb[in_idx[u - 1]];
}

/* 出力 i(極性つき参照 o[i] = 2*信号 + 極性)。 */
#define MDL_G_LUT_FLAT_OUT(sig, o, i) \
  ((unsigned char)((sig)[(o)[(i)] >> 1] ^ ((o)[(i)] & 1)))

/* lut_flat のバッチ軸パック版。信号表は u32 語 sig[信号*bwords + w]
 * (語 w のビット k = 枚 32w+k、CPU版 MDL_LUT_FLAT_PACKED_BODY と同一レイアウト)。 */
MDL_DEV inline unsigned int mdl_g_lut_flat_pk_init_word(
    MDL_GLOBAL const unsigned char *x, MDL_CONST int *in_idx, int in_width,
    int batch, int u, int w) {
  unsigned int word = 0u;
  int k;
  if (u == 0) return 0u;
  {
    const int col = in_idx[u - 1];
    const int s0 = w << 5;
    const int n = batch - s0 < 32 ? batch - s0 : 32;
    for (k = 0; k < n; k++) {
      word |= ((unsigned int)x[(s0 + k) * in_width + col] & 1u) << k;
    }
  }
  return word;
}

MDL_DEV inline unsigned char mdl_g_lut_flat_pk_out(
    MDL_GLOBAL const unsigned int *sig, MDL_CONST int *o, int bwords, int s,
    int i) {
  const int r = o[i];
  return (unsigned char)(((sig[(r >> 1) * bwords + (s >> 5)] >> (s & 31)) & 1u)
                         ^ (unsigned int)(r & 1));
}

MDL_DEV inline int mdl_g_group_sum_out(MDL_GLOBAL const unsigned char *x,
                                       int gsz, int g) {
  int s = 0, i;
  for (i = 0; i < gsz; i++) s += x[g * gsz + i];
  return s;
}

/* 並列縮約版のレーン単位部分和。group_sum は
 * `mdl_gpu_pick_r`(runtime/{cuda,metal}側に複製したホスト専用ヘルパ、
 * mdl_gpu_kernels.hはMetalのMSL文字列連結に巻き込まれるのでホスト専用
 * コードはここへ置かない)が選んだ R レーンへ gsz を分担する。R=1
 * (lane=0)なら mdl_g_group_sum_out と同じ値になる(整数加算は結合則を
 * 持つので、レーン分割と縮約の順序はどう選んでも最終値は変わらない)。 */
MDL_DEV inline int mdl_g_group_sum_partial(MDL_GLOBAL const unsigned char *x,
                                           int gsz, int g, int lane, int R) {
  int s = 0, i;
  for (i = lane; i < gsz; i += R) s += x[g * gsz + i];
  return s;
}

/* ---- dense のバッチ軸ビットプレーン化 ----
 *
 * 上の非パック層(mdl_g_lut_out等)は1スレッド=1ゲート・1bit=1byteで、
 * バッチはただの平坦展開(total=out_width*batch)にしかならず、ALU使用率が
 * batch=128でも8.5%止まりだった(タスク指示の実測)。CPU側には同じ
 * 変換(runtime/c/src/mdl_kernels.c::mdl_pack_batch/
 * mdl_lut_layer_packed/mdl_group_sum_packed)が既にあり実測9倍が出ている
 * ので、その式をそのままGPUの1スレッド=1語(32サンプル分)単位へ持ち込む。
 *
 * パックレイアウト(CPU版と同一): x[neuron * bwords + w]。語wはサンプル
 * 32w..32w+31、サンプルbのビットは 1<<(b&31)。bwords=ceil(batch/32)。
 *
 * 教訓(「1スレッド内で32個のスカラー値をマスクに畳む」はGPUで
 * 0.27〜0.5×に悪化)との違い: lut_dense_packed/group_sum_packedの本体は
 * **既にパック済みの語を読むだけ**(罠そのもの)。32個のスカラーを
 * 畳む処理が要るのは入力を作る mdl_g_pack_batch_word だけで、これは
 * 既存の mdl_k_pack(画像の空間パック、32画素/語)と同じ形の避けようが
 * ない1回限りの変換(パック後の全層はこの形を繰り返し使い回すので、
 * 償却されて層数ぶん薄まる)。 */

/* パック: 1スレッド = パック平面の1語(neuron i, word w)の**値**を計算する
 * (書き込み先アドレスはこの関数の外で計算すること)。呼び出し側は
 * total=width*bwords、idx=w*width+i で起動する(iを速い軸にすることで、
 * ワープ内の連続スレッドが同じレーンbについて隣接するx[b*width+i]を
 * 読む=コアレスアクセスになる)。
 *
 * **注意**: 戻り値の書き込み先は `out[idx]` ではなく `out[i*bwords+w]`
 * (このファイル冒頭のパックレイアウト規約、CPU版mdl_pack_batchと同一)。
 * idxの分解(w=idx/width、iが速い軸)は読み取り側のコアレスアクセスの
 * ためだけの都合で、書き込み先のneuron-majorレイアウトとは無関係
 * ─ `out[idx]`へそのまま書くとword-majorになってしまい、bwords==1
 * (batch<=32)のときだけ両者が偶然一致して壊れて見えない、という罠
 * (実装時に実際に踏んだ。bwords>=2の全ケースでdense_packedの
 * 出力がsimulateと不一致になっていた ─ tests/deploy/
 * test_metal_dense_packed.pyのbatch=33以降が検出)。呼び出し側
 * (mdl_k_pack_batch)は`i`/`w`を自前で再計算し、`out[i*bwords+w]`へ書く。 */
MDL_DEV inline unsigned int mdl_g_pack_batch_word(
    MDL_GLOBAL const unsigned char *x, int width, int batch, int idx) {
  const int w = idx / width;
  const int i = idx - w * width;
  const int base_b = w << 5;
  const int lanes = (batch - base_b < 32) ? (batch - base_b) : 32;
  unsigned int acc = 0u;
  int lane;
  for (lane = 0; lane < lanes; lane++) {
    acc |= (unsigned int)(x[(base_b + lane) * width + i] != 0u) << lane;
  }
  return acc;
}

/* lut_dense のバッチ軸パック版(1スレッド=1ゲート×1バッチ語)。式は
 * CPU版 MDL_LUT_LAYER_PACKED_BODY(mdl_kernels.c)と同一:
 *   n_inputs==2 は butterfly-select(3xor+2and/語)、3〜5は
 *   「立っている最小項の論理和」の汎用ループ。添字は int32/uint16_t の
 *   2バリアント(lut_dense_indices_fit_u16 判定を流用)。*/
MDL_DEV inline unsigned int mdl_g_lut_out_packed(
    MDL_GLOBAL const unsigned int *x, int n_inputs, int out_width,
    MDL_CONST int *indices, MDL_CONST unsigned int *tt, int bwords, int o,
    int w) {
  const unsigned int ttv = tt[o];
  if (n_inputs == 2) {
    const unsigned int t0 = (ttv & 1u) ? 0xffffffffu : 0u;
    const unsigned int t1 = (ttv & 2u) ? 0xffffffffu : 0u;
    const unsigned int t2 = (ttv & 4u) ? 0xffffffffu : 0u;
    const unsigned int t3 = (ttv & 8u) ? 0xffffffffu : 0u;
    const unsigned int a = x[indices[o] * bwords + w];
    const unsigned int b = x[indices[out_width + o] * bwords + w];
    const unsigned int u = t0 ^ (a & (t0 ^ t1));
    const unsigned int v = t2 ^ (a & (t2 ^ t3));
    return u ^ (b & (u ^ v));
  }
  {
    const int n_min = 1 << n_inputs;
    unsigned int v[5];
    unsigned int acc = 0u;
    int j, m;
    for (j = 0; j < n_inputs; j++) {
      v[j] = x[indices[j * out_width + o] * bwords + w];
    }
    for (m = 0; m < n_min; m++) {
      unsigned int term;
      if (((ttv >> m) & 1u) == 0u) continue;
      term = 0xffffffffu;
      for (j = 0; j < n_inputs; j++) {
        term &= ((m >> j) & 1) ? v[j] : ~v[j];
      }
      acc |= term;
      if (acc == 0xffffffffu) break;   /* 以降の最小項は結果を変えない */
    }
    return acc;
  }
}

/* 添字uint16_t版(同じ理由)。式は mdl_g_lut_out_packed と同一。*/
MDL_DEV inline unsigned int mdl_g_lut_out_packed_u16(
    MDL_GLOBAL const unsigned int *x, int n_inputs, int out_width,
    MDL_CONST unsigned short *indices, MDL_CONST unsigned int *tt,
    int bwords, int o, int w) {
  const unsigned int ttv = tt[o];
  if (n_inputs == 2) {
    const unsigned int t0 = (ttv & 1u) ? 0xffffffffu : 0u;
    const unsigned int t1 = (ttv & 2u) ? 0xffffffffu : 0u;
    const unsigned int t2 = (ttv & 4u) ? 0xffffffffu : 0u;
    const unsigned int t3 = (ttv & 8u) ? 0xffffffffu : 0u;
    const unsigned int a = x[indices[o] * bwords + w];
    const unsigned int b = x[indices[out_width + o] * bwords + w];
    const unsigned int u = t0 ^ (a & (t0 ^ t1));
    const unsigned int v = t2 ^ (a & (t2 ^ t3));
    return u ^ (b & (u ^ v));
  }
  {
    const int n_min = 1 << n_inputs;
    unsigned int v[5];
    unsigned int acc = 0u;
    int j, m;
    for (j = 0; j < n_inputs; j++) {
      v[j] = x[indices[j * out_width + o] * bwords + w];
    }
    for (m = 0; m < n_min; m++) {
      unsigned int term;
      if (((ttv >> m) & 1u) == 0u) continue;
      term = 0xffffffffu;
      for (j = 0; j < n_inputs; j++) {
        term &= ((m >> j) & 1) ? v[j] : ~v[j];
      }
      acc |= term;
      if (acc == 0xffffffffu) break;
    }
    return acc;
  }
}

/* group_sum のバッチ軸パック版(1スレッド=1出力グループ×1バッチ語)。
 * CPU版 mdl_group_sum_packedと同じ縦方向キャリーセーブ加算:
 * 32サンプルぶんのカウンタを同時に進め、最後に32個のint32スコアへ
 * 転置して書く。出力レイアウトは非パック版と同一
 * (scores[(sample)*k+g]、KIND_SCORES int32)なので、この層の後段に
 * アンパックは要らない(呼び出し側はこの関数の出力をそのまま
 * `mdl_metal_download`/`cudaMemcpy` で読める)。
 *
 * n_planes(カウンタに必要なビット面数=ceil(log2(gsz+1)))はホスト側で
 * 1回計算して渡す(`mdl_gpu_pick_r`まわりの他のパック版カーネルと同じ
 * 流儀。呼び出しのたびに全スレッドで再計算させない)。
 *
 * 並列度についての設計判断(実測により当初案から変更): 起動
 * スレッド数は k*bwords しかなく、dense系実モデル(har_dlgn_k600等、
 * k=6クラス)では batch=128(bwords=4)でもわずか24スレッド。
 * GPU時間の実測内訳(`mdl_metal_gpu_breakdown`、har_dlgn_k600 batch=128、
 * 30反復)でこの関数が**全体の78%**(9.33ms/11.93ms、次点の
 * lut_layer_packedは1.08ms)を占めることを確認した ─ 「lut_dense層群は
 * 常に十分な並列度を持つのでgroup_sumの並列度不足は全体のごく一部にしか
 * 効かない」という当初の見立て(v1設計時の予想)は実測で反証された。
 * そのため`mdl_g_group_sum_packed_partial`/`_finish`(下)で同型の
 * Rレーン縮約を追加する: gsz方向をRチャンクに分割し、各レーンが
 * 自分のチャンクだけをCSAして32サンプルぶんの部分和(整数、ビットプレーン
 * ではなく既にデコード済み)を返し、呼び出し側がthreadgroup共有メモリで
 * 単純に総和すれば元(R=1)と同じ整数値になる(整数加算は結合則を持つので
 * レーン分割の順序は最終値に影響しない ─ `mdl_g_spatial_group_sum_partial`
 * と同じ理由)。R=1(g_lo=0,g_hi=gsz)なら`mdl_g_group_sum_packed_word`と
 * 完全に同じ計算になる。 */
MDL_DEV inline void mdl_g_group_sum_packed_word(
    MDL_GLOBAL const unsigned int *x, MDL_GLOBAL int *scores, int gsz,
    int k, int batch, int bwords, int n_planes, int g, int w) {
  const int base_b = w << 5;
  const int lanes = (batch - base_b < 32) ? (batch - base_b) : 32;
  unsigned int planes[32];          /* n_planes <= 32(gsz < 2^32なので) */
  int p, i, l;

  for (p = 0; p < n_planes; p++) planes[p] = 0u;
  for (i = 0; i < gsz; i++) {
    unsigned int carry = x[(g * gsz + i) * bwords + w];
    for (p = 0; p < n_planes; p++) {
      const unsigned int t = planes[p] & carry;
      planes[p] ^= carry;
      carry = t;
    }
  }
  for (l = 0; l < lanes; l++) {
    int s = 0;
    for (p = 0; p < n_planes; p++) {
      s |= (int)((planes[p] >> l) & 1u) << p;
    }
    scores[(base_b + l) * k + g] = s;
  }
}

/* group_sum_packed の並列縮約版(同型)。レーン単位の部分和:
 * gszの[g_lo,g_hi)チャンクだけをCSAし、32サンプルぶんの整数部分和を
 * `out32`(呼び出し側のスレッドローカル配列)へ書く。`n_planes`は
 * このチャンクの最大値(g_hi-g_lo)に対するビット面数(ホスト側が
 * チャンク幅から計算して渡す ─ 全体のgszに対する面数より小さくて済む)。 */
MDL_DEV inline void mdl_g_group_sum_packed_partial(
    MDL_GLOBAL const unsigned int *x, int gsz, int bwords, int n_planes,
    int g, int w, int g_lo, int g_hi, MDL_LOCAL int *out32) {
  unsigned int planes[32];
  int p, i;

  for (p = 0; p < n_planes; p++) planes[p] = 0u;
  for (i = g_lo; i < g_hi; i++) {
    unsigned int carry = x[(g * gsz + i) * bwords + w];
    for (p = 0; p < n_planes; p++) {
      const unsigned int t = planes[p] & carry;
      planes[p] ^= carry;
      carry = t;
    }
  }
  for (i = 0; i < 32; i++) {
    int v = 0;
    for (p = 0; p < n_planes; p++) v |= (int)((planes[p] >> i) & 1u) << p;
    out32[i] = v;
  }
}

/* Rレーンぶんの部分和(スレッドグループ共有メモリで総和済み)を書き出す。
 * 境界チェックは要らない(group_sum_packedは空間タイルの端切れが無い ─
 * spatial_group_sumのmdl_g_spatial_group_sum_finishとの違い。batchの
 * 端数はlanes<32のガードだけで足りる、mdl_g_group_sum_packed_wordと同じ)。 */
MDL_DEV inline void mdl_g_group_sum_packed_finish(
    MDL_GLOBAL int *scores, int k, int batch, int g, int w,
    MDL_LOCAL const int *sum32) {
  const int base_b = w << 5;
  const int lanes = (batch - base_b < 32) ? (batch - base_b) : 32;
  int l;
  for (l = 0; l < lanes; l++) {
    scores[(base_b + l) * k + g] = sum32[l];
  }
}

/* ---- ストライド一括コピー ----
 * `deploy/gpubackend.py::copy_batched_ex` はバッチ(レーン)ぶんの独立した
 * D2Dコピーをホスト側forループでN回issueしていた ─ Metalは1回のコピー
 * (`mdl_metal_copy`、blitエンコーダ)ごとにcompute encoderをendEncodingして
 * blitを張り直すため、N回ぶんのエンコーダ切替コストがそのままかかる
 * (batch=1024なら1024回)。CUDA側もcudaMemcpyDeviceToDeviceをN回ホスト発行
 * する形は変わらない。ここでは1回のカーネル起動でバッチ全レーンぶんを
 * 一括して運ぶ ─ 1スレッド=1レーンの1バイト(total=n_bytes*batch)。
 * `src_stride=0`を渡すと全レーンが同じsrc範囲を読む(`copy_batched_ex`の
 * 既存の規約 ─ state_initの定数複製に使う ─ をそのまま維持する:
 * `lane*0=0`なのでどのレーンも`src_off+i`を読む)。 */
MDL_DEV inline void mdl_g_copy_strided_byte(
    MDL_GLOBAL const unsigned char *src, MDL_GLOBAL unsigned char *dst,
    int dst_off, int dst_stride, int src_off, int src_stride, int n_bytes,
    int idx) {
  const int lane = idx / n_bytes;
  const int i = idx - lane * n_bytes;
  dst[dst_off + lane * dst_stride + i] = src[src_off + lane * src_stride + i];
}

/* ---- RDDLGN自己回帰生成(decode)ハーネス専用カーネル ----
 * CPU版(runtime/c/src/mdl_rddlgn.c)のmdl_embed_lookup/mdl_rddlgn_argmaxと
 * 同じ式をGPU向けに1スレッド単位へ書き直したもの。IRのop語彙には無い
 * (argmax/表引きは論理回路で表せないblackbox相当。進捗記録
 * deploykit-plan.md節・節と同じ線引き)ので、
 * deploy/plan.pyのEMITTERS表には登録しない — CPU版のmdl_rddlgn.cが
 * cbackend.pyのEMITTERSの外に手書きされているのと対称の位置づけ。 */

/* 埋め込み表引き(バイト1個=ビット1個の非パック表、[vocab][d_emb])。
 * table はホスト側で bundle['embed_table']['bits'](u32 LSB-firstパック)を
 * 展開して作った0/1バイト表(折り畳み表と違い、GPU側は表引きの
 * たびに64bit演算や追加のアンパック計算を持ち込みたくないための選択—
 * 値はCPU版と同じ埋め込み表に由来するので中身は一致する)。 */
MDL_DEV inline unsigned char mdl_g_embed_lookup_out(
    MDL_GLOBAL const unsigned char *table, int d_emb, int token, int j) {
  return table[token * d_emb + j];
}

/* argmax(同点は先勝ち=strict >。runtime/c/src/mdl_kernels.c::mdl_argmaxと
 * runtime/c/src/mdl_rddlgn.c::mdl_rddlgn_argmaxと同じ規約)。 */
MDL_DEV inline int mdl_g_argmax_out(MDL_GLOBAL const int *scores, int width) {
  int best = 0, i;
  for (i = 1; i < width; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

#endif /* MDL_GPU_KERNELS_H */
