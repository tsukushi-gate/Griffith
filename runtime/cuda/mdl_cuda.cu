/* griffith-logic — CUDAランタイム
 *
 * カーネル本体は runtime/gpu/mdl_gpu_kernels.h(Metal と共有)。ここでは
 * CUDA の修飾子を与えてグリッド起動の殻をかぶせるだけにしてある。
 * 式を分けないので、CUDA と Metal と CPU の整数出力は同一になる。
 *
 * 並列化: 1スレッド = 出力ワード1個(= 32空間位置)。
 * conv の命令列だけは「スレッド i がスクラッチの列 i を命令順に舐める」形で、
 * スレッド間の依存が無い(WGSL版 webgpuBitplane.ts と同じ構造)。
 */
#include "mdl_cuda.h"

#define MDL_DEV __device__
#define MDL_GLOBAL
#define MDL_CONST const
#define MDL_LOCAL
#include "mdl_gpu_kernels.h"

#include <stdio.h>

#define MDL_CUDA_CHECK(expr)                                                  \
  do {                                                                        \
    cudaError_t err_ = (expr);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
              __FILE__, __LINE__);                                            \
      return -1;                                                              \
    }                                                                         \
  } while (0)

#define MDL_BLOCK 256
/* lut_flat 融合版の1ブロックのスレッド数(1ブロック=1枚)。*/
#define MDL_FLAT_FUSED_BLOCK 512
static inline int mdl_grid(int n) { return (n + MDL_BLOCK - 1) / MDL_BLOCK; }

/* ---- ストリーム・パイプライン化(Metal ミラー。詳細は
 * mdl_cuda.h のコメント参照)。
 *
 * `MDL_STREAM` は「現在選択中のストリーム」を返すマクロで、以下の
 * 全カーネル起動・cudaMemcpyAsync呼び出しがこれを4番目の起動引数
 * (/ストリーム引数)として使う。streams_init未実行(=n_streams<=1の
 * 生成物、従来どおりの経路)では常に0(レガシーデフォルトストリーム)を
 * 返すので、この変更は既存の非ストリーミング経路に対して**完全に
 * 無害**(`<<<grid, block>>>` と `<<<grid, block, 0, 0>>>` は同じ意味)。
 *
 * 関数本体(mdl_cuda_streams_init等)は本ファイル末尾のextern "C"ブロック
 * に置く(mdl_cuda.h で前方宣言済みなので、ここでのマクロ展開時点で
 * 呼び出せる)。 */
#define MDL_STREAM mdl_cuda_current_stream()

/* group_sum / spatial_group_sum の並列度不足対策。
 *
 * 1スレッド=1出力(group_sum)/1出力ワード(spatial_group_sum)のままだと、
 * har_dlgn_k6000_gumbel の group_sum は k=6 スレッドしか立たず1000要素を
 * 直列加算、det_face_lut_k256 の先頭 spatial_group_sum は32スレッドしか
 * 立たず256チャネルを直列CSA — どちらも実測でGPU時間の最大支配項だった
 * (P1再測定: har_k6000で72%、det_faceで39%)。
 *
 * ここでは「縮約対象(gsz / gs)をRレーンへ分担し、スレッドグループ共有
 * メモリで総和する」ための launch 幅Rを選ぶ。整数加算は結合則を持つので
 * レーン分割の仕方によらず最終値は変わらない(浮動小数点のような順序
 * 依存は無い)。既存の並列度(work_items)が既に target 相当あるなら
 * R=1(=旧カーネルをそのまま使う、変更ゼロ)を返す —
 * 「規模で切り替わる最適化はしきい値で自動判定する」方針
 * (.claude/skills/perf-hunt/SKILL.md)。
 *
 * ホスト専用の純粋な計算なので、MSL文字列連結に巻き込まれる
 * mdl_gpu_kernels.h には置かず、CUDA/Metal 双方の入口ファイルに
 * 小さく複製している(perf-huntスキルの言う「データの並べ替え」側の
 * コードで、GPU数式の同一性とは別軸)。 */
static inline int mdl_gpu_pick_r(int work_items, int elems_per_item,
                                 int target_threads, int max_r) {
  int r = 1;
  if (work_items <= 0 || elems_per_item <= 1) return 1;
  while (r * 2 <= elems_per_item && r * 2 <= max_r &&
        (long long)work_items * (r * 2) <= (long long)target_threads) {
    r *= 2;
  }
  return r;
}

#define MDL_GS_TARGET_THREADS 1024
#define MDL_GS_MAX_R 64

/* ------------------------------------------------------------------ */
/* カーネル                                                            */
/* ------------------------------------------------------------------ */

__global__ void mdl_k_pack(const unsigned char *src, unsigned int *dst,
                           int c, int h, int w, int wpr, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int rows = h * wpr;
    const int ch = idx / rows;
    const int rem = idx - ch * rows;
    const int y = rem / wpr;
    const int j = rem - y * wpr;
    const int x0 = j * 32;
    const int n = (w - x0) < 32 ? (w - x0) : 32;
    const unsigned char *row = src + ((size_t)ch * h + y) * w;
    unsigned int acc = 0u;
    int i;
    for (i = 0; i < n; i++) acc |= (unsigned int)(row[x0 + i] != 0) << i;
    dst[idx] = acc;
  }
}

__global__ void mdl_k_unpack(const unsigned int *src, unsigned char *dst,
                             int c, int h, int w, int wpr, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int x = idx % w;
    const int rest = idx / w;
    const int y = rest % h;
    const int ch = rest / h;
    dst[idx] = (unsigned char)
        ((src[((size_t)ch * h + y) * wpr + (x >> 5)] >> (x & 31)) & 1u);
  }
}

__global__ void mdl_k_preproc(const unsigned char *image, int c_img, int h,
                              int w, int wpr, const int *desc,
                              const short *thr, const unsigned int *qval,
                              unsigned int *out, int per, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int b = idx / per;              /* バッチ内の何枚目か */
    const int k = idx - b * per;
    out[idx] = mdl_g_preproc_word(image + (size_t)b * c_img * h * w,
                                  c_img, h, w, wpr, desc, thr, qval, k);
  }
}

/* conv: タップ平面プールの構築(1スレッド=平面の1ワード)。 */
__global__ void mdl_k_conv_planes(const unsigned int *in, unsigned int *scratch,
                                  mdl_cuda_conv_layout layout, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int per = layout.n_planes * layout.plane_stride;
    const int b = idx / per;              /* バッチ内の何枚目か */
    const int t = idx - b * per;
    const unsigned int *in_b = in + (size_t)b * layout.in_plane_words;
    unsigned int *sc = scratch + (size_t)b * layout.scratch_words;
    const int u = t / layout.plane_stride;
    const int k = t - u * layout.plane_stride;
    const int c_base = layout.plane_c[u] * layout.h * layout.wpr_in;
    unsigned int v;
    if (layout.use_fast_path) {
      v = mdl_g_shift_word(in_b, c_base, layout.h, layout.wpr_in, layout.p_h,
                           layout.plane_dx[u] - layout.p_w, k);
    } else {
      v = mdl_g_tap_word_strided(in_b, c_base, layout.h, layout.w,
                                 layout.wpr_in, layout.plane_dy[u],
                                 layout.plane_dx[u], layout.w_out,
                                 layout.wpr_out, layout.s_h, layout.s_w,
                                 layout.p_h, layout.p_w, k);
    }
    sc[layout.pool_base + t] = v;
  }
}

/* conv: 定数平面の初期化。 */
__global__ void mdl_k_conv_const(unsigned int *scratch, int tap_rows_words,
                                 int scratch_words, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int b = idx / tap_rows_words;
    const int i = idx - b * tap_rows_words;
    unsigned int *sc = scratch + (size_t)b * scratch_words;
    sc[i] = 0u;
    sc[tap_rows_words + i] = 0xffffffffu;
  }
}

/* conv: 命令列の実行と出力書き出し。スレッド i はスクラッチの列 i だけを
 * 触るので、命令間の依存はスレッド内で自然に守られる。 */
__global__ void mdl_k_conv_exec(unsigned int *scratch, unsigned int *out,
                                mdl_cuda_conv_layout layout, int total) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  {
    const int plane_words = layout.tap_rows_words;
    const int n = gid / plane_words;      /* バッチ内の何枚目か */
    const int i = gid - n * plane_words;
    unsigned int *sc = scratch + (size_t)n * layout.scratch_words;
    unsigned int *ou = out + (size_t)n * layout.out_plane_words;
    const unsigned int tail =
        ((i % layout.wpr_out) == layout.wpr_out - 1) ? layout.mask_out
                                                     : 0xffffffffu;
    int j, b;
    /* バケット0: 命令に依存しない出力(定数/タップ直結)。 */
    for (b = layout.write_start[0]; b < layout.write_start[1]; b++) {
      const int oc = layout.write_oc[b];
      ou[oc * plane_words + i] = sc[layout.oc_src_off[oc] + i] & tail;
    }
    for (j = 0; j < layout.n_instrs; j++) {
      const int *ins = layout.instr_flat + j * 4;
      sc[ins[3] + i] = mdl_g_gate2((unsigned int)ins[0], sc[ins[1] + i],
                                   sc[ins[2] + i]);
      for (b = layout.write_start[j + 1]; b < layout.write_start[j + 2]; b++) {
        const int oc = layout.write_oc[b];
        ou[oc * plane_words + i] = sc[layout.oc_src_off[oc] + i] & tail;
      }
    }
  }
}

/* conv: ocごとの命令列分割版。1スレッド=(batch, oc, ワード位置)で、
 * mdl_k_conv_exec(1スレッド=(batch, ワード位置)が全oc共通の命令列全体を
 * 直列実行)より並列度を out_channels 倍に増やす。命令オペランドは
 * `(value<<1)|is_reg` にパック済み: bit0=0なら共有スクラッチ(定数0/1平面+
 * タップ平面プール)の絶対ワードオフセット、bit0=1ならこのoc専用の
 * スレッドローカル配列(=レジスタ)`slots` の添字。各ocの依存閉包は
 * トポロジカル順の部分列なので、write_start/write_ocのような割込み
 * スケジュールは要らず、命令を順に実行して最後に oc_final_ref を
 * 読むだけでよい(Metal版 mdl_k_conv_exec_split、
 * layout.py::split_program_by_oc のコメント参照)。 */
__global__ void mdl_k_conv_exec_split(const unsigned int *scratch,
                                      unsigned int *out,
                                      mdl_cuda_conv_layout layout,
                                      const int *oc_instr_start,
                                      const int *instr_flat_split,
                                      const int *oc_final_ref, int total) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  {
    const int w = layout.tap_rows_words;    /* ocあたりのワード数 */
    const int n = gid / (layout.out_channels * w);   /* バッチ内の何枚目か */
    const int rem = gid - n * (layout.out_channels * w);
    const int oc = rem / w;
    const int i = rem - oc * w;
    const unsigned int *sc = scratch + (size_t)n * layout.scratch_words;
    unsigned int *ou = out + (size_t)n * layout.out_plane_words;
    const unsigned int tail =
        ((i % layout.wpr_out) == layout.wpr_out - 1) ? layout.mask_out
                                                     : 0xffffffffu;
    unsigned int slots[MDL_CONV_SPLIT_MAX_SLOTS];
    const int start = oc_instr_start[oc];
    const int end = oc_instr_start[oc + 1];
    int k;
    for (k = start; k < end; k++) {
      const int *ins = instr_flat_split + k * 4;
      const int ap = ins[1];
      const int bp = ins[2];
      const unsigned int av =
          (ap & 1) ? slots[ap >> 1] : sc[(ap >> 1) + i];
      const unsigned int bv =
          (bp & 1) ? slots[bp >> 1] : sc[(bp >> 1) + i];
      slots[ins[3]] = mdl_g_gate2((unsigned int)ins[0], av, bv);
    }
    {
      const int fr = oc_final_ref[oc];
      const unsigned int v = (fr & 1) ? slots[fr >> 1] : sc[(fr >> 1) + i];
      ou[oc * w + i] = v & tail;
    }
  }
}

__global__ void mdl_k_or_pool(const unsigned int *in, unsigned int *out,
                              int h, int w, int wpr, int h_out, int w_out,
                              int wpr_out, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] = mdl_g_or_pool_word(in, h, w, wpr, h_out, w_out, wpr_out, idx);
}

__global__ void mdl_k_upsample(const unsigned int *in, unsigned int *out,
                               int h, int w, int wpr, int wpr_out, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] = mdl_g_upsample_word(in, h, w, wpr, wpr_out, idx);
}

__global__ void mdl_k_permute(const unsigned int *in, unsigned int *out,
                              const int *order, int ch_words, int c_out,
                              int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int o = idx / ch_words;
    const int k = idx - o * ch_words;
    const int b = o / c_out;              /* バッチ内の何枚目か */
    const int oc = o - b * c_out;
    out[idx] = in[(size_t)(b * c_out + order[oc]) * ch_words + k];
  }
}

__global__ void mdl_k_sgs(const unsigned int *in, int *scores, int n_out,
                          int gs, int h, int w, int wpr, int n_planes,
                          int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  mdl_g_spatial_group_sum_word(in, scores, n_out, gs, h, w, wpr, n_planes, idx);
}

/* 並列縮約版。1ブロック=1出力ワード(o,y,j)、ブロック内のR
 * スレッド(レーン)が gs 方向をチャンク分担してCSAし、共有メモリで
 * 総和してから書き出す。mdl_cuda_spatial_group_sum が work_items
 * (=n_out*h*wpr)が小さいときだけこちらを選ぶ。 */
__global__ void mdl_k_sgs_r(const unsigned int *in, int *scores, int h,
                            int w, int wpr, int gs, int n_planes_chunk,
                            int chunk) {
  extern __shared__ int sdata[];        /* [R][32] */
  const int wi = blockIdx.x;
  const int lane = threadIdx.x;
  const int R = blockDim.x;
  const int rows = h * wpr;
  const int o = wi / rows;
  const int rem = wi - o * rows;
  const int y = rem / wpr;
  const int j = rem - y * wpr;
  const int g_lo = lane * chunk;
  const int g_hi = (g_lo + chunk < gs) ? (g_lo + chunk) : gs;
  int part[32];
  int i, stride;

  if (g_lo < gs) {
    mdl_g_spatial_group_sum_partial(in, gs, h, w, wpr, n_planes_chunk, o, y,
                                    j, g_lo, g_hi, part);
  } else {
    for (i = 0; i < 32; i++) part[i] = 0;
  }
  for (i = 0; i < 32; i++) sdata[lane * 32 + i] = part[i];
  __syncthreads();
  for (stride = R >> 1; stride > 0; stride >>= 1) {
    if (lane < stride) {
      for (i = 0; i < 32; i++) {
        sdata[lane * 32 + i] += sdata[(lane + stride) * 32 + i];
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    int sum32[32];
    for (i = 0; i < 32; i++) sum32[i] = sdata[i];
    mdl_g_spatial_group_sum_finish(scores, h, w, wpr, o, y, j, sum32);
  }
}

__global__ void mdl_k_lut_layer(const unsigned char *x, unsigned char *out,
                                int n_inputs, int out_width,
                                const int *indices, const unsigned int *tt,
                                int in_width, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int b = idx / out_width;        /* バッチ内の何枚目か */
    const int o = idx - b * out_width;
    out[idx] = mdl_g_lut_out(x + (size_t)b * in_width, n_inputs, out_width,
                             indices, tt, o);
  }
}

/* 添字uint16_t版。式は mdl_k_lut_layer と同一。*/
__global__ void mdl_k_lut_layer_u16(const unsigned char *x, unsigned char *out,
                                    int n_inputs, int out_width,
                                    const unsigned short *indices,
                                    const unsigned int *tt, int in_width,
                                    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int b = idx / out_width;        /* バッチ内の何枚目か */
    const int o = idx - b * out_width;
    out[idx] = mdl_g_lut_out_u16(x + (size_t)b * in_width, n_inputs, out_width,
                                 indices, tt, o);
  }
}

/* ---- lut_flat ----
 * 式は mdl_gpu_kernels.h の mdl_g_lut_flat_*(Metal版 mdl_k_lut_flat_* と同一)。
 * 表は `deploy/flatlevels.py` が段ごとに並べ替えたもの。 */
__global__ void mdl_k_lut_flat_init(const unsigned char *x, unsigned char *sig,
                                    const int *in_idx, int n_in, int in_width,
                                    int sig_stride, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int per = n_in + 1;
    const int b = idx / per;
    const int u = idx - b * per;
    sig[(size_t)b * sig_stride + u] =
        mdl_g_lut_flat_init_elem(x + (size_t)b * in_width, in_idx, u);
  }
}

__global__ void mdl_k_lut_flat_level(unsigned char *sig,
                                     const unsigned char *tt, const int *a,
                                     const int *b, int n_in, int sig_stride,
                                     int g0, int n, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int s = idx / n;
    const int g = g0 + (idx - s * n);
    unsigned char *sg = sig + (size_t)s * sig_stride;
    sg[1 + n_in + g] = mdl_g_lut_flat_gate(tt[g], sg[a[g]], sg[b[g]]);
  }
}

__global__ void mdl_k_lut_flat_out(const unsigned char *sig,
                                   unsigned char *out, const int *o,
                                   int out_width, int sig_stride, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int s = idx / out_width;
    const int i = idx - s * out_width;
    const unsigned char *sg = sig + (size_t)s * sig_stride;
    out[idx] = MDL_G_LUT_FLAT_OUT(sg, o, i);
  }
}

/* バッチ軸パック版: 信号表は u32 語 sig[信号*bw + w]。*/
__global__ void mdl_k_lut_flat_pk_init(const unsigned char *x,
                                       unsigned int *sig, const int *in_idx,
                                       int n_in, int in_width, int batch,
                                       int bw, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int u = idx / bw;
    const int w = idx - u * bw;
    sig[idx] = mdl_g_lut_flat_pk_init_word(x, in_idx, in_width, batch, u, w);
  }
}

__global__ void mdl_k_lut_flat_pk_level(unsigned int *sig,
                                        const unsigned char *tt, const int *a,
                                        const int *b, int n_in, int g0,
                                        int bw, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int j = idx / bw;
    const int w = idx - j * bw;
    const int g = g0 + j;
    sig[(size_t)(1 + n_in + g) * bw + w] =
        mdl_g_gate2(tt[g], sig[(size_t)a[g] * bw + w], sig[(size_t)b[g] * bw + w]);
  }
}

__global__ void mdl_k_lut_flat_pk_out(const unsigned int *sig,
                                      unsigned char *out, const int *o,
                                      int out_width, int bw, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int s = idx / out_width;
    const int i = idx - s * out_width;
    out[idx] = mdl_g_lut_flat_pk_out(sig, o, bw, s, i);
  }
}

/* 融合版: 1ブロック=1枚、信号表は動的共有メモリ(段の間だけ__syncthreads)。 */
__global__ void mdl_k_lut_flat_fused(const unsigned char *x,
                                     unsigned char *out,
                                     const unsigned char *tt, const int *a,
                                     const int *b, const int *o,
                                     const int *in_idx, const int *level_off,
                                     int n_in, int in_width, int out_width,
                                     int n_levels) {
  extern __shared__ unsigned char sg[];
  const int tid = threadIdx.x, tgs = blockDim.x, bid = blockIdx.x;
  const int base = 1 + n_in;
  const unsigned char *xb = x + (size_t)bid * in_width;
  int u, L, g, i;
  for (u = tid; u < base; u += tgs) sg[u] = mdl_g_lut_flat_init_elem(xb, in_idx, u);
  __syncthreads();
  for (L = 0; L < n_levels; L++) {
    const int g1 = level_off[L + 1];
    for (g = level_off[L] + tid; g < g1; g += tgs) {
      sg[base + g] = mdl_g_lut_flat_gate(tt[g], sg[a[g]], sg[b[g]]);
    }
    __syncthreads();
  }
  for (i = tid; i < out_width; i += tgs) {
    out[(size_t)bid * out_width + i] = MDL_G_LUT_FLAT_OUT(sg, o, i);
  }
}

__global__ void mdl_k_group_sum(const unsigned char *x, int *out, int gsz,
                                int k) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= k) return;
  out[g] = mdl_g_group_sum_out(x, gsz, g);
}

/* 並列縮約版。1ブロック=1出力グループ、ブロック内のRスレッドが
 * gsz を分担して部分和を出し、共有メモリで木構造に総和する。
 * har_dlgn_k6000_gumbel のような「クラス数k=6が極小、gsz=1000が極大」な
 * 形(k threads しか立たずGPU時間の72%を占めていた)向け。 */
__global__ void mdl_k_group_sum_r(const unsigned char *x, int *out, int gsz) {
  extern __shared__ int sdata[];        /* [R] */
  const int g = blockIdx.x;
  const int lane = threadIdx.x;
  const int R = blockDim.x;
  int stride;
  sdata[lane] = mdl_g_group_sum_partial(x, gsz, g, lane, R);
  __syncthreads();
  for (stride = R >> 1; stride > 0; stride >>= 1) {
    if (lane < stride) sdata[lane] += sdata[lane + stride];
    __syncthreads();
  }
  if (lane == 0) out[g] = sdata[0];
}

/* ---- dense のバッチ軸ビットプレーン化 ----
 * 計算式は mdl_gpu_kernels.h の mdl_g_pack_batch_word /
 * mdl_g_lut_out_packed[_u16] / mdl_g_group_sum_packed_word(Metalと共有)。
 * ここはCUDAのグリッド起動の殻だけ(Metal版 mdl_k_pack_batch/
 * mdl_k_lut_layer_packed[_u16]/mdl_k_group_sum_packed と対称)。 */

/* idxの分解(w=idx/width、iが速い軸)は読み取り側のコアレスアクセスの
 * ためだけの都合で、書き込み先はneuron-major(`out[i*bwords+w]`、
 * mdl_g_pack_batch_wordのコメント参照。Metal版mdl_k_pack_batchと対称)。 */
__global__ void mdl_k_pack_batch(const unsigned char *x, unsigned int *out,
                                 int width, int batch, int bwords,
                                 int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int w = idx / width;
    const int i = idx - w * width;
    out[i * bwords + w] = mdl_g_pack_batch_word(x, width, batch, idx);
  }
}

__global__ void mdl_k_lut_layer_packed(const unsigned int *x,
                                       unsigned int *out, int n_inputs,
                                       int out_width, const int *indices,
                                       const unsigned int *tt, int bwords,
                                       int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int o = idx / bwords;
    const int w = idx - o * bwords;
    out[idx] = mdl_g_lut_out_packed(x, n_inputs, out_width, indices, tt,
                                    bwords, o, w);
  }
}

/* 添字uint16_t版(同じ理由)。式は mdl_k_lut_layer_packed と同一。*/
__global__ void mdl_k_lut_layer_packed_u16(const unsigned int *x,
                                           unsigned int *out, int n_inputs,
                                           int out_width,
                                           const unsigned short *indices,
                                           const unsigned int *tt, int bwords,
                                           int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int o = idx / bwords;
    const int w = idx - o * bwords;
    out[idx] = mdl_g_lut_out_packed_u16(x, n_inputs, out_width, indices, tt,
                                        bwords, o, w);
  }
}

/* group_sum のバッチ軸パック版。出力は非パック版と同じint32レイアウト
 * (scores[sample*k+g])に直接書くので、この後段にアンパックは要らない。 */
__global__ void mdl_k_group_sum_packed(const unsigned int *x, int *scores,
                                       int gsz, int k, int batch, int bwords,
                                       int n_planes, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int g = idx / bwords;
    const int w = idx - g * bwords;
    mdl_g_group_sum_packed_word(x, scores, gsz, k, batch, bwords, n_planes, g,
                                w);
  }
}

/* group_sum_packed の並列縮約版(mdl_k_group_sum_r/mdl_k_sgs_rと同型、
 * Metal版mdl_k_group_sum_packed_rと対称)。実測(Metal実機、har_dlgn_k600、
 * batch=128)でmdl_k_group_sum_packedがGPU時間の78%を占めていた
 * (k*bwords=24スレッドしか立たないため)ことを受けて追加。1ブロック=
 * 1出力ワード(g,w)、ブロック内のRスレッド(レーン)がgszをチャンク分担
 * してCSAし、共有メモリで総和してから書き出す。 */
__global__ void mdl_k_group_sum_packed_r(const unsigned int *x, int *scores,
                                         int gsz, int k, int batch,
                                         int bwords, int n_planes_chunk,
                                         int chunk) {
  extern __shared__ int sdata[];        /* [R][32] */
  const int wi = blockIdx.x;
  const int lane = threadIdx.x;
  const int R = blockDim.x;
  const int gidx = wi / bwords;
  const int w = wi - gidx * bwords;
  const int g_lo = lane * chunk;
  const int g_hi = (g_lo + chunk < gsz) ? (g_lo + chunk) : gsz;
  int part[32];
  int i, stride;

  if (g_lo < gsz) {
    mdl_g_group_sum_packed_partial(x, gsz, bwords, n_planes_chunk, gidx, w,
                                   g_lo, g_hi, part);
  } else {
    for (i = 0; i < 32; i++) part[i] = 0;
  }
  for (i = 0; i < 32; i++) sdata[lane * 32 + i] = part[i];
  __syncthreads();
  for (stride = R >> 1; stride > 0; stride >>= 1) {
    if (lane < stride) {
      for (i = 0; i < 32; i++) {
        sdata[lane * 32 + i] += sdata[(lane + stride) * 32 + i];
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    int sum32[32];
    for (i = 0; i < 32; i++) sum32[i] = sdata[i];
    mdl_g_group_sum_packed_finish(scores, k, batch, gidx, w, sum32);
  }
}

/* ---- ストライド一括コピー ----
 * Metal版(mdl_metal.mm::mdl_metal_copy_strided)と対称。`copy_batched_ex`
 * がバッチぶん独立にcudaMemcpyDeviceToDeviceをホスト側forループでN回
 * 発行していたのを、1回のカーネル起動へ畳む(mdl_gpu_kernels.h::
 * mdl_g_copy_strided_byte参照、式はMetal版と同一)。 */
__global__ void mdl_k_copy_strided(const unsigned char *src,
                                   unsigned char *dst, int dst_off,
                                   int dst_stride, int src_off,
                                   int src_stride, int n_bytes, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  mdl_g_copy_strided_byte(src, dst, dst_off, dst_stride, src_off, src_stride,
                          n_bytes, idx);
}

/* ---- RDDLGN自己回帰生成(decode)ハーネス専用カーネル ----
 * gpubackend.pyのEMITTERSには載らない(IR opではない。
 * mdl_gpu_kernels.hのコメント参照)。呼び出しはdeploy/rddlgn_decode_gpu.py
 * が直接発行する。 */
__global__ void mdl_k_embed_lookup(const int *tokens, const unsigned char *table,
                                   unsigned char *out, int d_emb, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  {
    const int b = idx / d_emb;
    const int j = idx - b * d_emb;
    out[idx] = mdl_g_embed_lookup_out(table, d_emb, tokens[b], j);
  }
}

__global__ void mdl_k_argmax(const int *scores, int *tokens, int width,
                             int batch) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  tokens[b] = mdl_g_argmax_out(scores + b * width, width);
}

/* ------------------------------------------------------------------ */
/* ホスト側の薄いラッパ(生成コードから呼ぶ)                          */
/* ------------------------------------------------------------------ */

extern "C" {

int mdl_cuda_pack(const unsigned char *src, unsigned int *dst, int c, int h,
                  int w) {
  const int wpr = (w + 31) >> 5;
  const int total = c * h * wpr;
  mdl_k_pack<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(src, dst, c, h, w, wpr, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_unpack(const unsigned int *src, unsigned char *dst, int c, int h,
                    int w) {
  const int wpr = (w + 31) >> 5;
  const int total = c * h * w;
  mdl_k_unpack<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(src, dst, c, h, w, wpr, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_preproc(const unsigned char *image, int c_img, int h, int w,
                     const int *desc, int channels, const short *thr,
                     const unsigned int *qval, unsigned int *out, int batch) {
  const int wpr = (w + 31) >> 5;
  const int per = channels * h * wpr;
  const int total = per * batch;
  mdl_k_preproc<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(image, c_img, h, w, wpr, desc,
                                                thr, qval, out, per, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_conv_init(const mdl_cuda_conv_layout *layout,
                       unsigned int *scratch, int batch) {
  /* 定数平面(0 と 全1)を batch 枚ぶん埋める。 */
  const int total = layout->tap_rows_words * batch;
  mdl_k_conv_const<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      scratch, layout->tap_rows_words, layout->scratch_words, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  /* cudaDeviceSynchronize()を撤去。全カーネルは同一デフォルト
     ストリームに発行されるため発行順=実行順が保証され、後続の
     conv_planes/conv_execはこのカーネルの完了を自動的に待つ(GPU側で
     順序保証済み)。毎推論・毎convノードでホストとデバイスを同期する
     必要はなく、最終的な出力の可用性は呼び出し側の同期版
     cudaMemcpy(DeviceToHost)が保証する。エラー検出はcudaGetLastError()
     (非同期起動の失敗を拾う、同期を伴わない)で引き続き行う。 */
  return 0;
}

int mdl_cuda_conv_planes(const mdl_cuda_conv_layout *layout,
                         const unsigned int *in, unsigned int *scratch,
                         int batch) {
  const int planes_total = layout->n_planes * layout->plane_stride * batch;
  if (planes_total <= 0) return 0;
  mdl_k_conv_planes<<<mdl_grid(planes_total), MDL_BLOCK, 0, MDL_STREAM>>>(in, scratch,
                                                           *layout,
                                                           planes_total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_conv(const mdl_cuda_conv_layout *layout, const unsigned int *in,
                  unsigned int *scratch, unsigned int *out, int batch) {
  /* バッチはここで並列度そのものになる。conv_exec のスレッド数は
     画素数/32 しかなく(深い層で16〜32本)、単枚ではGPUが空回りしていた。 */
  const int exec_total = layout->tap_rows_words * batch;
  if (mdl_cuda_conv_planes(layout, in, scratch, batch) != 0) return -1;
  mdl_k_conv_exec<<<mdl_grid(exec_total), MDL_BLOCK, 0, MDL_STREAM>>>(
      scratch, out, *layout, exec_total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_conv_split(const mdl_cuda_conv_layout *layout,
                        const unsigned int *in, unsigned int *scratch,
                        unsigned int *out, const int *oc_instr_start,
                        const int *instr_flat_split, const int *oc_final_ref,
                        int batch) {
  /* タップ平面プールの構築(mdl_k_conv_planes)はcombined版と同じ関数を
     使い回す(幾何・タップは分割の影響を受けない共有データ)。 */
  const int exec_total = layout->out_channels * layout->tap_rows_words * batch;
  if (mdl_cuda_conv_planes(layout, in, scratch, batch) != 0) return -1;
  mdl_k_conv_exec_split<<<mdl_grid(exec_total), MDL_BLOCK, 0, MDL_STREAM>>>(
      scratch, out, *layout, oc_instr_start, instr_flat_split, oc_final_ref,
      exec_total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_or_pool(const unsigned int *in, unsigned int *out, int c, int h,
                     int w) {
  const int wpr = (w + 31) >> 5;
  const int h_out = ((h - 2) >> 1) + 1;
  const int w_out = ((w - 2) >> 1) + 1;
  const int wpr_out = (w_out + 31) >> 5;
  const int total = c * h_out * wpr_out;
  mdl_k_or_pool<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(in, out, h, w, wpr, h_out,
                                                w_out, wpr_out, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_upsample(const unsigned int *in, unsigned int *out, int c, int h,
                      int w) {
  const int wpr = (w + 31) >> 5;
  const int wpr_out = ((2 * w) + 31) >> 5;
  const int total = c * (2 * h) * wpr_out;
  mdl_k_upsample<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(in, out, h, w, wpr, wpr_out,
                                                 total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_permute(const unsigned int *in, unsigned int *out,
                     const int *order, int c_out, int h, int wpr, int batch) {
  const int ch_words = h * wpr;
  const int total = c_out * batch * ch_words;
  mdl_k_permute<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(in, out, order, ch_words,
                                                c_out, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_spatial_group_sum(const unsigned int *in, int *scores, int n_out,
                               int gs, int h, int w) {
  const int wpr = (w + 31) >> 5;
  int n_planes = 0, v = gs;
  const int total = n_out * h * wpr;
  while (v > 0) { n_planes++; v >>= 1; }
  {
    const int r = mdl_gpu_pick_r(total, gs, MDL_GS_TARGET_THREADS,
                                 MDL_GS_MAX_R);
    if (r <= 1) {
      mdl_k_sgs<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(in, scores, n_out, gs, h, w,
                                                wpr, n_planes, total);
    } else {
      const int chunk = (gs + r - 1) / r;
      int n_planes_chunk = 0;
      v = chunk;
      while (v > 0) { n_planes_chunk++; v >>= 1; }
      mdl_k_sgs_r<<<total, r, (size_t)r * 32 * sizeof(int), MDL_STREAM>>>(
          in, scores, h, w, wpr, gs, n_planes_chunk, chunk);
    }
  }
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_layer(const unsigned char *x, unsigned char *out,
                       int n_inputs, int out_width, const int *indices,
                       const unsigned int *tt, int in_width, int batch) {
  const int total = out_width * batch;
  mdl_k_lut_layer<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(x, out, n_inputs, out_width,
                                                  indices, tt, in_width,
                                                  total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_layer_u16(const unsigned char *x, unsigned char *out,
                           int n_inputs, int out_width,
                           const unsigned short *indices,
                           const unsigned int *tt, int in_width, int batch) {
  const int total = out_width * batch;
  mdl_k_lut_layer_u16<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(x, out, n_inputs,
                                                       out_width, indices, tt,
                                                       in_width, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_flat_init(const unsigned char *x, void *sig,
                           const int *in_idx, int n_in, int in_width,
                           int sig_stride, int batch) {
  const int total = (n_in + 1) * batch;
  mdl_k_lut_flat_init<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      x, (unsigned char *)sig, in_idx, n_in, in_width, sig_stride, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_flat_level(void *sig, const unsigned char *tt, const int *a,
                            const int *b, int n_in, int sig_stride, int g0,
                            int n, int batch) {
  const int total = n * batch;
  if (total <= 0) return 0;
  mdl_k_lut_flat_level<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      (unsigned char *)sig, tt, a, b, n_in, sig_stride, g0, n, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_flat_out(const void *sig, unsigned char *out, const int *o,
                          int out_width, int sig_stride, int batch) {
  const int total = out_width * batch;
  mdl_k_lut_flat_out<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      (const unsigned char *)sig, out, o, out_width, sig_stride, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_flat_pk_init(const unsigned char *x, void *sig,
                              const int *in_idx, int n_in, int in_width,
                              int batch) {
  const int bw = (batch + 31) >> 5;
  const int total = (n_in + 1) * bw;
  mdl_k_lut_flat_pk_init<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      x, (unsigned int *)sig, in_idx, n_in, in_width, batch, bw, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_flat_pk_level(void *sig, const unsigned char *tt,
                               const int *a, const int *b, int n_in, int g0,
                               int n, int batch) {
  const int bw = (batch + 31) >> 5;
  const int total = n * bw;
  if (total <= 0) return 0;
  mdl_k_lut_flat_pk_level<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      (unsigned int *)sig, tt, a, b, n_in, g0, bw, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_flat_pk_out(const void *sig, unsigned char *out,
                             const int *o, int out_width, int batch) {
  const int bw = (batch + 31) >> 5;
  const int total = out_width * batch;
  mdl_k_lut_flat_pk_out<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      (const unsigned int *)sig, out, o, out_width, bw, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_flat_fused(const unsigned char *x, unsigned char *out,
                            const unsigned char *tt, const int *a,
                            const int *b, const int *o, const int *in_idx,
                            const int *level_off, int n_in, int in_width,
                            int out_width, int n_levels, int sig_bytes,
                            int batch) {
  if (batch <= 0) return 0;
  mdl_k_lut_flat_fused<<<batch, MDL_FLAT_FUSED_BLOCK, (size_t)sig_bytes,
                         MDL_STREAM>>>(x, out, tt, a, b, o, in_idx, level_off,
                                       n_in, in_width, out_width, n_levels);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_group_sum(const unsigned char *x, int *out, int in_width, int k) {
  const int gsz = in_width / k;
  const int r = mdl_gpu_pick_r(k, gsz, MDL_GS_TARGET_THREADS, MDL_GS_MAX_R);
  if (r <= 1) {
    mdl_k_group_sum<<<mdl_grid(k), MDL_BLOCK, 0, MDL_STREAM>>>(x, out, gsz, k);
  } else {
    mdl_k_group_sum_r<<<k, r, (size_t)r * sizeof(int), MDL_STREAM>>>(x, out, gsz);
  }
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

/* ---- dense のバッチ軸ビットプレーン化 ----
 * `batch` 引数はいずれもこの呼び出しでの実行時サンプル数(既存の
 * バッチ透過カーネルの`batch`引数と同じ意味)。Metal版 mdl_metal_pack_batch/
 * mdl_metal_lut_layer_packed[_u16]/mdl_metal_group_sum_packed と対称。
 * **CUDA実機での実行は未検証**(このMacにCUDA実機/nvccが無いため。
 *タスク指示どおりローカルの構文検査まで)。*/

int mdl_cuda_pack_batch(const unsigned char *x, unsigned int *out, int width,
                        int batch) {
  const int bwords = (batch + 31) >> 5;
  const int total = width * bwords;
  mdl_k_pack_batch<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(x, out, width, batch,
                                                   bwords, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_layer_packed(const unsigned int *x, unsigned int *out,
                              int n_inputs, int out_width, const int *indices,
                              const unsigned int *tt, int batch) {
  const int bwords = (batch + 31) >> 5;
  const int total = out_width * bwords;
  mdl_k_lut_layer_packed<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      x, out, n_inputs, out_width, indices, tt, bwords, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_lut_layer_packed_u16(const unsigned int *x, unsigned int *out,
                                  int n_inputs, int out_width,
                                  const unsigned short *indices,
                                  const unsigned int *tt, int batch) {
  const int bwords = (batch + 31) >> 5;
  const int total = out_width * bwords;
  mdl_k_lut_layer_packed_u16<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      x, out, n_inputs, out_width, indices, tt, bwords, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

/* 並列度不足対策(Metal版mdl_metal_group_sum_packedと対称)。
 * work_items=k*bwordsが小さいモデルでは、既存のgroup_sum/
 * spatial_group_sumと同じ`mdl_gpu_pick_r`判定でRレーン縮約版を選ぶ。
 * r<=1(既に並列度が十分)なら従来のmdl_k_group_sum_packedのまま。 */
int mdl_cuda_group_sum_packed(const unsigned int *x, int *out, int in_width,
                              int k, int batch) {
  const int gsz = in_width / k;
  const int bwords = (batch + 31) >> 5;
  int n_planes = 0, v = gsz;
  while (v > 0) { n_planes++; v >>= 1; }
  if (n_planes == 0) n_planes = 1;
  {
    const int r = mdl_gpu_pick_r(k * bwords, gsz, MDL_GS_TARGET_THREADS,
                                 MDL_GS_MAX_R);
    if (r <= 1) {
      const int total = k * bwords;
      mdl_k_group_sum_packed<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
          x, out, gsz, k, batch, bwords, n_planes, total);
    } else {
      const int chunk = (gsz + r - 1) / r;
      int n_planes_chunk = 0;
      v = chunk;
      while (v > 0) { n_planes_chunk++; v >>= 1; }
      if (n_planes_chunk == 0) n_planes_chunk = 1;
      mdl_k_group_sum_packed_r<<<k * bwords, r, (size_t)r * 32 * sizeof(int), MDL_STREAM>>>(
          x, out, gsz, k, batch, bwords, n_planes_chunk, chunk);
    }
  }
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

/* copy_batched_exのバッチ一括版(Metal版mdl_metal_copy_strided対称)。*/
int mdl_cuda_copy_strided(const unsigned char *src, unsigned char *dst,
                          int dst_off, int dst_stride, int src_off,
                          int src_stride, int n_bytes, int batch) {
  const int total = n_bytes * batch;
  mdl_k_copy_strided<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(
      src, dst, dst_off, dst_stride, src_off, src_stride, n_bytes, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

/* ---- RDDLGN自己回帰生成(decode)ハーネス専用 ---- */
int mdl_cuda_embed_lookup(const int *tokens, const unsigned char *table,
                          unsigned char *out, int d_emb, int batch) {
  const int total = d_emb * batch;
  mdl_k_embed_lookup<<<mdl_grid(total), MDL_BLOCK, 0, MDL_STREAM>>>(tokens, table, out,
                                                     d_emb, total);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_argmax(const int *scores, int *tokens, int width, int batch) {
  mdl_k_argmax<<<mdl_grid(batch), MDL_BLOCK, 0, MDL_STREAM>>>(scores, tokens, width, batch);
  MDL_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int mdl_cuda_sync(void) {
  MDL_CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

/* ---- ストリーム・パイプライン化 ----
 * 状態はこのファイル内に閉じる(生成コード側は mdl_cuda_select_stream/
 * mdl_cuda_commit_async/mdl_cuda_wait_slot/mdl_cuda_upload_async/
 * mdl_cuda_download_async だけを呼ぶ)。詳細は mdl_cuda.h のコメント参照。 */
static cudaStream_t g_mdl_streams[MDL_CUDA_MAX_STREAMS];
static cudaEvent_t g_mdl_events[MDL_CUDA_MAX_STREAMS];
static int g_mdl_pending[MDL_CUDA_MAX_STREAMS] = {0};
static int g_mdl_n_streams = 0;
static cudaStream_t g_mdl_cur_stream = 0;   /* 0 = レガシーデフォルトストリーム */

cudaStream_t mdl_cuda_current_stream(void) { return g_mdl_cur_stream; }

int mdl_cuda_streams_init(int n_streams) {
  int i;
  if (n_streams < 1 || n_streams > MDL_CUDA_MAX_STREAMS) return -1;
  if (g_mdl_n_streams > 0) return 0;   /* 冪等 */
  for (i = 0; i < n_streams; i++) {
    MDL_CUDA_CHECK(cudaStreamCreate(&g_mdl_streams[i]));
    MDL_CUDA_CHECK(cudaEventCreateWithFlags(&g_mdl_events[i],
                                            cudaEventDisableTiming));
    g_mdl_pending[i] = 0;
  }
  g_mdl_n_streams = n_streams;
  return 0;
}

void mdl_cuda_streams_shutdown(void) {
  int i;
  if (g_mdl_n_streams <= 0) return;
  for (i = 0; i < g_mdl_n_streams; i++) {
    if (g_mdl_pending[i]) {
      cudaEventSynchronize(g_mdl_events[i]);
      g_mdl_pending[i] = 0;
    }
    cudaEventDestroy(g_mdl_events[i]);
    cudaStreamDestroy(g_mdl_streams[i]);
  }
  g_mdl_n_streams = 0;
  g_mdl_cur_stream = 0;
}

int mdl_cuda_select_stream(int slot) {
  if (slot < 0 || slot >= g_mdl_n_streams) return -1;
  g_mdl_cur_stream = g_mdl_streams[slot];
  return 0;
}

void mdl_cuda_select_default_stream(void) { g_mdl_cur_stream = 0; }

int mdl_cuda_commit_async(int slot) {
  if (slot < 0 || slot >= g_mdl_n_streams) return -1;
  if (g_mdl_pending[slot]) {
    /* 前回ぶんをまだ mdl_cuda_wait_slot していない。ここで黙って進むと
     * 同じ「バッファ共有の誤り」を起こしうるので拒否する
     * (Metal版 mdl_metal_commit_async と同じガード挙動)。 */
    fprintf(stderr,
           "mdl_cuda: スロット%d がまだpending(mdl_cuda_wait_slotを"
           "呼ばずに次のcommit_asyncを呼んだ)\n", slot);
    return -1;
  }
  MDL_CUDA_CHECK(cudaEventRecord(g_mdl_events[slot], g_mdl_streams[slot]));
  g_mdl_pending[slot] = 1;
  return 0;
}

int mdl_cuda_wait_slot(int slot) {
  if (slot < 0 || slot >= g_mdl_n_streams) return -1;
  if (!g_mdl_pending[slot]) return 0;   /* 何も積んでいなければ冪等に戻る */
  MDL_CUDA_CHECK(cudaEventSynchronize(g_mdl_events[slot]));
  g_mdl_pending[slot] = 0;
  return 0;
}

int mdl_cuda_upload_async(void *dst, const void *src, size_t n_bytes) {
  MDL_CUDA_CHECK(cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyHostToDevice,
                                 g_mdl_cur_stream));
  return 0;
}

int mdl_cuda_download_async(void *dst, const void *src, size_t n_bytes) {
  MDL_CUDA_CHECK(cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyDeviceToHost,
                                 g_mdl_cur_stream));
  return 0;
}

}  /* extern "C" */
