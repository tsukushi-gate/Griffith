/* griffith-logic — Cランタイム カーネル実装
 *
 * 移植元: tools/webbench/src/engines/bitplane.ts と
 *         tools/webbench/assembly/index.ts(GCなし・線形メモリ版)
 * 契約は mdl_kernels.h を参照。整数出力は JS / WASM / WebGPU と完全一致する。
 */
#include "mdl_kernels.h"
#include "mdl_simd.h"

/* libc への依存は memcpy / memset だけ。ベアメタル(フリースタンディング)では
 * <string.h> 自体が存在しないので、-DMDL_NO_LIBC で自前実装に切り替えられる
 * ようにしておく(C標準がフリースタンディング処理系に要求するヘッダに
 * string.h は含まれない)。通常のホスト環境では libc 版の方が速いので既定。 */
#ifdef MDL_NO_LIBC
void *mdl_memcpy(void *dst, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  size_t i;
  for (i = 0; i < n; i++) d[i] = s[i];
  return dst;
}
static void *mdl_memset(void *dst, int c, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  size_t i;
  for (i = 0; i < n; i++) d[i] = (unsigned char)c;
  return dst;
}
#define memcpy mdl_memcpy
#define memset mdl_memset
#else
#include <string.h>
/* generated model.c(アキュムレータのシードコピー)向けの公開版。
 * mdl_kernels.h 参照。ここでは通常のlibc memcpyへそのまま委譲するだけ
 * (このビルドモードではmdl_kernels.c自身も上のmemcpyマクロ経由ではなく
 * 直接libc memcpyを使っているので、内部の呼び出しには一切影響しない)。 */
void *mdl_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}
#endif

/* ------------------------------------------------------------------ */
/* 2入力ゲート                                                         */
/* ------------------------------------------------------------------ */

uint32_t mdl_gate2(uint32_t tt, uint32_t a, uint32_t b) {
  switch (tt & 0xfu) {
    case 0x0: return 0u;
    case 0x1: return ~(a | b);
    case 0x2: return a & ~b;
    case 0x3: return ~b;
    case 0x4: return ~a & b;
    case 0x5: return ~a;
    case 0x6: return a ^ b;
    case 0x7: return ~(a & b);
    case 0x8: return a & b;
    case 0x9: return ~(a ^ b);
    case 0xa: return a;
    case 0xb: return a | ~b;
    case 0xc: return b;
    case 0xd: return ~a | b;
    case 0xe: return a | b;
    default:  return 0xffffffffu;
  }
}

/* 分岐をワードループの外に出した区間版(bitplane.ts::gate2RangeInto と同形)。
 * 実測でここがconv実行時間の主要部になるため、tt判定を内側に置かない。
 *
 *(後半): `-mavx2`ビルドでのconv系0.79倍悪化(gccの生半可な自動
 * ベクタ化が疑い)を、mdl_simd.hの`mdl_gate2_vec_*`(AND/OR/XOR/NOT/
 * ANDNOTの5プリミティブ)を使った明示ベクタ化で潰す。`MDL_HAS_GATE2_VEC`
 * が立つアーキ(現状AVX2のみ)ではMDL_GATE2_VEC_WIDTH語ずつをベクタ演算で
 * 処理し、端数はスカラーで片付ける。ヘルパの無いアーキ(NEON/SSE2/
 * スカラー/-DMDL_NO_SIMD)は`MDL_GATE_LOOP2`が`MDL_GATE_LOOP`へそのまま
 * 委譲するので、生成コードは元のスカラーループと1バイトも変わらない。
 * 16分岐という構造そのものは変えていない(マスク式化はしない)。*/
void mdl_gate2_range(uint32_t *buf, int32_t dst_off, int32_t a_off,
                     int32_t b_off, uint32_t tt, int32_t n) {
  uint32_t *d = buf + dst_off;
  const uint32_t *a = buf + a_off;
  const uint32_t *b = buf + b_off;
  int32_t i;

#define MDL_GATE_LOOP(expr)                        \
  for (i = 0; i < n; i++) { d[i] = (expr); }       \
  break

#if defined(MDL_HAS_GATE2_VEC)
#define MDL_GATE_LOOP2(vexpr, sexpr)                                      \
  do {                                                                     \
    const int32_t nv = n - (n % MDL_GATE2_VEC_WIDTH);                     \
    for (i = 0; i < nv; i += MDL_GATE2_VEC_WIDTH) {                       \
      const mdl_gate2_vec av = mdl_gate2_vec_load(a + i);                 \
      const mdl_gate2_vec bv = mdl_gate2_vec_load(b + i);                 \
      /* 0x3/0x5/0xa/0xcはa,bの片方しか使わない(コピー/NOT単項)ので、    \
         その場合は未使用側が-Wunused-variableを出す。両方読み込んで     \
         おくのはロード命令を分岐間で共通化して分岐構造を単純に保つ     \
         ため(値としては使わない側を読んでも安全: 副作用が無い読み取り  \
         専用ロード)。(void)キャストで警告だけ黙らせる。 */              \
      (void)av; (void)bv;                                                \
      mdl_gate2_vec_store(d + i, (vexpr));                                \
    }                                                                      \
    for (; i < n; i++) { d[i] = (sexpr); }                                \
  } while (0);                                                             \
  break
#else
#define MDL_GATE_LOOP2(vexpr, sexpr) MDL_GATE_LOOP(sexpr)
#endif

  switch (tt & 0xfu) {
    case 0x0: memset(d, 0x00, (size_t)n * sizeof(uint32_t)); break;
    case 0x1: MDL_GATE_LOOP2(mdl_gate2_vec_not(mdl_gate2_vec_or(av, bv)),
                             ~(a[i] | b[i]));
    case 0x2: MDL_GATE_LOOP2(mdl_gate2_vec_andnot(bv, av), a[i] & ~b[i]);
    case 0x3: MDL_GATE_LOOP2(mdl_gate2_vec_not(bv), ~b[i]);
    case 0x4: MDL_GATE_LOOP2(mdl_gate2_vec_andnot(av, bv), ~a[i] & b[i]);
    case 0x5: MDL_GATE_LOOP2(mdl_gate2_vec_not(av), ~a[i]);
    case 0x6: MDL_GATE_LOOP2(mdl_gate2_vec_xor(av, bv), a[i] ^ b[i]);
    case 0x7: MDL_GATE_LOOP2(mdl_gate2_vec_not(mdl_gate2_vec_and(av, bv)),
                             ~(a[i] & b[i]));
    case 0x8: MDL_GATE_LOOP2(mdl_gate2_vec_and(av, bv), a[i] & b[i]);
    case 0x9: MDL_GATE_LOOP2(mdl_gate2_vec_not(mdl_gate2_vec_xor(av, bv)),
                             ~(a[i] ^ b[i]));
    case 0xa: MDL_GATE_LOOP2(av, a[i]);
    case 0xb: MDL_GATE_LOOP2(mdl_gate2_vec_or(av, mdl_gate2_vec_not(bv)),
                             a[i] | ~b[i]);
    case 0xc: MDL_GATE_LOOP2(bv, b[i]);
    case 0xd: MDL_GATE_LOOP2(mdl_gate2_vec_or(mdl_gate2_vec_not(av), bv),
                             ~a[i] | b[i]);
    case 0xe: MDL_GATE_LOOP2(mdl_gate2_vec_or(av, bv), a[i] | b[i]);
    default:  memset(d, 0xff, (size_t)n * sizeof(uint32_t)); break;
  }

#undef MDL_GATE_LOOP2
#undef MDL_GATE_LOOP
}

/* ------------------------------------------------------------------ */
/* pack / unpack                                                       */
/* ------------------------------------------------------------------ */

void mdl_pack_planes(const uint8_t *src, uint32_t *dst,
                     int32_t c, int32_t h, int32_t w) {
  const int32_t wpr = mdl_words_per_row(w);
  int32_t ch, y, j;
  for (ch = 0; ch < c; ch++) {
    for (y = 0; y < h; y++) {
      const uint8_t *row = src + ((size_t)ch * h + y) * w;
      uint32_t *out = dst + ((size_t)ch * h + y) * wpr;
      for (j = 0; j < wpr; j++) {
        const int32_t x0 = j * 32;
        const int32_t n = (w - x0) < 32 ? (w - x0) : 32;
        uint32_t acc = 0u;
        int32_t i = 0;
        /* 実モデルでは入力幅が数百万ビットになるので、ここもベクタ化する。
         * n==32(語の中身がまるごと使われる、行末以外はほぼ常にこちら)なら
         * `mdl_mask32` で32バイトを1回で32ビットへ詰める。同じ
         * 勝ち筋: AVX2では `_mm256_movemask_epi8` 1命令、それ以外の
         * 経路では下の16バイト版2回合成に自動で落ちるので、値はどちらでも
         * 元のループ(16バイト×2)と1ビットも変わらない。 */
        if (n == 32) {
          acc = mdl_mask32(row + x0);
          i = 32;
        } else {
          for (; i + 16 <= n; i += 16) {
            acc |= mdl_ne_zero_mask16(row + x0 + i) << i;
          }
        }
        for (; i < n; i++) {
          acc |= (uint32_t)(row[x0 + i] != 0) << i;
        }
        out[j] = acc; /* x>=w のビットは立てない = 端数ビット常0 */
      }
    }
  }
}

void mdl_unpack_planes(const uint32_t *src, uint8_t *dst,
                       int32_t c, int32_t h, int32_t w) {
  const int32_t wpr = mdl_words_per_row(w);
  int32_t ch, y, x;
  for (ch = 0; ch < c; ch++) {
    for (y = 0; y < h; y++) {
      const uint32_t *row = src + ((size_t)ch * h + y) * wpr;
      uint8_t *out = dst + ((size_t)ch * h + y) * w;
      for (x = 0; x < w; x++) {
        out[x] = (uint8_t)((row[x >> 5] >> (x & 31)) & 1u);
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* タップ平面                                                          */
/* ------------------------------------------------------------------ */

/* 1行ぶんの水平シフト(汎用)。out[x] = in[x + s]。行外は0。 */
static void shift_row_into(uint32_t *dst, const uint32_t *src, int32_t wpr,
                           int32_t s, int32_t dst_off, int32_t src_off) {
  int32_t j;
  const int32_t word_shift = s >> 5;              /* 算術シフト(負でも床) */
  const int32_t bit_shift = s - word_shift * 32;  /* [0,32) */
  for (j = 0; j < wpr; j++) {
    const int32_t j0 = j + word_shift;
    const uint32_t lo = (j0 >= 0 && j0 < wpr) ? src[src_off + j0] : 0u;
    uint32_t v = bit_shift == 0 ? lo : (lo >> bit_shift);
    if (bit_shift != 0) {
      const int32_t j1 = j0 + 1;
      const uint32_t hi = (j1 >= 0 && j1 < wpr) ? src[src_off + j1] : 0u;
      v |= hi << (32 - bit_shift);
    }
    dst[dst_off + j] = v;
  }
}

void mdl_shared_shift_plane(const uint32_t *src, int32_t src_base,
                            int32_t h, int32_t wpr, int32_t p_h, int32_t s,
                            uint32_t *dst, int32_t dst_base) {
  const int32_t n = h * wpr;
  const int32_t body = dst_base + p_h * wpr;
  int32_t i, y;

  memset(dst + dst_base, 0, (size_t)(p_h * wpr) * sizeof(uint32_t));
  memset(dst + body + n, 0, (size_t)(p_h * wpr) * sizeof(uint32_t));

  if (s == 0) {
    memcpy(dst + body, src + src_base, (size_t)n * sizeof(uint32_t));
    return;
  }
  if (s <= -32 || s >= 32) {
    /* |s|>=32(kW>33 等の想定外ジオメトリ)は行ごとの汎用路へ落とす。 */
    for (y = 0; y < h; y++) {
      shift_row_into(dst, src, wpr, s, body + y * wpr, src_base + y * wpr);
    }
    return;
  }
  {
    /* |s|<32 なので word_shift は 0(s>0)か -1(s<0)のみ。
     * 内部h行は連続領域なので行境界を無視した直線ループで合成し、
     * 行をまたいで混入した語だけを後から潰す(bitplane.ts の C5 と同じ)。 */
    const int32_t word_shift = s >> 5;
    const int32_t bit_shift = s - word_shift * 32;
    const int32_t inv = 32 - bit_shift;
    if (wpr == 1) {
      /* 1行=1語: 隣接語は常に行外 = 混入が起きない。 */
      if (word_shift == 0) {
        for (i = 0; i < n; i++) dst[body + i] = src[src_base + i] >> bit_shift;
      } else {
        for (i = 0; i < n; i++) dst[body + i] = src[src_base + i] << inv;
      }
      return;
    }
    if (word_shift == 0) {
      const int32_t last = n - 1;
      for (i = 0; i < last; i++) {
        dst[body + i] = (src[src_base + i] >> bit_shift)
                      | (src[src_base + i + 1] << inv);
      }
      dst[body + last] = src[src_base + last] >> bit_shift;
      for (y = 0; y < h - 1; y++) {
        i = y * wpr + (wpr - 1);
        dst[body + i] = src[src_base + i] >> bit_shift;
      }
    } else {
      dst[body] = src[src_base] << inv;
      for (i = 1; i < n; i++) {
        dst[body + i] = (src[src_base + i - 1] >> bit_shift)
                      | (src[src_base + i] << inv);
      }
      for (y = 1; y < h; y++) {
        i = y * wpr;
        dst[body + i] = src[src_base + i] << inv;
      }
    }
  }
}

void mdl_tap_plane_strided(const uint32_t *src, int32_t src_base,
                           int32_t h, int32_t w, int32_t wpr_in,
                           int32_t dy, int32_t dx,
                           int32_t h_out, int32_t w_out, int32_t wpr_out,
                           int32_t s_h, int32_t s_w, int32_t p_h, int32_t p_w,
                           uint32_t *dst, int32_t dst_base) {
  int32_t oy, ox;
  memset(dst + dst_base, 0, (size_t)(h_out * wpr_out) * sizeof(uint32_t));
  for (oy = 0; oy < h_out; oy++) {
    const int32_t y = oy * s_h - p_h + dy;
    if (y < 0 || y >= h) continue;
    {
      const int32_t row_base = src_base + y * wpr_in;
      const int32_t out_row = dst_base + oy * wpr_out;
      for (ox = 0; ox < w_out; ox++) {
        const int32_t xx = ox * s_w - p_w + dx;
        if (xx < 0 || xx >= w) continue;
        if ((src[row_base + (xx >> 5)] >> (xx & 31)) & 1u) {
          dst[out_row + (ox >> 5)] |= (uint32_t)1 << (ox & 31);
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* conv2d                                                              */
/* ------------------------------------------------------------------ */

void mdl_conv_init_scratch(const mdl_conv_layout *L, uint32_t *scratch) {
  memset(scratch, 0x00, (size_t)L->tap_rows_words * sizeof(uint32_t));
  memset(scratch + L->tap_rows_words, 0xff,
         (size_t)L->tap_rows_words * sizeof(uint32_t));
}

/* 出力バケットを書き出す。tap_wpr == wpr_out は経路によらず成り立つので、
 * 1つのocの出力平面は連続コピー + 各行末語への tail_mask で書ける。 */
static void write_bucket(const mdl_conv_layout *L, int32_t bucket,
                         const uint32_t *scratch, uint32_t *out) {
  const int32_t plane_words = L->h_out * L->wpr_out;
  int32_t i;
  for (i = L->write_start[bucket]; i < L->write_start[bucket + 1]; i++) {
    const int32_t oc = L->write_oc[i];
    const uint32_t *srcp = scratch + L->oc_src_off[oc];
    uint32_t *dstp = out + (size_t)oc * plane_words;
    int32_t oy;
    memcpy(dstp, srcp, (size_t)plane_words * sizeof(uint32_t));
    for (oy = 0; oy < L->h_out; oy++) {
      dstp[oy * L->wpr_out + L->wpr_out - 1] &= L->mask_out;
    }
  }
}

void mdl_conv_build_planes(const mdl_conv_layout *L, const uint32_t *in,
                           uint32_t *scratch) {
  int32_t u;
  for (u = 0; u < L->n_planes; u++) {
    const int32_t c_base = L->plane_c[u] * L->h * L->wpr_in;
    const int32_t dst_base = L->pool_base + u * L->plane_stride;
    if (L->use_fast_path) {
      mdl_shared_shift_plane(in, c_base, L->h, L->wpr_in, L->p_h,
                             L->plane_dx[u] - L->p_w, scratch, dst_base);
    } else {
      mdl_tap_plane_strided(in, c_base, L->h, L->w, L->wpr_in,
                            L->plane_dy[u], L->plane_dx[u],
                            L->h_out, L->w_out, L->wpr_out,
                            L->s_h, L->s_w, L->p_h, L->p_w,
                            scratch, dst_base);
    }
  }
}

void mdl_conv_mask_tails(const mdl_conv_layout *L, uint32_t *out) {
  const int32_t plane_words = L->h_out * L->wpr_out;
  int32_t oc, oy;
  if (L->mask_out == 0xffffffffu) return;      /* 幅が32の倍数なら不要 */
  for (oc = 0; oc < L->out_channels; oc++) {
    uint32_t *p = out + (size_t)oc * plane_words + L->wpr_out - 1;
    for (oy = 0; oy < L->h_out; oy++) p[oy * L->wpr_out] &= L->mask_out;
  }
}

void mdl_conv_exec(const mdl_conv_layout *L, const uint32_t *in,
                   uint32_t *scratch, uint32_t *out) {
  int32_t j;

  mdl_conv_build_planes(L, in, scratch);

  /* バケット0 = 出力が定数/タップ直結の oc(命令の実行前に書ける)。 */
  write_bucket(L, 0, scratch, out);
  for (j = 0; j < L->n_instrs; j++) {
    const int32_t *ins = L->instr_flat + (size_t)j * 4;
    mdl_gate2_range(scratch, ins[3], ins[1], ins[2], (uint32_t)ins[0],
                    L->tap_rows_words);
    /* 命令jの直後に「出力がjの oc」を書き出す(生存区間はここで閉じる)。 */
    if (L->write_start[j + 2] > L->write_start[j + 1]) {
      write_bucket(L, j + 1, scratch, out);
    }
  }
}

/* ------------------------------------------------------------------ */
/* ビット詰め替えの小道具                                              */
/* ------------------------------------------------------------------ */

/* 32bit中の偶数位置ビット(0,2,...,30)を連続する下位16bitへ圧縮する。
 * Hacker's Delight型のシフトカスケード(奇数位置ビットは結果に影響しない)。
 * bitplane.ts::compressEven32Shift と同一。 */
static uint32_t compress_even32(uint32_t v) {
  uint32_t x = v & 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

/* 下位16bitを偶数位置(0,2,...,30)へ拡散する(compress_even32の逆)。 */
static uint32_t spread_even16(uint32_t v) {
  uint32_t x = v & 0xffffu;
  x = (x | (x << 8)) & 0x00ff00ffu;
  x = (x | (x << 4)) & 0x0f0f0f0fu;
  x = (x | (x << 2)) & 0x33333333u;
  x = (x | (x << 1)) & 0x55555555u;
  return x;
}

/* 1入力ビットを出力2ビット(2i, 2i+1)へ複製する。 */
static uint32_t dilate16_to32(uint32_t v) {
  const uint32_t even = spread_even16(v);
  return even | (even << 1);
}

/* ------------------------------------------------------------------ */
/* or_pool2d(2x2, stride2, pad0)                                       */
/* ------------------------------------------------------------------ */

void mdl_or_pool2x2(const uint32_t *in, uint32_t *out,
                    int32_t c, int32_t h, int32_t w) {
  const int32_t wpr = mdl_words_per_row(w);
  const int32_t h_out = ((h - 2) >> 1) + 1;
  const int32_t w_out = ((w - 2) >> 1) + 1;
  const int32_t wpr_out = mdl_words_per_row(w_out);
  const uint32_t mask_out = mdl_tail_mask(w_out);
  int32_t ch, oy, wo, j;

  for (ch = 0; ch < c; ch++) {
    const uint32_t *cin = in + (size_t)ch * h * wpr;
    uint32_t *cout = out + (size_t)ch * h_out * wpr_out;
    for (oy = 0; oy < h_out; oy++) {
      const uint32_t *row_a = cin + (size_t)(2 * oy) * wpr;
      const uint32_t *row_b = cin + (size_t)(2 * oy + 1) * wpr;
      uint32_t *dst = cout + (size_t)oy * wpr_out;
      for (wo = 0; wo < wpr_out; wo++) {
        /* t = rowA | rowB(縦OR)。u(j) = t[j] | t[j]>>1 | t[j+1]<<31 で
         * 水平隣接ペア(2i, 2i+1)のORを偶数位置へ集約する。 */
        uint32_t lo, hi = 0u;
        j = 2 * wo;
        {
          const uint32_t t0 = j < wpr ? (row_a[j] | row_b[j]) : 0u;
          const uint32_t t1 = (j + 1) < wpr ? (row_a[j + 1] | row_b[j + 1]) : 0u;
          lo = t0 | (t0 >> 1) | (t1 << 31);
        }
        j = 2 * wo + 1;
        if (j < wpr) {
          const uint32_t t0 = row_a[j] | row_b[j];
          const uint32_t t1 = (j + 1) < wpr ? (row_a[j + 1] | row_b[j + 1]) : 0u;
          hi = t0 | (t0 >> 1) | (t1 << 31);
        }
        dst[wo] = compress_even32(lo) | (compress_even32(hi) << 16);
      }
      dst[wpr_out - 1] &= mask_out;
    }
  }
}

/* ------------------------------------------------------------------ */
/* upsample2d(scale=2, nearest)                                        */
/* ------------------------------------------------------------------ */

void mdl_upsample2d_x2(const uint32_t *in, uint32_t *out,
                       int32_t c, int32_t h, int32_t w) {
  const int32_t wpr = mdl_words_per_row(w);
  const int32_t h_out = h * 2;
  const int32_t w_out = w * 2;
  const int32_t wpr_out = mdl_words_per_row(w_out);
  const uint32_t mask_out = mdl_tail_mask(w_out);
  int32_t ch, y, ow;

  for (ch = 0; ch < c; ch++) {
    const uint32_t *cin = in + (size_t)ch * h * wpr;
    uint32_t *cout = out + (size_t)ch * h_out * wpr_out;
    for (y = 0; y < h; y++) {
      const uint32_t *row_in = cin + (size_t)y * wpr;
      uint32_t *row0 = cout + (size_t)(2 * y) * wpr_out;
      uint32_t *row1 = cout + (size_t)(2 * y + 1) * wpr_out;
      for (ow = 0; ow < wpr_out; ow++) {
        const int32_t si = ow >> 1;
        const uint32_t sw = si < wpr ? row_in[si] : 0u;
        const uint32_t half = (ow & 1) == 0 ? (sw & 0xffffu) : ((sw >> 16) & 0xffffu);
        const uint32_t v = dilate16_to32(half);
        row0[ow] = v;
        row1[ow] = v;
      }
      row0[wpr_out - 1] &= mask_out;
      row1[wpr_out - 1] &= mask_out;
    }
  }
}

/* ------------------------------------------------------------------ */
/* チャネル並べ替え / concat                                           */
/* ------------------------------------------------------------------ */

void mdl_permute_channels(const uint32_t *in, uint32_t *out,
                          const int32_t *order, int32_t c_out,
                          int32_t h, int32_t wpr) {
  const size_t ch_words = (size_t)h * wpr;
  int32_t o;
  for (o = 0; o < c_out; o++) {
    memcpy(out + (size_t)o * ch_words, in + (size_t)order[o] * ch_words,
           ch_words * sizeof(uint32_t));
  }
}

void mdl_concat_planes(const uint32_t *const *srcs, const int32_t *c_in,
                       int32_t n_in, uint32_t *out, int32_t h, int32_t wpr) {
  const size_t ch_words = (size_t)h * wpr;
  size_t off = 0;
  int32_t i;
  for (i = 0; i < n_in; i++) {
    const size_t n = (size_t)c_in[i] * ch_words;
    memcpy(out + off, srcs[i], n * sizeof(uint32_t));
    off += n;
  }
}

/* ------------------------------------------------------------------ */
/* SpatialGroupSum(読み出しヘッド)                                    */
/* ------------------------------------------------------------------ */

int32_t mdl_spatial_group_sum_planes(int32_t gs) {
  int32_t n = 0;
  while (gs > 0) { n++; gs >>= 1; }
  return n; /* = ceil(log2(gs+1)) */
}

void mdl_spatial_group_sum(const uint32_t *in, int32_t *scores,
                           int32_t n_out, int32_t gs, int32_t h, int32_t w,
                           uint32_t *planes) {
  const int32_t wpr = mdl_words_per_row(w);
  const int32_t n_planes = mdl_spatial_group_sum_planes(gs);
  const int32_t ch_words = h * wpr;
  const uint32_t last_mask = mdl_tail_mask(w);
  int32_t o, y, g, j, p;

  for (o = 0; o < n_out; o++) {
    const uint32_t *o_base = in + (size_t)o * gs * ch_words;
    for (y = 0; y < h; y++) {
      int32_t *out_row = scores + ((size_t)o * h + y) * w;
      /* 下の転置は立っているビットだけを `|=` で積むので、出力側を先に
       * 0 にしておく必要がある。呼び出し側のゼロ初期化に頼ると
       * 「2回目の推論で前回の結果が混ざる」「共有バッファだと前段の
       * 書き込みが残る」という形で静かに壊れる。 */
      memset(out_row, 0, (size_t)w * sizeof(int32_t));
      memset(planes, 0, (size_t)(n_planes * wpr) * sizeof(uint32_t));
      /* 垂直カウンタ加算(g方向)。総和は高々 gs なので最上位平面からの
       * 繰り上がりは常に0。 */
      for (g = 0; g < gs; g++) {
        const uint32_t *row = o_base + (size_t)g * ch_words + (size_t)y * wpr;
        for (j = 0; j < wpr; j++) {
          uint32_t carry = row[j];
          for (p = 0; carry != 0u && p < n_planes; p++) {
            const int32_t idx = p * wpr + j;
            const uint32_t t = planes[idx] & carry;
            planes[idx] ^= carry;
            carry = t;
          }
        }
      }
      /* ビットスライス平面 → スコアへ転置。立っているビットだけ走査するので
       * コストは 32*n_planes ではなく実際のポップカウントに比例する。 */
      for (j = 0; j < wpr; j++) {
        const int32_t x_base = j << 5;
        const uint32_t mask = (j == wpr - 1) ? last_mask : 0xffffffffu;
        for (p = 0; p < n_planes; p++) {
          uint32_t v = planes[(size_t)p * wpr + j] & mask;
          const int32_t bit = (int32_t)1 << p;
          while (v != 0u) {
            const uint32_t low = v & (~v + 1u); /* 最下位の立っているビット */
            int32_t pos = 0;
            uint32_t t = low;
            while (t > 1u) { t >>= 1; pos++; }
            out_row[x_base + pos] |= bit;
            v ^= low;
          }
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 非パック層                                                          */
/* ------------------------------------------------------------------ */

void mdl_lut_layer(const uint8_t *x, uint8_t *out, int32_t n_inputs,
                   int32_t out_width, const int32_t *indices,
                   const uint32_t *truth_tables) {
  int32_t o, j;
  for (o = 0; o < out_width; o++) {
    uint32_t addr = 0u;
    for (j = 0; j < n_inputs; j++) {
      addr |= (uint32_t)x[indices[(size_t)j * out_width + o]] << j;
    }
    out[o] = (uint8_t)((truth_tables[o] >> addr) & 1u);
  }
}

/* 入力数を**コンパイル時定数**にした本体。段数も語数も定数になるので、
 * 集めるループも三項演算の木も完全に展開される。
 *
 * 段数を実行時の値のままにすると展開されず、実測で 2.18倍 が 1.18倍 まで
 * 落ちる。3〜5入力(multilut / ABC合成後のモデル)でも同じことが
 * 起きていたので、N ごとに実体化する。
 *
 * `indices` に明示的な型を付けず、直接 `indices[...]` で読む。
 * このマクロは int32_t 版・uint16_t 版どちらの `mdl_lut_layer_slice*` からも
 * 展開されるので、囲む関数の `indices` 引数の型がそのままここへ伝わる
 * ——添字の幅ごとに本体を複製せずに済む(値は完全に同じでなければ
 * ならないので、複製すると経路間でずれるリスクを作る)。 */
#define MDL_SLICE_BODY(N)                                                     \
  do {                                                                        \
    const int32_t nmin_ = 1 << (N);                                           \
    const int32_t full_ = out_width & ~31;                                    \
    uint8_t bf_[(N)][32];                                                     \
    int32_t b_, j_, l_, m_, half_;                                            \
    for (b_ = 0; b_ < full_; b_ += 32) {                                      \
      const int32_t wi_ = b_ >> 5;                                            \
      uint32_t av_[(N)], s_[1 << (N)], w_;                                    \
      for (j_ = 0; j_ < (N); j_++) {                                          \
        const size_t ib_ = (size_t)j_ * out_width + b_;                       \
        for (l_ = 0; l_ < 32; l_++) bf_[j_][l_] = x[indices[ib_ + l_]];       \
        av_[j_] = mdl_mask32(bf_[j_]);                                        \
      }                                                                       \
      for (m_ = 0; m_ < nmin_; m_++)                                          \
        s_[m_] = tt_slices[(size_t)m_ * wpr + wi_];                           \
      for (j_ = 0, half_ = nmin_ >> 1; half_ >= 1; j_++, half_ >>= 1) {       \
        const uint32_t a_ = av_[j_];                                          \
        for (m_ = 0; m_ < half_; m_++) {                                      \
          const uint32_t u_ = s_[2 * m_], v_ = s_[2 * m_ + 1];                \
          s_[m_] = u_ ^ (a_ & (u_ ^ v_));                                     \
        }                                                                     \
      }                                                                       \
      w_ = s_[0];                                                             \
      for (l_ = 0; l_ < 32; l_++)                                             \
        out[b_ + l_] = (uint8_t)((w_ >> l_) & 1u);                            \
    }                                                                         \
    for (b_ = full_; b_ < out_width; b_++) {                                  \
      uint32_t ad_ = 0u;                                                      \
      for (j_ = 0; j_ < (N); j_++)                                            \
        ad_ |= (uint32_t)x[indices[(size_t)j_ * out_width + b_]] << j_;       \
      out[b_] = (uint8_t)                                                     \
        ((tt_slices[(size_t)ad_ * wpr + (b_ >> 5)] >> (b_ & 31)) & 1u);       \
    }                                                                         \
    return;                                                                   \
  } while (0)

/* `mdl_lut_layer_slice` / `mdl_lut_layer_slice_u16` の共通本体。
 * 添字配列 `indices` の型(int32_t* または uint16_t*)は呼び出し元の
 * 関数シグネチャで決まり、このマクロはそれをそのまま使う(型を持ち出さない
 * ので複製にならない)。値は両関数で1ビットも変わらない
 * ——tests/deploy/test_lut_slice.pyがu16/i32両経路の一致を固定する。 */
#define MDL_LUT_LAYER_SLICE_BODY                                             \
  const int32_t wpr = (out_width + 31) >> 5;                                 \
  const int32_t n_min = (int32_t)1 << n_inputs;                              \
  int32_t base, j, l, m, half;                                               \
  uint8_t buf[32];                                                           \
  switch (n_inputs) {                                                        \
    case 1: MDL_SLICE_BODY(1);                                               \
    case 3: MDL_SLICE_BODY(3);                                               \
    case 4: MDL_SLICE_BODY(4);                                               \
    case 5: MDL_SLICE_BODY(5);                                               \
    default: break;    /* 2 は下の手書き経路、それ以外は汎用へ落ちる */      \
  }                                                                          \
  if (n_inputs == 2) {                                                       \
    /* DLGN の素の形(2入力ゲート)。段数と語数が定数になるので、集める      \
     * ループも三項演算の木も完全に展開される。汎用経路の約2倍速い。 */     \
    const uint32_t *t0 = tt_slices;                                          \
    const uint32_t *t1 = tt_slices + wpr;                                    \
    const uint32_t *t2 = tt_slices + 2 * wpr;                                \
    const uint32_t *t3 = tt_slices + 3 * wpr;                                \
    const int32_t full = out_width & ~31;                                    \
    uint8_t bb[32];                                                          \
    for (base = 0; base < full; base += 32) {                                \
      const int32_t wi = base >> 5;                                          \
      uint32_t a, b, u, v, w;                                                \
      for (l = 0; l < 32; l++) buf[l] = x[indices[base + l]];                \
      for (l = 0; l < 32; l++) bb[l] = x[indices[out_width + base + l]];     \
      a = mdl_mask32(buf);                                                   \
      b = mdl_mask32(bb);                                                    \
      u = t0[wi] ^ (a & (t0[wi] ^ t1[wi]));                                  \
      v = t2[wi] ^ (a & (t2[wi] ^ t3[wi]));                                  \
      w = u ^ (b & (u ^ v));                                                 \
      for (l = 0; l < 32; l++) out[base + l] = (uint8_t)((w >> l) & 1u);     \
    }                                                                        \
    for (base = full; base < out_width; base++) {                           \
      const uint32_t addr = (uint32_t)x[indices[base]]                      \
                          | ((uint32_t)x[indices[out_width + base]] << 1);   \
      const uint32_t s = tt_slices[(size_t)addr * wpr + (base >> 5)];       \
      out[base] = (uint8_t)((s >> (base & 31)) & 1u);                       \
    }                                                                        \
    return;                                                                  \
  }                                                                          \
  for (base = 0; base < out_width; base += 32) {                             \
    const int32_t take = (out_width - base < 32) ? (out_width - base) : 32;  \
    const int32_t wi = base >> 5;                                            \
    uint32_t av[5];     /* 入力ごとの32ゲートぶんのビット列 */               \
    uint32_t sel[32];   /* 2^n_inputs <= 32 */                               \
    uint32_t w;                                                              \
    for (j = 0; j < n_inputs; j++) {                                         \
      const size_t ib2 = (size_t)j * out_width + base;                       \
      for (l = 0; l < take; l++) buf[l] = x[indices[ib2 + l]];               \
      for (l = take; l < 32; l++) buf[l] = 0u;                               \
      av[j] = mdl_mask32(buf);                                               \
    }                                                                        \
    for (m = 0; m < n_min; m++) sel[m] = tt_slices[(size_t)m * wpr + wi];    \
    /* 入力0から順に「まとめて三項演算」で畳む。段ごとに候補が半分になる。 */\
    for (j = 0, half = n_min >> 1; half >= 1; j++, half >>= 1) {             \
      const uint32_t a = av[j];                                              \
      for (m = 0; m < half; m++) {                                           \
        const uint32_t u = sel[2 * m], v = sel[2 * m + 1];                   \
        sel[m] = u ^ (a & (u ^ v));                                          \
      }                                                                      \
    }                                                                        \
    w = sel[0];                                                              \
    for (l = 0; l < take; l++) out[base + l] = (uint8_t)((w >> l) & 1u);     \
  }

void mdl_lut_layer_slice(const uint8_t *x, uint8_t *out, int32_t n_inputs,
                         int32_t out_width, const int32_t *indices,
                         const uint32_t *tt_slices) {
  MDL_LUT_LAYER_SLICE_BODY
}

/* uint16_t添字版。呼び出し側(cbackend)が in_width<=65536 のときだけ
 * 生成コードから呼ぶ。本体は上と1文字も変わらない(MDL_LUT_LAYER_SLICE_BODY
 * の展開で `indices` の型だけが差し替わる)。 */
void mdl_lut_layer_slice_u16(const uint8_t *x, uint8_t *out, int32_t n_inputs,
                             int32_t out_width, const uint16_t *indices,
                             const uint32_t *tt_slices) {
  MDL_LUT_LAYER_SLICE_BODY
}

/* 縮約済み dense 平坦ネットリストの表駆動評価。仕様は mdl_kernels.h。
 * 添字表の型(uint16_t / uint32_t)だけが違う2関数を、同じ本体マクロから
 * 作る(値は1ビットも変わらない)。 */
#define MDL_LUT_FLAT_BODY                                                    \
  const int32_t base = in_width + 1;                                         \
  int32_t g, i;                                                              \
  sig[0] = 0u;                                                               \
  for (i = 0; i < in_width; i++) sig[1 + i] = x[i];                          \
  for (g = 0; g < n_gates; g++) {                                            \
    const uint32_t addr = ((uint32_t)sig[b[g]] << 1) | (uint32_t)sig[a[g]];  \
    sig[base + g] = (uint8_t)(((uint32_t)tt[g] >> addr) & 1u);               \
  }                                                                          \
  for (i = 0; i < out_width; i++) {                                          \
    const uint32_t r = (uint32_t)o[i];                                       \
    out[i] = (uint8_t)(sig[r >> 1] ^ (r & 1u));                              \
  }

void mdl_lut_flat_u16(const uint8_t *x, uint8_t *out, uint8_t *sig,
                      int32_t in_width, int32_t n_gates, int32_t out_width,
                      const uint8_t *tt, const uint16_t *a, const uint16_t *b,
                      const uint16_t *o) {
  MDL_LUT_FLAT_BODY
}

void mdl_lut_flat_u32(const uint8_t *x, uint8_t *out, uint8_t *sig,
                      int32_t in_width, int32_t n_gates, int32_t out_width,
                      const uint8_t *tt, const uint32_t *a, const uint32_t *b,
                      const uint32_t *o) {
  MDL_LUT_FLAT_BODY
}

/* lut_flat のバッチ軸パック版。1ゲート = bw 語の butterfly-select。
 * 単枚版(上)は 1 ゲートごとに 2 バイト読み + 表引き + 1 バイト書きで、
 * 32 枚ぶん回すとこれを 32 回やる。こちらは同じ仕事を 32 枚 1 語で 1 回やる
 * (試作で HAR k6000 の網部分が 1 枚あたり 18.3µs → 1.2µs(bw=1)、
 * 0.14µs(bw=32)。入力の詰め替えを含めた実効は下の mdl_pack_batch を参照)。
 * bw==1 は語ループを外した特化(ESP32 のような小さいコアでの単語処理が主用途)。 */
#define MDL_LUT_FLAT_PACKED_BODY                                             \
  const size_t W = (size_t)bw;                                               \
  uint32_t *const gsig = sig + (size_t)(in_width + 1) * W;                   \
  int32_t g, i;                                                              \
  size_t w;                                                                  \
  for (w = 0; w < W; w++) sig[w] = 0u;                                       \
  if (x != sig + W) {                                                        \
    const size_t n_in = (size_t)in_width * W;                                \
    for (w = 0; w < n_in; w++) sig[W + w] = x[w];                            \
  }                                                                          \
  if (W == 1) {                                                              \
    for (g = 0; g < n_gates; g++) {                                          \
      const uint32_t t = (uint32_t)tt[g];                                    \
      const uint32_t t0 = 0u - (t & 1u), t1 = 0u - ((t >> 1) & 1u);          \
      const uint32_t t2 = 0u - ((t >> 2) & 1u), t3 = 0u - ((t >> 3) & 1u);   \
      const uint32_t av = sig[a[g]], bv = sig[b[g]];                         \
      const uint32_t u = t0 ^ (av & (t0 ^ t1)), v = t2 ^ (av & (t2 ^ t3));   \
      gsig[g] = u ^ (bv & (u ^ v));                                          \
    }                                                                        \
  } else {                                                                   \
    for (g = 0; g < n_gates; g++) {                                          \
      const uint32_t t = (uint32_t)tt[g];                                    \
      const uint32_t t0 = 0u - (t & 1u), t1 = 0u - ((t >> 1) & 1u);          \
      const uint32_t t2 = 0u - ((t >> 2) & 1u), t3 = 0u - ((t >> 3) & 1u);   \
      const uint32_t *pa = sig + (size_t)a[g] * W;                           \
      const uint32_t *pb = sig + (size_t)b[g] * W;                           \
      uint32_t *po = gsig + (size_t)g * W;                                   \
      for (w = 0; w < W; w++) {                                              \
        const uint32_t av = pa[w], bv = pb[w];                               \
        const uint32_t u = t0 ^ (av & (t0 ^ t1)), v = t2 ^ (av & (t2 ^ t3)); \
        po[w] = u ^ (bv & (u ^ v));                                          \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  for (i = 0; i < out_width; i++) {                                          \
    const uint32_t r = (uint32_t)o[i];                                       \
    const uint32_t m = 0u - (r & 1u);                                        \
    const uint32_t *ps = sig + (size_t)(r >> 1) * W;                         \
    uint32_t *pd = out + (size_t)i * W;                                      \
    for (w = 0; w < W; w++) pd[w] = ps[w] ^ m;                               \
  }

void mdl_lut_flat_packed_u16(const uint32_t *x, uint32_t *out, uint32_t *sig,
                             int32_t in_width, int32_t n_gates,
                             int32_t out_width, const uint8_t *tt,
                             const uint16_t *a, const uint16_t *b,
                             const uint16_t *o, int32_t bw) {
  MDL_LUT_FLAT_PACKED_BODY
}

void mdl_lut_flat_packed_u32(const uint32_t *x, uint32_t *out, uint32_t *sig,
                             int32_t in_width, int32_t n_gates,
                             int32_t out_width, const uint8_t *tt,
                             const uint32_t *a, const uint32_t *b,
                             const uint32_t *o, int32_t bw) {
  MDL_LUT_FLAT_PACKED_BODY
}

void mdl_group_sum(const uint8_t *x, int32_t *out, int32_t in_width, int32_t k) {
  const int32_t gsz = in_width / k;
  int32_t g, i;
  for (g = 0; g < k; g++) {
    int32_t s = 0;
    const uint8_t *base = x + (size_t)g * gsz;
    for (i = 0; i < gsz; i++) s += base[i];
    out[g] = s;
  }
}

/* ------------------------------------------------------------------ */
/* dense のバッチ軸ビットパック */
/* ------------------------------------------------------------------ */

/* 32x32 ビット行列の転置(LSB-first):m[r] のビット c → m[c] のビット r。
 * 交換を 16→8→4→2→1 の 5 段で行う再帰ブロック転置(Hacker's Delight 7-3 の
 * 向きを LSB-first に合わせたもの)。160 回ほどの語演算で 1024 ビットを動かす。 */
static void mdl_transpose32(uint32_t *m) {
  int32_t j, k;
  uint32_t mask = 0x0000ffffu, t;
  for (j = 16; j != 0; j >>= 1, mask ^= mask << j) {
    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
      t = ((m[k] >> j) ^ m[k + j]) & mask;
      m[k] ^= t << j;
      m[k + j] ^= t;
    }
  }
}

void mdl_pack_batch(const uint8_t *inputs, uint32_t *x, int32_t width,
                    int32_t batch) {
  /* 「32 枚 × 32 入力」の塊ごとに、各枚の連続 32 バイトを `mdl_mask32`
   * (SIMD なら 1〜2 命令)で 1 語にし、32x32 ビット転置で「入力ごとに 32 枚」
   * の語へ向きを変える。以前は列ごとに 32 枚ぶんを**枚の幅おきに**拾う
   * ギャザー(bw==1)/ ビットごとの分岐つき OR(bw>1)で、幅 4,608・32 枚で
   * 1 枚あたり 2.5µs、bw が増えると 5〜7µs かかり、平坦化回路(lut_flat)の
   * パック経路では網の評価より詰め替えのほうが重かった(試作の実測)。
   * こちらは連続読みだけで 0.6µs 前後。値は以前の定義と 1 ビットも変わらない
   * (端数の枚 >= batch のビットは 0。tests/deploy/test_lut_slice.py で固定)。 */
  const int32_t bwords = (batch + 31) >> 5;
  int32_t sw, jb, s, l;
  uint32_t blk[32];
  uint8_t tmp[32];
  for (sw = 0; sw < bwords; sw++) {
    const int32_t s0 = sw << 5;
    const int32_t ns = (batch - s0 < 32) ? (batch - s0) : 32;
    for (jb = 0; jb < width; jb += 32) {
      const int32_t take = (width - jb < 32) ? (width - jb) : 32;
      for (s = 0; s < ns; s++) {
        const uint8_t *row = inputs + (size_t)(s0 + s) * width + jb;
        if (take == 32) {
          blk[s] = mdl_mask32(row);
        } else {
          for (l = 0; l < take; l++) tmp[l] = row[l];
          for (; l < 32; l++) tmp[l] = 0u;
          blk[s] = mdl_mask32(tmp);
        }
      }
      for (; s < 32; s++) blk[s] = 0u;
      mdl_transpose32(blk);
      for (l = 0; l < take; l++) x[(size_t)(jb + l) * bwords + sw] = blk[l];
    }
  }
}

/* `mdl_lut_layer_packed` / `mdl_lut_layer_packed_u16` の共通本体。
 * MDL_LUT_LAYER_SLICE_BODYと同じ理由で、添字配列の型を持ち出さず直接
 * `indices[...]` で読む(囲む関数の引数型がそのまま伝わる)。 */
#define MDL_LUT_LAYER_PACKED_BODY                                            \
  const int32_t n_min = (int32_t)1 << n_inputs;                              \
  int32_t o, w, j, m;                                                        \
  if (n_inputs == 2) {                                                       \
    /* DLGN の素の形(2入力ゲート)の特化。汎用ループ(下、最小項の \
     * 論理和を分岐つきで足し込む)は n_inputs をコンパイル時定数にしておらず、\
     * 踏んだ「段数が実行時値だと展開されない」穴をそのまま持っていた \
     * ——実測(RDDLGN M-layer相当、n=20000〜480000、batch=16/32)で         \
     * 汎用ループは T回の mdl_lut_layer_slice 呼びより**遅い**(0.74〜0.99x)。\
     * mdl_lut_layer_sliceと同じ butterfly-select                      \
     * (`u=t0^(a&(t0^t1))`)へ、ゲートの4bit表を全ビット複製した4ワードとして\
     * 適用すると、分岐無しの3xor+2andまで縮む。同一入力でgeneric経路・     \
     * T回sliceの転置と完全一致することを確認済み(tests/deploy/            \
     * test_batch.py、test_lut_slice.pyの既存dense_batchテスト+            \
     * tests/deploy/test_c_rddlgn_packed_tf.py)。実測 9.3〜14.7倍           \
     * (charlm/paper_scale1規模)。 */                                       \
    for (o = 0; o < out_width; o++) {                                        \
      const uint32_t tt = truth_tables[o];                                   \
      const uint32_t t0 = (tt & 1u) ? 0xffffffffu : 0u;                      \
      const uint32_t t1 = (tt & 2u) ? 0xffffffffu : 0u;                      \
      const uint32_t t2 = (tt & 4u) ? 0xffffffffu : 0u;                      \
      const uint32_t t3 = (tt & 8u) ? 0xffffffffu : 0u;                      \
      const uint32_t *src_a = x + (size_t)indices[o] * bwords;               \
      const uint32_t *src_b = x + (size_t)indices[out_width + o] * bwords;   \
      uint32_t *dst = out + (size_t)o * bwords;                              \
      for (w = 0; w < bwords; w++) {                                         \
        const uint32_t a = src_a[w], b = src_b[w];                           \
        const uint32_t u = t0 ^ (a & (t0 ^ t1));                             \
        const uint32_t v = t2 ^ (a & (t2 ^ t3));                             \
        dst[w] = u ^ (b & (u ^ v));                                          \
      }                                                                      \
    }                                                                        \
    return;                                                                  \
  }                                                                          \
  for (o = 0; o < out_width; o++) {                                          \
    /* こちらは「1ゲートの表 × バッチ語ループ」なので、表はゲートあたり     \
     * 1語のまま読むのが速い。ビットスライス表現から組み直すと \
     * 実測で7%遅い(49,900 → 46,700 枚/秒)ので、単枚経路とは表を分ける。 */\
    const uint32_t tt = truth_tables[o];                                     \
    const uint32_t *src[5];                                                  \
    for (j = 0; j < n_inputs; j++) {                                         \
      src[j] = x + (size_t)indices[(size_t)j * out_width + o] * bwords;      \
    }                                                                        \
    for (w = 0; w < bwords; w++) {                                           \
      uint32_t v[5], acc = 0u;                                               \
      for (j = 0; j < n_inputs; j++) v[j] = src[j][w];                       \
      /* 真理値表を「立っている最小項の論理和」で評価する。                 \
       * 32画像ぶんが1つの語で同時に進む。 */                                \
      for (m = 0; m < n_min; m++) {                                          \
        uint32_t term;                                                       \
        if (((tt >> m) & 1u) == 0u) continue;                                \
        term = 0xffffffffu;                                                  \
        for (j = 0; j < n_inputs; j++) {                                     \
          term &= ((m >> j) & 1) ? v[j] : ~v[j];                             \
        }                                                                    \
        acc |= term;                                                         \
        if (acc == 0xffffffffu) break; /* 以降の最小項は結果を変えない */    \
      }                                                                      \
      out[(size_t)o * bwords + w] = acc;                                     \
    }                                                                        \
  }

void mdl_lut_layer_packed(const uint32_t *x, uint32_t *out, int32_t n_inputs,
                          int32_t out_width, const int32_t *indices,
                          const uint32_t *truth_tables, int32_t bwords) {
  MDL_LUT_LAYER_PACKED_BODY
}

/* uint16_t添字版。packed teacher forcing(RDDLGNのM-layers)が
 * in_width<=65536のときにこちらを呼ぶ。本体はmdl_lut_layer_packedと
 * 1文字も変わらない(MDL_LUT_LAYER_PACKED_BODYの展開で`indices`の型だけが
 * 差し替わる)。 */
void mdl_lut_layer_packed_u16(const uint32_t *x, uint32_t *out,
                              int32_t n_inputs, int32_t out_width,
                              const uint16_t *indices,
                              const uint32_t *truth_tables, int32_t bwords) {
  MDL_LUT_LAYER_PACKED_BODY
}

void mdl_group_sum_packed(const uint32_t *x, int32_t *scores, int32_t in_width,
                          int32_t k, int32_t batch, int32_t bwords) {
  const int32_t gsz = in_width / k;
  const int32_t n_planes = mdl_spatial_group_sum_planes(gsz);
  int32_t g, i, w, p, l;
  uint32_t planes[32];      /* gsz < 2^32 なので平面は高々32枚 */
  /* 1ワード=32画像なので、`mdl_spatial_group_sum` と同じ縦方向の
   * キャリーセーブ加算で **32画像ぶんのカウンタを同時に進める**。
   *
   * 以前は画像ごと・特徴ごとに1ビット抜き出して足していた。32画像が
   * 同じワードを共有しているのに同じワードを32回読み直す形で、実測で
   * パック経路の19%を食っていた(49,000 → 60,800 枚/秒。ゲート評価
   * 36,000個に対して集計は6,000ビットしか無いのに、である)。
   *
   * popcount 命令は使えない。1ワードの32ビットは別々の画像なので、
   * popcount だと画像をまたいで足してしまう(数える方向が90度違う)。
   *
   * 内側は「carry!=0の間だけ回す」早期終了ループだったが、carryが
   * いつ0になるかは入力依存で分岐予測が効かない。RDDLGNのM-layer
   * 出力(gsz=30〜358、n_planes=5〜9)に当ててみたところ、**早期終了を
   * 外し毎回n_planes全部を触る分岐無しループの方が速い**(実測1.46〜
   * 1.47倍、gsz=30/358の両方で再現。値は完全一致 — tests/deploy/
   * test_batch.py、test_c_rddlgn_packed_tf.pyで固定)。「分岐は少ない方が
   * 良い」の逆で、carryの成否が読めない以上、固定回数を回し切る方が
   * 分岐ミス予測のコストを避けられる。 */
  for (w = 0; w < bwords; w++) {
    const int32_t base_b = w << 5;
    const int32_t lanes = (batch - base_b < 32) ? (batch - base_b) : 32;
    for (g = 0; g < k; g++) {
      const uint32_t *src = x + (size_t)g * gsz * bwords + w;
      for (p = 0; p < n_planes; p++) planes[p] = 0u;
      for (i = 0; i < gsz; i++) {
        uint32_t carry = src[(size_t)i * bwords];
        for (p = 0; p < n_planes; p++) {
          const uint32_t t = planes[p] & carry;
          planes[p] ^= carry;
          carry = t;
        }
      }
      /* ビットスライス平面 → 画像ごとのスコアへ転置 */
      for (l = 0; l < lanes; l++) {
        int32_t s = 0;
        for (p = 0; p < n_planes; p++) {
          s |= (int32_t)((planes[p] >> l) & 1u) << p;
        }
        scores[(size_t)(base_b + l) * k + g] = s;
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 前処理(画像 u8 → ビットプレーン)                                   */
/* ------------------------------------------------------------------ */

/* 記述子のcut2語(int64二の補数)を復元する。 */
static int64_t preproc_cut(const int32_t *d) {
  const uint32_t lo = (uint32_t)d[5];
  const int32_t hi = d[6];
  return (int64_t)hi * (int64_t)4294967296 + (int64_t)lo;
}

static int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

/* ---- 差分比較を「u8のしきい値比較」へ落とす ----------------------------
 *
 * I(k) は k について**厳密に単調増加**する(mdl_preproc の前提。Python側
 * test_quant_table_monotonic が固定している)。よって
 *
 *     I(a) - I(b) >= cut   ⟺   a >= thr_ge[b]
 *     I(a) - I(b) <= cut   ⟺   a <  thr_lt[b]
 *
 * と、b ごとの1バイトしきい値へ厳密に還元できる。しきい値は b について
 * 単調なので、二重ループではなく**2ポインタで O(512)** で作れる
 * (画素数16384に対して無視できる)。
 * これで内側ループから int64 演算が消え、u8 の比較だけになる。 */
static void build_pair_threshold(const uint32_t *qval, int64_t cut,
                                 int32_t dir_le, int32_t *thr) {
  int32_t b, a;
  if (!dir_le) {
    /* thr[b] = min{ a : I(a) >= cut + I(b) }(無ければ256=常に偽) */
    a = 0;
    for (b = 0; b < 256; b++) {
      const int64_t need = cut + (int64_t)qval[b];
      while (a < 256 && (int64_t)qval[a] < need) a++;
      thr[b] = a;
    }
  } else {
    /* thr[b] = max{ a : I(a) <= cut + I(b) } + 1(無ければ0=常に偽) */
    a = 0;
    for (b = 0; b < 256; b++) {
      const int64_t limit = cut + (int64_t)qval[b];
      while (a < 256 && (int64_t)qval[a] <= limit) a++;
      thr[b] = a;   /* a は「条件を満たす最大値+1」 */
    }
  }
}

/* しきい値表を u8 化したもの + 飽和の開始位置。
 * thr[b] は [0,256] だが **b について単調非減少** なので、256 になるのは
 * b >= sat_from の範囲だけ。これで表そのものは u8 に収まり、
 * 16画素を1命令ぶんで引けるようになる(それまではここが完全スカラーで、
 * EDGE 1チャネルが thermometer の 8.7 倍かかっていた)。 */
typedef struct {
  uint8_t v[256];
  int32_t sat_from;      /* thr[b] == 256 となる最小の b(無ければ 256) */
#if defined(MDL_HAS_PAIR_GATHER)
  mdl_tbl256 tbl;
#endif
#if defined(MDL_HAS_PAIR_GATHER_AVX2)
  mdl_tbl256_avx2 tbl_avx2;   /*: x86 AVX2版(NEON分岐とは別名・並記) */
#endif
} mdl_pair_tbl;

/* u8化した表から元の [0,256] の値を復元する(端数処理用)。 */
static int32_t pair_thr(const mdl_pair_tbl *t, uint8_t b) {
  return b >= t->sat_from ? 256 : (int32_t)t->v[b];
}

static void pair_tbl_build(mdl_pair_tbl *t, const int32_t *thr) {
  int32_t b;
  t->sat_from = 256;
  for (b = 0; b < 256; b++) {
    if (thr[b] >= 256) {
      if (t->sat_from == 256) t->sat_from = b;
      t->v[b] = 255u;
    } else {
      t->v[b] = (uint8_t)thr[b];
    }
  }
#if defined(MDL_HAS_PAIR_GATHER)
  mdl_tbl256_load(&t->tbl, t->v);
#endif
#if defined(MDL_HAS_PAIR_GATHER_AVX2)
  mdl_tbl256_avx2_load(&t->tbl_avx2, t->v);
#endif
}

/* 1ワード(最大32画素)ぶんの「a >= thr[b]」/「a < thr[b]」を組む。 */
static uint32_t pair_word(const uint8_t *pa, const uint8_t *pb,
                          const mdl_pair_tbl *t, int32_t dir_le, int32_t n) {
  uint32_t acc = 0u;
  int32_t i = 0;
#if defined(MDL_HAS_PAIR_GATHER)
  {
    const uint8x16_t sat = vdupq_n_u8((uint8_t)(t->sat_from > 255
                                                ? 255 : t->sat_from));
    /* sat_from==256 のときは「飽和なし」。u8 では表せないので、
       その場合だけ飽和マスクを使わない経路にする。 */
    if (t->sat_from > 255) {
      for (; i + 16 <= n; i += 16) {
        const uint8x16_t ge = vcgeq_u8(vld1q_u8(pa + i),
                                       mdl_tbl256_get(&t->tbl,
                                                      vld1q_u8(pb + i)));
        acc |= mdl_movemask_u8(dir_le ? vmvnq_u8(ge) : ge) << i;
      }
    } else {
      for (; i + 16 <= n; i += 16) {
        acc |= mdl_pair_mask16(pa + i, pb + i, &t->tbl, sat, dir_le) << i;
      }
    }
  }
#elif defined(MDL_HAS_PAIR_GATHER_AVX2)
  /* NEON分岐(上)と対称のAVX2版。1レジスタ32バイト=32画素ぶん
   * 処理できるので、NEON(16画素×最大2回)と違いこのループは高々1回で
   * 終わる(呼び出し元で n<=32 が保証されている)。 */
  {
    const __m256i sat = _mm256_set1_epi8((char)(uint8_t)(t->sat_from > 255
                                                          ? 255 : t->sat_from));
    if (t->sat_from > 255) {
      for (; i + 32 <= n; i += 32) {
        acc |= mdl_pair_mask32_nosat_avx2(pa + i, pb + i, &t->tbl_avx2,
                                          dir_le) << i;
      }
    } else {
      for (; i + 32 <= n; i += 32) {
        acc |= mdl_pair_mask32_avx2(pa + i, pb + i, &t->tbl_avx2, sat,
                                    dir_le) << i;
      }
    }
  }
#endif
  /* 端数(と SIMD 無し環境)。**ベクタ版と同じ値を返す**ことがテストで固定。
   * 比較結果を 0/1 のまま シフトして積む(分岐を書かない)。
   * 実画像は画素値がほぼランダムなので、ここを if にすると分岐予測が
   * 半分外れて実測で2倍以上遅くなる(合成パターンでは気づけない)。 */
  if (!dir_le) {
    for (; i < n; i++) {
      acc |= (uint32_t)((int32_t)pa[i] >= pair_thr(t, pb[i])) << i;
    }
  } else {
    for (; i < n; i++) {
      acc |= (uint32_t)((int32_t)pa[i] < pair_thr(t, pb[i])) << i;
    }
  }
  return acc;
}

/* thermometer(単調 → k>=k0 の1閾値)。ここだけで実モデルの全チャネルの
 * 過半数を占めるので、16画素=1命令のベクタ比較を効かせる。 */
static void preproc_step_plane(const uint8_t *src, uint8_t k0,
                               int32_t h, int32_t w, int32_t wpr,
                               uint32_t *plane) {
  int32_t y, j, i;
  for (y = 0; y < h; y++) {
    const uint8_t *row = src + (size_t)y * w;
    uint32_t *dst = plane + (size_t)y * wpr;
    for (j = 0; j < wpr; j++) {
      const int32_t x0 = j * 32;
      const int32_t n = (w - x0) < 32 ? (w - x0) : 32;
      uint32_t acc = 0u;
      i = 0;
      for (; i + 16 <= n; i += 16) {
        acc |= mdl_ge_mask16(row + x0 + i, k0) << i;
      }
      for (; i < n; i++) {
        acc |= (uint32_t)(row[x0 + i] >= k0) << i;
      }
      dst[j] = acc;
    }
  }
}

/* thermometer(階段に畳めなかった場合)。256ビットLUT参照。 */
static void preproc_lut_plane(const uint8_t *src, const int32_t *lut,
                              int32_t h, int32_t w, int32_t wpr,
                              uint32_t *plane) {
  int32_t y, j, i;
  for (y = 0; y < h; y++) {
    const uint8_t *row = src + (size_t)y * w;
    uint32_t *dst = plane + (size_t)y * wpr;
    for (j = 0; j < wpr; j++) {
      const int32_t x0 = j * 32;
      const int32_t n = (w - x0) < 32 ? (w - x0) : 32;
      uint32_t acc = 0u;
      i = 0;
      for (; i + 16 <= n; i += 16) {
        acc |= mdl_lut_mask16(row + x0 + i, lut) << i;
      }
      for (; i < n; i++) {
        const uint32_t k = row[x0 + i];
        acc |= ((((uint32_t)lut[k >> 5]) >> (k & 31u)) & 1u) << i;
      }
      dst[j] = acc;
    }
  }
}

void mdl_preproc_exec(const uint8_t *image, int32_t c_img, int32_t h, int32_t w,
                      const int32_t *desc, int32_t channels,
                      const uint32_t *qval, uint32_t *out) {
  mdl_preproc_exec_range(image, c_img, h, w, desc, 0, 1, channels, qval, out);
}

void mdl_preproc_exec_range(const uint8_t *image, int32_t c_img, int32_t h,
                            int32_t w, const int32_t *desc, int32_t ch_start,
                            int32_t ch_stride, int32_t channels,
                            const uint32_t *qval, uint32_t *out) {
  const int32_t wpr = mdl_words_per_row(w);
  const size_t plane_px = (size_t)h * w;
  int32_t ch, y, j, i;
  int32_t thr[256];
  mdl_pair_tbl ptbl;

  for (ch = ch_start; ch < channels; ch += ch_stride) {
    const int32_t *d = desc + (size_t)ch * MDL_PREPROC_STRIDE;
    const int32_t op = d[0];
    const int32_t ca = d[1], cb = d[2];
    const int32_t dy = d[3], dx = d[4];
    const int32_t dir_le = d[7];
    uint32_t *plane = out + (size_t)ch * h * wpr;

    /* 種別の分岐はチャネル単位で済ませる(以前は画素ごとに判定していた)。 */
    if (op == MDL_PREPROC_OP_THERMO_STEP) {
      preproc_step_plane(image + (size_t)ca * plane_px, (uint8_t)cb,
                         h, w, wpr, plane);
      continue;
    }
    if (op == MDL_PREPROC_OP_THERMO) {
      preproc_lut_plane(image + (size_t)ca * plane_px, d + 8, h, w, wpr, plane);
      continue;
    }

    if (op != MDL_PREPROC_OP_CURV) {
      /* EDGE / CHROMA / SAT: しきい値表へ還元して u8 比較だけにする。 */
      const int64_t cut = preproc_cut(d);
      /* 水平方向は端の1〜数画素だけが replicate の対象なので、
       * 「語がまるごと内側か」で高速路と境界路を分ける。 */
      const int32_t lo = dx > 0 ? 0 : -dx;
      const int32_t hi = dx > 0 ? w - dx : w;
      build_pair_threshold(qval, cut, dir_le, thr);
      pair_tbl_build(&ptbl, thr);

      for (y = 0; y < h; y++) {
        const uint8_t *rb = image + (size_t)ca * plane_px + (size_t)y * w;
        const uint8_t *ra;
        uint32_t *dst = plane + (size_t)y * wpr;
        if (op == MDL_PREPROC_OP_EDGE) {
          ra = image + (size_t)ca * plane_px
             + (size_t)clamp_i32(y + dy, 0, h - 1) * w;
        } else if (op == MDL_PREPROC_OP_CHROMA) {
          ra = rb;
          rb = image + (size_t)cb * plane_px + (size_t)y * w;
        } else {
          ra = NULL;   /* SAT は下で画素ごとに max/min を作る */
        }

        for (j = 0; j < wpr; j++) {
          const int32_t x0 = j * 32;
          const int32_t n = (w - x0) < 32 ? (w - x0) : 32;
          uint32_t acc = 0u;
          if (op == MDL_PREPROC_OP_SAT) {
            /* 画素ごとの max/min を先に作ってから、EDGE と同じ
             * 「a >= thr[b]」へ落とす。max/min はチャネル方向の縦計算なので
             * そのままベクタ化できる。 */
            const uint8_t *base = image + (size_t)y * w + x0;
            uint8_t mxv[32], mnv[32];
            int32_t c;
            for (i = 0; i < n; i++) { mxv[i] = 0u; mnv[i] = 255u; }
            for (c = 0; c < c_img; c++) {
              const uint8_t *src = base + (size_t)c * plane_px;
              i = 0;
#if defined(MDL_HAS_PAIR_GATHER)
              for (; i + 16 <= n; i += 16) {
                const uint8x16_t v = vld1q_u8(src + i);
                vst1q_u8(mxv + i, vmaxq_u8(vld1q_u8(mxv + i), v));
                vst1q_u8(mnv + i, vminq_u8(vld1q_u8(mnv + i), v));
              }
#elif defined(MDL_HAS_PAIR_GATHER_AVX2)
              /* NEON分岐(上)と対称のAVX2版(32バイト/命令)。*/
              for (; i + 32 <= n; i += 32) {
                const __m256i v =
                    _mm256_loadu_si256((const __m256i *)(const void *)(src + i));
                _mm256_storeu_si256(
                    (__m256i *)(void *)(mxv + i),
                    _mm256_max_epu8(
                        _mm256_loadu_si256((const __m256i *)(const void *)(mxv + i)),
                        v));
                _mm256_storeu_si256(
                    (__m256i *)(void *)(mnv + i),
                    _mm256_min_epu8(
                        _mm256_loadu_si256((const __m256i *)(const void *)(mnv + i)),
                        v));
              }
#endif
              for (; i < n; i++) {
                const uint8_t v = src[i];
                if (v > mxv[i]) mxv[i] = v;
                if (v < mnv[i]) mnv[i] = v;
              }
            }
            acc = pair_word(mxv, mnv, &ptbl, dir_le, n);
          } else if (op == MDL_PREPROC_OP_CHROMA
                     || (x0 + dx >= lo && x0 + n + dx <= hi)) {
            /* 内側: 境界検査なしでまとめて処理する。 */
            acc = pair_word(ra + x0 + (op == MDL_PREPROC_OP_EDGE ? dx : 0),
                            rb + x0, &ptbl, dir_le, n);
          } else {
            for (i = 0; i < n; i++) {
              const int32_t xa = clamp_i32(x0 + i + dx, 0, w - 1);
              const int32_t a = ra[xa];
              const int32_t t = thr[rb[x0 + i]];
              acc |= (uint32_t)(dir_le ? (a < t) : (a >= t)) << i;
            }
          }
          dst[j] = acc;
        }
      }
      continue;
    }

    /* CURV: I(a) - 2I(m) + I(c) の3項なので2次元のしきい値表には落ちない。
     * 行ポインタと境界判定をループの外へ出して int64 のまま回す。 */
    {
      const int64_t cut = preproc_cut(d);
      const uint8_t *base = image + (size_t)ca * plane_px;
      const int32_t lo = dx > 0 ? dx : -dx;    /* |dx| */
      /* 中央項をしきい値へ畳む: Q[a] + Q[c] >= cut + 2*Q[m] ⟺ ... 。
       * 画素ごとの乗算と2回の加減算が、表引き1回と比較1回になる。 */
      int64_t tmid[256];
      for (i = 0; i < 256; i++) {
        tmid[i] = cut + 2 * (int64_t)qval[i];
      }
      for (y = 0; y < h; y++) {
        const uint8_t *rm = base + (size_t)y * w;
        const uint8_t *rp = base + (size_t)clamp_i32(y + dy, 0, h - 1) * w;
        const uint8_t *rn = base + (size_t)clamp_i32(y - dy, 0, h - 1) * w;
        uint32_t *dst = plane + (size_t)y * wpr;
        for (j = 0; j < wpr; j++) {
          const int32_t x0 = j * 32;
          const int32_t n = (w - x0) < 32 ? (w - x0) : 32;
          const int32_t inner = (x0 - lo >= 0) && (x0 + n + lo <= w);
          uint32_t acc = 0u;
          if (inner) {
            /* 境界判定をループの外へ出す。内側の語では clamp が要らない
             * ので、3本の連続ストリームを素直に舐めるだけになる。 */
            const uint8_t *pn = rn + x0 - dx;
            const uint8_t *pp = rp + x0 + dx;
            const uint8_t *pm = rm + x0;
            for (i = 0; i < n; i++) {
              const int64_t s = (int64_t)qval[pn[i]] + (int64_t)qval[pp[i]];
              acc |= (uint32_t)(dir_le ? (s <= tmid[pm[i]])
                                       : (s >= tmid[pm[i]])) << i;
            }
          } else {
            for (i = 0; i < n; i++) {
              const int32_t x = x0 + i;
              const int32_t xp = clamp_i32(x + dx, 0, w - 1);
              const int32_t xn = clamp_i32(x - dx, 0, w - 1);
              const int64_t s = (int64_t)qval[rn[xn]] + (int64_t)qval[rp[xp]];
              acc |= (uint32_t)(dir_le ? (s <= tmid[rm[x]])
                                       : (s >= tmid[rm[x]])) << i;
            }
          }
          dst[j] = acc;
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 読み出し                                                            */
/* ------------------------------------------------------------------ */

int32_t mdl_argmax(const int32_t *scores, int32_t n) {
  int32_t best = 0, i;
  for (i = 1; i < n; i++) {
    if (scores[i] > scores[best]) best = i; /* 同点は先勝ち */
  }
  return best;
}

void mdl_argmax_spatial(const int32_t *scores, int32_t n_out, int32_t n_pixels,
                        uint8_t *labels) {
  int32_t p, o;
  for (p = 0; p < n_pixels; p++) {
    int32_t best = 0;
    int32_t best_v = scores[p];
    for (o = 1; o < n_out; o++) {
      const int32_t v = scores[(size_t)o * n_pixels + p];
      if (v > best_v) { best_v = v; best = o; }
    }
    labels[p] = (uint8_t)best;
  }
}
