/* griffith-logic — Metalカーネルの入口
 *
 * 計算式そのものは runtime/gpu/mdl_gpu_kernels.h(CUDAと共有)にあり、
 * このファイルは「Metalの引数バインディングに合わせた殻」だけを持つ。
 * CUDA版(mdl_cuda.cu)との違いは:
 *
 *   - Metal は構造体メンバにデバイスポインタを持てないので、conv の
 *     レイアウトはスカラーだけの構造体 + 配列を別バッファで束ねる
 *   - アドレス空間修飾子(device / constant)が必須。共有ヘッダは
 *     MDL_GLOBAL / MDL_CONST マクロでそこを吸収している
 *
 * このソースは実行時に newLibraryWithSource: でコンパイルする
 * (オフラインの metal ツールチェーンを必要としない)。
 */

/* conv のスカラーパラメータ。配列は別バッファで渡す。 */
struct MdlConvParams {
  int h, w, h_out, w_out;
  int s_h, s_w, p_h, p_w;
  int out_channels;
  int wpr_in, wpr_out, tap_wpr, tap_rows_words;
  uint mask_out;
  int use_fast_path;
  int n_planes, plane_stride, plane_rows;
  int pool_base, n_instrs, n_write0;
  /* バッチ用のストライド(1枚ぶんのワード数)。 */
  int in_plane_words, out_plane_words, scratch_words;
};

struct MdlGeom {
  int c, h, w, wpr, h_out, w_out, wpr_out, extra;
  /* バッチ枚数と「1枚ぶんの入力要素数」(枚ごとに表を引き直すカーネル用)。
     チャネル方向に独立なカーネルは [batch][C][H][wpr] の並びのおかげで
     batch を見る必要がなく、チャネル数に batch を掛けて渡すだけでよい。 */
  int batch, in_stride;
};

/* copy_batched_exのバッチ一括コピー(mdl_gpu_kernels.hの
 * mdl_g_copy_strided_byte参照)。ホスト側(mdl_metal.mm)の
 * MdlCopyStridedParamsと同一レイアウトであること(MdlGeomと同じ約束)。 */
struct MdlCopyStridedParams {
  int dst_off, dst_stride, src_off, src_stride, n_bytes, batch;
};

kernel void mdl_k_copy_strided(device const uchar *src [[buffer(0)]],
                               device uchar *dst [[buffer(1)]],
                               constant MdlCopyStridedParams &p [[buffer(2)]],
                               uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= p.n_bytes * p.batch) return;
  mdl_g_copy_strided_byte(src, dst, p.dst_off, p.dst_stride, p.src_off,
                          p.src_stride, p.n_bytes, (int)idx);
}

kernel void mdl_k_pack(device const uchar *src [[buffer(0)]],
                       device uint *dst [[buffer(1)]],
                       constant MdlGeom &g [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
  const int rows = g.h * g.wpr;
  const int ch = (int)idx / rows;
  const int rem = (int)idx - ch * rows;
  const int y = rem / g.wpr;
  const int j = rem - y * g.wpr;
  const int x0 = j * 32;
  const int n = (g.w - x0) < 32 ? (g.w - x0) : 32;
  device const uchar *row = src + (ch * g.h + y) * g.w;
  uint acc = 0u;
  if ((int)idx >= g.c * rows) return;
  for (int i = 0; i < n; i++) acc |= (uint)(row[x0 + i] != 0) << i;
  dst[idx] = acc;
}

kernel void mdl_k_unpack(device const uint *src [[buffer(0)]],
                         device uchar *dst [[buffer(1)]],
                         constant MdlGeom &g [[buffer(2)]],
                         uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= g.c * g.h * g.w) return;
  {
    const int x = (int)idx % g.w;
    const int rest = (int)idx / g.w;
    const int y = rest % g.h;
    const int ch = rest / g.h;
    dst[idx] = (uchar)((src[(ch * g.h + y) * g.wpr + (x >> 5)] >> (x & 31)) & 1u);
  }
}

kernel void mdl_k_preproc(device const uchar *image [[buffer(0)]],
                          device uint *out [[buffer(1)]],
                          constant int *desc [[buffer(2)]],
                          constant short *thr [[buffer(3)]],
                          constant uint *qval [[buffer(4)]],
                          constant MdlGeom &g [[buffer(5)]],
                          uint idx [[thread_position_in_grid]]) {
  const int per = g.extra * g.h * g.wpr;
  if ((int)idx >= per * g.batch) return;
  {
    const int b = (int)idx / per;               /* バッチ内の何枚目か */
    const int k = (int)idx - b * per;
    out[idx] = mdl_g_preproc_word(image + b * g.c * g.h * g.w,
                                  g.c, g.h, g.w, g.wpr, desc, thr, qval, k);
  }
}

kernel void mdl_k_conv_const(device uint *scratch [[buffer(0)]],
                             constant MdlConvParams &p [[buffer(1)]],
                             constant int &total [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= total) return;
  {
    const int b = (int)gid / p.tap_rows_words;   /* バッチ内の何枚目か */
    const int i = (int)gid - b * p.tap_rows_words;
    device uint *sc = scratch + b * p.scratch_words;
    sc[i] = 0u;
    sc[p.tap_rows_words + i] = 0xffffffffu;
  }
}

kernel void mdl_k_conv_planes(device const uint *in [[buffer(0)]],
                              device uint *scratch [[buffer(1)]],
                              constant MdlConvParams &p [[buffer(2)]],
                              constant int *plane_c [[buffer(3)]],
                              constant int *plane_dy [[buffer(4)]],
                              constant int *plane_dx [[buffer(5)]],
                              constant int &total [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= total) return;
  {
    const int per = p.n_planes * p.plane_stride;
    const int b = (int)gid / per;               /* バッチ内の何枚目か */
    const int t = (int)gid - b * per;
    device const uint *in_b = in + b * p.in_plane_words;
    device uint *sc = scratch + b * p.scratch_words;
    const int u = t / p.plane_stride;
    const int k = t - u * p.plane_stride;
    const int c_base = plane_c[u] * p.h * p.wpr_in;
    uint v;
    if (p.use_fast_path) {
      v = mdl_g_shift_word(in_b, c_base, p.h, p.wpr_in, p.p_h,
                           plane_dx[u] - p.p_w, k);
    } else {
      v = mdl_g_tap_word_strided(in_b, c_base, p.h, p.w, p.wpr_in, plane_dy[u],
                                 plane_dx[u], p.w_out, p.wpr_out, p.s_h,
                                 p.s_w, p.p_h, p.p_w, k);
    }
    sc[p.pool_base + t] = v;
  }
}

kernel void mdl_k_conv_exec(device uint *scratch [[buffer(0)]],
                            device uint *out [[buffer(1)]],
                            constant MdlConvParams &p [[buffer(2)]],
                            constant int *instr [[buffer(3)]],
                            constant int *oc_src [[buffer(4)]],
                            constant int *wstart [[buffer(5)]],
                            constant int *woc [[buffer(6)]],
                            constant int &total [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= total) return;
  {
    const int plane_words = p.tap_rows_words;
    const int n = (int)gid / plane_words;       /* バッチ内の何枚目か */
    const int i = (int)gid - n * plane_words;
    device uint *sc = scratch + n * p.scratch_words;
    device uint *ou = out + n * p.out_plane_words;
    const uint tail = ((i % p.wpr_out) == p.wpr_out - 1) ? p.mask_out
                                                         : 0xffffffffu;
    for (int b = wstart[0]; b < wstart[1]; b++) {
      const int oc = woc[b];
      ou[oc * plane_words + i] = sc[oc_src[oc] + i] & tail;
    }
    for (int j = 0; j < p.n_instrs; j++) {
      constant int *ins = instr + j * 4;
      sc[ins[3] + i] = mdl_g_gate2((uint)ins[0], sc[ins[1] + i],
                                   sc[ins[2] + i]);
      for (int b = wstart[j + 1]; b < wstart[j + 2]; b++) {
        const int oc = woc[b];
        ou[oc * plane_words + i] = sc[oc_src[oc] + i] & tail;
      }
    }
  }
}

/* ocごとの命令列分割。1スレッド=(batch, oc, ワード列)で、旧来の
 * mdl_k_conv_exec(1スレッド=(batch, ワード列)が全oc共通の命令列全体を
 * 直列実行)より並列度を out_channels 倍に増やす。命令オペランド
 * (`instr[k*4+1]`/`instr[k*4+2]`)は `(value<<1)|is_reg` にパック済み:
 * bit0=0 なら共有スクラッチ(定数0/1平面+タップ平面プール、全スレッド
 * 共通の読み取り専用データ)の絶対ワードオフセット、bit0=1 なら
 * このoc専用のスレッドローカルレジスタ配列 `slots` の添字。
 * `oc_final_ref` も同じパック規則(最終値がconst/tap直結ならMEM、
 * 命令の結果ならREG)。ocの依存閉包はトポロジカル順の部分列なので
 * write_start/write_ocのような割込みスケジュールは要らず、
 * 命令を順に実行して最後に oc_final_ref を読むだけでよい
 * (`griffith_logic/deploy/layout.py::split_program_by_oc` のコメント
 * 参照 — 各ocの最終値は必ずローカル命令列の末尾になる)。 */
kernel void mdl_k_conv_exec_split(device const uint *scratch [[buffer(0)]],
                                  device uint *out [[buffer(1)]],
                                  constant MdlConvParams &p [[buffer(2)]],
                                  constant int *oc_instr_start [[buffer(3)]],
                                  constant int *instr [[buffer(4)]],
                                  constant int *oc_final_ref [[buffer(5)]],
                                  constant int &total [[buffer(6)]],
                                  uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= total) return;
  {
    const int w = p.tap_rows_words;           /* ocあたりのワード数 */
    const int n = (int)gid / (p.out_channels * w);   /* バッチ内の何枚目か */
    const int rem = (int)gid - n * (p.out_channels * w);
    const int oc = rem / w;
    const int i = rem - oc * w;
    device const uint *sc = scratch + n * p.scratch_words;
    device uint *ou = out + n * p.out_plane_words;
    const uint tail = ((i % p.wpr_out) == p.wpr_out - 1) ? p.mask_out
                                                         : 0xffffffffu;
    thread uint slots[MDL_CONV_SPLIT_MAX_SLOTS];
    const int start = oc_instr_start[oc];
    const int end = oc_instr_start[oc + 1];
    for (int k = start; k < end; k++) {
      constant int *ins = instr + k * 4;
      const int ap = ins[1];
      const int bp = ins[2];
      const uint av = (ap & 1) ? slots[ap >> 1] : sc[(ap >> 1) + i];
      const uint bv = (bp & 1) ? slots[bp >> 1] : sc[(bp >> 1) + i];
      slots[ins[3]] = mdl_g_gate2((uint)ins[0], av, bv);
    }
    {
      const int fr = oc_final_ref[oc];
      const uint v = (fr & 1) ? slots[fr >> 1] : sc[(fr >> 1) + i];
      ou[oc * w + i] = v & tail;
    }
  }
}

kernel void mdl_k_or_pool(device const uint *in [[buffer(0)]],
                          device uint *out [[buffer(1)]],
                          constant MdlGeom &g [[buffer(2)]],
                          uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= g.c * g.h_out * g.wpr_out) return;
  out[idx] = mdl_g_or_pool_word(in, g.h, g.w, g.wpr, g.h_out, g.w_out,
                                g.wpr_out, (int)idx);
}

kernel void mdl_k_upsample(device const uint *in [[buffer(0)]],
                           device uint *out [[buffer(1)]],
                           constant MdlGeom &g [[buffer(2)]],
                           uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= g.c * (2 * g.h) * g.wpr_out) return;
  out[idx] = mdl_g_upsample_word(in, g.h, g.w, g.wpr, g.wpr_out, (int)idx);
}

kernel void mdl_k_permute(device const uint *in [[buffer(0)]],
                          device uint *out [[buffer(1)]],
                          constant int *order [[buffer(2)]],
                          constant MdlGeom &g [[buffer(3)]],
                          uint idx [[thread_position_in_grid]]) {
  const int ch_words = g.h * g.wpr;
  if ((int)idx >= g.c * g.batch * ch_words) return;
  {
    const int o = (int)idx / ch_words;
    const int k = (int)idx - o * ch_words;
    const int b = o / g.c;                      /* バッチ内の何枚目か */
    const int oc = o - b * g.c;
    out[idx] = in[(b * g.c + order[oc]) * ch_words + k];
  }
}

kernel void mdl_k_sgs(device const uint *in [[buffer(0)]],
                      device int *scores [[buffer(1)]],
                      constant MdlGeom &g [[buffer(2)]],
                      uint idx [[thread_position_in_grid]]) {
  /* g.c=n_out, g.extra=gs, g.h_out=n_planes */
  if ((int)idx >= g.c * g.h * g.wpr) return;
  mdl_g_spatial_group_sum_word(in, scores, g.c, g.extra, g.h, g.w, g.wpr,
                               g.h_out, (int)idx);
}

/* 並列縮約版(mdl_gpu_kernels.hのコメント参照)。1スレッドグループ
 * = 出力ワード1個(o,y,j)、グループ内のRレーン(=スレッドグループ幅)が
 * gs 方向をチャンク分担してCSAし、threadgroupメモリで総和してから書き出す。
 * mdl_metal_spatial_group_sum が work_items(=n_out*h*wpr)が小さいときだけ
 * こちらを選ぶ(dispatchThreadgroups、host側で明示的にグループ整列)。
 * g.c=gs, g.h_out=n_planes(チャンク側), g.w_out=chunk, g.wpr_out=R
 * (thread_position_in_threadgroup等の組込み値に頼らず、host側が選んだRを
 * そのままデータとして渡す — CUDA版が blockDim.x を直接読むのと違い、
 * Metalの組込み属性の型/次元まわりの不確実性を避けるための選択)。 */
kernel void mdl_k_sgs_r(device const uint *in [[buffer(0)]],
                        device int *scores [[buffer(1)]],
                        constant MdlGeom &g [[buffer(2)]],
                        threadgroup int *sdata [[threadgroup(0)]],
                        uint lane [[thread_position_in_threadgroup]],
                        uint wi [[threadgroup_position_in_grid]]) {
  const int R = g.wpr_out;
  const int rows = g.h * g.wpr;
  const int o = (int)wi / rows;
  const int rem = (int)wi - o * rows;
  const int y = rem / g.wpr;
  const int j = rem - y * g.wpr;
  const int g_lo = (int)lane * g.w_out;
  const int g_hi = min(g_lo + g.w_out, g.c);
  thread int part[32];
  int i, stride;

  if (g_lo < g.c) {
    mdl_g_spatial_group_sum_partial(in, g.c, g.h, g.w, g.wpr, g.h_out, o, y,
                                    j, g_lo, g_hi, part);
  } else {
    for (i = 0; i < 32; i++) part[i] = 0;
  }
  for (i = 0; i < 32; i++) sdata[(int)lane * 32 + i] = part[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (stride = R >> 1; stride > 0; stride >>= 1) {
    if ((int)lane < stride) {
      for (i = 0; i < 32; i++) {
        sdata[(int)lane * 32 + i] += sdata[((int)lane + stride) * 32 + i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    thread int sum32[32];
    for (i = 0; i < 32; i++) sum32[i] = sdata[i];
    mdl_g_spatial_group_sum_finish(scores, g.h, g.w, g.wpr, o, y, j, sum32);
  }
}

kernel void mdl_k_lut_layer(device const uchar *x [[buffer(0)]],
                            device uchar *out [[buffer(1)]],
                            constant int *indices [[buffer(2)]],
                            constant uint *tt [[buffer(3)]],
                            constant MdlGeom &g [[buffer(4)]],
                            uint o [[thread_position_in_grid]]) {
  /* g.c=n_inputs, g.w=out_width, g.in_stride=1枚ぶんの入力幅 */
  if ((int)o >= g.w * g.batch) return;
  {
    const int b = (int)o / g.w;                 /* バッチ内の何枚目か */
    const int k = (int)o - b * g.w;
    out[o] = mdl_g_lut_out(x + b * g.in_stride, g.c, g.w, indices, tt, k);
  }
}

/* 添字uint16_t版。式は mdl_k_lut_layer と同一。*/
kernel void mdl_k_lut_layer_u16(device const uchar *x [[buffer(0)]],
                                device uchar *out [[buffer(1)]],
                                constant ushort *indices [[buffer(2)]],
                                constant uint *tt [[buffer(3)]],
                                constant MdlGeom &g [[buffer(4)]],
                                uint o [[thread_position_in_grid]]) {
  if ((int)o >= g.w * g.batch) return;
  {
    const int b = (int)o / g.w;                 /* バッチ内の何枚目か */
    const int k = (int)o - b * g.w;
    out[o] = mdl_g_lut_out_u16(x + b * g.in_stride, g.c, g.w, indices, tt, k);
  }
}

/* ---- lut_flat ----
 * 式は mdl_gpu_kernels.h の mdl_g_lut_flat_*。ホスト側(mdl_metal.mm)の
 * MdlFlatParams と同一レイアウトであること(MdlGeomと同じ約束)。
 *   n_in       信号表に載せる入力ビット数(詰め直し後)
 *   in_width   1枚ぶんの入力幅(x のストライド)
 *   sig_stride 1枚ぶんの信号表バイト数(段ごと起動版のみ)
 *   g0, n      この段のゲート範囲 [g0, g0+n)(段ごと起動版のみ)
 *   out_width  出力ビット数
 *   n_levels   段数(融合版のみ)
 *   batch      実行時枚数 */
struct MdlFlatParams {
  int n_in, in_width, sig_stride, g0, n, out_width, n_levels, batch;
};

kernel void mdl_k_lut_flat_init(device const uchar *x [[buffer(0)]],
                                device uchar *sig [[buffer(1)]],
                                constant int *in_idx [[buffer(2)]],
                                constant MdlFlatParams &p [[buffer(3)]],
                                uint idx [[thread_position_in_grid]]) {
  const int per = p.n_in + 1;
  if ((int)idx >= per * p.batch) return;
  {
    const int b = (int)idx / per;
    const int u = (int)idx - b * per;
    sig[b * p.sig_stride + u] =
        mdl_g_lut_flat_init_elem(x + b * p.in_width, in_idx, u);
  }
}

kernel void mdl_k_lut_flat_level(device uchar *sig [[buffer(0)]],
                                 constant uchar *tt [[buffer(1)]],
                                 constant int *a [[buffer(2)]],
                                 constant int *b [[buffer(3)]],
                                 constant MdlFlatParams &p [[buffer(4)]],
                                 uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= p.n * p.batch) return;
  {
    const int s = (int)idx / p.n;
    const int g = p.g0 + ((int)idx - s * p.n);
    device uchar *sg = sig + s * p.sig_stride;
    sg[1 + p.n_in + g] = mdl_g_lut_flat_gate(tt[g], sg[a[g]], sg[b[g]]);
  }
}

kernel void mdl_k_lut_flat_out(device const uchar *sig [[buffer(0)]],
                               device uchar *out [[buffer(1)]],
                               constant int *o [[buffer(2)]],
                               constant MdlFlatParams &p [[buffer(3)]],
                               uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= p.out_width * p.batch) return;
  {
    const int s = (int)idx / p.out_width;
    const int i = (int)idx - s * p.out_width;
    device const uchar *sg = sig + s * p.sig_stride;
    out[idx] = MDL_G_LUT_FLAT_OUT(sg, o, i);
  }
}

/* バッチ軸パック版: 信号表は u32 語(p.sig_stride = bwords)。*/
kernel void mdl_k_lut_flat_pk_init(device const uchar *x [[buffer(0)]],
                                   device uint *sig [[buffer(1)]],
                                   constant int *in_idx [[buffer(2)]],
                                   constant MdlFlatParams &p [[buffer(3)]],
                                   uint idx [[thread_position_in_grid]]) {
  const int bw = p.sig_stride;
  if ((int)idx >= (p.n_in + 1) * bw) return;
  {
    const int u = (int)idx / bw;
    const int w = (int)idx - u * bw;
    sig[idx] = mdl_g_lut_flat_pk_init_word(x, in_idx, p.in_width, p.batch, u, w);
  }
}

kernel void mdl_k_lut_flat_pk_level(device uint *sig [[buffer(0)]],
                                    constant uchar *tt [[buffer(1)]],
                                    constant int *a [[buffer(2)]],
                                    constant int *b [[buffer(3)]],
                                    constant MdlFlatParams &p [[buffer(4)]],
                                    uint idx [[thread_position_in_grid]]) {
  const int bw = p.sig_stride;
  if ((int)idx >= p.n * bw) return;
  {
    const int j = (int)idx / bw;
    const int w = (int)idx - j * bw;
    const int g = p.g0 + j;
    sig[(1 + p.n_in + g) * bw + w] =
        mdl_g_gate2(tt[g], sig[a[g] * bw + w], sig[b[g] * bw + w]);
  }
}

kernel void mdl_k_lut_flat_pk_out(device const uint *sig [[buffer(0)]],
                                  device uchar *out [[buffer(1)]],
                                  constant int *o [[buffer(2)]],
                                  constant MdlFlatParams &p [[buffer(3)]],
                                  uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= p.out_width * p.batch) return;
  {
    const int s = (int)idx / p.out_width;
    const int i = (int)idx - s * p.out_width;
    out[idx] = mdl_g_lut_flat_pk_out(sig, o, p.sig_stride, s, i);
  }
}

/* 融合版: 1スレッドグループ=1枚。信号表(1+n_in+n_gates バイト)を
 * threadgroupメモリに置き、段の間だけバリアを張る(起動1回で lut_flat 全体)。 */
kernel void mdl_k_lut_flat_fused(device const uchar *x [[buffer(0)]],
                                 device uchar *out [[buffer(1)]],
                                 constant uchar *tt [[buffer(2)]],
                                 constant int *a [[buffer(3)]],
                                 constant int *b [[buffer(4)]],
                                 constant int *o [[buffer(5)]],
                                 constant int *in_idx [[buffer(6)]],
                                 constant int *level_off [[buffer(7)]],
                                 constant MdlFlatParams &p [[buffer(8)]],
                                 threadgroup uchar *sg [[threadgroup(0)]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tgs [[threads_per_threadgroup]],
                                 uint bid [[threadgroup_position_in_grid]]) {
  const int base = 1 + p.n_in;
  device const uchar *xb = x + (int)bid * p.in_width;
  int u, L, g, i;
  for (u = (int)tid; u < base; u += (int)tgs) {
    sg[u] = mdl_g_lut_flat_init_elem(xb, in_idx, u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (L = 0; L < p.n_levels; L++) {
    const int g1 = level_off[L + 1];
    for (g = level_off[L] + (int)tid; g < g1; g += (int)tgs) {
      sg[base + g] = mdl_g_lut_flat_gate(tt[g], sg[a[g]], sg[b[g]]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (i = (int)tid; i < p.out_width; i += (int)tgs) {
    out[(int)bid * p.out_width + i] = MDL_G_LUT_FLAT_OUT(sg, o, i);
  }
}

kernel void mdl_k_group_sum(device const uchar *x [[buffer(0)]],
                            device int *out [[buffer(1)]],
                            constant MdlGeom &g [[buffer(2)]],
                            uint gidx [[thread_position_in_grid]]) {
  /* g.c=k, g.w=gsz */
  if ((int)gidx >= g.c) return;
  out[gidx] = mdl_g_group_sum_out(x, g.w, (int)gidx);
}

/* 並列縮約版。1スレッドグループ=1出力グループ、グループ内のR
 * レーン(=g.h、g.wpr_outと同様「thread_position_in_threadgroup系の
 * 組込み値に頼らずhost側の選んだRをデータとして渡す」流儀)が gsz を
 * 分担して部分和を出し、threadgroupメモリで木構造に総和する。
 * g.c=k(=グループ数=グリッド側の次元)、g.w=gsz、g.h=R。 */
kernel void mdl_k_group_sum_r(device const uchar *x [[buffer(0)]],
                              device int *out [[buffer(1)]],
                              constant MdlGeom &g [[buffer(2)]],
                              threadgroup int *sdata [[threadgroup(0)]],
                              uint lane [[thread_position_in_threadgroup]],
                              uint gidx [[threadgroup_position_in_grid]]) {
  const int R = g.h;
  int stride;
  sdata[lane] = mdl_g_group_sum_partial(x, g.w, (int)gidx, (int)lane, R);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (stride = R >> 1; stride > 0; stride >>= 1) {
    if ((int)lane < stride) sdata[lane] += sdata[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) out[gidx] = sdata[0];
}

/* ---- dense のバッチ軸ビットプレーン化 ----
 * 計算式は mdl_gpu_kernels.h の mdl_g_pack_batch_word /
 * mdl_g_lut_out_packed[_u16] / mdl_g_group_sum_packed_word(CUDA/Metal共有)。
 * ここはMetalの引数バインディングの殻だけ。 */

/* パック: 1スレッド=パック平面の1語(neuron, word)。g.w=width、
 * g.wpr=bwords、g.extra=batch(このdispatchでの実行時枚数n)。
 * total=width*bwords。idxの分解(w=idx/width、iが速い軸)は読み取り側の
 * コアレスアクセスのためだけの都合で、書き込み先は neuron-major
 * (`out[i*bwords+w]`、mdl_g_pack_batch_wordのコメント参照)。 */
kernel void mdl_k_pack_batch(device const uchar *x [[buffer(0)]],
                             device uint *out [[buffer(1)]],
                             constant MdlGeom &g [[buffer(2)]],
                             uint idx [[thread_position_in_grid]]) {
  const int width = g.w;
  const int bwords = g.wpr;
  const int w = (int)idx / width;
  const int i = (int)idx - w * width;
  out[i * bwords + w] = mdl_g_pack_batch_word(x, width, g.extra, (int)idx);
}

/* lut_dense のバッチ軸パック版。g.c=n_inputs、g.w=out_width、
 * g.extra=bwords。idxはo(ゲート)を遅い軸・w(バッチ語)を速い軸にして
 * 分解する(同じゲートを読む隣接スレッドがワープ内で連続するwを読み、
 * 添字/ttの参照は同一ワープ内で揃う)。total=out_width*bwords。 */
kernel void mdl_k_lut_layer_packed(device const uint *x [[buffer(0)]],
                                   device uint *out [[buffer(1)]],
                                   constant int *indices [[buffer(2)]],
                                   constant uint *tt [[buffer(3)]],
                                   constant MdlGeom &g [[buffer(4)]],
                                   uint idx [[thread_position_in_grid]]) {
  const int bwords = g.extra;
  const int o = (int)idx / bwords;
  const int w = (int)idx - o * bwords;
  out[idx] = mdl_g_lut_out_packed(x, g.c, g.w, indices, tt, bwords, o, w);
}

/* 添字uint16_t版(同じ理由)。式は mdl_k_lut_layer_packed と同一。*/
kernel void mdl_k_lut_layer_packed_u16(device const uint *x [[buffer(0)]],
                                       device uint *out [[buffer(1)]],
                                       constant ushort *indices [[buffer(2)]],
                                       constant uint *tt [[buffer(3)]],
                                       constant MdlGeom &g [[buffer(4)]],
                                       uint idx [[thread_position_in_grid]]) {
  const int bwords = g.extra;
  const int o = (int)idx / bwords;
  const int w = (int)idx - o * bwords;
  out[idx] = mdl_g_lut_out_packed_u16(x, g.c, g.w, indices, tt, bwords, o, w);
}

/* group_sum のバッチ軸パック版。g.w=gsz、g.c=k、g.batch=batch(実行時n)、
 * g.extra=bwords、g.h_out=n_planes(ホスト側で1回計算して渡す)。出力は
 * 非パック版と同じint32レイアウト(scores[sample*k+g])なので、この後段
 * にアンパックは要らない。total=k*bwords。 */
kernel void mdl_k_group_sum_packed(device const uint *x [[buffer(0)]],
                                   device int *scores [[buffer(1)]],
                                   constant MdlGeom &g [[buffer(2)]],
                                   uint idx [[thread_position_in_grid]]) {
  const int bwords = g.extra;
  const int gidx = (int)idx / bwords;
  const int w = (int)idx - gidx * bwords;
  mdl_g_group_sum_packed_word(x, scores, g.w, g.c, g.batch, bwords, g.h_out,
                              gidx, w);
}

/* group_sum_packed の並列縮約版(mdl_k_group_sum_r/mdl_k_sgs_rと同型)。
 * 実測(har_dlgn_k600、batch=128)でmdl_k_group_sum_packedがGPU時間の78%を
 * 占めていた(k*bwords=24スレッドしか立たないため)ことを受けて追加。
 * 1スレッドグループ=1出力ワード(g,w)、グループ内のRレーンがgszを
 * チャンク分担してCSAし、32サンプルぶんの整数部分和をthreadgroupメモリで
 * 総和してから書き出す。g.c=gsz、g.h=k、g.h_out=n_planes_chunk、
 * g.w_out=chunk、g.wpr_out=R、g.extra=bwords、g.batch=batch(実行時n)。 */
kernel void mdl_k_group_sum_packed_r(device const uint *x [[buffer(0)]],
                                     device int *scores [[buffer(1)]],
                                     constant MdlGeom &g [[buffer(2)]],
                                     threadgroup int *sdata [[threadgroup(0)]],
                                     uint lane [[thread_position_in_threadgroup]],
                                     uint wi [[threadgroup_position_in_grid]]) {
  const int bwords = g.extra;
  const int gsz = g.c;
  const int R = g.wpr_out;
  const int chunk = g.w_out;
  const int gidx = (int)wi / bwords;
  const int w = (int)wi - gidx * bwords;
  const int g_lo = (int)lane * chunk;
  const int g_hi = min(g_lo + chunk, gsz);
  thread int part[32];
  int i, stride;

  if (g_lo < gsz) {
    mdl_g_group_sum_packed_partial(x, gsz, bwords, g.h_out, gidx, w, g_lo,
                                   g_hi, part);
  } else {
    for (i = 0; i < 32; i++) part[i] = 0;
  }
  for (i = 0; i < 32; i++) sdata[(int)lane * 32 + i] = part[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (stride = R >> 1; stride > 0; stride >>= 1) {
    if ((int)lane < stride) {
      for (i = 0; i < 32; i++) {
        sdata[(int)lane * 32 + i] += sdata[((int)lane + stride) * 32 + i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    thread int sum32[32];
    for (i = 0; i < 32; i++) sum32[i] = sdata[i];
    mdl_g_group_sum_packed_finish(scores, g.h, g.batch, gidx, w, sum32);
  }
}

/* ---- RDDLGN自己回帰生成(decode)ハーネス専用カーネル ----
 * gpubackend.pyのEMITTERSには載らない(IR opではない、mdl_gpu_kernels.h
 * のコメント参照)。呼び出しはdeploy/rddlgn_decode_gpu.pyが直接発行する。 */

/* g.c = d_emb, g.batch = レーン数(=n)。tokens[b]をtable[token][j]へ引く。 */
kernel void mdl_k_embed_lookup(device const int *tokens [[buffer(0)]],
                               device const uchar *table [[buffer(1)]],
                               device uchar *out [[buffer(2)]],
                               constant MdlGeom &g [[buffer(3)]],
                               uint idx [[thread_position_in_grid]]) {
  if ((int)idx >= g.c * g.batch) return;
  {
    const int b = (int)idx / g.c;
    const int j = (int)idx - b * g.c;
    out[idx] = mdl_g_embed_lookup_out(table, g.c, tokens[b], j);
  }
}

/* g.c = width(vocab)、g.batch = レーン数。1スレッド=1レーン
 * (vocabは高々数百程度で並列縮約が要る規模ではない、設計メモ参照)。*/
kernel void mdl_k_argmax(device const int *scores [[buffer(0)]],
                         device int *tokens [[buffer(1)]],
                         constant MdlGeom &g [[buffer(2)]],
                         uint b [[thread_position_in_grid]]) {
  if ((int)b >= g.batch) return;
  tokens[b] = mdl_g_argmax_out(scores + (int)b * g.c, g.c);
}
