/* griffith-logic — CUDAランタイムの公開API
 *
 * 生成された model_cuda.cu から呼ぶ薄いラッパ群。カーネル本体は
 * runtime/gpu/mdl_gpu_kernels.h(Metalと共有)にあり、CUDA固有なのは
 * グリッド起動とメモリ管理だけ。
 *
 * ポインタは全てデバイスメモリ。呼び出し側(生成コード)が確保・解放する。
 */
#ifndef MDL_CUDA_H
#define MDL_CUDA_H

/* ストリームAPIは cudaStream_t を公開シグネチャで使うので、
 * CUDAランタイムヘッダが要る(このヘッダは常に .cu からnvccで
 * コンパイルされる前提 ─ mdl_cuda.cu自身と生成された model_cuda.cu の
 * 両方がnvccビルドなので問題ない)。 */
#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** conv2d 1ノードの実行レイアウト。中身はCPU版 mdl_conv_layout と同じ意味で、
 *  ポインタがデバイスメモリを指す点だけが違う。値渡しでカーネルへ入れるので
 *  POD にしてある。 */
typedef struct {
  int h, w, h_out, w_out;
  int s_h, s_w, p_h, p_w;
  int out_channels;
  int wpr_in, wpr_out, tap_wpr, tap_rows_words;
  unsigned int mask_out;
  int use_fast_path;
  const int *plane_c;      /* device */
  const int *plane_dy;     /* device */
  const int *plane_dx;     /* device */
  int n_planes, plane_stride, plane_rows;
  int in_plane_words, out_plane_words;
  const int *instr_flat;   /* device、[n_instrs*4] */
  int n_instrs;
  const int *oc_src_off;   /* device */
  const int *write_start;  /* device */
  const int *write_oc;     /* device */
  int pool_base, reg_base, scratch_words;
} mdl_cuda_conv_layout;

int mdl_cuda_pack(const unsigned char *src, unsigned int *dst,
                  int c, int h, int w);
int mdl_cuda_unpack(const unsigned int *src, unsigned char *dst,
                    int c, int h, int w);
/* --- バッチについて ---
 * ビットプレーンの並びは [batch][C][H][wpr] なので、**チャネル方向に
 * 独立なカーネルはバッチ透過**になる: pack / unpack / or_pool / upsample /
 * spatial_group_sum / group_sum は「チャネル数 × batch」を渡すだけでよく、
 * 引数は増えていない。バッチを明示的に受けるのは、記述子や順列表を
 * 枚ごとに引き直す必要がある下の5本だけ。 */
int mdl_cuda_preproc(const unsigned char *image, int c_img, int h, int w,
                     const int *desc, int channels, const short *thr,
                     const unsigned int *qval, unsigned int *out, int batch);
/** conv の定数平面(0 と 全1)を1回だけ用意する。scratch ごとに1回。 */
int mdl_cuda_conv_init(const mdl_cuda_conv_layout *layout,
                       unsigned int *scratch, int batch);
int mdl_cuda_conv(const mdl_cuda_conv_layout *layout, const unsigned int *in,
                  unsigned int *scratch, unsigned int *out, int batch);
/** ocごとの命令列分割版。conv並列度を(batch,ワード位置)から
 *  (batch,oc,ワード位置)へ拡張する。`mdl_cuda_conv_init` はそのまま
 *  共用できる(定数0/1平面の埋め方は scratch_words/tap_rows_words だけで
 *  決まるため。分割用は reg 平面が要らないぶん `layout->scratch_words` が
 *  小さい)。`layout` の `instr_flat`/`oc_src_off`/`write_start`/`write_oc`/
 *  `reg_base`/`n_instrs` は分割版では使わない(呼び出し側が0/NULLで
 *  埋めてよい)。`instr_flat_split` は [tt, a_packed, b_packed, dst_slot]
 *  の4語/命令(オペランドは `(value<<1)|is_reg` にパック済み、
 *  bit0=0で共有スクラッチのアドレス、bit0=1でスレッドローカル
 *  レジスタ`slots`の添字)。`oc_instr_start` はCSR(長さ
 *  out_channels+1)、`oc_final_ref` はocごとの最終値参照(同じパック規則)。 */
int mdl_cuda_conv_split(const mdl_cuda_conv_layout *layout,
                        const unsigned int *in, unsigned int *scratch,
                        unsigned int *out, const int *oc_instr_start,
                        const int *instr_flat_split, const int *oc_final_ref,
                        int batch);
/** タップ平面プールの構築だけ(モデル特化コード生成向け)。
 *  `mdl_cuda_conv`/`mdl_cuda_conv_split` が内部で使っている
 *  `mdl_k_conv_planes` 呼び出しを切り出したもの。生成側(`md deploy
 *  codegen-cuda --conv-codegen`)はこれを呼んだ後、命令列実行だけを
 *  生成した専用 `__global__` カーネル(ttをコンパイル時定数、slotsを
 *  スカラーレジスタへ展開したもの)へ直接差し替える
 *  (CUDAはAOTコンパイルなので、Metalの`mdl_metal_conv_exec_named`の
 *  ような名前引き実行時ディスパッチは不要 — 生成コードが直接
 *  `<<<>>>` で起動できる)。 */
int mdl_cuda_conv_planes(const mdl_cuda_conv_layout *layout,
                         const unsigned int *in, unsigned int *scratch,
                         int batch);
int mdl_cuda_or_pool(const unsigned int *in, unsigned int *out,
                     int c, int h, int w);
int mdl_cuda_upsample(const unsigned int *in, unsigned int *out,
                      int c, int h, int w);
int mdl_cuda_permute(const unsigned int *in, unsigned int *out,
                     const int *order, int c_out, int h, int wpr, int batch);
int mdl_cuda_spatial_group_sum(const unsigned int *in, int *scores,
                               int n_out, int gs, int h, int w);
int mdl_cuda_lut_layer(const unsigned char *x, unsigned char *out,
                       int n_inputs, int out_width, const int *indices,
                       const unsigned int *tt, int in_width, int batch);
/** 添字uint16_t版。in_width<=65536のときだけ選ばれる。*/
int mdl_cuda_lut_layer_u16(const unsigned char *x, unsigned char *out,
                           int n_inputs, int out_width,
                           const unsigned short *indices,
                           const unsigned int *tt, int in_width, int batch);
int mdl_cuda_group_sum(const unsigned char *x, int *out, int in_width, int k);
/** lut_flat(Metal版 mdl_metal_lut_flat_* と対称)。sig は1枚
 *  sig_stride バイト × batch の信号表(段ごと起動版のみ)。融合版は
 *  1ブロック=1枚で信号表(sig_bytes)を動的共有メモリに置く(<=48KB)。
 *  **CUDA実機での実行は未検証**(scripts/verify_cuda_remote.py で検証すること)。 */
int mdl_cuda_lut_flat_init(const unsigned char *x, void *sig,
                           const int *in_idx, int n_in, int in_width,
                           int sig_stride, int batch);
int mdl_cuda_lut_flat_level(void *sig, const unsigned char *tt, const int *a,
                            const int *b, int n_in, int sig_stride, int g0,
                            int n, int batch);
int mdl_cuda_lut_flat_out(const void *sig, unsigned char *out, const int *o,
                          int out_width, int sig_stride, int batch);
/** バッチ軸パック版: 信号表は u32 語 [信号][ceil(batch/32)]。*/
int mdl_cuda_lut_flat_pk_init(const unsigned char *x, void *sig,
                              const int *in_idx, int n_in, int in_width,
                              int batch);
int mdl_cuda_lut_flat_pk_level(void *sig, const unsigned char *tt,
                               const int *a, const int *b, int n_in, int g0,
                               int n, int batch);
int mdl_cuda_lut_flat_pk_out(const void *sig, unsigned char *out,
                             const int *o, int out_width, int batch);
int mdl_cuda_lut_flat_fused(const unsigned char *x, unsigned char *out,
                            const unsigned char *tt, const int *a,
                            const int *b, const int *o, const int *in_idx,
                            const int *level_off, int n_in, int in_width,
                            int out_width, int n_levels, int sig_bytes,
                            int batch);

/* ---- dense のバッチ軸ビットプレーン化(`md deploy codegen-cuda
 * --dense-packed`) ----
 * 32サンプル/語のビットプレーンへ詰め替え、lut_dense/group_sumをその形の
 * まま評価する(算術強度をバッチ方向に約32倍に上げる)。Metal版
 * (mdl_metal.h の同名関数群、`_metal_` → `_cuda_` の対称)と同じ式・
 * 同じレイアウト。適格性の判定(batch>=32・全主経路opがlut_dense/
 * group_sumのみ・conv/recurrent非混在)は `deploy/plan.py::
 * dense_packed_eligible` が生成時に行う。
 *
 * `batch` 引数はいずれもこの呼び出しでの**実行時**サンプル数(確保上限
 * ではない)。**CUDA実機での実行は未検証**(このMacにCUDA実機/nvccが
 * 無いため。ローカルの構文検査のみ実施)。 */

/** uchar入力([batch][width])→ u32パック平面(`out[neuron*bwords+w]`、
 *  CPU版mdl_pack_batchと同一レイアウト、bwords=ceil(batch/32))。
 *  `x`/`out`は別バッファであること(インプレース不可)。 */
int mdl_cuda_pack_batch(const unsigned char *x, unsigned int *out, int width,
                        int batch);
/** lut_dense のバッチ軸パック版。`x`/`out`はどちらもパック済みu32平面。 */
int mdl_cuda_lut_layer_packed(const unsigned int *x, unsigned int *out,
                              int n_inputs, int out_width,
                              const int *indices, const unsigned int *tt,
                              int batch);
/** 添字uint16_t版(同じ判定、in_width<=65536のときだけ選ばれる)。*/
int mdl_cuda_lut_layer_packed_u16(const unsigned int *x, unsigned int *out,
                                  int n_inputs, int out_width,
                                  const unsigned short *indices,
                                  const unsigned int *tt, int batch);
/** group_sum のバッチ軸パック版。`out`は非パック版と同じint32レイアウト
 *  (`scores[sample*k+g]`)に直接書く ─ この後段にアンパックは要らない。 */
int mdl_cuda_group_sum_packed(const unsigned int *x, int *out, int in_width,
                              int k, int batch);

/* `mdl_cuda_pack`等と同様にホスト側で呼ぶ。`copy_batched_ex`が
 * バッチ(レーン)ぶん独立にcudaMemcpyDeviceToDeviceをforループでN回
 * 発行していたのを、1回のカーネル起動で運ぶ(Metal版
 * mdl_metal_copy_strided対称)。`dst[dst_off+lane*dst_stride+i] =
 * src[src_off+lane*src_stride+i]`(i∈[0,n_bytes)、lane∈[0,batch))。
 * `src_stride=0`で全レーンが同じsrc範囲を読む(定数複製)。 */
int mdl_cuda_copy_strided(const unsigned char *src, unsigned char *dst,
                          int dst_off, int dst_stride, int src_off,
                          int src_stride, int n_bytes, int batch);

/* ---- RDDLGN自己回帰生成(decode)ハーネス専用 ----
 * IRのop語彙には無い(argmax/埋め込み表引きは論理回路で表せない
 * blackbox相当)。deploy/rddlgn_decode_gpu.pyが直接呼び出す。 */

/** tokens[b](int32、batch本)を table[vocab][d_emb](0/1バイト、行=トークン)
 *  で引き、out[b*d_emb+j] = table[tokens[b]][j] へ書く。 */
int mdl_cuda_embed_lookup(const int *tokens, const unsigned char *table,
                         unsigned char *out, int d_emb, int batch);
/** scores(int32、[batch][width])の行ごとにargmax(同点は先勝ち)を取り、
 *  tokens(int32、batch本)へ書く。 */
int mdl_cuda_argmax(const int *scores, int *tokens, int width, int batch);

/** 直前までのカーネルの完了を待つ(計測・結果読み出しの前に呼ぶ)。 */
int mdl_cuda_sync(void);

/* ---- ストリーム・パイプライン化(Metal(mdl_metal.h)の
 * mdl_metal_commit_async/mdl_metal_wait_slotのCUDA版ミラー)。
 *
 * 背景(Metal版と同じ): 1推論=1回のカーネル列発行+即
 * mdl_cuda_sync(cudaDeviceSynchronize)という既存の設計は、推論のたびに
 * ホスト側のwaitで止まる。連続する多数枚を流すとき「次の枚を別ストリームへ
 * 発行してから前の枚をwaitする」向きに並べ替えれば、CPUがwaitで止まって
 * いる時間を次の枚の発行作業と重ねられる。
 *
 * MetalとCUDAの違い: Metalは「エンコードしてから1つのcommandBufferとして
 * commit」という二段構えだが、CUDAのkernel launch(`<<<>>>`)はその場で
 * 指定したストリームへ非同期に積まれる(エンコードと同時にコミット済み)。
 * そのためCUDA版の「commit」は実質的に「そのスロットのストリームへ完了
 * イベントを記録し、pending扱いにする」だけでよい。
 *
 * 使い方(スロット0..MDL_CUDA_MAX_STREAMS-1は独立したバッファ集合を指す。
 * `md deploy codegen-cuda --n-streams N` で生成したモデルが
 * `mdl_cuda_infer_batch_submit`/`_collect`(入口の実体は生成コード側)を
 * 持つ):
 *
 *     mdl_cuda_infer_batch_submit(bits0, n0, 0);    // slot0へ発行(待たない)
 *     mdl_cuda_infer_batch_submit(bits1, n1, 1);    // slot1へ発行(待たない)
 *     mdl_cuda_infer_batch_collect(scores0, n0, 0); // slot0を待って読み出す
 *     mdl_cuda_infer_batch_submit(bits2, n2, 0);    // slot0を再利用(直前にcollect済みなので安全)
 *     mdl_cuda_infer_batch_collect(scores1, n1, 1);
 *     ...
 *
 * 安全性: Metal版と同じガード方式(教訓 ─ バッファ共有の誤りは
 * 1回きりの実行では見えない)。同じスロットを mdl_cuda_wait_slot(=collect
 * の内部)を挟まずに二重にcommit_asyncすると、前回ぶんをGPUがまだ読み書き
 * している可能性があるバッファへ新しい入力を上書きすることになり結果が
 * 壊れるので、`mdl_cuda_commit_async` は「そのスロットがまだpendingの
 * とき」エラー(-1)を返して拒否する(黙って壊れない)。 */

/** 同時に使えるスロット数の上限(Metal版 MDL_METAL_MAX_STREAMS と同じ値)。 */
#define MDL_CUDA_MAX_STREAMS 8

/** n_streams本のCUDAストリーム(+スロットごとの完了イベント)を作る。
 *  生成コードの mdl_cuda_init 内で(n_streams>1のときだけ)1回呼ばれる
 *  (冪等: 既に作成済みなら何もせず0を返す)。
 *  n<1 または n>MDL_CUDA_MAX_STREAMS は-1。 */
int mdl_cuda_streams_init(int n_streams);
/** 全スロットのpendingを待ってからストリーム・イベントを破棄する
 *  (mdl_cuda_shutdown相当のタイミングで生成コードが呼ぶ)。 */
void mdl_cuda_streams_shutdown(void);

/** これ以降の mdl_cuda_* 呼び出し(カーネル起動・D2Dコピー・非同期
 *  アップロード/ダウンロード)の宛先をスロットslotのストリームにする。
 *  範囲外/未初期化なら-1。 */
int mdl_cuda_select_stream(int slot);
/** 宛先をデフォルト(レガシー)ストリームへ戻す。streams_init前や
 *  n_streams<=1の生成物では常にこの状態(何もしなくても既定でこう)。 */
void mdl_cuda_select_default_stream(void);
/** 現在選択中のストリーム(mdl_cuda.cu内のカーネル起動・
 *  cudaMemcpyAsyncが読む)。生成コード(model_cuda.cu)が独自に発行する
 * カーネル(conv-codegenのdense-fuse)もこれを使えば、
 *  select_stream/select_default_streamで選んだストリームへ揃う。 */
cudaStream_t mdl_cuda_current_stream(void);

/** そのスロットに積んだ処理をコミットする(**待たない**)。
 *  戻り値: 0=OK、-1=範囲外/未初期化/そのスロットが既にpending
 *  (collectし忘れ)。 */
int mdl_cuda_commit_async(int slot);
/** そのスロットの直近の mdl_cuda_commit_async を待つ。何も積んでいなければ
 *  即0を返す(冪等)。 */
int mdl_cuda_wait_slot(int slot);

/** ホスト→デバイス/デバイス→ホストの非同期アップロード/ダウンロード
 *  (現在選択中のストリームへ積む)。mdl_cuda_select_stream(slot)の
 *  直後に呼べばそのスロットのストリームへ積まれ、選択していなければ
 *  デフォルトストリーム(従来のcudaMemcpyと同じ意味)になる。 */
int mdl_cuda_upload_async(void *dst, const void *src, size_t n_bytes);
int mdl_cuda_download_async(void *dst, const void *src, size_t n_bytes);

#ifdef __cplusplus
}
#endif

#endif /* MDL_CUDA_H */
