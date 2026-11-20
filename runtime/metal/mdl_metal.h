/* griffith-logic — Metal(Apple GPU)ランタイムの公開API
 *
 * 生成された model_metal.mm から呼ぶ。CUDA版(mdl_cuda.h)と同じ形にしてあり、
 * 違いは「シェーダを実行時にソースからコンパイルする」ことと、
 * デバイスバッファがハンドル(void*)であることだけ。
 *
 * Metalはオフラインの metal ツールチェーンが無くても
 * newLibraryWithSource: で動くので、Xcodeのコンポーネント追加は不要。
 */
#ifndef MDL_METAL_H
#define MDL_METAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *mdl_metal_buf;

/** ストリーム・パイプライン化で同時に使えるスロット数の上限。
 *  生成コード側(`md deploy codegen-metal` の n_streams)はこれ以下で
 *  なければならない(`mdl_metal_commit_async`/`mdl_metal_wait_slot` が
 *  範囲外を拒否する)。 */
#define MDL_METAL_MAX_STREAMS 8

/** conv のスカラーパラメータ。Metalの構造体にデバイスポインタは置けないので、
 *  配列は別バッファで渡す(mdl_metal_conv の引数)。 */
typedef struct {
  int h, w, h_out, w_out;
  int s_h, s_w, p_h, p_w;
  int out_channels;
  int wpr_in, wpr_out, tap_wpr, tap_rows_words;
  unsigned int mask_out;
  int use_fast_path;
  int n_planes, plane_stride, plane_rows;
  int pool_base, n_instrs, n_write0;
  /* バッチ用のストライド(1枚ぶんのワード数)。 */
  int in_plane_words, out_plane_words, scratch_words;
} mdl_metal_conv_params;

/** シェーダソース(MSL)からライブラリとパイプラインを作る。1回だけ呼ぶ。 */
int mdl_metal_init(const char *shader_source);

/** 事前コンパイル済み `.metallib` があればそれを使い、無ければ(または
 *  読めなければ)ソースからコンパイルする。
 *
 *  実測(Apple M3): 実行時コンパイルは「Metalコンパイラの起動 約51ms」+
 *  「シェーダ本体のコンパイル 約73ms」= **起動時のみ約124ms**。
 *  推論の定常性能(約0.9ms)には一切影響しない。短命プロセスや
 *  コールドスタートを詰めたい場合にだけ .metallib を用意する意味がある。
 *
 *  .metallib の作り方(Xcodeのmetalツールチェーンが要る):
 *      xcrun metal -c model.metal -o model.air
 *      xcrun metallib model.air -o model.metallib
 *  MSLソースは `md deploy codegen-metal` が生成した .mm に埋まっている。 */
int mdl_metal_init_ex(const char *metallib_path, const char *shader_source);
void mdl_metal_shutdown(void);
/** GPUの名前(ログ用)。未初期化なら NULL。 */
const char *mdl_metal_device_name(void);

mdl_metal_buf mdl_metal_alloc(size_t bytes);
/** 既存バッファの部分領域を指すビューを作る(メモリ計画がアリーナを
 *  切り出すのに使う)。ビューは実体を所有しないので、ベースより先に
 *  `mdl_metal_free` すること。offset はバイト単位で4の倍数。 */
mdl_metal_buf mdl_metal_view(mdl_metal_buf base, size_t offset);
void mdl_metal_free(mdl_metal_buf buf);
int mdl_metal_upload(mdl_metal_buf dst, const void *src, size_t bytes);
int mdl_metal_download(void *dst, mdl_metal_buf src, size_t bytes);
int mdl_metal_copy(mdl_metal_buf dst, size_t dst_off, mdl_metal_buf src,
                   size_t src_off, size_t bytes);
/** `mdl_metal_copy`をバッチ(レーン)ぶん独立にホスト側forループで
 *  N回呼ぶ代わりに、1回のcomputeカーネル起動で運ぶ。
 *  `dst[dst_off+lane*dst_stride+i] = src[src_off+lane*src_stride+i]`
 *  (i∈[0,n_bytes)、lane∈[0,batch))。`src_stride=0`で全レーンが同じsrc
 *  範囲を読む(定数の複製、`copy_batched_ex`の既存の規約)。 */
int mdl_metal_copy_strided(mdl_metal_buf dst, int dst_off, int dst_stride,
                           mdl_metal_buf src, int src_off, int src_stride,
                           int n_bytes, int batch);

int mdl_metal_pack(mdl_metal_buf src, mdl_metal_buf dst, int c, int h, int w);
int mdl_metal_unpack(mdl_metal_buf src, mdl_metal_buf dst, int c, int h, int w);
int mdl_metal_preproc(mdl_metal_buf image, int c_img, int h, int w,
                      mdl_metal_buf desc, int channels, mdl_metal_buf thr,
                      mdl_metal_buf qval, mdl_metal_buf out, int batch);
/** conv の定数平面(0 と 全1)を1回だけ用意する。scratch ごとに1回。 */
int mdl_metal_conv_init(const mdl_metal_conv_params *p, mdl_metal_buf scratch,
                        int batch);
int mdl_metal_conv(const mdl_metal_conv_params *p, mdl_metal_buf in,
                   mdl_metal_buf scratch, mdl_metal_buf out,
                   mdl_metal_buf plane_c, mdl_metal_buf plane_dy,
                   mdl_metal_buf plane_dx, mdl_metal_buf instr,
                   mdl_metal_buf oc_src, mdl_metal_buf wstart,
                   mdl_metal_buf woc, int batch);
/** ocごとの命令列分割版。`mdl_metal_conv_init` はそのまま共用できる
 *  (定数0/1平面の埋め方はscratch_words/tap_rows_wordsだけで決まるため)。
 *  `p->scratch_words` は分割用に小さくなっている(reg平面が要らない)ので、
 *  同じ `mdl_metal_conv_params` を組み立てるコード生成側が
 *  結合版と分割版で別の値を積む。`instr` は
 *  [tt, a_packed, b_packed, dst_slot] の4語/命令(オペランドは
 *  `(value<<1)|is_reg` にパック済み)、`oc_instr_start` はCSR
 *  (長さ out_channels+1)、`oc_final_ref` はocごとの最終値参照
 *  (同じパック規則)。 */
int mdl_metal_conv_split(const mdl_metal_conv_params *p, mdl_metal_buf in,
                         mdl_metal_buf scratch, mdl_metal_buf out,
                         mdl_metal_buf plane_c, mdl_metal_buf plane_dy,
                         mdl_metal_buf plane_dx, mdl_metal_buf oc_instr_start,
                         mdl_metal_buf instr, mdl_metal_buf oc_final_ref,
                         int batch);
/** タップ平面プールの構築だけ(モデル特化コード生成向け)。
 *  `mdl_metal_conv`/`mdl_metal_conv_split` が内部で使っている
 *  `mdl_k_conv_planes` 呼び出しをそのまま切り出したもの(幾何・タップは
 *  実行方式に関わらず共通)。生成側はこれを呼んだ後、命令列実行だけを
 *  `mdl_metal_conv_exec_named` の呼び出しへ差し替える。 */
int mdl_metal_conv_planes(const mdl_metal_conv_params *p, mdl_metal_buf in,
                          mdl_metal_buf scratch, mdl_metal_buf plane_c,
                          mdl_metal_buf plane_dy, mdl_metal_buf plane_dx,
                          int batch);
/** 名前で指定したカーネルを (scratch, out, total) の3引数で起動する。`md deploy codegen-metal --conv-codegen` が生成するモデル
 *  特化カーネル(tt即値+スロット静的展開、`mdl_k_conv_codegen_<tag>` )は
 *  この3引数だけを取る形に統一してあるので、汎用ディスパッチ1つで足りる
 *  (名前ごとにC関数を増やす必要が無い)。カーネル名は生成時にモデル特化
 *  シェーダソースへ埋め込まれ、`mdl_metal_init` が"mdl_k_"接頭辞の関数を
 *  自動でパイプライン登録するので、ここは名前引きするだけでよい。 */
int mdl_metal_conv_exec_named(const char *kernel_name, mdl_metal_buf scratch,
                              mdl_metal_buf out, int total);
int mdl_metal_or_pool(mdl_metal_buf in, mdl_metal_buf out, int c, int h, int w);
int mdl_metal_upsample(mdl_metal_buf in, mdl_metal_buf out, int c, int h,
                       int w);
int mdl_metal_permute(mdl_metal_buf in, mdl_metal_buf out, mdl_metal_buf order,
                      int c_out, int h, int wpr, int batch);
int mdl_metal_spatial_group_sum(mdl_metal_buf in, mdl_metal_buf scores,
                                int n_out, int gs, int h, int w);
int mdl_metal_lut_layer(mdl_metal_buf x, mdl_metal_buf out, int n_inputs,
                        int out_width, mdl_metal_buf indices,
                        mdl_metal_buf tt, int in_width, int batch);
/** 添字uint16_t版。in_width<=65536のときだけ選ばれる。*/
int mdl_metal_lut_layer_u16(mdl_metal_buf x, mdl_metal_buf out, int n_inputs,
                            int out_width, mdl_metal_buf indices,
                            mdl_metal_buf tt, int in_width, int batch);
int mdl_metal_group_sum(mdl_metal_buf x, mdl_metal_buf out, int in_width,
                        int k);
/** lut_flat(縮約済み dense 平坦ネットリスト)。表は
 *  `deploy/flatlevels.py` が段ごとに並べ替えたもの。段ごと起動版は
 *  init → level(段の数だけ)→ out を順に呼ぶ(信号表 sig は1枚 sig_stride
 *  バイト × batch)。融合版は1スレッドグループ=1枚で信号表(sig_bytes)を
 *  threadgroupメモリに置き、起動1回で全段を評価する。 */
int mdl_metal_lut_flat_init(mdl_metal_buf x, mdl_metal_buf sig,
                            mdl_metal_buf in_idx, int n_in, int in_width,
                            int sig_stride, int batch);
int mdl_metal_lut_flat_level(mdl_metal_buf sig, mdl_metal_buf tt,
                             mdl_metal_buf a, mdl_metal_buf b, int n_in,
                             int sig_stride, int g0, int n, int batch);
int mdl_metal_lut_flat_out(mdl_metal_buf sig, mdl_metal_buf out,
                           mdl_metal_buf o, int out_width, int sig_stride,
                           int batch);
/** バッチ軸パック版: 信号表は u32 語 [信号][ceil(batch/32)]。*/
int mdl_metal_lut_flat_pk_init(mdl_metal_buf x, mdl_metal_buf sig,
                               mdl_metal_buf in_idx, int n_in, int in_width,
                               int batch);
int mdl_metal_lut_flat_pk_level(mdl_metal_buf sig, mdl_metal_buf tt,
                                mdl_metal_buf a, mdl_metal_buf b, int n_in,
                                int g0, int n, int batch);
int mdl_metal_lut_flat_pk_out(mdl_metal_buf sig, mdl_metal_buf out,
                              mdl_metal_buf o, int out_width, int batch);
int mdl_metal_lut_flat_fused(mdl_metal_buf x, mdl_metal_buf out,
                             mdl_metal_buf tt, mdl_metal_buf a,
                             mdl_metal_buf b, mdl_metal_buf o,
                             mdl_metal_buf in_idx, mdl_metal_buf level_off,
                             int n_in, int in_width, int out_width,
                             int n_levels, int sig_bytes, int batch);

/** dense多層融合カーネル。`md deploy codegen-metal --dense-fuse` が
 *  生成するモデル特化カーネル(`mdl_k49_fused_<tag>`、全lut_dense層+
 *  group_sumを1個へ融合)を、名前引きで
 *  1スレッドグループ=1サンプル(グループ数=batch)で起動する。
 *  `idx_bufs`/`tt_bufs` は各n_layers本(層ごとの添字/真理値表、従来どおり
 *  デバイスバッファのまま)。`act_bytes` はping-pong活性1本ぶんの
 *  バイト数(`deploy/densefuse.py::act_buf_bytes`と同じ計算、
 *  [[threadgroup(0)]]/[[threadgroup(1)]]の両方に同じ長さを設定する)。 */
int mdl_metal_dense_fuse_exec(const char *kernel_name, mdl_metal_buf x,
                              mdl_metal_buf out, mdl_metal_buf *idx_bufs,
                              mdl_metal_buf *tt_bufs, int n_layers,
                              int act_bytes, int batch);
/* ---- dense のバッチ軸ビットプレーン化(`md deploy codegen-metal
 * --dense-packed`) ----
 * 32サンプル/語のビットプレーンへ詰め替え、lut_dense/group_sumをその形の
 * まま評価する(算術強度をバッチ方向に約32倍に上げる)。適格性の判定
 * (batch>=32・全主経路opがlut_dense/group_sumのみ・conv/recurrent非混在)
 * は `deploy/plan.py::dense_packed_eligible` が生成時に行う。
 *
 * `batch` 引数はいずれもこの呼び出しでの**実行時**サンプル数(確保上限
 * ではない。既存のバッチ透過カーネルの`batch`引数と同じ意味)。
 * bwords=ceil(batch/32) は各関数がホスト側で計算する。 */

/** uchar入力(現行アップロード形式のまま、[batch][width])→
 *  u32パック平面(`out[neuron*bwords+w]`、CPU版mdl_pack_batchと同一
 *  レイアウト)。`x`/`out`は別バッファであること(インプレース不可 ─
 *  読み取りが全レーンぶん終わる前に他スレッドが上書きしうる)。 */
int mdl_metal_pack_batch(mdl_metal_buf x, mdl_metal_buf out, int width,
                         int batch);
/** lut_dense のバッチ軸パック版。`x`/`out`はどちらもパック済みu32平面
 *  (`x[in_idx*bwords_in+w]`/`out[o*bwords+w]`)。n_inputs==2はbutterfly
 *  -select、3〜5は最小項の論理和(CPU版mdl_lut_layer_packedと同一式)。 */
int mdl_metal_lut_layer_packed(mdl_metal_buf x, mdl_metal_buf out,
                               int n_inputs, int out_width,
                               mdl_metal_buf indices, mdl_metal_buf tt,
                               int batch);
/** 添字uint16_t版(同じ判定、in_width<=65536のときだけ選ばれる)。*/
int mdl_metal_lut_layer_packed_u16(mdl_metal_buf x, mdl_metal_buf out,
                                   int n_inputs, int out_width,
                                   mdl_metal_buf indices, mdl_metal_buf tt,
                                   int batch);
/** group_sum のバッチ軸パック版。`x`はパック済みu32平面、`out`は非パック版
 *  と同じint32レイアウト(`scores[sample*k+g]`)に直接書く ─ この後段に
 * アンパックは要らない。CPU版mdl_group_sum_packed(縦カウンタ)と
 *  同じ式。 */
int mdl_metal_group_sum_packed(mdl_metal_buf x, mdl_metal_buf out,
                               int in_width, int k, int batch);

/* ---- RDDLGN自己回帰生成(decode)ハーネス専用 ----
 * IRのop語彙には無い(argmax/埋め込み表引きは論理回路で表せない
 * blackbox相当)。deploy/rddlgn_decode_gpu.pyが直接呼び出す。 */

/** tokens[b](int32、batch本)を table[vocab][d_emb](0/1バイト、行=トークン)
 *  で引き、out[b*d_emb+j] = table[tokens[b]][j] へ書く。 */
int mdl_metal_embed_lookup(mdl_metal_buf tokens, mdl_metal_buf table,
                           mdl_metal_buf out, int d_emb, int batch);
/** scores(int32、[batch][width])の行ごとにargmax(同点は先勝ち)を取り、
 *  tokens(int32、batch本)へ書く。 */
int mdl_metal_argmax(mdl_metal_buf scores, mdl_metal_buf tokens, int width,
                     int batch);

/** RDDLGN decodeのmegakernel経路(opt-in、`deploy/
 *  rddlgn_decode_megakernel.py`)。tickループ全体を1カーネル内(1
 *  スレッドグループ=1レーン、threadgroupバリアで層間同期)へ畳み、
 *  ホスト側は1回のdispatchで済ませる。`idx_bufs`/`tt_bufs`は
 *  dec_l_rec+dec_pm_recの全lut_dense層ぶん(層順)、`tg_bytes`は
 *  threadgroupバッファ1本(全中間値を詰めた単一アリーナ)の総バイト数。
 *  32KBに収まらないモデルはこの関数を使う経路自体を生成しない
 *  (生成時に従来経路へ自動フォールバックする)。*/
int mdl_metal_rddlgn_decode_megakernel_exec(
    const char *kernel_name, mdl_metal_buf d_token, mdl_metal_buf d_table,
    mdl_metal_buf d_context, mdl_metal_buf d_pstate, mdl_metal_buf d_out,
    mdl_metal_buf *idx_bufs, mdl_metal_buf *tt_bufs, int n_layers,
    int tg_bytes, int n);

/** 積んだコマンドをGPUへ流して完了を待つ(結果読み出しの前に呼ぶ)。 */
int mdl_metal_sync(void);

/* ---- ストリーム・パイプライン化。
 *
 * 背景: 1推論=1コマンドバッファ+即 mdl_metal_sync という既存の設計は、
 * 推論のたびに「ドライバ由来の床」(実測0.25〜0.35ms。wait方式を
 * 変えても消えないことは確認済み)を毎回払う。連続する多数枚を流すとき、
 * 「次の枚をencode+commitしてから前の枚をwaitする」向きに並べ替えれば、
 * CPUがwaitで止まっている時間を次の枚のencode作業と重ねられる
 * (GPU自体は同一コマンドキュー上でcommit順に実行される — 複数の
 * command bufferが同時に走るわけではない。隠れるのはCPU側の待ち時間)。
 *
 * 使い方(スロット0..MDL_METAL_MAX_STREAMS-1は独立したバッファ集合を指す。
 * `md deploy codegen-metal --n-streams N` で生成したモデルが
 * `mdl_metal_infer_batch_submit`/`_collect`(入口の実体は生成コード側)を
 * 持つ):
 *
 *     mdl_metal_infer_batch_submit(bits0, n0, 0);    // slot0へcommit(待たない)
 *     mdl_metal_infer_batch_submit(bits1, n1, 1);    // slot1へcommit(待たない)
 *     mdl_metal_infer_batch_collect(scores0, n0, 0); // slot0を待って読み出す
 *     mdl_metal_infer_batch_submit(bits2, n2, 0);    // slot0を再利用(直前にcollect済みなので安全)
 *     mdl_metal_infer_batch_collect(scores1, n1, 1);
 *     ...
 *
 * 安全性: 同じスロットを mdl_metal_wait_slot(=collectの内部)を挟まずに
 * 二重に commit_async すると、前回ぶんをGPUがまだ読み書きしている
 * 可能性があるバッファへ新しい入力を上書きすることになり結果が壊れる
 * (教訓 — バッファ共有の誤りは1回きりの実行では見えない)。
 * そのため `mdl_metal_commit_async` は「そのスロットがまだpendingのとき」
 * エラー(-1)を返して拒否する(黙って壊れない)。 */

/** そのスロットに積んだコマンドをコミットする(**待たない**)。
 *  戻り値: 0=OK、-1=範囲外/未初期化/そのスロットが既にpending
 *  (collectし忘れ)/積んだものが無い。 */
int mdl_metal_commit_async(int slot);
/** そのスロットの直近の mdl_metal_commit_async を待つ。何も積んでいなければ
 *  即0を返す(冪等)。 */
int mdl_metal_wait_slot(int slot);

/* ---- GPU時間の直接計測。既定OFF、計測時のみ呼び出し側が有効化する。
 *
 * 実測(Apple M3, macOS 26.6.2): device.counterSets に
 * MTLCommonCounterSetTimestamp が1本あり、supportsCounterSampling は
 * MTLCounterSamplingPointAtStageBoundary のみ true
 * (AtDispatchBoundary/AtDrawBoundary/AtBlitBoundary/AtTileDispatchBoundary は
 * すべて false)。つまり「エンコーダ単位」の開始/終了タイムスタンプしか
 * 取れず、1エンコーダに複数dispatchを積んだ場合のdispatchごとの内訳は
 * 取れない。そのため計測ON時は**カーネル呼び出し1回=1エンコーダ**に
 * 分割し直してサンプルする(既定OFF時の「1フレーム=複数dispatchを1つの
 * エンコーダにまとめ流す」経路とは別コードパスにして、計測しない限り
 * オーバーヘッドが既存経路に一切乗らないようにしてある)。
 *
 * resolveCounterRange: が返す MTLCounterResultTimestamp.timestamp は
 * 実測でナノ秒そのもの(cmd.GPUStartTime/GPUEndTime[秒]と桁を合わせて
 * 突き合わせ済み)。mach_absolute_time系(sampleTimestamps:gpuTimestamp:)の
 * ような timebase 変換は不要だった。 */

/** 計測のON/OFF。0=OFF(既定)。ONにすると内部でカウンタサンプルバッファを
 *  確保する(初回のみ)。mdl_metal_init系の後に呼ぶこと。
 *  環境変数 MDL_METAL_GPU_TIMING=1 でも既定を上書きできる(init時に読む)。 */
void mdl_metal_set_gpu_timing(int enable);
/** 現在の状態。 */
int mdl_metal_gpu_timing_enabled(void);
/** per-dispatchのカウンタサンプリングに対応しているデバイスか
 *  (0ならcmd.GPUStartTime/GPUEndTimeのコマンドバッファ粒度にフォールバック
 *  していて、カーネル別内訳は取れない)。 */
int mdl_metal_gpu_timing_supported(void);
/** 直近のリセット以降の累積GPU実行時間(ミリ秒)。 */
double mdl_metal_gpu_ms(void);
/** カーネル名別の累積内訳を整形した文字列を返す(次に呼ぶまで有効な
 *  内部バッファを指す)。mdl_metal_gpu_timing_supported()==0のときは
 *  内訳が無い旨の1行を返す。 */
const char *mdl_metal_gpu_breakdown(void);
/** GPU計測の累積値をリセットする(encode_ms/wait_msのmdl_metal_reset_statsとは
 *  別。既存の呼び出し規約を変えないため独立関数にしてある)。 */
void mdl_metal_reset_gpu_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* MDL_METAL_H */
