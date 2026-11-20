/* griffith-logic — RDDLGN自己回帰生成ハーネス(語彙折り畳み対応)
 *
 * 教師強制(mdl_infer/mdl_run)は生成コード(model.c)だけで完結するが、
 * 自己回帰生成(前ステップの出力トークンを次ステップの入力へ埋め込み直す
 * ループ)は「トークン列をargmaxで選ぶ」「埋め込み表を引く」という、
 * 論理ゲート回路では表せない処理を挟む(blackboxと同種の線引き。
 * デプロイキット設計メモ節の申し送り参照)。そのためIR化せず、
 * ここに手書きする。
 *
 * RDDLGN(export/rddlgn_ir.py::export_rddlgn_ir)が生成するIRは常に同じ
 * ノードid(enc_n_rec/enc_k_rec/dec_l_rec/dec_pm_rec、x_src)を使うため、
 * `deploy/cbackend.py::generate_rddlgn_c`でビルドしたmodel.h/model.cは
 * 常に下記のシンボルを公開する(このヘッダはmodel.hを直接includeし、
 * それらの宣言をそのまま使う):
 *
 *   エンコーダ:
 *     mdl_out_x_src(ctx) / MDL_BUF_WIDTH_X_SRC
 *     mdl_run_recurrent_enc_n_rec(ctx)
 *     mdl_out_enc_n_rec(ctx) / MDL_OUT_WIDTH_ENC_N_REC(追加、折り畳み用)
 *     mdl_run_recurrent_enc_k_rec(ctx)
 *     mdl_out_enc_k_rec(ctx) / MDL_OUT_WIDTH_ENC_K_REC
 *   デコーダ(L-layer単発評価):
 *     mdl_ext_dec_l_rec_sv_step(ctx) / MDL_EXT_WIDTH_DEC_L_REC_SV_STEP
 *     mdl_step_dec_l_rec(ctx, t)
 *     mdl_out_dec_l_rec(ctx)
 *   デコーダ(P+M+GroupSum単発評価):
 *     mdl_reset_state_dec_pm_rec(ctx)
 *     mdl_state_dec_pm_rec_sv_p(ctx) / MDL_STATE_WIDTH_DEC_PM_REC_SV_P
 *     mdl_ext_dec_pm_rec_sv_context(ctx) / MDL_EXT_WIDTH_DEC_PM_REC_SV_CONTEXT
 *     mdl_ext_dec_pm_rec_sv_l(ctx) / MDL_EXT_WIDTH_DEC_PM_REC_SV_L
 *     mdl_step_dec_pm_rec(ctx, t, scores) / MDL_OUT_WIDTH_DEC_PM_REC
 *     MDL_STEPS_DEC_PM_REC(教師強制のtgt_len = max_new_tokensの上限)
 *   埋め込み表:
 *     mdl_embed_lookup(token, out_bits) / MDL_VOCAB_SIZE / MDL_D_EMB
 * 語彙折り畳み表(export/rddlgn_ir.py::extract_folded_n_table/
 *   extract_folded_l_tableが自動判定で折り畳んだ場合だけ定義される):
 *     #ifdef MDL_HAS_FOLDED_N_TABLE
 *       mdl_folded_n_lookup(token, out_bits) / MDL_FOLDED_N_TABLE_WIDTH
 *     #ifdef MDL_HAS_FOLDED_L_TABLE
 *       mdl_folded_l_lookup(token, out_bits) / MDL_FOLDED_L_TABLE_WIDTH
 *
 * 依存はmodel.hだけ(コアmdl_kernels.cは非依存。pthreadも使わない)。
 * argmaxはここに自前で持つ(mdl_kernels.cのmdl_argmaxと同じ「同点は
 * 先勝ち(strict >)」規約だが、意図的に依存を増やさないため複製する)。
 */
#ifndef MDL_RDDLGN_H
#define MDL_RDDLGN_H

#include <stdint.h>

#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MDL_RDDLGN_WEIGHTS_DROPPED
/** 自己回帰貪欲デコード。
 *
 *  内部で毎回: (1) src_bitsからエンコーダを1回実行してcontextを得る、
 *  (2) デコーダのstate(P-layer出力)を初期値へ再初期化する — つまり
 *  **同じctxで複数回呼んでも呼び出しは独立**(前回の生成が残らない)。
 *
 *  Args:
 *    ctx: mdl_ctx_init済みのコンテキスト。作業領域は通常推論と共有する。
 *    src_bits: MDL_BUF_WIDTH_X_SRC(= src_len * MDL_D_EMB)個の0/1バイト
 *      (埋め込み・Heaviside済み。呼び出し側がsrcトークン列を
 *      mdl_embed_lookupで埋め込んでおく)。
 *    bos_token: デコーダの最初の入力トークン。
 *    eos_token: これが出力されたら生成を打ち切る(出力には含める)。
 *    max_new_tokens: 生成する最大トークン数。MDL_STEPS_DEC_PM_REC
 *      (コンパイル時上限、教師強制のtgt_len)を超えると-1を返す。
 *    out_tokens: 呼び出し側が MDL_STEPS_DEC_PM_REC 個ぶん確保する
 *      (ヒープ確保はしない)。
 *
 *  Returns:
 *    生成したトークン数(EOSを含む)。0以上。
 *    引数不正・max_new_tokens超過のときは負値。
 *
 *  `MDL_RDDLGN_WEIGHTS_DROPPED`が定義されている(*  deploy/cbackend.py::generate_rddlgn_cをdrop_folded_weights=Trueで
 *  呼んだ生成物)ときは、この関数自体が丸ごと消える — N-layers/
 *  L-layersの重みがmodel.cに無く、この関数はリンクできないため。
 *  そのビルドでは`mdl_rddlgn_generate_greedy_folded`だけを使うこと。
 */
int mdl_rddlgn_generate_greedy(mdl_ctx *ctx, const uint8_t *src_bits,
                               uint32_t bos_token, uint32_t eos_token,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens);
#endif /* !MDL_RDDLGN_WEIGHTS_DROPPED */

/** 自己回帰貪欲デコード、語彙折り畳み対応版。
 *
 *  export/rddlgn_ir.py::extract_folded_n_table/extract_folded_l_table が
 *  bundle['folded_n_table']/['folded_l_table']として焼き込んだ表が
 *  あれば(`MDL_HAS_FOLDED_N_TABLE`/`MDL_HAS_FOLDED_L_TABLE`が定義済み)、
 *  Embedding+N-layers(エンコーダ側)・Embedding+L-layers(デコーダ側)を
 *  評価する代わりに `mdl_folded_n_lookup`/`mdl_folded_l_lookup` の表引き
 *  1回に置き換える(N/L-layersのゲート評価そのものが消える)。マクロが
 *  無いモデル(語彙が大きく折り畳みを自動的に見送った場合)では、内部で
 *  `mdl_embed_lookup` + 通常のN/L-layers評価にフォールバックする —
 *  **この関数はどちらのモデルでも常に使え、`mdl_rddlgn_generate_greedy`
 *  (src_bits版)とビット厳密に同じ結果を返す**(折り畳み経路と
 *  非折り畳み経路の整数厳密一致は tests/deploy/
 *  test_c_rddlgn_folded_greedy.py で固定してある)。
 *
 *  `mdl_rddlgn_generate_greedy`との違いはただ1つ: srcを「呼び出し側が
 *  埋め込み済みのビット列」ではなく「トークンID列」で受け取る点
 *  (折り畳み経路はトークンIDが無いと表引きできないため。埋め込みは
 *  この関数の内部で行う — 折り畳みが効いている区間はN/L-layersごと
 *  スキップされ、効いていない区間は`mdl_embed_lookup`を内部で呼ぶ)。
 *
 *  Args:
 *    ctx: mdl_ctx_init済みのコンテキスト。
 *    src_tokens: MDL_STEPS_ENC_N_REC(= エンコーダに焼き込まれたsrc_len)
 *      個のトークンID。
 *    bos_token / eos_token / max_new_tokens / out_tokens:
 *      mdl_rddlgn_generate_greedyと同じ契約。
 *
 *  Returns:
 *    生成したトークン数(EOSを含む)。0以上。
 *    引数不正・max_new_tokens超過のときは負値。
 */
int mdl_rddlgn_generate_greedy_folded(mdl_ctx *ctx, const uint32_t *src_tokens,
                                      uint32_t bos_token, uint32_t eos_token,
                                      uint32_t max_new_tokens,
                                      uint32_t *out_tokens);

/* mdl_rddlgn_generate_greedy_packed32はmodel.hが
 * `packed_teacher_forcing='dec_pm_rec'`付きで生成されたときだけ
 * 定義される(mdl_rddlgn.c参照)。宣言もそれに合わせてガードする
 * (このヘッダはmodel.hを先にincludeしているので、model.h側のマクロが
 * ここで既に見えている)。フラグ無しで生成したmodel.cとリンクする既存の
 * 全呼び出し元(md deploy等)は無改造で通る。 */
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES

#define MDL_RDDLGN_LANES 32

/** greedy_packed32の作業バッファ(は関数内staticだったもの)。
 *
 *  呼び出し側(または呼び出し側が用意するスレッドローカル領域)が確保する。
 *  幅はすべてmodel.hの既存マクロ(mdl_rddlgn_generate_greedyが元から使う
 *  MDL_OUT_WIDTH_ENC_K_REC/MDL_STATE_WIDTH_DEC_PM_REC_SV_P/
 *  MDL_OUT_WIDTH_DEC_PM_REC、生やすMDL_DEC_PM_REC_ROW_WIDTH)だけで
 * 決まるので、cbackend.py側の変更は不要(関数内static配列を
 *  そのままcaller-owned structへ移しただけ)。
 *
 * 複数の`mdl_ctx`を複数コアで並行に回す(マルチコア化)ときは、
 *  ctx・pwork(tail用)と同様にこのscratchも**コアごとに別インスタンス**
 *  を渡すこと — mdl_run_poolの`ctxs[]`と同じ「読み取り専用は重みだけ、
 *  書き込みは全部own」規律に従えば、結果は逐次実行と完全に一致する
 *  (構造的に安全。tests/deploy/test_c_rddlgn_greedy_packed32.py の
 *  スレッドプール版テストで固定)。
 */
typedef struct {
  uint8_t context[MDL_RDDLGN_LANES][MDL_OUT_WIDTH_ENC_K_REC];
  uint8_t p_prev[MDL_RDDLGN_LANES][MDL_STATE_WIDTH_DEC_PM_REC_SV_P];
  uint8_t rows[MDL_RDDLGN_LANES][MDL_DEC_PM_REC_ROW_WIDTH];
  int32_t scores[MDL_RDDLGN_LANES][MDL_OUT_WIDTH_DEC_PM_REC];
} mdl_rddlgn_g32_scratch;

/** 32本の独立した系列を1ワードの32レーンで同時にgreedy生成する
 *  (教師強制の時間バッチ化と対をなす「系列バッチ化」)。
 *
 *  各系列のL/P(state更新)は真のリカレントなので系列ごとに単発評価する
 *  しかないが、M-layers+GroupSum(state_tだけの純関数)は32系列ぶん
 *  まとめて`mdl_<id>_tail_packed32`(deploy/cbackend.pyが
 *  `packed_teacher_forcing`と同時に生やす)で1回評価する。
 *  値は`mdl_rddlgn_generate_greedy`を32回独立に呼んだ場合と
 *  トークン列が完全一致する(tests/deploy/test_c_rddlgn_greedy_packed32.py)。
 *
 *  依存: model.hがdeploy/cbackend.py::generate_rddlgn_cを
 *  `packed_teacher_forcing='dec_pm_rec'`付きで生成したものであること
 *  (`mdl_step_dec_pm_rec_head`/`mdl_dec_pm_rec_tail_packed32`/
 *  `MDL_DEC_PM_REC_ROW_WIDTH`/`MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES`
 *  が公開されている前提)。row のバイト並びは
 *  `[p_t(MDL_STATE_WIDTH_DEC_PM_REC_SV_P)]
 *   [context(MDL_EXT_WIDTH_DEC_PM_REC_SV_CONTEXT)]
 *   [l_t(MDL_EXT_WIDTH_DEC_PM_REC_SV_L)]`
 *  で決め打ち(models/rddlgn.py::RDDLGNDecoder.forwardの
 *  `m_input = cat([p_t, context, l_t])`と同じ順序)。
 *
 * 内部の作業バッファ(はstatic)を呼び出し側が渡す
 *  `mdl_rddlgn_g32_scratch`へ一般化した。**別々の`scratch`(および
 *  別々の`ctx`/`pwork`)を渡す限り、この関数はスレッドセーフ**
 * ——複数コアで複数の「32本バッチ」を並行に処理できる(
 *  マルチコア化がこれを`mdl_run_pool`型のプールと組み合わせて使う)。
 *  paper_scale1規模で1インスタンス約7.5MBあるため、スタックではなく
 *  静的配列かヒープに置くこと。
 *
 *  Args:
 *    ctx: mdl_ctx_init済みのコンテキスト(1個で32系列ぶんを順に使い回す)。
 *    scratch: このバッチ専用のscratch(呼び出し側が用意)。他のバッチと
 *      共有しないこと。
 *    src_bits32: [32][MDL_BUF_WIDTH_X_SRC] の0/1バイト(系列ごとに別srcを
 *      許す。同じ配列を32回複製して渡せば「同一プロンプトから32本」も
 *      表現できる)。
 *    bos_tokens: 系列ごとの開始トークン([32]、全系列で同じ値でもよい)。
 *    eos_token: 全系列共通のEOS(これが出た系列はそれ以降の反復をスキップ
 *      するが、他系列の計算は続く — packed tailはlanes=32固定のため)。
 *    max_new_tokens: 生成する最大トークン数(MDL_STEPS_DEC_PM_REC以下)。
 *    pwork/pwork_bytes: packed tail用の作業領域(呼び出し側が用意する。
 *      MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES以上。このバッチ専用の
 *      インスタンスを渡すこと)。
 *    out_tokens: [32][MDL_STEPS_DEC_PM_REC] へ系列ごとの生成トークン列。
 *    out_lens: [32] へ系列ごとの生成トークン数(EOSを含む)。
 *
 *  Returns: 0=成功、負値=引数不正。
 */
int mdl_rddlgn_generate_greedy_packed32(
    mdl_ctx *ctx, mdl_rddlgn_g32_scratch *scratch,
    const uint8_t *src_bits32, const uint32_t *bos_tokens,
    uint32_t eos_token, uint32_t max_new_tokens,
    uint32_t *pwork, size_t pwork_bytes,
    uint32_t *out_tokens, int32_t *out_lens);

#endif /* MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES */

#ifdef __cplusplus
}
#endif

#endif /* MDL_RDDLGN_H */
