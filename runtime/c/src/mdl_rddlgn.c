/* griffith-logic — RDDLGN自己回帰生成ハーネス実装。
 *
 * 設計の要点(デプロイキット設計メモ節):
 *   - エンコーダはmdl_run_recurrent_enc_n_rec/enc_k_recを素直に1回ずつ
 *     呼ぶだけ(教師強制のmdl_run内で起きるのと同じ処理を単独で呼べる
 *     形にしてあるだけなので、値は完全に一致する)。
 *   - デコーダは「tを常に0に固定した単発呼び」で1トークンぶんずつ進める。
 *     dec_l_rec/dec_pm_recのstep関数はstacked出力バッファの[0, w)へしか
 *     書かないので、教師強制側のtgt_len回ぶんのバッファ全部を使わなくても
 *     成立する。
 *   - **他recurrentのstep関数を挟んでも、公開したバッファの生存が
 *     保証されているとは仮定しない**: メモリ計画(deploy/memplan.py)は
 *     教師強制の実行順序(dec_l_recの全tickが終わってからdec_pm_recが
 *     始まる)を前提にアリーナを詰めており、本ハーネスはdec_l_recの
 *     step関数とdec_pm_recのstep関数を**tickごとに交互に**呼ぶ
 *     (l→pm→l→pm→…)。この呼び出し順は教師強制側の前提と異なるため、
 *     dec_l_rec側のstep関数がdec_pm_recの永続バッファ(state/ext)の
 *     アリーナ領域を再利用している可能性がある。よって
 *     context(sv_context)とP-layerのstate(sv_p)はハーネス側の
 *     ローカル配列(スタック)に真実を持ち、dec_pm_recのstep関数を
 *     呼ぶ**直前に毎回書き戻す**(save/restore)。これでアリーナの
 *     再利用がどうなっていても値は必ず正しい — 「たまたま壊れていない」
 *     ではなく構造的に安全。
 */
#include "mdl_rddlgn.h"

#include <string.h>

/* 同点は先勝ち(strict >)。runtime/c/src/mdl_kernels.c::mdl_argmaxと
   同じ規約だが、意図的にコアへ依存を作らないためここに複製する。 */
static int32_t mdl_rddlgn_argmax(const int32_t *scores, int32_t n) {
  int32_t best = 0, i;
  for (i = 1; i < n; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

#ifndef MDL_RDDLGN_WEIGHTS_DROPPED
int mdl_rddlgn_generate_greedy(mdl_ctx *ctx, const uint8_t *src_bits,
                               uint32_t bos_token, uint32_t eos_token,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens) {
  uint8_t *x_src;
  uint8_t *ctx_out;
  uint8_t *l_ext;
  uint8_t *pm_ctx_ext;
  uint8_t *pm_l_ext;
  uint8_t *pm_state;
  uint8_t context_local[MDL_OUT_WIDTH_ENC_K_REC];
  uint8_t p_prev_local[MDL_STATE_WIDTH_DEC_PM_REC_SV_P];
  int32_t scores[MDL_OUT_WIDTH_DEC_PM_REC];
  uint32_t prev_token;
  uint32_t t;

  if (ctx == NULL || ctx->work == NULL || src_bits == NULL
      || out_tokens == NULL) {
    return -1;
  }
  if (max_new_tokens > (uint32_t)MDL_STEPS_DEC_PM_REC) return -1;

  /* 1. エンコーダ: src_bitsをx_srcバッファへ直接書き、N→Kを1回ずつ回す。 */
  x_src = mdl_out_x_src(ctx);
  memcpy(x_src, src_bits, MDL_BUF_WIDTH_X_SRC);
  mdl_run_recurrent_enc_n_rec(ctx);
  mdl_run_recurrent_enc_k_rec(ctx);
  ctx_out = mdl_out_enc_k_rec(ctx);
  memcpy(context_local, ctx_out, sizeof(context_local));

  /* 2. デコーダstateを毎回再初期化する(同じctxで複数回呼んでも
     前回の生成が残らないことを保証する)。 */
  mdl_reset_state_dec_pm_rec(ctx);
  pm_state = mdl_state_dec_pm_rec_sv_p(ctx);
  memcpy(p_prev_local, pm_state, sizeof(p_prev_local));

  l_ext = mdl_ext_dec_l_rec_sv_step(ctx);
  pm_ctx_ext = mdl_ext_dec_pm_rec_sv_context(ctx);
  pm_l_ext = mdl_ext_dec_pm_rec_sv_l(ctx);

  prev_token = bos_token;
  for (t = 0; t < max_new_tokens; t++) {
    uint32_t token;
    uint8_t *l_out;

    /* 3. L-layer単発評価: 前トークンの埋め込みビットを渡し、t=0固定で
       step関数を単発呼びする(stacked出力バッファの先頭スロットを
       使い捨てで再利用する)。 */
    mdl_embed_lookup(prev_token, l_ext);
    mdl_step_dec_l_rec(ctx, 0);
    l_out = mdl_out_dec_l_rec(ctx);
    memcpy(pm_l_ext, l_out, MDL_EXT_WIDTH_DEC_PM_REC_SV_L);

    /* 4. dec_pm_recの永続入力(context/state)を使う直前に必ず書き戻す
       (上のファイルコメント参照。アリーナ再利用の有無に依存しない)。 */
    memcpy(pm_ctx_ext, context_local, sizeof(context_local));
    pm_state = mdl_state_dec_pm_rec_sv_p(ctx);
    memcpy(pm_state, p_prev_local, sizeof(p_prev_local));

    /* 5. P+M+GroupSumを1tickぶん単発評価する。state_update(次のp_prev)
       はstep関数自身がpm_stateへ書き戻す。 */
    mdl_step_dec_pm_rec(ctx, 0, scores);
    memcpy(p_prev_local, pm_state, sizeof(p_prev_local));

    token = (uint32_t)mdl_rddlgn_argmax(scores, MDL_OUT_WIDTH_DEC_PM_REC);
    out_tokens[t] = token;
    if (token == eos_token) return (int)(t + 1);
    prev_token = token;
  }
  return (int)max_new_tokens;
}
#endif /* !MDL_RDDLGN_WEIGHTS_DROPPED */

/* ------------------------------------------------------------------ */
/* 32本の独立系列の系列バッチ化greedy生成 */
/* (scratchバッファを呼び出し側持ちへ一般化し、複数コアでの */
/*  並行実行に対応した)                                                */
/* ------------------------------------------------------------------ */

/* mdl_rddlgn.cはモデル非依存の共有ランタイムファイルなので、
 * `packed_teacher_forcing='dec_pm_rec'`無しで生成されたmodel.hを
 * 使うビルドでもそのままリンクできなければならない。この関数が要る
 * `mdl_step_dec_pm_rec_head`/`mdl_dec_pm_rec_tail_packed32`/
 * `MDL_DEC_PM_REC_ROW_WIDTH`はそのフラグを付けたときだけmodel.hに
 * 出るので、その代表マクロの有無で丸ごとガードする(既存経路を
 * 一切変えない選択制、mdl_rddlgn.hの宣言側ガードと対称)。本体は
 * `mdl_run_recurrent_enc_n_rec`/`mdl_step_dec_l_rec`(bits版と同じ生の
 * 公開API)を直接呼ぶため、`MDL_RDDLGN_WEIGHTS_DROPPED`
 * (語彙折り畳みの配信サイズ削減ビルド)ではこれらのシンボル自体が
 * 無くリンクできない —— ただし`packed_teacher_forcing`と
 * `drop_folded_recurrents`はcbackend.py::CBackend.__init__で同時指定が
 * ValueErrorになる(未対応の組み合わせ)ため、`MDL_DEC_PM_REC_TAIL_
 * PACKED32_WORK_BYTES`と`MDL_RDDLGN_WEIGHTS_DROPPED`が同じmodel.h内に
 * 両方定義されることは構造的に無い(選択の一意な起点はPython側のこの
 * ValueErrorだけにする——ここに二重ガードを足すと、その起点が2箇所に
 * 分散して将来どちらかだけ変更されたときに矛盾しうる)。 */
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES

int mdl_rddlgn_generate_greedy_packed32(
    mdl_ctx *ctx, mdl_rddlgn_g32_scratch *scratch,
    const uint8_t *src_bits32, const uint32_t *bos_tokens,
    uint32_t eos_token, uint32_t max_new_tokens,
    uint32_t *pwork, size_t pwork_bytes,
    uint32_t *out_tokens, int32_t *out_lens) {
  uint32_t prev_token[MDL_RDDLGN_LANES];
  uint8_t done[MDL_RDDLGN_LANES];
  uint8_t *x_src, *ctx_out, *l_ext, *pm_ctx_ext, *pm_l_ext, *pm_state;
  int lane;
  uint32_t t;

  if (ctx == NULL || ctx->work == NULL || scratch == NULL
      || src_bits32 == NULL || bos_tokens == NULL || pwork == NULL
      || out_tokens == NULL || out_lens == NULL) {
    return -1;
  }
  if (max_new_tokens > (uint32_t)MDL_STEPS_DEC_PM_REC) return -1;
  if (pwork_bytes < (size_t)MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES) {
    return -1;
  }

  /* 1. 32本それぞれのエンコーダを1回ずつ回し、context/stateを保存する。
     エンコーダ(N+K)はdecode(P+M+GroupSum)に比べて軽い
     (実測でdecode時間の1〜2%)ので、ここはパックしない
     — 支配項であるM+GroupSumのpackingに集中する。 */
  for (lane = 0; lane < MDL_RDDLGN_LANES; lane++) {
    x_src = mdl_out_x_src(ctx);
    memcpy(x_src, src_bits32 + (size_t)lane * MDL_BUF_WIDTH_X_SRC,
           MDL_BUF_WIDTH_X_SRC);
    mdl_run_recurrent_enc_n_rec(ctx);
    mdl_run_recurrent_enc_k_rec(ctx);
    ctx_out = mdl_out_enc_k_rec(ctx);
    memcpy(scratch->context[lane], ctx_out, sizeof(scratch->context[lane]));

    mdl_reset_state_dec_pm_rec(ctx);
    pm_state = mdl_state_dec_pm_rec_sv_p(ctx);
    memcpy(scratch->p_prev[lane], pm_state, sizeof(scratch->p_prev[lane]));

    prev_token[lane] = bos_tokens[lane];
    done[lane] = 0u;
    out_lens[lane] = 0;
  }

  l_ext = mdl_ext_dec_l_rec_sv_step(ctx);
  pm_ctx_ext = mdl_ext_dec_pm_rec_sv_context(ctx);
  pm_l_ext = mdl_ext_dec_pm_rec_sv_l(ctx);

  for (t = 0; t < max_new_tokens; t++) {
    int32_t rc;

    /* 2. 各系列のL単発評価+P(head、state更新まで)を順に評価し、
       tail(M+GroupSum)へ渡す行を組み立てる。P自体は真のリカレント
       (系列内で前トークンに依存)なので系列ごとに単発評価するしかないが、
       系列間には依存が無いので32本を順に回すだけでよい
       (mdl_rddlgn_generate_greedy.cのsave/restore規律と同じ:
       永続入力を使う直前に必ず書き戻す)。 */
    for (lane = 0; lane < MDL_RDDLGN_LANES; lane++) {
      uint8_t *l_out;
      uint8_t *row = scratch->rows[lane];

      if (done[lane]) continue;   /* 行は前回のまま(tail側は無害に空回り) */

      mdl_embed_lookup(prev_token[lane], l_ext);
      mdl_step_dec_l_rec(ctx, 0);
      l_out = mdl_out_dec_l_rec(ctx);
      memcpy(pm_l_ext, l_out, MDL_EXT_WIDTH_DEC_PM_REC_SV_L);

      memcpy(pm_ctx_ext, scratch->context[lane],
             sizeof(scratch->context[lane]));
      pm_state = mdl_state_dec_pm_rec_sv_p(ctx);
      memcpy(pm_state, scratch->p_prev[lane], sizeof(scratch->p_prev[lane]));

      mdl_step_dec_pm_rec_head(ctx);   /* p_tを計算しstate_bufへ書き戻す */

      pm_state = mdl_state_dec_pm_rec_sv_p(ctx);
      memcpy(scratch->p_prev[lane], pm_state, sizeof(scratch->p_prev[lane]));

      /* row = [p_t, context, l_t](concatの並び。models/rddlgn.py::
         RDDLGNDecoder.forwardのm_input = cat([p_t, context, l_t])と
         同じ順序をここで手で踏む。mdl_dec_pm_rec_tail_packed32は
         この並びで詰められたrowsを前提にしている)。 */
      memcpy(row, scratch->p_prev[lane], MDL_STATE_WIDTH_DEC_PM_REC_SV_P);
      memcpy(row + MDL_STATE_WIDTH_DEC_PM_REC_SV_P, scratch->context[lane],
             MDL_EXT_WIDTH_DEC_PM_REC_SV_CONTEXT);
      memcpy(row + MDL_STATE_WIDTH_DEC_PM_REC_SV_P
                 + MDL_EXT_WIDTH_DEC_PM_REC_SV_CONTEXT,
             l_out, MDL_EXT_WIDTH_DEC_PM_REC_SV_L);
    }

    /* 3. M+GroupSumを32本まとめて1回のpacked評価。 */
    rc = mdl_dec_pm_rec_tail_packed32(pwork, pwork_bytes, &scratch->rows[0][0],
                                      &scratch->scores[0][0]);
    if (rc != 0) return -2;

    /* 4. 系列ごとにargmaxし、EOS判定+次トークン埋め込みの準備。 */
    for (lane = 0; lane < MDL_RDDLGN_LANES; lane++) {
      uint32_t token;
      if (done[lane]) continue;
      token = (uint32_t)mdl_rddlgn_argmax(scratch->scores[lane],
                                          MDL_OUT_WIDTH_DEC_PM_REC);
      out_tokens[(size_t)lane * (uint32_t)MDL_STEPS_DEC_PM_REC + t] = token;
      out_lens[lane] = (int32_t)(t + 1);
      if (token == eos_token) {
        done[lane] = 1u;
      } else {
        prev_token[lane] = token;
      }
    }
  }
  return 0;
}

#endif /* MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES */

/* 語彙折り畳み対応版。Embedding+N-layers/Embedding+L-layersが
 * トークンIDのみの関数(状態非依存)であることを利用し、`MDL_HAS_
 * FOLDED_N_TABLE`/`MDL_HAS_FOLDED_L_TABLE`が定義されている(=
 * export/rddlgn_ir.pyの自動判定が「表のほうが元の重みより小さい」と
 * 判定した)区間だけ、N-layers/L-layersのゲート評価そのものを表引き
 * 1回に置き換える。マクロが無い区間は`mdl_embed_lookup`+通常の
 * recurrent呼び出しにフォールバックするので、この関数はどのモデルでも
 * 常に使え、`mdl_rddlgn_generate_greedy`とビット厳密に同じ結果を返す
 * (両者を突き合わせるテストがtests/deploy/
 * test_c_rddlgn_folded_greedy.pyにある)。 */
int mdl_rddlgn_generate_greedy_folded(mdl_ctx *ctx, const uint32_t *src_tokens,
                                      uint32_t bos_token, uint32_t eos_token,
                                      uint32_t max_new_tokens,
                                      uint32_t *out_tokens) {
  uint8_t *ctx_out;
  uint8_t *pm_ctx_ext;
  uint8_t *pm_l_ext;
  uint8_t *pm_state;
  uint8_t *l_ext;
  uint8_t context_local[MDL_OUT_WIDTH_ENC_K_REC];
  uint8_t p_prev_local[MDL_STATE_WIDTH_DEC_PM_REC_SV_P];
  int32_t scores[MDL_OUT_WIDTH_DEC_PM_REC];
  uint32_t prev_token;
  uint32_t t;

  if (ctx == NULL || ctx->work == NULL || src_tokens == NULL
      || out_tokens == NULL) {
    return -1;
  }
  if (max_new_tokens > (uint32_t)MDL_STEPS_DEC_PM_REC) return -1;

  /* 1. エンコーダ。N-layersが折り畳まれていれば、埋め込み+N-layers評価を
     まるごと飛ばし、`enc_n_rec`のstacked出力バッファ(=K-layersが読む
     ext入力そのもの)へ表引き結果を直接書く。K-layers(enc_k_rec)は
     どちらの経路でも完全に無改造で呼ぶ。 */
#ifdef MDL_HAS_FOLDED_N_TABLE
  {
    uint8_t *n_out = mdl_out_enc_n_rec(ctx);
    for (t = 0; t < (uint32_t)MDL_STEPS_ENC_N_REC; t++) {
      mdl_folded_n_lookup(src_tokens[t],
                          n_out + (size_t)t * MDL_OUT_WIDTH_ENC_N_REC);
    }
  }
#else
  {
    uint8_t *x_src = mdl_out_x_src(ctx);
    for (t = 0; t < (uint32_t)MDL_STEPS_ENC_N_REC; t++) {
      mdl_embed_lookup(src_tokens[t], x_src + (size_t)t * MDL_D_EMB);
    }
  }
  mdl_run_recurrent_enc_n_rec(ctx);
#endif
  mdl_run_recurrent_enc_k_rec(ctx);
  ctx_out = mdl_out_enc_k_rec(ctx);
  memcpy(context_local, ctx_out, sizeof(context_local));

  /* 2. デコーダstateを毎回再初期化する(mdl_rddlgn_generate_greedyと同じ)。 */
  mdl_reset_state_dec_pm_rec(ctx);
  pm_state = mdl_state_dec_pm_rec_sv_p(ctx);
  memcpy(p_prev_local, pm_state, sizeof(p_prev_local));

  pm_ctx_ext = mdl_ext_dec_pm_rec_sv_context(ctx);
  pm_l_ext = mdl_ext_dec_pm_rec_sv_l(ctx);
#ifdef MDL_HAS_FOLDED_L_TABLE
  /* L-layersが折り畳まれている(=weights dropped ビルドではdec_l_recの
     公開APIごと消えている可能性がある)ときは絶対に呼ばない。 */
  l_ext = NULL;
  (void)l_ext;
#else
  l_ext = mdl_ext_dec_l_rec_sv_step(ctx);
#endif

  prev_token = bos_token;
  for (t = 0; t < max_new_tokens; t++) {
    uint32_t token;

    /* 3. L-layer単発評価。折り畳まれていれば埋め込み+L-layers評価を
       まるごと表引き1回に置き換え、dec_pm_recのstep入力(sv_l)へ
       直接書く。 */
#ifdef MDL_HAS_FOLDED_L_TABLE
    mdl_folded_l_lookup(prev_token, pm_l_ext);
#else
    {
      uint8_t *l_out;
      mdl_embed_lookup(prev_token, l_ext);
      mdl_step_dec_l_rec(ctx, 0);
      l_out = mdl_out_dec_l_rec(ctx);
      memcpy(pm_l_ext, l_out, MDL_EXT_WIDTH_DEC_PM_REC_SV_L);
    }
#endif

    /* 4. dec_pm_recの永続入力(context/state)を使う直前に必ず書き戻す
       (mdl_rddlgn_generate_greedyと同じ理由。アリーナ再利用の有無に
       依存しない)。 */
    memcpy(pm_ctx_ext, context_local, sizeof(context_local));
    pm_state = mdl_state_dec_pm_rec_sv_p(ctx);
    memcpy(pm_state, p_prev_local, sizeof(p_prev_local));

    /* 5. P+M+GroupSumを1tickぶん単発評価する。 */
    mdl_step_dec_pm_rec(ctx, 0, scores);
    memcpy(p_prev_local, pm_state, sizeof(p_prev_local));

    token = (uint32_t)mdl_rddlgn_argmax(scores, MDL_OUT_WIDTH_DEC_PM_REC);
    out_tokens[t] = token;
    if (token == eos_token) return (int)(t + 1);
    prev_token = token;
  }
  return (int)max_new_tokens;
}
