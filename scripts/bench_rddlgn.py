#!/usr/bin/env python
"""RDDLGN デプロイキットの4指標ベースライン。

RDDLGN(教師強制=`mdl_run`、自己回帰生成=`mdl_rddlgn_generate_greedy`)が
deploykitへ載った。ここでは現在の最適化に先立つベースラインを、
`scripts/bench_deploy.py`と同じ4指標(レイテンシ/スループット/
配信サイズ/メモリ)で確立する。RDDLGNはrecurrent構造を持つため、
bench_deploy.pyのドライバをそのまま使えず、`mdl_rddlgn_generate_greedy`
(自己回帰生成)と`mdl_infer_ctx`(教師強制、既存の`mdl_run`と同一契約)の
両方を測る専用ドライバを新設する。

対象4モデル:
    copy_tiny / reverse_deep / charlm(主役) / paper_scale1(乱数初期化)

使い方:
    uv run python scripts/bench_rddlgn.py                    # 全モデル
    uv run python scripts/bench_rddlgn.py --models copy_tiny,charlm
    uv run python scripts/bench_rddlgn.py --skip-verify       # 検証を飛ばして高速反復
    uv run python scripts/bench_rddlgn.py --opt 1             # 最適化レベルを固定
"""

import argparse
import gzip
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from griffith_logic.convert.text_teacher import (  # noqa: E402
    decode_shakespeare_char,
    encode_shakespeare_char,
)
from griffith_logic.deploy.bundle import build_bundle  # noqa: E402
from griffith_logic.deploy.cbackend import generate_rddlgn_c  # noqa: E402
from griffith_logic.deploy.verify import RUNTIME_C  # noqa: E402
from griffith_logic.export.rddlgn_ir import (  # noqa: E402
    export_rddlgn_ir,
    extract_embed_table,
    extract_folded_l_table,
    extract_folded_n_table,
    simulate_rddlgn_greedy_bits,
)
from griffith_logic.ir.serialize import ir_to_dict  # noqa: E402
from griffith_logic.ir.simulate import simulate  # noqa: E402
from griffith_logic.ir.validate import validate  # noqa: E402
from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig  # noqa: E402

OUT_DIR = REPO / 'results' / 'deploy'
BUILD_ROOT = OUT_DIR / 'build_bench_rddlgn'

#: TOKEN_BOS/TOKEN_EOS(experiments/data/synthetic.py)。copy/reverseの
#: 学習規約(PAD=0, BOS=1, EOS=2)。
SYN_BOS, SYN_EOS = 1, 2
#: data/char_corpus.pyのBOS(66)。char-LMはEOSを明示的に学習していないので
#: 検証用には語彙外の大きな値を使う(=決して一致しない→打ち切りなしで
#: max_new_tokens全部を検証できる)。
CHAR_BOS = 66
#: 語彙に絶対含まれない値(perfドライバでeos早期終了を避け、生成ステップ数を
#: モデル間で揃えるため全モデル共通で使う)。
INVALID_EOS = 0xFFFFFFFE


# ---------------------------------------------------------------------------
# モデル読み込み
# ---------------------------------------------------------------------------
def load_ckpt_model(path: Path) -> tuple[RDDLGN, RDDLGNConfig]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = RDDLGNConfig(**ckpt['metadata']['config'])
    model = RDDLGN(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def load_paper_model(scale: float = 1.0, seed: int = 0) -> tuple[RDDLGN, RDDLGNConfig]:
    torch.manual_seed(seed)
    cfg = RDDLGNConfig.from_paper(scale=scale)
    model = RDDLGN(cfg)
    model.eval()
    return model, cfg


MODEL_SPECS = {
    'copy_tiny': dict(
        loader=lambda: load_ckpt_model(
            REPO / 'checkpoints/rddlgn_copy_tiny_20260815_074318/best.pt'),
        reps=100, bos=SYN_BOS, eos=SYN_EOS,
        desc='copy task, tiny (soft94.9/hard64.6, vocab=50, seq_len=4)'),
    'reverse_deep': dict(
        loader=lambda: load_ckpt_model(
            REPO / 'checkpoints/rddlgn_reverse_custom_20260815_081056/best.pt'),
        reps=100, bos=SYN_BOS, eos=SYN_EOS,
        desc='reverse task, custom/deep (soft98.1/hard81.5, 242K params)'),
    'charlm': dict(
        loader=lambda: load_ckpt_model(
            REPO / 'checkpoints/rddlgn_charlm_scale005/'
                   '2wsppkmu_alpha1.0_pureCE_best.pt'),
        reps=50, bos=CHAR_BOS, eos=66_000,
        desc='char-LM (paper scale=0.05, vocab=67, seq_len=32, 76,286 gates)'),
    'paper_scale1': dict(
        loader=lambda: load_paper_model(1.0),
        reps=20, bos=0, eos=16_000_000,
        desc='RDDLGNConfig.from_paper(scale=1.0), random init (整数一致+外挿のみ)'),
}


def gate_counts(cfg: RDDLGNConfig) -> dict:
    """N/K/L/P/M各層のゲート数(=出力幅の総和)。パラメータ数ではなくゲート数。

    softmaxパラメータ化は1ゲートあたり2^(2^n_inputs)個の重み(n_inputs=2なら16)
    を持つため、count_parameters()の比はゲート数比と一致しない。事前の
    予測M比率(copy_tiny 45.9%/reverse 33.9%/charlm・paper 83.9%)
    はこちらの定義で計算されたもの(実測して確認済み)。
    """
    n = sum(cfg.n_layer_sizes)
    k = sum(cfg.k_layer_sizes)
    lay = sum(cfg.l_layer_sizes)
    p = sum(cfg.p_layer_sizes)
    m = sum(cfg.m_layer_sizes)
    total = n + k + lay + p + m
    return {'N': n, 'K': k, 'L': lay, 'P': p, 'M': m, 'total': total,
            'm_ratio': m / total if total else 0.0}


# ---------------------------------------------------------------------------
# IR/バンドル/C生成
# ---------------------------------------------------------------------------
def build_doc(model: RDDLGN, cfg: RDDLGNConfig, *,
             force_n: bool | None = None, force_l: bool | None = None,
             optimize_degenerate: bool = False) -> dict:
    graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                             optimize_degenerate=optimize_degenerate)
    errs = validate(graph)
    assert errs == [], errs
    doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    # 語彙折り畳み(既定は自動判定。表サイズ>=N/L-layers重みサイズなら
    # None のままでbundleに節を追加しない — paper_scale1のような大語彙は
    # 自動的に対象外になる。force_n/force_lはbefore/after計測用)。
    doc['folded_n_table'] = extract_folded_n_table(model, force=force_n)
    doc['folded_l_table'] = extract_folded_l_table(model, force=force_l)
    return doc


def count_lut_gates(graph) -> dict:
    """IR中の全lut_layerノードのout_width合計(recurrent bodyへ再帰)。

    削減効果を「配信される物理ゲート数」で数える(analyze.pyの
    静的degen_rateはtt単体の退化率であり、層をまたいだ定数伝播の連鎖
    (デプロイキット設計メモ節)を含まないため、実際の削減率は
    これより大きくなりうる — before/afterはこの関数で直接数える)。
    """
    total = 0
    per_id: dict = {}

    def walk(g):
        nonlocal total
        for node in g.nodes:
            if node.type == 'lut_layer':
                w = int(node.params['out_width'])
                total += w
                per_id[node.id] = w
            elif node.type == 'recurrent':
                walk(node.params['body'])

    walk(graph)
    return {'total': total, 'per_id': per_id}


def measure_folded_delivery_savings(model: RDDLGN, cfg: RDDLGNConfig,
                                    cc: str, opt: int, work_root: Path) -> dict:
    """配信サイズbefore/after: 折り畳み無し(baseline)と
    折り畳み+重み削除(drop_folded_weights=True)の`.rodata`(__TEXT,__const)
    バイト数を比べる。

    本編のbench_one()が使うbundle(折り畳みは自動判定するが重みは残す
    ―mdl_infer/mdl_run・bits版greedyを引き続き使えるようにするための
    設計、デプロイキット設計メモ節参照)とは**別に**、この関数
    専用の使い捨てbundle2種をビルドする(教師強制検証やperf計測とは
    無関係、.rodataサイズだけが目的)。
    """
    baseline_doc = build_doc(model, cfg, force_n=False, force_l=False)
    baseline_bundle = build_bundle(baseline_doc)
    dropped_doc = build_doc(model, cfg)   # 自動判定(auto)
    dropped_bundle = build_bundle(dropped_doc)
    both_folded = (dropped_bundle['folded_n_table'] is not None
                  and dropped_bundle['folded_l_table'] is not None)

    def const_bytes(bundle: dict, tag: str, *, drop: bool) -> int:
        work = work_root / f'_sizeonly_{tag}'
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  drop_folded_weights=drop)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        for rel in ('include/mdl_kernels.h', 'include/mdl_simd.h',
                   'include/mdl_stdio.h', 'src/mdl_kernels.c'):
            shutil.copy2(RUNTIME_C / rel, work / Path(rel).name)
        cc_cmd = [cc, '-std=c99', f'-O{opt}', f'-I{work}', '-c',
                 str(work / 'model.c'), '-o', str(work / 'model.o')]
        r = sh(cc_cmd)
        if r.returncode != 0:
            raise RuntimeError(f'{tag}のビルドに失敗:\n{r.stderr[:2000]}')
        return section_sizes(work / 'model.o').get('const', 0)

    baseline_bytes = const_bytes(baseline_bundle, 'baseline', drop=False)
    result = {'baseline_const_bytes': baseline_bytes, 'both_folded': both_folded}
    if both_folded:
        dropped_bytes = const_bytes(dropped_bundle, 'dropped', drop=True)
        result['dropped_const_bytes'] = dropped_bytes
        result['savings_bytes'] = baseline_bytes - dropped_bytes
        result['savings_pct'] = (
            100.0 * result['savings_bytes'] / baseline_bytes
            if baseline_bytes else 0.0)
    else:
        result['dropped_const_bytes'] = None
        result['savings_bytes'] = 0
        result['savings_pct'] = 0.0
    return result


def emit_c_files(bundle: dict, work: Path) -> bool:
    """バンドルからCソース一式を生成する。packed_tfが使えたかを返す。

    packed_teacher_forcing='dec_pm_rec'は既存経路に追加する
    entrypoint mdl_infer_packed_tf_ctx を生やすだけ(mdl_infer_ctx/
    mdl_run本体は無改造)なので、常に有効化して同一バイナリで
    before/after(mdl_infer_ctx vs mdl_infer_packed_tf_ctx)を比較できる
    ようにする(デプロイキット設計メモ節)。

   (退化ゲート除去)を通したbundleは、pm_rec(P+M+GroupSum)のtailに
    concat(LutStackの縮約が作るアキュムレータ配線、export/rddlgn_optimize.py)
    を持ち込むことがあり、cbackend.py::_emit_packed_teacher_forcingは
    「tailはlut_dense連鎖のみ対応」という前提を崩すとValueErrorを送出する。
    併用は現状未対応(cbackend側の拡張が要る、実装スコープ
    外)なので、ここでは検出したら packed_teacher_forcing=None へ自動で
    フォールバックし、戻り値でその旨を呼び出し側(bench_one)へ伝える
    (PERF_C/VERIFY_TF_C/VERIFY_G32_C側は#ifdefで両ケースに対応済み)。
    """
    work.mkdir(parents=True, exist_ok=True)
    packed_available = True
    try:
        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  packed_teacher_forcing='dec_pm_rec')
    except ValueError as exc:
        if 'packed_teacher_forcing' not in str(exc):
            raise
        print(f'  [併用不可] packed_teacher_forcingを無効化して継続: '
              f'{exc}')
        packed_available = False
        files = generate_rddlgn_c(bundle, name='model', validate=False,
                                  packed_teacher_forcing=None)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    for rel in ('include/mdl_kernels.h', 'include/mdl_simd.h',
                'include/mdl_stdio.h',
                'src/mdl_kernels.c', 'include/mdl_threads.h',
                'src/mdl_threads.c', 'include/mdl_rddlgn.h',
                'src/mdl_rddlgn.c'):
        shutil.copy2(RUNTIME_C / rel, work / Path(rel).name)
    return packed_available


def sh(cmd: list[str], timeout: float | None = None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def compile_step(cc: str, work: Path, srcs: list[str], out: str, opt: int, *,
                  extra_flags: tuple[str, ...] = (), link: bool = False,
                  timeout: float | None = None):
    mode = [] if link else ['-c']
    cmd = [cc, '-std=c99', f'-O{opt}', f'-I{work}', *extra_flags, *mode,
           *[str(work / s) for s in srcs], '-o', str(work / out)]
    t0 = time.time()
    r = sh(cmd, timeout=timeout)
    return r, time.time() - t0


def decide_opt_level(cc: str, work: Path, opt_override: int | None,
                      timeout_s: float = 480.0) -> tuple[int, float]:
    """paper_scale1のmodel.cコンパイル時間から最適化レベルを決める。

    タスク注意点: 「コンパイル時間が10分を超えるようなら-O1で測り、その旨
    記録(最適化レベルは全モデルで統一)」。10分ちょうどでは危険側に振れる
    ので8分(480秒)を閾値にした。
    """
    if opt_override is not None:
        return opt_override, 0.0
    r, dt = compile_step(cc, work, ['model.c'], 'model.o', opt=2,
                          timeout=timeout_s)
    if r.returncode == 0 and dt <= timeout_s:
        print(f'  [opt決定] paper_scale1 -O2 コンパイル {dt:.1f}s → -O2を採用')
        return 2, dt
    print(f'  [opt決定] paper_scale1 -O2 が{timeout_s:.0f}s超/失敗 → -O1へ切替'
          f'(全モデル-O1で統一)')
    return 1, dt


# ---------------------------------------------------------------------------
# Cドライバ(検証・性能・レイヤ別プロファイル・スレッドプール・最小RSS)
# ---------------------------------------------------------------------------
VERIFY_TF_C = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>
#ifdef MDL_DEC_PM_REC_PACKED_WORK_BYTES
static uint32_t pwork[MDL_DEC_PM_REC_PACKED_WORK_BYTES / 4 + 4];
#endif
int main(void) {
  MDL_STDIO_BINARY();
  static uint8_t bits[MDL_INPUT_WIDTH];
  static int32_t scores[MDL_OUTPUT_WIDTH];
#ifdef MDL_DEC_PM_REC_PACKED_WORK_BYTES
  static int32_t scores_packed[MDL_OUTPUT_WIDTH];
#endif
  int32_t i;
  for (;;) {
    size_t got = fread(bits, 1, sizeof(bits), stdin);
    if (got == 0) break;
    if (got != sizeof(bits)) return 11;
    if (mdl_infer(bits, scores) != 0) return 2;
    /* mdl_infer_packed_tf_ctx(P headをsteps回、M+GroupSumをsteps本
       まとめて1回のpacked評価)がmdl_infer(既存経路)と一致することを
       ベンチ本体の検証フェーズでも固定する(専用テストは
       tests/deploy/test_c_rddlgn_packed_tf.py)。既定コンテキストは
       mdl_infer内部のstaticなものを使えないので別ctxを都度作る。
      (退化ゲート除去)併用時はこの節自体が生成されない
       (packed_teacher_forcing=Noneへ自動フォールバック、emit_c_files
       参照)ので#ifdefで丸ごと外す。 */
#ifdef MDL_DEC_PM_REC_PACKED_WORK_BYTES
    {
      static uint32_t work2[MDL_CTX_BYTES / 4];
      mdl_ctx ctx2;
      if (mdl_ctx_init(&ctx2, work2, sizeof(work2)) != 0) return 20;
      if (mdl_infer_packed_tf_ctx(&ctx2, bits, pwork, sizeof(pwork),
                                  scores_packed) != 0) return 21;
      if (memcmp(scores, scores_packed, sizeof(scores)) != 0) return 22;
    }
#endif
    for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
      unsigned char b[4];
      uint32_t v = (uint32_t)scores[i];
      b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
      b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
      fwrite(b, 1, 4, stdout);
    }
  }
  return 0;
}
'''

VERIFY_GEN_C = '''#include "model.h"
#include "mdl_rddlgn.h"
#include <stdio.h>
#include "mdl_stdio.h"
static uint32_t work[MDL_CTX_BYTES / 4];
int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  uint8_t src_bits[MDL_BUF_WIDTH_X_SRC];
  uint32_t out_tokens[MDL_STEPS_DEC_PM_REC];
  uint32_t bos, eos, max_new;
  int n, i;
  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 10;
  for (;;) {
    size_t got = fread(src_bits, 1, sizeof(src_bits), stdin);
    if (got == 0) break;
    if (got != sizeof(src_bits)) return 11;
    if (fread(&bos, sizeof(bos), 1, stdin) != 1) return 12;
    if (fread(&eos, sizeof(eos), 1, stdin) != 1) return 13;
    if (fread(&max_new, sizeof(max_new), 1, stdin) != 1) return 14;
    n = mdl_rddlgn_generate_greedy(&ctx, src_bits, bos, eos, max_new,
                                   out_tokens);
    printf("%d", n);
    if (n > 0) for (i = 0; i < n; i++) printf(" %u", out_tokens[i]);
    printf("\\n");
  }
  return 0;
}
'''

#:: greedy生成の系列バッチ化の検証。32本ぶんのクエリ(すべて
#: src_bits+bos、eos/max_newは1回ぶん共通)を受け取り、packed32の結果と
#: 32回逐次呼び出しの結果を突き合わせて不一致ならnonzeroで落ちる
#: (tests/deploy/test_c_rddlgn_greedy_packed32.pyと同じ手口)。
#:(退化ゲート除去)がpm_recのtailにflat_concatを持ち込むモデルでは
#: MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES自体が生成されない(節と
#: 同じ理由、emit_c_files参照)。#ifdefで実体/スタブを切り替える
#: (同じ入出力プロトコル(1件読んで"OK"を1行返す)を保つことで、Python側
#: (verify_greedy_packed32)は無改造で「スキップ」を「検証成功」として
#: 素通しできる — bench_one側でresult['verify']['greedy_packed32_skipped']
#: に実体があったかどうかを別途記録する)。
VERIFY_G32_C = '''#include "model.h"
#include "mdl_rddlgn.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>
#define LANES 32
static uint32_t work[MDL_CTX_BYTES / 4];
static uint8_t src_bits32[LANES][MDL_BUF_WIDTH_X_SRC];
static uint32_t bos_tokens[LANES];
static uint32_t out_tokens_seq[LANES][MDL_STEPS_DEC_PM_REC];
static int32_t out_lens_seq[LANES];
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
static uint32_t pwork[MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES / 4 + 4];
/* scratchが呼び出し側持ちへ一般化された(mdl_rddlgn_g32_scratch)。*/
static mdl_rddlgn_g32_scratch scratch;
static uint32_t out_tokens_packed[LANES][MDL_STEPS_DEC_PM_REC];
static int32_t out_lens_packed[LANES];
#endif
int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  int lane;
  uint32_t eos, max_new;
  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 10;
  for (;;) {
    size_t got = fread(src_bits32, 1, sizeof(src_bits32), stdin);
    if (got == 0) break;
    if (got != sizeof(src_bits32)) return 11;
    if (fread(bos_tokens, sizeof(uint32_t), LANES, stdin) != LANES) return 12;
    if (fread(&eos, sizeof(eos), 1, stdin) != 1) return 13;
    if (fread(&max_new, sizeof(max_new), 1, stdin) != 1) return 14;
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
    if (mdl_rddlgn_generate_greedy_packed32(
            &ctx, &scratch, &src_bits32[0][0], bos_tokens, eos, max_new,
            pwork, sizeof(pwork), &out_tokens_packed[0][0],
            out_lens_packed) != 0) return 2;
    for (lane = 0; lane < LANES; lane++) {
      int n = mdl_rddlgn_generate_greedy(&ctx, src_bits32[lane],
                                         bos_tokens[lane], eos, max_new,
                                         out_tokens_seq[lane]);
      if (n < 0) return 3;
      out_lens_seq[lane] = n;
    }
    for (lane = 0; lane < LANES; lane++) {
      if (out_lens_packed[lane] != out_lens_seq[lane]) return 5;
      if (memcmp(out_tokens_packed[lane], out_tokens_seq[lane],
                 sizeof(uint32_t) * (size_t)out_lens_seq[lane]) != 0) {
        return 6;
      }
    }
#else
    (void)src_bits32; (void)bos_tokens; (void)eos; (void)max_new;
    (void)out_tokens_seq; (void)out_lens_seq; (void)lane;
#endif
    printf("OK\\n");
  }
  return 0;
}
'''

#: 生成テキストサンプル用(char-LM専用)。1回の呼び出しでctx→32文字ぶんの
#: greedy生成をやり、複数回呼べば「直近32文字を再エンコードしてスライド」
#: できる(train_kd_charlm_rddlgn.py::generate_sampleと同じsliding window
#: 発想だが、1文字ごとの再エンコードではなく32文字ウィンドウをまとめて
#: 生成する — RDDLGNのP-layer状態は32ティック分自己回帰できるよう学習
#: されている、generate_greedy契約通り)。VERIFY_GEN_Cと同一。
SAMPLE_GEN_C = VERIFY_GEN_C

#: レイテンシ(encode/ttft/seq_e2e)+ レイヤ別プロファイル(N/K/L/PM/embed)+
#: 教師強制のスレッドプールスループット、を1バイナリで測る。
PERF_C = '''#include "model.h"
#include "mdl_rddlgn.h"
#include "mdl_threads.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CTX 8
#define N_ITEMS 16

static uint32_t work[MAX_CTX][MDL_CTX_BYTES / 4];
static mdl_ctx ctx[MAX_CTX];
static void *ctxp[MAX_CTX];
static uint8_t src_bits[MDL_BUF_WIDTH_X_SRC];
static uint32_t src_tokens[MDL_STEPS_ENC_N_REC];
static uint32_t out_tokens[MDL_STEPS_DEC_PM_REC];
static uint8_t tf_inputs[N_ITEMS * MDL_INPUT_WIDTH];
static int32_t tf_scores[N_ITEMS * MDL_OUTPUT_WIDTH];
/* greedy(通常)のmdl_run_poolラッパ(greedy_pool_fn)の出力先。
   N_ITEMS本ぶん、系列ごとにMDL_STEPS_DEC_PM_REC個のトークンを書く。 */
static uint32_t g_greedy_out[N_ITEMS][MDL_STEPS_DEC_PM_REC];

/* greedy生成の系列バッチ化(32本の独立系列)。*/
#define G32_LANES 32
static uint8_t g32_src_bits[G32_LANES][MDL_BUF_WIDTH_X_SRC];
static uint32_t g32_bos[G32_LANES];
static uint32_t g32_out_tokens[G32_LANES][MDL_STEPS_DEC_PM_REC];
static int32_t g32_out_lens[G32_LANES];
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
static uint32_t g32_pwork[MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES / 4 + 4];
/* scratchが呼び出し側持ちへ一般化された(1ctx計測ではこの1個を使い
   回す)。 */
static mdl_rddlgn_g32_scratch g32_scratch;
#endif

/* マルチコアスケーリング計測。mdl_run_poolへ、教師強制の
   packed tail版とgreedy32版をそれぞれ「1プールスロット =
   自前のctx/work一式」で乗せる。通常経路(mdl_infer_ctx_v/1系列ずつの
   mdl_rddlgn_generate_greedy)は既にmdl_run_poolへそのまま乗るので
   専用のラッパは要らない(tf_thr/後述のgreedy_pool_thrがそれ)。
   packed_tf/greedy32はいずれもcbackend.py::_emit_packed_teacher_forcing
   が1回で両方まとめて出す(節参照)ため、MDL_DEC_PM_REC_PACKED_WORK_BYTES
   とMDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTESは常に同時に有無が揃う。 */

#ifdef MDL_DEC_PM_REC_PACKED_WORK_BYTES
/* --- packed teacher forcingのプール版: ctx+pwork一式を1スロットに
   まとめ、mdl_infer_fn型のラッパを介してmdl_run_poolへ乗せる。 --- */
typedef struct {
  uint32_t ctxwork[MDL_CTX_BYTES / 4];
  mdl_ctx ctx;
  uint32_t pwork[MDL_DEC_PM_REC_PACKED_WORK_BYTES / 4 + 4];
} ptf_slot_t;
static ptf_slot_t ptf_slots[MAX_CTX];
static void *ptf_slotsp[MAX_CTX];

static int packed_tf_pool_fn(void *vctx, const uint8_t *input,
                             int32_t *scores) {
  ptf_slot_t *s = (ptf_slot_t *)vctx;
  return mdl_infer_packed_tf_ctx(&s->ctx, input, s->pwork, sizeof(s->pwork),
                                 scores);
}
#endif

/* --- 既存のmdl_rddlgn_generate_greedy(1系列ずつ)をmdl_infer_fn型に
   合わせるラッパ。ctx自体は既存のctx[MAX_CTX]をそのまま使う(mdl_run_pool
   の「N枚をNコアへ」がそのまま「N系列をNコアへ」になる)。
   flat_concat有無に依存しないので無条件でコンパイルする。 --- */
static int greedy_pool_fn(void *vctx, const uint8_t *input, int32_t *scores) {
  int n = mdl_rddlgn_generate_greedy((mdl_ctx *)vctx, input, 0, 0xFFFFFFFEu,
                                     (uint32_t)MDL_STEPS_DEC_PM_REC,
                                     (uint32_t *)scores);
  return n == MDL_STEPS_DEC_PM_REC ? 0 : -1;
}

#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
/* --- greedy32(32本まとめ)のプール版: ctx+scratch+pwork+出力を
   1スロットにまとめる(paper_scale1規模でscratch単体が約7.5MB、
   MAX_CTX=8で最大60MB前後になるがBSSなので実行ファイルサイズには
   乗らない)。1「item」= 32本まとめたバッチ1回ぶん。 --- */
typedef struct {
  uint32_t ctxwork[MDL_CTX_BYTES / 4];
  mdl_ctx ctx;
  mdl_rddlgn_g32_scratch scratch;
  uint32_t pwork[MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES / 4 + 4];
  uint32_t out_tokens[G32_LANES][MDL_STEPS_DEC_PM_REC];
  int32_t out_lens[G32_LANES];
} g32_slot_t;
static g32_slot_t g32_slots[MAX_CTX];
static void *g32_slotsp[MAX_CTX];
#define G32_POOL_BATCHES 32

static int g32_pool_fn(void *vctx, const uint8_t *input, int32_t *scores) {
  g32_slot_t *s = (g32_slot_t *)vctx;
  (void)input;
  (void)scores;
  return mdl_rddlgn_generate_greedy_packed32(
      &s->ctx, &s->scratch, &g32_src_bits[0][0], g32_bos, 0xFFFFFFFEu,
      (uint32_t)MDL_STEPS_DEC_PM_REC, s->pwork, sizeof(s->pwork),
      &s->out_tokens[0][0], s->out_lens);
}
#endif

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}

static double median_of(double *t, int reps) {
  qsort(t, (size_t)reps, sizeof(double), cmp_d);
  return t[reps / 2];
}

int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 50;
  const int max_thr = argc > 2 ? atoi(argv[2]) : 8;
  const uint32_t BOS = 0, EOS_INVALID = 0xFFFFFFFEu;
  const int full_steps = MDL_STEPS_DEC_PM_REC;
  double *t = (double *)malloc((size_t)reps * sizeof(double));
  int i, r, n;
  size_t k;

  for (k = 0; k < sizeof(src_bits); k++)
    src_bits[k] = (uint8_t)((k * 2654435761u) & 1);
  for (i = 0; i < MDL_STEPS_ENC_N_REC; i++)
    src_tokens[i] = (uint32_t)((i * 2654435761u) % MDL_VOCAB_SIZE);
  for (k = 0; k < sizeof(tf_inputs); k++)
    tf_inputs[k] = (uint8_t)((k * 2654435761u) & 1);
  for (i = 0; i < MAX_CTX; i++) {
    if (mdl_ctx_init(&ctx[i], work[i], sizeof(work[i])) != 0) return 3;
    ctxp[i] = &ctx[i];
  }
  /* マルチコアスケーリング計測用のプールスロット初期化。*/
#ifdef MDL_DEC_PM_REC_PACKED_WORK_BYTES
  for (i = 0; i < MAX_CTX; i++) {
    if (mdl_ctx_init(&ptf_slots[i].ctx, ptf_slots[i].ctxwork,
                     sizeof(ptf_slots[i].ctxwork)) != 0) return 4;
    ptf_slotsp[i] = &ptf_slots[i];
  }
#endif
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
  for (i = 0; i < MAX_CTX; i++) {
    if (mdl_ctx_init(&g32_slots[i].ctx, g32_slots[i].ctxwork,
                     sizeof(g32_slots[i].ctxwork)) != 0) return 5;
    g32_slotsp[i] = &g32_slots[i];
  }
#endif

  /* --- encode_ms: max_new_tokens=0(エンコーダのみ+state reset) --- */
  mdl_rddlgn_generate_greedy(&ctx[0], src_bits, BOS, EOS_INVALID, 0, out_tokens);
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_rddlgn_generate_greedy(&ctx[0], src_bits, BOS, EOS_INVALID, 0, out_tokens);
    t[r] = now_ms() - t0;
  }
  printf("encode_ms %.6f %.6f\\n", median_of(t, reps), t[0]);

  /* --- ttft_ms: encode + 1トークン --- */
  mdl_rddlgn_generate_greedy(&ctx[0], src_bits, BOS, EOS_INVALID, 1, out_tokens);
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_rddlgn_generate_greedy(&ctx[0], src_bits, BOS, EOS_INVALID, 1, out_tokens);
    t[r] = now_ms() - t0;
  }
  printf("ttft_ms %.6f %.6f\\n", median_of(t, reps), t[0]);

  /* --- seq_e2e_ms: encode + フル長生成(EOSは絶対一致しない値) --- */
  n = mdl_rddlgn_generate_greedy(&ctx[0], src_bits, BOS, EOS_INVALID,
                                 (uint32_t)full_steps, out_tokens);
  if (n != full_steps) return 20;   /* 早期終了していないことの自己チェック */
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_rddlgn_generate_greedy(&ctx[0], src_bits, BOS, EOS_INVALID,
                               (uint32_t)full_steps, out_tokens);
    t[r] = now_ms() - t0;
  }
  printf("seq_e2e_ms %.6f %.6f\\n", median_of(t, reps), t[0]);
  printf("gen_tokens %d\\n", full_steps);

  /* --- 語彙折り畳み版: mdl_rddlgn_generate_greedy_folded。
     折り畳みが効いていないモデルでもフォールバック経路として常に動く
     ので、無条件に測る(before/afterを同一バイナリ・同一条件で
     比較できる)。 --- */
  mdl_rddlgn_generate_greedy_folded(&ctx[0], src_tokens, BOS, EOS_INVALID, 0,
                                    out_tokens);
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_rddlgn_generate_greedy_folded(&ctx[0], src_tokens, BOS, EOS_INVALID, 0,
                                      out_tokens);
    t[r] = now_ms() - t0;
  }
  printf("encode_ms_folded %.6f %.6f\\n", median_of(t, reps), t[0]);

  mdl_rddlgn_generate_greedy_folded(&ctx[0], src_tokens, BOS, EOS_INVALID, 1,
                                    out_tokens);
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_rddlgn_generate_greedy_folded(&ctx[0], src_tokens, BOS, EOS_INVALID, 1,
                                      out_tokens);
    t[r] = now_ms() - t0;
  }
  printf("ttft_ms_folded %.6f %.6f\\n", median_of(t, reps), t[0]);

  n = mdl_rddlgn_generate_greedy_folded(&ctx[0], src_tokens, BOS, EOS_INVALID,
                                        (uint32_t)full_steps, out_tokens);
  if (n != full_steps) return 21;
  for (r = 0; r < reps; r++) {
    const double t0 = now_ms();
    mdl_rddlgn_generate_greedy_folded(&ctx[0], src_tokens, BOS, EOS_INVALID,
                                      (uint32_t)full_steps, out_tokens);
    t[r] = now_ms() - t0;
  }
  printf("seq_e2e_ms_folded %.6f %.6f\\n", median_of(t, reps), t[0]);
#ifdef MDL_HAS_FOLDED_N_TABLE
  printf("has_folded_n 1\\n");
#else
  printf("has_folded_n 0\\n");
#endif
#ifdef MDL_HAS_FOLDED_L_TABLE
  printf("has_folded_l 1\\n");
#else
  printf("has_folded_l 0\\n");
#endif

  /* --- レイヤ別プロファイル(N/K/L/PM/embed、単一tickの費用) --- */
  {
    uint8_t *x_src, *ctx_out, *l_ext, *pm_ctx_ext, *pm_l_ext, *pm_state;
    uint8_t context_local[MDL_OUT_WIDTH_ENC_K_REC];
    int32_t pm_scores[MDL_OUT_WIDTH_DEC_PM_REC];
    uint8_t embed_buf[MDL_D_EMB];

    x_src = mdl_out_x_src(&ctx[0]);
    memcpy(x_src, src_bits, MDL_BUF_WIDTH_X_SRC);

    for (r = 0; r < reps; r++) {
      const double t0 = now_ms();
      mdl_run_recurrent_enc_n_rec(&ctx[0]);
      t[r] = now_ms() - t0;
    }
    printf("prof_n_ms %.6f\\n", median_of(t, reps));

    for (r = 0; r < reps; r++) {
      const double t0 = now_ms();
      mdl_run_recurrent_enc_k_rec(&ctx[0]);
      t[r] = now_ms() - t0;
    }
    printf("prof_k_ms %.6f\\n", median_of(t, reps));

    ctx_out = mdl_out_enc_k_rec(&ctx[0]);
    memcpy(context_local, ctx_out, sizeof(context_local));
    mdl_reset_state_dec_pm_rec(&ctx[0]);
    pm_state = mdl_state_dec_pm_rec_sv_p(&ctx[0]);
    l_ext = mdl_ext_dec_l_rec_sv_step(&ctx[0]);
    pm_ctx_ext = mdl_ext_dec_pm_rec_sv_context(&ctx[0]);
    pm_l_ext = mdl_ext_dec_pm_rec_sv_l(&ctx[0]);
    memcpy(pm_ctx_ext, context_local, sizeof(context_local));

    mdl_embed_lookup(0u, l_ext);
    for (r = 0; r < reps; r++) {
      const double t0 = now_ms();
      mdl_embed_lookup((uint32_t)(r % MDL_VOCAB_SIZE), embed_buf);
      t[r] = now_ms() - t0;
    }
    printf("prof_embed_ms %.6f\\n", median_of(t, reps));

    mdl_embed_lookup(0u, l_ext);
    for (r = 0; r < reps; r++) {
      const double t0 = now_ms();
      mdl_step_dec_l_rec(&ctx[0], 0);
      t[r] = now_ms() - t0;
    }
    printf("prof_l_ms %.6f\\n", median_of(t, reps));

    memcpy(pm_l_ext, mdl_out_dec_l_rec(&ctx[0]), MDL_EXT_WIDTH_DEC_PM_REC_SV_L);
    for (r = 0; r < reps; r++) {
      const double t0 = now_ms();
      pm_state = mdl_state_dec_pm_rec_sv_p(&ctx[0]);
      (void)pm_state;
      mdl_step_dec_pm_rec(&ctx[0], 0, pm_scores);
      t[r] = now_ms() - t0;
    }
    printf("prof_pm_ms %.6f\\n", median_of(t, reps));
  }

  /* --- 教師強制(mdl_infer_ctx)をmdl_run_poolでN並列 --- */
  for (n = 1; n <= max_thr && n <= MAX_CTX; n *= 2) {
    double best = 1e18;
    for (r = 0; r < 7; r++) {
      const double t0 = now_ms();
      if (mdl_run_pool(mdl_infer_ctx_v, ctxp, n, tf_inputs,
                       (size_t)MDL_INPUT_WIDTH, tf_scores,
                       sizeof(int32_t) * MDL_OUTPUT_WIDTH, N_ITEMS) != 0)
        return 2;
      { const double dt = now_ms() - t0; if (dt < best) best = dt; }
    }
    printf("tf_thr %d %.2f\\n", n, N_ITEMS * 1000.0 / best);
  }

  /* --- greedy(通常、1系列ずつ)をmdl_run_poolでN並列。
     tf_thrと同じ枚数(N_ITEMS)を同じ枚数分割戦略で流し、集計tokens/sを
     報告する(既存のmdl_rddlgn_generate_greedyをctxごとに割り当てる
     だけなので、コアの変更は不要——ラッパ関数greedy_pool_fnのみ追加)。
     flat_concat有無に依存しないので無条件で計測する。--- */
  for (n = 1; n <= max_thr && n <= MAX_CTX; n *= 2) {
    double best = 1e18;
    for (r = 0; r < 5; r++) {
      const double t0 = now_ms();
      if (mdl_run_pool(greedy_pool_fn, ctxp, n, src_bits, 0,
                       (int32_t *)g_greedy_out,
                       MDL_STEPS_DEC_PM_REC * sizeof(uint32_t),
                       N_ITEMS) != 0)
        return 40;
      { const double dt = now_ms() - t0; if (dt < best) best = dt; }
    }
    printf("greedy_pool_thr %d %.2f\\n", n,
          (double)N_ITEMS * (double)MDL_STEPS_DEC_PM_REC * 1000.0 / best);
  }

  /* --- 教師強制の時間バッチ化(packed decode tail)、1ctx ---
   *(退化ゲート除去)がpm_recのtailにflat_concatを持ち込むモデルでは
   * MDL_DEC_PM_REC_PACKED_WORK_BYTES自体が生成されない(cbackend.py
   * _emit_packed_teacher_forcingがlut_dense連鎖のみ対応のため、
   * generate_rddlgn_cの時点でpacked_teacher_forcing=Noneへ自動フォール
   * バックする、bench_rddlgn.pyのemit_c_files参照)。その場合は-1で
   * 「対象外」を明示し、コンパイル自体は通す(has_folded_n/lと
   * 同じ#ifdef規約)。packed_tf_pool_thrも同じ条件でしか意味を
   * 持たないので、同じ#ifdefの中に含める。 */
#ifdef MDL_DEC_PM_REC_PACKED_WORK_BYTES
  /* --- packed teacher forcing(mdl_infer_packed_tf_ctx)を
     mdl_run_poolでN並列。tf_thrと同じ枚数・同じ入力バッファを使い、
     「packed tf(1ctx内2.09〜2.33倍)× マルチコア」の乗算効果を見る。 --- */
  for (n = 1; n <= max_thr && n <= MAX_CTX; n *= 2) {
    double best = 1e18;
    for (r = 0; r < 7; r++) {
      const double t0 = now_ms();
      if (mdl_run_pool(packed_tf_pool_fn, ptf_slotsp, n, tf_inputs,
                       (size_t)MDL_INPUT_WIDTH, tf_scores,
                       sizeof(int32_t) * MDL_OUTPUT_WIDTH, N_ITEMS) != 0)
        return 41;
      { const double dt = now_ms() - t0; if (dt < best) best = dt; }
    }
    printf("packed_tf_pool_thr %d %.2f\\n", n, N_ITEMS * 1000.0 / best);
  }

  {
    static uint32_t pwork[MDL_DEC_PM_REC_PACKED_WORK_BYTES / 4 + 4];
    static int32_t pscores[MDL_OUTPUT_WIDTH];
    mdl_infer_packed_tf_ctx(&ctx[0], tf_inputs, pwork, sizeof(pwork), pscores);
    for (r = 0; r < reps; r++) {
      const double t0 = now_ms();
      mdl_infer_packed_tf_ctx(&ctx[0], tf_inputs, pwork, sizeof(pwork),
                              pscores);
      t[r] = now_ms() - t0;
    }
    printf("packed_tf_ms %.6f\\n", median_of(t, reps));
    printf("packed_tf_work_bytes %d\\n",
          (int)MDL_DEC_PM_REC_PACKED_WORK_BYTES);
  }
#else
  printf("packed_tf_ms -1.0\\n");
  printf("packed_tf_work_bytes 0\\n");
#endif

  /* --- greedy生成の系列バッチ化(32本の独立系列)、集計tokens/s ---
   * 32回の逐次mdl_rddlgn_generate_greedy(=現状の「1本ずつ回す」)と、
   * 1回のmdl_rddlgn_generate_greedy_packed32(=32本まとめてtailをpacked
   * 評価)を比べる。大きいモデルでは逐次32回が高くつくので、この節だけ
   * repsを絞る(中央値を保ちつつ計測時間を抑える)。節と同じ理由で
   *併用時はMDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTESが無いことがある。*/
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
  {
    const int g32_reps = reps < 10 ? reps : 10;
    const uint32_t full_steps32 = (uint32_t)MDL_STEPS_DEC_PM_REC;
    int lane;
    for (lane = 0; lane < G32_LANES; lane++) {
      for (k = 0; k < sizeof(g32_src_bits[lane]); k++)
        g32_src_bits[lane][k] = (uint8_t)(((k + (size_t)lane) * 2654435761u) & 1);
      g32_bos[lane] = 0;
    }

    /* 逐次32回(既存のmdl_rddlgn_generate_greedyをそのまま32回)。 */
    mdl_rddlgn_generate_greedy(&ctx[0], g32_src_bits[0], 0, EOS_INVALID,
                              full_steps32, out_tokens);
    for (r = 0; r < g32_reps; r++) {
      const double t0 = now_ms();
      for (lane = 0; lane < G32_LANES; lane++) {
        mdl_rddlgn_generate_greedy(&ctx[0], g32_src_bits[lane], g32_bos[lane],
                                  EOS_INVALID, full_steps32, out_tokens);
      }
      t[r] = now_ms() - t0;
    }
    printf("g32_seq_ms %.6f\\n", median_of(t, g32_reps));

    /* packed32(1回で32本まとめて生成)。 */
    mdl_rddlgn_generate_greedy_packed32(
        &ctx[0], &g32_scratch, &g32_src_bits[0][0], g32_bos, EOS_INVALID,
        full_steps32, g32_pwork, sizeof(g32_pwork), &g32_out_tokens[0][0],
        g32_out_lens);
    for (r = 0; r < g32_reps; r++) {
      const double t0 = now_ms();
      mdl_rddlgn_generate_greedy_packed32(
          &ctx[0], &g32_scratch, &g32_src_bits[0][0], g32_bos, EOS_INVALID,
          full_steps32, g32_pwork, sizeof(g32_pwork), &g32_out_tokens[0][0],
          g32_out_lens);
      t[r] = now_ms() - t0;
    }
    printf("g32_packed_ms %.6f\\n", median_of(t, g32_reps));
    printf("g32_lanes %d\\n", G32_LANES);
  }
#else
  printf("g32_seq_ms -1.0\\n");
  printf("g32_packed_ms -1.0\\n");
  printf("g32_lanes %d\\n", G32_LANES);
#endif

  /* --- greedy32のscratchを呼び出し側持ちへ一般化した本体
     (mdl_rddlgn.c)を、mdl_run_poolでN並列。1「item」=32本まとめた
     バッチ1回ぶん(G32_POOL_BATCHES個)。集計tokens/sは
     G32_POOL_BATCHES*G32_LANES*full_stepsトークンを要した時間で割る。
     これが本タスク最大の見込み(2.09〜3.35倍のpacked経路 ×
     コア数効率)の実測。併用時はMDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
     が無いことがある(上のg32_seq_ms/g32_packed_msと同じ理由)ので、同じ
     #ifdefの中に含める。 */
#ifdef MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES
  for (n = 1; n <= max_thr && n <= MAX_CTX; n *= 2) {
    double best = 1e18;
    for (r = 0; r < 3; r++) {
      const double t0 = now_ms();
      /* g32_pool_fnはinput/scoresを使わない(組み立ては全部自分の
         スロット構造体で完結する)。in_stride/out_stride=0でNULL+0の
         ポインタ演算を避けるため、既存のtf_inputs/tf_scoresをダミーの
         非NULLポインタとして流用するだけ(中身は読み書きされない)。 */
      if (mdl_run_pool(g32_pool_fn, g32_slotsp, n, tf_inputs, 0, tf_scores, 0,
                       G32_POOL_BATCHES) != 0)
        return 42;
      { const double dt = now_ms() - t0; if (dt < best) best = dt; }
    }
    printf("g32_pool_thr %d %.2f\\n", n,
          (double)G32_POOL_BATCHES * (double)G32_LANES
              * (double)MDL_STEPS_DEC_PM_REC * 1000.0 / best);
  }
#endif

  printf("ctx_bytes %d\\n", MDL_CTX_BYTES);
  free(t);
  return 0;
}
'''

MINIMAL_C = '''#include "model.h"
#include "mdl_rddlgn.h"
static uint32_t work[MDL_CTX_BYTES / 4];
static uint8_t src_bits[MDL_BUF_WIDTH_X_SRC];
static uint32_t out_tokens[MDL_STEPS_DEC_PM_REC];
int main(void) {
  mdl_ctx ctx; size_t k; int r; int n = 0;
  for (k = 0; k < sizeof(src_bits); k++)
    src_bits[k] = (uint8_t)((k * 2654435761u) & 1);
  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 3;
  for (r = 0; r < 5; r++)
    n = mdl_rddlgn_generate_greedy(&ctx, src_bits, 0, 0xFFFFFFFEu,
                                   (uint32_t)MDL_STEPS_DEC_PM_REC, out_tokens);
  return n & 1;
}
'''


def section_sizes(obj: Path) -> dict:
    out = sh(['size', '-m', str(obj)]).stdout
    got = {}
    for name, key in (('__TEXT, __const', 'const'), ('__DATA, __bss', 'bss'),
                      ('__TEXT, __text', 'text')):
        m = re.search(re.escape(name) + r'\): (\d+)', out)
        if m:
            got[key] = int(m.group(1))
    return got


def peak_rss_kb(work: Path) -> int:
    r = sh(['/usr/bin/time', '-l', str(work / 'minimal')])
    m = re.search(r'(\d+)\s+maximum resident set size', r.stderr)
    return int(m.group(1)) // 1024 if m else 0


def kb(n: int) -> str:
    return f'{n / 1024:,.1f}KB'


# ---------------------------------------------------------------------------
# ビルド一式(model.cは1回だけコンパイル、他は小さいファイルなので都度)
# ---------------------------------------------------------------------------
def build_all(cc: str, work: Path, opt: int) -> dict:
    (work / 'verify_tf.c').write_text(VERIFY_TF_C, encoding='utf-8')
    (work / 'verify_gen.c').write_text(VERIFY_GEN_C, encoding='utf-8')
    (work / 'verify_g32.c').write_text(VERIFY_G32_C, encoding='utf-8')
    (work / 'perf.c').write_text(PERF_C, encoding='utf-8')
    (work / 'minimal.c').write_text(MINIMAL_C, encoding='utf-8')

    timings = {}
    r, dt = compile_step(cc, work, ['model.c'], 'model.o', opt=opt)
    if r.returncode != 0:
        raise RuntimeError(f'model.cのビルドに失敗:\n{r.stderr[:2000]}')
    timings['model_o'] = dt

    for src, out in (('mdl_kernels.c', 'mdl_kernels.o'),
                     ('mdl_rddlgn.c', 'mdl_rddlgn.o'),
                     ('mdl_threads.c', 'mdl_threads.o'),
                     ('verify_tf.c', 'verify_tf.o'),
                     ('verify_gen.c', 'verify_gen.o'),
                     ('verify_g32.c', 'verify_g32.o'),
                     ('perf.c', 'perf.o'),
                     ('minimal.c', 'minimal.o')):
        extra = ('-pthread',) if src == 'mdl_threads.c' else ()
        r, dt = compile_step(cc, work, [src], out, opt=opt, extra_flags=extra)
        if r.returncode != 0:
            raise RuntimeError(f'{src}のビルドに失敗:\n{r.stderr[:2000]}')
        timings[out] = dt

    r, _ = compile_step(cc, work, ['model.o', 'mdl_kernels.o', 'verify_tf.o'],
                        'verify_tf', opt=opt, link=True)
    if r.returncode != 0:
        raise RuntimeError(f'verify_tfのリンクに失敗:\n{r.stderr[:2000]}')
    r, _ = compile_step(
        cc, work, ['model.o', 'mdl_kernels.o', 'mdl_rddlgn.o', 'verify_gen.o'],
        'verify_gen', opt=opt, link=True)
    if r.returncode != 0:
        raise RuntimeError(f'verify_genのリンクに失敗:\n{r.stderr[:2000]}')
    r, _ = compile_step(
        cc, work, ['model.o', 'mdl_kernels.o', 'mdl_rddlgn.o', 'verify_g32.o'],
        'verify_g32', opt=opt, link=True)
    if r.returncode != 0:
        raise RuntimeError(f'verify_g32のリンクに失敗:\n{r.stderr[:2000]}')
    r, _ = compile_step(
        cc, work,
        ['model.o', 'mdl_kernels.o', 'mdl_rddlgn.o', 'mdl_threads.o', 'perf.o'],
        'perf', opt=opt, link=True, extra_flags=('-pthread',))
    if r.returncode != 0:
        raise RuntimeError(f'perfのリンクに失敗:\n{r.stderr[:2000]}')
    r, _ = compile_step(
        cc, work, ['model.o', 'mdl_kernels.o', 'mdl_rddlgn.o', 'minimal.o'],
        'minimal', opt=opt, link=True)
    if r.returncode != 0:
        raise RuntimeError(f'minimalのリンクに失敗:\n{r.stderr[:2000]}')
    return timings


# ---------------------------------------------------------------------------
# 検証(simulate ⇔ C、教師強制と自己回帰生成の両方)
# ---------------------------------------------------------------------------
def verify_teacher_forcing(work: Path, graph, n_samples: int = 5,
                           seed: int = 0) -> dict:
    in_width = int(graph.inputs[0].width)
    n_out = int(graph.outputs[0].width)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, in_width), dtype=np.uint8)
    want = simulate(graph, bits).astype(np.int64)

    proc = subprocess.run([str(work / 'verify_tf')], input=bits.tobytes(),
                          capture_output=True)
    if proc.returncode == 22:
        raise RuntimeError(
            'verify_tf: mdl_infer_packed_tf_ctx(教師強制時間バッチ化)'
            'がmdl_infer(既存経路)と不一致')
    if proc.returncode != 0:
        raise RuntimeError(f'verify_tf実行が失敗(rc={proc.returncode}): '
                          f'{proc.stderr.decode(errors="replace")}')
    got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
    got = got.reshape(n_samples, n_out)
    ok = bool(np.array_equal(got, want))
    return {'ok': ok, 'n_samples': n_samples, 'n_out': n_out,
            'max_abs_diff': int(np.abs(got - want).max()) if not ok else 0,
            'packed_tf_matches': True}


def verify_greedy(work: Path, model: RDDLGN, cfg: RDDLGNConfig, bos: int,
                  eos: int, n_samples: int = 6, seed: int = 11) -> dict:
    import struct as _struct

    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()

    rng = np.random.default_rng(seed)
    queries = []
    expected = []
    for _ in range(n_samples):
        src_tokens = rng.integers(0, cfg.vocab_size, size=(cfg.seq_len,))
        src_bits = embed[src_tokens].reshape(-1).astype(np.uint8)
        sim = simulate_rddlgn_greedy_bits(
            model, src_bits.astype(np.int64), bos_token=bos, eos_token=eos,
            max_new_tokens=cfg.seq_len)
        queries.append(src_bits.tobytes()
                       + _struct.pack('<III', int(bos), int(eos), cfg.seq_len))
        expected.append(sim)

    proc = subprocess.run([str(work / 'verify_gen')], input=b''.join(queries),
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f'verify_gen実行が失敗: '
                          f'{proc.stderr.decode(errors="replace")}')
    lines = proc.stdout.decode().strip('\n').split('\n')
    got = []
    for line in lines:
        parts = [int(v) for v in line.split()]
        n = parts[0]
        got.append(n if n < 0 else parts[1:1 + n])
    ok = got == expected
    return {'ok': ok, 'n_samples': n_samples,
            'got_sample0': got[0] if got else None,
            'want_sample0': expected[0] if expected else None}


def verify_greedy_packed32(work: Path, cfg: RDDLGNConfig, bos: int, eos: int,
                           seed: int = 13) -> dict:
    """mdl_rddlgn_generate_greedy_packed32(32本まとめて生成)が
    mdl_rddlgn_generate_greedy(既存、逐次)を32回呼んだものとトークン列
    レベルで完全一致することを、乱数の32本ぶんのsrc_bits/bosで確認する
    (Cの`VERIFY_G32_C`が内部でこの突き合わせをして0/非0を返す)。"""
    rng = np.random.default_rng(seed)
    in_width = int(cfg.seq_len * cfg.d_emb)
    src_bits32 = rng.integers(0, 2, size=(32, in_width), dtype=np.uint8)
    bos_tokens = np.full(32, int(bos), dtype='<u4')
    import struct as _struct
    payload = (src_bits32.tobytes() + bos_tokens.tobytes()
              + _struct.pack('<II', int(eos), cfg.seq_len))
    proc = subprocess.run([str(work / 'verify_g32')], input=payload,
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f'verify_g32実行が失敗(rc={proc.returncode}、'
            '5/6はpacked32が逐次と不一致): '
            f'{proc.stderr.decode(errors="replace")}')
    ok = proc.stdout.decode().strip() == 'OK'
    return {'ok': ok}


# ---------------------------------------------------------------------------
# 性能計測
# ---------------------------------------------------------------------------
def run_perf(work: Path, reps: int, max_thr: int = 8) -> dict:
    r = sh([str(work / 'perf'), str(reps), str(max_thr)])
    if r.returncode != 0:
        raise RuntimeError(f'perf実行が失敗(rc={r.returncode}): {r.stderr[:800]}')
    out: dict = {'tf_thr': {}, 'greedy_pool_thr': {}, 'packed_tf_pool_thr': {},
                'g32_pool_thr': {}}
    for line in r.stdout.splitlines():
        f = line.split()
        if f[0] in ('encode_ms', 'ttft_ms', 'seq_e2e_ms',
                    'encode_ms_folded', 'ttft_ms_folded', 'seq_e2e_ms_folded'):
            out[f[0]] = (float(f[1]), float(f[2]))
        elif f[0].startswith('prof_'):
            out[f[0]] = float(f[1])
        elif f[0] == 'tf_thr':
            out['tf_thr'][int(f[1])] = float(f[2])
        elif f[0] == 'greedy_pool_thr':
            out['greedy_pool_thr'][int(f[1])] = float(f[2])
        elif f[0] == 'packed_tf_pool_thr':
            out['packed_tf_pool_thr'][int(f[1])] = float(f[2])
        elif f[0] == 'g32_pool_thr':
            out['g32_pool_thr'][int(f[1])] = float(f[2])
        elif f[0] == 'gen_tokens':
            out['gen_tokens'] = int(f[1])
        elif f[0] == 'ctx_bytes':
            out['ctx_bytes'] = int(f[1])
        elif f[0] == 'packed_tf_ms':
            out['packed_tf_ms'] = float(f[1])
        elif f[0] == 'packed_tf_work_bytes':
            out['packed_tf_work_bytes'] = int(f[1])
        elif f[0] == 'g32_seq_ms':
            out['g32_seq_ms'] = float(f[1])
        elif f[0] == 'g32_packed_ms':
            out['g32_packed_ms'] = float(f[1])
        elif f[0] == 'g32_lanes':
            out['g32_lanes'] = int(f[1])
        elif f[0] in ('has_folded_n', 'has_folded_l'):
            out[f[0]] = bool(int(f[1]))
    return out


def run_perf_twice(work: Path, reps: int, max_thr: int = 8,
                    pause_s: float = 1.5) -> dict:
    """同一条件2回で再現性を確認する(タスク注意点。サーマル対策の休止つき)。"""
    a = run_perf(work, reps, max_thr)
    time.sleep(pause_s)
    b = run_perf(work, reps, max_thr)

    def combine_pair(key):
        va, vb = a[key][0], b[key][0]
        sd = statistics.pstdev([va, vb])
        return {'run1': va, 'run2': vb, 'median': statistics.median([va, vb]),
                'sd': sd, 'rel_sd_pct': 100.0 * sd / max(va, vb, 1e-9)}

    combined = {'encode_ms': combine_pair('encode_ms'),
                'ttft_ms': combine_pair('ttft_ms'),
                'seq_e2e_ms': combine_pair('seq_e2e_ms'),
                'encode_ms_folded': combine_pair('encode_ms_folded'),
                'ttft_ms_folded': combine_pair('ttft_ms_folded'),
                'seq_e2e_ms_folded': combine_pair('seq_e2e_ms_folded'),
                'gen_tokens': a.get('gen_tokens'),
                'ctx_bytes': a.get('ctx_bytes'),
                'has_folded_n': a.get('has_folded_n'),
                'has_folded_l': a.get('has_folded_l')}
    for key in ('prof_n_ms', 'prof_k_ms', 'prof_l_ms', 'prof_pm_ms',
               'prof_embed_ms', 'packed_tf_ms', 'g32_seq_ms', 'g32_packed_ms'):
        va, vb = a[key], b[key]
        sd = statistics.pstdev([va, vb])
        combined[key] = {'run1': va, 'run2': vb,
                         'median': statistics.median([va, vb]),
                         'rel_sd_pct': 100.0 * sd / max(va, vb, 1e-9)}
    combined['packed_tf_work_bytes'] = a.get('packed_tf_work_bytes')
    combined['g32_lanes'] = a.get('g32_lanes')
    combined['tf_thr'] = {
        n: statistics.median([a['tf_thr'][n], b['tf_thr'][n]])
        for n in a['tf_thr']}
    # プールスケーリング(N=1/2/4/8)。tf_thrと同じmedian-of-2規律。
    for pool_key in ('greedy_pool_thr', 'packed_tf_pool_thr', 'g32_pool_thr'):
        combined[pool_key] = {
            n: statistics.median([a[pool_key][n], b[pool_key][n]])
            for n in a.get(pool_key, {})}
    decode_step_ms = ((combined['seq_e2e_ms']['median']
                       - combined['encode_ms']['median'])
                      / max(1, combined['gen_tokens']))
    combined['decode_step_ms'] = decode_step_ms
    combined['tokens_per_s'] = 1000.0 / decode_step_ms if decode_step_ms > 0 else 0.0
    decode_step_ms_folded = ((combined['seq_e2e_ms_folded']['median']
                              - combined['encode_ms_folded']['median'])
                             / max(1, combined['gen_tokens']))
    combined['decode_step_ms_folded'] = decode_step_ms_folded
    combined['tokens_per_s_folded'] = (
        1000.0 / decode_step_ms_folded if decode_step_ms_folded > 0 else 0.0)
    combined['tf_seqs_per_s'] = combined['tf_thr']
    tgt_len = combined['gen_tokens']
    combined['tokens_per_s_agg'] = {
        n: v * tgt_len for n, v in combined['tf_thr'].items()}
    # packed teacher forcing(1ctx、mdl_infer_packed_tf_ctx)は
    # 「1回の呼び出しで教師強制1系列ぶんまるごと(全tick)」を計算するので
    # seqs/s = 1000/ms がそのまま比較対象(tf_thr[1]と同じ単位)。
    packed_tf_ms = combined['packed_tf_ms']['median']
    combined['packed_tf_seqs_per_s'] = (
        1000.0 / packed_tf_ms if packed_tf_ms > 0 else 0.0)
    combined['packed_tf_speedup_vs_1ctx'] = (
        combined['packed_tf_seqs_per_s'] / combined['tf_thr'][1]
        if combined['tf_thr'].get(1) else 0.0)

    # greedy生成の系列バッチ化。32本ぶんのトークンを生成しきる
    # 時間から、集計tokens/s(32系列分をまとめて数えたトークン/秒)を出す
    # (tf_seqs_per_sと違い、こちらはトークン単位。tf側のtokens_per_s_agg
    # と単位を揃えた)。
    lanes = combined['g32_lanes'] or 32
    g32_seq_ms = combined['g32_seq_ms']['median']
    g32_packed_ms = combined['g32_packed_ms']['median']
    total_tokens = lanes * tgt_len
    combined['g32_seq_tokens_per_s_agg'] = (
        1000.0 * total_tokens / g32_seq_ms if g32_seq_ms > 0 else 0.0)
    combined['g32_packed_tokens_per_s_agg'] = (
        1000.0 * total_tokens / g32_packed_ms if g32_packed_ms > 0 else 0.0)
    combined['g32_speedup'] = (
        combined['g32_packed_tokens_per_s_agg']
        / combined['g32_seq_tokens_per_s_agg']
        if combined['g32_seq_tokens_per_s_agg'] > 0 else 0.0)

    # プールスケーリング効率(実測/理想、N=1を基準にした線形スケーリング
    # 比)。3系統(greedy通常/packed teacher forcing/greedy32)それぞれで
    # N=1のスループットに対する倍率÷Nを効率とする(4コアで70%以上が
    # タスクの勝ち基準)。greedy_pool_thrはtokens/s(既にtgt_len倍済み)、
    # packed_tf_pool_thrは既存tf_thrと同じ系列/s単位、g32_pool_thrは
    # tokens/s(32レーン×tgt_len倍済み)。
    def pool_efficiency(pool_key: str) -> dict:
        thr = combined.get(pool_key, {})
        base = thr.get(1)
        if not base:
            return {}
        return {n: {'throughput': v, 'speedup_vs_n1': v / base,
                    'efficiency_pct': 100.0 * (v / base) / n}
                for n, v in thr.items()}

    combined['greedy_pool_scaling'] = pool_efficiency('greedy_pool_thr')
    combined['packed_tf_pool_scaling'] = pool_efficiency('packed_tf_pool_thr')
    combined['g32_pool_scaling'] = pool_efficiency('g32_pool_thr')

    # 最大の見込み: greedy32(packed32)をN pool並列した集計tokens/s
    # と、ベースライン(単一コンテキスト・非packed greedy、char-LMで
    # 20,145 tokens/s)比の総合倍率。
    g32_pool_max_n = max(combined['g32_pool_thr']) if combined['g32_pool_thr'] else None
    combined['g32_pool_max_n'] = g32_pool_max_n
    combined['g32_pool_tokens_per_s_agg_max'] = (
        combined['g32_pool_thr'].get(g32_pool_max_n) if g32_pool_max_n else 0.0)
    return combined


def delivery_sizes(work: Path, bundle: dict) -> dict:
    src = sum((work / n).stat().st_size for n in ('model.c', 'model.h'))
    raw = json.dumps(bundle, separators=(',', ':')).encode()
    stripped = work / 'perf.stripped'
    shutil.copy2(work / 'perf', stripped)
    sh(['strip', '-x', str(stripped)])
    binz = stripped.stat().st_size
    embed_bytes = len((bundle.get('embed_table') or {}).get('bits', [])) * 4
    folded_n_bytes = len((bundle.get('folded_n_table') or {}).get('bits', [])) * 4
    folded_l_bytes = len((bundle.get('folded_l_table') or {}).get('bits', [])) * 4
    return {
        'bundle_gz': len(gzip.compress(raw, 9)),
        'src': src,
        'bin': binz,
        'bin_gz': len(gzip.compress(stripped.read_bytes(), 9)),
        'embed_table_bytes': embed_bytes,
        'folded_n_table_bytes': folded_n_bytes,
        'folded_l_table_bytes': folded_l_bytes,
        'weights_minus_embed_bytes': None,   # section_sizesのconstから後で埋める
    }


# ---------------------------------------------------------------------------
# PyTorch比較(hard eval、CPU、1スレッド既定のまま)
# ---------------------------------------------------------------------------
def bench_pytorch(model: RDDLGN, cfg: RDDLGNConfig, bos: int, eos_invalid: int,
                  reps: int) -> dict:
    src_tokens = torch.randint(0, cfg.vocab_size, (cfg.seq_len,))
    tgt_tokens = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
    src_tokens_b = src_tokens.unsqueeze(0)

    def time_calls(fn, reps):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2], ts[0]

    with torch.no_grad():
        gen_med, gen_best = time_calls(
            lambda: model.generate_greedy(
                src_tokens, bos_token=bos, eos_token=eos_invalid,
                max_new_tokens=cfg.seq_len),
            reps)
        tf_med, tf_best = time_calls(
            lambda: model(src_tokens_b, tgt_tokens), reps)
    return {'generate_greedy_ms': (gen_med, gen_best),
            'teacher_forcing_ms': (tf_med, tf_best), 'reps': reps,
            'threads': torch.get_num_threads()}


def bench_simulate_reference(model: RDDLGN, cfg: RDDLGNConfig, bos: int,
                             eos_invalid: int, graph) -> dict:
    """ir/simulate参考点(1点のみ)。"""
    rng = np.random.default_rng(0)
    in_width = int(graph.inputs[0].width)
    bits = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
    t0 = time.perf_counter()
    simulate(graph, bits)
    tf_ms = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()
    src_tokens = rng.integers(0, cfg.vocab_size, size=(cfg.seq_len,))
    src_bits = embed[src_tokens].reshape(-1).astype(np.int64)
    t0 = time.perf_counter()
    simulate_rddlgn_greedy_bits(model, src_bits, bos_token=bos,
                                eos_token=eos_invalid,
                                max_new_tokens=cfg.seq_len)
    gen_ms = (time.perf_counter() - t0) * 1000.0
    return {'teacher_forcing_ms': tf_ms, 'generate_greedy_ms': gen_ms}


# ---------------------------------------------------------------------------
# char-LM生成テキストサンプル
# ---------------------------------------------------------------------------
def generate_charlm_sample(work: Path, model: RDDLGN, cfg: RDDLGNConfig,
                           target_chars: int = 100) -> str:
    import struct as _struct

    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()

    seed_text = 'First Citizen:\nBefore we proceed'[:cfg.seq_len]
    generated = encode_shakespeare_char(seed_text)
    assert len(generated) == cfg.seq_len

    eos_invalid = 66_000
    while len(generated) < cfg.seq_len + target_chars:
        ctx_tokens = np.array(generated[-cfg.seq_len:], dtype=np.int64)
        src_bits = embed[ctx_tokens].reshape(-1).astype(np.uint8)
        query = src_bits.tobytes() + _struct.pack(
            '<III', CHAR_BOS, eos_invalid, cfg.seq_len)
        proc = subprocess.run([str(work / 'verify_gen')], input=query,
                              capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors='replace'))
        parts = [int(v) for v in proc.stdout.decode().strip().split()]
        n = parts[0]
        if n <= 0:
            break
        generated.extend(parts[1:1 + n])

    seed_str = decode_shakespeare_char(generated[:cfg.seq_len])
    gen_str = decode_shakespeare_char(generated[cfg.seq_len:])
    return f'[seed]{seed_str}[/seed][gen]{gen_str}[/gen]'


# ---------------------------------------------------------------------------
# scripts/bench_deploy.py の統一ディスパッチから呼ばれるエントリ。
# ---------------------------------------------------------------------------
def rddlgn_bench_row(name: str, spec: dict, cc: str, opt: int, *,
                     reps: int | None = None, max_thr: int = 8,
                     skip_verify: bool = False,
                     optimize_degenerate: bool = False) -> dict:
    """`bench_one()`をそのまま実行し(計測ロジックは無改修)、結果へ
    `kind`/`backend`を足して返す薄いアダプタ。`scripts/bench_deploy.py`は
    このエントリだけを呼び、`bench_one`本体は複製しない(設計方針)。
    `reps`を渡すとspec既定を上書きする(通常は既定のまま=Noneで使う)。
    """
    local_spec = dict(spec, reps=reps) if reps is not None else spec
    result = bench_one(name, local_spec, cc, opt, skip_verify=skip_verify,
                       max_thr=max_thr, optimize_degenerate=optimize_degenerate)
    result['kind'] = 'rddlgn'
    result['backend'] = 'c'
    return result


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def bench_one(name: str, spec: dict, cc: str, opt: int, *, skip_verify: bool,
             max_thr: int, optimize_degenerate: bool = False) -> dict:
    print(f'== {name}: {spec["desc"]} ==')
    t0 = time.time()
    model, cfg = spec['loader']()
    gc = gate_counts(cfg)
    print(f'  gate_counts={gc}')

    doc = build_doc(model, cfg, optimize_degenerate=optimize_degenerate)
    from griffith_logic.ir.serialize import ir_from_dict
    graph = ir_from_dict(doc['ir'])
    bundle = build_bundle(doc)
    print(f'  IR/bundle構築 {time.time() - t0:.1f}s')

    k26_gates = None
    if optimize_degenerate:
        # before(最適化なし)/after(最適化あり)の物理ゲート数。
        # bundleのビルドとは無関係の比較専用グラフなので、ここでしか
        # 使わない(build_docのgraphはafter=optimize_degenerate=True)。
        graph_before = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                                        optimize_degenerate=False)
        before = count_lut_gates(graph_before)['total']
        after = count_lut_gates(graph)['total']
        k26_gates = {'before': before, 'after': after,
                    'reduction_pct': 100.0 * (1 - after / before) if before else 0.0}
        print(f' 退化ゲート除去: 物理ゲート数 {before:,} → {after:,} '
              f'({k26_gates["reduction_pct"]:.1f}%削減)')

    work = BUILD_ROOT / name
    if work.exists():
        shutil.rmtree(work)
    packed_tf_available = emit_c_files(bundle, work)
    src_kb = (work / 'model.c').stat().st_size / 1024
    print(f'  生成C: model.c={src_kb:,.0f}KB')

    t0 = time.time()
    build_timings = build_all(cc, work, opt)
    print(f'  ビルド完了 {time.time() - t0:.1f}s'
          f'(model.o={build_timings["model_o"]:.1f}s)')

    result: dict = {
        'name': name, 'desc': spec['desc'], 'opt_level': opt,
        'gate_counts': gc, 'src_kb': src_kb,
        'build_timings': build_timings,
        'params': model.count_parameters(),
        'seq_len': cfg.seq_len, 'vocab_size': cfg.vocab_size,
        'd_emb': cfg.d_emb, 'optimize_degenerate': optimize_degenerate,
        'k26_gates': k26_gates, 'packed_tf_available': packed_tf_available,
    }

    if not skip_verify:
        t0 = time.time()
        tf_check = verify_teacher_forcing(work, graph)
        gen_check = verify_greedy(work, model, cfg, spec['bos'], spec['eos'])
        g32_check = verify_greedy_packed32(work, cfg, spec['bos'], spec['eos'])
        g32_note = '' if packed_tf_available else '(packed無効化のためスキップ扱い)'
        print(f'  検証: 教師強制={"OK" if tf_check["ok"] else "NG!!"} '
              f'/ 自己回帰生成={"OK" if gen_check["ok"] else "NG!!"} '
              f'/ 系列バッチ生成={"OK" if g32_check["ok"] else "NG!!"}'
              f'{g32_note} ({time.time() - t0:.1f}s)')
        result['verify'] = {'teacher_forcing': tf_check, 'greedy': gen_check,
                            'greedy_packed32': g32_check}
        if not tf_check['ok'] or not gen_check['ok'] or not g32_check['ok']:
            raise RuntimeError(f'{name}: 検証に失敗した。ベンチを中止する。')

    t0 = time.time()
    perf = run_perf_twice(work, spec['reps'], max_thr=max_thr)
    if packed_tf_available:
        packed_line = (
            f'packed_tf={perf["packed_tf_seqs_per_s"]:.1f}系列/s '
            f'({perf["packed_tf_speedup_vs_1ctx"]:.2f}x vs 既存tf 1ctx), '
            f'packed32={perf["g32_packed_tokens_per_s_agg"]:.1f}tok/s '
            f'({perf["g32_speedup"]:.2f}x vs 32回逐次)')
    else:
        packed_line = 'packed_tf/packed32=N/A(併用のため無効化)'
    print(f'  perf計測完了 {time.time() - t0:.1f}s '
          f'(encode={perf["encode_ms"]["median"]:.4f}ms, '
          f'decode_step={perf["decode_step_ms"]:.4f}ms, '
          f'tokens/s={perf["tokens_per_s"]:.1f}, {packed_line})')
    result['perf'] = perf

    secs = section_sizes(work / 'model.o')
    sizes = delivery_sizes(work, bundle)
    sizes['weights_minus_embed_bytes'] = (
        secs.get('const', 0) - sizes['embed_table_bytes'])
    sizes['weights_minus_all_tables_bytes'] = (
        secs.get('const', 0) - sizes['embed_table_bytes']
        - sizes['folded_n_table_bytes'] - sizes['folded_l_table_bytes'])
    rss = peak_rss_kb(work)
    result['sizes'] = sizes
    result['section_sizes'] = secs
    result['rss_kb'] = rss
    print(f'  配信: bundle_gz={kb(sizes["bundle_gz"])} src={kb(sizes["src"])} '
          f'bin_gz={kb(sizes["bin_gz"])} weights={kb(secs.get("const", 0))} '
          f'ctx={kb(perf["ctx_bytes"])} rss={rss:,}KB')
    print(f' 語彙折り畳み: N={"fold" if perf["has_folded_n"] else "skip"}'
          f'({kb(sizes["folded_n_table_bytes"])}) '
          f'L={"fold" if perf["has_folded_l"] else "skip"}'
          f'({kb(sizes["folded_l_table_bytes"])}) '
          f'encode {perf["encode_ms"]["median"]:.4f}→'
          f'{perf["encode_ms_folded"]["median"]:.4f}ms '
          f'decode/tok {perf["decode_step_ms"]:.4f}→'
          f'{perf["decode_step_ms_folded"]:.4f}ms')

    t0 = time.time()
    savings = measure_folded_delivery_savings(model, cfg, cc, opt, BUILD_ROOT)
    result['folded_delivery_savings'] = savings
    print(f' 配信サイズ before/after {time.time() - t0:.1f}s: '
          f'baseline={kb(savings["baseline_const_bytes"])} '
          + (f'dropped={kb(savings["dropped_const_bytes"])} '
             f'savings={kb(savings["savings_bytes"])}'
             f'({savings["savings_pct"]:.1f}%)'
             if savings['both_folded'] else '(両方foldしないため対象外)'))

    t0 = time.time()
    pt = bench_pytorch(model, cfg, spec['bos'], INVALID_EOS, min(spec['reps'], 30))
    print(f'  PyTorch計測完了 {time.time() - t0:.1f}s '
          f'(generate={pt["generate_greedy_ms"][0]:.3f}ms, '
          f'teacher_forcing={pt["teacher_forcing_ms"][0]:.3f}ms, '
          f'threads={pt["threads"]})')
    result['pytorch'] = pt

    t0 = time.time()
    sim_ref = bench_simulate_reference(model, cfg, spec['bos'], INVALID_EOS, graph)
    print(f'  simulate参考点 {time.time() - t0:.1f}s')
    result['simulate_reference'] = sim_ref

    if name == 'charlm':
        sample = generate_charlm_sample(work, model, cfg)
        print(f'  生成サンプル: {sample}')
        result['charlm_sample'] = sample

    return result


def print_markdown(results: list[dict]) -> None:
    print('\n## レイテンシ・スループット(4指標、ベースライン)\n')
    print('| モデル | encode ms | ttft ms | seq_e2e ms | decode/tok ms | '
          'tokens/s | tf 1ctx 系列/s | tf 8ctx 系列/s |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in results:
        p = r['perf']
        print(f"| {r['name']} | {p['encode_ms']['median']:.4f} | "
              f"{p['ttft_ms']['median']:.4f} | {p['seq_e2e_ms']['median']:.4f} | "
              f"{p['decode_step_ms']:.4f} | {p['tokens_per_s']:.1f} | "
              f"{p['tf_thr'].get(1, 0):.1f} | {p['tf_thr'].get(8, 0):.1f} |")

    print('\n## 教師強制の時間バッチ化(packed decode tail)\n')
    print('mdl_infer_ctx(既存、1ctx)と mdl_infer_packed_tf_ctx('
          'P headをsteps回+M+GroupSumをsteps本まとめて1回のpacked評価)を'
          '同一バイナリ・同一モデルで比較。値は整数厳密一致(検証フェーズで固定)。\n')
    print('| モデル | 既存 tf 1ctx 系列/s | packed tf 系列/s | 倍率 | '
          'packed work bytes |')
    print('|---|---:|---:|---:|---:|')
    for r in results:
        p = r['perf']
        print(f"| {r['name']} | {p['tf_thr'].get(1, 0):.1f} | "
              f"{p['packed_tf_seqs_per_s']:.1f} | "
              f"{p['packed_tf_speedup_vs_1ctx']:.2f}x | "
              f"{kb(p.get('packed_tf_work_bytes', 0))} |")

    print('\n## greedy生成の系列バッチ化(32本の独立系列)\n')
    print('mdl_rddlgn_generate_greedy(既存、32回逐次)と '
          'mdl_rddlgn_generate_greedy_packed32(各系列のP+Lは単発'
          '評価しつつM+GroupSumを32本まとめて1回のpacked評価)を比較。'
          '集計tokens/sは32系列ぶんのトークン数を要した時間で割ったもの'
          '(値はトークン列レベルで整数厳密一致)。\n')
    print('| モデル | 既存 32回逐次 集計tokens/s | packed32 集計tokens/s | 倍率 |')
    print('|---|---:|---:|---:|')
    for r in results:
        p = r['perf']
        print(f"| {r['name']} | {p['g32_seq_tokens_per_s_agg']:.1f} | "
              f"{p['g32_packed_tokens_per_s_agg']:.1f} | "
              f"{p['g32_speedup']:.2f}x |")

    print('\n## 配信サイズとメモリ\n')
    print('| モデル | バンドルgz | 生成C | バイナリgz | 重み(.rodata) | '
          '埋め込み表 | 作業RAM | ピークRSS |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in results:
        s = r['sizes']
        print(f"| {r['name']} | {kb(s['bundle_gz'])} | {kb(s['src'])} | "
              f"{kb(s['bin_gz'])} | {kb(r['section_sizes'].get('const', 0))} | "
              f"{kb(s['embed_table_bytes'])} | {kb(r['perf']['ctx_bytes'])} | "
              f"{r['rss_kb']:,}KB |")

    print('\n## 語彙折り畳み before/after — レイテンシ(同一バイナリ内、'
         'mdl_rddlgn_generate_greedy(before) vs '
         'mdl_rddlgn_generate_greedy_folded(after))\n')
    print('| モデル | N fold | L fold | 表バイト(N+L) | encode前→後 ms | '
         'decode/tok前→後 ms | tokens/s前→後 |')
    print('|---|---|---|---:|---:|---:|---:|')
    for r in results:
        p = r['perf']
        s = r['sizes']
        table_bytes = s['folded_n_table_bytes'] + s['folded_l_table_bytes']
        print(f"| {r['name']} | {'○' if p['has_folded_n'] else '×'} | "
              f"{'○' if p['has_folded_l'] else '×'} | {kb(table_bytes)} | "
              f"{p['encode_ms']['median']:.4f}→{p['encode_ms_folded']['median']:.4f} | "
              f"{p['decode_step_ms']:.4f}→{p['decode_step_ms_folded']:.4f} | "
              f"{p['tokens_per_s']:.0f}→{p['tokens_per_s_folded']:.0f} |")

    print('\n## 語彙折り畳み before/after — 配信サイズ('
         '折り畳み無しbaseline vs 折り畳み+重み削除drop_folded_weights=True、'
         '別ビルド。.rodataバイト、`size -m`)\n')
    print('| モデル | baseline | dropped | 純減 | 純減% |')
    print('|---|---:|---:|---:|---:|')
    for r in results:
        sv = r['folded_delivery_savings']
        if sv['both_folded']:
            print(f"| {r['name']} | {kb(sv['baseline_const_bytes'])} | "
                  f"{kb(sv['dropped_const_bytes'])} | "
                  f"{kb(sv['savings_bytes'])} | {sv['savings_pct']:.1f}% |")
        else:
            print(f"| {r['name']} | {kb(sv['baseline_const_bytes'])} | "
                  f"(対象外) | - | - |")

    print('\n## PyTorch比\n')
    print('| モデル | PT generate ms | C seq_e2e ms | 倍率 | '
          'PT teacher-forcing ms | C infer(1ctx) ms |')
    print('|---|---:|---:|---:|---:|---:|')
    for r in results:
        pt = r['pytorch']
        p = r['perf']
        c_tf_ms = 1000.0 / max(p['tf_thr'].get(1, 1e-9), 1e-9)
        ratio = pt['generate_greedy_ms'][0] / max(p['seq_e2e_ms']['median'], 1e-9)
        print(f"| {r['name']} | {pt['generate_greedy_ms'][0]:.3f} | "
              f"{p['seq_e2e_ms']['median']:.3f} | {ratio:.1f}x | "
              f"{pt['teacher_forcing_ms'][0]:.3f} | {c_tf_ms:.4f} |")

    print('\n## レイヤ別プロファイル(単一tick、ms)\n')
    print('PMはP-layers+M-layers+GroupSumの合算(mdl_step_dec_pm_recが1関数に'
          '畳んでいるため実測ではP/M分離不可。ゲート数比較はdecode相当'
          '(L+P+M)の中でのPM比で揃えた — 予測M/全体比(全層中のM、'
          '事前予測はこちら)とは分母が異なる点に注意。\n')
    print('| モデル | N(全ticks) ms | K(全ticks) ms | L(1tick) ms | '
          'PM(1tick) ms | embed(1tok) ms | 実測PM/decode比 | '
          '予測PM/decode比(ゲート数) | 参考: 予測M/全体比 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in results:
        p = r['perf']
        gc = r['gate_counts']
        n_ms = p['prof_n_ms']['median']
        k_ms = p['prof_k_ms']['median']
        l_ms = p['prof_l_ms']['median']
        pm_ms = p['prof_pm_ms']['median']
        embed_ms = p['prof_embed_ms']['median']
        decode_total = l_ms + pm_ms + embed_ms
        measured_pm_share = pm_ms / decode_total if decode_total else 0.0
        gate_decode_total = gc['L'] + gc['P'] + gc['M']
        predicted_pm_share = ((gc['P'] + gc['M']) / gate_decode_total
                              if gate_decode_total else 0.0)
        print(f"| {r['name']} | {n_ms:.5f} | {k_ms:.5f} | "
              f"{l_ms:.5f} | {pm_ms:.5f} | "
              f"{embed_ms:.5f} | {100*measured_pm_share:.1f}% | "
              f"{100*predicted_pm_share:.1f}% | {100*gc['m_ratio']:.1f}% |")

    print('\n## マルチコアスケーリング(mdl_run_pool、N=1/2/4/8、'
         'Apple M3 4P+4E)\n')
    print('3系統: greedy(通常、1系列ずつ)/packed teacher forcing/'
         'greedy32(1item=32本バッチ)。efficiency%は'
         '(N倍率)/N×100(4コアで70%以上が勝ち基準)。\n')
    print('| モデル | 系統 | N=1 | N=2(eff%) | N=4(eff%) | N=8(eff%) |')
    print('|---|---|---:|---:|---:|---:|')
    for r in results:
        p = r['perf']
        for label, key in (('greedy(通常)', 'greedy_pool_scaling'),
                           ('packed tf', 'packed_tf_pool_scaling'),
                           ('greedy32', 'g32_pool_scaling')):
            sc = p.get(key, {})
            if not sc:
                continue
            cells = []
            for n in (2, 4, 8):
                e = sc.get(n)
                cells.append(f"{e['throughput']:.1f}({e['efficiency_pct']:.0f}%)"
                            if e else '-')
            n1 = sc.get(1)
            n1_s = f"{n1['throughput']:.1f}" if n1 else '-'
            print(f"| {r['name']} | {label} | {n1_s} | {cells[0]} | "
                  f"{cells[1]} | {cells[2]} |")

    print('\n## greedy32マルチコア化の総合倍率\n')
    print('ベースライン(単一コンテキスト・非packed greedy)比の総合倍率'
         '(packed経路の倍率×コア数効率の積)。\n')
    print('| モデル | 1ctx tokens/s | greedy32 N=1 tokens/s_agg | '
         '最大N tokens/s_agg | 総合倍率(vs 1ctx) |')
    print('|---|---:|---:|---:|---:|')
    for r in results:
        p = r['perf']
        base = p['tokens_per_s']
        n1 = p.get('g32_pool_thr', {}).get(1, 0.0)
        max_n = p.get('g32_pool_max_n')
        max_thr_val = p.get('g32_pool_tokens_per_s_agg_max', 0.0)
        overall = max_thr_val / base if base > 0 else 0.0
        print(f"| {r['name']} | {base:.1f} | {n1:.1f} | "
              f"N={max_n}: {max_thr_val:.1f} | {overall:.2f}x |")


# ---------------------------------------------------------------------------
# 非推奨の薄いラッパ。計測本体(bench_one/print_markdown/MODEL_SPECS等、
# 上のすべての関数)はscripts/bench_deploy.pyが`rddlgn_bench_row()`経由で
# そのままimportして呼ぶ「計測エンジン」として残る。このmain()自体は
# 旧CLI引数を受けて`bench_deploy.py`の統一引数へ変換し、統一実装を呼ぶだけ
# (挙動互換: 出力先の既定・print_markdownの全表は元のまま)。
# ---------------------------------------------------------------------------
def _load_bench_deploy():
    import importlib.util
    if 'bench_deploy' in sys.modules:
        return sys.modules['bench_deploy']
    spec = importlib.util.spec_from_file_location(
        'bench_deploy', Path(__file__).resolve().parent / 'bench_deploy.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['bench_deploy'] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', default=None,
                    help='カンマ区切り(既定: 全4モデル)')
    ap.add_argument('--opt', type=int, default=None,
                    help='最適化レベルを固定(既定: paper_scale1のコンパイル'
                         '時間から自動決定)')
    ap.add_argument('--skip-verify', action='store_true')
    ap.add_argument('--max-thr', type=int, default=8)
    ap.add_argument('--out', default=None)
    ap.add_argument('--optimize-degenerate', action='store_true',
                    help='LutStackの退化ゲート除去パスを有効化する'
                         '(既定off。packed_tf語彙折り畳みとは'
                         '独立に併用可能)')
    args = ap.parse_args()

    print('[非推奨] scripts/bench_rddlgn.py は scripts/bench_deploy.py '
          '--rddlgn(-only) へ統一された。今後は '
          '`uv run python scripts/bench_deploy.py --rddlgn-only ...` を'
          '使うこと。このラッパは統一実装をそのまま呼び出すだけ。',
          file=sys.stderr)

    deploy = _load_bench_deploy()
    forward = ['--backend', 'c', '--rddlgn-only', '--rddlgn-max-thr',
              str(args.max_thr)]
    if args.models:
        forward += ['--rddlgn-models', args.models]
    if args.opt is not None:
        forward += ['--rddlgn-opt', str(args.opt)]
    if args.skip_verify:
        forward += ['--rddlgn-skip-verify']
    if args.optimize_degenerate:
        forward += ['--rddlgn-optimize-degenerate']
    if args.out:
        forward += ['--rddlgn-out', args.out]
    return deploy.main(forward)


if __name__ == '__main__':
    raise SystemExit(main())
