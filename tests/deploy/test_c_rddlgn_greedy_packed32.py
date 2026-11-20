"""RDDLGN 自己回帰生成の系列バッチ化の整数厳密一致検証。

デプロイキット設計メモ節の仮説2: greedy生成では時間方向を
packできない(自己回帰、tがt-1に依存)が、**32本の独立した系列**なら
互いに依存が無いので、各系列のL/P(state更新、真のリカレント)は
系列ごとに単発評価しつつ、M-layers+GroupSum(state_tだけの純関数)は
32系列ぶんまとめて`mdl_<id>_tail_packed32`(教師強制の時間バッチ化
と同じ`packed_teacher_forcing`フラグが同時に生やす)で1回評価する
`mdl_rddlgn_generate_greedy_packed32`(runtime/c/src/mdl_rddlgn.c)を
追加した。

このテストは、32本まとめた生成結果が「`mdl_rddlgn_generate_greedy`
(既存)を32回独立に呼んだ場合」とトークン列レベルで完全一致する
ことを固定する — 値は変えず速さだけを変える最適化、という原則
(perf-huntスキル⑤)をここでも守る。
"""

import struct
import subprocess

import numpy as np
import pytest
import torch

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.cbackend import generate_rddlgn_c
from griffith_logic.export.rddlgn_ir import extract_embed_table

from .conftest import RUNTIME_C, build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc

LANES = 32

_DRIVER = f'''#include "model.h"
#include "mdl_rddlgn.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>

#define LANES {LANES}

static uint32_t work[MDL_CTX_BYTES / 4];
static uint32_t pwork[MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES / 4 + 4];
static mdl_rddlgn_g32_scratch scratch;
static uint8_t src_bits32[LANES][MDL_BUF_WIDTH_X_SRC];
static uint32_t bos_tokens[LANES];
static uint32_t out_tokens_packed[LANES][MDL_STEPS_DEC_PM_REC];
static int32_t out_lens_packed[LANES];
static uint32_t out_tokens_seq[LANES][MDL_STEPS_DEC_PM_REC];
static int32_t out_lens_seq[LANES];

int main(void) {{
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  int lane;
  uint32_t eos, max_new;

  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 10;
  for (;;) {{
    size_t got = fread(src_bits32, 1, sizeof(src_bits32), stdin);
    if (got == 0) break;
    if (got != sizeof(src_bits32)) return 11;
    if (fread(bos_tokens, sizeof(uint32_t), LANES, stdin) != LANES) return 12;
    if (fread(&eos, sizeof(eos), 1, stdin) != 1) return 13;
    if (fread(&max_new, sizeof(max_new), 1, stdin) != 1) return 14;

    if (mdl_rddlgn_generate_greedy_packed32(
            &ctx, &scratch, &src_bits32[0][0], bos_tokens, eos, max_new,
            pwork, sizeof(pwork), &out_tokens_packed[0][0],
            out_lens_packed) != 0) {{
      return 2;
    }}
    for (lane = 0; lane < LANES; lane++) {{
      int n = mdl_rddlgn_generate_greedy(
          &ctx, src_bits32[lane], bos_tokens[lane], eos, max_new,
          out_tokens_seq[lane]);
      if (n < 0) return 3;
      out_lens_seq[lane] = n;
    }}
    for (lane = 0; lane < LANES; lane++) {{
      if (out_lens_packed[lane] != out_lens_seq[lane]) return 5;
      if (memcmp(out_tokens_packed[lane], out_tokens_seq[lane],
                 sizeof(uint32_t) * (size_t)out_lens_seq[lane]) != 0) {{
        return 6;
      }}
    }}
    printf("OK\\n");
  }}
  return 0;
}}
'''

#:: 2つの`mdl_ctx`/`mdl_rddlgn_g32_scratch`/pworkを2スレッドで並行に
#: 走らせても、各スレッドが単独(逐次)で32本まとめて生成した場合と
#: トークン列レベルで完全一致することを固定する(スレッドごとに別々の
#: scratchを渡す限りスレッドセーフ、という設計の直接検証)。
_DRIVER_POOL = f'''#include "model.h"
#include "mdl_rddlgn.h"
#include <pthread.h>
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>

#define LANES {LANES}
#define NPOOL 2

static uint32_t work[NPOOL][MDL_CTX_BYTES / 4];
static mdl_ctx ctx[NPOOL];
static mdl_rddlgn_g32_scratch scratch[NPOOL];
static uint32_t pwork[NPOOL][MDL_DEC_PM_REC_TAIL_PACKED32_WORK_BYTES / 4 + 4];
static uint8_t src_bits32[NPOOL][LANES][MDL_BUF_WIDTH_X_SRC];
static uint32_t bos_tokens[NPOOL][LANES];
static uint32_t out_tokens_pool[NPOOL][LANES][MDL_STEPS_DEC_PM_REC];
static int32_t out_lens_pool[NPOOL][LANES];
static uint32_t out_tokens_seq[NPOOL][LANES][MDL_STEPS_DEC_PM_REC];
static int32_t out_lens_seq[NPOOL][LANES];
static uint32_t g_eos, g_max_new;
static int g_rc[NPOOL];

static void *worker(void *arg) {{
  int idx = (int)(size_t)arg;
  g_rc[idx] = mdl_rddlgn_generate_greedy_packed32(
      &ctx[idx], &scratch[idx], &src_bits32[idx][0][0], bos_tokens[idx],
      g_eos, g_max_new, pwork[idx], sizeof(pwork[idx]),
      &out_tokens_pool[idx][0][0], out_lens_pool[idx]);
  return NULL;
}}

int main(void) {{
  MDL_STDIO_BINARY();
  pthread_t tid[NPOOL];
  int idx, lane;

  for (idx = 0; idx < NPOOL; idx++) {{
    if (mdl_ctx_init(&ctx[idx], work[idx], sizeof(work[idx])) != 0) return 10;
  }}
  for (;;) {{
    size_t got = fread(src_bits32, 1, sizeof(src_bits32), stdin);
    if (got == 0) break;
    if (got != sizeof(src_bits32)) return 11;
    if (fread(bos_tokens, sizeof(uint32_t), NPOOL * LANES, stdin)
        != (size_t)(NPOOL * LANES)) return 12;
    if (fread(&g_eos, sizeof(g_eos), 1, stdin) != 1) return 13;
    if (fread(&g_max_new, sizeof(g_max_new), 1, stdin) != 1) return 14;

    /* 並行実行(NPOOL本のバッチを同時に処理、各自別ctx/scratch/pwork)。 */
    for (idx = 1; idx < NPOOL; idx++) {{
      if (pthread_create(&tid[idx], NULL, worker, (void *)(size_t)idx) != 0)
        return 15;
    }}
    worker((void *)(size_t)0);
    for (idx = 1; idx < NPOOL; idx++) pthread_join(tid[idx], NULL);
    for (idx = 0; idx < NPOOL; idx++) if (g_rc[idx] != 0) return 2;

    /* 突き合わせ用に、同じ入力を単独(逐次)のmdl_rddlgn_generate_greedyで
       再計算する(並行実行の結果がスレッド間で混線していないことの
       検証)。 */
    for (idx = 0; idx < NPOOL; idx++) {{
      for (lane = 0; lane < LANES; lane++) {{
        int n = mdl_rddlgn_generate_greedy(
            &ctx[0], src_bits32[idx][lane], bos_tokens[idx][lane], g_eos,
            g_max_new, out_tokens_seq[idx][lane]);
        if (n < 0) return 3;
        out_lens_seq[idx][lane] = n;
      }}
    }}
    for (idx = 0; idx < NPOOL; idx++) {{
      for (lane = 0; lane < LANES; lane++) {{
        if (out_lens_pool[idx][lane] != out_lens_seq[idx][lane]) return 5;
        if (memcmp(out_tokens_pool[idx][lane], out_tokens_seq[idx][lane],
                   sizeof(uint32_t) * (size_t)out_lens_seq[idx][lane]) != 0) {{
          return 6;
        }}
      }}
    }}
    printf("OK\\n");
  }}
  return 0;
}}
'''


def _build(bundle, work_dir, cc, *, driver_src=_DRIVER, driver_name='driver.c',
          extra_flags=()):
    files = generate_rddlgn_c(bundle, name='model', validate=False,
                              packed_teacher_forcing='dec_pm_rec')
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work_dir / name).write_text(text, encoding='utf-8')
    for rel in ('include/mdl_kernels.h', 'include/mdl_simd.h',
                'include/mdl_stdio.h',
                'src/mdl_kernels.c', 'include/mdl_rddlgn.h',
                'src/mdl_rddlgn.c'):
        import shutil
        shutil.copy2(RUNTIME_C / rel, work_dir / __import__('pathlib').Path(rel).name)
    (work_dir / driver_name).write_text(driver_src, encoding='utf-8')
    exe = work_dir / driver_name.removesuffix('.c')
    cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
          f'-I{work_dir}', *extra_flags,
          str(work_dir / 'model.c'), str(work_dir / 'mdl_kernels.c'),
          str(work_dir / 'mdl_rddlgn.c'), str(work_dir / driver_name),
          '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return exe


@pytest.fixture(scope='module')
def g32_model():
    return build_rddlgn_tiny_seq2seq(seed=0)


@pytest.fixture(scope='module')
def g32_bundle(g32_model):
    model, cfg = g32_model
    doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
    doc['embed_table'] = extract_embed_table(model.src_embedding)
    return build_bundle(doc)


@pytest.fixture(scope='module')
def g32_exe(cc, tmp_path_factory, g32_bundle):
    work = tmp_path_factory.mktemp('rddlgn_greedy_packed32')
    return _build(g32_bundle, work, cc)


@pytest.fixture(scope='module')
def g32_pool_exe(cc, tmp_path_factory, g32_bundle):
    """scratchを呼び出し側持ちへ一般化した後の、2スレッド並行実行の
    整数一致検証用ドライバ(-pthreadでリンク)。"""
    work = tmp_path_factory.mktemp('rddlgn_greedy_packed32_pool')
    return _build(g32_bundle, work, cc, driver_src=_DRIVER_POOL,
                 driver_name='driver_pool.c', extra_flags=('-pthread',))


def _embed_src_bits(model, tokens: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        embed = (torch.sigmoid(model.src_embedding.embedding.weight) > 0.5)
    embed = embed.to(torch.uint8).cpu().numpy()
    return embed[tokens].reshape(tokens.shape[0], -1).astype(np.uint8)


def _run_trial(exe, model, cfg, rng, eos: int, max_new: int) -> subprocess.CompletedProcess:
    src_tokens = rng.integers(0, cfg.vocab_size, size=(LANES, cfg.seq_len))
    src_bits = _embed_src_bits(model, src_tokens)
    bos = rng.integers(0, cfg.vocab_size, size=(LANES,)).astype('<u4')
    payload = (src_bits.tobytes() + bos.tobytes()
              + struct.pack('<II', int(eos), int(max_new)))
    return subprocess.run([str(exe)], input=payload, capture_output=True)


class TestPacked32MatchesSequential:
    """32本まとめた生成結果がgenerate_greedyを32回呼んだものと完全一致する。"""

    def test_random_samples_no_early_exit(self, g32_exe, g32_model):
        model, cfg = g32_model
        rng = np.random.default_rng(3)
        proc = _run_trial(g32_exe, model, cfg, rng, eos=999_999,
                          max_new=cfg.seq_len)
        assert proc.returncode == 0, (
            f'rc={proc.returncode}: {proc.stderr.decode(errors="replace")}')
        assert proc.stdout.decode().strip() == 'OK'

    @pytest.mark.parametrize('eos_val', [0, 5, 17])
    def test_mixed_early_exit(self, g32_exe, g32_model, eos_val):
        """EOSが系列によって早くヒットしたりしなかったりする混在ケース
        (done[lane]スキップ経路の固定)。"""
        model, cfg = g32_model
        rng = np.random.default_rng(11 + eos_val)
        proc = _run_trial(g32_exe, model, cfg, rng, eos=eos_val,
                          max_new=cfg.seq_len)
        assert proc.returncode == 0, (
            f'rc={proc.returncode}: {proc.stderr.decode(errors="replace")}')
        assert proc.stdout.decode().strip() == 'OK'

    def test_shorter_max_new_tokens(self, g32_exe, g32_model):
        model, cfg = g32_model
        rng = np.random.default_rng(5)
        proc = _run_trial(g32_exe, model, cfg, rng, eos=999_999,
                          max_new=max(1, cfg.seq_len - 2))
        assert proc.returncode == 0
        assert proc.stdout.decode().strip() == 'OK'

    def test_multiple_calls_same_ctx_do_not_leak_state(self, g32_exe,
                                                        g32_model):
        """同一プロセス内で複数回呼んでも前回の生成が残らない
        (mdl_ctxを使い回す実運用に対応、流儀を踏襲)。"""
        model, cfg = g32_model
        rng = np.random.default_rng(21)
        payload = b''
        for _ in range(3):
            src_tokens = rng.integers(0, cfg.vocab_size,
                                      size=(LANES, cfg.seq_len))
            src_bits = _embed_src_bits(model, src_tokens)
            bos = rng.integers(0, cfg.vocab_size,
                               size=(LANES,)).astype('<u4')
            payload += (src_bits.tobytes() + bos.tobytes()
                       + struct.pack('<II', 999_999, cfg.seq_len))
        proc = subprocess.run([str(g32_exe)], input=payload,
                              capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        lines = proc.stdout.decode().strip('\n').split('\n')
        assert lines == ['OK', 'OK', 'OK']


class TestPacked32ThreadPoolMatchesSequential:
    """`mdl_rddlgn_generate_greedy_packed32`のscratchバッファを
    呼び出し側持ちへ一般化(はstaticでスレッド安全でなかった)した
    ことの直接検証。2スレッドが別々のctx/scratch/pworkで**同時に**32本
    バッチを処理しても、各バッチが単独実行(逐次mdl_rddlgn_generate_greedy)
    した場合とトークン列レベルで完全一致することを固定する
    (`_DRIVER_POOL`が内部で突き合わせ、不一致ならnonzeroで落ちる)。"""

    NPOOL = 2

    def _run_pool_trial(self, exe, model, cfg, rng, eos: int, max_new: int):
        src_tokens = rng.integers(0, cfg.vocab_size,
                                  size=(self.NPOOL, LANES, cfg.seq_len))
        src_bits = np.stack([_embed_src_bits(model, src_tokens[i])
                             for i in range(self.NPOOL)])
        bos = rng.integers(0, cfg.vocab_size,
                           size=(self.NPOOL, LANES)).astype('<u4')
        payload = (src_bits.tobytes() + bos.tobytes()
                  + struct.pack('<II', int(eos), int(max_new)))
        return subprocess.run([str(exe)], input=payload, capture_output=True)

    def test_two_threads_no_early_exit(self, g32_pool_exe, g32_model):
        model, cfg = g32_model
        rng = np.random.default_rng(31)
        proc = self._run_pool_trial(g32_pool_exe, model, cfg, rng,
                                    eos=999_999, max_new=cfg.seq_len)
        assert proc.returncode == 0, (
            f'rc={proc.returncode}: {proc.stderr.decode(errors="replace")}')
        assert proc.stdout.decode().strip() == 'OK'

    def test_two_threads_mixed_early_exit(self, g32_pool_exe, g32_model):
        model, cfg = g32_model
        rng = np.random.default_rng(37)
        proc = self._run_pool_trial(g32_pool_exe, model, cfg, rng,
                                    eos=7, max_new=cfg.seq_len)
        assert proc.returncode == 0, (
            f'rc={proc.returncode}: {proc.stderr.decode(errors="replace")}')
        assert proc.stdout.decode().strip() == 'OK'

    def test_repeated_calls_do_not_leak_across_threads(self, g32_pool_exe,
                                                        g32_model):
        """同一プロセス内で複数回(スレッドを都度join/再生成)呼んでも、
        前回の並行実行の状態がスレッド間・呼び出し間で漏れないこと。"""
        model, cfg = g32_model
        rng = np.random.default_rng(41)
        payload = b''
        for _ in range(3):
            src_tokens = rng.integers(0, cfg.vocab_size,
                                      size=(self.NPOOL, LANES, cfg.seq_len))
            src_bits = np.stack([_embed_src_bits(model, src_tokens[i])
                                 for i in range(self.NPOOL)])
            bos = rng.integers(0, cfg.vocab_size,
                               size=(self.NPOOL, LANES)).astype('<u4')
            payload += (src_bits.tobytes() + bos.tobytes()
                       + struct.pack('<II', 999_999, cfg.seq_len))
        proc = subprocess.run([str(g32_pool_exe)], input=payload,
                              capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        lines = proc.stdout.decode().strip('\n').split('\n')
        assert lines == ['OK', 'OK', 'OK']


class TestGeneratedSourceProperties:
    def test_no_heap_allocation(self, g32_bundle):
        files = generate_rddlgn_c(g32_bundle, name='model', validate=False,
                                  packed_teacher_forcing='dec_pm_rec')
        for name in ('model.c',):
            for bad in ('malloc', 'calloc', 'realloc', 'free('):
                assert bad not in files[name], f'{name}に{bad}がある'

    def test_no_pthread_dependency(self, g32_bundle):
        files = generate_rddlgn_c(g32_bundle, name='model', validate=False,
                                  packed_teacher_forcing='dec_pm_rec')
        for text in files.values():
            assert 'pthread' not in text
