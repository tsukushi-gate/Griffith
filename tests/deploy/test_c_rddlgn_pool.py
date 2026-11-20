"""RDDLGN seq2seqのマルチコア推論。

tests/deploy/test_c_context.py(非recurrentモデル)と同じ性質を
recurrentノード(state_init/ext_init/flat_concat/recurrent_execを含む
mdl_ctx)を持つモデルでも固定する: **コンテキストが別なら並行に呼んでも
干渉しない**。recurrent固有のリスクは、state_bufs/ext_bufsが
「steps回のループの中で読み書きされ続ける」領域なので、他のコンテキストの
ワーカと領域を共有していれば結果が崩れる(tests/deploy/
test_rddlgn_memory_plan.py::TestRepeatedInferenceIsCleanの並列版)。

`mdl_run_pool`は既存のまま(コンテキストプールという設計自体は
確立済みで、値種別に関わらずコンテキスト単位で独立)。
recurrentを含むバンドルで初めて通すのがこのテストの役目。
"""

import subprocess

import numpy as np
import pytest

from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C
from .test_c_kernels import read_u32

#: 2コンテキストで4系列を分担して処理するドライバ(work-stealingで
#: どちらのコンテキストがどの系列を担当するかは実行時に決まる)。
POOL_DRIVER = '''#include "model.h"
#include "mdl_threads.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
#define N_ITEMS 4
#define N_CTX 2
static uint32_t work[N_CTX][MDL_CTX_BYTES / 4];
static mdl_ctx ctx[N_CTX];
static void *ctxp[N_CTX];
static uint8_t inputs[N_ITEMS][MDL_INPUT_WIDTH];
static int32_t scores[N_ITEMS][MDL_OUTPUT_WIDTH];
int main(void) {
  MDL_STDIO_BINARY();
  int i, j;
  for (i = 0; i < N_ITEMS; i++) {
    if (fread(inputs[i], 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH)
      return 1;
  }
  for (i = 0; i < N_CTX; i++) {
    if (mdl_ctx_init(&ctx[i], work[i], sizeof(work[i])) != 0) return 3;
    ctxp[i] = &ctx[i];
  }
  if (mdl_run_pool(mdl_infer_ctx_v, ctxp, N_CTX,
                   &inputs[0][0], (size_t)MDL_INPUT_WIDTH,
                   &scores[0][0], sizeof(int32_t) * MDL_OUTPUT_WIDTH,
                   N_ITEMS) != 0) return 2;
  for (i = 0; i < N_ITEMS; i++) {
    for (j = 0; j < MDL_OUTPUT_WIDTH; j++) {
      unsigned char b[4];
      uint32_t v = (uint32_t)scores[i][j];
      b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
      b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
      fwrite(b, 1, 4, stdout);
    }
  }
  return 0;
}
'''


def _compile(cc, work, driver, extra_src=(), extra_flags=()):
    (work / 'driver.c').write_text(driver, encoding='utf-8')
    cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
          f'-I{work}', f'-I{RUNTIME_C / "include"}',
          str(work / 'model.c'), *[str(p) for p in extra_src],
          str(work / 'driver.c'), '-o', str(work / 'app'), *extra_flags]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return work / 'app'


def _emit_model(bundle, tmp_path):
    files = generate_c(bundle, name='model', validate=False)
    work = tmp_path / 'gen'
    work.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    return work


@pytest.mark.slow
class TestThreadPoolMatchesSequential:
    """4系列を2コンテキストで並列処理した結果がsimulateと完全一致する
    (=コンテキスト間で干渉していない)。"""

    def test_pool_result_equals_sequential(self, cc, tmp_path,
                                           rddlgn_seq2seq_tiny_bundle):
        threads_c = RUNTIME_C / 'src' / 'mdl_threads.c'
        threads_h_dir = RUNTIME_C / 'include'
        bundle = rddlgn_seq2seq_tiny_bundle
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(20260815)
        x = rng.integers(0, 2, size=(4, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        work = _emit_model(bundle, tmp_path)
        try:
            exe = _compile(
                cc, work, POOL_DRIVER,
                extra_src=[RUNTIME_C / 'src' / 'mdl_kernels.c', threads_c],
                extra_flags=['-pthread', f'-I{threads_h_dir}'])
        except AssertionError as exc:
            if 'pthread' in str(exc):
                pytest.skip('pthread が使えない環境')
            raise
        proc = subprocess.run([str(exe)], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        got = read_u32(proc.stdout).astype(np.int64).reshape(want.shape)
        assert np.array_equal(got, want), (
            '2コンテキストの並列推論がsimulateと不一致'
            '(recurrentのstate_bufs/ext_bufsがコンテキスト間で漏れている疑い)')


@pytest.mark.slow
class TestRangeParallelExtraction:
    """既存 `mdl_preproc_parallel` から切り出した汎用プリミティブ
    `mdl_range_parallel`。系列内t並列はコード生成にはまだ
    組み込まないが(扱う)、プリミティブ自体はここで動作を固定する。

    n個の仕事をn_threads個のワーカへストライド分割し(担当iはi, i+nt,
    i+2nt, ...)、全部呼ばれる・二重に呼ばれないことを確認する。
    """

    DRIVER = '''#include "mdl_threads.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>
#define N 37
static int hit[N];
static int call(void *ctx, int start, int stride) {
  int i;
  (void)ctx;
  for (i = start; i < N; i += stride) hit[i] += 1;
  return 0;
}
int main(void) {
  MDL_STDIO_BINARY();
  int i, n_threads;
  for (n_threads = 1; n_threads <= 8; n_threads++) {
    memset(hit, 0, sizeof(hit));
    if (mdl_range_parallel(call, NULL, N, n_threads) != 0) return 2;
    for (i = 0; i < N; i++) {
      if (hit[i] != 1) return 3;   /* 未実行 or 二重実行 */
    }
  }
  printf("ok\\n");
  return 0;
}
'''

    def test_every_index_visited_exactly_once(self, cc, tmp_path):
        threads_c = RUNTIME_C / 'src' / 'mdl_threads.c'
        (tmp_path / 'driver.c').write_text(self.DRIVER, encoding='utf-8')
        cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
              f'-I{RUNTIME_C / "include"}', str(tmp_path / 'driver.c'),
              str(threads_c), '-pthread', '-o', str(tmp_path / 'app')]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            if 'pthread' in build.stderr:
                pytest.skip('pthread が使えない環境')
            raise AssertionError(build.stderr)
        proc = subprocess.run([str(tmp_path / 'app')], capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        assert proc.stdout.strip() == b'ok'
