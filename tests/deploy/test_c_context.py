"""コンテキスト化とマルチコアヘルパ。

作業バッファを `mdl_ctx` へ移したので、コンテキストを複数作れば同じモデルを
複数スレッドで同時に走らせられる。ここで固定するのは3点:

    1. **コンテキストは互いに干渉しない**(交互に別入力を推論しても崩れない)
    2. **`-DMDL_NO_DEFAULT_CTX` で static RAM が完全に消える**
       (数十KB RAM級のMCU向け。コアは相変わらずヒープもスレッドも使わない)
    3. **N枚をNコアで回した結果が逐次と完全一致する**

3 は「並列化したら答えが変わった」を防ぐ最重要の性質。DLGN は1推論が
純粋関数なので**本来一致するはず**で、一致しないならバッファ共有の誤りがある。
"""

import subprocess

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C
from .test_c_kernels import read_u32

#: 2つのコンテキストで**交互に**推論するドライバ。
#: どちらかが他方の作業領域を踏んでいれば結果が崩れる。
INTERLEAVE_DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
static uint32_t work_a[MDL_CTX_BYTES / 4];
static uint32_t work_b[MDL_CTX_BYTES / 4];
static uint8_t in_a[MDL_INPUT_WIDTH], in_b[MDL_INPUT_WIDTH];
static int32_t sa[MDL_OUTPUT_WIDTH], sb[MDL_OUTPUT_WIDTH];
static void emit(const int32_t *s) {
  int32_t i;
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
    unsigned char b[4];
    uint32_t v = (uint32_t)s[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }
}
int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx a, b;
  if (fread(in_a, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) return 1;
  if (fread(in_b, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) return 1;
  if (mdl_ctx_init(&a, work_a, sizeof(work_a)) != 0) return 3;
  if (mdl_ctx_init(&b, work_b, sizeof(work_b)) != 0) return 3;
  /* わざと交互に走らせる: a を1回 → b を1回 → a をもう1回。
     a の2回目が b の走行に影響されていれば、1回目と食い違う。 */
  if (mdl_infer_ctx(&a, in_a, sa) != 0) return 2;
  if (mdl_infer_ctx(&b, in_b, sb) != 0) return 2;
  emit(sa);
  emit(sb);
  if (mdl_infer_ctx(&a, in_a, sa) != 0) return 2;
  emit(sa);
  return 0;
}
'''

#: N枚をNスレッドで回して、そのまま結果を吐くドライバ。
POOL_DRIVER = '''#include "model.h"
#include "mdl_threads.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
#define N_ITEMS 8
#define N_CTX 4
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
           f'-I{work}', f'-I{RUNTIME_C / "include"}', *extra_flags,
           str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
           *[str(s) for s in extra_src],
           str(work / 'driver.c'), '-o', str(work / 'app')]
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        raise AssertionError(f'ビルド失敗:\n{build.stderr}')
    return work / 'app'


def _emit_model(bundle, tmp_path) -> object:
    files = generate_c(bundle, name='model', validate=False)
    work = tmp_path / 'gen'
    work.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    return work


class TestGeneratedApi:
    def test_header_declares_context_api(self, seg_doc):
        files = generate_c(build_bundle(seg_doc), name='model', validate=False)
        header = files['model.h']
        for sym in ('MDL_CTX_BYTES', 'mdl_ctx_init', 'mdl_infer_ctx',
                    'mdl_infer_ctx_v'):
            assert sym in header, f'{sym} がヘッダに無い'

    def test_weights_are_shared_not_per_context(self, seg_doc):
        """重み・命令列は `static const` のまま(全コンテキストで共有)。

        ここがコンテキストに移ると、スレッド数ぶんメモリを食う。
        """
        src = generate_c(build_bundle(seg_doc), name='model',
                         validate=False)['model.c']
        assert 'static const int32_t conv1_instr' in src or \
               'static const int32_t conv2_instr' in src, \
               '命令列が static const で置かれていない'
        assert 'ctx->work' in src, '作業領域がコンテキスト経由になっていない'

    def test_core_does_not_reference_threads(self):
        """コアはスレッドライブラリを一切参照しない(ベアメタル維持)。"""
        core = (RUNTIME_C / 'src' / 'mdl_kernels.c').read_text(encoding='utf-8')
        header = (RUNTIME_C / 'include' / 'mdl_kernels.h').read_text(
            encoding='utf-8')
        for bad in ('pthread', 'thread.h', '<threads.h>'):
            assert bad not in core, f'コアが {bad} を参照している'
            assert bad not in header, f'コアのヘッダが {bad} を参照している'


@pytest.mark.slow
class TestContextIndependence:
    def test_two_contexts_do_not_interfere(self, cc, tmp_path, seg_doc):
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(31)
        x = rng.integers(0, 2, size=(2, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        work = _emit_model(bundle, tmp_path)
        exe = _compile(cc, work, INTERLEAVE_DRIVER)
        proc = subprocess.run([str(exe)], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        got = read_u32(proc.stdout).astype(np.int64)
        n = want.shape[1]
        assert np.array_equal(got[:n], want[0]), 'ctx A の1回目が不一致'
        assert np.array_equal(got[n:2 * n], want[1]), 'ctx B が不一致'
        assert np.array_equal(got[2 * n:], want[0]), (
            'ctx A の2回目が1回目と違う(他のコンテキストに踏まれている)')

    def test_no_default_ctx_removes_all_static_ram(self, cc, tmp_path,
                                                   seg_doc):
        """`-DMDL_NO_DEFAULT_CTX` で生成物の static RAM が 0 になる。

        呼び出し側がメモリを用意する前提のMCU向け経路。
        """
        work = _emit_model(build_bundle(seg_doc), tmp_path)
        obj = work / 'model_noctx.o'
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
             '-DMDL_NO_DEFAULT_CTX', f'-I{work}',
             f'-I{RUNTIME_C / "include"}', '-c', str(work / 'model.c'),
             '-o', str(obj)],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        src = (work / 'model.c').read_text(encoding='utf-8')
        assert 'mdl_default_work' in src, '既定コンテキストの経路自体が無い'
        # 既定ctxを外した版では、モデル固有の可変 static が残らないこと
        nm = subprocess.run(['nm', str(obj)], capture_output=True, text=True)
        if nm.returncode != 0:
            pytest.skip('nm が使えない環境')
        # 小文字 b/d = ファイルローカルの .bss / .data。COFF(Windows)では節自体が
        # '.bss' / '.data' という b/d の疑似シンボルとして並ぶので、'.' 始まりは除く
        movable = [ln for ln in nm.stdout.splitlines()
                   if len(ln.split()) >= 2 and ln.split()[-2] in ('b', 'd')
                   and not ln.split()[-1].startswith('.')]
        assert not movable, (
            f'-DMDL_NO_DEFAULT_CTX でも可変staticが残っている: {movable}')


@pytest.mark.slow
class TestThreadPool:
    def test_pool_result_equals_sequential(self, cc, tmp_path, seg_doc):
        """8枚を4コンテキストで並列処理した結果が simulate と完全一致する。"""
        threads_c = RUNTIME_C / 'src' / 'mdl_threads.c'
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(1729)
        x = rng.integers(0, 2, size=(8, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        work = _emit_model(bundle, tmp_path)
        try:
            exe = _compile(cc, work, POOL_DRIVER, extra_src=[threads_c],
                           extra_flags=['-pthread'])
        except AssertionError as exc:
            if 'pthread' in str(exc):
                pytest.skip('pthread が使えない環境')
            raise
        proc = subprocess.run([str(exe)], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        got = read_u32(proc.stdout).astype(np.int64).reshape(want.shape)
        assert np.array_equal(got, want), (
            '並列実行の結果が逐次と違う'
            f'(不一致 {int((got != want).sum())}/{want.size} 要素)')
