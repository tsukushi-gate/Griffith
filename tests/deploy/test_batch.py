"""バッチ処理の等価性。

**バッチ化で最も起きやすい取り違えは「枚をまたいだ参照」**
(隣の枚のスクラッチを読む、順列表を枚ぶん引き直し忘れる、など)。
どれも「1枚だけなら正しい」ので単枚テストでは絶対に見つからない。

そこで固定するのは1点に尽きる:

    **batch=N の結果が、batch=1 を N 回回したものと完全に一致する**

基準は常に `ir/simulate.py`。ここでは C(逐次ループ)と Metal(実GPU)を
見る。CUDA は `scripts/verify_cuda_remote.py --batch` が同じことをやる。
"""

import subprocess

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.cudabackend import generate_cuda
from griffith_logic.deploy.memplan import check_no_overlap, plan_memory
from griffith_logic.deploy.metalbackend import generate_metal
from griffith_logic.deploy.plan import build_plan
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C
from .test_c_kernels import read_u32
from .test_metal import METAL_RUNTIME, _has_metal

N_BATCH = 5

C_BATCH_DRIVER = f'''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
#define N {N_BATCH}
static uint32_t work[MDL_CTX_BYTES / 4];
static uint8_t inputs[N * MDL_INPUT_WIDTH];
static int32_t scores[N * MDL_OUTPUT_WIDTH];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  int32_t i;
  if (fread(inputs, 1, sizeof(inputs), stdin) != sizeof(inputs)) return 1;
  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 3;
  if (mdl_infer_batch(&ctx, inputs, scores, N) != 0) return 2;
  for (i = 0; i < N * MDL_OUTPUT_WIDTH; i++) {{
    unsigned char b[4];
    uint32_t v = (uint32_t)scores[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }}
  return 0;
}}
'''

METAL_BATCH_DRIVER = f'''#include "mdl_metal.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
#define N {N_BATCH}
extern "C" {{
int mdl_metal_infer_batch(const unsigned char*, int*, int);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_metal_max_batch(void);
int mdl_model_init(void);
}}
int main(void) {{
  MDL_STDIO_BINARY();
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  unsigned char* buf = (unsigned char*)malloc((size_t)N * nin);
  int* scores = (int*)malloc((size_t)N * nout * sizeof(int));
  if (fread(buf, 1, (size_t)N * nin, stdin) != (size_t)N * nin) return 1;
  if (mdl_metal_max_batch() < N) return 5;
  if (mdl_model_init() != 0) return 3;
  if (mdl_metal_infer_batch(buf, scores, N) != 0) return 2;
  fwrite(scores, sizeof(int), (size_t)N * nout, stdout);
  return 0;
}}
'''


def _random_inputs(bundle, n, seed):
    graph = ir_from_dict(bundle['ir'])
    in_width = int(graph.inputs[0].width)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, in_width), dtype=np.uint8)
    return graph, x


class TestPlanScalesWithBatch:
    """プラン層がバッチをどう扱うか(バックエンド共通の土台)。"""

    def test_arena_scales_linearly(self, seg_doc):
        bundle = build_bundle(seg_doc)
        one = plan_memory(build_plan(bundle, batch=1))
        four = plan_memory(build_plan(bundle, batch=4))
        assert four.arena_words == 4 * one.arena_words, (
            'バッチ倍でアリーナが4倍になっていない')

    def test_no_overlap_holds_for_batched_plan(self, seg_doc):
        """バッチ化しても生存区間の重なりが出ない。"""
        plan = build_plan(build_bundle(seg_doc), batch=8)
        check_no_overlap(plan, plan_memory(plan))

    def test_op_sequence_is_identical(self, seg_doc):
        """op列そのものはバッチで変わらない(枚数は起動時の係数でしかない)。"""
        bundle = build_bundle(seg_doc)
        a = build_plan(bundle, batch=1)
        b = build_plan(bundle, batch=16)
        assert [(o.kind, o.inputs, o.output) for o in a.ops] == \
               [(o.kind, o.inputs, o.output) for o in b.ops]

    @pytest.mark.parametrize('bad', [0, -1])
    def test_invalid_batch_is_rejected(self, seg_doc, bad):
        with pytest.raises(ValueError, match='batch'):
            build_plan(build_bundle(seg_doc), batch=bad)


class TestGeneratedBatchSources:
    def test_gpu_sources_declare_max_batch(self, seg_doc):
        bundle = build_bundle(seg_doc)
        cuda = generate_cuda(bundle, name='m', batch=8)['m.cu']
        metal = generate_metal(bundle, name='m', batch=8)['m.mm']
        for src, who in ((cuda, 'CUDA'), (metal, 'Metal')):
            assert '#define MDL_MAX_BATCH 8' in src, f'{who}: 上限が出ていない'
            assert 'infer_batch' in src, f'{who}: バッチ入口が無い'

    def test_batch_one_keeps_single_image_entry(self, seg_doc):
        """batch=1 でも従来の単枚入口がそのまま生える(後方互換)。"""
        src = generate_cuda(build_bundle(seg_doc), name='m', batch=1)['m.cu']
        assert 'int mdl_cuda_infer(const unsigned char *input_bits' in src


@pytest.mark.slow
class TestCBatchMatchesSimulate:
    def test_c_batch_equals_simulate(self, cc, tmp_path, seg_doc):
        bundle = build_bundle(seg_doc)
        graph, x = _random_inputs(bundle, N_BATCH, seed=4242)
        want = simulate(graph, x).astype(np.int64)

        files = generate_c(bundle, name='model', validate=False)
        work = tmp_path / 'gen'
        work.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.c').write_text(C_BATCH_DRIVER, encoding='utf-8')
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
             str(work / 'driver.c'), '-o', str(work / 'app')],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        got = read_u32(proc.stdout).astype(np.int64).reshape(want.shape)
        assert np.array_equal(got, want), (
            f'C のバッチ実行が不一致({int((got != want).sum())}要素)')


@pytest.mark.slow
@pytest.mark.skipif(not _has_metal(), reason='macOS + Metal のある環境でのみ実行')
class TestMetalBatchMatchesSimulate:
    """**中核**: GPUでバッチ化した結果が単枚と1ビットも変わらない。"""

    def test_metal_batch_equals_simulate(self, tmp_path, seg_doc):
        bundle = build_bundle(seg_doc)
        graph, x = _random_inputs(bundle, N_BATCH, seed=99)
        want = simulate(graph, x).astype(np.int64)

        files = generate_metal(bundle, name='model_metal', batch=N_BATCH)
        work = tmp_path / 'gen'
        work.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.mm').write_text(METAL_BATCH_DRIVER, encoding='utf-8')
        build = subprocess.run(
            ['clang++', '-std=c++17', '-O2', '-fobjc-arc',
             f'-I{METAL_RUNTIME}', '-framework', 'Metal',
             '-framework', 'Foundation',
             str(work / 'model_metal.mm'),
             str(METAL_RUNTIME / 'mdl_metal.mm'),
             str(work / 'driver.mm'), '-o', str(work / 'app')],
            capture_output=True, text=True)
        if build.returncode != 0:
            pytest.skip(f'Metalのビルド不可: {build.stderr[:300]}')
        proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                              capture_output=True)
        if proc.returncode != 0:
            pytest.skip(
                f'Metal実行不可: {proc.stderr.decode(errors="replace")[:300]}')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        assert got.size == want.size, (
            f'出力サイズが違う: {got.size} != {want.size}')
        got = got.reshape(want.shape)
        assert np.array_equal(got, want), (
            'Metal のバッチ実行が simulate と不一致'
            f'({int((got != want).sum())}/{want.size} 要素)。'
            '枚をまたいだ参照(スクラッチ・順列表・記述子)を疑うこと')

    def test_partial_batch_matches(self, tmp_path, seg_doc):
        """確保した上限より**少ない**枚数でも正しい(起動スレッド数は n 依存)。"""
        bundle = build_bundle(seg_doc)
        graph, x = _random_inputs(bundle, N_BATCH, seed=7)
        want = simulate(graph, x).astype(np.int64)
        # 上限を大きめに取って、N_BATCH 枚だけ流す
        files = generate_metal(bundle, name='model_metal', batch=N_BATCH + 3)
        work = tmp_path / 'gen'
        work.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.mm').write_text(METAL_BATCH_DRIVER, encoding='utf-8')
        build = subprocess.run(
            ['clang++', '-std=c++17', '-O2', '-fobjc-arc',
             f'-I{METAL_RUNTIME}', '-framework', 'Metal',
             '-framework', 'Foundation',
             str(work / 'model_metal.mm'),
             str(METAL_RUNTIME / 'mdl_metal.mm'),
             str(work / 'driver.mm'), '-o', str(work / 'app')],
            capture_output=True, text=True)
        if build.returncode != 0:
            pytest.skip(f'Metalのビルド不可: {build.stderr[:300]}')
        proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                              capture_output=True)
        if proc.returncode != 0:
            pytest.skip('Metal実行不可')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        assert np.array_equal(got.reshape(want.shape), want), (
            '上限未満の枚数で結果が変わる')


@pytest.mark.slow
class TestDensePackedMatchesSequential:
    """dense のバッチ軸パック経路が逐次経路と**同じ値**を出す。

    1ワード=32画像でビットを詰め替えるので、詰め方や添字の取り違えは
    「一部の画像だけ壊れる」形で出る。逐次経路との全要素比較で押さえる。
    """

    DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
#include <string.h>
#define N 40
static uint32_t work[MDL_CTX_BYTES / 4];
static uint32_t dwork[MDL_DENSE_WORK_BYTES / 4];
static uint8_t in_[N * MDL_INPUT_WIDTH];
static int32_t s1[N * MDL_OUTPUT_WIDTH], s2[N * MDL_OUTPUT_WIDTH];
int main(void) {
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  int32_t i;
  if (fread(in_, 1, sizeof(in_), stdin) != sizeof(in_)) return 1;
  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 3;
  if (mdl_infer_batch(&ctx, in_, s1, N) != 0) return 2;
  if (mdl_infer_batch_packed(dwork, sizeof(dwork), in_, s2, N) != 0) return 4;
  if (memcmp(s1, s2, sizeof(s1)) != 0) return 5;
  for (i = 0; i < N * MDL_OUTPUT_WIDTH; i++) {
    unsigned char b[4];
    uint32_t v = (uint32_t)s2[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }
  return 0;
}
'''

    def test_packed_equals_sequential_and_simulate(self, cc, tmp_path,
                                                   dense_doc):
        bundle = build_bundle(dense_doc)
        graph, x = _random_inputs(bundle, 40, seed=2024)
        want = simulate(graph, x).astype(np.int64)

        files = generate_c(bundle, name='model', validate=False,
                           dense_batch=64)
        assert 'mdl_infer_batch_packed' in files['model.h'], (
            'dense専用モデルなのにパック経路が生えていない')
        work = tmp_path / 'gen'
        work.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.c').write_text(self.DRIVER, encoding='utf-8')
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
             str(work / 'driver.c'), '-o', str(work / 'app')],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode != 5, 'パック経路と逐次経路で値が違う'
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        got = read_u32(proc.stdout).astype(np.int64).reshape(want.shape)
        assert np.array_equal(got, want), (
            f'パック経路が simulate と不一致({int((got != want).sum())}要素)')

    def _run_packed(self, cc, tmp_path, bundle, x):
        files = generate_c(bundle, name='model', validate=False,
                           dense_batch=64)
        assert 'mdl_infer_batch_packed' in files['model.h']
        work = tmp_path / 'gen'
        work.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.c').write_text(self.DRIVER, encoding='utf-8')
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
             str(work / 'driver.c'), '-o', str(work / 'app')],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode != 5, 'パック経路と逐次経路で値が違う'
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        return files, read_u32(proc.stdout).astype(np.int64)

    def test_flat_packed_equals_sequential_and_simulate(self, cc, tmp_path,
                                                        dense_doc):
        """lut_flat(平坦化回路)もバッチ軸パック経路で動き、値が一致する。"""
        from griffith_logic.deploy.flatten_dense import flatten_dense_doc

        flat_doc, _ = flatten_dense_doc(dense_doc)
        bundle = build_bundle(flat_doc)
        graph, x = _random_inputs(bundle, 40, seed=77)
        want = simulate(ir_from_dict(build_bundle(dense_doc)['ir']),
                        x).astype(np.int64)
        files, got = self._run_packed(cc, tmp_path, bundle, x)
        assert 'mdl_lut_flat_packed_u16' in files['model.c']
        assert np.array_equal(got.reshape(want.shape), want), (
            f'flat パック経路が元IRの simulate と不一致'
            f'({int((got.reshape(want.shape) != want).sum())}要素)')

    def test_flat_packed_u32_index_path(self, cc, tmp_path):
        """信号数 > 32768(添字 uint32_t)でも flat パック経路の値は変わらない。"""
        from griffith_logic.deploy.flatten_dense import flatten_dense_doc
        from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
        from griffith_logic.ir.serialize import ir_to_dict

        rng = np.random.default_rng(78)
        in_w, width = 33000, 64
        idx = rng.integers(0, in_w, size=(2, width))
        tts = rng.integers(1, 15, size=width)
        g = IRGraph(
            metadata={'name': 'u32'},
            inputs=[IRPort(id='x', width=in_w, encoding='binary_threshold_0.5')],
            outputs=[IRPort(id='y', width=4, semantics='class_scores')],
            nodes=[IRNode(id='n1', type='lut_layer', params={
                       'in_width': in_w, 'out_width': width, 'n_inputs': 2,
                       'indices': idx,
                       'truth_tables': [f'0x{int(t):x}' for t in tts]}),
                   IRNode(id='n2', type='group_sum',
                          params={'in_width': width, 'k': 4, 'tau': 1.0})],
            edges=[IREdge(src='x', dst='n1'), IREdge(src='n1', dst='n2'),
                   IREdge(src='n2', dst='y')])
        flat_doc, _ = flatten_dense_doc(
            {'ir': ir_to_dict(g), 'preproc': {}, 'labels': []})
        bundle = build_bundle(flat_doc)
        x = rng.integers(0, 2, size=(40, in_w), dtype=np.uint8)
        want = simulate(g, x).astype(np.int64)
        files, got = self._run_packed(cc, tmp_path, bundle, x)
        assert 'mdl_lut_flat_packed_u32' in files['model.c']
        assert np.array_equal(got.reshape(want.shape), want)

    def test_conv_model_has_no_packed_entry(self, seg_doc):
        """conv を含むモデルには生やさない(空間軸で既にパック済み)。"""
        files = generate_c(build_bundle(seg_doc), name='model',
                           validate=False, dense_batch=64)
        assert 'mdl_infer_batch_packed' not in files['model.h']
