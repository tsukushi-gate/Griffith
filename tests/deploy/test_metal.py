"""Metal(Apple GPU)バックエンドの検証。

macOS + Apple GPU のある環境でのみ動く(それ以外は自動skip)。
CUDA と同じく、基準は常に `ir/simulate.py`。

シェーダは実行時にソースからコンパイルするので、Xcode の metal
ツールチェーン(オフラインコンパイラ)は不要。Metal.framework さえあれば動く。
"""

import platform
import shutil
import subprocess

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.metalbackend import build_shader_source, generate_metal
from griffith_logic.deploy.plan import UnsupportedNode
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C

RUNTIME = RUNTIME_C.parent
METAL_RUNTIME = RUNTIME / 'metal'

DRIVER = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
extern "C" {
int mdl_metal_infer(const unsigned char*, int*);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}
int main(void) {
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  unsigned char* buf = (unsigned char*)malloc(nin);
  int* scores = (int*)malloc(nout * sizeof(int));
  if (fread(buf, 1, nin, stdin) != (size_t)nin) return 1;
  if (mdl_model_init() != 0) return 3;
  if (mdl_metal_infer(buf, scores) != 0) return 2;
  fwrite(scores, sizeof(int), nout, stdout);
  return 0;
}
'''


def _has_metal() -> bool:
    if platform.system() != 'Darwin':
        return False
    return shutil.which('clang++') is not None and METAL_RUNTIME.is_dir()


pytestmark = pytest.mark.skipif(
    not _has_metal(), reason='macOS + Metal のある環境でのみ実行')


class TestShaderSource:
    def test_source_contains_shared_kernels_and_entries(self):
        """MSLソースは共有カーネルとMetal用入口の連結でできている。"""
        src = build_shader_source()
        assert '#include <metal_stdlib>' in src
        assert 'mdl_g_gate2' in src, '共有カーネルが入っていない'
        assert 'kernel void mdl_k_conv_exec' in src, 'Metal入口が入っていない'
        assert '#ifndef MDL_GPU_KERNELS_H' not in src, (
            'インクルードガードが残っている(MSLでは連結するので不要)')

    def test_no_64bit_integers(self):
        """Metal は64bit整数を持たないので、共有カーネルにも出してはいけない。"""
        src = build_shader_source()
        for bad in ('long long', 'int64_t', 'uint64_t'):
            assert bad not in src, f'MSLに {bad} がある(Metalは64bit整数非対応)'

    def test_no_reserved_identifiers(self):
        """`half` は MSL の予約型名なので変数名に使えない。"""
        src = build_shader_source()
        assert ' half ' not in src.replace('half_', '') or 'half2' in src, \
            'MSL予約語 half を識別子に使っている恐れ'


@pytest.mark.slow
class TestMetalMatchesSimulate:
    @pytest.mark.parametrize('shape', [
        pytest.param((6, 16, 16), id='seg_like'),
    ])
    def test_generated_metal_matches_simulate(self, tmp_path, seg_doc, shape):
        bundle = build_bundle(seg_doc)
        files = generate_metal(bundle, name='model_metal')
        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.mm').write_text(DRIVER, encoding='utf-8')

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

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.int64)

        proc = subprocess.run([str(work / 'app')], input=x[0].tobytes(),
                              capture_output=True)
        if proc.returncode != 0:
            pytest.skip(
                f'Metal実行不可(GPU無し等): '
                f'{proc.stderr.decode(errors="replace")[:300]}')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        assert got.size == want.size
        assert np.array_equal(got, want), (
            f'Metal ⇔ simulate 不一致: {int((got != want).sum())}/{want.size}')


@pytest.mark.slow
class TestMetalRddlgnMatchesSimulate:
    """RDDLGNClassifier(tiny)のrecurrentノードがMetalでもsimulateと一致する。CUDAはこのマシンに無いので検証できないが、GpuBackend共通の
    emit_recurrent_exec/emit_flat_concat/emit_state_initはCUDA/Metalで
    同一コードなので、Metalで通ればCUDA側の論理も同じ経路を通る。
    """

    def test_generated_metal_matches_simulate(self, tmp_path, rddlgn_tiny_bundle):
        bundle = rddlgn_tiny_bundle
        files = generate_metal(bundle, name='model_metal')
        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.mm').write_text(DRIVER, encoding='utf-8')

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

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(0)
        proc_ok = 0
        for i in range(4):
            x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
            want = simulate(graph, x)[0].astype(np.int64)
            proc = subprocess.run([str(work / 'app')], input=x[0].tobytes(),
                                  capture_output=True)
            if proc.returncode != 0:
                pytest.skip(
                    f'Metal実行不可(GPU無し等): '
                    f'{proc.stderr.decode(errors="replace")[:300]}')
            got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
            assert got.size == want.size
            assert np.array_equal(got, want), (
                f'サンプル{i}: Metal ⇔ simulate 不一致(recurrentノード): '
                f'{int((got != want).sum())}/{want.size}')
            proc_ok += 1
        assert proc_ok == 4

    def test_two_consecutive_calls_do_not_leak_state(self, tmp_path,
                                                      rddlgn_tiny_bundle):
        """1プロセス内で2回続けて推論しても、前回のstate/ext残留物が
        混ざらない(state_initが呼び出しごとに初期状態へ戻すこと、
        stateバッファ/extバッファのメモリ計画が使い回されても壊れない
        ことの確認)。tests/deploy/test_rddlgn_memory_plan.py のC版と対。
        """
        bundle = rddlgn_tiny_bundle
        files = generate_metal(bundle, name='model_metal')
        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        driver = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
extern "C" {
int mdl_metal_infer(const unsigned char*, int*);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}
int main(void) {
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  unsigned char* x0 = (unsigned char*)malloc(nin);
  unsigned char* x1 = (unsigned char*)malloc(nin);
  int* scores = (int*)malloc(nout * sizeof(int));
  if (fread(x0, 1, nin, stdin) != (size_t)nin) return 1;
  if (fread(x1, 1, nin, stdin) != (size_t)nin) return 1;
  if (mdl_model_init() != 0) return 3;
  if (mdl_metal_infer(x0, scores) != 0) return 2;
  fwrite(scores, sizeof(int), nout, stdout);
  if (mdl_metal_infer(x1, scores) != 0) return 2;
  fwrite(scores, sizeof(int), nout, stdout);
  return 0;
}
'''
        (work / 'driver.mm').write_text(driver, encoding='utf-8')

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

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        rng = np.random.default_rng(7)
        x0 = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        x1 = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        want0 = simulate(graph, x0)[0].astype(np.int64)
        want1 = simulate(graph, x1)[0].astype(np.int64)

        proc = subprocess.run([str(work / 'app')],
                              input=x0[0].tobytes() + x1[0].tobytes(),
                              capture_output=True)
        if proc.returncode != 0:
            pytest.skip(
                f'Metal実行不可(GPU無し等): '
                f'{proc.stderr.decode(errors="replace")[:300]}')
        out = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        assert out.size == 2 * n_out
        got0, got1 = out[:n_out], out[n_out:]
        assert np.array_equal(got0, want0), '1回目の呼び出しが不一致'
        assert np.array_equal(got1, want1), (
            '2回目の呼び出しが不一致(前回のstate/extが持ち越された疑い)')


@pytest.mark.slow
class TestMetalRddlgnSeq2seqMatchesSimulate:
    """RDDLGN seq2seq(教師強制、tiny)がMetalでもsimulateとビット厳密一致する。TestMetalRddlgnMatchesSimulate(分類器)の続きで、
    デコーダのrecurrent body(P+M+GroupSum)がKIND_SCORES(int32=4バイト/要素)
    のstacked出力を持つケースを踏む。plan.py::_plan_recurrentとcbackendは
    要素サイズ対応済みだったが、gpubackend.py::emit_recurrent_exec
    はstacked出力コピーを「1バイト=1要素」決め打ちのままだった
    (で `GpuBackend.elem_bytes` 経由に修正)。この修正が無いと
    デコーダの出力が4倍ずれたオフセットへ書かれ、simulateと一致しなくなる。
    """

    def test_generated_metal_matches_simulate(self, tmp_path,
                                               rddlgn_seq2seq_tiny_bundle):
        bundle = rddlgn_seq2seq_tiny_bundle
        files = generate_metal(bundle, name='model_metal')
        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.mm').write_text(DRIVER, encoding='utf-8')

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

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(0)
        proc_ok = 0
        for i in range(4):
            x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
            want = simulate(graph, x)[0].astype(np.int64)
            proc = subprocess.run([str(work / 'app')], input=x[0].tobytes(),
                                  capture_output=True)
            if proc.returncode != 0:
                pytest.skip(
                    f'Metal実行不可(GPU無し等): '
                    f'{proc.stderr.decode(errors="replace")[:300]}')
            got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
            assert got.size == want.size
            assert np.array_equal(got, want), (
                f'サンプル{i}: Metal ⇔ simulate 不一致(seq2seq recurrent、'
                f'stacked KIND_SCORES出力): '
                f'{int((got != want).sum())}/{want.size}')
            proc_ok += 1
        assert proc_ok == 4

    def test_two_consecutive_calls_do_not_leak_state(
            self, tmp_path, rddlgn_seq2seq_tiny_bundle):
        """1プロセス内で2回続けて推論しても、前回のstate/ext残留物が
        混ざらない(TestMetalRddlgnMatchesSimulateの同名テストのseq2seq版)。
        """
        bundle = rddlgn_seq2seq_tiny_bundle
        files = generate_metal(bundle, name='model_metal')
        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        driver = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
extern "C" {
int mdl_metal_infer(const unsigned char*, int*);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}
int main(void) {
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  unsigned char* x0 = (unsigned char*)malloc(nin);
  unsigned char* x1 = (unsigned char*)malloc(nin);
  int* scores = (int*)malloc(nout * sizeof(int));
  if (fread(x0, 1, nin, stdin) != (size_t)nin) return 1;
  if (fread(x1, 1, nin, stdin) != (size_t)nin) return 1;
  if (mdl_model_init() != 0) return 3;
  if (mdl_metal_infer(x0, scores) != 0) return 2;
  fwrite(scores, sizeof(int), nout, stdout);
  if (mdl_metal_infer(x1, scores) != 0) return 2;
  fwrite(scores, sizeof(int), nout, stdout);
  return 0;
}
'''
        (work / 'driver.mm').write_text(driver, encoding='utf-8')

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

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        rng = np.random.default_rng(7)
        x0 = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        x1 = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        want0 = simulate(graph, x0)[0].astype(np.int64)
        want1 = simulate(graph, x1)[0].astype(np.int64)

        proc = subprocess.run([str(work / 'app')],
                              input=x0[0].tobytes() + x1[0].tobytes(),
                              capture_output=True)
        if proc.returncode != 0:
            pytest.skip(
                f'Metal実行不可(GPU無し等): '
                f'{proc.stderr.decode(errors="replace")[:300]}')
        out = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        assert out.size == 2 * n_out
        got0, got1 = out[:n_out], out[n_out:]
        assert np.array_equal(got0, want0), '1回目の呼び出しが不一致'
        assert np.array_equal(got1, want1), (
            '2回目の呼び出しが不一致(前回のstate/extが持ち越された疑い)')


class TestMetalRejectsK26AccumulateGroup:
    """退化ゲート除去のaccumulate-group(alias化された
    PlanValue)はCUDA/Metal未対応であることを明示的に拒否する
    (`gpubackend.py::GpuBackend.__init__`)。

    `deploy/plan.py::_plan_accumulate_concat`が作るalias(親バッファ+
    オフセット)は`cbackend.py::_declare_buffers`だけが解釈する。GPU側で
    このガードが無いと、`declare_buffers`の「念のため」フォールバックが
    aliasを**独立バッファ**として静かに確保してしまい、後続層/境界層が
    読む親バッファには反映されない黙った誤りになる(perf-huntスキルの
    「同じバグでもバックエンドで出方が違う」を先回りする)。
    """

    def test_generate_metal_raises_on_optimize_degenerate_bundle(self):
        from griffith_logic.export.rddlgn_ir import export_rddlgn_ir
        from griffith_logic.ir.serialize import ir_to_dict
        from griffith_logic.ir.validate import validate

        from .conftest import build_rddlgn_tiny_seq2seq

        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        graph = export_rddlgn_ir(model, cfg.seq_len, cfg.seq_len,
                                 optimize_degenerate=True)
        assert validate(graph) == []
        doc = {'ir': ir_to_dict(graph), 'preproc': {}, 'labels': []}
        bundle = build_bundle(doc)

        with pytest.raises(UnsupportedNode, match='退化ゲート除去'):
            generate_metal(bundle, name='model_metal')


#: 同一プロセスで2回連続、別入力で推論するドライバ。
#: tests/deploy/test_c_context.py::test_two_contexts_do_not_interfere と
#: 同じ目的だが、Metalランタイムは(Cのmdl_ctxと違い)複数コンテキストを
#: 持たず1個のグローバル静的状態(mdl_metal.mmの`g`)を使い回す設計なので、
#: 対応する固定点は「並行コンテキストの独立性」ではなく「**逐次2推論の
#: 独立性**(1回目のGPUバッファの残骸が2回目の結果を汚さないか)」になる。
CONSECUTIVE_DRIVER = '''#include "mdl_metal.h"
#include <stdio.h>
#include <stdlib.h>
extern "C" {
int mdl_metal_infer(const unsigned char*, int*);
int mdl_metal_input_width(void);
int mdl_metal_output_width(void);
int mdl_model_init(void);
}
int main(void) {
  int nin = mdl_metal_input_width(), nout = mdl_metal_output_width();
  unsigned char* a = (unsigned char*)malloc(nin);
  unsigned char* b = (unsigned char*)malloc(nin);
  int* sa = (int*)malloc(nout * sizeof(int));
  int* sb = (int*)malloc(nout * sizeof(int));
  if (fread(a, 1, nin, stdin) != (size_t)nin) return 1;
  if (fread(b, 1, nin, stdin) != (size_t)nin) return 1;
  if (mdl_model_init() != 0) return 3;
  /* 1回目 → 2回目 → もう一度1回目と同じ入力、の3推論。3回目が1回目と
     食い違えば「2回目の実行が状態を汚した」ことになる。 */
  if (mdl_metal_infer(a, sa) != 0) return 2;
  fwrite(sa, sizeof(int), nout, stdout);
  if (mdl_metal_infer(b, sb) != 0) return 2;
  fwrite(sb, sizeof(int), nout, stdout);
  if (mdl_metal_infer(a, sa) != 0) return 2;
  fwrite(sa, sizeof(int), nout, stdout);
  return 0;
}
'''


@pytest.mark.slow
class TestConsecutiveInference:
    def test_two_consecutive_inferences_are_correct(self, tmp_path, seg_doc):
        """同一プロセス内で連続実行しても、それぞれ正しい(状態が漏れない)。

        1回目→2回目→1回目と同じ入力でもう一度、の3推論を1プロセスで流し、
        (1)それぞれがsimulateと一致、かつ(2)1回目と3回目(同じ入力)が
        一致することを見る。GPU側は静的に確保したスクラッチ/出力バッファを
        推論のたびに使い回す設計なので、前回の中間結果の残骸を消し忘れる
        バグがあれば入力次第でここに出る。
        """
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(31)
        x = rng.integers(0, 2, size=(2, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        files = generate_metal(bundle, name='model_metal')
        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.mm').write_text(CONSECUTIVE_DRIVER, encoding='utf-8')

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
                f'Metal実行不可(GPU無し等): '
                f'{proc.stderr.decode(errors="replace")[:300]}')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        n = want.shape[1]
        assert np.array_equal(got[:n], want[0]), '1回目の推論が不一致'
        assert np.array_equal(got[n:2 * n], want[1]), '2回目の推論が不一致'
        assert np.array_equal(got[2 * n:], want[0]), (
            '3回目(1回目と同じ入力)が1回目と食い違う'
            '(直前の推論の状態が漏れている)')


class TestMetallibFallback:
    """事前コンパイル済み .metallib のパスと、無いときのフォールバック。

    .metallib 自体は Xcode の metal ツールチェーンでしか作れず、この環境には
    未導入。よってここで固定できるのは「無い/読めないときに黙って壊れず
    ソースコンパイルへ落ちる」ことまで。**.metallib を実際に読む経路は
    未検証**であることを明示しておく。
    """

    def test_api_exposes_metallib_entry(self):
        header = (METAL_RUNTIME / 'mdl_metal.h').read_text(encoding='utf-8')
        assert 'mdl_metal_init_ex' in header
        assert 'metallib' in header

    @pytest.mark.slow
    def test_missing_metallib_falls_back_to_source(self, tmp_path, seg_doc):
        """存在しない .metallib を渡してもソースからコンパイルして動く。"""
        bundle = build_bundle(seg_doc)
        files = generate_metal(bundle, name='model_metal')
        work = tmp_path / 'gen'
        work.mkdir(parents=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        # init を .metallib 指定(存在しないパス)へ差し替えたドライバ
        (work / 'driver.mm').write_text(DRIVER.replace(
            'if (mdl_model_init() != 0) return 3;',
            'if (mdl_metal_init_ex("/nonexistent/model.metallib", NULL) == 0) '
            '  return 4;  /* ソースが無ければ失敗すべき */\n'
            '  if (mdl_model_init() != 0) return 3;'), encoding='utf-8')
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

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(1)
        x = rng.integers(0, 2, size=(1, in_width), dtype=np.uint8)
        want = simulate(graph, x)[0].astype(np.int64)
        proc = subprocess.run([str(work / 'app')], input=x[0].tobytes(),
                              capture_output=True)
        if proc.returncode == 4:
            pytest.fail('.metallib が無いのに成功を返した')
        if proc.returncode != 0:
            pytest.skip('Metal実行不可(GPU無し等)')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        assert np.array_equal(got, want), 'フォールバック経路で結果が変わった'
