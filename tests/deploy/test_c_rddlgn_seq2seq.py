"""RDDLGN seq2seq(教師強制、エンコーダ+デコーダ全体)のCランタイム端到端検証。

tests/deploy/test_c_rddlgn.py(エンコーダ+分類ヘッド)の続き。
bundle → generate_c → cc ビルド → 実行 が ir/simulate と整数厳密に一致する
ことを確かめる(deploykit-integrationスキル⑥の検証の鎖)。

decoderのP+M+GroupSumを1つのrecurrent bodyへ畳んだことで、recurrentの
出力そのものがグラフの最終出力(KIND_SCORES = int32スコア行)になる
固有のケースを踏む。deploy/plan.py::_plan_recurrentが出力値種別を
無条件にKIND_FLATへ決め打ちしていた点、deploy/cbackend.py::
emit_recurrent_execがstep関数内から呼び出し側の`scores`配列へ直接書けな
かった点(step関数のスコープには無い識別子だったため)をこのテストで
固定する。
"""

import numpy as np
import pytest

from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.verify import (
    build_c_model,
    run_c_model,
    verify_c_matches_simulate,
)
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C, build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc


class TestVerifyCMatchesSimulate:
    def test_random_samples_match_exactly(self, cc, tmp_path,
                                          rddlgn_seq2seq_tiny_bundle):
        result = verify_c_matches_simulate(
            rddlgn_seq2seq_tiny_bundle, n_samples=5, seed=0, cc=cc,
            work_dir=tmp_path / 'verify')
        assert result['max_abs_diff'] == 0
        assert result['path'] == 'bits'

    def test_extreme_inputs_match(self, cc, tmp_path,
                                  rddlgn_seq2seq_tiny_bundle):
        """全0/全1の埋め込みビット列でも一致する(端数ビットの扱い含む)。"""
        graph = ir_from_dict(rddlgn_seq2seq_tiny_bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        exe = build_c_model(rddlgn_seq2seq_tiny_bundle, tmp_path / 'gen', cc=cc)
        for label, x in (('zeros', np.zeros(in_width, dtype=np.uint8)),
                         ('ones', np.ones(in_width, dtype=np.uint8))):
            want = simulate(graph, x[None, :])[0].astype(np.int64)
            got = run_c_model(exe, x.tobytes(), n_out)
            assert np.array_equal(got, want), label


class TestMultipleModelConfigurations:
    """モデルの重みが違っても一致する(1つの固定シードで偶然揃った、を排除)。"""

    @pytest.mark.parametrize('seed', [1, 2])
    def test_different_seeds_match(self, cc, tmp_path, seed):
        model, cfg = build_rddlgn_tiny_seq2seq(seed=seed)
        doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
        from griffith_logic.deploy.bundle import build_bundle
        bundle = build_bundle(doc)
        result = verify_c_matches_simulate(
            bundle, n_samples=3, seed=seed, cc=cc,
            work_dir=tmp_path / f'verify{seed}')
        assert result['max_abs_diff'] == 0

    def test_init_mode_one_matches(self, cc, tmp_path):
        """init_mode='one'(全1初期状態)でも一致する(エンコーダK-layerと
        デコーダP-layerの両方に効く)。"""
        import torch

        from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig

        torch.manual_seed(0)
        cfg = RDDLGNConfig.tiny()
        cfg.init_mode = 'one'
        model = RDDLGN(cfg)
        model.eval()
        doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
        from griffith_logic.deploy.bundle import build_bundle
        bundle = build_bundle(doc)
        result = verify_c_matches_simulate(
            bundle, n_samples=3, seed=0, cc=cc, work_dir=tmp_path / 'verify_one')
        assert result['max_abs_diff'] == 0


class TestGeneratedSourceProperties:
    def test_no_heap_allocation(self, rddlgn_seq2seq_tiny_bundle):
        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False)
        source = files['model.c']
        for bad in ('malloc', 'calloc', 'realloc', 'free('):
            assert bad not in source, f'生成コードに {bad} がある'

    def test_no_pthread_dependency(self, rddlgn_seq2seq_tiny_bundle):
        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False)
        for text in files.values():
            assert 'pthread' not in text

    def test_step_functions_are_emitted(self, rddlgn_seq2seq_tiny_bundle):
        """4つのrecurrent(enc_n_rec/enc_k_rec/dec_l_rec/dec_pm_rec)全部が
        mdl_step_<id>へ切り出されている。"""
        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False)
        source = files['model.c']
        for node_id in ('enc_n_rec', 'enc_k_rec', 'dec_l_rec', 'dec_pm_rec'):
            assert f'static void mdl_step_{node_id}(' in source, node_id

    def test_pm_step_function_takes_scores_param(self,
                                                  rddlgn_seq2seq_tiny_bundle):
        """dec_pm_recの出力はグラフの最終出力そのもの(body内でgroup_sumを
        経由する固有のケース)なので、step関数は呼び出し側の
        `scores`配列を直接受け取る特別な引数を持つ。"""
        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False)
        source = files['model.c']
        assert 'static void mdl_step_dec_pm_rec(mdl_ctx *ctx, int32_t t, ' \
               'int32_t *scores)' in source

    def test_compiles_with_strict_warnings(self, cc, tmp_path,
                                           rddlgn_seq2seq_tiny_bundle):
        import subprocess

        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False)
        work = tmp_path / 'strict'
        work.mkdir()
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
              f'-I{work}', f'-I{RUNTIME_C / "include"}',
              '-c', str(work / 'model.c'), '-o', str(work / 'model.o')]
        build = subprocess.run(cmd, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr


@pytest.mark.slow
class TestBatchMatchesRepeatedSingle:
    """batch=N の結果が batch=1 を N 回回したものと完全に一致する
    (deploykit-integrationスキル: バッチ追加時に固定すべき性質)。"""

    N = 3

    DRIVER_TEMPLATE = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <string.h>
#define N {n}
static uint32_t work[MDL_CTX_BYTES / 4];
static uint8_t inputs[N * MDL_INPUT_WIDTH];
static int32_t scores_batch[N * MDL_OUTPUT_WIDTH];
static int32_t scores_seq[N * MDL_OUTPUT_WIDTH];
int main(void) {{
  MDL_STDIO_BINARY();
  mdl_ctx ctx;
  int32_t i;
  if (fread(inputs, 1, sizeof(inputs), stdin) != sizeof(inputs)) return 1;
  if (mdl_ctx_init(&ctx, work, sizeof(work)) != 0) return 3;
  if (mdl_infer_batch(&ctx, inputs, scores_batch, N) != 0) return 2;
  for (i = 0; i < N; i++) {{
    if (mdl_infer_ctx(&ctx, inputs + (size_t)i * MDL_INPUT_WIDTH,
                      scores_seq + (size_t)i * MDL_OUTPUT_WIDTH) != 0)
      return 4;
  }}
  if (memcmp(scores_batch, scores_seq, sizeof(scores_batch)) != 0) return 5;
  for (i = 0; i < N * MDL_OUTPUT_WIDTH; i++) {{
    unsigned char b[4];
    uint32_t v = (uint32_t)scores_batch[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }}
  return 0;
}}
'''

    def test_batch_equals_sequential_and_simulate(
            self, cc, tmp_path, rddlgn_seq2seq_tiny_bundle):
        import subprocess

        from .test_c_kernels import read_u32

        graph = ir_from_dict(rddlgn_seq2seq_tiny_bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(2026)
        x = rng.integers(0, 2, size=(self.N, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        files = generate_c(rddlgn_seq2seq_tiny_bundle, name='model',
                           validate=False)
        work = tmp_path / 'gen'
        work.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (work / name).write_text(text, encoding='utf-8')
        (work / 'driver.c').write_text(
            self.DRIVER_TEMPLATE.format(n=self.N), encoding='utf-8')
        build = subprocess.run(
            [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
             f'-I{work}', f'-I{RUNTIME_C / "include"}',
             str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
             str(work / 'driver.c'), '-o', str(work / 'app')],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode != 5, (
            'batch=N が batch=1 を N 回回したものと一致しない'
            '(前回tickの状態が持ち越されている等の疑い)')
        assert proc.returncode == 0, proc.stderr.decode(errors='replace')
        got = read_u32(proc.stdout).astype(np.int64).reshape(want.shape)
        assert np.array_equal(got, want), (
            f'Cのバッチ実行が simulate と不一致({int((got != want).sum())}要素)')
