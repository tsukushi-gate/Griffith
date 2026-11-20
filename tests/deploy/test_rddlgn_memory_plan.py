"""RDDLGNバンドルのメモリ計画がrecurrentの状態バッファでも壊れないこと。

`deploy/memplan.py`の設計コメントどおり、ノード型の知識を一切使わない
op列ベースの生存区間解析なので、recurrentを足しても無改修で正しく動く
はず、という主張を直接検査する。加えて、state_bufs/ext_bufsは
「steps回のループの中で読み書きされ続ける」という他のop(scratch含む)には
無い性質を持つので、tests/deploy/test_memory_plan.py と同じ
「同一プロセスで2回続けて別入力を推論しても両方simulateと一致する」
(状態の意図しない持ち越しが無い)テストも足す。
"""

import subprocess

import numpy as np
import pytest

from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.memplan import check_no_overlap, liveness, plan_memory
from griffith_logic.deploy.plan import build_plan
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C
from .test_c_kernels import read_u32

TWICE_DRIVER = '''#include "model.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
static void emit(const int32_t *scores) {
  int32_t i;
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) {
    unsigned char b[4];
    uint32_t v = (uint32_t)scores[i];
    b[0]=(unsigned char)(v&0xffu); b[1]=(unsigned char)((v>>8)&0xffu);
    b[2]=(unsigned char)((v>>16)&0xffu); b[3]=(unsigned char)((v>>24)&0xffu);
    fwrite(b, 1, 4, stdout);
  }
}
int main(void) {
  MDL_STDIO_BINARY();
  static uint8_t x0[MDL_INPUT_WIDTH];
  static uint8_t x1[MDL_INPUT_WIDTH];
  static int32_t scores[MDL_OUTPUT_WIDTH];
  if (fread(x0, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) return 1;
  if (fread(x1, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) return 1;
  if (mdl_infer(x0, scores) != 0) return 2;
  emit(scores);
  if (mdl_infer(x1, scores) != 0) return 3;
  emit(scores);
  return 0;
}
'''


def _build_and_run_twice(cc, tmp_path, bundle, x0: np.ndarray, x1: np.ndarray):
    files = generate_c(bundle, name='model', validate=False)
    work = tmp_path / 'gen'
    work.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.c').write_text(TWICE_DRIVER, encoding='utf-8')
    exe = work / 'app'
    cmd = [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
          f'-I{work}', f'-I{RUNTIME_C / "include"}',
          str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
          str(work / 'driver.c'), '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    payload = x0.astype(np.uint8).tobytes() + x1.astype(np.uint8).tobytes()
    proc = subprocess.run([str(exe)], input=payload, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    out = read_u32(proc.stdout).astype(np.int64)
    half = out.size // 2
    return out[:half], out[half:]


class TestCheckNoOverlap:
    def test_state_and_ext_buffers_do_not_overlap_with_live_values(
            self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        mem = plan_memory(plan, exclude={plan.input_value, plan.output_value})
        check_no_overlap(plan, mem)   # 重なっていればAssertionError

    def test_sharing_actually_happens(self, rddlgn_tiny_bundle):
        """バッファ共有が起きている(全値バラバラでは意味が無い)。"""
        plan = build_plan(rddlgn_tiny_bundle)
        mem = plan_memory(plan, exclude={plan.input_value, plan.output_value})
        assert mem.arena_words < mem.naive_words, (
            'recurrentの内部バッファが1本も共有されていない'
            '(生存区間解析が効いていない可能性)'
        )

    def test_state_buffer_liveness_spans_the_whole_recurrent_block(
            self, rddlgn_tiny_bundle):
        """state_initからrecurrent_execまでstateバッファが生き続ける。"""
        plan = build_plan(rddlgn_tiny_bundle)
        first, last = liveness(plan)
        state_init_ops = [op for op in plan.ops if op.kind == 'state_init']
        assert state_init_ops, 'state_initが無い'
        for op in state_init_ops:
            recurrent_idx = next(
                i for i, o in enumerate(plan.ops)
                if o.kind == 'recurrent_exec' and o.node_id == op.node_id
            )
            state_idx = plan.ops.index(op)
            assert first[op.output] == state_idx
            assert last[op.output] >= recurrent_idx


class TestStackedOutputLivenessSpansTheWholeRecurrentBlock:
    """`output_mode='stacked'`な出力の生存区間が、recurrent_exec
    opの実行時点だけでなくbody(scope)の先頭まで遡っていること。

    GPUバックエンドはCと異なり`plan.output_value`もプールする
    (`gpubackend.GpuBackend.__init__`は`plan_memory(plan)`にexcludeを
    渡さない)。stacked出力(RDDLGN seq2seqの`dec_pm_rec`がこれに該当。
    body内でgroup_sumを経由しKIND_SCORESを毎tick書き足す)の生存開始を
    「recurrent_exec自身の実行時点」のままにすると、生存区間解析が
    body自身のscratch(次tickの再実行で使い回される)より後に生まれる値
    だと誤判定し、同じ領域を割り当ててしまう。実行時は次tickのscratch
    書き込みが前tickにstackedへ書いた行を踏み潰す実バグになる
    (実装時にMetal/CUDA実行で発見。simulateと出力の75%が不一致に
    なった)。`memplan.py::liveness()`の後処理でstacked出力の生存開始を
    scope先頭まで遡らせて直す。
    """

    def test_stacked_output_first_use_is_backdated_to_scope_start(
            self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        first, _last = liveness(plan)
        stacked_ops = [op for op in plan.ops
                      if op.kind == 'recurrent_exec'
                      and op.attrs.get('output_mode') == 'stacked']
        assert stacked_ops, 'stacked出力のrecurrentが無い'
        for op in stacked_ops:
            scope_ops = [i for i, o in enumerate(plan.ops)
                        if o.scope == op.node_id]
            assert scope_ops, f'{op.node_id}のbodyが空'
            assert first[op.output] <= min(scope_ops), (
                f'{op.node_id}のstacked出力の生存開始がbodyの先頭より後'
                '(次tickのscratch再利用に上書きされる恐れ)'
            )

    def test_gpu_style_pooling_has_no_overlap_for_stacked_output(
            self, rddlgn_seq2seq_tiny_bundle):
        """GPUバックエンドと同じく`plan.output_value`もプールした状態で、
        stacked出力がbody内scratchと領域衝突していないこと(Metal/CUDAが
        simulateとビット厳密一致することの前提となる静的な性質)。"""
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        mem = plan_memory(plan)   # excludeなし(GpuBackendと同じ呼び方)
        check_no_overlap(plan, mem)   # 重なっていればAssertionError


@pytest.mark.slow
class TestRepeatedInferenceIsClean:
    """状態バッファが前回の推論を持ち越さない(固有のリスク)。"""

    def test_two_inferences_both_match_simulate(
            self, cc, tmp_path, rddlgn_tiny_bundle):
        graph = ir_from_dict(rddlgn_tiny_bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(20260815)
        x = rng.integers(0, 2, size=(2, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        got0, got1 = _build_and_run_twice(cc, tmp_path, rddlgn_tiny_bundle,
                                          x[0], x[1])
        assert np.array_equal(got0, want[0]), '1回目が simulate と不一致'
        assert np.array_equal(got1, want[1]), (
            '2回目が simulate と不一致(前回の推論のstateが残っている)'
        )

    def test_same_input_twice_is_idempotent(
            self, cc, tmp_path, rddlgn_tiny_bundle):
        graph = ir_from_dict(rddlgn_tiny_bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(3)
        x = rng.integers(0, 2, size=in_width, dtype=np.uint8)
        got0, got1 = _build_and_run_twice(cc, tmp_path, rddlgn_tiny_bundle, x, x)
        assert np.array_equal(got0, got1), '同じ入力で結果が変わる(状態が残っている)'
