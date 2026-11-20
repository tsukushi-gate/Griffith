"""メモリ計画の正しさと効果。

バッファを共有すると、誤りは**壊れて落ちる**のではなく「上書きされた値が
たまたままだ読まれていない」形で潜る。だからここでは

    ・生存区間の重なりを直接検査する(生成前のガード)
    ・**同一プロセスで2回続けて別入力を推論**して両方が simulate と一致する
      ことを確かめる(状態の漏れを検出する唯一の実効的な手段)

の両方をやる。後者は実際に潜在バグを1つ掘り当てた
(`mdl_spatial_group_sum` が出力を `|=` で積むだけでゼロ初期化しておらず、
呼び出し側がゼロ済みであることに暗黙依存していた)。
"""

import subprocess

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cbackend import generate_c
from griffith_logic.deploy.memplan import (
    check_no_overlap,
    liveness,
    plan_memory,
)
from griffith_logic.deploy.plan import KIND_SCRATCH, build_plan
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

from .conftest import RUNTIME_C
from .test_c_kernels import read_u32

#: 2枚を続けて推論して両方の結果を出すドライバ。バッファを共有したときに
#: 起きる「前回の推論の残りが混ざる」を捕まえるのが目的。
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
  static uint8_t a[MDL_INPUT_WIDTH], b[MDL_INPUT_WIDTH];
  /* 出力配列はわざと非ゼロで汚しておく(呼び出し側のゼロ初期化に
     依存していないことを確かめるため)。 */
  static int32_t sa[MDL_OUTPUT_WIDTH], sb[MDL_OUTPUT_WIDTH];
  int32_t i;
  if (fread(a, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) return 1;
  if (fread(b, 1, MDL_INPUT_WIDTH, stdin) != (size_t)MDL_INPUT_WIDTH) return 1;
  for (i = 0; i < MDL_OUTPUT_WIDTH; i++) { sa[i] = 0x5a5a5a; sb[i] = 0x5a5a5a; }
  if (mdl_infer(a, sa) != 0) return 2;
  if (mdl_infer(b, sb) != 0) return 2;
  emit(sa);
  emit(sb);
  return 0;
}
'''


def _build_and_run_twice(cc, tmp_path, bundle, x0, x1):
    files = generate_c(bundle, name='model', validate=False)
    work = tmp_path / 'gen'
    work.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work / name).write_text(text, encoding='utf-8')
    (work / 'driver.c').write_text(TWICE_DRIVER, encoding='utf-8')
    exe = work / 'app'
    build = subprocess.run(
        [cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror',
         f'-I{work}', f'-I{RUNTIME_C / "include"}',
         str(work / 'model.c'), str(RUNTIME_C / 'src' / 'mdl_kernels.c'),
         str(work / 'driver.c'), '-o', str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0:
        raise AssertionError(f'生成Cのビルドに失敗:\n{build.stderr}')
    payload = (x0.astype(np.uint8).tobytes() + x1.astype(np.uint8).tobytes())
    proc = subprocess.run([str(exe)], input=payload, capture_output=True)
    if proc.returncode != 0:
        raise AssertionError(
            f'実行失敗 (exit={proc.returncode}): '
            f'{proc.stderr.decode(errors="replace")}')
    got = read_u32(proc.stdout).astype(np.int64)
    half = got.size // 2
    return got[:half], got[half:]


class TestLiveness:
    @pytest.fixture(params=['seg', 'dense'])
    def any_plan(self, request, seg_doc, dense_doc):
        doc = seg_doc if request.param == 'seg' else dense_doc
        return build_plan(build_bundle(doc))

    def test_output_lives_to_the_end(self, any_plan):
        _, last = liveness(any_plan)
        assert last[any_plan.output_value] == len(any_plan.ops)

    def test_scratch_lives_for_exactly_one_op(self, any_plan):
        """スクラッチの生存区間が1opに閉じている(だから全部畳める)。"""
        first, last = liveness(any_plan)
        scratch = [v.id for v in any_plan.values if v.kind == KIND_SCRATCH]
        if not scratch:
            pytest.skip('スクラッチを持つopが無いモデル(dense経路)')
        for vid in scratch:
            assert first[vid] == last[vid], (
                f'v{vid} のスクラッチが複数opにまたがっている')

    def test_input_is_live_before_the_first_op(self, any_plan):
        first, _ = liveness(any_plan)
        assert first[any_plan.input_value] == -1


class TestAllocation:
    @pytest.fixture(params=['seg', 'dense'])
    def any_plan(self, request, seg_doc, dense_doc):
        doc = seg_doc if request.param == 'seg' else dense_doc
        return build_plan(build_bundle(doc))

    def test_no_live_overlap(self, any_plan):
        """同時に生きている値が領域を共有していない(正しさの中核)。"""
        mem = plan_memory(any_plan)
        check_no_overlap(any_plan, mem)          # 違反なら例外

    def test_output_never_aliases_its_own_inputs(self, any_plan):
        """opの出力が、そのopの入力と同じ領域に置かれない。

        `mdl_or_pool2x2(src, dst, ...)` は src を読みながら dst を書くので、
        ここが崩れると静かに間違う。
        """
        mem = plan_memory(any_plan)
        for op in any_plan.ops:
            if op.output not in mem.offsets:
                continue
            o_off = mem.offsets[op.output]
            o_size = mem.sizes[op.output]
            for vid in list(op.inputs) + list(op.attrs.get('scratch', ())):
                if vid not in mem.offsets:
                    continue
                i_off, i_size = mem.offsets[vid], mem.sizes[vid]
                assert not (o_off < i_off + i_size and i_off < o_off + o_size), (
                    f'{op.kind}({op.node_id}) の出力 v{op.output} が'
                    f'入力 v{vid} と同じ領域を使っている')

    def test_arena_is_smaller_than_naive(self, any_plan):
        mem = plan_memory(any_plan)
        assert mem.arena_words <= mem.naive_words
        assert mem.arena_words >= max(mem.sizes.values()), (
            'アリーナが最大の値より小さい(明らかに足りない)')

    def test_exclusions_are_not_placed(self, any_plan):
        mem = plan_memory(any_plan, exclude={any_plan.input_value})
        assert any_plan.input_value not in mem.offsets

    def test_reuse_actually_happens(self, any_plan):
        """再利用が実際に起きている(オフセットを共有する値の組がある)。

        削減率そのものはモデルの形で決まるので、ここでは「再利用が
        起きていること」だけを見る。実モデルでの削減率は
        `test_real_models.py::test_memory_plan_reuses_buffers` が押さえる。
        """
        mem = plan_memory(any_plan)
        assert mem.arena_words < mem.naive_words, '1つも再利用されていない'
        offs = list(mem.offsets.values())
        assert len(set(offs)) < len(offs), (
            'オフセットが全て別 = 領域を共有した値が1組も無い')


@pytest.mark.slow
class TestRepeatedInferenceIsClean:
    """同一プロセスで2回推論しても互いに影響しない。

    バッファ共有の誤りは1回きりの実行では見えないことが多い。ここが
    `mdl_spatial_group_sum` のゼロ初期化漏れを実際に検出した。
    """

    def test_two_inferences_both_match_simulate(self, cc, tmp_path, seg_doc):
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(20260805)
        x = rng.integers(0, 2, size=(2, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        got0, got1 = _build_and_run_twice(cc, tmp_path, bundle, x[0], x[1])
        assert np.array_equal(got0, want[0]), '1回目が simulate と不一致'
        assert np.array_equal(got1, want[1]), (
            '2回目が simulate と不一致(前回の推論の状態が残っている)')

    def test_same_input_twice_is_idempotent(self, cc, tmp_path, seg_doc):
        """同じ入力を2回入れたら同じ答えが2回出る。"""
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(7)
        x = rng.integers(0, 2, size=in_width, dtype=np.uint8)
        got0, got1 = _build_and_run_twice(cc, tmp_path, bundle, x, x)
        assert np.array_equal(got0, got1), (
            '同じ入力で結果が変わる(状態が残っている)')
