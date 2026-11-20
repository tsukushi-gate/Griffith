"""ABC(論理合成)最適化の等価性テスト(@abc、compile-pipeline)。

run_abc_optimize は IR core を ABC の等価保存フローで最適化し、
simulate(golden model)との完全一致を内部assertする。このテストは
さらに外側から simulate照合・cec・LUT/キューブ削減を検証する。

ABC非導入環境では自動skip。実行: uv run pytest -m abc
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from griffith_logic import GroupSum, LutLayer
from griffith_logic.convert.decompose_thermo import exact_adder_baseline_layer
from griffith_logic.export import export_ir
from griffith_logic.export.abc_opt import (
    abc_available,
    run_abc_cec,
    run_abc_optimize,
    valid_codewords,
)
from griffith_logic.export.blif import reattach_group_sum
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import tt_to_hex
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate

pytestmark = [
    pytest.mark.abc,
    pytest.mark.skipif(not abc_available(), reason='abc未導入'),
]


def _binary(batch: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, width), dtype=np.int64)


def _with_group_sum(core_graph: IRGraph) -> IRGraph:
    """core(x→lut鎖→y)へk=幅のgroup_sumを付けてrun_abc_optimize対象にする。"""
    n_out = core_graph.outputs[0].width
    gs = IRNode(id='gs', type='group_sum',
                params={'in_width': n_out, 'k': n_out, 'tau': 1.0})
    return reattach_group_sum(core_graph, gs)


def _single_lut_graph(tt: int) -> IRGraph:
    """x(3bit) → 1出力lut_layer(真理値表tt) → group_sum(k=1) → y"""
    node = IRNode(
        id='f', type='lut_layer',
        params={
            'in_width': 3, 'out_width': 1, 'n_inputs': 3,
            'indices': np.array([[0], [1], [2]], dtype=np.int64),
            'truth_tables': tt_to_hex([tt]),
        },
    )
    gs = IRNode(id='gs', type='group_sum',
                params={'in_width': 1, 'k': 1, 'tau': 1.0})
    graph = IRGraph(
        inputs=[IRPort(id='x', width=3)],
        outputs=[IRPort(id='y', width=1)],
        nodes=[node, gs],
        edges=[IREdge(src='x', dst='f'), IREdge(src='f', dst='gs'),
               IREdge(src='gs', dst='y')],
    )
    validate(graph)
    return graph


# f(t) = t2 ∧ XOR3(t0,t1,t2)。温度計有効符号語上では 1[level==3]=AND3 に一致。
# 総関数としては2キューブ、popcount射影との合成はAND3(1キューブ)へ簡約される
_TT_MESSY = (1 << 4) | (1 << 7)  # minterm {100, 111}(addr bit j=スロットj)


class TestRunAbcOptimize:
    def test_toy_chain_simulate_and_cec(self, tmp_path):
        """toy lut_layer鎖: simulate 1024本一致+cec equivalent+LUT非増加"""
        torch.manual_seed(41)
        model = nn.Sequential(
            LutLayer(16, 24, n_inputs=2),
            LutLayer(24, 24, n_inputs=2),
            GroupSum(k=4, tau=2.0),
        )
        model.eval()
        ir = export_ir(model, input_width=16, name='abc_toy')

        optimized, stats = run_abc_optimize(
            ir, workdir=str(tmp_path / 'opt'))

        x = _binary(1024, 16, seed=9)
        assert np.array_equal(simulate(ir, x), simulate(optimized, x)), (
            '最適化後IRがgolden modelと不一致'
        )
        assert stats['luts_after'] <= stats['luts_before'], stats

        cec = run_abc_cec(ir, optimized, workdir=str(tmp_path / 'cec'))
        assert cec['equivalent'], f'cecが非等価判定:\n{cec["stdout"]}'

    def test_adder_identity_copy_removal(self, tmp_path):
        """exact_adder_baseline_layerの小IR: 恒等コピーLUTがstrashで消える"""
        rng = np.random.default_rng(0)
        q = rng.choice([-1, 0, 1], size=(2, 8))
        q[:, 0] = 1  # 全ゼロ行を防ぐ
        thr = np.array([0.5, -0.5])
        base = exact_adder_baseline_layer(q, thr)
        graph = _with_group_sum(base)

        optimized, stats = run_abc_optimize(
            graph, workdir=str(tmp_path / 'opt'))

        assert stats['luts_after'] < stats['luts_before'], (
            f'恒等コピー除去でLUT数が減っていない: {stats}'
        )
        # 全256パターンでビット厳密一致
        x = np.array([[(k >> j) & 1 for j in range(8)]
                      for k in range(256)], dtype=np.int64)
        assert np.array_equal(simulate(graph, x), simulate(optimized, x))

    def test_projection_reduces_cubes(self, tmp_path):
        """射影: 温度計グループの無効パターンDCで射影なしより簡約される"""
        graph = _single_lut_graph(_TT_MESSY)
        groups = [[0, 1, 2]]

        _opt_plain, s_plain = run_abc_optimize(
            graph, workdir=str(tmp_path / 'plain'))
        opt_proj, s_proj = run_abc_optimize(
            graph, projection_groups=groups,
            workdir=str(tmp_path / 'proj'))

        assert s_proj['n_cubes_after'] < s_plain['n_cubes_after'], (
            f"射影でキューブが減っていない: "
            f"proj={s_proj['n_cubes_after']} plain={s_plain['n_cubes_after']}"
        )
        assert s_proj['luts_after'] <= s_plain['luts_after']
        assert s_proj['n_projection_luts'] == 3

        # 有効符号語上で元グラフと一致(射影は有効域では恒等)
        rng = np.random.default_rng(3)
        x = valid_codewords(rng, 256, 3, groups)
        assert np.array_equal(simulate(graph, x), simulate(opt_proj, x)), (
            '射影付き最適化が有効符号語上で元と不一致'
        )

    def test_calib_x_is_verified(self, tmp_path):
        """calib_xも等価性検証に使われる(正常系: assertを通過して返る)"""
        graph = _single_lut_graph(_TT_MESSY)
        calib = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=np.int64)
        optimized, stats = run_abc_optimize(
            graph, calib_x=calib, workdir=str(tmp_path / 'opt'))
        assert np.array_equal(simulate(graph, calib),
                              simulate(optimized, calib))
        assert stats['script']  # 使用スクリプトが記録される


class TestRunAbcCec:
    def test_equivalent_true_for_same_circuit(self, tmp_path):
        g1 = _single_lut_graph(_TT_MESSY)
        g2 = _single_lut_graph(_TT_MESSY)
        result = run_abc_cec(g1, g2, workdir=str(tmp_path))
        assert result['equivalent'], result['stdout']

    def test_equivalent_false_for_flipped_bit(self, tmp_path):
        g1 = _single_lut_graph(_TT_MESSY)
        g2 = _single_lut_graph(_TT_MESSY ^ 1)  # 1エントリだけ反転
        result = run_abc_cec(g1, g2, workdir=str(tmp_path))
        assert not result['equivalent'], result['stdout']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'abc'])
