"""dense多層融合カーネル(`--dense-fuse`)の等価性検証。

背景: GEM(DAC2025)のBoomerang層(複数論理段を共有メモリ常駐で1同期ごとに
処理)を、denseモデル(通常DLGN)のlut_dense列+group_sumへ適用する。
最小実験(scripts/gpu_calib流儀、har_dlgn_k6000_gumbel、Metal実機、
GPU busy時間、5セット中央値、`ir.simulate`との整数厳密一致を確認済み):

    batch=1    : 0.81x(悪化。1スレッドグループしか立たず占有率が下がる)
    batch=32   : 1.91x
    batch=128  : 2.51x(最良)
    batch=256  : 2.30x

判定基準(batch=1のGPU時間 または 最良バッチのスループットで1.3倍以上)を
最良バッチ側で満たしたため採用(opt-in、既定OFF)。

このファイルは3段+正しさ地雷修理の回帰テスト:
    1. `TestFuseDenseLayersDetection`: プラン層の検出条件(鎖の長さ・
       活性サイズが共有メモリに収まるか)をGPUを起動せず固定する。
       追加した2つのガード(n_inputs==2以外の層は鎖に含めない・
       鎖の長さは`DENSE_FUSE_MAX_LAYERS`以下)もここに含む。
    2. `TestGeneratedSourceStructure`: Metal/CUDA生成物の構造
       (既定OFFで新カーネルが混じらない、ONで鎖ごとに1カーネル)を
       GPUを起動せず固定する。
    3. `TestMetalDenseFuseMatchesSimulate`: 実際にMSLをビルド・実行して
       `ir/simulate.py` と整数厳密一致するか、既定(層別カーネル列)経路
       との新旧一致を見る(macOS + Metalのある環境のみ)。
    4. `TestAddrFormulaAssumesTwoInputs`: `densefuse._layer_lines`のaddr式が
       2入力決め打ちであることの直接証明(n_inputs==2ガードが
       無ければ何が壊れるかを示す故障注入型テスト、GPU不要)。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cudabackend import generate_cuda
from griffith_logic.deploy.densefuse import CODEGEN_KERNEL_PREFIX
from griffith_logic.deploy.metalbackend import generate_metal
from griffith_logic.deploy.plan import (
    DENSE_FUSE_MAX_LAYERS,
    build_plan,
    dense_fuse_act_bytes,
    fuse_dense_layers,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / 'tools' / 'webbench' / 'public' / 'models'
#: 対象4モデル(層活性が最大でも6,000バイト、共有メモリ32KBに
#: 全層余裕で収まる)。
HAR_MODEL_NAMES = [
    'har_dlgn_k600_gumbel', 'har_dlgn_k2400_gumbel', 'har_dlgn_dwn',
    'har_dlgn_k6000_gumbel',
]


def _model_bundle(name: str) -> dict | None:
    path = MODELS_DIR / f'{name}.json'
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding='utf-8'))
    return build_bundle(doc)


def _mixed_arity_dense_ir(seed: int, in_width: int, widths: list[int],
                          n_inputs_list: list[int], k: int,
                          name: str) -> IRGraph:
    """`webbench.build_dense_ir`のn_inputs=2決め打ちを層ごとに外せる版。

    n_inputs混在ガード(fuse_dense_layers)を検証するためだけの
    最小合成IR。lut_layer直列+group_sumという形自体はbuild_dense_irと
    同じ(手組み)。
    """
    assert len(widths) == len(n_inputs_list)
    rng = np.random.default_rng(seed)
    nodes, edges = [], [IREdge(src='x', dst='n1')]
    prev_w = in_width
    for li, (w, n_in) in enumerate(zip(widths, n_inputs_list, strict=True)):
        idx = rng.integers(0, prev_w, size=(n_in, w)).astype(np.int64)
        tts = [f'0x{int(v):x}'
              for v in rng.integers(0, 1 << (1 << n_in), size=w)]
        nodes.append(IRNode(id=f'n{li + 1}', type='lut_layer', params={
            'in_width': prev_w, 'out_width': w, 'n_inputs': n_in,
            'indices': idx, 'truth_tables': tts,
        }))
        edges.append(IREdge(src=f'n{li + 1}', dst=f'n{li + 2}'))
        prev_w = w
    gs_id = f'n{len(widths) + 1}'
    nodes.append(IRNode(id=gs_id, type='group_sum',
                        params={'in_width': prev_w, 'k': k, 'tau': 1.0}))
    edges.append(IREdge(src=gs_id, dst='y'))
    return IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=k, semantics='class_scores')],
        nodes=nodes, edges=edges,
    )


# ============================================================
# 1. プラン層の検出条件(GPUを起動しない、常時CI)
# ============================================================


class TestFuseDenseLayersDetection:
    def test_har_models_fuse_into_single_op(self):
        """har4モデルは全層+group_sumが1本の鎖(=1つのdense_fuse op)。"""
        found = 0
        for name in HAR_MODEL_NAMES:
            bundle = _model_bundle(name)
            if bundle is None:
                continue
            found += 1
            plan = build_plan(bundle, batch=4, dense_fuse=True)
            assert [op.kind for op in plan.ops] == ['dense_fuse'], name
        if found == 0:
            pytest.skip('配信har系モデルが無い環境')

    def test_dense_fuse_false_leaves_plan_unchanged(self):
        bundle = _model_bundle('har_dlgn_k600_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        plan = build_plan(bundle, batch=1, dense_fuse=False)
        kinds = [op.kind for op in plan.ops]
        assert 'dense_fuse' not in kinds
        assert kinds.count('lut_dense') == 4   # k600は4層
        assert kinds.count('reduce_scores') == 1

    def test_two_layer_chain_fuses_via_webbench_builder(self, webbench):
        """`scripts/export_webbench.py::build_dense_ir` の合成モデルでも
        同じ検出が働く(実モデル資産に依存しない最小ケース)。"""
        graph = webbench.build_dense_ir(seed=1, in_width=24, widths=[32, 24],
                                        k=3, name='k49_small')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)
        plan = build_plan(bundle, batch=1, dense_fuse=True)
        assert [op.kind for op in plan.ops] == ['dense_fuse']
        assert plan.ops[0].attrs['k'] == 3
        assert [layer['out_width'] for layer in plan.ops[0].attrs['layers']] \
            == [32, 24]

    def test_single_layer_chain_is_not_fused(self, webbench):
        """鎖の長さが1(lut_dense 1枚+group_sum)は融合しない
        (最小実験はn_layers>=2の鎖でしか検証していないため、
        対象を実測した範囲に限るスコープ制約)。"""
        graph = webbench.build_dense_ir(seed=2, in_width=24, widths=[24],
                                        k=3, name='k49_single')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)
        plan = build_plan(bundle, batch=1, dense_fuse=True)
        assert [op.kind for op in plan.ops] == ['lut_dense', 'reduce_scores']

    def test_wide_chain_exceeds_shared_memory_and_is_not_fused(self, webbench):
        """活性のping-pong2本(1本20,000B)が32KBを超える → 融合しない
        (mnist_adderの層幅45.9KBが対象外のまま、という仕様の合成版)。"""
        graph = webbench.build_dense_ir(seed=3, in_width=64,
                                        widths=[20000, 20000], k=4,
                                        name='k49_wide')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c', 'd'])
        bundle = build_bundle(doc)
        act_bytes = dense_fuse_act_bytes(
            [{'in_width': 64, 'out_width': 20000},
             {'in_width': 20000, 'out_width': 20000}])
        assert act_bytes == 20000
        assert 2 * act_bytes > 32768
        plan = build_plan(bundle, batch=1, dense_fuse=True)
        assert 'dense_fuse' not in [op.kind for op in plan.ops]
        assert [op.kind for op in plan.ops] == ['lut_dense', 'lut_dense',
                                                'reduce_scores']

    def test_dense_fuse_act_bytes_uses_first_layer_in_width(self):
        """1層目のin_widthが後続層のout_widthより大きい実例
        (har_dlgn_k600: in_width=4608 > out_width=600)を直接固定する。"""
        layers = [{'in_width': 4608, 'out_width': 600},
                 {'in_width': 600, 'out_width': 600}]
        assert dense_fuse_act_bytes(layers) == 4608

    def test_fuse_dense_layers_returns_fused_node_ids(self, webbench):
        graph = webbench.build_dense_ir(seed=4, in_width=24, widths=[32, 24],
                                        k=3, name='k49_ids')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)
        plan = build_plan(bundle, batch=1, dense_fuse=False)
        fused = fuse_dense_layers(plan)
        assert fused == ['n3']   # group_sumノードid(2層なのでn3)

    # --: n_inputs!=2の層を鎖に含めないガード ---------------------

    @pytest.mark.parametrize('n_in', [3, 4])
    def test_mixed_arity_layer_is_excluded_from_fusion(self, webbench, n_in):
        """n_inputs=2の層とn_inputs=3/4の層が混在する鎖は、後者の層が
        原因で一切融合されない(densefuse._layer_linesのaddr式が2入力
        決め打ちなので、混ぜて融合すると誤答になる。欠陥1)。

        既定経路(dense_fuse=False)と`--dense-fuse`相当(True)が
        同一プランになることも合わせて固定する。
        """
        graph = _mixed_arity_dense_ir(
            seed=10 + n_in, in_width=24, widths=[16, 12],
            n_inputs_list=[2, n_in], k=3, name=f'k52a_mixed_{n_in}')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)

        plan_default = build_plan(bundle, batch=1, dense_fuse=False)
        plan_fused = build_plan(bundle, batch=1, dense_fuse=True)
        kinds_default = [op.kind for op in plan_default.ops]
        kinds_fused = [op.kind for op in plan_fused.ops]

        assert kinds_default == ['lut_dense', 'lut_dense', 'reduce_scores']
        assert kinds_fused == kinds_default
        assert 'dense_fuse' not in kinds_fused
        # n_inputsが本当に混在していたことの前提チェック
        assert [int(op.attrs['n_inputs']) for op in plan_default.ops
               if op.kind == 'lut_dense'] == [2, n_in]

    def test_mixed_arity_bad_layer_at_chain_head_still_blocks_it_alone(
            self, webbench):
        """n_inputs!=2の層が鎖の先頭に来ても、後続の2入力のみの層は
        (group_sumへ直接繋がっていれば)独立して融合されうる ─
        ガードが「その層だけ」を弾き、周辺の正常な層を巻き込みすぎない
        ことの確認。"""
        graph = _mixed_arity_dense_ir(
            seed=20, in_width=24, widths=[16, 12, 12],
            n_inputs_list=[4, 2, 2], k=3, name='k52a_head_bad')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)
        plan = build_plan(bundle, batch=1, dense_fuse=True)
        kinds = [op.kind for op in plan.ops]
        # 1層目(n_inputs=4)は単独のlut_denseのまま、2・3層目はgroup_sumと
        # 融合されて1本のdense_fuseになる。
        assert kinds == ['lut_dense', 'dense_fuse']
        assert [len(op.attrs['layers']) for op in plan.ops
               if op.kind == 'dense_fuse'] == [2]

    # --: 鎖の最大層数ガード(Metalバッファ引数上限=14層) ----------

    @pytest.mark.parametrize('n_layers', [15, 16, 20])
    def test_long_chain_is_capped_at_max_layers(self, webbench, n_layers):
        """15層以上の連続dense鎖は、融合されても`DENSE_FUSE_MAX_LAYERS`
        (=14)層以下の区間ごとに切られる(欠陥2。Metalのカーネル
        引数テーブル index 0..30 に buffer(2+2i)/buffer(3+2i) が収まる
        上限)。"""
        widths = [8] * n_layers
        graph = webbench.build_dense_ir(
            seed=100 + n_layers, in_width=8, widths=widths, k=4,
            name=f'k52a_long_{n_layers}')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c', 'd'])
        bundle = build_bundle(doc)
        plan = build_plan(bundle, batch=1, dense_fuse=True)

        fused_ops = [op for op in plan.ops if op.kind == 'dense_fuse']
        assert fused_ops, '融合が1つも起きていない(テストの前提が崩れている)'
        for op in fused_ops:
            assert len(op.attrs['layers']) <= DENSE_FUSE_MAX_LAYERS

        # 貪欲な鎖延伸を14層で打ち切ると、group_sumの直前に来る最大14層の
        # 区間だけが融合され、それより手前の (n_layers - 14) 層は
        # lut_denseのまま残る(fuse_dense_layersのdocstring参照)。
        kinds = [op.kind for op in plan.ops]
        assert kinds.count('lut_dense') == n_layers - DENSE_FUSE_MAX_LAYERS
        assert kinds.count('dense_fuse') == 1

    def test_max_layers_param_is_configurable_for_unit_testing(self, webbench):
        """`fuse_dense_layers`は`max_shared_bytes`と同じ流儀で`max_layers`
        も引数化してある ─ 小さい上限を直接渡して境界を最小コストで
        確認する(GPUを起動しない)。"""
        graph = webbench.build_dense_ir(
            seed=200, in_width=8, widths=[8, 8, 8, 8], k=4, name='k52a_small_cap')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c', 'd'])
        bundle = build_bundle(doc)
        plan = build_plan(bundle, batch=1, dense_fuse=False)
        fused = fuse_dense_layers(plan, max_layers=2)
        assert fused == ['n5']
        dense_fuse_ops = [op for op in plan.ops if op.kind == 'dense_fuse']
        assert len(dense_fuse_ops) == 1
        assert len(dense_fuse_ops[0].attrs['layers']) == 2
        # 4層 - 上限2層 = 2層がlut_denseのまま残る
        assert [op.kind for op in plan.ops].count('lut_dense') == 2


# ============================================================
# 2. 生成ソースの構造(GPUを起動しない、常時CI)
# ============================================================


class TestGeneratedSourceStructure:
    def test_metal_default_has_no_dense_fuse_kernel(self):
        bundle = _model_bundle('har_dlgn_k600_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        text = generate_metal(bundle, name='m')['m.mm']
        assert CODEGEN_KERNEL_PREFIX not in text
        assert 'mdl_metal_dense_fuse_exec' not in text
        assert 'mdl_metal_lut_layer' in text

    def test_metal_dense_fuse_true_emits_one_kernel(self):
        bundle = _model_bundle('har_dlgn_k600_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        text = generate_metal(bundle, name='m', dense_fuse=True)['m.mm']
        assert text.count(f'kernel void {CODEGEN_KERNEL_PREFIX}') == 1
        assert text.count('mdl_metal_dense_fuse_exec(') == 1
        # 層別カーネル呼び出し(lut_layer/group_sum)は融合経路では
        # 生成されない(実行時に読まないため)。
        assert 'mdl_metal_lut_layer' not in text
        assert 'mdl_metal_group_sum(' not in text
        assert 'mdl_metal_group_sum_r' not in text

    def test_cuda_default_has_no_dense_fuse_kernel(self):
        bundle = _model_bundle('har_dlgn_k600_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        text = generate_cuda(bundle, name='m')['m.cu']
        assert CODEGEN_KERNEL_PREFIX not in text
        assert 'mdl_cuda_lut_layer' in text

    def test_cuda_dense_fuse_true_emits_one_kernel(self):
        bundle = _model_bundle('har_dlgn_k600_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        text = generate_cuda(bundle, name='m', dense_fuse=True)['m.cu']
        assert text.count(f'__global__ void {CODEGEN_KERNEL_PREFIX}') == 1
        assert 'mdl_cuda_lut_layer' not in text
        assert 'mdl_cuda_group_sum' not in text

    # --: n_inputs混在モデルは--dense-fuseを付けても融合カーネルが
    #    生成されない ------------------------------------------------

    @pytest.mark.parametrize('n_in', [3, 4])
    def test_metal_mixed_arity_emits_no_fuse_kernel(self, webbench, n_in):
        graph = _mixed_arity_dense_ir(
            seed=30 + n_in, in_width=24, widths=[16, 12],
            n_inputs_list=[2, n_in], k=3, name=f'k52a_metal_mixed_{n_in}')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)
        text = generate_metal(bundle, name='m', dense_fuse=True)['m.mm']
        assert CODEGEN_KERNEL_PREFIX not in text
        assert 'mdl_metal_dense_fuse_exec' not in text
        assert 'mdl_metal_lut_layer' in text

    @pytest.mark.parametrize('n_in', [3, 4])
    def test_cuda_mixed_arity_emits_no_fuse_kernel(self, webbench, n_in):
        graph = _mixed_arity_dense_ir(
            seed=40 + n_in, in_width=24, widths=[16, 12],
            n_inputs_list=[2, n_in], k=3, name=f'k52a_cuda_mixed_{n_in}')
        doc = webbench.build_model_doc(
            graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c'])
        bundle = build_bundle(doc)
        text = generate_cuda(bundle, name='m', dense_fuse=True)['m.cu']
        assert CODEGEN_KERNEL_PREFIX not in text
        assert 'mdl_cuda_lut_layer' in text

    def test_dense_fuse_batch_only_changes_grid_not_source(self):
        """batchはホスト側の起動グリッド(グループ数)だけに効く
        ─ カーネル本体のテキストはbatchに依存しない。"""
        bundle = _model_bundle('har_dlgn_k600_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        text1 = generate_metal(bundle, name='m', batch=1,
                               dense_fuse=True)['m.mm']
        text128 = generate_metal(bundle, name='m', batch=128,
                                 dense_fuse=True)['m.mm']

        def _extract_kernel(text):
            start = text.index(f'kernel void {CODEGEN_KERNEL_PREFIX}')
            end = text.index('}', text.index('{', start))
            return text[start:end]
        assert _extract_kernel(text1) == _extract_kernel(text128)


# ============================================================
# 3. 実機ビルド・実行(Metal、macOSのみ)
# ============================================================

from griffith_logic.deploy.metalverify import (  # noqa: E402
    build_metal_model,
    has_metal,
    run_metal_model,
    verify_metal_matches_simulate,
)

pytestmark_metal = pytest.mark.skipif(
    not has_metal(), reason='macOS + Metal のある環境でのみ実行')


@pytestmark_metal
@pytest.mark.slow
class TestMetalDenseFuseMatchesSimulate:
    def test_dense_doc_fuse_matches_simulate(self, dense_doc, tmp_path):
        result = verify_metal_matches_simulate(
            build_bundle(dense_doc), n_samples=3, work_dir=tmp_path,
            verbose=False, dense_fuse=True)
        assert result['max_abs_diff'] == 0

    def test_fuse_and_interp_agree(self, dense_doc, tmp_path):
        """dense_fuse ON/OFFが互いに一致し、両方ともsimulateと一致する
        (test_conv_codegen.pyのcodegen/interp一致テストと同じ形)。"""
        from griffith_logic.ir.serialize import ir_from_dict
        from griffith_logic.ir.simulate import simulate

        bundle = build_bundle(dense_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        rng = np.random.default_rng(21)
        x = rng.integers(0, 2, size=in_width, dtype=np.uint8)
        want = simulate(graph, x[None])[0].astype(np.int64)

        exe_interp = build_metal_model(bundle, tmp_path / 'interp',
                                       dense_fuse=False)
        exe_fused = build_metal_model(bundle, tmp_path / 'fused',
                                      dense_fuse=True)
        got_interp = run_metal_model(exe_interp, x.tobytes(),
                                     n_out).astype(np.int64)
        got_fused = run_metal_model(exe_fused, x.tobytes(),
                                    n_out).astype(np.int64)
        assert np.array_equal(got_interp, want)
        assert np.array_equal(got_fused, want)
        assert np.array_equal(got_interp, got_fused)

    @pytest.mark.parametrize('name', HAR_MODEL_NAMES)
    def test_har_models_fuse_matches_simulate(self, name, tmp_path):
        bundle = _model_bundle(name)
        if bundle is None:
            pytest.skip(f'配信モデル {name} が無い環境')
        result = verify_metal_matches_simulate(
            bundle, n_samples=3, work_dir=tmp_path / name, verbose=False,
            dense_fuse=True)
        assert result['max_abs_diff'] == 0

    @pytest.mark.parametrize('name', HAR_MODEL_NAMES)
    def test_har_models_default_still_matches_simulate(self, name, tmp_path):
        """dense_fuse機能追加が既定経路(OFF)を壊していないことの回帰確認。"""
        bundle = _model_bundle(name)
        if bundle is None:
            pytest.skip(f'配信モデル {name} が無い環境')
        result = verify_metal_matches_simulate(
            bundle, n_samples=2, work_dir=tmp_path / f'{name}_default',
            verbose=False, dense_fuse=False)
        assert result['max_abs_diff'] == 0


#: `test_batch.py::METAL_BATCH_DRIVER` と同型(1スレッドグループ=1サンプル
#: という設計は、batch=N が「グループ数=N」に直結するのでバッチを
#: またいだ参照ミスが一番出やすい経路 ─ バッチ一致テストと同じ理由で
#: 独立に固定する)。
_N_BATCH = 37


def _metal_batch_driver(max_batch: int, n: int) -> str:
    return f'''#include "mdl_metal.h"
#include <stdio.h>
#include "mdl_stdio.h"
#include <stdlib.h>
#define N {n}
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


@pytestmark_metal
@pytest.mark.slow
class TestMetalDenseFuseBatch:
    """設計(1スレッドグループ=1サンプル、グループ数=batch)固有の
    「枚をまたいだ参照」チェック(`test_batch.py`と同じ理由・同じ形)。"""

    def _run(self, bundle, max_batch, n, seed, tmp_path):
        from griffith_logic.ir.serialize import ir_from_dict
        from griffith_logic.ir.simulate import simulate

        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=(n, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        work = tmp_path / 'gen'
        work.mkdir(parents=True, exist_ok=True)
        files = generate_metal(bundle, name='model_metal', batch=max_batch,
                               dense_fuse=True)
        for name_, text in files.items():
            (work / name_).write_text(text, encoding='utf-8')
        (work / 'driver.mm').write_text(
            _metal_batch_driver(max_batch, n), encoding='utf-8')
        from griffith_logic.deploy.metalverify import RUNTIME_METAL
        build = subprocess.run(
            ['clang++', '-std=c++17', '-O2', '-fobjc-arc',
             f'-I{RUNTIME_METAL}', '-framework', 'Metal',
             '-framework', 'Foundation', str(work / 'model_metal.mm'),
             str(RUNTIME_METAL / 'mdl_metal.mm'), str(work / 'driver.mm'),
             '-o', str(work / 'app')],
            capture_output=True, text=True)
        assert build.returncode == 0, f'ビルド失敗:\n{build.stderr}'
        proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                              capture_output=True)
        assert proc.returncode == 0, (
            f'実行失敗: {proc.stderr.decode(errors="replace")}')
        got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
        got = got.reshape(want.shape)
        assert np.array_equal(got, want), (
            f'dense_fuse のバッチ実行が simulate と不一致'
            f'({int((got != want).sum())}/{want.size} 要素)。'
            '枚をまたいだ参照(threadgroup活性・スレッドグループ境界)を疑うこと')

    def test_dense_fuse_batch_equals_simulate(self, tmp_path):
        bundle = _model_bundle('har_dlgn_k6000_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        self._run(bundle, max_batch=_N_BATCH, n=_N_BATCH, seed=99,
                  tmp_path=tmp_path)

    def test_dense_fuse_partial_batch_matches(self, tmp_path):
        """確保した上限(64)より少ない枚数(37)でも一致する
        (グループ数=nであってmax_batchではないことの固定)。"""
        bundle = _model_bundle('har_dlgn_k6000_gumbel')
        if bundle is None:
            pytest.skip('配信har系モデルが無い環境')
        self._run(bundle, max_batch=64, n=_N_BATCH, seed=7, tmp_path=tmp_path)


# ============================================================
# 4. 故障注入: n_inputs==2ガードが無ければ何が起きるか(GPU不要)
# ============================================================


class TestAddrFormulaAssumesTwoInputs:
    """欠陥1の直接証明。

    `densefuse._layer_lines` が生成するaddr式は

        addr = cur[idx[o]] | (cur[idx[ow + o]] << 1)

    という2入力決め打ちで、3本目以降の入力添字(`indices`の3行目以降)を
    一切読まない。`plan.py::fuse_dense_layers`のn_inputs==2ガード
    (このファイルの`TestFuseDenseLayersDetection`にある
    `test_mixed_arity_layer_is_excluded_from_fusion`等)が無ければ、
    n_inputs=3以上の層がそのまま`dense_fuse`のlayersへ混入し、
    実機のkernelはここで再現する通り黙って間違った答えを返す。

    Metal/CUDAを実際にビルド・実行しなくても、addr式の形はこの2行
    (`_layer_lines`の該当行そのまま)でPython上に再現できるので、ここでは
    GPUを起動せず常時CIで固定する(実機での故障注入はビルド一式の手組みが
    要り重いため見送り。上のTestGeneratedSourceStructure/
    TestFuseDenseLayersDetectionが「ガードが効いてこの入力は実際には
    dense_fuseへ渡らない」ことを別途固定している)。
    """

    def test_third_input_is_silently_ignored_by_addr_formula(self):
        w = 5
        # 3入力目(idx2)がaddrに一切現れないことを、addrに影響する形で作る:
        # idx0/idx1は常にx[0]=x[1]=0を指し、idx2だけがx[3]=1を指す。
        idx0 = [0] * w
        idx1 = [1] * w
        idx2 = [3] * w
        # tt: bit0-3(idx2側ビットが立たないaddr)=0、bit4-7(立つaddr)=1。
        # 「3入力目だけで出力が変わる」関数(3入力目がゲートなら常に
        # 出力に効く。2入力扱いだと絶対に見えない)。
        tt = [0xF0] * w
        x = [0, 0, 0, 1]     # x[idx2[o]] = x[3] = 1

        # 正しい出力(n_inputs=3のフル評価。ir/simulateと同じ意味論
        # ─ addr = x[idx0] | x[idx1]<<1 | x[idx2]<<2)。
        correct = []
        for o in range(w):
            addr3 = x[idx0[o]] | (x[idx1[o]] << 1) | (x[idx2[o]] << 2)
            correct.append((tt[o] >> addr3) & 1)
        assert correct == [1] * w   # x[3]=1なので全ゲート1が正解

        # `densefuse._layer_lines`が生成するC式そのまま(2入力決め打ち、
        # idx2を一切参照しない):
        #   addr = cur[idx{ln}[o]] | (cur[idx{ln}[{ow}+o]] << 1);
        #   nxt[o] = (tt{ln}[o] >> addr) & 1;
        naive = []
        for o in range(w):
            addr_naive = x[idx0[o]] | (x[idx1[o]] << 1)
            naive.append((tt[o] >> addr_naive) & 1)

        assert naive == [0] * w
        assert naive != correct, (
            'addr式が3本目の入力を無視して常に間違った側の真理値表ビット'
            '(bit0-3)しか読まない。plan.pyのn_inputs==2ガード'
            '(fuse_dense_layers)を外すとdense_fuseはこの通り静かに'
            '誤答を返す ─ このテストはガードの再発防止線')
