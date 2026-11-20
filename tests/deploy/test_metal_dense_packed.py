"""dense のバッチ軸ビットプレーン化(`--dense-packed`)の等価性検証。

背景: GPU dense経路の現行実装は1スレッド=1ゲート・1bit=1byteで、バッチは
`total=out_width*batch`へ平坦展開されるだけ(ALU使用率は最良でもbatch=128で
8.5%)。CPU側には同じ変換(runtime/c/src/mdl_kernels.c::mdl_pack_batch/
mdl_lut_layer_packed/mdl_group_sum_packed)が既にあり実測9倍が出ているので、
同じ式をGPUの1スレッド=1語(32サンプル分)単位へ持ち込む
(`runtime/gpu/mdl_gpu_kernels.h::mdl_g_pack_batch_word`/
`mdl_g_lut_out_packed[_u16]`/`mdl_g_group_sum_packed_word`)。

このファイルは4段の回帰テスト(`tests/deploy/test_dense_fuse.py`と同じ構成):
    1. `TestDensePackedEligibility`: `plan.py::dense_packed_eligible`の判定
       条件(batch>=32・非画像入力・主経路opがlut_dense/group_sumのみ・
       recurrent非混在)をGPUを起動せず固定する。`generate_metal`/
       `generate_cuda`側の併用禁止(dense_fuse/n_streams>1)もここに含む。
    2. `TestGeneratedSourceStructure`: Metal/CUDA生成物の構造(既定OFFで
       パックカーネルが混じらない、ONで層ごとにpacked版カーネル呼び出しへ
       差し替わる)をGPUを起動せず固定する。
    3. `TestMetalDensePackedMatchesSimulate`: 実際にMSLをビルド・実行して
       `ir/simulate.py`と整数厳密一致するか(macOS + Metalのある環境のみ)。
       n_inputs=2のみの鎖・3〜5混在の鎖の両方、batch∈{32,33,64,100,128}。
    4. `TestFaultInjection`: 真理値表を1ビット壊すと出力が
       (壊していない)simulateの結果と食い違うことを確認する(テストが
       常に一致してしまう空振りでないことの証拠、GPU必要)。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.cudabackend import generate_cuda
from griffith_logic.deploy.metalbackend import generate_metal
from griffith_logic.deploy.plan import (
    DENSE_PACKED_MIN_BATCH,
    KIND_FLAT,
    KIND_SCORES,
    ExecPlan,
    PlanOp,
    PlanValue,
    build_plan,
    dense_packed_eligible,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate

REPO_ROOT = Path(__file__).resolve().parents[2]


def _mixed_arity_dense_ir(seed: int, in_width: int, widths: list[int],
                          n_inputs_list: list[int], k: int,
                          name: str) -> IRGraph:
    """層ごとにn_inputsを変えられるdense IR(手組み、test_dense_fuse.pyの
    同名ヘルパと同じ形。n_inputs=3/4/5混在の鎖を作るのに使う)。"""
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


def _dense_bundle(webbench, *, seed: int, in_width: int, widths: list[int],
                  n_inputs_list: list[int] | None, k: int, name: str) -> dict:
    if n_inputs_list is None:
        graph = webbench.build_dense_ir(seed=seed, in_width=in_width,
                                        widths=widths, k=k, name=name)
    else:
        graph = _mixed_arity_dense_ir(seed=seed, in_width=in_width,
                                      widths=widths,
                                      n_inputs_list=n_inputs_list, k=k,
                                      name=name)
    doc = webbench.build_model_doc(
        graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
        labels=[f'c{i}' for i in range(k)])
    return build_bundle(doc)


# ============================================================
# 1. 適格性判定(GPUを起動しない、常時CI)
# ============================================================


class TestDensePackedEligibility:
    def _flat_plan(self, *, batch: int) -> ExecPlan:
        """`dense_packed_eligible`を直接単体テストするための最小プラン
        (lut_dense 1層+group_sum、build_plan一式を経由しない手組み)。"""
        values = [
            PlanValue(id=0, kind=KIND_FLAT, width=8, origin='input'),
            PlanValue(id=1, kind=KIND_FLAT, width=4, origin='n1'),
            PlanValue(id=2, kind=KIND_SCORES, width=2, origin='n2'),
        ]
        ops = [
            PlanOp(kind='lut_dense', inputs=[0], output=1,
                  attrs={'n_inputs': 2, 'out_width': 4,
                        'indices': [0, 1, 2, 3, 4, 5, 6, 7],
                        'truth_tables': [6, 6, 6, 6]},
                  node_id='n1'),
            PlanOp(kind='reduce_scores', inputs=[1], output=2,
                  attrs={'op': 'group_sum', 'in_width': 4, 'k': 2},
                  node_id='n2'),
        ]
        return ExecPlan(values=values, ops=ops, input_value=0, output_value=2,
                        input_width=8, output_width=2, batch=batch)

    def test_min_batch_is_32(self):
        assert DENSE_PACKED_MIN_BATCH == 32

    def test_batch_below_minimum_rejected(self):
        plan = self._flat_plan(batch=31)
        with pytest.raises(ValueError, match='batch'):
            dense_packed_eligible(plan)

    def test_batch_at_minimum_accepted(self):
        plan = self._flat_plan(batch=32)
        dense_packed_eligible(plan)   # 例外を投げなければOK

    def test_batch_above_minimum_accepted(self):
        plan = self._flat_plan(batch=128)
        dense_packed_eligible(plan)

    def test_preproc_present_rejected(self):
        plan = self._flat_plan(batch=64)
        plan.preproc = {'kind': 'thermometer'}
        with pytest.raises(ValueError, match='画像|preproc'):
            dense_packed_eligible(plan)

    def test_non_lut_dense_op_rejected(self):
        plan = self._flat_plan(batch=64)
        plan.ops.append(PlanOp(kind='plane_op', inputs=[1], output=1,
                               attrs={'op': 'or_pool2x2'}, node_id='bad'))
        with pytest.raises(ValueError, match='lut_dense|group_sum'):
            dense_packed_eligible(plan)

    def test_reduce_scores_non_group_sum_rejected(self):
        plan = self._flat_plan(batch=64)
        plan.ops[-1].attrs['op'] = 'spatial_group_sum'
        with pytest.raises(ValueError, match='lut_dense|group_sum'):
            dense_packed_eligible(plan)

    def test_recurrent_scoped_op_rejected(self):
        plan = self._flat_plan(batch=64)
        plan.ops.append(PlanOp(kind='lut_dense', inputs=[1], output=1,
                               attrs={'n_inputs': 2, 'out_width': 4,
                                     'indices': [0] * 8,
                                     'truth_tables': [0] * 4},
                               node_id='body1', scope='rec1'))
        with pytest.raises(ValueError, match='recurrent'):
            dense_packed_eligible(plan)

    def test_recurrent_exec_op_rejected(self):
        plan = self._flat_plan(batch=64)
        plan.ops.append(PlanOp(kind='recurrent_exec', inputs=[1], output=1,
                               attrs={'steps': 3}, node_id='rec1'))
        with pytest.raises(ValueError, match='recurrent'):
            dense_packed_eligible(plan)

    def test_pure_dense_model_via_build_plan_is_eligible(self, webbench):
        bundle = _dense_bundle(webbench, seed=1, in_width=24,
                               widths=[32, 24], n_inputs_list=None, k=3,
                               name='k53_ok')
        plan = build_plan(bundle, batch=32)
        dense_packed_eligible(plan)

    def test_conv_model_rejected(self, seg_doc):
        """conv2dを含むモデル(seg_doc)は主経路opがgate_program等を含む
        ため拒否される(preprocも画像なので二重に非対象)。"""
        bundle = build_bundle(seg_doc)
        plan = build_plan(bundle, batch=32)
        with pytest.raises(ValueError):
            dense_packed_eligible(plan)

    # -- generate_metal/generate_cuda 側の引数検証 ----------------------

    def test_generate_metal_batch_below_min_raises(self, webbench):
        bundle = _dense_bundle(webbench, seed=2, in_width=24, widths=[16],
                               n_inputs_list=None, k=4, name='k53_lowbatch')
        with pytest.raises(ValueError, match='batch'):
            generate_metal(bundle, batch=16, dense_packed=True)

    def test_generate_metal_dense_fuse_and_packed_conflict(self, webbench):
        bundle = _dense_bundle(webbench, seed=3, in_width=24, widths=[16],
                               n_inputs_list=None, k=4, name='k53_conflict')
        with pytest.raises(ValueError, match='dense_fuse'):
            generate_metal(bundle, batch=32, dense_fuse=True,
                           dense_packed=True)

    def test_generate_metal_n_streams_and_packed_conflict(self, webbench):
        bundle = _dense_bundle(webbench, seed=4, in_width=24, widths=[16],
                               n_inputs_list=None, k=4, name='k53_streams')
        with pytest.raises(ValueError, match='n_streams'):
            generate_metal(bundle, batch=32, n_streams=2, dense_packed=True)

    def test_generate_cuda_batch_below_min_raises(self, webbench):
        bundle = _dense_bundle(webbench, seed=5, in_width=24, widths=[16],
                               n_inputs_list=None, k=4, name='k53_cuda_low')
        with pytest.raises(ValueError, match='batch'):
            generate_cuda(bundle, batch=8, dense_packed=True)

    def test_generate_cuda_dense_fuse_and_packed_conflict(self, webbench):
        bundle = _dense_bundle(webbench, seed=6, in_width=24, widths=[16],
                               n_inputs_list=None, k=4, name='k53_cuda_conf')
        with pytest.raises(ValueError, match='dense_fuse'):
            generate_cuda(bundle, batch=32, dense_fuse=True,
                         dense_packed=True)


# ============================================================
# 2. 生成ソースの構造(GPUを起動しない、常時CI)
# ============================================================


class TestGeneratedSourceStructure:
    def test_metal_default_has_no_packed_kernel(self, webbench):
        """このモデルはbatch=32でdense_fuseの自動ON条件も満たすため
        (2層lut_dense+group_sumの鎖が共有メモリに収まる)、「packedカーネル
        だけが無い」旧来の既定挙動を見るには`auto='off'`で自動選択自体を
        切る必要がある(元テストの意図=dense_packed未指定時の挙動、を
        保ったまま更新)。"""
        bundle = _dense_bundle(webbench, seed=10, in_width=24, widths=[32, 24],
                               n_inputs_list=None, k=3, name='k53_struct1')
        text = generate_metal(bundle, name='m', batch=32, auto='off')['m.mm']
        assert 'mdl_metal_pack_batch(' not in text
        assert 'mdl_metal_lut_layer_packed' not in text
        assert 'mdl_metal_group_sum_packed(' not in text
        # in_width<=65536なので添字はu16版が選ばれる(
        # lut_dense_indices_fit_u16判定、非パック経路も同じ判定を使う)。
        assert 'mdl_metal_lut_layer_u16(' in text
        assert 'mdl_metal_group_sum(' in text

    def test_metal_dense_packed_true_emits_packed_calls(self, webbench):
        bundle = _dense_bundle(webbench, seed=11, in_width=24, widths=[32, 24],
                               n_inputs_list=None, k=3, name='k53_struct2')
        text = generate_metal(bundle, name='m', batch=32,
                              dense_packed=True)['m.mm']
        assert text.count('mdl_metal_pack_batch(') == 1
        assert text.count('mdl_metal_lut_layer_packed_u16(') == 2   # 2層
        assert text.count('mdl_metal_group_sum_packed(') == 1
        # 非パック経路の呼び出しはpackedモードでは一切出ない
        assert 'mdl_metal_lut_layer_u16(' not in text
        assert 'mdl_metal_lut_layer(' not in text
        assert 'mdl_metal_group_sum(' not in text

    def test_metal_dense_packed_u16_index_variant(self, webbench):
        """in_widthが小さい(<=65536)ので添字はu16版が選ばれる
        (lut_dense_indices_fit_u16判定をpacked経路でも共有)。"""
        bundle = _dense_bundle(webbench, seed=12, in_width=24, widths=[16],
                               n_inputs_list=None, k=2, name='k53_u16')
        text = generate_metal(bundle, name='m', batch=32,
                              dense_packed=True)['m.mm']
        assert 'mdl_metal_lut_layer_packed_u16(' in text

    def test_cuda_default_has_no_packed_kernel(self, webbench):
        """Metal版の同名テストと同じ理由で`auto='off'`にする
        (このモデルはbatch=32でdense_fuseの自動ON条件も満たすため)。"""
        bundle = _dense_bundle(webbench, seed=13, in_width=24, widths=[32, 24],
                               n_inputs_list=None, k=3, name='k53_cstruct1')
        text = generate_cuda(bundle, name='m', batch=32, auto='off')['m.cu']
        assert 'mdl_cuda_pack_batch(' not in text
        assert 'mdl_cuda_lut_layer_packed' not in text
        assert 'mdl_cuda_group_sum_packed(' not in text
        assert 'mdl_cuda_lut_layer_u16(' in text

    def test_cuda_dense_packed_true_emits_packed_calls(self, webbench):
        bundle = _dense_bundle(webbench, seed=14, in_width=24, widths=[32, 24],
                               n_inputs_list=None, k=3, name='k53_cstruct2')
        text = generate_cuda(bundle, name='m', batch=32,
                             dense_packed=True)['m.cu']
        assert text.count('mdl_cuda_pack_batch(') == 1
        assert text.count('mdl_cuda_lut_layer_packed_u16(') == 2
        assert text.count('mdl_cuda_group_sum_packed(') == 1
        assert 'mdl_cuda_lut_layer_u16(' not in text
        assert 'mdl_cuda_lut_layer(' not in text
        assert 'mdl_cuda_group_sum(' not in text


# ============================================================
# 3. 実機ビルド・実行(Metal、macOSのみ)
# ============================================================

from griffith_logic.deploy.metalverify import (  # noqa: E402
    RUNTIME_METAL,
    has_metal,
)

pytestmark_metal = pytest.mark.skipif(
    not has_metal(), reason='macOS + Metal のある環境でのみ実行')


def _batch_driver(n: int) -> str:
    """`test_dense_fuse.py::_metal_batch_driver`と同型(mdl_metal_infer_batch
    をn枚で1回呼ぶだけの最小ドライバ)。dense_packedはbatch>=32必須なので
    単枚(mdl_metal_infer)経路は使えない。"""
    return f'''#include "mdl_metal.h"
#include <stdio.h>
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


def _build_and_run(bundle: dict, tmp_path: Path, *, max_batch: int, n: int,
                   x: np.ndarray, dense_packed: bool,
                   tag: str) -> np.ndarray:
    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    files = generate_metal(bundle, name='model_metal', batch=max_batch,
                           dense_packed=dense_packed)
    for name_, text in files.items():
        (work / name_).write_text(text, encoding='utf-8')
    (work / 'driver.mm').write_text(_batch_driver(n), encoding='utf-8')
    build = subprocess.run(
        ['clang++', '-std=c++17', '-O2', '-fobjc-arc',
         f'-I{RUNTIME_METAL}', '-framework', 'Metal', '-framework',
         'Foundation', str(work / 'model_metal.mm'),
         str(RUNTIME_METAL / 'mdl_metal.mm'), str(work / 'driver.mm'),
         '-o', str(work / 'app')],
        capture_output=True, text=True)
    assert build.returncode == 0, f'ビルド失敗({tag}):\n{build.stderr}'
    proc = subprocess.run([str(work / 'app')], input=x.tobytes(),
                          capture_output=True)
    assert proc.returncode == 0, (
        f'実行失敗({tag}, exit={proc.returncode}): '
        f'{proc.stderr.decode(errors="replace")}')
    got = np.frombuffer(proc.stdout, dtype='<i4').astype(np.int64)
    return got


@pytestmark_metal
@pytest.mark.slow
class TestMetalDensePackedMatchesSimulate:
    _BATCHES = [32, 33, 64, 100, 128]

    def _check(self, bundle: dict, tmp_path: Path, *, batch: int,
              seed: int, tag: str) -> None:
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=(batch, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        got_packed = _build_and_run(
            bundle, tmp_path, max_batch=batch, n=batch, x=x,
            dense_packed=True, tag=f'{tag}_packed_b{batch}')
        n_out = int(graph.outputs[0].width)
        got_packed = got_packed.reshape(batch, n_out)
        assert np.array_equal(got_packed, want), (
            f'dense_packed が simulate と不一致(batch={batch}): '
            f'{int((got_packed != want).sum())}/{want.size} 要素不一致')

        got_plain = _build_and_run(
            bundle, tmp_path, max_batch=batch, n=batch, x=x,
            dense_packed=False, tag=f'{tag}_plain_b{batch}').reshape(
            batch, n_out)
        assert np.array_equal(got_plain, want)
        assert np.array_equal(got_packed, got_plain)

    @pytest.mark.parametrize('batch', _BATCHES)
    def test_pure_two_input_chain(self, webbench, tmp_path, batch):
        bundle = _dense_bundle(webbench, seed=101, in_width=48,
                               widths=[40, 32, 24], n_inputs_list=None,
                               k=4, name='k53_pure2')
        self._check(bundle, tmp_path, batch=batch, seed=200 + batch,
                   tag='pure2')

    @pytest.mark.parametrize('batch', _BATCHES)
    def test_mixed_arity_chain(self, webbench, tmp_path, batch):
        """n_inputs=2/3/4/5が層ごとに混在する鎖(dense_fuseのn_inputs==2
        決め打ちガードと違い、packed経路は3〜5もそのまま評価できる
        ─ CPU版MDL_LUT_LAYER_PACKED_BODYの汎用ループを踏襲しているため)。"""
        bundle = _dense_bundle(webbench, seed=102, in_width=40,
                               widths=[48, 40, 32, 24],
                               n_inputs_list=[2, 3, 4, 5], k=6,
                               name='k53_mixed')
        self._check(bundle, tmp_path, batch=batch, seed=300 + batch,
                   tag='mixed')

    def test_partial_batch_less_than_max(self, webbench, tmp_path):
        """確保した上限(128)より少ない枚数(33)で実行しても一致する
        ─ bwords=ceil(n/32)は実行時nから毎回計算するので、確保上限
        (plan.batch)とは独立に正しく振る舞う必要がある(設計の要)。"""
        bundle = _dense_bundle(webbench, seed=103, in_width=48,
                               widths=[40, 32], n_inputs_list=None, k=4,
                               name='k53_partial')
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        rng = np.random.default_rng(999)
        n = 33
        x = rng.integers(0, 2, size=(n, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)
        got = _build_and_run(bundle, tmp_path, max_batch=128, n=n, x=x,
                             dense_packed=True, tag='partial').reshape(
            n, n_out)
        assert np.array_equal(got, want)

    def test_single_layer_chain(self, webbench, tmp_path):
        """鎖の長さ1(lut_dense 1枚+group_sum)。dense_fuseは鎖長>=2を
        要求するが、packed経路は各層を独立に評価するので1層でも対象になる。"""
        bundle = _dense_bundle(webbench, seed=104, in_width=32, widths=[24],
                               n_inputs_list=None, k=3, name='k53_single')
        self._check(bundle, tmp_path, batch=32, seed=555, tag='single')


@pytestmark_metal
@pytest.mark.slow
class TestFaultInjection:
    """真理値表を壊すとdense_packedの出力が(壊していない)simulateの結果と
    食い違うことを確認する ─ 一致テストが空振りでないことの証拠。

    1ゲートだけ(tt[0]のみ)壊す設計は実際に試すと**空振りしうる**ことが
    判明した: 24ゲートの層で1本だけ壊しても、そのゲートの出力を後段の
    どのゲートも読んでいなければ(indicesがランダムなので約(23/24)^32≈26%
    の確率で起こりうる)、最終出力に一切影響しない
    ─ `ir/simulate.py`自身で確認済み(壊したゲートの生出力は完全に
    反転するのに、group_sum後の最終スコアは1ビットも変わらなかった)。
    そのため1層目の**全ゲート**の真理値表を反転させ、後段の最低1本かは
    ほぼ確実にそれを読むようにする(2^-24程度でしか空振りしない)。"""

    def test_corrupted_truth_table_is_detected(self, webbench, tmp_path):
        graph_ok = webbench.build_dense_ir(seed=201, in_width=32,
                                           widths=[24, 16], k=4,
                                           name='k53_fault_ok')
        doc_ok = webbench.build_model_doc(
            graph_ok, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
            labels=['a', 'b', 'c', 'd'])
        bundle_ok = build_bundle(doc_ok)

        # 壊す: 1層目の全ゲートの真理値表を反転(補数)させた壊れIRを作る
        # (ノード構造・添字は完全に同一、truth_tablesだけが違う)。
        import copy

        doc_bad = copy.deepcopy(doc_ok)
        n1 = next(n for n in doc_bad['ir']['nodes'] if n['id'] == 'n1')
        n_inputs = int(n1['params']['n_inputs'])
        mask = (1 << (1 << n_inputs)) - 1
        n1['params']['truth_tables'] = [
            f'0x{(~int(tt, 16)) & mask:x}'
            for tt in n1['params']['truth_tables']]
        bundle_bad = build_bundle(doc_bad)

        graph = ir_from_dict(bundle_ok['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        batch = 32
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=(batch, in_width), dtype=np.uint8)
        want = simulate(graph, x).astype(np.int64)

        got_bad = _build_and_run(
            bundle_bad, tmp_path, max_batch=batch, n=batch, x=x,
            dense_packed=True, tag='fault_bad').reshape(batch, n_out)
        assert not np.array_equal(got_bad, want), (
            '真理値表を1ビット壊しても出力が変わらなかった ─ 壊した'
            'tt[0]が実際にはpackedカーネルへ伝わっていない(添字/tt配線の'
            '欠陥をこのテストが検出できていない)')

        # 対照: 壊していないbundleは一致する(壊し方自体が無関係な差分を
        # 生んでいないことの確認)。
        got_ok = _build_and_run(
            bundle_ok, tmp_path, max_batch=batch, n=batch, x=x,
            dense_packed=True, tag='fault_ok').reshape(batch, n_out)
        assert np.array_equal(got_ok, want)
