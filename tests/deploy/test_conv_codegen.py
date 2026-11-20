"""conv2dのモデル特化コード生成(`--conv-codegen`)の等価性検証。

SASS解析(進捗記録節)で、
`mdl_k_conv_exec_split`(インタプリタ、命令列を実行時にバッファから読む)
には (1) ttが実行時値のためのマスク抽出LOP3×4、(2) `slots[8]` の動的添字
によるlocal memory落ち、という2つの理論的な残り余地が定量化された。
最小実験(scripts/gpu_calib/gen_conv_codegen_ab.py、Metal実機、
seg_convdlgn_k32の実ノード3本、5セット中央値、出力bit厳密一致確認済み)で
1.3倍ゲートを通過(interp比1.39〜1.44倍@batch=1、2.11〜2.56倍@batch大)した
ため、本実装(`convcodegen.py` + `metalbackend.py`/`cudabackend.py`の
`--conv-codegen`)へ進んだ。

このファイルは3段:
    1. `TestPureLogicEquivalence`: `convcodegen.py` が生成する
       GATE_EXPR文字列をPythonで評価した結果が、`test_conv_oc_split.py`の
       アドレス解決レベル実行(`_run_split_layout`)と一致することを見る
       (GPUを起動せずに常時CIで走る)。
    2. `TestMetalCodegenMatchesSimulate`: 実際にMSLをビルド・実行して
       `ir/simulate.py` と整数厳密一致するか、既定(インタプリタ)経路との
       新旧一致を見る(macOS + Metalのある環境のみ)。
    3. `TestDefaultUnchangedAndGuards`: `conv_codegen` 未指定/False時に
       生成物へ新カーネルが一切混じらないこと(既定動作不変)と、
       `conv_engine='combined'` との組み合わせを拒否すること。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.convcodegen import (
    CODEGEN_KERNEL_PREFIX,
    GATE_EXPR,
    build_switch_body,
    oc_function_body,
    ref_expr,
)
from griffith_logic.deploy.cudabackend import generate_cuda
from griffith_logic.deploy.layout import build_conv_layout_split
from griffith_logic.deploy.metalbackend import generate_metal
from griffith_logic.ir.serialize import ir_from_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / 'tools' / 'webbench' / 'public' / 'models'
REAL_MODEL_NAMES = [
    'seg_convdlgn_k32', 'seg_convdlgn_k128',
    'det_face_lut_k256', 'det_multimnist_mono_lut_k640',
]


def _model_bundle(name: str) -> dict | None:
    path = MODELS_DIR / f'{name}.json'
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding='utf-8'))
    return build_bundle(doc)


# ============================================================
# 1. 純粋ロジックの等価性(GPUを起動しない、常時CI)
# ============================================================


#: 生成される式は「1スレッド=1ワード内の1ビット位置i」を表す `sc[off + i]`
#: というC/MSL構文。テストでは `state` 配列が既に(scratch_words, n_trials)
#: の「1trial=1論理ビット値」でベクトル化されている(test_conv_oc_split.py
#: の `_run_split_layout` と同じ流儀)ので、`+ i` は取り除いて
#: `sc[off]`(行=state配列の該当ワード全体)に読み替える。
_SC_IDX_RE = re.compile(r'sc\[(\d+) \+ i\]')
#: `0u` / `0xffffffffu` のC風符号無しサフィックスはPythonの整数リテラルに
#: 無いので取り除く(数値としての意味は変わらない)。
_UINT_SUFFIX_RE = re.compile(r'\b(0x[0-9a-fA-F]+|\d+)u\b')


def _sanitize_for_python(expr: str) -> str:
    expr = _SC_IDX_RE.sub(r'sc[\1]', expr)
    expr = _UINT_SUFFIX_RE.sub(r'\1', expr)
    return expr


def _eval_codegen_oc(layout, oc: int, state: np.ndarray) -> np.ndarray:
    """`oc_function_body`が返すGATE_EXPR文字列をnumpy boolで評価する。

    生成される式はビット演算(&, |, ^, ~)のみ(`cbackend.GATE_EXPR`)なので、
    Python の bool ndarray でも同じ演算子がそのまま使える。
    """
    n_trials = state.shape[1]

    def _as_bool_array(v) -> np.ndarray:
        # 全定数の部分式(両オペランドがCONST平面)はPythonのbare intに
        # なりうる(例: `~0` は numpy無しだと-1)。実配列と揃えるため、
        # ndarrayでなければ真偽値へ落としてから複製する。
        if isinstance(v, np.ndarray):
            return v.astype(bool)
        return np.full(n_trials, bool(v))

    body, final_expr = oc_function_body(layout, oc)
    ns: dict[str, np.ndarray] = {'sc': state}
    lines = [ln for ln in body.split('\n') if ln.strip()
            and not ln.strip().startswith('unsigned int')]
    for line in lines:
        stmt = line.strip()
        assert stmt.endswith(';')
        m = re.match(r'(s\d+)\s*=', stmt)
        assert m, f'想定外の代入文: {stmt}'
        var = m.group(1)
        exec(_sanitize_for_python(stmt[:-1]), {}, ns)  # noqa: S102
        ns[var] = _as_bool_array(ns[var])
    return _as_bool_array(eval(_sanitize_for_python(final_expr), {}, ns))  # noqa: S307


def _run_codegen_layout(layout, state: np.ndarray) -> list[np.ndarray]:
    return [_eval_codegen_oc(layout, oc, state)
           for oc in range(layout.out_channels)]


def _run_split_layout_ref(layout, state: np.ndarray) -> list[np.ndarray]:
    """`test_conv_oc_split.py::_run_split_layout` と同じデコード規則。"""
    from griffith_logic.export.optimize import _gate2

    outs: list[np.ndarray] = []
    for oc in range(layout.out_channels):
        start, end = layout.oc_instr_start[oc], layout.oc_instr_start[oc + 1]
        slots = [None] * max(1, layout.max_slots_per_oc)

        def rv(packed: int, _slots=slots) -> np.ndarray:
            if packed & 1:
                v = _slots[packed >> 1]
                assert v is not None
                return v
            return state[packed >> 1]

        for k in range(start, end):
            tt, ap, bp, dst = layout.instr_flat[4 * k:4 * k + 4]
            slots[dst] = _gate2(tt, rv(ap), rv(bp))
        outs.append(rv(layout.oc_final_ref[oc]))
    return outs


def _make_shared_scratch(layout, rng: np.random.Generator,
                         n_trials: int) -> np.ndarray:
    """定数0/1平面(mdl_k_conv_constの不変条件)+タップ平面プールへ
    ランダム内容を詰めた共有スクラッチ。codegenのoff==0/off==trw即値化
    最適化はこの不変条件が前提(最小実験で一度これを破って偽の不一致を
    出した実例あり、`scripts/gpu_calib/gen_conv_codegen_ab.py` 参照)。
    """
    state = np.zeros((layout.scratch_words, n_trials), dtype=bool)
    state[layout.tap_rows_words:2 * layout.tap_rows_words, :] = True
    pool_end = layout.pool_base + layout.n_planes * layout.plane_stride
    if pool_end > layout.pool_base:
        state[layout.pool_base:pool_end, :] = rng.integers(
            0, 2, size=(pool_end - layout.pool_base, n_trials)).astype(bool)
    return state


class TestPureLogicEquivalence:
    """`convcodegen.py`が生成する式 ⇔ インタプリタのアドレス解決レベル実行。"""

    @pytest.mark.parametrize('name', REAL_MODEL_NAMES)
    def test_real_model_conv_nodes(self, name):
        bundle = _model_bundle(name)
        if bundle is None:
            pytest.skip(f'配信モデル {name} が無い環境')
        graph = ir_from_dict(bundle['ir'])
        rng = np.random.default_rng(hash(name) & 0xffff)
        checked = 0
        for node in graph.nodes:
            if node.type != 'conv2d':
                continue
            layout = build_conv_layout_split(node.params,
                                             bundle['programs'][node.id])
            state = _make_shared_scratch(layout, rng, n_trials=24)
            want = _run_split_layout_ref(layout, state)
            got = _run_codegen_layout(layout, state)
            for oc, (w, g) in enumerate(zip(want, got, strict=True)):
                assert np.array_equal(w, g), (
                    f'{name}/{node.id}: oc={oc} でcodegen式が'
                    'インタプリタと不一致')
            checked += 1
        assert checked > 0, f'{name}: conv2dノードが無い'

    def test_ref_expr_uses_const_invariant(self):
        """off==0/off==trwはCONST-0/1平面の即値化(mdl_k_conv_constの
        不変条件前提)、それ以外はsc[off+i]。"""
        assert ref_expr(0 << 1, trw=16) == '0u'
        assert ref_expr(16 << 1, trw=16) == '0xffffffffu'
        assert ref_expr(20 << 1, trw=16) == 'sc[20 + i]'
        assert ref_expr((3 << 1) | 1, trw=16) == 's3'

    def test_gate_expr_table_matches_cbackend(self):
        """CPUと同じ表を流用している(共有化の固定)。"""
        from griffith_logic.deploy.cbackend import GATE_EXPR as CPU_TABLE

        assert GATE_EXPR is CPU_TABLE

    def test_switch_body_lists_every_oc_once(self):
        bundle = _model_bundle('seg_convdlgn_k32')
        if bundle is None:
            pytest.skip('配信モデル seg_convdlgn_k32 が無い環境')
        graph = ir_from_dict(bundle['ir'])
        node = next(n for n in graph.nodes if n.type == 'conv2d')
        layout = build_conv_layout_split(node.params,
                                         bundle['programs'][node.id])
        body = build_switch_body(layout, lambda oc: f'fn{oc}')
        for oc in range(layout.out_channels):
            assert f'case {oc}: v = fn{oc}(scratch, i); break;' in body


# ============================================================
# 2. 生成ソースの構造(GPUを起動しない、常時CI)
# ============================================================


class TestGeneratedSourceStructure:
    def test_metal_default_has_no_codegen_kernel(self):
        """既定(conv_codegen省略)は従来のインタプリタ経路のまま
        (既定動作を変えない、というタスクの制約の固定)。"""
        bundle = _model_bundle('seg_convdlgn_k32')
        if bundle is None:
            pytest.skip('配信モデル seg_convdlgn_k32 が無い環境')
        text = generate_metal(bundle, name='m')['m.mm']
        assert CODEGEN_KERNEL_PREFIX not in text
        assert 'mdl_metal_conv_exec_named' not in text
        assert 'mdl_metal_conv_split(' in text

    def test_metal_conv_codegen_true_emits_kernel_per_node(self):
        bundle = _model_bundle('seg_convdlgn_k32')
        if bundle is None:
            pytest.skip('配信モデル seg_convdlgn_k32 が無い環境')
        graph = ir_from_dict(bundle['ir'])
        n_conv = sum(1 for n in graph.nodes if n.type == 'conv2d')
        text = generate_metal(bundle, name='m', conv_codegen=True)['m.mm']
        assert text.count(f'kernel void {CODEGEN_KERNEL_PREFIX}') == n_conv
        assert text.count('mdl_metal_conv_exec_named(') == n_conv
        # インタプリタの命令列アップロード(instr/oc_final)は
        # codegen経路では生成されない(実行時に読まないため)。
        assert 'mdl_metal_conv_split(' not in text

    def test_cuda_default_has_no_codegen_kernel(self):
        bundle = _model_bundle('seg_convdlgn_k32')
        if bundle is None:
            pytest.skip('配信モデル seg_convdlgn_k32 が無い環境')
        text = generate_cuda(bundle, name='m')['m.cu']
        assert CODEGEN_KERNEL_PREFIX not in text
        assert 'mdl_cuda_conv_split(' in text

    def test_cuda_conv_codegen_true_emits_kernel_per_node(self):
        bundle = _model_bundle('seg_convdlgn_k32')
        if bundle is None:
            pytest.skip('配信モデル seg_convdlgn_k32 が無い環境')
        graph = ir_from_dict(bundle['ir'])
        n_conv = sum(1 for n in graph.nodes if n.type == 'conv2d')
        text = generate_cuda(bundle, name='m', conv_codegen=True)['m.cu']
        assert text.count(f'__global__ void {CODEGEN_KERNEL_PREFIX}') == n_conv
        assert 'mdl_cuda_conv_split(' not in text

    def test_conv_codegen_rejects_combined_engine(self, seg_doc):
        bundle = build_bundle(seg_doc)
        with pytest.raises(ValueError, match='conv_codegen'):
            generate_metal(bundle, conv_engine='combined', conv_codegen=True)
        with pytest.raises(ValueError, match='conv_codegen'):
            generate_cuda(bundle, conv_engine='combined', conv_codegen=True)


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
class TestMetalCodegenMatchesSimulate:
    def test_seg_doc_codegen_matches_simulate(self, seg_doc, tmp_path):
        result = verify_metal_matches_simulate(
            build_bundle(seg_doc), n_samples=3, work_dir=tmp_path,
            verbose=False, conv_codegen=True)
        assert result['max_abs_diff'] == 0

    def test_codegen_and_interp_agree(self, seg_doc, tmp_path):
        """codegen ON/OFFが互いに一致し、両方ともsimulateと一致する
        (test_metal_conv_split.pyのsplit/combined一致テストと同じ形)。"""
        from griffith_logic.ir.simulate import simulate

        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        in_width = int(graph.inputs[0].width)
        n_out = int(graph.outputs[0].width)
        rng = np.random.default_rng(11)
        x = rng.integers(0, 2, size=in_width, dtype=np.uint8)
        want = simulate(graph, x[None])[0].astype(np.int64)

        exe_interp = build_metal_model(bundle, tmp_path / 'interp',
                                       conv_codegen=False)
        exe_codegen = build_metal_model(bundle, tmp_path / 'codegen',
                                        conv_codegen=True)
        got_interp = run_metal_model(exe_interp, x.tobytes(),
                                     n_out).astype(np.int64)
        got_codegen = run_metal_model(exe_codegen, x.tobytes(),
                                      n_out).astype(np.int64)
        assert np.array_equal(got_interp, want)
        assert np.array_equal(got_codegen, want)
        assert np.array_equal(got_interp, got_codegen)

    @pytest.mark.parametrize('name', REAL_MODEL_NAMES)
    def test_real_models_codegen_matches_simulate(self, name, tmp_path):
        bundle = _model_bundle(name)
        if bundle is None:
            pytest.skip(f'配信モデル {name} が無い環境')
        result = verify_metal_matches_simulate(
            bundle, n_samples=2, work_dir=tmp_path / name, verbose=False,
            conv_codegen=True)
        assert result['max_abs_diff'] == 0
