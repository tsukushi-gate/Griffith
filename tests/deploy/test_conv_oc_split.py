"""conv2dの「ocごとの命令列分割」レイアウトの等価性検証(Step1)。

WebGPU版に入った「ocごとの簡約済み命令列」をCUDA/Metalネイティブ
ランタイムへ移植する前段として、Python側のスライス変換
(`griffith_logic/deploy/layout.py::split_program_by_oc` /
`build_conv_layout_split`)が**全oc結合の命令列と全数一致する**ことを、
ランタイム(GPUカーネル)を1行も書く前に固定する。

2段階で検証する:

    1. ref参照レベル(アドレス解決前): `split_program_by_oc` が返す
       ocごとのローカル命令列を素朴に解釈実行し、全oc結合の
       `ConvProgram` をそのまま解釈実行した結果と全数一致することを見る
       (`TestRefLevelEquivalence`)。ここが変換ロジックの核心。
    2. アドレス解決レベル: `build_conv_layout` / `build_conv_layout_split`
       が実際に作る絶対ワードオフセット/パック済みref込みで、共有スクラッチ
       (定数0/1平面+タップ平面プール)に対して**同じランダム内容**を
       与えたときに両者が同じ最終値を書くことを見る
       (`TestAddressLevelEquivalence`)。ここは実際のCUDA/Metalカーネルが
       やる仕事(`sc[addr+i]` / `slots[ix]` のデコード)をPythonで先取りして
       検証している。

対象は実モデル(seg_convdlgn_k32等、無ければ自動skip)+ランダム真理値表の
合成conv(hypothesis property test)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from griffith_logic.deploy import build_bundle
from griffith_logic.deploy.layout import (
    CONV_SPLIT_MAX_SLOTS,
    ConvLayout,
    ConvLayoutSplit,
    OcProgramSplit,
    build_conv_layout,
    build_conv_layout_split,
    conv_layout_split_stats,
    split_program_by_oc,
)
from griffith_logic.deploy.programs import program_from_dict, program_to_dict
from griffith_logic.export.optimize import (
    REF_KIND_CONST,
    REF_KIND_TAP,
    ConvProgram,
    ConvTreeSpec,
    _gate2,
    build_conv_program,
    ref_idx,
    ref_kind,
)
from griffith_logic.ir.serialize import ir_from_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / 'tools' / 'webbench' / 'public' / 'models'

REAL_MODEL_NAMES = [
    'seg_convdlgn_k32', 'seg_convdlgn_k128',
    'det_face_lut_k256', 'det_multimnist_mono_lut_k640',
]


def _model_path(name: str) -> Path:
    return MODELS_DIR / f'{name}.json'


# ============================================================
# 1. ref参照レベルの解釈実行(アドレス解決前)
# ============================================================


def _eval_combined_ref_level(prog: ConvProgram,
                             tap_vectors: list[np.ndarray]) -> list[np.ndarray]:
    """全oc結合の ConvProgram を素朴に(全命令0..n-1の順で)解釈実行する。

    実際の mdl_k_conv_exec(write_start/write_ocのCSRで逐次書き出す)と
    「命令を順に評価する」骨格は同じだが、ここでは全命令の値を
    保持しておいて最後に oc_out を読むだけの単純な形にしてある
    (書き出しタイミングの都合はアドレス解決レベルの方で見る)。
    """
    n = tap_vectors[0].shape[0] if tap_vectors else 1
    vals: dict[int, np.ndarray] = {}

    def rv(r: int) -> np.ndarray:
        kind = ref_kind(r)
        if kind == REF_KIND_CONST:
            return np.full(n, bool(ref_idx(r)))
        if kind == REF_KIND_TAP:
            return tap_vectors[ref_idx(r)]
        return vals[ref_idx(r)]

    for j, ins in enumerate(prog.instrs):
        vals[j] = _gate2(ins.tt, rv(ins.a), rv(ins.b))
    return [rv(r) for r in prog.oc_out]


def _eval_split_ref_level(splits: list[OcProgramSplit],
                          tap_vectors: list[np.ndarray]) -> list[np.ndarray]:
    """`split_program_by_oc` が返すocごとのローカル命令列を解釈実行する。"""
    n = tap_vectors[0].shape[0] if tap_vectors else 1
    outs: list[np.ndarray] = []
    for sp in splits:
        local: list[np.ndarray] = []

        def rv(r: int, _local=local) -> np.ndarray:
            kind = ref_kind(r)
            if kind == REF_KIND_CONST:
                return np.full(n, bool(ref_idx(r)))
            if kind == REF_KIND_TAP:
                return tap_vectors[ref_idx(r)]
            return _local[ref_idx(r)]

        for ins in sp.instrs:
            local.append(_gate2(ins.tt, rv(ins.a), rv(ins.b)))
        outs.append(rv(sp.out_ref))
    return outs


def _assert_ref_level_matches(prog: ConvProgram, *, n_trials: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tap_vectors = [rng.integers(0, 2, size=n_trials).astype(bool)
                  for _ in prog.taps]
    want = _eval_combined_ref_level(prog, tap_vectors)
    splits = split_program_by_oc(prog)
    assert len(splits) == len(prog.oc_out)
    got = _eval_split_ref_level(splits, tap_vectors)
    for oc, (w, g) in enumerate(zip(want, got, strict=True)):
        assert np.array_equal(w, g), (
            f'oc={oc} でref解釈実行が不一致(seed={seed}, n_trials={n_trials})')


class TestRefLevelEquivalence:
    """`split_program_by_oc` の純粋なグラフ変換としての正しさ。"""

    def test_known_passthrough_tree(self):
        """全パススルー(tt=0xa)木: oc_outが直接タップを指す(命令ゼロ)。"""
        spec = ConvTreeSpec(
            out_channels=2, tree_depth=2,
            indices_flat=[[10, 11, 12, 13], [20, 21, 22, 23]],
            truth_tables=[0xa, 0xa, 0xa, 0xa, 0xa, 0xa],
        )
        prog = build_conv_program(spec)
        splits = split_program_by_oc(prog)
        for sp in splits:
            assert sp.instrs == []
            assert sp.n_slots == 0
            assert ref_kind(sp.out_ref) == REF_KIND_TAP
        _assert_ref_level_matches(prog, n_trials=64, seed=0)

    @pytest.mark.parametrize('tt_const,expect_bit', [(0x0, False), (0xf, True)])
    def test_known_const_tree(self, tt_const, expect_bit):
        """全定数木: oc_outが直接CONSTを指す(命令ゼロ、タップも読まない)。"""
        spec = ConvTreeSpec(
            out_channels=3, tree_depth=2,
            indices_flat=[[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]],
            truth_tables=[tt_const] * 9,
        )
        prog = build_conv_program(spec)
        splits = split_program_by_oc(prog)
        for sp in splits:
            assert sp.instrs == []
            assert ref_kind(sp.out_ref) == REF_KIND_CONST
            assert bool(ref_idx(sp.out_ref)) == expect_bit
        _assert_ref_level_matches(prog, n_trials=8, seed=1)

    @given(
        depth=st.integers(min_value=1, max_value=4),
        oc=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        data=st.data(),
    )
    @settings(max_examples=60, deadline=None)
    def test_random_tree_matches_combined_program(self, depth, oc, seed, data):
        """ランダム真理値表の合成木: 分割前後で全ワード一致する(property test)。

        タップIDのプールをわざと狭くして、複数ocが同じタップ/同じ部分木を
        共有するケース(CSE・複数ocが同一rootを指すケース)を誘発する。
        """
        n_leaves = 1 << depth
        n_internal = n_leaves - 1
        n_tap_pool = data.draw(
            st.integers(min_value=1, max_value=max(1, n_leaves * oc // 2)),
            label='n_tap_pool',
        )
        indices_flat = data.draw(
            st.lists(
                st.lists(st.integers(min_value=0, max_value=n_tap_pool - 1),
                        min_size=n_leaves, max_size=n_leaves),
                min_size=oc, max_size=oc,
            ),
            label='indices_flat',
        )
        truth_tables = data.draw(
            st.lists(st.integers(min_value=0, max_value=15),
                    min_size=oc * n_internal, max_size=oc * n_internal),
            label='truth_tables',
        )
        spec = ConvTreeSpec(out_channels=oc, tree_depth=depth,
                            indices_flat=indices_flat, truth_tables=truth_tables)
        prog = build_conv_program(spec)
        _assert_ref_level_matches(prog, n_trials=32, seed=seed)

        # ローカルスロット数の健全性: 命令が無ければ0、あれば1以上。
        splits = split_program_by_oc(prog)
        for sp in splits:
            if sp.instrs:
                assert 1 <= sp.n_slots <= len(sp.instrs)
            else:
                assert sp.n_slots == 0

    @pytest.mark.parametrize('name', REAL_MODEL_NAMES)
    def test_real_model_conv_nodes(self, name):
        """実モデルのconv2dノード全部でref解釈実行が一致する。"""
        path = _model_path(name)
        if not path.is_file():
            pytest.skip(f'配信モデル {name} が無い環境')
        doc = json.loads(path.read_text(encoding='utf-8'))
        bundle = build_bundle(doc)
        graph = ir_from_dict(bundle['ir'])
        checked = 0
        for node in graph.nodes:
            if node.type != 'conv2d':
                continue
            prog = program_from_dict(bundle['programs'][node.id])
            _assert_ref_level_matches(prog, n_trials=16, seed=hash(node.id) & 0xffff)
            checked += 1
        assert checked > 0, f'{name}: conv2dノードが1つも無い'


# ============================================================
# 2. アドレス解決レベルの解釈実行(実際のGPUカーネルが読む形)
# ============================================================


def _make_shared_scratch(layout: ConvLayout, rng: np.random.Generator,
                         n_trials: int) -> np.ndarray:
    """定数0/1平面+タップ平面プールへランダム内容を詰めた共有スクラッチ。

    `ConvLayout`(全oc結合)の scratch_words は reg 平面ぶん `ConvLayoutSplit`
    より必ず大きい(同じ `_conv_geometry` を使っているため pool_base/タップ
    プールは完全に同一)ので、この1本を両方の解釈実行で使い回せる。
    """
    state = np.zeros((layout.scratch_words, n_trials), dtype=bool)
    state[layout.tap_rows_words:2 * layout.tap_rows_words, :] = True
    pool_end = layout.pool_base + layout.n_planes * layout.plane_stride
    if pool_end > layout.pool_base:
        state[layout.pool_base:pool_end, :] = rng.integers(
            0, 2, size=(pool_end - layout.pool_base, n_trials)).astype(bool)
    return state


def _run_combined_layout(layout: ConvLayout, state: np.ndarray) -> list[np.ndarray]:
    """`instr_flat`(絶対アドレス、reg平面へ書く)を順に実行し oc_src_off を読む。

    combined設計はレジスタ平面(slot)を全oc・全命令にわたって使い回す
    (生存区間解析で自由リストから取り合う)ので、**write_start/write_ocの
    CSRが指すタイミングで即座に読み出さないと**、そのスロットが後続の
    命令で上書きされてから読むことになり値が壊れる
    (実際の mdl_k_conv_exec / mdl_k_conv_exec の write_start ループと
    厳密に同じ実行順にする必要がある — 全命令を先に流してから
    oc_src_off を読むのは誤り)。
    """
    st_ = state.copy()
    flat = layout.instr_flat
    ws = layout.write_start
    woc = layout.write_oc
    outs: list[np.ndarray | None] = [None] * layout.out_channels

    def flush(bucket_start: int, bucket_end: int) -> None:
        for b in range(bucket_start, bucket_end):
            oc = woc[b]
            outs[oc] = st_[layout.oc_src_off[oc]].copy()

    flush(ws[0], ws[1])
    for j in range(layout.n_instrs):
        tt, a, b, d = flat[4 * j:4 * j + 4]
        st_[d] = _gate2(tt, st_[a], st_[b])
        flush(ws[j + 1], ws[j + 2])
    assert all(o is not None for o in outs), 'write_start/write_ocに漏れがある'
    return outs


def _run_split_layout(layout: ConvLayoutSplit, state: np.ndarray) -> list[np.ndarray]:
    """パック済みref(`(value<<1)|is_reg`)をデコードしながらocごとに実行する。

    実際の mdl_k_conv_exec_split カーネル(1スレッド=(oc,ワード列))と
    同じデコード規則: bit0=0→共有スクラッチのアドレス、bit0=1→
    このoc内ローカルなレジスタ配列の添字。
    """
    n_trials = state.shape[1]
    flat = layout.instr_flat
    outs: list[np.ndarray] = []
    for oc in range(layout.out_channels):
        start, end = layout.oc_instr_start[oc], layout.oc_instr_start[oc + 1]
        slots = [None] * max(1, layout.max_slots_per_oc)

        def rv(packed: int, _slots=slots) -> np.ndarray:
            if packed & 1:
                v = _slots[packed >> 1]
                assert v is not None, '未計算のローカルスロットを読んだ'
                return v
            return state[packed >> 1]

        for k in range(start, end):
            tt, ap, bp, dst = flat[4 * k:4 * k + 4]
            slots[dst] = _gate2(tt, rv(ap), rv(bp))
        outs.append(rv(layout.oc_final_ref[oc]))
    return outs


def _assert_address_level_matches(params: dict, prog_dict: dict, *,
                                  n_trials: int, seed: int) -> None:
    layout = build_conv_layout(params, prog_dict)
    layout_split = build_conv_layout_split(params, prog_dict)
    rng = np.random.default_rng(seed)
    state = _make_shared_scratch(layout, rng, n_trials)
    want = _run_combined_layout(layout, state)
    got = _run_split_layout(layout_split, state)
    assert len(want) == len(got) == layout.out_channels
    for oc, (w, g) in enumerate(zip(want, got, strict=True)):
        assert np.array_equal(w, g), (
            f'oc={oc}: アドレス解決レベルで combined ⇔ split 不一致'
            f'(seed={seed})')


class TestAddressLevelEquivalence:
    """`build_conv_layout` vs `build_conv_layout_split` の実行結果一致。"""

    def test_seg_doc_conv_nodes(self, seg_doc):
        """conftest.pyのseg_doc(小さな合成セグIR、命令列が実際に残る)。"""
        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        checked = 0
        for node in graph.nodes:
            if node.type != 'conv2d':
                continue
            _assert_address_level_matches(
                node.params, bundle['programs'][node.id],
                n_trials=48, seed=hash(node.id) & 0xffff)
            checked += 1
        assert checked > 0

    @pytest.mark.slow
    @pytest.mark.parametrize('name', REAL_MODEL_NAMES)
    def test_real_model_conv_nodes(self, name):
        path = _model_path(name)
        if not path.is_file():
            pytest.skip(f'配信モデル {name} が無い環境')
        doc = json.loads(path.read_text(encoding='utf-8'))
        bundle = build_bundle(doc)
        graph = ir_from_dict(bundle['ir'])
        checked = 0
        for node in graph.nodes:
            if node.type != 'conv2d':
                continue
            _assert_address_level_matches(
                node.params, bundle['programs'][node.id],
                n_trials=24, seed=hash(node.id) & 0xffff)
            checked += 1
        assert checked > 0, f'{name}: conv2dノードが1つも無い'

    @given(
        depth=st.integers(min_value=1, max_value=3),
        oc=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        data=st.data(),
    )
    @settings(max_examples=25, deadline=None)
    def test_random_synthetic_conv_node(self, depth, oc, seed, data):
        """ランダム真理値表の合成convを、実際の(params, program)形へ組んで検証する。

        kernel_size を tree_depth に合わせて選ぶことで(in_channels=1、
        kh*kw==n_leaves)、indices_flat がそのまま受容野内の平坦位置になる
        (`spec_from_node_params`/`build_conv_layout` の前提と一致させる)。
        """
        n_leaves = 1 << depth
        n_internal = n_leaves - 1
        kh = 1
        kw = n_leaves
        indices_flat = data.draw(
            st.lists(
                st.lists(st.integers(min_value=0, max_value=n_leaves - 1),
                        min_size=n_leaves, max_size=n_leaves),
                min_size=oc, max_size=oc,
            ),
            label='indices_flat',
        )
        truth_tables = data.draw(
            st.lists(st.integers(min_value=0, max_value=15),
                    min_size=oc * n_internal, max_size=oc * n_internal),
            label='truth_tables',
        )
        spec = ConvTreeSpec(out_channels=oc, tree_depth=depth,
                            indices_flat=indices_flat, truth_tables=truth_tables)
        prog = build_conv_program(spec)
        prog_dict = program_to_dict(prog)
        params = {
            'in_shape': [1, 1, kw], 'kernel_size': [kh, kw],
            'stride': 1, 'padding': 0, 'out_channels': oc,
        }
        _assert_address_level_matches(params, prog_dict, n_trials=32, seed=seed)


# ============================================================
# 3. 統計・ガードレール
# ============================================================


class TestStatsAndGuard:
    @pytest.mark.parametrize('name', REAL_MODEL_NAMES)
    def test_stats_are_sane_for_real_models(self, name):
        path = _model_path(name)
        if not path.is_file():
            pytest.skip(f'配信モデル {name} が無い環境')
        doc = json.loads(path.read_text(encoding='utf-8'))
        bundle = build_bundle(doc)
        graph = ir_from_dict(bundle['ir'])
        for node in graph.nodes:
            if node.type != 'conv2d':
                continue
            prog_dict = bundle['programs'][node.id]
            layout_split = build_conv_layout_split(node.params, prog_dict)
            stats = conv_layout_split_stats(prog_dict, layout_split)
            assert stats['n_instrs_split_total'] >= stats['n_instrs_combined']
            assert stats['duplication_pct'] >= 0.0
            assert 0 <= stats['max_slots_per_oc'] <= CONV_SPLIT_MAX_SLOTS
            assert 0 <= stats['zero_instr_oc_count'] <= stats['out_channels']

    def test_max_slots_guard_rejects_oversized_layout(self, seg_doc, monkeypatch):
        """CONV_SPLIT_MAX_SLOTSを実測未満へ下げると、明示的にValueErrorで落ちる。

        「未検証のレジスタ数を黙って使わない」という設計方針の固定
        (タスク指示: 上限を超えたら生成時に落とす)。
        """
        import griffith_logic.deploy.layout as layout_mod

        bundle = build_bundle(seg_doc)
        graph = ir_from_dict(bundle['ir'])
        conv_nodes = [n for n in graph.nodes if n.type == 'conv2d']
        assert conv_nodes, 'seg_docにconv2dノードが無い'

        # 実際のmax_slots_per_ocを一度測ってから、それ未満へ下げる。
        node = conv_nodes[0]
        actual = build_conv_layout_split(node.params, bundle['programs'][node.id])
        monkeypatch.setattr(layout_mod, 'CONV_SPLIT_MAX_SLOTS',
                            max(0, actual.max_slots_per_oc - 1))
        if actual.max_slots_per_oc <= 0:
            pytest.skip('このノードは命令ゼロでスロットを使わない')
        with pytest.raises(ValueError, match='CONV_SPLIT_MAX_SLOTS'):
            layout_mod.build_conv_layout_split(node.params,
                                               bundle['programs'][node.id])
