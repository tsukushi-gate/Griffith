"""最適化の自動選択(auto-config)の判定表テスト。

`plan.py::resolve_auto_config`の各判定(conv_engine自動フォールバック・
dense_fuse自動ON・dense_packed自動ON・両者の優先順位)と、
`generate_metal`/`generate_cuda`の`auto`引数による合成(自動選択の既定ON・
明示引数が常に勝つ・`auto='off'`で無効化)を、GPUを起動せず構造だけで
固定する(このファイルは常時CI)。判定表そのもの・実測根拠は
`docs/guides/deploy-optimization.md` 参照。

このファイルは3段:
    1. `TestResolveAutoConfigDenseFuse`/`DensePacked`/`ConvEngine`:
       `resolve_auto_config`単体の判定ロジック(閾値の境界・優先順位・
       理由文字列)。
    2. `TestGenerateAutoWiring`: `generate_metal`/`generate_cuda`が
       `resolve_auto_config`をどう使うか(既定=auto ON・明示が勝つ・
       `auto='off'`で無効化・生成ソース冒頭コメント・標準出力サマリ)。
    3. `TestMegakernelDefaultChanged`: decode megakernelの既定値が
       False→Trueへ変わったことの回帰固定(rddlgn系は別ファイル
       `test_rddlgn_decode_megakernel.py`が実機込みで詳しく見るので、
       ここでは既定値の変更点だけを軽く確認する)。
"""

from __future__ import annotations

import numpy as np
import pytest

from griffith_logic.deploy.bundle import build_bundle
from griffith_logic.deploy.cudabackend import generate_cuda
from griffith_logic.deploy.densefuse import CODEGEN_KERNEL_PREFIX
from griffith_logic.deploy.metalbackend import generate_metal
from griffith_logic.deploy.plan import (
    DENSE_FUSE_AUTO_MIN_BATCH,
    DENSE_PACKED_AUTO_MIN_BATCH,
    build_plan,
    resolve_auto_config,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort


def _dense_bundle(webbench, *, seed: int, in_width: int, widths: list[int],
                  k: int, name: str) -> dict:
    """n_inputs=2固定の純denseバンドル(dense_fuseが融合できる形)。"""
    graph = webbench.build_dense_ir(seed=seed, in_width=in_width,
                                    widths=widths, k=k, name=name)
    doc = webbench.build_model_doc(
        graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
        labels=[f'c{i}' for i in range(k)])
    return build_bundle(doc)


def _mixed_arity_dense_bundle(webbench, *, seed: int, in_width: int,
                              widths: list[int], n_inputs_list: list[int],
                              k: int, name: str) -> dict:
    """n_inputsが2以外を含む純denseバンドル(dense_packedはeligibleだが、
    dense_fuseのn_inputs==2ガードで融合対象外になる形。
    `test_metal_dense_packed.py::_mixed_arity_dense_ir`と同じ手組み方式)。
    """
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
    graph = IRGraph(
        metadata={'name': name},
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=k, semantics='class_scores')],
        nodes=nodes, edges=edges,
    )
    doc = webbench.build_model_doc(
        graph, preproc={'kind': 'thermometer', 'input_kind': 'bits'},
        labels=[f'c{i}' for i in range(k)])
    return build_bundle(doc)


# ============================================================
# 1. resolve_auto_config: 判定ロジック単体(GPUを起動しない)
# ============================================================

class TestResolveAutoConfigDenseFuse:
    def test_at_threshold_batch_enables_and_packed_is_skipped(self, webbench):
        bundle = _dense_bundle(webbench, seed=1, in_width=24, widths=[32, 24],
                               k=3, name='ac_fuse_ok')
        batch = DENSE_FUSE_AUTO_MIN_BATCH
        plan = build_plan(bundle, batch=batch)
        decision = resolve_auto_config(plan, batch=batch, backend='metal')
        assert decision['dense_fuse'] is True
        # dense_fuseとdense_packedは併用不可 ─ dense_fuseが自動ONになった
        # ら、dense_packedは見送られる(優先順位)。
        assert decision['dense_packed'] is False
        assert '見送り' in decision['reasons']['dense_packed']

    def test_below_threshold_batch_does_not_enable(self, webbench):
        bundle = _dense_bundle(webbench, seed=2, in_width=24, widths=[32, 24],
                               k=3, name='ac_fuse_low')
        batch = DENSE_FUSE_AUTO_MIN_BATCH - 1
        plan = build_plan(bundle, batch=batch)
        decision = resolve_auto_config(plan, batch=batch, backend='metal')
        assert decision['dense_fuse'] is False
        assert str(batch) in decision['reasons']['dense_fuse']

    def test_single_layer_chain_never_fuses(self, webbench):
        """鎖の長さが1(lut_dense1枚+group_sum)は融合対象にならない ─
        batchが十分大きくてもdense_fuseはONにならない。"""
        bundle = _dense_bundle(webbench, seed=3, in_width=24, widths=[16],
                               k=4, name='ac_fuse_single')
        plan = build_plan(bundle, batch=256)
        decision = resolve_auto_config(plan, batch=256, backend='metal')
        assert decision['dense_fuse'] is False
        assert '融合可能な鎖なし' in decision['reasons']['dense_fuse']

    def test_conv_model_never_fuses(self, seg_doc):
        bundle = build_bundle(seg_doc)
        plan = build_plan(bundle, batch=256)
        decision = resolve_auto_config(plan, batch=256, backend='metal')
        assert decision['dense_fuse'] is False

    def test_does_not_mutate_input_plan(self, webbench):
        """`resolve_auto_config`は下見のためだけにdense_fuseを試すので、
        渡した`plan`自体は書き換えない(呼び出し側が同じplanを複数回
        判定に使い回せる)。"""
        bundle = _dense_bundle(webbench, seed=4, in_width=24, widths=[32, 24],
                               k=3, name='ac_fuse_no_mutate')
        batch = DENSE_FUSE_AUTO_MIN_BATCH
        plan = build_plan(bundle, batch=batch)
        kinds_before = [op.kind for op in plan.ops]
        resolve_auto_config(plan, batch=batch, backend='metal')
        assert [op.kind for op in plan.ops] == kinds_before
        assert 'dense_fuse' not in kinds_before


class TestResolveAutoConfigDensePacked:
    def test_wide_chain_too_big_for_fuse_falls_back_to_packed(self, webbench):
        """鎖の活性が共有メモリ上限(CUDA48KB/Metal32KB)を超えると
        dense_fuseは対象外になり、優先順位でdense_packedへ
        フォールバックする(mnist_adderの層幅45.9KBが対象外になる、
        という仕様の合成版。`test_dense_fuse.py::
        test_wide_chain_exceeds_shared_memory_and_is_not_fused`と同じ
        widthを使う)。"""
        bundle = _dense_bundle(webbench, seed=5, in_width=64,
                               widths=[30000, 24000], k=4,
                               name='ac_packed_wide')
        batch = DENSE_PACKED_AUTO_MIN_BATCH
        plan = build_plan(bundle, batch=batch)
        for backend in ('metal', 'cuda'):
            decision = resolve_auto_config(plan, batch=batch, backend=backend)
            assert decision['dense_fuse'] is False, backend
            assert decision['dense_packed'] is True, backend

    def test_mixed_arity_not_fusible_but_packed_eligible(self, webbench):
        """n_inputs=3の層はdense_fuseのn_inputs==2ガードで鎖の
        先頭候補にすらならないが、dense_packedは3〜5入力も評価できるので
        適格のまま(`test_metal_dense_packed.py::test_mixed_arity_chain`と
        同じ理屈)。"""
        bundle = _mixed_arity_dense_bundle(
            webbench, seed=6, in_width=24, widths=[16, 12],
            n_inputs_list=[3, 3], k=3, name='ac_packed_mixed')
        batch = DENSE_PACKED_AUTO_MIN_BATCH
        plan = build_plan(bundle, batch=batch)
        decision = resolve_auto_config(plan, batch=batch, backend='metal')
        assert decision['dense_fuse'] is False
        assert decision['dense_packed'] is True

    def test_below_threshold_batch_does_not_enable(self, webbench):
        bundle = _mixed_arity_dense_bundle(
            webbench, seed=7, in_width=24, widths=[16, 12],
            n_inputs_list=[3, 3], k=3, name='ac_packed_low')
        batch = DENSE_PACKED_AUTO_MIN_BATCH - 1
        plan = build_plan(bundle, batch=batch)
        decision = resolve_auto_config(plan, batch=batch, backend='metal')
        assert decision['dense_packed'] is False
        assert str(batch) in decision['reasons']['dense_packed']

    def test_conv_model_is_ineligible(self, seg_doc):
        bundle = build_bundle(seg_doc)
        plan = build_plan(bundle, batch=DENSE_PACKED_AUTO_MIN_BATCH)
        decision = resolve_auto_config(
            plan, batch=DENSE_PACKED_AUTO_MIN_BATCH, backend='metal')
        assert decision['dense_packed'] is False
        assert '不適格' in decision['reasons']['dense_packed']


class TestResolveAutoConfigConvEngine:
    def test_small_model_stays_split(self, seg_doc):
        bundle = build_bundle(seg_doc)
        plan = build_plan(bundle, batch=1)
        decision = resolve_auto_config(plan, batch=1, backend='metal')
        assert decision['conv_engine'] == 'split'

    def test_oversized_slots_falls_back_to_combined(self, seg_doc, monkeypatch):
        """実測の再現: max_slots_per_ocが`CONV_SPLIT_MAX_SLOTS`を超える
        conv2dノードが1つでもあれば、モデル全体がcombinedへ自動
        フォールバックする。`CONV_SPLIT_MAX_SLOTS`を人工的に下げて壁を
        再現する手法は`test_conv_oc_split.py::
        test_max_slots_guard_rejects_oversized_layout`と同じ。"""
        import griffith_logic.deploy.layout as layout_mod
        from griffith_logic.deploy.layout import build_conv_layout_split

        bundle = build_bundle(seg_doc)
        plan = build_plan(bundle, batch=1)
        conv_op = next(op for op in plan.ops if op.kind == 'gate_program')
        actual = build_conv_layout_split(conv_op.attrs['params'],
                                         conv_op.attrs['program'])
        if actual.max_slots_per_oc <= 0:
            pytest.skip('このモデルの最大スロット数が0で下げ余地がない')
        monkeypatch.setattr(layout_mod, 'CONV_SPLIT_MAX_SLOTS',
                            actual.max_slots_per_oc - 1)
        decision = resolve_auto_config(plan, batch=1, backend='metal')
        assert decision['conv_engine'] == 'combined'
        assert 'combined' in decision['reasons']['conv_engine']
        assert conv_op.node_id in decision['reasons']['conv_engine']


# ============================================================
# 2. generate_metal/generate_cuda: autoの合成(GPUを起動しない)
# ============================================================

class TestGenerateAutoWiring:
    def test_metal_auto_enables_dense_fuse_at_threshold_batch(self, webbench):
        bundle = _dense_bundle(webbench, seed=20, in_width=24,
                               widths=[32, 24], k=3, name='ac_gen_fuse')
        text = generate_metal(bundle, name='m',
                              batch=DENSE_FUSE_AUTO_MIN_BATCH)['m.mm']
        assert 'mdl_metal_dense_fuse_exec(' in text
        assert 'auto-config' in text
        assert 'dense_fuse=True' in text
        assert 'dense_packed=False' in text

    def test_cuda_auto_enables_dense_fuse_at_threshold_batch(self, webbench):
        bundle = _dense_bundle(webbench, seed=21, in_width=24,
                               widths=[32, 24], k=3, name='ac_gen_cuda_fuse')
        text = generate_cuda(bundle, name='m',
                             batch=DENSE_FUSE_AUTO_MIN_BATCH)['m.cu']
        assert text.count(f'__global__ void {CODEGEN_KERNEL_PREFIX}') == 1
        assert 'auto-config' in text
        assert 'dense_fuse=True' in text

    def test_explicit_false_overrides_auto(self, webbench):
        """明示引数が常に勝つ: batchが閾値を満たしていても
        `dense_fuse=False`を明示すれば自動ONにならない。"""
        bundle = _dense_bundle(webbench, seed=22, in_width=24,
                               widths=[32, 24], k=3, name='ac_gen_fuse_off')
        text = generate_metal(bundle, name='m',
                              batch=DENSE_FUSE_AUTO_MIN_BATCH,
                              dense_fuse=False)['m.mm']
        assert 'mdl_metal_dense_fuse_exec(' not in text
        assert 'dense_fuse=False' in text

    def test_explicit_true_overrides_auto_off_batch(self, webbench):
        """明示引数が常に勝つ(逆方向): batchが閾値未満でも
        `dense_fuse=True`を明示すれば有効になる。"""
        bundle = _dense_bundle(webbench, seed=23, in_width=24,
                               widths=[32, 24], k=3, name='ac_gen_fuse_force')
        text = generate_metal(bundle, name='m', batch=1,
                              dense_fuse=True)['m.mm']
        assert 'mdl_metal_dense_fuse_exec(' in text

    def test_auto_off_restores_legacy_defaults(self, webbench):
        """`auto='off'`は、明示しなかった引数を旧版の既定値
        (conv_engine='split'/dense_fuse・dense_packed=False)のまま
        生成する(batchが自動ON閾値を満たしていても無視する)。"""
        bundle = _dense_bundle(webbench, seed=24, in_width=24,
                               widths=[32, 24], k=3, name='ac_gen_off')
        text = generate_metal(bundle, name='m',
                              batch=DENSE_FUSE_AUTO_MIN_BATCH,
                              auto='off')['m.mm']
        assert 'mdl_metal_dense_fuse_exec(' not in text
        assert 'mdl_metal_lut_layer' in text
        assert 'auto-config' in text
        assert "auto='off'" in text
        assert 'auto=off のため判定表は評価していない' in text

    def test_explicit_dense_packed_forces_dense_fuse_off(self, webbench):
        """dense_fuseとdense_packedは生成側で併用不可。片方を明示的に
        Trueにしたら、もう一方は未指定(auto)でも自動ONの対象から外れる
        (mdl_backend.__init__のValueErrorに頼らず、そもそも併用を試みない)。
        """
        bundle = _dense_bundle(webbench, seed=25, in_width=24,
                               widths=[32, 24], k=3, name='ac_gen_packed_wins')
        text = generate_metal(bundle, name='m',
                              batch=DENSE_PACKED_AUTO_MIN_BATCH,
                              dense_packed=True)['m.mm']
        assert 'mdl_metal_dense_fuse_exec(' not in text
        assert text.count('mdl_metal_pack_batch(') == 1

    def test_explicit_conv_engine_wins_over_auto(self, seg_doc):
        """明示指定した`conv_engine`は判定表の結果によらず常にそのまま
        使われる(このモデルはmax_slots_per_ocが小さくautoならsplitのはず
        だが、明示的にcombinedを指定すればcombinedになる)。"""
        bundle = build_bundle(seg_doc)
        text = generate_metal(bundle, name='m',
                              conv_engine='combined')['m.mm']
        assert 'mdl_metal_conv_split(' not in text
        assert "conv_engine='combined'" in text

    def test_auto_falls_back_conv_engine_end_to_end(self, seg_doc, monkeypatch):
        """壁の再現をgenerate_metal経由で確認する ─ auto既定のまま
        生成しても、max_slots_per_oc超過ノードがあれば自動的にcombinedへ
        フォールバックし、split版特有の呼び出しは1つも出ない。"""
        import griffith_logic.deploy.layout as layout_mod
        from griffith_logic.deploy.layout import build_conv_layout_split

        bundle = build_bundle(seg_doc)
        plan = build_plan(bundle, batch=1)
        conv_op = next(op for op in plan.ops if op.kind == 'gate_program')
        actual = build_conv_layout_split(conv_op.attrs['params'],
                                         conv_op.attrs['program'])
        if actual.max_slots_per_oc <= 0:
            pytest.skip('このモデルの最大スロット数が0で下げ余地がない')
        monkeypatch.setattr(layout_mod, 'CONV_SPLIT_MAX_SLOTS',
                            actual.max_slots_per_oc - 1)
        text = generate_metal(bundle, name='m')['m.mm']
        assert 'mdl_metal_conv_split(' not in text
        assert "conv_engine='combined'" in text

    def test_measured_prints_summary_line(self, webbench, capsys):
        bundle = _dense_bundle(webbench, seed=26, in_width=24,
                               widths=[16], k=4, name='ac_gen_print')
        generate_metal(bundle, name='m', batch=1)
        out = capsys.readouterr().out
        assert 'auto-config' in out

    def test_auto_off_prints_nothing(self, webbench, capsys):
        bundle = _dense_bundle(webbench, seed=27, in_width=24,
                               widths=[16], k=4, name='ac_gen_noprint')
        generate_metal(bundle, name='m', batch=1, auto='off')
        out = capsys.readouterr().out
        assert 'auto-config' not in out

    def test_invalid_auto_value_rejected(self, webbench):
        bundle = _dense_bundle(webbench, seed=28, in_width=24,
                               widths=[16], k=4, name='ac_gen_badauto')
        with pytest.raises(ValueError, match='auto'):
            generate_metal(bundle, name='m', auto='yes')
        with pytest.raises(ValueError, match='auto'):
            generate_cuda(bundle, name='m', auto='yes')

    def test_both_explicit_true_conflict_still_raises(self, webbench):
        """dense_fuse/dense_packedが両方とも明示的にTrueなら、autoの値に
        よらず即ValueError(タスク指示どおりの既存挙動を保つ)。"""
        bundle = _dense_bundle(webbench, seed=29, in_width=24,
                               widths=[16], k=4, name='ac_gen_bothtrue')
        with pytest.raises(ValueError, match='dense_fuse'):
            generate_metal(bundle, name='m', batch=32, dense_fuse=True,
                          dense_packed=True)


# ============================================================
# 3. decode megakernelの既定値変更の回帰固定
# ============================================================

class TestMegakernelDefaultChanged:
    """rddlgn decodeの実機検証・golden一致は`test_rddlgn_decode_gpu.py`/
    `test_rddlgn_decode_megakernel.py`が別途詳しく見る。ここでは
    既定値がFalse→Trueへ変わったことだけを軽く固定する(GPU不要、
    ソース生成のみ)。"""

    def test_cuda_default_now_matches_explicit_true(self):
        from griffith_logic.deploy.bundle import build_bundle as _bb
        from griffith_logic.deploy.rddlgn_decode_gpu import (
            generate_rddlgn_decode_cuda,
        )
        from griffith_logic.export.rddlgn_ir import extract_embed_table

        from .conftest import build_rddlgn_tiny_seq2seq, rddlgn_seq2seq_doc

        model, cfg = build_rddlgn_tiny_seq2seq(seed=0)
        doc = rddlgn_seq2seq_doc(model, cfg.seq_len, cfg.seq_len)
        doc['embed_table'] = extract_embed_table(model.src_embedding)
        bundle = _bb(doc)

        default_text = generate_rddlgn_decode_cuda(
            bundle, name='m', batch=4)['m.cu']
        explicit_text = generate_rddlgn_decode_cuda(
            bundle, name='m', batch=4, megakernel=True)['m.cu']
        legacy_text = generate_rddlgn_decode_cuda(
            bundle, name='m', batch=4, megakernel=False)['m.cu']
        assert default_text == explicit_text
        assert default_text != legacy_text
        assert 'megakernel: enabled' in default_text
