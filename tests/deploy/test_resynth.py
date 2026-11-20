"""``deploy/resynth.py`` の lossless 再合成テスト。

seg_doc フィクスチャ(conv2d/or_pool2d/upsample2d/group_sum を含む小さな
合成セグIR、真理値表はランダム化済み)を使う。tree_depth=2(木あたり最大
3ゲート)しかない小さな木なので、探索上限 max_cost=5(既定値)でも
分枝限定探索は一瞬で終わる(実測: 全テスト込みで1秒未満)。そのため
「max_cost=3に落とす」調整は不要だった。

故障注入(fault injection)は tests/deploy/test_verify_c.py の方針を踏襲:
「ガード自体が仕事をしているか」を、内部の resynth_node_params をすり替えて
壊れた再合成結果を返させ、resynth_doc_lossless の検証がそれを検出して
ValueError を送出することで確かめる。
"""

import numpy as np
import pytest

from griffith_logic.deploy import build_bundle, validate_bundle
from griffith_logic.deploy.resynth import resynth_doc_lossless
from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.simulate import simulate


def _conv_gates_naive(graph) -> int:
    """conv2dノードの素朴な木ゲート数合計(2^depth - 1 per oc)。"""
    total = 0
    for n in graph.nodes:
        if n.type != 'conv2d':
            continue
        depth = int(n.params['tree_depth'])
        oc = int(n.params['out_channels'])
        total += oc * ((1 << depth) - 1)
    return total


class TestResynthDocLossless:
    def test_gates_do_not_increase(self, seg_doc):
        new_doc, summary = resynth_doc_lossless(seg_doc, max_cost=5, seed=0)
        assert summary['gates_after'] <= summary['gates_before']
        assert summary['gates_before'] > 0, 'conv2dノードが無いテストになっている'

    def test_summary_matches_about_provenance(self, seg_doc):
        new_doc, summary = resynth_doc_lossless(seg_doc, max_cost=5, seed=0)
        prov = new_doc['about']['resynth']
        assert prov['mode'] == 'lossless'
        assert prov['max_cost'] == 5
        assert prov['gates_before'] == summary['gates_before']
        assert prov['gates_after'] == summary['gates_after']

    def test_original_doc_is_not_mutated(self, seg_doc):
        import copy
        before = copy.deepcopy(seg_doc)
        resynth_doc_lossless(seg_doc, max_cost=5, seed=0)
        assert seg_doc == before, '入力docを書き換えてはいけない'

    def test_independent_equivalence_check_on_fresh_random_inputs(self, seg_doc):
        """テスト自身が(実装内部の検証を信用せず)独立に再検証する。"""
        old_graph = ir_from_dict(seg_doc['ir'])
        new_doc, _ = resynth_doc_lossless(seg_doc, max_cost=5, seed=0)
        new_graph = ir_from_dict(new_doc['ir'])

        in_width = int(old_graph.inputs[0].width)
        rng = np.random.default_rng(12345)  # 実装の検証(seed=0)とは別の乱数
        bits = rng.integers(0, 2, size=(64, in_width), dtype=np.uint8)
        old_out = simulate(old_graph, bits)
        new_out = simulate(new_graph, bits)
        assert np.array_equal(old_out, new_out)

    def test_resources_refreshed(self, seg_doc):
        new_doc, _ = resynth_doc_lossless(seg_doc, max_cost=5, seed=0)
        assert new_doc['resources']['json_bytes'] > 0
        assert new_doc['resources']['gzip_bytes'] > 0
        assert 'full_bytes' in new_doc['resources']
        assert 'full_gzip_bytes' in new_doc['resources']
        # ゲート数が減っていれば素朴展開のLUT総数も減っているはず(増えはしない)
        assert new_doc['resources']['total_luts'] <= seg_doc['resources']['total_luts']

    def test_bundle_builds_and_validates_strict(self, seg_doc):
        new_doc, _ = resynth_doc_lossless(seg_doc, max_cost=5, seed=0)
        bundle = build_bundle(new_doc)
        warnings = validate_bundle(bundle, strict=True)
        assert warnings == []


class TestVerifyModes:
    def test_library_default_is_simulate(self, seg_doc, monkeypatch):
        """ライブラリの既定は simulate(golden model)照合。外部の呼び出し元の検証を弱めない。"""
        import griffith_logic.deploy.resynth as resynth_mod

        calls = []
        real = resynth_mod.simulate

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(resynth_mod, 'simulate', spy)
        _doc, summary = resynth_mod.resynth_doc_lossless(seg_doc, seed=0)
        assert summary['verify'] == 'simulate'
        assert calls

    def test_fast_does_not_lower_conv(self, seg_doc, monkeypatch):
        """fast(CLI の --compress fast が明示で渡す)は simulate(conv2d展開)を呼ばずに検証を終える。"""
        import griffith_logic.deploy.resynth as resynth_mod

        def boom(*a, **k):
            raise AssertionError('fast 検証で simulate が呼ばれた')

        monkeypatch.setattr(resynth_mod, 'simulate', boom)
        _doc, summary = resynth_mod.resynth_doc_lossless(seg_doc, seed=0, verify='fast')
        assert summary['verify'] == 'fast'
        assert summary['seconds'] >= 0

    def test_fast_and_simulate_give_identical_docs(self, seg_doc):
        fast_doc, fast_s = resynth_doc_lossless(seg_doc, verify='fast')
        sim_doc, sim_s = resynth_doc_lossless(seg_doc, verify='simulate')
        assert sim_s['verify'] == 'simulate'
        assert fast_doc['ir'] == sim_doc['ir']
        for key in ('gates_before', 'gates_after', 'n_trees', 'n_trees_changed'):
            assert fast_s[key] == sim_s[key]
        # 資源見積りは両モードで一致する(構造のみ依存)
        assert fast_doc['resources'] == sim_doc['resources']

    def test_invalid_verify_rejected(self, seg_doc):
        with pytest.raises(ValueError, match='verify'):
            resynth_doc_lossless(seg_doc, verify='bogus')

    def test_fast_detects_corrupted_root_gate(self, seg_doc, monkeypatch):
        """再合成後の木を壊すと per-tree 照合が検出する(simulate 不使用でも)。"""
        import griffith_logic.deploy.resynth as resynth_mod
        from griffith_logic.export.dc_resynth import (
            resynth_node_params as real,
        )

        def fake(params, care_sets, *, max_cost):
            new_params, st = real(params, care_sets, max_cost=max_cost)
            # oc=0 の根ゲート tt を反転 → その木の出力が全入力で反転する
            root = (1 << int(new_params['tree_depth'])) - 2
            tts = list(new_params['truth_tables'])
            if tts and isinstance(tts[0], str):
                tts[root] = hex(int(tts[root], 16) ^ 0xf)
            else:
                tts[root] = int(tts[root]) ^ 0xf
            return dict(new_params, truth_tables=tts), st

        monkeypatch.setattr(resynth_mod, 'resynth_node_params', fake)
        with pytest.raises(ValueError, match='一致しない'):
            resynth_mod.resynth_doc_lossless(seg_doc, verify='fast')


class TestFaultInjection:
    """内部検証(多重防御)が実際に機能していることを故障注入で確かめる。

    tests/deploy/test_verify_c.py の TestFaultInjection と同じ狙い:
    resynth_node_params が返す再合成結果を1本だけ意味的に壊し、
    resynth_doc_lossless がそれを検出して ValueError を送出することを見る。
    """

    def test_corrupted_truth_table_is_detected(self, seg_doc, monkeypatch):
        import griffith_logic.deploy.resynth as resynth_mod
        from griffith_logic.export.dc_resynth import (
            resynth_node_params as real_resynth_node_params,
        )

        state = {'done': False}

        def fake_resynth_node_params(params, care_sets, *, max_cost):
            new_params, st = real_resynth_node_params(
                params, care_sets, max_cost=max_cost)
            if not state['done']:
                state['done'] = True
                # 最初のconv2dノードの最初のoc(木)の根ゲートtt(=そのocの
                # セグメント最後のtt要素)をXOR 0xfで反転する。
                # gate2_mask(tt^0xf, a, b, full) == full ^ gate2_mask(tt, a, b, full)
                # が恒等式として成り立つ(2入力の4パターンは互いに排他的な
                # 被覆なので、真理値表の全ビット反転はその木が計算する
                # 出力ビットの全パターン反転と等価)ため、木の深さや構造に
                # 関わらず必ずそのocの出力が全入力で反転し、検出できる。
                depth = int(new_params['tree_depth'])
                n_int = (1 << depth) - 1
                tts = list(new_params['truth_tables'])
                is_hex = bool(tts) and isinstance(tts[0], str)
                root_idx = n_int - 1  # oc=0のセグメントの根(最後のゲート)
                if is_hex:
                    val = int(tts[root_idx], 16)
                    tts[root_idx] = hex(val ^ 0xf)
                else:
                    tts[root_idx] = int(tts[root_idx]) ^ 0xf
                new_params = dict(new_params)
                new_params['truth_tables'] = tts
            return new_params, st

        monkeypatch.setattr(
            resynth_mod, 'resynth_node_params', fake_resynth_node_params)

        with pytest.raises(ValueError, match='一致しない'):
            resynth_mod.resynth_doc_lossless(seg_doc, max_cost=5, seed=0)


@pytest.mark.slow
class TestChainToC:
    """再合成 → バンドル化 → Cランタイム検証の鎖が壊れていないことを確認する。"""

    def test_resynth_bundle_verify_c(self, seg_doc, tmp_path, cc):
        from griffith_logic.deploy.verify import verify_c_matches_simulate

        new_doc, summary = resynth_doc_lossless(seg_doc, max_cost=5, seed=0)
        bundle = build_bundle(new_doc)
        result = verify_c_matches_simulate(
            bundle, n_samples=2, work_dir=tmp_path, verbose=False)
        assert result['max_abs_diff'] == 0
        assert summary['gates_after'] <= summary['gates_before']
