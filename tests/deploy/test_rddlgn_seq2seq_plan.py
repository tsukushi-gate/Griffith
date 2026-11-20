"""RDDLGN seq2seqバンドルのプラン層。

test_rddlgn_plan.py(エンコーダ+分類ヘッド)と同じ4つの不変条件
(単一代入・使用前定義・全値がサイズを持つ・全値が到達可能)がseq2seq全体
(4つのrecurrentノード: enc_n_rec/enc_k_rec/dec_l_rec/dec_pm_rec)でも
壊れていないことを固定する。

加えて固有の性質: **dec_pm_recのbody内でgroup_sumを経由するので、
recurrentノード自身の出力値種別がKIND_FLATではなくKIND_SCORES(int32スコア
行)になる**。deploy/plan.py::_plan_recurrentがこれを無条件にKIND_FLATへ
決め打ちしていた(旧設計)ため、この一点を明示的に固定する
(退行すればtest_output_is_scoresが即座に落ちる)。
"""

from griffith_logic.deploy.plan import (
    KIND_BITPLANE,
    KIND_FLAT,
    KIND_SCORES,
    KIND_SCRATCH,
    build_plan,
)


class TestRecurrentOpsShape:
    def test_four_recurrent_execs(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        kinds = [op.kind for op in plan.ops]
        assert kinds.count('recurrent_exec') == 4, (
            'enc_n_rec/enc_k_rec/dec_l_rec/dec_pm_recの4つが要る'
        )
        node_ids = {op.node_id for op in plan.ops if op.kind == 'recurrent_exec'}
        assert node_ids == {'enc_n_rec', 'enc_k_rec', 'dec_l_rec', 'dec_pm_rec'}

    def test_state_init_count(self, rddlgn_seq2seq_tiny_bundle):
        """state_initはstate役scan_varを持つrecurrent(enc_k_rec/dec_pm_rec)
        の2回のみ(enc_n_rec/dec_l_recはstate 0個)。"""
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        kinds = [op.kind for op in plan.ops]
        assert kinds.count('state_init') == 2

    def test_reduce_scores_is_scoped_to_decoder_recurrent(
            self, rddlgn_seq2seq_tiny_bundle):
        """group_sumがdec_pm_recのbody内にあるので、reduce_scores opの
        scopeはdec_pm_recになる(トップレベルのop列には出ない)。"""
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        reduce_ops = [op for op in plan.ops if op.kind == 'reduce_scores']
        assert len(reduce_ops) == 1
        assert reduce_ops[0].scope == 'dec_pm_rec'
        assert reduce_ops[0].attrs['op'] == 'group_sum'

    def test_body_ops_are_scoped_to_their_recurrent_node(
            self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        scoped = [op for op in plan.ops if op.scope]
        assert scoped, 'scope付きのopが1つも無い'
        node_ids = {op.node_id for op in plan.ops if op.kind == 'recurrent_exec'}
        scopes = {op.scope for op in scoped}
        assert scopes <= node_ids
        for op in scoped:
            assert op.kind in ('lut_dense', 'flat_concat', 'reduce_scores'), (
                f'body内に想定外のop種別: {op.kind}'
            )


class TestRecurrentOutputKindTracksBody:
    """中核: recurrentの出力値種別はbody出力(output_source)に
    追従する(body内がlut_layer/concatのみならKIND_FLAT、group_sumを
    経由するならKIND_SCORES)。"""

    def test_encoder_recurrents_stay_flat(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        by_node = {op.node_id: op for op in plan.ops if op.kind == 'recurrent_exec'}
        for node_id in ('enc_n_rec', 'enc_k_rec', 'dec_l_rec'):
            v = plan.value(by_node[node_id].output)
            assert v.kind == KIND_FLAT, f'{node_id}: {v.kind}'

    def test_decoder_pm_recurrent_is_scores(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        by_node = {op.node_id: op for op in plan.ops if op.kind == 'recurrent_exec'}
        v = plan.value(by_node['dec_pm_rec'].output)
        assert v.kind == KIND_SCORES

    def test_output_is_scores(self, rddlgn_seq2seq_tiny_bundle):
        """グラフの最終出力はdec_pm_recの出力そのもの(to_scoresの追加変換
        op無しに直接KIND_SCORES)。"""
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        assert plan.value(plan.output_value).kind == KIND_SCORES
        assert plan.value(plan.output_value).width == plan.output_width
        assert not any(op.kind == 'to_scores' for op in plan.ops), (
            'dec_pm_recの出力が既にKIND_SCORESなのでto_scoresは不要のはず'
        )


class TestPlanInvariantsHoldForSeq2Seq:
    """test_rddlgn_plan.py::TestPlanInvariantsHoldForRddlgnと同じ4条件。"""

    def test_single_assignment(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        written = [op.output for op in plan.ops]
        assert len(written) == len(set(written))

    def test_defined_before_use(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        available = {plan.input_value}
        if plan.input_bitplane is not None:
            available.add(plan.input_bitplane)
        for i, op in enumerate(plan.ops):
            for vid in op.inputs:
                assert vid in available, (
                    f'op[{i}] {op.kind} が未定義の値 v{vid} を読んでいる'
                )
            available.add(op.output)

    def test_every_value_has_a_size(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        for v in plan.values:
            assert v.nbytes() > 0, f'v{v.id}({v.kind}) のサイズが0'

    def test_all_values_reachable(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        used = {plan.output_value, plan.input_value}
        for op in plan.ops:
            used.update(op.inputs)
            used.update(op.attrs.get('scratch', ()))
            used.add(op.output)
        orphans = [v.id for v in plan.values if v.id not in used]
        assert not orphans, f'どのopからも参照されない値: {orphans}'

    def test_kinds_are_from_the_closed_set(self, rddlgn_seq2seq_tiny_bundle):
        plan = build_plan(rddlgn_seq2seq_tiny_bundle)
        allowed = {KIND_BITPLANE, KIND_FLAT, KIND_SCORES, KIND_SCRATCH}
        for v in plan.values:
            assert v.kind in allowed, f'未知の値種別 {v.kind}'
