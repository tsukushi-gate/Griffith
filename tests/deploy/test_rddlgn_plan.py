"""RDDLGNバンドルのプラン層。

recurrentノードが正しくop列へ写ること(state_init/recurrent_exec/
flat_concat)、bodyから生まれたopにscope=recurrentノードidが付くこと、
そしてtest_plan_extensibility.py::TestPlanInvariantsと同じ4つの不変条件
(単一代入・使用前定義・全値がサイズを持つ・全値が到達可能)が
recurrentを含むプランでも壊れていないことを固定する。
"""

from griffith_logic.deploy.plan import (
    KIND_BITPLANE,
    KIND_FLAT,
    KIND_SCORES,
    KIND_SCRATCH,
    build_plan,
)


class TestRecurrentOpsShape:
    def test_state_init_recurrent_exec_flat_concat_present(self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        kinds = [op.kind for op in plan.ops]
        assert kinds.count('state_init') == 1, (
            'state_init はK-layers(state 1個)ぶんの1回のみのはず'
        )
        assert kinds.count('recurrent_exec') == 2, (
            'N-layers/K-layers それぞれに1つずつrecurrent_execが要る'
        )
        assert kinds.count('flat_concat') == 1, (
            'K-layers bodyのconcat(state+step)が1つあるはず'
        )

    def test_recurrent_exec_ops_are_not_scoped(self, rddlgn_tiny_bundle):
        """recurrent_exec自身とstate_initは主経路のop(scope=='')。"""
        plan = build_plan(rddlgn_tiny_bundle)
        for op in plan.ops:
            if op.kind in ('recurrent_exec', 'state_init'):
                assert op.scope == '', (
                    f'{op.kind} は主経路のはずだがscope={op.scope!r}'
                )

    def test_body_ops_are_scoped_to_their_recurrent_node(self, rddlgn_tiny_bundle):
        """bodyから生まれたop(lut_dense/flat_concat)にscopeが付く。"""
        plan = build_plan(rddlgn_tiny_bundle)
        scoped = [op for op in plan.ops if op.scope]
        assert scoped, 'scope付きのopが1つも無い'
        scopes = {op.scope for op in scoped}
        node_ids = {op.node_id for op in plan.ops if op.kind == 'recurrent_exec'}
        assert scopes <= node_ids, (
            f'scope値がrecurrent_execのnode_idの集合に含まれない: '
            f'{scopes - node_ids}'
        )
        # 各scopeの中身がlut_dense/flat_concat(bodyの許容ノード型)のみ
        for op in scoped:
            assert op.kind in ('lut_dense', 'flat_concat'), (
                f'body内に想定外のop種別: {op.kind}'
            )

    def test_recurrent_exec_appears_after_its_body_ops_in_op_order(
            self, rddlgn_tiny_bundle):
        """recurrent_execは自分のbody opより後ろに来る(トポ順の帰結)。

        emit_recurrent_execは「scoped opをbuildしてstep関数へ切り出し、
        主経路にはrecurrent_execだけ残す」実装なので、この順序が崩れると
        値が定義される前に参照される事故になる。
        """
        plan = build_plan(rddlgn_tiny_bundle)
        for i, op in enumerate(plan.ops):
            if op.kind != 'recurrent_exec':
                continue
            body_indices = [j for j, o in enumerate(plan.ops)
                            if o.scope == op.node_id]
            assert body_indices, f'{op.node_id}: bodyのopが見つからない'
            assert max(body_indices) < i, (
                f'{op.node_id}: recurrent_execがbody opより前に来ている'
            )


class TestPlanInvariantsHoldForRddlgn:
    """test_plan_extensibility.py::TestPlanInvariantsと同じ4条件。"""

    def test_single_assignment(self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        written = [op.output for op in plan.ops]
        assert len(written) == len(set(written))

    def test_defined_before_use(self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        available = {plan.input_value}
        if plan.input_bitplane is not None:
            available.add(plan.input_bitplane)
        for i, op in enumerate(plan.ops):
            for vid in op.inputs:
                assert vid in available, (
                    f'op[{i}] {op.kind} が未定義の値 v{vid} を読んでいる'
                )
            available.add(op.output)

    def test_every_value_has_a_size(self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        for v in plan.values:
            assert v.nbytes() > 0, f'v{v.id}({v.kind}) のサイズが0'

    def test_all_values_reachable(self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        used = {plan.output_value, plan.input_value}
        for op in plan.ops:
            used.update(op.inputs)
            used.update(op.attrs.get('scratch', ()))
            used.add(op.output)
        orphans = [v.id for v in plan.values if v.id not in used]
        assert not orphans, f'どのopからも参照されない値: {orphans}'

    def test_kinds_are_from_the_closed_set(self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        allowed = {KIND_BITPLANE, KIND_FLAT, KIND_SCORES, KIND_SCRATCH}
        for v in plan.values:
            assert v.kind in allowed, f'未知の値種別 {v.kind}'

    def test_output_is_scores(self, rddlgn_tiny_bundle):
        plan = build_plan(rddlgn_tiny_bundle)
        assert plan.value(plan.output_value).kind == KIND_SCORES
        assert plan.value(plan.output_value).width == plan.output_width
