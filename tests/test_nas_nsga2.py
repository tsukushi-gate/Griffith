"""tools/nas/nsga2.py のテスト。

方針:
    - non_dominated_sort / crowding_distance は、本ファイル内の素朴な
      参照実装(全対比較のみで組む、最適化なし)と一致することを
      hypothesisのproperty testで検証する。
    - strategy='nsga2' での小規模XOR探索が決定的に完走し、パレート集合が
      世代を追うごとに単調非劣化する(既出の最良acc/最小total_lutsが
      悪化しない)ことを確認する。
    - 既定戦略(strategy未指定='evolution')のゴールデンには一切
      触れない(test_nas_search.py が担当)。
"""

import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tools.nas.nsga2 import crowding_distance, non_dominated_sort, select_parent
from tools.nas.search import SearchConfig, SearchSpace, pareto_front, run_search

# ------------------------------------------------------------------
# 参照実装(素朴なO(n^2)全対比較、search.py/nsga2.pyの実装からは独立)
# ------------------------------------------------------------------


def _ref_dominates(a, b):
    """aがbを支配するか(acc最大化・total_luts最小化)。"""
    better_or_eq = a[0] >= b[0] and a[1] <= b[1]
    strictly = a[0] > b[0] or a[1] < b[1]
    return better_or_eq and strictly


def _ref_non_dominated_sort(points):
    """素朴な非劣ソート: 各フロントをその都度全対比較で洗い出す。"""
    remaining = set(range(len(points)))
    fronts = []
    while remaining:
        front = [
            i for i in remaining
            if not any(_ref_dominates(points[j], points[i]) for j in remaining if j != i)
        ]
        fronts.append(sorted(front))
        remaining -= set(front)
    return fronts


def _ref_crowding_distance(front):
    """素朴な混雑度: 定義通り軸ごとにソートして両端inf、内側は正規化差分の和。"""
    n = len(front)
    if n <= 2:
        return [float('inf')] * n
    distance = [0.0] * n
    for obj_idx in range(2):
        order = sorted(range(n), key=lambda i: front[i][obj_idx])
        distance[order[0]] = float('inf')
        distance[order[-1]] = float('inf')
        lo, hi = front[order[0]][obj_idx], front[order[-1]][obj_idx]
        span = hi - lo
        if span == 0:
            continue
        for k in range(1, n - 1):
            distance[order[k]] += (
                front[order[k + 1]][obj_idx] - front[order[k - 1]][obj_idx]
            ) / span
    return distance


_point_strategy = st.tuples(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=1000).map(float),
)


class TestNonDominatedSortMatchesReference:
    @given(points=st.lists(_point_strategy, min_size=0, max_size=25))
    def test_fronts_match_as_sets(self, points):
        got = non_dominated_sort(points)
        want = _ref_non_dominated_sort(points)
        # フロントの「集合の並び」が一致すること(各フロント内の要素集合が一致)
        assert [set(f) for f in got] == [set(f) for f in want]

    @given(points=st.lists(_point_strategy, min_size=1, max_size=25))
    def test_every_point_assigned_exactly_once(self, points):
        fronts = non_dominated_sort(points)
        seen = [i for f in fronts for i in f]
        assert sorted(seen) == list(range(len(points)))

    def test_simple_case(self):
        points = [(0.5, 10.0), (0.9, 5.0), (1.0, 8.0), (0.9, 8.0)]
        fronts = non_dominated_sort(points)
        assert fronts[0] == [1, 2]
        assert set(fronts[-1]) | {i for f in fronts[:-1] for i in f} == {0, 1, 2, 3}


class TestCrowdingDistanceMatchesReference:
    @given(points=st.lists(_point_strategy, min_size=0, max_size=15))
    def test_matches_reference(self, points):
        # 単一フロントとして扱う(参照実装もfront全体に対して定義)
        got = crowding_distance(points)
        want = _ref_crowding_distance(points)
        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            if g == float('inf') or w == float('inf'):
                assert g == w
            else:
                assert abs(g - w) < 1e-9

    def test_boundary_points_are_infinite(self):
        front = [(0.1, 10.0), (0.5, 5.0), (0.9, 1.0)]
        dist = crowding_distance(front)
        assert dist[0] == float('inf')
        assert dist[2] == float('inf')
        assert dist[1] < float('inf')

    def test_small_fronts_are_all_infinite(self):
        assert crowding_distance([]) == []
        assert crowding_distance([(0.1, 1.0)]) == [float('inf')]
        assert crowding_distance([(0.1, 1.0), (0.2, 2.0)]) == [float('inf'), float('inf')]


# ------------------------------------------------------------------
# select_parent: 常にstate=='done' かつ partial=False な候補のみを返す
# ------------------------------------------------------------------


class _FakeTrial:
    def __init__(self, trial_id, acc, total_luts, state='done', partial=False):
        self.trial_id = trial_id
        self.acc = acc
        self.total_luts = total_luts
        self.state = state
        self.partial = partial


class TestSelectParent:
    def test_returns_none_when_no_candidates(self):
        import random
        rng = random.Random(0)
        assert select_parent(rng, []) is None
        assert select_parent(rng, [_FakeTrial(0, 0.5, 1, state='failed')]) is None

    def test_only_picks_from_first_front(self):
        import random
        ledger = [
            _FakeTrial(0, 0.5, 10),  # 支配される
            _FakeTrial(1, 0.9, 5),   # 非劣
            _FakeTrial(2, 1.0, 8),   # 非劣
        ]
        rng = random.Random(0)
        for _ in range(30):
            pid = select_parent(rng, ledger)
            assert pid in (1, 2)

    def test_excludes_partial_and_failed(self):
        import random
        ledger = [
            _FakeTrial(0, 1.0, 1, partial=True),
            _FakeTrial(1, 0.9, 1, state='failed'),
            _FakeTrial(2, 0.5, 3),
        ]
        rng = random.Random(0)
        assert select_parent(rng, ledger) == 2


# ------------------------------------------------------------------
# 探索ループへの組み込み: 決定的完走+パレートの単調非劣化
# ------------------------------------------------------------------


_SMALL_SPACE = SearchSpace(
    layers_choices=(1, 2), width_choices=(4, 8), n_inputs_choices=(2,)
)


def _nsga2_cfg(out_root, **kw):
    base = dict(
        search_id='nsga2t', dataset='logic:xor', model_kind='logic_mlp',
        max_trials=6, epochs=3, seed=0, out_root=str(out_root), space=_SMALL_SPACE,
        strategy='nsga2',
    )
    base.update(kw)
    return SearchConfig(**base)


@pytest.mark.learning
class TestNsga2SearchIntegration:
    def test_completes_deterministically(self, tmp_path):
        ledger = run_search(_nsga2_cfg(tmp_path))
        assert len(ledger) == 6
        for t in ledger:
            assert t.state == 'done'
            assert 0.0 <= t.acc <= 1.0
            assert t.total_luts >= 1

    def test_repeatable_given_same_seed(self, tmp_path):
        a = run_search(_nsga2_cfg(tmp_path / 'a'))
        b = run_search(_nsga2_cfg(tmp_path / 'b'))
        for ta, tb in zip(a, b, strict=True):
            assert ta.genome == tb.genome
            assert ta.parent_id == tb.parent_id
            assert ta.total_luts == tb.total_luts

    def test_pareto_quality_is_monotonically_non_degrading(self, tmp_path):
        """世代を追うごとに、既出archiveの最良acc/最小total_lutsが悪化しない。

        (前世代のパレート集合が、後世代に「支配されて全体品質が下がる」
        ことがない=単調非劣化。ledgerはtrial追加のみのアーカイブなので、
        各時点のパレートの持つ最良値は非減少/非増加であるべき。)
        """
        ledger = run_search(_nsga2_cfg(tmp_path))
        best_acc_so_far = -1.0
        best_luts_so_far = float('inf')
        for k in range(1, len(ledger) + 1):
            front_ids = pareto_front(ledger[:k])
            if not front_ids:
                continue
            front_trials = [t for t in ledger[:k] if t.trial_id in front_ids]
            max_acc = max(t.acc for t in front_trials)
            min_luts = min(t.total_luts for t in front_trials)
            assert max_acc >= best_acc_so_far - 1e-9
            if best_luts_so_far != float('inf'):
                assert min_luts <= best_luts_so_far + 1e-9
            best_acc_so_far = max(best_acc_so_far, max_acc)
            best_luts_so_far = min(best_luts_so_far, min_luts)


# ------------------------------------------------------------------
# 既定戦略(strategy未指定)への非干渉確認(ゴールデン不変の裏取り)
# ------------------------------------------------------------------


@pytest.mark.learning
class TestDefaultStrategyUnaffected:
    def test_default_strategy_field_value(self):
        cfg = SearchConfig(search_id='x')
        assert cfg.strategy == 'evolution'
