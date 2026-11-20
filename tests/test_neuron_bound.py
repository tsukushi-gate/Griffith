"""ニューロン単位LUT下界解析(analysis/neuron_bound)のテスト。

正しさの機械的保証:
    1. majority_fill_agreement を素朴二重ループ(辞書集計)と厳密照合
    2. 選択法の単調性: best_subset ≥ greedy ≥ 任意の固定部分集合
    3. planted部分集合の回復(best_subsetが厳密最良である証拠)
    4. min_circuit_search d=1 のSAT/UNSATを全部分集合列挙と照合
       (偽UNSAT独立検証。IDPool.occupy教訓)
    5. 既知境界: parity7はd=1(arity6)でUNSAT、d=2でSAT
    6. conf_budget極小で unknown が返り、unsatと混同されないこと
"""

import itertools

import numpy as np
import pytest

from griffith_logic.analysis.neuron_bound import (
    best_subset_select,
    consistency_ceiling,
    greedy_agreement_select,
    majority_fill_agreement,
)

try:
    import pysat  # noqa: F401

    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False

needs_pysat = pytest.mark.skipif(not HAS_PYSAT, reason='pysat未導入')


def _naive_majority_fill(X: np.ndarray, y: np.ndarray,
                         subset: list[int]) -> float:
    """辞書集計による素朴照合実装。"""
    votes: dict[tuple, list[int]] = {}
    for row, label in zip(X.tolist(), y.tolist()):
        key = tuple(row[c] for c in subset)
        votes.setdefault(key, []).append(int(label))
    agree = sum(max(v.count(0), v.count(1)) for v in votes.values())
    return agree / len(y)


def _random_dataset(seed: int, n: int = 300, m: int = 8):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, m), dtype=np.int64)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    return X, y


class TestMajorityFill:
    """①閉形式majority-fillが素朴照合と厳密一致。"""

    @pytest.mark.parametrize('seed', range(4))
    @pytest.mark.parametrize('k', [1, 2, 4, 6])
    def test_matches_naive(self, seed, k):
        X, y = _random_dataset(seed)
        rng = np.random.default_rng(100 + seed)
        subset = rng.choice(X.shape[1], size=k, replace=False)
        got = majority_fill_agreement(X, y, subset)
        want = _naive_majority_fill(X, y, list(subset))
        assert got == pytest.approx(want, abs=1e-12)

    def test_perfect_neuron_is_one(self):
        """yが部分集合の決定関数ならagreement==1.0(等値)。"""
        X, _ = _random_dataset(7, n=500, m=8)
        subset = [1, 3, 4]
        y = (X[:, 1] ^ X[:, 3] & X[:, 4]).astype(np.int64)
        assert majority_fill_agreement(X, y, subset) == 1.0

    def test_monotone_in_subset_growth(self):
        """列追加(分割の細分)でagreementは非減少。"""
        X, y = _random_dataset(11)
        subset: list[int] = []
        prev = 0.0
        for c in [0, 2, 5, 7]:
            subset.append(c)
            cur = majority_fill_agreement(X, y, subset)
            assert cur >= prev - 1e-12
            prev = cur

    def test_consistency_ceiling_matches_full_subset(self):
        """全列部分集合のmajority-fillと一致(m≤16のbincount経路照合)。"""
        X, y = _random_dataset(13, n=400, m=8)
        want = majority_fill_agreement(X, y, list(range(8)))
        assert consistency_ceiling(X, y) == pytest.approx(want, abs=1e-12)

    def test_consistency_ceiling_wide_input(self):
        """m>16(bincount不可幅)でもunique経路で計算できる。"""
        rng = np.random.default_rng(17)
        X = rng.integers(0, 2, size=(50, 24), dtype=np.int64)
        y = rng.integers(0, 2, size=50, dtype=np.int64)
        got = consistency_ceiling(X, y)
        want = _naive_majority_fill(X, y, list(range(24)))
        assert got == pytest.approx(want, abs=1e-12)


class TestSubsetSelection:
    """②③選択法の単調性とplanted回復。"""

    @pytest.mark.parametrize('seed', range(3))
    def test_best_geq_greedy_geq_fixed(self, seed):
        X, y = _random_dataset(seed, n=200, m=8)
        sub_best, v_best = best_subset_select(X, y, k=4)
        sub_greedy, v_greedy = greedy_agreement_select(X, y, k=4)
        v_fixed = majority_fill_agreement(X, y, [0, 1, 2, 3])
        assert v_best >= v_greedy - 1e-12
        assert v_greedy >= v_fixed - 1e-12
        # 返り値整合: 部分集合を再評価すると同じ値
        assert majority_fill_agreement(X, y, sub_best) == \
            pytest.approx(v_best, abs=1e-12)
        assert majority_fill_agreement(X, y, sub_greedy) == \
            pytest.approx(v_greedy, abs=1e-12)

    def test_best_subset_is_exhaustive_optimum(self):
        """C(m,k)全列挙の素朴max と一致(厳密最良の証拠)。"""
        X, y = _random_dataset(23, n=150, m=7)
        _, v_best = best_subset_select(X, y, k=3)
        want = max(
            _naive_majority_fill(X, y, list(sub))
            for sub in itertools.combinations(range(7), 3)
        )
        assert v_best == pytest.approx(want, abs=1e-12)

    def test_planted_subset_recovery(self):
        """planted 3列パリティをbest_subsetが厳密回復(agreement=1.0)。"""
        rng = np.random.default_rng(31)
        X = rng.integers(0, 2, size=(300, 10), dtype=np.int64)
        planted = (2, 5, 8)
        y = (X[:, 2] ^ X[:, 5] ^ X[:, 8]).astype(np.int64)
        sub, val = best_subset_select(X, y, k=6)
        assert val == 1.0
        assert set(planted).issubset(set(sub.tolist()))

    def test_valid_indices(self):
        X, y = _random_dataset(37, n=100, m=8)
        for fn in (best_subset_select, greedy_agreement_select):
            sub, _ = fn(X, y, k=6)
            assert len(sub) == 6
            assert len(set(sub.tolist())) == 6
            assert all(0 <= c < 8 for c in sub.tolist())


def _threshold_neuron_dataset(seed: int, m: int = 8, n: int = 64,
                              support: int | None = None):
    """ランダムternaryニューロン y=1[Σu·x>T] のN点部分関数を生成。"""
    rng = np.random.default_rng(seed)
    u = np.zeros(m, dtype=np.int64)
    sup = m if support is None else support
    cols = rng.choice(m, size=sup, replace=False)
    u[cols] = rng.choice([-1, 1], size=sup)
    thr = float(rng.integers(-2, 3)) + 0.5
    X = rng.integers(0, 2, size=(n, m), dtype=np.int64)
    y = (X @ u > thr).astype(np.int64)
    return X, y


@needs_pysat
class TestMinCircuitSearch:
    """④⑤⑥ SAT回路探索の独立検証。"""

    @pytest.mark.parametrize('seed', range(6))
    def test_depth1_matches_subset_enumeration(self, seed):
        """d=1(単一arity-6 LUT・自由配線)のSAT/UNSATが
        『agreement=1.0となる6列部分集合が存在するか』の全列挙と一致。
        偽UNSAT対策の独立検証(ブルートフォース回帰)。
        """
        from griffith_logic.analysis.neuron_bound import min_circuit_search

        # supportを6〜8で振ってSAT/UNSAT両ケースを混ぜる
        X, y = _threshold_neuron_dataset(seed, m=8, n=64,
                                         support=6 + seed % 3)
        res = min_circuit_search(X, y, arity=6, depths=(1,))
        exists = any(
            _naive_majority_fill(X, y, list(sub)) == 1.0
            for sub in itertools.combinations(range(8), 6)
        )
        status = res['per_depth'][1]['status']
        assert status in ('sat', 'unsat'), status  # 小規模でunknown禁止
        assert (status == 'sat') == exists

    def test_parity7_depth_boundary(self):
        """parity7全128点: d=1(arity6)はUNSAT、d=2はSAT。既知境界の再現。"""
        from griffith_logic.analysis.neuron_bound import min_circuit_search

        X = np.array(list(itertools.product([0, 1], repeat=7)),
                     dtype=np.int64)
        y = (X.sum(axis=1) % 2).astype(np.int64)
        res = min_circuit_search(X, y, arity=6, depths=(1, 2),
                                 width_grid=(1, 2))
        assert res['per_depth'][1]['status'] == 'unsat'
        assert res['per_depth'][2]['status'] == 'sat'
        best = res['best']
        assert best is not None and best['depth'] == 2
        # 総LUT数 = (d-1)*k + 1
        assert best['total_luts'] == (2 - 1) * best['width'] + 1

    def test_depth1_sat_stops_early(self):
        """d=1でSATなら1LUTが大域最小 → 後続深さは探索しない。"""
        from griffith_logic.analysis.neuron_bound import min_circuit_search

        X, y = _threshold_neuron_dataset(101, m=8, n=32, support=4)
        res = min_circuit_search(X, y, arity=6, depths=(1, 2, 3))
        if res['per_depth'][1]['status'] == 'sat':
            assert res['best']['total_luts'] == 1
            assert 2 not in res['per_depth']

    def test_conf_budget_exhaustion_yields_unknown(self):
        """conf_budget極小 → unknown。unsatと誤報しない(下界の健全性)。"""
        from griffith_logic.analysis.neuron_bound import min_circuit_search

        X = np.array(list(itertools.product([0, 1], repeat=7)),
                     dtype=np.int64)
        y = (X.sum(axis=1) % 2).astype(np.int64)
        res = min_circuit_search(X, y, arity=6, depths=(2,),
                                 width_grid=(2,), conf_budget=1)
        st = res['per_depth'][2]['status']
        assert st in ('sat', 'unknown')  # 予算1で証明できたらsatでも良い
        assert st != 'unsat' or res['per_depth'][2]['tried'][2] == 'unsat'

    def test_wall_budget_skips_remaining(self):
        """壁時計予算0 → 全判定がスキップされunknown扱い。"""
        from griffith_logic.analysis.neuron_bound import min_circuit_search

        X, y = _threshold_neuron_dataset(7, m=8, n=64)
        res = min_circuit_search(X, y, depths=(1, 2), wall_budget_s=0.0)
        for d in (1, 2):
            assert res['per_depth'][d]['status'] == 'unknown'
        assert res['best'] is None
