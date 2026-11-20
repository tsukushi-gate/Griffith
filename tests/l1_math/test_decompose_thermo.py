"""温度計部分和分解(convert/decompose_thermo)のL1テスト。

正しさの機械的保証:
    1. optimal_thermo_quantizer を全分割ブルートフォースと厳密照合
       (max誤差最小、ビット増で非増加、b≥|V|-1で厳密)
    2. reachable_sums を素朴2^n列挙と照合
    3. 誤差上界の全数保証: G≤12全2^G掃引で、マージン>error_bound の
       全入力点において回路出力==元ニューロン出力(等値assert)
    4. G≤6の単一LUT base caseは全点bit厳密(error_bound==0)
    5. 大マージンニューロンは粗い量子化でも較正集合上bit厳密
       (budget_met=True → 較正agreement==1.0)
    6. 回路の構造制約: 全LUT fan-in≤6、wire数が層ごとに厳密減少、
       n_luts/depthのmeta整合
"""

import itertools

import numpy as np
import pytest

from griffith_logic.convert.decompose_thermo import (
    DecomposeSpec,
    decompose_threshold_neuron,
    eval_circuit,
    optimal_thermo_quantizer,
    reachable_sums,
)


def _all_patterns(g: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=g)),
                    dtype=np.int64)


def _brute_quantize_max_err(values: np.ndarray, n_thresholds: int) -> float:
    """連続区間分割の全列挙でmax誤差の最小値を求める素朴照合。"""
    v = np.sort(values)
    n = len(v)
    n_bins = min(n_thresholds + 1, n)
    best = np.inf
    # n値をn_bins個の連続区間へ分割する境界位置の全列挙
    for cuts in itertools.combinations(range(1, n), n_bins - 1):
        edges = [0, *cuts, n]
        err = max((v[e - 1] - v[s]) / 2.0
                  for s, e in zip(edges[:-1], edges[1:]))
        best = min(best, err)
    return best


class TestOptimalThermoQuantizer:
    """①離散DP量子化の厳密性。"""

    @pytest.mark.parametrize('seed', range(4))
    @pytest.mark.parametrize('b', [1, 2, 3, 4])
    def test_matches_brute_force(self, seed, b):
        rng = np.random.default_rng(seed)
        n_vals = int(rng.integers(3, 9))
        values = np.unique(rng.integers(-6, 7, size=n_vals)).astype(float)
        probs = rng.dirichlet(np.ones(len(values)))
        thresholds, recon, max_err, _ = optimal_thermo_quantizer(
            values, probs, b)
        assert max_err == pytest.approx(
            _brute_quantize_max_err(values, b), abs=1e-12)
        # 復元レベルは単調増加、閾値はビット数以下
        assert len(thresholds) <= b
        assert (np.diff(recon) > 0).all()

    def test_monotone_in_bits(self):
        values = np.arange(-3, 4).astype(float)
        probs = np.full(7, 1 / 7)
        prev = np.inf
        for b in range(1, 7):
            _, _, e, _ = optimal_thermo_quantizer(values, probs, b)
            assert e <= prev + 1e-12
            prev = e

    def test_exact_when_enough_bits(self):
        values = np.array([-2.0, -1.0, 0.5, 2.0, 3.0])
        probs = np.full(5, 0.2)
        _, recon, e, _ = optimal_thermo_quantizer(values, probs, 4)
        assert e == 0.0
        assert np.allclose(np.sort(recon), values)

    def test_reconstruction_consistency(self):
        """ŝ(v) = c0 + Σδ·1[v≥θ] が各binの復元レベルと一致する。"""
        values = np.arange(-4, 5).astype(float)
        probs = np.full(9, 1 / 9)
        thresholds, recon, max_err, _ = optimal_thermo_quantizer(
            values, probs, 2)
        deltas = np.diff(recon)
        assert (deltas > 0).all()
        for v in values:
            bits = (v >= thresholds).astype(float)
            s_hat = recon[0] + (deltas * bits[:len(deltas)]).sum() \
                if len(deltas) else recon[0]
            # binの復元レベルとの一致(=温度計符号での復元が正しい)
            bin_idx = int((v >= thresholds).sum())
            assert s_hat == pytest.approx(recon[bin_idx], abs=1e-12)
            assert abs(s_hat - v) <= max_err + 1e-12


class TestReachableSums:
    """②到達可能部分和の列挙。"""

    @pytest.mark.parametrize('seed', range(3))
    def test_matches_enumeration_ternary(self, seed):
        rng = np.random.default_rng(seed)
        w = rng.choice([-1.0, 1.0], size=6)
        got = reachable_sums(w)
        want = np.unique([
            (w * np.array(bits)).sum()
            for bits in itertools.product([0, 1], repeat=6)
        ])
        assert np.allclose(np.sort(got), want)

    def test_matches_enumeration_real(self):
        w = np.array([0.5, 0.5, 1.5, 2.0])
        got = reachable_sums(w)
        want = np.unique([
            (w * np.array(bits)).sum()
            for bits in itertools.product([0, 1], repeat=4)
        ])
        assert np.allclose(np.sort(got), want)

    def test_ternary_contiguous_integers(self):
        """±1重みの到達集合は[-q, p]の連続整数。"""
        w = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
        got = np.sort(reachable_sums(w))
        assert np.allclose(got, np.arange(-2, 4))


def _neuron_cases():
    """(seed, G, spec) の組。uniform/adaptive×粗細を掃引。"""
    specs = [
        DecomposeSpec(alloc='uniform', bits_uniform=1),
        DecomposeSpec(alloc='uniform', bits_uniform=2),
        DecomposeSpec(alloc='uniform', bits_uniform=3),
        DecomposeSpec(alloc='adaptive', margin_alpha=0.5),
        DecomposeSpec(alloc='adaptive', margin_alpha=0.9,
                      grouping='sorted'),
    ]
    return [(seed, g, spec)
            for seed in range(2) for g in (8, 11) for spec in specs]


class TestErrorBoundSweep:
    """③全2^G掃引での誤差上界保証(本手法の中核性質)。"""

    @pytest.mark.parametrize('seed,g,spec', _neuron_cases())
    def test_bound_guarantees_agreement(self, seed, g, spec):
        rng = np.random.default_rng(seed)
        u = rng.choice([-1.0, 1.0], size=g)
        thr = float(rng.integers(-2, 3)) + 0.5
        X = _all_patterns(g)
        calib = X[rng.permutation(len(X))[:200]]
        circ = decompose_threshold_neuron(u, thr, calib, spec)
        bound = circ.meta['error_bound']
        assert bound >= 0.0

        y_true = (X @ u > thr).astype(np.int64)
        y_hat = eval_circuit(circ, X)
        margins = np.abs(X @ u - thr)
        # マージンが上界を超える全点で厳密一致(等値assert、閾値緩和禁止)
        safe = margins > bound + 1e-9
        assert (y_hat[safe] == y_true[safe]).all(), (
            f'誤差上界{bound}を満たすのに不一致: '
            f'{int((y_hat[safe] != y_true[safe]).sum())}点')

    @pytest.mark.parametrize('g', [2, 4, 6])
    def test_base_case_exact(self, g):
        """④G≤6は単一LUTで全点厳密、error_bound==0。"""
        rng = np.random.default_rng(g)
        u = rng.choice([-1.0, 1.0], size=g)
        thr = 0.5
        X = _all_patterns(g)
        circ = decompose_threshold_neuron(u, thr, X[:32], DecomposeSpec())
        assert circ.meta['error_bound'] == 0.0
        assert circ.meta['n_luts'] == 1
        assert circ.meta['depth'] == 1
        y_true = (X @ u > thr).astype(np.int64)
        assert (eval_circuit(circ, X) == y_true).all()

    def test_large_margin_calib_exact(self):
        """⑤マージンが大きい較正点のみなら粗い量子化でもbit厳密。"""
        rng = np.random.default_rng(7)
        g = 12
        u = np.ones(g)  # 全+1: s = popcount
        thr = 5.5
        X = _all_patterns(g)
        margins = np.abs(X.sum(axis=1) - thr)
        calib = X[margins >= 3.5][:300]  # 大マージン点のみ較正
        spec = DecomposeSpec(alloc='adaptive', margin_alpha=0.9)
        circ = decompose_threshold_neuron(u, thr, calib, spec)
        assert circ.meta['budget_met']
        y_true = (calib @ u > thr).astype(np.int64)
        assert (eval_circuit(circ, calib) == y_true).all()


class TestCircuitStructure:
    """⑥構造制約(fan-in≤6、wire数減少、meta整合)。"""

    @pytest.mark.parametrize('seed', range(3))
    def test_fanin_and_shrink(self, seed):
        rng = np.random.default_rng(seed)
        g = 24
        u = rng.choice([-1.0, 1.0], size=g)
        thr = 0.5
        calib = rng.integers(0, 2, size=(300, g)).astype(np.int64)
        spec = DecomposeSpec(alloc='uniform', bits_uniform=2)
        circ = decompose_threshold_neuron(u, thr, calib, spec)
        widths = [g] + [len(lv) for lv in circ.levels]
        # 各levelのwire数は厳密減少(発散防止)
        assert all(a > b for a, b in zip(widths[:-1], widths[1:])), widths
        assert widths[-1] == 1  # 出力ビット
        n_luts = 0
        for li, level in enumerate(circ.levels):
            for node in level:
                assert 1 <= len(node.srcs) <= 6
                assert 0 <= node.tt < (1 << (1 << len(node.srcs)))
                assert all(0 <= s < widths[li] for s in node.srcs)
                n_luts += 1
        assert circ.meta['n_luts'] == n_luts
        assert circ.meta['depth'] == len(circ.levels)

    def test_eval_matches_manual_two_level(self):
        """手組みの2段回路期待値とeval_circuitの一致(評価器自体の検算)。"""
        # 6+6=12本, uniform 2bit → 第1段4LUT+第2段1LUT想定の小回路
        rng = np.random.default_rng(11)
        u = rng.choice([-1.0, 1.0], size=12)
        thr = -0.5
        X = _all_patterns(12)[:512]
        circ = decompose_threshold_neuron(
            u, thr, X[:200], DecomposeSpec(alloc='uniform', bits_uniform=2))
        # 素朴評価: wireを1本ずつpython intで追う
        for row in X[:64]:
            wires = list(row)
            for level in circ.levels:
                nxt = []
                for node in level:
                    addr = sum(int(wires[s]) << j
                               for j, s in enumerate(node.srcs))
                    nxt.append((node.tt >> addr) & 1)
                wires = nxt
            got = eval_circuit(circ, row[None, :])[0]
            assert got == wires[0]
