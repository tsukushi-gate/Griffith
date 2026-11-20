"""ternary層→多段LUT IRの組み立て(decompose_ternary_layer /
exact_adder_baseline_layer)のL1テスト。

正しさの機械的保証:
    1. 生成IRが validate を通過し、simulate出力形状が(batch, out)である
    2. 分解IR: マージン>誤差上界の全点で simulate == 参照 1[q·x>thr]
       (誤差上界0の設定では全点bit厳密の等値assert)
    3. baseline B(NOT反転+popcount加算木): 全点bit厳密(構成的保証)
    4. 深さ不揃いのニューロン混載(G=3とG=18)でも列順・厳密性が保たれる
    5. metadata にLUT数・深さが記録される
"""

import itertools

import numpy as np
import pytest

from griffith_logic.convert.decompose_thermo import (
    DecomposeSpec,
    decompose_ternary_layer,
    exact_adder_baseline_layer,
)
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _random_ternary_layer(seed: int, n_out: int, n_in: int,
                          density: float = 0.8):
    rng = np.random.default_rng(seed)
    q = rng.choice([-1, 0, 1], size=(n_out, n_in),
                   p=[density / 2, 1 - density, density / 2]).astype(np.int64)
    # 全ゼロ行を防ぐ
    for r in range(n_out):
        if (q[r] != 0).sum() < 2:
            q[r, rng.choice(n_in, 2, replace=False)] = 1
    thr = rng.integers(-2, 3, size=n_out).astype(np.float64) + 0.5
    return q, thr


def _reference(q: np.ndarray, thr: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (X @ q.T > thr[None, :]).astype(np.int64)


class TestDecomposeLayer:
    @pytest.mark.parametrize('seed', range(3))
    def test_validate_and_bound(self, seed):
        q, thr = _random_ternary_layer(seed, n_out=5, n_in=14)
        rng = np.random.default_rng(100 + seed)
        X = rng.integers(0, 2, size=(400, 14), dtype=np.int64)
        spec = DecomposeSpec(alloc='uniform', bits_uniform=2)
        graph = decompose_ternary_layer(q, thr, X[:200], spec)
        validate(graph)
        out = simulate(graph, X)
        assert out.shape == (400, 5)
        ref = _reference(q, thr, X)
        # ニューロンごとの誤差上界で安全点の厳密一致を確認
        bounds = graph.metadata['decompose']['error_bounds']
        for r in range(5):
            margins = np.abs(X @ q[r] - thr[r])
            safe = margins > bounds[r] + 1e-9
            assert (out[safe, r] == ref[safe, r]).all()

    def test_exact_when_bounds_zero(self):
        """G≤6のみの層(全base case)は誤差上界0で全点厳密。"""
        rng = np.random.default_rng(5)
        n_in = 10
        q = np.zeros((4, n_in), dtype=np.int64)
        for r in range(4):
            cols = rng.choice(n_in, 5, replace=False)
            q[r, cols] = rng.choice([-1, 1], size=5)
        thr = np.array([0.5, -0.5, 1.5, 0.5])
        X = np.array(list(itertools.product([0, 1], repeat=n_in)),
                     dtype=np.int64)
        graph = decompose_ternary_layer(q, thr, X[:256],
                                        DecomposeSpec())
        validate(graph)
        assert all(b == 0.0
                   for b in graph.metadata['decompose']['error_bounds'])
        assert (simulate(graph, X) == _reference(q, thr, X)).all()

    def test_mixed_depth_neurons(self):
        """④G=3(深さ1)とG=18(深さ≥2)の混載で列順・厳密性を保つ。"""
        rng = np.random.default_rng(9)
        n_in = 18
        q = np.zeros((3, n_in), dtype=np.int64)
        q[0, :3] = [1, -1, 1]                      # G=3 → base case
        q[1] = rng.choice([-1, 1], size=n_in)      # G=18 → 多段
        q[2, :7] = rng.choice([-1, 1], size=7)     # G=7 → 2段
        thr = np.array([0.5, 0.5, -0.5])
        X = rng.integers(0, 2, size=(500, n_in), dtype=np.int64)
        spec = DecomposeSpec(alloc='uniform', bits_uniform=3)
        graph = decompose_ternary_layer(q, thr, X[:250], spec)
        validate(graph)
        out = simulate(graph, X)
        ref = _reference(q, thr, X)
        bounds = graph.metadata['decompose']['error_bounds']
        assert bounds[0] == 0.0  # base caseニューロンは厳密
        assert (out[:, 0] == ref[:, 0]).all()
        for r in (1, 2):
            margins = np.abs(X @ q[r] - thr[r])
            safe = margins > bounds[r] + 1e-9
            assert (out[safe, r] == ref[safe, r]).all()

    def test_metadata_counts(self):
        q, thr = _random_ternary_layer(3, n_out=4, n_in=20)
        rng = np.random.default_rng(3)
        X = rng.integers(0, 2, size=(200, 20), dtype=np.int64)
        graph = decompose_ternary_layer(
            q, thr, X, DecomposeSpec(alloc='uniform', bits_uniform=2))
        md = graph.metadata['decompose']
        assert md['n_luts_total'] >= md['n_luts_core'] > 0
        assert md['depth'] >= 1
        assert len(md['error_bounds']) == 4


class TestExactAdderBaseline:
    """③baseline B: 全点bit厳密の構成的保証。"""

    @pytest.mark.parametrize('seed', range(3))
    def test_bit_exact(self, seed):
        q, thr = _random_ternary_layer(seed, n_out=4, n_in=12)
        graph = exact_adder_baseline_layer(q, thr)
        validate(graph)
        X = np.array(list(itertools.product([0, 1], repeat=12)),
                     dtype=np.int64)[::7]  # 間引き全域サンプル
        assert (simulate(graph, X) == _reference(q, thr, X)).all()

    def test_mixed_fanin_bit_exact(self):
        """fan-in不揃い(定数0パディング)でも厳密。"""
        n_in = 10
        q = np.zeros((3, n_in), dtype=np.int64)
        q[0, :2] = [1, -1]
        q[1, :9] = [1, 1, -1, 1, -1, -1, 1, 1, -1]
        q[2, 3:6] = [-1, -1, -1]
        thr = np.array([-0.5, 1.5, -2.5])
        graph = exact_adder_baseline_layer(q, thr)
        validate(graph)
        X = np.array(list(itertools.product([0, 1], repeat=n_in)),
                     dtype=np.int64)
        assert (simulate(graph, X) == _reference(q, thr, X)).all()

    def test_lut_count_recorded(self):
        q, thr = _random_ternary_layer(1, n_out=3, n_in=16)
        graph = exact_adder_baseline_layer(q, thr)
        assert graph.metadata['decompose']['n_luts_total'] > 0
