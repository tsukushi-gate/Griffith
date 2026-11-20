"""SAT第2段合成(decompose_thermo stage2='sat')の独立検証テスト。

偽UNSAT/偽SAT対策の3点セット(設計制約5):
    1. can_realize(verify=True) 内蔵のwitness→IRシミュレート照合(SAT側)
    2. 合成後回路を eval_circuit(独立実装の評価器)で較正全点照合
    3. recursive分解との突き合わせ: 同一ニューロン・同一第1段に対して
       SAT合成版とrecursive版が「マージン>誤差上界」の全点で同じ答え
       (=参照ニューロン)を出す
    4. SAT予算切れ(unknown)時はrecursiveへフォールバックして完走する
"""

import numpy as np
import pytest

pytest.importorskip('pysat')

from griffith_logic.convert.decompose_thermo import (  # noqa: E402
    DecomposeSpec,
    decompose_threshold_neuron,
    eval_circuit,
)


def _neuron(seed: int, g: int = 24):
    rng = np.random.default_rng(seed)
    u = rng.choice([-1.0, 1.0], size=g)
    thr = float(rng.integers(-2, 3)) + 0.5
    X = rng.integers(0, 2, size=(300, g), dtype=np.int64)
    return u, thr, X


class TestSatStage2:
    @pytest.mark.parametrize('seed', range(3))
    def test_sat_vs_recursive_agree_on_safe_points(self, seed):
        """③SAT合成とrecursiveの突き合わせ+②独立評価器での照合。"""
        u, thr, X = _neuron(seed)
        calib = X[:200]
        spec_sat = DecomposeSpec(alloc='uniform', bits_uniform=2,
                                 stage2='sat')
        spec_rec = DecomposeSpec(alloc='uniform', bits_uniform=2,
                                 stage2='recursive')
        c_sat = decompose_threshold_neuron(u, thr, calib, spec_sat)
        c_rec = decompose_threshold_neuron(u, thr, calib, spec_rec)

        y_true = (X @ u > thr).astype(np.int64)
        margins = np.abs(X @ u - thr)

        # 較正点上: SAT版は量子化後関数を厳密実現するため、
        # 安全マージン較正点で参照と一致する(recursiveと同保証)
        for circ in (c_sat, c_rec):
            bound = circ.meta['error_bound']
            safe = margins[:200] > bound + 1e-9
            got = eval_circuit(circ, calib)
            assert (got[safe] == y_true[:200][safe]).all()

        # SAT合成が実際に使われたか(なければfallback記録があるはず)
        assert c_sat.meta.get('stage2') in ('sat', 'recursive_fallback')

    def test_sat_reduces_or_matches_luts(self):
        """SAT成功時はrecursive以下のLUT数(最小化の趣旨確認、非厳密)。"""
        u, thr, X = _neuron(11, g=24)
        calib = X[:150]
        c_sat = decompose_threshold_neuron(
            u, thr, calib,
            DecomposeSpec(alloc='uniform', bits_uniform=2, stage2='sat'))
        c_rec = decompose_threshold_neuron(
            u, thr, calib,
            DecomposeSpec(alloc='uniform', bits_uniform=2,
                          stage2='recursive'))
        if c_sat.meta.get('stage2') == 'sat':
            assert c_sat.meta['n_luts'] <= c_rec.meta['n_luts']

    def test_unknown_falls_back_to_recursive(self):
        """④conf_budget=1でSATがunknown → recursiveで完走。"""
        u, thr, X = _neuron(5, g=24)
        spec = DecomposeSpec(alloc='uniform', bits_uniform=2,
                             stage2='sat', sat_conf_budget=1)
        circ = decompose_threshold_neuron(u, thr, X[:100], spec)
        assert circ.meta.get('stage2') == 'recursive_fallback'
        # 完走して評価可能
        assert eval_circuit(circ, X[:50]).shape == (50,)

    def test_calib_exact_bruteforce_regression(self):
        """②較正全点でSAT版回路==参照(全較正点照合の回帰)。

        SAT tailは較正パターン上でのみ厳密(off_calib_guarantee=False)
        なので、較正集合の安全マージン点全点をeval_circuit(SATと独立の
        評価器)でassertする。sat_max_inputs=8でtail窓に限定し、
        意図された使い方(第1段縮約後の小さな残り関数の最小化)を検証。
        """
        rng = np.random.default_rng(3)
        g = 24
        u = rng.choice([-1.0, 1.0], size=g)
        thr = 0.5
        X = rng.integers(0, 2, size=(1500, g), dtype=np.int64)
        spec = DecomposeSpec(alloc='uniform', bits_uniform=2,
                             stage2='sat', sat_max_inputs=8)
        circ = decompose_threshold_neuron(u, thr, X, spec)
        y_true = (X @ u > thr).astype(np.int64)
        margins = np.abs(X @ u - thr)
        bound = circ.meta['error_bound']
        safe = margins > bound + 1e-9
        got = eval_circuit(circ, X)
        assert (got[safe] == y_true[safe]).all()
        if circ.meta['stage2'] == 'sat':
            assert not circ.meta['off_calib_guarantee']
