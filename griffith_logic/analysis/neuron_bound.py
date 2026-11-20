"""ニューロン単位のLUT下界解析(multilut-decomposition Phase 1)。

1ニューロンのN点部分関数 (X, y) — 較正データを流したときの
(入力ビットパターン, 出力ビット) ペア — に対して:

- 固定入力部分集合を持つ単一LUTの最良agreementは majority-fill
  (アドレスごとに多数決ラベルを置く)で閉形式厳密に求まる。したがって
  「最良のk本選択」はSAT不要で部分集合列挙(best_subset_select)により
  厳密に得られる。SATは深さ≥2・複数LUT回路の実現可能性判定
  (min_circuit_search)のみに使い、sat_capacity.can_realize
  (readout='bit', dataset部分関数モード)へ委譲する。
- 下界の健全性: UNSATのみを「実現不能」の証拠とする。conflict予算・
  壁時計予算の超過は 'unknown' として厳格に区別する(偽UNSAT教訓)。

pysatはmin_circuit_search内で遅延importする(analysisパッケージ規約)。
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

# bincount直接集計を使う部分集合サイズの上限(2^k個のカウンタ)
_MAX_BINCOUNT_BITS = 16


def _addresses(X: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """部分集合列を束ねたLUTアドレス(バタフライ基底順: bit j = 列j)。"""
    Xi = np.asarray(X, dtype=np.int64)
    sub = np.asarray(subset, dtype=np.int64)
    weights = np.left_shift(1, np.arange(len(sub), dtype=np.int64))
    return Xi[:, sub] @ weights


def majority_fill_agreement(X: np.ndarray, y: np.ndarray,
                            subset) -> float:
    """入力部分集合subsetを持つ単一LUTの最良agreement(閉形式・厳密)。

    アドレスごとに多数決ラベルを置くLUTが最適であることから、
    Σ_addr max(count0, count1) / N を返す。SAT不要。

    Args:
        X: (N, M) の0/1入力パターン
        y: (N,) の0/1出力ビット
        subset: LUT入力に使う列インデックス列(長さ≤16)

    Returns:
        最良agreement ∈ [0.5, 1.0]付近(データ依存)
    """
    sub = np.asarray(subset, dtype=np.int64)
    assert len(sub) <= _MAX_BINCOUNT_BITS, (
        f'部分集合サイズ{len(sub)}はbincount上限{_MAX_BINCOUNT_BITS}超')
    yv = np.asarray(y, dtype=np.float64)
    addr = _addresses(X, sub)
    n_addr = 1 << len(sub)
    cnt = np.bincount(addr, minlength=n_addr).astype(np.float64)
    ones = np.bincount(addr, weights=yv, minlength=n_addr)
    return float(np.maximum(ones, cnt - ones).sum() / len(yv))


def consistency_ceiling(X: np.ndarray, y: np.ndarray) -> float:
    """全候補入力を見た場合の情報理論的天井。

    同一入力パターンに異なるラベルが付く点(ラベル衝突)は、どんな
    決定的回路でも片方しか当てられない。unique行グルーピングで計算
    するため入力幅Mに上限はない。
    """
    Xi = np.asarray(X, dtype=np.int64)
    yv = np.asarray(y, dtype=np.float64)
    _, inv = np.unique(Xi, axis=0, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    ones = np.bincount(inv, weights=yv)
    return float(np.maximum(ones, cnt - ones).sum() / len(yv))


def greedy_agreement_select(X: np.ndarray, y: np.ndarray,
                            k: int = 6) -> tuple[np.ndarray, float]:
    """前向き貪欲特徴選択: 各stepでmajority-fill agreementを最大化する列を追加。

    Returns:
        (選択列インデックス (k,), 最終agreement)
    """
    m = np.asarray(X).shape[1]
    assert k <= m, (k, m)
    chosen: list[int] = []
    best_val = 0.0
    for _ in range(k):
        step_best, step_col = -1.0, -1
        for c in range(m):
            if c in chosen:
                continue
            v = majority_fill_agreement(X, y, chosen + [c])
            if v > step_best:
                step_best, step_col = v, c
        chosen.append(step_col)
        best_val = step_best
    return np.array(chosen, dtype=np.int64), best_val


def best_subset_select(X: np.ndarray, y: np.ndarray, k: int = 6,
                       max_subsets: int = 2_000_000
                       ) -> tuple[np.ndarray, float]:
    """C(M,k)全列挙による厳密最良k本(単一LUTの大域最適)。

    majority-fillが閉形式なので、部分集合列挙だけで「SATが見つけ得る
    最良のk本」と同じ答えが厳密に出る(M≤32, k=6でC(32,6)≈906k)。

    Returns:
        (最良列インデックス (k,), そのagreement)
    """
    Xi = np.asarray(X, dtype=np.int64)
    yv = np.asarray(y, dtype=np.float64)
    n, m = Xi.shape
    n_sub = math.comb(m, k)
    assert n_sub <= max_subsets, (
        f'C({m},{k})={n_sub}が上限{max_subsets}超(Mを絞るか上限を上げる)')
    weights = np.left_shift(1, np.arange(k, dtype=np.int64))
    n_addr = 1 << k
    best_val, best_sub = -1.0, None
    for sub in itertools.combinations(range(m), k):
        addr = Xi[:, sub] @ weights
        cnt = np.bincount(addr, minlength=n_addr).astype(np.float64)
        ones = np.bincount(addr, weights=yv, minlength=n_addr)
        v = float(np.maximum(ones, cnt - ones).sum())
        if v > best_val:
            best_val, best_sub = v, sub
    return np.array(best_sub, dtype=np.int64), best_val / n


def min_circuit_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    arity: int = 6,
    depths: tuple[int, ...] = (1, 2, 3),
    width_grid: tuple[int, ...] = (1, 2, 4, 8, 16),
    conf_budget: int | None = 200_000,
    wall_budget_s: float = 540.0,
    seed: int = 0,
    solver: str = 'cadical195',
) -> dict:
    """部分関数(X, y)を厳密実現する最小(深さd, 幅k)をSATで探索する。

    深さdごとに layer_widths=[k]*(d-1)+[1](readout='bit', 自由配線)を
    幅グリッド昇順に判定し、最初のSATを最小幅とする。d=1でSATなら
    1LUTが大域最小なので後続深さは探索しない。

    健全性の規約:
        - 'unsat' のみが「その(d,k)で実現不能」の証拠
        - conflict予算超過は 'unknown'、壁時計予算超過は 'skipped(wall)'
          として記録し、いずれも unsat とは区別する

    Returns:
        {'per_depth': {d: {'status', 'min_width', 'largest_unsat_width',
                           'tried': {k: status}, 'total_luts'}},
         'best': {'depth', 'width', 'total_luts'} | None,
         'witness': 最良回路のRealizeResult | None(JSON非対応、呼び手で
         取り出す), 'topology': そのLutTopology | None,
         'wall_s': 総経過秒}
    """
    from .sat_capacity import LutTopology, TargetSpec, can_realize

    Xi = np.asarray(X, dtype=np.int64)
    yv = np.asarray(y, dtype=np.int64)
    m = Xi.shape[1]
    target = TargetSpec(dataset=(Xi, yv))
    t0 = time.monotonic()
    per_depth: dict[int, dict] = {}
    best: dict | None = None
    best_witness = None
    best_topology = None

    for d in depths:
        tried: dict[int, str] = {}
        min_w: int | None = None
        largest_unsat: int | None = None
        has_undecided = False
        # d=1は最終層のみ(幅1固定)なのでグリッドは[1]
        grid = (1,) if d == 1 else width_grid
        for w in grid:
            if time.monotonic() - t0 >= wall_budget_s:
                tried[w] = 'skipped(wall)'
                has_undecided = True
                continue
            widths = [w] * (d - 1) + [1]
            topo = LutTopology(n_in=m, layer_widths=widths,
                               lut_arity=arity, readout='bit')
            r = can_realize(topo, target, wiring='free', seed=seed,
                            solver=solver, conf_budget=conf_budget)
            tried[w] = r.status
            if r.status == 'sat':
                min_w = w
                sat_result, sat_topo = r, topo
                break
            if r.status == 'unsat':
                largest_unsat = w
            else:
                has_undecided = True
        total = None if min_w is None else (d - 1) * min_w + 1
        status = ('sat' if min_w is not None
                  else 'unknown' if has_undecided else 'unsat')
        per_depth[d] = {
            'status': status,
            'min_width': min_w,
            'largest_unsat_width': largest_unsat,
            'tried': tried,
            'total_luts': total,
        }
        if total is not None and (best is None
                                  or total < best['total_luts']):
            best = {'depth': d, 'width': min_w, 'total_luts': total}
            best_witness, best_topology = sat_result, sat_topo
        if d == 1 and min_w is not None:
            break  # 1LUTより小さい回路は無い
    return {'per_depth': per_depth, 'best': best,
            'witness': best_witness, 'topology': best_topology,
            'wall_s': time.monotonic() - t0}
