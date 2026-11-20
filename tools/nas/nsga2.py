"""NAS探索戦略プラグイン: NSGA-II系親選択。

非劣ソート(fast non-dominated sort)+混雑度(crowding distance)による
親選択を提供する。目的は2軸: (最大化: hard acc, 最小化: total_luts)。

tools/nas/search.py の探索ループへ ``SearchConfig.strategy='nsga2'`` として
選択的に組み込まれる(既定 ``strategy='evolution'`` の挙動・乱数消費・
ゴールデンには一切影響しない。search.py の next_genome 参照)。

参照文献: Deb, K., et al. (2002). "A fast and elitist multiobjective
genetic algorithm: NSGA-II". IEEE TEC.
"""

from __future__ import annotations

import random

Point = tuple[float, float]  # (acc, total_luts) = (最大化, 最小化)


def _dominates(a: Point, b: Point) -> bool:
    """aがbを支配するか(acc最大化・total_luts最小化)。"""
    better_or_eq = a[0] >= b[0] and a[1] <= b[1]
    strictly = a[0] > b[0] or a[1] < b[1]
    return better_or_eq and strictly


def non_dominated_sort(points: list[Point]) -> list[list[int]]:
    """点群を非劣フロントへ分割する(fast non-dominated sort, Deb et al. 2002)。

    各点について「自分を支配する点の数」と「自分が支配する点の集合」を1回の
    全対比較(O(n^2))で求め、支配数0の点を第1フロントとして確定後、
    そのフロントが支配していた点の支配数を減算しながら次フロントを剥がしていく。

    Args:
        points: (acc, total_luts) のリスト。

    Returns:
        フロントのリスト。fronts[0] が第1フロント(非劣解)。各フロントは
        points へのインデックスの昇順リスト。全点がいずれか1つのフロントに
        属し、重複はない。
    """
    n = len(points)
    dominates_count = [0] * n
    dominated_by_me: list[list[int]] = [[] for _ in range(n)]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(points[p], points[q]):
                dominated_by_me[p].append(q)
            elif _dominates(points[q], points[p]):
                dominates_count[p] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if dominates_count[i] == 0]
    while current:
        fronts.append(sorted(current))
        nxt: list[int] = []
        for p in current:
            for q in dominated_by_me[p]:
                dominates_count[q] -= 1
                if dominates_count[q] == 0:
                    nxt.append(q)
        current = nxt
    return fronts


def crowding_distance(front: list[Point]) -> list[float]:
    """フロント内各点の混雑度を計算する(Deb et al. 2002)。

    各目的軸ごとに点をソートし、両端(その軸の最大/最小)は無限大とする。
    内側の点は隣接2点の差分をその軸のレンジで正規化して積算する
    (値が大きいほど周囲が疎=多様性への寄与が大きい)。

    Args:
        front: 単一フロント内の (acc, total_luts) のリスト。

    Returns:
        front と同じ順序・長さの混雑度リスト。要素数が2以下なら全てinf
        (境界の判別ができないため)。
    """
    n = len(front)
    if n <= 2:
        return [float('inf')] * n

    distance = [0.0] * n
    for obj_idx in range(2):
        order = sorted(range(n), key=lambda i: front[i][obj_idx])
        distance[order[0]] = float('inf')
        distance[order[-1]] = float('inf')
        lo = front[order[0]][obj_idx]
        hi = front[order[-1]][obj_idx]
        span = hi - lo
        if span == 0:
            continue
        for k in range(1, n - 1):
            prev_v = front[order[k - 1]][obj_idx]
            next_v = front[order[k + 1]][obj_idx]
            distance[order[k]] += (next_v - prev_v) / span
    return distance


def select_parent(rng: random.Random, ledger: list) -> int | None:
    """非劣ソート+混雑度で ledger から親 trial_id を1つ選ぶ(NSGA-II風)。

    第1フロント(非劣解)のみを親候補とし、混雑度が大きい(周囲が疎=より
    多様性に寄与する)個体ほど選ばれやすいよう重み付きサンプリングする。
    フロントが1点しかない場合はそれをそのまま返す。candidate(state=='done'
    かつ partial=False)が無ければNoneを返す(search.py側でランダム初期化に
    フォールバックする)。

    Args:
        rng: 呼び出し側で (seed, trial_id) から決定的に作られたRandom。
        ledger: tools/nas/search.py の Trial のリスト。

    Returns:
        選ばれた親のtrial_id。候補が無ければNone。
    """
    candidates = [t for t in ledger if t.state == 'done' and not t.partial]
    if not candidates:
        return None

    points: list[Point] = [(t.acc, t.total_luts) for t in candidates]
    fronts = non_dominated_sort(points)
    first_front = fronts[0]
    if len(first_front) == 1:
        return candidates[first_front[0]].trial_id

    front_points = [points[i] for i in first_front]
    dist = crowding_distance(front_points)
    finite = [d for d in dist if d != float('inf')]
    # infは有限最大の10倍相当の重みとして扱う(重み付き選択に使うため有限化)
    inf_weight = (max(finite) * 10.0) if finite else 1.0
    weights = [inf_weight if d == float('inf') else d for d in dist]
    if sum(weights) <= 0:
        weights = [1.0] * len(weights)

    chosen = rng.choices(range(len(first_front)), weights=weights, k=1)[0]
    return candidates[first_front[chosen]].trial_id
