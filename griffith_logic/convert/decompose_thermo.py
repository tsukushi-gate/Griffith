"""温度計部分和分解: 1閾値ニューロン → fan-in≤6 LUT小回路(Phase 2)。

対象は重み付き閾値ニューロン y = 1[Σ_i w_i·x_i > T](x∈{0,1})。
第1段(ternary転写)では w∈{-1,+1}、再帰段では w=温度計復元差分(正実数)。

分解の構造:
    1. 入力をグループ(各≤group_size本)へ分割
    2. 各グループの部分和 s_g を温度計符号 t_{g,k}=1[s_g ≥ θ_{g,k}] で
       粗量子化する。各温度計ビットはfan-in≤6の閾値関数=**1個のLUT**
    3. 復元 ŝ_g = c_{g,0} + Σ_k δ_{g,k}·t_{g,k}(δ>0)により、次段は
       y = 1[Σδ·t > T'] の正重み閾値ニューロンに帰着 → 同じ手続きを再帰
    4. 入力本数≤group_sizeで base case: 単一LUTに厳密列挙

誤差保証(本手法の中核):
    |ŝ − s| ≤ Σ_{level} Σ_g e_g^max が構成的に成り立つため、
    マージン |s−T| > error_bound の入力では回路出力が元ニューロンと
    厳密一致する。較正マージン分位点 M_q に対し error_bound ≤ α·M_q
    となるよう適応ビット割当(allocate_bits)する。

隠れ層に算術は置かない(集約はすべてLUT)。回路は純関数表現
(NeuronCircuit)で構築し、IR化は上位で _lut_layer_from_cols に委譲する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_ROUND = 9  # 部分和値の同値判定桁(δは2進小数なので厳密に丸まる)


# ============================================================
# 回路の純関数表現
# ============================================================

@dataclass
class LutNodeSpec:
    """1個のLUT: 前levelワイヤsrcsを入力とする真理値表tt(butterfly順)。"""

    tt: int
    srcs: list[int]


@dataclass
class NeuronCircuit:
    """1ニューロン分の多段LUT回路。levels[-1]は長さ1(出力ビット)。"""

    levels: list[list[LutNodeSpec]]
    meta: dict = field(default_factory=dict)


def eval_circuit(circ: NeuronCircuit, X: np.ndarray) -> np.ndarray:
    """回路の参照評価。X: (N, G) 0/1 → (N,) 出力ビット。"""
    wires = np.asarray(X, dtype=np.int64)
    for level in circ.levels:
        cols = []
        for node in level:
            n = len(node.srcs)
            addr = np.zeros(wires.shape[0], dtype=np.int64)
            for j, s in enumerate(node.srcs):
                addr |= wires[:, s] << j
            # ttはPython int(N=6で64bit)。LUT展開でnumpy overflowを回避
            table = np.array([(node.tt >> k) & 1 for k in range(1 << n)],
                             dtype=np.int64)
            cols.append(table[addr])
        wires = np.stack(cols, axis=1)
    assert wires.shape[1] == 1, '最終levelは1ビットのはず'
    return wires[:, 0]


# ============================================================
# 部品: 到達和・最適温度計量子化・グループ分割・ビット割当
# ============================================================

def reachable_sums(weights: np.ndarray) -> np.ndarray:
    """部分集合和の到達可能値集合(昇順)。len(weights)≤6想定(≤64値)。"""
    w = np.asarray(weights, dtype=np.float64)
    assert len(w) <= 20, f'到達和列挙は本数≤20のみ: {len(w)}'
    sums = np.array([0.0])
    for wi in w:
        sums = np.unique(np.round(np.concatenate([sums, sums + wi]),
                                  _ROUND))
    return sums


def optimal_thermo_quantizer(
    values: np.ndarray,
    probs: np.ndarray,
    n_thresholds: int,
    objective: str = 'max_err',
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """離散値集合のbビット温度計量子化の厳密最適(DP)。

    値集合を n_thresholds+1 個の連続区間(bin)へ分割する。

    objective='max_err': max誤差(bin半径の最大)を第一目的、期待絶対誤差を
        第二目的で最小化。復元レベルはbin中点(max誤差最適)
    objective='exp_err': 較正分布上の期待絶対誤差のみを最小化。復元レベルは
        bin内の確率加重平均(不偏 → 多群の誤差が相殺しやすい)。max_err
        (保証上界)は事後計算で返す

    Returns:
        (thresholds 昇順(≤b,), recon 各binの復元レベル(bins,),
         max_err, exp_err)
    """
    assert objective in ('max_err', 'exp_err'), objective
    order = np.argsort(values)
    v = np.asarray(values, dtype=np.float64)[order]
    pr = np.asarray(probs, dtype=np.float64)[order]
    n = len(v)
    assert n <= 128, f'値集合が大きすぎる: {n}(グループ本数≤6なら≤64)'
    n_bins = min(n_thresholds + 1, n)

    # bin[i..j] のコスト前計算(cumsumでO(n^2)、復元レベル別)
    cp = np.concatenate([[0.0], np.cumsum(pr)])          # P[k] = Σ_{<k} pr
    cpv = np.concatenate([[0.0], np.cumsum(pr * v)])     # Σ_{<k} pr·v
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    half = (v[jj] - v[ii]) / 2.0                          # 中点復元のmax誤差
    seg_p = cp[jj + 1] - cp[ii]
    seg_pv = cpv[jj + 1] - cpv[ii]
    mid = (v[ii] + v[jj]) / 2.0
    with np.errstate(invalid='ignore'):
        mu = np.where(seg_p > 0, seg_pv / np.where(seg_p > 0, seg_p, 1.0),
                      mid)
    mean_rec = mu

    def _exp_err_matrix(rec: np.ndarray) -> np.ndarray:
        """復元レベル行列recでの期待絶対誤差 Σ_{k∈[i,j]} pr_k|v_k−rec_ij|。

        split = 最初にv_k≥recとなるk。|v−rec|の和を上下に分けてcumsumで取る。
        """
        split = np.searchsorted(v, rec.ravel()).reshape(rec.shape)
        split = np.clip(split, ii, jj + 1)  # binの内側へ
        p_lo = cp[split] - cp[ii]
        pv_lo = cpv[split] - cpv[ii]
        p_hi = cp[jj + 1] - cp[split]
        pv_hi = cpv[jj + 1] - cpv[split]
        return (rec * p_lo - pv_lo) + (pv_hi - rec * p_hi)

    exp_mid = _exp_err_matrix(mid)
    exp_mean = _exp_err_matrix(mu)
    max_mean = np.maximum(np.abs(v[ii] - mu), np.abs(v[jj] - mu))

    inf = float('inf')

    if objective == 'max_err':
        # DP1: max誤差の最小値(内側ループをnumpyで縮約)
        dp = np.full((n_bins + 1, n + 1), inf)
        dp[0][0] = 0.0
        for b in range(1, n_bins + 1):
            for j in range(1, n + 1):
                lo = b - 1
                if lo >= j:
                    continue
                dp[b][j] = np.maximum(dp[b - 1][lo:j],
                                      half[lo:j, j - 1]).min()
        best_max = dp[n_bins][n]
        cost, gate = exp_mid, best_max
    else:
        cost, gate = exp_mean, None

    # DP2: (max_err時はmax誤差≤最適の制約下で)期待誤差最小の分割を復元
    dp2 = np.full((n_bins + 1, n + 1), inf)
    back = np.full((n_bins + 1, n + 1), -1, dtype=np.int64)
    dp2[0][0] = 0.0
    for b in range(1, n_bins + 1):
        for j in range(1, n + 1):
            lo = b - 1
            if lo >= j:
                continue
            cand = dp2[b - 1][lo:j] + cost[lo:j, j - 1]
            if gate is not None:
                cand = np.where(half[lo:j, j - 1] > gate + 1e-12, inf,
                                cand)
            k = int(np.argmin(cand))
            if cand[k] < dp2[b][j]:
                dp2[b][j] = cand[k]
                back[b][j] = lo + k
    # 分割境界の復元
    bounds = [n]
    j = n
    for b in range(n_bins, 0, -1):
        j = int(back[b][j])
        bounds.append(j)
    bounds = bounds[::-1]  # [0, ..., n]

    if objective == 'max_err':
        recon = np.array([(v[s] + v[e - 1]) / 2.0
                          for s, e in zip(bounds[:-1], bounds[1:], strict=True)])
        max_err = float(max(half[s][e - 1]
                            for s, e in zip(bounds[:-1], bounds[1:], strict=True)))
    else:
        recon = np.array([mean_rec[s][e - 1]
                          for s, e in zip(bounds[:-1], bounds[1:], strict=True)])
        max_err = float(max(max_mean[s][e - 1]
                            for s, e in zip(bounds[:-1], bounds[1:], strict=True)))
        # 復元レベルの単調性(温度計の前提)を保証: 平均は bin順で
        # 単調非減少だが同値になり得るため、微小に分離する
        for k in range(1, len(recon)):
            if recon[k] <= recon[k - 1]:
                recon[k] = recon[k - 1] + 1e-9
    exp_err = float(dp2[n_bins][n])
    return (np.array([(v[e - 1] + v[e]) / 2.0 for e in bounds[1:-1]]),
            recon, max_err, exp_err)


def partition_support(weights: np.ndarray, spec: DecomposeSpec
                      ) -> list[np.ndarray]:
    """入力をグループ(各≤group_size本)へ分割する。

    balanced: 符号を交互配置(グループ内部分和の分布を0近傍へ寄せ、
              適応binの期待誤差を下げる)。sorted: |w|降順の連続チャンク。
              random: シャッフル。
    """
    w = np.asarray(weights, dtype=np.float64)
    m = len(w)
    if spec.grouping == 'sorted':
        order = np.argsort(-np.abs(w), kind='stable')
    elif spec.grouping == 'random':
        order = np.random.default_rng(spec.seed).permutation(m)
    else:  # balanced
        pos = np.argsort(-np.where(w > 0, np.abs(w), -np.inf),
                         kind='stable')[:int((w > 0).sum())]
        neg = np.argsort(-np.where(w <= 0, np.abs(w), -np.inf),
                         kind='stable')[:int((w <= 0).sum())]
        order = np.empty(m, dtype=np.int64)
        inter = []
        pi = ni = 0
        for i in range(m):
            take_pos = (pi < len(pos)) and (i % 2 == 0 or ni >= len(neg))
            if take_pos:
                inter.append(pos[pi])
                pi += 1
            else:
                inter.append(neg[ni])
                ni += 1
        order = np.array(inter, dtype=np.int64)
    return [order[i:i + spec.group_size]
            for i in range(0, m, spec.group_size)]


def allocate_bits(errs_by_bits: list[list[float]], budget: float,
                  spec: DecomposeSpec) -> list[int]:
    """適応ビット割当: Σ_g e_g(b_g) ≤ budget を最小総ビットで満たす貪欲。

    Args:
        errs_by_bits: errs_by_bits[g][b] = 群gにbビット割当時のmax誤差
                      (b=0..len-1、単調非増加、末尾は0=厳密)
        budget: このlevelの誤差予算
    Returns:
        群ごとのビット数(予算を満たせない場合は全群厳密=最大ビット)
    """
    n_groups = len(errs_by_bits)
    if spec.alloc == 'uniform':
        return [min(spec.bits_uniform, len(e) - 1) for e in errs_by_bits]
    bits = [0] * n_groups
    total = sum(e[0] for e in errs_by_bits)
    while total > budget + 1e-12:
        best_gain, best_g = 0.0, -1
        for g, e in enumerate(errs_by_bits):
            if bits[g] + 1 < len(e):
                gain = e[bits[g]] - e[bits[g] + 1]
                if gain > best_gain:
                    best_gain, best_g = gain, g
        if best_g < 0:
            break  # 全群厳密でも予算超過(margin極小)→現状bestで返す
        total -= best_gain
        bits[best_g] += 1
    return bits


# ============================================================
# 分解本体
# ============================================================

@dataclass
class DecomposeSpec:
    """温度計分解の設定。"""

    group_size: int = 6
    alloc: str = 'adaptive'          # 'adaptive' | 'uniform'
    bits_uniform: int = 3
    margin_quantile: float = 0.05    # 予算の基準分位点
    margin_alpha: float = 0.5        # 予算 = α × マージン分位点
    grouping: str = 'balanced'       # 'balanced' | 'sorted' | 'random'
    objective: str = 'max_err'       # 'max_err'(保証優先) | 'exp_err'(不偏)
    max_bits_per_group: int = 6      # 群あたり温度計ビット数の上限
    shrink_ratio: float = 0.75       # 各levelの幾何縮小保証(Σbits≤比×幅)
    stage2: str = 'recursive'        # 'recursive' | 'sat'
    sat_max_inputs: int = 16         # SAT合成を試す残り入力本数の上限
    max_levels: int = 64
    sat_conf_budget: int = 200_000
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.group_size <= 6, 'LUT fan-in上限は6'
        assert self.alloc in ('adaptive', 'uniform'), self.alloc
        assert self.grouping in ('balanced', 'sorted', 'random'), \
            self.grouping
        assert self.objective in ('max_err', 'exp_err'), self.objective


def _witness_to_levels(result, topology) -> list[list[LutNodeSpec]]:
    """can_realizeのSAT witnessをLutNodeSpecのlevel列へ変換する。"""
    levels = []
    for li, width in enumerate(topology.layer_widths):
        tts = result.truth_tables[li]
        wiring = result.wiring[li]  # (arity, width)
        levels.append([
            LutNodeSpec(tt=int(tts[o]),
                        srcs=[int(wiring[j][o])
                              for j in range(topology.lut_arity)])
            for o in range(width)
        ])
    return levels


def _synthesize_sat_tail(w: np.ndarray, thr: float, wires: np.ndarray,
                         spec: DecomposeSpec):
    """残り関数 1[Σw·t > thr] の較正パターン部分関数をSATで最小回路化する。

    幅グリッドは再帰tailのLUT数(≈次level+base)を超えない範囲
    ([1], [2,1], [4,1])に限定。SAT時は can_realize(verify=True) の
    witness→IR照合が内蔵で走る(偽SAT対策)。unknown/unsatはNone
    (再帰へフォールバック)。

    注意: SAT tailは較正パターン上でのみ厳密。未観測パターンの挙動は
    任意なので、error_boundの安全マージン保証は較正集合上に限定される。
    """
    try:
        from ..analysis.sat_capacity import LutTopology, TargetSpec, can_realize
    except ImportError:  # pragma: no cover
        return None
    y = (wires @ w > thr).astype(np.int64)
    for widths in ([1], [2, 1], [4, 1]):
        topo = LutTopology(n_in=len(w), layer_widths=widths,
                           lut_arity=6, readout='bit')
        try:
            r = can_realize(topo, TargetSpec(dataset=(wires, y)),
                            wiring='free',
                            conf_budget=spec.sat_conf_budget, verify=True)
        except RuntimeError:  # pysat未導入
            return None
        if r.status == 'sat':
            return _witness_to_levels(r, topo)
    return None


def _threshold_tt(w: np.ndarray, thr: float) -> int:
    """閾値関数 1[Σw·x > thr] のfan-in≤6真理値表(butterfly順)。"""
    n = len(w)
    assert n <= 6, n
    tt = 0
    for addr in range(1 << n):
        s = sum(w[j] for j in range(n) if (addr >> j) & 1)
        if s > thr:
            tt |= 1 << addr
    return tt


def _empirical_probs(vals: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """到達値集合vals上の経験確率(較正データobservedから)。"""
    vr = np.round(vals, _ROUND)
    obs = np.round(observed, _ROUND)
    idx = np.searchsorted(vr, obs)
    idx = np.clip(idx, 0, len(vr) - 1)
    ok = np.isclose(vr[idx], obs)
    counts = np.bincount(idx[ok], minlength=len(vr)).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else np.full(len(vr), 1 / len(vr))


def decompose_threshold_neuron(
    w: np.ndarray,
    thr: float,
    calib_bits: np.ndarray,
    spec: DecomposeSpec,
) -> NeuronCircuit:
    """重み付き閾値ニューロン 1[Σw·x > thr] を多段LUT回路へ分解する。

    Args:
        w: (G,) 実数重み(第1段はternary符号±1)
        thr: 閾値
        calib_bits: (N, G) 較正二値入力(マージン予算と適応割当に使用)
        spec: 分解設定

    Returns:
        NeuronCircuit。meta: error_bound(構成的誤差上界)、budget_met、
        n_luts、depth、margin_q、margin_min、level_bounds
    """
    w = np.asarray(w, dtype=np.float64)
    calib = np.asarray(calib_bits, dtype=np.int64)
    assert calib.ndim == 2 and calib.shape[1] == len(w), calib.shape

    s_cal = calib @ w
    margins = np.abs(s_cal - thr)
    m_q = float(np.quantile(margins, spec.margin_quantile)) if len(margins) \
        else 0.0
    budget_total = spec.margin_alpha * m_q

    levels: list[list[LutNodeSpec]] = []
    level_bounds: list[float] = []
    cur_w, cur_thr = w, float(thr)
    cur_wires = calib
    bound = 0.0
    stage2_used = 'recursive'

    while len(cur_w) > spec.group_size:
        assert len(levels) < spec.max_levels, (
            f'分解が{spec.max_levels}段で収束しない(G={len(w)})')

        # SAT第2段合成: 残り入力が窓内なら部分関数の最小回路化を試す
        if spec.stage2 == 'sat' and len(cur_w) <= spec.sat_max_inputs:
            tail = _synthesize_sat_tail(cur_w, cur_thr, cur_wires, spec)
            if tail is not None:
                levels.extend(tail)
                level_bounds.extend([0.0] * len(tail))
                return _finish(levels, level_bounds, bound, budget_total,
                               m_q, margins, stage2='sat')
            stage2_used = 'recursive_fallback'
        level_budget = max(budget_total - bound, 0.0) * 0.5
        groups = partition_support(cur_w, spec)

        # 群ごとの到達値・経験確率・全ビット数の量子化前計算
        ginfos = []
        for gidx in groups:
            vals = reachable_sums(cur_w[gidx])
            gsum = cur_wires[:, gidx] @ cur_w[gidx]
            probs = _empirical_probs(vals, gsum)
            n_q = min(len(vals), spec.max_bits_per_group + 1)
            quants = [optimal_thermo_quantizer(vals, probs, b,
                                               objective=spec.objective)
                      for b in range(n_q)]
            ginfos.append({'idx': gidx, 'vals': vals, 'gsum': gsum,
                           'quants': quants})

        # 適応割当の基準: max_err目的はmax誤差(保証と同軸)、exp_err目的は
        # 期待誤差(DP最適性よりbに単調 — max誤差は平均復元では非単調になり
        # 得て、貪欲gainが負→全群0bitの崩壊を起こすため使わない)
        err_idx = 2 if spec.objective == 'max_err' else 3
        errs = [[q[err_idx] for q in gi['quants']] for gi in ginfos]
        bits = allocate_bits(errs, level_budget, spec)

        # 収束保証: 幾何縮小(Σbits ≤ shrink_ratio×幅)までビットを削る。
        # 幅-1ずつの縮小を許すとG≈1600でmax_levels超過するため、
        # 予算未達(budget_met=False)を受け入れて深さを対数に抑える。
        # 削る対象は誤差増加が最小の群
        shrink_cap = max(1, int(spec.shrink_ratio * len(cur_w)))
        while sum(bits) > shrink_cap:
            best_loss, best_g = float('inf'), -1
            for g in range(len(bits)):
                if bits[g] > 0:
                    loss = errs[g][bits[g] - 1] - errs[g][bits[g]]
                    if loss < best_loss:
                        best_loss, best_g = loss, g
            assert best_g >= 0, 'ビットを削れない(group_size>1なら不可能)'
            bits[best_g] -= 1

        if sum(bits) == 0:
            # 全群定数寄与 → 出力は定数。現在ワイヤ0番に定数LUTを張る
            offset = sum(float(gi['quants'][0][1][0]) for gi in ginfos)
            lvl_err = sum(e[0] for e in errs)
            bound += lvl_err
            level_bounds.append(lvl_err)
            const = 1 if offset > cur_thr else 0
            levels.append([LutNodeSpec(tt=0b11 if const else 0b00,
                                       srcs=[0])])
            return _finish(levels, level_bounds, bound, budget_total,
                           m_q, margins, stage2=stage2_used)

        # LUT出力(温度計ビット)の生成
        nodes: list[LutNodeSpec] = []
        new_w: list[float] = []
        new_cols: list[np.ndarray] = []
        offset = 0.0
        lvl_err = 0.0
        for gi, b in zip(ginfos, bits, strict=True):
            thresholds, recon, max_err, _ = gi['quants'][b]
            lvl_err += max_err
            offset += float(recon[0])
            deltas = np.diff(recon)
            srcs = [int(c) for c in gi['idx']]
            for k, theta in enumerate(thresholds):
                # θは到達値の狭間にあるため 1[s≥θ] == 1[s>θ]
                nodes.append(LutNodeSpec(
                    tt=_threshold_tt(cur_w[gi['idx']], float(theta)),
                    srcs=srcs))
                new_w.append(float(deltas[k]))
                new_cols.append((gi['gsum'] > theta).astype(np.int64))

        levels.append(nodes)
        level_bounds.append(lvl_err)
        bound += lvl_err
        cur_thr = cur_thr - offset
        cur_w = np.array(new_w)
        cur_wires = np.stack(new_cols, axis=1)

    # base case: 残り≤group_size本 → 単一LUTで厳密
    levels.append([LutNodeSpec(tt=_threshold_tt(cur_w, cur_thr),
                               srcs=list(range(len(cur_w))))])
    level_bounds.append(0.0)
    return _finish(levels, level_bounds, bound, budget_total, m_q, margins,
                   stage2=stage2_used)


def _finish(levels, level_bounds, bound, budget_total, m_q, margins,
            stage2: str = 'recursive') -> NeuronCircuit:
    meta = {
        'error_bound': float(bound),
        'budget_met': bool(bound <= budget_total + 1e-12),
        'budget_total': float(budget_total),
        'margin_q': float(m_q),
        'margin_min': float(margins.min()) if len(margins) else 0.0,
        'n_luts': sum(len(lv) for lv in levels),
        'depth': len(levels),
        'level_bounds': [float(b) for b in level_bounds],
        'stage2': stage2,
        # SAT tailは較正パターン上のみ厳密(安全マージン保証の適用範囲)
        'off_calib_guarantee': stage2 != 'sat',
    }
    return NeuronCircuit(levels=levels, meta=meta)


# ============================================================
# 層単位のIR組み立て
# ============================================================

_TT_ID, _TT_NOT, _TT_ZERO = 0x2, 0x1, 0x0  # fan-in1の恒等/否定/定数0


def _ternary_layer_nodes(
    q: np.ndarray,
    thr: np.ndarray,
    calib_bits: np.ndarray,
    spec: DecomposeSpec,
    *,
    node_id_prefix: str,
    input_id: str,
    in_width: int,
) -> tuple[list, list, str, dict]:
    """ternary層の分解ノード列を構築する(グラフ組み込み用コア)。

    Returns:
        (nodes, edges, last_node_id, decompose_meta)
    """
    from ..ir.schema import IREdge
    from .binarize import _lut_layer_from_cols

    q = np.asarray(q, dtype=np.int64)
    thr = np.asarray(thr, dtype=np.float64)
    calib = np.asarray(calib_bits, dtype=np.int64)
    n_out = q.shape[0]

    circs: list[NeuronCircuit] = []
    sups: list[np.ndarray] = []
    for r in range(n_out):
        sup = np.nonzero(q[r])[0]
        assert len(sup) >= 1, f'ニューロン{r}の支持集合が空'
        circs.append(decompose_threshold_neuron(
            q[r, sup].astype(np.float64), float(thr[r]),
            calib[:, sup], spec))
        sups.append(sup)

    depth = max(c.meta['depth'] for c in circs)
    n_pad = 0
    per_levels: list[list[list[LutNodeSpec]]] = []
    for c in circs:
        lv = [list(level) for level in c.levels]
        while len(lv) < depth:  # 出力ビットを恒等LUTで最終levelまで延長
            lv.append([LutNodeSpec(tt=_TT_ID, srcs=[0])])
            n_pad += 1
        per_levels.append(lv)

    nodes, edges = [], []
    prev_id, prev_width = input_id, in_width
    prev_offsets: list[int] | None = None  # 前levelの各ニューロン開始列
    for li in range(depth):
        cols: list[tuple[int, list[int]]] = []
        offsets: list[int] = []
        for r in range(n_out):
            offsets.append(len(cols))
            for spec_node in per_levels[r][li]:
                if li == 0:
                    srcs = [int(sups[r][s]) for s in spec_node.srcs]
                else:
                    srcs = [prev_offsets[r] + s for s in spec_node.srcs]
                cols.append((spec_node.tt, srcs))
        node_id = f'{node_id_prefix}_l{li}'
        nodes.append(_lut_layer_from_cols(
            node_id, prev_width, cols,
            {'kind': 'thermo_decompose', 'level': li}))
        edges.append(IREdge(src=prev_id, dst=node_id))
        prev_id, prev_width, prev_offsets = node_id, len(cols), offsets

    assert prev_width == n_out, (prev_width, n_out)
    n_core = sum(c.meta['n_luts'] for c in circs)
    meta = {
        'error_bounds': [c.meta['error_bound'] for c in circs],
        'budget_met': [c.meta['budget_met'] for c in circs],
        'stage2': [c.meta['stage2'] for c in circs],
        'n_luts_core': int(n_core),
        'n_luts_total': int(n_core + n_pad),
        'n_luts_pad': int(n_pad),
        'depth': int(depth),
        'spec': {
            'group_size': spec.group_size, 'alloc': spec.alloc,
            'bits_uniform': spec.bits_uniform,
            'margin_quantile': spec.margin_quantile,
            'margin_alpha': spec.margin_alpha,
            'grouping': spec.grouping, 'stage2': spec.stage2,
        },
    }
    return nodes, edges, prev_id, meta


def decompose_ternary_layer(
    q: np.ndarray,
    thr: np.ndarray,
    calib_bits: np.ndarray,
    spec: DecomposeSpec,
    *,
    node_id_prefix: str = 'dt',
):
    """ternary層 y_r = 1[Σ_j q_rj·x_j > thr_r] を多段LUT層IRへ分解する。

    ニューロンごとに decompose_threshold_neuron を適用し、深さ最大値に
    そろえて(不足levelは出力ビットの恒等LUTで延長)、level単位の
    lut_layerノード列として組み立てる。隠れに算術ノードは置かない。

    Args:
        q: (out, in) の{-1,0,+1}行列(0=非接続)
        thr: (out,) の実数閾値
        calib_bits: (N, in) の較正二値入力
        spec: 分解設定

    Returns:
        IRGraph('x'→lut_layer列→'y')。metadata['decompose'] に
        error_bounds / n_luts_core / n_luts_total / depth / budget_met
    """
    from ..ir.schema import IREdge, IRGraph, IRPort

    n_out, n_in = np.asarray(q).shape
    nodes, edges, last_id, meta = _ternary_layer_nodes(
        q, thr, calib_bits, spec,
        node_id_prefix=node_id_prefix, input_id='x', in_width=n_in)
    edges.append(IREdge(src=last_id, dst='y'))
    return IRGraph(
        metadata={'source': 'convert.decompose_ternary_layer',
                  'decompose': meta},
        inputs=[IRPort(id='x', width=n_in,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=n_out, semantics='neuron_bits')],
        nodes=nodes,
        edges=edges,
    )


def _adder_layer_nodes(
    q: np.ndarray,
    thr: np.ndarray,
    *,
    node_id_prefix: str,
    input_id: str,
    in_width: int,
) -> tuple[list, list, str, dict]:
    """baseline Bの層ノード列(グラフ組み込み用コア)。"""
    from ..ir.schema import IREdge
    from .binarize import _lut_layer_from_cols, popcount_threshold_nodes

    q = np.asarray(q, dtype=np.int64)
    thr = np.asarray(thr, dtype=np.float64)
    n_out = q.shape[0]
    sups = [np.nonzero(q[r])[0] for r in range(n_out)]
    assert all(len(s) >= 1 for s in sups), '支持集合が空のニューロン'
    width = max(len(s) for s in sups)

    pre_cols: list[tuple[int, list[int]]] = []
    input_cols = np.zeros((n_out, width), dtype=np.int64)
    for r, sup in enumerate(sups):
        for k, j in enumerate(sup):
            input_cols[r, k] = len(pre_cols)
            pre_cols.append((_TT_ID if q[r, j] > 0 else _TT_NOT, [int(j)]))
        for k in range(len(sup), width):  # 定数0パディング(popcount不変)
            input_cols[r, k] = len(pre_cols)
            pre_cols.append((_TT_ZERO, [0]))
    pre_id = f'{node_id_prefix}_pre'
    pre = _lut_layer_from_cols(pre_id, in_width, pre_cols,
                               {'kind': 'sign_fold'})

    thetas = [int(np.floor(thr[r] + int((q[r] < 0).sum()))) + 1
              for r in range(n_out)]
    pc_nodes, pc_edges, last_id, out_w = popcount_threshold_nodes(
        width=width,
        thresholds=thetas,
        node_id_prefix=f'{node_id_prefix}_pc',
        in_width=pre.params['out_width'],
        input_cols=input_cols,
    )
    assert out_w == n_out, (out_w, n_out)

    nodes = [pre, *pc_nodes]
    edges = [IREdge(src=input_id, dst=pre_id),
             IREdge(src=pre_id, dst=f'{node_id_prefix}_pc_p0'),
             *pc_edges]
    n_luts = sum(n.params['out_width'] for n in nodes
                 if n.type == 'lut_layer')
    meta = {
        'error_bounds': [0.0] * n_out,
        'n_luts_total': int(n_luts),
        'depth': int(sum(1 for n in nodes if n.type == 'lut_layer')),
    }
    return nodes, edges, last_id, meta


def exact_adder_baseline_layer(
    q: np.ndarray,
    thr: np.ndarray,
    *,
    node_id_prefix: str = 'pb',
):
    """baseline B: 符号折り込み+popcount加算木による厳密LUT化。

    ternaryは負重み入力をNOT反転すると
        1[Σq·x > T] = 1[popcount(z) ≥ ⌊T + |neg|⌋ + 1]
    に厳密帰着する(z_j = x_j if q=+1 else ¬x_j)。集約は既存
    popcount_threshold_nodes(carry-save加算木、全LUT実装)へ委譲。
    agreement=1.0が構成的に保証されるコスト上限アンカー。

    Returns:
        IRGraph('x'→符号折り込みlut_layer→popcount木→'y')
    """
    from ..ir.schema import IREdge, IRGraph, IRPort

    n_out, n_in = np.asarray(q).shape
    nodes, edges, last_id, meta = _adder_layer_nodes(
        q, thr, node_id_prefix=node_id_prefix, input_id='x',
        in_width=n_in)
    edges.append(IREdge(src=last_id, dst='y'))
    return IRGraph(
        metadata={'source': 'convert.exact_adder_baseline_layer',
                  'decompose': meta},
        inputs=[IRPort(id='x', width=n_in,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=n_out, semantics='neuron_bits')],
        nodes=nodes,
        edges=edges,
    )


# ============================================================
# MLP全体の分解(C3互換の分類グラフ)
# ============================================================

def ternary_from_linear(weight, bias) -> tuple[np.ndarray, np.ndarray]:
    """ternary値のLinear重みから (q, thr) を厳密復元する。

    重みが行ごとに q_rj·scale_r(q∈{-1,0,+1})の形であることをassertし、
    ニューロン意味論 1[W·x+b>0] ⟺ 1[Σq·x > thr](thr=-b/scale)へ
    正規化する。scaleはゼロでない|W|の行平均(厳密ternaryなら定数)。
    """
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    mask = w != 0
    assert mask.any(axis=1).all(), '全ゼロ行のニューロンがある'
    scale = np.where(mask, np.abs(w), 0.0).sum(axis=1) / mask.sum(axis=1)
    q = np.sign(w).astype(np.int64)
    assert np.allclose(w, q * scale[:, None], atol=1e-10), (
        '重みが厳密ternary(q·scale)でない: 事前にbakeすること')
    thr = -b / scale
    return q, thr


def decompose_mlp(
    model,
    calib_x: np.ndarray,
    spec: DecomposeSpec,
    *,
    hidden_arm: str = 'thermo',
    weight_bits_final: int = 8,
):
    """ternary値MLP(Linear列+Heaviside隠れ活性)を多段LUT分類IRへ分解する。

    隠れ層: ternary_from_linear で (q, thr) を厳密復元し、
        hidden_arm='thermo': 温度計部分和分解(arm C)
        hidden_arm='adder' : popcount加算木の厳密lowering(arm B)
    最終層: transcribe._final_layer_nodes(argmax厳密保存)+ GroupSum。
    較正は層ごとに厳密参照 1[q·x>thr] で伝搬する。

    Returns:
        IRGraph('x'→隠れ層列→final→group_sum→'y')。
        metadata['decompose']['layers'] に層別meta。
    """
    import torch

    from ..ir.schema import IREdge, IRGraph, IRPort
    from .transcribe import _final_layer_nodes

    assert hidden_arm in ('thermo', 'adder'), hidden_arm
    linears = [m for m in model.modules()
               if isinstance(m, torch.nn.Linear)]
    assert len(linears) >= 1, 'nn.Linearが見つからない'
    hidden, final = linears[:-1], linears[-1]

    calib = np.asarray(calib_x, dtype=np.int64)
    nodes, edges = [], []
    prev_id, prev_width = 'x', int(linears[0].in_features)
    layer_metas = []
    h = calib
    for li, lin in enumerate(hidden):
        w = lin.weight.detach().cpu().double().numpy()
        b = (lin.bias.detach().cpu().double().numpy()
             if lin.bias is not None else np.zeros(w.shape[0]))
        q, thr = ternary_from_linear(w, b)
        if hidden_arm == 'thermo':
            n_, e_, last_id, meta = _ternary_layer_nodes(
                q, thr, h, spec,
                node_id_prefix=f'n{li}_dt', input_id=prev_id,
                in_width=prev_width)
        else:
            n_, e_, last_id, meta = _adder_layer_nodes(
                q, thr, node_id_prefix=f'n{li}_pb', input_id=prev_id,
                in_width=prev_width)
        nodes += n_
        edges += e_
        layer_metas.append(meta)
        h = (h @ q.T > thr[None, :]).astype(np.int64)  # 厳密参照で較正伝搬
        prev_id, prev_width = last_id, q.shape[0]

    fw = final.weight.detach().cpu().double()
    fb = (final.bias.detach().cpu().double() if final.bias is not None
          else torch.zeros(fw.shape[0], dtype=torch.float64))
    lut_node, gs_node, g = _final_layer_nodes(
        fw, fb, weight_bits=weight_bits_final,
        node_id=f'n{len(hidden)}_final')
    nodes += [lut_node, gs_node]
    edges += [IREdge(src=prev_id, dst=lut_node.id),
              IREdge(src=lut_node.id, dst=gs_node.id),
              IREdge(src=gs_node.id, dst='y')]

    total = sum(m['n_luts_total'] for m in layer_metas) \
        + lut_node.params['out_width']
    return IRGraph(
        metadata={
            'source': 'convert.decompose_mlp',
            'decompose': {
                'hidden_arm': hidden_arm,
                'layers': layer_metas,
                'n_luts_total': int(total),
                'group_width': g,
            },
        },
        inputs=[IRPort(id='x', width=int(linears[0].in_features),
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=int(final.out_features),
                        semantics='class_scores')],
        nodes=nodes,
        edges=edges,
    )
