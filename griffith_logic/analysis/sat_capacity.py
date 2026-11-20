"""SAT容量判定器: 「このLUTトポロジーで関数Fを実現できるか」の厳密判定。

Boolean matching(∃gates ∀inputs の2-QBF)の落とし込み。入力パターンを
全列挙(または訓練データ点のみ)して制約に展開することで、QBFを平坦な
SATに帰着する。ゲート変数はニューロンごとの2^arity真理値表ビット
(arity=2で4ビット=16ゲートの自由選択、arity=4/6でLUT4/LUT6。one-hot不要)。

用途:
    - 固定ランダム配線トポロジーの表現可能性証明(SAT/UNSAT)
    - 自由配線モード(配線もSAT変数化)→「配線税」の定量化
    - witness(ゲート割当)の抽出 → LutLayerモデルへの復元(SAT-init)
    - 部分関数モード: データ点のみ制約、残りはdon't care

規約(リポジトリ全体と一致):
    - LUTアドレス: addr = Σ_j x[indices[j,o]] << j (slot 0 = LSB)
    - 真理値表: bit k = butterfly基底順パターンkへの出力
    - GroupSum: 連続チャンクでkグループに分割、argmax同点はindex 0優先
      (torch.argmaxの規約) → y=0 ⟺ sum0 ≥ sum1、y=1 ⟺ sum1 ≥ sum0+1

依存: python-sat(optional extra 'sat')。未インストール環境では
import時でなく使用時にRuntimeErrorを送出する。
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:
    from pysat.card import CardEnc, EncType
    from pysat.formula import IDPool
    from pysat.solvers import Solver
    _PYSAT_OK = True
except ImportError:  # pragma: no cover - optional依存
    _PYSAT_OK = False


def _require_pysat() -> None:
    if not _PYSAT_OK:
        raise RuntimeError(
            'python-satが必要です: uv sync --extra sat でインストール'
        )


SOLVER_FALLBACKS = ('cadical195', 'glucose42', 'minisat22')


@dataclass
class LutTopology:
    """判定対象のLUT網トポロジー(arity∈{2,4,6}、GroupSum読み出し)。"""

    n_in: int
    layer_widths: list[int]
    lut_arity: int = 2
    readout: str = 'groupsum'  # 'groupsum' | 'bit'
    groupsum_k: int = 2

    def __post_init__(self) -> None:
        assert self.lut_arity in (2, 4, 6), (
            f'lut_arityは2/4/6のみ: {self.lut_arity}'
        )
        assert self.readout in ('groupsum', 'bit'), self.readout
        assert len(self.layer_widths) >= 1, '少なくとも1層必要'
        if self.readout == 'groupsum':
            assert self.groupsum_k >= 2, self.groupsum_k
            assert self.layer_widths[-1] % self.groupsum_k == 0, (
                f'最終層幅({self.layer_widths[-1]})はk({self.groupsum_k})の倍数'
            )


@dataclass
class TargetSpec:
    """標的関数の指定。3形式のうち1つを与える。

    - fn: bitsタプル -> ラベル(0/1)。全2^n_inパターンを制約
    - truth_table: 2^n_inビット整数(bit p = パターンpのラベル)。全列挙
    - dataset: (X, y)。Xの各行のみ制約(部分関数)。重複は多数決で解決
    dont_care_mask: 全列挙形式でパターンpを制約から外すマスク(True=DC)
    """

    fn: Callable | None = None
    truth_table: int | None = None
    dataset: tuple[np.ndarray, np.ndarray] | None = None
    dont_care_mask: np.ndarray | None = None

    def patterns_labels(self, n_in: int) -> tuple[np.ndarray, np.ndarray]:
        """制約対象の(パターン(P, n_in) 0/1配列, ラベル(P,))を返す。"""
        given = sum(v is not None
                    for v in (self.fn, self.truth_table, self.dataset))
        assert given == 1, 'fn / truth_table / dataset のどれか1つを指定'

        if self.dataset is not None:
            x = np.asarray(self.dataset[0], dtype=np.int64)
            y = np.asarray(self.dataset[1], dtype=np.int64)
            assert x.ndim == 2 and x.shape[1] == n_in, x.shape
            assert set(np.unique(x)).issubset({0, 1}), '入力は0/1のみ'
            # 重複行のdedup + ラベル衝突は多数決(同数は1優先しない=0優先)
            votes: dict[tuple, list[int]] = {}
            for row, label in zip(x.tolist(), y.tolist(), strict=True):
                votes.setdefault(tuple(row), []).append(int(label))
            pats = np.array(sorted(votes), dtype=np.int64)
            labels = np.array(
                [int(np.mean(votes[tuple(r)]) > 0.5) for r in pats.tolist()],
                dtype=np.int64,
            )
            return pats, labels

        pats = np.array(
            list(itertools.product([0, 1], repeat=n_in)), dtype=np.int64,
        )
        if self.truth_table is not None:
            labels = np.array(
                [(self.truth_table >> p) & 1 for p in range(len(pats))],
                dtype=np.int64,
            )
        else:
            labels = np.array(
                [int(self.fn(tuple(row))) for row in pats.tolist()],
                dtype=np.int64,
            )
        if self.dont_care_mask is not None:
            keep = ~np.asarray(self.dont_care_mask, dtype=bool)
            pats, labels = pats[keep], labels[keep]
        return pats, labels


@dataclass
class RealizeResult:
    """can_realizeの結果。"""

    status: str  # 'sat' | 'unsat' | 'unknown'
    truth_tables: list[list[int]] | None = None  # 層ごと(width,)のTT整数
    wiring: list[np.ndarray] | None = None       # 層ごと(2, width)のindices
    stats: dict = field(default_factory=dict)


def _fixed_wiring(topology: LutTopology, seed: int) -> list[np.ndarray]:
    """torchのget_random_connectionsと同一の固定ランダム配線を生成する。"""
    import torch

    from ..core.connections import get_random_connections

    wiring = []
    d = topology.n_in
    torch.manual_seed(seed)
    for w in topology.layer_widths:
        wiring.append(
            get_random_connections(d, w, topology.lut_arity).numpy().copy()
        )
        d = w
    return wiring


class _Cnf:
    """変数プール+節バッファの薄いラッパ。

    注意: pysatのIDPool.occupyは pool.top を更新しないため、
    CardEncのtop_idにpool.topをそのまま渡すと**totalizer補助変数が
    衝突して過剰制約(偽UNSAT)になる**。self._topで自前管理する。
    """

    def __init__(self) -> None:
        self.pool = IDPool()
        self.clauses: list[list[int]] = []
        self._top = 0

    @property
    def top(self) -> int:
        return max(self._top, self.pool.top)

    def var(self, *key) -> int:
        v = self.pool.id(key)
        if v > self._top:
            self._top = v
        return v

    def add(self, clause: list[int]) -> None:
        self.clauses.append(clause)

    def card_atleast(self, lits: list[int], bound: int) -> None:
        if bound <= 0:
            return
        if bound > len(lits):
            self.add([])  # 充足不能
            return
        enc = CardEnc.atleast(lits=lits, bound=bound,
                              top_id=self.top, encoding=EncType.totalizer)
        self._absorb(enc)

    def card_atmost(self, lits: list[int], bound: int) -> None:
        if bound >= len(lits):
            return
        if bound < 0:
            self.add([])
            return
        enc = CardEnc.atmost(lits=lits, bound=bound,
                             top_id=self.top, encoding=EncType.totalizer)
        self._absorb(enc)

    def _absorb(self, enc) -> None:
        if enc.nv > self._top:
            self.pool.occupy(self._top + 1, enc.nv)
            self._top = enc.nv
        self.clauses.extend(enc.clauses)


def _guard(lit: int, bit: int) -> int:
    """条件「変数=bit」が不成立のとき真になるガードリテラル。"""
    return lit if bit == 0 else -lit


def _encode(cnf: _Cnf, topology: LutTopology, patterns: np.ndarray,
            labels: np.ndarray, wiring: list[np.ndarray] | None) -> None:
    """CNF本体を構築する。wiring=Noneなら自由配線モード。"""
    free = wiring is None
    arity = topology.lut_arity

    # t[l,o,k]: 層l(1始まり)ニューロンoの真理値表ビットk(butterfly順)
    def t(layer: int, o: int, k: int) -> int:
        return cnf.var('t', layer, o, k)

    # 自由配線: s[l,o,j,i] ピンjが前層iを選ぶone-hot
    if free:
        d = topology.n_in
        for li, w in enumerate(topology.layer_widths, start=1):
            for o in range(w):
                for j in range(arity):
                    svars = [cnf.var('s', li, o, j, i) for i in range(d)]
                    cnf.add(svars)  # at-least-one
                    for a, b in itertools.combinations(svars, 2):
                        cnf.add([-a, -b])  # pairwise at-most-one
                # ピン対称性破り: 隣接ピンjのindex ≤ ピンj+1のindex
                # (任意のピン置換はTTビットの対応置換で吸収できるため健全)
                for j in range(arity - 1):
                    for i in range(d):
                        for i2 in range(i):
                            cnf.add([-cnf.var('s', li, o, j, i),
                                     -cnf.var('s', li, o, j + 1, i2)])
            d = w
        _add_neuron_symmetry_breaking(cnf, topology)

    # パターンごとの信号伝搬
    for p, (row, label) in enumerate(
            zip(patterns.tolist(), labels.tolist(), strict=True)):
        # prev_lits[i]: 前層出力iのリテラル。層0(入力)は定数なのでNone扱い
        prev_lits: list[int] | None = None  # None=入力層(定数)
        for li, w in enumerate(topology.layer_widths, start=1):
            cur: list[int] = []
            for o in range(w):
                if not free:
                    srcs = [int(wiring[li - 1][j][o]) for j in range(arity)]
                    if prev_lits is None:
                        # 入力は定数 → 出力リテラルはTTビットそのもの(alias)
                        k = sum(row[s] << j for j, s in enumerate(srcs))
                        cur.append(t(li, o, k))
                        continue
                    pin_lits = [prev_lits[s] for s in srcs]
                    v = cnf.var('v', p, li, o)
                    for alphas in itertools.product((0, 1), repeat=arity):
                        k = sum(a << j for j, a in enumerate(alphas))
                        g = [_guard(l, a) for l, a in zip(pin_lits, alphas, strict=True)]
                        cnf.add(g + [-v, t(li, o, k)])
                        cnf.add(g + [v, -t(li, o, k)])
                    cur.append(v)
                else:
                    d = topology.n_in if prev_lits is None \
                        else len(prev_lits)
                    pins = []
                    for j in range(arity):
                        pin = cnf.var('pin', p, li, o, j)
                        for i in range(d):
                            s = cnf.var('s', li, o, j, i)
                            if prev_lits is None:
                                # 入力定数: s → pin=x_p[i]
                                cnf.add([-s, pin] if row[i] == 1
                                        else [-s, -pin])
                            else:
                                cnf.add([-s, -pin, prev_lits[i]])
                                cnf.add([-s, pin, -prev_lits[i]])
                        pins.append(pin)
                    v = cnf.var('v', p, li, o)
                    for alphas in itertools.product((0, 1), repeat=arity):
                        k = sum(a << j for j, a in enumerate(alphas))
                        g = [_guard(l, a) for l, a in zip(pins, alphas, strict=True)]
                        cnf.add(g + [-v, t(li, o, k)])
                        cnf.add(g + [v, -t(li, o, k)])
                    cur.append(v)
            prev_lits = cur

        # 読み出し制約
        out = prev_lits
        if topology.readout == 'bit':
            cnf.add([out[0]] if label == 1 else [-out[0]])
        else:
            # k-class GroupSum: y=c ⟺ 全j≠cで sum_c {≥|>} sum_j
            # (torch argmaxの同点=若番優先: j<c なら strict >、j>c なら ≥)。
            # sum_c ≥ sum_j + δ ⟺ count(group_c ∪ ¬group_j) ≥ |group_j| + δ
            k = topology.groupsum_k
            gsz = len(out) // k
            groups = [out[i * gsz:(i + 1) * gsz] for i in range(k)]
            c = int(label)
            assert 0 <= c < k, (c, k)
            for j in range(k):
                if j == c:
                    continue
                lits = list(groups[c]) + [-v for v in groups[j]]
                delta = 1 if j < c else 0
                cnf.card_atleast(lits, gsz + delta)


def _lex_leq(cnf: _Cnf, a_vec: list[int], b_vec: list[int], tag) -> None:
    """A ≤lex B の標準符号化(プレフィックス一致変数e_i付き)。"""
    e_prev = None
    for i, (a, b) in enumerate(zip(a_vec, b_vec, strict=True)):
        if e_prev is None:
            cnf.add([-a, b])  # a_0 ≤ b_0
        else:
            cnf.add([-e_prev, -a, b])
        e = cnf.var('lex_e', tag, i)
        # e ← e_prev ∧ (a ↔ b)  (プレフィックス一致ならeを真に強制)
        if e_prev is None:
            cnf.add([-a, -b, e])
            cnf.add([a, b, e])
        else:
            cnf.add([-e_prev, -a, -b, e])
            cnf.add([-e_prev, a, b, e])
        e_prev = e


def _add_neuron_symmetry_breaking(cnf: _Cnf, topology: LutTopology) -> None:
    """自由配線モードのニューロン置換対称性破り(層内lex順序)。

    自由配線では同一層内のニューロン置換(+下流selectionの対応置換)が
    解を保存するため、隣接ニューロンの変数ベクトル(配線one-hot+TT)に
    lex順序を課してよい。最終層はGroupSum読み出しの場合グループを跨ぐ
    置換が和を変えるため、**グループ内のみ**順序付けする。
    """
    n_layers = len(topology.layer_widths)
    arity = topology.lut_arity
    n_entries = 1 << arity
    d = topology.n_in
    for li, w in enumerate(topology.layer_widths, start=1):
        def vec(o: int, li: int = li, d: int = d) -> list[int]:
            v = []
            for j in range(arity):
                v.extend(cnf.var('s', li, o, j, i) for i in range(d))
            v.extend(cnf.var('t', li, o, k) for k in range(n_entries))
            return v

        is_last = li == n_layers
        if is_last and topology.readout == 'groupsum':
            k = topology.groupsum_k
            gsz = w // k
            groups = [range(i * gsz, (i + 1) * gsz) for i in range(k)]
        else:
            groups = [range(w)]
        for grp in groups:
            grp = list(grp)
            for o1, o2 in zip(grp, grp[1:], strict=False):
                _lex_leq(cnf, vec(o1), vec(o2), ('nsym', li, o1))
        d = w


def _extract_witness(model_vals: set[int], cnf: _Cnf, topology: LutTopology,
                     free: bool) -> tuple[list[list[int]], list[np.ndarray]]:
    """SATモデルからTTと配線を取り出す。"""
    tts: list[list[int]] = []
    wiring: list[np.ndarray] = []
    arity = topology.lut_arity
    n_entries = 1 << arity
    d = topology.n_in
    for li, w in enumerate(topology.layer_widths, start=1):
        layer_tts = []
        for o in range(w):
            tt = 0
            for k in range(n_entries):
                if cnf.var('t', li, o, k) in model_vals:
                    tt |= (1 << k)
            layer_tts.append(tt)
        tts.append(layer_tts)
        if free:
            idx = np.zeros((arity, w), dtype=np.int64)
            for o in range(w):
                for j in range(arity):
                    chosen = [i for i in range(d)
                              if cnf.var('s', li, o, j, i) in model_vals]
                    assert len(chosen) == 1, (li, o, j, chosen)
                    idx[j][o] = chosen[0]
            wiring.append(idx)
        d = w
    return tts, wiring


def witness_to_model(result: RealizeResult, topology: LutTopology):
    """witnessからLutLayer網(nn.Sequential)を構築する(hard初期化)。"""
    import torch.nn as nn

    from ..group_sum import GroupSum
    from ..ir.weights import restore_lut_layer
    from ..layers import LutLayer

    assert result.status == 'sat' and result.truth_tables is not None
    layers: list = []
    arity = topology.lut_arity
    # N=2は16ゲートsoftmax(従来互換)。N>2のhard真理値表はsoftmax(合計1)では
    # 表現できないためindependent(エントリ独立sigmoid)で復元する。
    param = 'softmax' if arity == 2 else 'independent'
    d = topology.n_in
    for li, w in enumerate(topology.layer_widths):
        layer = LutLayer(d, w, n_inputs=arity, parameterization=param)
        restore_lut_layer(layer, result.wiring[li], result.truth_tables[li])
        layers.append(layer)
        d = w
    if topology.readout == 'groupsum':
        layers.append(GroupSum(k=topology.groupsum_k,
                               tau=max(1.0, (d // topology.groupsum_k) ** 0.5)))
    return nn.Sequential(*layers)


def witness_to_ir(result: RealizeResult, topology: LutTopology):
    """witnessをIRへ変換する(export_ir経由)。"""
    from ..export import export_ir

    model = witness_to_model(result, topology)
    return export_ir(model, input_width=topology.n_in, name='sat_witness')


def _verify_witness(result: RealizeResult, topology: LutTopology,
                    patterns: np.ndarray, labels: np.ndarray) -> bool:
    """witnessをIRシミュレータで標的と照合する(エンコーダ非依存の検証)。"""
    from ..ir import simulate

    ir = witness_to_ir(result, topology)
    scores = simulate(ir, patterns)
    if topology.readout == 'bit':
        pred = scores[:, 0]
    else:
        pred = scores.argmax(-1)
    return bool((pred == labels).all())


def can_realize(
    topology: LutTopology,
    target: TargetSpec,
    *,
    wiring: str | list[np.ndarray] = 'fixed_random',
    seed: int = 0,
    solver: str = 'cadical195',
    verify: bool = True,
    conf_budget: int | None = None,
) -> RealizeResult:
    """トポロジーが標的関数を実現できるかをSATで判定する。

    Args:
        topology: 対象トポロジー
        target: 標的関数
        wiring: 'fixed_random'(seedから生成) | 'free'(配線もSAT変数) |
            明示的なindicesリスト(層ごと(2, width))
        seed: fixed_random時の配線seed
        solver: pysatソルバ名(失敗時はSOLVER_FALLBACKSへ)
        verify: SAT時にwitnessをIRシミュレートで標的と照合する
        conf_budget: conflict予算(超過でstatus='unknown'。UNSAT証明が
            重い大規模自由配線セルの打ち切り用。None=無制限)

    Returns:
        RealizeResult(status, truth_tables, wiring, stats)
    """
    _require_pysat()
    patterns, labels = target.patterns_labels(topology.n_in)

    free = wiring == 'free'
    if free:
        fixed = None
    elif wiring == 'fixed_random':
        fixed = _fixed_wiring(topology, seed)
    else:
        fixed = [np.asarray(w, dtype=np.int64) for w in wiring]
        for li, w in enumerate(fixed):
            assert w.shape == (topology.lut_arity,
                               topology.layer_widths[li]), w.shape

    cnf = _Cnf()
    t0 = time.time()
    _encode(cnf, topology, patterns, labels, fixed)
    t_encode = time.time() - t0

    names = [solver] + [s for s in SOLVER_FALLBACKS if s != solver]
    sat = None
    used = None
    for name in names:
        try:
            with Solver(name=name, bootstrap_with=cnf.clauses) as s:
                t1 = time.time()
                if conf_budget is not None:
                    s.conf_budget(conf_budget)
                    sat = s.solve_limited()
                else:
                    sat = s.solve()
                t_solve = time.time() - t1
                model = s.get_model() if sat else None
            used = name
            break
        except Exception:
            continue
    if used is None:
        raise RuntimeError(f'利用可能なSATソルバが無い: {names}')

    stats = {
        'n_vars': cnf.top,
        'n_clauses': len(cnf.clauses),
        'n_patterns': len(patterns),
        'encode_time_s': t_encode,
        'solve_time_s': t_solve,
        'solver': used,
    }

    if sat is None:  # conflict予算超過
        return RealizeResult(status='unknown', stats=stats)
    if not sat:
        return RealizeResult(status='unsat', stats=stats)

    model_vals = {v for v in model if v > 0}
    tts, free_wiring = _extract_witness(model_vals, cnf, topology, free)
    result = RealizeResult(
        status='sat', truth_tables=tts,
        wiring=free_wiring if free else fixed, stats=stats,
    )
    if verify:
        ok = _verify_witness(result, topology, patterns, labels)
        stats['verified'] = ok
        if not ok:
            raise AssertionError(
                'SAT witnessがIRシミュレートで標的と不一致(エンコードバグ)'
            )
    return result


def find_sat_wiring(
    topology: LutTopology,
    target: TargetSpec,
    *,
    max_tries: int = 32,
    seed0: int = 0,
    conf_budget: int | None = None,
) -> dict:
    """UNSATになる固定ランダム配線を棄却し、SAT配線を探す(配線ガード)。

    「UNSATになる配線をしない」ための構成時スクリーニング。実タスク規模では
    targetを部分関数(訓練サブサンプル)にして必要条件チェックとして使う。

    Returns:
        {'wiring': indicesリスト|None, 'seed': 当たりseed|None,
         'tried': 試行数, 'statuses': 各seedのstatus}
    """
    statuses = []
    for i in range(max_tries):
        seed = seed0 + i
        r = can_realize(topology, target, wiring='fixed_random', seed=seed,
                        conf_budget=conf_budget, verify=False)
        statuses.append(r.status)
        if r.status == 'sat':
            return {'wiring': _fixed_wiring(topology, seed), 'seed': seed,
                    'tried': i + 1, 'statuses': statuses}
    return {'wiring': None, 'seed': None, 'tried': max_tries,
            'statuses': statuses}


def min_width_search(
    n_in: int,
    depth: int,
    target: TargetSpec,
    *,
    widths: tuple[int, ...] = (2, 4, 8, 16, 32),
    wiring: str = 'free',
    seed: int = 0,
    readout: str = 'groupsum',
    lut_arity: int = 2,
    conf_budget: int | None = None,
) -> dict:
    """幅候補を昇順に試し、最初にSATになる幅を返す。"""
    results = {}
    min_w = None
    for w in widths:
        topo = LutTopology(n_in=n_in, layer_widths=[w] * depth,
                           readout=readout, lut_arity=lut_arity)
        r = can_realize(topo, target, wiring=wiring, seed=seed,
                        conf_budget=conf_budget)
        results[w] = r.status
        if r.status == 'sat' and min_w is None:
            min_w = w
            break
    return {'min_width': min_w, 'tried': results}


def wiring_tax(
    n_in: int,
    depth: int,
    width: int,
    target: TargetSpec,
    *,
    n_wiring_seeds: int = 16,
    readout: str = 'groupsum',
    lut_arity: int = 2,
) -> dict:
    """固定ランダム配線のSAT率と自由配線の判定を比較する(配線税)。"""
    topo = LutTopology(n_in=n_in, layer_widths=[width] * depth,
                       readout=readout, lut_arity=lut_arity)
    free = can_realize(topo, target, wiring='free')
    fixed_sat = []
    for s in range(n_wiring_seeds):
        r = can_realize(topo, target, wiring='fixed_random', seed=s)
        fixed_sat.append(r.status == 'sat')
    return {
        'free_sat': free.status == 'sat',
        'fixed_sat_frac': float(np.mean(fixed_sat)),
        'fixed_sat': fixed_sat,
    }
