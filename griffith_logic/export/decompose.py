"""n入力ゲートの conv2d ノードを、等価な2入力ゲートの conv2d ノードへ書き換える。

なぜ要るか
----------
`LutConv2d` は n_inputs∈{2,4,6} を学習できるが、配信側の速い経路
(packed / bitplane / wasm SIMD / WebGPU)と後段簡約(`export/optimize.py`)は
いずれも **n_inputs=2 固定**を前提にしている。n を上げて精度が出ても、
そのままでは遅い汎用経路にしか載らず簡約も効かない。

ここで Shannon 展開して2入力へ落としておけば、ノードは conv2d のままなので
速い経路にも簡約にも戻れる。実測(ランダム表=論理合成の最悪ケース):

    n=4: 1ゲート → 二分深さ5・31ゲート → 簡約後 平均 10.1
    n=6: 1ゲート → 二分深さ9・511ゲート → 簡約後 平均 45.4

分解の考え方
------------
    mux(s, f1, f0) = OR( AND(s, f1), ANDNOT(s, f0) )
s を2回参照するが、木の葉は同じタップを重複して指してよいので木のまま書ける。
conv2d は**完全 n分木**を要求するので、タップ側はパススルーゲート f(a,b)=a で
深さを揃える。1ゲートを置き換える二分部分木の深さは g(2)=1, g(n)=g(n-1)+2。

制約
----
n分木(深さ d)全体は深さ **d×g(n)** の完全二分木になる。葉数は 2^(d×g(n)) で、
パディング分だけ指数的に膨らむ(簡約で畳まれるが、中間表現は実体化される)。
そのため d×g(n) に上限を設けている。n=4,d=2 → 10(葉1,024)は実用範囲、
n=6,d=3 → 27(葉1.3億)は不可。
"""

from __future__ import annotations

import functools

import numpy as np

from griffith_logic.ir.schema import IRGraph, IRNode
from griffith_logic.ir.serialize import tt_from_hex, tt_to_hex

__all__ = [
    'binary_depth_for',
    'decompose_gate',
    'decompose_conv_params',
    'decompose_graph',
]

# bit index = (b<<1)|a、値 = f(a,b)
TT_PASS_A = 0b1010        # f = a （パススルー）
TT_NOT_A = 0b0101         # f = ~a
TT_AND = 0b1000           # f = a & b
TT_OR = 0b1110            # f = a | b
TT_NOTA_AND_B = 0b0100    # f = (~a) & b
TT_NOTA_OR_B = 0b1101     # f = (~a) | b
TT_XOR = 0b0110           # f = a ^ b

# d×g(n) の上限。10 で葉 1,024/oc。これを超えると中間表現が実用外になる。
MAX_BINARY_DEPTH = 12


def binary_depth_for(n: int) -> int:
    """n入力ゲート1個を置き換える完全二分部分木の深さ g(n)。"""
    if n < 1:
        raise ValueError(f'n は1以上: {n}')
    return 1 if n <= 2 else binary_depth_for(n - 1) + 2


def _shift(base_hid: int, rel_hid: int) -> int:
    """部分木の相対heap idを絶対heap idへ移す。

    相対idが深さkにあるとき(rel_hid ∈ [2^k, 2^(k+1)))、絶対idは
    base_hid を根とする部分木の同じ位置、つまり base*2^k + (rel-2^k)。
    """
    k = rel_hid.bit_length() - 1
    return base_hid * (1 << k) + (rel_hid - (1 << k))


def _cofactors(bits: int, m: int, i: int) -> tuple[int, int]:
    """m変数関数 bits を圧縮位置 i で Shannon 余因子に分ける。

    戻り値は (f0, f1)、どちらも変数 i を除いた m-1 変数の関数。
    """
    f0 = f1 = 0
    pos = 0
    for addr in range(1 << m):
        if (addr >> i) & 1:
            continue
        f0 |= ((bits >> addr) & 1) << pos
        f1 |= ((bits >> (addr | (1 << i))) & 1) << pos
        pos += 1
    return f0, f1


@functools.cache
def _plan_gate(bits: int, m: int) -> tuple[int, int, tuple]:
    """m変数関数を2入力ゲート木で作る最良手順を返す。

    素朴な Shannon 展開（常に最上位変数で分割し、常に mux=3ゲートを出す）は
    6変数関数で平均 32.8 ゲートになった。実際には次の2点で大きく減る:

    1. **分割変数の選択**: どの変数で割るかでコストが変わる。全変数を試して
       最小を採る（(bits, m) でメモ化するので指数爆発しない）。
    2. **縮退した mux の特殊形**: mux(s, f1, f0) は一般には3ゲート要るが、
       余因子が退化していれば1ゲートで済む。
           f0 == f1  → 変数 s に依存しない（ゲート0個、s を捨てる）
           f0 == 0   → s & f1
           f1 == 0   → (~s) & f0
           f0 == all → (~s) | f1
           f1 == all → s | f0
           f1 == ~f0 → s ^ f0

    Returns:
        (cost, depth, plan)。cost は A1 簡約後に残る実ゲート数の見積り
        （パススルー/定数/NOT は簡約で 0 ゲートに畳まれるので数えない）。
        plan 内の位置は 0..m-1 の圧縮位置で、木を組む側が実タップへ写す。
    """
    full = (1 << (1 << m)) - 1
    bits &= full
    if bits == 0:
        return 0, 1, ('const', 0)
    if bits == full:
        return 0, 1, ('const', 1)
    if m == 0:                      # 定数は上で処理済み
        return 0, 1, ('const', bits & 1)
    if m == 1:
        # 0b10 = x, 0b01 = ~x（0b00/0b11 は定数として上で返っている）
        return (0, 0, ('lit', 0, False)) if bits == 0b10 \
            else (0, 1, ('lit', 0, True))
    if m == 2:
        # 2変数関数はそのまま1ゲート。アドレス規約が tt 規約と一致している
        # （addr = x0 | x1<<1、tt の index = (b<<1)|a、a=x0, b=x1）。
        return 1, 1, ('g2', bits)

    best: tuple[int, int, tuple] | None = None
    for i in range(m):
        f0, f1 = _cofactors(bits, m, i)
        sub_full = (1 << (1 << (m - 1))) - 1

        if f0 == f1:
            # 変数 i に依存しない。捨てるのが常に最良なので即決。
            c, d, p = _plan_gate(f0, m - 1)
            return c, d, ('skip', i, p)

        if f0 == 0:
            c, d, p = _plan_gate(f1, m - 1)
            cand = (c + 1, d + 1, ('and', i, False, p))
        elif f1 == 0:
            c, d, p = _plan_gate(f0, m - 1)
            cand = (c + 1, d + 1, ('and', i, True, p))
        elif f0 == sub_full:
            c, d, p = _plan_gate(f1, m - 1)
            cand = (c + 1, d + 1, ('or', i, True, p))
        elif f1 == sub_full:
            c, d, p = _plan_gate(f0, m - 1)
            cand = (c + 1, d + 1, ('or', i, False, p))
        elif f1 == (~f0 & sub_full):
            c, d, p = _plan_gate(f0, m - 1)
            cand = (c + 1, d + 1, ('xor', i, p))
        else:
            c1, d1, p1 = _plan_gate(f1, m - 1)
            c0, d0, p0 = _plan_gate(f0, m - 1)
            cand = (c1 + c0 + 3, max(d1, d0) + 2, ('mux', i, p1, p0))

        if best is None or cand[:2] < best[:2]:
            best = cand

    assert best is not None
    return best


def decompose_gate(table_bits: int, n: int) -> tuple[dict, dict]:
    """n入力の真理値表を、深さ g(n) の完全二分木へ Shannon 展開する。

    Args:
        table_bits: 2^n ビットの真理値表。bit j がアドレス j の出力。
            アドレスは Σ x_i << i（IRの conv2d/lut_layer 規約)。
        n: 入力数。

    Returns:
        (gtt, leafslot)
        gtt:      相対heap id(根=1) → 2入力真理値表(0..15)
        leafslot: 葉の相対heap id → その位置が参照する入力番号(0..n-1)
    """
    d = binary_depth_for(n)
    gtt: dict[int, int] = {}
    leafslot: dict[int, int] = {}

    def pad(hid: int, depth: int, slot: int) -> None:
        """パススルーゲートで深さを揃えつつ、葉に入力slotを置く。"""
        if depth == 0:
            leafslot[hid] = slot
            return
        gtt[hid] = TT_PASS_A
        pad(2 * hid, depth - 1, slot)
        pad(2 * hid + 1, depth - 1, slot)

    def emit(plan: tuple, hid: int, depth: int, varlist: list[int]) -> None:
        kind = plan[0]
        if kind == 'const':
            gtt[hid] = 0b1111 if plan[1] else 0b0000
            pad(2 * hid, depth - 1, varlist[0] if varlist else 0)
            pad(2 * hid + 1, depth - 1, varlist[0] if varlist else 0)
            return
        if kind == 'lit':
            tap = varlist[plan[1]]
            if not plan[2]:
                pad(hid, depth, tap)
            else:
                gtt[hid] = TT_NOT_A
                pad(2 * hid, depth - 1, tap)
                pad(2 * hid + 1, depth - 1, tap)
            return
        if kind == 'g2':
            gtt[hid] = plan[1]
            pad(2 * hid, depth - 1, varlist[0])
            pad(2 * hid + 1, depth - 1, varlist[1])
            return
        if kind == 'skip':
            i = plan[1]
            emit(plan[2], hid, depth, varlist[:i] + varlist[i + 1:])
            return

        i = plan[1]
        tap = varlist[i]
        rest = varlist[:i] + varlist[i + 1:]
        if kind in ('and', 'or', 'xor'):
            if kind == 'and':
                gtt[hid] = TT_NOTA_AND_B if plan[2] else TT_AND
                child = plan[3]
            elif kind == 'or':
                gtt[hid] = TT_NOTA_OR_B if plan[2] else TT_OR
                child = plan[3]
            else:
                gtt[hid] = TT_XOR
                child = plan[2]
            pad(2 * hid, depth - 1, tap)
            emit(child, 2 * hid + 1, depth - 1, rest)
            return

        # mux: OR( s & f1, (~s) & f0 )
        gtt[hid] = TT_OR
        gtt[2 * hid] = TT_AND
        gtt[2 * hid + 1] = TT_NOTA_AND_B
        pad(4 * hid, depth - 2, tap)
        emit(plan[2], 4 * hid + 1, depth - 2, rest)
        pad(4 * hid + 2, depth - 2, tap)
        emit(plan[3], 4 * hid + 3, depth - 2, rest)

    _, need, plan = _plan_gate(int(table_bits), n)
    if need > d:  # 全 mux の最悪ケースがちょうど g(n)。超えたら実装バグ
        raise AssertionError(f'分解が深すぎる: {need} > {d} (n={n})')
    emit(plan, 1, d, list(range(n)))
    return gtt, leafslot


def decompose_conv_params(params: dict) -> dict:
    """conv2d ノードの params を n_inputs=2 の等価な params へ書き換える。

    n_inputs=2 のときは（コピーを返すだけで）何もしない。
    """
    n = int(params.get('n_inputs', 2))
    if n == 2:
        return dict(params)
    if n not in (4, 6):
        raise ValueError(f'n_inputs は 2/4/6 のみ: {n}')

    d = int(params['tree_depth'])
    gn = binary_depth_for(n)
    new_depth = d * gn
    if new_depth > MAX_BINARY_DEPTH:
        raise ValueError(
            f'分解後の二分木が深すぎる: tree_depth={d} × g({n})={gn} '
            f'= {new_depth} > {MAX_BINARY_DEPTH}（葉 2^{new_depth} /出力ch）。'
            'パディングで指数的に膨らむため、浅い木で学習し直すこと。'
        )

    oc = int(params['out_channels'])
    n_leaves = n ** d
    n_internal = (n_leaves - 1) // (n - 1)
    idxf = np.asarray(params['indices_flat'], dtype=np.int64)
    tts_raw = params['truth_tables']
    flat = tt_from_hex(list(tts_raw)) if tts_raw and isinstance(tts_raw[0], str) \
        else [int(v) for v in tts_raw]
    tts = np.array(flat, dtype=object).reshape(oc, n_internal)

    # レベルLの先頭ゲート番号（level0 = 葉の直上）
    level_offset = []
    acc = 0
    for lv in range(d):
        level_offset.append(acc)
        acc += n ** (d - 1 - lv)

    # 各ゲートの分解結果は表にしか依存しないので使い回す
    cache: dict[int, tuple[dict, dict]] = {}

    new_tts: list[int] = []
    new_idx = np.empty((oc, 1 << new_depth), dtype=np.int64)

    def emit(c: int, level: int, g: int, base_hid: int,
             gtt_abs: dict, leaf_abs: dict) -> None:
        """n分木のノード(level, g)を、base_hid を根とする二分部分木へ展開する。"""
        table = int(tts[c, level_offset[level] + g])
        got = cache.get(table)
        if got is None:
            got = decompose_gate(table, n)
            cache[table] = got
        rel_gtt, rel_leaf = got
        for rel, t in rel_gtt.items():
            gtt_abs[_shift(base_hid, rel)] = t
        for rel, slot in rel_leaf.items():
            child_hid = _shift(base_hid, rel)
            child = g * n + slot
            if level == 0:
                leaf_abs[child_hid] = int(idxf[c, child])
            else:
                emit(c, level - 1, child, child_hid, gtt_abs, leaf_abs)

    for c in range(oc):
        gtt_abs: dict[int, int] = {}
        leaf_abs: dict[int, int] = {}
        emit(c, d - 1, 0, 1, gtt_abs, leaf_abs)

        # レベル昇順（level0 = 葉の直上）に並べ替え
        for lv in range(new_depth):
            lo, hi = 1 << (new_depth - 1 - lv), 1 << (new_depth - lv)
            new_tts.extend(gtt_abs[hid] for hid in range(lo, hi))
        base = 1 << new_depth
        for i in range(1 << new_depth):
            new_idx[c, i] = leaf_abs[base + i]

    out = dict(params)
    out['n_inputs'] = 2
    out['tree_depth'] = new_depth
    out['indices_flat'] = new_idx
    out['truth_tables'] = tt_to_hex(new_tts) \
        if tts_raw and isinstance(tts_raw[0], str) else new_tts
    return out


def decompose_graph(graph: IRGraph) -> IRGraph:
    """グラフ中の n_inputs>2 な conv2d を、すべて2入力へ書き換えた新グラフを返す。"""
    nodes = []
    for node in graph.nodes:
        if node.type == 'conv2d' and int(node.params.get('n_inputs', 2)) != 2:
            nodes.append(IRNode(
                id=node.id,
                type=node.type,
                params=decompose_conv_params(node.params),
                train_info=node.train_info,
            ))
        else:
            nodes.append(node)
    return IRGraph(
        metadata=dict(graph.metadata),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        nodes=nodes,
        edges=list(graph.edges),
        format=graph.format,
        version=graph.version,
    )
