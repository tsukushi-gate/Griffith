"""conv2d 重み共有木の lossless 命令列化(ゲート削減)— Pythonミラー。

ブラウザ側ベンチ実装(TypeScript版 convProgram.ts)の縮約パス
(NOT吸収→定数畳み込み→退化tt→同一参照→可換正規化→CSE→dead除去)を
純Python(numpy可)で忠実に再実装したもの。**挙動の真実源はTS側**であり、
このモジュールはそのミラー(公式ゲート数集計 + クロス検証用)。

用途:
    1. reduce_conv_tree / graph_conv_stats — IRのconv2dノードの縮約後
       ゲート数をPython側の「公式」統計として算出する(HDL非依存の見積り)。
    2. verify_conv_program — 縮約前後の関数等価性を全数照合(numpyビット
       並列)でセルフチェックする(TSのverifyConvProgramのミラー)。
    3. TS版のゲート数集計スクリプト(--json 出力)とのクロス検証
       (tests/l1_math/test_optimize.py)。

規約(convProgram.tsと同一。ズレやすい箇所に注記):
    - tt のビット規約: bit index = (b<<1)|a、値 = f(a,b)。
      IR側(ir/schema.py)の「bit k = パターンk、bit j = 入力スロットjの値」
      規約とも一致するため、IRのtruth_tables(hex)はそのまま使える
      (ビット並べ替え不要)
    - n_inputs=2 固定(convProgram.tsの前提と同じ)
    - Ref エンコード: (idx << 2) | kind、kind∈{CONST=0, TAP=1, INSTR=2}
    - 可換正規化は「入替時にrefの整数値(idx<<2|kind)でソート」であり、
      taps/instrsのdedup後の番号に依存する。dead除去後の詰め直しで
      番号が変わるため、TS同様に**dead除去後にもう一度**可換正規化を
      やり直す(そうしないとdead除去前後でCSEの正規形が変わり得る)
"""

from dataclasses import asdict, dataclass, fields

import numpy as np

from ..ir.schema import IRGraph
from ..ir.serialize import tt_from_hex

# ============================================================
# 参照(Ref)エンコード: (idx << 2) | kind
# ============================================================

REF_KIND_CONST = 0
REF_KIND_TAP = 1
REF_KIND_INSTR = 2


def make_ref(kind: int, idx: int) -> int:
    return (idx << 2) | kind


def ref_kind(r: int) -> int:
    return r & 3


def ref_idx(r: int) -> int:
    return r >> 2


REF_ZERO = make_ref(REF_KIND_CONST, 0)
REF_ONE = make_ref(REF_KIND_CONST, 1)

# ============================================================
# 2入力 truth table (tt) の代数(convProgram.ts と同一の機械生成表)
# ============================================================


def tt_eval(tt: int, a: int, b: int) -> int:
    """tt のビット規約: bit index = (b<<1)|a、値 = f(a,b)。"""
    return (tt >> (((b & 1) << 1) | (a & 1))) & 1


def _tt_from(f) -> int:
    tt = 0
    for b in range(2):
        for a in range(2):
            if f(a, b):
                tt |= 1 << ((b << 1) | a)
    return tt & 0xf


def _build_table(mapper) -> list:
    return [mapper(tt) for tt in range(16)]


# g(a,b) = f(b,a) の tt(オペランド入替)。
TT_SWAP: list = _build_table(lambda tt: _tt_from(lambda a, b: tt_eval(tt, b, a)))
# g(a,b) = f(~a,b) の tt(a入力の反転吸収)。
TT_NOT_A: list = _build_table(lambda tt: _tt_from(lambda a, b: tt_eval(tt, a ^ 1, b)))
# g(a,b) = f(a,~b) の tt(b入力の反転吸収)。
TT_NOT_B: list = _build_table(lambda tt: _tt_from(lambda a, b: tt_eval(tt, a, b ^ 1)))


def tt_swap(tt: int) -> int:
    return TT_SWAP[tt & 0xf]


def tt_is_commutative(tt: int) -> bool:
    return TT_SWAP[tt & 0xf] == (tt & 0xf)


# --- 単項簡約コード ---------------------------------------------------------
RED_CONST0 = 0
RED_CONST1 = 1
RED_ID = 2
RED_NOT = 3
RED_BINARY = 4


def _red_of(g0: int, g1: int) -> int:
    """(g0,g1) = (f(0),f(1)) から単項コードへ。"""
    if g0 == 0 and g1 == 0:
        return RED_CONST0
    if g0 == 1 and g1 == 1:
        return RED_CONST1
    return RED_ID if g0 == 0 else RED_NOT


def _build_const_fold_b() -> list:
    t = [0] * 32
    for tt in range(16):
        for kb in range(2):
            t[tt * 2 + kb] = _red_of(tt_eval(tt, 0, kb), tt_eval(tt, 1, kb))
    return t


def _build_const_fold_a() -> list:
    t = [0] * 32
    for tt in range(16):
        for ka in range(2):
            t[tt * 2 + ka] = _red_of(tt_eval(tt, ka, 0), tt_eval(tt, ka, 1))
    return t


# 「b が定数 kb のとき、残る a についての簡約コード」表。添字 = tt*2 + kb。
CONST_FOLD_B: list = _build_const_fold_b()
# 「a が定数 ka のとき、残る b についての簡約コード」表。添字 = tt*2 + ka。
CONST_FOLD_A: list = _build_const_fold_a()


def _depends(tt: int) -> tuple:
    depends_a = (tt_eval(tt, 0, 0) != tt_eval(tt, 1, 0)
                 or tt_eval(tt, 0, 1) != tt_eval(tt, 1, 1))
    depends_b = (tt_eval(tt, 0, 0) != tt_eval(tt, 0, 1)
                 or tt_eval(tt, 1, 0) != tt_eval(tt, 1, 1))
    return depends_a, depends_b


def _tt_degen_entry(tt: int) -> int:
    depends_a, depends_b = _depends(tt)
    if not depends_a and not depends_b:
        return RED_CONST1 if tt_eval(tt, 0, 0) else RED_CONST0
    if depends_a and depends_b:
        return RED_BINARY
    # 依存する側の値がf(0)|f(*) — a/bどちらの片依存でも同一式
    return RED_NOT if tt_eval(tt, 0, 0) else RED_ID


def _tt_degen_on_b_entry(tt: int) -> int:
    depends_a, _ = _depends(tt)
    return 0 if depends_a else 1


# tt 自体が片方の入力に依存しない(退化)場合のコード。RED_CONST0/1のときは
# 入力に依存しない。TT_DEGEN_ON_B[tt]=1ならb側、0ならa側が対象入力。
TT_DEGEN: list = _build_table(_tt_degen_entry)
TT_DEGEN_ON_B: list = _build_table(_tt_degen_on_b_entry)

# 同一参照が両入力に来たとき f(a,a) の簡約コード(g0=bit0, g1=bit3)。
SAME_FOLD: list = _build_table(lambda tt: _red_of(tt & 1, (tt >> 3) & 1))

# NOT を表す単項命令の tt(¬a、b は a と同一参照を置く)。
TT_NOT = 0x5


# ============================================================
# 入出力の型
# ============================================================


@dataclass
class ConvTreeSpec:
    """命令列化の入力(IRのconv2dノードparamsからビットプレーン非依存の情報だけ抜いたもの)。"""

    out_channels: int
    tree_depth: int
    indices_flat: list  # [oc][leaf] のタップID(indices_flatのまま)
    truth_tables: list  # [OC*nInternal] の数値化truth table(0..15、oc-major・レベル昇順)


@dataclass
class ConvInstr:
    tt: int
    a: int
    b: int
    slot: int = -1


@dataclass
class ConvProgramStats:
    gates_before: int
    gates_after: int
    const_folded: int
    alias_folded: int
    not_folded: int
    cse_hits: int
    emitted: int
    instrs_created: int
    dead_removed: int
    not_instrs: int
    taps_before: int
    n_taps: int
    n_slots: int
    oc_const: int
    oc_tap: int
    oc_unique: int


@dataclass
class ConvProgram:
    taps: list
    instrs: list  # list[ConvInstr]
    n_slots: int
    oc_out: list
    write_start: list
    write_oc: list
    stats: ConvProgramStats


_STATS_FIELDS = [f.name for f in fields(ConvProgramStats)]


# ============================================================
# ConvTreeSpec の構築ヘルパー(IRノードparams → ConvTreeSpec)
# ============================================================


def spec_from_node_params(params: dict) -> ConvTreeSpec:
    """IR conv2dノードのparamsからConvTreeSpecを作る。

    truth_tablesはhex文字列(IR JSON規約)・整数列のどちらでも受け付ける。
    n_inputsは2固定を前提とする(convProgram.tsの前提と同じ)。
    """
    n_inputs = int(params.get('n_inputs', 2))
    if n_inputs != 2:
        raise ValueError(
            f'spec_from_node_params: n_inputs={n_inputs} は非対応'
            '(convProgram縮約はn_inputs=2固定)'
        )
    tts_raw = params['truth_tables']
    truth_tables = tt_from_hex(list(tts_raw)) if tts_raw and isinstance(tts_raw[0], str) \
        else [int(v) for v in tts_raw]
    indices_flat = [[int(v) for v in row] for row in params['indices_flat']]
    return ConvTreeSpec(
        out_channels=int(params['out_channels']),
        tree_depth=int(params['tree_depth']),
        indices_flat=indices_flat,
        truth_tables=truth_tables,
    )


# ============================================================
# 命令列ビルダ(convProgram.ts::buildConvProgram のミラー)
# ============================================================


def build_conv_program(spec: ConvTreeSpec) -> ConvProgram:
    """conv2dの重み共有木(全oc)をlosslessに命令列へ縮約する。

    元のバランス木の評価順(レベル0: 葉(2g,2g+1) → 上位レベル)は
    convProgram.ts::buildConvProgram と厳密に同一。
    """
    oc_count = spec.out_channels
    n_leaves = 1 << spec.tree_depth
    n_internal = n_leaves - 1  # n_inputs=2固定: (nLeaves-1)/(2-1)

    taps: list = []
    tap_id_of: dict = {}
    draft: list = []  # each: [tt, a, b] (mutable draft instr)
    cse: dict = {}
    oc_out: list = [0] * oc_count

    counters = {
        'const_folded': 0, 'alias_folded': 0, 'not_folded': 0,
        'cse_hits': 0, 'emitted': 0,
    }
    state = {'last_was_new': False}

    def raw_emit(tt: int, a: int, b: int) -> int:
        key = (tt, a, b)
        hit = cse.get(key)
        if hit is not None:
            state['last_was_new'] = False
            return make_ref(REF_KIND_INSTR, hit)
        idx = len(draft)
        draft.append([tt, a, b])
        cse[key] = idx
        state['last_was_new'] = True
        return make_ref(REF_KIND_INSTR, idx)

    def emit_not(x: int) -> int:
        if ref_kind(x) == REF_KIND_CONST:
            state['last_was_new'] = False
            return REF_ONE if ref_idx(x) == 0 else REF_ZERO
        if ref_kind(x) == REF_KIND_INSTR:
            d = draft[ref_idx(x)]
            if d[0] == TT_NOT and d[1] == d[2]:
                state['last_was_new'] = False
                return d[1]  # ¬¬y = y
        return raw_emit(TT_NOT, x, x)

    def apply_red(code: int, x: int) -> int:
        if code == RED_CONST0:
            return REF_ZERO
        if code == RED_CONST1:
            return REF_ONE
        if code == RED_ID:
            return x
        return emit_not(x)

    def tally(code: int) -> None:
        if code in (RED_CONST0, RED_CONST1):
            counters['const_folded'] += 1
        elif code == RED_ID:
            counters['alias_folded'] += 1
        else:
            counters['not_folded'] += 1

    def emit_gate(tt0: int, a0: int, b0: int) -> int:
        tt = tt0 & 0xf
        a = a0
        b = b0

        # (3) NOT吸収
        if ref_kind(a) == REF_KIND_INSTR:
            d = draft[ref_idx(a)]
            if d[0] == TT_NOT and d[1] == d[2]:
                a = d[1]
                tt = TT_NOT_A[tt]
        if ref_kind(b) == REF_KIND_INSTR:
            d = draft[ref_idx(b)]
            if d[0] == TT_NOT and d[1] == d[2]:
                b = d[1]
                tt = TT_NOT_B[tt]

        # (1) 定数畳み込み
        a_const = ref_kind(a) == REF_KIND_CONST
        b_const = ref_kind(b) == REF_KIND_CONST
        if a_const and b_const:
            counters['const_folded'] += 1
            return REF_ONE if tt_eval(tt, ref_idx(a), ref_idx(b)) else REF_ZERO
        if b_const:
            code = CONST_FOLD_B[tt * 2 + ref_idx(b)]
            r = apply_red(code, a)
            tally(code)
            return r
        if a_const:
            code = CONST_FOLD_A[tt * 2 + ref_idx(a)]
            r = apply_red(code, b)
            tally(code)
            return r

        # (2) tt自体の退化
        degen = TT_DEGEN[tt]
        if degen != RED_BINARY:
            target = b if TT_DEGEN_ON_B[tt] else a
            r = apply_red(degen, target)
            tally(degen)
            return r

        # 同一参照が両入力に来た場合
        if a == b:
            code = SAME_FOLD[tt]
            r = apply_red(code, a)
            tally(code)
            return r

        # (4) 可換正規化
        if b < a:
            a, b = b, a
            tt = TT_SWAP[tt]

        # (5) CSE
        r = raw_emit(tt, a, b)
        if state['last_was_new']:
            counters['emitted'] += 1
        else:
            counters['cse_hits'] += 1
        return r

    for oc in range(oc_count):
        row = spec.indices_flat[oc]
        cur = [0] * n_leaves
        for k in range(n_leaves):
            pp = row[k]
            tid = tap_id_of.get(pp)
            if tid is None:
                tid = len(taps)
                taps.append(pp)
                tap_id_of[pp] = tid
            cur[k] = make_ref(REF_KIND_TAP, tid)
        g_offset = 0
        while len(cur) > 1:
            g_out = len(cur) >> 1
            nxt = [0] * g_out
            for g in range(g_out):
                tt = spec.truth_tables[oc * n_internal + g_offset + g] & 0xf
                nxt[g] = emit_gate(tt, cur[g << 1], cur[(g << 1) + 1])
            g_offset += g_out
            cur = nxt
        oc_out[oc] = cur[0]

    instrs_created = len(draft)

    # ---- (6) dead命令/未使用タップの除去 + トポロジカル番号付け ----
    live = bytearray(instrs_created)
    tap_used = bytearray(len(taps))
    stack: list = []

    def push_ref(r: int) -> None:
        kind = ref_kind(r)
        if kind == REF_KIND_TAP:
            tap_used[ref_idx(r)] = 1
        elif kind == REF_KIND_INSTR and not live[ref_idx(r)]:
            live[ref_idx(r)] = 1
            stack.append(ref_idx(r))

    for oc in range(oc_count):
        push_ref(oc_out[oc])
    while stack:
        j = stack.pop()
        push_ref(draft[j][1])
        push_ref(draft[j][2])

    tap_remap = [-1] * len(taps)
    live_taps: list = []
    for t in range(len(taps)):
        if not tap_used[t]:
            continue
        tap_remap[t] = len(live_taps)
        live_taps.append(taps[t])

    # draftは生成順=トポロジカル順なので、liveな命令を昇順に詰め直せば順序は保たれる。
    remap = [-1] * instrs_created
    instrs: list = []  # each: [tt, a, b, slot]
    for j in range(instrs_created):
        if not live[j]:
            continue
        remap[j] = len(instrs)
        d = draft[j]
        instrs.append([d[0], d[1], d[2], -1])

    def fix_ref(r: int) -> int:
        kind = ref_kind(r)
        if kind == REF_KIND_INSTR:
            return make_ref(REF_KIND_INSTR, remap[ref_idx(r)])
        if kind == REF_KIND_TAP:
            return make_ref(REF_KIND_TAP, tap_remap[ref_idx(r)])
        return r

    for ins in instrs:
        ins[1] = fix_ref(ins[1])
        ins[2] = fix_ref(ins[2])
        # 命令番号の詰め直しで参照IDの大小関係が変わりうるので、可換正規形を張り直す
        if ins[2] < ins[1]:
            ins[1], ins[2] = ins[2], ins[1]
            ins[0] = TT_SWAP[ins[0]]

    for oc in range(oc_count):
        oc_out[oc] = fix_ref(oc_out[oc])

    n = len(instrs)

    # ---- 出力書き出しスケジュール(CSR) ----
    bucket_count = [0] * (n + 2)
    for oc in range(oc_count):
        r = oc_out[oc]
        bucket_count[ref_idx(r) + 1 if ref_kind(r) == REF_KIND_INSTR else 0] += 1
    write_start = [0] * (n + 2)
    acc = 0
    for i in range(n + 1):
        write_start[i] = acc
        acc += bucket_count[i]
    write_start[n + 1] = acc
    cursor = write_start[:n + 1]
    write_oc = [0] * oc_count
    for oc in range(oc_count):
        r = oc_out[oc]
        bucket = ref_idx(r) + 1 if ref_kind(r) == REF_KIND_INSTR else 0
        write_oc[cursor[bucket]] = oc
        cursor[bucket] += 1

    # ---- 生存区間解析(スロット割当) ----
    last_use = list(range(n))
    for j in range(n):
        ins = instrs[j]
        if ref_kind(ins[1]) == REF_KIND_INSTR:
            last_use[ref_idx(ins[1])] = j
        if ref_kind(ins[2]) == REF_KIND_INSTR:
            last_use[ref_idx(ins[2])] = j
    free: list = []
    n_slots = 0
    for j in range(n):
        if free:
            slot = free.pop()
        else:
            slot = n_slots
            n_slots += 1
        instrs[j][3] = slot
        ia, ib = instrs[j][1], instrs[j][2]
        if ref_kind(ia) == REF_KIND_INSTR and last_use[ref_idx(ia)] == j:
            free.append(instrs[ref_idx(ia)][3])
        if ref_kind(ib) == REF_KIND_INSTR and ib != ia and last_use[ref_idx(ib)] == j:
            free.append(instrs[ref_idx(ib)][3])
        if last_use[j] == j:
            free.append(slot)

    not_instrs = sum(1 for ins in instrs if ins[0] == TT_NOT and ins[1] == ins[2])
    oc_const = 0
    oc_tap = 0
    uniq = set()
    for oc in range(oc_count):
        uniq.add(oc_out[oc])
        k = ref_kind(oc_out[oc])
        if k == REF_KIND_CONST:
            oc_const += 1
        elif k == REF_KIND_TAP:
            oc_tap += 1

    stats = ConvProgramStats(
        gates_before=oc_count * n_internal,
        gates_after=n,
        const_folded=counters['const_folded'],
        alias_folded=counters['alias_folded'],
        not_folded=counters['not_folded'],
        cse_hits=counters['cse_hits'],
        emitted=counters['emitted'],
        instrs_created=instrs_created,
        dead_removed=instrs_created - n,
        not_instrs=not_instrs,
        taps_before=len(taps),
        n_taps=len(live_taps),
        n_slots=n_slots,
        oc_const=oc_const,
        oc_tap=oc_tap,
        oc_unique=len(uniq),
    )

    return ConvProgram(
        taps=live_taps,
        instrs=[ConvInstr(tt=i[0], a=i[1], b=i[2], slot=i[3]) for i in instrs],
        n_slots=n_slots,
        oc_out=oc_out,
        write_start=write_start,
        write_oc=write_oc,
        stats=stats,
    )


# ============================================================
# セルフチェック(全数真理値表照合、numpyビット並列)
# ============================================================


def _gate2(tt: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """gate2の16式(convProgram.ts::gate2Bits と同一)をnumpy bool配列で評価する。"""
    tt &= 0xf
    if tt == 0x0:
        return np.zeros_like(a)
    if tt == 0x1:
        return ~(a | b)
    if tt == 0x2:
        return a & ~b
    if tt == 0x3:
        return ~b
    if tt == 0x4:
        return ~a & b
    if tt == 0x5:
        return ~a
    if tt == 0x6:
        return a ^ b
    if tt == 0x7:
        return ~(a & b)
    if tt == 0x8:
        return a & b
    if tt == 0x9:
        return ~(a ^ b)
    if tt == 0xa:
        return a
    if tt == 0xb:
        return a | ~b
    if tt == 0xc:
        return b
    if tt == 0xd:
        return ~a | b
    if tt == 0xe:
        return a | b
    return np.ones_like(a)  # 0xf


def verify_conv_program(spec: ConvTreeSpec, prog: ConvProgram) -> None:
    """命令列が元のバランス木と全数一致することを検証する(TS版verifyConvProgramのミラー)。

    各ocについて「その木のユニークタップ数u(≤nLeaves)」に対し2^uパターンを
    numpy bool配列で評価し、元の木と縮約後の命令列(oc出力の依存錐だけ)を
    突き合わせる。不一致ならValueErrorを投げる。
    """
    oc_count = spec.out_channels
    n_leaves = 1 << spec.tree_depth
    n_internal = n_leaves - 1

    tap_id_of = {pp: i for i, pp in enumerate(prog.taps)}

    for oc in range(oc_count):
        row = spec.indices_flat[oc]
        var_of_pp: dict = {}
        for k in range(n_leaves):
            pp = row[k]
            if pp not in var_of_pp:
                var_of_pp[pp] = len(var_of_pp)
        u = len(var_of_pp)
        n_patterns = 1 << u
        patterns = np.arange(n_patterns, dtype=np.int64)
        leaf_cols = {v: ((patterns >> v) & 1).astype(bool) for v in range(u)}

        # このocの葉に現れるユニークタップへパターン変数を割り当てる。命令列側で
        # 落ちたタップにも変数を与えるので、「元の木が本当にそのタップに
        # 依存していない」ことまで検証できる。
        tap_vals: dict = {}
        for pp, v in var_of_pp.items():
            tid = tap_id_of.get(pp)
            if tid is not None:
                tap_vals[tid] = leaf_cols[v]

        # --- 元のバランス木を評価 ---
        cur = [leaf_cols[var_of_pp[row[k]]] for k in range(n_leaves)]
        g_offset = 0
        while len(cur) > 1:
            g_out = len(cur) >> 1
            nxt = []
            for g in range(g_out):
                tt = spec.truth_tables[oc * n_internal + g_offset + g] & 0xf
                nxt.append(_gate2(tt, cur[g << 1], cur[(g << 1) + 1]))
            g_offset += g_out
            cur = nxt
        want = cur[0]

        # --- 縮約後の命令列(出力の依存錐のみ)を評価 ---
        out0 = prog.oc_out[oc]
        in_cone: set = set()
        cone: list = []
        if ref_kind(out0) == REF_KIND_INSTR:
            in_cone.add(ref_idx(out0))
            cone.append(ref_idx(out0))
            dfs = [ref_idx(out0)]
            while dfs:
                j = dfs.pop()
                for r in (prog.instrs[j].a, prog.instrs[j].b):
                    if ref_kind(r) == REF_KIND_INSTR and ref_idx(r) not in in_cone:
                        in_cone.add(ref_idx(r))
                        dfs.append(ref_idx(r))
                        cone.append(ref_idx(r))
                    if ref_kind(r) == REF_KIND_TAP and ref_idx(r) not in tap_vals:
                        raise ValueError(
                            f'verify_conv_program: oc={oc} の依存錐に木の外の'
                            f'タップ {prog.taps[ref_idx(r)]} が現れた'
                        )
        cone.sort()  # トポロジカル順(生成順=昇順)

        instr_vals: dict = {}

        def ref_vals(r: int, _n=n_patterns, _taps=tap_vals, _instrs=instr_vals) -> np.ndarray:
            kind = ref_kind(r)
            if kind == REF_KIND_CONST:
                return (np.ones(_n, dtype=bool) if ref_idx(r) == 1
                        else np.zeros(_n, dtype=bool))
            if kind == REF_KIND_TAP:
                return _taps[ref_idx(r)]
            return _instrs[ref_idx(r)]

        for j in cone:
            ins = prog.instrs[j]
            instr_vals[j] = _gate2(ins.tt, ref_vals(ins.a), ref_vals(ins.b))

        got = ref_vals(out0)
        if not np.array_equal(want, got):
            raise ValueError(
                f'verify_conv_program: oc={oc} で真理値表不一致(u={u})'
            )


# ============================================================
# IR接続: reduce_conv_tree / graph_conv_stats
# ============================================================


def reduce_conv_tree(node_params: dict, *, verify: bool = True) -> ConvProgramStats:
    """IR conv2dノードのparamsを縮約し、統計(ConvProgramStats)を返す。

    Args:
        node_params: IRNode(type='conv2d').params(またはそれと同じキーを
            持つdict)。truth_tablesはhex文字列/整数列どちらでも可
        verify: Trueなら縮約前後の全数真理値表照合(verify_conv_program)を
            実行し、不一致ならValueErrorを送出する(既定True。ゲート数を
            「公式の真実源」として使う以上、既定でセルフチェックする)

    Returns:
        ConvProgramStats
    """
    spec = spec_from_node_params(node_params)
    prog = build_conv_program(spec)
    if verify:
        verify_conv_program(spec, prog)
    return prog.stats


def graph_conv_stats(graph: IRGraph, *, verify: bool = True) -> dict:
    """IR中の全conv2dノードを縮約し、ノード別+合計の統計を返す。

    Returns:
        {'nodes': [{'id': ..., **ConvProgramStatsのフィールド}, ...],
         'total': {ConvProgramStatsのフィールド: 合計値}}
        conv2dノードが無いグラフでは nodes=[]、total は全フィールド0。
    """
    nodes: list = []
    total = {name: 0 for name in _STATS_FIELDS}
    for node in graph.nodes:
        if node.type != 'conv2d':
            continue
        stats = reduce_conv_tree(node.params, verify=verify)
        row = {'id': node.id, **asdict(stats)}
        nodes.append(row)
        for name, value in asdict(stats).items():
            total[name] += value
    return {'nodes': nodes, 'total': total}
