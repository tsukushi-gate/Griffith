"""conv2d 重み共有木の don't care 再合成。

lossless 縮約(``optimize.py`` / ``convProgram.ts``)は
「木を評価しながらの peephole 簡約」であり、木の**構造そのものは変えない**。
本モジュールは各 (conv2dノード, 出力チャネル) の木を1つの論理関数とみなし、
**care set 上で等価な最小ゲート木を張り直す**(再合成)。

出力は IR フォーマット無変更:
    深さDのバランス木(葉 2^D、内部ゲート 2^D-1)という conv2d の表現の中で、
    余ったゲートを「パススルー(tt=0xa)」「定数(tt=0x0/0xf)」にしておけば
    A1 の縮約パス(定数畳み込み・退化tt・dead除去)がそれらを 0 ゲートに
    畳む。したがって TS ランタイム側は一切変更不要で、A1 縮約後の
    命令数がそのまま減る。

care set の2系統:
    (i) 全パターン(``care=None``)= **lossless**。A1 のような局所簡約では
        到達できない大域最小化(例 (a&b)|(a&c) → a&(b|c))が効く。
        精度は厳密に不変。
    (ii) 観測 care set = **lossy**。キャリブレーション入力を流して各木が
        実際に観測したタップパターンだけを care とし、未観測パターンを
        don't care として最小化する。精度は要検証。

探索(``synth_tree``):
    深さDのバランス木が表現できる回路は「木(ファンアウト共有なし)・
    深さ≦D」に限られる(D=3 なら最大7ゲート)。この位相の上で、根を
    ``gate(左部分木, 右部分木)`` に分けて片側を列挙・片側を逆算する
    分枝限定探索を行う。カバーする分割(左コスト, 右コスト)は:
        cost0  タップ直結 / 定数
        cost1  gate(tap, tap)
        cost2  (0,1)
        cost3  (1,1) / (0,2)
        cost4  (1,2) / (0,3)
        cost5  (2,2)
    (1,3)(2,3)(3,3) は右側辞書 |c1|^2*16 の構築が重いので張らない。
    **探索が空振りなら必ず現状維持**なので、再合成でゲート数が増えることは
    ない(厳密最小は保証しないが、退化もしない)。

    署名(signature)は「care パターン列に対する出力ビット列」の Python int。
    care に制限した時点で don't care は自動的に吸収されるため、探索本体に
    DC の特別扱いは要らない(部分DCが出るのは「解を逆算した相手オペランド」
    側だけで、そこは (req, care) の対で扱う)。
"""

from dataclasses import dataclass, field

# 探索で相手オペランドが部分don't careになったときに線形走査する上限。
# これを超える辞書に対する部分DC照合は諦める(=解を見逃すが、退化はしない)。
_SCAN_LIMIT = 8192

# パススルー(f(a,b)=a)と定数の tt。A1 の縮約で 0 ゲートに畳まれる。
TT_PASS_A = 0xa
TT_CONST0 = 0x0
TT_CONST1 = 0xf
TT_NOT_A_ONLY = 0x5  # f(a,b) = ~a


# ============================================================
# 解の木表現
# ============================================================
# ('t', v)          葉: taps[v]
# ('c', 0|1)        定数(根のみ)
# ('g', tt, l, r)   2入力ゲート


def tree_cost(node) -> int:
    """ゲート数(A1縮約後に残る命令数の見積り)。"""
    if node[0] != 'g':
        return 0
    return 1 + tree_cost(node[2]) + tree_cost(node[3])


def tree_depth(node) -> int:
    if node[0] != 'g':
        return 0
    return 1 + max(tree_depth(node[2]), tree_depth(node[3]))


def tree_taps(node, out: set) -> set:
    if node[0] == 't':
        out.add(node[1])
    elif node[0] == 'g':
        tree_taps(node[2], out)
        tree_taps(node[3], out)
    return out


# ============================================================
# ビットマスク上の 2入力ゲート
# ============================================================


def gate2_mask(tt: int, a: int, b: int, full: int) -> int:
    """tt のビット規約は optimize.py と同一(bit index = (b<<1)|a)。"""
    na = full ^ a
    nb = full ^ b
    r = 0
    if tt & 1:
        r |= na & nb
    if tt & 2:
        r |= a & nb
    if tt & 4:
        r |= na & b
    if tt & 8:
        r |= a & b
    return r


def _all_gate2(a: int, b: int, full: int) -> list:
    """16通りの tt について gate2 を一括算出する。"""
    na = full ^ a
    nb = full ^ b
    m0 = na & nb
    m1 = a & nb
    m2 = na & b
    m3 = a & b
    out = [0] * 16
    for tt in range(16):
        r = 0
        if tt & 1:
            r |= m0
        if tt & 2:
            r |= m1
        if tt & 4:
            r |= m2
        if tt & 8:
            r |= m3
        out[tt] = r
    return out


def _solve_b(tt: int, a: int, target: int, full: int, care_out: int) -> tuple:
    """gate2(tt, a, B) が ``care_out`` 上で ``target`` と一致する B を解く。

    Returns:
        (req, care) — 「care のビット位置で B が req と一致すればよい」。
        充足不能なら (None, None)。care==0 は「Bが何でもよい」
        (=ゲートが退化しており、より低コストの解が別途あるはず)。
    """
    na = full ^ a
    req = 0
    care = 0
    # a=0 の群: B=0 → tt bit0、B=1 → tt bit2
    for group, f0, f1 in ((na & care_out, tt & 1, (tt >> 2) & 1),
                          (a & care_out, (tt >> 1) & 1, (tt >> 3) & 1)):
        if group == 0:
            continue
        if f0 == f1:  # B に依存しない: target が定数 f0 でなければ不能
            want = group if f0 else 0
            if (target & group) != want:
                return None, None
        elif f0 == 0:  # 恒等: B = target
            req |= target & group
            care |= group
        else:  # 反転: B = ~target
            req |= (full ^ target) & group
            care |= group
    return req, care


def _lookup(table: dict, req: int, care: int, full: int, max_cost: int):
    """(req, care) に整合する最小コスト要素を table(sig -> (cost, tree))から探す。"""
    if care == full:
        hit = table.get(req)
        if hit is not None and hit[0] <= max_cost:
            return hit
        return None
    if len(table) > _SCAN_LIMIT:
        return None
    best = None
    for sig, entry in table.items():
        if entry[0] > max_cost:
            continue
        if (sig ^ req) & care == 0 and (best is None or entry[0] < best[0]):
            best = entry
    return best


# ============================================================
# 合成本体
# ============================================================


@dataclass
class SynthResult:
    tree: tuple
    cost: int
    n_taps: int


def synth_tree(target: int, cols: list, full: int, *, max_cost: int = 5):
    """care 制限された署名 ``target`` を最小ゲート数で実現する木を返す。

    Args:
        target: care パターン列上の目標出力ビットマスク
        cols: cols[v] = タップ v の care パターン列上の値ビットマスク
        full: (1 << ncare) - 1
        max_cost: 探索するゲート数の上限(0..5)

    Returns:
        SynthResult(cost <= max_cost) または None(上限内に解なし)
    """
    u = len(cols)

    # ---- cost 0 ----
    if target == 0:
        return SynthResult(('c', 0), 0, 0)
    if target == full:
        return SynthResult(('c', 1), 0, 0)
    for v in range(u):
        if cols[v] == target:
            return SynthResult(('t', v), 0, 1)
    if max_cost < 1:
        return None

    # ---- cost<=1 の辞書(子として使う。定数は含めない) ----
    d1: dict = {}

    def put(d, sig, cost, tree):
        cur = d.get(sig)
        if cur is None or cost < cur[0]:
            d[sig] = (cost, tree)

    for v in range(u):
        put(d1, cols[v], 0, ('t', v))
    for v in range(u):
        put(d1, full ^ cols[v], 1, ('g', TT_NOT_A_ONLY, ('t', v), ('t', v)))
    for v in range(u):
        for w in range(v + 1, u):
            gs = _all_gate2(cols[v], cols[w], full)
            for tt in range(16):
                put(d1, gs[tt], 1, ('g', tt, ('t', v), ('t', w)))

    hit = d1.get(target)
    if hit is not None and hit[0] <= 1:
        return SynthResult(hit[1], hit[0], len(tree_taps(hit[1], set())))
    if max_cost < 2:
        return None

    c1_items = [(sig, e) for sig, e in d1.items() if e[0] == 1]
    c0_items = [(cols[v], (0, ('t', v))) for v in range(u)]

    def split(req, care, left_items, resolve_right, budget):
        """(req,care) = gate(tt, L, R) を L 側列挙 + R 側逆算で探す(最小コスト)。"""
        best = None
        for lsig, lentry in left_items:
            rb = budget - 1 - lentry[0]
            if rb < 0:
                continue
            for tt in range(16):
                rreq, rcare = _solve_b(tt, lsig, req, full, care)
                if rreq is None or rcare == 0:
                    continue
                r = resolve_right(rreq, rcare, rb)
                if r is None:
                    continue
                total = 1 + lentry[0] + r[0]
                if best is None or total < best[0]:
                    best = (total, ('g', tt, lentry[1], r[1]))
        return best

    def res_d1(req, care, b):
        return _lookup(d1, req, care, full, b)

    best = split(target, full, c0_items, res_d1, max_cost)   # (0, 0..1)
    got = split(target, full, c1_items, res_d1, max_cost)    # (1, 0..1)
    if got is not None and (best is None or got[0] < best[0]):
        best = got
    if best is not None and best[0] <= 2:
        return SynthResult(best[1], best[0], len(tree_taps(best[1], set())))
    if max_cost < 3:
        return None

    # ---- cost<=2 の level2 辞書 T(2) ----
    d2 = dict(d1)
    for lsig, lentry in c1_items:
        for v in range(u):
            gs = _all_gate2(lsig, cols[v], full)
            for tt in range(16):
                put(d2, gs[tt], 2, ('g', tt, lentry[1], ('t', v)))

    def res_d2(req, care, b):
        return _lookup(d2, req, care, full, b)

    for left in (c0_items, c1_items):                        # (0,2) / (1,2)
        got = split(target, full, left, res_d2, max_cost)
        if got is not None and (best is None or got[0] < best[0]):
            best = got
    if best is not None and best[0] <= 3:
        return SynthResult(best[1], best[0], len(tree_taps(best[1], set())))
    if max_cost < 4:
        return None

    # level2 の cost3 = gate(cost1, cost1)。辞書化すると |c1|^2*16 で重いので
    # 「相手を逆算してから c1×c1 分解できるか」を入れ子で判定する。
    def res_g11(req, care, b):
        if b < 3:
            return None
        return split(req, care, c1_items, res_d1, 3)

    got = split(target, full, c0_items, res_g11, max_cost)   # (0,3)
    if got is not None and (best is None or got[0] < best[0]):
        best = got
    if best is not None and best[0] <= 4:
        return SynthResult(best[1], best[0], len(tree_taps(best[1], set())))
    if max_cost < 5:
        return None

    d2_c2 = [(sig, e) for sig, e in d2.items() if e[0] == 2]
    got = split(target, full, d2_c2, res_d2, max_cost)       # (2,2)
    if got is not None and (best is None or got[0] < best[0]):
        best = got
    if best is None:
        return None
    return SynthResult(best[1], best[0], len(tree_taps(best[1], set())))


# ============================================================
# バランス木への埋め込み(IRフォーマット無変更で出力する)
# ============================================================


def embed_balanced(node, taps: list, tree_depth_d: int) -> tuple:
    """解の木を深さ ``tree_depth_d`` のバランス木(葉/tt列)へ埋め込む。

    余ったゲートはパススルー(0xa)と定数(0x0)にする。A1 の縮約が
    それらを 0 ゲートに畳むため、実効ゲート数は ``tree_cost(node)``。

    Returns:
        (leaves, tts) — leaves は長さ 2^D のタップID(indices_flat の値)、
        tts は長さ 2^D-1 の tt(レベル昇順・optimize.py の並び)
    """
    n_leaves = 1 << tree_depth_d
    leaves = [taps[0]] * n_leaves
    tts = [TT_CONST0] * (n_leaves - 1)
    # レベルℓ(1..D)のゲート開始オフセット
    offsets = [0] * (tree_depth_d + 1)
    acc = 0
    for lvl in range(1, tree_depth_d + 1):
        offsets[lvl] = acc
        acc += 1 << (tree_depth_d - lvl)

    if node[0] == 'c':
        # 定数: 全ゲートを 0x0 / 0xf にする(A1 が定数畳み込みで消す)
        fill = TT_CONST1 if node[1] else TT_CONST0
        return leaves, [fill] * (n_leaves - 1)

    def place(nd, level, pos):
        if level == 0:
            assert nd[0] == 't', f'葉位置に非葉ノード: {nd[0]}'
            leaves[pos] = taps[nd[1]]
            return
        gi = offsets[level] + pos
        if nd[0] == 't' or tree_depth(nd) < level:
            # パススルー: 左に本体、右はダミー(定数0のまま)
            tts[gi] = TT_PASS_A
            place(nd, level - 1, pos * 2)
            return
        tts[gi] = nd[1]
        place(nd[2], level - 1, pos * 2)
        place(nd[3], level - 1, pos * 2 + 1)

    place(node, tree_depth_d, 0)
    return leaves, tts


# ============================================================
# 木の関数抽出
# ============================================================


def eval_tree(indices_row: list, tts_row: list, tree_depth_d: int,
              mask_of: dict, full: int) -> int:
    """タップID→値ビットマスクの割当のもとで木を評価する(ビット並列)。"""
    cur = [mask_of[pp] for pp in indices_row]
    g_offset = 0
    while len(cur) > 1:
        g_out = len(cur) >> 1
        nxt = []
        for g in range(g_out):
            tt = tts_row[g_offset + g] & 0xf
            nxt.append(gate2_mask(tt, cur[g << 1], cur[(g << 1) + 1], full))
        g_offset += g_out
        cur = nxt
    return cur[0]


def tree_function(indices_row: list, tts_row: list, tree_depth_d: int) -> tuple:
    """1本の木から (ユニークタップ列, 2^u パターンの真理値表) を得る。

    Returns:
        (taps, onset) — taps はタップID列(出現順)、onset は
        ``bit p = f(パターンp)``、パターン p の bit v = taps[v] の値
    """
    n_leaves = 1 << tree_depth_d
    taps: list = []
    var_of: dict = {}
    for k in range(n_leaves):
        pp = indices_row[k]
        if pp not in var_of:
            var_of[pp] = len(taps)
            taps.append(pp)
    u = len(taps)
    n_pat = 1 << u
    fullp = (1 << n_pat) - 1
    # 各変数の「全パターン上の値」マスク
    var_mask = []
    for v in range(u):
        mm = 0
        for p in range(n_pat):
            if (p >> v) & 1:
                mm |= 1 << p
        var_mask.append(mm)

    cur = [var_mask[var_of[indices_row[k]]] for k in range(n_leaves)]
    g_offset = 0
    while len(cur) > 1:
        g_out = len(cur) >> 1
        nxt = []
        for g in range(g_out):
            tt = tts_row[g_offset + g] & 0xf
            nxt.append(gate2_mask(tt, cur[g << 1], cur[(g << 1) + 1], fullp))
        g_offset += g_out
        cur = nxt
    return taps, cur[0]


# ============================================================
# ノード単位の再合成
# ============================================================


# ============================================================
# キャリブレーション用の高速 numpy フォワード(care set 採取用)
# ============================================================


def _pair(v) -> tuple:
    if isinstance(v, list | tuple):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _g2np(tt: int, a, b):
    """0/1 の uint8 配列に対する 2入力ゲート(optimize._gate2 と同一規約)。"""
    import numpy as np
    tt &= 0xf
    if tt == 0x0:
        return np.zeros_like(a)
    if tt == 0xf:
        return np.ones_like(a)
    if tt == 0xa:
        return a
    if tt == 0xc:
        return b
    if tt == 0x5:
        return a ^ 1
    if tt == 0x3:
        return b ^ 1
    if tt == 0x8:
        return a & b
    if tt == 0xe:
        return a | b
    if tt == 0x6:
        return a ^ b
    if tt == 0x9:
        return (a ^ b) ^ 1
    if tt == 0x7:
        return (a & b) ^ 1
    if tt == 0x1:
        return (a | b) ^ 1
    if tt == 0x2:
        return a & (b ^ 1)
    if tt == 0x4:
        return (a ^ 1) & b
    if tt == 0xb:
        return a | (b ^ 1)
    return (a ^ 1) | b  # 0xd


def _conv_tap_planes(h, params, tap_ids):
    """conv2dノードの入力から、指定タップIDの平面 (N, Hout, Wout) を作る。"""
    import numpy as np
    c_in, hh, ww = (int(v) for v in params['in_shape'])
    k_h, k_w = _pair(params['kernel_size'])
    s_h, s_w = _pair(params['stride'])
    p_h, p_w = _pair(params['padding'])
    h_out = (hh + 2 * p_h - k_h) // s_h + 1
    w_out = (ww + 2 * p_w - k_w) // s_w + 1
    n = h.shape[0]
    xin = h.reshape(n, c_in, hh, ww)
    if p_h or p_w:
        xp = np.zeros((n, c_in, hh + 2 * p_h, ww + 2 * p_w), dtype=np.uint8)
        xp[:, :, p_h:p_h + hh, p_w:p_w + ww] = xin
    else:
        xp = xin
    planes = {}
    for pp in tap_ids:
        c, rem = divmod(int(pp), k_h * k_w)
        dy, dx = divmod(rem, k_w)
        planes[pp] = xp[:, c, dy:dy + h_out * s_h:s_h, dx:dx + w_out * s_w:s_w]
    return planes, h_out, w_out


def conv2d_forward(h, params):
    """conv2dノードを評価する。h: (N, in_width) uint8 → (N, out_width) uint8。"""
    import numpy as np

    from ..ir.serialize import tt_from_hex
    tts_raw = params['truth_tables']
    tts = (tt_from_hex(list(tts_raw)) if tts_raw and isinstance(tts_raw[0], str)
           else [int(v) for v in tts_raw])
    depth = int(params['tree_depth'])
    oc_count = int(params['out_channels'])
    n_int = (1 << depth) - 1
    idx = params['indices_flat']
    tap_ids = {int(v) for row in idx for v in row}
    planes, h_out, w_out = _conv_tap_planes(h, params, tap_ids)
    n = h.shape[0]
    out = np.empty((n, oc_count, h_out, w_out), dtype=np.uint8)
    for oc in range(oc_count):
        cur = [planes[int(v)] for v in idx[oc]]
        g_off = 0
        while len(cur) > 1:
            g_out = len(cur) >> 1
            cur = [_g2np(tts[oc * n_int + g_off + g] & 0xf,
                         cur[g << 1], cur[(g << 1) + 1])
                   for g in range(g_out)]
            g_off += g_out
        out[:, oc] = cur[0]
    return out.reshape(n, -1)


def conv2d_care_sets(h, params):
    """conv2dノードの各ocについて、観測されたタップパターン集合を返す。

    Returns:
        list[set[int]] — care_sets[oc] は 0..2^u-1 のパターン集合。
        u とパターンのビット順は ``tree_function`` の taps 出現順に一致する。
    """
    import numpy as np
    idx = params['indices_flat']
    tap_ids = {int(v) for row in idx for v in row}
    planes, _, _ = _conv_tap_planes(h, params, tap_ids)
    out = []
    for row in idx:
        taps: list = []
        var_of: dict = {}
        for v in row:
            v = int(v)
            if v not in var_of:
                var_of[v] = len(taps)
                taps.append(v)
        pat = np.zeros_like(planes[taps[0]], dtype=np.uint16)
        for i, pp in enumerate(taps):
            pat |= planes[pp].astype(np.uint16) << i
        seen = np.bincount(pat.ravel(), minlength=1 << len(taps))
        out.append({int(p) for p in np.nonzero(seen)[0]})
    return out


_FORWARD_TYPES = ('permute', 'conv2d', 'or_pool2d', 'upsample2d',
                  'concat', 'group_sum')


def forward_graph(graph, x, *, care_nodes=None):
    """IRを numpy で直接実行する(lower_to_primitives を経由しない高速路)。

    ``ir/simulate.py`` と整数一致することを ``tests/l1_math/test_dc_resynth.py``
    で検証している。conv2d を展開しないためメモリ・時間が桁違いに軽い。

    Args:
        graph: IRGraph(conv2d ネイティブノードを含んでよい)
        x: (N, in_width) の 0/1 配列
        care_nodes: care set を採取する conv2d ノードid の集合(Noneなら採取しない)

    Returns:
        (out, care) — out は最終出力(group_sum終端なら整数スコア)、
        care は {node_id: list[set[int]]}
    """
    import numpy as np
    vals = {p.id: np.asarray(x, dtype=np.uint8) for p in graph.inputs}
    grouped: dict = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    in_srcs = {dst: [e.src for e in sorted(es, key=lambda e: e.dst_slot)]
               for dst, es in grouped.items()}
    care: dict = {}

    for node in graph.topological_order():
        ins = [vals[s] for s in in_srcs.get(node.id, [])]
        h = ins[0] if ins else None
        p = node.params
        t = node.type
        if t == 'permute':
            dims = [int(v) for v in p['in_shape']]
            perm = [int(v) for v in p['perm']]
            out = h.reshape(h.shape[0], *dims).transpose(
                0, *[q + 1 for q in perm]).reshape(h.shape[0], -1)
        elif t == 'conv2d':
            if care_nodes is not None and node.id in care_nodes:
                care[node.id] = conv2d_care_sets(h, p)
            out = conv2d_forward(h, p)
        elif t == 'or_pool2d':
            c_in, hh, ww = (int(v) for v in p['in_shape'])
            k = int(p['kernel_size'])
            s = int(p['stride'])
            pad = int(p.get('padding', 0))
            h_out = (hh + 2 * pad - k) // s + 1
            w_out = (ww + 2 * pad - k) // s + 1
            xin = h.reshape(h.shape[0], c_in, hh, ww)
            if pad:
                xp = np.zeros((h.shape[0], c_in, hh + 2 * pad, ww + 2 * pad),
                              dtype=np.uint8)
                xp[:, :, pad:pad + hh, pad:pad + ww] = xin
            else:
                xp = xin
            acc = np.zeros((h.shape[0], c_in, h_out, w_out), dtype=np.uint8)
            for dy in range(k):
                for dx in range(k):
                    acc |= xp[:, :, dy:dy + h_out * s:s, dx:dx + w_out * s:s]
            out = acc.reshape(h.shape[0], -1)
        elif t == 'upsample2d':
            c_in, hh, ww = (int(v) for v in p['in_shape'])
            sc = int(p['scale'])
            xin = h.reshape(h.shape[0], c_in, hh, ww)
            out = np.repeat(np.repeat(xin, sc, axis=2), sc, axis=3)
            out = out.reshape(h.shape[0], -1)
        elif t == 'concat':
            out = np.concatenate(ins, axis=1)
        elif t == 'group_sum':
            out = h.reshape(h.shape[0], int(p['k']), -1).sum(axis=-1)
        else:
            raise NotImplementedError(f'forward_graph 未対応ノード型: {t}')
        vals[node.id] = out

    src = graph.output_source()
    last = vals[src] if src is not None else vals[graph.topological_order()[-1].id]
    return last, care


@dataclass
class ResynthStats:
    n_trees: int = 0
    n_changed: int = 0
    cost_before: int = 0
    cost_after: int = 0
    taps_before: int = 0
    taps_after: int = 0
    n_care_full: int = 0
    care_patterns: int = 0
    care_total: int = 0
    per_cost_saved: dict = field(default_factory=dict)

    def merge(self, other) -> None:
        self.n_trees += other.n_trees
        self.n_changed += other.n_changed
        self.cost_before += other.cost_before
        self.cost_after += other.cost_after
        self.taps_before += other.taps_before
        self.taps_after += other.taps_after
        self.n_care_full += other.n_care_full
        self.care_patterns += other.care_patterns
        self.care_total += other.care_total
        for k, v in other.per_cost_saved.items():
            self.per_cost_saved[k] = self.per_cost_saved.get(k, 0) + v


def _solo_cost(indices_row: list, tts_row: list, tree_depth_d: int) -> int:
    """A1 縮約を単木に適用したときの残存ゲート数。"""
    from .optimize import ConvTreeSpec, build_conv_program
    spec = ConvTreeSpec(
        out_channels=1,
        tree_depth=tree_depth_d,
        indices_flat=[list(indices_row)],
        truth_tables=[t & 0xf for t in tts_row],
    )
    return build_conv_program(spec).stats.gates_after


def _var_mask(j: int, u: int) -> int:
    """2^u パターン上で「変数 j が 1」のビットマスク。"""
    block = ((1 << (1 << j)) - 1) << (1 << j)       # 幅 2^(j+1) の上半分
    mm = 0
    for b in range(1 << (u - j - 1)):
        mm |= block << (b << (j + 1))
    return mm


def essential_vars(onset: int, u: int) -> list:
    """真理値表 ``onset``(2^u ビット)が実際に依存する変数番号(昇順)。

    変数 j に依存 ⇔ j=0 のパターンで f(p) != f(p | 1<<j) となるものがある。
    """
    out = []
    for j in range(u):
        hi = _var_mask(j, u)
        if ((onset >> (1 << j)) ^ onset) & (((1 << (1 << u)) - 1) ^ hi):
            out.append(j)
    return out


def resynth_tree(indices_row: list, tts_row: list, tree_depth_d: int,
                 care_patterns=None, *, max_cost: int = 5, prune: bool = True):
    """1本の木を care set 上で再合成する。

    Args:
        indices_row: 長さ 2^D のタップID列
        tts_row: 長さ 2^D-1 の tt 列
        tree_depth_d: 木の深さ D
        care_patterns: care とするパターン(0..2^u-1 の集合)。None なら全パターン
            (=lossless)。u は ``tree_function`` が返すユニークタップ数
        max_cost: 探索するゲート数の上限
        prune: True(既定)なら lossless(care=None)のとき、(a) 厳密な下界
            ``本質変数 e 個 ⇒ ≥ e-1 ゲート`` により既に下界の木は探索を省き、
            (b) 探索を本質変数だけに制限する。探索結果(ゲート数)は不変で
            速いだけ(実測 7〜20 倍)。False は旧経路(全タップ全探索。回帰
            テストの基準として残す)。care ありのときは常に旧経路

    Returns:
        (leaves, tts, before_cost, after_cost) — 改善が無ければ入力をそのまま返す
    """
    taps, onset = tree_function(indices_row, tts_row, tree_depth_d)
    u = len(taps)
    before = _solo_cost(indices_row, tts_row, tree_depth_d)

    if prune and care_patterns is None:
        return _resynth_tree_pruned(indices_row, tts_row, tree_depth_d,
                                    taps, onset, before, max_cost)

    if care_patterns is None:
        pats = list(range(1 << u))
    else:
        pats = sorted(care_patterns)
    m = len(pats)
    if m == 0:
        # 一度も観測されない木: 定数0で潰す
        leaves, tts = embed_balanced(('c', 0), taps, tree_depth_d)
        return leaves, tts, before, 0

    full = (1 << m) - 1
    cols = []
    for v in range(u):
        mm = 0
        for i, p in enumerate(pats):
            if (p >> v) & 1:
                mm |= 1 << i
        cols.append(mm)
    target = 0
    for i, p in enumerate(pats):
        if (onset >> p) & 1:
            target |= 1 << i

    budget = min(max_cost, before - 1)
    if budget < 0:
        return list(indices_row), list(tts_row), before, before
    res = synth_tree(target, cols, full, max_cost=budget)
    if res is None or res.cost >= before:
        return list(indices_row), list(tts_row), before, before

    leaves, tts = embed_balanced(res.tree, taps, tree_depth_d)
    # 埋め込み後に A1 縮約でのゲート数を実測(パススルー畳み込みの検算)
    after = _solo_cost(leaves, tts, tree_depth_d)
    if after >= before:
        return list(indices_row), list(tts_row), before, before
    return leaves, tts, before, after


def resynth_node_params(params: dict, care_sets=None, *, max_cost: int = 5):
    """IR conv2d ノードの params を再合成した新 params と統計を返す。

    Args:
        params: conv2d ノードの params(truth_tables は hex 文字列/整数どちらも可)
        care_sets: care_sets[oc] = そのocの木の care パターン集合。None なら全 oc lossless
        max_cost: 探索ゲート数上限

    Returns:
        (new_params, ResynthStats)
    """
    from ..ir.serialize import tt_from_hex, tt_to_hex
    tts_raw = params['truth_tables']
    is_hex = bool(tts_raw) and isinstance(tts_raw[0], str)
    tts_all = tt_from_hex(list(tts_raw)) if is_hex else [int(v) for v in tts_raw]
    depth = int(params['tree_depth'])
    oc_count = int(params['out_channels'])
    n_int = (1 << depth) - 1

    new_idx: list = []
    new_tts: list = []
    st = ResynthStats()
    for oc in range(oc_count):
        row = [int(v) for v in params['indices_flat'][oc]]
        trow = tts_all[oc * n_int:(oc + 1) * n_int]
        care = None if care_sets is None else care_sets[oc]
        leaves, tts, before, after = resynth_tree(
            row, trow, depth, care, max_cost=max_cost)
        new_idx.append(leaves)
        new_tts.extend(tts)
        st.n_trees += 1
        st.cost_before += before
        st.cost_after += after
        if after < before:
            st.n_changed += 1
            st.per_cost_saved[before] = st.per_cost_saved.get(before, 0) + (before - after)
        u = len(set(row))
        st.care_total += 1 << u
        if care is None:
            st.n_care_full += 1
            st.care_patterns += 1 << u
        else:
            st.care_patterns += len(care)

    new_params = dict(params)
    new_params['indices_flat'] = new_idx
    new_params['truth_tables'] = tt_to_hex(new_tts) if is_hex else new_tts
    return new_params, st


def _resynth_tree_pruned(indices_row, tts_row, tree_depth_d, taps, onset,
                         before, max_cost):
    """lossless 専用の枝刈り版(``resynth_tree`` から呼ぶ)。

    旧経路と同じ ``synth_tree`` を、本質変数だけの列で呼ぶ。非本質変数は
    最小回路で使う意味がないので探索結果は変わらない。
    """
    keep = (list(indices_row), list(tts_row), before, before)
    budget = min(max_cost, before - 1)
    if budget < 0:
        return keep
    ev = essential_vars(onset, len(taps))
    e = len(ev)
    if budget < max(e - 1, 0):          # 既に下界(これ以上減らせない)
        return keep
    n_pat = 1 << e
    full = (1 << n_pat) - 1
    cols = [_var_mask(v, e) for v in range(e)]
    target = 0
    for i in range(n_pat):
        fp = 0
        for v, ov in enumerate(ev):
            if (i >> v) & 1:
                fp |= 1 << ov
        if (onset >> fp) & 1:
            target |= 1 << i
    res = synth_tree(target, cols, full, max_cost=budget)
    if res is None or res.cost >= before:
        return keep
    leaves, tts = embed_balanced(
        res.tree, [taps[j] for j in ev] or taps, tree_depth_d)
    after = _solo_cost(leaves, tts, tree_depth_d)
    if after >= before:
        return keep
    return leaves, tts, before, after
