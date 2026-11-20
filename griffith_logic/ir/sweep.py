"""IR sweep: 測定正規化パス(SM2)。

EDAツールの合成が掃除する論理(定数・縮退・重複・未参照ゲート)をIR上で
除去した正規形を作る。目的は**最適化ではなく測定の分母合わせ**:
pre-EDAの生ゲート数で学習法・トポロジーを評価すると「EDAがどうせ消す分」を
二重計上するため、指標は本パスの `swept_gates` を分母にする
(構造メトリクスの設計方針)。HDL出力経路には接続しない(出荷は従来どおり
無最適化でEDAに委ねる)。

パス(不動点まで反復):
    1. 縮退スロットの束縛: 同一ワイヤを指す複数スロットのttを片方に特殊化
    2. 定数伝播: 定数ゲート(サポート0)/constantノードの値を下流lut_layerの
       ttへ特殊化して吸収
    3. CSE: 同一機能ゲート(使用ワイヤ集合+正準化tt)への参照付け替え
    4. 未参照ゲート除去: どの消費者からも実効参照されないゲートを層から削除
       (消費者が全てlut_layerの場合のみ。group_sum/concat等の位置依存
       消費者・出力ポートに繋がる層は幅を変えない)
    5. 死ノード除去: 消費者が一切ないノードの削除

層IRの構造的制約: 恒等/否定ゲート(サポート1)は「前層入力の転送」そのもの
であり、層をまたぐ配線エイリアスを表現できないため**グラフからは除去しない**。
EDA(平坦ネットリスト)では消えるため、reportの `swept_gates` は
サポート0/1の残存ゲートを差し引いた値を報告する。

正しさの定義: 任意入力で simulate(graph) == simulate(graph') のbit一致
(tests/integration/test_ir_sweep.py)。
"""

import copy

import numpy as np

from .schema import IRGraph
from .serialize import tt_from_hex, tt_to_hex

# lowering展開が必要なノード型は export/lowering.LOWERABLE_TYPES が単一情報源。
# ir→exportのモジュールレベル循環importを避けるため、参照は関数内の遅延importで行う
# (複製すると upsample2d/permute 等のノード型追加に追従できないため)

_MAX_ITERS = 100


# ---------------------------------------------------------------------------
# 真理値表のビット操作
# ---------------------------------------------------------------------------

def _slot_used(tt: int, j: int, n: int) -> bool:
    """ttがスロットjに依存するか(∃k: tt[k] != tt[k^(1<<j)])。"""
    for k in range(1 << n):
        if ((tt >> k) & 1) != ((tt >> (k ^ (1 << j))) & 1):
            return True
    return False


def _used_slots(tt: int, n: int) -> list[int]:
    return [j for j in range(n) if _slot_used(tt, j, n)]


def _specialize_slot(tt: int, j: int, v: int, n: int) -> int:
    """スロットjを定数vに固定した真理値表(lowering._specialize_zeroの0/1両対応)。"""
    out = 0
    for k in range(1 << n):
        src = (k & ~(1 << j)) | (v << j)
        if (tt >> src) & 1:
            out |= 1 << k
    return out


def _tie_slots(tt: int, j_dst: int, j_src: int, n: int) -> int:
    """スロットj_dstをj_srcと同値に束縛した真理値表(縮退スロットの吸収)。"""
    out = 0
    for k in range(1 << n):
        b = (k >> j_src) & 1
        src = (k & ~(1 << j_dst)) | (b << j_dst)
        if (tt >> src) & 1:
            out |= 1 << k
    return out


def _permute_tt(tt: int, perm: list[int], n: int) -> int:
    """スロット順の置換: 新スロットpが旧スロットperm[p]を読む真理値表。"""
    out = 0
    for k_new in range(1 << n):
        k_old = 0
        for p in range(n):
            if (k_new >> p) & 1:
                k_old |= 1 << perm[p]
        if (tt >> k_old) & 1:
            out |= 1 << k_new
    return out


def _canonical_gate(wires: list[int], tt: int, n: int) -> tuple:
    """CSE用の正準形: 使用ワイヤを昇順に並べ替えたときの(ワイヤ列, tt)。

    未使用スロットは除外した上で、残りをワイヤ昇順に正準化する
    (XOR(a,b)とXOR(b,a)を同一視)。定数ゲートは呼び出し側で除外する。
    """
    used = _used_slots(tt, n)
    order = sorted(used, key=lambda j: (wires[j], j))
    # 使用スロットを先頭に詰め、残りは任意固定値0に特殊化した縮約ttを作る
    perm = order + [j for j in range(n) if j not in used]
    tt_p = _permute_tt(tt, perm, n)
    for p in range(len(order), n):
        tt_p = _specialize_slot(tt_p, p, 0, n)
    return (tuple(wires[j] for j in order), tt_p)


# ---------------------------------------------------------------------------
# グラフ側ヘルパ
# ---------------------------------------------------------------------------

def _lut_nodes(graph):
    return [n for n in graph.nodes if n.type == 'lut_layer']


def _parse(node) -> tuple[np.ndarray, list[int], int]:
    p = node.params
    idx = np.asarray(p['indices'], dtype=np.int64).copy()
    tts = tt_from_hex(p['truth_tables'])
    return idx, tts, int(p['n_inputs'])


def _write(node, idx: np.ndarray, tts: list[int]) -> None:
    node.params['indices'] = [[int(v) for v in row] for row in idx]
    node.params['truth_tables'] = tt_to_hex(tts)
    node.params['out_width'] = len(tts)


def _consumers(graph) -> dict[str, list]:
    """src id → [(consumer nodeまたは'__output__', kind)] のマップ。"""
    out_ids = {p.id for p in graph.outputs}
    nodes = {n.id: n for n in graph.nodes}
    result: dict[str, list] = {n.id: [] for n in graph.nodes}
    for p in graph.inputs:
        result[p.id] = []
    for e in graph.edges:
        if e.src not in result:
            continue
        if e.dst in out_ids:
            result[e.src].append(('__output__', 'output'))
        elif e.dst in nodes:
            result[e.src].append((nodes[e.dst], nodes[e.dst].type))
    return result


def _const_wires(graph) -> dict[str, dict[int, int]]:
    """src id → {出力ビット位置: 定数値} のマップ(定数ゲート+constantノード)。"""
    result: dict[str, dict[int, int]] = {}
    for node in graph.nodes:
        if node.type == 'lut_layer':
            _, tts, n = _parse(node)
            consts = {
                o: tt & 1 for o, tt in enumerate(tts)
                if not _used_slots(tt, n)
            }
            if consts:
                result[node.id] = consts
        elif node.type == 'constant':
            result[node.id] = {
                o: int(b) for o, b in enumerate(node.params['bits'])
            }
    return result


# ---------------------------------------------------------------------------
# 各パス(いずれも変更があればTrueを返す)
# ---------------------------------------------------------------------------

def _pass_tie_degenerate(graph, report) -> bool:
    changed = False
    for node in _lut_nodes(graph):
        idx, tts, n = _parse(node)
        dirty = False
        for o in range(len(tts)):
            for j2 in range(1, n):
                if not _slot_used(tts[o], j2, n):
                    continue
                for j1 in range(j2):
                    if idx[j1, o] == idx[j2, o]:
                        tts[o] = _tie_slots(tts[o], j2, j1, n)
                        report['degenerate_tied'] += 1
                        dirty = True
                        break
        if dirty:
            _write(node, idx, tts)
            changed = True
    return changed


def _pass_const_absorb(graph, report) -> bool:
    consts = _const_wires(graph)
    if not consts:
        return False
    srcs_of = {}
    for e in graph.edges:
        srcs_of.setdefault(e.dst, []).append(e.src)
    changed = False
    for node in _lut_nodes(graph):
        src_ids = srcs_of.get(node.id, [])
        if len(src_ids) != 1 or src_ids[0] not in consts:
            continue
        cmap = consts[src_ids[0]]
        idx, tts, n = _parse(node)
        dirty = False
        for o in range(len(tts)):
            for j in range(n):
                w = int(idx[j, o])
                if w in cmap and _slot_used(tts[o], j, n):
                    tts[o] = _specialize_slot(tts[o], j, cmap[w], n)
                    report['constants_absorbed'] += 1
                    dirty = True
        if dirty:
            _write(node, idx, tts)
            changed = True
    return changed


def _pass_cse(graph, report) -> bool:
    consumers = _consumers(graph)
    changed = False
    for node in _lut_nodes(graph):
        idx, tts, n = _parse(node)
        canon_of: dict[tuple, int] = {}
        redirect: dict[int, int] = {}
        for o in range(len(tts)):
            used = _used_slots(tts[o], n)
            if not used:
                continue  # 定数は定数伝播が担当
            key = _canonical_gate([int(v) for v in idx[:, o]], tts[o], n)
            if key in canon_of:
                redirect[o] = canon_of[key]
            else:
                canon_of[key] = o
        if not redirect:
            continue
        applied = False
        for consumer, kind in consumers[node.id]:
            if kind != 'lut_layer':
                continue
            c_idx, c_tts, _ = _parse(consumer)
            mask = np.isin(c_idx, list(redirect))
            if mask.any():
                c_idx = c_idx.copy()
                for dup, rep in redirect.items():
                    c_idx[c_idx == dup] = rep
                _write(consumer, c_idx, c_tts)
                applied = True
        if applied:
            # 実際に付け替えた反復でのみ数える(残存重複の再計上を防ぐ)
            report['cse_merged'] += len(redirect)
            changed = True
    return changed


def _pass_remove_unreferenced(graph, report) -> bool:
    consumers = _consumers(graph)
    changed = False
    for node in _lut_nodes(graph):
        out_width = int(node.params['out_width'])
        referenced = np.zeros(out_width, dtype=bool)
        removable = True
        for consumer, kind in consumers[node.id]:
            if kind != 'lut_layer':
                # 出力ポート・位置依存ノード(group_sum/concat/add等)は
                # 全ビットを位置で参照する: 幅を変えない
                referenced[:] = True
                removable = False
                break
            c_idx, c_tts, c_n = _parse(consumer)
            for o in range(len(c_tts)):
                for j in _used_slots(c_tts[o], c_n):
                    referenced[c_idx[j, o]] = True
        if not removable or referenced.all():
            continue
        if not referenced.any():
            referenced[0] = True  # 空レイヤ化はしない(死ノード除去が担当)
        new_width = int(referenced.sum())
        if new_width == out_width:
            continue

        remap = np.full(out_width, -1, dtype=np.int64)
        remap[referenced] = np.arange(new_width)
        idx, tts, _ = _parse(node)
        _write(node, idx[:, referenced],
               [t for o, t in enumerate(tts) if referenced[o]])
        for consumer, _kind in consumers[node.id]:
            c_idx, c_tts, c_n = _parse(consumer)
            c_idx = remap[c_idx]
            c_idx[c_idx < 0] = 0  # 未使用スロットの参照先消滅は任意位置でよい
            _write(consumer, c_idx, c_tts)
            consumer.params['in_width'] = new_width
        report['gates_removed'] += out_width - new_width
        changed = True
    return changed


def _pass_remove_dead_nodes(graph, report) -> bool:
    consumers = _consumers(graph)
    dead = [
        n.id for n in graph.nodes
        if not consumers[n.id] and n.type != 'blackbox'
    ]
    if not dead:
        return False
    dead_set = set(dead)
    graph.nodes = [n for n in graph.nodes if n.id not in dead_set]
    graph.edges = [e for e in graph.edges if e.dst not in dead_set]
    report['nodes_removed'] += len(dead)
    return True


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

def _count_gates(graph) -> int:
    return sum(int(n.params['out_width']) for n in _lut_nodes(graph))


def sweep(graph: IRGraph) -> tuple[IRGraph, dict]:
    """IRを測定正規形へ変換し、(正規化済みグラフ, レポート) を返す。

    conv2d/or_pool2d/upsample2d/permute(=LOWERABLE_TYPES)を含む場合は
    lut_layer列へ展開してから処理する
    (ゲート粒度=HDLと同じ粒度で測るため。ノードidは展開後のもの)。

    Returns:
        graph': simulate出力が元とbit一致する正規化済みIR(入力は不変)
        report: {
            'gates_before', 'gates_after', 'iterations',
            'degenerate_tied', 'constants_absorbed', 'cse_merged',
            'gates_removed', 'nodes_removed',
            'passthrough_remaining',  # 残存サポート1(EDAが配線へ畳む)
            'const_remaining',        # 残存サポート0(EDAが定数へ畳む)
            'swept_gates',  # gates_after - passthrough/const_remaining
            'sweep_ratio',  # swept_gates / gates_before
        }
    """
    from ..export.lowering import LOWERABLE_TYPES, lower_to_primitives
    if any(n.type in LOWERABLE_TYPES for n in graph.nodes):
        g = lower_to_primitives(graph)
    else:
        g = copy.deepcopy(graph)

    report = {
        'gates_before': _count_gates(g),
        'iterations': 0,
        'degenerate_tied': 0,
        'constants_absorbed': 0,
        'cse_merged': 0,
        'gates_removed': 0,
        'nodes_removed': 0,
    }

    for _ in range(_MAX_ITERS):
        report['iterations'] += 1
        changed = _pass_tie_degenerate(g, report)
        changed |= _pass_const_absorb(g, report)
        changed |= _pass_cse(g, report)
        changed |= _pass_remove_unreferenced(g, report)
        changed |= _pass_remove_dead_nodes(g, report)
        if not changed:
            break

    passthrough = 0
    const = 0
    for node in _lut_nodes(g):
        _, tts, n = _parse(node)
        for tt in tts:
            s = len(_used_slots(tt, n))
            if s == 0:
                const += 1
            elif s == 1:
                passthrough += 1

    report['gates_after'] = _count_gates(g)
    report['passthrough_remaining'] = passthrough
    report['const_remaining'] = const
    report['swept_gates'] = report['gates_after'] - passthrough - const
    report['sweep_ratio'] = (
        report['swept_gates'] / report['gates_before']
        if report['gates_before'] else 0.0
    )
    return g, report
