"""IRレベルの真理値表flip修復(貪欲座標降下、C-系)。

`griffith_logic/analysis/dontcare_repair.py` / `guided_flip.py` は
torchのLutLayer(16ゲートモード等)前提だが、本モジュールはIR(JSON)の
`lut_layer` ノードを直接対象にする(torch不要、numpy+標準ライブラリのみ)。
設計思想(fit採択/val報告、RepairReport構造)はdontcare_repair.pyを踏襲する。

対象IRの構造: x → lut_layer鎖(concatを挟む場合あり) → group_sum → y。
group_sumは in_width を k 分割した各群のpopcountをクラススコアとする
(`griffith_logic/ir/simulate.py` の実装通り: reshape(batch,k,-1).sum(-1))。

目的関数は2種:
    objective='argmax' — group_sum後のargmaxが教師ラベルと一致する率
    objective='bits'   — group_sumを無視し、core出力(group_sumへの入力。
                         group_sumが無いグラフでは出力ポート直結ノード)の
                         ビット列が教師ビット列と一致する率

アルゴリズム:
    1. 出力からn_layers_from_output以内のlut_layerノードのみを対象にする
       (`target_nodes`、トポロジ順の末尾K件)
    2. 候補生成: 各対象ノードの各出力列oについて、fitデータで実際に
       訪問されたアドレスのみを候補にする(未訪問アドレスは一切変更しない)。
       アドレス集合はパスの先頭で1回のsimulate_traceから計算する
       (パスをまたぐと上流ノードのflipで真の入力が変わりうるため、
       「1回」はパス単位で解釈し、パスごとに再計算して鮮度を保つ。
       これによりパス間の陳腐化バグを避ける — 設計判断は本ファイル末尾コメント参照)
    3. 出力に直結するlut_layer(target_id自身、またはgroup_sumの直接入力)は
       スコア/ビットキャッシュによる閉形式評価で貪欲flip(再シミュレーション不要、
       O(該当サンプル数)で厳密)
    4. それ以外の対象ノード(深い層)は、誤分類サンプルの訪問数で候補を
       topk件に絞り、「アドレスを踏むサンプルのみ」を対象に
       simulate_traceで部分再評価する(サンプル方向のみ限定。ノード方向の
       部分再シミュレーションは行わず全ノードを再実行するが、行数を
       絞ることでコストを制御する — 設計判断は末尾コメント参照)
    5. fitの目的値が厳密に増加する場合のみ採択(タイ不採択)。valは報告のみ
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ir.schema import IRGraph
from ..ir.serialize import tt_from_hex, tt_to_hex
from ..ir.simulate import simulate_trace
from ..ir.validate import validate


@dataclass
class IrRepairReport:
    """IR flip修復の記録。"""

    fit_acc_before: float
    fit_acc_after: float
    val_acc_before: float
    val_acc_after: float
    n_flips: int = 0
    n_candidates: int = 0
    passes: int = 0
    per_pass: list[dict] = field(default_factory=list)


def _single_input_src(graph: IRGraph, node_id: str) -> str:
    """arity=1ノードの唯一の入力元idを返す。"""
    srcs = [e.src for e in graph.edges if e.dst == node_id]
    assert len(srcs) == 1, f'{node_id}: 入力エッジが1本ではない({len(srcs)}本)'
    return srcs[0]


def _compute_addr(h: np.ndarray, params: dict) -> np.ndarray:
    """lut_layerの各出力列oについてサンプルごとの訪問アドレスを計算する。

    simulate.pyと同一規則: addr = Σ_j x[indices[j][o]] << j
    """
    indices = np.asarray(params['indices'], dtype=np.int64)  # (n_inputs, out_width)
    n_inputs = int(params['n_inputs'])
    gathered = h[:, indices]  # (batch, n_inputs, out_width)
    weights = (1 << np.arange(n_inputs, dtype=np.int64))[None, :, None]
    return (gathered * weights).sum(axis=1).astype(np.int64)  # (batch, out_width)


def _target_setup(graph: IRGraph, objective: str) -> tuple[str, dict | None]:
    """objectiveに応じてtarget_id(目的値を読むノードid)とgroup_info(argmaxのみ)を決める。"""
    gs_nodes = [n for n in graph.nodes if n.type == 'group_sum']

    if objective == 'argmax':
        assert len(gs_nodes) == 1, (
            "objective='argmax'にはgroup_sumノードがちょうど1つ必要"
            f'(見つかった数: {len(gs_nodes)})'
        )
        gs = gs_nodes[0]
        k = int(gs.params['k'])
        in_width = int(gs.params['in_width'])
        assert in_width % k == 0
        group_info = {'k': k, 'group_size': in_width // k}
        return gs.id, group_info

    if objective == 'bits':
        if gs_nodes:
            assert len(gs_nodes) == 1, (
                f'group_sumノードが複数ある: {len(gs_nodes)}'
            )
            target_id = _single_input_src(graph, gs_nodes[0].id)
        else:
            target_id = graph.output_source()
            assert target_id is not None, (
                "objective='bits'は出力ポート接続が必要(group_sumが無い場合)"
            )
        return target_id, None

    raise ValueError(f"未知のobjective: {objective}(argmax/bitsのいずれか)")


def _contrib_from_trace(
    trace: dict, target_id: str, y: np.ndarray, objective: str,
) -> np.ndarray:
    """サンプルごとの目的達成量を返す(argmax: bool正解、bits: 一致ビット数)。"""
    out = trace[target_id]
    if objective == 'argmax':
        pred = out.argmax(axis=1)
        return pred == np.asarray(y)
    return (out == np.asarray(y)).sum(axis=1)


def _acc_from_contrib(contrib: np.ndarray, objective: str, width: int = 1) -> float:
    if objective == 'argmax':
        return float(contrib.mean())
    return float(contrib.sum() / (contrib.shape[0] * width))


def _eval_full(
    graph: IRGraph, x: np.ndarray, y: np.ndarray, objective: str, target_id: str,
) -> float:
    trace = simulate_trace(graph, x)
    contrib = _contrib_from_trace(trace, target_id, y, objective)
    width = trace[target_id].shape[1] if objective == 'bits' else 1
    return _acc_from_contrib(contrib, objective, width)


def _greedy_output_layer(
    node, addr: np.ndarray, trace: dict, contrib: np.ndarray,
    y: np.ndarray, objective: str, target_id: str, group_info: dict | None,
    budget: int,
) -> tuple[int, int]:
    """出力直結lut_layerの閉形式貪欲flip。戻り値: (採択flip数, 検討候補数)。

    contribは呼び出し元の配列を直接書き換える(採用行のみ更新)。
    """
    if budget <= 0:
        return 0, 0
    out_width = int(node.params['out_width'])
    tts = tt_from_hex(node.params['truth_tables'])
    col_addrs = [addr[:, o] for o in range(out_width)]
    col_uniques = [np.unique(c) for c in col_addrs]

    flips = 0
    n_considered = 0

    if objective == 'argmax':
        group_size = group_info['group_size']
        scores = np.asarray(trace[target_id], dtype=np.int64).copy()  # (N, k)
        pred = scores.argmax(axis=1)
        correct = pred == np.asarray(y)

        changed = True
        while changed and flips < budget:
            changed = False
            best = None  # (net, o, a, mask, delta, g, new_pred, new_correct)
            for o in range(out_width):
                col = col_addrs[o]
                g = o // group_size
                for a in col_uniques[o]:
                    a = int(a)
                    mask = col == a
                    n_considered += 1
                    bit = (tts[o] >> a) & 1
                    delta = 1 - 2 * bit
                    sub_scores = scores[mask].copy()
                    sub_scores[:, g] += delta
                    new_pred = sub_scores.argmax(axis=1)
                    new_correct = new_pred == np.asarray(y)[mask]
                    net = int(new_correct.sum()) - int(correct[mask].sum())
                    if net > 0 and (best is None or net > best[0]):
                        best = (net, o, a, mask, delta, g, new_pred, new_correct)
            if best is not None:
                _net, o, a, mask, delta, g, new_pred, new_correct = best
                tts[o] ^= (1 << a)
                scores[mask, g] += delta
                pred[mask] = new_pred
                correct[mask] = new_correct
                contrib[mask] = new_correct
                flips += 1
                changed = True
        node.params['truth_tables'] = tt_to_hex(tts)
        return flips, n_considered

    # objective == 'bits'
    bits = np.asarray(trace[node.id], dtype=np.int64).copy()  # (N, out_width)
    y_arr = np.asarray(y)
    match = bits == y_arr

    changed = True
    while changed and flips < budget:
        changed = False
        best = None  # (net, o, a, mask, new_bit, new_match_col)
        for o in range(out_width):
            col = col_addrs[o]
            for a in col_uniques[o]:
                a = int(a)
                mask = col == a
                n_considered += 1
                bit = (tts[o] >> a) & 1
                new_bit = 1 - bit
                new_match_col = new_bit == y_arr[mask, o]
                net = int(new_match_col.sum()) - int(match[mask, o].sum())
                if net > 0 and (best is None or net > best[0]):
                    best = (net, o, a, mask, new_bit, new_match_col)
        if best is not None:
            _net, o, a, mask, new_bit, new_match_col = best
            tts[o] ^= (1 << a)
            bits[mask, o] = new_bit
            match[mask, o] = new_match_col
            contrib[mask] = match[mask].sum(axis=1)
            flips += 1
            changed = True
    node.params['truth_tables'] = tt_to_hex(tts)
    return flips, n_considered


def _greedy_deep_layer(
    working: IRGraph, node, addr: np.ndarray, contrib: np.ndarray,
    x_fit: np.ndarray, y_fit: np.ndarray, objective: str, target_id: str,
    topk: int, budget: int,
) -> tuple[int, int]:
    """出力から離れたlut_layerの部分再シミュレーション貪欲flip。

    候補は「誤分類サンプルの訪問数」上位topkに限定する。各候補の効果は、
    そのアドレスを踏むサンプル(行)だけをsimulate_traceで再評価して厳密に
    計算する(ノード方向の部分再シムは行わず全ノードを再実行するが、
    行を絞ることでコストを抑える)。戻り値: (採択flip数, 検討候補数)。
    """
    if budget <= 0:
        return 0, 0
    out_width = int(node.params['out_width'])
    tts = tt_from_hex(node.params['truth_tables'])

    # 優先度付け用の誤分類判定: argmaxはbool正解の否定、bitsは「target_idの
    # 出力幅に満たない一致数」。nodeがtarget_idそのものでない場合(深い層)は
    # nodeのout_widthをtarget幅の代理として使う(保守的な優先度付けとして
    # 機能すれば十分。厳密な採否判定はsimulate_traceでの再評価が行う)
    if objective == 'argmax':
        misclassified = ~contrib.astype(bool)
    else:
        misclassified = contrib < out_width

    masks: dict[tuple[int, int], np.ndarray] = {}
    counts: list[tuple[int, int, int]] = []  # (count, o, a)
    for o in range(out_width):
        col = addr[:, o]
        for a in np.unique(col):
            a = int(a)
            mask = col == a
            masks[(o, a)] = mask
            counts.append((int((mask & misclassified).sum()), o, a))

    counts.sort(key=lambda t: -t[0])
    cand_list = counts[:topk]
    n_considered = len(cand_list)

    flips = 0
    changed = True
    while changed and flips < budget:
        changed = False
        best = None  # (net, o, a, mask, new_contrib_sub)
        for _cnt, o, a in cand_list:
            mask = masks[(o, a)]
            if not mask.any():
                continue
            old_tt_o = tts[o]
            tts[o] = old_tt_o ^ (1 << a)
            node.params['truth_tables'] = tt_to_hex(tts)

            sub_x = x_fit[mask]
            sub_y = y_fit[mask]
            sub_trace = simulate_trace(working, sub_x)
            new_contrib_sub = _contrib_from_trace(sub_trace, target_id, sub_y, objective)
            old_contrib_sub = contrib[mask]
            net = int(np.asarray(new_contrib_sub).astype(np.int64).sum()) - \
                int(np.asarray(old_contrib_sub).astype(np.int64).sum())

            tts[o] = old_tt_o
            node.params['truth_tables'] = tt_to_hex(tts)

            if net > 0 and (best is None or net > best[0]):
                best = (net, o, a, mask, new_contrib_sub)
        if best is not None:
            _net, o, a, mask, new_contrib_sub = best
            tts[o] ^= (1 << a)
            node.params['truth_tables'] = tt_to_hex(tts)
            contrib[mask] = new_contrib_sub
            flips += 1
            changed = True

    return flips, n_considered


def _count_initial_candidates(
    graph: IRGraph, target_nodes: list, x_fit: np.ndarray,
) -> int:
    """報告用: 修復開始時点での訪問アドレス候補の総数(1回のtraceから)。"""
    if not target_nodes:
        return 0
    trace = simulate_trace(graph, x_fit)
    total = 0
    for node in target_nodes:
        src_id = _single_input_src(graph, node.id)
        addr = _compute_addr(trace[src_id], node.params)
        for o in range(int(node.params['out_width'])):
            total += len(np.unique(addr[:, o]))
    return total


def repair_ir_flip(
    graph: IRGraph,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    objective: str = 'argmax',
    n_layers_from_output: int = 2,
    max_flips: int = 512,
    passes: int = 2,
    topk: int = 64,
) -> tuple[IRGraph, IrRepairReport]:
    """IRの真理値表を貪欲flipで修復する(graphは変更せずコピーを返す)。

    Args:
        graph: 修復対象のIR(検証済み想定)。変更しない
        x_fit: (N_fit, in_width) の0/1配列。採否判定に使う
        y_fit: objective='argmax'なら(N_fit,)のラベル整数列、
               'bits'なら(N_fit, core_out_width)の0/1ビット行列
        x_val: (N_val, in_width) の0/1配列。報告のみ(選択に使わない)
        y_val: x_valに対応する教師(形式はy_fitと同じ)
        objective: 'argmax' | 'bits'
        n_layers_from_output: 出力寄り何層のlut_layerを訪問対象にするか
        max_flips: 全体で許す最大flip数
        passes: 最大パス数(改善が無くなれば早期終了)
        topk: 深い層で厳密評価する候補数の上限

    Returns:
        (修復後のIRGraph(deepcopy)、IrRepairReport)
    """
    assert objective in ('argmax', 'bits'), f'未知のobjective: {objective}'

    working = copy.deepcopy(graph)
    target_id, group_info = _target_setup(working, objective)

    topo = working.topological_order()
    lut_nodes_topo = [n for n in topo if n.type == 'lut_layer']
    if n_layers_from_output > 0:
        target_nodes = lut_nodes_topo[-n_layers_from_output:]
    else:
        target_nodes = []
    # 出力に最も近い順(target_id自身/group_sumの直接入力から遠ざかる順)
    proximity_order = list(reversed(target_nodes))

    ids = {n.id for n in working.nodes}
    output_lut = None
    if target_id in ids and working.node_by_id(target_id).type == 'lut_layer':
        cand = working.node_by_id(target_id)
        if cand in target_nodes:
            output_lut = cand
    elif objective == 'argmax':
        # target_id はgroup_sum自身。その直接入力がlut_layerならoutput_lut候補
        src_id = _single_input_src(working, target_id)
        if src_id in ids and working.node_by_id(src_id).type == 'lut_layer':
            cand = working.node_by_id(src_id)
            if cand in target_nodes:
                output_lut = cand

    n_candidates = _count_initial_candidates(working, target_nodes, x_fit)

    fit_acc_before = _eval_full(working, x_fit, y_fit, objective, target_id)
    val_acc_before = _eval_full(working, x_val, y_val, objective, target_id)

    n_flips = 0
    per_pass: list[dict[str, Any]] = []

    for p in range(passes):
        if n_flips >= max_flips or not target_nodes:
            break
        flips_before_pass = n_flips

        # パス先頭で1回だけsimulate_trace(候補生成+contribベースライン)。
        # 上流ノードのflipで真の入力が変わりうるため、パスごとに鮮度を保つ
        # (モジュールdocstring末尾の設計判断コメント参照)
        trace = simulate_trace(working, x_fit)
        node_addr: dict[str, np.ndarray] = {}
        for node in target_nodes:
            src_id = _single_input_src(working, node.id)
            node_addr[node.id] = _compute_addr(trace[src_id], node.params)

        contrib = _contrib_from_trace(trace, target_id, y_fit, objective)

        if output_lut is not None:
            flips, _n = _greedy_output_layer(
                output_lut, node_addr[output_lut.id], trace, contrib,
                y_fit, objective, target_id, group_info,
                budget=max_flips - n_flips,
            )
            n_flips += flips

        for node in proximity_order:
            if node is output_lut:
                continue
            if n_flips >= max_flips:
                break
            flips, _n = _greedy_deep_layer(
                working, node, node_addr[node.id], contrib,
                x_fit, y_fit, objective, target_id, topk,
                budget=max_flips - n_flips,
            )
            n_flips += flips

        fit_acc_now = _eval_full(working, x_fit, y_fit, objective, target_id)
        per_pass.append({
            'pass': p,
            'fit_acc': fit_acc_now,
            'n_flips_pass': n_flips - flips_before_pass,
            'n_flips_cum': n_flips,
        })
        if n_flips == flips_before_pass:
            break

    fit_acc_after = _eval_full(working, x_fit, y_fit, objective, target_id)
    val_acc_after = _eval_full(working, x_val, y_val, objective, target_id)

    validate(working)

    report = IrRepairReport(
        fit_acc_before=fit_acc_before,
        fit_acc_after=fit_acc_after,
        val_acc_before=val_acc_before,
        val_acc_after=val_acc_after,
        n_flips=n_flips,
        n_candidates=n_candidates,
        passes=len(per_pass),
        per_pass=per_pass,
    )
    return working, report


# ============================================================
# 設計判断メモ(報告用)
# ============================================================
#
# 1. 「候補生成はsimulate_traceを1回実行」の解釈:
#    仕様書の記述を字義通り「呼び出し全体で1回」と実装すると、深い層の
#    flipで上流ノードの出力が変わった後に出力直結層の閉形式評価が
#    陳腐化したアドレス配列を使い続けるバグになる(パス2以降で誤ったflipを
#    行いうる)。本実装は「パスごとに1回」に緩め、パス内では
#    処理順(出力に近い層→遠い層)により対象ノードの入力が処理中に
#    変化しないことを保証する。これにより正確性と「候補生成をO(候補数)回
#    resimulateしない」という効率目標の両方を満たす。
#
# 2. 深い層の部分再シミュレーションは「ノード方向」ではなく「サンプル方向」
#    のみ絞る: candidate (o, a) の効果評価は、そのアドレスを踏む行だけを
#    simulate_trace(working, x_fit[mask]) で全ノード再実行して求める。
#    ノードL以降のみをトポロジ順に手動伝播する実装(必要なメモリ/バグ面の
#    複雑さが大きい)は見送り、行を絞ることで計算量を抑える設計とした。
#    候補プール自体もtopk(既定64、誤分類サンプルの訪問数で優先度付け)に
#    限定するため、実質コストは O(topk × |該当行| × グラフ評価) に収まる。
