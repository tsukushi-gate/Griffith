"""活性率・トグル率・動的電力proxy(SM5)。

`ir/simulate.py simulate_trace` でデータセットを流し、全ノード出力ビットの
統計を取る(numpyのみ・バッチ分割で省メモリ)。

指標(構造メトリクスの設計方針):
    - 活性率 p_i: ビットiが1になる率
    - トグルproxy 2·p_i·(1−p_i): i.i.d.サンプル間の期待スイッチ率。
      入力順序に依存しない標準指標(実系列トグルは sequential=True で別途)
    - 動的電力proxy Σ_i toggle_i × fanout_i: スイッチが伝播する配線量の重み付け
    - データ上定数率: p∈{0,1} のビット率。**構造的dead(SM2 sweep)とは別物**
      (訓練データ分布上でたまたま動かない=don't care側の情報)

対象は二値信号ノードのみ(group_sum の整数スコア、add mode='sum' は除外)。
"""

import numpy as np

from ..ir.validate import node_in_width, node_out_width

# 出力が{0,1}に収まるノード型(add はmode='sum'のみ除外)
_BINARY_TYPES = ('lut_layer', 'binarize', 'flatten', 'concat', 'add', 'constant')


def _binary_node_ids(graph) -> list[str]:
    ids = []
    for n in graph.nodes:
        if n.type not in _BINARY_TYPES:
            continue
        if n.type == 'add' and n.params.get('mode') == 'sum':
            continue
        ids.append(n.id)
    return ids


def ir_bit_fanout(graph) -> dict[str, np.ndarray]:
    """各ノード(+入力ポート)の出力ビット別ファンアウトを返す。

    lut_layer消費者は indices の参照回数(スロット重複込み)、
    位置参照ノード(group_sum/concat/add/binarize/flatten)・出力ポートは
    各ビット1回として数える。
    """
    nodes = {n.id: n for n in graph.nodes}
    out_ids = {p.id for p in graph.outputs}
    fanout: dict[str, np.ndarray] = {
        n.id: np.zeros(node_out_width(n), dtype=np.int64) for n in graph.nodes
    }
    for p in graph.inputs:
        fanout[p.id] = np.zeros(p.width, dtype=np.int64)

    for e in graph.edges:
        if e.src not in fanout:
            continue
        if e.dst in out_ids:
            fanout[e.src][:] += 1
            continue
        dst = nodes.get(e.dst)
        if dst is None:
            continue
        if dst.type == 'lut_layer':
            idx = np.asarray(dst.params['indices'], dtype=np.int64)
            fanout[e.src] += np.bincount(
                idx.ravel(), minlength=fanout[e.src].size,
            )
        elif dst.type == 'constant':
            continue
        else:
            w = min(node_in_width(dst, e.dst_slot), fanout[e.src].size)
            fanout[e.src][:w] += 1
    return fanout


def activity_stats(
    graph,
    x: np.ndarray,
    batch_size: int = 4096,
    sequential: bool = False,
) -> dict:
    """データセットxを流して活性・トグル・電力proxyを計測する。

    Args:
        graph: IRGraph(conv系はsimulate側で自動lowering)
        x: (num_samples, in_width) の0/1配列
        batch_size: simulateのバッチ分割サイズ
        sequential: Trueならxの行順を実系列とみなし、隣接サンプル間の
            実測トグル率も併せて返す(順序に意味がある場合のみ)

    Returns:
        {'n_samples',
         'per_node': {id: {'act_mean', 'toggle_mean', 'const_frac',
                           'power_proxy', ('seq_toggle_mean')}},
         'totals': {'power_proxy', 'toggle_mean', 'data_const_frac'}}
        いずれもJSON化可能(値はPython float/int)。
    """
    from ..ir.simulate import simulate_trace

    x = np.asarray(x)
    n_samples = x.shape[0]
    if n_samples == 0:
        raise ValueError('サンプルが空')

    node_ids = _binary_node_ids(graph)
    ones: dict[str, np.ndarray] = {}
    seq_flips: dict[str, np.ndarray] = {}
    prev_last: dict[str, np.ndarray] = {}

    for start in range(0, n_samples, batch_size):
        trace = simulate_trace(graph, x[start:start + batch_size])
        for nid in node_ids:
            v = trace[nid]
            ones[nid] = ones.get(nid, 0) + v.sum(axis=0)
            if sequential:
                flips = (np.diff(v, axis=0) != 0).sum(axis=0)
                if nid in prev_last:
                    flips = flips + (v[0] != prev_last[nid]).astype(np.int64)
                seq_flips[nid] = seq_flips.get(nid, 0) + flips
                prev_last[nid] = v[-1]

    fanout = ir_bit_fanout(graph)
    per_node: dict[str, dict] = {}
    power_total = 0.0
    toggle_sum = 0.0
    const_sum = 0
    bit_total = 0
    for nid in node_ids:
        p = ones[nid] / n_samples
        toggle = 2.0 * p * (1.0 - p)
        const = (p == 0.0) | (p == 1.0)
        power = float((toggle * fanout[nid]).sum())
        entry = {
            'act_mean': float(p.mean()),
            'toggle_mean': float(toggle.mean()),
            'const_frac': float(const.mean()),
            'power_proxy': power,
        }
        if sequential:
            entry['seq_toggle_mean'] = float(
                (seq_flips[nid] / max(n_samples - 1, 1)).mean()
            )
        per_node[nid] = entry
        power_total += power
        toggle_sum += float(toggle.sum())
        const_sum += int(const.sum())
        bit_total += int(p.size)

    return {
        'n_samples': int(n_samples),
        'per_node': per_node,
        'totals': {
            'power_proxy': power_total,
            'toggle_mean': toggle_sum / bit_total if bit_total else 0.0,
            'data_const_frac': const_sum / bit_total if bit_total else 0.0,
        },
    }
