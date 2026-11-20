"""IRのハードウェアリソース見積り。

HDL生成前のIRから、シンセサイザに依存しない一次見積りを出す:
    - lut_counts:  n_inputs別のLUT個数(lut_layerの各出力ニューロン=1 LUT)
    - total_luts:  LUT総数
    - logic_depth: 論理深さ=入力ポートから各ノードへの最長路上の
                   lut_layerノード数(concat/add/flatten/binarize等は0段。
                   lut_flat は中の平坦ネットリストの段数を足す)
    - wire_count:  論理信号線の本数(入力ポート幅+各ノードの論理出力幅の
                   総和。group_sumはk本と数える。fanoutは同一ワイヤの再利用
                   なので加算しない)
    - group_sums:  group_sumノード数(popcount加算木のコストは別勘定のため
                   個数のみ報告)
    - node_counts: ノード型別の個数(展開後)
    - lowered:     conv2d/or_pool2dをlower_to_primitivesで展開して計測したか

estimate_resources_with_conv_reduction: conv2d重み共有木の
lossless縮約(export/optimize.py)後のゲート数を reduced_gates として
併記するオプション関数。既存のestimate_resources(total_luts等)の意味・
呼び出しは変えない(素朴展開のゲート数のまま)。
"""

from ..ir.schema import IRGraph
from ..ir.validate import node_out_width


def _lut_flat_depth(params: dict) -> int:
    """`lut_flat` の論理段数(出力が参照する信号のゲート段の最大)。

    信号番号は 0 = 定数0、1..in_width = 入力、以降 = ゲート(トポロジカル順)。
    定数と入力は 0 段、ゲートは 1 + 入力の段の最大(ASAP)。出力 `outputs[i]` は
    `2*信号番号 + 極性` で、極性(NOT)は段に数えない。
    """
    in_w = int(params['in_width'])
    level = [0] * (1 + in_w)
    for g in params['gates']:
        a, b = int(g[1]), int(g[2])
        level.append(1 + max(level[a], level[b]))
    return max((level[int(o) >> 1] for o in params['outputs']), default=0)


def estimate_resources(graph: IRGraph) -> dict:
    """IRのリソース見積りを返す。

    conv2d/or_pool2dネイティブノードを含む場合はlower_to_primitivesで
    lut_layer列へ展開してから計測する(HDL生成後の実体と一致させる)。

    Args:
        graph: 検証済みIR

    Returns:
        {'total_luts', 'lut_counts', 'logic_depth', 'wire_count',
         'group_sums', 'node_counts', 'lowered'} のdict
    """
    from .lowering import LOWERABLE_TYPES, lower_to_primitives

    lowered = any(n.type in LOWERABLE_TYPES for n in graph.nodes)
    if lowered:
        graph = lower_to_primitives(graph)

    lut_counts: dict[int, int] = {}
    node_counts: dict[str, int] = {}
    wire_count = sum(p.width for p in graph.inputs)
    for n in graph.nodes:
        node_counts[n.type] = node_counts.get(n.type, 0) + 1
        wire_count += node_out_width(n)
        if n.type == 'lut_layer':
            key = int(n.params['n_inputs'])
            lut_counts[key] = lut_counts.get(key, 0) + int(n.params['out_width'])
        elif n.type == 'lut_flat':
            # 縮約済み平坦ネットリストの実効ゲート(2入力)数
            lut_counts[2] = lut_counts.get(2, 0) + len(n.params['gates'])

    # 論理深さ: 入力ポート(深さ0)からの最長路。lut_layerは1段、lut_flatは
    # 中のネットリストの段数(`_lut_flat_depth`)と数える
    in_srcs: dict[str, list[str]] = {}
    for e in graph.edges:
        in_srcs.setdefault(e.dst, []).append(e.src)
    depth: dict[str, int] = {p.id: 0 for p in graph.inputs}
    logic_depth = 0
    for node in graph.topological_order():
        d = max((depth.get(s, 0) for s in in_srcs.get(node.id, [])), default=0)
        if node.type == 'lut_layer':
            d += 1
        elif node.type == 'lut_flat':
            d += _lut_flat_depth(node.params)
        depth[node.id] = d
        logic_depth = max(logic_depth, d)

    return {
        'total_luts': sum(lut_counts.values()),
        'lut_counts': lut_counts,
        'logic_depth': logic_depth,
        'wire_count': wire_count,
        'group_sums': node_counts.get('group_sum', 0),
        'node_counts': node_counts,
        'lowered': lowered,
    }


def estimate_resources_with_conv_reduction(graph: IRGraph, *, verify: bool = True) -> dict:
    """estimate_resourcesの結果に、conv2d木縮約後ゲート数(reduced_gates)を併記する。

    estimate_resources自体はlower_to_primitivesで素朴展開したlut_layer列を
    数えるため(total_luts等は変えない)、conv2dの重み共有木に対する
    NOT吸収/定数畳み込み/CSE/dead除去(export/optimize.py、TS版
    convProgram.tsのPythonミラー)による削減後ゲート数は別勘定として
    reduced_gates に載せる。

    Args:
        graph: 検証済みIR(lowering前。conv2dノードを保持している必要がある)
        verify: reduce_conv_treeへ渡す(既定True。縮約の全数照合セルフチェック)

    Returns:
        estimate_resourcesの戻り値 + {'reduced_gates', 'reduced_gates_detail'}。
        graphにconv2dノードが無ければ reduced_gates=None、
        reduced_gates_detail=graph_conv_statsの戻り値(nodes=[]・total全0)。
    """
    from .optimize import graph_conv_stats

    result = estimate_resources(graph)
    conv_stats = graph_conv_stats(graph, verify=verify)
    result['reduced_gates'] = (
        conv_stats['total']['gates_after'] if conv_stats['nodes'] else None
    )
    result['reduced_gates_detail'] = conv_stats
    return result
