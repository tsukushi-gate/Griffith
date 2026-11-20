"""IR → Verilog-2001 生成(v0.2でDAG対応)。

生成戦略:
    - ベンダ非依存: LUTは「真理値表リテラルの右シフト+LSBマスク」で表現。
      Quartus/Vivadoのシンセサイザが自動でLUTプリミティブにマップする。
          assign l1[5] = TT >> {x[i2], x[i1], x[i0]};  (LSB=スロット0)
      ビット順はIRの規約(bit k = butterfly基底順パターンk)を厳守
    - group_sum: 1bit線の整数加算でpopcount(加算木はシンセサイザに委ねる)。
      tauは整数argmaxに不変なので回路では無視し、生スコアを出力
    - ワイヤ名=サニタイズ済みnode id(w_<id>)。fanoutは同一ワイヤの複数参照
    - concat: {slotN-1, ..., slot1, slot0} 連結。IRのビット順規約
      (slot 0が下位ビット側)に合わせ、Verilogの{}は左がMSBなので
      slot降順に並べる
    - add: or/xor=ビット演算。sumは幅拡張なしで表現不能のため
      validate(target='hdl')が事前に拒否する
    - constant: ハードワイヤ定数(width'b リテラル。bits[0]がLSB)
    - 純組み合わせ回路(pipeline化は次段階)

出力: <name>.v 単一ファイル(topモジュール)
"""

import math
import re

from ..ir.schema import IRGraph
from ..ir.serialize import tt_from_hex
from ..ir.validate import node_out_width, validate


def _sanitize(node_id: str) -> str:
    """node idをHDL識別子に使える形へ変換する"""
    return re.sub(r'\W', '_', node_id)


def _lut_layer_lines(node, src_wire: str, wire: str) -> list[str]:
    p = node.params
    n = p['n_inputs']
    out_width = p['out_width']
    n_entries = 1 << n
    indices = p['indices']
    tables = tt_from_hex(p['truth_tables'])

    lines = [f'    // {node.id}: lut_layer N={n}, out={out_width}',
             f'    wire [{out_width - 1}:0] {wire};']
    for o in range(out_width):
        # アドレス: {スロットn-1, ..., スロット0}(連結はMSB→LSB順)
        addr = ', '.join(
            f'{src_wire}[{int(indices[j][o])}]' for j in reversed(range(n))
        )
        tt = tables[o]
        lines.append(
            f"    assign {wire}[{o}] = {n_entries}'h{tt:x} >> {{{addr}}};"
        )
    return lines


def _group_sum_lines(node, src_wire: str, wire: str) -> tuple[list[str], int]:
    p = node.params
    k = p['k']
    in_width = p['in_width']
    group = in_width // k
    score_bits = max(1, math.ceil(math.log2(group + 1)))

    lines = [f'    // {node.id}: group_sum k={k}, group={group} (popcount)']
    lines.append(f'    wire [{k * score_bits - 1}:0] {wire};')
    for c in range(k):
        if score_bits == 1:
            terms = f'{src_wire}[{c * group}]'
        else:
            terms = ' + '.join(
                f"{{{score_bits - 1}'b0, {src_wire}[{c * group + i}]}}"
                for i in range(group)
            )
        lines.append(
            f'    assign {wire}[{(c + 1) * score_bits - 1}:{c * score_bits}]'
            f' = {terms};'
        )
    return lines, score_bits


def _concat_lines(node, src_wires: list[str], wire: str) -> list[str]:
    out_width = int(sum(node.params['in_widths']))
    # ビット順規約: slot 0が下位ビット側。Verilogの{}は左がMSBなのでslot降順
    parts = ', '.join(reversed(src_wires))
    return [
        f'    // {node.id}: concat widths={list(node.params["in_widths"])}'
        ' (slot0=下位ビット)',
        f'    wire [{out_width - 1}:0] {wire};',
        f'    assign {wire} = {{{parts}}};',
    ]


def _add_lines(node, src_wires: list[str], wire: str) -> list[str]:
    p = node.params
    op = {'or': ' | ', 'xor': ' ^ '}[p['mode']]  # sumはvalidateで拒否済み
    return [
        f"    // {node.id}: add mode={p['mode']}, width={p['width']}",
        f"    wire [{p['width'] - 1}:0] {wire};",
        f'    assign {wire} = {op.join(src_wires)};',
    ]


def _constant_lines(node, wire: str) -> list[str]:
    p = node.params
    width = int(p['width'])
    # ビット順規約: bits[j]が出力ビットj。Verilogリテラルは左がMSBなので降順
    literal = ''.join(str(int(b)) for b in reversed(list(p['bits'])))
    return [
        f'    // {node.id}: constant width={width} (ハードワイヤ定数)',
        f'    wire [{width - 1}:0] {wire};',
        f"    assign {wire} = {width}'b{literal};",
    ]


def to_verilog(graph: IRGraph, module_name: str = 'logic_core') -> str:
    """IRからVerilogソース文字列を生成する。

    出力ポート:
        group_sumで終わる場合: y = kクラス × score_bits の連結スコア
        それ以外: 最終レイヤのビット列
    """
    validate(graph, target='hdl')
    for n in graph.nodes:
        if n.type in ('conv2d', 'or_pool2d'):
            raise NotImplementedError(
                f'{n.id}: {n.type}ノードはHDL直接生成不可。'
                'export/lowering.lower_to_primitives(graph) を先に適用すること'
            )

    in_width = graph.inputs[0].width
    body: list[str] = []
    wire_of = {graph.inputs[0].id: 'x'}
    width_of = {graph.inputs[0].id: in_width}

    # dst_slot昇順の入力元マップ(fanout=同一ワイヤの複数参照)
    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    in_srcs = {
        dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
        for dst, edges in grouped.items()
    }

    last_wire, last_width = 'x', in_width
    score_bits = None
    for node in graph.topological_order():
        # constant等のarity 0ノードは入力エッジを持たない
        src_wires = [wire_of[s] for s in in_srcs.get(node.id, [])]
        wire = f'w_{_sanitize(node.id)}'
        if wire in wire_of.values():
            raise ValueError(f'ワイヤ名が衝突している: {wire}({node.id})')
        if node.type == 'lut_layer':
            lines = _lut_layer_lines(node, src_wires[0], wire)
        elif node.type == 'group_sum':
            lines, score_bits = _group_sum_lines(node, src_wires[0], wire)
        elif node.type == 'concat':
            lines = _concat_lines(node, src_wires, wire)
        elif node.type == 'add':
            lines = _add_lines(node, src_wires, wire)
        elif node.type == 'constant':
            lines = _constant_lines(node, wire)
        elif node.type == 'flatten':
            wire_of[node.id] = src_wires[0]
            width_of[node.id] = width_of[in_srcs[node.id][0]]
            continue
        elif node.type == 'binarize':
            # 論理領域内では恒等(入力は既に0/1)
            wire_of[node.id] = src_wires[0]
            width_of[node.id] = width_of[in_srcs[node.id][0]]
            continue
        else:
            raise NotImplementedError(f'HDL未対応ノード型: {node.type}')
        width = node_out_width(node) if node.type != 'group_sum' \
            else node.params['k'] * score_bits
        body.extend(lines)
        body.append('')
        wire_of[node.id] = wire
        width_of[node.id] = width
        last_wire, last_width = wire, width

    # 出力ポートに接続されたノードを優先(未接続なら最終トポノード)
    out_src = graph.output_source()
    if out_src is not None and out_src in wire_of:
        last_wire, last_width = wire_of[out_src], width_of[out_src]

    name = graph.metadata.get('name', 'model')
    header = [
        f'// griffith-logic 自動生成 (source: {name})',
        f'// 入力: {in_width}bit / 出力: {last_width}bit'
        + (f' (score_bits={score_bits})' if score_bits else ''),
        f'module {module_name} (',
        f'    input  wire [{in_width - 1}:0] x,',
        f'    output wire [{last_width - 1}:0] y',
        ');',
        '',
    ]
    footer = [
        f'    assign y = {last_wire};',
        '',
        'endmodule',
        '',
    ]
    return '\n'.join(header + body + footer)


def write_verilog(graph: IRGraph, path: str,
                  module_name: str = 'logic_core') -> str:
    src = to_verilog(graph, module_name)
    # 生成HDLはUTF-8固定(Windowsの既定cp932に依存させない)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    return path
