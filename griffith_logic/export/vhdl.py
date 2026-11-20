"""IR → VHDL-2008 生成(v0.2でDAG対応)。

verilog.pyと同じ生成戦略のVHDL版:
    - LUT: 真理値表定数(std_logic_vector)をアドレス整数でインデックス。
      アドレスは {スロットn-1 & ... & スロット0}(MSB=最上位スロット、
      Verilog版と同一のビット規約)
    - group_sum: unsigned加算のpopcount(加算木はシンセサイザに委ねる)
    - シグナル名=サニタイズ済みnode id(w_<id>)。fanoutは同一シグナルの複数参照
    - concat: slotN-1 & ... & slot0 連結(&の左がMSB。IRのビット順規約:
      slot 0が下位ビット側)
    - add: or/xor=ビット演算。sumはvalidate(target='hdl')が事前に拒否
    - constant: ハードワイヤ定数(ビット文字列リテラル。bits[0]がLSB)
    - 純組み合わせ回路

出力: <name>.vhdl 単一ファイル(entity logic_core)
"""

import math
import re

from ..ir.schema import IRGraph
from ..ir.serialize import tt_from_hex
from ..ir.validate import node_out_width, validate


def _sanitize(node_id: str) -> str:
    """node idをHDL識別子に使える形へ変換する"""
    return re.sub(r'\W', '_', node_id)


def _tt_literal(tt: int, n_entries: int) -> str:
    """真理値表をstd_logic_vectorリテラル(降順ビット文字列)に変換"""
    bits = ''.join('1' if (tt >> k) & 1 else '0'
                   for k in reversed(range(n_entries)))
    return f'"{bits}"'


def _lut_layer_lines(node, src_sig: str, sig: str):
    p = node.params
    n = p['n_inputs']
    out_width = p['out_width']
    n_entries = 1 << n
    indices = p['indices']
    tables = tt_from_hex(p['truth_tables'])

    decls = [f'    signal {sig} : std_logic_vector({out_width - 1} downto 0);']
    consts = []
    body = [f'    -- {node.id}: lut_layer N={n}, out={out_width}']
    for o in range(out_width):
        cname = f'TT_{_sanitize(node.id)}_{o}'
        consts.append(
            f'    constant {cname} : std_logic_vector({n_entries - 1} downto 0)'
            f' := {_tt_literal(tables[o], n_entries)};'
        )
        # アドレス: スロットn-1 & ... & スロット0(&の左がMSB)
        addr = ' & '.join(
            f'{src_sig}({int(indices[j][o])})' for j in reversed(range(n))
        )
        if n == 1:
            addr_expr = f"unsigned'(\"\" & {addr})"
        else:
            addr_expr = f"unsigned(std_logic_vector'({addr}))"
        body.append(
            f'    {sig}({o}) <= {cname}(to_integer({addr_expr}));'
        )
    return decls, consts, body


def _group_sum_lines(node, src_sig: str, sig: str):
    p = node.params
    if any(int(v) != 0 for v in p.get('bias') or []):
        raise ValueError(
            f'{node.id}: group_sum の判定バイアス(bias)はVHDL生成未対応。'
            '比較器 + 定数は FPGA 側の終端部品で付けること'
        )
    k = p['k']
    in_width = p['in_width']
    group = in_width // k
    score_bits = max(1, math.ceil(math.log2(group + 1)))

    decls = [f'    signal {sig} : '
             f'std_logic_vector({k * score_bits - 1} downto 0);']
    body = [f'    -- {node.id}: group_sum k={k}, group={group} (popcount)']
    for c in range(k):
        terms = ' + '.join(
            f'resize(unsigned\'("" & {src_sig}({c * group + i})), {score_bits})'
            for i in range(group)
        )
        body.append(
            f'    {sig}({(c + 1) * score_bits - 1} downto {c * score_bits})'
            f' <= std_logic_vector({terms});'
        )
    return decls, body, score_bits


def _concat_lines(node, src_sigs: list, sig: str):
    out_width = int(sum(node.params['in_widths']))
    decls = [f'    signal {sig} : std_logic_vector({out_width - 1} downto 0);']
    # ビット順規約: slot 0が下位ビット側。&の左がMSBなのでslot降順
    parts = ' & '.join(reversed(src_sigs))
    body = [
        f'    -- {node.id}: concat widths={list(node.params["in_widths"])}'
        ' (slot0=下位ビット)',
        f'    {sig} <= {parts};',
    ]
    return decls, body


def _add_lines(node, src_sigs: list, sig: str):
    p = node.params
    op = {'or': ' or ', 'xor': ' xor '}[p['mode']]  # sumはvalidateで拒否済み
    decls = [f"    signal {sig} : std_logic_vector({p['width'] - 1} downto 0);"]
    body = [
        f"    -- {node.id}: add mode={p['mode']}, width={p['width']}",
        f'    {sig} <= {op.join(src_sigs)};',
    ]
    return decls, body


def _constant_lines(node, sig: str):
    p = node.params
    width = int(p['width'])
    # ビット順規約: bits[j]が出力ビットj。リテラルは左がMSBなので降順
    literal = ''.join(str(int(b)) for b in reversed(list(p['bits'])))
    decls = [f'    signal {sig} : std_logic_vector({width - 1} downto 0);']
    body = [
        f'    -- {node.id}: constant width={width} (ハードワイヤ定数)',
        f'    {sig} <= "{literal}";',
    ]
    return decls, body


def to_vhdl(graph: IRGraph, entity_name: str = 'logic_core') -> str:
    """IRからVHDL-2008ソース文字列を生成する。"""
    validate(graph, target='hdl')
    for n in graph.nodes:
        if n.type in ('conv2d', 'or_pool2d'):
            raise NotImplementedError(
                f'{n.id}: {n.type}ノードはHDL直接生成不可。'
                'export/lowering.lower_to_primitives(graph) を先に適用すること'
            )

    in_width = graph.inputs[0].width
    sig_of = {graph.inputs[0].id: 'x'}
    width_of = {graph.inputs[0].id: in_width}

    # dst_slot昇順の入力元マップ(fanout=同一シグナルの複数参照)
    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    in_srcs = {
        dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
        for dst, edges in grouped.items()
    }

    all_decls, all_consts, all_body = [], [], []
    last_sig, last_width = 'x', in_width
    score_bits = None
    for node in graph.topological_order():
        # constant等のarity 0ノードは入力エッジを持たない
        src_sigs = [sig_of[s] for s in in_srcs.get(node.id, [])]
        sig = f'w_{_sanitize(node.id)}'
        if sig in sig_of.values():
            raise ValueError(f'シグナル名が衝突している: {sig}({node.id})')
        consts = []
        if node.type == 'lut_layer':
            decls, consts, body = _lut_layer_lines(node, src_sigs[0], sig)
            width = node_out_width(node)
        elif node.type == 'group_sum':
            decls, body, score_bits = _group_sum_lines(node, src_sigs[0], sig)
            width = node.params['k'] * score_bits
        elif node.type == 'concat':
            decls, body = _concat_lines(node, src_sigs, sig)
            width = node_out_width(node)
        elif node.type == 'add':
            decls, body = _add_lines(node, src_sigs, sig)
            width = node_out_width(node)
        elif node.type == 'constant':
            decls, body = _constant_lines(node, sig)
            width = node_out_width(node)
        elif node.type in ('flatten', 'binarize'):
            sig_of[node.id] = src_sigs[0]
            width_of[node.id] = width_of[in_srcs[node.id][0]]
            continue
        else:
            raise NotImplementedError(f'VHDL未対応ノード型: {node.type}')
        all_decls.extend(decls)
        all_consts.extend(consts)
        all_body.extend(body + [''])
        sig_of[node.id] = sig
        width_of[node.id] = width
        last_sig, last_width = sig, width

    # 出力ポートに接続されたノードを優先(未接続なら最終トポノード)
    out_src = graph.output_source()
    if out_src is not None and out_src in sig_of:
        last_sig, last_width = sig_of[out_src], width_of[out_src]

    name = graph.metadata.get('name', 'model')
    lines = [
        f'-- griffith-logic 自動生成 (source: {name})',
        f'-- 入力: {in_width}bit / 出力: {last_width}bit'
        + (f' (score_bits={score_bits})' if score_bits else ''),
        'library ieee;',
        'use ieee.std_logic_1164.all;',
        'use ieee.numeric_std.all;',
        '',
        f'entity {entity_name} is',
        '    port (',
        f'        x : in  std_logic_vector({in_width - 1} downto 0);',
        f'        y : out std_logic_vector({last_width - 1} downto 0)',
        '    );',
        f'end entity {entity_name};',
        '',
        f'architecture rtl of {entity_name} is',
        *all_consts,
        *all_decls,
        'begin',
        *all_body,
        f'    y <= {last_sig};',
        'end architecture rtl;',
        '',
    ]
    return '\n'.join(lines)


def write_vhdl(graph: IRGraph, path: str,
               entity_name: str = 'logic_core') -> str:
    src = to_vhdl(graph, entity_name)
    # 生成HDLはUTF-8固定(Windowsの既定cp932に依存させない)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    return path
