"""IR → コア Verilog 生成(安定命名。組み合わせ / パイプラインの2モード)。

ライブラリ `export/verilog.py` と**ビット同一挙動**であることが契約
(check の第4辺 = 生成器二重化 bit 一致で CI 検証する)。その上で本生成器は
  - 安定命名: wire 名 = `w_<IRノードid>`(サニタイズ済み)+ 名前マップ JSON
  - 観測点: 指定ノードの wire に (* keep *) / synthesis keep を付与
  - モジュール名可変(オーバーレイ/ラッパ統合用)
  - **pipelined=True**: 全ノード直後に FF(`w_<id>_c` = comb、`w_<id>` = reg)。
    II=1、latency = トポロジカル深さ。name_map の wire 名は不変のため、
    ビューアでは「値が 1clk ごとに層を進む」様子がそのまま見える
を生成時に組み込む(後付け計装は難路、という評価済み判断による)。

pipelined=False の出力は従来と byte-identical(target 変数の置換のみで実現。
スナップショットテストで固定)。

ビット順規約(IR と同一):
  - lut_layer: bit k = butterfly 基底順パターン k。アドレスは {slot n-1,...,slot 0}
  - concat: slot 0 が下位ビット側
  - group_sum: クラス c のスコアを score_bits 幅で c*score_bits..(c+1)*score_bits-1 に配置

パイプラインのスキュー均衡: 多入力ノード(concat/add)の入力段が揃わない場合、
浅い側に遅延レジスタ列 `w_<src>_d{k}` を挿入して到達段を揃える(メモ化)。
constant は時不変のため FF・遅延とも不要。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.ir.schema import IRGraph
from griffith_logic.ir.serialize import tt_from_hex
from griffith_logic.ir.validate import node_out_width, validate

# FF を挿入しないノード型(constant=時不変、flatten/binarize=恒等エイリアス)
_UNREGISTERED_TYPES = ('constant', 'flatten', 'binarize')


def sanitize(node_id: str) -> str:
    return re.sub(r'\W', '_', node_id)


@dataclass
class CoreGen:
    verilog: str
    module_name: str
    in_width: int
    out_width: int
    score_bits: int | None
    name_map: dict[str, dict] = field(default_factory=dict)
    pipelined: bool = False
    latency: int = 0          # 0 = 純組み合わせ

    @property
    def n_gates(self) -> int:
        return sum(m.get('n_gates', 0) for m in self.name_map.values())


def _compute_stages(graph: IRGraph, in_srcs: dict) -> dict[str, int | None]:
    """パイプライン段(FF 通過数)。constant は None(時不変)。"""
    types = {n.id: n.type for n in graph.nodes}
    stage: dict[str, int | None] = {graph.inputs[0].id: 0}
    for node in graph.topological_order():
        srcs = in_srcs.get(node.id, [])
        src_stages = [stage[s] for s in srcs if stage.get(s) is not None]
        if node.type == 'constant':
            stage[node.id] = None
        elif node.type in ('flatten', 'binarize'):
            stage[node.id] = src_stages[0] if src_stages else 0
        else:
            stage[node.id] = (max(src_stages) if src_stages else 0) + 1
    # 出力に届かない constant のみのグラフ等は考えない(validate 済み前提)
    _ = types
    return stage


def generate_core(graph: IRGraph, module_name: str = 'mdl_core',
                  observe: tuple[str, ...] = (),
                  pipelined: bool = False) -> CoreGen:
    """IR からコア Verilog を生成する(conv/pool は内部で lowering)。"""
    graph = lower_to_primitives(graph)
    validate(graph, target='hdl')

    in_width = graph.inputs[0].width
    input_id = graph.inputs[0].id
    wire_of: dict[str, str] = {input_id: 'x'}
    width_of: dict[str, int] = {input_id: in_width}
    name_map: dict[str, dict] = {
        input_id: {'wire': 'x', 'width': in_width, 'type': 'input'},
    }

    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    in_srcs = {
        dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
        for dst, edges in grouped.items()
    }

    stage_of: dict[str, int | None] = {}
    if pipelined:
        stage_of = _compute_stages(graph, in_srcs)
        name_map[input_id]['stage'] = 0

    body: list[str] = []
    last_wire, last_width = 'x', in_width
    score_bits: int | None = None

    def decl(wire: str, width: int, keep: bool) -> str:
        attr = '(* keep *) ' if keep else ''
        note = '  /* synthesis keep */' if keep else ''
        return f'    {attr}wire [{width - 1}:0] {wire};{note}'

    def decl_reg(wire: str, width: int, keep: bool) -> str:
        attr = '(* keep *) ' if keep else ''
        note = '  /* synthesis keep */' if keep else ''
        return (f'    {attr}reg [{width - 1}:0] {wire} = '
                f'{{{width}{{1\'b0}}}};{note}')

    # スキュー均衡ディレイ(メモ化): (src_id, k) → wire 名
    delayed: dict[tuple[str, int], str] = {}

    def get_delayed(src_id: str, need_stage: int) -> str:
        """src の値を need_stage 段目で使えるワイヤ名を返す(必要なら遅延挿入)。"""
        s = stage_of.get(src_id)
        if s is None:                       # constant: 時不変
            return wire_of[src_id]
        k = need_stage - s
        assert k >= 0, (src_id, s, need_stage)
        if k == 0:
            return wire_of[src_id]
        if (src_id, k) in delayed:
            return delayed[(src_id, k)]
        prev = get_delayed(src_id, need_stage - 1) if k > 1 \
            else wire_of[src_id]
        w = width_of[src_id]
        name = f'{wire_of[src_id]}_d{k}'
        body.append(f'    // スキュー均衡ディレイ: {src_id} +{k}clk')
        body.append(decl_reg(name, w, keep=False))
        body.append(f'    always @(posedge clk) {name} <= {prev};')
        body.append('')
        delayed[(src_id, k)] = name
        return name

    for node in graph.topological_order():
        srcs = in_srcs.get(node.id, [])
        wire = f'w_{sanitize(node.id)}'
        if wire in wire_of.values():
            raise ValueError(f'ワイヤ名が衝突: {wire}({node.id})')
        keep = node.id in observe
        entry: dict = {'wire': wire, 'type': node.type, 'observed': keep}

        registered = pipelined and node.type not in _UNREGISTERED_TYPES
        target = f'{wire}_c' if registered else wire
        if registered:
            # 入力は「自段-1」で揃える(スキュー均衡)
            need = stage_of[node.id] - 1
            src_wires = [get_delayed(s, need) for s in srcs]
        else:
            src_wires = [wire_of[s] for s in srcs]

        if node.type == 'lut_layer':
            p = node.params
            n = p['n_inputs']
            out_w = p['out_width']
            indices = p['indices']
            tables = tt_from_hex(p['truth_tables'])
            body.append(f'    // {node.id}: lut_layer N={n}, out={out_w}')
            body.append(decl(target, out_w, keep and not registered))
            n_entries = 1 << n
            for o in range(out_w):
                addr = ', '.join(
                    f'{src_wires[0]}[{int(indices[j][o])}]'
                    for j in reversed(range(n)))
                body.append(
                    f"    assign {target}[{o}] = "
                    f"{n_entries}'h{tables[o]:x} >> {{{addr}}};")
            width = out_w
            entry['n_gates'] = out_w
        elif node.type == 'group_sum':
            p = node.params
            k = p['k']
            group = p['in_width'] // k
            score_bits = max(1, math.ceil(math.log2(group + 1)))
            width = k * score_bits
            body.append(
                f'    // {node.id}: group_sum k={k}, group={group} (popcount, '
                f'score_bits={score_bits})')
            body.append(decl(target, width, keep and not registered))
            # 1 式あたりの加算項数上限。数百項の単一式は Windows 版 yosys の
            # Verilog フロントエンドをスタックオーバーフローさせる実測があるため
            # (mnist24k group=400、0xC00000FD)、超える場合は部分和 wire に分割
            # する。popcount 意味論では全ての途中値が総和以下(< 2^score_bits)
            # なので score_bits 幅の部分和は bit-exact(presim/照合(c)辺で検証)
            gs_chunk = 64
            for c in range(k):
                if score_bits == 1:
                    terms_list = [f'{src_wires[0]}[{c * group}]']
                else:
                    terms_list = [
                        f"{{{score_bits - 1}'b0, {src_wires[0]}[{c * group + i}]}}"
                        for i in range(group)]
                level = 0
                while len(terms_list) > gs_chunk:
                    parts = []
                    for j in range(0, len(terms_list), gs_chunk):
                        pname = f'{target}_c{c}_l{level}p{len(parts)}'
                        body.append(
                            f'    wire [{score_bits - 1}:0] {pname} = '
                            + ' + '.join(terms_list[j:j + gs_chunk]) + ';')
                        parts.append(pname)
                    terms_list = parts
                    level += 1
                body.append(
                    f'    assign {target}[{(c + 1) * score_bits - 1}'
                    f':{c * score_bits}] = ' + ' + '.join(terms_list) + ';')
            entry['score_bits'] = score_bits
        elif node.type == 'concat':
            width = int(sum(node.params['in_widths']))
            parts = ', '.join(reversed(src_wires))  # slot0=下位ビット
            body.append(f'    // {node.id}: concat (slot0=下位ビット)')
            body.append(decl(target, width, keep and not registered))
            body.append(f'    assign {target} = {{{parts}}};')
        elif node.type == 'add':
            p = node.params
            op = {'or': ' | ', 'xor': ' ^ '}[p['mode']]  # sumはvalidateで拒否済み
            width = p['width']
            body.append(f"    // {node.id}: add mode={p['mode']}")
            body.append(decl(target, width, keep and not registered))
            body.append(f'    assign {target} = {op.join(src_wires)};')
        elif node.type == 'constant':
            p = node.params
            width = int(p['width'])
            literal = ''.join(str(int(b)) for b in reversed(list(p['bits'])))
            body.append(f'    // {node.id}: constant')
            body.append(decl(target, width, keep))
            body.append(f"    assign {target} = {width}'b{literal};")
        elif node.type in ('flatten', 'binarize'):
            # 論理領域内では恒等。ワイヤを増やさず別名にする
            wire_of[node.id] = src_wires[0]
            width_of[node.id] = width_of[in_srcs[node.id][0]]
            name_map[node.id] = {'wire': src_wires[0], 'type': node.type,
                                 'width': width_of[node.id], 'alias': True}
            if pipelined:
                name_map[node.id]['stage'] = stage_of[node.id]
            continue
        else:
            raise NotImplementedError(
                f'HDL 未対応ノード型: {node.type}({node.id})。'
                'v1 は順伝播プリミティブ(lut_layer/group_sum/concat/add/'
                'constant/flatten/binarize + conv lowering)のみ対応')

        if registered:
            body.append(decl_reg(wire, width, keep))
            body.append(f'    always @(posedge clk) {wire} <= {target};')

        body.append('')
        entry['width'] = width
        if pipelined:
            entry['stage'] = stage_of[node.id]
        wire_of[node.id] = wire
        width_of[node.id] = width
        name_map[node.id] = entry
        last_wire, last_width = wire, width

    out_src = graph.output_source()
    if out_src is not None and out_src in wire_of:
        last_wire, last_width = wire_of[out_src], width_of[out_src]

    latency = 0
    if pipelined:
        out_id = out_src if out_src in wire_of else \
            graph.topological_order()[-1].id
        latency = int(stage_of.get(out_id) or 0)

    name = graph.metadata.get('name', 'model')
    if pipelined:
        header = [
            f'// mdl-fpga 自動生成コア (source IR: {name})',
            f'// 入力: {in_width}bit / 出力: {last_width}bit'
            + (f' (score_bits={score_bits})' if score_bits else ''),
            f'// 純パイプライン(全ノード直後に FF、II=1)。latency={latency}clk',
            f'module {module_name} (',
            '    input  wire clk,',
            f'    input  wire [{in_width - 1}:0] x,',
            f'    output wire [{last_width - 1}:0] y',
            ');',
            '',
        ]
    else:
        header = [
            f'// mdl-fpga 自動生成コア (source IR: {name})',
            f'// 入力: {in_width}bit / 出力: {last_width}bit'
            + (f' (score_bits={score_bits})' if score_bits else ''),
            '// 純組み合わせ。挙動はライブラリ export/verilog.py と bit 一致(check 第4辺)',
            f'module {module_name} (',
            f'    input  wire [{in_width - 1}:0] x,',
            f'    output wire [{last_width - 1}:0] y',
            ');',
            '',
        ]
    footer = [f'    assign y = {last_wire};', '', 'endmodule', '']
    return CoreGen(
        verilog='\n'.join(header + body + footer),
        module_name=module_name,
        in_width=in_width,
        out_width=last_width,
        score_bits=score_bits,
        name_map=name_map,
        pipelined=pipelined,
        latency=latency,
    )
