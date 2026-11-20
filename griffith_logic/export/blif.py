"""IR(lut_layer/concat)⇔BLIF変換(compile-pipeline)。

対象は `x(入力ポート) → lut_layer/concat鎖 → group_sum → y` の純鎖IR
(decompose_mlp等の出力)。BLIF化するのはgroup_sum手前のlut_layer部分のみ
(group_sumは整数popcountでBLIF(論理ゲート網)の意味論に乗らないため)。

    split_lut_core   IR → (coreグラフ, 切り離したgroup_sumノード)
    write_blif       coreグラフ(lut_layer/concatのみ) → .blifファイル
    read_blif        .blifファイル → BlifNetwork(素朴なパース結果)
    blif_to_ir       BlifNetwork → IR(lut_layer/concat列。ASAP levelize+
                     concatバスでDAGを層化)
    reattach_group_sum  coreグラフ + group_sumノード → 元と同型のIR

ビット順規約(simulate.pyと完全統一):
    - lut_layerのaddr = Σ_j x[indices[j,o]] << j(スロットjがアドレスのbit j)
    - `.names` の行: ヘッダ位置j(左からj番目のfanin) = スロットj。
      cube文字位置j = アドレスkのbit j。ヘッダとcubeの位置対応が
      そのままスロット対応になるため、書き出し/読み込みで反転処理は不要
      (ただし逆順に書いてしまう実装ミスは非対称な真理値表で顕在化する)
    - concat: slot 0が出力ビット列の先頭(下位ビット側)

blif_to_irの層化(ASAP levelize):
    1. 各`.names`出力信号のlevelを「faninの最大level+1」で確定
       (プライマリ入力はlevel 0)
    2. 各primary outputのドライバは専用の最終レベル(既存最大level+1)へ
       *常に複製*する。これにより最終lut_layerの列順=y_0..y_{W-1}が
       保証され、内部fanoutを持つドライバも複製元は元のlevelに残るので
       壊れない
    3. 層をまたぐ配線(level Lが level<L-1 の信号を参照)は、恒等LUTを
       挿入せず「concatバス」で解決する: bus_0=x、level L処理後
       bus_{L+1}=concat(bus_L, level L出力)。各levelのlut_layerは
       bus_L全体を入力とし、indicesはバス内オフセットを直接指す
    4. 層内fan-inパディング: layer内 n_inputs = 実fan-inの最大値。
       fan-inが少ないノードはslot0を複製し、pad_ttで
       「パディングスロットの値に依存しない」よう真理値表を拡張する。
       定数ノード(fan-in 0)はfan-in 1(任意の既存信号、x_0を使う)へ
       正規化してから同じ処理に載せる
"""

import os
from dataclasses import dataclass

import numpy as np

from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.serialize import tt_from_hex, tt_to_hex
from ..ir.validate import validate

# ============================================================
# split_lut_core / reattach_group_sum
# ============================================================

def split_lut_core(graph: IRGraph) -> tuple[IRGraph, IRNode]:
    """group_sum手前までのlut_layer/concat純部分(core)を切り出す。

    対象IRは x → lut_layer/concat鎖 → group_sum → y の純鎖を仮定する。
    core側にlut_layer/concat以外のノード型が現れたらAssertionError。
    core出力の幅・ビット順はgroup_sumの入力と同一に保つ。

    Returns:
        (core_graph, group_sum_node)。core_graphの出力ポートは
        id='core_out', width=group_sumのin_width
    """
    validate(graph)
    assert len(graph.outputs) == 1, 'split_lut_coreは単一出力ポートのみ対応'

    gs_nodes = [n for n in graph.nodes if n.type == 'group_sum']
    assert len(gs_nodes) <= 1, (
        f'group_sumノードが2個以上ある(現在{len(gs_nodes)}個)'
    )
    if not gs_nodes:
        # group_sum無し(bits系: core出力=グラフ出力)はそのままcoreとして扱う
        for n in graph.nodes:
            assert n.type in ('lut_layer', 'concat'), (
                f'{n.id}: coreにlut_layer/concat以外のノード型が含まれる'
                f'(type={n.type})'
            )
        return graph, None
    gs_node = gs_nodes[0]
    for n in graph.nodes:
        if n.id == gs_node.id:
            continue
        assert n.type in ('lut_layer', 'concat'), (
            f'{n.id}: coreにlut_layer/concat以外のノード型が含まれる'
            f'(type={n.type})'
        )

    in_edges = [e for e in graph.edges if e.dst == gs_node.id]
    assert len(in_edges) == 1, (
        f'group_sumの入力エッジが1本でない(現在{len(in_edges)}本)'
    )
    core_src = in_edges[0].src
    core_width = int(gs_node.params['in_width'])

    out_port_ids = {p.id for p in graph.outputs}
    core_nodes = [n for n in graph.nodes if n.id != gs_node.id]
    core_edges = [
        e for e in graph.edges
        if e.dst != gs_node.id and e.dst not in out_port_ids
    ]
    core_out = IRPort(id='core_out', width=core_width)
    core_edges.append(IREdge(src=core_src, dst=core_out.id))

    core_graph = IRGraph(
        metadata={**graph.metadata, 'split_from': 'split_lut_core'},
        inputs=list(graph.inputs),
        outputs=[core_out],
        nodes=core_nodes,
        edges=core_edges,
    )
    validate(core_graph)
    return core_graph, gs_node


def reattach_group_sum(core_graph: IRGraph, group_sum_node: IRNode) -> IRGraph:
    """coreグラフへgroup_sum+出力ポートを再接続し、元と同型のIRへ戻す。

    core_graphの出力ポート幅はgroup_sum_node.params['in_width']と
    一致していなければならない(split_lut_coreの逆操作)。
    group_sum_node=None(group_sum無しグラフのsplit結果)の場合は
    core_graphをそのまま返す。
    """
    if group_sum_node is None:
        return core_graph
    assert len(core_graph.outputs) == 1, (
        'reattach_group_sumは単一出力ポートのみ対応'
    )
    out_port = core_graph.outputs[0]
    assert out_port.width == group_sum_node.params['in_width'], (
        f'core出力幅({out_port.width}) != '
        f"group_sumのin_width({group_sum_node.params['in_width']})"
    )
    core_src = core_graph.output_source()
    assert core_src is not None, 'core_graphの出力が未接続'

    edges = [e for e in core_graph.edges if e.dst != out_port.id]
    edges.append(IREdge(src=core_src, dst=group_sum_node.id))
    edges.append(IREdge(src=group_sum_node.id, dst='y'))

    y_port = IRPort(id='y', width=int(group_sum_node.params['k']))
    graph = IRGraph(
        metadata={**core_graph.metadata},
        inputs=list(core_graph.inputs),
        outputs=[y_port],
        nodes=[*core_graph.nodes, group_sum_node],
        edges=edges,
    )
    validate(graph)
    return graph


# ============================================================
# write_blif
# ============================================================

def write_blif(graph: IRGraph, path: str, *, model_name: str = 'core') -> dict:
    """lut_layer/concatのみのIRをBLIF形式で書き出す。

    信号名: 入力 x_{j}(全入力ポートを順に連結)、各LUT出力
    {node_id}__{o}、primary output y_{i}(グラフ出力ポートのビット順)。
    POは内部信号のバッファ(`.names 内部信号 y_i` / `1 1`)で固定する。

    Returns:
        統計dict(n_names, n_lut_names, n_po_names, n_inputs, n_outputs,
        n_lut_nodes)
    """
    validate(graph)
    assert len(graph.outputs) == 1, 'write_blifは単一出力ポートのみ対応'
    for n in graph.nodes:
        assert n.type in ('lut_layer', 'concat'), (
            f'{n.id}: write_blifはlut_layer/concatのみ対応(type={n.type})'
        )

    # 入力信号名: 全入力ポートを順に連結してx_0..を割り当てる
    sig_table: dict[str, list[str]] = {}
    offset = 0
    for p in graph.inputs:
        sig_table[p.id] = [f'x_{offset + i}' for i in range(p.width)]
        offset += p.width
    in_width = offset

    # dst_slot昇順の入力元マップ(simulate.pyと同じ組み立て)
    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    in_srcs = {
        dst: [e.src for e in sorted(es, key=lambda e: e.dst_slot)]
        for dst, es in grouped.items()
    }

    body: list[str] = []
    n_lut_names = 0
    for node in graph.topological_order():
        srcs = in_srcs.get(node.id, [])
        if node.type == 'concat':
            # slot0が出力ビット列の先頭(下位ビット側)
            sig_table[node.id] = [sig for s in srcs for sig in sig_table[s]]
            continue

        p = node.params
        in_sig = sig_table[srcs[0]]
        n_inputs = int(p['n_inputs'])
        assert n_inputs <= 6, f'{node.id}: n_inputs={n_inputs}が6を超える'
        indices = np.asarray(p['indices'])
        tts = tt_from_hex(p['truth_tables'])
        full = (1 << (1 << n_inputs)) - 1

        out_sigs = []
        for o in range(int(p['out_width'])):
            out_sig = f'{node.id}__{o}'
            out_sigs.append(out_sig)
            tt = int(tts[o])
            if tt == 0:
                # 定数0: 入力なし(行なし=定数0)
                body.append(f'.names {out_sig}')
            elif tt == full:
                # 定数1: 入力なし+cube行1本
                body.append(f'.names {out_sig}')
                body.append('1')
            else:
                fanins = [in_sig[int(indices[j][o])] for j in range(n_inputs)]
                body.append('.names ' + ' '.join([*fanins, out_sig]))
                for k in range(1 << n_inputs):
                    if (tt >> k) & 1:
                        # 文字位置j(左から) = アドレスkのbit j = スロットj
                        pattern = ''.join(
                            str((k >> j) & 1) for j in range(n_inputs))
                        body.append(f'{pattern} 1')
            n_lut_names += 1
        sig_table[node.id] = out_sigs

    out_port = graph.outputs[0]
    out_src = graph.output_source()
    assert out_src is not None, '出力ポートが未接続'
    core_out_sigs = sig_table[out_src]
    assert len(core_out_sigs) == out_port.width, (
        f'出力幅不整合: {len(core_out_sigs)} != {out_port.width}'
    )
    for i, sig in enumerate(core_out_sigs):
        body.append(f'.names {sig} y_{i}')
        body.append('1 1')

    lines = [
        f'.model {model_name}',
        '.inputs ' + ' '.join(f'x_{i}' for i in range(in_width)),
        '.outputs ' + ' '.join(f'y_{i}' for i in range(out_port.width)),
        *body,
        '.end',
    ]
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return {
        'n_names': n_lut_names + out_port.width,
        'n_lut_names': n_lut_names,
        'n_po_names': out_port.width,
        'n_inputs': in_width,
        'n_outputs': out_port.width,
        'n_lut_nodes': sum(1 for n in graph.nodes if n.type == 'lut_layer'),
    }


# ============================================================
# read_blif
# ============================================================

@dataclass
class BlifNetwork:
    """パース済みBLIFネットワーク(フラット、.latch/.subckt/.gateは非対応)。

    names: [(fanins, out, cubes), ...]。
        fanins: `.names`ヘッダのfanin信号名列(位置j=スロットj)
        out: 出力信号名
        cubes: [(入力パターン文字列, 出力値'0'|'1'), ...]。
            0-arity(fanins=[])の行は入力パターンを''とする
    """
    inputs: list[str]
    outputs: list[str]
    names: list[tuple[list[str], str, list[tuple[str, str]]]]


def _read_logical_lines(path: str) -> list[str]:
    """コメント除去+行継続(`\\`)を解決した論理行列を返す。"""
    with open(path) as f:
        raw = f.read().splitlines()
    logical: list[str] = []
    buf = ''
    for line in raw:
        if '#' in line:
            line = line[:line.index('#')]
        line = line.rstrip()
        if line.endswith('\\'):
            buf += line[:-1] + ' '
            continue
        buf += line
        stripped = buf.strip()
        if stripped:
            logical.append(stripped)
        buf = ''
    if buf.strip():
        logical.append(buf.strip())
    return logical


def read_blif(path: str) -> BlifNetwork:
    """BLIFファイルをパースする(`.model/.inputs/.outputs/.names/.end`)。

    `.latch`/`.subckt`/`.gate`が現れたらValueError。
    """
    lines = _read_logical_lines(path)
    inputs: list[str] = []
    outputs: list[str] = []
    names: list[tuple[list[str], str, list[tuple[str, str]]]] = []
    cur: tuple[list[str], str, list[tuple[str, str]]] | None = None
    seen_model = False

    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        d = tokens[0]
        if d == '.model':
            seen_model = True
            cur = None
        elif d == '.inputs':
            inputs.extend(tokens[1:])
            cur = None
        elif d == '.outputs':
            outputs.extend(tokens[1:])
            cur = None
        elif d == '.names':
            toks = tokens[1:]
            assert toks, '.namesに出力名がない'
            fanins, out = toks[:-1], toks[-1]
            cur = (fanins, out, [])
            names.append(cur)
        elif d == '.end':
            cur = None
        elif d in ('.latch', '.subckt', '.gate'):
            raise ValueError(f'未対応のBLIFディレクティブ: {d}')
        elif d.startswith('.'):
            cur = None  # 未知の他ディレクティブ(.default等)は無視
        else:
            assert cur is not None, f'.namesの外にあるcube行: {line}'
            fanins, _out, cubes = cur
            if not fanins:
                assert len(tokens) == 1, f'0入力.namesのcube行が不正: {line}'
                cubes.append(('', tokens[0]))
            else:
                assert len(tokens) == 2, f'cube行が不正: {line}'
                cubes.append((tokens[0], tokens[1]))

    assert seen_model, '.modelが見つからない'
    return BlifNetwork(inputs=inputs, outputs=outputs, names=names)


# ============================================================
# blif_to_ir
# ============================================================

def pad_tt(tt: int, n_real: int, n_target: int) -> int:
    """n_real本の実スロットの真理値表をn_targetスロットへ拡張する。

    パディングスロット(j=n_real..n_target-1)の値に出力が依存しないよう
    tt'[addr] = tt[addr & ((1<<n_real)-1)] とする
    (convert/binarize.py の _extend_tt と同一アルゴリズム)。
    """
    assert 0 <= n_real <= n_target, (n_real, n_target)
    if n_real == n_target:
        return tt
    mask = (1 << n_real) - 1
    out = 0
    for addr in range(1 << n_target):
        if (tt >> (addr & mask)) & 1:
            out |= 1 << addr
    return out


def _cubes_to_tt(fanins: list[str], cubes: list[tuple[str, str]]) -> int:
    """(fanins, cubes) から2^len(fanins)ビットの真理値表(python int)を計算する。

    値'1'の行が1つでもあればON-set解釈(既定0)、無ければ値'0'の行の
    OFF-set解釈(既定1)。cube無しは定数0。'-'はドントケア展開。
    """
    n = len(fanins)
    if not cubes:
        return 0
    mode = '1' if any(v == '1' for _, v in cubes) else '0'
    default_one = (mode == '0')
    tt = (1 << (1 << n)) - 1 if default_one else 0
    for pattern, val in cubes:
        if val != mode:
            continue
        zero_mask = 0
        one_mask = 0
        for j, c in enumerate(pattern):
            if c == '0':
                zero_mask |= (1 << j)
            elif c == '1':
                one_mask |= (1 << j)
        for k in range(1 << n):
            if (k & zero_mask) == 0 and (k & one_mask) == one_mask:
                if default_one:
                    tt &= ~(1 << k)
                else:
                    tt |= (1 << k)
    return tt


def blif_to_ir(net: BlifNetwork, *, in_width: int) -> IRGraph:
    """BlifNetwork(可変fan-in DAG)をlut_layer/concat列のIRへ層化する。

    手順(モジュールdocstring参照): ASAP levelize → PO driver複製で
    専用最終レベルを作る → concatバスで層またぎ配線を解決 →
    層内fan-inパディング(pad_tt)。戻したIRはvalidate()を通す。
    """
    assert len(net.inputs) == in_width, (
        f'net.inputsの本数({len(net.inputs)}) != in_width({in_width})'
    )

    gates: dict[str, tuple[list[str], int]] = {}
    for fanins, out, cubes in net.names:
        assert out not in gates, f'{out}: .namesの重複定義'
        gates[out] = (list(fanins), _cubes_to_tt(fanins, cubes))

    input_set = set(net.inputs)
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def level(name: str) -> int:
        if name in input_set:
            return 0
        if name in memo:
            return memo[name]
        if name not in gates:
            raise ValueError(f'未定義の信号を参照している: {name}')
        if name in visiting:
            raise ValueError(f'循環参照を検出した: {name}')
        visiting.add(name)
        fanins, _tt = gates[name]
        lv = 1 + max((level(f) for f in fanins), default=0)
        visiting.discard(name)
        memo[name] = lv
        return lv

    for out in gates:
        level(out)
    lmax = max(memo.values(), default=0)

    by_level: dict[int, list[str]] = {}
    for _fanins, out, _cubes in net.names:
        by_level.setdefault(memo[out], []).append(out)

    bus_pos: dict[str, int] = {name: i for i, name in enumerate(net.inputs)}
    bus_id = 'x'
    bus_width = in_width
    nodes: list[IRNode] = []
    edges: list[IREdge] = []

    def emit_level(gate_names: list[str], node_id: str) -> IRNode:
        real: list[tuple[list[int], int]] = []
        for name in gate_names:
            fanins, tt = gates[name]
            if not fanins:
                # 定数ノード(fan-in0)はfan-in1へ正規化(任意の既存信号を使う)
                srcs = [bus_pos[net.inputs[0]]]
                tt_use = 0x3 if (tt & 1) else 0x0
            else:
                srcs = [bus_pos[f] for f in fanins]
                tt_use = tt
            real.append((srcs, tt_use))

        n_inputs = max(len(s) for s, _ in real)
        assert n_inputs <= 6, f'{node_id}: fan-in {n_inputs} が6を超える'
        indices = np.zeros((n_inputs, len(gate_names)), dtype=np.int64)
        tts: list[int] = []
        for o, (srcs, tt) in enumerate(real):
            n_r = len(srcs)
            padded = srcs + [srcs[0]] * (n_inputs - n_r)
            for j in range(n_inputs):
                indices[j, o] = padded[j]
            tts.append(pad_tt(tt, n_r, n_inputs))

        node = IRNode(
            id=node_id,
            type='lut_layer',
            params={
                'in_width': bus_width,
                'out_width': len(gate_names),
                'n_inputs': n_inputs,
                'indices': indices,
                'truth_tables': tt_to_hex(tts),
            },
            train_info={'source': 'blif_to_ir'},
        )
        nodes.append(node)
        edges.append(IREdge(src=bus_id, dst=node_id))
        for o, name in enumerate(gate_names):
            bus_pos[name] = bus_width + o
        return node

    def extend_bus(lut_node: IRNode, n_new: int, concat_id: str) -> None:
        nonlocal bus_id, bus_width
        concat_node = IRNode(
            id=concat_id, type='concat',
            params={'in_widths': [bus_width, n_new]},
        )
        nodes.append(concat_node)
        edges.append(IREdge(src=bus_id, dst=concat_id, dst_slot=0))
        edges.append(IREdge(src=lut_node.id, dst=concat_id, dst_slot=1))
        bus_width += n_new
        bus_id = concat_id

    for lv in range(1, lmax + 1):
        names_here = by_level.get(lv, [])
        if not names_here:
            continue
        lut_node = emit_level(names_here, f'L{lv}')
        extend_bus(lut_node, len(names_here), f'L{lv}_bus')

    # PO専用最終レベル: 各primary outputのドライバを複製する
    def resolve_driver(name: str) -> tuple[list[str], int]:
        if name in gates:
            return gates[name]
        if name in input_set:
            return [name], 0x2  # 恒等(パススルー)
        raise ValueError(f'未定義の出力ドライバ: {name}')

    po_names: list[str] = []
    for i, out_name in enumerate(net.outputs):
        fanins, tt = resolve_driver(out_name)
        virt = f'__po_{i}'
        gates[virt] = (fanins, tt)
        po_names.append(virt)

    po_node = emit_level(po_names, 'Lpo')
    edges.append(IREdge(src=po_node.id, dst='y'))

    graph = IRGraph(
        metadata={'source': 'blif_to_ir'},
        inputs=[IRPort(id='x', width=in_width)],
        outputs=[IRPort(id='y', width=len(net.outputs))],
        nodes=nodes,
        edges=edges,
    )
    validate(graph)
    return graph
