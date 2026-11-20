"""IR → 実行可能Pythonコード生成(E2、v0.2でDAG対応)。

fastapi非依存の純関数。生成コードは `build_model()` を定義し、
exec → build_model() → export_ir がビット厳密に元のIRへ戻ること
(往復同一性)を tests/test_editor_roundtrip.py で担保する。

規約:
    - モジュール順は graph.topological_order()
    - 学習済みlut_layerは restore_lut_layer(m{i}, _M{i}_INDICES, _M{i}_TT)
      で復元、draft(train_info.initialized=False)は「# 未学習(ランダム初期化)」
    - indices/truth_tablesリテラルはファイル末尾に定数として配置
    - flattenノードがIRにあればnn.Flatten()を生成、なければ前置しない
      (往復対称性: export_irはnn.Flattenをノード化しない)
    - 直列グラフは従来どおりnn.Sequentialを返すコードを生成(出力完全一致)。
      DAGグラフはnn.Moduleサブクラス(GraphModel)を生成し、forwardで
      topological_order順に実行する。concat/addは関数的に計算:
      concat=torch.cat(slot 0が先頭=下位ビット側)、
      add: or→1-(1-a)(1-b)、xor→a+b-2ab、sum→a+b
"""

import numpy as np

from griffith_logic.ir.build import FIXED_LUT_KINDS, LUT_LAYER_N_INPUTS, is_serial_graph
from griffith_logic.ir.schema import IRGraph

# 復元関数名 → import 元(ヘッダ生成用)
_RESTORE_IMPORT = 'from griffith_logic.ir.weights import '


def _table_constants(node, const_prefix: str, constants: list[str]) -> None:
    """indices / truth_tables の定数定義をファイル末尾の定数列へ足す。"""
    p = node.params
    indices = np.asarray(p['indices'], dtype=np.int64).tolist()
    tts = [t if isinstance(t, str) else hex(int(t)) for t in p['truth_tables']]
    constants.append(f'{const_prefix}_INDICES = {indices!r}')
    constants.append(f'{const_prefix}_TT = {tts!r}')


def _fixed_lut_lines(node, var: str, const_prefix: str, constants: list[str],
                     imports: set, indent: str) -> list[str]:
    """LutLayer が受けない幅 / kind='thermometer' の lut_layer → FixedLutLayer。

    build_torch と同じ規則(ir/build.py の _build_fixed_lut_layer)。真理値表が無い
    draft は組めないので NotImplementedError。
    """
    p = node.params
    ti = node.train_info
    if ti.get('initialized', True) is False or p.get('truth_tables') is None:
        raise NotImplementedError(
            f'{node.id}: n_inputs={p["n_inputs"]} の lut_layer は凍結 LUT として生成する'
            'ため真理値表が必要です(draft は不可)')
    imports.add('FixedLutLayer')
    _table_constants(node, const_prefix, constants)
    info = {k: v for k, v in ti.items() if k != 'initialized'}
    return [
        f"{indent}{var} = FixedLutLayer("
        f"{p['in_width']}, {p['out_width']}, {p['n_inputs']}, "
        f'{const_prefix}_INDICES, {const_prefix}_TT, train_info={info!r})'
    ]


def _dwn_layer_lines(node, var: str, const_prefix: str, constants: list[str],
                     imports: set, indent: str) -> list[str]:
    """kind='dwn' / 'ltt' の lut_layer → DWNLayer + restore_dwn_layer(build_torch と同じ復元)。"""
    p = node.params
    ti = node.train_info
    imports.add('DWNLayer')
    lines = [
        f"{indent}{var} = DWNLayer("
        f"{p['in_width']}, {p['out_width']}, n_inputs={p['n_inputs']}, "
        f"mapping={ti.get('mapping', 'random')!r}, efd={ti.get('efd', 'paper')!r})"
    ]
    if ti.get('initialized', True) and p.get('truth_tables') is not None:
        _table_constants(node, const_prefix, constants)
        lines.append(
            f'{indent}restore_dwn_layer({var}, '
            f'{const_prefix}_INDICES, {const_prefix}_TT)'
        )
    else:
        lines[0] += '  # 未学習(ランダム初期化)'
    return lines


def _lut_layer_lines(node, var: str, const_prefix: str,
                     constants: list[str], imports: set,
                     indent: str = '    ') -> list[str]:
    """lut_layerノードの構築行(+必要なら復元行)を生成する。"""
    p = node.params
    ti = node.train_info
    if ti.get('kind') in ('dwn', 'ltt'):
        return _dwn_layer_lines(node, var, const_prefix, constants, imports, indent)
    if ti.get('kind') in FIXED_LUT_KINDS or p['n_inputs'] not in LUT_LAYER_N_INPUTS:
        return _fixed_lut_lines(node, var, const_prefix, constants, imports, indent)
    imports.add('LutLayer')
    lines = [
        f"{indent}{var} = LutLayer("
        f"{p['in_width']}, {p['out_width']}, n_inputs={p['n_inputs']}, "
        f"connections={ti.get('connections', 'random')!r}, "
        f"parameterization={ti.get('parameterization', 'softmax')!r}, "
        f"selection={ti.get('selection', 'soft')!r})"
    ]
    initialized = ti.get('initialized', True)
    if initialized and p.get('truth_tables') is not None:
        _table_constants(node, const_prefix, constants)
        lines.append(
            f'{indent}restore_lut_layer({var}, '
            f'{const_prefix}_INDICES, {const_prefix}_TT)'
        )
    else:
        lines[0] += '  # 未学習(ランダム初期化)'
    return lines


def _module_lines(node, var: str, const_prefix: str, constants: list[str],
                  imports: set, indent: str = '    ') -> list[str]:
    """モジュール構築行を生成する(直列/DAG共通)。"""
    if node.type == 'lut_layer':
        return _lut_layer_lines(node, var, const_prefix, constants, imports, indent)
    if node.type == 'group_sum':
        imports.add('GroupSum')
        p = node.params
        bias = [int(v) for v in (p.get('bias') or [])]
        bias_arg = f', bias={bias!r}' if any(bias) else ''
        return [
            f"{indent}{var} = GroupSum(k={p['k']}, tau={float(p.get('tau', 1.0))!r}{bias_arg})"
        ]
    if node.type == 'permute':
        imports.add('Permute')
        p = node.params
        return [
            f"{indent}{var} = Permute({[int(v) for v in p['in_shape']]!r}, "
            f"{[int(v) for v in p['perm']]!r})"
        ]
    if node.type == 'binarize':
        imports.add('Binarize')
        return [
            f"{indent}{var} = Binarize("
            f"threshold={float(node.params.get('threshold', 0.5))!r})"
        ]
    if node.type == 'flatten':
        return [f'{indent}{var} = nn.Flatten()']
    raise NotImplementedError(
        f'{node.id}: {node.type} ノードはコード生成不可'
    )


def _restore_names(lines: list[str]) -> set[str]:
    """構築行が使う復元関数名(restore_lut_layer / restore_dwn_layer)を集める。"""
    return {
        name for name in ('restore_lut_layer', 'restore_dwn_layer')
        if any(f'{name}(' in ln for ln in lines)
    }


def _header_lines(graph: IRGraph, imports: set, restores: set,
                  needs_torch: bool) -> list[str]:
    name = str(graph.metadata.get('name', 'model'))
    header = [
        '"""griffith-logic ビジュアルエディタ生成モデル。',
        '',
        f'IR: {name}',
        '実行例: model = build_model(); model.eval()',
        '"""',
        '',
    ]
    if needs_torch:
        header.append('import torch')
    header.append('import torch.nn as nn')
    if imports:
        header.append('')
        header.append(f"from griffith_logic import {', '.join(sorted(imports))}")
    if restores:
        header.append(_RESTORE_IMPORT + ', '.join(sorted(restores)))
    return header


def _generate_serial(graph: IRGraph) -> str:
    """直列グラフ: nn.Sequentialを返すbuild_model()を生成(従来互換)。"""
    order = graph.topological_order()

    imports: set = set()
    restores: set = set()
    body: list[str] = []
    constants: list[str] = []
    var_names: list[str] = []

    for i, node in enumerate(order):
        var = f'm{i}'
        var_names.append(var)
        lines = _module_lines(node, var, f'_M{i}', constants, imports)
        restores |= _restore_names(lines)
        body.extend(lines)

    header = _header_lines(graph, imports, restores, needs_torch=False)
    lines = header + [
        '',
        '',
        'def build_model() -> nn.Sequential:',
        *body,
        f"    return nn.Sequential({', '.join(var_names)})",
    ]
    if constants:
        lines += ['', ''] + constants
    return '\n'.join(lines) + '\n'


def _add_forward_lines(node, var: str, in_vars: list[str]) -> list[str]:
    """addノードのforward行(soft意味論、hard evalでビット厳密)。"""
    mode = node.params['mode']
    a, rest = in_vars[0], in_vars[1:]
    if mode == 'or':
        lines = [f'{var} = 1 - (1 - {a}) * (1 - {rest[0]})'
                 f'  # {node.id}: add(or) 確率OR']
        for t in rest[1:]:
            lines.append(f'{var} = 1 - (1 - {var}) * (1 - {t})')
        return lines
    if mode == 'xor':
        lines = [f'{var} = {a} + {rest[0]} - 2 * {a} * {rest[0]}'
                 f'  # {node.id}: add(xor) 確率XOR']
        for t in rest[1:]:
            lines.append(f'{var} = {var} + {t} - 2 * {var} * {t}')
        return lines
    # sum: 整数和(論理領域の終端専用)
    return [f"{var} = {' + '.join(in_vars)}  # {node.id}: add(sum) 整数和"]


def _generate_dag(graph: IRGraph) -> str:
    """DAGグラフ: nn.Moduleサブクラス(GraphModel)を生成。"""
    if len(graph.inputs) != 1:
        raise NotImplementedError(
            f'コード生成は単一入力ポートのみ対応(現在{len(graph.inputs)}個)'
        )
    order = graph.topological_order()

    # dst_slot昇順の入力元マップ
    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    in_srcs = {
        dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
        for dst, edges in grouped.items()
    }

    imports: set = set()
    restores: set = set()
    needs_torch = False
    init_body: list[str] = []
    fwd_body: list[str] = []
    constants: list[str] = []
    var_of = {graph.inputs[0].id: 'x'}

    for i, node in enumerate(order):
        var = f'v{i}'
        in_vars = [var_of[s] for s in in_srcs[node.id]]
        if node.type == 'concat':
            needs_torch = True
            fwd_body.append(
                f"{var} = torch.cat([{', '.join(in_vars)}], dim=-1)"
                f'  # {node.id}: concat(slot0が先頭=下位ビット側)'
            )
        elif node.type == 'add':
            fwd_body.extend(_add_forward_lines(node, var, in_vars))
        else:
            mod = f'self.m{i}'
            lines = _module_lines(node, mod, f'_M{i}', constants,
                                  imports, indent='        ')
            restores |= _restore_names(lines)
            init_body.extend(lines)
            fwd_body.append(f'{var} = {mod}({in_vars[0]})')
        var_of[node.id] = var

    out_src = graph.output_source()
    out_var = var_of[out_src] if out_src in var_of else var_of[order[-1].id]

    header = _header_lines(graph, imports, restores, needs_torch)
    lines = header + [
        '',
        '',
        'class GraphModel(nn.Module):',
        '    """DAG構造モデル(自動生成、topological_order順にforward)"""',
        '',
        '    def __init__(self):',
        '        super().__init__()',
        *init_body,
        '',
        '    def forward(self, x):',
        *[f'        {ln}' for ln in fwd_body],
        f'        return {out_var}',
        '',
        '',
        'def build_model() -> nn.Module:',
        '    return GraphModel()',
    ]
    if constants:
        lines += ['', ''] + constants
    return '\n'.join(lines) + '\n'


def generate_code(graph: IRGraph) -> str:
    """IRからモデル定義コード(build_model()を含む単一ファイル)を生成する。"""
    if is_serial_graph(graph):
        return _generate_serial(graph)
    return _generate_dag(graph)
