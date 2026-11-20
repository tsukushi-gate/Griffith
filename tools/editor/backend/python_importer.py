"""Python由来モデルのエディタimport処理。

Pythonモデルを正として読み込み、IRは表示・監視・コード生成のための
契約として作る。IRへ完全にrebuildできないAttention等はblackbox表示に
し、実際の学習は読み込んだPyTorchモデルを使う。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from griffith_logic.editor_models import (
    EditorModelSpec,
    get_editor_model_spec,
    infer_width,
)
from griffith_logic.ir import ir_to_dict
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort

from .demo import PROJECT_ROOT


CAP_LOGIC = {
    'python_executable': True,
    'trainable': True,
    'ir_buildable': True,
    'gate_expandable': True,
    'hdl_exportable': True,
}

CAP_FUNCTIONAL = {
    'python_executable': True,
    'trainable': False,
    'ir_buildable': True,
    'gate_expandable': False,
    'hdl_exportable': True,
}

CAP_BLACKBOX = {
    'python_executable': True,
    'trainable': True,
    'ir_buildable': False,
    'gate_expandable': False,
    'hdl_exportable': False,
}


@dataclass
class PythonImportResult:
    model: nn.Module
    ir: dict
    input_width: int
    spec: dict


def node_capabilities(node_type: str, *, python_executable: bool = False) -> dict:
    """ノード種別ごとの標準capabilities。meta側の明示値があれば上書きする。"""
    if node_type in ('lut_layer', 'conv2d', 'or_pool2d'):
        caps = dict(CAP_LOGIC)
    elif node_type in ('group_sum', 'binarize', 'flatten', 'concat', 'add', 'constant'):
        caps = dict(CAP_FUNCTIONAL)
    else:
        caps = dict(CAP_BLACKBOX)
    if python_executable:
        caps['python_executable'] = True
    return caps


def annotate_capabilities(doc: dict, *, python_executable: bool = False) -> dict:
    """IR dictの各nodeへ ``meta.capabilities`` を補完する。"""
    for node in doc.get('nodes', []):
        meta = node.setdefault('meta', {})
        existing = meta.get('capabilities') if isinstance(meta, dict) else None
        caps = node_capabilities(node.get('type', ''), python_executable=python_executable)
        if isinstance(existing, dict):
            caps.update(existing)
        meta['capabilities'] = caps
    return doc


def resolve_python_path(path: str, *, workspace_root: Path = PROJECT_ROOT) -> Path:
    """workspace内のPythonファイルだけを許可する。"""
    root = workspace_root.resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if root != p and root not in p.parents:
        raise ValueError(f'workspace外のPythonファイルは読み込めません: {p}')
    if p.suffix != '.py':
        raise ValueError(f'Pythonファイル(.py)を指定してください: {p}')
    if not p.exists():
        raise FileNotFoundError(f'ファイルが存在しません: {p}')
    return p


def _module_name_for_path(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:16]
    return f'_magicdelta_editor_model_{digest}'


def import_module_from_path(path: Path):
    """指定パスを一意な一時モジュール名でimportする。"""
    module_name = _module_name_for_path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'import specを作成できません: {path}')
    module = importlib.util.module_from_spec(spec)
    # 相対importや同階層importを許可するため一時的に親をsys.pathへ追加する。
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    return module


def _infer_module_io(model: nn.Module) -> tuple[int | None, int | None]:
    """fallback build_model用に既知モジュールからI/O幅を推定する。"""
    in_width = None
    out_width = None
    for m in model.modules():
        if hasattr(m, 'in_dim'):
            in_width = int(m.in_dim)
            break
        if hasattr(m, 'in_features'):
            in_width = int(m.in_features)
            break
    for m in reversed(list(model.modules())):
        if hasattr(m, 'out_dim'):
            out_width = int(m.out_dim)
            break
        if hasattr(m, 'out_features'):
            out_width = int(m.out_features)
            break
        if hasattr(m, 'k'):
            out_width = int(m.k)
            break
    return in_width, out_width


def _build_from_symbol(module, symbol: str | None) -> tuple[EditorModelSpec, nn.Module]:
    """module/symbolからEditorModelSpecとモデル実体を得る。"""
    if symbol:
        if not hasattr(module, symbol):
            raise AttributeError(f'シンボルが存在しません: {symbol}')
        obj = getattr(module, symbol)
        spec = get_editor_model_spec(obj)
        if spec is not None:
            model = spec.build_model()
            return spec, model
        if isinstance(obj, nn.Module):
            model = obj
        elif callable(obj):
            model = obj()
        else:
            raise TypeError(f'{symbol} はモデルまたはbuilderではありません')
        in_width, out_width = _infer_module_io(model)
        return _fallback_spec(symbol, in_width, out_width, builder=lambda: model), model

    candidates: list[tuple[str, EditorModelSpec]] = []
    for name, obj in vars(module).items():
        spec = get_editor_model_spec(obj)
        if spec is not None:
            candidates.append((name, spec))
    if len(candidates) == 1:
        spec = candidates[0][1]
        model = spec.build_model()
        return spec, model
    if len(candidates) > 1:
        names = [name for name, _ in candidates]
        raise ValueError(f'複数のeditor modelがあります。symbolを指定してください: {names}')

    if hasattr(module, 'EDITOR_MODEL'):
        spec = get_editor_model_spec(module.EDITOR_MODEL)
        if spec is None:
            raise TypeError('EDITOR_MODEL はEditorModelSpecである必要があります')
        model = spec.build_model()
        return spec, model

    if hasattr(module, 'build_model'):
        builder = module.build_model
        if not callable(builder):
            raise TypeError('build_model はcallableである必要があります')
        model = builder()
        in_width = getattr(module, 'INPUT_WIDTH', None)
        out_width = getattr(module, 'OUTPUT_WIDTH', None)
        inferred_in, inferred_out = _infer_module_io(model)
        return _fallback_spec(
            'build_model',
            int(in_width) if in_width is not None else inferred_in,
            int(out_width) if out_width is not None else inferred_out,
            builder=builder,
        ), model

    raise ValueError('editor_model付きbuilder、EDITOR_MODEL、build_modelのいずれも見つかりません')


def _fallback_spec(
    name: str,
    in_width: int | None,
    out_width: int | None,
    *,
    builder,
) -> EditorModelSpec:
    if in_width is None or out_width is None:
        raise ValueError('fallback build_modelではINPUT_WIDTH/OUTPUT_WIDTHまたは推定可能な層が必要です')
    return EditorModelSpec(
        id=name,
        name=name,
        input_spec={'width': in_width, 'encoding': 'binary_threshold_0.5'},
        output_spec={'width': out_width},
        capabilities={'python_executable': True, 'trainable': True},
        build=builder,
    )


def _io_spec(spec: EditorModelSpec) -> tuple[int, int]:
    return (
        infer_width(spec.input_spec, field_name='input_spec'),
        infer_width(spec.output_spec, field_name='output_spec'),
    )


def _stage_nodes(spec: EditorModelSpec, in_width: int, out_width: int) -> tuple[list[IRNode], list[IREdge]]:
    """spec.stagesからblackbox直列グラフを作る。stages未指定なら1ノード。"""
    stages = list(spec.stages) or [{
        'id': spec.id,
        'label': spec.name,
        'torch_class': spec.name,
        'in_width': in_width,
        'out_width': out_width,
    }]
    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    prev = 'x'
    prev_width = in_width
    for i, stage in enumerate(stages):
        sid = str(stage.get('id') or f'n{i}_{spec.id}')
        nid = sid if sid.startswith('n') else f'n{i}_{sid}'
        stage_out = int(stage.get('out_width', out_width if i == len(stages) - 1 else prev_width))
        stage_in = int(stage.get('in_width', prev_width))
        caps = dict(CAP_BLACKBOX)
        caps.update(stage.get('capabilities') or {})
        node = IRNode(
            id=nid,
            type='blackbox',
            params={
                'torch_class': str(stage.get('torch_class') or stage.get('label') or nid),
                'config': dict(stage.get('config') or {}),
                'in_width': stage_in,
                'out_width': stage_out,
            },
            train_info={
                'contract': {
                    'source': 'python_model',
                    'display': stage.get('label') or nid,
                    'ir_buildable': False,
                }
            },
            meta={
                'editor': {
                    'label': stage.get('label') or nid,
                    'hierarchy_id': stage.get('hierarchy_id') or sid,
                    'level': stage.get('level', 'L2'),
                    'detail': stage.get('detail'),
                },
                'capabilities': caps,
            },
        )
        nodes.append(node)
        edges.append(IREdge(src=prev, dst=nid))
        prev = nid
        prev_width = stage_out
    edges.append(IREdge(src=prev, dst='y'))
    return nodes, edges


def _display_ir(spec: EditorModelSpec, path: Path, symbol: str | None) -> dict:
    """Pythonモデル表示用のblackbox IRを作る。"""
    in_width, out_width = _io_spec(spec)
    nodes, edges = _stage_nodes(spec, in_width, out_width)
    metadata = {
        'name': spec.name,
        'source': 'python',
        'editor': {
            'model_ref': {
                'kind': 'python',
                'path': str(path),
                'symbol': symbol,
                'spec': spec.to_metadata(),
            },
            'hierarchy': {
                'levels': ['L3:model', 'L2:layer', 'L1:op', 'L0:gate'],
                'mode': 'python-first',
                'nodes': [dict(n) for n in spec.hierarchy],
            },
            'lowering': {
                str(k): dict(v) for k, v in spec.lowering.items()
            },
            'paper_ref': dict(spec.paper_ref) if spec.paper_ref is not None else None,
            'dataset_presets': {
                str(k): dict(v) for k, v in spec.dataset_presets.items()
            },
        },
    }
    graph = IRGraph(
        metadata=metadata,
        inputs=[IRPort(
            id='x',
            width=in_width,
            shape=spec.input_spec.get('shape'),
            encoding=spec.input_spec.get('encoding'),
            semantics=spec.input_spec.get('semantics'),
        )],
        outputs=[IRPort(
            id='y',
            width=out_width,
            shape=spec.output_spec.get('shape'),
            encoding=spec.output_spec.get('encoding'),
            semantics=spec.output_spec.get(
                'semantics',
                'class_scores' if spec.task == 'classification' else None,
            ),
        )],
        nodes=nodes,
        edges=edges,
    )
    doc = ir_to_dict(graph)
    doc['version'] = '0.3.0'
    return annotate_capabilities(doc, python_executable=True)


def _editor_meta(doc: dict | None) -> dict:
    if not isinstance(doc, dict):
        return {}
    editor = doc.get('metadata', {}).get('editor')
    return editor if isinstance(editor, dict) else {}


def _spec_meta(doc: dict | None) -> dict:
    ref = _editor_meta(doc).get('model_ref')
    if not isinstance(ref, dict):
        return {}
    spec = ref.get('spec')
    return spec if isinstance(spec, dict) else {}


def hierarchy_from_ir(doc: dict | None) -> dict:
    """現在IRの階層メタデータをUI向けに返す。"""
    editor = _editor_meta(doc)
    hierarchy = editor.get('hierarchy')
    if not isinstance(hierarchy, dict):
        hierarchy = {}
    nodes = hierarchy.get('nodes')
    if nodes is None:
        nodes = _spec_meta(doc).get('hierarchy', [])
    if not isinstance(nodes, list):
        nodes = []
    return {
        'levels': hierarchy.get('levels') or ['L3:model', 'L2:layer', 'L1:op', 'L0:gate'],
        'mode': hierarchy.get('mode') or 'ir',
        'nodes': nodes,
        'paper_ref': editor.get('paper_ref') or _spec_meta(doc).get('paper_ref'),
        'dataset_presets': (
            editor.get('dataset_presets') or _spec_meta(doc).get('dataset_presets') or {}
        ),
    }


def _lowering_defs(doc: dict | None) -> dict:
    editor = _editor_meta(doc)
    lowering = editor.get('lowering')
    if isinstance(lowering, dict):
        return lowering
    spec_lowering = _spec_meta(doc).get('lowering')
    return spec_lowering if isinstance(spec_lowering, dict) else {}


def _hierarchy_node(doc: dict | None, node_id: str) -> dict | None:
    for node in hierarchy_from_ir(doc).get('nodes', []):
        if isinstance(node, dict) and node.get('id') == node_id:
            return node
    return None


def _child_hierarchy_nodes(doc: dict | None, node_id: str) -> list[dict]:
    out = []
    for node in hierarchy_from_ir(doc).get('nodes', []):
        if isinstance(node, dict) and node.get('parent') == node_id:
            out.append(node)
    return out


def _repeat_gate_pattern(out_width: int, pattern: list[int] | None = None) -> list[str]:
    base = pattern or [0x6, 0x8, 0xE, 0xA]
    return [hex(base[i % len(base)]) for i in range(out_width)]


def _repeat_indices(in_width: int, out_width: int, n_inputs: int) -> list[list[int]]:
    if in_width <= 0:
        return [[0 for _ in range(out_width)] for _ in range(n_inputs)]
    return [
        [int((o * n_inputs + slot) % in_width) for o in range(out_width)]
        for slot in range(n_inputs)
    ]


def virtual_lut_node_from_lowering(node_id: str, lowering: dict, hierarchy_node: dict | None) -> dict:
    """Lowering定義からDeckGLが読める仮想lut_layer IRノードを作る。

    Python実モデルは破壊しない。これは大規模LUTの構造理解用ビューで、
    実重みの確率分布は今後hooksから上書きする。
    """
    in_width = int(lowering.get('in_width') or lowering.get('input_width') or 1)
    out_width = int(lowering.get('out_width') or lowering.get('output_width') or 1)
    n_inputs = int(lowering.get('n_inputs') or 2)
    pattern = lowering.get('truth_table_pattern')
    if isinstance(pattern, list):
        gate_pattern = [int(v) for v in pattern]
    else:
        gate_pattern = None
    indices = lowering.get('indices')
    if not isinstance(indices, list):
        indices = _repeat_indices(in_width, out_width, n_inputs)
    truth_tables = lowering.get('truth_tables')
    if not isinstance(truth_tables, list):
        truth_tables = _repeat_gate_pattern(out_width, gate_pattern)
    caps = dict(CAP_LOGIC)
    caps.update((hierarchy_node or {}).get('capabilities') or {})
    label = (hierarchy_node or {}).get('label') or lowering.get('label') or node_id
    return {
        'id': node_id,
        'type': 'lut_layer',
        'params': {
            'in_width': in_width,
            'out_width': out_width,
            'n_inputs': n_inputs,
            'indices': indices,
            'truth_tables': truth_tables,
        },
        'train_info': {
            'parameterization': lowering.get('parameterization') or 'softmax',
            'selection': lowering.get('selection') or 'soft',
            'virtual': True,
        },
        'meta': {
            'editor': {
                'label': label,
                'level': 'L0',
                'detail': lowering.get('detail') or (hierarchy_node or {}).get('detail'),
            },
            'capabilities': caps,
        },
    }


def lowering_view_from_ir(doc: dict | None, node_id: str, level: str = 'L0') -> dict:
    """階層ノードのlowering表示を返す。"""
    if level not in ('L0', 'L1', 'L2'):
        raise ValueError(f'未対応level: {level}(対応: L0/L1/L2)')
    lowerings = _lowering_defs(doc)
    lowering = lowerings.get(node_id)
    hnode = _hierarchy_node(doc, node_id)
    if lowering is None and hnode is None:
        raise KeyError(f'階層ノードが存在しません: {node_id}')
    if lowering is not None and not isinstance(lowering, dict):
        raise ValueError(f'lowering定義が不正です: {node_id}')
    lowering = lowering or {}
    caps = dict(CAP_BLACKBOX)
    caps.update((hnode or {}).get('capabilities') or lowering.get('capabilities') or {})
    virtual_node = None
    summary = None
    if level == 'L0' and str(lowering.get('kind') or '').lower() == 'lut_layer':
        virtual_node = virtual_lut_node_from_lowering(node_id, lowering, hnode)
        p = virtual_node['params']
        tts = p.get('truth_tables') or []
        gate_ids = [int(str(tt), 16) for tt in tts[: int(p['out_width'])]]
        summary = {
            'type': 'layer_summary',
            'layer': node_id,
            'step': 0,
            'n_inputs': int(p['n_inputs']),
            'gate_ids': gate_ids,
            'confidence': [float(v) for v in lowering.get('confidence', [])]
            if isinstance(lowering.get('confidence'), list)
            else [1.0 for _ in gate_ids],
            # §4-1: 静的プレビューでは迷い統計は未計測(Noneで型を揃える)
            'entropy_mean': None,
            'entropy_max': None,
        }
    return {
        'node_id': node_id,
        'level': level,
        'title': lowering.get('label') or (hnode or {}).get('label') or node_id,
        'description': (hnode or {}).get('detail') or lowering.get('detail'),
        'capabilities': caps,
        'shape': (hnode or {}).get('shape') or lowering.get('shape'),
        'children': _child_hierarchy_nodes(doc, node_id),
        'lowering': lowering,
        'virtual_ir_node': virtual_node,
        'summary': summary,
        'actions': lowering.get('actions') or (
            ['open_l0'] if virtual_node is not None else []
        ),
    }


def load_python_editor_model(
    path: str,
    *,
    symbol: str | None = None,
    mode: str = 'auto',
    workspace_root: Path = PROJECT_ROOT,
) -> PythonImportResult:
    """Pythonファイルからモデルを読み込み、STATEへ入れられる形へ変換する。"""
    if mode not in ('auto', 'display'):
        raise ValueError(f'未対応mode: {mode}(対応: auto/display)')
    py_path = resolve_python_path(path, workspace_root=workspace_root)
    module = import_module_from_path(py_path)
    spec, model = _build_from_symbol(module, symbol)
    if not isinstance(model, nn.Module):
        raise TypeError(f'builderの戻り値はnn.Moduleである必要があります: {type(model)}')
    ir = _display_ir(spec, py_path, symbol)
    return PythonImportResult(
        model=model,
        ir=ir,
        input_width=ir['inputs'][0]['width'],
        spec=spec.to_metadata(),
    )


def preflight_python_import(path: str, *, symbol: str | None = None, timeout: float = 10.0) -> None:
    """別プロセスでimport/buildを試し、サーバ本体のクラッシュを避ける。"""
    cmd = [
        sys.executable,
        '-m',
        'tools.editor.backend.python_importer',
        '--inspect',
        path,
    ]
    if symbol:
        cmd.extend(['--symbol', symbol])
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f'exit={proc.returncode}'
        raise ValueError(f'Pythonモデルのpreflightに失敗: {detail}')


def generate_python_ref_code(model_ref: dict) -> str:
    """Python由来モデル用のcompanionコードを生成する。"""
    path = str(model_ref['path'])
    symbol = model_ref.get('symbol')
    spec = model_ref.get('spec') or {}
    name = str(spec.get('name') or spec.get('id') or 'python_model')
    symbol_expr = repr(symbol)
    return f'''"""griffith-logic エディタ生成Pythonモデル参照。

元モデル: {path}
モデル名: {name}
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SOURCE_PATH = Path({path!r})
SYMBOL = {symbol_expr}
EDITOR_MODEL = {spec!r}


def _load_module():
    spec = importlib.util.spec_from_file_location("_magicdelta_generated_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"import specを作成できません: {{SOURCE_PATH}}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SOURCE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(SOURCE_PATH.parent))
        except ValueError:
            pass
    return module


def build_model():
    module = _load_module()
    if SYMBOL:
        obj = getattr(module, SYMBOL)
    else:
        decorated = [
            obj for obj in vars(module).values()
            if hasattr(obj, "__magicdelta_editor_model__")
        ]
        obj = decorated[0] if len(decorated) == 1 else getattr(module, "build_model")
    editor_spec = getattr(obj, "__magicdelta_editor_model__", None)
    if editor_spec is not None:
        return editor_spec.build_model()
    return obj()
'''


def _inspect_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--symbol')
    parser.add_argument('path')
    args = parser.parse_args()
    result = load_python_editor_model(args.path, symbol=args.symbol)
    print(json.dumps({'input_width': result.input_width, 'spec': result.spec}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(_inspect_cli())
