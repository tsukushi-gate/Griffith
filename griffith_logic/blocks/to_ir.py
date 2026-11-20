"""blocks部品で組んだモデル → IRグラフ直接組み立て(A3)。

export/extract.export_ir はnn.Sequentialの直列走査しかできず、
Residual等のカスタムforwardは捕まらない。blocks部品を含むモデルは
この blocks_to_ir でDAGとして直接組み立てる(既存export_irは無変更)。

対応範囲(○=IRネイティブ、△=blackbox、—=痕跡なし、×=非対応):
    ○ export/extract.pyのEXTRACTORS登録型
      (LutLayer/GroupSum/Binarize/DifferentiableLutBlock等)
    ○ Residual(inner, merge) — fanout + concat/addノードの部分グラフ。
      構造部品のネストは1段のみ(Residual内のResidualはNotImplementedError)
    ○ LogicPositionalEncoding — constantノード + concatノード
    ○ LutLayer→LogicBatchNorm→Binarize — プローブfoldingで
      単一lut_layerノードへ折り畳み(truth_tables_from_composition)
    △ LogicBatchNorm(折り畳み不能な位置)/LogicLayerNorm — blackboxノード
    △ LogicAttention/LogicTransformerBlock — blackbox+contract
      (HDL不可の理由はblocks/attention.py docstring参照)
    — LogicDropout — eval時identityのためスキップ(IRに痕跡なし)
    × 素のConcat/Add — 多入力部品は直列走査で意味を持たない
      (Residual経由で使うこと。NotImplementedError)
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import torch.nn as nn

from ..binarize import Binarize
from ..export.discretize import truth_tables_from_composition
from ..export.extract import ExtractCtx, find_extractor
from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.serialize import tt_to_hex
from ..layers.lut import LutLayer
from .attention import LogicAttention
from .dropout import LogicDropout
from .norm import LogicBatchNorm, LogicLayerNorm
from .positional import LogicPositionalEncoding
from .structural import Add, Concat, Residual
from .transformer import LogicTransformerBlock

JST = timezone(timedelta(hours=9))


class _Walker:
    """走査状態(ノード列・エッジ列・id採番)"""

    def __init__(self):
        self.nodes: list[IRNode] = []
        self.edges: list[IREdge] = []
        self._counter = 0

    def next_id(self, label: str) -> str:
        nid = f'n{self._counter}_{label}'
        self._counter += 1
        return nid

    def add_serial(self, node: IRNode, src: str) -> str:
        """直列ノードを追加してエッジで接続し、新しいsrc idを返す"""
        self.nodes.append(node)
        self.edges.append(IREdge(src=src, dst=node.id))
        return node.id


def _emit_extractor_chain(w: _Walker, module: nn.Module,
                          src: str, width: int) -> tuple[str, int]:
    """EXTRACTORS登録型を既存抽出器でノード列化する"""
    extractor = find_extractor(module)
    if extractor is None:
        raise NotImplementedError(
            f'{type(module).__name__} はblocks_to_ir未対応'
            '(export/extract.pyのregister_extractorで追加するか、'
            'blocks部品で構成すること)'
        )
    node_id = w.next_id(type(module).__name__.lower())
    new_nodes, new_width, _ = extractor(
        module, node_id, ExtractCtx(width=width, shape=None),
    )
    for node in new_nodes:
        src = w.add_serial(node, src)
    return src, new_width


def _emit_residual(w: _Walker, module: Residual,
                   src: str, width: int, depth: int) -> tuple[str, int]:
    """Residual → fanout + concat/addノードの部分グラフ"""
    inner = module.inner
    inner_modules = list(inner) if isinstance(inner, nn.Sequential) else [inner]
    inner_src, inner_width = _walk(w, inner_modules, src, width, depth + 1)

    if module.merge == 'concat':
        node = IRNode(
            id=w.next_id('concat'),
            type='concat',
            params={'in_widths': [width, inner_width]},
            train_info={'from': 'Residual'},
        )
        out_width = width + inner_width
    else:  # or / xor
        if inner_width != width:
            raise ValueError(
                f'Residual(merge={module.merge}): innerの出力幅'
                f'({inner_width})が入力幅({width})と一致しない'
            )
        node = IRNode(
            id=w.next_id('add'),
            type='add',
            params={'width': width, 'mode': module.merge},
            train_info={'from': 'Residual'},
        )
        out_width = width
    w.nodes.append(node)
    # slot 0=恒等枝(torchのcat([x, inner(x)])と同一ビット順)
    w.edges.append(IREdge(src=src, dst=node.id, dst_slot=0))
    w.edges.append(IREdge(src=inner_src, dst=node.id, dst_slot=1))
    return node.id, out_width


def _emit_positional(w: _Walker, module: LogicPositionalEncoding,
                     src: str, width: int) -> tuple[str, int]:
    """LogicPositionalEncoding → constantノード + concatノード"""
    const_width = module.seq_len * module.width
    const = IRNode(
        id=w.next_id('constant'),
        type='constant',
        params={'width': const_width, 'bits': module.bits_flat},
        train_info={'from': 'LogicPositionalEncoding',
                    'code': module.code,
                    'seq_len': module.seq_len},
    )
    cat = IRNode(
        id=w.next_id('concat'),
        type='concat',
        params={'in_widths': [width, const_width]},
        train_info={'from': 'LogicPositionalEncoding'},
    )
    w.nodes.extend([const, cat])
    w.edges.append(IREdge(src=src, dst=cat.id, dst_slot=0))
    w.edges.append(IREdge(src=const.id, dst=cat.id, dst_slot=1))
    return cat.id, width + const_width


def _emit_folded_lut(w: _Walker, lut: LutLayer, norm: LogicBatchNorm,
                     binarize: Binarize, src: str, width: int) -> tuple[str, int]:
    """LutLayer→LogicBatchNorm→Binarize をプローブfoldingで折り畳む"""
    assert lut.in_dim == width, (
        f'LutLayer.in_dim({lut.in_dim}) != 前段幅({width})'
    )
    tables = truth_tables_from_composition(lut, [norm, binarize])
    node = IRNode(
        id=w.next_id('lutlayer'),
        type='lut_layer',
        params={
            'in_width': lut.in_dim,
            'out_width': lut.out_dim,
            'n_inputs': lut.n_inputs,
            'indices': lut.indices.cpu().numpy().astype(np.int64),
            'truth_tables': tt_to_hex(tables.tolist()),
        },
        train_info={
            'folded_from': 'LutLayer+LogicBatchNorm+Binarize',
            'parameterization': lut.parameterization,
        },
    )
    return w.add_serial(node, src), lut.out_dim


def _emit_norm_blackbox(w: _Walker, module: nn.Module,
                        src: str, width: int) -> tuple[str, int]:
    """折り畳み不能な位置のNorm部品 → blackboxノード(幅不変)"""
    assert module.width == width, (
        f'{type(module).__name__}.width({module.width}) != 前段幅({width})'
    )
    node = IRNode(
        id=w.next_id(type(module).__name__.lower()),
        type='blackbox',
        params={
            'torch_class': type(module).__name__,
            'config': {'width': module.width},
            'in_width': width,
            'out_width': width,
        },
    )
    return w.add_serial(node, src), width


def _emit_attention_blackbox(w: _Walker, module: nn.Module,
                             src: str, width: int) -> tuple[str, int]:
    """LogicAttention/LogicTransformerBlock → blackbox+contract"""
    if isinstance(module, LogicTransformerBlock):
        d_model = module.d_model
        config = {
            'd_model': d_model,
            'n_heads': module.attn.n_heads,
            'norm': module.norm_mode,
            'merge': module.merge_mode,
        }
        out_width = width  # (B,S,D)→(B,S,D)で幅不変
    else:  # LogicAttention(self-attention契約: q=k=v=x)
        d_model = module.d_model
        config = {
            'd_model': d_model,
            'n_heads': module.n_heads,
            'mode': module.mode,
            'similarity': module.similarity,
        }
        out_width = (width // d_model) * module.d_v
    if width % d_model != 0:
        raise ValueError(
            f'{type(module).__name__}: 前段幅({width})が'
            f'd_model({d_model})の倍数でない'
        )
    node = IRNode(
        id=w.next_id(type(module).__name__.lower()),
        type='blackbox',
        params={
            'torch_class': type(module).__name__,
            'config': config,
            'in_width': width,
            'out_width': out_width,
        },
        train_info={
            'contract': {
                'input': f'(B, S, {d_model})のNCHW flatten(S={width // d_model})',
                'output': f'(B, S, {out_width // (width // d_model)})のflatten',
                'hdl': 'softmax/argmaxを含むためHDL生成不可',
            },
        },
    )
    return w.add_serial(node, src), out_width


def _walk(w: _Walker, modules: list, src: str, width: int,
          depth: int) -> tuple[str, int]:
    """モジュール列を走査してノード・エッジを追加し、(最終src, 幅)を返す"""
    i = 0
    while i < len(modules):
        m = modules[i]
        if isinstance(m, LogicDropout | nn.Flatten):
            i += 1  # eval時identity(dropout)/幅不変(flatten): 痕跡なし
            continue
        if isinstance(m, nn.Sequential):
            src, width = _walk(w, list(m), src, width, depth)
            i += 1
            continue
        if isinstance(m, Residual):
            if depth >= 1:
                raise NotImplementedError(
                    '構造部品のネストは1段まで'
                    '(Residual内のResidualは未対応)'
                )
            src, width = _emit_residual(w, m, src, width, depth)
            i += 1
            continue
        if isinstance(m, Concat | Add):
            raise NotImplementedError(
                f'素の{type(m).__name__}は直列走査では意味を持たない'
                '(多入力部品。Residual経由で使うこと)'
            )
        if isinstance(m, LogicPositionalEncoding):
            src, width = _emit_positional(w, m, src, width)
            i += 1
            continue
        if (isinstance(m, LutLayer) and i + 2 < len(modules)
                and isinstance(modules[i + 1], LogicBatchNorm)
                and isinstance(modules[i + 2], Binarize)):
            src, width = _emit_folded_lut(
                w, m, modules[i + 1], modules[i + 2], src, width,
            )
            i += 3
            continue
        if isinstance(m, LogicLayerNorm | LogicBatchNorm):
            src, width = _emit_norm_blackbox(w, m, src, width)
            i += 1
            continue
        if isinstance(m, LogicAttention | LogicTransformerBlock):
            src, width = _emit_attention_blackbox(w, m, src, width)
            i += 1
            continue
        src, width = _emit_extractor_chain(w, m, src, width)
        i += 1
    return src, width


def blocks_to_ir(
    model: nn.Module,
    input_width: int,
    *,
    name: str = 'blocks_model',
    metadata: dict | None = None,
) -> IRGraph:
    """blocks部品で組んだモデルをIRグラフへ変換する。

    Args:
        model: nn.Sequential(または単一Module)。対応範囲はモジュール
            docstringの○×表を参照
        input_width: 入力ビット幅
        name: metadata.name

    Returns:
        IRGraph(Residual等を含む場合はDAG。build_torchでIRGraphModule化)
    """
    # loweringモジュールのextractor登録を確実に行う(循環import回避の遅延)
    from ..export import lowering  # noqa: F401

    modules = list(model) if isinstance(model, nn.Sequential) else [model]
    w = _Walker()
    last_src, out_width = _walk(w, modules, 'x', input_width, 0)
    if not w.nodes:
        raise ValueError('IRノードを1つも生成しないモデルは変換できない')
    w.edges.append(IREdge(src=last_src, dst='y'))

    last_node = next(n for n in w.nodes if n.id == last_src)
    out_semantics = 'class_scores' if last_node.type == 'group_sum' else None

    return IRGraph(
        metadata={
            'name': name,
            'created_at': datetime.now(JST).isoformat(),
            **(metadata or {}),
        },
        inputs=[IRPort(id='x', width=input_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=out_width, semantics=out_semantics)],
        nodes=w.nodes,
        edges=w.edges,
    )
