"""torch.fx トレースによる DAG モデル → IR 抽出器。

export/extract.export_ir は nn.Sequential の直列走査しかできず、
U-Net 風の分岐(torch.cat スキップ接続)を持つモデル(SegConvDLGN 等)は
抽出できない。この export_dag_ir は torch.fx でモデルの forward を
シンボリックトレースし、得られた計算グラフを IR v0.2 の DAG へ写す。

方針(既存資産の最大再利用):
    - ゲート抽出は export/extract.py と export/lowering.py の
      (NATIVE_)EXTRACTORS 登録関数をそのまま使う(真理値表化の一元管理)。
    - 分岐/合流は torch.fx が捕える。torch.cat(dim=1) はチャネル方向の
      連結 = IR concat ノード(dst_slot 順=引数順、slot 0 が下位ビット)。
      NCHW flatten 上で cat([A, B], dim=1) の平坦順は [A_flat, B_flat] で
      あり concat 規約(slot 0 が先頭)と厳密一致する。
    - leaf module = 抽出器が登録された部品(LutConv2d/OrPooling2d/
      SpatialGroupSum/UpSample2d 等)。それ以外の nn.Module は fx が中へ
      トレースする(将来の合成部品に自然対応)。

対応 fx ノード:
    placeholder   → 入力ポート x
    call_module   → 抽出器で (conv2d|or_pool2d|lut_layer|group_sum ...) ノード列
    call_function → torch.cat(dim=1) を concat ノードへ(他関数は明示エラー)
    output        → 出力ポート y へ接続

正しさは PyTorch hard eval ⇔ ir/simulate.py の整数完全一致で固定する
(tests/test_ir_dag_seg.py。3段等価性テストの流儀)。
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.fx as fx

from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from .extract import ExtractCtx, find_extractor

JST = timezone(timedelta(hours=9))


class _LeafTracer(fx.Tracer):
    """抽出器を持つ部品を leaf として扱う fx トレーサ。

    LutConv2d 等のカスタム forward(非トレース可能な実数演算を含む)へは
    降りず、call_module ノードとして捕える。標準の leaf 判定も併用する。
    """

    def __init__(self, native: bool):
        super().__init__()
        self._native = native

    def is_leaf_module(self, m, qualified_name):
        if find_extractor(m, native=self._native) is not None:
            return True
        return super().is_leaf_module(m, qualified_name)


def _cat_channel_shape(shapes: list[tuple]) -> tuple:
    """チャネル方向 concat の出力形状 (Csum, H, W) を返す(H,W 一致を検証)。"""
    hw = {(int(s[1]), int(s[2])) for s in shapes}
    assert len(hw) == 1, f'concat の空間形状が不一致: {shapes}'
    csum = sum(int(s[0]) for s in shapes)
    h, w = shapes[0][1], shapes[0][2]
    return (csum, int(h), int(w))


def export_dag_ir(
    model: torch.nn.Module,
    input_shape: tuple,
    *,
    name: str = 'dag_model',
    metadata: dict | None = None,
    native_conv: bool = True,
) -> IRGraph:
    """torch.fx で DAG モデルを IR へ変換する(conv 系 U-Net 等)。

    Args:
        model: 単一入力・単一出力の nn.Module。forward は fx でトレース可能で
            あること(分岐は torch.cat(dim=1) のみ対応)。
        input_shape: 入力形状 (C, H, W)。入力幅 = prod(input_shape)。
        name: metadata.name。
        native_conv: True(既定)なら LutConv2d/OrPooling2d をネイティブノード
            (conv2d/or_pool2d)として抽出する。False なら lut_layer 列へ展開。

    Returns:
        IRGraph(concat を含む DAG)。simulate/build_torch でそのまま実行可能。
    """
    # loweringモジュールのextractor登録を確実に行う(循環import回避の遅延)
    from . import lowering  # noqa: F401

    model.eval()
    input_shape = tuple(int(v) for v in input_shape)
    in_width = int(np.prod(input_shape))

    graph = _LeafTracer(native_conv).trace(model)
    submodules = dict(model.named_modules())

    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    # fxノード名 → (IR src id, 幅, 形状(C,H,W)|None)
    env: dict[str, tuple] = {}
    counter = 0
    out_width: int | None = None
    # 多出力(dict/tuple ヘッド)の分割メタ。単一出力なら None のまま
    output_heads: list[dict] | None = None

    def next_id(label: str) -> str:
        nonlocal counter
        counter += 1
        return f'n{counter}_{label}'

    for fn in graph.nodes:
        if fn.op == 'placeholder':
            env[fn.name] = ('x', in_width, input_shape)

        elif fn.op == 'call_module':
            sub = submodules[fn.target]
            if len(fn.args) != 1 or not isinstance(fn.args[0], fx.Node):
                raise NotImplementedError(
                    f'{fn.target}: 単一テンソル入力の部品のみ対応')
            src_id, width, shape = env[fn.args[0].name]
            extractor = find_extractor(sub, native=native_conv)
            if extractor is None:
                raise NotImplementedError(
                    f'{fn.target}: {type(sub).__name__} の抽出器が未登録')
            node_id = next_id(type(sub).__name__.lower())
            new_nodes, new_width, new_shape = extractor(
                sub, node_id, ExtractCtx(width=width, shape=shape))
            prev = src_id
            for node in new_nodes:
                nodes.append(node)
                edges.append(IREdge(src=prev, dst=node.id))
                prev = node.id
            env[fn.name] = (prev, new_width, new_shape)

        elif fn.op == 'call_function':
            if fn.target is not torch.cat:
                raise NotImplementedError(
                    f'未対応の関数呼び出し: {fn.target}'
                    '(DAG抽出は torch.cat(dim=1) のみ対応)')
            dim = fn.kwargs.get('dim', fn.args[1] if len(fn.args) > 1 else 0)
            if int(dim) != 1:
                raise NotImplementedError(
                    f'torch.cat は dim=1(チャネル方向)のみ対応(got dim={dim})')
            arg_nodes = fn.args[0]
            in_widths, shapes, srcs = [], [], []
            for a in arg_nodes:
                s_id, w, sh = env[a.name]
                if sh is None:
                    raise NotImplementedError(
                        'concat 入力に空間形状(C,H,W)が必要')
                in_widths.append(int(w))
                shapes.append(sh)
                srcs.append(s_id)
            node_id = next_id('concat')
            nodes.append(IRNode(
                id=node_id, type='concat',
                params={'in_widths': in_widths},
                train_info={'from': 'torch.cat'}))
            for slot, s_id in enumerate(srcs):
                edges.append(IREdge(src=s_id, dst=node_id, dst_slot=slot))
            env[fn.name] = (node_id, sum(in_widths), _cat_channel_shape(shapes))

        elif fn.op == 'output':
            out_arg = fn.args[0]
            # 出力形態を [(名前, fx.Node), ...] に正規化する:
            #   単一テンソル / 1要素tuple → 従来どおり単一出力ポート y
            #   dict(検出3ヘッド hm/size/off 等) / 多要素tuple → concat で
            #   単一ポート y へ束ね、各ヘッドの幅/形状を output_heads に記録
            if isinstance(out_arg, dict):
                head_items = list(out_arg.items())
            elif isinstance(out_arg, tuple | list):
                head_items = [(f'out{i}', a) for i, a in enumerate(out_arg)]
            else:
                head_items = [('y', out_arg)]

            if len(head_items) == 1:
                out_id, out_width, _ = env[head_items[0][1].name]
                edges.append(IREdge(src=out_id, dst='y'))
            else:
                # 多出力: 各ヘッド(通常 group_sum の整数スコア)を concat で
                # 束ねる。dst_slot=引数順、slot0 が出力の先頭(concat規約)。
                heads, srcs, total = [], [], 0
                for hname, a in head_items:
                    if not isinstance(a, fx.Node):
                        raise NotImplementedError(
                            f'出力ヘッド {hname} はテンソル(fx.Node)のみ対応')
                    s_id, w, sh = env[a.name]
                    heads.append({
                        'name': str(hname), 'width': int(w),
                        'shape': ([int(v) for v in sh] if sh else None),
                    })
                    srcs.append(s_id)
                    total += int(w)
                cat_id = next_id('outcat')
                nodes.append(IRNode(
                    id=cat_id, type='concat',
                    params={'in_widths': [h['width'] for h in heads]},
                    train_info={'from': 'multi_output_heads'}))
                for slot, s_id in enumerate(srcs):
                    edges.append(IREdge(src=s_id, dst=cat_id, dst_slot=slot))
                edges.append(IREdge(src=cat_id, dst='y'))
                out_width = total
                output_heads = heads

        elif fn.op == 'get_attr':
            raise NotImplementedError(
                'get_attr(定数/バッファ参照)は DAG 抽出未対応')
        else:
            raise NotImplementedError(f'未対応の fx ノード: {fn.op}')

    if not nodes or out_width is None:
        raise ValueError('IRノード/出力を生成しないモデルは変換できない')

    last = next(n for n in nodes if n.id == edges[-1].src)
    if output_heads is not None:
        out_semantics = 'multi_head'
    elif last.type == 'group_sum':
        out_semantics = 'class_scores'
    else:
        out_semantics = None

    meta = {
        'name': name,
        'created_at': datetime.now(JST).isoformat(),
        **(metadata or {}),
    }
    if output_heads is not None:
        # ヘッド分割メタ(concat スロット順=出力ビット列順)。TS デコード側が
        # 単一スコア列を hm/size/off へ分割するのに使う。
        meta['output_heads'] = output_heads

    return IRGraph(
        metadata=meta,
        inputs=[IRPort(id='x', width=in_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=out_width, semantics=out_semantics)],
        nodes=nodes,
        edges=edges,
    )
