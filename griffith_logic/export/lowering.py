"""conv系・BinaryBrain系モジュールのIR lowering(A2でIR→IR純関数化)。

いずれも既存の `lut_layer` / `group_sum` IRノードへ展開(unroll)する:

    LutConv2d      → 木の各レベルを1つのlut_layerノードに展開。
                     重みは全空間位置で共有 = 同一真理値表の繰り返し。
                     ニューロン順は (oc*L + pos)*gates + g(最終レベルで
                     NCHW flattenと一致する)
    OrPooling2d    → 窓k×kのORゲート(窓≦6入力のみ。真理値表 2^(2^n)−2)
    SpatialGroupSum→ ワイヤ順を(o,y,x,g)へ並べ替える恒等置換lut_layer
                     (n_inputs=1, tt=0x2)+ group_sum
                     (native_conv=True時はpermuteネイティブノード+
                     group_sum)
    DifferentiableLutBlock → ブロック自身のeval()に全2^Nパターンを流す
                     プローブで真理値表化(Unbinarize+raw LUT+BatchNorm+
                     Binarizeの合成を厳密に畳み込む。数式の複製なし)
    UpSample2d     → nearest拡大は固定配線(複製)なので、パススルーゲート
                     (n_inputs=1, tt=0x2)のインデックス複製lut_layer。
                     native_conv=True時はupsample2dネイティブノード
    BinaryBrainConv2d → im2col→ブロック連鎖→col2imを窓ごとに平坦化し、
                     ブロックごとのlut_layer列へ展開

padding>0の範囲外参照は「真理値表の特殊化」で処理する:
範囲外スロットjは入力0に固定できるため、tt'[k] = tt[k & ~(1<<j)] と
畳み込み、スロットはダミー(index 0)に繋ぐ。定数線・スキーマ拡張が
不要で、モデルのどの深さでも動作する。

A2での二段構え:
    torch→IR: export_ir(native_conv=True) がネイティブノード
              (conv2d/or_pool2d)を生成(NATIVE_EXTRACTORS。既定は従来lowering)
    IR→IR:    lower_to_primitives(graph) がネイティブノードをlut_layer列へ
              展開する純関数パス。HDL生成(to_verilog/to_vhdl)の前段で呼ぶ

恒等配線ノード(upsample2d/permute)もconv2d/or_pool2dと同型の
二段構えに揃える。UpSample2d/SpatialGroupSumの恒等配線部は
NATIVE_EXTRACTORSでは小さいparams(in_shape+scale、in_shape+perm)の
ネイティブノードを生成し、lower_to_primitivesがHDL生成/リソース見積り
向けにlut_layer列へ実行時展開する(indexはJSONに保存しない)。
"""

import copy

import numpy as np
import torch

from ..bb_conv2d import BinaryBrainConv2d
from ..conv import LutConv2d
from ..differentiable_lut_block import DifferentiableLutBlock
from ..dwn_conv import DWNConv2d
from ..ir.schema import IREdge, IRGraph, IRNode
from ..ir.serialize import tt_from_hex, tt_to_hex
from ..pooling import OrPooling2d
from ..spatial_group_sum import SpatialGroupSum
from ..upsample import UpSample2d
from .discretize import truth_tables_from_conv
from .extract import (
    _IDENTITY_TT_1IN,
    ExtractCtx,
    register_extractor,
    register_native_extractor,
)

# lower_to_primitivesの展開対象ノード型
LOWERABLE_TYPES = ('conv2d', 'or_pool2d', 'upsample2d', 'permute')


def _specialize_zero(tt: int, j: int, n_inputs: int) -> int:
    """入力スロットjを定数0に固定した真理値表を返す。

    tt'[k] = tt[k & ~(1<<j)] — 出力はスロットjに依存しなくなるため、
    スロットjは任意の有効indexに繋いでよい(padding処理の中核)。
    """
    mask = ~(1 << j)
    out = 0
    for k in range(1 << n_inputs):
        if (tt >> (k & mask)) & 1:
            out |= 1 << k
    return out


def _lut_node(node_id, in_width, out_width, n_inputs, indices, tts,
              lowered_from):
    return IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': in_width,
            'out_width': out_width,
            'n_inputs': n_inputs,
            'indices': np.asarray(indices, dtype=np.int64),
            'truth_tables': tt_to_hex(tts),
        },
        train_info={'lowered_from': lowered_from},
    )


def _unroll_conv_tree(
    node_id: str,
    in_width: int,
    in_shape: tuple,
    out_channels: int,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    tree_depth: int,
    n_inputs: int,
    idxf: np.ndarray,
    tts_gates: np.ndarray,
) -> tuple[list, int, tuple]:
    """conv木をlut_layerノード列へ展開する純関数(torch経路とIR経路で共有)。

    Args:
        idxf: (out_channels, n_leaves) の受容野内フラット接続テーブル
        tts_gates: (out_channels, n_internal) の内部ゲート真理値表(object配列)

    Returns:
        (lut_layerノード列, 出力幅, 出力shape(OC, H_out, W_out))
    """
    C, H, W = in_shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    L = H_out * W_out
    OC = out_channels
    N = n_inputs

    nodes = []
    n_gates = N ** tree_depth   # 現レベルの「子」の数(=n_leaves)
    gate_offset = 0
    prev_width = in_width       # レベル0の入力幅(実入力幅、定数線は範囲外index)

    for level in range(tree_depth):
        g_out = n_gates // N
        width_out = OC * L * g_out
        indices = np.empty((N, width_out), dtype=np.int64)
        tts = []

        for oc in range(OC):
            for pos in range(L):
                oy, ox = divmod(pos, W_out)
                base_out = (oc * L + pos) * g_out
                for g in range(g_out):
                    out_idx = base_out + g
                    tt = int(tts_gates[oc, gate_offset + g])
                    for j in range(N):
                        child = g * N + j
                        if level == 0:
                            p = int(idxf[oc, child])
                            c, rem = divmod(p, kH * kW)
                            dy, dx = divmod(rem, kW)
                            y = oy * sH - pH + dy
                            x = ox * sW - pW + dx
                            if 0 <= y < H and 0 <= x < W:
                                src = c * H * W + y * W + x
                            else:
                                # padding: スロットを0固定でttへ畳み込み
                                tt = _specialize_zero(tt, j, N)
                                src = 0
                        else:
                            src = (oc * L + pos) * n_gates + child
                        indices[j, out_idx] = src
                    tts.append(tt)

        nodes.append(_lut_node(
            f'{node_id}_lv{level}', prev_width, width_out, N,
            indices, tts, 'LutConv2d',
        ))
        prev_width = width_out
        n_gates = g_out
        gate_offset += g_out

    return nodes, OC * L, (OC, H_out, W_out)


def _or_pool_lut_node(
    node_id: str,
    in_width: int,
    in_shape: tuple,
    kernel_size: int,
    stride: int,
    padding: int,
) -> tuple:
    """窓ORのlut_layerノードを生成する純関数(torch経路とIR経路で共有)。

    Returns:
        (lut_layerノード, 出力幅, 出力shape(C, H_out, W_out))
    """
    C, H, W = in_shape
    k = kernel_size
    s = stride
    p = padding
    n_in = k * k
    if n_in > 6:
        raise NotImplementedError(
            f'{node_id}: 窓{k}x{k}({n_in}入力)のORは未対応(≦6入力のみ)。'
            'OR木への分解は次段階'
        )
    H_out = (H + 2 * p - k) // s + 1
    W_out = (W + 2 * p - k) // s + 1
    width_out = C * H_out * W_out
    or_tt = (1 << (1 << n_in)) - 2  # 全パターン1、パターン0のみ0

    indices = np.empty((n_in, width_out), dtype=np.int64)
    tts = []
    for c in range(C):
        for oy in range(H_out):
            for ox in range(W_out):
                out_idx = c * H_out * W_out + oy * W_out + ox
                tt = or_tt
                for j in range(n_in):
                    dy, dx = divmod(j, k)
                    y = oy * s - p + dy
                    x = ox * s - p + dx
                    if 0 <= y < H and 0 <= x < W:
                        src = c * H * W + y * W + x
                    else:
                        tt = _specialize_zero(tt, j, n_in)
                        src = 0
                    indices[j, out_idx] = src
                tts.append(tt)

    node = _lut_node(node_id, in_width, width_out, n_in,
                     indices, tts, 'OrPooling2d')
    return node, width_out, (C, H_out, W_out)


def _effective_indices_flat(module: LutConv2d):
    """export用の実効入力配線 (OC, n_leaves)。

    wiring='learnable'（DWN §3.2）時は forward と同じハード規則
    leaves[o,i] = candidate[o, argmax_j mapping_weights[o,j,i]] を解決し、
    wiring_candidate_indices を argmax 選択で確定する。固定配線時は
    従来の indices_flat をそのまま返す。これを怠ると学習配線モデルの
    IRが固定配線で展開され PyTorch hard eval ⇔ simulate が不一致になる。
    """
    if getattr(module, 'wiring', 'fixed') == 'learnable':
        mapping = module.mapping_weights.argmax(dim=1)   # (OC, n_leaves)
        cand = module.wiring_candidate_indices           # (OC, n_candidates)
        return torch.gather(cand, 1, mapping)            # (OC, n_leaves)
    return module.indices_flat


@register_extractor(LutConv2d)
def _extract_lut_conv2d(module: LutConv2d, node_id: str, ctx: ExtractCtx):
    assert ctx.shape is not None, (
        f'{node_id}: conv系のexport_irには input_shape=(C,H,W) が必要'
    )
    C, _, _ = ctx.shape
    assert C == module.in_channels, (
        f'{node_id}: in_channels({module.in_channels}) != 前段C({C})'
    )
    tts_gates = truth_tables_from_conv(module)          # (OC, n_internal)
    idxf = _effective_indices_flat(module).cpu().numpy()  # (OC, n_leaves)
    return _unroll_conv_tree(
        node_id, ctx.width, ctx.shape,
        module.out_channels, module.kernel_size, module.stride,
        module.padding, module.tree_depth, module.n_inputs,
        idxf, tts_gates,
    )


@register_extractor(OrPooling2d)
def _extract_or_pooling(module: OrPooling2d, node_id: str, ctx: ExtractCtx):
    assert ctx.shape is not None, f'{node_id}: pooling系には形状追跡が必要'
    node, width_out, shape_out = _or_pool_lut_node(
        node_id, ctx.width, ctx.shape,
        module.kernel_size, module.stride, module.padding,
    )
    return [node], width_out, shape_out


# --- ネイティブ抽出器: export_ir(native_conv=True) 経路 ---

@register_native_extractor(LutConv2d)
def _extract_lut_conv2d_native(module: LutConv2d, node_id: str, ctx: ExtractCtx):
    """LutConv2d → conv2dネイティブノード(unrollしない)"""
    assert ctx.shape is not None, (
        f'{node_id}: conv系のexport_irには input_shape=(C,H,W) が必要'
    )
    C, H, W = ctx.shape
    assert C == module.in_channels, (
        f'{node_id}: in_channels({module.in_channels}) != 前段C({C})'
    )
    kH, kW = module.kernel_size
    sH, sW = module.stride
    pH, pW = module.padding
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    OC = module.out_channels

    tts_gates = truth_tables_from_conv(module)          # (OC, n_internal)
    node = IRNode(
        id=node_id,
        type='conv2d',
        params={
            'in_shape': [C, H, W],
            'out_channels': OC,
            'kernel_size': [kH, kW],
            'stride': [sH, sW],
            'padding': [pH, pW],
            'tree_depth': module.tree_depth,
            'n_inputs': module.n_inputs,
            'indices_flat':
                _effective_indices_flat(module).cpu().numpy().astype(np.int64),
            # oc優先(oc-major)で木のレベル順=LutConv2d.weightsのdim1順
            'truth_tables': tt_to_hex(
                [int(t) for t in tts_gates.reshape(-1)]
            ),
        },
        train_info={
            'parameterization': module.parameterization,
            'input_channels_per_tree': module.input_channels_per_tree,
            # 推定器は既定(sigmoid)以外のときだけ載せる(既定経路の IR は不変)
            **({'estimator': module.estimator} if module.estimator != 'sigmoid' else {}),
        },
    )
    return [node], OC * H_out * W_out, (OC, H_out, W_out)


@register_native_extractor(OrPooling2d)
def _extract_or_pooling_native(module: OrPooling2d, node_id: str,
                               ctx: ExtractCtx):
    """OrPooling2d → or_pool2dネイティブノード(unrollしない)"""
    assert ctx.shape is not None, f'{node_id}: pooling系には形状追跡が必要'
    C, H, W = ctx.shape
    k = module.kernel_size
    s = module.stride
    p = module.padding
    H_out = (H + 2 * p - k) // s + 1
    W_out = (W + 2 * p - k) // s + 1
    node = IRNode(
        id=node_id,
        type='or_pool2d',
        params={
            'in_shape': [C, H, W],
            'kernel_size': k,
            'stride': s,
            'padding': p,
        },
    )
    return [node], C * H_out * W_out, (C, H_out, W_out)


# --- IR→IR lowering ---

def lowered_output_id(node) -> str:
    """展開後グラフでの、元ノードの出力に対応するノードid。

    conv2d は最終レベルのlut_layer(id_lv{d-1})、or_pool2dは同一id。
    simulateのconvフォールバックがtraceキーの集約に使う。
    """
    if node.type == 'conv2d':
        return f"{node.id}_lv{int(node.params['tree_depth']) - 1}"
    return node.id


def _pair(v) -> tuple[int, int]:
    if isinstance(v, list | tuple):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _conv2d_node_to_luts(node) -> list:
    """conv2dネイティブノードをlut_layerノード列へ展開する"""
    p = node.params
    in_shape = tuple(int(v) for v in p['in_shape'])
    oc = int(p['out_channels'])
    d = int(p['tree_depth'])
    n = int(p['n_inputs'])
    n_leaves = n ** d
    n_internal = d if n == 1 else (n_leaves - 1) // (n - 1)
    idxf = np.asarray(p['indices_flat'], dtype=np.int64)
    tts_gates = np.array(
        tt_from_hex(p['truth_tables']), dtype=object,
    ).reshape(oc, n_internal)
    in_width = int(np.prod(in_shape))
    nodes, _, _ = _unroll_conv_tree(
        node.id, in_width, in_shape, oc,
        _pair(p['kernel_size']), _pair(p['stride']), _pair(p['padding']),
        d, n, idxf, tts_gates,
    )
    return nodes


def _or_pool2d_node_to_lut(node):
    """or_pool2dネイティブノードを窓ORのlut_layerノードへ展開する(同一id)"""
    p = node.params
    in_shape = tuple(int(v) for v in p['in_shape'])
    in_width = int(np.prod(in_shape))
    lut, _, _ = _or_pool_lut_node(
        node.id, in_width, in_shape,
        int(p['kernel_size']), int(p['stride']), int(p.get('padding', 0)),
    )
    return lut


def _upsample2d_node_to_lut(node):
    """upsample2dネイティブノードを恒等複製lut_layerノードへ展開する(同一id)。

    out[c,y,x] = in[c, y//scale, x//scale] のindexを実行時にJIT計算する
    (JSONにindexを保存しないための核心。_extract_upsample2dの旧展開と
    ビット厳密一致)。
    """
    p = node.params
    c_in, h_in, w_in = (int(v) for v in p['in_shape'])
    s = int(p['scale'])
    h_out, w_out = h_in * s, w_in * s
    width_out = c_in * h_out * w_out
    in_width = c_in * h_in * w_in

    indices = np.empty((1, width_out), dtype=np.int64)
    for c in range(c_in):
        for y in range(h_out):
            for x in range(w_out):
                out_idx = (c * h_out + y) * w_out + x
                indices[0, out_idx] = (c * h_in + y // s) * w_in + x // s

    return _lut_node(node.id, in_width, width_out, 1, indices,
                     [_IDENTITY_TT_1IN] * width_out, 'UpSample2d')


def _permute_node_to_lut(node):
    """permuteネイティブノードを恒等置換lut_layerノードへ展開する(同一id)。

    入力をin_shape形状のreshape viewとみなし、permに従って軸並べ替えした
    row-major出力のindexを実行時にJIT計算する
    (_extract_spatial_group_sumの旧展開・export_webbenchのthermo channel
    permuteとビット厳密一致)。
    """
    p = node.params
    dims = [int(v) for v in p['in_shape']]
    perm = [int(v) for v in p['perm']]
    width = int(np.prod(dims))
    out_dims = [dims[q] for q in perm]

    # 入力側strides(row-major、末尾軸のstride=1)
    in_strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        in_strides[i] = in_strides[i + 1] * dims[i + 1]

    # 出力flat位置をrow-majorでunravelし、対応する入力flat位置を合成する
    out_idx = np.arange(width)
    coords = np.unravel_index(out_idx, out_dims)
    src = np.zeros(width, dtype=np.int64)
    for i, ax in enumerate(perm):
        src += coords[i].astype(np.int64) * in_strides[ax]

    indices = src.reshape(1, width)
    return _lut_node(node.id, width, width, 1, indices,
                     [_IDENTITY_TT_1IN] * width, 'Permute')


def lower_to_primitives(graph: IRGraph) -> IRGraph:
    """ネイティブノードをlut_layer列へ展開するIR→IRパス(恒等配線ノードにも対応)。

    純関数: 入力graphは変更しない(展開対象が無ければgraphをそのまま返す)。
    HDL生成(to_verilog/to_vhdl)やリソース見積り(export/estimate.py)、
    ir/simulate.pyの内部フォールバックの前段として呼ぶ。展開後ノードの
    idは conv2d → {id}_lv{level}、それ以外(or_pool2d/upsample2d/permute)
    → 同一id(export_irの従来lowering経路と同一規約)。
    """
    if not any(n.type in LOWERABLE_TYPES for n in graph.nodes):
        return graph

    new_nodes = []
    src_remap: dict[str, str] = {}   # 元ノードid → 展開後の最終ノードid
    dst_remap: dict[str, str] = {}   # 元ノードid → 展開後の先頭ノードid
    chain_edges: list[IREdge] = []
    for node in graph.nodes:
        if node.type == 'conv2d':
            chain = _conv2d_node_to_luts(node)
            new_nodes.extend(chain)
            dst_remap[node.id] = chain[0].id
            src_remap[node.id] = chain[-1].id
            chain_edges.extend(
                IREdge(src=a.id, dst=b.id)
                for a, b in zip(chain, chain[1:], strict=False)
            )
        elif node.type == 'or_pool2d':
            new_nodes.append(_or_pool2d_node_to_lut(node))  # 同一id
        elif node.type == 'upsample2d':
            new_nodes.append(_upsample2d_node_to_lut(node))  # 同一id
        elif node.type == 'permute':
            new_nodes.append(_permute_node_to_lut(node))  # 同一id
        else:
            new_nodes.append(node)

    edges = [
        IREdge(
            src=src_remap.get(e.src, e.src),
            dst=dst_remap.get(e.dst, e.dst),
            dst_slot=e.dst_slot,
        )
        for e in graph.edges
    ] + chain_edges

    return IRGraph(
        metadata={**graph.metadata},
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        nodes=new_nodes,
        edges=edges,
        format=graph.format,
        version=graph.version,
    )


@register_extractor(SpatialGroupSum)
def _extract_spatial_group_sum(module: SpatialGroupSum, node_id: str,
                               ctx: ExtractCtx):
    assert ctx.shape is not None, f'{node_id}: SpatialGroupSumには形状追跡が必要'
    C, H, W = ctx.shape
    n_out = module.n_outputs
    assert C % n_out == 0
    gs = C // n_out
    width = C * H * W

    # ワイヤ順を (c=o*gs+g, y, x) → (o, y, x, g) へ並べ替える恒等置換
    indices = np.empty((1, width), dtype=np.int64)
    for o in range(n_out):
        for y in range(H):
            for x in range(W):
                for g in range(gs):
                    out_idx = ((o * H + y) * W + x) * gs + g
                    indices[0, out_idx] = (o * gs + g) * H * W + y * W + x

    permute = _lut_node(f'{node_id}_permute', ctx.width, width, 1,
                        indices, [_IDENTITY_TT_1IN] * width, 'SpatialGroupSum')
    group = IRNode(
        id=node_id,
        type='group_sum',
        params={
            'in_width': width,
            'k': n_out * H * W,
            # scaleは実数係数。整数スコアの比較には影響しないため参考情報
            'tau': 1.0 / module.scale if module.scale != 0 else 1.0,
        },
        train_info={'lowered_from': 'SpatialGroupSum'},
    )
    return [permute, group], n_out * H * W, (n_out, H, W)


@register_native_extractor(SpatialGroupSum)
def _extract_spatial_group_sum_native(module: SpatialGroupSum, node_id: str,
                                      ctx: ExtractCtx):
    """SpatialGroupSum → permuteネイティブノード + group_sum(unrollしない)。

    ワイヤ順を (c=o*gs+g, y, x) → (o, y, x, g) へ並べ替える恒等置換は、
    入力を(n_out, gs, H, W)のreshape viewとみなしperm=[0,2,3,1]の
    permuteノードで表現できる(_extract_spatial_group_sumの旧lut_layer
    展開とビット厳密一致。indexはJSONへ保存せず実行時JIT計算)。
    """
    assert ctx.shape is not None, f'{node_id}: SpatialGroupSumには形状追跡が必要'
    C, H, W = ctx.shape
    n_out = module.n_outputs
    assert C % n_out == 0
    gs = C // n_out
    width = C * H * W

    permute = IRNode(
        id=f'{node_id}_permute',
        type='permute',
        params={'in_shape': [n_out, gs, H, W], 'perm': [0, 2, 3, 1]},
        train_info={'from': 'SpatialGroupSum'},
    )
    group = IRNode(
        id=node_id,
        type='group_sum',
        params={
            'in_width': width,
            'k': n_out * H * W,
            # scaleは実数係数。整数スコアの比較には影響しないため参考情報
            'tau': 1.0 / module.scale if module.scale != 0 else 1.0,
        },
        train_info={'lowered_from': 'SpatialGroupSum'},
    )
    return [permute, group], n_out * H * W, (n_out, H, W)


@torch.no_grad()
def truth_tables_from_bb_block(block: DifferentiableLutBlock) -> np.ndarray:
    """DifferentiableLutBlockの合成真理値表を抽出する。

    Unbinarize→LUT(raw連続)→BatchNorm(running統計)→Binarize の合成を、
    ブロック自身のeval()に全2^Nパターンを流すプローブで厳密に評価する
    (BN折り畳みの数式を複製しない)。BNのrunning統計が必要なため、
    未訓練ブロック(初期統計)でも動作はするが意味を持つのは訓練後。
    """
    n = block.lut.n_inputs
    out_dim = block.lut.out_dim

    probe = copy.deepcopy(block).to('cpu')
    probe.eval()
    probe.lut.indices = torch.arange(n, dtype=torch.int64).unsqueeze(1) \
        .expand(n, out_dim).contiguous()
    probe.lut.in_dim = n

    patterns = torch.tensor(
        [[float((k >> j) & 1) for j in range(n)] for k in range(2 ** n)]
    )
    outs = probe(patterns)  # (2^n, out_dim) — Binarize済みで厳密に{0,1}
    bits = (outs > 0.5).to(torch.int64)

    tables = []
    for o in range(out_dim):
        t = 0
        for k in range(2 ** n):
            t |= int(bits[k, o].item()) << k
        tables.append(t)
    return np.array(tables, dtype=object)


@register_extractor(DifferentiableLutBlock)
def _extract_bb_block(module: DifferentiableLutBlock, node_id: str,
                      ctx: ExtractCtx):
    assert module.lut.in_dim == ctx.width, (
        f'{node_id}: in_dim({module.lut.in_dim}) != 前段幅({ctx.width})'
    )
    tables = truth_tables_from_bb_block(module)
    node = _lut_node(
        node_id, ctx.width, module.lut.out_dim, module.lut.n_inputs,
        module.lut.indices.cpu().numpy().astype(np.int64),
        tables.tolist(), 'DifferentiableLutBlock',
    )
    return [node], module.lut.out_dim, None


@register_extractor(UpSample2d)
def _extract_upsample2d(module: UpSample2d, node_id: str, ctx: ExtractCtx):
    """UpSample2d(nearest) → 恒等置換lut_layer。

    nearest拡大は out[c,y,x] = in[c, y//s, x//s] の固定配線(複製)なので、
    パススルーゲート(n_inputs=1, tt=0x2)のインデックス複製で表現する
    (SpatialGroupSumの恒等置換loweringと同一手法)。HDL上は配線複製になる。
    """
    assert ctx.shape is not None, f'{node_id}: UpSample2dには形状追跡が必要'
    c_in, h_in, w_in = ctx.shape
    s = int(module.scale_factor)
    h_out, w_out = h_in * s, w_in * s
    width_out = c_in * h_out * w_out

    indices = np.empty((1, width_out), dtype=np.int64)
    for c in range(c_in):
        for y in range(h_out):
            for x in range(w_out):
                out_idx = (c * h_out + y) * w_out + x
                indices[0, out_idx] = (c * h_in + y // s) * w_in + x // s

    node = _lut_node(node_id, ctx.width, width_out, 1, indices,
                     [_IDENTITY_TT_1IN] * width_out, 'UpSample2d')
    return [node], width_out, (c_in, h_out, w_out)


@register_native_extractor(UpSample2d)
def _extract_upsample2d_native(module: UpSample2d, node_id: str,
                               ctx: ExtractCtx):
    """UpSample2d(nearest) → upsample2dネイティブノード(unrollしない)。

    params(in_shape+scale)のみでout[c,y,x]=in[c,y//s,x//s]を表現でき、
    indexはJSONへ保存せず実行時JIT計算する(_extract_upsample2dの旧
    lut_layer展開とビット厳密一致)。
    """
    assert ctx.shape is not None, f'{node_id}: UpSample2dには形状追跡が必要'
    c_in, h_in, w_in = ctx.shape
    s = int(module.scale_factor)
    h_out, w_out = h_in * s, w_in * s
    node = IRNode(
        id=node_id,
        type='upsample2d',
        params={'in_shape': [c_in, h_in, w_in], 'scale': s},
        train_info={'from': 'UpSample2d'},
    )
    return [node], c_in * h_out * w_out, (c_in, h_out, w_out)


@register_extractor(BinaryBrainConv2d)
def _extract_bb_conv2d(module: BinaryBrainConv2d, node_id: str,
                       ctx: ExtractCtx):
    """BinaryBrainConv2d → ブロックごとのlut_layer列へ展開する。

    forward(bb_conv2d.py)の im2col → DifferentiableLutBlock連鎖 → col2im を、
    窓(空間位置pos)ごとに平坦化したindicesで表現する(stride=1固定):
        第1ブロック: 受容野フラット q = c*kH*kW + dy*kW + dx を入力画素
            (y,x) = (oy−pad+dy, ox−pad+dx) へ写像。範囲外は
            _specialize_zeroでttへ畳み込む(F.unfoldの0パディングと厳密一致)
        中間ブロック: 位置優先(pos-major)順 pos*dim + f で前段と接続
        最終ブロック: col2imのNCHW順 o*L + pos へ並べ替えて出力
    各ブロックの真理値表は truth_tables_from_bb_block のプローブで確定する
    (BatchNormなし最終ブロック(_DifferentiableLutBlockNoBN)も同じ
    プローブで畳み込める)。
    """
    assert ctx.shape is not None, (
        f'{node_id}: conv系のexport_irには input_shape=(C,H,W) が必要'
    )
    if module._lut_blocks is None:
        raise NotImplementedError(
            f'{node_id}: BinaryBrainConv2dが未構築(in_channels未確定)。'
            'in_channelsを指定して構築するか、forwardを一度実行すること'
        )
    c_in, h_in, w_in = ctx.shape
    assert module.in_channels == c_in, (
        f'{node_id}: in_channels({module.in_channels}) != 前段C({c_in})'
    )
    k = module.kernel_size
    pad = k // 2 if module.padding == 'same' else int(module.padding)
    h_out = h_in + 2 * pad - k + 1
    w_out = w_in + 2 * pad - k + 1
    n_pos = h_out * w_out  # = L(im2colの窓数)

    nodes = []
    prev_width = ctx.width
    prev_dim = 0  # 第1ブロックでは未使用(受容野から直接接続する)
    n_blocks = len(module._lut_blocks)
    for i, block in enumerate(module._lut_blocks):
        n = block.lut.n_inputs
        out_dim = block.lut.out_dim
        idx_blk = block.lut.indices.cpu().numpy().astype(np.int64)  # (n, out)
        tts_blk = truth_tables_from_bb_block(block)                 # (out,)
        is_last = (i == n_blocks - 1)

        width_out = out_dim * n_pos
        indices = np.empty((n, width_out), dtype=np.int64)
        tts = [0] * width_out
        for pos in range(n_pos):
            oy, ox = divmod(pos, w_out)
            for f in range(out_dim):
                # 最終ブロックのみcol2imのNCHW順(o*L+pos)へ並べ替える
                out_idx = f * n_pos + pos if is_last else pos * out_dim + f
                tt = int(tts_blk[f])
                for j in range(n):
                    q = int(idx_blk[j, f])
                    if i == 0:
                        c, rem = divmod(q, k * k)
                        dy, dx = divmod(rem, k)
                        y = oy - pad + dy
                        x = ox - pad + dx
                        if 0 <= y < h_in and 0 <= x < w_in:
                            src = (c * h_in + y) * w_in + x
                        else:
                            # padding: スロットを0固定でttへ畳み込み
                            tt = _specialize_zero(tt, j, n)
                            src = 0
                    else:
                        src = pos * prev_dim + q
                    indices[j, out_idx] = src
                tts[out_idx] = tt

        nodes.append(_lut_node(f'{node_id}_b{i}', prev_width, width_out, n,
                               indices, tts, 'BinaryBrainConv2d'))
        prev_width = width_out
        prev_dim = out_dim

    out_channels = module.out_channels
    return nodes, out_channels * n_pos, (out_channels, h_out, w_out)


# ---------------------------------------------------------------------------
# DWNConv2d（レバー4: 離散のまま学習するRAMノードの畳み込み版）
# ---------------------------------------------------------------------------
def _dwn_truth_table(lut_row: np.ndarray) -> int:
    """表の実数値を真理値表（整数ビット列）へ。

    DWN の推論は「表を引いて >0 で二値化」。表の各エントリ i が正なら
    真理値表のビット i を立てればよく、アドレスの並び（LSB=スロット0）は
    ``dwn_conv.DWNConv2d`` と ``layers/dwn.py`` で共通。
    """
    tt = 0
    for i, v in enumerate(lut_row):
        if v > 0:
            tt |= 1 << i
    return tt


@register_extractor(DWNConv2d)
def _extract_dwn_conv2d(module: DWNConv2d, node_id: str, ctx: ExtractCtx):
    """DWNConv2d を lut_layer 1枚へ展開する。

    木構造（LutConv2d）は深さdぶんの lut_layer に分かれるが、DWN は
    出力チャネルあたり RAMノード1個なので **1枚で済む**。
    受容野のはみ出し（padding）は、木と同じく該当スロットを定数0に
    畳み込んだ真理値表へ特殊化することで表現する。
    """
    assert ctx.shape is not None, f'{node_id}: input_shape=(C,H,W) が必要'
    C, H, W = ctx.shape
    assert C == module.in_channels, (node_id, C, module.in_channels)

    kH, kW = module.kernel_size
    sH, sW = module.stride
    pH, pW = module.padding
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    L = H_out * W_out
    OC = module.out_channels
    n = module.n_inputs

    taps = module.taps.detach().cpu().numpy()          # (OC, n)
    luts = module.luts.detach().cpu().numpy()          # (OC, 2^n)
    base_tt = [_dwn_truth_table(luts[o]) for o in range(OC)]

    in_width = C * H * W
    out_width = OC * L
    indices = np.empty((n, out_width), dtype=np.int64)
    tts = []
    for oc in range(OC):
        for pos in range(L):
            oy, ox = divmod(pos, W_out)
            out_idx = oc * L + pos
            tt = base_tt[oc]
            for j in range(n):
                p = int(taps[oc, j])
                c, rem = divmod(p, kH * kW)
                dy, dx = divmod(rem, kW)
                y = oy * sH - pH + dy
                x = ox * sW - pW + dx
                if 0 <= y < H and 0 <= x < W:
                    src = c * H * W + y * W + x
                else:
                    # padding: そのスロットを定数0に固定した表へ特殊化
                    tt = _specialize_zero(tt, j, n)
                    src = 0
                indices[j, out_idx] = src
            tts.append(tt)

    node = _lut_node(node_id, in_width, out_width, n, indices, tts, 'DWNConv2d')
    return [node], out_width, (OC, H_out, W_out)
