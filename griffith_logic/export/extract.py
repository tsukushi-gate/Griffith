"""PyTorchモデル → IR抽出器(形状追跡+lowering対応)。

レジストリ方式: モジュール型ごとの抽出関数を @register_extractor で
登録し、nn.Sequential の子を順に走査してIRノード列を構築する。
lookupはMRO走査(exact一致が最優先)なので、サブクラス
(例: LutConv2dHardST)は親クラスの抽出器で処理される。

抽出関数のシグネチャ(v0.2):
    fn(module, node_id, ctx: ExtractCtx) -> (list[IRNode], new_width, new_shape)

対応範囲:
    dense系: LutLayer / GroupSum / Binarize / nn.Flatten /
             DWNLayer / LearnableMapping(真理値表・接続の確定)
    conv系(export/lowering.py): LutConv2d / OrPooling2d / SpatialGroupSum /
                                 DifferentiableLutBlock(BN折り畳み) /
                                 UpSample2d / BinaryBrainConv2d

padding>0のconv/poolは「真理値表の特殊化」で処理する
(export/lowering.py参照。スキーマ拡張・定数線は不要)。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import torch.nn as nn

from ..binarize import Binarize
from ..encoding.thermometer import ThermometerEncoder
from ..group_sum import GroupSum
from ..ir.schema import IREdge, IRGraph, IRNode, IRPort
from ..ir.serialize import tt_to_hex
from ..layers.dwn import DWNLayer, LearnableMapping
from ..layers.fixed_lut import FixedLutLayer
from ..layers.ltt import LTTLayer
from ..layers.lut import LutLayer
from ..layers.permute import Permute
from .discretize import (
    truth_tables_from_dwn,
    truth_tables_from_ltt,
    truth_tables_from_lut_layer,
)

JST = timezone(timedelta(hours=9))

# 恒等1入力LUT: bit0=0→0, bit1=1→1(固定配線をlut_layerで表す際の共通定数)
_IDENTITY_TT_1IN = 0x2

EXTRACTORS: dict[type, callable] = {}

# ネイティブ抽出器: conv/pool等をlowering展開せずネイティブIRノード
# (conv2d/or_pool2d)として抽出する。export_ir(native_conv=True)時のみ
# EXTRACTORSより優先して参照される(既定Falseで従来lowering経路を維持)
NATIVE_EXTRACTORS: dict[type, callable] = {}

# ============================================================
# 抽出方針表
#   公開nn.Module型(griffith_logic._SYMBOL_MODULES)は、
#   EXTRACTORS/NATIVE_EXTRACTORS登録(MRO込み)があるか、以下のいずれかの
#   表に載っていなければならない(tests/test_extractor_coverage_matrix.py
#   が機械検査する。未分類の新型を公開するとテストが落ちる)。
# 型オブジェクトではなくクラス名文字列で持つ(遅延インポート方針
#   崩さず、export/extract.pyのimportを軽く保つため)
# ============================================================

# blackbox方針: softmax/argmax等の実数演算や特徴横断の正規化を含み、
# 真理値表へ確定できない(HDL化不能)。blocks_to_ir がblackboxノード
# (+contract)として痕跡を残す(blocks/to_ir.py参照)
BLACKBOX_TYPES = (
    'LogicAttention',        # softmax/類似度計算を含む(blocks/attention.py参照)
    'LogicTransformerBlock',  # LogicAttentionを内包する複合部品
    'LogicLayerNorm',        # 特徴横断の正規化のためlut折り畳み不能
    'LogicBatchNorm',        # 単独では折り畳み不能(LutLayer直後のみfolding対象)
    'ConditionalSum',        # 符号化値のfp32/量子化matmul(実数演算のため真理値表化不能)
)

# 抽出対象外の完成モデル/ハイブリッド複合部品。
# IR化するときは内部の論理部分(LutLayer列等)のみをexport_irに渡す
MODEL_TYPES = (
    'RDDLGN',
    'RDDLGNEncoder',
    'RDDLGNDecoder',
    'RDDLGNClassifier',      # RDDLGNEncoder+分類ヘッドの完成モデル
    'LogicGPT',              # decoder-only 因果Transformer char-LM
    'LutStack',              # RDDLGN内部のLutLayer連鎖(モデル部品)
    'DWNClassifier',         # ThermometerEncoder境界を内包する完成モデル
    'LogicLinear',           # ToLogic/FromLogic境界を内包するハイブリッド層
    'LogicMLP',
    'DLGNBooleanAttention',
    'DLGNMultiQueryAttention',
    'DLGNPositionalAttention',
    'DLGNAttentionSeq2Seq',
    'DLGNPointerAttention',
    'DLGNAttentionTranslator',
    'LLViT',                 # LL-ViT完成モデル(MHA+論理チャネルミキサの複合)
    'LLViTChannelMixer',     # thermometer境界+DWN論理+ConditionalSumのハイブリッド複合
)

# 実数⇔論理の境界(encoding系)。回路の外側の前処理/後処理であり、
# IRでは入出力ポートのencoding属性の領分(ノードにしない)。
# ThermometerEncoder は専用の抽出器を持つ(input_bits 指定時に B 入力 lut_layer。
# 未指定なら従来どおり境界として NotImplementedError)ので、この表には載せない
BOUNDARY_TYPES = (
    'ToLogic',
    'FromLogic',
    'RealToBinary',
    'BinaryToReal',
    'LogicEmbedding',        # トークンID→ビット列の境界(RDDLGN用)
)

# グラフ構造部品: blocks_to_ir がconcat/add/constantノード等として
# 直接処理する(直列走査のexport_irでは意味を持たない。blocks/to_ir.py参照)
STRUCTURAL_TYPES = (
    'Concat',
    'Add',
    'Residual',
    'LogicPositionalEncoding',  # constant+concatノードへ展開
    'LogicDropout',             # eval時identity(IRに痕跡なし)
)


@dataclass
class ExtractCtx:
    """抽出走査の状態"""
    width: int                      # 現在の論理幅
    shape: tuple | None             # 4D経路の(C, H, W)。平坦ならNone


def register_extractor(module_type: type):
    """モジュール型 → IRノード抽出関数を登録するデコレータ"""
    def deco(fn):
        EXTRACTORS[module_type] = fn
        return fn
    return deco


def register_native_extractor(module_type: type):
    """モジュール型 → ネイティブIRノード抽出関数を登録するデコレータ"""
    def deco(fn):
        NATIVE_EXTRACTORS[module_type] = fn
        return fn
    return deco


def find_extractor(module: nn.Module, *, native: bool = False):
    """MRO順に抽出器を探す(exact一致が最優先、次に近い親クラス)。

    native=Trueならネイティブ抽出器(NATIVE_EXTRACTORS)を優先し、
    見つからなければ従来のEXTRACTORSへフォールバックする。
    未登録型はNoneを返し、呼び出し側が従来どおりエラーにする。
    """
    for cls in type(module).__mro__:
        if native and cls in NATIVE_EXTRACTORS:
            return NATIVE_EXTRACTORS[cls]
        if cls in EXTRACTORS:
            return EXTRACTORS[cls]
    return None


@register_extractor(LutLayer)
def _extract_lut_layer(module: LutLayer, node_id: str, ctx: ExtractCtx):
    assert module.in_dim == ctx.width, (
        f'{node_id}: in_dim({module.in_dim}) != 前段幅({ctx.width})'
    )
    tables = truth_tables_from_lut_layer(module)
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': module.in_dim,
            'out_width': module.out_dim,
            'n_inputs': module.n_inputs,
            # ndarrayのまま保持(serializeが大配列をnpzサイドカーへ退避する)
            'indices': module.indices.cpu().numpy().astype(np.int64),
            'truth_tables': tt_to_hex(tables.tolist()),
        },
        train_info={
            'parameterization': module.parameterization,
            'selection': module.selection,
        },
    )
    return [node], module.out_dim, None


def _group_sum_stride_permutation(width: int, k: int) -> np.ndarray:
    """stride 割当(ニューロン i → クラス i mod k)を contig の group_sum に写す転置の添字。

    出力位置 c·m + j(クラス c の j 番目)← 入力位置 j·k + c(m = width / k)。
    (m, k) → (k, m) の転置と同じ。(1, width) の int64。
    """
    m = width // k
    src = np.empty(width, dtype=np.int64)
    for c in range(k):
        for j in range(m):
            src[c * m + j] = j * k + c
    return src.reshape(1, width)


def _group_sum_node(module: GroupSum, node_id: str, ctx: ExtractCtx,
                    train_info: dict | None = None) -> IRNode:
    params = {'in_width': ctx.width, 'k': module.k, 'tau': float(module.tau)}
    # 判定バイアス。全 0 / 未設定なら params に出さない(従来の IR と同一)
    bias = module.bias_ints() if hasattr(module, 'bias_ints') else None
    if bias is not None:
        params['bias'] = bias
    node = IRNode(id=node_id, type='group_sum', params=params)
    if train_info:
        node.train_info = train_info
    return node


@register_extractor(GroupSum)
def _extract_group_sum(module: GroupSum, node_id: str, ctx: ExtractCtx):
    """GroupSum → group_sum ノード。

    assign='stride'は手前に恒等 lut_layer(n_inputs=1)の転置 (m, k) → (k, m)
    を挟み、group_sum 自体は contig のまま(SpatialGroupSum の非ネイティブ抽出と同型。
    ネイティブ版は permute ノード)。assign='contig'(既定)の IR は従来と同一。
    """
    if getattr(module, 'assign', 'contig') != 'stride':
        return [_group_sum_node(module, node_id, ctx)], module.k, None
    width = ctx.width
    assert width % module.k == 0, (
        f'{node_id}: 入力幅({width})はクラス数({module.k})で割り切れる必要があります'
    )
    permute = IRNode(
        id=f'{node_id}_permute',
        type='lut_layer',
        params={
            'in_width': width,
            'out_width': width,
            'n_inputs': 1,
            'indices': _group_sum_stride_permutation(width, module.k),
            'truth_tables': tt_to_hex([_IDENTITY_TT_1IN] * width),
        },
        train_info={'lowered_from': 'GroupSum(assign=stride)'},
    )
    group = _group_sum_node(module, node_id, ctx,
                            train_info={'lowered_from': 'GroupSum(assign=stride)'})
    return [permute, group], module.k, None


@register_native_extractor(GroupSum)
def _extract_group_sum_native(module: GroupSum, node_id: str, ctx: ExtractCtx):
    """GroupSum(assign='stride') → permute ネイティブノード + group_sum。

    入力を (m, k) の reshape view とみなし perm=[1, 0] で (k, m) に転置する
    (非ネイティブ版の恒等 lut_layer 展開とビット厳密一致。添字は実行時 JIT 計算)。
    assign='contig' は非ネイティブ版と同じ group_sum 1 個。
    """
    if getattr(module, 'assign', 'contig') != 'stride':
        return _extract_group_sum(module, node_id, ctx)
    width = ctx.width
    assert width % module.k == 0, (
        f'{node_id}: 入力幅({width})はクラス数({module.k})で割り切れる必要があります'
    )
    permute = IRNode(
        id=f'{node_id}_permute',
        type='permute',
        params={'in_shape': [width // module.k, module.k], 'perm': [1, 0]},
        train_info={'from': 'GroupSum(assign=stride)'},
    )
    group = _group_sum_node(module, node_id, ctx,
                            train_info={'lowered_from': 'GroupSum(assign=stride)'})
    return [permute, group], module.k, None


@register_extractor(Binarize)
def _extract_binarize(module: Binarize, node_id: str, ctx: ExtractCtx):
    node = IRNode(
        id=node_id,
        type='binarize',
        params={'width': ctx.width, 'threshold': float(module.threshold)},
    )
    # 要素ごとの閾値処理なので幅・形状は前段からそのまま伝播する
    return [node], ctx.width, ctx.shape


@register_extractor(DWNLayer)
def _extract_dwn_layer(module: DWNLayer, node_id: str, ctx: ExtractCtx):
    """DWNLayer → lut_layerノード。

    DWNのeval意味論(layers/dwn.py):
        addr = Σ_j 1[x[m(j,u)] > 0] << j    (gather+アドレス化、LSB=スロット0)
        y_u  = 1[luts[u, addr] > 0]          (BinarizeSTE、閾値0)
    は、二値入力{0,1}では「>0」=ビット値そのものなので、
    tt_u の bit k = 1[luts[u,k] > 0](truth_tables_from_dwn)とした
    lut_layer意味論 y_u = (tt_u >> addr) & 1 と厳密一致する。
    mapping='learnable'の接続は argmax(W) で確定する
    (LearnableMappingFunction.forwardのハード選択と同一規則)。
    """
    assert module.in_bits == ctx.width, (
        f'{node_id}: in_bits({module.in_bits}) != 前段幅({ctx.width})'
    )
    if not module.ste:
        raise NotImplementedError(
            f'{node_id}: ste=FalseのDWNLayerは実数出力のため論理IRへ抽出できない'
        )
    if module.mapping_mode == 'learnable':
        # mapping[q] = argmax(W[:, q])、q = u*n_inputs + j → indices[j, u]
        mapping = module.learnable_mapping.weights.detach().argmax(dim=0)
        indices = mapping.reshape(module.n_luts, module.n_inputs).T \
            .cpu().numpy().astype(np.int64)
    else:
        indices = module.indices.cpu().numpy().astype(np.int64)
    tables = truth_tables_from_dwn(module)
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': module.in_bits,
            'out_width': module.n_luts,
            'n_inputs': module.n_inputs,
            'indices': indices,
            'truth_tables': tt_to_hex(tables.tolist()),
        },
        train_info={
            'kind': 'dwn',
            'mapping': module.mapping_mode,
            'efd': module.efd_mode,
        },
    )
    return [node], module.n_luts, None


@register_extractor(ThermometerEncoder)
def _extract_thermometer(module: ThermometerEncoder, node_id: str, ctx: ExtractCtx):
    """ThermometerEncoder → B 入力 lut_layer。

    IR の入力ポートは特徴 d ごとの B = input_bits ビット符号 c(LSB 先頭、位置 d·B + j)で、
    実数 q = c / (2^B − 1)(ToTensor の uint8 → [0, 1])。出力 (d, i)(位置 d·z + i、
    forward の (..., D·z) 平坦順と同じ)の真理値表は bit c = 1[q(c) > t_{i,d}]
    (比較は forward と同じ float32・厳密な >)。distributive は fit 済みが必要。
    input_bits=None は従来どおり境界(IR の外)として NotImplementedError。
    """
    import torch

    b = module.input_bits
    if b is None:
        raise NotImplementedError(
            f'{node_id}: ThermometerEncoder は input_bits 未設定なので境界(IR の外)。'
            'IR に含めるには ThermometerEncoder(bits, mode, input_bits=8) のように'
            '入力の量子化ビット数を指定すること'
        )
    if ctx.width % b != 0:
        raise ValueError(
            f'{node_id}: 入力幅({ctx.width})は input_bits({b})の倍数が必要'
            '(特徴ごとに B ビット符号)'
        )
    n_feat = ctx.width // b
    z = module.bits
    thresholds = module._resolve_thresholds(torch.zeros(1, n_feat, dtype=torch.float32))
    thresholds = thresholds.detach().to(torch.float32)                 # (z, D)
    full = (1 << b) - 1
    codes = torch.arange(1 << b, dtype=torch.float32) / float(full)   # (2^B,)
    hit = (codes.unsqueeze(0).unsqueeze(0) > thresholds.unsqueeze(-1)).numpy()   # (z, D, 2^B)
    indices = np.empty((b, n_feat * z), dtype=np.int64)
    tables = []
    for d in range(n_feat):
        for i in range(z):
            o = d * z + i
            indices[:, o] = d * b + np.arange(b)
            # 2^B ビットの真理値表は Python int で組む(B=8 で 256 bit。uint64 に収まらない)
            tables.append(sum(1 << c for c in np.flatnonzero(hit[i, d]).tolist()))
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': ctx.width,
            'out_width': n_feat * z,
            'n_inputs': b,
            'indices': indices,
            'truth_tables': tt_to_hex(tables),
        },
        train_info={'kind': 'thermometer', 'mode': module.mode, 'input_bits': b},
    )
    return [node], n_feat * z, None


@register_extractor(LTTLayer)
def _extract_ltt_layer(module: LTTLayer, node_id: str, ctx: ExtractCtx):
    """LTTLayer(線形閾値ゲート)→ lut_layer ノード。

    eval 意味論 y_u = 1[Σ_j w[u,j]·x[idx[j,u]] + b[u] > 0] を、全 2^n パターンの列挙
    (truth_tables_from_ltt。bit k の bit j = 入力スロット j)で真理値表に焼く。
    connections='learnable' の配線は argmax(W) で確定する(DWNLayer と同じ規則)。
    """
    assert module.in_dim == ctx.width, (
        f'{node_id}: in_dim({module.in_dim}) != 前段幅({ctx.width})'
    )
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': module.in_dim,
            'out_width': module.out_dim,
            'n_inputs': module.n_inputs,
            'indices': module.effective_indices().cpu().numpy().astype(np.int64),
            'truth_tables': tt_to_hex(truth_tables_from_ltt(module).tolist()),
        },
        train_info={
            'kind': 'ltt',
            'connections': module.mapping_mode,
            'ste': module.ste,
        },
    )
    return [node], module.out_dim, None


@register_extractor(LearnableMapping)
def _extract_learnable_mapping(module: LearnableMapping, node_id: str,
                               ctx: ExtractCtx):
    """LearnableMapping単体 → 恒等置換lut_layer。

    eval意味論 y_i = x[argmax(W[:,i])] は固定配線(複製あり)なので、
    パススルーゲート(n_inputs=1, tt=0x2)のインデックス選択で表現する。
    """
    assert module.in_features == ctx.width, (
        f'{node_id}: in_features({module.in_features}) != 前段幅({ctx.width})'
    )
    mapping = module.weights.detach().argmax(dim=0) \
        .cpu().numpy().astype(np.int64)
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': module.in_features,
            'out_width': module.out_features,
            'n_inputs': 1,
            'indices': mapping.reshape(1, -1),
            'truth_tables': tt_to_hex([_IDENTITY_TT_1IN] * module.out_features),
        },
        train_info={'kind': 'learnable_mapping'},
    )
    return [node], module.out_features, None


def _node_out_width(node: IRNode) -> int:
    if node.type == 'lut_layer':
        return node.params['out_width']
    if node.type == 'group_sum':
        return node.params['k']
    if node.type in ('binarize', 'flatten'):
        return node.params['width']
    return node.params['out_width']


def export_ir(
    model: nn.Module,
    input_width: int | None = None,
    *,
    input_shape: tuple | None = None,
    name: str = 'model',
    metadata: dict | None = None,
    native_conv: bool = False,
) -> IRGraph:
    """nn.Sequential直列モデルをIRへ変換する。

    Args:
        model: nn.Sequential(またはiterableなModule列)
        input_width: 入力ビット幅(平坦入力のモデル用)
        input_shape: 入力形状(C, H, W)(conv系モデル用。widthは自動計算)
        name: metadata.name
        native_conv: TrueならLutConv2d/OrPooling2dをネイティブノード
            (conv2d/or_pool2d)として抽出する。
            既定False=従来のlowering展開(lut_layer列)で後方互換
    """
    # loweringモジュールのextractor登録を確実に行う(循環import回避の遅延)
    from . import lowering  # noqa: F401

    assert (input_width is None) != (input_shape is None), (
        'input_width か input_shape のどちらか一方を指定すること'
    )
    if input_shape is not None:
        input_shape = tuple(input_shape)
        real_width = int(np.prod(input_shape))
    else:
        real_width = input_width

    modules = list(model) if isinstance(model, nn.Sequential) else [model]

    ctx = ExtractCtx(width=real_width, shape=input_shape)

    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    prev_id = 'x'

    for i, module in enumerate(modules):
        if isinstance(module, nn.Flatten):
            ctx.shape = None  # NCHWフラット順 == torch.flatten順(幅不変)
            continue
        extractor = find_extractor(module, native=native_conv)
        node_id = f'n{i}_{type(module).__name__.lower()}'
        if extractor is None:
            raise NotImplementedError(
                f'{node_id}: {type(module).__name__} の抽出器が未登録。'
                'register_extractor で追加するか、論理部分のみを渡すこと'
            )
        new_nodes, new_width, new_shape = extractor(module, node_id, ctx)
        for node in new_nodes:
            nodes.append(node)
            edges.append(IREdge(src=prev_id, dst=node.id))
            prev_id = node.id
        ctx.width = new_width
        ctx.shape = new_shape

    edges.append(IREdge(src=prev_id, dst='y'))

    graph_metadata = {
        'name': name,
        'created_at': datetime.now(JST).isoformat(),
        **(metadata or {}),
    }
    in_port_width = real_width

    last = nodes[-1] if nodes else None
    out_semantics = 'class_scores' if last and last.type == 'group_sum' else None

    return IRGraph(
        metadata=graph_metadata,
        inputs=[IRPort(id='x', width=in_port_width,
                       encoding='binary_threshold_0.5')],
        outputs=[IRPort(id='y', width=ctx.width, semantics=out_semantics)],
        nodes=nodes,
        edges=edges,
    )


# --- build_torch が組む凍結部品の抽出器。build → export_ir の往復でノードが不変 ---

@register_extractor(FixedLutLayer)
def _extract_fixed_lut_layer(module: FixedLutLayer, node_id: str, ctx: ExtractCtx):
    """FixedLutLayer → lut_layer。params と train_info を組んだときのまま IR へ戻す。

    ir/build.py が LutLayer の受けない幅(thermometer の 8 入力、stride 割当の
    1 入力恒等)や kind='thermometer' を凍結 LUT で復元したものを、エディタの
    build(再抽出)で元のノードに戻すための対(ついで)の抽出器。
    """
    assert module.in_dim == ctx.width, (
        f'{node_id}: in_dim({module.in_dim}) != 前段幅({ctx.width})'
    )
    node = IRNode(
        id=node_id,
        type='lut_layer',
        params={
            'in_width': module.in_dim,
            'out_width': module.out_dim,
            'n_inputs': module.n_inputs,
            'indices': module.indices.cpu().numpy().astype(np.int64),
            'truth_tables': tt_to_hex(module.truth_tables),
        },
        train_info=dict(module.train_info),
    )
    return [node], module.out_dim, None


@register_extractor(Permute)
def _extract_permute(module: Permute, node_id: str, ctx: ExtractCtx):
    """Permute → permute ノード(恒等配線。HDL 生成前は lowering が lut_layer へ展開する)。"""
    assert module.width == ctx.width, (
        f'{node_id}: 入力幅({module.width}) != 前段幅({ctx.width})'
    )
    node = IRNode(
        id=node_id,
        type='permute',
        params={'in_shape': list(module.in_shape), 'perm': list(module.perm)},
    )
    return [node], module.width, module.out_shape


def _nchw_adapter_extractor(native: bool):
    """NCHWAdapter(build_torch が conv2d / or_pool2d を包む平坦⇔NCHW の橋)の抽出器。

    内包モジュールの抽出器にそのまま委ね、ctx.shape が無ければ adapter の in_shape を
    使う(エディタの build は input_width だけで再抽出するため)。ノード id は内包
    モジュールの型名で付け直す(n{i}_lutconv2d 等。export_ir の規約と同じ)。
    """
    def extractor(module, node_id: str, ctx: ExtractCtx):
        inner = module.module
        in_shape = tuple(int(v) for v in module.in_shape)
        if ctx.shape is None:
            assert ctx.width == int(np.prod(in_shape)), (
                f'{node_id}: 前段幅({ctx.width}) != in_shape の積({in_shape})'
            )
            ctx.shape = in_shape
        else:
            assert tuple(ctx.shape) == in_shape, (
                f'{node_id}: 前段形状({tuple(ctx.shape)}) != in_shape({in_shape})'
            )
        fn = find_extractor(inner, native=native)
        if fn is None:
            raise NotImplementedError(
                f'{node_id}: {type(inner).__name__} の抽出器が未登録')
        inner_id = node_id.replace('nchwadapter', type(inner).__name__.lower())
        return fn(inner, inner_id, ctx)
    return extractor


# build.py は本モジュールを import しないので、ここから import しても循環しない
from ..ir.build import NCHWAdapter  # noqa: E402

register_extractor(NCHWAdapter)(_nchw_adapter_extractor(native=False))
register_native_extractor(NCHWAdapter)(_nchw_adapter_extractor(native=True))
