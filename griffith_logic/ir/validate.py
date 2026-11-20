"""IRの検証パス(v0.2でDAG対応)。HDL生成前に必ず実行する。

検証項目:
    1. format/version/ノード型
    2. エッジ端点の存在・slot整合(重複/欠番/範囲外、ノード型別arity表)
    3. エッジ幅整合(srcのout_width == dstのslot別in_width)
    4. indices境界(< in_width)
    5. 真理値表値域(< 2^(2^N))
    6. DAG検証(トポロジカルソートで循環検出)
    7. 未使用ノード(警告)。fanout(同一srcから複数エッジ)は許容
    8. group_sumの in_width % k == 0、concat/addのparams整合
    9. HDLターゲット時: blackbox・add(mode='sum')が存在すればエラー
   10. conv2d/or_pool2dのshape伝播(in_shapeから出力shape計算→幅整合。
       幅規約 width=C*H*W flatten)
   11. upsample2d(in_shape+scale)/permute(in_shape+permの妥当な順列)の
       params整合(indexは持たずJIT計算するため実行時ではなくここで
       形状の整合のみ検査する)
   12. lut_flat: gates=[tt,a,b]の値域・前方参照なし(トポロジカル順)、
       outputsの長さ=out_width・参照範囲
"""

import numpy as np

from .schema import ADD_MODES, NODE_TYPES, IRGraph
from .serialize import tt_from_hex


class IRValidationError(ValueError):
    pass


# ノード型別の入力エッジ本数(min, max)。Noneは上限なし。
# 'recurrent' の実際の本数はscan_vars(step/invariant)の数に依存するため
# ここでは形だけ持ち、実チェックは_check_slotsの専用分岐で行う。
NODE_ARITY: dict[str, tuple[int, int | None]] = {
    'lut_layer': (1, 1),
    'group_sum': (1, 1),
    'binarize': (1, 1),
    'flatten': (1, 1),
    'blackbox': (1, 1),
    'concat': (2, None),
    'add': (2, None),
    'conv2d': (1, 1),
    'or_pool2d': (1, 1),
    'constant': (0, 0),
    'upsample2d': (1, 1),
    'permute': (1, 1),
    'recurrent': (0, None),
    'scan_var': (0, 0),
    'lut_flat': (1, 1),
}

# or_pool2dの窓入力数上限(loweringのORゲート真理値表が2^(2^n)のため)
OR_POOL_MAX_WINDOW = 6


def _pair(v) -> tuple[int, int]:
    """int または [h, w] を (h, w) タプルへ正規化する"""
    if isinstance(v, list | tuple):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def conv2d_out_shape(params: dict) -> tuple[int, int, int]:
    """conv2dノードの出力形状 (out_channels, H_out, W_out) を計算する"""
    _, h, w = (int(v) for v in params['in_shape'])
    kh, kw = _pair(params['kernel_size'])
    sh, sw = _pair(params['stride'])
    ph, pw = _pair(params['padding'])
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    return int(params['out_channels']), h_out, w_out


def or_pool2d_out_shape(params: dict) -> tuple[int, int, int]:
    """or_pool2dノードの出力形状 (C, H_out, W_out) を計算する"""
    c, h, w = (int(v) for v in params['in_shape'])
    k = int(params['kernel_size'])
    s = int(params['stride'])
    p = int(params.get('padding', 0))
    h_out = (h + 2 * p - k) // s + 1
    w_out = (w + 2 * p - k) // s + 1
    return c, h_out, w_out


def upsample2d_out_shape(params: dict) -> tuple[int, int, int]:
    """upsample2dノードの出力形状 (C, H_out, W_out) を計算する"""
    c, h, w = (int(v) for v in params['in_shape'])
    s = int(params['scale'])
    return c, h * s, w * s


def permute_out_shape(params: dict) -> list[int]:
    """permuteノードの出力形状 [in_shape[p] for p in perm] を計算する"""
    dims = [int(v) for v in params['in_shape']]
    perm = [int(v) for v in params['perm']]
    return [dims[q] for q in perm]


def recurrent_ext_scan_vars(params: dict) -> list[dict]:
    """recurrentノードの外部入力に対応するscan_vars(role∈step/invariant)。

    親グラフからの入力エッジ(dst_slot昇順)はこの並びに1対1対応する
    (schema.py参照)。role='state'のscan_varはbody内部でのみ初期化され、
    親からの入力エッジを持たない。
    """
    return [sv for sv in params.get('scan_vars', [])
            if sv.get('role') in ('step', 'invariant')]


def node_out_width(node) -> int:
    """ノードの出力幅"""
    p = node.params
    if node.type == 'lut_layer':
        return p['out_width']
    if node.type == 'group_sum':
        return p['k']
    if node.type == 'binarize':
        return p['width']
    if node.type == 'flatten':
        return p['width']
    if node.type == 'blackbox':
        return p['out_width']
    if node.type == 'concat':
        return int(sum(p['in_widths']))
    if node.type == 'add':
        return p['width']
    if node.type == 'conv2d':
        oc, h_out, w_out = conv2d_out_shape(p)
        return oc * h_out * w_out
    if node.type == 'or_pool2d':
        c, h_out, w_out = or_pool2d_out_shape(p)
        return c * h_out * w_out
    if node.type == 'constant':
        return p['width']
    if node.type == 'upsample2d':
        c, h_out, w_out = upsample2d_out_shape(p)
        return c * h_out * w_out
    if node.type == 'permute':
        return int(np.prod([int(v) for v in p['in_shape']]))
    if node.type == 'scan_var':
        return int(p['width'])
    if node.type == 'lut_flat':
        return int(p['out_width'])
    if node.type == 'recurrent':
        w = int(p['output_width'])
        return int(p['steps']) * w if p['output_mode'] == 'stacked' else w
    raise IRValidationError(f'未知のノード型: {node.type}')


def node_in_width(node, slot: int) -> int:
    """ノードのslot別入力幅"""
    p = node.params
    if node.type == 'concat':
        return int(p['in_widths'][slot])
    if node.type == 'add':
        return p['width']
    if node.type == 'lut_layer':
        return p['in_width']
    if node.type in ('group_sum', 'lut_flat'):
        return p['in_width']
    if node.type in ('binarize', 'flatten'):
        return p['width']
    if node.type == 'blackbox':
        return p['in_width']
    if node.type in ('conv2d', 'or_pool2d', 'upsample2d'):
        # 幅規約: 入力幅 = C*H*W flatten
        return int(np.prod([int(v) for v in p['in_shape']]))
    if node.type == 'permute':
        return int(np.prod([int(v) for v in p['in_shape']]))
    if node.type in ('constant', 'scan_var'):
        raise IRValidationError(f'{node.id}: {node.type}ノードは入力を持たない')
    if node.type == 'recurrent':
        ext = recurrent_ext_scan_vars(p)
        if slot < 0 or slot >= len(ext):
            raise IRValidationError(
                f'{node.id}: slot {slot} は範囲外'
                f'(外部入力は{len(ext)}本、role step/invariantのscan_vars数)')
        sv = ext[slot]
        w = int(sv['width'])
        return int(p['steps']) * w if sv['role'] == 'step' else w
    raise IRValidationError(f'未知のノード型: {node.type}')


def _check_node_params(node) -> None:
    """ノード固有のparams整合(3-5, 8)"""
    p = node.params
    if node.type == 'lut_layer':
        indices = np.asarray(p['indices'])
        if indices.shape != (p['n_inputs'], p['out_width']):
            raise IRValidationError(
                f'{node.id}: indices形状 {indices.shape} != '
                f"({p['n_inputs']}, {p['out_width']})"
            )
        if indices.min() < 0 or indices.max() >= p['in_width']:
            raise IRValidationError(f'{node.id}: indicesが入力幅を超えている')
        tts = tt_from_hex(p['truth_tables'])
        if len(tts) != p['out_width']:
            raise IRValidationError(f'{node.id}: truth_tables数が不一致')
        limit = 1 << (1 << p['n_inputs'])
        if any(t < 0 or t >= limit for t in tts):
            raise IRValidationError(f'{node.id}: 真理値表が値域外')
    elif node.type == 'group_sum':
        if p['in_width'] % p['k'] != 0:
            raise IRValidationError(
                f"{node.id}: in_width({p['in_width']}) % k({p['k']}) != 0"
            )
        bias = p.get('bias')
        if bias is not None:
            # 判定バイアス: クラスごとの整数、長さ k
            if (not isinstance(bias, list | tuple | np.ndarray) or len(bias) != int(p['k'])
                    or any(int(v) != v for v in bias)):
                raise IRValidationError(
                    f"{node.id}: bias は長さ k({p['k']}) の整数リストが必要: {bias}"
                )
    elif node.type == 'concat':
        widths = p.get('in_widths')
        if (not isinstance(widths, list | tuple) or len(widths) < 2
                or any(int(w) <= 0 for w in widths)):
            raise IRValidationError(
                f'{node.id}: in_widthsは2要素以上の正整数リストが必要'
            )
    elif node.type == 'add':
        if int(p.get('width', 0)) <= 0:
            raise IRValidationError(f'{node.id}: widthは正整数が必要')
        if p.get('mode') not in ADD_MODES:
            raise IRValidationError(
                f"{node.id}: 未知のaddモード {p.get('mode')}"
                f"(有効: {'/'.join(ADD_MODES)})"
            )
    elif node.type == 'conv2d':
        _check_conv2d_params(node)
    elif node.type == 'or_pool2d':
        _check_or_pool2d_params(node)
    elif node.type == 'upsample2d':
        _check_upsample2d_params(node)
    elif node.type == 'permute':
        _check_permute_params(node)
    elif node.type == 'constant':
        width = int(p.get('width', 0))
        if width <= 0:
            raise IRValidationError(f'{node.id}: widthは正整数が必要')
        bits = p.get('bits')
        if not isinstance(bits, list | tuple | np.ndarray) or len(bits) != width:
            raise IRValidationError(
                f'{node.id}: bitsはwidth({width})要素のリストが必要'
            )
        if any(int(b) not in (0, 1) for b in bits):
            raise IRValidationError(f'{node.id}: bitsは0/1のみ')
    elif node.type == 'scan_var':
        _check_scan_var_params(node)
    elif node.type == 'recurrent':
        _check_recurrent_params(node)
    elif node.type == 'lut_flat':
        _check_lut_flat_params(node)


def _check_lut_flat_params(node) -> None:
    """lut_flatノードのparams整合(信号番号規約は schema.py 参照)。"""
    p = node.params
    in_width = p.get('in_width')
    out_width = p.get('out_width')
    for name, v in (('in_width', in_width), ('out_width', out_width)):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise IRValidationError(f'{node.id}: {name}は正整数が必要')
    gates = np.asarray(p.get('gates', []), dtype=np.int64)
    if gates.size == 0:
        gates = gates.reshape(0, 3)
    if gates.ndim != 2 or gates.shape[1] != 3:
        raise IRValidationError(
            f'{node.id}: gatesは[tt,a,b]のリスト(形状(n,3))が必要: '
            f'{gates.shape}')
    n_gates = gates.shape[0]
    if n_gates:
        if gates[:, 0].min() < 0 or gates[:, 0].max() > 0xf:
            raise IRValidationError(f'{node.id}: gatesのttが0..15の外')
        own = int(in_width) + 1 + np.arange(n_gates, dtype=np.int64)
        for col, name in ((1, 'a'), (2, 'b')):
            ref = gates[:, col]
            if ref.min() < 0 or np.any(ref >= own):
                raise IRValidationError(
                    f'{node.id}: gatesの{name}参照が不正'
                    '(負数、または自分以降の信号への前方参照)')
    outputs = np.asarray(p.get('outputs', []), dtype=np.int64)
    if outputs.ndim != 1 or outputs.shape[0] != out_width:
        raise IRValidationError(
            f'{node.id}: outputsは長さout_width({out_width})のリストが必要: '
            f'{outputs.shape}')
    n_signals = int(in_width) + 1 + n_gates
    if outputs.min() < 0 or (outputs >> 1).max() >= n_signals:
        raise IRValidationError(
            f'{node.id}: outputsの参照が信号数{n_signals}の範囲外')


def _check_scan_var_params(node) -> None:
    """scan_varノードのparams整合。

    role='state'のときのみinit={'kind':'const','bits':[...]}を持つ
    (schema.py参照。kind!='const'・role!='state'でのinit保持はエラー)。
    """
    p = node.params
    width = p.get('width')
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise IRValidationError(f'{node.id}: widthは正整数が必要')
    role = p.get('role')
    if role not in ('state', 'step', 'invariant'):
        raise IRValidationError(
            f"{node.id}: roleは'state'/'step'/'invariant'のいずれかが必要: "
            f'{role!r}'
        )
    init = p.get('init')
    if role == 'state':
        if not isinstance(init, dict):
            raise IRValidationError(
                f"{node.id}: role='state'にはinit(dict)が必要"
            )
        if init.get('kind') != 'const':
            raise IRValidationError(
                f"{node.id}: init.kind={init.get('kind')!r}は未対応"
                "(現状'const'のみ実装済み)"
            )
        bits = init.get('bits')
        if (not isinstance(bits, list | tuple | np.ndarray)
                or len(bits) != width):
            raise IRValidationError(
                f'{node.id}: init.bitsはwidth({width})要素のリストが必要'
            )
        if any(int(b) not in (0, 1) for b in bits):
            raise IRValidationError(f'{node.id}: init.bitsは0/1のみ')
    elif init is not None:
        raise IRValidationError(
            f"{node.id}: role={role!r}はinitを持てない('state'のみ持てる)"
        )


def _check_recurrent_params(node) -> None:
    """recurrentノードのparams整合。

    bodyそのものはvalidate(body, target='generic')で再帰検証する
    (recurrent自身は現状HDL非対応なので、targetをhdlへ伝播しない。
    9節のHDLチェックで別途recurrentそのものを拒否する)。
    """
    p = node.params
    steps = p.get('steps')
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise IRValidationError(f'{node.id}: stepsは1以上の整数が必要')

    body = p.get('body')
    if not isinstance(body, IRGraph):
        raise IRValidationError(f'{node.id}: bodyはIRGraphインスタンスが必要')
    validate(body, target='generic')

    scan_vars = p.get('scan_vars')
    if not isinstance(scan_vars, list | tuple) or not scan_vars:
        raise IRValidationError(f'{node.id}: scan_varsは1要素以上のリストが必要')

    body_ids = {n.id for n in body.nodes}
    body_scan_var_ids = {n.id for n in body.nodes if n.type == 'scan_var'}
    seen_ids: set = set()
    for sv in scan_vars:
        if (not isinstance(sv, dict) or 'id' not in sv or 'role' not in sv
                or 'width' not in sv):
            raise IRValidationError(
                f'{node.id}: scan_varsの各要素はid/role/widthを持つ必要がある'
            )
        if sv['id'] in seen_ids:
            raise IRValidationError(f"{node.id}: scan_var idが重複: {sv['id']}")
        seen_ids.add(sv['id'])
        if sv['role'] not in ('state', 'step', 'invariant'):
            raise IRValidationError(
                f"{node.id}: scan_var {sv['id']} の役割が不正: {sv['role']!r}"
            )
        if sv['id'] not in body_scan_var_ids:
            raise IRValidationError(
                f"{node.id}: scan_var {sv['id']!r} がbody内のscan_varノード"
                'に無い'
            )
        bn = body.node_by_id(sv['id'])
        if bn.params.get('role') != sv['role']:
            raise IRValidationError(
                f"{node.id}: scan_var {sv['id']} のroleがbody内ノードと不一致"
                f"(recurrent={sv['role']!r}, body={bn.params.get('role')!r})"
            )
        if bn.params.get('width') != sv['width']:
            raise IRValidationError(
                f"{node.id}: scan_var {sv['id']} のwidthがbody内ノードと不一致"
                f"(recurrent={sv['width']!r}, body={bn.params.get('width')!r})"
            )

    if body_scan_var_ids != seen_ids:
        raise IRValidationError(
            f'{node.id}: scan_vars({sorted(seen_ids)})がbody内のscan_var'
            f'ノード({sorted(body_scan_var_ids)})と1対1対応しない'
        )

    state_svs = [sv for sv in scan_vars if sv['role'] == 'state']
    state_update = p.get('state_update')
    if not isinstance(state_update, dict):
        raise IRValidationError(f'{node.id}: state_updateはdictが必要')
    if set(state_update.keys()) != {sv['id'] for sv in state_svs}:
        raise IRValidationError(
            f"{node.id}: state_updateのキー({sorted(state_update.keys())})が"
            f"role='state'のscan_var({sorted(sv['id'] for sv in state_svs)})"
            'を過不足なく網羅していない'
        )
    for sv_id, bnid in state_update.items():
        if bnid not in body_ids:
            raise IRValidationError(
                f"{node.id}: state_update[{sv_id!r}] がbody内に無いノード"
                f'を指している: {bnid!r}'
            )
        sv_width = next(sv['width'] for sv in state_svs if sv['id'] == sv_id)
        if node_out_width(body.node_by_id(bnid)) != sv_width:
            raise IRValidationError(
                f"{node.id}: state_update[{sv_id!r}]={bnid!r}の出力幅が"
                f'state幅({sv_width})と一致しない'
            )

    output_mode = p.get('output_mode')
    if output_mode not in ('final_state', 'stacked'):
        raise IRValidationError(
            f'{node.id}: output_modeは final_state/stacked のいずれか: '
            f'{output_mode!r}'
        )
    output_source_id = p.get('output_source_id')
    if output_source_id not in body_ids:
        raise IRValidationError(
            f"{node.id}: output_source_idがbody内に無いノード: "
            f'{output_source_id!r}'
        )
    output_width = p.get('output_width')
    if not isinstance(output_width, int) or output_width <= 0:
        raise IRValidationError(f'{node.id}: output_widthは正整数が必要')
    if node_out_width(body.node_by_id(output_source_id)) != output_width:
        raise IRValidationError(
            f'{node.id}: output_source_id({output_source_id})の出力幅が'
            f'output_width({output_width})と一致しない'
        )


def _check_in_shape(node) -> tuple[int, int, int]:
    """in_shapeが[C,H,W]の正整数3要素であることを検証して返す"""
    shape = node.params.get('in_shape')
    if (not isinstance(shape, list | tuple) or len(shape) != 3
            or any(int(v) <= 0 for v in shape)):
        raise IRValidationError(
            f'{node.id}: in_shapeは[C,H,W]の正整数3要素が必要: {shape}'
        )
    return tuple(int(v) for v in shape)


def _check_conv2d_params(node) -> None:
    """conv2dノードのparams整合(shape伝播・indices_flat・真理値表)"""
    p = node.params
    c, _, _ = _check_in_shape(node)
    oc = int(p['out_channels'])
    d = int(p['tree_depth'])
    n = int(p['n_inputs'])
    if oc <= 0 or d <= 0 or n <= 0:
        raise IRValidationError(
            f'{node.id}: out_channels/tree_depth/n_inputsは正整数が必要'
        )
    _, h_out, w_out = conv2d_out_shape(p)
    if h_out <= 0 or w_out <= 0:
        raise IRValidationError(
            f'{node.id}: 出力形状が不正(H_out={h_out}, W_out={w_out})。'
            'kernel_size/stride/paddingとin_shapeを確認すること'
        )
    kh, kw = _pair(p['kernel_size'])
    n_leaves = n ** d
    n_internal = d if n == 1 else (n_leaves - 1) // (n - 1)

    idxf = np.asarray(p['indices_flat'])
    if idxf.shape != (oc, n_leaves):
        raise IRValidationError(
            f'{node.id}: indices_flat形状 {idxf.shape} != '
            f'({oc}, {n_leaves})(=out_channels, n_inputs^tree_depth)'
        )
    if idxf.min() < 0 or idxf.max() >= c * kh * kw:
        raise IRValidationError(
            f'{node.id}: indices_flatが受容野(C*kH*kW={c * kh * kw})を超えている'
        )

    tts = tt_from_hex(p['truth_tables'])
    if len(tts) != oc * n_internal:
        raise IRValidationError(
            f'{node.id}: truth_tables数({len(tts)}) != '
            f'out_channels*内部ノード数({oc}*{n_internal}={oc * n_internal})'
        )
    limit = 1 << (1 << n)
    if any(t < 0 or t >= limit for t in tts):
        raise IRValidationError(f'{node.id}: 真理値表が値域外')


def _check_or_pool2d_params(node) -> None:
    """or_pool2dノードのparams整合(shape伝播・窓サイズ上限)"""
    p = node.params
    _check_in_shape(node)
    k = int(p['kernel_size'])
    s = int(p['stride'])
    pad = int(p.get('padding', 0))
    if k <= 0 or s <= 0 or pad < 0:
        raise IRValidationError(
            f'{node.id}: kernel_size/strideは正整数、paddingは非負整数が必要'
        )
    if k * k > OR_POOL_MAX_WINDOW:
        raise IRValidationError(
            f'{node.id}: 窓{k}x{k}({k * k}入力)のORは未対応'
            f'(≦{OR_POOL_MAX_WINDOW}入力のみ。OR木への分解は次段階)'
        )
    _, h_out, w_out = or_pool2d_out_shape(p)
    if h_out <= 0 or w_out <= 0:
        raise IRValidationError(
            f'{node.id}: 出力形状が不正(H_out={h_out}, W_out={w_out})'
        )


def _check_upsample2d_params(node) -> None:
    """upsample2dノードのparams整合(shape伝播・scale)"""
    p = node.params
    _check_in_shape(node)
    s = int(p.get('scale', 0))
    if s <= 0:
        raise IRValidationError(f'{node.id}: scaleは正整数が必要')


def _check_permute_params(node) -> None:
    """permuteノードのparams整合(in_shape正整数リスト・permが有効な順列)"""
    p = node.params
    dims = p.get('in_shape')
    if (not isinstance(dims, list | tuple | np.ndarray) or len(dims) < 1
            or any(int(v) <= 0 for v in dims)):
        raise IRValidationError(
            f'{node.id}: in_shapeは1要素以上の正整数リストが必要: {dims}'
        )
    n = len(dims)
    perm = p.get('perm')
    if (not isinstance(perm, list | tuple | np.ndarray) or len(perm) != n
            or sorted(int(v) for v in perm) != list(range(n))):
        raise IRValidationError(
            f'{node.id}: permは0..{n - 1}の順列({n}要素)が必要: {perm}'
        )


def _check_slots(node, in_edges: list) -> None:
    """slot重複・欠番・範囲外・arityを検証する(2)"""
    lo, hi = NODE_ARITY[node.type]
    if hi == 0:
        # arity 0(constant/scan_var): 入力エッジを持てない
        if in_edges:
            raise IRValidationError(
                f'{node.id}: {node.type}ノードは入力エッジを持てない'
                f'(現在{len(in_edges)}本)'
            )
        return

    if node.type == 'recurrent':
        # 本数はscan_vars(role step/invariant)の数に依存する。0本もあり得る
        # (state役のscan_varのみで完結するbody)ので、先に処理して以降の
        # 「入力エッジがない→エラー」を通らないようにする。
        n_slots = len(recurrent_ext_scan_vars(node.params))
        slots = sorted(e.dst_slot for e in in_edges)
        seen = set()
        for s in slots:
            if s in seen:
                raise IRValidationError(
                    f'{node.id}: slot {s} に複数のエッジが接続されている(重複)'
                )
            seen.add(s)
        for s in slots:
            if s < 0 or s >= n_slots:
                raise IRValidationError(
                    f'{node.id}: slot {s} は範囲外(外部入力は{n_slots}本)'
                )
        for s in range(n_slots):
            if s not in seen:
                raise IRValidationError(f'{node.id}: slot {s} が欠番')
        return

    if not in_edges:
        raise IRValidationError(f'{node.id}: 入力エッジがない')

    slots = sorted(e.dst_slot for e in in_edges)
    seen = set()
    for s in slots:
        if s in seen:
            raise IRValidationError(
                f'{node.id}: slot {s} に複数のエッジが接続されている(重複)'
            )
        seen.add(s)

    if node.type == 'concat':
        n_slots = len(node.params['in_widths'])
        for s in slots:
            if s < 0 or s >= n_slots:
                raise IRValidationError(
                    f'{node.id}: slot {s} は範囲外(in_widthsは{n_slots}要素)'
                )
        for s in range(n_slots):
            if s not in seen:
                raise IRValidationError(f'{node.id}: slot {s} が欠番')
        return

    if node.type == 'add':
        if len(slots) < lo:
            raise IRValidationError(
                f'{node.id}: addノードは入力エッジが{lo}本以上必要'
                f'(現在{len(slots)}本)'
            )
        # slotは0..m-1の連番であること
        for i, s in enumerate(slots):
            if s != i:
                raise IRValidationError(f'{node.id}: slot {i} が欠番')
        return

    # 単一入力ノード: slot 0 のみ1本
    if len(in_edges) > hi:
        raise IRValidationError(
            f'{node.id}: {node.type}ノードの入力エッジは1本のみ'
            f'(現在{len(in_edges)}本)'
        )
    if slots[0] != 0:
        raise IRValidationError(
            f'{node.id}: {node.type}ノードの入力はslot 0のみ'
            f'(slot {slots[0]} は不正)'
        )


def validate(graph: IRGraph, *, target: str = 'generic') -> list[str]:
    """検証を実行し警告リストを返す。エラーはIRValidationErrorを送出。

    Args:
        target: 'generic' | 'hdl'
            (hdlはblackbox・add(mode='sum')を禁止)
    """
    warnings: list[str] = []

    # 1. 型
    for n in graph.nodes:
        if n.type not in NODE_TYPES:
            raise IRValidationError(f'{n.id}: 未知のノード型 {n.type}')

    ids = {n.id for n in graph.nodes}
    in_ports = {p.id: p for p in graph.inputs}
    out_ports = {p.id: p for p in graph.outputs}
    if len(ids) != len(graph.nodes):
        raise IRValidationError('ノードidが重複している')

    # 2. エッジ端点の存在・出力ポートのslot
    for e in graph.edges:
        if e.src not in ids and e.src not in in_ports:
            raise IRValidationError(f'エッジsrcが未定義: {e.src}')
        if e.dst not in ids and e.dst not in out_ports:
            raise IRValidationError(f'エッジdstが未定義: {e.dst}')
        if e.dst in out_ports and e.dst_slot != 0:
            raise IRValidationError(
                f'出力ポート {e.dst} へのエッジはslot 0のみ(slot {e.dst_slot})'
            )

    # 3-5, 8. ノード固有params
    for n in graph.nodes:
        _check_node_params(n)

    # 2-3. slot整合+幅整合(fanoutは許容: 同一srcから複数エッジ可)
    def width_of_src(src_id: str) -> int:
        if src_id in in_ports:
            return in_ports[src_id].width
        return node_out_width(graph.node_by_id(src_id))

    in_edges_of: dict[str, list] = {}
    for e in graph.edges:
        if e.dst in ids:
            in_edges_of.setdefault(e.dst, []).append(e)

    for n in graph.nodes:
        _check_slots(n, in_edges_of.get(n.id, []))
        for e in in_edges_of.get(n.id, []):
            w_src = width_of_src(e.src)
            w_dst = node_in_width(n, e.dst_slot)
            if w_src != w_dst:
                if n.type == 'concat':
                    raise IRValidationError(
                        f'幅不整合: {e.src}(out={w_src}) → '
                        f'{n.id}.slot{e.dst_slot}'
                        f'(in_widths[{e.dst_slot}]={w_dst})'
                    )
                if n.type == 'add':
                    raise IRValidationError(
                        f'幅不整合: {e.src}(out={w_src}) → '
                        f'{n.id}.slot{e.dst_slot}(width={w_dst})'
                    )
                raise IRValidationError(
                    f'幅不整合: {e.src}(out={w_src}) → {e.dst}(in={w_dst})'
                )

    # 出力ポートへの幅整合
    for e in graph.edges:
        if e.dst in out_ports:
            w_src = width_of_src(e.src)
            w_dst = out_ports[e.dst].width
            if w_src != w_dst:
                raise IRValidationError(
                    f'幅不整合: {e.src}(out={w_src}) → {e.dst}(in={w_dst})'
                )

    # 6. DAG(トポロジカル順に全ノードが現れるか)
    order = graph.topological_order()
    if len(order) != len(graph.nodes):
        raise IRValidationError('循環または到達不能な依存が存在する')

    # 7. 未使用ノード(警告)
    connected_src = {e.src for e in graph.edges}
    for n in graph.nodes:
        if n.id not in connected_src:
            warnings.append(f'{n.id}: 出力が未使用')

    # 9. HDLターゲット
    if target == 'hdl':
        for n in graph.nodes:
            if n.type == 'blackbox':
                raise IRValidationError(
                    f'{n.id}: blackboxノードはHDL生成不可'
                    '(ハイブリッドモデルは論理部分のみエクスポートすること)'
                )
            if n.type == 'add' and n.params.get('mode') == 'sum':
                raise IRValidationError(
                    f"{n.id}: add(mode='sum')はHDL生成不可"
                    '(整数和は幅拡張が必要。group_sum同様に論理領域の終端で処理すること)'
                )
            if n.type == 'lut_flat':
                raise IRValidationError(
                    f'{n.id}: lut_flatノードはHDL生成不可(現時点)。'
                    '縮約前のlut_layer列(flatten_dense適用前のIR)をHDLへ渡すこと')
            if n.type == 'recurrent':
                raise IRValidationError(
                    f'{n.id}: recurrentノードはHDL生成不可(現時点)。'
                    'bodyをunrollしない設計のため、HDL化には別途ステートマシン'
                    '生成トラックが要る'
                )
            if n.type == 'group_sum' and any(int(v) != 0 for v in n.params.get('bias') or []):
                raise IRValidationError(
                    f'{n.id}: group_sum の判定バイアス(bias)はHDL生成未対応。'
                    'HDL ではバイアス無しの生スコアを出し、比較器 + 定数は FPGA 側の'
                    '終端部品で付けること'
                )

    return warnings
