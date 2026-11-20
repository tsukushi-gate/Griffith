"""NAS用IR変異オペレータ群。

IR(``griffith_logic.ir.IRGraph``)→IRの純関数変異オペレータを提供する。
本体ライブラリ(``griffith_logic/``)は一切変更せず、``ir/schema.py`` の
``IRGraph``/``IRNode``/``IREdge`` と ``ir/validate.py`` の ``validate`` を
import して呼び出すだけに留める。

設計判断:
    - IR表現は ``IRGraph`` オブジェクトをそのまま使う(dict化はしない)。
      ``validate``/``build_torch`` が ``IRGraph`` を受け取る形が本体の
      自然なインターフェースであり、dict版(``ir_to_dict``/``ir_from_dict``)
      は主にHTTP境界用の変換なので、内部変異ロジックでは往復コストが
      無駄になる。dictを渡したい呼び出し元は
      ``ir_from_dict``/``ir_to_dict`` で変換してから/した後に本モジュールへ
      渡せばよい
    - 各オペレータは ``copy.deepcopy`` した新しいIRを返し、入力IRには
      一切触れない(非破壊)
    - 乱数は呼び出し側が渡す ``rng``(``random.Random`` または
      ``numpy.random.Generator``)経由のみで消費し、モジュール内で
      グローバル乱数を使わない(決定的再現性)
    - 変異後IRが ``validate()`` を通らない場合は内部で棄却し、
      別の乱数選択で再試行する。上限回数(``max_attempts``)に達したら
      ``None`` を返す(失敗を例外にしない=探索ループ側でスキップしやすくする)

対応スコープ(あえて絞っている点、詳細は各関数docstring):
    - 幅変更系(層幅/convチャネル)は「変異対象ノードの出力エッジが
      ちょうど1本」の場合のみ対応する(fanoutがある場合は複数の後続を
      同時に整合させる必要がありスコープ外。棄却して再試行する)
    - 後続ノードが ``output`` ポート直結の場合は外部インターフェース
      (グラフの入出力幅)を変えないために棄却する
    - 後続ノードが ``add``/``blackbox``/``conv2d``/``or_pool2d`` の場合も
      幅整合の再計算が非自明なため棄却する(対応後続型は
      ``lut_layer``/``group_sum``/``concat``/``flatten`` のみ)
"""

import copy

from griffith_logic.ir.schema import IREdge, IRGraph, IRNode
from griffith_logic.ir.serialize import tt_to_hex
from griffith_logic.ir.validate import IRValidationError, conv2d_out_shape, validate

# LutLayerが対応するLUT入力数(tests/strategies.py の n_inputs_st と同じ集合)
N_INPUTS_CHOICES = (2, 4, 6)

# 幅変更系オペレータが安全に整合を再計算できる後続ノード型
_REWIRABLE_SUCCESSOR_TYPES = ('lut_layer', 'group_sum', 'concat', 'flatten')

DEFAULT_MAX_ATTEMPTS = 20


# --------------------------------------------------------------------------
# 乱数ヘルパー(random.Random / numpy.random.Generator の両対応)
# --------------------------------------------------------------------------

def _randint(rng, lo: int, hi: int) -> int:
    """[lo, hi] 両端含む一様整数(rngはrandom.Randomまたはnp.random.Generator)"""
    if hasattr(rng, 'integers'):
        return int(rng.integers(lo, hi + 1))
    return rng.randint(lo, hi)


def _choice(rng, seq):
    """seqから1要素を選ぶ(rng種類に依存しない共通実装)"""
    return seq[_randint(rng, 0, len(seq) - 1)]


def _shuffled_indices(rng, n: int) -> list[int]:
    """0..n-1 のFisher-Yatesシャッフル(_randintのみで実装し決定性を統一)"""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = _randint(rng, 0, i)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def _truth_table_family(parameterization: str, n_inputs: int, *,
                         is_conv: bool = False) -> str:
    """parameterization/n_inputsから、build_torch可能な真理値表の集合を選ぶ。

    ir/weights.py の weights_from_truth_tables / conv_weights_from_truth_tables
    が離散化復元できる真理値表の範囲(popcount制約)に合わせて生成する
    (validate()自体はこの制約をチェックしないが、変異後IRを
    ``build_torch`` できることを保証しNAS trialで実際に使えるIRにするため)。

        'arbitrary'              — 制約なし(softmax16g/N=2、independent等)
        'single_minterm'         — popcount==1のみ(iwp、conv全般のsoftmax)
        'single_minterm_or_false'— popcount in {0,1}(lut_layerのsoftmax、N>2)
    """
    if is_conv:
        return 'single_minterm' if parameterization == 'softmax' else 'arbitrary'
    if parameterization == 'softmax':
        return 'arbitrary' if n_inputs == 2 else 'single_minterm_or_false'
    if parameterization == 'iwp':
        return 'single_minterm'
    return 'arbitrary'


def _random_truth_table(rng, n_inputs: int, family: str = 'arbitrary') -> str:
    """n_inputs入力LUTの真理値表をfamily制約に従いランダム生成(hex文字列)"""
    n_entries = 1 << n_inputs
    if family == 'single_minterm':
        k = _randint(rng, 0, n_entries - 1)
        return tt_to_hex([1 << k])[0]
    if family == 'single_minterm_or_false':
        if _randint(rng, 0, 3) == 0:
            return tt_to_hex([0])[0]
        k = _randint(rng, 0, n_entries - 1)
        return tt_to_hex([1 << k])[0]
    limit = 1 << n_entries
    return tt_to_hex([_randint(rng, 0, limit - 1)])[0]


# --------------------------------------------------------------------------
# グラフ走査ヘルパー
# --------------------------------------------------------------------------

def _out_edges(graph: IRGraph, node_id: str) -> list[IREdge]:
    return [e for e in graph.edges if e.src == node_id]


def _output_port_ids(graph: IRGraph) -> set[str]:
    return {p.id for p in graph.outputs}


def _unique_node_id(graph: IRGraph, prefix: str) -> str:
    existing = {n.id for n in graph.nodes}
    i = 0
    while f'{prefix}_{i}' in existing:
        i += 1
    return f'{prefix}_{i}'


def _rewire_successor_width(graph: IRGraph, edge: IREdge, new_width: int,
                             rng) -> bool:
    """edge.dstノードのslot別入力幅をnew_widthへ更新する(成功可否を返す)。

    対応する後続型は _REWIRABLE_SUCCESSOR_TYPES のみ。group_sumは
    in_width % k == 0 を満たさない場合に失敗(False)を返す。
    """
    dst_ids = {n.id: n for n in graph.nodes}
    dst_node = dst_ids.get(edge.dst)
    if dst_node is None:
        # output ポート直結: 外部インターフェースは変えない方針のため不可
        return False
    if dst_node.type not in _REWIRABLE_SUCCESSOR_TYPES:
        return False

    if dst_node.type == 'lut_layer':
        old_in_width = dst_node.params['in_width']
        dst_node.params['in_width'] = new_width
        if new_width < old_in_width:
            # 範囲外を指すindicesを新範囲内へ再サンプルする
            indices = dst_node.params['indices']
            for row in indices:
                for k, v in enumerate(row):
                    if v >= new_width:
                        row[k] = _randint(rng, 0, new_width - 1)
        return True

    if dst_node.type == 'group_sum':
        if new_width % dst_node.params['k'] != 0:
            return False
        dst_node.params['in_width'] = new_width
        return True

    if dst_node.type == 'concat':
        widths = dst_node.params['in_widths']
        widths[edge.dst_slot] = new_width
        return True

    if dst_node.type == 'flatten':
        dst_node.params['width'] = new_width
        return True

    return False  # pragma: no cover — _REWIRABLE_SUCCESSOR_TYPESと矛盾しない


def _validated_or_none(graph: IRGraph) -> IRGraph | None:
    try:
        validate(graph)
    except IRValidationError:
        return None
    return graph


# --------------------------------------------------------------------------
# 1. 層幅(lut_layer.out_width)の増減
# --------------------------------------------------------------------------

def mutate_lut_width(ir: IRGraph, rng, *, max_delta: int = 2,
                      max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> IRGraph | None:
    """lut_layerの out_width をランダムに増減し、後続層のindices/幅を整合させる。

    対象ノードの条件(棄却規則):
        - 出力エッジがちょうど1本であること(fanoutは非対応)
        - 後続が _REWIRABLE_SUCCESSOR_TYPES のいずれかであること
        - 変異後 out_width >= 1
        - 変異後IRが validate() を通過すること

    Returns:
        変異後の新規IR。条件を満たす変異が見つからなければ None。
    """
    lut_ids = [n.id for n in ir.nodes if n.type == 'lut_layer']
    if not lut_ids:
        return None

    for _ in range(max_attempts):
        node_id = _choice(rng, lut_ids)
        out_edges = _out_edges(ir, node_id)
        if len(out_edges) != 1:
            continue

        delta = _choice(rng, [d for d in range(-max_delta, max_delta + 1)
                              if d != 0])
        graph = copy.deepcopy(ir)
        node = graph.node_by_id(node_id)
        old_width = node.params['out_width']
        new_width = old_width + delta
        if new_width < 1:
            continue

        n_inputs = node.params['n_inputs']
        in_width = node.params['in_width']
        indices = node.params['indices']
        tts = node.params['truth_tables']
        family = _truth_table_family(
            node.train_info.get('parameterization', 'softmax'), n_inputs)

        if new_width > old_width:
            for row in indices:
                for _k in range(new_width - old_width):
                    row.append(_randint(rng, 0, in_width - 1))
            for _k in range(new_width - old_width):
                tts.append(_random_truth_table(rng, n_inputs, family))
        else:
            drop = old_width - new_width
            drop_cols = set(_shuffled_indices(rng, old_width)[:drop])
            for j, row in enumerate(indices):
                indices[j] = [v for k, v in enumerate(row) if k not in drop_cols]
            node.params['truth_tables'] = [
                v for k, v in enumerate(tts) if k not in drop_cols
            ]
        node.params['out_width'] = new_width

        edge = next(e for e in graph.edges if e.src == node_id)
        if not _rewire_successor_width(graph, edge, new_width, rng):
            continue

        result = _validated_or_none(graph)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------
# 2a. 深さ増(pass-through lut_layer挿入)
# --------------------------------------------------------------------------

def mutate_insert_layer(ir: IRGraph, rng, *,
                         max_attempts: int = DEFAULT_MAX_ATTEMPTS
                         ) -> IRGraph | None:
    """既存エッジ上に pass-through 初期化のlut_layerを挿入し、深さを1増やす。

    挿入層は n_inputs=2、indices[0]=[0..w-1](恒等)・indices[1]=indices[0]の
    複製(無視される)、truth_tables=全出力ビットで「slot0を素通し」する
    定数0xA(4パターン中 f(00)=0,f(01)=1,f(10)=0,f(11)=1。butterfly基底の
    bit j=入力スロットj規約でbit0=slot0を素通しし、slot1は無視する)。

    対象エッジの条件: dstがノード(出力ポート直結は不可。挿入後も
    出力ポート幅は不変のため実は挿入可能だが、他オペレータとの対称性・
    実装単純化のため一律ノード間エッジのみ対象とする)。
    """
    candidates = [e for e in ir.edges if e.dst in {n.id for n in ir.nodes}]
    if not candidates:
        return None

    for _ in range(max_attempts):
        edge = _choice(rng, candidates)
        graph = copy.deepcopy(ir)
        target_edge = next(
            e for e in graph.edges
            if e.src == edge.src and e.dst == edge.dst
            and e.dst_slot == edge.dst_slot
        )

        src_id = target_edge.src
        in_ports = {p.id: p for p in graph.inputs}
        if src_id in in_ports:
            width = in_ports[src_id].width
        else:
            width = graph.node_by_id(src_id).params.get('out_width')
            if width is None:
                # out_widthを持たない型(conv2d等)からの挿入は非対応
                continue

        new_id = _unique_node_id(graph, 'nas_ins')
        identity_row = list(range(width))
        new_node = IRNode(
            id=new_id,
            type='lut_layer',
            params={
                'in_width': width,
                'out_width': width,
                'n_inputs': 2,
                'indices': [identity_row, list(identity_row)],
                'truth_tables': ['0xa'] * width,
            },
            train_info={'parameterization': 'softmax', 'selection': 'soft'},
            meta={'nas_inserted': True},
        )
        graph.nodes.append(new_node)
        target_edge.src = new_id
        graph.edges.append(IREdge(src=src_id, dst=new_id, dst_slot=0))

        result = _validated_or_none(graph)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------
# 2b. 深さ減(lut_layer削除+再配線)
# --------------------------------------------------------------------------

def mutate_remove_layer(ir: IRGraph, rng, *,
                         max_attempts: int = DEFAULT_MAX_ATTEMPTS
                         ) -> IRGraph | None:
    """in_width==out_widthのlut_layerを1つ削除し、前後を直結して再配線する。

    対象ノードの条件(棄却規則):
        - type=='lut_layer' かつ in_width == out_width(削除しても
          前段の出力幅と後続の要求幅の整合が保たれる)
        - 入力エッジがちょうど1本(fanin 1、lut_layerのarity契約どおり)
        - このノードからの出力エッジ本数は任意(全ての後続を前段へ繋ぎ替える)
    """
    lut_ids = [
        n.id for n in ir.nodes
        if n.type == 'lut_layer' and n.params['in_width'] == n.params['out_width']
    ]
    if not lut_ids:
        return None

    for _ in range(max_attempts):
        node_id = _choice(rng, lut_ids)
        graph = copy.deepcopy(ir)
        in_edges = [e for e in graph.edges if e.dst == node_id]
        if len(in_edges) != 1:
            continue
        pred_src = in_edges[0].src

        out_edges = [e for e in graph.edges if e.src == node_id]
        for e in out_edges:
            e.src = pred_src
        graph.edges = [e for e in graph.edges if e.dst != node_id]
        graph.nodes = [n for n in graph.nodes if n.id != node_id]

        result = _validated_or_none(graph)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------
# 3. n_inputs変更(truth_table/indicesの整合再生成)
# --------------------------------------------------------------------------

def mutate_n_inputs(ir: IRGraph, rng, *,
                     max_attempts: int = DEFAULT_MAX_ATTEMPTS
                     ) -> IRGraph | None:
    """lut_layerの n_inputs を変更し、indices/truth_tablesを再生成する。

    out_width/in_widthは変えないため後続の整合は不要(ローカルな変異)。
    """
    lut_ids = [n.id for n in ir.nodes if n.type == 'lut_layer']
    if not lut_ids:
        return None

    for _ in range(max_attempts):
        node_id = _choice(rng, lut_ids)
        graph = copy.deepcopy(ir)
        node = graph.node_by_id(node_id)
        current = node.params['n_inputs']
        choices = [n for n in N_INPUTS_CHOICES if n != current]
        if not choices:
            continue
        new_n = _choice(rng, choices)
        in_width = node.params['in_width']
        out_width = node.params['out_width']

        node.params['n_inputs'] = new_n
        node.params['indices'] = [
            [_randint(rng, 0, in_width - 1) for _ in range(out_width)]
            for _ in range(new_n)
        ]
        family = _truth_table_family(
            node.train_info.get('parameterization', 'softmax'), new_n)
        node.params['truth_tables'] = [
            _random_truth_table(rng, new_n, family) for _ in range(out_width)
        ]

        result = _validated_or_none(graph)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------
# 4. conv2dチャネル増減
# --------------------------------------------------------------------------

def mutate_conv_channels(ir: IRGraph, rng, *, max_delta: int = 2,
                          max_attempts: int = DEFAULT_MAX_ATTEMPTS
                          ) -> IRGraph | None:
    """conv2dノードのout_channelsを増減し、indices_flat/truth_tables/後続幅を整合させる。

    対象ノードの条件(棄却規則):
        - 出力エッジがちょうど1本(_rewire_successor_width と同じ制約)
        - 後続が _REWIRABLE_SUCCESSOR_TYPES のいずれか
          (or_pool2d/conv2dへの連鎖はshape再計算が非自明なため非対応。
          conv2d→flattenのような構成を想定)
        - 変異後 out_channels >= 1
    """
    conv_ids = [n.id for n in ir.nodes if n.type == 'conv2d']
    if not conv_ids:
        return None

    for _ in range(max_attempts):
        node_id = _choice(rng, conv_ids)
        out_edges = _out_edges(ir, node_id)
        if len(out_edges) != 1:
            continue

        delta = _choice(rng, [d for d in range(-max_delta, max_delta + 1)
                              if d != 0])
        graph = copy.deepcopy(ir)
        node = graph.node_by_id(node_id)
        p = node.params
        old_oc = int(p['out_channels'])
        new_oc = old_oc + delta
        if new_oc < 1:
            continue

        c_in, kh_kw = p['in_shape'][0], p['kernel_size']
        kh = kh_kw[0] if isinstance(kh_kw, list | tuple) else kh_kw
        kw = kh_kw[1] if isinstance(kh_kw, list | tuple) else kh_kw
        n = int(p['n_inputs'])
        d = int(p['tree_depth'])
        n_leaves = n ** d
        n_internal = d if n == 1 else (n_leaves - 1) // (n - 1)
        receptive = int(c_in) * int(kh) * int(kw)

        new_indices_flat = [
            [_randint(rng, 0, receptive - 1) for _ in range(n_leaves)]
            for _ in range(new_oc)
        ]
        family = _truth_table_family(
            node.train_info.get('parameterization', 'softmax'), n, is_conv=True)
        new_tts = [
            _random_truth_table(rng, n, family)
            for _ in range(new_oc * n_internal)
        ]

        p['out_channels'] = new_oc
        p['indices_flat'] = new_indices_flat
        p['truth_tables'] = new_tts

        # 出力幅 = out_channels * H_out * W_out(H_out/W_outはout_channels非依存)
        _, h_out, w_out = conv2d_out_shape(p)
        new_width = new_oc * h_out * w_out

        edge = next(e for e in graph.edges if e.src == node_id)
        if not _rewire_successor_width(graph, edge, new_width, rng):
            continue

        result = _validated_or_none(graph)
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------
# 統一入口
# --------------------------------------------------------------------------

MUTATION_OPS: dict[str, callable] = {
    'lut_width': mutate_lut_width,
    'insert_layer': mutate_insert_layer,
    'remove_layer': mutate_remove_layer,
    'n_inputs': mutate_n_inputs,
    'conv_channels': mutate_conv_channels,
}


def mutate(ir: IRGraph, rng, ops: list | None = None, *,
           max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> IRGraph | None:
    """変異オペレータをランダムに1つ選び適用する統一入口。

    Args:
        ir: 入力IR(非破壊)
        rng: random.Random または numpy.random.Generator
        ops: 使用するオペレータ名のリスト(既定=MUTATION_OPS全キー)。
            'conv_channels' はIRにconv2dノードが無い場合、'remove_layer'は
            in_width==out_widthのlut_layerが無い場合に自動的に除外される
        max_attempts: 各オペレータ呼び出し内の再試行上限(そのまま委譲)

    Returns:
        いずれかのオペレータが変異に成功した新規IR。全滅なら None。
    """
    op_names = list(ops) if ops is not None else list(MUTATION_OPS.keys())

    has_conv = any(n.type == 'conv2d' for n in ir.nodes)
    has_removable = any(
        n.type == 'lut_layer' and n.params['in_width'] == n.params['out_width']
        for n in ir.nodes
    )
    applicable = []
    for name in op_names:
        if name == 'conv_channels' and not has_conv:
            continue
        if name == 'remove_layer' and not has_removable:
            continue
        applicable.append(name)
    if not applicable:
        return None

    order = _shuffled_indices(rng, len(applicable))
    for idx in order:
        op = MUTATION_OPS[applicable[idx]]
        result = op(ir, rng, max_attempts=max_attempts)
        if result is not None:
            return result
    return None
