"""IR → PyTorchモデル再構築(E2、v0.2でDAG対応)。

export/extract.py(モデル→IR)の逆方向。レジストリ方式でノード型ごとの
ビルダーを登録し、直列グラフはtopological_order()に従ってnn.Sequentialを、
DAGグラフはIRGraphModule(topo順実行+values辞書)を構築する。

規約:
    - 返すのは純論理モデル(入力は(batch, in_width))。MNIST等の画像入力に
      対するnn.Flatten前置はエディタバックエンド側の責務
    - lut_layerは train_info.initialized(既定True)が真かつtruth_tablesが
      あれば restore_lut_layer で重み・接続を復元し、偽(draft)なら
      ランダム初期化のまま返す
    - train_info.kind='dwn' のlut_layerはDWNLayerとして復元する(非可逆: 真理値表の±1符号化。eval等価のみ保証)
    - 直列グラフ(全ノード単一入力・fanoutなし・concat/addなし)は従来どおり
      nn.Sequentialを返す(既存テスト・エディタが依存するfast-path)
    - concat/add/constantはモジュールを持たずIRGraphModule.forward内で
      計算する。soft(学習時)意味論: or→1-(1-a)(1-b)、xor→a+b-2ab、
      sum→a+b(微分可能。hard evalでは{0,1}入力に対しビット厳密に一致)。
      constantはbitsをバッチ方向へexpandした定数行
"""

import re

import torch
import torch.nn as nn

from ..binarize import Binarize
from ..conv import LutConv2d
from ..core.connections import WIRING_SPEC_KEYS, make_connections
from ..group_sum import GroupSum
from ..layers.dwn import DWNLayer
from ..layers.fixed_lut import FixedLutLayer
from ..layers.lut import LutLayer
from ..layers.permute import Permute
from ..pooling import OrPooling2d
from .schema import IRGraph
from .weights import restore_dwn_layer, restore_lut_conv2d, restore_lut_layer

BUILDERS: dict[str, callable] = {}

# forward内で関数的に計算するノード型(モジュールを構築しない)
FUNCTIONAL_NODE_TYPES = ('concat', 'add', 'constant')

# LutLayer が受ける入力数。それ以外の幅の lut_layer は凍結 LUT(FixedLutLayer)で復元する
LUT_LAYER_N_INPUTS = (2, 4, 6)
# 幅に関わらず凍結 LUT で復元する train_info.kind(学習し直す意味がない部品)
FIXED_LUT_KINDS = ('thermometer',)
# LutLayer 自身が文字列で受ける配線種。それ以外は make_connections で明示 Tensor にする
_LUT_LAYER_STRING_CONNECTIONS = ('random', 'unique')


def register_builder(node_type: str):
    """IRノード型 → モジュール構築関数を登録するデコレータ"""
    def deco(fn):
        BUILDERS[node_type] = fn
        return fn
    return deco


@register_builder('lut_layer')
def _build_lut_layer(node) -> nn.Module:
    p = node.params
    ti = node.train_info
    if ti.get('kind') in ('dwn', 'ltt'):
        # LTT(線形閾値ゲート)の真理値表は閾値関数で、softmax の LutLayer(N>2)では
        # 表現できない。w / b への逆算はしない(IR は確定済み回路のみを持つ)ので、DWN と同じ
        # 「任意の表を持てる」LUT 保持層として eval 等価に復元する
        return _build_dwn_lut_layer(node)
    if ti.get('kind') in FIXED_LUT_KINDS or p['n_inputs'] not in LUT_LAYER_N_INPUTS:
        return _build_fixed_lut_layer(node)
    initialized = ti.get('initialized', True)
    layer = LutLayer(
        p['in_width'],
        p['out_width'],
        n_inputs=p['n_inputs'],
        connections=_draft_connections(node, initialized),
        parameterization=ti.get('parameterization', 'softmax'),
        selection=ti.get('selection', 'soft'),
    )
    if initialized and p.get('truth_tables') is not None:
        restore_lut_layer(layer, p['indices'], p['truth_tables'])
    return layer


def _draft_connections(node, initialized: bool):
    """lut_layer の train_info.connections を LutLayer の connections 引数に解決する。

    'random' / 'unique' は LutLayer が文字列のまま受ける。構造化配線(rot / butterfly /
    local2d 等)は train_info.wiring の引数で make_connections を呼んで明示 Tensor にする。
    学習済み(initialized)ノードは indices を restore_lut_layer が上書きするので生成しない。
    """
    p = node.params
    ti = node.train_info
    kind = ti.get('connections', 'random')
    if kind in _LUT_LAYER_STRING_CONNECTIONS:
        return kind
    if initialized and p.get('truth_tables') is not None:
        return 'random'
    if kind not in WIRING_SPEC_KEYS:
        raise ValueError(
            f'{node.id}: draft ノードの配線種 {kind!r} は未対応'
            f'(対応: {", ".join(WIRING_SPEC_KEYS)})')
    wiring = ti.get('wiring') or {}
    if not isinstance(wiring, dict):
        raise ValueError(f'{node.id}: train_info.wiring は dict が必要です: {wiring!r}')
    spec = {k: v for k, v in wiring.items() if k != 'dedupe'}
    try:
        return make_connections(
            kind, p['in_width'], p['out_width'], p['n_inputs'],
            dedupe=bool(wiring.get('dedupe')), **spec)
    except ValueError as e:
        raise ValueError(f'{node.id}: {e}') from e


def _build_fixed_lut_layer(node) -> nn.Module:
    """lut_layer → FixedLutLayer。

    LutLayer が受けない幅(ThermometerEncoder の 8 入力、GroupSum(assign='stride') を
    展開した 1 入力の恒等)と kind='thermometer' を、学習パラメータを持たない凍結 LUT
    として復元する。真理値表の無い draft は組めない。
    """
    p = node.params
    ti = node.train_info
    if ti.get('initialized', True) is False or p.get('truth_tables') is None:
        raise ValueError(
            f'{node.id}: n_inputs={p["n_inputs"]} の lut_layer は凍結 LUT として復元する'
            'ため真理値表が必要です(draft は不可)')
    info = {k: v for k, v in ti.items() if k != 'initialized'}
    return FixedLutLayer(
        p['in_width'], p['out_width'], p['n_inputs'], p['indices'], p['truth_tables'],
        train_info=info,
    )


@register_builder('permute')
def _build_permute(node) -> nn.Module:
    """permute → Permute。GroupSum(assign='stride') のネイティブ抽出などの恒等配線。"""
    p = node.params
    return Permute(p['in_shape'], p['perm'])


def _build_dwn_lut_layer(node) -> nn.Module:
    """train_info.kind='dwn' のlut_layer → DWNLayer復元。

    **非可逆な復元**: lutsは真理値表の±1符号化で合成する
    (restore_dwn_layer参照)。eval等価のみ保証し、元の訓練時実数重みは
    復元しない(IRは確定済み回路のみを持つため)。
    """
    p = node.params
    ti = node.train_info
    layer = DWNLayer(
        p['in_width'],
        p['out_width'],
        n_inputs=p['n_inputs'],
        mapping=ti.get('mapping', 'random'),
        efd=ti.get('efd', 'paper'),
    )
    initialized = ti.get('initialized', True)
    if initialized and p.get('truth_tables') is not None:
        restore_dwn_layer(layer, p['indices'], p['truth_tables'])
    return layer


@register_builder('group_sum')
def _build_group_sum(node) -> nn.Module:
    p = node.params
    return GroupSum(k=p['k'], tau=float(p.get('tau', 1.0)), bias=p.get('bias'))


@register_builder('binarize')
def _build_binarize(node) -> nn.Module:
    return Binarize(threshold=float(node.params.get('threshold', 0.5)))


@register_builder('flatten')
def _build_flatten(node) -> nn.Module:
    return nn.Flatten()


class NCHWAdapter(nn.Module):
    """(batch, C*H*W)平坦入力をNCHWへ戻して内包モジュールを適用し、再び平坦化する。

    IRの幅規約(width=C*H*W flatten)と、conv/pool系モジュールの4D入力を
    橋渡しする。simulate(NCHW flatten規約)とビット順が厳密に一致する。
    """

    def __init__(self, module: nn.Module, in_shape):
        super().__init__()
        self.module = module
        self.in_shape = tuple(int(v) for v in in_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.module(x.reshape(x.shape[0], *self.in_shape))
        return y.flatten(1)

    def extra_repr(self):
        return f'in_shape={self.in_shape}'


def _pair(v) -> tuple[int, int]:
    if isinstance(v, list | tuple):
        return int(v[0]), int(v[1])
    return int(v), int(v)


@register_builder('conv2d')
def _build_conv2d(node) -> nn.Module:
    p = node.params
    ti = node.train_info
    c, _, _ = (int(v) for v in p['in_shape'])
    conv = LutConv2d(
        c,
        int(p['out_channels']),
        kernel_size=_pair(p['kernel_size']),
        tree_depth=int(p['tree_depth']),
        stride=_pair(p['stride']),
        padding=_pair(p['padding']),
        n_inputs=int(p['n_inputs']),
        parameterization=ti.get('parameterization', 'softmax'),
        residual_init=False,
        input_channels_per_tree=int(ti.get('input_channels_per_tree', 2)),
        # 推定器。IR に無ければ既定の sigmoid(従来の IR と同じ)
        estimator=ti.get('estimator', 'sigmoid'),
    )
    initialized = ti.get('initialized', True)
    if initialized and p.get('truth_tables') is not None:
        restore_lut_conv2d(conv, p['indices_flat'], p['truth_tables'])
    return NCHWAdapter(conv, p['in_shape'])


@register_builder('or_pool2d')
def _build_or_pool2d(node) -> nn.Module:
    p = node.params
    pool = OrPooling2d(
        kernel_size=int(p['kernel_size']),
        stride=int(p['stride']),
        padding=int(p.get('padding', 0)),
    )
    return NCHWAdapter(pool, p['in_shape'])


@register_builder('blackbox')
def _build_blackbox(node) -> nn.Module:
    raise NotImplementedError(
        f'{node.id}: blackboxノードはIRからの再構築不可'
        '(元のtorchモジュールをIRに保存していないため)'
    )


@register_builder('lut_flat')
def _build_lut_flat(node) -> nn.Module:
    raise NotImplementedError(
        f'{node.id}: lut_flatノードはIRからの再構築不可'
        '(deploy/flatten_dense.py が縮約して作る配備専用ノードで、'
        '重み共有・学習可能パラメータを持たない。元のlut_layer列のIRから'
        'build_torchすること)'
    )


# -- recurrent -----------------------------------------------------
#: bodyグラフ内でのみ関数的に計算するノード型。scan_varは値を持たず、
#: _BodyGraphModule.forward が渡された辞書からそのまま読む
_BODY_FUNCTIONAL_TYPES = FUNCTIONAL_NODE_TYPES + ('scan_var',)


class _BodyGraphModule(nn.Module):
    """recurrentのbodyグラフを実行するモジュール。

    IRGraphModuleとほぼ同じ走査だが、次の2点が違う:
      - 入力ポートを持たない(scan_varノードが「入力の穴」を表すため、
        forwardは辞書 {scan_var_id: Tensor} を受け取る)
      - 全ノードの値を返す(state_update・output_source_idの参照先を
        _RecurrentModule側で自由に選べるようにするため)
    """

    def __init__(self, graph: IRGraph):
        super().__init__()
        in_edges: dict[str, list] = {}
        for e in graph.edges:
            in_edges.setdefault(e.dst, []).append(e)
        self._in_srcs: dict[str, list[str]] = {
            dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
            for dst, edges in in_edges.items()
        }
        self._order: list[str] = [n.id for n in graph.topological_order()]
        self._node_of = {n.id: n for n in graph.nodes}
        self._key_of: dict[str, str] = {}
        mods: dict[str, nn.Module] = {}
        for node in graph.topological_order():
            if node.type in _BODY_FUNCTIONAL_TYPES:
                continue
            key = _sanitize_id(node.id)
            if key in mods:
                raise ValueError(
                    f'ノードid {node.id} のサニタイズ結果 {key} が衝突している'
                )
            mods[key] = _build_module(node)
            self._key_of[node.id] = key
        self.node_modules = nn.ModuleDict(mods)

    def forward(self, scan_values: dict) -> dict:
        """scan_var_id → Tensor の辞書を受け取り、全ノードの値を返す。"""
        values: dict = dict(scan_values)
        for nid in self._order:
            if nid in values:
                continue          # scan_varとして注入済み
            node = self._node_of[nid]
            ins = [values[s] for s in self._in_srcs.get(nid, [])]
            if node.type == 'constant':
                # constantはarity 0だが、batchサイズをscan_valuesから借りる
                any_val = next(iter(scan_values.values()))
                bits = any_val.new_tensor(
                    [float(b) for b in node.params['bits']]
                )
                out = bits.unsqueeze(0).expand(any_val.shape[0], -1)
            elif node.type == 'concat':
                out = torch.cat(ins, dim=-1)
            elif node.type == 'add':
                mode = node.params['mode']
                if mode == 'or':
                    out = _soft_or(ins)
                elif mode == 'xor':
                    out = _soft_xor(ins)
                else:
                    out = sum(ins)
            else:
                out = self.node_modules[self._key_of[nid]](ins[0])
            values[nid] = out
        return values


class _RecurrentModule(nn.Module):
    """IRの`recurrent`ノード → bodyをPython forループでsteps回実行する薄いラッパ。

    往復検証(build_torch→再抽出)用の最小実装。学習には使わない
    (勾配疎通はM2以降の課題)。
    """

    def __init__(self, node):
        super().__init__()
        p = node.params
        self.body = _BodyGraphModule(p['body'])
        self.steps = int(p['steps'])
        self.scan_vars = list(p['scan_vars'])
        self.state_update = dict(p['state_update'])
        self.output_mode = p['output_mode']
        self.output_source_id = p['output_source_id']
        self._ext_svs = [sv for sv in self.scan_vars
                         if sv['role'] in ('step', 'invariant')]
        self._state_svs = [sv for sv in self.scan_vars if sv['role'] == 'state']

    def forward(self, *ext_ins) -> torch.Tensor:
        assert len(ext_ins) == len(self._ext_svs), (
            f'scan_vars(step/invariant)={len(self._ext_svs)}本 だが '
            f'入力={len(ext_ins)}本'
        )
        device = ext_ins[0].device if ext_ins else None
        batch = ext_ins[0].shape[0] if ext_ins else None

        state_vals: dict[str, torch.Tensor] = {}
        for sv in self._state_svs:
            init = self.body._node_of[sv['id']].params['init']
            bits = torch.tensor([float(b) for b in init['bits']],
                                device=device)
            state_vals[sv['id']] = bits.unsqueeze(0).expand(batch, -1)

        outputs = []
        out_t = None
        for t in range(self.steps):
            seed = dict(state_vals)
            for sv, x in zip(self._ext_svs, ext_ins, strict=True):
                w = int(sv['width'])
                if sv['role'] == 'step':
                    seed[sv['id']] = x[:, t * w:(t + 1) * w]
                else:
                    seed[sv['id']] = x
            trace = self.body(seed)
            for sv_id, body_node_id in self.state_update.items():
                state_vals[sv_id] = trace[body_node_id]
            out_t = trace[self.output_source_id]
            if self.output_mode == 'stacked':
                outputs.append(out_t)
        if self.output_mode == 'stacked':
            return torch.cat(outputs, dim=-1)
        return out_t


@register_builder('recurrent')
def _build_recurrent(node) -> nn.Module:
    return _RecurrentModule(node)


def _build_module(node) -> nn.Module:
    builder = BUILDERS.get(node.type)
    if builder is None:
        raise NotImplementedError(f'{node.id}: 未対応ノード型 {node.type}')
    return builder(node)


def is_serial_graph(graph: IRGraph) -> bool:
    """純直列グラフ(全ノード単一入力slot0・fanoutなし・concat/addなし)か"""
    node_ids = {n.id for n in graph.nodes}
    in_count: dict[str, int] = {}
    src_count: dict[str, int] = {}
    for e in graph.edges:
        if e.dst_slot != 0:
            return False
        if e.dst in node_ids:
            in_count[e.dst] = in_count.get(e.dst, 0) + 1
        src_count[e.src] = src_count.get(e.src, 0) + 1
    for n in graph.nodes:
        if n.type in FUNCTIONAL_NODE_TYPES:
            return False
        if n.type == 'recurrent':
            # recurrentは外部入力が0本や複数本のこともあり、常に
            # 多入力ノードと同じ扱い(*ins渡し)にする。
            # nn.SequentialのX=module(X)呼び出し規約と衝突しないよう、
            # 外部入力が1本の場合でも常にIRGraphModule経由にする
            return False
        if in_count.get(n.id, 0) != 1:
            return False
    return all(c <= 1 for c in src_count.values())


def _sanitize_id(node_id: str) -> str:
    """node idをModuleDictキーに使える識別子へ変換する"""
    return re.sub(r'\W', '_', node_id)


def _soft_or(ins: list) -> torch.Tensor:
    """確率OR: 1-(1-a)(1-b)。{0,1}入力に対してはビットORと厳密一致"""
    out = ins[0]
    for t in ins[1:]:
        out = 1 - (1 - out) * (1 - t)
    return out


def _soft_xor(ins: list) -> torch.Tensor:
    """確率XOR: a+b-2ab。{0,1}入力に対してはビットXORと厳密一致"""
    out = ins[0]
    for t in ins[1:]:
        out = out + t - 2 * out * t
    return out


class IRGraphModule(nn.Module):
    """DAG構造のIRを実行するモジュール(v0.2)。

    topological_order()順に各ノードを実行し、values辞書(node_id→Tensor)を
    引き回す。多入力ノードはdst_slot昇順に入力を集める(slot 0が先頭)。
    入力ポートは単一のみ対応(複数入力ポートは将来拡張)。
    """

    def __init__(self, graph: IRGraph):
        super().__init__()
        if len(graph.inputs) != 1:
            raise NotImplementedError(
                f'入力ポートは単一のみ対応(現在{len(graph.inputs)}個)'
            )
        self._input_id = graph.inputs[0].id

        # dst_slot昇順の入力元マップ
        in_edges: dict[str, list] = {}
        for e in graph.edges:
            in_edges.setdefault(e.dst, []).append(e)
        self._in_srcs: dict[str, list[str]] = {
            dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
            for dst, edges in in_edges.items()
        }

        self._order: list[str] = []
        self._node_of = {n.id: n for n in graph.nodes}
        self._key_of: dict[str, str] = {}
        mods: dict[str, nn.Module] = {}
        for node in graph.topological_order():
            self._order.append(node.id)
            if node.type in FUNCTIONAL_NODE_TYPES:
                continue
            key = _sanitize_id(node.id)
            if key in mods:
                raise ValueError(
                    f'ノードid {node.id} のサニタイズ結果 {key} が衝突している'
                )
            mods[key] = _build_module(node)
            self._key_of[node.id] = key
        self.node_modules = nn.ModuleDict(mods)

        # 出力ポートに接続されたノードを優先、なければ最終トポノード
        out_src = graph.output_source()
        self._output_id = out_src if out_src is not None else self._order[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_trace(x)[self._output_id]

    def forward_trace(self, x: torch.Tensor) -> dict:
        """全ノードの出力を返す(node_id→Tensor。推論プローブA5用)"""
        values: dict[str, torch.Tensor] = {self._input_id: x}
        for nid in self._order:
            node = self._node_of[nid]
            # constant等のarity 0ノードは入力エッジを持たない
            ins = [values[s] for s in self._in_srcs.get(nid, [])]
            if node.type == 'constant':
                # 定数行: 全バッチ同一のビット列(dtype/deviceはxに合わせる)
                bits = x.new_tensor(
                    [float(b) for b in node.params['bits']]
                )
                out = bits.unsqueeze(0).expand(x.shape[0], -1)
            elif node.type == 'concat':
                # ビット順規約: slot 0が出力ビット列の先頭(下位ビット側)
                out = torch.cat(ins, dim=-1)
            elif node.type == 'add':
                mode = node.params['mode']
                if mode == 'or':
                    out = _soft_or(ins)
                elif mode == 'xor':
                    out = _soft_xor(ins)
                else:  # sum: 整数和(論理領域の終端専用)
                    out = sum(ins)
            elif node.type == 'recurrent':
                # 外部入力が0〜複数本ありうるので*insで渡す
                out = self.node_modules[self._key_of[nid]](*ins)
            else:
                out = self.node_modules[self._key_of[nid]](ins[0])
            values[nid] = out
        return values


def build_torch(graph: IRGraph) -> nn.Module:
    """IRからPyTorchモデルを構築する。

    Args:
        graph: 検証済みIR

    Returns:
        直列グラフ: topological_order()順のnn.Sequential(従来互換)
        DAGグラフ: IRGraphModule
    """
    if is_serial_graph(graph):
        modules = [_build_module(node) for node in graph.topological_order()]
        return nn.Sequential(*modules)
    return IRGraphModule(graph)
