"""DLGN構造統計ライブラリ(SM1)。

学習前トポロジー・学習中ダイナミクス・学習後構造の計測を、エディタと
headless実験の両方から同一実装で行うための単一情報源。
ゲート確率・エントロピー・確信度の計算は tools/editor/backend/extract.py
からここへ昇格した(editor側は本モジュールを再exportする薄いラッパ)。

方針(構造メトリクスの設計方針):
    - 返り値はJSON化可能なplain dictを原則とする(numpyスカラは組込み型へ変換)
    - torch層(LutLayer)とIRGraphの両対応が自然なものは両対応
    - sweep(ir/sweep.py)や実験スクリプトから流用できる内部API
      (ndarrayを返す *_masks / *_counts)は分離して公開する
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    'tt_to_gate_id_n2',
    'get_gate_ids',
    'get_confidence',
    'gate_probs_from_weights',
    'gate_entropy',
    'neuron_uncertainty',
    'layer_entropy',
    'summary_from_weights',
    'lut_layers_with_ids',
    'fanout_counts',
    'fanout_stats',
    'support_sizes',
    'gate_class_rates',
    'degenerate_input_stats',
    'wiring_locality',
    'reachable_masks',
    'reach_cone',
    'soft_eval_acc',
    'selection_gap_eval',
    'metric_layers_with_ids',
    'dwn_uncertainty',
    'wiring_indices',
    'StructMetricsTracker',
    'struct_final_report',
]


# ---------------------------------------------------------------------------
# ゲート確率・確信度(tools/editor/backend/extract.py から昇格)
# ---------------------------------------------------------------------------

def tt_to_gate_id_n2(tt: int) -> int:
    """N=2真理値表(bit k=butterfly基底順)を16ゲートIDへ変換する。

    ゲートID規約: 真理値表を(0,0),(0,1),(1,0),(1,1)順でMSBから読んだ2進数
    (core/basis.py の _build_gate_table_n2 と同一)。
    """
    b00 = tt & 1
    b10 = (tt >> 1) & 1
    b01 = (tt >> 2) & 1
    b11 = (tt >> 3) & 1
    return (b00 << 3) | (b01 << 2) | (b10 << 1) | b11


def get_gate_ids(layer) -> np.ndarray:
    """LutLayerのゲートIDを取得する。

    Returns:
        (out_dim,) int配列。N=2では0-15、N>2ではargmaxインデックス。
    """
    with torch.no_grad():
        weights = layer.weights.detach().cpu()

        if layer.parameterization == 'iwp':
            # IWP: sigmoid → 0/1 で等価ゲートの真理値表を計算
            p = (torch.sigmoid(weights) > 0.5).long()
            gate_ids = torch.zeros(layer.out_dim, dtype=torch.long)
            for j in range(layer.n_inputs):
                gate_ids += p[:, j] << j
            return gate_ids.numpy()

        if getattr(layer, '_use_16gate', False):
            # N=2 softmax: 16ゲートモード
            return weights.argmax(dim=-1).numpy()

        if layer.parameterization in ('warp', 'raw'):
            # 真理値表を確定してからIDへ(離散化はレイヤ自身のevalパスに委譲)
            from griffith_logic.export import truth_tables_from_lut_layer
            tables = truth_tables_from_lut_layer(layer)
            if layer.n_inputs == 2:
                ids = [tt_to_gate_id_n2(int(t)) for t in tables]
            else:
                # N>2は色分け用の安定なintに丸める
                ids = [int(t) & 0xFFFF for t in tables]
            return np.array(ids, dtype=np.int64)

        # softmax(N>2) / independent: エントリのargmax
        return weights.argmax(dim=-1).numpy()


def get_confidence(layer) -> np.ndarray:
    """各ニューロンの確信度(0.0〜1.0)を取得する。"""
    with torch.no_grad():
        weights = layer.weights.detach().cpu()

        if layer.parameterization == 'softmax':
            probs = F.softmax(weights, dim=-1)
            return probs.max(dim=-1).values.numpy()

        if layer.parameterization == 'independent':
            sig = torch.sigmoid(weights)
            # 各エントリの0/1への近さの平均
            conf = torch.max(sig, 1.0 - sig).mean(dim=-1)
            return conf.numpy()

        if layer.parameterization == 'iwp':
            sig = torch.sigmoid(weights)
            conf = torch.max(sig, 1.0 - sig).prod(dim=-1)
            return conf.numpy()

    # warp / raw: 確信度の定義が自明でないため中立値
    return np.ones(layer.out_dim, dtype=np.float32) * 0.5


def gate_probs_from_weights(
    weights: torch.Tensor, parameterization: str, n_inputs: int,
) -> torch.Tensor | None:
    """weightsテンソルからゲート確率分布を導出する。

    softmax: 全16(または2^N)ゲート上の確率分布 / independent・iwp: sigmoid /
    raw: [0,1]クランプ。warpはWH係数のため確率解釈がなくNone。
    """
    with torch.no_grad():
        w = weights.detach().to(torch.float32).cpu()
        if parameterization == 'softmax':
            return F.softmax(w, dim=-1)
        if parameterization in ('independent',):
            return torch.sigmoid(w)
        if parameterization == 'iwp':
            return torch.sigmoid(w)
        if parameterization == 'raw':
            return w.clamp(0.0, 1.0)
    return None


def gate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """ゲート確率分布のニューロン毎カテゴリカルエントロピー(自然対数、nat)。

    最終次元を正規化された確率分布とみなす(softmax専用の解釈)。
    gate_distributionフレームの既存表示と互換のため無次元化しない。
    """
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def neuron_uncertainty(
    weights: torch.Tensor, parameterization: str, n_inputs: int,
) -> torch.Tensor | None:
    """ニューロン毎の「迷い」を正規化[0,1]で返す(§4-1)。

    パラメータ化ごとに確率モデルが異なるため式を使い分ける:
    - softmax: 分布サイズKのカテゴリカルエントロピー / ln(K)
      (16ゲートモードはK=16、エントリ直接パラメータ化はK=2^N)
    - independent/iwp/raw: 各エントリは独立ベルヌーイ。
      二値エントロピー h(p)=-(p·ln p+(1-p)·ln(1-p)) の平均 / ln(2)
    - warp: 確率解釈がないためNone
    正規化により層間・パラメータ化間で比較可能(uncertain_gatesの順位付け前提)。
    """
    probs = gate_probs_from_weights(weights, parameterization, n_inputs)
    if probs is None:
        return None
    if parameterization == 'softmax':
        k = int(probs.shape[-1])
        if k <= 1:
            return torch.zeros(probs.shape[0])
        return gate_entropy(probs) / math.log(k)
    # float32で1-epsが1.0に丸められないよう1e-7を使う(1e-12だとlog(0)=NaN)
    eps = 1e-7
    p = probs.clamp(eps, 1.0 - eps)
    h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return (h / math.log(2.0)).mean(dim=-1)


def layer_entropy(layer) -> torch.Tensor | None:
    """レイヤのニューロン毎の迷い(正規化[0,1])。確率解釈不能ならNone(warp)。"""
    with torch.no_grad():
        return neuron_uncertainty(
            layer.weights, layer.parameterization, int(layer.n_inputs),
        )


def summary_from_weights(
    weights: torch.Tensor, parameterization: str, n_inputs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """スナップショットのweightsからgate_ids/confidenceを導出する。

    get_gate_ids/get_confidence のスナップショット版(レイヤオブジェクト不要)。
    warpは真理値表確定にレイヤ本体が要るため中立値を返す。
    """
    with torch.no_grad():
        w = weights.detach().to(torch.float32).cpu()
        out_dim = w.shape[0]

        if parameterization == 'iwp':
            p = (torch.sigmoid(w) > 0.5).long()
            gate_ids = torch.zeros(out_dim, dtype=torch.long)
            for j in range(n_inputs):
                gate_ids += p[:, j] << j
            conf = torch.max(torch.sigmoid(w), 1.0 - torch.sigmoid(w)).prod(dim=-1)
            return gate_ids.numpy(), conf.numpy()

        if parameterization == 'softmax':
            probs = F.softmax(w, dim=-1)
            return (
                w.argmax(dim=-1).numpy(),
                probs.max(dim=-1).values.numpy(),
            )

        if parameterization == 'independent':
            sig = torch.sigmoid(w)
            conf = torch.max(sig, 1.0 - sig).mean(dim=-1)
            return w.argmax(dim=-1).numpy(), conf.numpy()

        if parameterization == 'raw':
            bits = (w.clamp(0.0, 1.0) > 0.5).long()
            if n_inputs == 2:
                ids = []
                for row in bits:
                    tt = int(sum(int(b) << k for k, b in enumerate(row)))
                    ids.append(tt_to_gate_id_n2(tt))
                gate_ids = np.array(ids, dtype=np.int64)
            else:
                gate_ids = w.argmax(dim=-1).numpy()
            dist = torch.min((w - 0.0).abs(), (w - 1.0).abs()).mean(dim=-1)
            conf = (1.0 - 2.0 * dist).clamp(0.0, 1.0)
            return gate_ids, conf.numpy()

    # warp等: 中立値
    return (
        np.zeros(weights.shape[0], dtype=np.int64),
        np.ones(weights.shape[0], dtype=np.float32) * 0.5,
    )


def lut_layers_with_ids(model) -> list[tuple[str, object]]:
    """モデルから (IRノードid, LutLayer) の一覧を返す。

    直列(Sequential): export_ir と同じ規約 `n{i}_{クラス名小文字}` で採番する。
    DAG(IRGraphModule内包): 元のIRノードidをそのまま使う(build時にidを
    張り替えない規約と対応)。いずれもWS layer_summary/購読の突き合わせ用。
    """
    from griffith_logic import LutLayer
    from griffith_logic.ir import IRGraphModule

    gmod = next(
        (m for m in model.modules() if isinstance(m, IRGraphModule)), None,
    )
    if gmod is not None:
        return [
            (node_id, gmod.node_modules[key])
            for node_id, key in gmod._key_of.items()
            if isinstance(gmod.node_modules[key], LutLayer)
        ]

    layers = []
    modules = list(model) if isinstance(model, torch.nn.Sequential) else [model]
    for i, module in enumerate(modules):
        if isinstance(module, LutLayer):
            layers.append((f'n{i}_{type(module).__name__.lower()}', module))
    return layers


# ---------------------------------------------------------------------------
# 配線トポロジー統計(SM1新規): fanout / 縮退 / 局所性
# ---------------------------------------------------------------------------

def _to_indices_array(indices) -> np.ndarray:
    """indices(torch.Tensor / ndarray / list)を (n_inputs, out_dim) int64 へ。"""
    if isinstance(indices, torch.Tensor):
        arr = indices.detach().cpu().numpy()
    else:
        arr = np.asarray(indices)
    arr = arr.astype(np.int64, copy=False)
    if arr.ndim != 2:
        raise ValueError(f'indicesは2次元 (n_inputs, out_dim) が必要: {arr.shape}')
    return arr


def _tt_ints(truth_tables) -> list[int]:
    """真理値表列(int / hex文字列 混在可)をint列へ。"""
    return [int(t, 16) if isinstance(t, str) else int(t) for t in truth_tables]


def fanout_counts(indices, in_width: int) -> np.ndarray:
    """各入力ワイヤの被参照数(スロット重複込み)を返す。

    build_reverse_index(core/connections.py)の rev_starts.diff() と一致する
    (テストで照合)。

    Returns:
        (in_width,) int64。counts[x] = ワイヤxを参照するスロットの総数
    """
    arr = _to_indices_array(indices)
    if arr.size and (arr.min() < 0 or arr.max() >= in_width):
        raise ValueError('indicesがin_widthの範囲外')
    return np.bincount(arr.ravel(), minlength=in_width).astype(np.int64)


def fanout_stats(indices, in_width: int, top_n: int = 10) -> dict:
    """1層分のファンアウト分布サマリ(JSON化可能)。

    Returns:
        {'in_width', 'mean', 'max', 'zero_frac', 'hist': {fanout値: ワイヤ本数},
         'top': [[wire, count], ...]}
    """
    counts = fanout_counts(indices, in_width)
    vals, freq = np.unique(counts, return_counts=True)
    order = np.argsort(counts)[::-1][:top_n]
    return {
        'in_width': int(in_width),
        'mean': float(counts.mean()) if in_width else 0.0,
        'max': int(counts.max()) if in_width else 0,
        'zero_frac': float((counts == 0).mean()) if in_width else 0.0,
        'hist': {int(v): int(f) for v, f in zip(vals, freq, strict=True)},
        'top': [[int(i), int(counts[i])] for i in order],
    }


def degenerate_input_stats(indices) -> dict:
    """縮退ゲート(複数スロットが同一ワイヤを指す=実効アリティ低下)の統計。"""
    arr = _to_indices_array(indices)
    n_inputs, out_dim = arr.shape
    if out_dim == 0 or n_inputs < 2:
        return {'rate': 0.0, 'count': 0, 'out_dim': int(out_dim)}
    srt = np.sort(arr, axis=0)
    degenerate = (np.diff(srt, axis=0) == 0).any(axis=0)
    return {
        'rate': float(degenerate.mean()),
        'count': int(degenerate.sum()),
        'out_dim': int(out_dim),
    }


def wiring_locality(indices, in_width: int) -> dict:
    """配線局所性のproxy統計。

    - spread: ゲートごとの入力インデックス幅 (max-min)/(in_width-1)。
      ※インデックス空間は物理配置の代理にすぎない(placerが並べ替えるため、
      SM6の配置配線後実測との相関検証が前提の指標)
    - pair_share_expected: ランダムに選んだ2ゲートが共有する入力ワイヤ数の期待値
      = Σ_x C(g_x, 2) / C(out_dim, 2)(g_x = ワイヤxを使う異なりゲート数)
    """
    arr = _to_indices_array(indices)
    n_inputs, out_dim = arr.shape
    if out_dim == 0:
        return {'spread_mean': 0.0, 'spread_p50': 0.0, 'spread_p90': 0.0,
                'pair_share_expected': 0.0}
    denom = max(in_width - 1, 1)
    spread = (arr.max(axis=0) - arr.min(axis=0)) / denom
    # ワイヤごとの「異なりゲート数」g_x(同一ゲートの重複スロットは1と数える)
    gate_ids = np.tile(np.arange(out_dim, dtype=np.int64), n_inputs)
    pairs = np.unique(np.stack([arr.ravel(), gate_ids], axis=1), axis=0)
    g_x = np.bincount(pairs[:, 0], minlength=in_width).astype(np.float64)
    share_pairs = (g_x * (g_x - 1) / 2.0).sum()
    total_pairs = out_dim * (out_dim - 1) / 2.0
    return {
        'spread_mean': float(spread.mean()),
        'spread_p50': float(np.percentile(spread, 50)),
        'spread_p90': float(np.percentile(spread, 90)),
        'pair_share_expected': float(share_pairs / total_pairs)
        if total_pairs else 0.0,
    }


# ---------------------------------------------------------------------------
# 真理値表統計(SM1新規): 実効サポート(定数・恒等の検出)
# ---------------------------------------------------------------------------

def support_sizes(truth_tables, n_inputs: int) -> np.ndarray:
    """各ゲートの実効依存入力数(サポートサイズ)を返す。

    定義: 入力jに非依存 ⇔ 全パターンkで tt[k] == tt[k ^ (1<<j)]。
    support=0が定数(FALSE/TRUE)、=1が恒等/否定(パススルー)。
    ゲートID表に依存しないためN=2/4/6すべて同一コード。

    Args:
        truth_tables: int列またはhex文字列列(IRのparams['truth_tables']そのまま可)
        n_inputs: N(パターン数=2^N)

    Returns:
        (num_gates,) int64
    """
    tts = np.array(_tt_ints(truth_tables), dtype=np.uint64)
    size = 1 << n_inputs
    ks = np.arange(size, dtype=np.uint64)
    bits = ((tts[:, None] >> ks[None, :]) & np.uint64(1)).astype(np.uint8)
    support = np.zeros(len(tts), dtype=np.int64)
    for j in range(n_inputs):
        flipped = ks ^ np.uint64(1 << j)
        depends = (bits != bits[:, flipped.astype(np.int64)]).any(axis=1)
        support += depends.astype(np.int64)
    return support


def gate_class_rates(truth_tables, n_inputs: int) -> dict:
    """定数ゲート率・パススルーゲート率とサポート分布(JSON化可能)。

    EDAツールが掃除する論理(定数伝播・恒等除去)の量の直接指標。
    学習中のスパース正則化が「EDAが消せる形」に効いているかを測る。
    """
    s = support_sizes(truth_tables, n_inputs)
    if len(s) == 0:
        return {'const_rate': 0.0, 'passthrough_rate': 0.0,
                'support_hist': {}, 'num_gates': 0}
    vals, freq = np.unique(s, return_counts=True)
    return {
        'const_rate': float((s == 0).mean()),
        'passthrough_rate': float((s == 1).mean()),
        'support_hist': {int(v): int(f) for v, f in zip(vals, freq, strict=True)},
        'num_gates': int(len(s)),
    }


# ---------------------------------------------------------------------------
# 到達コーン(SM1新規): IR上の後方到達可能性+コーン深度
# ---------------------------------------------------------------------------

def _in_srcs(graph) -> dict[str, list[str]]:
    """ノードid → dst_slot昇順の入力元idリスト(simulate.pyと同一規約)。"""
    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    return {
        dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
        for dst, edges in grouped.items()
    }


def _maybe_lowered(graph):
    """LOWERABLE_TYPES(conv2d/or_pool2d/upsample2d/permute)を含む場合は
    lut_layer列へ展開したグラフを返す。

    到達・深度はゲート粒度の指標のため、展開後(=HDLと同じ粒度)で計測する。
    ノード型の一覧は export/lowering.LOWERABLE_TYPES が単一情報源(複製しない)。
    """
    from ..export.lowering import LOWERABLE_TYPES, lower_to_primitives
    if any(n.type in LOWERABLE_TYPES for n in graph.nodes):
        return lower_to_primitives(graph)
    return graph


def reachable_masks(graph) -> dict[str, np.ndarray]:
    """出力ポートから後方到達可能な出力ビットのマスクを全ノードについて返す。

    ビット粒度の正確な後方伝播:
        lut_layer: 出力ビットo → 入力ビット indices[:, o]
        group_sum: 出力グループg → 入力ビット [g*gs, (g+1)*gs)
        binarize/flatten: 同一ビット
        concat: スロット範囲へ分配 / add: 全スロットの同一ビット
        blackbox: 保守的に全ビット(過大評価side=安全)
    conv2d/or_pool2dは展開後グラフで計測(ノードidは展開後のもの)。

    Returns:
        {node_id: (out_width,) bool}。内部API(ndarrayを返す)。
        JSONサマリは reach_cone() を使う。
    """
    from ..ir.validate import node_out_width

    graph = _maybe_lowered(graph)
    nodes = {n.id: n for n in graph.nodes}
    input_ids = {p.id for p in graph.inputs}
    needed: dict[str, np.ndarray] = {
        n.id: np.zeros(node_out_width(n), dtype=bool) for n in graph.nodes
    }
    for p in graph.inputs:
        needed[p.id] = np.zeros(p.width, dtype=bool)

    in_srcs = _in_srcs(graph)

    # 出力ポートに接続されたノードは全ビット必要
    out_ids = {p.id for p in graph.outputs}
    for e in graph.edges:
        if e.dst in out_ids and e.src not in input_ids:
            needed[e.src][:] = True

    for node in reversed(graph.topological_order()):
        mask = needed[node.id]
        if not mask.any():
            continue
        srcs = in_srcs.get(node.id, [])
        if not srcs:
            continue  # constant等
        p = node.params
        if node.type == 'lut_layer':
            idx = np.asarray(p['indices'], dtype=np.int64)
            used = np.unique(idx[:, mask])
            needed[srcs[0]][used] = True
        elif node.type == 'group_sum':
            gs = p['in_width'] // p['k']
            src_mask = needed[srcs[0]]
            for g in np.nonzero(mask)[0]:
                src_mask[g * gs:(g + 1) * gs] = True
        elif node.type in ('binarize', 'flatten'):
            needed[srcs[0]][:len(mask)] |= mask
        elif node.type == 'concat':
            offset = 0
            for slot, src in enumerate(srcs):
                w = int(p['in_widths'][slot])
                needed[src][:w] |= mask[offset:offset + w]
                offset += w
        elif node.type == 'add':
            for src in srcs:
                needed[src][:len(mask)] |= mask
        else:  # blackbox等: 保守的に全ビット
            for src in srcs:
                needed[src][:] = True

    # ノード分のみ返す(入力ポートは全ビット到達扱いにしない=情報として返す)
    return {nid: needed[nid] for nid in list(nodes) + list(input_ids)}


def reach_cone(graph) -> dict:
    """到達コーンのJSONサマリ: 到達不能ゲート率+出力コーン深度分布。

    深度 = 入力からその出力ビットまでに通るlut_layer段数の最大値
    (estimate_resourcesのlogic_depth=グラフ最大値を、出力ビット別分布に拡張)。
    """
    from ..ir.validate import node_out_width

    graph = _maybe_lowered(graph)
    masks = reachable_masks(graph)

    per_node = {}
    total = 0
    unreachable = 0
    for n in graph.nodes:
        if n.type != 'lut_layer':
            continue
        m = masks[n.id]
        cnt = int((~m).sum())
        per_node[n.id] = {
            'out_width': int(m.size),
            'unreachable': cnt,
            'unreachable_frac': float(cnt / m.size) if m.size else 0.0,
        }
        total += int(m.size)
        unreachable += cnt

    # 前方パスでビット別深度
    depth: dict[str, np.ndarray] = {
        p.id: np.zeros(p.width, dtype=np.int64) for p in graph.inputs
    }
    in_srcs = _in_srcs(graph)
    for node in graph.topological_order():
        srcs = in_srcs.get(node.id, [])
        p = node.params
        if not srcs:
            depth[node.id] = np.zeros(node_out_width(node), dtype=np.int64)
            continue
        if node.type == 'lut_layer':
            din = depth[srcs[0]]
            idx = np.asarray(p['indices'], dtype=np.int64)
            depth[node.id] = din[idx].max(axis=0) + 1
        elif node.type == 'group_sum':
            din = depth[srcs[0]]
            depth[node.id] = din.reshape(p['k'], -1).max(axis=1)
        elif node.type in ('binarize', 'flatten'):
            depth[node.id] = depth[srcs[0]].copy()
        elif node.type == 'concat':
            depth[node.id] = np.concatenate([depth[s] for s in srcs])
        elif node.type == 'add':
            depth[node.id] = np.stack([depth[s] for s in srcs]).max(axis=0)
        else:  # blackbox等: 入力の最大深度を引き継ぐ(段数は加算しない)
            dmax = max((int(depth[s].max()) if depth[s].size else 0)
                       for s in srcs)
            depth[node.id] = np.full(
                node_out_width(node), dmax, dtype=np.int64,
            )

    out_src = graph.output_source()
    if out_src is not None and out_src in depth:
        out_depths = depth[out_src]
    else:
        last = graph.topological_order()[-1].id
        out_depths = depth[last]
    vals, freq = np.unique(out_depths, return_counts=True)

    return {
        'total_lut_gates': total,
        'unreachable_gates': unreachable,
        'unreachable_frac': float(unreachable / total) if total else 0.0,
        'per_node': per_node,
        'output_depth': {
            'max': int(out_depths.max()) if out_depths.size else 0,
            'mean': float(out_depths.mean()) if out_depths.size else 0.0,
            'hist': {int(v): int(f) for v, f in zip(vals, freq, strict=True)},
        },
    }


# ---------------------------------------------------------------------------
# 学習中メトリクス(SM4): flip率・selection gap・run末レポート
# ---------------------------------------------------------------------------

def soft_eval_acc(model, loader, metric_fn, device: str = 'cpu') -> float:
    """train mode(連続緩和)のままno_gradで精度を測る(soft精度)。"""
    was = model.training
    model.train()
    total = 0
    acc_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            n = int(y.shape[0])
            acc_sum += float(metric_fn(model(x), y)) * n
            total += n
    model.train(mode=was)
    return acc_sum / total if total else float('nan')


def selection_gap_eval(model, loader, metric_fn, device: str = 'cpu') -> dict:
    """soft/hard 2パス評価(editor backend/training.py §4-1と同定義)。

    hard: model.eval()=離散化+入力round / soft: model.train()=連続緩和。
    selection_gap = soft_acc − hard_acc(hard化で失う精度)。
    """
    was = model.training
    model.eval()
    total = 0
    acc_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(device).round()
            y = y.to(device)
            n = int(y.shape[0])
            acc_sum += float(metric_fn(model(x), y)) * n
            total += n
    model.train(mode=was)
    hard = acc_sum / total if total else float('nan')
    soft = soft_eval_acc(model, loader, metric_fn, device)
    return {'hard_acc': hard, 'soft_acc': soft,
            'selection_gap': soft - hard}


def metric_layers_with_ids(model) -> list[tuple[str, object, str]]:
    """モデルから (IRノードid, 層, 種別) の一覧を返す(SM10)。

    種別: 'lut'=LutLayer / 'dwn'=DWNLayer。採番規約は lut_layers_with_ids と
    同一(Sequential=`n{i}_{クラス名小文字}`、IRGraphModuleは元ノードid)。
    LutConv2d等はまだ対象外。
    """
    from griffith_logic import LutLayer
    from griffith_logic.ir import IRGraphModule
    from griffith_logic.layers.dwn import DWNLayer

    def _kind(m) -> str | None:
        if isinstance(m, LutLayer):
            return 'lut'
        if isinstance(m, DWNLayer):
            return 'dwn'
        return None

    gmod = next(
        (m for m in model.modules() if isinstance(m, IRGraphModule)), None,
    )
    if gmod is not None:
        return [
            (node_id, gmod.node_modules[key], _kind(gmod.node_modules[key]))
            for node_id, key in gmod._key_of.items()
            if _kind(gmod.node_modules[key])
        ]

    result = []
    modules = list(model) if isinstance(model, torch.nn.Sequential) else [model]
    for i, module in enumerate(modules):
        kind = _kind(module)
        if kind:
            result.append(
                (f'n{i}_{type(module).__name__.lower()}', module, kind)
            )
    return result


def dwn_uncertainty(layer) -> torch.Tensor:
    """DWNLayerのノード毎の迷い(正規化[0,1])。

    テーブル実数値の二値化閾値0からの距離: 1 − |clamp(luts, −1, 1)| の
    ノード平均。LutLayer系のneuron_uncertaintyと同じスケールで集約できる。
    """
    with torch.no_grad():
        t = layer.luts.detach().cpu().clamp(-1.0, 1.0)
        return (1.0 - t.abs()).mean(dim=-1)


def wiring_indices(layer) -> np.ndarray | None:
    """学習配線(LearnableMapping)の現在のargmax配線を返す。

    DWNLayer(mapping='learnable')とLutLayer(wiring='learnable')の共通属性
    `learnable_mapping.weights`(in, out×n)の列ごとargmax。固定配線層はNone。

    Returns:
        (out×n,) int64 または None
    """
    lm = getattr(layer, 'learnable_mapping', None)
    if lm is None:
        return None
    with torch.no_grad():
        return lm.weights.detach().cpu().argmax(dim=0).numpy()


def _tt_bits(truth_tables, n_inputs: int) -> np.ndarray:
    """真理値表列を平坦なビット配列 (num_gates×2^n,) uint8 へ展開する。"""
    size = 1 << n_inputs
    arr = np.array([int(t) for t in truth_tables], dtype=np.uint64)
    ks = np.arange(size, dtype=np.uint64)
    return ((arr[:, None] >> ks[None, :]) & np.uint64(1)).astype(np.uint8).ravel()


class StructMetricsTracker:
    """学習中の構造メトリクス収集(SM4、SM10でDWN対応)。

    前回計測時の離散状態を保持し、flip率(構造収束の指標)を出す。
    トレーナ非依存(nn.Moduleを直接受ける。runner/experimentsの両方から使える)。
    対象はLutLayerとDWNLayer(LutConv2d等は対象外)。
    返り値は `struct/` プレフィックスのflat dict(wandb.logへそのまま渡せる)。

    flip率の決定単位は層種別で異なる(全体値は決定単位数の加重平均):
        LutLayer: ニューロン単位(argmaxゲートIDの変化)
        DWNLayer: 真理値表ビット単位(二値化テーブルbitの変化)
    """

    def __init__(self, model, prefix: str = 'struct/'):
        self.prefix = prefix
        self.layers = metric_layers_with_ids(model)
        self._prev_state: dict[str, np.ndarray] = {}
        self._prev_wiring: dict[str, np.ndarray] = {}

    def collect(self) -> dict:
        """軽量メトリクス一式を返す(eval間隔ごとに呼ぶ)。

        - flip_rate: 前回collect時と離散状態が変わった決定単位率
          (初回呼び出しでは比較対象がないためキー省略)
        - wiring_flip_rate: 学習配線(LearnableMapping)のargmax変化スロット率。
          学習配線を持つ層のみ(DWNのLM収束・LutLayer learnable wiring)
        - entropy_mean / uncertainty_p90: 正規化[0,1]の迷い
          (LutLayerはエントロピー系、DWNは1−|テーブル値|。warp層は対象外)
        - const_rate / passthrough_rate: 現時点の離散化真理値表のサポート0/1率
          (=EDAが消せる形にどれだけ寄っているか。スパース正則化の直接指標)
        """
        from griffith_logic.export import truth_tables_from_lut_layer
        from griffith_logic.export.discretize import truth_tables_from_dwn

        p = self.prefix
        out: dict = {}
        flip_num = 0
        flip_den = 0
        wf_num = 0
        wf_den = 0
        unc_parts = []
        const_num = 0
        pass_num = 0
        gates = 0
        for node_id, layer, kind in self.layers:
            n_in = int(layer.n_inputs)
            if kind == 'lut':
                tts = truth_tables_from_lut_layer(layer)
                state = np.asarray(get_gate_ids(layer))
                unc = layer_entropy(layer)
            else:  # dwn
                tts = truth_tables_from_dwn(layer)
                state = _tt_bits(tts, n_in)
                unc = dwn_uncertainty(layer)

            prev = self._prev_state.get(node_id)
            if prev is not None and prev.shape == state.shape:
                flips = int((state != prev).sum())
                out[f'{p}flip_rate/{node_id}'] = flips / state.size
                flip_num += flips
                flip_den += state.size
            self._prev_state[node_id] = state

            w = wiring_indices(layer)
            if w is not None:
                pw = self._prev_wiring.get(node_id)
                if pw is not None and pw.shape == w.shape:
                    wf = int((w != pw).sum())
                    out[f'{p}wiring_flip_rate/{node_id}'] = wf / w.size
                    wf_num += wf
                    wf_den += w.size
                self._prev_wiring[node_id] = w

            if unc is not None and unc.numel():
                unc_parts.append(unc.numpy())

            s = support_sizes([int(t) for t in tts], n_in)
            const_num += int((s == 0).sum())
            pass_num += int((s == 1).sum())
            gates += len(s)

        if flip_den:
            out[f'{p}flip_rate'] = flip_num / flip_den
        if wf_den:
            out[f'{p}wiring_flip_rate'] = wf_num / wf_den
        if unc_parts:
            unc_all = np.concatenate(unc_parts)
            out[f'{p}entropy_mean'] = float(unc_all.mean())
            out[f'{p}uncertainty_p90'] = float(np.percentile(unc_all, 90))
        if gates:
            out[f'{p}const_rate'] = const_num / gates
            out[f'{p}passthrough_rate'] = pass_num / gates
        return out

    def gate_histograms(self) -> dict:
        """{node_id: {gate_id: 個数}}(LutLayerのみ。DWNの64bit表は対象外)。

        wandb.Histogram化は呼び出し側で行う。
        """
        result = {}
        for node_id, layer, kind in self.layers:
            if kind != 'lut':
                continue
            gids = get_gate_ids(layer)
            vals, freq = np.unique(gids, return_counts=True)
            result[node_id] = {
                int(v): int(f) for v, f in zip(vals, freq, strict=True)
            }
        return result


def struct_final_report(
    model,
    input_width: int,
    *,
    lut_k: int | None = None,
    yosys: str | None = None,
    prefix: str = 'struct/',
) -> dict:
    """run末の重い構造レポート: IR抽出 → sweep(SM2)→ 到達コーン。

    lut_k指定時はpacked LUT proxy(SM3)も試みる(yosys未解決なら通知の上
    スキップ。キー自体を省略する=黙って0を返さない)。

    Returns:
        `struct/` プレフィックスのflat dict(wandb.log/JSON保存の両用)
    """
    from griffith_logic.export import export_ir

    from ..ir.sweep import sweep

    graph = export_ir(model, input_width)
    _, srep = sweep(graph)
    cone = reach_cone(graph)
    p = prefix
    out = {
        f'{p}gates_raw': srep['gates_before'],
        f'{p}sweep_gates': srep['swept_gates'],
        f'{p}sweep_ratio': srep['sweep_ratio'],
        f'{p}passthrough_remaining': srep['passthrough_remaining'],
        f'{p}const_remaining': srep['const_remaining'],
        f'{p}unreachable_frac': cone['unreachable_frac'],
        f'{p}logic_depth_max': cone['output_depth']['max'],
        f'{p}logic_depth_mean': cone['output_depth']['mean'],
    }
    if lut_k is not None:
        from .lut_pack import packed_lut_stats
        try:
            packed = packed_lut_stats(graph, k=lut_k, yosys=yosys)
        except RuntimeError as e:
            print(f'packed LUT proxyスキップ: {e}')
        else:
            out[f'{p}packed_luts'] = packed['lut_count']
            if packed['gates_per_lut'] is not None:
                out[f'{p}gates_per_lut'] = packed['gates_per_lut']
            if packed['depth'] is not None:
                out[f'{p}packed_depth'] = packed['depth']
    return out
