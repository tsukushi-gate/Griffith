"""native DLGN転写IRの幅拡張(widen、ROADMAP-v5)。

fan-in2転写初期化パイプラインは、枝刈り済みMLPを厳密転写して
得た「teacher由来の細いnative DLGN IR」を出発点にする。転写IRは各隠れ
ニューロンが正確にfan-in2のLUTになっており、rebuild_trainable(softmax16g)で
学習可能DLGNへ復元できる(convert/recover.py)。

本モジュールはその転写IRを**teacher初期化を保存したまま f 倍に広げる**。
native DLGN(Petersen系)はスクラッチだと6k〜24kゲートを使うが、転写IRは
数百〜千ゲートしかない。teacher由来ノードを捨てずに幅だけ増やすことで、
「事前学習の骨格 + 追加容量」という初期化を作る(接続最適化2507.06173の
Top-K候補固定+重み学習、LILogic Netの候補集合先固定と同型の発想)。

保存則(この関数が保証する不変量):
    1. teacher由来ノードは各隠れ層の**先頭**に配置される。よって層l+1の
       teacher由来ノードが持つ層l向けindicesは値が不変のまま有効
       (先頭W個のスロットは元のノードを指し続ける)。
    2. init_new_final='const0' なら最終層に追加する新ゲートは tt=0
       (常に0を出力)で初期化される。group_sumのスコアに寄与しないため、
       widen直後の hard argmax は元IRと一致する(初期精度を保存)。

新ノードの初期化(隠れ層):
    確率 jitter_frac で「teacherノードのクローン + 片方のindexを一様再サンプル」、
    それ以外は完全ランダム(tt一様16ゲート、indices一様)。
    どちらもseed決定的。

制約:
    - 対象は transcribe_mlp / prune転写が生成する直列IR
      (入力'x' → lut_layer列(n_inputs=2) → 最終lut_layer → group_sum → 'y')。
    - 隠れ層LUTはfan-in2(n_inputs=2)前提。最終層は縮退LUT(n_inputs=2)群。
    - 元IRは無改造(deepcopyして返す)。
"""

from __future__ import annotations

import copy

import numpy as np

from ..ir.schema import IRGraph, IRNode
from ..ir.serialize import tt_from_hex, tt_to_hex

INIT_NEW_FINAL_POLICIES = ('const0', 'random')


def _hidden_and_final_ids(graph: IRGraph) -> tuple[list[str], str, str]:
    """直列転写IRから (隠れlut_layer id列, 最終lut_layer id, group_sum id) を返す。

    最終lut_layer = 後続が group_sum のlut_layer。隠れlut_layer = それ以外の
    lut_layer(topo順)。
    """
    succ: dict[str, list[str]] = {}
    for e in graph.edges:
        succ.setdefault(e.src, []).append(e.dst)
    gs_id = next((n.id for n in graph.nodes if n.type == 'group_sum'), None)
    if gs_id is None:
        raise ValueError('group_sumノードが無い(転写IRでない)')
    # group_sumへ入るlut_layer = 最終層
    final_id = None
    for e in graph.edges:
        if e.dst == gs_id:
            final_id = e.src
    if final_id is None:
        raise ValueError('group_sumへ接続するlut_layerが無い')
    lut_ids = [n.id for n in graph.topological_order() if n.type == 'lut_layer']
    hidden_ids = [nid for nid in lut_ids if nid != final_id]
    return hidden_ids, final_id, gs_id


def _downstream_lut(graph: IRGraph, node_id: str) -> str | None:
    """node_id の出力を受け取る唯一のlut_layer(次段)のidを返す(直列前提)。"""
    for e in graph.edges:
        if e.src == node_id:
            dst = next((n for n in graph.nodes if n.id == e.dst), None)
            if dst is not None and dst.type == 'lut_layer':
                return dst.id
    return None


def _widen_hidden_layer(
    node: IRNode, factor: int, jitter_frac: float, rng: np.random.Generator,
) -> None:
    """隠れlut_layerノードを in-place で factor倍に広げる(teacher先頭保存)。

    元out_width=W → 新out_width=f*W。先頭W個はteacherノードをそのまま保存、
    残りは jitter_frac 確率でクローン+片index再サンプル、それ以外は完全ランダム。
    """
    p = node.params
    in_width = int(p['in_width'])
    w0 = int(p['out_width'])
    idx = np.asarray(p['indices'], dtype=np.int64)  # (2, W)
    tts = tt_from_hex(p['truth_tables'])            # list[int] len W
    n_new = w0 * (factor - 1)

    new_idx = np.empty((2, n_new), dtype=np.int64)
    new_tts: list[int] = []
    for i in range(n_new):
        if rng.random() < jitter_frac:
            # teacherノードのクローン + 片スロットのindexを一様再サンプル
            src = int(rng.integers(0, w0))
            slot = int(rng.integers(0, 2))
            a = int(idx[0, src])
            b = int(idx[1, src])
            if slot == 0:
                a = int(rng.integers(0, in_width))
            else:
                b = int(rng.integers(0, in_width))
            new_idx[0, i] = a
            new_idx[1, i] = b
            new_tts.append(int(tts[src]))
        else:
            # 完全ランダム: tt一様16ゲート、indices一様
            new_idx[0, i] = int(rng.integers(0, in_width))
            new_idx[1, i] = int(rng.integers(0, in_width))
            new_tts.append(int(rng.integers(0, 16)))

    p['indices'] = np.concatenate([idx, new_idx], axis=1)
    p['out_width'] = w0 * factor
    p['truth_tables'] = tt_to_hex(list(tts) + new_tts)


def _remap_downstream_indices_noop() -> None:
    """teacher先頭保存の帰結: 下流indicesは不変(先頭スロットが元ノードを指す)。

    明示的なドキュメント用のno-op。実際の再マップは不要
    (widen前後で teacher由来ノードのスロット番号 0..W-1 が同一のため)。
    """
    return None


def _widen_final_layer(
    node: IRNode, gs_node: IRNode, factor: int, init_new_final: str,
    rng: np.random.Generator,
) -> None:
    """最終lut_layer(クラス群構造)を in-place で広げる。

    転写の最終層は n_classes 個のクラス群(各幅 g)を持つ縮退LUT列で、
    group_sum(k=n_classes) が群ごとに集計する。各群にランダムゲートを
    (factor-1)*g 個追加し、群幅を g*factor に拡張する。init_new_final='const0'
    なら新ゲートtt=0(group_sumスコアに寄与せず初期精度保存)。
    """
    p = node.params
    gp = gs_node.params
    in_width = int(p['in_width'])
    k = int(gp['k'])
    out0 = int(p['out_width'])
    if out0 % k != 0:
        raise ValueError(
            f'最終層out_width={out0}がk={k}で割り切れない(クラス群構造でない)'
        )
    g = out0 // k
    idx = np.asarray(p['indices'], dtype=np.int64)  # (2, k*g)
    tts = tt_from_hex(p['truth_tables'])
    g_new = g * factor

    new_idx = np.zeros((2, k * g_new), dtype=np.int64)
    new_tts: list[int] = []
    add_per_group = g_new - g
    for c in range(k):
        # 元の g ゲートを先頭に保存
        for j in range(g):
            src_col = c * g + j
            dst_col = c * g_new + j
            new_idx[0, dst_col] = idx[0, src_col]
            new_idx[1, dst_col] = idx[1, src_col]
            new_tts.append(int(tts[src_col]))
        # (factor-1)*g 個の新ゲートを追加
        for j in range(add_per_group):
            dst_col = c * g_new + g + j
            a = int(rng.integers(0, in_width))
            b = int(rng.integers(0, in_width))
            new_idx[0, dst_col] = a
            new_idx[1, dst_col] = b
            if init_new_final == 'const0':
                new_tts.append(0)
            else:  # random
                new_tts.append(int(rng.integers(0, 16)))

    p['indices'] = new_idx
    p['out_width'] = k * g_new
    p['truth_tables'] = tt_to_hex(new_tts)
    gp['in_width'] = k * g_new
    # group_sum の k は不変(クラス数)。群幅だけ拡大


def widen_native_ir(
    graph: IRGraph,
    factor: int,
    seed: int = 0,
    *,
    jitter_frac: float = 0.5,
    init_new_final: str = 'const0',
) -> IRGraph:
    """転写native DLGN IRを teacher初期化を保存したまま factor倍に広げる。

    隠れ層を factor倍(teacher由来ノードを先頭配置で保存、新ノードは
    jitter_frac確率でクローン+片index再サンプル、それ以外完全ランダム)、
    最終層は各クラス群に (factor-1)*g 個のランダムゲートを追加する。
    init_new_final='const0' なら新最終ゲートを tt=0 初期化して widen直後の
    hard精度を元IRと一致させる。

    Args:
        graph: transcribe_mlp/prune転写が生成した直列IR
            (入力'x' → lut_layer(n_inputs=2)列 → 最終lut_layer → group_sum → 'y')
        factor: 幅拡張倍率(f>=1の整数。f=1は恒等)
        seed: 乱数シード(決定的)
        jitter_frac: 隠れ新ノードをクローン+jitterする確率([0,1])
        init_new_final: 'const0'(新最終ゲートtt=0) | 'random'

    Returns:
        新しいIRGraph(元graphは無改造)。widen後もvalidate通過。

    Raises:
        ValueError: factor<1 / 不正なinit_new_final / 転写IRでない
    """
    if factor < 1:
        raise ValueError(f'factorは1以上: {factor}')
    if init_new_final not in INIT_NEW_FINAL_POLICIES:
        raise ValueError(
            f'init_new_finalは{INIT_NEW_FINAL_POLICIES}のみ: {init_new_final}'
        )
    if not (0.0 <= jitter_frac <= 1.0):
        raise ValueError(f'jitter_fracは[0,1]: {jitter_frac}')

    g = copy.deepcopy(graph)
    if factor == 1:
        return g

    rng = np.random.default_rng(seed)
    hidden_ids, final_id, gs_id = _hidden_and_final_ids(g)
    node_by_id = {n.id: n for n in g.nodes}

    # 隠れ層を順に広げる。teacher先頭保存なので下流indicesの再マップは不要。
    for nid in hidden_ids:
        _widen_hidden_layer(node_by_id[nid], factor, jitter_frac, rng)
        # 下流lut_layerの in_width を新out_widthへ更新
        down = _downstream_lut(g, nid)
        if down is not None:
            node_by_id[down].params['in_width'] = int(
                node_by_id[nid].params['out_width']
            )
    _remap_downstream_indices_noop()

    # 最終層(クラス群)を広げる
    _widen_final_layer(
        node_by_id[final_id], node_by_id[gs_id], factor, init_new_final, rng,
    )

    g.metadata = {
        **g.metadata,
        'widen': {
            'factor': factor, 'seed': seed, 'jitter_frac': jitter_frac,
            'init_new_final': init_new_final,
        },
    }
    return g
