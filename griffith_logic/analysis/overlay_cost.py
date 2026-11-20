"""overlay実装(メモリ格納ゲートの逐次実行エンジン)向けコスト指標(SM8)。

spatial実装(全ゲート並列)と違い、overlayではエンジン回路が固定で、
モデル依存コストは「メモリ量・サイクル数・メモリアクセスの局所性」に移る。
エンジン自体のP&R特性はSM6で一度だけ較正すればよく、モデルごとのEDA解析は
不要になる(構造メトリクスの設計方針)。

実行モデル(単純化した参照エンジン):
    - ゲートはトポロジカル順(ノード順×ノード内出力順)にスケジュール
    - 層バリア: 次のlut_layerは前ノード完了後に開始
    - 1サイクルに gates_per_cycle 個のゲートを評価
    - ゲート出力はスケジュール位置=アドレスの平坦メモリに書かれ、
      bank = アドレス % banks のインターリーブで読まれる
    - 一次入力(入力ポート・constant)は別メモリ(バンク競合対象外)

指標:
    - memory: 真理値表bit + indexビット幅×アリティ の総和(モデル格納量)
    - cycles: Σ_ノード ceil(out_width / gates_per_cycle)
    - reuse: 各ゲート入力の「生産位置との距離」分布(ワーキングセットproxy)
    - banking: 同一サイクル群内の同一バンク読出しの競合率・直列化係数
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class OverlayEngineSpec:
    """参照エンジンのパラメータ。実機エンジン(gen_engine系)に合わせて較正する。"""
    gates_per_cycle: int = 1
    banks: int = 1
    index_bits: int | None = None  # Noneなら層ごとに ceil(log2(in_width))


def _lut_nodes_in_topo(graph):
    return [n for n in graph.topological_order() if n.type == 'lut_layer']


def _in_srcs(graph) -> dict[str, list[str]]:
    grouped: dict[str, list] = {}
    for e in graph.edges:
        grouped.setdefault(e.dst, []).append(e)
    return {
        dst: [e.src for e in sorted(edges, key=lambda e: e.dst_slot)]
        for dst, edges in grouped.items()
    }


def overlay_cost(graph, spec: OverlayEngineSpec | None = None) -> dict:
    """IRのoverlay実行コストを見積る(JSON化可能なdict)。

    conv系を含むIRは事前に lower_to_primitives 済みであること
    (ゲート粒度=エンジンが実行する粒度で測るため)。
    """
    from ..ir.validate import node_out_width

    spec = spec or OverlayEngineSpec()
    gpc = max(int(spec.gates_per_cycle), 1)
    banks = max(int(spec.banks), 1)

    in_srcs = _in_srcs(graph)

    # --- 出力ビット別の「生産位置」(スケジュール位置=アドレス。一次入力=-1)
    pos: dict[str, np.ndarray] = {
        p.id: np.full(p.width, -1, dtype=np.int64) for p in graph.inputs
    }
    cursor = 0
    mem_tt_bits = 0
    mem_index_bits = 0
    mem_per_node: dict[str, int] = {}
    cycles_per_node: dict[str, int] = {}
    reuse: list[int] = []
    primary_reads = 0
    conflict_extra = 0
    serialization: list[int] = []
    banked_reads = 0

    for node in graph.topological_order():
        srcs = in_srcs.get(node.id, [])
        p = node.params
        if node.type == 'lut_layer':
            src_pos = pos[srcs[0]]
            idx = np.asarray(p['indices'], dtype=np.int64)
            n = int(p['n_inputs'])
            out_width = int(p['out_width'])
            in_width = int(p['in_width'])

            ibits = spec.index_bits
            if ibits is None:
                ibits = max(math.ceil(math.log2(max(in_width, 2))), 1)
            node_bits = out_width * ((1 << n) + n * ibits)
            mem_per_node[node.id] = node_bits
            mem_tt_bits += out_width * (1 << n)
            mem_index_bits += out_width * n * ibits

            cycles_per_node[node.id] = math.ceil(out_width / gpc)

            gate_pos = cursor + np.arange(out_width, dtype=np.int64)
            prod = src_pos[idx]                      # (n, out_width)
            # 再利用距離(一次入力は別カウント)
            mask = prod >= 0
            reuse.extend((gate_pos[None, :] - prod)[mask].tolist())
            primary_reads += int((~mask).sum())

            # バンク競合: gpc個ずつのサイクル群で同一バンク読出しを数える
            for start in range(0, out_width, gpc):
                group_prod = prod[:, start:start + gpc]
                addrs = group_prod[group_prod >= 0]
                if addrs.size == 0:
                    continue
                counts = np.bincount(addrs % banks, minlength=banks)
                conflict_extra += int(np.maximum(counts - 1, 0).sum())
                serialization.append(int(counts.max()))
                banked_reads += int(addrs.size)

            pos[node.id] = gate_pos
            cursor += out_width
        elif node.type == 'constant':
            pos[node.id] = np.full(node_out_width(node), -1, dtype=np.int64)
        elif node.type == 'concat':
            pos[node.id] = np.concatenate([pos[s] for s in srcs])
        elif node.type in ('binarize', 'flatten'):
            pos[node.id] = pos[srcs[0]].copy()
        elif node.type == 'add':
            pos[node.id] = np.stack([pos[s] for s in srcs]).max(axis=0)
        elif node.type == 'group_sum':
            gs = int(p['in_width']) // int(p['k'])
            pos[node.id] = pos[srcs[0]].reshape(int(p['k']), gs).max(axis=1)
        else:  # blackbox等: 位置情報は伝播しない(下流lutは全て一次扱い)
            pos[node.id] = np.full(node_out_width(node), -1, dtype=np.int64)

    reuse_arr = np.asarray(reuse, dtype=np.int64)
    return {
        'engine': {'gates_per_cycle': gpc, 'banks': banks,
                   'index_bits': spec.index_bits},
        'memory': {
            'total_bits': mem_tt_bits + mem_index_bits,
            'tt_bits': mem_tt_bits,
            'index_bits': mem_index_bits,
            'per_node': mem_per_node,
        },
        'cycles': {
            'total': int(sum(cycles_per_node.values())),
            'per_node': cycles_per_node,
        },
        'reuse': {
            'reads': int(reuse_arr.size),
            'primary_reads': int(primary_reads),
            'mean': float(reuse_arr.mean()) if reuse_arr.size else 0.0,
            'p50': float(np.percentile(reuse_arr, 50)) if reuse_arr.size else 0.0,
            'p90': float(np.percentile(reuse_arr, 90)) if reuse_arr.size else 0.0,
            'max': int(reuse_arr.max()) if reuse_arr.size else 0,
        },
        'banking': {
            'conflict_rate': conflict_extra / banked_reads if banked_reads else 0.0,
            'mean_serialization': float(np.mean(serialization))
            if serialization else 0.0,
        },
    }
