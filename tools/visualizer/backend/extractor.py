"""DDLGNモデルからネットワーク構造を抽出し、可視化用データに変換する。"""

import math
import struct
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# N=2用の16ゲート名マッピング
# ゲートID: 真理値表を(0,0),(0,1),(1,0),(1,1)順で読んだ2進数
GATE_NAMES = {
    0: 'FALSE', 1: 'AND', 2: 'A>B', 3: 'A',
    4: 'B>A', 5: 'B', 6: 'XOR', 7: 'OR',
    8: 'NOR', 9: 'XNOR', 10: '!B', 11: 'A>=B',
    12: '!A', 13: 'B>=A', 14: 'NAND', 15: 'TRUE',
}

# ゲート種別ごとの色 (R, G, B, A) — 0-255
GATE_COLORS = {
    0: (140, 140, 140, 255),   # FALSE — グレー
    15: (180, 180, 180, 255),  # TRUE — グレー
    3: (100, 149, 237, 255),   # A — 青系
    5: (65, 105, 225, 255),    # B — 青系
    12: (135, 206, 250, 255),  # !A — 青系薄
    10: (176, 196, 222, 255),  # !B — 青系薄
    1: (220, 60, 60, 255),     # AND — 赤
    14: (255, 127, 80, 255),   # NAND — オレンジ
    2: (255, 99, 71, 255),     # A>B — 赤系
    4: (233, 116, 81, 255),    # B>A — 赤系
    11: (205, 92, 92, 255),    # A>=B — 赤系
    13: (240, 128, 128, 255),  # B>=A — 赤系
    6: (50, 205, 50, 255),     # XOR — 緑
    9: (144, 238, 144, 255),   # XNOR — 緑薄
    7: (148, 103, 189, 255),   # OR — 紫
    8: (186, 152, 217, 255),   # NOR — 紫薄
}

# N>2のゲートはインデックスに応じたカラーマップを使用
DEFAULT_COLOR = (100, 100, 200, 255)


def _get_gate_ids(layer) -> np.ndarray:
    """LutLayerのゲートIDを取得する。

    Returns:
        (out_dim,) int配列。N=2では0-15、N>2ではargmaxインデックス。
    """
    with torch.no_grad():
        weights = layer.weights.detach().cpu()

        if layer.parameterization == 'iwp':
            # IWP: sigmoid → 0/1 で等価ゲートの真理値表を計算
            p = (torch.sigmoid(weights) > 0.5).long()
            # 2進数としてゲートIDを計算
            gate_ids = torch.zeros(layer.out_dim, dtype=torch.long)
            for j in range(layer.n_inputs):
                gate_ids += p[:, j] << j
            return gate_ids.numpy()

        if hasattr(layer, '_use_16gate') and layer._use_16gate:
            # N=2 softmax: 16ゲートモード
            return weights.argmax(dim=-1).numpy()

        # softmax / independent: エントリのargmax
        return weights.argmax(dim=-1).numpy()


def _get_confidence(layer) -> np.ndarray:
    """各ニューロンの確信度（0.0〜1.0）を取得する。"""
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

    return np.ones(layer.out_dim, dtype=np.float32) * 0.5


def _get_gate_color(gate_id: int, n_inputs: int) -> tuple[int, int, int, int]:
    """ゲートIDから色を返す。"""
    if n_inputs == 2:
        return GATE_COLORS.get(gate_id, DEFAULT_COLOR)
    # N>2: ハッシュベースの色分け
    h = (gate_id * 137 + 43) % 360
    # HSV → RGB (簡易版: S=0.6, V=0.85)
    s, v = 0.6, 0.85
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255), 255)


def extract_layers(model) -> list[dict[str, Any]]:
    """モデルから全LutLayerの情報を抽出する。

    Returns:
        レイヤー情報のリスト。各要素:
        {
            'name': str,
            'in_dim': int,
            'out_dim': int,
            'n_inputs': int,
            'gate_ids': np.ndarray (out_dim,),
            'confidence': np.ndarray (out_dim,),
            'indices': np.ndarray (n_inputs, out_dim),
        }
    """
    from griffith_logic import LutLayer

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, LutLayer):
            layers.append({
                'name': name,
                'in_dim': module.in_dim,
                'out_dim': module.out_dim,
                'n_inputs': module.n_inputs,
                'gate_ids': _get_gate_ids(module),
                'confidence': _get_confidence(module),
                'indices': module.indices.detach().cpu().numpy(),
            })
    return layers


def compute_layout(layers: list[dict]) -> dict[str, Any]:
    """レイヤー情報からノード座標とエッジ座標を計算する。

    論文のニューラルネットワーク構造図のように、各レイヤーのノードを
    縦一列に配置し、レイヤーは左から右に並べる。Y方向は中央揃え。

    Returns:
        {
            'nodes': np.ndarray (total_nodes, 8),
                # [x, y, gate_id, confidence, r, g, b, a]
            'edges': np.ndarray (total_edges, 4),
                # [x1, y1, x2, y2]
            'layer_boundaries': list[dict],
        }
    """
    if not layers:
        return {
            'nodes': np.empty((0, 8), dtype=np.float32),
            'edges': np.empty((0, 4), dtype=np.float32),
            'layer_boundaries': [],
        }

    NODE_SPACING_Y = 3.0      # ノード間の縦間隔
    LAYER_SPACING_X = 100.0   # レイヤー間の横間隔

    # 全レイヤー（入力層含む）のノード数を収集
    first_layer = layers[0]
    input_dim = first_layer['in_dim']
    all_dims = [input_dim] + [l['out_dim'] for l in layers]

    # 最大レイヤーの高さ（中央揃え用）
    max_dim = max(all_dims)
    max_height = max_dim * NODE_SPACING_Y

    def _place_column(n: int, layer_x: float) -> np.ndarray:
        """レイヤーのノードを縦一列に配置する（Y中央揃え）。"""
        positions = np.zeros((n, 2), dtype=np.float32)
        height = n * NODE_SPACING_Y
        y_offset = (max_height - height) / 2.0
        positions[:, 0] = layer_x
        positions[:, 1] = y_offset + np.arange(n) * NODE_SPACING_Y
        return positions

    all_nodes = []
    all_edges = []
    layer_boundaries = []

    # --- 入力層 ---
    input_x = 0.0
    input_positions = _place_column(input_dim, input_x)

    input_nodes = np.zeros((input_dim, 8), dtype=np.float32)
    input_nodes[:, 0] = input_positions[:, 0]
    input_nodes[:, 1] = input_positions[:, 1]
    input_nodes[:, 2] = -1  # gate_id (入力ノード)
    input_nodes[:, 3] = 1.0
    input_nodes[:, 4] = 200  # r
    input_nodes[:, 5] = 200  # g
    input_nodes[:, 6] = 200  # b
    input_nodes[:, 7] = 255  # a
    all_nodes.append(input_nodes)

    layer_boundaries.append({
        'name': 'input',
        'node_start': 0,
        'node_count': input_dim,
        'x_min': float(input_x),
        'x_max': float(input_x),
        'y_min': float(input_positions[:, 1].min()),
        'y_max': float(input_positions[:, 1].max()),
    })

    node_offset = input_dim
    prev_positions = input_positions

    # --- 各LutLayer ---
    for layer_idx, layer in enumerate(layers):
        out_dim = layer['out_dim']
        n_inputs = layer['n_inputs']
        gate_ids = layer['gate_ids']
        confidence = layer['confidence']
        indices = layer['indices']  # (n_inputs, out_dim)

        layer_x = (layer_idx + 1) * LAYER_SPACING_X
        positions = _place_column(out_dim, layer_x)

        # ノードデータ: [x, y, gate_id, confidence, r, g, b, a]
        nodes = np.zeros((out_dim, 8), dtype=np.float32)
        nodes[:, 0] = positions[:, 0]
        nodes[:, 1] = positions[:, 1]
        nodes[:, 2] = gate_ids.astype(np.float32)
        nodes[:, 3] = confidence.astype(np.float32)

        # ゲートごとの色をベクトル化で設定
        for gid in np.unique(gate_ids):
            mask = gate_ids == gid
            r, g, b, a = _get_gate_color(int(gid), n_inputs)
            nodes[mask, 4] = r
            nodes[mask, 5] = g
            nodes[mask, 6] = b
            nodes[mask, 7] = a

        all_nodes.append(nodes)

        # エッジデータ: [x1, y1, x2, y2]（ベクトル化）
        edges = np.zeros((n_inputs * out_dim, 4), dtype=np.float32)
        for j in range(n_inputs):
            src_indices = indices[j]
            start = j * out_dim
            end = (j + 1) * out_dim
            edges[start:end, 0] = prev_positions[src_indices, 0]
            edges[start:end, 1] = prev_positions[src_indices, 1]
            edges[start:end, 2] = positions[:, 0]
            edges[start:end, 3] = positions[:, 1]

        all_edges.append(edges)

        layer_boundaries.append({
            'name': layer['name'],
            'node_start': node_offset,
            'node_count': out_dim,
            'n_inputs': n_inputs,
            'x_min': float(layer_x),
            'x_max': float(layer_x),
            'y_min': float(positions[:, 1].min()),
            'y_max': float(positions[:, 1].max()),
        })

        node_offset += out_dim
        prev_positions = positions

    nodes_array = np.concatenate(all_nodes, axis=0).astype(np.float32)
    edges_array = np.concatenate(all_edges, axis=0).astype(np.float32) if all_edges else np.empty((0, 4), dtype=np.float32)

    return {
        'nodes': nodes_array,
        'edges': edges_array,
        'layer_boundaries': layer_boundaries,
    }


def extract_network(model) -> dict[str, Any]:
    """モデルからネットワーク全体の可視化データを抽出する。

    Args:
        model: PyTorchモデル（LutLayerを含む）

    Returns:
        {
            'nodes': np.ndarray (N, 8) — [x, y, gate_id, confidence, r, g, b, a]
            'edges': np.ndarray (E, 4) — [x1, y1, x2, y2]
            'layer_boundaries': list[dict]
            'gate_names': dict
            'total_nodes': int
            'total_edges': int
        }
    """
    layers = extract_layers(model)
    layout = compute_layout(layers)

    return {
        **layout,
        'gate_names': GATE_NAMES,
        'total_nodes': int(layout['nodes'].shape[0]),
        'total_edges': int(layout['edges'].shape[0]),
    }


def encode_binary(network: dict) -> bytes:
    """ネットワークデータをバイナリ形式にエンコードする。

    フォーマット:
        ヘッダ (12 bytes):
            total_nodes: uint32
            total_edges: uint32
            layer_count: uint32
        ノードデータ: float32[] × (total_nodes × 8)
        エッジデータ: float32[] × (total_edges × 4)
    """
    nodes = network['nodes']
    edges = network['edges']
    n_layers = len(network['layer_boundaries'])

    header = struct.pack('<III', nodes.shape[0], edges.shape[0], n_layers)
    node_bytes = nodes.astype(np.float32).tobytes()
    edge_bytes = edges.astype(np.float32).tobytes()

    return header + node_bytes + edge_bytes
