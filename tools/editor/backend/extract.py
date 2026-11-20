"""LutLayerからゲートID・確信度を抽出する(表示用資産+WSフレーム構築)。

統計ロジック本体は griffith_logic.analysis.structure へ昇格した(SM1)。
本モジュールは後方互換のための再exportと、エディタ固有の表示資産
(16ゲート名/色テーブル=フロントエンド src/ir/gates.ts と同一定義)、
WS配信フレーム(layer_summary)のみを持つ。
"""

from griffith_logic.analysis.structure import (
    gate_entropy,
    gate_probs_from_weights,
    get_confidence,
    get_gate_ids,
    layer_entropy,
    lut_layers_with_ids,
    neuron_uncertainty,
    summary_from_weights,
    tt_to_gate_id_n2,
)

__all__ = [
    'GATE_NAMES',
    'GATE_COLORS',
    'DEFAULT_COLOR',
    'tt_to_gate_id_n2',
    'get_gate_ids',
    'get_confidence',
    'gate_probs_from_weights',
    'gate_entropy',
    'neuron_uncertainty',
    'layer_entropy',
    'summary_from_weights',
    'lut_layers_with_ids',
    'layer_summary',
]

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

DEFAULT_COLOR = (100, 100, 200, 255)


def layer_summary(node_id: str, layer, step: int) -> dict:
    """WS配信用のlayer_summaryフレームを構築する。

    選択ギャップ・ダッシュボード(§4-1)向けに、レイヤ別の迷い統計
    (ニューロン毎ゲート分布エントロピーの平均・最大)も載せる。
    """
    gate_ids = get_gate_ids(layer)
    confidence = get_confidence(layer)
    ent = layer_entropy(layer)
    if ent is not None and ent.numel() > 0:
        entropy_mean = round(float(ent.mean().item()), 4)
        entropy_max = round(float(ent.max().item()), 4)
    else:
        entropy_mean = None
        entropy_max = None
    return {
        'type': 'layer_summary',
        'layer': node_id,
        'step': step,
        'n_inputs': int(layer.n_inputs),
        'gate_ids': [int(g) for g in gate_ids],
        'confidence': [round(float(c), 4) for c in confidence],
        'entropy_mean': entropy_mean,
        'entropy_max': entropy_max,
    }
