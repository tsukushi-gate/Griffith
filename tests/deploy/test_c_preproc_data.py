"""前処理テストで共有する preproc 宣言(K4)。

配信中の実モデルが使う2種類(image_thermometer / image_lowlevel)を、
小さな画像サイズで再現したもの。カーネル方向・チャネル対は
griffith_logic/encoding/image_features.py の定数をそのまま使う
(ここで数値を書き写すと契約が二重管理になるため)。
"""

from griffith_logic.encoding.image_features import (
    CHROMA_PAIRS,
    CURV_OFFSETS,
    EDGE_OFFSETS,
)

_N_THR = 3

THERMO_PREPROC = {
    'kind': 'image_thermometer',
    'rgb_channels': 3,
    'n_thresholds': _N_THR,
    'height': 7,
    'width': 11,
    'mean': [0.0, 0.0, 0.0],
    'std': [1.0, 1.0, 1.0],
    'thresholds': [[(i + 1) / (_N_THR + 1) for i in range(_N_THR)]
                   for _ in range(3)],
    'bit_order': 'rgb,thr,y,x',
}

LOWLEVEL_PREPROC = {
    'kind': 'image_lowlevel',
    'rgb_channels': 3,
    'n_thresholds': _N_THR,
    'height': 6,
    'width': 9,
    'mean': [0.0, 0.0, 0.0],
    'std': [1.0, 1.0, 1.0],
    'thresholds': [[(i + 1) / (_N_THR + 1) for i in range(_N_THR)]
                   for _ in range(3)],
    'edge_thresholds': [0.1, 0.3],
    'curv_thresholds': [0.2],
    'chroma_thresholds': [0.15, 0.4],
    'edge_offsets': [list(o) for o in EDGE_OFFSETS],
    'curv_offsets': [list(o) for o in CURV_OFFSETS],
    'chroma_pairs': [list(p) for p in CHROMA_PAIRS],
    'bit_order': 'block,detector,thr,sign,c,y,x',
}
