"""tools/nas: NAS(構造自動探索)用ツール群(N8系列、本体ライブラリ外)。

ROADMAP-v5-editor-nnc.md の N8方針:
    - griffith_logic/ 本体は import して呼び出すだけ(改変しない)
    - IRスキーマは変更しない(既存ノード型のparamsを書き換えるのみ)
"""

from .mutate import (
    MUTATION_OPS,
    mutate,
    mutate_conv_channels,
    mutate_insert_layer,
    mutate_lut_width,
    mutate_n_inputs,
    mutate_remove_layer,
)
from .nsga2 import crowding_distance, non_dominated_sort, select_parent

__all__ = [
    'MUTATION_OPS',
    'mutate',
    'mutate_conv_channels',
    'mutate_insert_layer',
    'mutate_lut_width',
    'mutate_n_inputs',
    'mutate_remove_layer',
    'crowding_distance',
    'non_dominated_sort',
    'select_parent',
]
