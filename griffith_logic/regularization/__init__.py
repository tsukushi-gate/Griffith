"""正則化サブパッケージ(DWNのSpectral Regularization等)。"""

from .apply import (
    iter_spectral_layers,
    soft_truth_table,
    spectral_norm_gram,
    spectral_penalty,
)
from .spectral import build_spectral_coefficient_matrix, spectral_norm

__all__ = [
    'build_spectral_coefficient_matrix',
    'iter_spectral_layers',
    'soft_truth_table',
    'spectral_norm',
    'spectral_norm_gram',
    'spectral_penalty',
]
