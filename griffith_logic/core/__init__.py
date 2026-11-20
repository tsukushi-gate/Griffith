"""数式そのもの(nn.Module無し)を置くコアサブパッケージ。

M3のパッケージ再編で basis / functions / connections / ste が
ここへ移動する。現時点では参照実装のみ。
"""

from .estimators import ESTIMATORS, estimator_rho, estimator_rho_prime
from .reference import (
    reference_basis,
    reference_gather,
    reference_grads,
    reference_ltt_forward,
    reference_ltt_grads,
    reference_ltt_pre_activation,
    reference_ltt_ste_surrogate,
    reference_lut_forward,
    reference_lut_warp_hard_forward,
    reference_lut_warp_hard_st_surrogate,
    reference_soft_merge,
    reference_warp_entry_logits,
    reference_wh_basis,
    reference_xnor_attention,
)

__all__ = [
    'ESTIMATORS',
    'estimator_rho',
    'estimator_rho_prime',
    'reference_basis',
    'reference_gather',
    'reference_grads',
    'reference_ltt_forward',
    'reference_ltt_grads',
    'reference_ltt_pre_activation',
    'reference_ltt_ste_surrogate',
    'reference_lut_forward',
    'reference_lut_warp_hard_forward',
    'reference_lut_warp_hard_st_surrogate',
    'reference_warp_entry_logits',
    'reference_soft_merge',
    'reference_wh_basis',
    'reference_xnor_attention',
]
