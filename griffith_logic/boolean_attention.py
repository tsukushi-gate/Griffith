"""後方互換シム。実体は models/attention/boolean_attention.py に移動した。"""

from .models.attention.boolean_attention import (
    BooleanAttentionConfig,
    DLGNBooleanAttention,
)

__all__ = [
    'BooleanAttentionConfig',
    'DLGNBooleanAttention',
]
