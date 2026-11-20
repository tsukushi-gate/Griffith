"""後方互換シム。実体は models/attention/positional_attention.py に移動した。"""

from .models.attention.positional_attention import (
    DLGNPositionalAttention,
    PositionalAttentionConfig,
)

__all__ = [
    'DLGNPositionalAttention',
    'PositionalAttentionConfig',
]
