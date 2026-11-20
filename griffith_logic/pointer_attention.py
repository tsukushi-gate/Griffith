"""後方互換シム。実体は models/attention/pointer_attention.py に移動した。"""

from .models.attention.pointer_attention import (
    DLGNPointerAttention,
    PointerAttentionConfig,
)

__all__ = [
    'DLGNPointerAttention',
    'PointerAttentionConfig',
]
