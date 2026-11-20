"""後方互換シム。実体は models/attention/multi_query_attention.py に移動した。"""

from .models.attention.multi_query_attention import (
    DLGNMultiQueryAttention,
    MultiQueryAttentionConfig,
)

__all__ = [
    'DLGNMultiQueryAttention',
    'MultiQueryAttentionConfig',
]
