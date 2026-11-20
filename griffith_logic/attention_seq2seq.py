"""後方互換シム。実体は models/attention/attention_seq2seq.py に移動した。"""

from .models.attention.attention_seq2seq import (
    AttentionSeq2SeqConfig,
    DLGNAttentionSeq2Seq,
)

__all__ = [
    'AttentionSeq2SeqConfig',
    'DLGNAttentionSeq2Seq',
]
