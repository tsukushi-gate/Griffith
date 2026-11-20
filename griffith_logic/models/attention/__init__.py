"""Attention系モデルのサブパッケージ。"""

from .attention_seq2seq import AttentionSeq2SeqConfig, DLGNAttentionSeq2Seq
from .attention_translator import DLGNAttentionTranslator, TranslatorConfig
from .boolean_attention import BooleanAttentionConfig, DLGNBooleanAttention
from .multi_query_attention import DLGNMultiQueryAttention, MultiQueryAttentionConfig
from .pointer_attention import DLGNPointerAttention, PointerAttentionConfig
from .positional_attention import DLGNPositionalAttention, PositionalAttentionConfig
