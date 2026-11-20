"""後方互換シム。実体は models/attention/attention_translator.py に移動した。"""

from .models.attention.attention_translator import (
    DLGNAttentionTranslator,
    TranslatorConfig,
)

__all__ = [
    'DLGNAttentionTranslator',
    'TranslatorConfig',
]
