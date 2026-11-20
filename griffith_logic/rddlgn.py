"""後方互換シム。実体は models/rddlgn.py に移動した。"""

from .models.rddlgn import (
    RDDLGN,
    LogicEmbedding,
    LutStack,
    RDDLGNConfig,
    RDDLGNDecoder,
    RDDLGNEncoder,
    _init_hidden,  # noqa: F401 — 後方互換の再エクスポート(test_backcompat)
)

__all__ = [
    'LogicEmbedding',
    'LutStack',
    'RDDLGN',
    'RDDLGNConfig',
    'RDDLGNDecoder',
    'RDDLGNEncoder',
]
