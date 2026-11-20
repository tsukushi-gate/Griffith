"""後方互換シム。実体は core/connections.py に移動した。"""

from .core.connections import (
    _unique_connections_general,  # noqa: F401 — 後方互換の再エクスポート(test_backcompat)
    _unique_connections_n2,  # noqa: F401 — 同上
    build_reverse_index,
    get_random_connections,
    get_unique_connections,
)

__all__ = [
    'build_reverse_index',
    'get_random_connections',
    'get_unique_connections',
]
