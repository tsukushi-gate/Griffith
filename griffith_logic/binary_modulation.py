"""後方互換シム。実体は encoding/modulation.py に移動した。"""

from .encoding.modulation import BinaryToReal, RealToBinary

__all__ = ['BinaryToReal', 'RealToBinary']
