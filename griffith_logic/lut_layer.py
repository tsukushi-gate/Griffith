"""後方互換シム。実体は core/ と layers/ に移動した。

    基底関数        → core/basis.py
    Function群      → core/functions.py
    GradFactor      → core/ste.py
    LutLayer        → layers/lut.py
"""

from .core.basis import (
    _build_gate_table_n2,  # noqa: F401 — 後方互換の再エクスポート(test_backcompat)
    _compute_basis_butterfly,  # noqa: F401 — 同上
    _compute_wh_basis_butterfly,  # noqa: F401 — 同上
)
from .core.functions import (
    CudaLutFunction,
    CudaLutFunctionIndependent,
    CudaLutFunctionIWP,
    CudaLutFunctionWARP,
    LutFunction,
    LutFunctionHardST,
    LutFunctionIndependent,
    LutFunctionIWP,
    LutFunctionRaw,
    LutFunctionWARP,
    LutFunctionWARPHardST,
)
from .core.ste import GradFactor
from .layers.lut import LutLayer

__all__ = [
    'CudaLutFunction',
    'CudaLutFunctionIndependent',
    'CudaLutFunctionIWP',
    'CudaLutFunctionWARP',
    'GradFactor',
    'LutFunction',
    'LutFunctionHardST',
    'LutFunctionIWP',
    'LutFunctionIndependent',
    'LutFunctionRaw',
    'LutFunctionWARPHardST',
    'LutFunctionWARP',
    'LutLayer',
]
