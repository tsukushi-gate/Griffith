"""magicdelta-ir: モデル中間表現(M7)。

Python API ⇔ エディタ ⇔ HDLトランスパイラの共通契約。
"""

from .build import (
    BUILDERS,
    IRGraphModule,
    build_torch,
    is_serial_graph,
    register_builder,
)
from .probe import probe, probe_soft
from .schema import IR_FORMAT, IR_VERSION, IREdge, IRGraph, IRNode, IRPort
from .serialize import ir_from_dict, ir_to_dict, load_ir, save_ir
from .simulate import simulate, simulate_trace
from .validate import IRValidationError, validate
from .weights import (
    TT_EDITABLE,
    gate_id_from_tt_n2,
    restore_dwn_layer,
    restore_lut_conv2d,
    restore_lut_conv2d_fair_init,
    restore_lut_layer,
    weights_from_truth_tables,
    weights_from_truth_tables_fair_init,
)

__all__ = [
    'BUILDERS',
    'IR_FORMAT',
    'IR_VERSION',
    'IREdge',
    'IRGraph',
    'IRGraphModule',
    'IRNode',
    'IRPort',
    'IRValidationError',
    'TT_EDITABLE',
    'build_torch',
    'gate_id_from_tt_n2',
    'ir_from_dict',
    'ir_to_dict',
    'is_serial_graph',
    'load_ir',
    'probe',
    'probe_soft',
    'register_builder',
    'restore_dwn_layer',
    'restore_lut_conv2d',
    'restore_lut_conv2d_fair_init',
    'restore_lut_layer',
    'save_ir',
    'simulate',
    'simulate_trace',
    'validate',
    'weights_from_truth_tables',
    'weights_from_truth_tables_fair_init',
]
