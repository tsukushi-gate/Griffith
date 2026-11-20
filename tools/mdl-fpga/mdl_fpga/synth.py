"""互換シム: 実体は backends/quartus.py へ移設(M7 マルチベンダ化)。

既存の import(tests / examples / bundle 等)を無改修で維持するための
再エクスポート。新規コードは mdl_fpga.backends を使うこと。
"""

from __future__ import annotations

from .backends.base import FlowResult, write_summary as _write_summary_impl  # noqa: F401
from .backends.quartus import (  # noqa: F401
    STAGES, QuartusBackend, _cpf_args, _KNOWN_ERRORS, _STA_CORNER_DEFAULT,
    _STA_CORNERS, _STA_TCL, convert_sof, diagnose_failure, parse_fit_entities,
    parse_fit_summary, parse_fmax, parse_fmax_all, run_flow,
    summarize_entity_m10k, write_project_files)


def _write_summary(projdir, res):  # 旧名互換(私的関数だがテストが参照し得る)
    return _write_summary_impl(projdir, res)
