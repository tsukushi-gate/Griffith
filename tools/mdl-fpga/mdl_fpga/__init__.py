"""mdl-fpga: griffith-logic IR の FPGA 実装支援ツール。

構想書: mdl-fpga 構想書(契約面 = magicdelta-ir v0.2.0)。
ライブラリからは ir.serialize / ir.validate / ir.simulate / export.verify /
export.estimate を流用し、HDL 生成・フロー駆動・可視化は本ツールが所有する。
"""

__version__ = '0.1.0.dev0'
