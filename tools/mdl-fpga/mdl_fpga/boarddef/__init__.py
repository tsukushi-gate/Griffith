"""board-def: ボード定義のオンデマンド機械生成(AUTOFLOW 工程3)。

- providers: 外部ソース(litex-boards / .cst / .qsf / part.hpp)のパーサ
- fetch:     レシピ駆動の 2 ソース突合 → schema_version=2 draft 生成
- recipes/:  ボード別レシピ(信号→役割の対応付けのみ。ピン値は持たない)
"""

from .fetch import CrossCheckError, FetchResult, run_fetch  # noqa: F401
