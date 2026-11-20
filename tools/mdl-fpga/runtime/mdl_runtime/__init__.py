"""mdl_runtime — 配布用スタンドアローンランタイム(STANDALONE.md)。

griffith-logic / mdl-fpga に依存しない(import-lint テストで機械保証)。
bundle zip の app/ に vendor 同梱され、io_contract.json だけを頼りに
書込(openFPGALoader)・UART 通信・セルフテスト・GUI/CUI を提供する。

コア依存は Python 標準ライブラリ+pyserial のみ(pygame はテンプレのみ)。
"""

__version__ = '0.1.0'
