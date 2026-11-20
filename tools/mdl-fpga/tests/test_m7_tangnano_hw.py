"""M7: Tang Nano 20K 実機統合(requires_hw、MDL_FPGA_HW_TESTS=1 で実行)。

JTAG は排他資源のため CI では走らせない(明示 opt-in)。他ボード接続時は
skip(検出結果が対象ボードでない場合。DE0-CV 用 requires_hw テストと併存可)。
初回受入の実測記録は進捗記録 / boards/NOTICE が正。
"""

import pytest
from conftest import requires_hw

from mdl_fpga.detect import run_detect


@requires_hw
def test_detect_tangnano20k_full_confidence():
    res = run_detect()
    best = res.best()
    if best is None or best.key != 'tang-nano-20k':
        pytest.skip(f'Tang Nano 20K 非接続(best={best.key if best else None})')
    # 実機受入と同水準: IDCODE+部品名+USB+ボード名の4署名
    assert best.confidence == 1.0
    assert any(d.idcode == '0x0000081b' for d in res.observation.jtag)
    assert best.tier == 1
