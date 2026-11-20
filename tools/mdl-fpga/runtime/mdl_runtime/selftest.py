"""セルフテスト(同梱 golden の実機再生= hw-check の配布版)。

golden は bundle 時に参照シミュレータから生成されたもの。全ベクトル
bit 一致なら、実機は参照実装と同じ関数を計算している。
"""

from __future__ import annotations

from .link import UartLink


def run_selftest(link: UartLink, goldens: list[dict]) -> dict:
    n_ok = 0
    first_fail = None
    for i, v in enumerate(goldens):
        try:
            got = link.query(i, v['x'])
        except TimeoutError as e:
            if first_fail is None:
                first_fail = f'vec {i}: {e}'
            continue
        if got == v['y']:
            n_ok += 1
        elif first_fail is None:
            first_fail = f'vec {i}: got=0x{got:x} exp=0x{v["y"]:x}'
    return {'passed': n_ok == len(goldens), 'n_ok': n_ok,
            'n': len(goldens), 'first_fail': first_fail}
