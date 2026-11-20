"""合成バックエンドのレジストリ(board JSON の toolchain 欄でディスパッチ)。

新ベンダの追加手順: backends/<name>.py に SynthBackend 実装を書き、
_REGISTRY に1行登録するだけ(build.py は無改修で載る)。
将来登録欄(未実装): 'vivado' / 'openxc7'(Xilinx)、
'lattice-oss'(nextpnr-ecp5/ice40)、'radiant'(Lattice 公式 EDA)。

toolchain 欄が無い board JSON(外部・旧版)は family 名から推定する。
"""

from __future__ import annotations

import re

from .base import FlowResult, SynthBackend, write_summary  # noqa: F401
from .gowin_oss import GowinOssBackend
from .quartus import QuartusBackend

_REGISTRY: dict[str, type] = {
    'quartus': QuartusBackend,
    'gowin-oss': GowinOssBackend,
}

# family 名 → バックエンドの推定(toolchain 欄が無いボード用フォールバック)
_FAMILY_FALLBACK: tuple[tuple[str, str], ...] = (
    (r'cyclone|max ?10|arria|stratix', 'quartus'),
    (r'^gw', 'gowin-oss'),
)


def backend_names() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(board: dict) -> SynthBackend:
    """board dict から合成バックエンドを解決する。

    優先順: board['toolchain'](明示)→ family 正規表現推定 → 明示エラー。
    """
    key = board.get('toolchain')
    if key:
        cls = _REGISTRY.get(key)
        if cls is None:
            raise KeyError(
                f'未知の toolchain: {key!r}(ボード {board.get("key")})。'
                f'利用可能: {", ".join(backend_names())}。'
                '新ベンダは backends/ に SynthBackend 実装を追加して登録する')
        return cls()
    family = board.get('family') or ''
    for pat, name in _FAMILY_FALLBACK:
        if re.search(pat, family, re.I):
            return _REGISTRY[name]()
    raise KeyError(
        f'ボード {board.get("key")}(family={family!r})の合成バックエンドを'
        '決定できない。boards/*.json に toolchain 欄を明示する'
        f'(利用可能: {", ".join(backend_names())})')
