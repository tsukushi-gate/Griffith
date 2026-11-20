"""ビューアレイアウトの永続化(<model>.viewer.json サイドカー)。

IR 本体は汚さない(マニフェスト思想と同じ)。壊れたファイルは既定値で
継続し、保存はアトミック(tmp → os.replace)。

注意: viewer 配下では `from __future__ import annotations` を使わない。
"""

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from .probes import ProbeSpec


class ViewerConfig(BaseModel):
    version: int = 1
    probes: list[ProbeSpec] = Field(default_factory=list)
    positions: dict[str, dict] = Field(default_factory=dict)  # id -> {x,y}
    expanded: list[str] = Field(default_factory=list)
    viewport: dict | None = None       # {x, y, zoom}
    board: str | None = None           # 選択ボード key(モデルごとに記憶)


def sidecar_path(model: str, mode: str = 'core') -> Path:
    """model パス → レイアウトサイドカーのパス。

    'x.ir.json' → 'x.viewer.json'(コア)/ 'x.system.viewer.json'(システム)。
    モード別に分けることでプローブの相互孤児化を防ぐ。
    """
    tag = '.system.viewer.json' if mode == 'system' else '.viewer.json'
    p = Path(model)
    name = p.name
    for suffix in ('.ir.json', '.json'):
        if name.endswith(suffix):
            return p.with_name(name[:-len(suffix)] + tag)
    return p.with_name(name + tag)


def load_config(path: Path) -> ViewerConfig:
    if not path.exists():
        return ViewerConfig()
    try:
        with open(path, encoding='utf-8') as f:
            return ViewerConfig(**json.load(f))
    except Exception as e:  # noqa: BLE001 - 壊れていても view を起動させる
        print(f'[mdl-fpga view] レイアウト読込失敗({path.name}): {e} → 既定値で継続')
        return ViewerConfig()


def save_config(path: Path, cfg: ViewerConfig) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(
        cfg.model_dump_json(indent=1, by_alias=True), encoding='utf-8')
    os.replace(tmp, path)
