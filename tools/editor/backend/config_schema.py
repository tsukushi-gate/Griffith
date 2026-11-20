"""学習設定(TrainConfig)JSON SchemaAPI。

CONFIGタブのスキーマ駆動フォーム(SchemaForm.tsx)がフィールド一覧を
機械的に取得するためのエンドポイント。TrainConfig自体は本体
(griffith_logic/training/config.py)のスキーマをそのまま返すだけで、
本体のバリデーションロジックには一切手を加えない。

main.py への組み込み(循環importを避けるためSTATE参照はconfigureで注入。
本routerは現状STATE未使用だが、probe.py/hdl.py/projects_router.py同様の
パターンを踏襲するため引数として受け取っておく):
    from . import config_schema
    app.include_router(config_schema.configure(STATE))
"""

from __future__ import annotations

from fastapi import APIRouter

from griffith_logic.training.config import TrainConfig

router = APIRouter()

_STATE = None  # main.pyのAppState参照(configureで注入。本routerは現状未使用)


def configure(state) -> APIRouter:
    """STATE参照を注入し、includeすべきrouterを返す。"""
    global _STATE
    _STATE = state
    return router


@router.get('/api/config/schema')
def get_config_schema() -> dict:
    """TrainConfigのJSON Schema(pydantic v2 model_json_schema)をそのまま返す。"""
    return TrainConfig.model_json_schema()
