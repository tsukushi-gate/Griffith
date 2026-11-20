"""エディタのプロジェクト形式。

複数ネットワーク・複数学習設定・データセット/run参照をまとめる上位コンテナ。
既存の単一IR保存経路(STATE.ir、/api/models/current/save等)とは独立に共存する。

永続化は training.py の PROJECT_ROOT 規約(outputs/editor_*)に倣い、
outputs/editor_projects/<project_id>.json に1プロジェクト1ファイルで保存する
(runs.pyの流儀: ensure_ascii=False、壊れたファイルはスキップ/None)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from griffith_logic.ir import IRValidationError, ir_from_dict, validate
from griffith_logic.training.config import TrainConfig

# プロジェクト一覧のサマリに含めるキー
SUMMARY_KEYS = ('id', 'name', 'description', 'created_at', 'updated_at')


class ProjectSchema(BaseModel):
    """プロジェクトのルートスキーマ。

    networks/train_configsの値自体はdictのまま保持し、
    妥当性検証(IR/TrainConfigのパース)は保存/読込時に本モジュールの
    validate_network/validate_train_configで別途行う。
    """

    model_config = ConfigDict(extra='forbid')

    id: str
    name: str
    description: str = ''
    networks: dict[str, dict] = Field(default_factory=dict)
    train_configs: dict[str, dict] = Field(default_factory=dict)
    dataset_refs: list[str] = Field(default_factory=list)
    run_refs: list[str] = Field(default_factory=list)
    optimizers: dict = Field(default_factory=dict)  # 後続タスク用の空枠
    monitors: dict = Field(default_factory=dict)  # 後続タスク用の空枠
    executors: dict = Field(default_factory=dict)  # 後続タスク用の空枠
    created_at: float
    updated_at: float
    deleted: bool = False  # 論理削除フラグ


def validate_network(name: str, ir_doc: dict) -> None:
    """ネットワーク1件分のIR dictを検証する(不正ならValueError)。"""
    try:
        graph = ir_from_dict(ir_doc)
        validate(graph)
    except (ValueError, KeyError, TypeError, AssertionError, IRValidationError) as e:
        raise ValueError(f"ネットワーク '{name}' のIRが不正です: {e}") from e


def validate_train_config(name: str, cfg_doc: dict) -> None:
    """学習設定1件分のdictを検証する(不正ならValueError)。"""
    try:
        TrainConfig(**cfg_doc)
    except Exception as e:  # pydantic.ValidationError他をまとめて日本語化
        raise ValueError(f"学習設定 '{name}' が不正です: {e}") from e


def save_project(project: ProjectSchema, root: Path) -> Path:
    """プロジェクトを <root>/<id>.json へ保存する。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{project.id}.json'
    path.write_text(
        json.dumps(project.model_dump(mode='json'), ensure_ascii=False),
        encoding='utf-8',
    )
    return path


def list_projects(root: Path) -> list[dict[str, Any]]:
    """論理削除済みを除くプロジェクト一覧(サマリ)を返す。壊れたファイルはスキップする。"""
    root = Path(root)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob('*.json')):
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get('deleted'):
            continue
        out.append({k: doc.get(k) for k in SUMMARY_KEYS})
    out.sort(key=lambda r: r.get('updated_at') or 0, reverse=True)
    return out


def load_project(root: Path, project_id: str) -> ProjectSchema | None:
    """プロジェクトを読み込む。存在しない/壊れている場合はNone(deleted=Trueでも返す)。"""
    root = Path(root)
    path = root / f'{project_id}.json'
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
        return ProjectSchema(**doc)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
