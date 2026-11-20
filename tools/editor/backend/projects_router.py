"""プロジェクトCRUD API。

複数ネットワーク・複数学習設定をまとめる「プロジェクト」の作成/一覧/取得/
更新/論理削除を扱う。既存の単一IR保存経路(STATE.ir)とは独立に共存する。

main.py への組み込み(循環importを避けるためSTATE参照はconfigureで注入):
    from . import projects_router
    app.include_router(projects_router.configure(STATE))
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .project import (
    ProjectSchema,
    list_projects,
    load_project,
    save_project,
    validate_network,
    validate_train_config,
)

# projects_router.py -> backend -> editor -> tools -> リポジトリルート
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECTS_ROOT = PROJECT_ROOT / 'outputs' / 'editor_projects'

router = APIRouter()

_STATE = None  # main.pyのAppState参照(configureで注入。本routerは現状未使用だが
# probe.py/hdl.py同様のパターンを踏襲するため引数として受け取っておく)


def configure(state) -> APIRouter:
    """STATE参照を注入し、includeすべきrouterを返す。"""
    global _STATE
    _STATE = state
    return router


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    description: str = ''


class UpdateProjectRequest(BaseModel):
    """部分更新的なPUTボディ。省略項目は既存値を維持する。"""

    model_config = ConfigDict(extra='forbid')

    name: str | None = None
    description: str | None = None
    networks: dict[str, dict] | None = None
    train_configs: dict[str, dict] | None = None
    dataset_refs: list[str] | None = None
    run_refs: list[str] | None = None
    optimizers: dict | None = None
    monitors: dict | None = None
    executors: dict | None = None


def _validate_all(networks: dict[str, dict], train_configs: dict[str, dict]) -> None:
    """全ネットワークIR・全TrainConfigを検証する(不正なら422用ValueError)。"""
    for name, ir_doc in networks.items():
        validate_network(name, ir_doc)
    for name, cfg_doc in train_configs.items():
        validate_train_config(name, cfg_doc)


def _get_or_404(project_id: str) -> ProjectSchema:
    project = load_project(PROJECTS_ROOT, project_id)
    if project is None or project.deleted:
        raise HTTPException(status_code=404, detail=f'プロジェクトが見つかりません: {project_id}')
    return project


@router.post('/api/projects')
def create_project(req: CreateProjectRequest):
    """プロジェクトを新規作成する。"""
    now = time.time()
    project = ProjectSchema(
        id=uuid.uuid4().hex,
        name=req.name,
        description=req.description,
        created_at=now,
        updated_at=now,
    )
    save_project(project, PROJECTS_ROOT)
    return project.model_dump(mode='json')


@router.get('/api/projects')
def get_projects():
    """プロジェクト一覧(論理削除済みは除外)を返す。"""
    return list_projects(PROJECTS_ROOT)


@router.get('/api/projects/{project_id}')
def get_project(project_id: str):
    """プロジェクト単体を取得する(存在しない/論理削除済みなら404)。"""
    project = _get_or_404(project_id)
    return project.model_dump(mode='json')


@router.put('/api/projects/{project_id}')
def update_project(project_id: str, req: UpdateProjectRequest):
    """プロジェクトを更新する(全ネットワークIR・全TrainConfigを検証してから保存)。"""
    project = _get_or_404(project_id)

    data = project.model_dump()
    for field in (
        'name', 'description', 'networks', 'train_configs',
        'dataset_refs', 'run_refs', 'optimizers', 'monitors', 'executors',
    ):
        value = getattr(req, field)
        if value is not None:
            data[field] = value

    networks = data['networks']
    train_configs = data['train_configs']
    try:
        _validate_all(networks, train_configs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    data['updated_at'] = time.time()
    updated = ProjectSchema(**data)
    save_project(updated, PROJECTS_ROOT)
    return updated.model_dump(mode='json')


@router.delete('/api/projects/{project_id}')
def delete_project(project_id: str):
    """プロジェクトを論理削除する(ファイルは物理削除しない)。"""
    project = _get_or_404(project_id)
    data = project.model_dump()
    data['deleted'] = True
    data['updated_at'] = time.time()
    deleted = ProjectSchema(**data)
    save_project(deleted, PROJECTS_ROOT)
    return {'ok': True, 'id': project_id}
