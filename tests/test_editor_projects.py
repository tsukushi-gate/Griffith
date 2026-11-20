"""プロジェクトCRUD API(tools/editor/backend/project.py, projects_router.py)のテスト。

方針: projects_router.router は main.STATE に依存しない(現状STATE未使用)ため、
main.pyには一切触れず、専用の小さなFastAPIアプリにmountしてテストする
(test_editor_hdl.pyの流儀に倣う)。永続化先(PROJECTS_ROOT)はテスト用の
一時ディレクトリへmonkeypatchし、実outputs/editor_projects/は汚さない。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import ir_to_dict
from tools.editor.backend import project as project_mod
from tools.editor.backend import projects_router

IN_WIDTH = 8
HIDDEN = 6
CLASSES = 2


def _valid_ir_doc(seed: int = 1) -> dict:
    """export_ir経由で作る有効な最小IR dict。"""
    torch.manual_seed(seed)
    model = nn.Sequential(
        LutLayer(IN_WIDTH, HIDDEN, n_inputs=2, parameterization='softmax'),
        GroupSum(k=CLASSES, tau=1.0),
    )
    model.eval()
    graph = export_ir(model, input_width=IN_WIDTH, name=f'net{seed}')
    return ir_to_dict(graph)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(projects_router, 'PROJECTS_ROOT', tmp_path / 'editor_projects')
    app = FastAPI()
    app.include_router(projects_router.configure(state=None))
    return TestClient(app, base_url='http://localhost:8765')


class TestProjectCrud:
    def test_create_put_get_roundtrip(self, client):
        created = client.post(
            '/api/projects', json={'name': 'proj-a', 'description': 'desc'},
        )
        assert created.status_code == 200
        body = created.json()
        pid = body['id']
        assert body['name'] == 'proj-a'
        assert body['networks'] == {}
        assert body['train_configs'] == {}

        networks = {'net1': _valid_ir_doc(1), 'net2': _valid_ir_doc(2)}
        train_configs = {
            'default': {},
            'small': {'model': {'layers': 2}},
        }
        put_res = client.put(
            f'/api/projects/{pid}',
            json={'networks': networks, 'train_configs': train_configs},
        )
        assert put_res.status_code == 200, put_res.text
        put_body = put_res.json()
        assert set(put_body['networks'].keys()) == {'net1', 'net2'}
        assert set(put_body['train_configs'].keys()) == {'default', 'small'}
        assert put_body['train_configs']['small']['model']['layers'] == 2

        get_res = client.get(f'/api/projects/{pid}')
        assert get_res.status_code == 200
        get_body = get_res.json()
        assert get_body['networks']['net1']['metadata']['name'] == 'net1'
        assert get_body['networks']['net2']['metadata']['name'] == 'net2'
        assert get_body['train_configs'] == put_body['train_configs']

    def test_put_rejects_invalid_ir(self, client):
        created = client.post('/api/projects', json={'name': 'proj-b'})
        pid = created.json()['id']

        bad_ir = {
            'format': 'magicdelta-ir', 'version': '0.2.0',
            'metadata': {'name': 'bad'},
            'inputs': [{'id': 'x', 'width': 4, 'shape': None,
                        'encoding': None, 'semantics': None}],
            'outputs': [{'id': 'y', 'width': 4, 'shape': None,
                         'encoding': None, 'semantics': None}],
            'nodes': [],
            # 'y' の生成元ノードが存在しない不整合エッジ
            'edges': [{'src': 'no_such_node', 'src_port': None,
                       'dst': 'y', 'dst_port': None}],
        }
        res = client.put(
            f'/api/projects/{pid}', json={'networks': {'broken': bad_ir}},
        )
        assert res.status_code == 422
        detail = res.json()['detail']
        assert 'broken' in detail

    def test_put_rejects_invalid_train_config(self, client):
        created = client.post('/api/projects', json={'name': 'proj-c'})
        pid = created.json()['id']

        res = client.put(
            f'/api/projects/{pid}',
            json={'train_configs': {'bad_cfg': {'not_a_real_key': 1}}},
        )
        assert res.status_code == 422
        detail = res.json()['detail']
        assert 'bad_cfg' in detail

    def test_delete_is_logical_and_hidden_from_list(self, client):
        created = client.post('/api/projects', json={'name': 'proj-d'})
        pid = created.json()['id']

        list_before = client.get('/api/projects').json()
        assert any(p['id'] == pid for p in list_before)

        del_res = client.delete(f'/api/projects/{pid}')
        assert del_res.status_code == 200

        list_after = client.get('/api/projects').json()
        assert all(p['id'] != pid for p in list_after)

        # 個別GETは論理削除済みを404として扱う方針
        get_res = client.get(f'/api/projects/{pid}')
        assert get_res.status_code == 404

        # ファイル自体は物理削除されず残っている(論理削除)
        deleted_doc = project_mod.load_project(
            projects_router.PROJECTS_ROOT, pid,
        )
        assert deleted_doc is not None
        assert deleted_doc.deleted is True

    def test_get_and_delete_unknown_project_404(self, client):
        assert client.get('/api/projects/does-not-exist').status_code == 404
        assert client.delete('/api/projects/does-not-exist').status_code == 404


class TestProjectModuleUnit:
    """project.py の関数を直接呼ぶユニットテスト。"""

    def test_save_and_load_project(self, tmp_path):
        import time

        now = time.time()
        proj = project_mod.ProjectSchema(
            id='abc123', name='n', created_at=now, updated_at=now,
        )
        project_mod.save_project(proj, tmp_path)
        loaded = project_mod.load_project(tmp_path, 'abc123')
        assert loaded is not None
        assert loaded.name == 'n'

    def test_load_missing_returns_none(self, tmp_path):
        assert project_mod.load_project(tmp_path, 'nope') is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'broken.json').write_text('{not json')
        assert project_mod.load_project(tmp_path, 'broken') is None

    def test_validate_network_rejects_bad_ir(self):
        bad_ir = {
            'format': 'magicdelta-ir', 'version': '0.2.0',
            'metadata': {}, 'nodes': [],
            'inputs': [{'id': 'x', 'width': 4, 'shape': None,
                        'encoding': None, 'semantics': None}],
            'outputs': [{'id': 'y', 'width': 4, 'shape': None,
                         'encoding': None, 'semantics': None}],
            'edges': [{'src': 'no_such_node', 'src_port': None,
                       'dst': 'y', 'dst_port': None}],
        }
        with pytest.raises(ValueError, match='xyz'):
            project_mod.validate_network('xyz', bad_ir)

    def test_validate_train_config_rejects_unknown_key(self):
        with pytest.raises(ValueError, match='badcfg'):
            project_mod.validate_train_config('badcfg', {'nope': 1})
