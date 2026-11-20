"""config schema API(tools/editor/backend/config_schema.py)のテスト。

方針: config_schema.router は main.STATE に依存しない(現状STATE未使用)ため、
main.pyには一切触れず、専用の小さなFastAPIアプリにmountしてテストする
(test_editor_projects.py/test_editor_hdl.pyの流儀に倣う)。
"""

import os
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tools.editor.backend import config_schema


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(config_schema.configure(state=None))
    return TestClient(app, base_url='http://localhost:8765')


class TestConfigSchema:
    def test_get_schema_ok(self):
        res = _client().get('/api/config/schema')
        assert res.status_code == 200
        schema = res.json()
        assert 'properties' in schema
        for key in ('data', 'model', 'optim', 'run', 'expected_metrics'):
            assert key in schema['properties'], f'{key} がproperties直下にない'

    def test_schema_has_defs_for_nested_models(self):
        """data/model/optim/runはネストしたpydanticモデルなので$defs+$ref形式で出る。"""
        schema = _client().get('/api/config/schema').json()
        assert '$defs' in schema
        for key in ('DataConfig', 'ModelConfig', 'OptimConfig', 'RunConfig'):
            assert key in schema['$defs']
