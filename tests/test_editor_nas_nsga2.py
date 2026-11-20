"""NAS API(tools/editor/backend/nas.py)の strategy パラメータのテスト。

既存 tests/test_editor_nas.py はstrategy未指定(既定値)の従来動作を検証する
ため、新規パラメータ('nsga2')の受理・レスポンス互換のみをここで確認する。
"""

import os
import sys
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tools.editor.backend import nas


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(nas.configure(state=None))
    return TestClient(app, base_url='http://localhost:8765')


def _wait_done(client, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get('/api/nas/status').json()
        if st['state'] != 'running':
            return st
        time.sleep(0.1)
    raise AssertionError('探索がタイムアウトした')


class TestNasStrategyParam:
    def test_default_strategy_omitted_still_works(self, tmp_path):
        client = _client()
        body = {
            'search_id': 'default_strategy', 'dataset': 'logic:xor',
            'model_kind': 'logic_mlp', 'max_trials': 3, 'epochs': 2, 'seed': 0,
            'n_random': 4, 'out_root': str(tmp_path),
            'layers_choices': [1, 2], 'width_choices': [4, 8], 'n_inputs_choices': [2],
        }
        res = client.post('/api/nas/start', json=body)
        assert res.status_code == 200, res.text
        st = _wait_done(client)
        assert st['state'] == 'done'

    def test_nsga2_strategy_accepted_and_completes(self, tmp_path):
        client = _client()
        body = {
            'search_id': 'nsga2', 'dataset': 'logic:xor', 'model_kind': 'logic_mlp',
            'max_trials': 3, 'epochs': 2, 'seed': 0, 'n_random': 1,
            'out_root': str(tmp_path),
            'layers_choices': [1, 2], 'width_choices': [4, 8], 'n_inputs_choices': [2],
            'strategy': 'nsga2',
        }
        res = client.post('/api/nas/start', json=body)
        assert res.status_code == 200, res.text
        st = _wait_done(client)
        assert st['state'] == 'done'
        assert st['n_trials'] == 3

        trials = client.get('/api/nas/trials').json()
        assert len(trials['trials']) == 3

    def test_unknown_strategy_value_is_not_rejected_by_schema(self, tmp_path):
        # strategyはstr型のフリーフォームなので、search.py側で従来分岐に
        # フォールバックする(evolution以外は _select_parent 経路のまま)。
        client = _client()
        body = {
            'search_id': 'weird', 'dataset': 'logic:xor', 'model_kind': 'logic_mlp',
            'max_trials': 2, 'epochs': 2, 'seed': 0, 'n_random': 1,
            'out_root': str(tmp_path),
            'layers_choices': [1, 2], 'width_choices': [4, 8], 'n_inputs_choices': [2],
            'strategy': 'unknown_value',
        }
        res = client.post('/api/nas/start', json=body)
        assert res.status_code == 200, res.text
        st = _wait_done(client)
        assert st['state'] == 'done'
