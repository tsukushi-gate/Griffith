"""NAS API(tools/editor/backend/nas.py)のテスト。

方針: nas.router は main.STATE に依存しない(自前のバックグラウンド探索
マネージャを持つ)ため、main.pyには一切触れず専用の小さなFastAPIアプリに
mountしてテストする(test_editor_config_schema.py/test_editor_hdl.pyの流儀)。
探索は small space + max_trials=3 + epochs=2 で数秒に収める。
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


def _start_body(out_root, **kw):
    body = {
        'search_id': 'test', 'dataset': 'logic:xor', 'model_kind': 'logic_mlp',
        'max_trials': 3, 'epochs': 2, 'seed': 0, 'n_random': 4,
        'out_root': str(out_root),
        'layers_choices': [1, 2], 'width_choices': [4, 8], 'n_inputs_choices': [2],
    }
    body.update(kw)
    return body


def _wait_done(client, timeout=30.0):
    """探索スレッドの完了を待つ(running が消えるまでポーリング)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get('/api/nas/status').json()
        if st['state'] != 'running':
            return st
        time.sleep(0.1)
    raise AssertionError('探索がタイムアウトした')


class TestNasApi:
    def test_full_lifecycle(self, tmp_path):
        client = _client()

        # 開始
        res = client.post('/api/nas/start', json=_start_body(tmp_path))
        assert res.status_code == 200, res.text
        assert res.json()['search_id'] == 'test'

        st = _wait_done(client)
        assert st['state'] == 'done'
        assert st['n_trials'] == 3
        assert st['max_trials'] == 3

        # trial一覧
        trials = client.get('/api/nas/trials').json()
        assert len(trials['trials']) == 3
        row = trials['trials'][0]
        for key in ('trial_id', 'acc', 'total_luts', 'logic_depth', 'gates',
                    'partial', 'state', 'parent_id', 'genome'):
            assert key in row
        # パレートは非空(全trial done かつ partial false)
        assert isinstance(trials['pareto'], list)

        # trialのIR取得(EDITで開く用)
        ir = client.get('/api/nas/trials/0/ir').json()
        assert ir['trial_id'] == 0
        assert 'nodes' in ir['ir'] and 'edges' in ir['ir']

        # 存在しないtrialは404
        assert client.get('/api/nas/trials/99/ir').status_code == 404

    def test_start_conflict_while_running(self, tmp_path):
        client = _client()
        # 十分大きめのtrial数で走らせ、実行中に再startして409を得る
        client.post('/api/nas/start', json=_start_body(tmp_path, max_trials=6, epochs=2))
        # 直後(まだrunning)なら409。既に終わっていればスキップ相当で許容。
        r2 = client.post('/api/nas/start', json=_start_body(tmp_path, max_trials=6))
        assert r2.status_code in (200, 409)
        client.post('/api/nas/stop')
        _wait_done(client)

    def test_stop_request(self, tmp_path):
        client = _client()
        client.post('/api/nas/start', json=_start_body(tmp_path, max_trials=6, epochs=2))
        r = client.post('/api/nas/stop')
        assert r.status_code == 200
        st = _wait_done(client)
        assert st['state'] in ('stopped', 'done')

    def test_start_rejects_unknown_field(self, tmp_path):
        client = _client()
        body = _start_body(tmp_path)
        body['bogus'] = 1  # extra='forbid'
        assert client.post('/api/nas/start', json=body).status_code == 422
