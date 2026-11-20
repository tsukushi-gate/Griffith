"""runレジストリのテスト。"""

import time

import pytest
import torch
from fastapi.testclient import TestClient

from griffith_logic import GroupSum, LutLayer
from tools.editor.backend.main import STATE, app
from tools.editor.backend.runs import build_record, list_runs, load_run, save_run
from tools.editor.backend.training import SessionManager, TrainingSession


def _tiny_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        LutLayer(32, 40, n_inputs=2, connections='random'),
        GroupSum(k=10, tau=1.0),
    )


def _loader_factory():
    x = torch.rand(256, 32)
    y = torch.randint(0, 10, (256,))
    ds = torch.utils.data.TensorDataset(x, y)

    def factory():
        return torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True)

    return factory


def _wait(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


class TestRunRegistry:
    def test_session_persists_record(self, tmp_path):
        """セッション終了時にrunレコードが保存され、一覧/個別取得できる。"""
        session = TrainingSession(
            _tiny_model(), lr=0.01, batch_size=32, epochs=1,
            loader_factory=_loader_factory(),
            history_root=tmp_path / 'hist', runs_root=tmp_path / 'runs',
            ir_snapshot={'metadata': {'name': 'test_model'}, 'nodes': []},
        )
        session.start()
        assert _wait(lambda: session.state in ('finished', 'stopped', 'error'))
        session._thread.join(timeout=5)
        session.history.close()

        runs = list_runs(tmp_path / 'runs')
        assert len(runs) == 1
        summary = runs[0]
        assert summary['run_id'] == session.sid
        assert summary['state'] == 'finished'
        assert summary['model_name'] == 'test_model'
        assert summary['final_step'] == session.step

        full = load_run(tmp_path / 'runs', session.sid)
        assert full is not None
        assert len(full['series']['steps']) > 0
        assert len(full['series']['steps']) <= 500  # 間引き上限
        assert full['ir']['metadata']['name'] == 'test_model'

    def test_series_downsampled(self, tmp_path):
        """長い系列は500点以下に間引かれる。"""
        session = TrainingSession(
            _tiny_model(), lr=0.01, batch_size=32, epochs=1,
            loader_factory=_loader_factory(),
            history_root=tmp_path / 'h', runs_root=tmp_path / 'r',
        )
        session.series = [(i, 1.0 / (i + 1), i / 2000) for i in range(2000)]
        session.state = 'finished'
        record = build_record(session, None)
        assert len(record['series']['steps']) == 500

    def test_list_skips_broken_files(self, tmp_path):
        root = tmp_path / 'runs'
        root.mkdir()
        (root / 'broken.json').write_text('{not json')
        save_run({'run_id': 'ok1', 'ended_at': 1.0, 'state': 'finished'}, root)
        runs = list_runs(root)
        assert [r['run_id'] for r in runs] == ['ok1']


class TestRunsRestAPI:
    @pytest.fixture(autouse=True)
    def reset_state(self, tmp_path, monkeypatch):
        STATE.model = None
        STATE.ir = None
        STATE.ir_dirty = False
        STATE.input_width = None
        STATE.sessions = SessionManager()
        import tools.editor.backend.training as training_mod
        monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path / 'hist')
        monkeypatch.setattr(training_mod, 'RUNS_ROOT', tmp_path / 'runs')
        yield

    @pytest.fixture
    def client(self):
        return TestClient(app, base_url='http://localhost:8765')

    def test_runs_endpoints(self, client, tmp_path, monkeypatch):
        # 空一覧
        assert client.get('/api/runs').json() == {'runs': []}
        # 存在しないrun
        assert client.get('/api/runs/nope').status_code == 404
        # レコードを直接置いて一覧に出ることを確認
        save_run(
            {'run_id': 'r1', 'ended_at': 2.0, 'state': 'finished',
             'series': {'steps': [1], 'loss': [0.5], 'acc': [0.9]}},
            tmp_path / 'runs',
        )
        runs = client.get('/api/runs').json()['runs']
        assert runs[0]['run_id'] == 'r1'
        full = client.get('/api/runs/r1').json()
        assert full['series']['acc'] == [0.9]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
