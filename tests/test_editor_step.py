"""ステップ実行(step_n)とcapture統合のテスト。

MNIST非依存: loader_factory注入で合成データを使い _train_loop を実走する。
"""

import time

import pytest
import torch
from fastapi.testclient import TestClient

from griffith_logic import GroupSum, LutLayer
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager, TrainingSession


def _tiny_model() -> torch.nn.Module:
    """合成データ用の小規模モデル(32次元入力→10クラス)。"""
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        LutLayer(32, 40, n_inputs=2, connections='random'),
        GroupSum(k=10, tau=1.0),
    )


def _synthetic_loader(n_batches: int = 64, in_dim: int = 32, batch: int = 8):
    """再利用可能な合成DataLoaderを返すfactory。"""
    x = torch.rand(n_batches * batch, in_dim)
    y = torch.randint(0, 10, (n_batches * batch,))
    ds = torch.utils.data.TensorDataset(x, y)

    def factory():
        return torch.utils.data.DataLoader(ds, batch_size=batch, drop_last=True)

    return factory


def _wait(cond, timeout: float = 10.0, interval: float = 0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def session(tmp_path):
    s = TrainingSession(
        _tiny_model(),
        lr=0.01,
        batch_size=8,
        epochs=100,  # 合成loaderを繰り返しても止まらない十分な量
        loader_factory=_synthetic_loader(),
        history_root=tmp_path,
    )
    yield s
    s.stop()
    s._thread.join(timeout=5)
    s.history.close()


class TestStepExecution:
    def test_step_n_advances_exactly_and_pauses(self, session):
        session.start()
        assert _wait(lambda: session.step >= 3)
        session.pause()
        assert _wait(lambda: session.state == 'paused')
        s0 = session.step

        session.step_n(3)
        # ちょうど3ステップ進んで自動pauseする
        assert _wait(lambda: session.state == 'paused' and session.step == s0 + 3)
        time.sleep(0.2)  # 追加で進んでいないことを確認
        assert session.step == s0 + 3
        assert session.status()['step_budget'] == -1

    def test_step_n_rejected_unless_paused(self, session):
        session.start()
        assert _wait(lambda: session.state == 'running')
        with pytest.raises(RuntimeError):
            session.step_n(1)

    def test_resume_clears_budget(self, session):
        session.start()
        assert _wait(lambda: session.step >= 1)
        session.pause()
        assert _wait(lambda: session.state == 'paused')
        session.resume()
        assert _wait(lambda: session.state == 'running')
        assert session.status()['step_budget'] == -1


class TestCaptureIntegration:
    def test_history_captures_every_step(self, session):
        """小規模モデル(fullレベル)は全ステップがcaptureされる。"""
        assert session.capture_level == 'full'
        session.start()
        assert _wait(lambda: session.step >= 5)
        session.pause()
        assert _wait(lambda: session.state == 'paused')

        idx = session.history.index()
        assert idx['count'] == session.step  # stride=1で全step
        assert idx['min_step'] == 1
        assert idx['max_step'] == session.step

        # fullレベルはgradも保存されている
        snap = session.history.get(session.step)
        assert snap is not None
        rec = snap.layers['n1_lutlayer']
        assert rec.grad is not None
        assert rec.weights.shape == (40, 16)

    def test_status_extended_keys(self, session):
        session.start()
        assert _wait(lambda: session.step >= 1)
        st = session.status()
        assert st['method'] == 'bp'
        assert st['capture']['level'] == 'full'
        assert st['history']['count'] >= 1
        # 既存キーも不変(後方互換)
        for key in ('session_id', 'state', 'step', 'epoch', 'loss', 'acc', 'error'):
            assert key in st


class TestStepRestAPI:
    """REST経由のstep/history(合成loaderをdefaultに差し替えて実走)。"""

    @pytest.fixture(autouse=True)
    def reset_state(self, tmp_path, monkeypatch):
        STATE.model = None
        STATE.ir = None
        STATE.ir_dirty = False
        STATE.input_width = None
        STATE.sessions = SessionManager()
        # MNISTの代わりに合成データ(モデル入力784に合わせ(1,28,28)形状)
        x = torch.rand(512, 1, 28, 28)
        y = torch.randint(0, 10, (512,))
        ds = torch.utils.data.TensorDataset(x, y)

        def fake_loader(self):
            return torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True)

        monkeypatch.setattr(TrainingSession, '_default_loader', fake_loader)
        # 履歴の書き込み先をtmpへ
        import tools.editor.backend.training as training_mod
        monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path)
        yield

    @pytest.fixture
    def client(self):
        return TestClient(app, base_url='http://localhost:8765')

    def test_step_and_history_endpoints(self, client):
        demo = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        assert demo.status_code == 200
        started = client.post('/api/train/start', json={
            'dataset': 'mnist', 'lr': 0.01, 'batch_size': 32, 'epochs': 100,
        })
        assert started.status_code == 200
        sid = started.json()['session_id']

        # running中のstepは409
        assert _wait(
            lambda: client.get(f'/api/train/{sid}/status').json()['step'] >= 1
        )
        rejected = client.post(f'/api/train/{sid}/step', json={'n': 1})
        assert rejected.status_code == 409

        # pause → step(2) → ちょうど2進んでpaused
        client.post(f'/api/train/{sid}/pause')
        assert _wait(
            lambda: client.get(f'/api/train/{sid}/status').json()['state'] == 'paused'
        )
        s0 = client.get(f'/api/train/{sid}/status').json()['step']
        stepped = client.post(f'/api/train/{sid}/step', json={'n': 2})
        assert stepped.status_code == 200
        assert _wait(lambda: (
            lambda st: st['state'] == 'paused' and st['step'] == s0 + 2
        )(client.get(f'/api/train/{sid}/status').json()))

        # 履歴インデックス
        hist = client.get(f'/api/train/{sid}/history').json()
        assert hist['max_step'] == s0 + 2
        assert hist['capture']['level'] in ('full', 'weights', 'metrics')

        client.post(f'/api/train/{sid}/stop')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
