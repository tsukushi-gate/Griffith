"""巻き戻し(閲覧スクラブ+branch復元)のテスト。

MNIST非依存(合成loader)。u8エンベロープの量子化誤差検証を含む。
"""

import time

import pytest
import torch
from fastapi.testclient import TestClient

from griffith_logic import GroupSum, LutLayer
from tools.editor.backend.main import STATE, app
from tools.editor.backend.monitor import decode_u8, encode_u8
from tools.editor.backend.training import SessionManager, TrainingSession


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        LutLayer(32, 40, n_inputs=2, connections='random'),
        GroupSum(k=10, tau=1.0),
    )


def _synthetic_loader(n_batches: int = 64, in_dim: int = 32, batch: int = 8):
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
    torch.manual_seed(42)
    s = TrainingSession(
        _tiny_model(),
        lr=0.01,
        batch_size=8,
        epochs=100,
        loader_factory=_synthetic_loader(),
        history_root=tmp_path,
    )
    yield s
    s.stop()
    s._thread.join(timeout=5)
    s.history.close()


class TestBranchRestore:
    def test_restore_weights_exact_and_continue(self, session):
        """20step進めてrestore(10): weights厳密一致→さらに5step続行できる。"""
        session.start()
        assert _wait(lambda: session.step >= 20)
        session.pause()
        assert _wait(lambda: session.state == 'paused')

        snap10 = session.history.get(10)
        assert snap10 is not None and snap10.step == 10
        expected = snap10.layers['n1_lutlayer'].weights.clone()

        result = session.restore_to(10)
        assert result['restored_step'] == 10
        assert result['history_truncated_from'] == 11
        layer = session.lut_layers[0][1]
        assert torch.equal(layer.weights.detach().cpu(), expected)
        assert session.step == 10

        # truncate検証: 10より後の履歴は消えている
        idx = session.history.index()
        assert idx['max_step'] == 10

        # そこから5step続行できる(branch再学習)
        session.step_n(5)
        assert _wait(lambda: session.state == 'paused' and session.step == 15)
        assert session.history.index()['max_step'] == 15

    def test_restore_rejected_unless_paused(self, session):
        session.start()
        assert _wait(lambda: session.step >= 1)
        with pytest.raises(RuntimeError):
            session.restore_to(1)

    def test_restore_unknown_step(self, session):
        session.start()
        assert _wait(lambda: session.step >= 1)
        session.pause()
        assert _wait(lambda: session.state == 'paused')
        with pytest.raises(ValueError):
            session.restore_to(-1)


class TestU8Envelope:
    def test_quantization_error_bound(self):
        t = torch.rand(50, 16) * 3.0
        env = encode_u8(t)
        restored = decode_u8(env)
        assert (restored - t).abs().max().item() <= env['scale'] / 255.0 + 1e-6

    def test_probability_scale_fixed(self):
        probs = torch.softmax(torch.randn(20, 16), dim=-1)
        env = encode_u8(probs, scale=1.0)
        assert env['scale'] == 1.0
        restored = decode_u8(env)
        assert (restored - probs).abs().max().item() <= 1.0 / 255.0 + 1e-6
        assert env['shape'] == [20, 16]


class TestHistoryStepRestAPI:
    @pytest.fixture(autouse=True)
    def reset_state(self, tmp_path, monkeypatch):
        STATE.model = None
        STATE.ir = None
        STATE.ir_dirty = False
        STATE.input_width = None
        STATE.sessions = SessionManager()
        x = torch.rand(512, 1, 28, 28)
        y = torch.randint(0, 10, (512,))
        ds = torch.utils.data.TensorDataset(x, y)

        def fake_loader(self):
            return torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True)

        monkeypatch.setattr(TrainingSession, '_default_loader', fake_loader)
        import tools.editor.backend.training as training_mod
        monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path)
        yield

    @pytest.fixture
    def client(self):
        return TestClient(app, base_url='http://localhost:8765')

    def test_history_step_view_and_restore(self, client):
        demo = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        assert demo.status_code == 200
        started = client.post('/api/train/start', json={
            'dataset': 'mnist', 'lr': 0.01, 'batch_size': 32, 'epochs': 100,
        })
        sid = started.json()['session_id']
        assert _wait(
            lambda: client.get(f'/api/train/{sid}/status').json()['step'] >= 5
        )

        # 閲覧スクラブはrunning中でも可能
        view = client.get(
            f'/api/train/{sid}/history/3',
            params={'fields': 'metrics,summary,gate_distribution'},
        )
        assert view.status_code == 200
        body = view.json()
        assert body['resolved_step'] <= 3
        assert 'loss' in body['metrics']
        layer = body['layers']['n1_lutlayer']
        assert len(layer['gate_ids']) == 400
        assert layer['gate_distribution']['shape'] == [400, 16]
        assert layer['gate_distribution']['scale'] == 1.0
        # fullレベルなのでgrad_normも付く
        assert layer['grad_norm']['shape'] == [400]

        # 存在しない未来のstepは直近下位へ丸め、負数は404
        missing = client.get(f'/api/train/{sid}/history/0')
        assert missing.status_code == 404

        # running中のrestoreは409
        rejected = client.post(f'/api/train/{sid}/restore', json={'step': 2})
        assert rejected.status_code == 409

        # pause → restore → 巻き戻り確認
        client.post(f'/api/train/{sid}/pause')
        assert _wait(
            lambda: client.get(f'/api/train/{sid}/status').json()['state'] == 'paused'
        )
        s_now = client.get(f'/api/train/{sid}/status').json()['step']
        target = max(1, s_now - 2)
        restored = client.post(f'/api/train/{sid}/restore', json={'step': target})
        assert restored.status_code == 200
        assert restored.json()['restored_step'] == target
        st = client.get(f'/api/train/{sid}/status').json()
        assert st['step'] == target
        assert st['history']['max_step'] == target

        client.post(f'/api/train/{sid}/stop')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
