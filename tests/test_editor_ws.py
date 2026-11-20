"""エディタ訓練WebSocketの回帰テスト。

Vite proxy 層は対象外。FastAPI TestClient で handshake と
metrics/session_end 配信を検証する。
subscribe v2(detail購読)と新フレームの後方互換テストを含む。
"""

import json
import time
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from tools.editor.backend.main import STATE, app
from tools.editor.backend.monitor import DETAIL_MAX_ELEMS, build_gate_distribution
from tools.editor.backend.training import SessionManager, TrainingSession


@pytest.fixture(autouse=True)
def reset_state():
    """テスト間で共有STATEを初期化する。"""
    STATE.model = None
    STATE.ir = None
    STATE.ir_dirty = False
    STATE.input_width = None
    STATE.sessions = SessionManager()
    yield


@pytest.fixture
def client():
    return TestClient(app, base_url='http://localhost:8765')


def _instant_train(self: TrainingSession):
    """MNIST不要の即時終了訓練(WS配信検証用)。"""
    self.state = 'running'
    self.step = 1
    self.last_loss = 2.31
    self.last_acc = 0.11
    self._put_metrics({
        'type': 'metrics',
        'step': 1,
        'epoch': 0,
        'loss': 2.31,
        'acc': 0.11,
        'weight_delta': {'n1_lutlayer': 0.79},
    })
    self.state = 'finished'


def _wait_session_done(client: TestClient, sid: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f'/api/train/{sid}/status').json()
        if st['state'] in ('finished', 'stopped', 'error'):
            return st
        time.sleep(0.05)
    raise TimeoutError(f'訓練セッション {sid} が {timeout}s 以内に終了しなかった')


def _small_ir() -> dict:
    return {
        'format': 'magicdelta-ir',
        'version': '0.1.0',
        'metadata': {'name': 'small_xor'},
        'inputs': [
            {'id': 'x', 'width': 3, 'shape': None, 'encoding': None, 'semantics': None},
        ],
        'outputs': [
            {'id': 'y', 'width': 1, 'shape': None, 'encoding': None, 'semantics': None},
        ],
        'nodes': [
            {
                'id': 'n1_lutlayer',
                'type': 'lut_layer',
                'params': {
                    'in_width': 3,
                    'out_width': 1,
                    'n_inputs': 2,
                    'indices': [[0], [1]],
                    'truth_tables': ['0x6'],
                },
                'train_info': {'parameterization': 'softmax', 'selection': 'soft'},
                'meta': {},
            },
        ],
        'edges': [
            {'src': 'x', 'dst': 'n1_lutlayer'},
            {'src': 'n1_lutlayer', 'dst': 'y'},
        ],
    }


class TestTrainStartValidation:
    def test_rejects_non_mnist_ir_before_creating_session(self, client):
        put = client.put('/api/models/current/ir', json={'ir': _small_ir()})
        assert put.status_code == 200

        started = client.post('/api/train/start', json={
            'dataset': 'mnist',
            'lr': 0.01,
            'batch_size': 128,
            'epochs': 1,
        })

        assert started.status_code == 400
        assert '入力幅784' in started.json()['detail']
        assert STATE.sessions.sessions == {}


class TestEditorWebSocket:
    def test_ws_handshake_unknown_session(self, client):
        """存在しない sid でも upgrade は 400 にならず accept→close できる。"""
        with client.websocket_connect('/ws/train/no_such_session') as ws:
            # サーバ側で即 close される。クライアントは handshake 成功のみ確認。
            pass

    def test_ws_receives_metrics_or_session_end(self, client):
        demo = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        assert demo.status_code == 200

        with patch.object(TrainingSession, '_run', _instant_train):
            started = client.post('/api/train/start', json={
                'dataset': 'mnist',
                'lr': 0.01,
                'batch_size': 128,
                'epochs': 1,
            })
        assert started.status_code == 200
        sid = started.json()['session_id']
        _wait_session_done(client, sid)

        with client.websocket_connect(f'/ws/train/{sid}') as ws:
            ws.send_text(json.dumps({'type': 'subscribe', 'layers': ['n1_lutlayer']}))
            seen: set[str] = set()
            deadline = time.time() + 5.0
            while time.time() < deadline and 'session_end' not in seen:
                msg = json.loads(ws.receive_text())
                seen.add(msg['type'])
                if msg['type'] == 'metrics':
                    assert msg['step'] == 1
                    assert 'loss' in msg
            assert 'metrics' in seen or 'session_end' in seen
            # 旧subscribe(v1)ではdetail系フレームは流れない(後方互換)
            assert 'gate_distribution' not in seen
            assert 'activations' not in seen


class TestSubscribeV2:
    """subscribe v2(detail購読)とcapture_config/gate_distribution。"""

    @pytest.fixture(autouse=True)
    def synthetic_loader(self, tmp_path, monkeypatch):
        x = torch.rand(512, 1, 28, 28)
        y = torch.randint(0, 10, (512,))
        ds = torch.utils.data.TensorDataset(x, y)

        def fake_loader(self):
            return torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True)

        monkeypatch.setattr(TrainingSession, '_default_loader', fake_loader)
        import tools.editor.backend.training as training_mod
        monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path)
        yield

    def test_capture_config_sent_on_connect(self, client):
        client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        started = client.post('/api/train/start', json={
            'dataset': 'mnist', 'lr': 0.01, 'batch_size': 32, 'epochs': 100,
        })
        sid = started.json()['session_id']
        try:
            with client.websocket_connect(f'/ws/train/{sid}') as ws:
                first = json.loads(ws.receive_text())
                assert first['type'] == 'capture_config'
                assert first['level'] in ('full', 'weights', 'metrics')
                assert first['stride'] >= 1
                assert first['method'] == 'bp'
        finally:
            client.post(f'/api/train/{sid}/stop')

    def test_v2_subscribe_receives_gate_distribution(self, client):
        client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        started = client.post('/api/train/start', json={
            'dataset': 'mnist', 'lr': 0.01, 'batch_size': 32, 'epochs': 100,
        })
        sid = started.json()['session_id']
        try:
            with client.websocket_connect(f'/ws/train/{sid}') as ws:
                ws.send_text(json.dumps({
                    'type': 'subscribe',
                    'layers': ['n1_lutlayer'],
                    'detail': {
                        'gate_distribution': ['n1_lutlayer'],
                        'activations': True,
                    },
                    'detail_interval_ms': 100,
                }))
                got: dict[str, dict] = {}
                deadline = time.time() + 10.0
                while time.time() < deadline and 'gate_distribution' not in got:
                    msg = json.loads(ws.receive_text())
                    got[msg['type']] = msg
                frame = got['gate_distribution']
                assert frame['layer'] == 'n1_lutlayer'
                assert frame['mode'] == 'dense'
                assert frame['probs']['shape'] == [400, 16]
                assert frame['probs']['scale'] == 1.0
                assert frame['entropy_mean'] is not None
        finally:
            client.post(f'/api/train/{sid}/stop')


class TestGateDistributionFrame:
    def test_aggregate_fallback_for_large_layer(self):
        """DETAIL_MAX_ELEMS超の層は集約モード(argmaxヒスト)へ格下げされる。"""
        out_dim = DETAIL_MAX_ELEMS // 16 + 1
        weights = torch.randn(out_dim, 16)
        frame = build_gate_distribution('n1_lutlayer', weights, 'softmax', 2, 5)
        assert frame['mode'] == 'aggregate'
        assert len(frame['hist']) == 16
        assert sum(frame['hist']) == out_dim
        assert 'probs' not in frame

    def test_dense_for_small_layer(self):
        weights = torch.randn(40, 16)
        frame = build_gate_distribution('n1_lutlayer', weights, 'softmax', 2, 5)
        assert frame['mode'] == 'dense'
        assert frame['probs']['shape'] == [40, 16]
