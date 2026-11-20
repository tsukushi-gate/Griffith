"""エディタ学習の評価機構とデータセットカタログのテスト。

- hard eval(model.eval()=離散化)がエポック毎+セッション終了時に走り、
  metricsフレームへval_loss/val_accがスパースに載ること
- classificationタスクで混同行列がGET /api/train/{sid}/confusionから取れること
- GET /api/datasets がLOGIC_DATASETSから自動生成されたカタログを返すこと
"""

import time

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from griffith_logic.ir import build_torch, ir_from_dict
from tools.editor.backend.datasets import list_datasets
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager, TrainingSession


def _xor_ir() -> dict:
    return {
        'format': 'magicdelta-ir', 'version': '0.1.0',
        'metadata': {'name': 'test_xor'},
        'inputs': [{'id': 'x', 'width': 2, 'shape': None, 'encoding': None, 'semantics': None}],
        'outputs': [{'id': 'y', 'width': 1, 'shape': None, 'encoding': None, 'semantics': None}],
        'nodes': [{
            'id': 'n1_lutlayer', 'type': 'lut_layer',
            'params': {'in_width': 2, 'out_width': 1, 'n_inputs': 2,
                       'indices': [[0], [1]], 'truth_tables': ['0x0']},
            'train_info': {'parameterization': 'softmax', 'selection': 'soft',
                           'initialized': False},
            'meta': {},
        }],
        'edges': [{'src': 'x', 'dst': 'n1_lutlayer'},
                  {'src': 'n1_lutlayer', 'dst': 'y'}],
    }


def _build_xor_model() -> nn.Module:
    graph = ir_from_dict(_xor_ir())
    core = build_torch(graph)
    model = nn.Sequential(nn.Flatten(), *core)
    model.train()
    return model


def _drain_metrics(session: TrainingSession) -> list[dict]:
    frames = []
    while not session.metrics_queue.empty():
        frames.append(session.metrics_queue.get_nowait())
    return frames


def _wait_state(session: TrainingSession, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if session.state in ('finished', 'stopped', 'error'):
            return session.state
        time.sleep(0.05)
    raise TimeoutError(f'セッションが{timeout}s以内に終了しなかった: {session.state}')


class TestListDatasets:
    def test_contains_mnist_and_logic(self):
        items = list_datasets()
        by_name = {d['name']: d for d in items}
        assert by_name['mnist'] == {
            'name': 'mnist', 'label': 'MNIST',
            'in_width': 784, 'out_width': 10, 'kind': 'classification',
            'input_spec': {'shape': [1, 28, 28], 'width': 784},
            'output_spec': {'width': 10, 'semantics': 'class_scores'},
        }
        assert by_name['logic:xor']['in_width'] == 2
        assert by_name['logic:xor']['out_width'] == 1
        assert by_name['logic:xor']['kind'] == 'binary_vector'
        # LOGIC_DATASETS全キーがlogic:プレフィクスで載る
        from griffith_logic.datasets import LOGIC_DATASETS
        for key in LOGIC_DATASETS:
            assert f'logic:{key}' in by_name

    def test_endpoint_returns_catalog(self):
        # 現在、カタログは組込+取込データセットの合成になった。
        # 組込分が欠けず先頭に全て含まれることを検証する(取込分は環境依存なので件数を固定しない)
        client = TestClient(app, base_url='http://localhost:8765')
        res = client.get('/api/datasets')
        assert res.status_code == 200
        payload = res.json()
        builtin = list_datasets()
        assert payload[: len(builtin)] == builtin
        for extra in payload[len(builtin):]:
            assert extra.get('imported') is True


class TestEvalOnLogicDataset:
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

    def test_xor_training_reaches_val_acc(self, client):
        """北極星: XORサンプル+logic:xorでhard eval val_acc=1.0に到達する。"""
        put = client.put('/api/models/current/ir', json={'ir': _xor_ir()})
        assert put.status_code == 200

        started = client.post('/api/train/start', json={
            'dataset': 'logic:xor', 'lr': 0.1, 'batch_size': 32, 'epochs': 60,
            'loss': 'bce',
        })
        assert started.status_code == 200, started.json()
        sid = started.json()['session_id']

        deadline = time.time() + 30.0
        st = None
        while time.time() < deadline:
            st = client.get(f'/api/train/{sid}/status').json()
            if st['state'] in ('finished', 'stopped', 'error'):
                break
            if st.get('last_val_acc') is not None and st['last_val_acc'] >= 1.0:
                client.post(f'/api/train/{sid}/stop')
            time.sleep(0.05)

        assert st is not None and st['state'] in ('finished', 'stopped'), st
        assert st['last_val_acc'] is not None
        assert st['last_val_acc'] >= 0.999, (
            f'XORのhard evalが1.0に到達していない: {st["last_val_acc"]}'
        )
        # binary_vectorタスクでは混同行列は作られない(分類のみ)
        conf = client.get(f'/api/train/{sid}/confusion')
        assert conf.status_code == 404

    def test_metrics_frames_carry_sparse_val_keys(self, tmp_path):
        """valキーは評価直後のフレームのみ(スパース)に載る。"""
        session = TrainingSession(
            _build_xor_model(), dataset='logic:xor', lr=0.05,
            batch_size=32, epochs=2, loss='bce',
            history_root=tmp_path / 'h', runs_root=tmp_path / 'r',
        )
        session.start()
        _wait_state(session)

        frames = [f for f in _drain_metrics(session) if f['type'] == 'metrics']
        assert frames, 'metricsフレームが1件も流れていない'
        val_frames = [f for f in frames if 'val_loss' in f]
        assert val_frames, 'val_loss付きフレームが無い'
        assert all('val_acc' in f for f in val_frames)
        # 全フレームには載らない(評価は1エポック毎+終了時のみ)
        assert len(val_frames) < len(frames)
        assert session.last_val_acc is not None
        assert session.last_val_step is not None
        session.history.close()

    def test_eval_interval_steps(self, tmp_path):
        """eval_interval_steps指定時はそのstep間隔で評価される。"""
        session = TrainingSession(
            _build_xor_model(), dataset='logic:xor', lr=0.05,
            batch_size=32, epochs=1, loss='bce',
            eval_interval_steps=4,
            history_root=tmp_path / 'h', runs_root=tmp_path / 'r',
        )
        session.start()
        _wait_state(session)

        frames = [f for f in _drain_metrics(session) if f['type'] == 'metrics']
        val_steps = [f['step'] for f in frames if 'val_loss' in f]
        assert val_steps, 'val付きフレームが無い'
        # 終了時評価(最終step)以外は4の倍数のstepで評価される
        last_step = max(f['step'] for f in frames)
        for s in val_steps:
            assert s % 4 == 0 or s == last_step, f'評価stepが間隔外: {s}'
        session.history.close()

    def test_confusion_for_classification(self, client, tmp_path):
        """classificationタスクのeval後、混同行列APIが10×10行列を返す。"""
        demo = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        assert demo.status_code == 200

        # 合成データ(MNIST形状)を注入して短時間で1エポック回す
        x = torch.rand(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        train_ds = torch.utils.data.TensorDataset(x, y)
        ex = torch.rand(50, 1, 28, 28)
        ey = torch.randint(0, 10, (50,))
        eval_ds = torch.utils.data.TensorDataset(ex, ey)

        session = STATE.sessions.create(
            STATE.model, dataset='mnist', lr=0.01, batch_size=32, epochs=1,
            loader_factory=lambda: torch.utils.data.DataLoader(
                train_ds, batch_size=32, drop_last=True),
            eval_loader_factory=lambda: torch.utils.data.DataLoader(
                eval_ds, batch_size=32),
            history_root=tmp_path / 'h', runs_root=tmp_path / 'r',
        )
        _wait_state(session)

        res = client.get(f'/api/train/{session.sid}/confusion')
        assert res.status_code == 200, res.json()
        doc = res.json()
        assert doc['classes'] == 10
        assert len(doc['matrix']) == 10
        assert all(len(row) == 10 for row in doc['matrix'])
        # 全セル合計 = 評価サンプル数
        assert sum(sum(row) for row in doc['matrix']) == 50
        assert doc['step'] == session.last_val_step
        session.history.close()

    def test_confusion_404_before_eval(self, client, tmp_path):
        """評価前(eval loaderなし)は混同行列APIが404を返す。"""
        demo = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        assert demo.status_code == 200
        x = torch.rand(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        ds = torch.utils.data.TensorDataset(x, y)
        session = STATE.sessions.create(
            STATE.model, dataset='mnist', lr=0.01, batch_size=32, epochs=1,
            loader_factory=lambda: torch.utils.data.DataLoader(
                ds, batch_size=32, drop_last=True),
            eval_loader_factory=lambda: None,  # 評価なし
            history_root=tmp_path / 'h', runs_root=tmp_path / 'r',
        )
        _wait_state(session)
        res = client.get(f'/api/train/{session.sid}/confusion')
        assert res.status_code == 404
        session.history.close()

    def test_stop_triggers_final_eval(self, client):
        """途中stopでもセッション終了時のhard evalが走りlast_val_accが立つ。"""
        put = client.put('/api/models/current/ir', json={'ir': _xor_ir()})
        assert put.status_code == 200
        started = client.post('/api/train/start', json={
            'dataset': 'logic:xor', 'lr': 0.05, 'batch_size': 32, 'epochs': 1000,
            'loss': 'bce',
        })
        assert started.status_code == 200
        sid = started.json()['session_id']

        # 数step進んだのを確認してからstop
        deadline = time.time() + 10.0
        while time.time() < deadline:
            st = client.get(f'/api/train/{sid}/status').json()
            if st['step'] >= 3:
                break
            time.sleep(0.05)
        client.post(f'/api/train/{sid}/stop')

        deadline = time.time() + 10.0
        st = None
        while time.time() < deadline:
            st = client.get(f'/api/train/{sid}/status').json()
            if st['state'] in ('finished', 'stopped', 'error'):
                break
            time.sleep(0.05)
        assert st is not None and st['state'] == 'stopped', st
        assert st['last_val_acc'] is not None, '終了時評価が走っていない'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
