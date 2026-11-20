"""セッション再開始の信頼性テスト(「学習2回目が動かない」回帰防止)。

MNIST非依存: 合成データ・合成スナップショットのみ使用する(高速枠)。
狙う不具合:
  1. HistoryStoreの二重close/close後flushでwriterキューjoinが無限ブロックする
  2. SessionManagerが旧履歴を毎回二重closeし、再学習(2〜3回目)でハングする
"""

import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient

from griffith_logic import GroupSum, LutLayer
from tools.editor.backend.history import HistoryStore, LayerRecord, StepSnapshot
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager


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


def _snap(step: int, *, out_dim: int = 8) -> StepSnapshot:
    gen = torch.Generator().manual_seed(step)
    return StepSnapshot(
        step=step, epoch=0, loss=1.0 / (step + 1), acc=step / 100.0,
        ts=time.time(), method='bp',
        layers={'n1_lutlayer': LayerRecord(weights=torch.randn(out_dim, 16, generator=gen))},
    )


def _run_with_timeout(fn, timeout: float = 5.0) -> bool:
    """fnを別スレッドで実行し、timeout内に完了すればTrue(=ハングしない)。"""
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(timeout)
    return not t.is_alive()


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        LutLayer(32, 40, n_inputs=2, connections='random'),
        GroupSum(k=10, tau=1.0),
    )


def _synthetic_loader(n_batches: int = 16, in_dim: int = 32, batch: int = 8):
    x = torch.rand(n_batches * batch, in_dim)
    y = torch.randint(0, 10, (n_batches * batch,))
    ds = torch.utils.data.TensorDataset(x, y)

    def factory():
        return torch.utils.data.DataLoader(ds, batch_size=batch, drop_last=True)

    return factory


def _wait(cond, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


class TestHistoryCloseIdempotent:
    def test_double_close_and_flush_after_close_do_not_hang(self, tmp_path):
        # ディスク退避を強制してwriterキューに実書き込みを積む
        store = HistoryStore('hc', tmp_path, ring_bytes=4096, chunk_steps=4)
        for i in range(1, 41):
            store.append(_snap(i))

        def sequence():
            store.flush()      # 実チャンクを掃く
            store.close()      # sentinel投入
            store.close()      # 二重close(冪等・no-op)
            store.flush()      # close後のjoin: sentinelがtask_done済みでなければ無限待ち

        assert _run_with_timeout(sequence, timeout=5.0), (
            'close二重呼び / close後flushでwriterキューのjoinがハングした'
        )


class TestSessionRestart:
    def test_start_finish_start_thrice_no_hang(self, tmp_path):
        mgr = SessionManager()

        def loop():
            for _ in range(3):
                s = mgr.create(
                    _tiny_model(), lr=0.01, batch_size=8, epochs=1,
                    loader_factory=_synthetic_loader(), history_root=tmp_path,
                )
                # 終端まで待つ(次のcreateは終了済みセッションをpruneして開始できる)
                assert _wait(
                    lambda s=s: s.state in ('finished', 'stopped', 'error'),
                    timeout=10.0,
                )

        assert _run_with_timeout(loop, timeout=30.0), (
            'start→finish→start の反復でハングした(旧履歴の二重close等)'
        )

    def test_terminated_sessions_capped(self, tmp_path):
        mgr = SessionManager()
        mgr.MAX_TERMINATED_KEPT = 2  # テスト高速化のため上限を小さく
        sids = []
        for _ in range(4):
            s = mgr.create(
                _tiny_model(), lr=0.01, batch_size=8, epochs=1,
                loader_factory=_synthetic_loader(), history_root=tmp_path,
            )
            sids.append(s.sid)
            assert _wait(lambda s=s: s.state in ('finished', 'stopped', 'error'))
        # 上限2件のため最古(sids[0])は刈られ、直近の終了済みは一覧に残る
        assert sids[0] not in mgr.sessions
        assert sids[-1] in mgr.sessions
        assert len(mgr.sessions) <= mgr.MAX_TERMINATED_KEPT + 1

    def test_parallel_sessions_run_independently(self, tmp_path):
        # create()はアクティブでも例外を投げず、複数ランが同時に走る
        mgr = SessionManager()
        s1 = mgr.create(
            _tiny_model(), lr=0.01, batch_size=8, epochs=100,
            loader_factory=_synthetic_loader(), history_root=tmp_path,
        )
        s2 = mgr.create(
            _tiny_model(), lr=0.01, batch_size=8, epochs=100,
            loader_factory=_synthetic_loader(), history_root=tmp_path,
        )
        try:
            assert _wait(lambda: s1.state == 'running' and s2.state == 'running')
            assert s1.sid != s2.sid
            assert s1.model is not s2.model  # 独立したモデル実体
            assert len(mgr.list_sessions()) == 2
        finally:
            s1.stop()
            s2.stop()
            s1._thread.join(timeout=5)
            s2._thread.join(timeout=5)


class TestParallelRunsAPI:
    @pytest.fixture(autouse=True)
    def reset_state(self, tmp_path, monkeypatch):
        STATE.model = None
        STATE.ir = None
        STATE.ir_dirty = False
        STATE.input_width = None
        STATE.sessions = SessionManager()
        import tools.editor.backend.training as training_mod
        monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path)
        yield
        for s in list(STATE.sessions.sessions.values()):
            s.stop()
            s._thread.join(timeout=5)

    @pytest.fixture
    def client(self):
        return TestClient(app, base_url='http://localhost:8765')

    def test_two_parallel_starts_both_succeed_and_listed(self, client):
        assert client.put('/api/models/current/ir', json={'ir': _xor_ir()}).status_code == 200
        body = {'dataset': 'logic:xor', 'lr': 0.1, 'batch_size': 32, 'epochs': 50, 'loss': 'bce'}
        r1 = client.post('/api/train/start', json=body)
        r2 = client.post('/api/train/start', json=body)  # アクティブでも409にならない
        assert r1.status_code == 200 and r2.status_code == 200, (r1.json(), r2.json())
        sid1, sid2 = r1.json()['session_id'], r2.json()['session_id']
        assert sid1 != sid2

        listed = client.get('/api/train/sessions').json()['sessions']
        sids = {s['session_id'] for s in listed}
        assert {sid1, sid2} <= sids
        assert all('started_at' in s for s in listed)

    def test_apply_adopts_trained_weights_into_model(self, client):
        assert client.put('/api/models/current/ir', json={'ir': _xor_ir()}).status_code == 200
        started = client.post('/api/train/start', json={
            'dataset': 'logic:xor', 'lr': 0.2, 'batch_size': 32, 'epochs': 60, 'loss': 'bce',
        })
        assert started.status_code == 200
        sid = started.json()['session_id']
        deadline = time.time() + 30.0
        while time.time() < deadline:
            st = client.get(f'/api/train/{sid}/status').json()
            if st['acc'] is not None and st['acc'] >= 1.0 and st['step'] > 50:
                client.post(f'/api/train/{sid}/stop')
            if st['state'] in ('finished', 'stopped', 'error'):
                break
            time.sleep(0.05)

        applied = client.post(f'/api/train/{sid}/apply')
        assert applied.status_code == 200, applied.json()
        ir = applied.json()['ir']
        # 学習済み重みがIRへ書き戻る=truth_tablesが初期の全0(0x0)から変化する
        tt = ir['nodes'][0]['params']['truth_tables']
        assert tt_hex_nonzero(tt), f'apply後もtruth_tablesが未学習のまま: {tt}'


def tt_hex_nonzero(tt) -> bool:
    """truth_tablesのいずれかが非0(=学習で更新された)か。"""
    return any(int(str(h), 16) != 0 for h in tt)
