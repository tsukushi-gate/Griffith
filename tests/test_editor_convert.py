"""エディタ変換パイプラインのREST契約テスト。

既存 test_editor_*.py の流儀(TestClient・合成データ・monkeypatch)に従う:
resolve_teacher を合成ミニMLPへ差し替え、データセット解決を合成0/1ローダへ
差し替えて、start→statusポーリング→done→STATE.ir 反映+GET /models/current/ir
取得を固定する。エラー系(不正preset/学習中409/cancel)とpresets一覧も検証。

数秒規模に抑えるため spec(mnist-intlower-v1)を overrides で極小化する。
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import tools.editor.backend.convert as convert_mod
from tools.editor.backend import schemas
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager


class _MiniMLP(nn.Module):
    """8-6-3 のReLU-MLP(合成teacherのモデル本体)。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _FakeTeacher:
    def __init__(self) -> None:
        self.model = _MiniMLP()
        self.num_classes = 3
        self.input_size = (8,)
        self.fingerprint = 'fake-mini-mlp'


def _synthetic_loaders():
    rng = np.random.default_rng(0)
    x = (rng.random((96, 8)) > 0.5).astype(np.float32)
    w = rng.standard_normal((8, 3))
    y = (x.astype(np.float64) @ w).argmax(axis=1).astype(np.int64)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return (DataLoader(ds, batch_size=16, shuffle=True),
            DataLoader(ds, batch_size=16, shuffle=False))


# 極小overrides(prune1段fan-in4・epoch1・QAT無効・gate無効)で数秒に抑える
_TINY_OVERRIDES = {
    'fanins': [4], 'epochs_per_stage': 1, 'lr': 1e-2,
    'weight_bits_hidden': 4, 'weight_bits_final': 8,
    'qat_epochs': 0, 'gate_acc': 0.0, 'batch_size': 16, 'seed': 0,
}


@pytest.fixture(autouse=True)
def reset_and_patch(monkeypatch):
    """STATE初期化 + teacher解決/データセット解決を合成へ差し替える。"""
    STATE.model = None
    STATE.ir = None
    STATE.ir_dirty = False
    STATE.input_width = None
    STATE.sessions = SessionManager()
    STATE.conversions = convert_mod.ConversionManager()

    monkeypatch.setattr(
        convert_mod, 'resolve_teacher',
        lambda source, device='cpu': _FakeTeacher())
    monkeypatch.setattr(
        convert_mod, '_binarized_dataset_loaders',
        lambda dataset, batch_size, binarize_threshold: _synthetic_loaders())
    yield


@pytest.fixture
def client():
    return TestClient(app, base_url='http://localhost:8765')


def _poll_until_done(client, cid, timeout=30.0):
    deadline = time.time() + timeout
    st = None
    while time.time() < deadline:
        st = client.get(f'/api/convert/{cid}/status').json()
        if st['phase'] in ('done', 'error', 'cancelled'):
            return st
        time.sleep(0.05)
    return st


class TestConvertPresets:
    def test_presets_list(self, client):
        res = client.get('/api/convert/presets')
        assert res.status_code == 200
        names = [p['name'] for p in res.json()['presets']]
        assert 'mnist-intlower-v1' in names
        assert 'fashion-intlower-v1' in names
        # 勝ちレシピの主要spec値
        mnist = next(p for p in res.json()['presets']
                     if p['name'] == 'mnist-intlower-v1')
        assert mnist['spec']['weight_bits_hidden'] == 4
        assert mnist['dataset'] == 'mnist'
        # スキーマ契約(ドリフト検出): ConvertPresetInfo でvalidateできる
        for p in res.json()['presets']:
            schemas.ConvertPresetInfo.model_validate(p)


class TestConvertFlow:
    def test_start_poll_done_reflects_ir(self, client):
        started = client.post('/api/convert/start', json={
            'source': 'ckpt:fake', 'preset': 'mnist-intlower-v1',
            'overrides': _TINY_OVERRIDES,
        })
        assert started.status_code == 200, started.json()
        cid = started.json()['convert_id']

        st = _poll_until_done(client, cid)
        assert st is not None and st['phase'] == 'done', st
        assert st['metrics']['exact'] is True
        assert isinstance(st['metrics']['hard_acc'], float)
        assert st['has_ir'] is True
        # スキーマ契約(ドリフト検出): ConvertStatus でvalidateできる
        schemas.ConvertStatus.model_validate(st)

        # done検知でSTATEへ反映済み → GET /models/current/ir で取得できる
        ir_res = client.get('/api/models/current/ir')
        assert ir_res.status_code == 200
        ir = ir_res.json()['ir']
        assert ir_res.json()['dirty'] is True
        assert any(n['type'] == 'lut_layer' for n in ir['nodes'])
        assert STATE.model is None  # 変換IRは未build

    def test_cancel(self, client):
        started = client.post('/api/convert/start', json={
            'source': 'ckpt:fake', 'preset': 'mnist-intlower-v1',
            'overrides': {**_TINY_OVERRIDES, 'epochs_per_stage': 3},
        })
        cid = started.json()['convert_id']
        # 即キャンセル要求 → phase境界で cancelled へ抜ける
        client.post(f'/api/convert/{cid}/cancel')
        st = _poll_until_done(client, cid)
        assert st['phase'] in ('cancelled', 'done'), st
        # cancelledなら error に理由が載る(極小specで先にdoneになる競合も許容)
        if st['phase'] == 'cancelled':
            assert st['error'] is not None


class TestConvertErrors:
    def test_unknown_preset_400(self, client):
        res = client.post('/api/convert/start', json={
            'source': 'ckpt:fake', 'preset': 'no-such-preset',
        })
        assert res.status_code == 400

    def test_bad_overrides_400(self, client):
        res = client.post('/api/convert/start', json={
            'source': 'ckpt:fake', 'preset': 'mnist-intlower-v1',
            'overrides': {'nonexistent_field': 1},
        })
        assert res.status_code == 400

    def test_missing_source_422(self, client):
        # sourceは必須(pydantic必須欠落=422)
        res = client.post('/api/convert/start', json={
            'preset': 'mnist-intlower-v1',
        })
        assert res.status_code == 422

    def test_status_404(self, client):
        assert client.get('/api/convert/nope/status').status_code == 404

    def test_start_rejected_while_training(self, client, monkeypatch):
        """学習セッション実行中は409(モデル差し替え不可)。"""
        class _FakeActive:
            sid = 'train123'
            state = 'running'

        monkeypatch.setattr(STATE.sessions, 'active_session',
                            lambda: _FakeActive())
        res = client.post('/api/convert/start', json={
            'source': 'ckpt:fake', 'preset': 'mnist-intlower-v1',
        })
        assert res.status_code == 409


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
