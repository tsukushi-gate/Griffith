"""推論プローブAPI(tools/editor/backend/probe.py)のテスト。

routerを単体のFastAPIアプリにincludeし、main.py非依存で検証する
(STATE注入はprobe.configure()。main.pyへの統合は別途)。
"""

import json
import os
import sys

import pytest
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir import ir_to_dict
from tools.editor.backend import probe as probe_api

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'ir_v01')


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, f'{name}.json')) as f:
        return json.load(f)


class _NoSessions:
    """SessionManager代替(アクティブセッションなし)"""

    def active_session(self):
        return None


class _RunningSession:
    state = 'running'
    sid = 'sess-test'


class _RunningSessions:
    def active_session(self):
        return _RunningSession()


class FakeState:
    """main.AppState互換の最小状態(main.py非依存)"""

    def __init__(self):
        self.model = None
        self.ir = None
        self.ir_dirty = False
        self.input_width = None
        self.sessions = _NoSessions()


@pytest.fixture
def ctx():
    state = FakeState()
    app = FastAPI()
    app.include_router(probe_api.configure(state))
    return state, TestClient(app, base_url='http://localhost:8765')


class TestProbeEndpointXor:
    def test_xor_all_patterns_truth_table(self, ctx):
        """XORサンプルIRの4パターン全てで真理値一致(hard+softビット化)"""
        state, client = ctx
        fx = _load_fixture('xor')
        state.ir = fx['ir']

        res = client.post('/api/probe', json={
            'inputs': fx['golden_inputs'], 'mode': 'both',
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['input_width'] == 2
        assert body['batch'] == 4
        assert body['hard']['n1_lutlayer'] == fx['golden_outputs']
        soft_bits = [[1 if v > 0.5 else 0 for v in row]
                     for row in body['soft']['n1_lutlayer']]
        assert soft_bits == fx['golden_outputs']
        # 入力ポート値も含む
        assert body['hard']['x'] == fx['golden_inputs']

    def test_mode_hard_only(self, ctx):
        state, client = ctx
        state.ir = _load_fixture('xor')['ir']
        res = client.post('/api/probe', json={
            'inputs': [[0, 1]], 'mode': 'hard',
        })
        assert res.status_code == 200
        body = res.json()
        assert 'hard' in body and 'soft' not in body

    def test_mode_soft_only_rounded(self, ctx):
        state, client = ctx
        state.ir = _load_fixture('xor')['ir']
        res = client.post('/api/probe', json={
            'inputs': [[1, 1]], 'mode': 'soft',
        })
        assert res.status_code == 200
        body = res.json()
        assert 'hard' not in body and 'soft' in body
        for rows in body['soft'].values():
            for row in rows:
                for v in row:
                    assert v == round(v, 4)  # 小数4桁丸め


class TestProbeEndpointModelPaths:
    def _editor_model_and_ir(self):
        """エディタ規約のモデル(Flatten前置)+export_ir由来のIR dict"""
        torch.manual_seed(3)
        model = nn.Sequential(
            nn.Flatten(),
            LutLayer(4, 8, n_inputs=2, parameterization='independent'),
            GroupSum(k=2, tau=1.0),
        )
        model.eval()
        ir = ir_to_dict(export_ir(model, input_width=4))
        return model, ir

    def test_uses_built_model_and_restores_mode(self, ctx):
        """STATE.model使用時: n{i}規約で一致し、train/evalモードを復元する"""
        state, client = ctx
        model, ir = self._editor_model_and_ir()
        state.model = model
        state.ir = ir
        model.train()  # プローブ後に復元されるべきモード

        res = client.post('/api/probe', json={
            'inputs': [[0, 1, 0, 1], [1, 1, 1, 1]], 'mode': 'both',
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert set(body['hard']) == {'x', 'n1_lutlayer', 'n2_groupsum'}
        bits = [[1 if v > 0.5 else 0 for v in row]
                for row in body['soft']['n1_lutlayer']]
        assert bits == body['hard']['n1_lutlayer']
        assert model.training is True  # モード復元

    def test_running_session_does_not_touch_model(self, ctx):
        """訓練running中はSTATE.modelに触らず独立モデルで応答する(安全性)"""
        state, client = ctx
        model, ir = self._editor_model_and_ir()
        state.model = model
        state.ir = ir
        state.sessions = _RunningSessions()
        model.train()  # 学習中モデルのモードが切り替わってはいけない

        res = client.post('/api/probe', json={
            'inputs': [[1, 0, 1, 0]], 'mode': 'both',
        })
        assert res.status_code == 200, res.text
        assert model.training is True  # eval切替されていない

    def test_builds_from_ir_when_model_missing(self, ctx):
        """モデル未構築ならIRからbuild_torchで直接構築する"""
        state, client = ctx
        state.ir = _load_fixture('parity3')['ir']
        res = client.post('/api/probe', json={
            'inputs': _load_fixture('parity3')['golden_inputs'],
            'mode': 'both',
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert body['hard']['n2_lutlayer'] == \
            _load_fixture('parity3')['golden_outputs']
        bits = [[1 if v > 0.5 else 0 for v in row]
                for row in body['soft']['n2_lutlayer']]
        assert bits == body['hard']['n2_lutlayer']


class TestProbeEndpointErrors:
    def test_no_ir_400(self, ctx):
        _, client = ctx
        res = client.post('/api/probe', json={'inputs': [[0, 1]]})
        assert res.status_code == 400
        assert 'IR未設定' in res.json()['detail']

    def test_wrong_width_422(self, ctx):
        state, client = ctx
        state.ir = _load_fixture('xor')['ir']
        res = client.post('/api/probe', json={'inputs': [[0, 1, 1]]})
        assert res.status_code == 422
        assert '入力幅' in res.json()['detail']

    def test_non_binary_422(self, ctx):
        state, client = ctx
        state.ir = _load_fixture('xor')['ir']
        res = client.post('/api/probe', json={'inputs': [[0, 2]]})
        assert res.status_code == 422
        assert '0/1' in res.json()['detail']

    def test_empty_inputs_422(self, ctx):
        state, client = ctx
        state.ir = _load_fixture('xor')['ir']
        res = client.post('/api/probe', json={'inputs': []})
        assert res.status_code == 422

    def test_invalid_mode_422(self, ctx):
        state, client = ctx
        state.ir = _load_fixture('xor')['ir']
        res = client.post('/api/probe', json={
            'inputs': [[0, 1]], 'mode': 'fuzzy',
        })
        assert res.status_code == 422  # pydanticパターン検証


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
