"""エディタバックエンドのDAG(IR v0.2)build+学習の統合テスト(引き継ぎ解消)。

残差構造(fanout+add(xor))のIRを PUT→build→学習(logic:half_adder) まで通す。
"""

import time

import pytest
import torch
from fastapi.testclient import TestClient

from tools.editor.backend.extract import lut_layers_with_ids
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager


def _residual_ir(draft: bool) -> dict:
    """x(2) → lut(2→2) → add(xor) ← x(skip) → y(2) の残差IR。"""
    train_info = {'parameterization': 'softmax', 'connections': 'random'}
    if draft:
        train_info['initialized'] = False
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': 'residual_test'},
        'inputs': [{'id': 'x', 'width': 2}],
        'outputs': [{'id': 'y', 'width': 2}],
        'nodes': [
            {
                'id': 'lut1', 'type': 'lut_layer',
                'params': {
                    'in_width': 2, 'out_width': 2, 'n_inputs': 2,
                    'indices': [[0, 0], [1, 1]],
                    'truth_tables': ['0x6', '0x8'],  # XOR, AND
                },
                'train_info': train_info,
            },
            {
                'id': 'add1', 'type': 'add',
                'params': {'width': 2, 'mode': 'xor'},
            },
        ],
        'edges': [
            {'src': 'x', 'dst': 'lut1'},
            {'src': 'lut1', 'dst': 'add1'},           # slot 0(キー省略)
            {'src': 'x', 'dst': 'add1', 'dst_slot': 1},  # skip接続
            {'src': 'add1', 'dst': 'y'},              # 出力ポートへの接続
        ],
    }


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
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
def client():
    return TestClient(app, base_url='http://localhost:8765')


class TestDagBuild:
    def test_build_preserves_ids_and_matches_simulate(self, client):
        """DAGのbuildが成功し、ノードid不変+hard eval⇔simulate一致。"""
        import numpy as np

        from griffith_logic.ir import ir_from_dict, simulate

        res = client.put('/api/models/current/ir', json={'ir': _residual_ir(draft=False)})
        assert res.status_code == 200, res.text
        res = client.post('/api/models/current/build')
        assert res.status_code == 200, res.text
        ir = res.json()['ir']

        # ノードidが張り替わらない(DAG経路の規約)
        assert {n['id'] for n in ir['nodes']} == {'lut1', 'add1'}
        lut = next(n for n in ir['nodes'] if n['id'] == 'lut1')
        assert 'initialized' not in (lut.get('train_info') or {})

        # 全4パターンで hard eval == simulate(3段等価性のうち2段)
        graph = ir_from_dict(ir)
        x_bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
        sim = simulate(graph, x_bits)
        with torch.no_grad():
            out = STATE.model(torch.tensor(x_bits, dtype=torch.float32))
        assert np.array_equal((out.numpy() > 0.5).astype(np.int64), sim)

    def test_draft_dag_build_initializes(self, client):
        """draftノード入りDAGもbuildで確定する(truth_tablesが更新される)。"""
        torch.manual_seed(0)
        res = client.put('/api/models/current/ir', json={'ir': _residual_ir(draft=True)})
        assert res.status_code == 200, res.text
        res = client.post('/api/models/current/build')
        assert res.status_code == 200, res.text
        lut = next(n for n in res.json()['ir']['nodes'] if n['id'] == 'lut1')
        assert len(lut['params']['truth_tables']) == 2
        assert 'initialized' not in (lut.get('train_info') or {})

    def test_lut_layers_with_ids_uses_node_ids(self, client):
        """DAGモデルの層列挙が元IRノードidを返す(監視の突き合わせ用)。"""
        client.put('/api/models/current/ir', json={'ir': _residual_ir(draft=False)})
        client.post('/api/models/current/build')
        ids = [nid for nid, _ in lut_layers_with_ids(STATE.model)]
        assert ids == ['lut1']

    @pytest.mark.learning
    def test_dag_trains_half_adder(self, client):
        """残差DAGがlogic:half_adderで学習できる(北極星のDAG版)。

        解析解: lut0=B(0xc)→XOR⊕a=b?の系で表現可能(skipがxorのため)。
        """
        torch.manual_seed(42)
        client.put('/api/models/current/ir', json={'ir': _residual_ir(draft=True)})
        client.post('/api/models/current/build')
        res = client.post('/api/train/start', json={
            'dataset': 'logic:half_adder', 'lr': 0.1,
            'batch_size': 4, 'epochs': 100,
        })
        assert res.status_code == 200, res.text
        sid = res.json()['session_id']

        deadline = time.time() + 60
        status = {}
        while time.time() < deadline:
            status = client.get(f'/api/train/{sid}/status').json()
            if status['state'] in ('finished', 'stopped', 'error'):
                break
            time.sleep(0.1)
        assert status['state'] == 'finished', status
        assert status['acc'] is not None and status['acc'] >= 0.99, status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
