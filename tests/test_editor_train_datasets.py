"""エディタ学習の一般化のテスト。

サンプル回路(XOR)が論理データセット+BCEで実際に学習できること
(=第4次実装の北極星デモのバックエンド側)を固定する。
"""

import time

import pytest
from fastapi.testclient import TestClient

from tools.editor.backend.datasets import LLVIT_IN_WIDTH, LOGIC_TARGET_ROWS, resolve_dataset
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


class TestResolveDataset:
    def test_logic_xor(self):
        r = resolve_dataset('logic:xor', batch_size=32)
        assert (r.kind, r.in_width, r.out_width) == ('binary_vector', 2, 1)
        loader = r.loader_factory()
        x, y = next(iter(loader))
        assert x.shape[1] == 2 and y.shape[1] == 1
        assert len(loader.dataset) == LOGIC_TARGET_ROWS

    @pytest.mark.parametrize(
        'name,in_width,out_width',
        [
            ('logic:majority3', 3, 1),
            ('logic:kv_attention2', 5, 1),
            ('logic:seq101', 3, 1),
        ],
    )
    def test_new_logic_datasets(self, name, in_width, out_width):
        r = resolve_dataset(name, batch_size=16)
        assert (r.kind, r.in_width, r.out_width) == ('binary_vector', in_width, out_width)
        loader = r.loader_factory()
        assert len(loader.dataset) == LOGIC_TARGET_ROWS

    def test_mnist_meta(self):
        r = resolve_dataset('mnist', batch_size=32)
        assert (r.kind, r.in_width, r.out_width) == ('classification', 784, 10)

    @pytest.mark.parametrize(
        'name,classes',
        [
            ('llvit:cifar10', 10),
            ('llvit:cifar100', 100),
            ('llvit:flowers102', 102),
            ('llvit:tinyimagenet', 200),
        ],
    )
    def test_llvit_dataset_fake_injection(self, name, classes):
        r = resolve_dataset(name, batch_size=2, params={'fake': True, 'fake_size': 4})
        assert (r.kind, r.in_width, r.out_width) == ('classification', LLVIT_IN_WIDTH, classes)
        x, y = next(iter(r.loader_factory()))
        assert x.shape == (2, 3, 224, 224)
        assert y.shape == (2,)
        assert int(y.max()) < classes

    def test_llvit_synthetic_debug_alias(self):
        r = resolve_dataset('llvit_synthetic_debug', batch_size=16)
        assert (r.kind, r.in_width, r.out_width) == ('classification', 32, 4)
        x, y = next(iter(r.loader_factory()))
        assert x.shape[1:] == (4, 8)
        assert y.ndim == 1

    def test_tinyimagenet_workspace_outside_path_rejected(self, tmp_path):
        outside = tmp_path.parent / 'tiny-imagenet-200'
        with pytest.raises(ValueError, match='workspace外'):
            resolve_dataset(
                'llvit:tinyimagenet',
                batch_size=2,
                params={'root': str(outside)},
            ).loader_factory()

    def test_truth_table(self):
        # AND関数の真理値表(パターンk順: 00,10,01,11)
        r = resolve_dataset('truth_table', batch_size=8, params={
            'table': [[0], [0], [0], [1]], 'n_inputs': 2, 'n_outputs': 1,
        })
        assert (r.kind, r.in_width, r.out_width) == ('binary_vector', 2, 1)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_dataset('logic:no_such', batch_size=8)
        with pytest.raises(ValueError):
            resolve_dataset('cifar', batch_size=8)


@pytest.mark.learning
class TestEditorTrainOnLogicDataset:
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

    @pytest.fixture
    def client(self):
        return TestClient(app, base_url='http://localhost:8765')

    def test_xor_sample_learns_with_bce(self, client):
        """北極星: XORサンプル+logic:xor+BCEでbit acc=1.0に到達する。"""
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
            if st['acc'] is not None and st['acc'] >= 1.0 and st['step'] > 50:
                client.post(f'/api/train/{sid}/stop')
            time.sleep(0.05)

        assert st is not None and st['state'] in ('finished', 'stopped'), st
        assert st['task'] == 'binary_vector'
        assert st['loss'] == 'bce'
        assert st['total_steps'] == (LOGIC_TARGET_ROWS // 32) * 60
        assert st['acc'] is not None and st['acc'] >= 0.999, (
            f'XORが学習できていない: acc={st["acc"]}'
        )
        # 履歴(タイムマシン)も論理データセット学習で機能している
        hist = client.get(f'/api/train/{sid}/history').json()
        assert hist['count'] > 0

    def test_io_width_mismatch_rejected(self, client):
        client.put('/api/models/current/ir', json={'ir': _xor_ir()})
        # 2入力モデルにparity3(3入力)は400
        res = client.post('/api/train/start', json={
            'dataset': 'logic:parity3', 'lr': 0.1, 'batch_size': 32, 'epochs': 1,
        })
        assert res.status_code == 400
        assert '入力幅3' in res.json()['detail']
        assert STATE.sessions.sessions == {}

    def test_invalid_loss_for_classification(self, client):
        client.put('/api/models/current/ir', json={'ir': _xor_ir()})
        res = client.post('/api/train/start', json={
            'dataset': 'logic:xor', 'lr': 0.1, 'batch_size': 32, 'epochs': 1,
            'loss': 'ce',  # binary_vectorにceは不整合
        })
        assert res.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
