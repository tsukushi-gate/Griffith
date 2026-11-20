"""データセット取込API(tools/editor/backend/dataset_import.py)のテスト。

routerを単体のFastAPIアプリにincludeし、main.py非依存で検証する。
STORE_ROOTはtmp_pathへ差し替え、実データ(outputs/editor_datasets)には触れない。
"""

import io
import os
import sys
import zipfile

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tools.editor.backend import dataset_import as di


@pytest.fixture
def client(tmp_path, monkeypatch):
    """STORE_ROOTをtmp_pathへ差し替えた取込APIクライアント。"""
    monkeypatch.setattr(di, 'STORE_ROOT', tmp_path / 'editor_datasets')
    di._reset_for_test()
    app = FastAPI()
    app.include_router(di.router)
    yield TestClient(app, base_url='http://localhost:8765')
    di._reset_for_test()


# XOR相当CSV(binary_vector、in=2/out=1)
XOR_CSV = b'x__0,x__1,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n'
# NNC形式 分類CSV(y:label → classification)
LABEL_CSV = b'x__0,x__1,y:label\n0,0,0\n0,1,1\n1,0,1\n1,1,2\n'


class TestCsvImport:
    def test_xor_csv_registers_binary_vector(self, client):
        resp = client.post(
            '/api/datasets/import/csv',
            files={'file': ('xor.csv', XOR_CSV, 'text/csv')},
        )
        assert resp.status_code == 200, resp.text
        ds = resp.json()['dataset']
        assert ds['name'].startswith('imported:')
        assert ds['kind'] == 'binary_vector'
        assert ds['in_width'] == 2 and ds['out_width'] == 1
        assert ds['imported'] is True

    def test_appears_in_catalog(self, client):
        client.post('/api/datasets/import/csv',
                    files={'file': ('xor.csv', XOR_CSV, 'text/csv')})
        # list_imported_datasets はレジストリ経由(main.pyが GET /api/datasets へ合成)
        cat = di.list_imported_datasets()
        assert len(cat) == 1
        assert cat[0]['kind'] == 'binary_vector'

    def test_label_csv_is_classification(self, client):
        resp = client.post('/api/datasets/import/csv',
                           files={'file': ('lbl.csv', LABEL_CSV, 'text/csv')})
        ds = resp.json()['dataset']
        assert ds['kind'] == 'classification'
        assert ds['out_width'] == 3  # クラス 0,1,2

    def test_generic_csv_last_column_is_target(self, client):
        raw = b'a,b,c\n0.1,0.2,0\n0.3,0.4,1\n'
        resp = client.post('/api/datasets/import/csv',
                           files={'file': ('g.csv', raw, 'text/csv')})
        assert resp.status_code == 200, resp.text
        ds = resp.json()['dataset']
        assert ds['in_width'] == 2

    def test_empty_csv_rejected(self, client):
        resp = client.post('/api/datasets/import/csv',
                           files={'file': ('e.csv', b'x__0,y\n', 'text/csv')})
        assert resp.status_code == 400


class TestResolveAndTrain:
    def test_resolve_imported_runs_one_epoch(self, client):
        """取込XORをresolve→1エポック学習が回る(受け入れ: resolve経路)。"""
        import torch

        name = client.post('/api/datasets/import/csv',
                           files={'file': ('xor.csv', XOR_CSV, 'text/csv')}
                           ).json()['dataset']['name']
        resolved = di.resolve_imported(name, batch_size=4)
        assert resolved.in_width == 2 and resolved.out_width == 1
        loader = resolved.loader_factory()
        # 1バッチ取り出して形状を確認、簡単な線形モデルで1ステップ学習
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        seen = 0
        for xb, yb in loader:
            assert xb.shape[1] == 2
            loss = torch.nn.functional.mse_loss(model(xb), yb.float().reshape(xb.shape[0], 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += xb.shape[0]
        assert seen > 0

    def test_resolve_unknown_raises_valueerror(self, client):
        with pytest.raises(ValueError):
            di.resolve_imported('imported:does_not_exist', batch_size=4)


class TestSplitDeterminism:
    def test_split_is_deterministic(self, client):
        idx1 = di._split_indices(100, {'train': 0.8, 'val': 0.2, 'test': 0.0}, seed=7)
        idx2 = di._split_indices(100, {'train': 0.8, 'val': 0.2, 'test': 0.0}, seed=7)
        assert idx1 == idx2
        assert len(idx1['train']) == 80 and len(idx1['val']) == 20
        # 別seedでは並びが変わる
        idx3 = di._split_indices(100, {'train': 0.8, 'val': 0.2, 'test': 0.0}, seed=8)
        assert idx3['train'] != idx1['train']

    def test_config_update_changes_split_counts(self, client):
        name = client.post('/api/datasets/import/csv',
                           files={'file': ('xor.csv', XOR_CSV, 'text/csv')}
                           ).json()['dataset']['name']
        slug = name.split(':', 1)[1]
        resp = client.put(f'/api/datasets/imported/{slug}/config',
                          json={'split': {'train': 0.5, 'val': 0.5, 'test': 0.0}, 'seed': 3})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert abs(body['split']['train'] - 0.5) < 1e-9


class TestAugmentation:
    def test_bit_flip_noise_changes_train_only(self, client):
        import torch

        # bit_flip_noise=1.0 なら全ビット反転 → 元と必ず異なる
        x = torch.tensor([[0.0, 1.0, 0.0]])
        aug = {'flip': False, 'shift': 0, 'bit_flip_noise': 1.0}
        out = di._apply_augmentation(x, aug, None, seed=1)
        assert torch.allclose(out, 1.0 - x)
        # noise=0 なら不変
        out0 = di._apply_augmentation(x, {'bit_flip_noise': 0.0}, None, seed=1)
        assert torch.allclose(out0, x)


class TestImageImport:
    def _make_zip(self) -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for cls, val in (('cat', 0), ('dog', 255)):
                for k in range(2):
                    img = Image.new('L', (8, 8), color=val)
                    ib = io.BytesIO()
                    img.save(ib, format='PNG')
                    zf.writestr(f'{cls}/img{k}.png', ib.getvalue())
        return buf.getvalue()

    def test_image_folder_import(self, client):
        resp = client.post(
            '/api/datasets/import/images',
            files={'file': ('imgs.zip', self._make_zip(), 'application/zip')},
            data={'resize': '8', 'grayscale': 'true', 'binarize': '0.5'},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        ds = body['dataset']
        assert ds['kind'] == 'classification'
        assert ds['out_width'] == 2
        assert ds['in_width'] == 8 * 8
        assert set(body['classes']) == {'cat', 'dog'}

    def test_image_samples_return_png(self, client):
        name = client.post(
            '/api/datasets/import/images',
            files={'file': ('imgs.zip', self._make_zip(), 'application/zip')},
            data={'resize': '8', 'grayscale': 'true', 'binarize': '0.5'},
        ).json()['dataset']['name']
        slug = name.split(':', 1)[1]
        resp = client.get(f'/api/datasets/imported/{slug}/samples')
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body['kind'] == 'image'
        assert body['samples'][0]['image'].startswith('data:image/png;base64,')

    def test_binarized_tensor_is_zero_one(self, client):
        name = client.post(
            '/api/datasets/import/images',
            files={'file': ('imgs.zip', self._make_zip(), 'application/zip')},
            data={'resize': '8', 'grayscale': 'true', 'binarize': '0.5'},
        ).json()['dataset']['name']
        x, _y = di._load_arrays(name)
        assert set(np.unique(x).tolist()).issubset({0.0, 1.0})


class TestSamples:
    def test_logic_samples_paging(self, client):
        name = client.post('/api/datasets/import/csv',
                           files={'file': ('xor.csv', XOR_CSV, 'text/csv')}
                           ).json()['dataset']['name']
        slug = name.split(':', 1)[1]
        resp = client.get(f'/api/datasets/imported/{slug}/samples',
                          params={'page': 0, 'page_size': 2})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body['num_data'] == 4
        assert len(body['samples']) == 2
        assert body['samples'][0]['inputs'] is not None

    def test_split_filter(self, client):
        name = client.post('/api/datasets/import/csv',
                           files={'file': ('xor.csv', XOR_CSV, 'text/csv')}
                           ).json()['dataset']['name']
        slug = name.split(':', 1)[1]
        resp = client.get(f'/api/datasets/imported/{slug}/samples', params={'split': 'train'})
        assert resp.status_code == 200
        # 既定split train=0.8 → 4件中3件程度
        assert resp.json()['num_data'] <= 4

    def test_unknown_dataset_404(self, client):
        resp = client.get('/api/datasets/imported/nope/samples')
        assert resp.status_code == 404


class TestEncodingPreview:
    def test_binarize(self, client):
        resp = client.post('/api/datasets/encoding-preview',
                           json={'values': [0.2, 0.8], 'mode': 'binarize', 'threshold': 0.5})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body['bits_per_value'] == 1
        assert body['grid'] == [[0], [1]]

    def test_thermometer(self, client):
        resp = client.post('/api/datasets/encoding-preview',
                           json={'values': [0.5], 'mode': 'thermometer', 'bits': 4})
        body = resp.json()
        # bit_j = 1 if 0.5 > (j+1)/5 → j=0:0.5>0.2=1, j=1:0.5>0.4=1, j=2:0.5>0.6=0, j=3:0
        assert body['grid'] == [[1, 1, 0, 0]]
        assert body['bits_per_value'] == 4

    def test_threshold_changes_bits(self, client):
        low = client.post('/api/datasets/encoding-preview',
                          json={'values': [0.3], 'mode': 'binarize', 'threshold': 0.2}).json()
        high = client.post('/api/datasets/encoding-preview',
                           json={'values': [0.3], 'mode': 'binarize', 'threshold': 0.5}).json()
        assert low['grid'] == [[1]] and high['grid'] == [[0]]

    def test_invalid_mode_422(self, client):
        resp = client.post('/api/datasets/encoding-preview',
                           json={'values': [0.1], 'mode': 'bogus'})
        assert resp.status_code == 422
