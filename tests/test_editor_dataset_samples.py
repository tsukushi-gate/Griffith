"""データセットサンプル閲覧API(tools/editor/backend/dataset_samples.py)のテスト。

routerを単体のFastAPIアプリにincludeし、main.py非依存で検証する
(STATE不要のためconfigure()は無し)。MNISTはネットワーク/ダウンロード依存の
ため対象外とし、logic系を中心に検証する。
"""

import os
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tools.editor.backend import dataset_samples as dataset_samples_api

app = FastAPI()
app.include_router(dataset_samples_api.router)
client = TestClient(app, base_url='http://localhost:8765')


class TestLogicSamples:
    def test_xor_returns_4_rows(self):
        resp = client.get('/api/datasets/logic:xor/samples')
        assert resp.status_code == 200
        body = resp.json()
        assert body['kind'] == 'logic'
        assert body['num_data'] == 4
        assert body['num_columns'] == 2
        assert len(body['samples']) == 4
        for item in body['samples']:
            assert item['inputs'] is not None and len(item['inputs']) == 2
            assert item['outputs'] is not None and len(item['outputs']) == 1
            assert item['image'] is None and item['label'] is None

    def test_xor_class_distribution(self):
        resp = client.get('/api/datasets/logic:xor/samples')
        body = resp.json()
        # XORは2^2=4パターン、出力0/1がそれぞれ2件ずつ
        assert body['class_distribution_approx'] is False
        assert sum(body['class_distribution'].values()) == 4
        assert set(body['class_distribution']) == {'0', '1'}
        assert body['class_distribution']['0'] == 2
        assert body['class_distribution']['1'] == 2

    def test_paging_is_consistent(self):
        first_page = client.get(
            '/api/datasets/logic:xor/samples', params={'page': 0, 'page_size': 2}
        ).json()
        second_page = client.get(
            '/api/datasets/logic:xor/samples', params={'page': 1, 'page_size': 2}
        ).json()
        assert first_page['num_pages'] == 2
        assert len(first_page['samples']) == 2
        assert len(second_page['samples']) == 2
        first_indices = [s['index'] for s in first_page['samples']]
        second_indices = [s['index'] for s in second_page['samples']]
        assert first_indices == [0, 1]
        assert second_indices == [2, 3]

    def test_page_beyond_range_returns_empty_samples(self):
        resp = client.get(
            '/api/datasets/logic:xor/samples', params={'page': 5, 'page_size': 2}
        )
        assert resp.status_code == 200
        assert resp.json()['samples'] == []

    def test_multi_output_dataset(self):
        # half_adderは2出力(sum, carry)
        resp = client.get('/api/datasets/logic:half_adder/samples')
        assert resp.status_code == 200
        body = resp.json()
        assert body['num_columns'] == 2
        for item in body['samples']:
            assert len(item['outputs']) == 2


class TestUnknownDataset:
    def test_unknown_id_returns_404_japanese(self):
        resp = client.get('/api/datasets/logic:not_a_real_dataset/samples')
        assert resp.status_code == 404
        assert '不明なデータセット' in resp.json()['detail']

    def test_unknown_top_level_id_returns_404(self):
        resp = client.get('/api/datasets/not_a_real_dataset/samples')
        assert resp.status_code == 404


class TestValidation:
    def test_negative_page_is_422(self):
        resp = client.get('/api/datasets/logic:xor/samples', params={'page': -1})
        assert resp.status_code == 422

    def test_page_size_too_large_is_422(self):
        resp = client.get(
            '/api/datasets/logic:xor/samples', params={'page_size': 10_000}
        )
        assert resp.status_code == 422
