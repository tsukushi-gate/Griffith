"""一括評価API(tools/editor/backend/evaluate.py)のテスト。

routerを単体のFastAPIアプリにincludeし、main.py非依存で検証する
(STATE注入はevaluate.configure())。MNIST等の画像classificationは
ネットワーク/ダウンロード依存のため対象外とし、logic系(binary_vector)を中心に
検証する。classificationのpred/margin算出は純関数を直接叩いて検証する。
"""

import copy
import json
import os
import sys

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tools.editor.backend import evaluate as evaluate_api

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'ir_v01')


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, f'{name}.json')) as f:
        return json.load(f)


class FakeState:
    """main.AppState互換の最小状態(main.py非依存)。"""

    def __init__(self):
        self.model = None
        self.ir = None
        self.ir_dirty = False
        self.input_width = None


@pytest.fixture
def ctx(tmp_path):
    state = FakeState()
    app = FastAPI()
    app.include_router(evaluate_api.configure(state, runs_root=tmp_path))
    return state, TestClient(app, base_url='http://localhost:8765'), tmp_path


def _wrong_xor_ir() -> dict:
    """XORのIRを壊してAND(0x8)にする(誤答が出るようにする)。"""
    ir = copy.deepcopy(_load_fixture('xor')['ir'])
    ir['nodes'][0]['params']['truth_tables'] = ['0x8']  # AND
    return ir


class TestEvaluateXor:
    def test_correct_xor_all_correct(self, ctx):
        """正しいXOR IRを評価すると全サンプルcorrect=True、accuracy=1.0。"""
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        resp = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
            'page': 0, 'page_size': 2000,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body['kind'] == 'binary_vector'
        assert body['num_samples'] > 0
        assert body['accuracy'] == 1.0
        assert body['num_errors'] == 0
        assert not body['on_demand']
        for it in body['items']:
            assert it['correct'] is True
            assert 0.0 <= it['soft_confidence'] <= 1.0
            assert 0.0 <= it['soft_margin'] <= 1.0
            assert it['input_bits'] is not None and len(it['input_bits']) == 2
            assert it['label_bits'] is not None and it['pred_bits'] is not None
            assert it['image'] is None

    def test_truth_table_matches(self, ctx):
        """各入力パターンのpred_bitsがXOR真理値表(x0^x1)と一致する。"""
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        body = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
            'page': 0, 'page_size': 2000,
        }).json()
        for it in body['items']:
            x0, x1 = it['input_bits']
            assert it['pred_bits'] == [x0 ^ x1]
            assert it['label_bits'] == [x0 ^ x1]

    def test_wrong_ir_has_errors(self, ctx):
        """AND化した誤りIRでは半数程度が誤答(correct=False)になる。"""
        state, client, _ = ctx
        state.ir = _wrong_xor_ir()
        body = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
            'page': 0, 'page_size': 2000,
        }).json()
        assert body['num_errors'] > 0
        assert body['accuracy'] < 1.0
        assert any(not it['correct'] for it in body['items'])

    def test_paging_consistent(self, ctx):
        """ページングは全件と一貫(index連番・件数整合)。"""
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        page0 = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
            'page': 0, 'page_size': 10,
        }).json()
        assert len(page0['items']) == 10
        assert [it['index'] for it in page0['items']] == list(range(10))
        page1 = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
            'page': 1, 'page_size': 10,
        }).json()
        assert [it['index'] for it in page1['items']] == list(range(10, 20))

    def test_cache_fingerprint_hit(self, ctx):
        """2回目は同一指紋でキャッシュヒット(cached=True)。"""
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        req = {'source': {'kind': 'current'}, 'dataset': 'logic:xor', 'page': 0, 'page_size': 10}
        first = client.post('/api/evaluate', json=req).json()
        assert first['cached'] is False
        second = client.post('/api/evaluate', json=req).json()
        assert second['cached'] is True
        assert second['fingerprint'] == first['fingerprint']

    def test_ir_change_invalidates_cache(self, ctx):
        """IRを変えると指紋が変わりキャッシュミス(再評価)。"""
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        req = {'source': {'kind': 'current'}, 'dataset': 'logic:xor', 'page': 0, 'page_size': 10}
        fp_xor = client.post('/api/evaluate', json=req).json()['fingerprint']
        state.ir = _wrong_xor_ir()
        r2 = client.post('/api/evaluate', json=req).json()
        assert r2['fingerprint'] != fp_xor
        assert r2['cached'] is False


class TestDiscretizationGap:
    def test_mismatch_and_margin_fields_present(self, ctx):
        """soft_margin と hard_soft_mismatch が全行に付く。"""
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        body = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor', 'page': 0, 'page_size': 10,
        }).json()
        for it in body['items']:
            assert isinstance(it['hard_soft_mismatch'], bool)
            assert 'soft_margin' in it


class TestClassificationHelpers:
    """画像classificationはネットワーク非依存で純関数を直接検証する。"""

    def test_classification_item_argmax_and_margin(self):
        hard = np.array([0, 3, 1])          # argmax=1 → pred=1
        soft = torch.tensor([0.1, 2.0, 0.5])  # softmax argmax=1
        item = evaluate_api._classification_item(7, hard, soft, label=1)
        assert item['pred'] == 1
        assert item['correct'] is True
        assert item['hard_soft_mismatch'] is False
        assert 0.0 <= item['soft_confidence'] <= 1.0
        assert item['soft_margin'] >= 0.0

    def test_classification_hard_soft_mismatch(self):
        hard = np.array([5, 0, 0])          # hard argmax=0
        soft = torch.tensor([0.1, 3.0, 0.2])  # soft argmax=1 → mismatch
        item = evaluate_api._classification_item(0, hard, soft, label=0)
        assert item['pred'] == 0
        assert item['correct'] is True
        assert item['hard_soft_mismatch'] is True

    def test_binary_item_confidence_bounds(self):
        hard = np.array([1, 0])
        soft = torch.tensor([0.9, 0.1])
        item = evaluate_api._binary_item(0, hard, soft, label_bits=[1, 0])
        assert item['pred_bits'] == [1, 0]
        assert item['correct'] is True
        assert item['hard_soft_mismatch'] is False
        assert item['soft_confidence'] == pytest.approx(0.8, abs=1e-6)
        assert item['soft_margin'] == pytest.approx(0.8, abs=1e-6)


class TestRunSource:
    def test_run_snapshot_evaluated(self, ctx):
        """source.kind='run': runスナップショットのIRで評価できる。"""
        state, client, tmp_path = ctx
        from tools.editor.backend.runs import save_run

        save_run({'run_id': 'r1', 'ir': _load_fixture('xor')['ir']}, tmp_path)
        body = client.post('/api/evaluate', json={
            'source': {'kind': 'run', 'run_id': 'r1'}, 'dataset': 'logic:xor',
            'page': 0, 'page_size': 2000,
        }).json()
        assert body['source_kind'] == 'run'
        assert body['accuracy'] == 1.0

    def test_run_missing_404(self, ctx):
        state, client, _ = ctx
        resp = client.post('/api/evaluate', json={
            'source': {'kind': 'run', 'run_id': 'nope'}, 'dataset': 'logic:xor',
        })
        assert resp.status_code == 404

    def test_current_without_ir_400(self, ctx):
        state, client, _ = ctx
        resp = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
        })
        assert resp.status_code == 400


class TestCsvExport:
    def test_csv_export(self, ctx):
        """CSVがヘッダ+サンプル毎行で出力される。"""
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        resp = client.post('/api/evaluate/export.csv', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
        })
        assert resp.status_code == 200
        assert resp.headers['content-type'].startswith('text/csv')
        lines = resp.text.strip().splitlines()
        assert lines[0].startswith('index,correct,label,pred')
        # ヘッダ + 全サンプル行
        assert len(lines) > 1


class TestSchemaValidation:
    def test_extra_forbidden(self, ctx):
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        resp = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor', 'bogus': 1,
        })
        assert resp.status_code == 422

    def test_unknown_dataset_400(self, ctx):
        state, client, _ = ctx
        state.ir = _load_fixture('xor')['ir']
        resp = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'does_not_exist',
        })
        assert resp.status_code == 400
