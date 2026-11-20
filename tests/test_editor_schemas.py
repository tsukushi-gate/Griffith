"""エディタAPI契約の整合テスト。

実際のWSフレーム/RESTレスポンス(TestClient経由)が
tools/editor/backend/schemas.py のpydanticモデルでvalidateできることを検証する。
モデルは原則extra='forbid'のため、実装がキーを増減/型変更した場合に
ここで落ちる(=スキーマと実装のドリフト検出)。

MNIST非依存: 学習・評価とも合成データへ差し替えて実走する
(tests/test_editor_step.py と同じパターン)。
"""

import json
import os
import time
from contextlib import contextmanager

import pytest
import torch
from fastapi.testclient import TestClient

from tools.editor.backend import schemas
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager, TrainingSession


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    """共有STATEの初期化+データ/永続化先の差し替え。"""
    STATE.model = None
    STATE.ir = None
    STATE.ir_dirty = False
    STATE.input_width = None
    STATE.sessions = SessionManager()

    # 追加分のテストが実outputs/を汚さないよう、プロジェクト永続化先をtmpへ差し替える
    import tools.editor.backend.projects_router as projects_router_mod
    monkeypatch.setattr(projects_router_mod, 'PROJECTS_ROOT', tmp_path / 'editor_projects')

    # 学習データ: 合成MNIST形状(モデル入力784に合わせ(1,28,28))
    x = torch.rand(512, 1, 28, 28)
    y = torch.randint(0, 10, (512,))
    ds = torch.utils.data.TensorDataset(x, y)

    def fake_loader(self):
        return torch.utils.data.DataLoader(ds, batch_size=32, drop_last=True)

    # hard eval用も合成データ(CIでのMNISTダウンロードを避ける)
    eval_ds = torch.utils.data.TensorDataset(
        torch.rand(64, 1, 28, 28), torch.randint(0, 10, (64,))
    )

    def fake_eval_loader(self):
        return torch.utils.data.DataLoader(eval_ds, batch_size=32)

    monkeypatch.setattr(TrainingSession, '_default_loader', fake_loader)
    monkeypatch.setattr(TrainingSession, '_default_eval_loader', fake_eval_loader)
    import tools.editor.backend.training as training_mod
    monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path / 'history')
    monkeypatch.setattr(training_mod, 'RUNS_ROOT', tmp_path / 'runs')
    yield


@pytest.fixture
def client():
    return TestClient(app, base_url='http://localhost:8765')


def _wait(cond, timeout: float = 15.0, interval: float = 0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def _state_of(client: TestClient, sid: str) -> str:
    return client.get(f'/api/train/{sid}/status').json()['state']


@contextmanager
def _session(client: TestClient, **overrides):
    """デモモデル生成→訓練開始→(テスト本体)→停止のヘルパ。"""
    demo = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
    assert demo.status_code == 200
    body = {'dataset': 'mnist', 'lr': 0.01, 'batch_size': 32, 'epochs': 100,
            **overrides}
    started = client.post('/api/train/start', json=body)
    assert started.status_code == 200
    sid = started.json()['session_id']
    try:
        yield sid
    finally:
        client.post(f'/api/train/{sid}/stop')
        assert _wait(lambda: _state_of(client, sid) in
                     ('stopped', 'finished', 'error'))


class TestRestResponseSchemas:
    """主要RESTレスポンスのスキーマ整合。"""

    def test_datasets_catalog(self, client):
        items = client.get('/api/datasets').json()
        assert len(items) >= 5  # mnist + 論理データセット群
        for item in items:
            schemas.DatasetInfo.model_validate(item)

    def test_hierarchy_and_lowering_responses(self, client):
        imported = client.post('/api/models/import/python', json={
            'path': 'tools/editor/samples/llvit_ivitt_cifar10.py',
        })
        assert imported.status_code == 200, imported.json()

        hierarchy = client.get('/api/models/current/hierarchy').json()
        parsed_hierarchy = schemas.HierarchyResponse.model_validate(hierarchy)
        assert parsed_hierarchy.paper_ref is not None

        lower = client.get('/api/models/current/lower', params={
            'node_id': 'encoder_0.channel_mixer.lut0',
            'level': 'L0',
        }).json()
        parsed_lower = schemas.LoweringView.model_validate(lower)
        assert parsed_lower.virtual_ir_node is not None

    def test_train_status_running_and_paused(self, client):
        with _session(client) as sid:
            assert _wait(
                lambda: client.get(f'/api/train/{sid}/status').json()['step'] >= 1
            )
            st = client.get(f'/api/train/{sid}/status').json()
            parsed = schemas.TrainStatus.model_validate(st)
            assert parsed.capture is not None
            assert parsed.history is not None
            assert parsed.total_steps is not None

            # pause/step/resumeのレスポンスもTrainStatusと同形
            paused = client.post(f'/api/train/{sid}/pause').json()
            schemas.TrainStatus.model_validate(paused)

    def test_history_index_and_step_view(self, client):
        with _session(client) as sid:
            assert _wait(
                lambda: client.get(f'/api/train/{sid}/status').json()['step'] >= 3
            )
            client.post(f'/api/train/{sid}/pause')
            assert _wait(lambda: _state_of(client, sid) == 'paused')

            idx = client.get(f'/api/train/{sid}/history').json()
            parsed_idx = schemas.HistoryIndex.model_validate(idx)
            assert parsed_idx.capture is not None  # RESTは必ずcaptureを載せる

            view = client.get(
                f'/api/train/{sid}/history/{idx["max_step"]}',
                params={'fields': 'metrics,summary,gate_distribution,trace'},
            ).json()
            parsed_view = schemas.HistoryStepView.model_validate(view)
            assert 'n1_lutlayer' in parsed_view.layers
            layer = parsed_view.layers['n1_lutlayer']
            assert layer.gate_ids is not None
            assert layer.gate_distribution is not None
            # k=400はfullキャプチャなのでbackpropトレースも載る
            assert parsed_view.trace is not None

    def test_runs_summary(self, client):
        with _session(client) as sid:
            assert _wait(
                lambda: client.get(f'/api/train/{sid}/status').json()['step'] >= 1
            )
        # runレコードは訓練スレッドのfinallyで書かれるため出現を待つ
        assert _wait(
            lambda: len(client.get('/api/runs').json()['runs']) >= 1
        )
        runs = client.get('/api/runs').json()['runs']
        for run in runs:
            parsed = schemas.RunSummary.model_validate(run)
            assert parsed.run_id == sid


_IR_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'ir_v01')


def _load_ir_fixture(name: str) -> dict:
    with open(os.path.join(_IR_FIXTURE_DIR, f'{name}.json'), encoding='utf-8') as f:
        return json.load(f)['ir']


class TestNSeriesApiSchemas:
    """N系列の主要レスポンスのスキーマ整合。"""

    def test_stats_response(self, client):
        demo = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
        assert demo.status_code == 200
        body = client.get('/api/stats').json()
        parsed = schemas.StatsResponse.model_validate(body)
        assert parsed.total_luts > 0

    def test_dataset_samples_response(self, client):
        body = client.get('/api/datasets/logic:xor/samples').json()
        parsed = schemas.DatasetSamplesResponse.model_validate(body)
        assert parsed.num_data == 4
        for item in parsed.samples:
            schemas.DatasetSampleItem.model_validate(item.model_dump())

    def test_encoding_preview_response(self, client):
        body = client.post('/api/datasets/encoding-preview', json={
            'values': [0.1, 0.6, 0.9], 'mode': 'thermometer', 'bits': 3,
        }).json()
        parsed = schemas.EncodingPreviewResponse.model_validate(body)
        assert parsed.bits_per_value == 3

    def test_project_crud_response(self, client):
        created = client.post('/api/projects', json={'name': 'n10-2テスト'})
        assert created.status_code == 200, created.text
        parsed = schemas.ProjectSchema.model_validate(created.json())

        fetched = client.get(f'/api/projects/{parsed.id}').json()
        schemas.ProjectSchema.model_validate(fetched)

        updated = client.put(f'/api/projects/{parsed.id}', json={
            'description': '更新後の説明',
        }).json()
        parsed_updated = schemas.ProjectSchema.model_validate(updated)
        assert parsed_updated.description == '更新後の説明'

    def test_config_schema_default_matches_trainconfig_model(self, client):
        """GET /api/config/schemaはTrainConfig.model_json_schema()そのものであり、
        TrainConfig自体をALL_MODELSへ登録していることのドリフト検出。"""
        from griffith_logic.training.config import TrainConfig as RealTrainConfig

        schema = client.get('/api/config/schema').json()
        assert schema == RealTrainConfig.model_json_schema()
        # デフォルトインスタンスがschemas.TrainConfigとしてもvalidateできる
        schemas.TrainConfig.model_validate(RealTrainConfig().model_dump())

    def test_evaluate_response(self, client):
        STATE.ir = _load_ir_fixture('xor')
        body = client.post('/api/evaluate', json={
            'source': {'kind': 'current'}, 'dataset': 'logic:xor',
            'page': 0, 'page_size': 10,
        }).json()
        parsed = schemas.EvaluateResponse.model_validate(body)
        assert parsed.kind == 'binary_vector'
        assert len(parsed.items) > 0
        for item in parsed.items:
            schemas.EvalItem.model_validate(item.model_dump())

    def test_network_transfer_response(self, client):
        source_ir = _load_ir_fixture('xor')
        target_ir = _load_ir_fixture('xor')
        body = client.post('/api/networks/transfer', json={
            'source_ir': source_ir, 'target_ir': target_ir,
        }).json()
        parsed = schemas.TransferResponse.model_validate(body)
        assert parsed.transferred

    def test_nas_status_and_trials_response(self, client):
        status = client.get('/api/nas/status').json()
        schemas.NasStatusResponse.model_validate(status)
        trials = client.get('/api/nas/trials').json()
        schemas.NasTrialsResponse.model_validate(trials)


class TestWsFrameSchemas:
    """WSフレームのスキーマ整合(subscribe v2で全detailを購読して収集する)。"""

    FRAME_MODELS = {
        'capture_config': schemas.CaptureConfigFrame,
        'metrics': schemas.MetricsFrame,
        'layer_summary': schemas.LayerSummaryFrame,
        'gate_distribution': schemas.GateDistributionFrame,
        'activations': schemas.ActivationsFrame,
        'grad_flow': schemas.GradFlowFrame,
        'history_index': schemas.HistoryIndexFrame,
        'session_end': schemas.SessionEndFrame,
    }

    def test_all_ws_frame_types_validate(self, client):
        wanted = set(self.FRAME_MODELS) - {'session_end'}
        collected: dict[str, list[dict]] = {}

        def receive_until(ws, deadline, done):
            while time.time() < deadline and not done():
                frame = json.loads(ws.receive_text())
                collected.setdefault(frame['type'], []).append(frame)

        with _session(client) as sid:
            with client.websocket_connect(f'/ws/train/{sid}') as ws:
                ws.send_text(json.dumps({
                    'type': 'subscribe',
                    'layers': ['n1_lutlayer'],
                    'detail': {
                        'gate_distribution': ['n1_lutlayer'],
                        'activations': True,
                        'grad_flow': True,
                    },
                    'detail_interval_ms': 100,
                }))
                deadline = time.time() + 30.0
                receive_until(ws, deadline, lambda: wanted <= set(collected))
                # 停止するとキューを掃き切った後session_endが届く
                client.post(f'/api/train/{sid}/stop')
                receive_until(ws, deadline, lambda: 'session_end' in collected)

        missing = set(self.FRAME_MODELS) - set(collected)
        assert not missing, f'受信できなかったフレームtype: {missing}'

        n_validated = 0
        for ftype, frames in collected.items():
            model = self.FRAME_MODELS.get(ftype)
            # スキーマ未定義のフレームtypeが増えた場合もドリフトとして検出する
            assert model is not None, f'スキーマ未定義のフレームtype: {ftype}'
            for frame in frames:
                model.model_validate(frame)
                n_validated += 1
        assert n_validated >= len(self.FRAME_MODELS)

        # 個別の要点: capture_config / gate_distribution の中身
        cc = schemas.CaptureConfigFrame.model_validate(
            collected['capture_config'][0]
        )
        assert cc.level in ('full', 'weights', 'metrics')
        gd = schemas.GateDistributionFrame.model_validate(
            collected['gate_distribution'][0]
        )
        assert (gd.mode == 'dense') == (gd.probs is not None)


class TestBuilderFrameSchemas:
    """WS実走で出しにくいフレームはビルダー関数(実配信経路と同一)の出力を検証する。"""

    def test_ap_delta_frame(self):
        # AP法セッションの実走は重いため、_update_detailsが使う
        # monitor.build_ap_delta の出力を直接validateする
        from tools.editor.backend.monitor import build_ap_delta

        frame = build_ap_delta('n1_lutlayer', torch.randn(40, 16), step=7)
        parsed = schemas.ApDeltaFrame.model_validate(frame)
        assert parsed.delta is not None
        assert parsed.delta.encoding == 'u8'

    def test_u8_envelope_roundtrip_shape(self):
        from tools.editor.backend.monitor import encode_u8

        env = encode_u8(torch.rand(8, 16), scale=1.0)
        parsed = schemas.U8Envelope.model_validate(env)
        assert parsed.shape == [8, 16]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
