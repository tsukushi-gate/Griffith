"""Python由来モデルimportのバックエンドテスト。"""

import textwrap
import time

import pytest
from fastapi.testclient import TestClient

from tools.editor.backend.main import STATE, app
from tools.editor.backend.python_importer import (
    hierarchy_from_ir,
    load_python_editor_model,
    lowering_view_from_ir,
    resolve_python_path,
)
from tools.editor.backend.training import SessionManager


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    STATE.model = None
    STATE.ir = None
    STATE.ir_dirty = False
    STATE.input_width = None
    STATE.sessions = SessionManager()
    import tools.editor.backend.training as training_mod

    monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path / 'history')
    monkeypatch.setattr(training_mod, 'RUNS_ROOT', tmp_path / 'runs')
    yield


def _write(path, body: str):
    path.write_text(textwrap.dedent(body), encoding='utf-8')
    return path


def test_decorator_model_import(tmp_path):
    py = _write(tmp_path / 'decorated.py', """
        import torch.nn as nn
        from griffith_logic import editor_model

        @editor_model(
            id='decorated',
            name='Decorated',
            input_spec={'width': 4},
            output_spec={'width': 2, 'semantics': 'class_scores'},
            datasets=['toy'],
        )
        def make_model():
            return nn.Sequential(nn.Linear(4, 2))
    """)

    result = load_python_editor_model(str(py), workspace_root=tmp_path)
    assert result.input_width == 4
    assert result.ir['metadata']['editor']['model_ref']['spec']['id'] == 'decorated'
    assert result.ir['nodes'][0]['type'] == 'blackbox'
    assert result.ir['nodes'][0]['meta']['capabilities']['python_executable'] is True


def test_fallback_build_model_and_explicit_symbol(tmp_path):
    py = _write(tmp_path / 'fallback.py', """
        import torch.nn as nn

        INPUT_WIDTH = 3
        OUTPUT_WIDTH = 1

        def other():
            return nn.Sequential(nn.Linear(3, 2))

        def build_model():
            return nn.Sequential(nn.Linear(3, 1))
    """)

    fallback = load_python_editor_model(str(py), workspace_root=tmp_path)
    assert fallback.input_width == 3
    assert fallback.ir['outputs'][0]['width'] == 1

    explicit = load_python_editor_model(
        str(py), symbol='other', workspace_root=tmp_path
    )
    assert explicit.ir['outputs'][0]['width'] == 2


def test_workspace_outside_path_rejected(tmp_path):
    outside = tmp_path.parent / 'outside_model.py'
    outside.write_text('def build_model(): pass', encoding='utf-8')
    with pytest.raises(ValueError, match='workspace外'):
        resolve_python_path(str(outside), workspace_root=tmp_path / 'workspace')


def test_endpoint_imports_llvit_sample_and_preserves_model_on_put():
    client = TestClient(app, base_url='http://localhost:8765')
    res = client.post('/api/models/import/python', json={
        'path': 'tools/editor/samples/llvit_tiny.py',
    })
    assert res.status_code == 200, res.json()
    body = res.json()
    assert body['input_width'] == 32
    assert body['ir']['version'] == '0.3.0'
    assert body['ir']['metadata']['editor']['model_ref']['kind'] == 'python'
    assert STATE.model is not None

    before = STATE.model
    put = client.put('/api/models/current/ir', json={'ir': body['ir']})
    assert put.status_code == 200
    assert STATE.model is before
    assert STATE.ir_dirty is False

    codegen = client.post('/api/models/current/codegen')
    assert codegen.status_code == 200
    assert 'def build_model()' in codegen.json()['code']


def test_llvit_ivitt_paper_config_and_lowering():
    result = load_python_editor_model('tools/editor/samples/llvit_ivitt_cifar10.py')
    spec = result.spec
    assert result.input_width == 3 * 224 * 224
    assert spec['id'] == 'llvit_ivitt_cifar10'
    cfg = spec['paper_ref']['config']
    assert cfg['dim'] == 192
    assert cfg['encoders'] == 12
    assert cfg['patch_size'] == 16
    assert cfg['token_count'] == 197
    assert cfg['channel_mixer_luts'] == [768, 192]
    assert cfg['thermometer_bits'] == 8
    assert cfg['lut_inputs'] == 6
    assert cfg['lut_mapping'] == 'learnable'
    assert cfg['lut_efd'] == 'official'
    assert cfg['encoded_value'] == 'int4'

    hierarchy = hierarchy_from_ir(result.ir)
    ids = {n['id'] for n in hierarchy['nodes']}
    assert 'encoder_0.channel_mixer' in ids
    assert 'encoder_0.token_mixer' in ids
    assert 'encoder_0.channel_mixer.lut0' in ids

    token_mixer = next(n for n in hierarchy['nodes'] if n['id'] == 'encoder_0.token_mixer')
    assert token_mixer['capabilities']['python_executable'] is True
    assert token_mixer['capabilities']['gate_expandable'] is False

    lower = lowering_view_from_ir(
        result.ir, 'encoder_0.channel_mixer.lut0', level='L0'
    )
    assert lower['virtual_ir_node']['type'] == 'lut_layer'
    assert lower['virtual_ir_node']['params']['in_width'] == 1536
    assert lower['virtual_ir_node']['params']['out_width'] == 768
    assert lower['summary']['n_inputs'] == 6
    assert len(lower['summary']['gate_ids']) == 768


def test_hierarchy_and_lowering_endpoints_for_llvit():
    client = TestClient(app, base_url='http://localhost:8765')
    res = client.post('/api/models/import/python', json={
        'path': 'tools/editor/samples/llvit_ivitt_cifar10.py',
    })
    assert res.status_code == 200, res.json()

    hierarchy = client.get('/api/models/current/hierarchy')
    assert hierarchy.status_code == 200, hierarchy.json()
    assert hierarchy.json()['paper_ref']['baseline'] == 'I-ViT-T'
    assert any(
        n['id'] == 'encoder_0.channel_mixer.lut1'
        for n in hierarchy.json()['nodes']
    )

    lower = client.get('/api/models/current/lower', params={
        'node_id': 'encoder_0.channel_mixer.lut1',
        'level': 'L0',
    })
    assert lower.status_code == 200, lower.json()
    body = lower.json()
    assert body['virtual_ir_node']['params']['out_width'] == 192
    assert body['summary']['gate_ids']


def test_python_model_put_structural_change_invalidates_loaded_model():
    client = TestClient(app, base_url='http://localhost:8765')
    res = client.post('/api/models/import/python', json={
        'path': 'tools/editor/samples/llvit_tiny.py',
    })
    assert res.status_code == 200, res.json()
    ir = res.json()['ir']
    assert STATE.model is not None

    edited = {
        **ir,
        'nodes': [
            *ir['nodes'],
            {
                'id': 'unused_constant',
                'type': 'constant',
                'params': {'width': 1, 'bits': [0]},
                'train_info': {},
                'meta': {},
            },
        ],
    }
    put = client.put('/api/models/current/ir', json={'ir': edited})
    assert put.status_code == 200, put.json()
    assert STATE.model is None
    assert STATE.ir_dirty is True


def test_llvit_python_model_can_start_training():
    client = TestClient(app, base_url='http://localhost:8765')
    res = client.post('/api/models/import/python', json={
        'path': 'tools/editor/samples/llvit_tiny.py',
    })
    assert res.status_code == 200, res.json()

    started = client.post('/api/train/start', json={
        'dataset': 'llvit_synthetic_debug',
        'lr': 0.001,
        'logic_lr': 0.02,
        'ann_lr': 0.001,
        'batch_size': 32,
        'epochs': 1,
        'loss': 'ce',
    })
    assert started.status_code == 200, started.json()
    sid = started.json()['session_id']

    deadline = time.time() + 20
    status = None
    while time.time() < deadline:
        status = client.get(f'/api/train/{sid}/status').json()
        if status['state'] == 'error' or status['step'] >= 2:
            break
        time.sleep(0.05)
    client.post(f'/api/train/{sid}/stop')

    assert status is not None
    assert status['state'] != 'error', status
    assert status['step'] >= 1


def test_llvit_structure_preview_rejects_local_training():
    client = TestClient(app, base_url='http://localhost:8765')
    res = client.post('/api/models/import/python', json={
        'path': 'tools/editor/samples/llvit_ivitt_cifar10.py',
    })
    assert res.status_code == 200, res.json()

    started = client.post('/api/train/start', json={
        'dataset': 'llvit:cifar10',
        'lr': 0.0001,
        'logic_lr': 0.02,
        'ann_lr': 0.0001,
        'batch_size': 1,
        'epochs': 1,
        'loss': 'ce',
    })
    assert started.status_code == 400
    assert 'experiments/train_llvit.py' in started.json()['detail']
