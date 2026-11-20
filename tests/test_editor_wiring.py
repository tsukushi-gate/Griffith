"""エディタ対応のバックエンド検証。

- POST /api/wiring/generate が core.connections.make_connections と同じ indices と
  被覆統計を返し、不正な引数は 422
- 本ブランチが足した IR 形(thermometer の 8 入力 LUT、stride 割当の恒等 / permute、
  group_sum の bias、draft の構造化配線)を PUT → build できて、返る IR がノード型を保つ
- コード生成が FixedLutLayer / Permute / GroupSum(bias) / DWNLayer(LTT)を出し、
  exec → export_ir → simulate が元の IR と一致する
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from griffith_logic import GroupSum, LTTLayer, LutLayer
from griffith_logic.core.connections import (
    connection_stats,
    get_butterfly_connections,
    get_rot_connections,
)
from griffith_logic.encoding.thermometer import ThermometerEncoder
from griffith_logic.export import export_ir
from griffith_logic.ir import ir_from_dict, ir_to_dict, simulate
from tools.editor.backend.codegen import generate_code
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager


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


def _exec_build_model(code: str):
    ns = {}
    exec(compile(code, '<codegen>', 'exec'), ns)
    model = ns['build_model']()
    model.eval()
    return model


def _bits(batch, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(batch, width)).astype(np.int64)


class TestWiringGenerate:
    def test_butterfly_matches_generator_and_reports_stats(self, client):
        res = client.post('/api/wiring/generate', json={
            'kind': 'butterfly', 'in_width': 64, 'out_width': 64, 'n_inputs': 2,
            'spec': {'stride': 4}, 'dedupe': False,
        })
        assert res.status_code == 200, res.text
        body = res.json()
        want = get_butterfly_connections(64, 64, 2, 4)
        assert body['indices'] == want.tolist()
        assert body['stats'] == connection_stats(want, 64)
        assert body['stats']['coverage'] == 1.0

    def test_rot_step_and_dedupe(self, client):
        res = client.post('/api/wiring/generate', json={
            'kind': 'rot', 'in_width': 12, 'out_width': 12, 'n_inputs': 2,
            'spec': {'s_a': 0, 's_b': 12, 'step': 2}, 'dedupe': True,
        })
        assert res.status_code == 200, res.text
        idx = np.asarray(res.json()['indices'])
        assert not (idx[0] == idx[1]).any()
        assert res.json()['stats']['same_input'] == 0
        raw = get_rot_connections(12, 12, 0, 12, 'cpu', step=2)
        assert (raw[0] == raw[1]).all()  # dedupe 前は s_b ≡ s_a (mod 12) で全ゲートが重複
        assert idx[0].tolist() == raw[0].tolist()  # dedupe はスロット 1 だけを +1 ずらす
        assert idx[1].tolist() == ((raw[0] + 1) % 12).tolist()

    @pytest.mark.parametrize('body,needle', [
        ({'kind': 'spiral', 'spec': {}}, '未知の配線種'),
        ({'kind': 'rot_n', 'spec': {'step': 2}}, 'shifts'),
        ({'kind': 'rot_n', 'spec': {'shifts': [0, 1, 2]}}, '一致しません'),
        ({'kind': 'butterfly', 'spec': {'stride': 1, 'grid': [2, 2]}}, '受け付けません'),
    ])
    def test_invalid_requests_are_422(self, client, body, needle):
        res = client.post('/api/wiring/generate', json={
            'in_width': 16, 'out_width': 8, 'n_inputs': 2, **body,
        })
        assert res.status_code == 422, res.text
        assert needle in res.json()['detail']


def _put_and_build(client, doc):
    put = client.put('/api/models/current/ir', json={'ir': doc})
    assert put.status_code == 200, put.text
    built = client.post('/api/models/current/build')
    assert built.status_code == 200, built.text
    return built.json()['ir']


class TestBuildNewIrForms:
    def test_thermometer_lut8_builds_and_keeps_node(self, client):
        torch.manual_seed(0)
        model = nn.Sequential(
            ThermometerEncoder(3, input_bits=8), LutLayer(12, 8, n_inputs=2), GroupSum(k=2),
        ).eval()
        doc = ir_to_dict(export_ir(model, input_width=32))
        ir = _put_and_build(client, doc)
        assert [n['type'] for n in ir['nodes']] == ['lut_layer', 'lut_layer', 'group_sum']
        assert ir['nodes'][0]['params']['n_inputs'] == 8
        assert ir['nodes'][0]['train_info']['kind'] == 'thermometer'
        x = _bits(16, 32)
        assert np.array_equal(simulate(ir_from_dict(ir), x), simulate(ir_from_dict(doc), x))

    def test_stride_permute_builds_and_keeps_node(self, client):
        torch.manual_seed(1)
        model = nn.Sequential(LutLayer(16, 12, n_inputs=2), GroupSum(k=4, assign='stride')).eval()
        doc = ir_to_dict(export_ir(model, input_width=16, native_conv=True))
        ir = _put_and_build(client, doc)
        assert [n['type'] for n in ir['nodes']] == ['lut_layer', 'permute', 'group_sum']
        x = _bits(16, 16)
        assert np.array_equal(simulate(ir_from_dict(ir), x), simulate(ir_from_dict(doc), x))

    def test_group_sum_bias_survives_build(self, client):
        torch.manual_seed(2)
        model = nn.Sequential(LutLayer(8, 8, n_inputs=2), GroupSum(k=2, bias=[0, 1])).eval()
        doc = ir_to_dict(export_ir(model, input_width=8))
        ir = _put_and_build(client, doc)
        assert ir['nodes'][-1]['params']['bias'] == [0, 1]

    def test_draft_structured_wiring_is_generated_on_build(self, client):
        doc = {
            'format': 'magicdelta-ir', 'version': '0.2.0', 'metadata': {'name': 'draft'},
            'inputs': [{'id': 'x', 'width': 64}], 'outputs': [{'id': 'y', 'width': 4}],
            'nodes': [
                {
                    'id': 'u1', 'type': 'lut_layer',
                    'params': {
                        'in_width': 64, 'out_width': 64, 'n_inputs': 2,
                        'indices': [[(o * 2 + j) % 64 for o in range(64)] for j in range(2)],
                        'truth_tables': ['0xa'] * 64,
                    },
                    'train_info': {
                        'parameterization': 'softmax', 'connections': 'butterfly',
                        'wiring': {'stride': 4}, 'selection': 'soft', 'initialized': False,
                    },
                },
                {'id': 'u2', 'type': 'group_sum',
                 'params': {'in_width': 64, 'k': 4, 'tau': 1.0}},
            ],
            'edges': [{'src': 'x', 'dst': 'u1'}, {'src': 'u1', 'dst': 'u2'},
                      {'src': 'u2', 'dst': 'y'}],
        }
        ir = _put_and_build(client, doc)
        lut = ir['nodes'][0]
        assert lut['params']['indices'] == get_butterfly_connections(64, 64, 2, 4).tolist()
        assert 'initialized' not in lut['train_info']


class TestCodegenNewNodes:
    @staticmethod
    def _roundtrip(ir, input_width, batch=32):
        code = generate_code(ir)
        model = _exec_build_model(code)
        ir2 = export_ir(model, input_width=input_width, native_conv=True)
        x = _bits(batch, input_width)
        assert np.array_equal(simulate(ir2, x), simulate(ir, x))
        assert [n.type for n in ir2.nodes] == [n.type for n in ir.nodes]
        return code

    def test_fixed_lut_thermometer(self):
        torch.manual_seed(3)
        model = nn.Sequential(
            ThermometerEncoder(3, input_bits=8), LutLayer(12, 8, n_inputs=2), GroupSum(k=2),
        ).eval()
        ir = export_ir(model, input_width=32)
        code = self._roundtrip(ir, 32)
        assert 'FixedLutLayer(32, 12, 8' in code
        assert "'kind': 'thermometer'" in code

    def test_permute_and_bias(self):
        torch.manual_seed(4)
        model = nn.Sequential(
            LutLayer(16, 12, n_inputs=2), GroupSum(k=4, assign='stride', bias=[1, 0, -1, 0]),
        ).eval()
        ir = export_ir(model, input_width=16, native_conv=True)
        code = self._roundtrip(ir, 16)
        assert 'Permute([3, 4], [1, 0])' in code
        assert 'bias=[1, 0, -1, 0]' in code

    def test_identity_lut_non_native(self):
        torch.manual_seed(5)
        model = nn.Sequential(LutLayer(16, 12, n_inputs=2), GroupSum(k=4, assign='stride')).eval()
        ir = export_ir(model, input_width=16)
        code = generate_code(ir)
        model2 = _exec_build_model(code)
        x = _bits(32, 16)
        assert np.array_equal(simulate(export_ir(model2, input_width=16), x), simulate(ir, x))
        assert 'FixedLutLayer(12, 12, 1' in code

    def test_ltt_as_dwn_layer(self):
        torch.manual_seed(6)
        model = nn.Sequential(LTTLayer(16, 8, n_inputs=4, connections='random'), GroupSum(k=2)).eval()
        ir = export_ir(model, input_width=16)
        code = generate_code(ir)
        assert 'DWNLayer(16, 8, n_inputs=4' in code
        assert 'restore_dwn_layer' in code
        model2 = _exec_build_model(code)
        x = _bits(32, 16)
        assert np.array_equal(simulate(export_ir(model2, input_width=16), x), simulate(ir, x))
