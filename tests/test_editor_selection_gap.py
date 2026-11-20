"""選択ギャップ・ダッシュボード(§4-1)のテスト。

- gate_entropy: カテゴリカルエントロピーの数式性質(一様=log(n)、one-hot=0)
- neuron_uncertainty: パラメータ化別の正規化迷い[0,1]
  (softmax=カテゴリカル/ln(K)、independent系=二値エントロピー/ln(2)、warp=None)
- layer_summary: レイヤ別の迷い統計(entropy_mean/entropy_max)が載る
- hard/soft eval: metricsフレームに val_acc_soft / selection_gap がスパースに載る
- uncertain_gates: 低確信ゲートTop-Nがentropy降順で取れ、スキーマ整合する
"""

import math
import time

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from griffith_logic.ir import build_torch, ir_from_dict
from tools.editor.backend import schemas
from tools.editor.backend.extract import (
    gate_entropy,
    layer_entropy,
    layer_summary,
    lut_layers_with_ids,
    neuron_uncertainty,
)
from tools.editor.backend.main import STATE, app
from tools.editor.backend.training import SessionManager, TrainingSession


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


def _build_xor_model() -> nn.Module:
    graph = ir_from_dict(_xor_ir())
    core = build_torch(graph)
    model = nn.Sequential(nn.Flatten(), *core)
    model.train()
    return model


def _wait_state(session: TrainingSession, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if session.state in ('finished', 'stopped', 'error'):
            return session.state
        time.sleep(0.05)
    raise TimeoutError(f'セッションが{timeout}s以内に終了しなかった: {session.state}')


def _drain_metrics(session: TrainingSession) -> list[dict]:
    frames = []
    while not session.metrics_queue.empty():
        frames.append(session.metrics_queue.get_nowait())
    return frames


class TestGateEntropy:
    def test_uniform_distribution_reaches_log_n(self):
        """一様分布のエントロピーは log(n)(natural log)。"""
        probs = torch.full((3, 4), 0.25)
        ent = gate_entropy(probs)
        assert ent.shape == (3,)
        assert torch.allclose(ent, torch.full((3,), math.log(4)), atol=1e-5)

    def test_one_hot_is_zero(self):
        """確信の高い(one-hot)分布のエントロピーは0。"""
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        ent = gate_entropy(probs)
        assert torch.allclose(ent, torch.zeros(2), atol=1e-5)

    def test_uniform_greater_than_peaked(self):
        """一様分布 > ピーク分布(迷いの大小関係)。"""
        uniform = gate_entropy(torch.full((1, 4), 0.25))
        peaked = gate_entropy(torch.tensor([[0.7, 0.1, 0.1, 0.1]]))
        assert float(uniform[0]) > float(peaked[0])


class TestNeuronUncertainty:
    def test_softmax_uniform_is_one(self):
        """softmax: 一様分布(logits全て等しい)の正規化迷いは1。"""
        w = torch.zeros(3, 16)  # 16ゲートモード相当
        u = neuron_uncertainty(w, 'softmax', 2)
        assert u is not None
        assert torch.allclose(u, torch.ones(3), atol=1e-5)

    def test_softmax_peaked_near_zero(self):
        """softmax: 1ゲートに集中したlogitsの正規化迷いは0に近い。"""
        w = torch.full((2, 16), -50.0)
        w[:, 3] = 50.0
        u = neuron_uncertainty(w, 'softmax', 2)
        assert u is not None
        assert torch.all(u < 1e-3)

    def test_independent_half_is_one(self):
        """independent: 全エントリp=0.5(logits=0)の正規化迷いは1。

        独立ベルヌーイ列なのでカテゴリカル式ではなく二値エントロピーで
        正規化されることを固定する(p=0.5で最大=ln2)。
        """
        w = torch.zeros(3, 4)  # sigmoid(0)=0.5
        u = neuron_uncertainty(w, 'independent', 2)
        assert u is not None
        assert torch.allclose(u, torch.ones(3), atol=1e-5)

    def test_independent_saturated_near_zero(self):
        """independent: 飽和したlogits(確定した真理値表)の迷いは0に近い。"""
        w = torch.full((2, 4), 50.0)
        w[1] = -50.0
        u = neuron_uncertainty(w, 'independent', 2)
        assert u is not None
        assert torch.all(u < 1e-3)

    def test_range_is_normalized(self):
        """任意のlogitsで[0,1]に収まる(層間比較の前提)。"""
        gen = torch.Generator().manual_seed(0)
        for param, cols in (('softmax', 16), ('independent', 4), ('iwp', 2)):
            w = torch.randn(8, cols, generator=gen) * 5
            u = neuron_uncertainty(w, param, 2)
            assert u is not None, param
            assert torch.all(u >= 0.0) and torch.all(u <= 1.0 + 1e-6), param

    def test_warp_is_none(self):
        """warp: 確率解釈がないためNone(layer_summaryはNone伝播)。"""
        assert neuron_uncertainty(torch.randn(2, 4), 'warp', 2) is None


class TestLayerSummaryEntropy:
    def test_summary_carries_entropy_stats(self):
        """layer_summaryにentropy_mean/entropy_maxが載り、schemaでvalidateできる。"""
        model = _build_xor_model()
        layers = lut_layers_with_ids(model)
        assert layers, 'LutLayerが見つからない'
        name, layer = layers[0]
        frame = layer_summary(name, layer, step=3)
        parsed = schemas.LayerSummaryFrame.model_validate(frame)
        assert parsed.entropy_mean is not None
        assert parsed.entropy_max is not None
        # 平均 <= 最大、いずれも非負
        assert 0.0 <= parsed.entropy_mean <= parsed.entropy_max + 1e-6

    def test_layer_entropy_matches_summary_mean(self):
        """layer_entropyのニューロン毎平均がsummaryのentropy_meanと一致する。"""
        model = _build_xor_model()
        name, layer = lut_layers_with_ids(model)[0]
        ent = layer_entropy(layer)
        assert ent is not None
        frame = layer_summary(name, layer, step=0)
        assert frame['entropy_mean'] == round(float(ent.mean().item()), 4)
        assert frame['entropy_max'] == round(float(ent.max().item()), 4)


class TestSelectionGapEval:
    def test_metrics_carry_soft_and_gap(self, tmp_path):
        """hard eval直後のフレームに val_acc_soft / selection_gap が載る。"""
        session = TrainingSession(
            _build_xor_model(), dataset='logic:xor', lr=0.05,
            batch_size=32, epochs=2, loss='bce',
            history_root=tmp_path / 'h', runs_root=tmp_path / 'r',
        )
        session.start()
        _wait_state(session)

        frames = [f for f in _drain_metrics(session) if f['type'] == 'metrics']
        val_frames = [f for f in frames if 'val_acc' in f]
        assert val_frames, 'val付きフレームが無い'
        gap_frames = [f for f in val_frames if 'selection_gap' in f]
        assert gap_frames, 'selection_gap付きフレームが無い'
        for f in gap_frames:
            assert 'val_acc_soft' in f
            # gap = soft - hard の恒等式(丸め誤差許容)
            assert f['selection_gap'] == pytest.approx(
                round(f['val_acc_soft'] - f['val_acc'], 6), abs=1e-6
            )
            schemas.MetricsFrame.model_validate(f)
        assert session.last_val_acc_soft is not None
        assert session.last_selection_gap is not None
        session.history.close()


class TestUncertainGatesEndpoint:
    @pytest.fixture(autouse=True)
    def reset_state(self, tmp_path, monkeypatch):
        STATE.model = None
        STATE.ir = None
        STATE.ir_dirty = False
        STATE.input_width = None
        STATE.sessions = SessionManager()
        import tools.editor.backend.training as training_mod
        monkeypatch.setattr(training_mod, 'HISTORY_ROOT', tmp_path / 'hist')
        monkeypatch.setattr(training_mod, 'RUNS_ROOT', tmp_path / 'runs')
        yield

    def test_method_returns_sorted_topn(self, tmp_path):
        session = TrainingSession(
            _build_xor_model(), dataset='logic:xor', lr=0.05,
            batch_size=32, epochs=1, loss='bce',
            history_root=tmp_path / 'h', runs_root=tmp_path / 'r',
        )
        gates = session.uncertain_gates(top_n=3)
        assert len(gates) <= 3
        assert gates, '低確信ゲートが1件も返らない'
        # entropy降順
        ents = [g['entropy'] for g in gates]
        assert ents == sorted(ents, reverse=True)
        for g in gates:
            assert g['layer_id'] == 'n1_lutlayer'
            assert g['neuron_index'] >= 0
        session.history.close()

    def test_endpoint_schema(self):
        client = TestClient(app, base_url='http://localhost:8765')
        put = client.put('/api/models/current/ir', json={'ir': _xor_ir()})
        assert put.status_code == 200
        started = client.post('/api/train/start', json={
            'dataset': 'logic:xor', 'lr': 0.05, 'batch_size': 32, 'epochs': 1000,
            'loss': 'bce',
        })
        assert started.status_code == 200
        sid = started.json()['session_id']
        try:
            res = client.get(f'/api/train/{sid}/uncertain_gates', params={'top_n': 5})
            assert res.status_code == 200
            parsed = schemas.UncertainGatesResponse.model_validate(res.json())
            assert len(parsed.gates) <= 5
        finally:
            client.post(f'/api/train/{sid}/stop')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
