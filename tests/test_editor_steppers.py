"""Stepper抽象(BP/AP/BB)とAP法delta捕捉のテスト。

BPStepper抽出のリファクタ安全性は既存スイート
(test_editor_ws/step/restore)が無修正グリーンであることで担保する。
"""

import time

import pytest
import torch

from griffith_logic import GroupSum, LutLayer
from tools.editor.backend.monitor import decode_u8
from tools.editor.backend.steppers import (
    APStepper,
    BBStepper,
    BPStepper,
    make_stepper,
)
from tools.editor.backend.training import TrainingSession

IN_DIM = 32
BATCH = 8


def _model(parameterization: str = 'softmax') -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        LutLayer(IN_DIM, 40, n_inputs=2, connections='random',
                 parameterization=parameterization),
        GroupSum(k=10, tau=1.0),
    )


def _batch():
    torch.manual_seed(1)
    return torch.rand(BATCH, IN_DIM), torch.randint(0, 10, (BATCH,))


def _loader_factory(in_dim: int = IN_DIM):
    x = torch.rand(64 * BATCH, in_dim)
    y = torch.randint(0, 10, (64 * BATCH,))
    ds = torch.utils.data.TensorDataset(x, y)

    def factory():
        return torch.utils.data.DataLoader(ds, batch_size=BATCH, drop_last=True)

    return factory


def _wait(cond, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


class TestMakeStepper:
    def test_bp_default(self):
        model = _model()
        lut_layers = [('n1_lutlayer', model[1])]
        s = make_stepper('bp', model, 0.01, lut_layers)
        assert isinstance(s, BPStepper) and s.method == 'bp'
        assert s.optimizer is not None

    def test_unknown_method_raises(self):
        model = _model()
        with pytest.raises(ValueError):
            make_stepper('dfa', model, 0.01, [])

    def test_bp_step_updates_weights(self):
        model = _model()
        s = make_stepper('bp', model, 0.05, [('n1_lutlayer', model[1])])
        before = model[1].weights.detach().clone()
        x, y = _batch()
        loss, acc = s.step(x, y)
        assert isinstance(loss, float) and 0.0 <= acc <= 1.0
        assert not torch.equal(model[1].weights.detach(), before)


class TestBBStepper:
    def test_raw_weights_clamped(self):
        model = _model('raw')
        s = make_stepper('bb', model, 1.0, [('n1_lutlayer', model[1])])
        assert isinstance(s, BBStepper) and s.method == 'bb'
        x, y = _batch()
        for _ in range(5):
            s.step(x, y)
        w = model[1].weights.detach()
        assert w.min() >= 0.0 and w.max() <= 1.0


class TestAPStepper:
    def test_delta_captured(self):
        """AP法の更新量delta(.grad非経由)が層別に捕捉される。"""
        model = _model('independent')  # AP法はindependent必須
        lut_layers = [('n1_lutlayer', model[1])]
        s = make_stepper('ap', model, 0.1, lut_layers)
        assert isinstance(s, APStepper) and s.optimizer is None

        x, y = _batch()
        loss, acc = s.step(x, y)
        assert isinstance(loss, float) and 0.0 <= acc <= 1.0

        deltas = s.last_deltas_by_name
        assert set(deltas) == {'n1_lutlayer'}
        delta = deltas['n1_lutlayer']
        assert delta.shape == (40, 4)  # independent: (out_dim, 2^N)
        assert float(delta.abs().sum()) > 0.0  # 実際に更新されている

    def test_ap_requires_independent(self):
        model = _model('softmax')
        with pytest.raises(ValueError):
            make_stepper('ap', model, 0.1, [('n1_lutlayer', model[1])])

    def test_ap_delta_frame_roundtrip(self):
        from tools.editor.backend.monitor import build_ap_delta

        model = _model('independent')
        s = make_stepper('ap', model, 0.1, [('n1_lutlayer', model[1])])
        x, y = _batch()
        s.step(x, y)
        delta = s.last_deltas_by_name['n1_lutlayer']
        frame = build_ap_delta('n1_lutlayer', delta, 3)
        assert frame['type'] == 'ap_delta'
        assert frame['layer'] == 'n1_lutlayer'
        assert frame['delta_norm'] >= 0.0
        assert decode_u8(frame['delta']).shape == (40, 4)


class TestSessionWithMethod:
    def test_ap_session_snapshots_delta(self, tmp_path):
        """method='ap'のTrainingSessionでスナップショットにdeltaが載る。"""
        session = TrainingSession(
            _model('independent'),
            lr=0.1,
            batch_size=BATCH,
            epochs=100,
            method='ap',
            loader_factory=_loader_factory(),
            history_root=tmp_path,
        )
        try:
            session.start()
            assert _wait(lambda: session.step >= 3)
            session.pause()
            assert _wait(lambda: session.state == 'paused')

            assert session.status()['method'] == 'ap'
            snap = session.history.get(session.step)
            assert snap is not None and snap.method == 'ap'
            rec = snap.layers['n1_lutlayer']
            assert rec.delta is not None
            assert float(rec.delta.abs().sum()) > 0.0
            # AP法は.gradが無い(gradは保存されない)
            assert rec.grad is None
        finally:
            session.stop()
            session._thread.join(timeout=5)
            session.history.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
