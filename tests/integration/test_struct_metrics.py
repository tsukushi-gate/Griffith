"""学習中構造メトリクス(SM4: StructMetricsTracker / selection_gap / runnerフック)。

- Tracker単体: flip率の手計算一致・定数/パススルー率・初回はflipキー無し
- selection_gap_eval: soft/hard 2パスの定義(gap = soft − hard)
- runnerスモーク: xor学習後の result['struct_metrics'] に構造レポートが載る
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic import LutLayer
from griffith_logic.analysis.structure import (
    StructMetricsTracker,
    dwn_uncertainty,
    selection_gap_eval,
    soft_eval_acc,
    struct_final_report,
    wiring_indices,
)
from griffith_logic.layers.dwn import DWNLayer
from griffith_logic.training.config import load_config
from griffith_logic.training.runner import run_training


def _one_hot_gates(layer: LutLayer, gate_ids: list[int]) -> None:
    """16ゲートモードのweightsを指定ゲートIDのone-hotに設定する。"""
    with torch.no_grad():
        layer.weights.data.zero_()
        for o, g in enumerate(gate_ids):
            layer.weights.data[o, g] = 10.0


class TestStructMetricsTracker:
    def _model(self):
        torch.manual_seed(0)
        return nn.Sequential(
            LutLayer(4, 8, n_inputs=2),
            LutLayer(8, 4, n_inputs=2),
        )

    def test_first_collect_has_no_flip_key(self):
        tracker = StructMetricsTracker(self._model())
        out = tracker.collect()
        assert 'struct/flip_rate' not in out
        assert 'struct/entropy_mean' in out
        assert 'struct/const_rate' in out
        assert 'struct/passthrough_rate' in out

    def test_flip_rate_hand_computed(self):
        model = self._model()
        _one_hot_gates(model[0], [1] * 8)   # 全部AND
        _one_hot_gates(model[1], [7] * 4)   # 全部OR
        tracker = StructMetricsTracker(model)
        tracker.collect()

        # 層0の8ゲート中3個だけXORへ変更、層1は不変
        _one_hot_gates(model[0], [6, 6, 6, 1, 1, 1, 1, 1])
        out = tracker.collect()
        layer0_id = tracker.layers[0][0]
        layer1_id = tracker.layers[1][0]
        assert out[f'struct/flip_rate/{layer0_id}'] == pytest.approx(3 / 8)
        assert out[f'struct/flip_rate/{layer1_id}'] == 0.0
        assert out['struct/flip_rate'] == pytest.approx(3 / 12)

    def test_const_and_passthrough_rates(self):
        model = self._model()
        _one_hot_gates(model[0], [0] * 8)  # FALSE(定数)
        _one_hot_gates(model[1], [3] * 4)  # A(パススルー)
        tracker = StructMetricsTracker(model)
        out = tracker.collect()
        assert out['struct/const_rate'] == pytest.approx(8 / 12)
        assert out['struct/passthrough_rate'] == pytest.approx(4 / 12)

    def test_gate_histograms(self):
        model = self._model()
        _one_hot_gates(model[0], [1, 1, 6, 6, 6, 7, 7, 7])
        tracker = StructMetricsTracker(model)
        hist = tracker.gate_histograms()
        layer0_id = tracker.layers[0][0]
        assert hist[layer0_id] == {1: 2, 6: 3, 7: 3}


class TestSelectionGapEval:
    def test_gap_definition(self):
        torch.manual_seed(1)
        model = nn.Sequential(LutLayer(4, 4, n_inputs=2))
        x = torch.randint(0, 2, (32, 4)).float()
        y = torch.randint(0, 2, (32, 4)).float()
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        def metric_fn(out, target):
            return float(((out > 0.5).float() == target).float().mean())

        result = selection_gap_eval(model, loader, metric_fn)
        assert set(result) == {'hard_acc', 'soft_acc', 'selection_gap'}
        assert result['selection_gap'] == pytest.approx(
            result['soft_acc'] - result['hard_acc']
        )
        assert result['soft_acc'] == pytest.approx(
            soft_eval_acc(model, loader, metric_fn)
        )

    def test_model_mode_restored(self):
        model = nn.Sequential(LutLayer(2, 2, n_inputs=2))
        x = torch.randint(0, 2, (8, 2)).float()
        loader = DataLoader(TensorDataset(x, x), batch_size=4)
        model.eval()
        selection_gap_eval(model, loader, lambda o, t: 0.0)
        assert not model.training
        model.train()
        selection_gap_eval(model, loader, lambda o, t: 0.0)
        assert model.training


class TestDwnTracker:
    """SM10: DWNLayerの計測対応(ttビットflip率・uncertainty・const率)。"""

    def _model(self, mapping='random'):
        torch.manual_seed(0)
        # n_inputs=3 → 8エントリ×4ノード = 32決定ビット
        return nn.Sequential(DWNLayer(8, 4, n_inputs=3, mapping=mapping))

    def test_first_collect_has_metrics_but_no_flip(self):
        tracker = StructMetricsTracker(self._model())
        out = tracker.collect()
        assert 'struct/flip_rate' not in out
        assert 'struct/entropy_mean' in out
        assert 'struct/const_rate' in out

    def test_bit_flip_rate_hand_computed(self):
        model = self._model()
        layer = model[0]
        with torch.no_grad():
            layer.luts.fill_(1.0)  # 全ビット1
        tracker = StructMetricsTracker(model)
        tracker.collect()
        with torch.no_grad():
            layer.luts.view(-1)[:5] = -1.0  # 5ビットだけ0へ反転
        out = tracker.collect()
        node_id = tracker.layers[0][0]
        assert out[f'struct/flip_rate/{node_id}'] == pytest.approx(5 / 32)
        assert out['struct/flip_rate'] == pytest.approx(5 / 32)

    def test_uncertainty_hand_computed(self):
        model = self._model()
        with torch.no_grad():
            model[0].luts.fill_(0.5)  # |0.5| → 迷い 0.5
        unc = dwn_uncertainty(model[0])
        np.testing.assert_allclose(unc.numpy(), 0.5)
        out = StructMetricsTracker(model).collect()
        assert out['struct/entropy_mean'] == pytest.approx(0.5)

    def test_all_negative_tables_are_const(self):
        model = self._model()
        with torch.no_grad():
            model[0].luts.fill_(-1.0)  # 全ノードtt=0(FALSE)
        out = StructMetricsTracker(model).collect()
        assert out['struct/const_rate'] == pytest.approx(1.0)

    def test_mixed_model_weighted_global(self):
        # 決定単位: LutLayer=4ニューロン + DWN=2ノード×8bit=16 → 計20
        torch.manual_seed(0)
        model = nn.Sequential(
            LutLayer(4, 4, n_inputs=2),
            DWNLayer(4, 2, n_inputs=3),
        )
        _one_hot_gates(model[0], [1] * 4)
        with torch.no_grad():
            model[1].luts.fill_(1.0)
        tracker = StructMetricsTracker(model)
        tracker.collect()
        _one_hot_gates(model[0], [6, 6, 1, 1])  # lut側2ニューロン変更
        with torch.no_grad():
            model[1].luts.view(-1)[:4] = -1.0   # dwn側4ビット変更
        out = tracker.collect()
        assert out['struct/flip_rate'] == pytest.approx((2 + 4) / 20)


class TestWiringFlip:
    """SM10: 学習配線(LearnableMapping)のargmax変化率。"""

    def _set_wiring(self, lm, targets):
        with torch.no_grad():
            lm.weights.zero_()
            for col, row in enumerate(targets):
                lm.weights[row, col] = 10.0

    def test_dwn_learnable_mapping_flip(self):
        torch.manual_seed(0)
        model = nn.Sequential(DWNLayer(6, 2, n_inputs=3, mapping='learnable'))
        lm = model[0].learnable_mapping  # weights: (6, 2*3=6)
        self._set_wiring(lm, [0, 1, 2, 3, 4, 5])
        tracker = StructMetricsTracker(model)
        out = tracker.collect()
        assert 'struct/wiring_flip_rate' not in out  # 初回は比較対象なし
        self._set_wiring(lm, [1, 0, 2, 3, 4, 5])  # 2スロット変更
        out = tracker.collect()
        node_id = tracker.layers[0][0]
        assert out[f'struct/wiring_flip_rate/{node_id}'] == pytest.approx(2 / 6)
        assert out['struct/wiring_flip_rate'] == pytest.approx(2 / 6)

    def test_lut_learnable_wiring_flip(self):
        torch.manual_seed(0)
        model = nn.Sequential(LutLayer(4, 3, n_inputs=2, wiring='learnable'))
        lm = model[0].learnable_mapping  # weights: (4, 3*2=6)
        assert wiring_indices(model[0]) is not None
        self._set_wiring(lm, [0, 1, 2, 3, 0, 1])
        tracker = StructMetricsTracker(model)
        tracker.collect()
        self._set_wiring(lm, [3, 1, 2, 3, 0, 1])  # 1スロット変更
        out = tracker.collect()
        assert out['struct/wiring_flip_rate'] == pytest.approx(1 / 6)

    def test_fixed_wiring_has_no_key(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            LutLayer(4, 4, n_inputs=2),
            DWNLayer(4, 2, n_inputs=3, mapping='random'),
        )
        assert wiring_indices(model[0]) is None
        tracker = StructMetricsTracker(model)
        tracker.collect()
        out = tracker.collect()
        assert 'struct/wiring_flip_rate' not in out


class TestDwnFinalReport:
    def test_dwn_model_final_report(self):
        torch.manual_seed(0)
        model = nn.Sequential(DWNLayer(8, 4, n_inputs=3, mapping='random'))
        sm = struct_final_report(model, 8)
        assert sm['struct/gates_raw'] == 4
        assert 'struct/sweep_ratio' in sm
        assert sm['struct/logic_depth_max'] >= 1


class TestRunnerHook:
    def test_result_contains_struct_metrics(self, tmp_path):
        """runnerスモーク(wandb無効): run末の構造レポートがresultに載る。"""
        cfg = load_config(None, [
            'data.name=logic:xor',
            'data.batch_size=32',
            'model.kind=logic_mlp',
            'model.k=8',
            'model.layers=2',
            'optim.lr=0.05',
            'run.epochs=2',
            'run.seed=0',
            'run.device=cpu',
        ])
        result = run_training(cfg, out_dir=str(tmp_path / 'run'))
        sm = result['struct_metrics']
        assert sm['struct/gates_raw'] > 0
        assert 0.0 <= sm['struct/sweep_ratio'] <= 1.0
        assert sm['struct/sweep_gates'] <= sm['struct/gates_raw']
        assert sm['struct/logic_depth_max'] >= 1
        # packed LUT系はlut_k未指定のためキーごと存在しない
        assert 'struct/packed_luts' not in sm
