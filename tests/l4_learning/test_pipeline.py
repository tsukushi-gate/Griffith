"""convert/pipeline.pyの統合テスト。

小さな合成データ+ミニMLPで convert_pretrained_mlp の全phaseが走ることを固定する:
  - pruning→quantizing→lowering→verifying のphase遷移(progressコールバック)
  - metricsに各段acc・最終hard_acc・回路統計が載る
  - 厳密性(量子化torch参照⇔simulate 全件argmax一致)が verifying phase内でassert済み
    =例外なく返れば厳密lowering成立

数秒規模に抑えるため spec は極小(fanins=(4,), epochs=1, 少データ)。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from griffith_logic.convert.pipeline import (
    PIPELINE_PHASES,
    ConvertPipelineSpec,
    convert_pretrained_mlp,
)
from griffith_logic.ir import IRGraph

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


class _MiniMLP(nn.Module):
    """8-6-3 の素朴なReLU-MLP(nn.Linear列。pipeline側が発火判定として吸収する)。"""

    def __init__(self, in_dim: int = 8, hidden: int = 6, n_classes: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _synthetic_loaders(in_dim=8, n_classes=3, n=96, batch=16, seed=0):
    """0/1特徴量+線形分離可能なラベルの合成データ(train/eval同分布)。"""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, in_dim)) > 0.5).astype(np.int64)
    # 決定的なラベル(特徴の重み付き和のargmax)。学習可能な信号を持たせる。
    w = rng.standard_normal((in_dim, n_classes))
    y = (x.astype(np.float64) @ w).argmax(axis=1).astype(np.int64)
    ds = TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.int64))
    train = DataLoader(ds, batch_size=batch, shuffle=True)
    ev = DataLoader(ds, batch_size=batch, shuffle=False)
    return train, ev


def _tiny_spec(**overrides) -> ConvertPipelineSpec:
    base = dict(fanins=(4,), epochs_per_stage=1, lr=1e-2,
                weight_bits_hidden=4, weight_bits_final=8,
                qat_epochs=0, gate_acc=0.0, batch_size=16, seed=0)
    base.update(overrides)
    return ConvertPipelineSpec(**base)


class TestConvertPipeline:
    def test_full_pipeline_runs_and_reports(self):
        model = _MiniMLP()
        train, ev = _synthetic_loaders()
        ir, metrics = convert_pretrained_mlp(model, train, ev, _tiny_spec())

        # 検証済みIRが返る(verifying phase内でvalidate+厳密性assert済み)
        assert isinstance(ir, IRGraph)
        assert metrics['exact'] is True
        assert isinstance(metrics['hard_acc'], float)
        assert 0.0 <= metrics['hard_acc'] <= 1.0
        # 各段のacc・回路統計がmetricsに載る
        assert len(metrics['pruning_stage_acc']) == 1
        assert metrics['quant_acc'] is not None
        assert metrics['total_lut_outputs'] > 0
        assert metrics['n_lut_layers'] > 0
        assert metrics['final_max_fanin'] <= 4

    def test_progress_phase_transitions(self):
        model = _MiniMLP()
        train, ev = _synthetic_loaders()
        seen: list[tuple[str, float]] = []

        def progress(phase: str, prog: float, m: dict) -> None:
            seen.append((phase, prog))
            assert 0.0 <= prog <= 1.0
            assert isinstance(m, dict)

        convert_pretrained_mlp(model, train, ev, _tiny_spec(), progress=progress)

        phases_in_order = [p for p, _ in seen]
        # 4 phase全てが少なくとも1回出現し、規定順で初出する
        first_idx = {}
        for i, p in enumerate(phases_in_order):
            first_idx.setdefault(p, i)
        assert set(PIPELINE_PHASES).issubset(first_idx.keys())
        ordered = sorted(PIPELINE_PHASES, key=lambda p: first_idx[p])
        assert tuple(ordered) == PIPELINE_PHASES
        # 各phaseで最終的にprogress=1.0に到達する
        for phase in PIPELINE_PHASES:
            assert any(p == phase and prog == pytest.approx(1.0)
                       for p, prog in seen), f'{phase}が1.0で終わっていない'

    def test_qat_fires_when_below_gate(self):
        """gate_accを高く設定するとQATが発火し、metricsに記録される。"""
        model = _MiniMLP()
        train, ev = _synthetic_loaders()
        # gate_acc=1.1(絶対に下回る)+ qat_epochs=1 でQAT発火を強制
        ir, metrics = convert_pretrained_mlp(
            model, train, ev,
            _tiny_spec(gate_acc=1.1, qat_epochs=1))
        assert metrics['qat_ran'] is True
        assert metrics['qat_acc'] is not None
        assert metrics['exact'] is True  # QAT後も厳密lowering成立

    def test_binary_input_required(self):
        """0/1でない入力は明確なValueError。"""
        model = _MiniMLP()
        x = torch.randn(16, 8)  # 実数(0/1でない)
        y = torch.randint(0, 3, (16,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=8)
        with pytest.raises(ValueError, match='0/1'):
            convert_pretrained_mlp(model, loader, loader, _tiny_spec())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
