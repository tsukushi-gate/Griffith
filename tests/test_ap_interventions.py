"""experiments/ap_interventions.py の介入トレーナー変種のテスト。

高速・CPU・slowマーカーなし。コア実装は変更せず、既存 APTrainer/CNNAPTrainer を
継承したアダプタ(単調射影・DFA-lite符号)を検証する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from ap_interventions import (
    DFAliteAPTrainer,
    MonotoneAPTrainer,
    isotonic_project_lut_,
)
from monotonicity import collect_lut_layers, monotonicity_violation_V

from griffith_logic import GroupSum, LutLayer

# e=Σx_j2^j 順の代表的な真理値表(n_inputs=2)
TABLE_AND = torch.tensor([0.0, 0.0, 0.0, 1.0])   # x0 AND x1（単調増加）
TABLE_NOT0 = torch.tensor([1.0, 0.0, 1.0, 0.0])  # NOT x0（単調減少）


def _make_model(in_dim, out_dim, num_classes, n_layers):
    """Flatten + LutLayer×n_layers + GroupSum の independent モデル。"""
    layers = [torch.nn.Flatten()]
    layers.append(LutLayer(in_dim, out_dim, n_inputs=2, parameterization='independent'))
    for _ in range(n_layers - 1):
        layers.append(
            LutLayer(out_dim, out_dim, n_inputs=2, parameterization='independent')
        )
    layers.append(GroupSum(num_classes, tau=1.0))
    return torch.nn.Sequential(*layers)


def _table_layer(table):
    """(1, 4) の table を持つ n_inputs=2 の LutLayer を作り V 計測に使う。"""
    layer = LutLayer(in_dim=2, out_dim=1, n_inputs=2, parameterization='independent')
    layer.weights.data.copy_(table.unsqueeze(0))
    return layer


# ============================================================
# 1. isotonic_project_lut_ : 減少テーブルの単調化 / 増加テーブルの不変
# ============================================================

def test_isotonic_project_decreasing_to_monotone():
    """単調減少(NOT)テーブルを射影すると V≈0 になる。"""
    w = TABLE_NOT0.clone().unsqueeze(0)  # (1, 4)
    isotonic_project_lut_(w, n_inputs=2, iters=3)

    layer = _table_layer(w[0])
    v = monotonicity_violation_V(layer, ap_mode=True)
    assert v == pytest.approx(0.0, abs=1e-6)


def test_isotonic_project_increasing_unchanged():
    """既に単調増加(AND)なテーブルは射影で不変。"""
    w = TABLE_AND.clone().unsqueeze(0)
    before = w.clone()
    isotonic_project_lut_(w, n_inputs=2, iters=3)
    assert torch.allclose(w, before, atol=1e-6)


def test_isotonic_project_multineuron_exact():
    """複数ニューロン(N>1)・n_inputs=2 はベクトル化射影で厳密に V=0 になる。

    n=2 は dim0→dim1 の1巡で単調錐へ収まる(全ペア制約が満たされる)。
    """
    torch.manual_seed(0)
    w = torch.rand(8, 4)  # (N=8, 2^2)
    isotonic_project_lut_(w, n_inputs=2, iters=1)

    layer = LutLayer(in_dim=2, out_dim=8, n_inputs=2, parameterization='independent')
    layer.weights.data.copy_(w)
    v = monotonicity_violation_V(layer, ap_mode=True)
    assert v == pytest.approx(0.0, abs=1e-6)


def test_isotonic_project_higher_arity_reduces_violation():
    """n_inputs=4 でも反復スイープで単調度違反 V が十分小さくなる。"""
    torch.manual_seed(0)
    w = torch.rand(4, 16)  # (N=4, 2^4)
    layer = LutLayer(in_dim=4, out_dim=4, n_inputs=4, parameterization='independent')
    layer.weights.data.copy_(w)
    v_before = monotonicity_violation_V(layer, ap_mode=True)

    isotonic_project_lut_(w, n_inputs=4, iters=50)
    layer.weights.data.copy_(w)
    v_after = monotonicity_violation_V(layer, ap_mode=True)

    assert v_before > 0.0
    assert v_after < 1e-4


# ============================================================
# 2. MonotoneAPTrainer : 各stepの射影後に全層 V≈0
# ============================================================

def test_monotone_ap_trainer_keeps_monotone():
    torch.manual_seed(0)
    model = _make_model(in_dim=8, out_dim=20, num_classes=2, n_layers=2)
    trainer = MonotoneAPTrainer(model, lr=0.1, update_mode=1)

    x = torch.rand(16, 8)
    y = torch.randint(0, 2, (16,))
    for _ in range(5):
        trainer.train_step(x, y)

    for layer in collect_lut_layers(model):
        v = monotonicity_violation_V(layer, ap_mode=True)
        assert v == pytest.approx(0.0, abs=1e-6)


# ============================================================
# 3. DFAliteAPTrainer : 層固定±1符号が一定
# ============================================================

def test_dfalite_signs_fixed_and_pm1():
    torch.manual_seed(0)
    model = _make_model(in_dim=8, out_dim=12, num_classes=2, n_layers=2)
    trainer = DFAliteAPTrainer(model, lr=0.1, dfa_seed=0)

    before = [s.clone() for s in trainer._dfa_signs]
    # 値が ±1 のみ
    for s in trainer._dfa_signs:
        assert set(s.unique().tolist()) <= {-1.0, 1.0}

    x = torch.rand(16, 8)
    y = torch.randint(0, 2, (16,))
    trainer.train_step(x, y)
    trainer.train_step(x, y)

    # 2回 train_step しても符号は不変
    for s0, s1 in zip(before, trainer._dfa_signs):
        assert torch.equal(s0, s1)
