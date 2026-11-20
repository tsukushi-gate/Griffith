"""L5: ハイブリッドモデルの勾配フロー検証。

ToLogic/FromLogic境界で勾配が死なないことを直接検証する。
LogicLinearをnn.Linearで挟んだ「サンドイッチ」で、前段Linearまで
勾配が届けばハイブリッド構成は学習可能。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import FromLogic, LogicLinear, LogicMLP, LogicSpec, ToLogic


def _assert_all_grads_nonzero(model, exclude_substr=()):
    dead = []
    for name, p in model.named_parameters():
        if p.grad is None:
            dead.append(f'{name}: grad=None')
        elif p.grad.abs().sum().item() == 0.0:
            if not any(s in name for s in exclude_substr):
                dead.append(f'{name}: grad全ゼロ')
    assert not dead, '勾配が届いていないパラメータ:\n' + '\n'.join(dead)


class TestSandwichGradientFlow:
    @pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp', 'warp'])
    def test_linear_logic_linear(self, param, seed):
        """Linear→LogicLinear→Linear で全パラメータに勾配が流れる"""
        spec = LogicSpec(parameterization=param, depth=2, neurons_per_output=4)
        model = nn.Sequential(
            nn.Linear(12, 16),
            LogicLinear(16, 8, spec=spec),
            nn.Linear(8, 3),
        )
        x = torch.randn(5, 12)
        loss = model(x).square().mean()
        loss.backward()
        _assert_all_grads_nonzero(model)

    def test_3d_input(self, seed):
        """(B, S, D)入力のflatten処理と形状復元"""
        layer = LogicLinear(16, 8, spec=LogicSpec(neurons_per_output=4))
        x = torch.randn(3, 7, 16, requires_grad=True)
        y = layer(x)
        assert y.shape == (3, 7, 8)
        y.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_residual_connection(self, seed):
        """residual接続(x + LogicMLP(x))でスケールが暴れず勾配も流れる"""
        mlp = LogicMLP(16, spec=LogicSpec(neurons_per_output=8))
        x = torch.randn(64, 16)
        y = x + mlp(x)
        # groupsum_centeredにより出力はゼロ中心・O(1)スケール
        assert y.std().item() < 5.0
        y.square().mean().backward()
        _assert_all_grads_nonzero(mlp)

    def test_encoder_tau_is_learnable(self, seed):
        """ToLogicの温度τが学習可能パラメータとして勾配を持つ"""
        layer = LogicLinear(8, 4)
        x = torch.randn(4, 8)
        layer(x).sum().backward()
        assert layer.encode.log_tau.grad is not None
        assert layer.encode.log_tau.grad.abs().item() > 0


class TestBridgeContracts:
    def test_to_logic_range(self, seed):
        """ToLogic出力は常に[0,1](LUT層の入力契約)"""
        for mode in ('sigmoid', 'binarize_ste'):
            bridge = ToLogic(mode)
            x = torch.randn(100, 8) * 10
            y = bridge(x)
            assert y.min() >= 0.0 and y.max() <= 1.0, mode

    def test_from_logic_centered(self, seed):
        """groupsum_centeredは一様ランダム二値入力でゼロ中心・O(1)"""
        bridge = FromLogic(64, 4, mode='groupsum_centered')
        x = torch.randint(0, 2, (1000, 64)).float()
        y = bridge(x)
        assert y.mean().abs().item() < 0.1
        assert 0.1 < y.std().item() < 2.0

    def test_from_logic_shape(self):
        bridge = FromLogic(24, 6)
        y = bridge(torch.rand(2, 5, 24))
        assert y.shape == (2, 5, 6)

    def test_from_logic_rejects_bad_width(self):
        with pytest.raises(AssertionError):
            FromLogic(10, 3)  # 10 % 3 != 0


class TestTrainEvalGap:
    def test_hybrid_train_eval_consistency(self, seed):
        """飽和前の初期状態でもtrain/evalの出力が発散していない(回帰監視)"""
        model = nn.Sequential(
            nn.Linear(12, 16),
            LogicLinear(16, 8, spec=LogicSpec(neurons_per_output=8)),
        )
        x = torch.randn(32, 12)
        model.train()
        y_train = model(x)
        model.eval()
        y_eval = model(x)
        # 初期状態では乖離するが、桁が合っていることを監視
        gap = (y_train - y_eval).abs().mean().item()
        assert gap < 5.0, f'train-eval乖離が異常: {gap}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
