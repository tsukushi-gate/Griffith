"""L1: NaN/inf数値安定性の系統テスト。

対象レイヤと検証ケース:
    LutLayer(softmax/independent/iwp/raw/warp)
        (a) 重みに±30の大振幅logits(softmax/sigmoidの飽和域)
        (b) 入力0/1境界値(LUT入力の契約域[0,1]の端点)
        (d) backward後の全grad(入力・重み)有限性
    GroupSum
        (c) 温度tau極小(1e-3)/極大(1e3) + backward
    Binarize / ThermometerEncoder
        実数契約のレイヤ: ±1e30級の極端入力 + backward

方針:
    - LutLayerの入力契約は[0,1]域なので入力はクランプ域のみ。
      極端値は重み側(±30 logits)を主対象とする
    - 検証は conftest.assert_all_finite で統一
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest用

from conftest import assert_all_finite

from griffith_logic.binarize import Binarize
from griffith_logic.encoding import ThermometerEncoder
from griffith_logic.group_sum import GroupSum
from griffith_logic.lut_layer import LutLayer

LUT_PARAMS = ['softmax', 'independent', 'iwp', 'raw', 'warp']
NS = [2, 4, 6]

WEIGHT_AMPLITUDE = 30.0   # softmax/sigmoidが完全飽和するロジット振幅
EXTREME_INPUT = 1e30      # 実数契約レイヤ(Binarize/Thermometer)向けの極端入力


def _boundary_inputs(in_dim: int, batch: int = 8) -> torch.Tensor:
    """[0,1]契約域の入力。0/1境界値の行を明示的に含む"""
    x = torch.rand(batch, in_dim)
    x[0] = 0.0                    # 全ゼロ境界
    x[1] = 1.0                    # 全イチ境界
    x[2, 0::2] = 0.0              # 境界の混在
    x[2, 1::2] = 1.0
    return x


def _saturate_weights(layer: LutLayer) -> None:
    """重みへ±30の大振幅logitsをランダム符号で書き込む"""
    with torch.no_grad():
        layer.weights.copy_(
            torch.sign(torch.randn_like(layer.weights)) * WEIGHT_AMPLITUDE
        )


def _extreme_real_inputs(batch: int = 4, dim: int = 8) -> torch.Tensor:
    """実数契約レイヤ向け: ±1e30・境界0/1・閾値近傍を含む入力"""
    x = torch.randn(batch, dim)
    x[0, 0] = EXTREME_INPUT
    x[0, 1] = -EXTREME_INPUT
    x[1] = 0.0
    x[2] = 1.0
    x[3] = 0.5  # Binarize閾値ちょうど
    return x


class TestLutLayerStability:
    """(a)大振幅重み × (b)境界入力 × (d)grad有限性"""

    @pytest.mark.parametrize('n', NS)
    @pytest.mark.parametrize('param', LUT_PARAMS)
    def test_train_forward_backward_finite(self, param, n, seed):
        """train forward + backward: 大振幅重み・境界入力で全値/全gradが有限"""
        layer = LutLayer(max(8, n), 6, n_inputs=n, parameterization=param)
        _saturate_weights(layer)
        layer.train()
        x = _boundary_inputs(layer.in_dim).requires_grad_(True)

        y = layer(x)
        assert_all_finite(y, f'LutLayer({param}, n={n}) train forward')

        y.sum().backward()
        assert_all_finite(x.grad, f'LutLayer({param}, n={n}) input grad')
        assert_all_finite(layer.weights.grad, f'LutLayer({param}, n={n}) weight grad')

    @pytest.mark.parametrize('n', NS)
    @pytest.mark.parametrize('param', LUT_PARAMS)
    def test_eval_forward_finite(self, param, n, seed):
        """eval forward(離散化パス): 大振幅重み・境界入力で有限"""
        layer = LutLayer(max(8, n), 6, n_inputs=n, parameterization=param)
        _saturate_weights(layer)
        layer.eval()
        y = layer(_boundary_inputs(layer.in_dim))
        assert_all_finite(y, f'LutLayer({param}, n={n}) eval forward')

    @pytest.mark.parametrize('param', ['softmax', 'warp'])
    def test_hard_st_selection_finite(self, param, seed):
        """hard_st選択(CAGE方式)でも大振幅重みで有限(softmax/warpのみ対応)"""
        layer = LutLayer(8, 6, n_inputs=2, parameterization=param,
                         selection='hard_st')
        _saturate_weights(layer)
        layer.train()
        x = _boundary_inputs(layer.in_dim).requires_grad_(True)
        y = layer(x)
        assert_all_finite(y, f'LutLayer({param}, hard_st) train forward')
        y.sum().backward()
        assert_all_finite(x.grad, f'LutLayer({param}, hard_st) input grad')
        assert_all_finite(layer.weights.grad, f'LutLayer({param}, hard_st) weight grad')


class TestGroupSumStability:
    """(c)温度tau極小/極大 + (d)grad有限性"""

    @pytest.mark.parametrize('tau', [1e-3, 1.0, 1e3])
    def test_extreme_tau_forward_backward_finite(self, tau, seed):
        """tau=1e-3(×1000スケール)/1e3(×0.001スケール)で出力とgradが有限"""
        gs = GroupSum(k=4, tau=tau)
        x = _boundary_inputs(32).requires_grad_(True)
        y = gs(x)
        assert_all_finite(y, f'GroupSum(tau={tau}) forward')
        y.sum().backward()
        assert_all_finite(x.grad, f'GroupSum(tau={tau}) input grad')


class TestBinarizeStability:
    """実数契約: 極端入力(±1e30)・閾値境界 + STE backward"""

    def test_train_forward_backward_finite(self, seed):
        layer = Binarize()
        layer.train()
        x = _extreme_real_inputs().requires_grad_(True)
        y = layer(x)
        assert_all_finite(y, 'Binarize train forward')
        y.sum().backward()
        assert_all_finite(x.grad, 'Binarize input grad')

    def test_eval_forward_finite(self, seed):
        layer = Binarize()
        layer.eval()
        y = layer(_extreme_real_inputs())
        assert_all_finite(y, 'Binarize eval forward')


class TestThermometerStability:
    """実数契約: 極端入力・退化データfit + STE backward"""

    def test_unfitted_uniform_extreme_inputs_finite(self, seed):
        """fit未実行(既定[0,1]しきい値)で±1e30入力でも有限"""
        enc = ThermometerEncoder(bits=4)
        x = _extreme_real_inputs().requires_grad_(True)
        y = enc(x)
        assert_all_finite(y, 'Thermometer(uniform, unfit) forward')
        y.sum().backward()
        assert_all_finite(x.grad, 'Thermometer(uniform, unfit) input grad')

    @pytest.mark.parametrize('mode', ['uniform', 'distributive'])
    def test_fit_with_extreme_data_finite(self, mode, seed):
        """±1e30を含むデータでfitしてもしきい値・出力が有限"""
        data = torch.randn(32, 8)
        data[0, 0] = EXTREME_INPUT
        data[1, 0] = -EXTREME_INPUT
        enc = ThermometerEncoder(bits=4, mode=mode).fit(data)
        assert_all_finite(enc.thresholds, f'Thermometer({mode}) thresholds')
        y = enc(_extreme_real_inputs())
        assert_all_finite(y, f'Thermometer({mode}) forward')

    def test_fit_degenerate_constant_data_finite(self, seed):
        """全定数データ(max−min=0)でfitしても有限(しきい値退化ケース)"""
        data = torch.full((16, 4), 0.5)
        enc = ThermometerEncoder(bits=4, mode='uniform').fit(data)
        assert_all_finite(enc.thresholds, 'Thermometer(定数データfit) thresholds')
        y = enc(torch.full((2, 4), 0.5))
        assert_all_finite(y, 'Thermometer(定数データfit) forward')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
