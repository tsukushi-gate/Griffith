"""L2: LutLayer.forwardのルーティング検証。

LutLayerが各parameterization/selectionで正しいFunctionに
ディスパッチしていること、grad_factor・gumbelの各機構が
仕様どおり動くことをレイヤAPIレベルで検証する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F32_EQ

from griffith_logic.lut_layer import (
    LutFunction,
    LutFunctionIndependent,
    LutFunctionIWP,
    LutFunctionRaw,
    LutFunctionWARP,
    LutLayer,
)


class TestForwardRouting:
    """layer(x)がFunction直接呼び出しと同一出力(CPU/train)"""

    @pytest.mark.parametrize('param', ['softmax', 'independent', 'raw', 'iwp'])
    def test_train_mode_routes_to_function(self, param, seed):
        layer = LutLayer(8, 4, n_inputs=2, parameterization=param)
        layer.train()
        x = torch.rand(3, 8)
        y = layer(x)

        gate_table = getattr(layer, 'gate_table', None)
        if param == 'softmax':
            y_direct = LutFunction.apply(
                x, layer.weights, layer.indices, 2, gate_table,
            )
        elif param == 'independent':
            y_direct = LutFunctionIndependent.apply(
                x, layer.weights, layer.indices, 2,
            )
        elif param == 'raw':
            y_direct = LutFunctionRaw.apply(x, layer.weights, layer.indices, 2)
        else:
            y_direct = LutFunctionIWP.apply(x, layer.weights, layer.indices, 2)

        assert torch.allclose(y, y_direct, atol=TOL_F32_EQ, rtol=0)

    def test_warp_routes_with_gumbel_off(self, seed):
        layer = LutLayer(8, 4, n_inputs=4, parameterization='warp')
        layer.gumbel = False
        layer.train()
        x = torch.rand(3, 8)
        y = layer(x)
        y_direct = LutFunctionWARP.apply(
            x, layer.weights, layer.indices, 4, layer.tau_warp, False,
        )
        assert torch.allclose(y, y_direct, atol=TOL_F32_EQ, rtol=0)


class TestWarpGumbel:
    def test_gumbel_is_stochastic(self, seed):
        """gumbel=True では呼び出しごとに出力が変わる"""
        layer = LutLayer(8, 4, n_inputs=4, parameterization='warp')
        assert layer.gumbel  # デフォルトON
        layer.train()
        x = torch.rand(3, 8)
        y1, y2 = layer(x), layer(x)
        assert not torch.allclose(y1, y2), 'gumbelノイズが効いていない'

    def test_gumbel_mean_approximates_deterministic(self, seed):
        """gumbelノイズは平均的には決定的出力の近傍(logisticノイズ、中央値一致)"""
        layer = LutLayer(8, 4, n_inputs=4, parameterization='warp')
        layer.train()
        x = torch.rand(16, 8)

        layer.gumbel = False
        y_det = layer(x)

        layer.gumbel = True
        samples = torch.stack([layer(x) for _ in range(400)])
        # logisticノイズは対称 → サンプル中央値 ≈ 決定的sigmoid出力
        y_med = samples.median(dim=0).values
        assert (y_med - y_det).abs().max() < 0.1, (
            f'中央値乖離 {(y_med - y_det).abs().max():.3f}'
        )


class TestGradFactor:
    def test_grad_factor_scales_input_gradient(self, seed):
        """grad_factor=f はgrad_xをf倍する(forwardは不変)"""
        torch.manual_seed(31)
        layer1 = LutLayer(8, 4, n_inputs=2, grad_factor=1.0)
        torch.manual_seed(31)
        layer2 = LutLayer(8, 4, n_inputs=2, grad_factor=2.0)

        x = torch.rand(3, 8)
        x1 = x.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)

        y1, y2 = layer1(x1), layer2(x2)
        assert torch.allclose(y1, y2, atol=TOL_F32_EQ)  # forward不変

        y1.sum().backward()
        y2.sum().backward()
        assert torch.allclose(x2.grad, 2.0 * x1.grad, atol=TOL_F32_EQ), (
            'grad_factorが入力勾配をスケールしていない'
        )

    def test_grad_factor_does_not_scale_weight_gradient(self, seed):
        """grad_factorはweightsの勾配には影響しない(入力側STEのみ)"""
        torch.manual_seed(37)
        layer1 = LutLayer(8, 4, n_inputs=2, grad_factor=1.0)
        torch.manual_seed(37)
        layer2 = LutLayer(8, 4, n_inputs=2, grad_factor=2.0)

        x = torch.rand(3, 8)
        layer1(x).sum().backward()
        layer2(x).sum().backward()
        assert torch.allclose(
            layer1.weights.grad, layer2.weights.grad, atol=TOL_F32_EQ,
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
