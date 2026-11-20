"""既知バグの起票テスト。

調査で判明した既存実装の欠陥を xfail(strict=True) で固定する。
M1で修正されたら xfail マーカーを外して正式テストに昇格する。

既知バグ一覧はリポジトリ内部の進捗記録「調査で判明した既知バグ」を参照。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.core import reference_grads
from griffith_logic.lut_layer import LutFunctionIWP, LutLayer


class TestIWPZeroFactorGradient:
    """回帰テスト(旧・既知バグ1、修正済み)。

    旧実装の output/factors.clamp(min=1e-8) 方式は、factor_j ≈ 0
    (入力x_jが厳密に0/1かつp_jが飽和)のとき grad_x の絶対誤差が
    O(1) になっていた。prefix/suffix積方式への書き換えで解消。
    このテストはゼロfactor入力での勾配正当性を恒久的に監視する。
    """

    def _zero_factor_setup(self):
        torch.manual_seed(3)
        in_dim, out_dim, n, batch = 8, 4, 4, 3
        layer = LutLayer(in_dim, out_dim, n_inputs=n, parameterization='iwp')
        indices = layer.indices.clone()
        indices[0, :] = 0  # 全ニューロンのslot0を入力0に固定
        x = torch.rand(batch, in_dim, dtype=torch.float64) * 0.8 + 0.1
        x[:, 0] = 0.0  # 入力0を厳密に0
        # α=40 → p=sigmoid(40)≈1-4e-18 → factor_0=(1-p)≈4e-18 < clamp閾値1e-8
        weights = torch.full((out_dim, n), 40.0, dtype=torch.float64)
        return x, weights, indices, n

    def test_grad_x_matches_gradcheck(self):
        x, weights, indices, n = self._zero_factor_setup()
        x = x.requires_grad_(True)
        fn = lambda x: LutFunctionIWP.apply(x, weights, indices, n)
        assert torch.autograd.gradcheck(fn, (x,), atol=1e-6, rtol=1e-3)

    def test_grad_x_matches_reference(self):
        x, weights, indices, n = self._zero_factor_setup()
        grad_output = torch.ones(x.shape[0], indices.shape[1], dtype=torch.float64)

        xr = x.clone().requires_grad_(True)
        y = LutFunctionIWP.apply(xr, weights, indices, n)
        (grad_x,) = torch.autograd.grad(y, (xr,), grad_outputs=grad_output)

        _, grad_x_ref, _ = reference_grads(
            x, weights, indices, n, 'iwp', grad_output,
        )
        assert torch.allclose(grad_x, grad_x_ref, atol=1e-10), (
            f'max diff = {(grad_x - grad_x_ref).abs().max().item():.3e}'
        )


# 既知バグ2(CUDAテストの旧シグネチャ)は修正済み:
# tests/l2_grad/test_cuda_parity.py に現行シグネチャで移設した。


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
