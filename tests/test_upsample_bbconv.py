"""UpSample2d / BinaryBrainConv2d の検証。

UpSample2d:
    nearest意味論 y[b,c,h,w] = x[b,c,h//s,w//s] の素朴照合と勾配疎通。
BinaryBrainConv2d:
    最小構成でのforward形状、二値入力→二値域出力{0,1}、勾配疎通・有限性。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.bb_conv2d import BinaryBrainConv2d
from griffith_logic.upsample import UpSample2d


class TestUpSample2d:
    def test_nearest_semantics_2x2_to_4x4(self):
        """2x2→4x4の手計算照合(各画素が2x2ブロックに複製される)"""
        x = torch.tensor([[[[1.0, 2.0],
                            [3.0, 4.0]]]])
        y = UpSample2d(scale_factor=2)(x)
        expected = torch.tensor([[[[1.0, 1.0, 2.0, 2.0],
                                   [1.0, 1.0, 2.0, 2.0],
                                   [3.0, 3.0, 4.0, 4.0],
                                   [3.0, 3.0, 4.0, 4.0]]]])
        assert y.shape == (1, 1, 4, 4)
        assert torch.equal(y, expected)

    @pytest.mark.parametrize('scale', [2, 3])
    def test_matches_naive_loop(self, scale):
        """nearest意味論 y[b,c,h,w] == x[b,c,h//s,w//s] の素朴照合"""
        torch.manual_seed(3)
        x = torch.rand(2, 3, 2, 2)
        y = UpSample2d(scale_factor=scale)(x)
        assert y.shape == (2, 3, 2 * scale, 2 * scale)
        for b in range(2):
            for c in range(3):
                for h in range(2 * scale):
                    for w in range(2 * scale):
                        assert y[b, c, h, w] == x[b, c, h // scale, w // scale]

    def test_grad_flow(self):
        """勾配疎通: Σyのxに対する勾配は各画素の複製数(s²)"""
        x = torch.rand(1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        y = UpSample2d(scale_factor=2)(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.full_like(x, 4.0))


class TestBinaryBrainConv2d:
    def _make(self, seed=43):
        torch.manual_seed(seed)
        return BinaryBrainConv2d(
            lut_sizes=[8, 4], in_channels=1, kernel_size=3, n_inputs=2,
        )

    def test_forward_shape_train(self):
        conv = self._make()
        conv.train()
        torch.manual_seed(1)
        x = torch.randint(0, 2, (2, 1, 4, 4)).float()
        y = conv(x)
        # padding='same' なので空間サイズ保持、チャンネルはlut_sizes[-1]
        assert y.shape == (2, 4, 4, 4)

    def test_binary_input_gives_binary_output(self):
        """二値入力→二値域出力{0,1}(最終段Binarizeにより厳密)"""
        conv = self._make()
        conv.train()
        torch.manual_seed(2)
        x = torch.randint(0, 2, (2, 1, 4, 4)).float()
        y_train = conv(x)
        assert torch.all((y_train == 0.0) | (y_train == 1.0))

        # evalモード(BN running統計使用)でも二値域
        conv.eval()
        with torch.no_grad():
            y_eval = conv(x)
        assert y_eval.shape == y_train.shape
        assert torch.all((y_eval == 0.0) | (y_eval == 1.0))

    def test_grad_flow_finite(self):
        """勾配疎通: 全パラメータのgradが存在し全有限、かつ一部は非ゼロ"""
        conv = self._make()
        conv.train()
        torch.manual_seed(5)
        x = torch.randint(0, 2, (4, 1, 4, 4)).float()
        y = conv(x)
        y.sum().backward()

        total_abs = 0.0
        for name, p in conv.named_parameters():
            assert p.grad is not None, f'{name} のgradがNone'
            assert torch.isfinite(p.grad).all(), f'{name} のgradに非有限値'
            total_abs += p.grad.abs().sum().item()
        assert total_abs > 0.0, '全パラメータの勾配がゼロ(Binarize STEが遮断)'

    def test_lazy_in_channels_build(self):
        """in_channels未指定でもforward時に自動構築される"""
        torch.manual_seed(7)
        conv = BinaryBrainConv2d(lut_sizes=[8, 4], kernel_size=3, n_inputs=2)
        x = torch.randint(0, 2, (2, 2, 4, 4)).float()
        y = conv(x)
        assert conv.in_channels == 2
        assert y.shape == (2, 4, 4, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
