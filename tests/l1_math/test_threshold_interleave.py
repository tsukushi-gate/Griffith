"""L1: ThresholdTransform の平坦化順オプション(先行実験 S3')。

検証する仕様:
    interleave=False(既定) — index = ((t*C + c)*H + row)*W + col（しきい値major）
    interleave=True        — index = ((c*H + row)*W + col)*T + t（しきい値minor）
                             = 社内先行実験の凍結符号化仕様 v1
                               (cifar/encode_cifar.py: ((c*32+row)*32+col)*3+t)
    どちらもビットの多重集合は同一で、順序だけが置換になる。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.datasets.vision import ThresholdTransform

C, H, W = 3, 32, 32
T = 3


def _synthetic_image(seed=0):
    """[0,1] の合成画像 (C,H,W)。しきい値 1/4・2/4・3/4 の全域を跨ぐ値を含む"""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(C, H, W, generator=gen)


class TestIndexFormula:
    def test_interleaved_index_formula(self):
        """flat[((c*H+row)*W+col)*T+t] == (x[c,row,col] > (t+1)/(T+1))"""
        x = _synthetic_image()
        flat = ThresholdTransform(T, interleave=True)(x).reshape(-1)
        assert flat.numel() == C * H * W * T
        for c in range(C):
            for row in (0, 7, 31):
                for col in (0, 13, 31):
                    for t in range(T):
                        idx = ((c * H + row) * W + col) * T + t
                        expected = float(x[c, row, col] > (t + 1) / (T + 1))
                        assert flat[idx].item() == expected, (c, row, col, t)

    def test_default_index_formula(self):
        """既定は flat[((t*C+c)*H+row)*W+col]（従来のしきい値major）"""
        x = _synthetic_image(1)
        flat = ThresholdTransform(T)(x).reshape(-1)
        for c in range(C):
            for row in (0, 31):
                for col in (0, 31):
                    for t in range(T):
                        idx = ((t * C + c) * H + row) * W + col
                        expected = float(x[c, row, col] > (t + 1) / (T + 1))
                        assert flat[idx].item() == expected, (c, row, col, t)

    def test_neighbourhood_semantics(self):
        """しきい値minor順では +T が右1画素、+W*T が1行下になる（rot 配線の前提）"""
        x = _synthetic_image(2)
        flat = ThresholdTransform(T, interleave=True)(x).reshape(-1)
        c, row, col, t = 1, 5, 10, 0
        base = ((c * H + row) * W + col) * T + t
        right = ((c * H + row) * W + col + 1) * T + t
        down = ((c * H + row + 1) * W + col) * T + t
        assert right - base == T          # 先行実験 CIFAR: +3
        assert down - base == W * T       # 先行実験 CIFAR: +96
        assert flat[right].item() == float(x[c, row, col + 1] > (t + 1) / (T + 1))
        assert flat[down].item() == float(x[c, row + 1, col] > (t + 1) / (T + 1))


class TestSameBitsDifferentOrder:
    def test_same_multiset(self):
        x = _synthetic_image(3)
        a = ThresholdTransform(T)(x).reshape(-1)
        b = ThresholdTransform(T, interleave=True)(x).reshape(-1)
        assert a.numel() == b.numel()
        assert a.sum().item() == b.sum().item()
        assert torch.equal(a.sort().values, b.sort().values)

    def test_is_a_permutation(self):
        """順序だけの置換であることを index 対応で厳密確認する"""
        x = _synthetic_image(4)
        a = ThresholdTransform(T)(x).reshape(-1)
        b = ThresholdTransform(T, interleave=True)(x).reshape(-1)
        perm = torch.empty(C * H * W * T, dtype=torch.int64)
        for c in range(C):
            for row in range(H):
                for col in range(W):
                    for t in range(T):
                        src = ((t * C + c) * H + row) * W + col
                        dst = ((c * H + row) * W + col) * T + t
                        perm[dst] = src
        assert torch.equal(b, a[perm])
        assert torch.equal(perm.sort().values,
                           torch.arange(C * H * W * T))


class TestShapesAndBackCompat:
    def test_default_shape_unchanged(self):
        x = _synthetic_image(5)
        out = ThresholdTransform(T)(x)
        assert out.shape == (T * C, H, W)

    def test_interleaved_shape(self):
        x = _synthetic_image(5)
        out = ThresholdTransform(T, interleave=True)(x)
        assert out.shape == (C, H, W * T)
        assert out.numel() == T * C * H * W

    def test_default_is_bitwise_identical_to_legacy(self):
        """既定パスは torch.cat(dim=0) のままであること"""
        x = _synthetic_image(6)
        legacy = torch.cat(
            [(x > (i + 1) / (T + 1)).float() for i in range(T)], dim=0)
        assert torch.equal(ThresholdTransform(T)(x), legacy)
        assert torch.equal(ThresholdTransform(T, interleave=False)(x), legacy)

    @pytest.mark.parametrize('count', [1, 2, 31])
    def test_other_threshold_counts(self, count):
        x = _synthetic_image(7)
        out = ThresholdTransform(count, interleave=True)(x)
        assert out.shape == (C, H, W * count)
        flat = out.reshape(-1)
        idx = ((2 * H + 3) * W + 4) * count + (count - 1)
        assert flat[idx].item() == float(
            x[2, 3, 4] > count / (count + 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
