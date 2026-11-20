"""L1: GroupSum / SpatialGroupSum の出力定義の厳密検証。

検証する数式:
    GroupSum:        logits[g] = Σ_{i=g·n_per}^{(g+1)·n_per-1} x_i / τ
                     (連続したn_per個ずつをkクラスに分割して和)
    τスケール性質:   logits(τ) == logits(1) / τ
    SpatialGroupSum: y[b,o,h,w] = Σ_{c=o·gs}^{(o+1)·gs-1} x[b,c,h,w] × scale
                     (チャンネルをグループ和、空間次元は保持)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest定数用

from conftest import GRADCHECK_KW, TOL_F64_EQ

from griffith_logic.group_sum import GroupSum
from griffith_logic.spatial_group_sum import SpatialGroupSum


class TestGroupSumSemantics:
    def test_matches_naive_loop(self):
        """出力定義(k分割和/τ)を素朴な二重ループで厳密照合"""
        torch.manual_seed(3)
        k, n_per, tau = 3, 4, 2.5
        x = torch.rand(5, k * n_per, dtype=torch.float64)
        y = GroupSum(k=k, tau=tau)(x)

        expected = torch.zeros(5, k, dtype=torch.float64)
        for b in range(5):
            for g in range(k):
                s = 0.0
                for i in range(n_per):
                    s += x[b, g * n_per + i]
                expected[b, g] = s / tau

        assert y.shape == (5, k)
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0)

    def test_matches_naive_loop_extra_dims(self):
        """先頭次元が複数あっても最終次元のみ分割される"""
        torch.manual_seed(5)
        k, n_per, tau = 2, 3, 1.5
        x = torch.rand(2, 4, k * n_per, dtype=torch.float64)
        y = GroupSum(k=k, tau=tau)(x)

        expected = torch.zeros(2, 4, k, dtype=torch.float64)
        for b in range(2):
            for t in range(4):
                for g in range(k):
                    expected[b, t, g] = x[b, t, g * n_per:(g + 1) * n_per].sum() / tau

        assert y.shape == (2, 4, k)
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0)

    def test_gradcheck(self):
        torch.manual_seed(7)
        gs = GroupSum(k=3, tau=2.0)
        x = torch.rand(4, 12, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(gs, (x,), **GRADCHECK_KW)

    @pytest.mark.parametrize('tau', [1.0, 2.0, 7.5, 30.0])
    def test_tau_scaling(self, tau):
        """τスケール性質: logits(τ) == logits(1) / τ"""
        torch.manual_seed(11)
        x = torch.rand(6, 8, dtype=torch.float64)
        y_tau = GroupSum(k=4, tau=tau)(x)
        y_one = GroupSum(k=4, tau=1.0)(x)
        assert torch.allclose(y_tau, y_one / tau, atol=TOL_F64_EQ, rtol=0)

    def test_rejects_indivisible_dim(self):
        with pytest.raises(AssertionError):
            GroupSum(k=3)(torch.rand(2, 8))


class TestSpatialGroupSumSemantics:
    def test_matches_naive_loop(self):
        """空間保持の分割和意味論を素朴な四重ループで厳密照合"""
        torch.manual_seed(13)
        n_outputs, group_size, scale = 2, 3, 0.5
        x = torch.rand(2, n_outputs * group_size, 3, 4, dtype=torch.float64)
        y = SpatialGroupSum(n_outputs=n_outputs, scale=scale)(x)

        expected = torch.zeros(2, n_outputs, 3, 4, dtype=torch.float64)
        for b in range(2):
            for o in range(n_outputs):
                for h in range(3):
                    for w in range(4):
                        s = 0.0
                        for c in range(group_size):
                            s += x[b, o * group_size + c, h, w]
                        expected[b, o, h, w] = s * scale

        assert y.shape == (2, n_outputs, 3, 4)
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0)

    def test_gradcheck(self):
        torch.manual_seed(17)
        layer = SpatialGroupSum(n_outputs=2, scale=0.5)
        x = torch.rand(2, 4, 2, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(layer, (x,), **GRADCHECK_KW)

    @pytest.mark.parametrize('scale', [0.25, 1.0, 3.0])
    def test_scale_linearity(self, scale):
        """scaleは出力に線形: y(scale) == y(1) × scale"""
        torch.manual_seed(19)
        x = torch.rand(2, 6, 2, 2, dtype=torch.float64)
        y_s = SpatialGroupSum(n_outputs=3, scale=scale)(x)
        y_1 = SpatialGroupSum(n_outputs=3, scale=1.0)(x)
        assert torch.allclose(y_s, y_1 * scale, atol=TOL_F64_EQ, rtol=0)

    def test_rejects_indivisible_channels(self):
        with pytest.raises(AssertionError):
            SpatialGroupSum(n_outputs=4)(torch.rand(1, 6, 2, 2))


class TestGroupSumStrideAssign:
    """assign='stride': ニューロン i → クラス i mod k
    (内部調査リポジトリ先行実験の GroupSum)。既定 contig は従来と不変。"""

    def test_matches_naive_loop(self):
        torch.manual_seed(23)
        k, m, tau = 3, 4, 2.5
        x = torch.rand(5, k * m, dtype=torch.float64)
        y = GroupSum(k=k, tau=tau, assign='stride')(x)

        expected = torch.zeros(5, k, dtype=torch.float64)
        for b in range(5):
            for i in range(k * m):
                expected[b, i % k] += x[b, i]

        assert y.shape == (5, k)
        assert torch.allclose(y, expected / tau, atol=TOL_F64_EQ, rtol=0)

    def test_contig_default_is_unchanged(self):
        torch.manual_seed(27)
        x = torch.rand(4, 12, dtype=torch.float64)
        assert torch.equal(GroupSum(k=3, tau=2.0)(x),
                           GroupSum(k=3, tau=2.0, assign='contig')(x))

    def test_stride_equals_contig_after_transpose(self):
        """stride の和 = (m, k) → (k, m) に転置した contig の和(IR の permute と同じ恒等式)"""
        torch.manual_seed(29)
        k, m = 4, 6
        x = torch.rand(3, k * m, dtype=torch.float64)
        xt = x.reshape(3, m, k).transpose(1, 2).reshape(3, k * m)
        assert torch.allclose(GroupSum(k=k, assign='stride')(x), GroupSum(k=k)(xt),
                              atol=TOL_F64_EQ, rtol=0)

    def test_gradcheck(self):
        torch.manual_seed(31)
        gs = GroupSum(k=3, tau=2.0, assign='stride')
        x = torch.rand(4, 12, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(gs, (x,), **GRADCHECK_KW)

    def test_rejects_unknown_assign(self):
        with pytest.raises(AssertionError):
            GroupSum(k=2, assign='random')

    def test_extra_repr(self):
        assert GroupSum(k=2, tau=1.0).extra_repr() == 'k=2, tau=1.0'
        assert GroupSum(k=2, tau=1.0, assign='stride').extra_repr() == (
            'k=2, tau=1.0, assign=stride')


class TestGroupSumBias:
    """判定バイアス: logits = (Σ + bias) / τ。NID の s1 > s0 + t は bias = [t, 0]"""

    def test_bias_added_before_tau(self):
        torch.manual_seed(37)
        x = torch.rand(5, 12, dtype=torch.float64)
        plain = GroupSum(k=3, tau=2.0)(x)
        biased = GroupSum(k=3, tau=2.0, bias=[1.0, -2.0, 0.5])(x)
        expected = plain + torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64) / 2.0
        assert torch.allclose(biased, expected, atol=TOL_F64_EQ, rtol=0)

    def test_none_and_zero_bias_unchanged(self):
        torch.manual_seed(41)
        x = torch.rand(4, 8, dtype=torch.float64)
        assert torch.equal(GroupSum(k=2)(x), GroupSum(k=2, bias=[0, 0])(x))
        assert GroupSum(k=2).bias is None
        assert GroupSum(k=2, bias=[0, 0]).bias_ints() is None

    def test_nid_decision_rule(self):
        """s1 > s0 + t  ⇔  argmax((s0 + t, s1)) == 1(同点は 0 = normal)"""
        gs = GroupSum(k=2, tau=1.0, bias=[3, 0])
        x = torch.zeros(3, 8)
        x[0, 4:8] = 1.0                     # s0=0, s1=4 → 4 > 0+3 → 1
        x[1, 4:7] = 1.0                     # s0=0, s1=3 → 同点 → 0
        x[2, 0:2] = 1.0
        x[2, 4:8] = 1.0                     # s0=2, s1=4 → 4 > 5 でない → 0
        assert gs(x).argmax(1).tolist() == [1, 0, 0]

    def test_bias_ints_and_validation(self):
        assert GroupSum(k=2, bias=[3, 0]).bias_ints() == [3, 0]
        with pytest.raises(ValueError):
            GroupSum(k=2, bias=[1.5, 0]).bias_ints()
        with pytest.raises(ValueError):
            GroupSum(k=3, bias=[1, 0])
        assert 'bias=[3.0, 0.0]' in GroupSum(k=2, bias=[3, 0]).extra_repr()

    def test_gradcheck_with_bias(self):
        torch.manual_seed(43)
        gs = GroupSum(k=2, tau=1.5, bias=[2.0, -1.0])
        x = torch.rand(3, 6, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(gs, (x,), **GRADCHECK_KW)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
