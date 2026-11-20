"""L1: Thermometer符号化の数学的性質(DWN, Bacellar et al. 2024, Appendix B)。

検証する数式:
    符号化:      T(q) = (1_{q>t_1}, ..., 1_{q>t_z}),  t_{i+1} > t_i
    単調性:      unary code — bit_i ≥ bit_{i+1}(しきい値昇順のため)
    しきい値数:  z = bits 個/特徴
    distributive: t_i = sort(data)[⌊N·i/(z+1)⌋](経験分位点、補間なし。
                  公式実装 torch_dwn/binarization.py DistributiveThermometer と同一)
    uniform:      t_i = min + i·(max−min)/(z+1)(等間隔)
    STE:          ∂L/∂x = Σ_i ∂L/∂T_i(素通し合算)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest定数用

from conftest import TOL_F64_EQ

from griffith_logic.encoding.thermometer import ThermometerEncoder

BITS = [1, 3, 8]


class TestThresholds:
    @pytest.mark.parametrize('bits', BITS)
    def test_uniform_evenly_spaced(self, bits):
        """uniform: [min,max]を(z+1)等分する内点がしきい値"""
        data = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
        enc = ThermometerEncoder(bits, mode='uniform').fit(data)
        expected = torch.tensor(
            [[2.0 * i / (bits + 1)] for i in range(1, bits + 1)],
            dtype=torch.float64,
        )
        assert enc.thresholds.shape == (bits, 1)
        assert torch.allclose(enc.thresholds, expected, atol=TOL_F64_EQ, rtol=0)

    @pytest.mark.parametrize('bits', BITS)
    def test_distributive_matches_quantile(self, bits, seed):
        """distributive: しきい値 = sort(data)[⌊N·i/(z+1)⌋](公式実装の経験分位点)"""
        data = torch.rand(100, 5, dtype=torch.float64)
        enc = ThermometerEncoder(bits, mode='distributive').fit(data)
        sorted_data = torch.sort(data, dim=0)[0]
        idx = [int(data.shape[0] * i / (bits + 1)) for i in range(1, bits + 1)]
        expected = sorted_data[idx]
        assert enc.thresholds.shape == (bits, 5)
        assert torch.equal(enc.thresholds, expected)

    def test_thresholds_ascending(self, seed):
        """しきい値は昇順 t_{i+1} ≥ t_i(両モード)"""
        data = torch.rand(200, 4, dtype=torch.float64)
        for mode in ('uniform', 'distributive'):
            enc = ThermometerEncoder(5, mode=mode).fit(data)
            diff = enc.thresholds[1:] - enc.thresholds[:-1]
            assert (diff >= 0).all(), mode

    def test_unfitted_uniform_defaults_to_unit_range(self):
        """fit前のuniformは[0,1]範囲のしきい値で動作する"""
        enc = ThermometerEncoder(3, mode='uniform')
        out = enc.encode(torch.tensor([[0.5]]))
        assert out.shape == (1, 1, 3)
        # t = 0.25, 0.5, 0.75 → 0.5 は t_1 のみ超える
        assert out.tolist() == [[[1.0, 0.0, 0.0]]]

    def test_distributive_requires_fit(self):
        """distributiveはfit必須(未fitはエラー)"""
        enc = ThermometerEncoder(3, mode='distributive')
        with pytest.raises(RuntimeError):
            enc.encode(torch.rand(2, 2))

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            ThermometerEncoder(3, mode='gaussian')


class TestEncoding:
    @pytest.mark.parametrize('bits', BITS)
    def test_monotone_unary_code(self, bits, seed):
        """単調性: bit_i ≥ bit_{i+1}(1が連続するunary code)"""
        data = torch.rand(64, 7, dtype=torch.float64)
        enc = ThermometerEncoder(bits, mode='distributive').fit(data)
        code = enc.encode(torch.rand(32, 7, dtype=torch.float64))
        assert code.shape == (32, 7, bits)
        assert (code[..., :-1] >= code[..., 1:]).all()

    def test_values_binary(self, seed):
        """符号は{0,1}のみ"""
        data = torch.rand(64, 3)
        enc = ThermometerEncoder(4, mode='distributive').fit(data)
        code = enc.encode(torch.rand(16, 3))
        assert ((code == 0) | (code == 1)).all()

    def test_popcount_equals_rank(self):
        """1の個数 = 超えたしきい値の個数(定義通り)"""
        enc = ThermometerEncoder(4, mode='uniform')  # t = 0.2,0.4,0.6,0.8
        x = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        code = enc.encode(x)
        assert code.sum(-1).tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0]]

    def test_forward_flattens(self, seed):
        """forward: (batch, D) → (batch, D*bits)、特徴優先の並び"""
        data = torch.rand(32, 4)
        enc = ThermometerEncoder(3, mode='distributive').fit(data)
        x = torch.rand(8, 4)
        out = enc(x)
        assert out.shape == (8, 12)
        assert torch.equal(out, enc.encode(x).reshape(8, 12))


class TestSTE:
    def test_gradient_pass_through(self):
        """STE: ∂L/∂x_d = Σ_i ∂L/∂T_{d,i}(しきい値比較を素通し)"""
        enc = ThermometerEncoder(3, mode='uniform')
        x = torch.tensor([[0.3, 0.9]], dtype=torch.float64, requires_grad=True)
        out = enc(x)  # (1, 6)
        grad_out = torch.arange(1.0, 7.0, dtype=torch.float64).reshape(1, 6)
        out.backward(grad_out)
        expected = torch.tensor([[1.0 + 2.0 + 3.0, 4.0 + 5.0 + 6.0]], dtype=torch.float64)
        assert torch.allclose(x.grad, expected, atol=TOL_F64_EQ, rtol=0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
