"""L1: Gumbel-STの数学的性質。

    1. Gumbel-max性質: argmax(w+G)のサンプル頻度 ≈ softmax(w)
    2. evalは決定的(ノイズなしargmax)でsoft/hard_stのevalと同一
    3. 訓練forwardの出力は常に「単一ゲートのhard関数」の値
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.lut_layer import LutLayer  # noqa: E402


class TestGumbelMaxProperty:
    def test_sample_frequency_matches_softmax(self):
        """argmax(w+G)の経験頻度がsoftmax(w)に一致(Gumbel-max trick)。"""
        torch.manual_seed(0)
        w = torch.tensor([1.5, 0.0, -1.0, 0.5])
        n_samples = 40000
        u = torch.rand(n_samples, 4)
        g = -torch.log((-torch.log(u + 1e-20)).clamp_min(1e-20))
        counts = torch.bincount((w + g).argmax(-1), minlength=4).float()
        freq = counts / n_samples
        probs = F.softmax(w, dim=-1)
        assert torch.allclose(freq, probs, atol=0.01), (freq, probs)


class TestEvalDeterminism:
    def test_eval_no_noise_and_matches_hard_st_eval(self):
        torch.manual_seed(5)
        layer = LutLayer(8, 6, n_inputs=2, parameterization='softmax',
                         selection='gumbel_st')
        x = torch.rand(4, 8)
        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        assert torch.equal(y1, y2), 'evalが確率的になっている'

        # selectionはeval経路に影響しない(全selectionでeval同一)
        for sel in ('soft', 'hard_st'):
            other = LutLayer(8, 6, n_inputs=2, parameterization='softmax',
                             selection=sel)
            with torch.no_grad():
                other.weights.copy_(layer.weights)
                other.indices.copy_(layer.indices)
            other.eval()
            assert torch.equal(y1, other(x)), f'eval経路が{sel}と不一致'


class TestForwardIsHardFunction:
    def test_binary_input_gives_binary_output(self):
        """0/1入力に対する訓練forward出力は{0,1}(単一ゲートのhard関数)。"""
        torch.manual_seed(9)
        layer = LutLayer(6, 8, n_inputs=2, parameterization='softmax',
                         selection='gumbel_st')
        x = (torch.rand(16, 6) > 0.5).float()
        layer.train()
        y = layer(x)
        assert set(y.unique().tolist()).issubset({0.0, 1.0}), (
            'gumbel_stのforwardがhard関数になっていない'
        )
