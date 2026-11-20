"""L1: LutConv2d のサンプラ指定・sin 推定器・Ω 空間 Residual Init。

    - sampler='dilated8' 等は conv_samplers の座標を bit 反転で写した indices_flat を持つ
    - 窓・木・入力数・配線が腕と食い違えば ValueError
    - estimator='sin' の eval は sin01(w) > 0.5 で離散化し(sigmoid と食い違う w で確認)、
      export の真理値表(truth_tables_from_conv)も同じ規則
    - RI(sin)は第 1 入力の pass-through = 木の出力が葉 0 の値になる
    - 既定(sampler='random', estimator='sigmoid')の構築は従来と同じ経路
"""

import math

import pytest
import torch

from griffith_logic.conv import LutConv2d
from griffith_logic.conv_samplers import build_leaf_selection, leaf_indices_flat
from griffith_logic.export.discretize import truth_tables_from_conv


def _images(batch=6, c=2, h=7, w=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, c, h, w), generator=g).float()


class TestSamplerWiring:
    @pytest.mark.parametrize('arm,kernel,depth', [
        ('dilated8', 5, 3), ('cross5', 5, 3), ('checker13', 5, 4), ('random8', 3, 3)])
    def test_indices_flat_matches_sampler(self, arm, kernel, depth):
        g = torch.Generator().manual_seed(5)
        conv = LutConv2d(4, 6, kernel_size=kernel, tree_depth=depth, padding=kernel // 2,
                         sampler=arm, layer_idx=2, generator=g)
        g2 = torch.Generator().manual_seed(5)
        ch, dy, dx = build_leaf_selection(arm, 6, 4, layer_idx=2, num_chn=2, generator=g2)
        expected = leaf_indices_flat(ch, dy, dx, kernel, kernel)
        assert torch.equal(conv.indices_flat, expected)
        assert f'sampler={arm}' in conv.extra_repr()

    def test_forward_shapes_with_pattern_sampler(self):
        conv = LutConv2d(2, 5, kernel_size=5, tree_depth=3, padding=2, sampler='dilated8')
        y = conv.eval()(_images())
        assert y.shape == (6, 5, 7, 7)
        assert set(y.unique().tolist()) <= {0.0, 1.0}

    def test_mismatched_arguments_rejected(self):
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=3, tree_depth=3, sampler='dilated8')      # 窓 5 が要る
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=5, tree_depth=2, sampler='dilated8')      # 葉 8 が要る
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=5, tree_depth=3, sampler='checker13')     # 葉 16 が要る
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=5, tree_depth=1, n_inputs=4, sampler='cross5')
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=5, tree_depth=3, sampler='dilated8', wiring='learnable')
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=3, sampler='grid9')

    def test_default_sampler_is_random(self):
        conv = LutConv2d(2, 4, kernel_size=3)
        assert conv.sampler == 'random'
        assert conv.estimator == 'sigmoid'
        assert 'sampler=' not in conv.extra_repr()
        assert 'est=' not in conv.extra_repr()


class TestSinEstimator:
    def test_requires_independent(self):
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=3, estimator='sin')             # softmax × sin
        with pytest.raises(ValueError):
            LutConv2d(2, 4, kernel_size=3, parameterization='iwp', estimator='sin')

    def test_eval_threshold_follows_sin(self):
        """w = 4: sigmoid(4) = 0.98 > 0.5 だが sin01(4) = 0.12 < 0.5。eval は sin に従う"""
        torch.manual_seed(0)
        conv = LutConv2d(1, 1, kernel_size=3, tree_depth=1, parameterization='independent',
                         estimator='sin', residual_init=False)
        with torch.no_grad():
            conv.weights.fill_(4.0)           # 全エントリ: sigmoid なら 1、sin なら 0
        y = conv.eval()(_images(c=1))
        assert float(y.max()) == 0.0
        conv_sig = LutConv2d(1, 1, kernel_size=3, tree_depth=1, parameterization='independent',
                             residual_init=False)
        with torch.no_grad():
            conv_sig.weights.fill_(4.0)
        assert float(conv_sig.eval()(_images(c=1)).min()) == 1.0

    def test_truth_tables_match_eval_rule(self):
        torch.manual_seed(1)
        conv = LutConv2d(2, 3, kernel_size=5, tree_depth=3, padding=2, sampler='cross5',
                         parameterization='independent', estimator='sin', residual_init=False)
        with torch.no_grad():
            conv.weights.copy_(torch.randn_like(conv.weights) * 3.0)   # |w| > π も混ざる
        tts = truth_tables_from_conv(conv)
        w = conv.weights.detach()
        for oc in range(3):
            for g in range(conv.n_internal):
                expected = 0
                for k in range(4):
                    if 0.5 + 0.5 * math.sin(float(w[oc, g, k])) > 0.5:
                        expected |= 1 << k
                assert int(tts[oc, g]) == expected

    def test_train_forward_uses_sin(self):
        torch.manual_seed(2)
        conv = LutConv2d(1, 1, kernel_size=3, tree_depth=1, parameterization='independent',
                         estimator='sin', residual_init=False)
        with torch.no_grad():
            conv.weights.zero_()               # sin01(0) = 0.5 → soft 出力は定数 0.5
        y = conv.train()(_images(c=1))
        assert torch.allclose(y, torch.full_like(y, 0.5))


class TestResidualInitSin:
    def test_omega_space_init_statistics(self):
        g = torch.Generator().manual_seed(11)
        conv = LutConv2d(2, 64, kernel_size=3, tree_depth=3, parameterization='independent',
                         estimator='sin', residual_init=True, ri_mu=1.2, ri_sigma=0.25,
                         generator=g)
        w = conv.weights.detach()
        a1 = w[:, :, [1, 3]]                   # bit 0 が立つエントリ(第 1 入力 = 1)
        a0 = w[:, :, [0, 2]]
        assert abs(float(a1.mean()) - 1.2) < 0.05
        assert abs(float(a0.mean()) + 1.2) < 0.05
        assert abs(float(a1.std()) - 0.25) < 0.05

    def test_ri_tree_passes_first_input_through(self):
        """RI 直後の hard 木は毎段「a 素通し」なので、出力 = 葉 0(隣接対の最初)の値"""
        g = torch.Generator().manual_seed(12)
        conv = LutConv2d(2, 4, kernel_size=3, tree_depth=3, padding=1,
                         parameterization='independent', estimator='sin', residual_init=True,
                         sampler='random8', generator=g)
        x = _images(batch=4, c=2, h=6, w=6)
        y = conv.eval()(x)
        patches = torch.nn.functional.unfold(x, (3, 3), padding=1)    # (B, C·9, L)
        leaf0 = patches[:, conv.indices_flat[:, 0]]                     # (B, OC, L)
        expected = leaf0.reshape(4, 4, 6, 6)
        assert torch.equal(y, expected)

    def test_generator_makes_init_deterministic(self):
        a = LutConv2d(2, 8, kernel_size=3, parameterization='independent', estimator='sin',
                      sampler='random8', generator=torch.Generator().manual_seed(3))
        b = LutConv2d(2, 8, kernel_size=3, parameterization='independent', estimator='sin',
                      sampler='random8', generator=torch.Generator().manual_seed(3))
        assert torch.equal(a.weights, b.weights)
        assert torch.equal(a.indices_flat, b.indices_flat)
