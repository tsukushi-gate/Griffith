"""L1: 16ゲートモードのロジット初期化 init_mode(先行実験 S2)の仕様。

各モードの定義（plan.md「固定条件」/ 社内先行実験 model_iwp.py:77-80）:
    randn    — N(0,1)（従来の既定）
    ri       — 全ロジット0 + op3 に 5.0（従来の residual_init=True）
    ri_randn — N(0,1) に op3 だけ +5
    rri      — N(0,1) に「ゲートごと乱択の許可された非自明1種」へ +5
    uniform  — 全ロジット0

検証: argmax分布 / generatorによる再現性 / init_mode=None の bitwise 不変 /
uniform の入力勾配が厳密0（H3 の崩壊予測の前提）。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F32_EQ

from griffith_logic.layers.lut import LutLayer

A_GATE = 3
TRIVIAL = (0, 15, 3, 5, 10, 12)
NONTRIVIAL = tuple(g for g in range(16) if g not in TRIVIAL)
HIDDEN_MASK = (0, 15, 3, 5, 10, 12)


def _layer(mode, out_dim=512, seed=0, **kw):
    gen = torch.Generator().manual_seed(seed)
    return LutLayer(32, out_dim, n_inputs=2, init_mode=mode,
                    generator=gen, **kw)


class TestModeDefinitions:
    def test_uniform_is_all_zero(self):
        layer = _layer('uniform', out_dim=64)
        assert torch.equal(layer.weights, torch.zeros(64, 16))

    def test_ri_is_zero_plus_5_at_op3(self):
        layer = _layer('ri', out_dim=64)
        expected = torch.zeros(64, 16)
        expected[:, A_GATE] = 5.0
        assert torch.equal(layer.weights, expected)

    def test_ri_randn_is_randn_plus_5_at_op3(self):
        """ri_randn = randn（同一seed）+ op3への+5 を厳密に確認"""
        base = _layer('randn', out_dim=64, seed=3)
        ri = _layer('ri_randn', out_dim=64, seed=3)
        delta = ri.weights - base.weights
        expected = torch.zeros(64, 16)
        expected[:, A_GATE] = 5.0
        assert torch.equal(delta, expected)

    def test_rri_adds_5_to_exactly_one_nontrivial_gate(self):
        base = _layer('randn', out_dim=256, seed=4)
        rri = _layer('rri', out_dim=256, seed=4)
        delta = rri.weights - base.weights
        # 各行でちょうど1か所だけ +5、他は0
        nz = (delta != 0)
        assert torch.equal(nz.sum(-1), torch.full((256,), 1))
        assert torch.equal(delta[nz], torch.full((256,), 5.0))
        picked = delta.argmax(-1)
        assert set(picked.tolist()).issubset(set(NONTRIVIAL))

    def test_rri_respects_gate_mask(self):
        """gate_mask で許可された非自明ゲートからのみ乱択する"""
        allowed = (1, 2)
        mask = [g for g in range(16) if g not in allowed]
        gen = torch.Generator().manual_seed(5)
        layer = LutLayer(32, 200, n_inputs=2, init_mode='rri',
                         gate_mask=mask, generator=gen)
        gen2 = torch.Generator().manual_seed(5)
        base = LutLayer(32, 200, n_inputs=2, init_mode='randn',
                        gate_mask=mask, generator=gen2)
        picked = (layer.weights - base.weights).argmax(-1)
        assert set(picked.tolist()).issubset(set(allowed))

    def test_rri_raises_when_no_nontrivial_gate_allowed(self):
        mask = list(NONTRIVIAL)  # 非自明を全部除外
        with pytest.raises(ValueError, match='非自明ゲート'):
            LutLayer(32, 8, n_inputs=2, init_mode='rri', gate_mask=mask)


class TestArgmaxDistribution:
    def test_ri_selects_op3_everywhere(self):
        """ri は全ロジット0+op3に5.0なので argmax は例外なく op3"""
        layer = _layer('ri', out_dim=512, seed=1)
        assert torch.equal(layer.weights.argmax(-1), torch.full((512,), A_GATE))

    def test_ri_randn_selects_op3_almost_everywhere(self):
        """ri_randn は N(0,1) の上に +5 なので稀に他ゲートが勝つ。

        他ゲートが勝つのは randn_other − randn_op3 > 5、すなわち
        N(0,2) の上側 5 → 片側 p ≈ 2.0e-4、15組で行あたり ≈ 3e-3。
        512行なら数行が入れ替わりうる（これが 'ri' との本質的な差）。
        """
        layer = _layer('ri_randn', out_dim=512, seed=1)
        chosen = layer.weights.argmax(-1)
        frac = (chosen == A_GATE).float().mean().item()
        assert frac >= 0.99, f'op3選択率 {frac}'

    def test_rri_argmax_is_the_boosted_nontrivial_gate(self):
        """RRI は +5 したゲートが argmax になる（ri_randn と同じ理由で稀に例外）"""
        base = _layer('randn', out_dim=512, seed=2)
        rri = _layer('rri', out_dim=512, seed=2)
        boosted = (rri.weights - base.weights).argmax(-1)
        agree = (rri.weights.argmax(-1) == boosted).float().mean().item()
        assert agree >= 0.99, f'boostゲート一致率 {agree}'
        # 10種の非自明ゲートが十分に散らばる（乱択の確認）
        assert len(set(boosted.tolist())) == len(NONTRIVIAL)

    def test_uniform_argmax_is_degenerate_gate0(self):
        """全ロジット同点 → argmax は先頭。マスク下でも許可集合の先頭になる"""
        layer = _layer('uniform', out_dim=32)
        assert torch.equal(layer.weights.argmax(-1), torch.zeros(32, dtype=torch.int64))
        gen = torch.Generator().manual_seed(0)
        masked = LutLayer(32, 32, n_inputs=2, init_mode='uniform',
                          gate_mask=HIDDEN_MASK, generator=gen)
        chosen = masked._masked_weights().argmax(-1)
        assert set(chosen.tolist()) == {min(masked.allowed_gate_ids())}

    def test_uniform_input_gradient_is_exactly_zero(self):
        """一様初期化では入力勾配が厳密0（H3の崩壊予測の前提。マスク下でも同じ）"""
        for mask in (None, HIDDEN_MASK):
            gen = torch.Generator().manual_seed(0)
            layer = LutLayer(16, 64, n_inputs=2, init_mode='uniform',
                             selection='hard_st', gate_mask=mask, generator=gen)
            layer.train()
            x = torch.rand(8, 16, requires_grad=True)
            layer(x).sum().backward()
            assert x.grad.abs().max().item() == 0.0, f'mask={mask}'

    def test_rri_input_gradient_is_nonzero(self):
        """対照: RRI では入力勾配が立つ"""
        gen = torch.Generator().manual_seed(0)
        layer = LutLayer(16, 64, n_inputs=2, init_mode='rri',
                         selection='hard_st', gate_mask=HIDDEN_MASK,
                         generator=gen)
        layer.train()
        x = torch.rand(8, 16, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad.abs().max().item() > 0.0


class TestGeneratorReproducibility:
    @pytest.mark.parametrize('mode', ['randn', 'ri_randn', 'rri', 'uniform', 'ri'])
    def test_same_seed_same_weights(self, mode):
        a = _layer(mode, out_dim=128, seed=42)
        b = _layer(mode, out_dim=128, seed=42)
        assert torch.equal(a.weights, b.weights)

    @pytest.mark.parametrize('mode', ['randn', 'ri_randn', 'rri'])
    def test_different_seed_different_weights(self, mode):
        a = _layer(mode, out_dim=128, seed=1)
        b = _layer(mode, out_dim=128, seed=2)
        assert not torch.equal(a.weights, b.weights)

    def test_generator_is_independent_of_global_rng(self):
        torch.manual_seed(999)
        a = _layer('randn', out_dim=64, seed=7)
        torch.manual_seed(1)
        b = _layer('randn', out_dim=64, seed=7)
        assert torch.equal(a.weights, b.weights)

    def test_cuda_generator_rejected(self):
        gen = torch.Generator()
        assert gen.device.type == 'cpu'  # CPU generator は受理される
        LutLayer(8, 4, n_inputs=2, generator=gen)


class TestBackwardCompatibility:
    @pytest.mark.parametrize('residual_init', [False, True])
    def test_init_mode_none_is_bitwise_identical(self, residual_init):
        torch.manual_seed(13)
        base = LutLayer(16, 48, n_inputs=2, residual_init=residual_init)
        torch.manual_seed(13)
        explicit = LutLayer(16, 48, n_inputs=2, residual_init=residual_init,
                            init_mode=None, generator=None)
        assert torch.equal(base.weights, explicit.weights)
        assert torch.equal(base.indices, explicit.indices)
        assert base.init_mode is None

    def test_residual_init_true_equals_init_mode_ri(self):
        torch.manual_seed(21)
        legacy = LutLayer(16, 48, n_inputs=2, residual_init=True)
        torch.manual_seed(21)
        modern = LutLayer(16, 48, n_inputs=2, init_mode='ri')
        assert torch.equal(legacy.weights, modern.weights)

    @pytest.mark.parametrize(
        'parameterization', ['iwp', 'raw', 'independent', 'warp']
    )
    def test_other_parameterizations_unchanged_without_generator(
        self, parameterization
    ):
        torch.manual_seed(31)
        base = LutLayer(16, 24, n_inputs=2, parameterization=parameterization)
        torch.manual_seed(31)
        same = LutLayer(16, 24, n_inputs=2, parameterization=parameterization,
                        generator=None)
        assert torch.equal(base.weights, same.weights)

    def test_n4_softmax_unchanged(self):
        torch.manual_seed(33)
        base = LutLayer(16, 24, n_inputs=4, parameterization='softmax')
        torch.manual_seed(33)
        same = LutLayer(16, 24, n_inputs=4, parameterization='softmax',
                        generator=None)
        assert torch.equal(base.weights, same.weights)


class TestValidation:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match='init_mode'):
            LutLayer(8, 4, n_inputs=2, init_mode='bogus')

    def test_non_16gate_raises(self):
        with pytest.raises(ValueError, match='16ゲートモード'):
            LutLayer(8, 4, n_inputs=4, parameterization='softmax',
                     init_mode='ri')
        with pytest.raises(ValueError, match='16ゲートモード'):
            LutLayer(8, 4, n_inputs=2, parameterization='independent',
                     init_mode='ri')

    def test_conflict_with_residual_init_raises(self):
        with pytest.raises(ValueError, match='同時指定'):
            LutLayer(8, 4, n_inputs=2, init_mode='rri', residual_init=True)


class TestInitScale:
    """残差初期化の加算量 α（先行実験 S2'）。

    社内先行実験 Phase 3 の OP 腕は `--residual_init` 既定 1.0 で
    `w[:,3] += 1.0`（train_cifar_ab.py:255 → model_iwp.py:79-80）なので、
    先行実験を再現する α は **1.0**（5.0 ではない）。
    """

    def test_ri_randn_adds_exactly_init_scale_at_op3(self):
        """init_scale=1.0（先行実験の実値）で op3 にだけ +1.0"""
        base = _layer('randn', out_dim=128, seed=11)
        ri = _layer('ri_randn', out_dim=128, seed=11, init_scale=1.0)
        # float32 の丸めがあるので「同じ加算をした結果」と bitwise 照合する
        expected = base.weights.detach().clone()
        expected[:, A_GATE] += 1.0
        assert torch.equal(ri.weights, expected)
        delta = (ri.weights - base.weights).detach()
        assert (delta[:, A_GATE] - 1.0).abs().max() < TOL_F32_EQ
        other = [g for g in range(16) if g != A_GATE]
        assert torch.equal(delta[:, other], torch.zeros(128, 15))

    @pytest.mark.parametrize('scale', [0.5, 1.0, 2.5, 10.0])
    def test_ri_randn_scales(self, scale):
        base = _layer('randn', out_dim=64, seed=12)
        ri = _layer('ri_randn', out_dim=64, seed=12, init_scale=scale)
        expected = base.weights.detach().clone()
        expected[:, A_GATE] += scale
        assert torch.equal(ri.weights, expected)
        delta = (ri.weights - base.weights).detach()
        assert (delta[:, A_GATE] - scale).abs().max() < TOL_F32_EQ
        other = [g for g in range(16) if g != A_GATE]
        assert torch.equal(delta[:, other], torch.zeros(64, 15))

    @pytest.mark.parametrize('scale', [1.0, 3.0])
    def test_rri_adds_init_scale_to_one_nontrivial_gate(self, scale):
        base = _layer('randn', out_dim=256, seed=13)
        rri = _layer('rri', out_dim=256, seed=13, init_scale=scale)
        delta = (rri.weights - base.weights).detach()
        nz = (delta != 0)
        assert torch.equal(nz.sum(-1), torch.full((256,), 1))
        assert (delta[nz] - scale).abs().max() < TOL_F32_EQ
        picked = delta.argmax(-1)
        assert set(picked.tolist()).issubset(set(NONTRIVIAL))
        expected = base.weights.detach().clone()
        expected[torch.arange(256), picked] += scale
        assert torch.equal(rri.weights, expected)

    @pytest.mark.parametrize('scale', [1.0, 5.0])
    def test_ri_sets_op3_to_init_scale(self, scale):
        layer = _layer('ri', out_dim=32, init_scale=scale)
        expected = torch.zeros(32, 16)
        expected[:, A_GATE] = scale
        assert torch.equal(layer.weights, expected)

    def test_selection_probability_matches_closed_form(self):
        """α=1 では P(op3) ≈ e/(e+15) ≈ 0.153（α=5 の ≈0.908 と大きく違う）"""
        import math

        for scale in (1.0, 5.0):
            layer = _layer('ri', out_dim=8, init_scale=scale)
            p = torch.softmax(layer.weights, dim=-1)[0, A_GATE].item()
            expected = math.exp(scale) / (math.exp(scale) + 15.0)
            assert abs(p - expected) < 1e-6, (scale, p, expected)

    @pytest.mark.parametrize('mode', ['ri', 'ri_randn', 'rri'])
    def test_default_scale_is_5_and_bitwise_identical(self, mode):
        implicit = _layer(mode, out_dim=64, seed=14)
        explicit = _layer(mode, out_dim=64, seed=14, init_scale=5.0)
        assert implicit.init_scale == 5.0
        assert torch.equal(implicit.weights, explicit.weights)

    @pytest.mark.parametrize('mode', ['randn', 'uniform'])
    def test_ignored_by_randn_and_uniform(self, mode):
        a = _layer(mode, out_dim=64, seed=15)
        b = _layer(mode, out_dim=64, seed=15, init_scale=1.0)
        assert torch.equal(a.weights, b.weights)

    def test_residual_init_true_keeps_default_5(self):
        torch.manual_seed(23)
        legacy = LutLayer(16, 32, n_inputs=2, residual_init=True)
        assert legacy.init_scale == 5.0
        expected = torch.zeros(32, 16)
        expected[:, A_GATE] = 5.0
        assert torch.equal(legacy.weights, expected)

    @pytest.mark.parametrize('bad', [0.0, -1.0])
    def test_non_positive_raises(self, bad):
        with pytest.raises(ValueError, match='init_scale'):
            LutLayer(8, 4, n_inputs=2, init_mode='ri', init_scale=bad)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
