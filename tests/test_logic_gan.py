"""LogicGAN(GANの論理ゲート化)のテスト。

検証項目:
    1. shape契約(Generator/Discriminator/generate)
    2. 勾配疎通: g_loss で Generator の全 LutLayer に勾配 / d_loss で Discriminator に勾配
       (fake は detach され Generator へ流れないことも確認)
    3. hard離散性: eval時 Generator 出力が厳密に {0,1}(純ブール回路)
    4. 学習ダイナミクスのスモーク: 4パターンのトイ分布で敵対学習が崩壊しない健全性
       (G loss が発散せず有界 / D精度が完全分離に張り付かない)
    5. 再現性: generator 固定で generate が一致
"""

import math
import statistics

import pytest
import torch

from griffith_logic.layers.lut import LutLayer
from griffith_logic.models.logic_gan import LogicGAN, LogicGANConfig


def _tiny_config(**kwargs) -> LogicGANConfig:
    base = dict(
        image_size=8, in_channels=1, noise_bits=16,
        hidden=32, depth=2, disc_hidden=32, disc_depth=2,
    )
    base.update(kwargs)
    return LogicGANConfig(**base)


def _lut_layers(module: torch.nn.Module) -> list[LutLayer]:
    """module 配下の全 LutLayer を列挙する。"""
    return [m for m in module.modules() if isinstance(m, LutLayer)]


def _toy_patterns(size: int = 8) -> torch.Tensor:
    """互いに異なる4つの二値トイ画像(上/下/左/右の半分)。"""
    pats = torch.zeros(4, 1, size, size)
    half = size // 2
    pats[0, 0, :half, :] = 1.0   # 上半分
    pats[1, 0, half:, :] = 1.0   # 下半分
    pats[2, 0, :, :half] = 1.0   # 左半分
    pats[3, 0, :, half:] = 1.0   # 右半分
    return pats


# ------------------------------------------------------------------
# 1. shape契約
# ------------------------------------------------------------------
def test_shape_contract():
    cfg = _tiny_config()
    gan = LogicGAN(cfg)

    gan.train()
    z = gan.generator.sample_noise(5)
    assert z.shape == (5, cfg.noise_bits)
    # 乱数ビット列は {0,1}
    assert set(torch.unique(z).tolist()) <= {0.0, 1.0}

    fake = gan.generator(z)
    assert fake.shape == (5, cfg.in_channels, cfg.image_size, cfg.image_size)
    # soft出力は画素確率 [0,1]
    assert float(fake.detach().min()) >= 0.0 and float(fake.detach().max()) <= 1.0

    logits = gan.discriminator(fake)
    assert logits.shape == (5, 1)

    gan.eval()
    gen = gan.generate(7)
    assert gen.shape == (7, cfg.in_channels, cfg.image_size, cfg.image_size)


# ------------------------------------------------------------------
# 2. 勾配疎通(g_loss→Generator全層 / d_loss→Discriminatorのみ)
# ------------------------------------------------------------------
def test_g_loss_grad_reaches_all_generator_luts():
    cfg = _tiny_config()
    gan = LogicGAN(cfg)
    gan.train()

    z = gan.generator.sample_noise(8)
    fake = gan.generator(z)
    fake_logits = gan.discriminator(fake)  # detach しない(勾配を G まで通す)

    gan.zero_grad()
    gan.g_loss(fake_logits).backward()

    gen_luts = _lut_layers(gan.generator)
    assert len(gen_luts) == cfg.depth + 1  # body(depth) + readout
    for i, lut in enumerate(gen_luts):
        assert lut.weights.grad is not None, f'Generator LUT[{i}] に勾配が届いていない'
        assert torch.isfinite(lut.weights.grad).all()
        assert float(lut.weights.grad.abs().sum()) > 0.0, \
            f'Generator LUT[{i}] の勾配がゼロ'


def test_d_loss_grad_to_discriminator_and_detached_from_generator():
    cfg = _tiny_config()
    gan = LogicGAN(cfg)
    gan.train()

    real = (torch.rand(8, 1, cfg.image_size, cfg.image_size) > 0.5).float()
    z = gan.generator.sample_noise(8)
    fake = gan.generator(z)

    gan.zero_grad()
    gan.d_loss(real, fake).backward()

    # Discriminator には勾配が届く
    for lut in _lut_layers(gan.discriminator):
        assert lut.weights.grad is not None
        assert float(lut.weights.grad.abs().sum()) > 0.0
    assert gan.discriminator.logit_bias.grad is not None

    # Generator へは流れない(fake.detach() の確認)
    for lut in _lut_layers(gan.generator):
        assert lut.weights.grad is None, \
            'd_loss の勾配が Generator へ漏れている(detach されていない)'


# ------------------------------------------------------------------
# 3. hard離散性(eval時 Generator 出力が厳密 {0,1})
# ------------------------------------------------------------------
def test_generator_hard_output_is_binary():
    cfg = _tiny_config()
    gan = LogicGAN(cfg)

    gan.eval()
    z = gan.generator.sample_noise(6)
    fake = gan.generator(z)
    uniq = torch.unique(fake.detach())
    assert set(uniq.tolist()) <= {0.0, 1.0}, f'eval時に非二値の出力: {uniq}'

    # generate 経路も厳密に {0,1}
    gen = gan.generate(6)
    assert set(torch.unique(gen).tolist()) <= {0.0, 1.0}


# ------------------------------------------------------------------
# 4. 学習ダイナミクスのスモーク(4トイパターンで敵対学習が崩壊しない)
# ------------------------------------------------------------------
@pytest.mark.learning
def test_training_dynamics_smoke():
    """極小タスクで敵対学習を回し、健全性(崩壊しない)を確認する。

    GAN の完全収束は要求しない。既知の失敗モードは「Discriminator が Generator を
    圧倒し G loss が発散(本設定の未調整版で 0.85→5.9)」する状態。安定化(G:D=3:1、
    D側lr↓、入力ビット反転ノイズ、label smoothing)を入れると、seed=0 で
    late g_loss≈0.76・late D精度≈0.78 に収まる(実測較正)。以下はその健全域を
    余裕をもって外れないことを確認する(発散しない=Gが淘汰されない健全性)。
    """
    torch.manual_seed(0)
    size = 8
    pats = _toy_patterns(size)
    cfg = LogicGANConfig(
        image_size=size, in_channels=1, noise_bits=16,
        hidden=64, depth=2, disc_hidden=32, disc_depth=1,
    )
    gan = LogicGAN(cfg)
    opt_g = torch.optim.Adam(gan.generator.parameters(), lr=0.01)
    opt_d = torch.optim.Adam(gan.discriminator.parameters(), lr=0.003)

    def bit_flip(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """D入力へのインスタンスノイズ(ビット反転)。D過学習を抑え game を拮抗させる。"""
        mask = (torch.rand_like(x) < p).float()
        return x * (1.0 - mask) + (1.0 - x) * mask

    g_losses: list[float] = []
    d_losses: list[float] = []
    d_accs: list[float] = []
    gan.train()
    for _ in range(400):
        real = pats[torch.randint(0, 4, (16,))]
        z = gan.generator.sample_noise(16)
        fake = gan.generator(z)
        opt_d.zero_grad()
        ld = gan.d_loss(bit_flip(real), bit_flip(fake), label_smoothing=0.1)
        ld.backward()
        opt_d.step()

        # G:D = 3:1(Generator を多めに更新して敵対バランスを保つ)
        for _ in range(3):
            z = gan.generator.sample_noise(16)
            fake = gan.generator(z)
            lg = gan.g_loss(gan.discriminator(fake))
            opt_g.zero_grad()
            lg.backward()
            opt_g.step()

        g_losses.append(lg.item())
        d_losses.append(ld.item())
        with torch.no_grad():
            real_acc = (gan.discriminator(real) > 0).float().mean().item()
            fake_acc = (gan.discriminator(fake) < 0).float().mean().item()
            d_accs.append((real_acc + fake_acc) / 2.0)

    # 全損失が有限(NaN/Inf で崩壊していない)
    assert all(math.isfinite(v) for v in g_losses + d_losses + d_accs)

    late_gloss = statistics.mean(g_losses[-50:])
    late_dacc = statistics.mean(d_accs[-50:])
    # 健全性1: G loss が有界(未調整版の発散 >5 に対し、拮抗していれば < 2.0)
    assert late_gloss < 2.0, f'G loss が発散(敵対バランス崩壊): {late_gloss}'
    # 健全性2: D が完全分離に張り付いていない(game が拮抗している)
    assert late_dacc < 0.99, f'Discriminator が Generator を圧倒(D精度={late_dacc})'


# ------------------------------------------------------------------
# 5. 再現性(generator 固定で generate 一致)
# ------------------------------------------------------------------
def test_generate_reproducible():
    cfg = _tiny_config()
    gan = LogicGAN(cfg)
    gan.eval()

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out1 = gan.generate(5, generator=g1)
    out2 = gan.generate(5, generator=g2)
    assert torch.equal(out1, out2), 'generator 固定でも generate が一致しない'

    # 異なるseedでは(ほぼ)異なる出力になる
    g3 = torch.Generator().manual_seed(7)
    out3 = gan.generate(5, generator=g3)
    assert not torch.equal(out1, out3)
