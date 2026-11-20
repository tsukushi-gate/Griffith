"""LogicVAE(二値潜在VAEの論理ゲート化)のテスト。

検証項目:
    1. shape契約(encode/decode/forward/generate)
    2. 勾配疎通: サンプラーを通ってエンコーダまで勾配が流れる(STの要)
    3. 潜在の離散性: z が常に {0,1}
    4. KL解析形の正しさ(q=0.5でKL=0、q→0/1で増加、torch参照と数値照合)
    5. 極小収束: 4パターンのトイ画像で再構成BCEが実測較正値以下+潜在がパターンを分離
"""

import math

import torch
import torch.nn.functional as F

from griffith_logic.models.logic_vae import LogicVAE, LogicVAEConfig


def _tiny_config(**kwargs) -> LogicVAEConfig:
    base = dict(image_size=8, latent_bits=16, hidden=32, depth=2)
    base.update(kwargs)
    return LogicVAEConfig(**base)


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
    model = LogicVAE(cfg)
    x = (torch.rand(5, cfg.in_channels, cfg.image_size, cfg.image_size) > 0.5).float()

    model.train()
    logits = model.encode(x)
    assert logits.shape == (5, cfg.latent_bits)

    z = model.sample_latent(logits, hard=True)
    assert z.shape == (5, cfg.latent_bits)

    recon = model.decode(z)
    assert recon.shape == (5, cfg.in_channels, cfg.image_size, cfg.image_size)

    out = model(x)
    assert set(out) == {'recon_logits', 'latent_logits', 'z'}
    assert out['recon_logits'].shape == x.shape
    assert out['latent_logits'].shape == (5, cfg.latent_bits)
    assert out['z'].shape == (5, cfg.latent_bits)

    model.eval()
    gen = model.generate(7)
    assert gen.shape == (7, cfg.in_channels, cfg.image_size, cfg.image_size)
    # 生成は画素確率([0,1])
    assert float(gen.min()) >= 0.0 and float(gen.max()) <= 1.0


# ------------------------------------------------------------------
# 2. 勾配疎通(サンプラーを通ってエンコーダまで)
# ------------------------------------------------------------------
def test_gradient_flows_through_sampler_to_encoder():
    cfg = _tiny_config()
    model = LogicVAE(cfg)
    model.train()
    x = (torch.rand(4, 1, cfg.image_size, cfg.image_size) > 0.5).float()

    out = model(x)
    losses = model.elbo_loss(x, out)
    # recon 経路のみで backward(KL経由でなくサンプラー経由の勾配を検証)
    model.zero_grad()
    losses['recon'].backward()

    enc_grad = model.encoder_body[0].weights.grad
    assert enc_grad is not None, 'エンコーダに勾配が届いていない(STが機能していない)'
    assert torch.isfinite(enc_grad).all()
    assert float(enc_grad.abs().sum()) > 0.0, 'サンプラー経由の勾配がゼロ'


# ------------------------------------------------------------------
# 3. 潜在の離散性
# ------------------------------------------------------------------
def test_latent_is_binary():
    cfg = _tiny_config()
    model = LogicVAE(cfg)
    x = (torch.rand(6, 1, cfg.image_size, cfg.image_size) > 0.5).float()

    def _assert_binary(z):
        uniq = torch.unique(z.detach())
        assert set(uniq.tolist()) <= {0.0, 1.0}, f'非二値の潜在: {uniq}'

    # 学習時(ST forward)
    model.train()
    logits = model.encode(x)
    _assert_binary(model.sample_latent(logits, hard=True))
    _assert_binary(model(x)['z'])

    # eval時(決定的・確率的の両方)
    model.eval()
    logits = model.encode(x)
    _assert_binary(model.sample_latent(logits, hard=True))
    gen = torch.Generator().manual_seed(0)
    _assert_binary(model.sample_latent(logits, hard=False, generator=gen))


# ------------------------------------------------------------------
# 4. KL解析形の正しさ
# ------------------------------------------------------------------
def test_kl_analytic_form():
    # q=0.5 (logit=0) で KL=0
    kl_half = LogicVAE.kl_bernoulli_uniform(torch.zeros(10))
    assert torch.allclose(kl_half, torch.zeros(10), atol=1e-6)

    # q→0/1 で単調増加し、上限は log2
    logits = torch.tensor([0.0, 1.0, 2.0, 4.0, 8.0])
    kl = LogicVAE.kl_bernoulli_uniform(logits)
    # logitが大きいほどKLが大きい(単調増加)
    assert torch.all(kl[1:] > kl[:-1])
    assert float(kl[-1]) < math.log(2.0) + 1e-6
    assert float(kl[-1]) > math.log(2.0) - 1e-2  # 十分飽和

    # torch参照(Bernoulli KL)と数値照合
    logits = torch.tensor([-3.0, -1.0, 0.0, 0.5, 2.5])
    q = torch.sigmoid(logits)
    ref = torch.distributions.kl_divergence(
        torch.distributions.Bernoulli(probs=q),
        torch.distributions.Bernoulli(probs=torch.full_like(q, 0.5)),
    )
    got = LogicVAE.kl_bernoulli_uniform(logits)
    assert torch.allclose(got, ref, atol=1e-5)

    # elbo_loss の kl はビット総和・バッチ平均
    cfg = _tiny_config()
    model = LogicVAE(cfg)
    x = (torch.rand(3, 1, cfg.image_size, cfg.image_size) > 0.5).float()
    out = model(x)
    losses = model.elbo_loss(x, out)
    expected_kl = LogicVAE.kl_bernoulli_uniform(out['latent_logits']).sum(dim=1).mean()
    assert torch.allclose(losses['kl'], expected_kl)
    # loss = recon + β·kl
    assert torch.allclose(
        losses['loss'], losses['recon'] + cfg.kl_weight * losses['kl']
    )


# ------------------------------------------------------------------
# 5. 極小収束(4トイパターンの再構成+潜在分離)
# ------------------------------------------------------------------
def test_tiny_convergence_and_latent_separation():
    torch.manual_seed(0)
    size = 8
    pats = _toy_patterns(size)
    cfg = _tiny_config(
        image_size=size, latent_bits=16, hidden=64, depth=2, kl_weight=0.1
    )
    model = LogicVAE(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(600):
        out = model(pats)
        loss = model.elbo_loss(pats, out)['loss']
        opt.zero_grad()
        loss.backward()
        opt.step()

    # eval: 決定的潜在で再構成
    model.eval()
    with torch.no_grad():
        logits = model.encode(pats)
        z = model.sample_latent(logits, hard=True)
        recon_logits = model.decode(z)
        bce = F.binary_cross_entropy_with_logits(
            recon_logits, pats, reduction='none'
        ).mean().item()

    # 実測較正: 400step で ~0.086, 800step で ~0.071(seed=0)。余裕を持って 0.25 未満
    assert bce < 0.25, f'再構成BCE/px が高すぎる: {bce}'

    # 潜在がパターンを分離(4パターンで異なる z)
    distinct = {tuple(zz.tolist()) for zz in z}
    assert len(distinct) == 4, f'潜在がパターンを分離できていない: {len(distinct)}/4'
