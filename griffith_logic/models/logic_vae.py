"""LogicVAE — a VAE with a binary latent and logic-circuit encoder/decoder.

VAE(変分オートエンコーダ)を微分可能論理ゲート(LUTツリー)で構成した生成モデル。
生成系のBNN化(BinaryVAE等)は存在するが、エンコーダ/デコーダを LUT スタックで組み、
潜在を Bernoulli ビット列にした構成の先行研究は 2026-09 時点で確認できていない
(最近接は Latent Bernoulli AE(ICML 2020)= 二値潜在×通常NN、DiffLogic CA 2506.04912
= 論理ゲートだが VAE ではない)。主張の強度・最近接文献は監査ページ world-firsts(B2)を
参照し、対外的な文言はそちらの限定形に従うこと。

技術的困難と本実装の主張:
    連続ガウス潜在の再パラメータ化(z = μ + σ·ε)は乗算・平方根を要し、論理回路と相性が最悪。
    本実装は **潜在を Bernoulli ビット列** にすることでこの困難を回避する。
    Bernoulli サンプリングは「u ~ U(0,1) に対する確率的閾値比較 z = 1_{q>u}」であり、
    これは**比較器**そのもの — 論理回路と厳密に整合する(オートエンコーダの前例あり:
    experiments/train_autoencoder.py, MNIST MSE 0.0863)。

設計要点:
    - エンコーダ: 画素[0,1] → Thermometer符号化(encode_bits) → LutLayer スタック(幅 hidden × depth)
      → readout LutLayer + GroupSum で latent_bits 本のロジット q(z=1|x) を実数化する。
    - サンプラー: straight-through Bernoulli(比較器 forward + sigmoid-STE backward)。
      理由は _STBernoulli の docstring 参照。潜在は常に {0,1} を維持する。
    - デコーダ: 潜在ビット {0,1} → LutLayer スタック → readout LutLayer + GroupSum で
      画素ロジットを実数化する(周辺FP方針: 最終の和のみ実数、中間は論理)。
    - ELBO: recon = BCEWithLogits、KL = 解析形 KL(Bern(q) ‖ Bern(0.5))
      (prior = 各ビット独立 Bernoulli(0.5))。
    - eval時、全 LutLayer は真理値表を離散化した純ブール回路として動くため、{0,1}入力に対し
      中間表現は {0,1} を維持する。GroupSum の和のみ整数ロジットになる(周辺FP)。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding.thermometer import ThermometerEncoder
from ..group_sum import GroupSum
from ..layers.lut import LutLayer

# readout(GroupSum)の1出力あたりに割り当てる LUT チャンネル数。大きいほど実数化された
# ロジットの分解能が上がる(sum([0,1]×GROUP) を中央化して符号付きロジットにする)。
_ENC_GROUP = 8   # 潜在ロジット用
_DEC_GROUP = 8   # 画素ロジット用


@dataclass
class LogicVAEConfig:
    """LogicVAE のハイパーパラメータ。"""

    image_size: int = 28
    in_channels: int = 1
    encode_bits: int = 4         # Thermometer符号化のビット数(画素あたり)
    latent_bits: int = 64        # 二値潜在ビット数
    hidden: int = 256            # エンコーダ/デコーダのLutLayer幅
    depth: int = 2               # LutLayerスタックの段数
    n_inputs: int = 2            # LUTの入力数(2/4/6)
    parameterization: str = 'softmax'
    kl_weight: float = 1.0       # βに相当(β-VAE)


class _STBernoulli(torch.autograd.Function):
    """Bernoulliサンプリングの straight-through 推定量。

    Forward: q = σ(logits) と一様ノイズ u に対し z = 1_{q>u} を返す。これは
        「確率的閾値比較=比較器」であり、Bernoulli(q) からの厳密なサンプルである
        (E[z] = P(u<q) = q)。出力は厳密に {0,1}(離散潜在を保証)。
    Backward: ∂z/∂logits ≈ ∂q/∂logits = q(1-q) の STE。ハードなサンプルを
        通しつつ勾配をエンコーダまで流す(離散サンプリングの非微分性を回避)。

    Gumbel-Sigmoid ではなくこの方式を採る理由: 本モデルの主張は「サンプリング=比較器」
    であり、u~U(0,1) の閾値比較そのものが論理回路(比較器)に対応する。Gumbel は
    logistic ノイズの加算+温度付き sigmoid で近似的だが、比較器という主張を素直に
    体現するのは STE-Bernoulli であり、実装も温度スケジュール不要で単純。
    """

    @staticmethod
    def forward(ctx, logits, noise):
        q = torch.sigmoid(logits)
        z = (q > noise).to(logits.dtype)  # 比較器: 確率的閾値比較
        ctx.save_for_backward(q)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        (q,) = ctx.saved_tensors
        # STE: z を q で近似して勾配を流す(∂q/∂logits = q(1-q))
        return grad_output * q * (1.0 - q), None


class LogicVAE(nn.Module):
    """二値潜在VAEの論理ゲート化。

    (B, C, H, W)[0,1] を受け、Bernoulli 二値潜在 (B, latent_bits){0,1} を経由して
    画素を再構成する。prior Bernoulli(0.5) からのサンプリングで新規画像を生成できる。
    """

    def __init__(self, config: LogicVAEConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError(f'depth は1以上: {config.depth}')
        if config.latent_bits < 1:
            raise ValueError(f'latent_bits は1以上: {config.latent_bits}')
        self.config = config

        self.pixel_dim = config.in_channels * config.image_size * config.image_size
        self.latent_bits = config.latent_bits

        llkw = dict(
            n_inputs=config.n_inputs,
            parameterization=config.parameterization,
            connections='random',
        )

        # --- エンコーダ: 画素 → Thermometer → LutLayer スタック → 潜在ロジット ---
        self.encoder_input = ThermometerEncoder(bits=config.encode_bits)
        enc_layers: list[nn.Module] = []
        prev = self.pixel_dim * config.encode_bits
        for _ in range(config.depth):
            enc_layers.append(LutLayer(in_dim=prev, out_dim=config.hidden, **llkw))
            prev = config.hidden
        self.encoder_body = nn.Sequential(*enc_layers)
        # readout: latent_bits 本のロジットを GroupSum で実数化
        self.encoder_readout = LutLayer(
            in_dim=config.hidden, out_dim=config.latent_bits * _ENC_GROUP, **llkw
        )
        self.encoder_group = GroupSum(k=config.latent_bits)
        # クラス毎(=ビット毎)の基準発火率を較正する学習可能バイアス(readout の一部)
        self.latent_bias = nn.Parameter(torch.zeros(config.latent_bits))

        # --- デコーダ: 潜在ビット → LutLayer スタック → 画素ロジット ---
        dec_layers: list[nn.Module] = []
        prev = config.latent_bits
        for _ in range(config.depth):
            dec_layers.append(LutLayer(in_dim=prev, out_dim=config.hidden, **llkw))
            prev = config.hidden
        self.decoder_body = nn.Sequential(*dec_layers)
        self.decoder_readout = LutLayer(
            in_dim=config.hidden, out_dim=self.pixel_dim * _DEC_GROUP, **llkw
        )
        self.decoder_group = GroupSum(k=self.pixel_dim)
        self.pixel_bias = nn.Parameter(torch.zeros(self.pixel_dim))

    # ------------------------------------------------------------------
    # エンコード / サンプリング / デコード
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W)[0,1] → q(z=1|x) のロジット (B, latent_bits)。"""
        h = x.reshape(x.shape[0], -1)          # (B, pixel_dim)
        h = self.encoder_input(h)              # (B, pixel_dim*encode_bits) {0,1}
        h = self.encoder_body(h)               # (B, hidden)
        h = self.encoder_readout(h)            # (B, latent_bits*ENC_GROUP)
        h = self.encoder_group(h)              # (B, latent_bits), 値域[0, ENC_GROUP]
        # 中央化して符号付きロジット化(+較正バイアス)
        return h - _ENC_GROUP / 2.0 + self.latent_bias

    def sample_latent(
        self, logits: torch.Tensor, *, hard: bool, generator=None
    ) -> torch.Tensor:
        """Bernoulli(σ(logits)) から {0,1} をサンプルする。

        学習時(training): straight-through Bernoulli(微分可能、比較器 forward)。
            勾配はサンプラーを通ってエンコーダまで流れる(STの要)。
        eval時: generator を与えれば確率的サンプル(比較器)、無ければ
            hard=True で決定的(σ>0.5)、hard=False で既定RNGの確率的サンプル。
        """
        if self.training:
            noise = torch.rand(
                logits.shape, generator=generator,
                device=logits.device, dtype=logits.dtype,
            )
            return _STBernoulli.apply(logits, noise)

        q = torch.sigmoid(logits)
        if generator is not None:
            noise = torch.rand(
                logits.shape, generator=generator,
                device=logits.device, dtype=logits.dtype,
            )
            return (q > noise).to(logits.dtype)
        if hard:
            return (q > 0.5).to(logits.dtype)
        noise = torch.rand_like(logits)
        return (q > noise).to(logits.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_bits){0,1} → 画素ロジット (B, C, H, W)。"""
        h = self.decoder_body(z)               # (B, hidden)
        h = self.decoder_readout(h)            # (B, pixel_dim*DEC_GROUP)
        h = self.decoder_group(h)              # (B, pixel_dim), 値域[0, DEC_GROUP]
        h = h - _DEC_GROUP / 2.0 + self.pixel_bias
        return h.reshape(
            z.shape[0], self.config.in_channels,
            self.config.image_size, self.config.image_size,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """再構成の1パス。{'recon_logits','latent_logits','z'} を返す。"""
        logits = self.encode(x)
        z = self.sample_latent(logits, hard=True)
        recon_logits = self.decode(z)
        return {'recon_logits': recon_logits, 'latent_logits': logits, 'z': z}

    # ------------------------------------------------------------------
    # 損失(ELBO)
    # ------------------------------------------------------------------
    @staticmethod
    def kl_bernoulli_uniform(logits: torch.Tensor) -> torch.Tensor:
        """解析形 KL(Bern(σ(logits)) ‖ Bern(0.5)) を要素ごとに返す。

        KL = q·log(q/0.5) + (1-q)·log((1-q)/0.5)。q=0.5 で 0、q→0/1 で log2 に増加。
        """
        eps = 1e-7
        q = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        return q * torch.log(q / 0.5) + (1.0 - q) * torch.log((1.0 - q) / 0.5)

    def elbo_loss(self, x: torch.Tensor, out: dict) -> dict:
        """ELBO 損失。{'loss','recon','kl'} を返す(すべてサンプル平均のスカラ)。"""
        # recon: BCEWithLogits を画素で総和、バッチで平均
        recon_per = F.binary_cross_entropy_with_logits(
            out['recon_logits'], x, reduction='none'
        )
        recon = recon_per.flatten(1).sum(dim=1).mean()
        # KL: 潜在ビットで総和、バッチで平均
        kl = self.kl_bernoulli_uniform(out['latent_logits']).sum(dim=1).mean()
        loss = recon + self.config.kl_weight * kl
        return {'loss': loss, 'recon': recon, 'kl': kl}

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, n: int, *, generator=None) -> torch.Tensor:
        """prior Bernoulli(0.5) から z をサンプルしデコード → (n,C,H,W) 画素確率。"""
        device = self.pixel_bias.device
        # 比較器で prior サンプル(u<0.5 で 1)。{0,1} を厳密に維持
        noise = torch.rand(
            (n, self.latent_bits), generator=generator,
            device=device, dtype=self.pixel_bias.dtype,
        )
        z = (noise < 0.5).to(self.pixel_bias.dtype)
        was_training = self.training
        self.eval()  # 純ブール回路としてデコード
        logits = self.decode(z)
        if was_training:
            self.train()
        return torch.sigmoid(logits)
