"""LogicGAN — a GAN whose generator and discriminator are both learned logic circuits.

GAN(敵対的生成ネットワーク)を微分可能論理ゲート(LUTツリー)で構成した生成モデル。
生成系のBNN化(二値化GAN)には断片的な先行例があるが、Generator/Discriminator の**双方**を
LUTスタックで組み、**乱数ビット列→論理回路→画像ビット** とした構成の先行研究は
2026-09 時点で確認できていない(最近接は BinaryGAN 1810.04714 = 出力層のみ二値で G/D は MLP)。
主張の強度・最近接文献は監査ページ world-firsts(B3)を参照し、対外的な文言はそちらの
限定形に従うこと。

技術的困難:
    GANは「敵対的勾配(min-max)」×「離散出力」の二重の不安定性を抱える。二値化GANすら
    学習が崩れやすいことが知られており、完全論理ゲート化はさらに困難である。

本実装の設計方針:
    - Generator を「ランダムビット列 → 論理回路 → 画像ビット」とする。ノイズ源 z は
      連続ガウスではなく **乱数ビット列**(Bernoulli(0.5))であり、論理回路にとって最も
      自然な生成器の形になる(乱数ビット源 = 物理乱数/LFSR で実装可能)。
    - Generator は Thermometer 等の前処理なしに z(既に {0,1})を直接 LutLayer スタックへ
      通し、最終 LutLayer の出力をそのまま画素確率とする。学習時(soft経路)は LutLayer の
      連続出力([0,1] = 画素確率)、eval時(hard経路)は純ブール回路として厳密に {0,1} を返す。
    - Discriminator は画素 [0,1] を LutLayer スタックへ通し、readout LutLayer + GroupSum で
      real/fake ロジットを実数化する(周辺FP方針: 最終の和のみ実数、中間は論理)。
    - 損失は non-saturating GAN loss(BCEWithLogits)。real 側も binary MNIST(>0.5)で
      分布を揃える(Generator の hard 出力 {0,1} と整合)。

品質・安定性:
    敵対的勾配×離散出力の不安定性は既知の研究課題であり、本実装の受け入れは構造契約+
    学習ダイナミクスの健全性(スモーク)までとする。完全収束の品質基準は本走で較正する。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..group_sum import GroupSum
from ..layers.lut import LutLayer

# Discriminator の readout: 1つの real/fake ロジットに割り当てる LUT チャンネル数。
# GroupSum で {0,1}×_DISC_GROUP を合計し中央化して符号付きロジットにする(大きいほど分解能↑)。
_DISC_GROUP = 64


@dataclass
class LogicGANConfig:
    """LogicGAN のハイパーパラメータ。"""

    image_size: int = 28
    in_channels: int = 1
    noise_bits: int = 64          # 入力乱数ビット列の長さ
    hidden: int = 512             # Generator の LutLayer 幅
    depth: int = 3                # Generator の LutLayer スタック段数
    n_inputs: int = 2             # LUT の入力数(2/4/6)
    parameterization: str = 'softmax'
    disc_hidden: int = 256        # Discriminator の LutLayer 幅
    disc_depth: int = 2           # Discriminator の LutLayer スタック段数


def _lut_kwargs(config: LogicGANConfig) -> dict:
    """LutLayer 共通引数(接続はランダム)。"""
    return dict(
        n_inputs=config.n_inputs,
        parameterization=config.parameterization,
        connections='random',
    )


class LogicGenerator(nn.Module):
    """乱数ビット列 → 論理回路 → 画像ビット の Generator。

    z:(B, noise_bits){0,1} を Thermometer なしで直接 LutLayer スタックへ通し、
    最終 LutLayer の出力をそのまま画素確率/ビット (B, C, H, W) とする。
    学習時は soft([0,1] = LutLayer の連続出力 = 画素確率)、eval時は hard({0,1} =
    純ブール回路)。中間表現は eval 時 {0,1} を維持する(最終出力も {0,1})。
    """

    def __init__(self, config: LogicGANConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError(f'depth は1以上: {config.depth}')
        if config.noise_bits < 1:
            raise ValueError(f'noise_bits は1以上: {config.noise_bits}')
        self.config = config
        self.pixel_dim = config.in_channels * config.image_size * config.image_size
        self.noise_bits = config.noise_bits

        llkw = _lut_kwargs(config)

        # body: 乱数ビット → LutLayer スタック(幅 hidden × depth)
        body: list[nn.Module] = []
        prev = config.noise_bits
        for _ in range(config.depth):
            body.append(LutLayer(in_dim=prev, out_dim=config.hidden, **llkw))
            prev = config.hidden
        self.body = nn.Sequential(*body)
        # readout: 画素数ぶんの LUT。soft出力[0,1]=画素確率、hard出力{0,1}=画素ビット
        self.readout = LutLayer(in_dim=config.hidden, out_dim=self.pixel_dim, **llkw)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z:(B, noise_bits){0,1} → (B, C, H, W)。train=soft[0,1] / eval=hard{0,1}。"""
        h = self.body(z)                # (B, hidden)
        h = self.readout(h)             # (B, pixel_dim), soft[0,1] or hard{0,1}
        return h.reshape(
            z.shape[0], self.config.in_channels,
            self.config.image_size, self.config.image_size,
        )

    def sample_noise(self, n: int, *, generator=None) -> torch.Tensor:
        """乱数ビット列 z ~ Bernoulli(0.5) を (n, noise_bits){0,1} で返す(比較器サンプル)。"""
        device = self.readout.weights.device
        dtype = self.readout.weights.dtype
        noise = torch.rand(
            (n, self.noise_bits), generator=generator, device=device, dtype=dtype,
        )
        return (noise < 0.5).to(dtype)  # 比較器: u<0.5 で 1(乱数ビット源)

    @torch.no_grad()
    def generate(self, n: int, *, generator=None) -> torch.Tensor:
        """乱数ビット列をサンプルし hard 経路(純ブール回路)で生成 → (n, C, H, W){0,1}。"""
        z = self.sample_noise(n, generator=generator)
        was_training = self.training
        self.eval()  # 純ブール回路として生成(厳密に {0,1})
        img = self.forward(z)
        if was_training:
            self.train()
        return img


class LogicDiscriminator(nn.Module):
    """画素 [0,1] → real/fake ロジットの Discriminator。

    (B, C, H, W)[0,1] を平坦化し LutLayer スタックへ通し、readout LutLayer + GroupSum で
    1本の real/fake ロジット (B, 1) を実数化する(周辺FP: 和のみ実数、中間は論理)。
    """

    def __init__(self, config: LogicGANConfig) -> None:
        super().__init__()
        if config.disc_depth < 1:
            raise ValueError(f'disc_depth は1以上: {config.disc_depth}')
        self.config = config
        self.pixel_dim = config.in_channels * config.image_size * config.image_size

        llkw = _lut_kwargs(config)

        body: list[nn.Module] = []
        prev = self.pixel_dim
        for _ in range(config.disc_depth):
            body.append(LutLayer(in_dim=prev, out_dim=config.disc_hidden, **llkw))
            prev = config.disc_hidden
        self.body = nn.Sequential(*body)
        # readout: 1本のロジットを GroupSum で実数化
        self.readout = LutLayer(in_dim=config.disc_hidden, out_dim=_DISC_GROUP, **llkw)
        self.group = GroupSum(k=1)
        # 基準発火率を較正する学習可能バイアス(readout の一部)
        self.logit_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W)[0,1] → real/fake ロジット (B, 1)。"""
        h = x.reshape(x.shape[0], -1)   # (B, pixel_dim)
        h = self.body(h)                # (B, disc_hidden)
        h = self.readout(h)             # (B, _DISC_GROUP)
        h = self.group(h)               # (B, 1), 値域[0, _DISC_GROUP]
        return h - _DISC_GROUP / 2.0 + self.logit_bias


class LogicGAN(nn.Module):
    """LogicGAN: Generator + Discriminator の束ね + 損失ヘルパ。

    Generator は乱数ビット列から画像を生成し、Discriminator が real/fake を判定する。
    損失は non-saturating GAN loss(BCEWithLogits)。
    """

    def __init__(self, config: LogicGANConfig) -> None:
        super().__init__()
        self.config = config
        self.generator = LogicGenerator(config)
        self.discriminator = LogicDiscriminator(config)

    # ------------------------------------------------------------------
    # 損失(non-saturating GAN loss)
    # ------------------------------------------------------------------
    def d_loss(
        self, real: torch.Tensor, fake: torch.Tensor, *, label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """Discriminator の損失: BCE(D(real), 1) + BCE(D(fake.detach()), 0)。

        fake は detach するため、この損失の勾配は Generator へ流れない(D のみ更新)。
        label_smoothing>0 で real ターゲットを 1-label_smoothing に緩める(D 過学習抑制)。
        """
        real_logits = self.discriminator(real)
        fake_logits = self.discriminator(fake.detach())
        real_target = torch.full_like(real_logits, 1.0 - label_smoothing)
        fake_target = torch.zeros_like(fake_logits)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_target)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_target)
        return loss_real + loss_fake

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator の non-saturating 損失: BCE(D(fake), 1) = softplus(-fake_logits)。

        飽和する min log(1-D) ではなく max log(D) を採り、初期勾配消失を回避する。
        fake_logits は detach しない D(G(z)) を渡すこと(勾配が G まで流れる)。
        """
        target = torch.ones_like(fake_logits)
        return F.binary_cross_entropy_with_logits(fake_logits, target)

    # ------------------------------------------------------------------
    # 生成(Generator へ委譲)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, n: int, *, generator=None) -> torch.Tensor:
        """乱数ビット列から hard 経路で生成 → (n, C, H, W){0,1}。"""
        return self.generator.generate(n, generator=generator)
