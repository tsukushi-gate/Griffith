"""LogicUNet(U-Netの完全論理ゲート化・密予測)。

U-Net(エンコーダ/デコーダ+skip接続)を全段 LutConv2d で構成した密予測モデル。
セグメンテーション系のBNN化は存在するが、畳み込み・プーリング・アップサンプル・
skip結合まで含めて完全に論理ゲート(LUTツリー)で組んだU-Netは世界に存在しない
(網羅調査)。密予測(画素毎出力)の論理化はオートエンコーダの前例
(experiments/train_autoencoder_cnn.py)
があり、部品(LutConv2d/OrPooling2d/UpSample2d/SpatialGroupSum)は揃っている。

設計要点:
    - エンコーダ: 各stageで LutConv2d(same padding)→ skip保存 → OrPooling2d で 1/2 に縮小。
      チャンネル幅はstageごとに倍化(base → base*2 → ... → base*2^depth=ボトルネック)。
    - デコーダ: 各stageで UpSample2d(×2, nearest)→ 対応skipを **channel次元でConcat** →
      LutConv2d。幅はエンコーダを鏡写しに縮小する。
    - skip接続の連結はチャンネル方向(dim=1)。blocks.structural.Concat は dim=-1
      (系列/特徴ベクトル向け)なので、空間版として torch.cat(dim=1) を直接使う。
    - 出力ヘッド: LutConv2d で out_channels*head_group チャンネルを作り、SpatialGroupSum で
      クラス毎に集約 → 画素毎ロジット(B, out_channels, H, W)。BCEWithLogits で学習する。
    - eval時、全 LutConv2d は真理値表を one-hot 化した純ブール回路として動くため、
      {0,1}入力に対し中間表現は {0,1} を厳密に維持する(OrPooling=max, UpSample=nearest,
      Concat も {0,1} を保つ)。最終の SpatialGroupSum の和のみ整数ロジットになる。

アップサンプルとプーリングの寸法整合:
    floorプーリングと×2アップサンプルは奇数解像度で寸法がずれる(例 7→3→6≠7)。
    デコーダでは対応するエンコーダskipの空間サイズへ nearest で厳密に合わせ直すため、
    任意の入力H,Wで出力H,Wが保存される({0,1}もnearestで保たれる)。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conv import LutConv2d
from ..pooling import OrPooling2d
from ..spatial_group_sum import SpatialGroupSum
from ..upsample import UpSample2d

# LutConv2d のツリー深さ(リーフ数 = n_inputs^depth)。密予測では受容野より
# ツリー表現力を優先し、計算量を抑えるため浅めに固定する。
_TREE_DEPTH = 2
# 畳み込みカーネルサイズ(same paddingで空間サイズ保存)。
_KERNEL_SIZE = 3
# 出力ヘッドで1クラスあたりに割り当てる LUT チャンネル数(SpatialGroupSum のグループ幅)。
# 大きいほど画素ロジットの分解能が上がる。sum([0,1]×head_group) を中央化してロジット化する。
_HEAD_GROUP = 8


@dataclass
class LogicUNetConfig:
    """LogicUNet のハイパーパラメータ。"""

    in_channels: int = 1
    encode_bits: int = 4
    base_channels: int = 32     # 論理チャネル幅(最浅stage)
    depth: int = 2              # ダウン/アップサンプル段数
    n_inputs: int = 2
    parameterization: str = 'softmax'
    out_channels: int = 1       # セグメンテーションマスクのクラス数(1=前景/背景)
    use_skip: bool = True       # skip接続の有無(効果検証・アブレーション用)


class LogicUNet(nn.Module):
    """完全論理ゲート版U-Net(密予測)。

    (B, in_channels, H, W)[0,1] を受け、画素毎ロジット (B, out_channels, H, W) を返す。
    """

    def __init__(self, config: LogicUNetConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError(f'depth は1以上: {config.depth}')
        if config.base_channels < 1:
            raise ValueError(f'base_channels は1以上: {config.base_channels}')
        self.config = config

        pad = _KERNEL_SIZE // 2  # same padding

        # チャンネル幅: エンコーダ各stageの出力(=skip幅)は base*2^i、
        # ボトルネックは base*2^depth。
        enc_channels = [config.base_channels * (2 ** i) for i in range(config.depth)]
        bottleneck_channels = config.base_channels * (2 ** config.depth)

        def make_conv(in_ch: int, out_ch: int) -> LutConv2d:
            return LutConv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=_KERNEL_SIZE,
                tree_depth=_TREE_DEPTH,
                padding=pad,
                n_inputs=config.n_inputs,
                parameterization=config.parameterization,
            )

        # --- エンコーダ(LutConv + OrPooling)× depth ---
        self.enc_convs = nn.ModuleList()
        in_ch = config.in_channels
        for i in range(config.depth):
            self.enc_convs.append(make_conv(in_ch, enc_channels[i]))
            in_ch = enc_channels[i]
        self.pool = OrPooling2d(kernel_size=2)

        # --- ボトルネック ---
        self.bottleneck = make_conv(enc_channels[-1], bottleneck_channels)

        # --- デコーダ(UpSample + skip Concat + LutConv)× depth ---
        self.up = UpSample2d(scale_factor=2)
        self.dec_convs = nn.ModuleList()
        prev_ch = bottleneck_channels
        for i in range(config.depth):
            skip_ch = enc_channels[config.depth - 1 - i]
            # skip有効時はConcatで幅が加算される。無効時はupの出力のみ。
            concat_ch = prev_ch + (skip_ch if config.use_skip else 0)
            out_ch = skip_ch  # エンコーダ幅を鏡写しに縮小
            self.dec_convs.append(make_conv(concat_ch, out_ch))
            prev_ch = out_ch

        # --- 出力ヘッド(SpatialGroupSum系で集約) ---
        # prev_ch(= base_channels)から out_channels*head_group を作り、クラス毎に集約。
        self.head_conv = make_conv(prev_ch, config.out_channels * _HEAD_GROUP)
        self.group_sum = SpatialGroupSum(n_outputs=config.out_channels, scale=1.0)
        # sum は [0, head_group] の範囲。中央化(-head_group/2)して符号付きロジットにする。
        self._logit_offset = _HEAD_GROUP / 2.0
        # クラス毎の学習可能バイアス(readout の一部)。基準発火率を較正し、密予測で
        # 前景/背景の偏りを吸収する(rddlgn/reservoir が nn.Linear readout を持つのと同趣旨)。
        self.logit_bias = nn.Parameter(torch.zeros(config.out_channels))

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """x の空間サイズを ref に nearest で合わせる({0,1}を保存)。"""
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, H, W)[0,1] → (B, out_channels, H, W) 画素毎ロジット。"""
        # --- エンコーダ ---
        skips: list[torch.Tensor] = []
        for conv in self.enc_convs:
            x = conv(x)          # same padding: 空間サイズ保存
            skips.append(x)      # プーリング前をskipとして保存(対応解像度)
            x = self.pool(x)     # 1/2 に縮小

        # --- ボトルネック ---
        x = self.bottleneck(x)

        # --- デコーダ ---
        for i, conv in enumerate(self.dec_convs):
            skip = skips[self.config.depth - 1 - i]
            x = self.up(x)                 # ×2 アップサンプル
            x = self._match_size(x, skip)  # skip解像度へ厳密整合(奇数対策)
            if self.config.use_skip:
                x = torch.cat([x, skip], dim=1)  # channel次元でConcat
            x = conv(x)

        # --- 出力ヘッド ---
        x = self.head_conv(x)              # (B, out_channels*head_group, H, W)
        logits = self.group_sum(x)         # (B, out_channels, H, W)
        logits = logits - self._logit_offset
        return logits + self.logit_bias.view(1, -1, 1, 1)
