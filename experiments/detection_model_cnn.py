"""
CenterNet 物体検出モデル（通常CNNベースライン）。

CenterNetLUT（微分可能論理ゲート版）と同一の入出力仕様を持つ、
通常のConv+BN+ReLUによる小型CenterNet。DLGN版の比較対照・KD teacher候補。

出力ストライド=4、3つの検出ヘッド（ヒートマップ・サイズ・オフセット）。
出力値域をLUT版と揃えるため:
  - ヒートマップ: sigmoid で [0,1]（focal loss互換）
  - サイズ/オフセット: sigmoid で [0,1]（GT正規化値域と一致）

W14ladder: CNN→DLGN帰属実験（train_detection.py --ladder）のL0/L1/L2段として
使う。downsample='orpool'（既定'stride'から変更）で2x2 stride2 convの代わりに
(stride1 conv + OrPooling2d)を使う（LUT版CenterNetLUT(downsample='orpool')の
OR飽和の影響だけを切り出して測るための対応物）。
"""

import torch
import torch.nn as nn

from griffith_logic import OrPooling2d

DOWNSAMPLES = ('stride', 'orpool')


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CenterNetCNN(nn.Module):
    """
    通常CNN版CenterNet（~1-2Mパラメータ）。

    CenterNetLUTと同じforward署名（dict出力: hm/size/off）。
    バックボーンは出力ストライド4（96→24）。
    """

    def __init__(
        self,
        n_classes: int = 10,
        in_channels: int = 3,
        width: int = 64,
        head_ch: int = 64,
        downsample: str = 'stride',
        **_ignored,
    ):
        """
        Args:
            n_classes: 検出クラス数
            in_channels: 入力チャンネル数（閾値二値化後）
            width: バックボーン基本チャンネル幅
            head_ch: 検出ヘッドの中間チャンネル数
            downsample: バックボーンのダウンサンプル方式（W14ladder、既定
                'stride'で従来挙動と数値不変）。'orpool'では2x2 stride2 conv
                の代わりに(stride1 conv + OrPooling2d(2))を使う
                （CenterNetLUT(downsample='orpool')との対応物、OR飽和の
                影響だけを切り出して測る=--ladder L2）
            _ignored: LUT版と共通のargsを受け流すためのダミー
        """
        super().__init__()
        if downsample not in DOWNSAMPLES:
            raise ValueError(f'downsample は {DOWNSAMPLES} のいずれか: {downsample}')
        self.n_classes = n_classes
        self.downsample = downsample

        w = width
        # バックボーン: 出力ストライド4（/2 を2回）
        if downsample == 'stride':
            self.backbone = nn.Sequential(
                _conv_bn_relu(in_channels, w, stride=1),
                _conv_bn_relu(w, w, stride=2),          # /2
                _conv_bn_relu(w, 2 * w, stride=1),
                _conv_bn_relu(2 * w, 2 * w, stride=2),  # /4
                _conv_bn_relu(2 * w, 4 * w, stride=1),
                _conv_bn_relu(4 * w, 4 * w, stride=1),
            )
        else:
            # 'orpool': 元の6層構成(チャンネル変換)はそのまま残し、ストライドに
            # よる空間ダウンサンプルだけをOrPooling2d(2)へ差し替える(stride
            # convは重み形状がstrideに依らないため、パラメータ数は'stride'と
            # 同一。'stride'のconv(w,w,stride=2)がconv(w,w,stride=1)+
            # OrPooling2d(2)になる、という1変数だけの差し替え)。
            self.backbone = nn.Sequential(
                _conv_bn_relu(in_channels, w, stride=1),
                _conv_bn_relu(w, w, stride=1),
                OrPooling2d(2),                          # /2
                _conv_bn_relu(w, 2 * w, stride=1),
                _conv_bn_relu(2 * w, 2 * w, stride=1),
                OrPooling2d(2),                           # /4
                _conv_bn_relu(2 * w, 4 * w, stride=1),
                _conv_bn_relu(4 * w, 4 * w, stride=1),
            )
        feat_ch = 4 * w

        def head(out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                _conv_bn_relu(feat_ch, head_ch, stride=1),
                nn.Conv2d(head_ch, out_ch, kernel_size=1),
            )

        self.hm_head = head(n_classes)
        self.size_head = head(2)
        self.off_head = head(2)

        # ヒートマップのbias初期化（focal loss安定化: 初期確率を低くする）
        self.hm_head[-1].bias.data.fill_(-2.19)  # sigmoid(-2.19)≈0.1

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, H, W) 入力画像 [0,1]

        Returns:
            'hm': (B, n_classes, H/4, W/4) ヒートマップ [0,1]
            'size': (B, 2, H/4, W/4) 正規化サイズ [0,1]
            'off': (B, 2, H/4, W/4) サブピクセルオフセット [0,1]
        """
        feat = self.backbone(x)
        hm = torch.sigmoid(self.hm_head(feat))
        size = torch.sigmoid(self.size_head(feat))
        off = torch.sigmoid(self.off_head(feat))
        return {'hm': hm, 'size': size, 'off': off}

    @property
    def num_gates(self) -> int:
        """LUT版とのAPI互換のためのダミー（CNNはゲートを持たない）。"""
        return 0
