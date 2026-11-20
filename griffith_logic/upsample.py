import torch.nn as nn
import torch.nn.functional as F


class UpSample2d(nn.Module):
    """
    最近傍補間によるアップサンプリング（BinaryBrainのUpSamplingと等価）。

    MaxPoolingの逆操作として、空間次元をscale_factor倍に拡大する。
    BinaryBrainのfill=True（デフォルト）と同じ動作。

    Args:
        scale_factor: 拡大倍率（デフォルト2）
    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 入力テンソル

        Returns:
            (B, C, H*scale_factor, W*scale_factor) アップサンプリング後のテンソル
        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

    def extra_repr(self):
        return f'scale_factor={self.scale_factor}'
