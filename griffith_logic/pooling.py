import torch
import torch.nn as nn
import torch.nn.functional as F


class OrPooling2d(nn.Module):
    """
    論理ORプーリング（CDDLGN Section 3.1）。

    max poolingによる確率的論理ORの連続緩和。
    L_max(a, b) = max(a, b) は maximum t-conorm。

    推論時のバイナリ値では完全なBoolean OR演算として動作。
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None, padding: int = 0):
        """
        Args:
            kernel_size: プーリングウィンドウのサイズ
            stride: ストライド（Noneの場合はkernel_sizeと同じ）
            padding: ゼロパディング
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width) 入力テンソル、値は[0,1]

        Returns:
            プーリング後のテンソル
        """
        assert x.dim() == 4, f'入力は4次元テンソルである必要があります: {x.dim()}次元'
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'
