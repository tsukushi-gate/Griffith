import torch
import torch.nn as nn


class SpatialGroupSum(nn.Module):
    """
    チャンネルをグループ化して合計し、空間次元を保持する層。

    LUT出力（[0,1]）を複数チャンネル分合計することで、
    [0, group_size]の範囲の回帰値を表現する。
    scaleで最終的な値域を調整する。

    GroupSumの空間版（2D特徴マップ対応）。
    """

    def __init__(self, n_outputs: int, scale: float = 1.0):
        """
        Args:
            n_outputs: 出力チャンネル数
            scale: 合計後に掛けるスケール係数
        """
        super().__init__()
        self.n_outputs = n_outputs
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 入力テンソル。C = n_outputs * group_size

        Returns:
            (B, n_outputs, H, W) グループ合計後のテンソル
        """
        B, C, H, W = x.shape
        assert C % self.n_outputs == 0, (
            f'チャンネル数({C})は出力数({self.n_outputs})で割り切れる必要があります'
        )
        group_size = C // self.n_outputs
        return x.reshape(B, self.n_outputs, group_size, H, W).sum(dim=2) * self.scale

    def extra_repr(self):
        return f'n_outputs={self.n_outputs}, scale={self.scale}'
