import torch
import torch.nn as nn


class BinarizeFunction(torch.autograd.Function):
    """
    二値化のカスタムautograd（HardTanh STE）。

    BinaryBrainのDifferentiableLutN.h backward（1209-1211行）を忠実に再現:
    - Forward: 閾値で二値化 (x > threshold → 1.0, otherwise → 0.0)
    - Backward: HardTanh STE（hardtanh_min < x < hardtanh_max のとき勾配を通す）
    """

    @staticmethod
    def forward(ctx, x, threshold, hardtanh_min, hardtanh_max):
        ctx.save_for_backward(x)
        ctx.hardtanh_min = hardtanh_min
        ctx.hardtanh_max = hardtanh_max
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # HardTanh: 範囲外の勾配をゼロに
        # BinaryBrain: if (tanh_x <= 0.0) { dy = 0.0; } if (tanh_x >= 1.0) { dy = 0.0; }
        mask = (x > ctx.hardtanh_min) & (x < ctx.hardtanh_max)
        return grad_output * mask.to(grad_output.dtype), None, None, None


class Binarize(nn.Module):
    """
    二値化層（HardTanh STE）。

    BinaryBrainのDifferentiableLut内部で使用される活性化関数。
    訓練時はSTE（Straight-Through Estimator）で勾配を近似、
    推論時は純粋な閾値二値化。

    Args:
        threshold: 二値化の閾値（デフォルト0.5）
        hardtanh_min: HardTanh勾配の下限（デフォルト0.0）
        hardtanh_max: HardTanh勾配の上限（デフォルト1.0）
    """

    def __init__(self, threshold: float = 0.5,
                 hardtanh_min: float = 0.0, hardtanh_max: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return BinarizeFunction.apply(
                x, self.threshold, self.hardtanh_min, self.hardtanh_max,
            )
        return (x > self.threshold).to(x.dtype)

    def extra_repr(self):
        return (f'threshold={self.threshold}, '
                f'hardtanh=[{self.hardtanh_min}, {self.hardtanh_max}]')
