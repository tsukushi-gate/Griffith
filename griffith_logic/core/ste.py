"""勾配操作系のカスタムautograd(STE等)。"""

import torch


class GradFactor(torch.autograd.Function):
    """勾配スケーリング: forwardは恒等、backwardで勾配にfを乗算"""

    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None
