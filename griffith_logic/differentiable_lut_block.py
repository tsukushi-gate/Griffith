import torch
import torch.nn as nn

from .binarize import Binarize
from .lut_layer import LutLayer


class UnbinarizeFunction(torch.autograd.Function):
    """
    Unbinarize: バイナリ入力{0,1}を{0.5-bias, 0.5+bias}にリマップ。

    BinaryBrainのDifferentiableLutN.h:557を忠実に再現:
        x = 0.5 + ((x > 0.5) ? +unbinarize_bias : -unbinarize_bias)

    Forward: 不連続（閾値判定）
    Backward: STE（勾配をそのまま通す）
    """

    @staticmethod
    def forward(ctx, x, unbinarize_bias):
        return 0.5 + torch.where(
            x > 0.5,
            torch.tensor(unbinarize_bias, device=x.device, dtype=x.dtype),
            torch.tensor(-unbinarize_bias, device=x.device, dtype=x.dtype),
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DifferentiableLutBlock(nn.Module):
    """
    BinaryBrainのDifferentiableLutと等価なコンポジット層。

    内部構成（DifferentiableLutN.h Forward計算を忠実に再現）:
        1. Unbinarize: {0,1} → {0.25, 0.75}（入力リマップ）
        2. LutLayer(raw): N入力LUT（重みを直接[0,1]で保持）
        3. BatchNorm1d: gamma=0.3, beta=0.5 で初期化
        4. Binarize: 閾値0.5で二値化、backward時はHardTanh [0, 1]

    Args:
        in_dim: 入力次元数
        out_dim: 出力ニューロン数
        n_inputs: LUTの入力数（デフォルト6）
        device: 計算デバイス
        connections: 接続方式 ('random' or 'unique')
        unbinarize_bias: 入力リマップのバイアス値（デフォルト0.25）
        bn_gamma: BatchNormのgamma初期値（デフォルト0.3）
        bn_beta: BatchNormのbeta初期値（デフォルト0.5）
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_inputs: int = 6,
        device: str = 'cpu',
        connections: str = 'random',
        unbinarize_bias: float = 0.25,
        bn_gamma: float = 0.3,
        bn_beta: float = 0.5,
    ):
        super().__init__()
        self.unbinarize_bias = unbinarize_bias

        self.lut = LutLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            n_inputs=n_inputs,
            device=device,
            connections=connections,
            parameterization='raw',
        )

        # BatchNorm1d: momentum=1.0 はPyTorch規約でBinaryBrainのmomentum=0.0と等価
        # （running_mean = batch_mean、EMAなし）
        self.bn = nn.BatchNorm1d(out_dim, momentum=1.0, device=device)
        # BinaryBrainのデフォルト初期値
        nn.init.constant_(self.bn.weight, bn_gamma)
        nn.init.constant_(self.bn.bias, bn_beta)

        self.binarize = Binarize(
            threshold=0.5, hardtanh_min=0.0, hardtanh_max=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) バイナリまたは実数入力

        Returns:
            (B, out_dim) バイナリ出力 {0, 1}
        """
        # 1. Unbinarize: {0,1} → {0.25, 0.75}
        if self.training:
            x = UnbinarizeFunction.apply(x, self.unbinarize_bias)
        else:
            x = 0.5 + torch.where(
                x > 0.5,
                torch.tensor(self.unbinarize_bias, device=x.device, dtype=x.dtype),
                torch.tensor(-self.unbinarize_bias, device=x.device, dtype=x.dtype),
            )

        # 2. LUT Forward
        x = self.lut(x)

        # 3. BatchNorm
        x = self.bn(x)

        # 4. Binarize
        x = self.binarize(x)

        return x

    def clamp_weights(self):
        """重みを[0,1]にクランプ。BinaryBrainのm_flagClampと等価。"""
        with torch.no_grad():
            self.lut.weights.clamp_(0.0, 1.0)

    def extra_repr(self):
        return f'unbinarize_bias={self.unbinarize_bias}'
