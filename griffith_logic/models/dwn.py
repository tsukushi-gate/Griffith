"""DWN分類器(Bacellar et al., ICML 2024)。

構成: ThermometerEncoder → DWNLayer × L → GroupSum(popcount集約)

公式実装 examples/mnist.py の構成に準拠:
    DistributiveThermometer(3) → LUTLayer(in, 2000, n=6, mapping='learnable')
    → LUTLayer(2000, 1000, n=6) → GroupSum(k=10, tau=1/0.3)
- Learnable Mappingは第1層のみ(公式READMEの推奨。訓練速度と精度のバランス)
- 集約は各クラスに割り当てたLUT出力のpopcount(=和)をτで割ったロジット
"""

import torch
import torch.nn as nn

from ..encoding.thermometer import ThermometerEncoder
from ..group_sum import GroupSum
from ..layers.dwn import DWNLayer
from ..regularization.spectral import spectral_norm

_DEFAULT_TAU = 1.0 / 0.3  # 公式mnist.pyのGroupSum温度


class DWNClassifier(nn.Module):
    """Thermometer符号化 + DWN多層LUT + popcount集約の分類器。

    Args:
        in_features: 実数値入力の特徴数 D
        n_classes: クラス数 k
        layer_widths: 各DWNLayerのLUT数(最終層はn_classesで割り切れること)
        bits: Thermometer符号化のビット数/特徴
        encoder_mode: 'uniform' | 'distributive'(distributiveはfit_encoder必須)
        n_inputs: LUTあたりの入力数 n
        mapping: 第1層の接続方式 'random' | 'learnable'(第2層以降はrandom固定)
        efd: EFD係数 'paper'(1/(H+1)) | 'official'(α·β^H)
        tau: GroupSumの温度(公式既定値 1/0.3)
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        layer_widths: tuple[int, ...] = (1000, 500),
        bits: int = 3,
        encoder_mode: str = 'distributive',
        n_inputs: int = 6,
        mapping: str = 'learnable',
        efd: str = 'paper',
        tau: float = _DEFAULT_TAU,
        device: str = 'cpu',
    ):
        super().__init__()
        assert layer_widths, 'layer_widthsは1層以上'
        assert layer_widths[-1] % n_classes == 0, (
            f'最終層のLUT数({layer_widths[-1]})はクラス数({n_classes})で'
            f'割り切れる必要があります'
        )
        self.in_features = in_features
        self.n_classes = n_classes

        self.encoder = ThermometerEncoder(bits, mode=encoder_mode)
        layers = []
        in_bits = in_features * bits
        for i, width in enumerate(layer_widths):
            layers.append(DWNLayer(
                in_bits, width, n_inputs=n_inputs,
                mapping=mapping if i == 0 else 'random',
                efd=efd, device=device,
            ))
            in_bits = width
        self.layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(k=n_classes, tau=tau)

    def fit_encoder(self, data: torch.Tensor) -> 'DWNClassifier':
        """Thermometerしきい値を訓練データから学習する(メソッドチェーン可)"""
        self.encoder.fit(data)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features) 実数値入力

        Returns:
            (batch, n_classes) 分類ロジット
        """
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h)
        return self.group_sum(h)

    def spectral_penalty(self) -> torch.Tensor:
        """全DWN層のSpectral Regularization項の合計(損失に加算して使う)"""
        return torch.stack([
            spectral_norm(layer.luts, layer.n_inputs) for layer in self.layers
        ]).sum()
