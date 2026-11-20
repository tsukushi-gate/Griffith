"""正規化の独立部品。

[0,1]論理域の途中に挟める正規化層。IRノードにはしない:
    - LutLayer→LogicBatchNorm→Binarize の並びは blocks_to_ir が
      真理値表への折り畳み(プローブfolding)で単一lut_layerノード化する
    - 折り畳めない位置のNorm部品はblackboxノード化する
      (LogicLayerNormは特徴横断の正規化のため常に折り畳み不能)
"""

import torch
import torch.nn as nn


class LogicLayerNorm(nn.Module):
    """[0,1]論理域向けLayerNorm。

    内部で 2x-1 へ写像 → nn.LayerNorm → sigmoid で[0,1]へ戻す
    (FromLogic('layernorm')と同じくnn.LayerNormを正規化の核に使う)。
    LayerNormは特徴横断のため出力ビットが全入力に依存し、
    真理値表への折り畳みは不可能(blocks_to_irではblackboxノード)。
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.norm(2.0 * x - 1.0))

    def extra_repr(self):
        return f'width={self.width}'


class LogicBatchNorm(nn.Module):
    """DifferentiableLutBlockのBN部を独立化した[0,1]論理域向けBatchNorm。

    BatchNorm1d(gamma=0.3, beta=0.5 初期化。BinaryBrain規約)の出力を
    STEクランプで[0,1]へ復帰する(forward=clamp、backward=恒等)。
    eval時はrunning統計による特徴ごとのaffine+クランプで決定的。

    特徴ごとに独立のため、後続がBinarizeなら前段LutLayerの真理値表へ
    折り畳み可能(clampは閾値0.5の比較を変えない)。blocks_to_ir参照。
    3D入力(B, S, D)は(B×S, D)にflattenして処理する。
    """

    def __init__(self, width: int, momentum: float = 0.1,
                 bn_gamma: float = 0.3, bn_beta: float = 0.5):
        super().__init__()
        self.width = width
        self.bn = nn.BatchNorm1d(width, momentum=momentum)
        nn.init.constant_(self.bn.weight, bn_gamma)
        nn.init.constant_(self.bn.bias, bn_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        h = self.bn(x.reshape(-1, self.width))
        # STEクランプ: forward=[0,1]へclamp、backward=恒等
        h = h + (h.clamp(0.0, 1.0) - h).detach()
        return h.reshape(*lead, self.width)

    def extra_repr(self):
        return f'width={self.width}'
