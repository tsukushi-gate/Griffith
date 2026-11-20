"""残差論理ConvNet(LogicResNet)。

ResNet型の「残差ブロックを積む深いConvNet」の論理ゲート版。CDDLGN(LogicTreeNet)は
Residual Initialization(ゲートをpass-through寄りに初期化)を使うが、構造としての
skip接続(x OR f(x) / x XOR f(x))を明示的に持つ名前付きモデルはライブラリに無かった。
既存部品(LutConv2d/OrPooling2d/blocks.structural.Residual/SpatialGroupSum/GroupSum/
ThermometerEncoder)を組み合わせるのみで、新規のautograd Functionは追加しない。

forward:
    (B,C,H,W)[0,1]
    → thermometer(チャネルごとにencode_bitsビット化)
    → stem LutConv2d
    → [残差ブロック(LutConv2d×2 + Residual(or|xor))]×n を stage_blocks の各stage分繰り返し
      (stage間でOrPooling2dダウンサンプル。幅はResidualの制約上、全stage共通=stem_channels)
    → head LutConv2d(1x1) で num_classes*head_group_size チャネルへ
    → SpatialGroupSum(空間次元を保ったままnum_classesへ縮約)
    → Flatten → GroupSum(空間分をクラスごとに合算) → (B, num_classes)

Residual merge modeの設計判断(or vs xor):
    'or'  — 1-(1-a)(1-b)。inner(x)=0のとき出力=x(恒等)に落ちるため、
            Residual Initializationとの相性が良く恒等写像への収束が容易。
            ただし飽和(a,bが1に近いと勾配が消えやすい)の弱点がある。
    'xor' — a+b-2ab。恒等になる条件が inner(x)=0 のみでは不十分
            (xとinner(x)がビットごとに一致すれば0を出す必要がある=単なる
            pass-through最適解にはなりにくい)反面、飽和しにくく勾配レンジが安定。
    既定は 'or'(ResNetの加算残差に最も近い意味論のため)。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..blocks.structural import Residual
from ..conv import LutConv2d
from ..encoding.thermometer import ThermometerEncoder
from ..group_sum import GroupSum
from ..pooling import OrPooling2d
from ..spatial_group_sum import SpatialGroupSum

# head 1x1 convの出力チャネル数は num_classes*_HEAD_GROUP_SIZE とし、
# SpatialGroupSumで割り切れることを構造的に保証する(表現力のための余剰チャネル)。
_HEAD_GROUP_SIZE = 4


@dataclass
class LogicResNetConfig:
    """LogicResNet のハイパーパラメータ。"""

    in_channels: int = 3
    encode_bits: int = 4          # thermometer符号化/チャネル
    stem_channels: int = 32       # 論理チャネル幅(残差ブロックは幅不変のため全stage共通)
    stage_blocks: tuple = (2, 2)  # 各stageの残差ブロック数(stage間でOrPooling2dダウンサンプル)
    residual_mode: str = 'or'     # 'or' | 'xor'(blocks.structural.Add の語彙)
    n_inputs: int = 2
    parameterization: str = 'softmax'
    num_classes: int = 10

    # --- 以下は契約外の拡張パラメータ(既定値のみで完結、呼び出し側は省略可) ---
    kernel_size: int = 3
    tree_depth: int = 2
    head_group_size: int = _HEAD_GROUP_SIZE
    input_channels_per_tree: int = 2
    residual_init: bool = True
    grad_factor: float = 1.0
    tau: float = 1.0
    device: str = 'cpu'


def _stage_conv(config: LogicResNetConfig, channels: int) -> LutConv2d:
    """残差ブロック内部で使う幅不変(channels→channels)のLutConv2d。"""
    return LutConv2d(
        channels, channels,
        kernel_size=config.kernel_size,
        tree_depth=config.tree_depth,
        padding=config.kernel_size // 2,
        n_inputs=config.n_inputs,
        device=config.device,
        grad_factor=config.grad_factor,
        residual_init=config.residual_init,
        input_channels_per_tree=config.input_channels_per_tree,
        parameterization=config.parameterization,
    )


class LogicResBlock(nn.Module):
    """残差ブロック: Residual(x, LutConv2d×2)。幅不変(in==out==channels)。

    blocks/structural.py の Residual をそのまま再利用する(自前で二重実装しない)。
    """

    def __init__(self, config: LogicResNetConfig, channels: int):
        super().__init__()
        inner = nn.Sequential(
            _stage_conv(config, channels),
            _stage_conv(config, channels),
        )
        self.residual = Residual(inner, merge=config.residual_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x)


class LogicResNet(nn.Module):
    """残差論理ConvNet。"""

    def __init__(self, config: LogicResNetConfig | None = None):
        super().__init__()
        config = config or LogicResNetConfig()
        if config.residual_mode not in ('or', 'xor'):
            raise ValueError(
                f"residual_mode は 'or'|'xor' のいずれか: {config.residual_mode}"
            )
        if len(config.stage_blocks) == 0:
            raise ValueError('stage_blocks は1要素以上必要')
        if any(n < 1 for n in config.stage_blocks):
            raise ValueError(f'stage_blocks の各要素は1以上: {config.stage_blocks}')
        self.config = config

        # 実数値[0,1]画素をチャネルごとに encode_bits ビットのthermometer符号へ
        self.thermometer = ThermometerEncoder(bits=config.encode_bits, mode='uniform')

        stem_in = config.in_channels * config.encode_bits
        self.stem = LutConv2d(
            stem_in, config.stem_channels,
            kernel_size=config.kernel_size,
            tree_depth=config.tree_depth,
            padding=config.kernel_size // 2,
            n_inputs=config.n_inputs,
            device=config.device,
            grad_factor=config.grad_factor,
            residual_init=config.residual_init,
            input_channels_per_tree=config.input_channels_per_tree,
            parameterization=config.parameterization,
        )

        # stage間(最終stage末尾は除く)でOrPooling2dダウンサンプル
        n_stages = len(config.stage_blocks)
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                LogicResBlock(config, config.stem_channels)
                for _ in range(n_blocks)
            ])
            for n_blocks in config.stage_blocks
        ])
        self.downsamples = nn.ModuleList([
            OrPooling2d(2) for _ in range(n_stages - 1)
        ])

        head_out = config.num_classes * config.head_group_size
        self.head = LutConv2d(
            config.stem_channels, head_out,
            kernel_size=1, tree_depth=1, padding=0,
            n_inputs=config.n_inputs,
            device=config.device,
            grad_factor=config.grad_factor,
            residual_init=False,
            input_channels_per_tree=min(config.input_channels_per_tree, config.stem_channels),
            parameterization=config.parameterization,
        )
        self.spatial_group_sum = SpatialGroupSum(config.num_classes)
        self.flatten = nn.Flatten()
        self.group_sum = GroupSum(k=config.num_classes, tau=config.tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W)[0,1] → (B, C*encode_bits, H, W) thermometer符号"""
        b, c, h, w = x.shape
        code = self.thermometer(x.permute(0, 2, 3, 1))  # (B,H,W,C*bits)
        return code.permute(0, 3, 1, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """head直後(SpatialGroupSum前)のテンソルを返す(hard離散性の検証用)。"""
        h = self.encode(x)
        h = self.stem(h)
        for i, stage in enumerate(self.stages):
            h = stage(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
        return self.head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        h = self.spatial_group_sum(h)  # (B, num_classes, H', W')
        h = self.flatten(h)            # (B, num_classes*H'*W')
        return self.group_sum(h)       # (B, num_classes)

    @property
    def num_gates(self) -> int:
        """総ゲート数(全LutConv2dのnum_gatesの合計)"""
        return sum(m.num_gates for m in self.modules() if isinstance(m, LutConv2d))

    def extra_repr(self) -> str:
        c = self.config
        return (
            f'stem_channels={c.stem_channels}, stage_blocks={c.stage_blocks}, '
            f'residual_mode={c.residual_mode}, n_inputs={c.n_inputs}, '
            f'num_classes={c.num_classes}'
        )


__all__ = ['LogicResBlock', 'LogicResNet', 'LogicResNetConfig']
