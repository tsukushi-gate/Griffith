"""nn.Linear / Transformer FFN のdrop-in論理ゲート型レイヤ。

実数in/実数outの契約を内部で完結させる:

    x(実数) → ToLogic → [LutLayer × depth] → FromLogic → y(実数)

既存のnn.Moduleの途中にそのまま挟める。3D入力(B, S, D)は
(B×S, D)にflattenして処理する(Transformer MLP置換に必須)。
"""

import torch
import torch.nn as nn

from ..encoding.bridges import FromLogic, ToLogic
from ..hybrid.policy import LogicSpec
from .lut import LutLayer


class LogicLinear(nn.Module):
    """nn.Linear互換の論理ゲート型レイヤ。

    Args:
        in_features: 入力特徴数
        out_features: 出力特徴数
        spec: 生成レシピ(省略時はLogicSpec()のデフォルト)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        spec: LogicSpec | None = None,
    ):
        super().__init__()
        spec = spec or LogicSpec()
        self.in_features = in_features
        self.out_features = out_features
        self.spec = spec
        width = spec.resolve_width(out_features)

        self.encode = ToLogic(spec.encoder, spec.encoder_tau,
                              bits=spec.encoder_bits)
        encoded_in = self.encode.out_width(in_features)
        luts = []
        if spec.parameterization == 'dwn':
            # DWN: RAMノード層(FD/EFD勾配、第1層のみLearnable Mapping)
            from .dwn import DWNLayer
            for i in range(spec.depth):
                luts.append(DWNLayer(
                    in_bits=encoded_in if i == 0 else width,
                    n_luts=width,
                    n_inputs=spec.n_inputs,
                    mapping='learnable' if i == 0 else 'random',
                ))
        else:
            for i in range(spec.depth):
                luts.append(LutLayer(
                    in_dim=encoded_in if i == 0 else width,
                    out_dim=width,
                    n_inputs=spec.n_inputs,
                    connections=spec.connections,
                    parameterization=spec.parameterization,
                    selection=spec.selection,
                    grad_factor=spec.grad_factor,
                ))
        self.luts = nn.Sequential(*luts)
        self.decode = FromLogic(width, out_features, spec.decoder, spec.decoder_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        h = x.reshape(-1, self.in_features)
        h = self.decode(self.luts(self.encode(h)))
        return h.reshape(*lead, self.out_features)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


class LogicMLP(LogicLinear):
    """Transformer FFN(Linear→活性化→Linear)全体の置換体。

    FFNを2つのLogicLinearで置き換えるよりLUTスタック1本の方が
    効率的なため、独立クラスとして提供する。out_features省略時は
    in_featuresと同じ(d_model→d_modelのFFN置換)。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        *,
        spec: LogicSpec | None = None,
    ):
        super().__init__(in_features, out_features or in_features, spec=spec)
