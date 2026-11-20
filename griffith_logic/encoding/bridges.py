"""実数テンソル⇔論理領域[0,1]の境界ブリッジ。

ハイブリッドNN(PyTorch標準層 + 論理ゲート層)の接合部で使う。
docs/archive/hybrid-lut-ann-attention.md のANNToLut/LutToANN構想を実装。

    ToLogic:   実数 → [0,1]   (LUT層の入力契約を満たす)
    FromLogic: [0,1] → 実数   (GroupSum集約+ゼロ中心化で後段の実数層へ)
"""

import math

import torch
import torch.nn as nn

from ..binarize import BinarizeFunction
from .thermometer import ThermometerEncoder

TO_LOGIC_MODES = ('sigmoid', 'binarize_ste', 'identity', 'thermometer')
FROM_LOGIC_MODES = ('groupsum_centered', 'groupsum', 'layernorm', 'identity')


class ToLogic(nn.Module):
    """実数 → [0,1] 変換。

    mode:
        'sigmoid'      — sigmoid(x/τ)。τは学習可能(飽和による勾配消失を緩和)
        'binarize_ste' — 閾値0で二値化、backward はHardTanh STE
        'identity'     — 変換なし(入力が既に[0,1]のとき)
        'thermometer'  — Thermometer符号化(D → D×bits に拡張、STE)。
                         しきい値は初回訓練バッチで自動fit(uniform)

    Note:
        thermometerは出力幅が変わる。後段の幅は out_width() で取得すること。
    """

    def __init__(self, mode: str = 'sigmoid', tau_init: float = 1.0,
                 bits: int = 8):
        super().__init__()
        assert mode in TO_LOGIC_MODES, f'mode は {TO_LOGIC_MODES} のいずれか: {mode}'
        self.mode = mode
        self.bits = bits
        if mode == 'sigmoid':
            # 正値制約のためlogパラメータ化
            self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())
        elif mode == 'thermometer':
            self.thermo = ThermometerEncoder(bits, mode='uniform')

    def out_width(self, in_features: int) -> int:
        """このエンコーダを通した後の特徴幅"""
        if self.mode == 'thermometer':
            return in_features * self.bits
        return in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'sigmoid':
            return torch.sigmoid(x / self.log_tau.exp())
        if self.mode == 'binarize_ste':
            return BinarizeFunction.apply(x, 0.0, -1.0, 1.0)
        if self.mode == 'thermometer':
            if self.thermo.thresholds is None:
                if not self.training:
                    raise RuntimeError(
                        'ToLogic(thermometer): 初回訓練バッチでの自動fit前に'
                        'eval実行された。先にtrainモードでforwardするか'
                        'self.thermo.fit(data)を呼ぶこと'
                    )
                self.thermo.fit(x.detach().reshape(-1, x.shape[-1]))
            return self.thermo(x)
        return x

    def extra_repr(self):
        return f'mode={self.mode}'


class FromLogic(nn.Module):
    """[0,1]論理出力 → 実数変換(グループ集約)。

    in_width個の論理出力を out_features グループに分けて合計し、
    ゼロ中心化+スケーリングして実数表現に戻す。

    mode:
        'groupsum_centered' — (Σ - G/2)/τ。residual接続でも平均が暴れない(推奨)
        'groupsum'          — Σ/τ(従来のGroupSumと同じ、非負)
        'layernorm'         — groupsum_centered + LayerNorm
        'identity'          — 変換なし(in_width == out_features 必須)

    τのデフォルトは √G(G=グループサイズ)。二値出力の合計の標準偏差が
    √G/2 のため、出力がO(1)に正規化される。
    """

    def __init__(
        self,
        in_width: int,
        out_features: int,
        mode: str = 'groupsum_centered',
        tau: float | None = None,
    ):
        super().__init__()
        assert mode in FROM_LOGIC_MODES, f'mode は {FROM_LOGIC_MODES} のいずれか: {mode}'
        if mode == 'identity':
            assert in_width == out_features, (
                f'identity時は in_width == out_features: {in_width} != {out_features}'
            )
        else:
            assert in_width % out_features == 0, (
                f'in_width({in_width}) は out_features({out_features}) の倍数が必要'
            )
        self.in_width = in_width
        self.out_features = out_features
        self.mode = mode
        self.group_size = in_width // out_features if mode != 'identity' else 1
        self.tau = tau if tau is not None else max(1.0, math.sqrt(self.group_size))
        if mode == 'layernorm':
            self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'identity':
            return x
        g = x.reshape(*x.shape[:-1], self.out_features, self.group_size).sum(dim=-1)
        if self.mode in ('groupsum_centered', 'layernorm'):
            g = g - self.group_size / 2.0
        y = g / self.tau
        if self.mode == 'layernorm':
            y = self.norm(y)
        return y

    def extra_repr(self):
        return (f'in_width={self.in_width}, out={self.out_features}, '
                f'mode={self.mode}, tau={self.tau:.3g}')
