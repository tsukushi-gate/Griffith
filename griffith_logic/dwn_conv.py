"""DWN RAMノードの畳み込み版（レバー4）。

なぜ作るか
----------
現行の ``LutConv2d`` は 16 種の2入力ゲートに対する softmax 分布を学び、評価時に
argmax で1つへ確定する。つまり「連続で学んで最後に離散化」する方式で、学習時と
評価時で計算が変わる（この不一致が繰り返し精度の壁になった）。

DWN (Bacellar et al., ICML 2024) は LUT の中身を離散のまま扱い、勾配を有限差分で
求める。forward は評価時とまったく同じ「アドレスで表を引く」操作であり、
**学習時と評価時が最初から一致する**。dense 版は
``griffith_logic/layers/dwn.py`` に実装済み（公式実装と照合済み）なので、
本モジュールはその ``DWNFunction`` をそのまま使い、畳み込みの入力収集
（unfold → 木ごとのタップ集め）だけを与える。

``LutConv2d`` との対応:
    LutConv2d   : 出力チャネルごとに深さdのゲート木（2入力ゲート × n_internal個）
    DWNConv2d   : 出力チャネルごとに1個の n入力 RAMノード（表サイズ 2^n）

n=6 のとき表は 64 エントリで、深さ3のゲート木（7ゲート×16通り）より表現力が高い
一方、パラメータ数は 64 と小さい。空間方向は重み共有（同じ表を全画素で使う）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.dwn import (
    BinarizeSTEFunction,
    DWNFunction,
    build_efd_coefficients,
)


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def generate_conv_taps(
    in_channels: int,
    out_channels: int,
    kernel_size,
    n_inputs: int,
    input_channels_per_tree: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """出力チャネルごとに、受容野の中から n_inputs 個のタップ位置を選ぶ。

    ``LutConv2d._generate_tree_connections`` と同じ規約:
    まず木ごとに ``input_channels_per_tree`` 本の入力チャネルを選び、その中から
    (channel, kh, kw) を引く。返り値は unfold 後の平坦índex
    ``c*kH*kW + h*kW + w``。

    Returns:
        (out_channels, n_inputs) の long テンソル
    """
    kH, kW = _pair(kernel_size)
    g = generator
    ict = min(input_channels_per_tree, in_channels)
    taps = torch.empty(out_channels, n_inputs, dtype=torch.long)
    for o in range(out_channels):
        chans = torch.randperm(in_channels, generator=g)[:ict]
        for i in range(n_inputs):
            c = chans[torch.randint(ict, (1,), generator=g).item()].item()
            h = torch.randint(kH, (1,), generator=g).item()
            w = torch.randint(kW, (1,), generator=g).item()
            taps[o, i] = c * kH * kW + h * kW + w
    return taps


class DWNConv2d(nn.Module):
    """n入力RAMノードを畳み込みカーネルとして使う層（離散のまま学習）。

    ``LutConv2d`` と同じ入出力契約: (B, in_ch, H, W) の {0,1} → (B, out_ch, H', W')
    の {0,1}。空間方向は重み共有なので、ゲート数（＝配信サイズ）は解像度に依存しない。

    Args:
        in_channels/out_channels/kernel_size/stride/padding: 畳み込みの通常引数
        n_inputs: RAMノード1個あたりの入力数 n（表サイズ 2^n）
        input_channels_per_tree: 1ノードが参照してよい入力チャネル数
        efd: EFD係数 'paper'(1/(H+1)) | 'official'(α·β^H)
        ste: 出力を{0,1}へ二値化する
        clamp_luts: 訓練時に表を[−1,1]へclamp（公式実装準拠）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        *,
        stride=1,
        padding=1,
        n_inputs: int = 6,
        input_channels_per_tree: int = 8,
        efd: str = 'paper',
        ste: bool = True,
        clamp_luts: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.n_inputs = n_inputs
        self.n_entries = 2 ** n_inputs
        self.input_channels_per_tree = input_channels_per_tree
        self.ste = ste
        self.clamp_luts = clamp_luts

        # LUT表: 一様U(−1,1)初期化（dwn.py の DWNLayer と同一）
        self.luts = nn.Parameter(torch.rand(out_channels, self.n_entries) * 2.0 - 1.0)
        self.register_buffer('c_efd', build_efd_coefficients(n_inputs, mode=efd))

        g = None
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        self.register_buffer(
            'taps',
            generate_conv_taps(in_channels, out_channels, self.kernel_size,
                               n_inputs, input_channels_per_tree, generator=g),
        )

    @property
    def num_gates(self) -> int:
        """重み共有のノード数（LutConv2d.num_gates と同じ意味で使える）。"""
        return self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # torch.fx の symbolic trace では shape が Proxy になり、比較を
        # bool として使うと TraceError になる（export_dag_ir がここを通る）。
        # 具体値のときだけ検査する。
        if isinstance(C, int) and C != self.in_channels:
            raise ValueError(f'in_channels 不一致: {C} != {self.in_channels}')
        kH, kW = self.kernel_size

        if self.training and self.clamp_luts:
            with torch.no_grad():
                self.luts.clamp_(-1.0, 1.0)

        # (B, C*kH*kW, L) — L は出力画素数
        patches = F.unfold(x, (kH, kW), padding=self.padding, stride=self.stride)
        L = patches.shape[-1]

        # 各出力チャネルのタップを集める: (B, out_ch, n_inputs, L)
        gathered = patches[:, self.taps.reshape(-1)].reshape(
            B, self.out_channels, self.n_inputs, L)
        # DWNFunction は (batch, n_luts, n) を期待するので画素をバッチ側へ畳む
        gathered = gathered.permute(0, 3, 1, 2).reshape(
            B * L, self.out_channels, self.n_inputs)

        y = DWNFunction.apply(gathered, self.luts, self.c_efd)  # (B*L, out_ch)
        if self.ste:
            y = BinarizeSTEFunction.apply(y)

        H_out = (H + 2 * self.padding[0] - kH) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - kW) // self.stride[1] + 1
        return y.reshape(B, L, self.out_channels).permute(0, 2, 1).reshape(
            B, self.out_channels, H_out, W_out)

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'n_inputs={self.n_inputs}, entries={self.n_entries}, '
                f'ict={self.input_channels_per_tree}')
