"""conv-DLGN(重み共有の論理ゲート木 conv + OR プール + 全結合頭部)の組み立て。

内部調査リポジトリ先行実験 `ConvLogicNet` / cdlgn(Petersen et al., NeurIPS 2024)の構成を、
ライブラリの部品(LutConv2d / OrPooling2d / LutLayer / GroupSum)だけで組む:

    conv 木ブロック × 段(各段で空間 1/2: stride-2 conv か stride-1 conv + OR プール 2×2)
    → Flatten → 全結合論理層 × 段(配線 rot2 / random / unique)→ GroupSum(割当 contig / stride)

docs/architecture/model-architectures.md §8 の LogicTreeNet(experiments/train_cnn.py)と同じ
cdlgn 型で、こちらは学習側の部品(葉サンプラ・IWP(independent × sin × Ω 空間 RI)・
歩進付き rot(rot2)の頭部・stride 割当の GroupSum)を足し、`training/config.py` の
`model.kind: conv_dlgn` から組めるようにしたもの。

IR は `export_ir(net, input_shape=cfg.in_shape, native_conv=True)` で conv2d / or_pool2d
ネイティブノードになる(重み共有は native ノードにしか残らない。FPGA 側との契約 §0.3 の 3)。
入力は DataLoader の (B, C, H, W) をそのまま受ける(先頭に Flatten を置かない)。

先行実験からの写し方の注意:
- 木の対(ペアリング)は先行実験の半割を隣接に写しただけ(同型。conv_samplers.py)。
- 頭部の LutLayer の Residual Init は最後の入力(b)の pass-through(ライブラリの既定)、
  conv 木は第 1 入力(a)の pass-through(先行実験と同じ)。どちらも「素通し」で等価。
- 先行実験の学習時 forward は mode='ste'(値 hard / 勾配 soft)が既定、先行実験は soft。
  本ビルダは parameterization='independent'(soft)か 'independent_ste'(値 hard)で選ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..conv import LutConv2d
from ..conv_samplers import SAMPLER_SPECS
from ..core.connections import get_rot_connections
from ..group_sum import GroupSum
from ..layers.lut import LutLayer
from ..pooling import OrPooling2d

HEAD_WIRINGS = ('rot2', 'random', 'unique')
DOWNSAMPLES = ('orpool', 'stride')


@dataclass
class ConvDLGNConfig:
    """conv-DLGN の構成。既定値は先行実験 `cifar10_s`(cdlgn k=32)。

    Attributes:
        in_shape: 入力 (C, H, W)。CIFAR-10 3 閾値(閾値 major)なら (9, 32, 32)
        channels: conv 木ブロックの出力チャネル列(cdlgn k=32: k, 4k, 16k, 32k)
        sampler: 葉の読み座標('random' / 'random8' / 'cross5' / 'dilated8' / 'checker13')
        kernel_size / tree_depth / padding: sampler='random' のときだけ使う(他は腕ごとに固定)
        input_channels_per_tree: 木 1 本が読む入力チャネル数(cdlgn num_chn)
        downsample: 'orpool'(stride-1 conv + OR プール 2×2)/ 'stride'(stride-2 conv)
        head_dims: 頭部の隠れ幅列(入力 flat を含まない)。None なら cdlgn 頭部則
            [10·flat, 5·flat, 2.5·flat] × head_scale(最終層は num_classes の倍数)
        head_wiring: 'rot2'(歩進付き rot。先行実験 `rot2_schedule`)/ 'random' / 'unique'
        num_classes / tau / group_assign: GroupSum
        parameterization / estimator / residual_init / ri_mu / ri_sigma: conv 木と頭部に共通
        selection: 頭部 LutLayer のゲート選択('soft' / 'hard_st' / 'gumbel_st')
    """

    in_shape: tuple[int, int, int]
    channels: tuple[int, ...] = (32, 128, 512, 1024)
    sampler: str = 'random8'
    kernel_size: int = 3
    tree_depth: int = 3
    padding: int = 1
    input_channels_per_tree: int = 2
    downsample: str = 'orpool'
    head_dims: tuple[int, ...] | None = None
    head_wiring: str = 'rot2'
    head_scale: float = 1.0
    num_classes: int = 10
    tau: float = 20.0
    group_assign: str = 'stride'
    parameterization: str = 'independent'
    estimator: str = 'sin'
    residual_init: bool = True
    ri_mu: float = 1.2
    ri_sigma: float = 0.25
    selection: str = 'soft'


def head_dims_cdlgn(flat: int, num_classes: int = 10, extra_layers: int = 1,
                    scale: float = 1.0) -> list[int]:
    """cdlgn ConvLogicCifarModel の頭部幅則(先行実験 `head_dims_cdlgn`)。入力 flat を先頭に含む。

    scale=1.0 が cdlgn どおり [flat, 10·flat, 5·flat, 2.5·flat]。scale は入力 flat には
    掛けない(flat は conv 段の出力形で決まる)ので縮小層の比は変わらず rot2 の歩進 r も不変。
    """
    dims = [int(flat),
            extra_layers * flat * 10,
            extra_layers * flat * 5,
            extra_layers * (flat * num_classes) // 4]
    if scale != 1.0:
        dims = [dims[0]] + [int(round(v * scale)) for v in dims[1:]]
    if dims[-1] % num_classes != 0:
        raise ValueError(
            f'head_scale={scale} だと最終層 {dims[-1]} が num_classes={num_classes} で'
            '割り切れません')
    if any(v <= 0 for v in dims):
        raise ValueError(f'頭部の幅が正でない: {dims}')
    return dims


def rot_schedule(dims: list[int]) -> list[tuple[int, int]]:
    """頭部 rot 配線のオフセット則(先行実験 `rot_schedule`)。層 l(入力幅 W_in、全 L 層):
    s_a = 0, s_b = max(1, round(W_in^((l+1)/(L+1))))(細かい近傍から粗い混合へ)。"""
    n = len(dims) - 1
    return [(0, max(1, int(round(dims[layer] ** ((layer + 1) / (n + 1))))))
            for layer in range(n)]


def rot2_schedule(dims: list[int]) -> list[tuple[int, int]]:
    """歩進付き rot(rot2)のオフセット則(先行実験 `rot2_schedule`)。層 l ごとに (r, s_b)。

        r   = max(1, W_in // W_out)   入力を W_out 個のゲートで舐めきる歩進
        s_b = rot_schedule の幾何級数を、r ≥ 2 で s_b mod r == 0 なら +1(別の剰余類に落とす)

    cifar10_s 頭部 [4096, 40960, 20480, 10240] → [(1, 8), (2, 203), (2, 1713)]、
    cifar10_m 頭部 [32768, 327680, 163840, 81920] → [(1, 13), (2, 573), (2, 8145)]。
    縮小層は r=2・s_b 奇数なので被覆率 100%・fan-out ちょうど 1。
    """
    out = []
    for layer, (_, s_b) in enumerate(rot_schedule(dims)):
        w_in, w_out = dims[layer], dims[layer + 1]
        r = max(1, w_in // w_out)
        if r > 1 and s_b % r == 0:
            s_b += 1
        out.append((r, s_b))
    return out


def conv_out_hw(h: int, w: int, kernel: int, stride: int, padding: int) -> tuple[int, int]:
    return (h + 2 * padding - kernel) // stride + 1, (w + 2 * padding - kernel) // stride + 1


def conv_dlgn_shape(cfg: ConvDLGNConfig) -> dict:
    """構成から形(ブロックごとの出力 (C, H, W)、flat、頭部の幅列、rot2 の (r, s_b))を計算する。"""
    if cfg.sampler == 'random':
        kernel, depth, pad = int(cfg.kernel_size), int(cfg.tree_depth), int(cfg.padding)
    else:
        if cfg.sampler not in SAMPLER_SPECS:
            raise ValueError(
                f'未知のサンプラ: {cfg.sampler!r}(対応: random / {sorted(SAMPLER_SPECS)})')
        spec = SAMPLER_SPECS[cfg.sampler]
        kernel, depth, pad = spec['kernel'], spec['depth'], spec['padding']
    if cfg.downsample not in DOWNSAMPLES:
        raise ValueError(f"downsample は {DOWNSAMPLES} のいずれか: {cfg.downsample!r}")
    if cfg.head_wiring not in HEAD_WIRINGS:
        raise ValueError(f"head_wiring は {HEAD_WIRINGS} のいずれか: {cfg.head_wiring!r}")
    stride = 2 if cfg.downsample == 'stride' else 1
    c, h, w = (int(v) for v in cfg.in_shape)
    blocks = []
    for co in cfg.channels:
        h, w = conv_out_hw(h, w, kernel, stride, pad)
        if cfg.downsample == 'orpool':
            h, w = h // 2, w // 2
        if h < 1 or w < 1:
            raise ValueError(
                f'空間サイズが 0 になります: channels={cfg.channels}, in_shape={cfg.in_shape}')
        blocks.append((int(co), h, w))
    flat = blocks[-1][0] * blocks[-1][1] * blocks[-1][2]
    if cfg.head_dims is None:
        dims = head_dims_cdlgn(flat, cfg.num_classes, 1, cfg.head_scale)
    else:
        dims = [flat] + [int(v) for v in cfg.head_dims]
        if dims[-1] % cfg.num_classes != 0:
            raise ValueError(
                f'head_dims の最終層 {dims[-1]} は num_classes={cfg.num_classes} の倍数が必要です')
    return {
        'kernel': kernel, 'tree_depth': depth, 'padding': pad, 'stride': stride,
        'blocks': blocks, 'flat': flat, 'head_dims': dims,
        'rot2': rot2_schedule(dims) if cfg.head_wiring == 'rot2' else None,
    }


def build_conv_dlgn(cfg: ConvDLGNConfig, device: str = 'cpu',
                    generator: torch.Generator | None = None) -> nn.Sequential:
    """conv-DLGN を nn.Sequential として組む(export_ir にそのまま渡せる直列構造)。

    Args:
        cfg: 構成
        device: 配置先
        generator: random8 の葉座標・sin の RI・頭部 random 配線に使う generator
            (None ならグローバル RNG)
    """
    shape = conv_dlgn_shape(cfg)
    layers: list[nn.Module] = []
    c_in = int(cfg.in_shape[0])
    for i, (co, _, _) in enumerate(shape['blocks']):
        layers.append(LutConv2d(
            c_in, co, kernel_size=shape['kernel'], tree_depth=shape['tree_depth'],
            stride=shape['stride'], padding=shape['padding'], n_inputs=2, device=device,
            residual_init=cfg.residual_init,
            input_channels_per_tree=cfg.input_channels_per_tree,
            parameterization=cfg.parameterization, sampler=cfg.sampler, layer_idx=i,
            estimator=cfg.estimator, ri_mu=cfg.ri_mu, ri_sigma=cfg.ri_sigma,
            generator=generator,
        ))
        if cfg.downsample == 'orpool':
            layers.append(OrPooling2d(kernel_size=2))
        c_in = co
    layers.append(nn.Flatten())
    dims = shape['head_dims']
    for layer_idx in range(len(dims) - 1):
        d_in, d_out = dims[layer_idx], dims[layer_idx + 1]
        if cfg.head_wiring == 'rot2':
            r, s_b = shape['rot2'][layer_idx]
            connections = get_rot_connections(d_in, d_out, 0, s_b, device, step=r)
        else:
            connections = cfg.head_wiring
        layers.append(LutLayer(
            d_in, d_out, n_inputs=2, device=device, connections=connections,
            parameterization=cfg.parameterization, estimator=cfg.estimator,
            residual_init=cfg.residual_init, ri_mu=cfg.ri_mu, ri_sigma=cfg.ri_sigma,
            selection=cfg.selection,
        ))
    layers.append(GroupSum(cfg.num_classes, tau=cfg.tau, assign=cfg.group_assign))
    return nn.Sequential(*layers)
