"""LogicMixer — an MLP-Mixer whose token/channel mixing MLPs are logic-gate stacks.

MLP-Mixer(Tolstikhin et al., 2021)はattentionを使わず、パッチ埋め込み後に
「token-mixing MLP(パッチ間、転置して適用)」と「channel-mixing MLP(チャネル内)」
を交互に積む純MLP構成。本実装はそのMLPをLutLayerスタック(LutStack)に置換した
論理ゲート版で、token-mix / channel-mix の分解を学習論理ゲートで行う先行研究は
2026-09 時点で確認できていない(最近接は CDDLGN 2411.04732 = 畳込であり mixer 分解ではない)。
ただし実タスクの数値は未較正(MNIST hard 0.671 で飽和)であり、新規性ではなく数値で
語る段階にある。主張の強度・最近接文献は監査ページ world-firsts(B4a)を参照し、
対外的な文言はそちらの限定形に従うこと。

設計判断:
    - **residual結合はsoft-XOR** ``a + b - 2ab`` を採用する。両入力が{0,1}の
      とき厳密にbitwise XORと一致するため、eval時のhard離散性(中間表現が
      {0,1})を保ったまま学習時は微分可能。GF(2)上の加算(=XOR)は実数残差接続
      の加算 ``x + f(x)`` の直接の類似物であり、``a XOR b XOR b = a`` で
      情報を破壊しない。soft-ORは飽和して段数を重ねるほど1に潰れるため不採用。
    - **token-mixing**はMLP-Mixerと同じく、チャネル軸をバッチ側へ畳み込んで
      token軸方向に「チャネル間で重み共有」のLutStackを適用する(転置して
      1回のLutStack呼び出しで全チャネルを処理)。
    - **patch embedding**は全パッチで重み共有の単一LutLayer
      (Conv2d(stride=patch_size)の論理ゲート類似物)。
    - **分類ヘッド**は本ライブラリの慣習(差分論理ゲートNN/LogicReservoirの
      readout='lut')に従い、フラット化した最終表現にLutLayer
      (out=num_classes×group_width)→GroupSumを適用する。
      d_tokens×d_channelsがnum_classesで割り切れる保証がないため、
      GroupSum直前に次元をnum_classesの倍数へ整える専用LUT層を挟む。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..encoding.thermometer import ThermometerEncoder
from ..group_sum import GroupSum
from ..layers.lut import LutLayer

# 分類ヘッドでクラスあたりに割り当てるLUTニューロン数(LogicReservoirのreadout='lut'を踏襲)
_HEAD_GROUP_WIDTH = 16


def _soft_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """微分可能なXOR残差結合。a,b∈{0,1}のとき厳密にbitwise XORと一致する。"""
    return a + b - 2.0 * a * b


class _LutStack(nn.Module):
    """同一in/out幅のLutLayerを`depth`段積んだ論理ゲートMLP相当のブロック。"""

    def __init__(self, width: int, depth: int, n_inputs: int, parameterization: str,
                 residual_init: bool = False):
        super().__init__()
        if depth < 1:
            raise ValueError(f'depthは1以上: {depth}')
        self.layers = nn.ModuleList([
            LutLayer(
                in_dim=width, out_dim=width, n_inputs=n_inputs,
                parameterization=parameterization, connections='random',
                residual_init=residual_init,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class LogicMixerConfig:
    """LogicMixerのハイパーパラメータ。

    d_tokens(=(image_size//patch_size)**2)はimage_size/patch_sizeからの
    導出値のため、独立フィールドではなくpropertyとして提供する。
    """

    image_size: int = 28
    patch_size: int = 7
    in_channels: int = 1
    encode_bits: int = 8       # thermometer符号化ビット/画素
    d_channels: int = 64       # パッチ表現のビット幅
    n_blocks: int = 4
    depth: int = 2             # 各mixing LutStackの段数
    n_inputs: int = 2
    parameterization: str = 'softmax'
    num_classes: int = 10
    # --- パッチ埋め込みのfan-in制御(MIXER-1 C) ---
    # 1段埋め込みは patch_pixels*encode_bits 本の入力を d_channels*n_inputs 本の
    # タップでしか受けられず、既定(392入力/128タップ)では入力の大半が構造的に
    # 捨てられる。embed_hiddenを指定すると 入力→embed_hidden→d_channels の
    # 2段LUTになり、1段目を全入力被覆(connections='unique')にできる。
    embed_hidden: int | None = None
    embed_connections: str = 'random'   # 'random' | 'unique'(全入力被覆を保証)
    # --- 分類ヘッドの幅(MIXER-1) ---
    # ヘッドは d_tokens×d_channels 本の入力を num_classes×head_group_width 個の
    # ゲートで受けるため、d_channelsを増やすほど被覆率が下がる(絞りが悪化する)。
    # 既定16はLogicReservoirのreadout='lut'を踏襲した値。
    head_group_width: int = _HEAD_GROUP_WIDTH
    # --- 混合LutLayerの恒等(素通し)近傍初期化(MIXER-2 B) ---
    # LutLayerのresidual_init: ゲート'A'(pass-through)のロジットを立てて、
    # 学習開始時に各ブロックがほぼ恒等写像になるようにする。深いブロック段数
    # (n_blocks=8)で勾配が消えて学習が始まらない問題への対処。
    # 注意: parameterization='iwp'では未対応(LutLayer側が警告して無視する)。
    residual_init: bool = False
    # --- 残差結合の方式(MIXER-2) ---
    # 'xor'       : 従来の soft-XOR `a+b-2ab`
    # 'none'      : 残差なし(素の直列)。soft-XORの ∂out/∂a = 1-2b は b→0.5 で0に
    #               なり勾配を殺すため、その影響を切り分ける/回避するための対照
    # 'alternate' : 2ブロックに1回だけXOR残差を入れる(中間案)
    residual_mode: str = 'xor'

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f'image_sizeはpatch_sizeで割り切れる必要: '
                f'{self.image_size} % {self.patch_size} != 0'
            )
        if self.n_blocks < 1:
            raise ValueError(f'n_blocksは1以上: {self.n_blocks}')
        if self.embed_connections not in ('random', 'unique'):
            raise ValueError(
                f"embed_connectionsは'random'か'unique': {self.embed_connections}")
        if self.embed_hidden is not None and self.embed_hidden < 1:
            raise ValueError(f'embed_hiddenは1以上かNone: {self.embed_hidden}')
        if self.head_group_width < 1:
            raise ValueError(f'head_group_widthは1以上: {self.head_group_width}')
        if self.residual_mode not in ('xor', 'none', 'alternate'):
            raise ValueError(
                f"residual_modeは'xor'/'none'/'alternate': {self.residual_mode}")
        if self.embed_connections == 'unique':
            # 'unique'は out_dim*n_inputs >= in_dim を要求する(全入力被覆の必要条件)
            first_out = self.embed_hidden or self.d_channels
            in_bits = self.patch_pixels * self.encode_bits
            if first_out * self.n_inputs < in_bits:
                raise ValueError(
                    f"embed_connections='unique'には埋め込み1段目の"
                    f'out_dim×n_inputs({first_out}×{self.n_inputs}='
                    f'{first_out * self.n_inputs}) >= 入力ビット数({in_bits}) '
                    f'が必要。embed_hiddenを{-(-in_bits // self.n_inputs)}以上にすること'
                )

    @property
    def side(self) -> int:
        """パッチ格子の一辺の枚数(image_size // patch_size)。"""
        return self.image_size // self.patch_size

    @property
    def d_tokens(self) -> int:
        """パッチ(token)数 = side ** 2。"""
        return self.side ** 2

    @property
    def patch_pixels(self) -> int:
        """1パッチあたりの生画素数(in_channels × patch_size²)。"""
        return self.in_channels * self.patch_size ** 2


class LogicMixerBlock(nn.Module):
    """token-mixing → channel-mixing の1ブロック(残差方式はconfigで切替)。"""

    def __init__(self, config: LogicMixerConfig, block_index: int = 0):
        super().__init__()
        self.config = config
        self.block_index = block_index
        self.token_mix = _LutStack(
            config.d_tokens, config.depth, config.n_inputs, config.parameterization,
            residual_init=config.residual_init,
        )
        self.channel_mix = _LutStack(
            config.d_channels, config.depth, config.n_inputs, config.parameterization,
            residual_init=config.residual_init,
        )

    def _combine(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """残差の合流。mode='none'なら混合出力をそのまま返す(素の直列)。

        soft-XORは ∂out/∂x = 1-2h であり、h→0.5 で勾配が厳密に0になる。
        混合出力は初期化直後に1ブロックで0.5へ潰れるため、これが深さ方向の
        指数的な勾配消失(1ブロックあたり約1e-2)の直接原因になっている。
        """
        mode = self.config.residual_mode
        if mode == 'none':
            return h
        if mode == 'alternate' and self.block_index % 2 == 1:
            return h
        return _soft_xor(x, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_tokens, d_channels) → 同形状。"""
        batch, tokens, channels = x.shape

        # --- token-mixing: チャネル軸をバッチへ畳み込み、token軸方向に
        #     チャネル間で重み共有のLutStackを適用(MLP-Mixerの転置MLPに相当) ---
        h = x.transpose(1, 2).reshape(batch * channels, tokens)
        h = self.token_mix(h)
        h = h.reshape(batch, channels, tokens).transpose(1, 2)
        x = self._combine(x, h)

        # --- channel-mixing: token軸をバッチへ畳み込み、チャネル軸方向に混合 ---
        h = x.reshape(batch * tokens, channels)
        h = self.channel_mix(h)
        h = h.reshape(batch, tokens, channels)
        x = self._combine(x, h)

        return x


class LogicMixer(nn.Module):
    """MLP-Mixerの論理ゲート化モデル。

    x: (B,C,H,W)実数[0,1] → thermometer符号化 → patch化 →
    [token-mix(転置してLutStack) → channel-mix(LutStack)]×n_blocks →
    分類ヘッド(LutLayer) → GroupSum → (B,num_classes) logits
    """

    def __init__(self, config: LogicMixerConfig):
        super().__init__()
        self.config = config

        self.thermometer = ThermometerEncoder(config.encode_bits, mode='uniform')

        # パッチ埋め込み: 全パッチ共有(Conv2d stride=patch_sizeの論理ゲート類似物)。
        # embed_hiddenを指定すると 入力→embed_hidden→d_channels の2段LUTになり、
        # 1段目のfan-inを増やして入力ビットの取りこぼしを減らせる(MIXER-1 C)。
        embed_in = config.patch_pixels * config.encode_bits
        embed_dims = ([embed_in, config.embed_hidden, config.d_channels]
                      if config.embed_hidden is not None
                      else [embed_in, config.d_channels])
        self.patch_embed = nn.Sequential(*[
            LutLayer(
                in_dim=embed_dims[i], out_dim=embed_dims[i + 1],
                n_inputs=config.n_inputs,
                parameterization=config.parameterization,
                # 全入力被覆が意味を持つのは生入力を受ける1段目のみ
                connections=config.embed_connections if i == 0 else 'random',
            )
            for i in range(len(embed_dims) - 1)
        ])

        self.blocks = nn.ModuleList([
            LogicMixerBlock(config, block_index=i) for i in range(config.n_blocks)
        ])

        # 分類ヘッド: d_tokens×d_channels → num_classes×group_width → GroupSum
        head_hidden = config.num_classes * config.head_group_width
        self.head_lut = LutLayer(
            in_dim=config.d_tokens * config.d_channels,
            out_dim=head_hidden,
            n_inputs=config.n_inputs,
            parameterization=config.parameterization,
            connections='random',
        )
        self.group_sum = GroupSum(
            k=config.num_classes, tau=float(config.head_group_width) ** 0.5)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → (B, d_tokens, patch_pixels) の非重複パッチに分割する。"""
        cfg = self.config
        if x.dim() != 4:
            raise ValueError(f'xは(B,C,H,W)の4次元: {tuple(x.shape)}')
        batch, channels, height, width = x.shape
        if (channels, height, width) != (cfg.in_channels, cfg.image_size, cfg.image_size):
            raise ValueError(
                f'期待する入力形状 (B,{cfg.in_channels},{cfg.image_size},'
                f'{cfg.image_size}): {tuple(x.shape)}'
            )
        p = cfg.patch_size
        side = cfg.side
        h = x.reshape(batch, channels, side, p, side, p)
        h = h.permute(0, 2, 4, 1, 3, 5)  # (B, side, side, C, p, p)
        return h.reshape(batch, cfg.d_tokens, cfg.patch_pixels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = x.shape[0]

        patches = self._patchify(x)  # (B, d_tokens, patch_pixels) 実数[0,1]
        coded = self.thermometer(patches)  # (B, d_tokens, patch_pixels*encode_bits) {0,1}

        h = self.patch_embed(coded.reshape(-1, coded.shape[-1]))
        h = h.reshape(batch, cfg.d_tokens, cfg.d_channels)

        for block in self.blocks:
            h = block(h)

        h = h.reshape(batch, cfg.d_tokens * cfg.d_channels)
        h = self.head_lut(h)
        return self.group_sum(h)

    def embed_input_coverage(self) -> float:
        """パッチ埋め込み1段目が実際に参照する入力ビットの割合(0〜1)。

        1.0未満なら、その分の入力ビットは構造的にモデルへ到達しない。
        """
        first = self.patch_embed[0]
        return int(torch.unique(first.indices).numel()) / int(first.in_dim)

    def head_input_coverage(self) -> float:
        """分類ヘッドが実際に参照する入力(d_tokens×d_channels)の割合(0〜1)。

        ヘッドは幅を増やしても out_dim=num_classes×16 で固定なので、
        d_channelsを増やすほどこの値は下がる(絞りが悪化する)。
        """
        return int(torch.unique(self.head_lut.indices).numel()) / int(self.head_lut.in_dim)

    def extra_repr(self) -> str:
        cfg = self.config
        embed = (f'{cfg.embed_hidden}->' if cfg.embed_hidden is not None else '')
        return (
            f'image_size={cfg.image_size}, patch_size={cfg.patch_size}, '
            f'd_tokens={cfg.d_tokens}, d_channels={cfg.d_channels}, '
            f'n_blocks={cfg.n_blocks}, depth={cfg.depth}, n_inputs={cfg.n_inputs}, '
            f'embed={embed}{cfg.d_channels}({cfg.embed_connections})'
        )


__all__ = [
    'LogicMixer',
    'LogicMixerBlock',
    'LogicMixerConfig',
]
