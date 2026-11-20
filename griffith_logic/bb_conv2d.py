import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .differentiable_lut_block import DifferentiableLutBlock


class BinaryBrainConv2d(nn.Module):
    """
    BinaryBrainのConvolution2d忠実再現。

    im2col でパッチ抽出 → DifferentiableLutBlock チェーン → col2im で空間に戻す。

    BinaryBrainのConvolution2dは以下の3段構成:
        1. ConvolutionIm2Col: (B, C, H, W) → パッチ (B, C*kH*kW, L)
        2. sub_layer: DifferentiableLutBlockチェーン（各位置独立に処理）
        3. ConvolutionCol2Im: パッチ → (B, out_ch, H_out, W_out)

    Args:
        lut_sizes: DifferentiableLutBlockの出力サイズリスト（例: [192, 32]）
            最後の要素が出力チャンネル数になる
        in_channels: 入力チャンネル数（set_input_shape相当、Noneの場合はforward時に自動決定）
        kernel_size: カーネルサイズ（デフォルト3）
        padding: パディング方式（'same' or int）
        n_inputs: LUTの入力数（デフォルト6）
        device: 計算デバイス
        unbinarize_bias: DifferentiableLutBlockのunbinarize bias
        batch_norm: BatchNormを有効にするか（最終層でFalseにする場合あり）
    """

    def __init__(
        self,
        lut_sizes: list[int],
        in_channels: int | None = None,
        kernel_size: int = 3,
        padding: str | int = 'same',
        n_inputs: int = 6,
        device: str = 'cpu',
        unbinarize_bias: float = 0.25,
        batch_norm: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        assert len(lut_sizes) >= 1, 'lut_sizesは1つ以上の要素が必要'

        self.lut_sizes = lut_sizes
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.n_inputs = n_inputs
        self.unbinarize_bias = unbinarize_bias
        self.batch_norm = batch_norm
        self.use_checkpoint = use_checkpoint
        self.out_channels = lut_sizes[-1]

        # in_channelsが指定されている場合はDifferentiableLutBlockを即座に構築
        self._lut_blocks: nn.ModuleList | None = None
        if in_channels is not None:
            self._build_blocks(in_channels, device)

    def _build_blocks(self, in_channels: int, device: str = 'cpu'):
        """DifferentiableLutBlockチェーンを構築する。"""
        kH = kW = self.kernel_size
        flat_in = in_channels * kH * kW

        blocks = []
        prev_dim = flat_in
        for i, out_dim in enumerate(self.lut_sizes):
            # 最終ブロック以外、またはbatch_norm=Trueの場合はデフォルト設定
            # BinaryBrainでは最終層でbatch_norm=Falseを指定可能
            is_last = (i == len(self.lut_sizes) - 1)
            if is_last and not self.batch_norm:
                block = _DifferentiableLutBlockNoBN(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    n_inputs=self.n_inputs,
                    device=device,
                    unbinarize_bias=self.unbinarize_bias,
                )
            else:
                block = DifferentiableLutBlock(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    n_inputs=self.n_inputs,
                    device=device,
                    connections='random',
                    unbinarize_bias=self.unbinarize_bias,
                )
            blocks.append(block)
            prev_dim = out_dim

        self._lut_blocks = nn.ModuleList(blocks)
        self.in_channels = in_channels

    def _compute_max_chunk(self) -> int:
        """INT_MAX超過防止 + メモリ節約のためのチャンクサイズ。

        LutLayerのbasisテンソル (batch, out_dim, 2^n_inputs) がボトルネック。
        MPSでは16384以下に抑えないとメモリ不足する傾向があるため小さめに設定。
        """
        max_out_dim = max(self.lut_sizes)
        n_entries = 2 ** self.n_inputs
        # INT_MAX制約からの理論上限
        max_elements = int(2 ** 31 * 0.8)
        int_limit = max(1, max_elements // (max_out_dim * n_entries))
        # MPS/CUDAメモリ節約のため実用的な上限を設ける
        memory_limit = 16384
        return min(int_limit, memory_limit)

    def _apply_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """DifferentiableLutBlockチェーンを適用する（checkpoint対象）。"""
        for block in self._lut_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 入力テンソル

        Returns:
            (B, out_channels, H_out, W_out) 出力テンソル
        """
        B, C, H, W = x.shape

        # 遅延初期化（in_channelsが未指定の場合）
        if self._lut_blocks is None:
            self._build_blocks(C, x.device.type)
            self._lut_blocks = self._lut_blocks.to(x.device)

        # パディングの計算
        if self.padding == 'same':
            pad = self.kernel_size // 2
        else:
            pad = int(self.padding)

        kH = kW = self.kernel_size

        # 1. im2col: (B, C*kH*kW, L)
        patches = F.unfold(x, (kH, kW), padding=(pad, pad))
        L = patches.shape[2]

        # 2. (B, C*kH*kW, L) → (B*L, C*kH*kW)
        patches = patches.permute(0, 2, 1).reshape(B * L, -1)

        # 3. DifferentiableLutBlockチェーンで処理
        # メモリ節約のためチャンク分割 + gradient checkpointing
        # （MPSではautogradが全中間活性を保持するため、全空間位置を一括処理すると
        #   メモリ不足になる）
        total = patches.shape[0]
        max_chunk = self._compute_max_chunk()
        use_ckpt = self.use_checkpoint and self.training and patches.requires_grad

        if total <= max_chunk:
            if use_ckpt:
                out = checkpoint(self._apply_blocks, patches, use_reentrant=False)
            else:
                out = self._apply_blocks(patches)
        else:
            chunks = patches.split(max_chunk, dim=0)
            processed = []
            for chunk in chunks:
                if use_ckpt:
                    chunk = checkpoint(self._apply_blocks, chunk, use_reentrant=False)
                else:
                    chunk = self._apply_blocks(chunk)
                processed.append(chunk)
            out = torch.cat(processed, dim=0)

        # 4. (B*L, out_ch) → (B, out_ch, H_out, W_out)
        H_out = H + 2 * pad - kH + 1
        W_out = W + 2 * pad - kW + 1
        out = out.reshape(B, L, self.out_channels)
        out = out.permute(0, 2, 1).reshape(B, self.out_channels, H_out, W_out)

        return out

    def clamp_weights(self):
        """全DifferentiableLutBlockの重みを[0,1]にクランプ。"""
        if self._lut_blocks is not None:
            for block in self._lut_blocks:
                if hasattr(block, 'clamp_weights'):
                    block.clamp_weights()

    def extra_repr(self):
        return (
            f'lut_sizes={self.lut_sizes}, in_channels={self.in_channels}, '
            f'kernel_size={self.kernel_size}, padding={self.padding}, '
            f'n_inputs={self.n_inputs}, batch_norm={self.batch_norm}'
        )


class _DifferentiableLutBlockNoBN(nn.Module):
    """
    BatchNormなしのDifferentiableLutBlock。

    BinaryBrainの最終Conv層で batch_norm=False が指定される場合に使用。
    Unbinarize → LUT → Binarize（BatchNormスキップ）。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_inputs: int = 6,
        device: str = 'cpu',
        connections: str = 'random',
        unbinarize_bias: float = 0.25,
    ):
        super().__init__()
        from .binarize import Binarize
        from .differentiable_lut_block import UnbinarizeFunction
        from .lut_layer import LutLayer

        self.unbinarize_bias = unbinarize_bias
        self._unbinarize_fn = UnbinarizeFunction

        self.lut = LutLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            n_inputs=n_inputs,
            device=device,
            connections=connections,
            parameterization='raw',
        )

        self.binarize = Binarize(
            threshold=0.5, hardtanh_min=0.0, hardtanh_max=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self._unbinarize_fn.apply(x, self.unbinarize_bias)
        else:
            x = 0.5 + torch.where(
                x > 0.5,
                torch.tensor(self.unbinarize_bias, device=x.device, dtype=x.dtype),
                torch.tensor(-self.unbinarize_bias, device=x.device, dtype=x.dtype),
            )
        x = self.lut(x)
        x = self.binarize(x)
        return x

    def clamp_weights(self):
        """重みを[0,1]にクランプ。"""
        with torch.no_grad():
            self.lut.weights.clamp_(0.0, 1.0)
