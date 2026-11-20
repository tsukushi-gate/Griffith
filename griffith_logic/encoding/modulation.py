import torch
import torch.nn as nn


class RealToBinary(nn.Module):
    """
    バイナリ変調: 実数入力を複数閾値で二値化し、バッチ方向にフレーム数を増やす。

    BinaryBrainのRealToBinaryを忠実に再現。
    閾値: (f + 0.5) / frame_modulation_size  (f = 0, 1, ..., frame_modulation_size-1)

    パラメータなし。勾配は流さない（前処理）。

    Args:
        frame_modulation_size: フレーム変調数（デフォルト15）
    """

    # register_bufferで登録する属性の型宣言(mypyのModule.__getattr__曖昧性回避)
    thresholds: torch.Tensor

    def __init__(self, frame_modulation_size: int = 15):
        super().__init__()
        self.frame_modulation_size = frame_modulation_size
        # 閾値を事前計算してバッファに登録
        thresholds = torch.tensor(
            [(f + 0.5) / frame_modulation_size for f in range(frame_modulation_size)],
            dtype=torch.float32,
        )
        self.register_buffer('thresholds', thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) または (B, C, H, W) 実数入力 [0, 1]

        Returns:
            (B*M, D) または (B*M, C, H, W) バイナリ出力 {0, 1}
        """
        M = self.frame_modulation_size
        if x.dim() == 2:
            # (B, D) → (B*M, D)
            B, D = x.shape
            binary_frames = (x.unsqueeze(1) > self.thresholds.view(1, -1, 1)).to(x.dtype)
            return binary_frames.reshape(B * M, D)
        elif x.dim() == 4:
            # (B, C, H, W) → (B*M, C, H, W)
            B, C, H, W = x.shape
            # x: (B, 1, C, H, W), thresholds: (1, M, 1, 1, 1) → (B, M, C, H, W)
            binary_frames = (
                x.unsqueeze(1) > self.thresholds.view(1, -1, 1, 1, 1)
            ).to(x.dtype)
            return binary_frames.reshape(B * M, C, H, W)
        else:
            raise ValueError(
                f'入力は2次元(B,D)または4次元(B,C,H,W)である必要があります: {x.dim()}次元'
            )

    def extra_repr(self):
        return f'frame_modulation_size={self.frame_modulation_size}'


class BinaryToReal(nn.Module):
    """
    フレーム積算: 変調フレームを平均して実数に復元。

    BinaryBrainのBinaryToRealを忠実に再現。
    reshape + mean で実装し、PyTorchのautogradで自動的に勾配が流れる。

    Args:
        frame_modulation_size: フレーム変調数（RealToBinaryと同じ値を指定）
    """

    def __init__(self, frame_modulation_size: int = 15):
        super().__init__()
        self.frame_modulation_size = frame_modulation_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*M, D) または (B*M, C, H, W) バイナリまたは実数値

        Returns:
            (B, D) または (B, C, H, W) 実数値（フレーム方向の平均）
        """
        M = self.frame_modulation_size
        if x.dim() == 2:
            # (B*M, D) → (B, M, D) → mean over M → (B, D)
            BM, D = x.shape
            B = BM // M
            return x.reshape(B, M, D).mean(dim=1)
        elif x.dim() == 4:
            # (B*M, C, H, W) → (B, M, C, H, W) → mean over M → (B, C, H, W)
            BM, C, H, W = x.shape
            B = BM // M
            return x.reshape(B, M, C, H, W).mean(dim=1)
        else:
            raise ValueError(
                f'入力は2次元(B*M,D)または4次元(B*M,C,H,W)である必要があります: {x.dim()}次元'
            )

    def extra_repr(self):
        return f'frame_modulation_size={self.frame_modulation_size}'
