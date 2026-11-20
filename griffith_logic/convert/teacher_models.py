"""教師CNNのモデル定義(ROADMAP-v5)。

`experiments/train_teacher_cnn.py` が学習するteacherのアーキテクチャ。
`convert/teacher.py` の `resolve_teacher('ckpt:<path>')` は
metadataの `model_class` を `torch.load` 時にunpickleするため、
モジュール直下(import可能なパス)にクラスとして定義する必要がある
(experimentsスクリプト内で定義したクラスは `__main__.ClassName` に
束縛され、別プロセスからの読込で解決できなくなるため)。

`convert/__init__.py` からは意図的に再エクスポートしない
(teacher読込専用の内部部品であり、Track D公開APIの一部ではないため)。
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ['SmallTeacherCNN', 'TeacherMLP', 'TeacherMLPSte', 'HeavisideSTE']


class HeavisideSTE(torch.autograd.Function):
    """発火判定二値化 y=1[x>0] のStraight-Through Estimator(ROADMAP-v5)。

    forwardは `transcribe_mlp` の隠れ層意味論 y_i=1[Wx+b>0] と厳密に一致する
    (境界 x=0 は 0)。backwardはhardtanh的な勾配(|x|<=1 で 1、外側で 0)を
    通す(BNN学習の標準STE、Hubara 2016 / Courbariaux 2016)。

    この二値化で学習したteacherは、隠れ活性が転写後(1bit)と学習時で一致する
    ため、転写ギャップが fan-in疎化誤差のみに縮む(仮説H2)。
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        (x,) = ctx.saved_tensors
        return grad_y * (x.abs() <= 1).to(grad_y.dtype)


class TeacherMLP(nn.Module):
    """784-300-100-10 ReLU MLP(float活性teacher、teacher-float)。

    `transcribe_mlp` / `lower_bnn_mlp` は nn.Linear 列を children/modules 順に
    走査するため、Linear と活性化を素の nn.Sequential(`self.net`)に並べる。
    入力は (B, 1, 28, 28) でも (B, 784) でも受け付ける(先頭で flatten)。

    Args:
        in_width: 入力次元(既定 784)
        hidden: 隠れ層幅のタプル(既定 (300, 100))
        num_classes: 出力クラス数(既定 10)
    """

    def __init__(
        self,
        in_width: int = 28 * 28,
        hidden: tuple[int, ...] = (300, 100),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_width = in_width
        self.num_classes = num_classes
        layers: list[nn.Module] = []
        prev = in_width
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class TeacherMLPSte(nn.Module):
    """784-300-100-10 MLPだが隠れ活性を y=1[preact>0] のSTEで二値化する
    teacher(teacher-ste)。

    隠れ活性が `transcribe_mlp` の意味論(y_i=1[Wx+b>0])と学習時に厳密一致
    するため、転写後のギャップは fan-in疎化誤差のみになる(仮説H2)。
    最終層(logits)は二値化しない(argmax保存のスコア)。

    構造は TeacherMLP と同じ nn.Linear 列を持つ(`self.net` の Linear のみを
    転写器が拾う)。ReLU の代わりに HeavisideSTE を隠れ活性に挟む。
    """

    def __init__(
        self,
        in_width: int = 28 * 28,
        hidden: tuple[int, ...] = (300, 100),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_width = in_width
        self.num_classes = num_classes
        # 転写器(transcribe_mlp / lower_bnn_mlp)が nn.Linear を modules() 順に
        # 一意に走査できるよう、Linear は ModuleList に1度だけ登録する
        # (同一 Linear を複数コンテナに入れると modules() が重複列挙してしまう)。
        self.linears = nn.ModuleList()
        prev = in_width
        for h in hidden:
            self.linears.append(nn.Linear(prev, h))
            prev = h
        self.linears.append(nn.Linear(prev, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.reshape(x.shape[0], -1)
        for lin in self.linears[:-1]:
            h = HeavisideSTE.apply(lin(h))
        return self.linears[-1](h)


class SmallTeacherCNN(nn.Module):
    """Conv(32)-Conv(64)-FCの小型CNN(MNIST/Fashion-MNIST蒸留teacher用)。

    BatchNorm+Dropoutで軽く正則化した標準的な2conv+2fc構成。
    28x28x`in_channels`入力を想定(MNIST/Fashion-MNIST共通)。

    Args:
        num_classes: 出力クラス数(既定10)
        in_channels: 入力チャネル数(既定1、グレースケール)
        dropout: 全結合層前のdropout率
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
