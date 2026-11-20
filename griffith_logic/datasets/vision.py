"""視覚データセット中央ローダ。

experiments/train.py の load_dataset 相当をパッケージへ集約する
(editor/experiments/recipeの重複解消の受け皿。既存スクリプトは無改造)。

- torchvisionはoptional依存: 関数内で遅延importする(コア依存に増やさない)
- データ配置はrepo直下の data-mnist / data-fashion-mnist / data-cifar /
  data-cifar-100 を自動発見する(root引数で明示指定も可能)
- VISION_DATASETS: name → メタ情報(in_width/classes)+ローダファクトリ
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

# repo直下(このファイルの2階層上)。editableインストールのレイアウト前提で
# data-mnist 等の自動発見に使う
_REPO_ROOT = Path(__file__).resolve().parents[2]

# 既定の二値化閾値(LUT回路は{0,1}入力が前提のため既定で二値化する)
DEFAULT_BINARIZE_THRESHOLD = 0.5

MNIST_IN_WIDTH = 28 * 28
MNIST20_IN_WIDTH = 20 * 20
CIFAR_PIXELS = 3 * 32 * 32


def _torchvision():
    """torchvisionを遅延import(未導入なら導入方法つきのエラー)"""
    try:
        import torchvision
    except ImportError as e:
        raise ImportError(
            '視覚データセットにはtorchvisionが必要です'
            '(uv sync --extra cpu または --extra cu128)'
        ) from e
    return torchvision


def resolve_data_root(dirname: str, root: str | None = None) -> str:
    """データ格納ディレクトリを解決する。

    root指定があればそれを優先し、なければ repo直下 → カレントディレクトリ の順に
    既存ディレクトリを探す。見つからなければrepo直下(ダウンロード先)を返す。
    """
    if root is not None:
        return str(root)
    for base in (_REPO_ROOT, Path.cwd()):
        candidate = base / dirname
        if candidate.is_dir():
            return str(candidate)
    return str(_REPO_ROOT / dirname)


class BinarizeTransform:
    """ピクセル値を閾値で{0,1}に二値化する変換(pickle対応)"""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.threshold).float()


class ThresholdTransform:
    """ピクセル値を複数の閾値で二値化し結合する変換(pickle対応)。

    experiments/train.py と同一のCIFAR前処理(difflogic互換)。

    平坦化順(Flatten後のビット位置)が2通りある:
        interleave=False(既定) — **しきい値major**。torch.cat(dim=0) なので
            出力は (T*C, H, W)、index = ((t*C + c)*H + row)*W + col
        interleave=True        — **しきい値minor**。出力は (C, H, W*T)、
            index = ((c*H + row)*W + col)*T + t
            社内先行実験の凍結符号化仕様 v1
            (`cifar/encode_cifar.py`: idx = ((c*32+row)*32+col)*3+t)と同一で、
            この順序でだけ「+3 = 右1画素 / +96 = 1行下」という近傍が成立する
            (先行実験 rot 配線 s_a=3 / s_b=96 の幾何の前提)。
    どちらもビットの集合は同一で、並び順だけが違う。
    """

    def __init__(self, threshold_count: int, interleave: bool = False):
        self.threshold_count = threshold_count
        self.interleave = interleave

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        bits = [(x > (i + 1) / (self.threshold_count + 1)).float()
                for i in range(self.threshold_count)]
        if not self.interleave:
            return torch.cat(bits, dim=0)
        # (C, H, W, T) → (C, H, W*T)。Flatten後にしきい値がminorになる
        stacked = torch.stack(bits, dim=-1)
        return stacked.reshape(x.shape[0], x.shape[1], -1)


class MNISTRemoveBorderTransform:
    """28x28 MNIST画像から黒いボーダーを適応的に除去して20x20にする変換。

    difflogicと同一のアルゴリズム: 各画像ごとに上下左右の黒行/列を検出し、
    合計8ピクセルずつ除去する。Resize(補間)と異なり元のピクセル値を保持する。
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # 水平方向: 完全に黒い行を検出
        horizontal_black_lines = (image == 0.).all(dim=2)
        top_black = 0
        while horizontal_black_lines[0, top_black] and top_black < 14:
            top_black += 1
        bottom_black = 0
        while horizontal_black_lines[0, -bottom_black - 1] and bottom_black < 14:
            bottom_black += 1
        assert top_black + bottom_black >= 8, (top_black, bottom_black)
        while top_black + bottom_black >= 10:
            if top_black > 0:
                top_black -= 1
            if bottom_black > 0:
                bottom_black -= 1
        if top_black + bottom_black == 9:
            if top_black > 0:
                top_black -= 1
            else:
                bottom_black -= 1
        assert top_black + bottom_black == 8, (top_black, bottom_black)
        image = image[:, top_black:28 - bottom_black]

        # 垂直方向: 完全に黒い列を検出
        vertical_black_lines = (image == 0.).all(dim=1)
        left_black = 0
        while vertical_black_lines[0, left_black] and left_black < 14:
            left_black += 1
        right_black = 0
        while vertical_black_lines[0, -right_black - 1] and right_black < 14:
            right_black += 1
        assert left_black + right_black >= 8, (left_black, right_black)
        while left_black + right_black >= 10:
            if left_black > 0:
                left_black -= 1
            if right_black > 0:
                right_black -= 1
        if left_black + right_black == 9:
            if left_black > 0:
                left_black -= 1
            else:
                right_black -= 1
        assert left_black + right_black == 8, (left_black, right_black)
        image = image[:, :, left_black:28 - right_black]

        return image


def _compose(tv, binarize_threshold: float | None, extra=None):
    """ToTensor(+ボーダー除去等)+任意の二値化のComposeを構築する"""
    transforms = [tv.transforms.ToTensor()]
    if extra is not None:
        transforms.append(extra)
    if binarize_threshold is not None:
        transforms.append(BinarizeTransform(binarize_threshold))
    return tv.transforms.Compose(transforms)


def load_mnist(
    root: str | None = None,
    binarize_threshold: float | None = DEFAULT_BINARIZE_THRESHOLD,
    download: bool = True,
):
    """MNIST(28x28=784入力/10クラス)の(train_set, test_set)を返す。

    Args:
        root: データディレクトリ(省略時はrepo直下 data-mnist を自動発見)
        binarize_threshold: 二値化閾値(Noneでグレースケール値を保持)
        download: データが無い場合にダウンロードするか
    """
    tv = _torchvision()
    transform = _compose(tv, binarize_threshold)
    data_root = resolve_data_root('data-mnist', root)
    train_set = tv.datasets.MNIST(
        data_root, train=True, download=download, transform=transform
    )
    test_set = tv.datasets.MNIST(data_root, train=False, transform=transform)
    return train_set, test_set


def load_mnist20x20(
    root: str | None = None,
    binarize_threshold: float | None = None,
    download: bool = True,
):
    """MNIST 20x20(difflogic互換ボーダー除去、400入力/10クラス)。

    補間はピクセル値を中間値にしてしまうため、crop(値保持)で20x20にする。
    """
    tv = _torchvision()
    transform = _compose(tv, binarize_threshold, extra=MNISTRemoveBorderTransform())
    data_root = resolve_data_root('data-mnist', root)
    train_set = tv.datasets.MNIST(
        data_root, train=True, download=download, transform=transform
    )
    test_set = tv.datasets.MNIST(data_root, train=False, transform=transform)
    return train_set, test_set


def load_fashion_mnist(
    root: str | None = None,
    binarize_threshold: float | None = DEFAULT_BINARIZE_THRESHOLD,
    download: bool = True,
):
    """Fashion-MNIST(784入力/10クラス)の(train_set, test_set)を返す"""
    tv = _torchvision()
    transform = _compose(tv, binarize_threshold)
    data_root = resolve_data_root('data-fashion-mnist', root)
    train_set = tv.datasets.FashionMNIST(
        data_root, train=True, download=download, transform=transform
    )
    test_set = tv.datasets.FashionMNIST(data_root, train=False, transform=transform)
    return train_set, test_set


def load_cifar10(
    root: str | None = None,
    threshold_count: int = 3,
    download: bool = True,
    interleave_thresholds: bool = False,
):
    """CIFAR-10(3x32x32×threshold_count入力/10クラス)。

    difflogic互換の複数閾値二値化(ThresholdTransform)を適用する。

    interleave_thresholds=True で平坦化順を「チャネル→行→列→しきい値minor」
    (社内先行実験の凍結符号化仕様)にする。既定Falseは従来の
    しきい値major順(bitwise不変)。
    """
    tv = _torchvision()
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        ThresholdTransform(threshold_count, interleave=interleave_thresholds),
    ])
    data_root = resolve_data_root('data-cifar', root)
    train_set = tv.datasets.CIFAR10(
        data_root, train=True, download=download, transform=transform
    )
    test_set = tv.datasets.CIFAR10(data_root, train=False, transform=transform)
    return train_set, test_set


def load_cifar100(
    root: str | None = None,
    threshold_count: int = 3,
    download: bool = True,
    interleave_thresholds: bool = False,
):
    """CIFAR-100(3x32x32×threshold_count入力/100クラス)。

    interleave_thresholds は load_cifar10 と同じ(既定Falseで従来順)。
    """
    tv = _torchvision()
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        ThresholdTransform(threshold_count, interleave=interleave_thresholds),
    ])
    data_root = resolve_data_root('data-cifar-100', root)
    train_set = tv.datasets.CIFAR100(
        data_root, train=True, download=download, transform=transform
    )
    test_set = tv.datasets.CIFAR100(data_root, train=False, transform=transform)
    return train_set, test_set


# name → (torchvision datasetクラス属性名, data-*ディレクトリ名)。
# load_raw が既存ローダと同一の基底データセット・同一引数で未二値化版を返すのに使う。
_RAW_DATASET_INFO: dict[str, tuple[str, str]] = {
    'mnist': ('MNIST', 'data-mnist'),
    'mnist20x20': ('MNIST', 'data-mnist'),
    'fashion-mnist': ('FashionMNIST', 'data-fashion-mnist'),
    'cifar-10-3-thresholds': ('CIFAR10', 'data-cifar'),
    'cifar-10-31-thresholds': ('CIFAR10', 'data-cifar'),
    'cifar-100-3-thresholds': ('CIFAR100', 'data-cifar-100'),
    'cifar-100-31-thresholds': ('CIFAR100', 'data-cifar-100'),
}


def load_raw(name: str, data_root: str | None = None, download: bool = True):
    """VISION_DATASETS の同名データセットを**未二値化**で返す。

    ToTensor のみを適用し、二値化(BinarizeTransform/ThresholdTransform)も
    teacher 側正規化(TEACHER_TRANSFORMS)も**適用しない**。teacher への正規化は
    teacher 側で行う(convert/teacher.py の transform)前提で、ここは生の
    [0,1] 画像テンソルを返す。

    既存ローダ(load_mnist 等)と**同一の基底データセットクラス・同一の
    コンストラクタ引数**で構築するため、`__getitem__(i)` のインデックス順が
    既存の(二値化済み)ローダと一致する。これが convert/cache.py の
    インデックス整列(teacherロジット[i] ↔ student[i])の前提になる。

    Args:
        name: VISION_DATASETS のキー(例 'mnist')。
        data_root: データディレクトリ(省略時は repo直下 data-* を自動発見)。
        download: データが無い場合にダウンロードするか。

    Returns:
        (train_set, test_set)。各サンプルは (image[C,H,W] float in [0,1], label)。

    Raises:
        ValueError: name が未対応。
    """
    if name not in _RAW_DATASET_INFO:
        raise ValueError(
            f'load_raw 未対応のデータセット: {name}(対応: {sorted(_RAW_DATASET_INFO)})'
        )
    tv = _torchvision()
    class_name, dirname = _RAW_DATASET_INFO[name]
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    resolved_root = resolve_data_root(dirname, data_root)
    dataset_cls = getattr(tv.datasets, class_name)
    train_set = dataset_cls(
        resolved_root, train=True, download=download, transform=transform
    )
    test_set = dataset_cls(resolved_root, train=False, transform=transform)
    return train_set, test_set


@dataclass(frozen=True)
class VisionDatasetSpec:
    """視覚データセットのメタ情報+ローダファクトリ(レジストリの単位)。

    load(root=None, **params) は (train_set, test_set) を返す。
    """

    name: str
    in_width: int  # Flatten後の入力ビット幅
    classes: int
    load: Callable[..., tuple]


# name → メタ情報+ローダ(experiments/train.py の DATASET_INFO と同じ名前空間)
VISION_DATASETS: dict[str, VisionDatasetSpec] = {
    'mnist': VisionDatasetSpec('mnist', MNIST_IN_WIDTH, 10, load_mnist),
    'mnist20x20': VisionDatasetSpec(
        'mnist20x20', MNIST20_IN_WIDTH, 10, load_mnist20x20
    ),
    'fashion-mnist': VisionDatasetSpec(
        'fashion-mnist', MNIST_IN_WIDTH, 10, load_fashion_mnist
    ),
    'cifar-10-3-thresholds': VisionDatasetSpec(
        'cifar-10-3-thresholds', CIFAR_PIXELS * 3, 10,
        partial(load_cifar10, threshold_count=3),
    ),
    'cifar-10-31-thresholds': VisionDatasetSpec(
        'cifar-10-31-thresholds', CIFAR_PIXELS * 31, 10,
        partial(load_cifar10, threshold_count=31),
    ),
    'cifar-100-3-thresholds': VisionDatasetSpec(
        'cifar-100-3-thresholds', CIFAR_PIXELS * 3, 100,
        partial(load_cifar100, threshold_count=3),
    ),
    'cifar-100-31-thresholds': VisionDatasetSpec(
        'cifar-100-31-thresholds', CIFAR_PIXELS * 31, 100,
        partial(load_cifar100, threshold_count=31),
    ),
}
