"""
合成Multi-MNIST検出データセット。

MNIST数字をランダムに96×96キャンバスに配置し、
CenterNet形式のGTヒートマップ・バウンディングボックスを生成する。

color=True でカラー版: 背景=CIFAR-10画像（自然画像）、
数字=ランダムな鮮やか色でアルファ合成。前景/背景分離に色が効く設定。
"""

import colorsys
import os
import random
import sys

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detection_utils import encode_centernet_targets

from griffith_logic.encoding.image_features import encode_image_features


def _random_vivid_color(rng: random.Random) -> tuple[float, float, float]:
    """彩度・明度の高いランダムRGB色を返す（前景/背景分離用の鮮やか色）。"""
    hue = rng.random()
    sat = 0.85 + 0.15 * rng.random()
    val = 0.85 + 0.15 * rng.random()
    return colorsys.hsv_to_rgb(hue, sat, val)


class ThresholdBinarize:
    """閾値による二値化変換（train_cnn.pyと共通）。"""

    def __init__(self, n_thresholds: int):
        self.n_thresholds = n_thresholds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [(x > (i + 1) / (self.n_thresholds + 1)).float()
             for i in range(self.n_thresholds)], dim=0
        )


class LowLevelFeatureBinarize:
    """サーモメータ + 固定の低レベル特徴検出器（Conv-DLGN §3.5 相当）。

    ``ThresholdBinarize`` の上位互換で、edge/curv/chroma を全て空にすると
    ``ThresholdBinarize(n_thresholds)`` と厳密に同じ出力になる（テストで担保）。
    検出器は学習しない固定カーネル。チャネル順の契約は
    ``griffith_logic.encoding.image_features.channel_names`` が真実源。
    """

    def __init__(
        self,
        n_thresholds: int,
        edge_thresholds: tuple[float, ...] = (),
        curv_thresholds: tuple[float, ...] = (),
        chroma_thresholds: tuple[float, ...] = (),
    ):
        self.n_thresholds = n_thresholds
        self.edge_thresholds = tuple(edge_thresholds)
        self.curv_thresholds = tuple(curv_thresholds)
        self.chroma_thresholds = tuple(chroma_thresholds)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return encode_image_features(
            x,
            n_thresholds=self.n_thresholds,
            edge_thresholds=self.edge_thresholds,
            curv_thresholds=self.curv_thresholds,
            chroma_thresholds=self.chroma_thresholds,
        )


def make_binarize(
    n_thresholds: int,
    edge_thresholds=(),
    curv_thresholds=(),
    chroma_thresholds=(),
):
    """入力二値化器のファクトリ。

    低レベル特徴（edge/curv/chroma）が1つでも指定されていれば
    ``LowLevelFeatureBinarize``、そうでなければ従来どおり
    ``ThresholdBinarize``（n_thresholds<=0 なら None）を返す。

    multimnist_color / fewclass / face の3データセットが同じ規則で
    二値化器を作れるようにするための共通経路。個別に書くと
    「あるデータセットだけ低レベル特徴が黙って無視される」事故になる
    （実際 fewclass に配管漏れがあり、arm が起動即エラーになった）。
    """
    if edge_thresholds or curv_thresholds or chroma_thresholds:
        return LowLevelFeatureBinarize(
            n_thresholds,
            edge_thresholds=edge_thresholds,
            curv_thresholds=curv_thresholds,
            chroma_thresholds=chroma_thresholds,
        )
    return ThresholdBinarize(n_thresholds) if n_thresholds > 0 else None


class MultiMNISTDetection(torch.utils.data.Dataset):
    """
    合成Multi-MNIST物体検出データセット。

    MNIST数字を大きなキャンバスにランダム配置し、
    CenterNet形式のGT（ヒートマップ、サイズ、オフセット）を生成する。
    """

    def __init__(
        self,
        mnist_root: str = './data-mnist',
        train: bool = True,
        canvas_size: int = 96,
        min_objects: int = 1,
        max_objects: int = 3,
        n_thresholds: int = 3,
        output_stride: int = 4,
        n_classes: int = 10,
        num_samples: int = 10000,
        seed: int = 0,
        kd_teacher_input: bool = False,
        kd_teacher_n_thresholds: int | None = None,
        color: bool = False,
        use_rgb: bool = False,
        cifar_root: str = './data-cifar10',
        edge_thresholds: tuple[float, ...] = (),
        curv_thresholds: tuple[float, ...] = (),
        chroma_thresholds: tuple[float, ...] = (),
    ):
        """
        Args:
            mnist_root: MNISTデータのルートディレクトリ
            train: 訓練セットを使うか
            canvas_size: キャンバスサイズ（正方形）
            min_objects: 画像あたりの最小オブジェクト数
            max_objects: 画像あたりの最大オブジェクト数
            n_thresholds: 閾値二値化のチャンネル数（0=二値化なし）
            output_stride: ヒートマップの出力ストライド
            n_classes: クラス数（MNIST=10）
            num_samples: データセットサイズ
            seed: ランダムシード
            kd_teacher_input: True で __getitem__ が (image, targets, teacher_image)
                の3つ組を返す（W3-det KD用）。color=False では teacher(CNN) も
                student(LUT) も同一の n_thresholds 二値化入力を消費するため
                teacher_image は student image と同一テンソル。color=True では
                VOCと同型で teacher_image=生RGB、student image=二値化。
            color: True でカラー版。背景=CIFAR-10画像（リサイズ）、
                数字=ランダム鮮やか色でアルファ合成した3chキャンバスになる。
            use_rgb: True で __getitem__ が生RGB 3ch を返す（CNN用）。
                False（既定）は ThresholdBinarize 適用後（LUT用、3*n_thresholds
                ch）。color=True 専用（color=Falseでは無意味なためエラー）。
            cifar_root: color=True のときのCIFAR-10データルート
                （torchvision標準ディレクトリ構造、未取得なら自動DL）。
            edge_thresholds: 1階差分（edge）検出器のしきい値。空で無効。
            curv_thresholds: 2階差分（curvature）検出器のしきい値。空で無効。
            chroma_thresholds: チャネル間比較・彩度のしきい値。空で無効（color=True専用）。
                以上3つは Conv-DLGN §3.5 の低レベル特徴検出器に対応する固定（非学習）
                前処理で、全て空なら従来の ThresholdBinarize と厳密に同一。
        """
        if use_rgb and not color:
            raise ValueError('use_rgb=True は color=True 専用です')
        if kd_teacher_input and use_rgb:
            raise ValueError(
                'kd_teacher_input=True は use_rgb=False（LUT student 二値化入力）専用です'
            )
        self.kd_teacher_input = kd_teacher_input
        # グレースケール(color=False)のKDでは teacher も二値化入力を消費するが、
        # teacher は自分が訓練された閾値数(通常3)のチャネル数しか受け取れない。
        # student だけ 31閾値(5bit)へ上げると teacher の conv1 が
        # 「expected input to have 3 channels, but got 31」で落ちるため、
        # teacher 用に別の ThresholdBinarize を持たせて student と独立させる。
        # None のときは従来通り student と同一テンソルを共有する。
        self.kd_teacher_n_thresholds = kd_teacher_n_thresholds
        self._kd_teacher_binarize = (
            ThresholdBinarize(kd_teacher_n_thresholds)
            if (kd_teacher_input and not color and kd_teacher_n_thresholds is not None
                and kd_teacher_n_thresholds != n_thresholds)
            else None
        )
        self.canvas_size = canvas_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.n_thresholds = n_thresholds
        self.output_stride = output_stride
        self.n_classes = n_classes
        self.num_samples = num_samples
        self.seed = seed
        self.color = color
        self.use_rgb = use_rgb

        # ヒートマップサイズ
        self.hm_h = canvas_size // output_stride
        self.hm_w = canvas_size // output_stride

        # MNISTデータを読み込み（画像とラベルを個別に保持）
        mnist = torchvision.datasets.MNIST(
            mnist_root, train=train, download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        self.mnist_images = mnist.data.float() / 255.0  # (N, 28, 28) [0,1]
        self.mnist_labels = mnist.targets  # (N,)

        # カラー版: CIFAR-10画像を背景として読み込み
        if self.color:
            cifar = torchvision.datasets.CIFAR10(
                cifar_root, train=train, download=True,
            )
            # (N, 32, 32, 3) uint8 → (N, 3, 32, 32) float [0,1]
            self.cifar_images = torch.from_numpy(cifar.data).float().permute(
                0, 3, 1, 2,
            ) / 255.0

        # 閾値二値化（+ 任意で固定の低レベル特徴検出器）
        self.edge_thresholds = tuple(edge_thresholds)
        self.curv_thresholds = tuple(curv_thresholds)
        self.chroma_thresholds = tuple(chroma_thresholds)
        if self.chroma_thresholds and not color:
            raise ValueError('chroma_thresholds は color=True 専用です（RGB 3ch前提）')
        self.binarize = make_binarize(
            n_thresholds,
            edge_thresholds=self.edge_thresholds,
            curv_thresholds=self.curv_thresholds,
            chroma_thresholds=self.chroma_thresholds,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 再現性のあるランダム状態
        rng = random.Random(self.seed * self.num_samples + idx)

        n_objects = rng.randint(self.min_objects, self.max_objects)

        # キャンバス作成（color=True: CIFAR-10背景、それ以外: 黒キャンバス）
        if self.color:
            canvas = self._make_color_background(rng)
        else:
            canvas = torch.zeros(1, self.canvas_size, self.canvas_size)

        # GT格納用
        gt_boxes = []  # (cx, cy, w, h) ピクセル座標
        gt_labels = []

        placed_rects = []  # 重複回避用 (x1, y1, x2, y2)

        for _ in range(n_objects):
            # ランダムにMNIST数字を選択
            digit_idx = rng.randint(0, len(self.mnist_images) - 1)
            digit_img = self.mnist_images[digit_idx]  # (28, 28)
            digit_label = self.mnist_labels[digit_idx].item()

            dh, dw = digit_img.shape

            # 配置可能な位置を探す（最大10回試行）
            placed = False
            for _ in range(10):
                x = rng.randint(0, self.canvas_size - dw)
                y = rng.randint(0, self.canvas_size - dh)

                # 重複チェック
                new_rect = (x, y, x + dw, y + dh)
                overlap = False
                for rect in placed_rects:
                    if (new_rect[0] < rect[2] and new_rect[2] > rect[0] and
                            new_rect[1] < rect[3] and new_rect[3] > rect[1]):
                        overlap = True
                        break

                if not overlap:
                    placed = True
                    break

            if not placed:
                continue

            if self.color:
                # ランダム鮮やか色でアルファ合成（alpha=数字強度、背景を透過）
                color = _random_vivid_color(rng)
                color_t = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
                alpha = digit_img.unsqueeze(0)  # (1, dh, dw)
                region = canvas[:, y:y + dh, x:x + dw]
                canvas[:, y:y + dh, x:x + dw] = (
                    region * (1 - alpha) + color_t * alpha
                )
            else:
                # キャンバスに配置（max演算で重ね合わせ）
                canvas[0, y:y + dh, x:x + dw] = torch.max(
                    canvas[0, y:y + dh, x:x + dw], digit_img,
                )
            placed_rects.append((x, y, x + dw, y + dh))

            # 中心座標とサイズ
            cx = x + dw / 2.0
            cy = y + dh / 2.0
            gt_boxes.append((cx, cy, float(dw), float(dh)))
            gt_labels.append(digit_label)

        # 入力エンコード
        if self.color:
            if self.use_rgb:
                image = canvas  # 生RGB 3ch（CNN用）
            elif self.binarize is not None:
                image = self.binarize(canvas)  # 3*n_thresholds ch（LUT用）
            else:
                image = canvas
        elif self.binarize is not None:
            image = self.binarize(canvas)
        else:
            image = canvas

        # GT生成
        targets = self._make_targets(gt_boxes, gt_labels)

        if self.kd_teacher_input:
            if self.color:
                # teacher(CNN)=生RGB、student(LUT)=二値化（VOCと同型）
                return image, targets, canvas
            # teacher(CNN)も student(LUT)も二値化入力を消費する。ただし閾値数が
            # 異なる場合(student 5bit × teacher 3bit)は teacher 用に作り直す。
            if self._kd_teacher_binarize is not None:
                return image, targets, self._kd_teacher_binarize(canvas)
            return image, targets, image
        return image, targets

    def _make_color_background(self, rng: random.Random) -> torch.Tensor:
        """CIFAR-10画像をランダムに1枚選びキャンバスサイズへリサイズする。"""
        bg_idx = rng.randint(0, self.cifar_images.shape[0] - 1)
        bg = self.cifar_images[bg_idx].unsqueeze(0)  # (1, 3, 32, 32)
        bg_resized = F.interpolate(
            bg, size=(self.canvas_size, self.canvas_size),
            mode='bilinear', align_corners=False,
        )
        return bg_resized.squeeze(0).clone()

    def _make_targets(
        self,
        gt_boxes: list[tuple[float, float, float, float]],
        gt_labels: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        CenterNet形式のGTを生成する。

        Args:
            gt_boxes: [(cx, cy, w, h), ...] ピクセル座標
            gt_labels: [class_id, ...]

        Returns:
            encode_centernet_targets() と同じ辞書（detection_utils で共通化）。
        """
        return encode_centernet_targets(
            gt_boxes, gt_labels,
            n_classes=self.n_classes,
            hm_h=self.hm_h, hm_w=self.hm_w,
            output_stride=self.output_stride,
            max_objects=self.max_objects,
        )


def collate_detection(batch):
    """
    検出データセット用のcollate関数。

    画像はスタック、ターゲットの各フィールドもスタックする。
    KDモード（各要素が3つ組 (image, targets, teacher_image)）では
    teacher_image もスタックして (images, targets, teacher_images) を返す。
    """
    images = torch.stack([item[0] for item in batch])
    target_keys = batch[0][1].keys()
    targets = {
        key: torch.stack([item[1][key] for item in batch])
        for key in target_keys
    }
    if len(batch[0]) == 3:
        teacher_images = torch.stack([item[2] for item in batch])
        return images, targets, teacher_images
    return images, targets
