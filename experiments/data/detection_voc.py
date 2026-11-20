"""
Pascal VOC 物体検出データセット（CenterNet形式）。

torchvision.datasets.VOCDetection をラップし、Multi-MNIST 検出と同一の
CenterNet ターゲット（ヒートマップ・サイズ・オフセット）を生成する。

- 訓練: VOC2007 trainval + VOC2012 trainval（~16.5k枚）
- 評価: VOC2007 test
- 20クラス、画像は img_size×img_size に単純リサイズ（box座標も同倍率）
- output_stride=4 → (img_size/4)×(img_size/4) ヒートマップ
- LUT入力: チャンネル毎 ThresholdBinarize（3ch RGB × n_thresholds = 3n ch）
- CNN入力: 生RGB 3ch（use_rgb=True）
- データルートは相対パス ./data-voc（Modal/リモート規約）
"""

import os
import sys

import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detection_utils import encode_centernet_targets

# Pascal VOC 20クラス（標準順序、class_id = index）
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
VOC_CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


class ThresholdBinarize:
    """閾値による二値化変換（チャンネル毎、data/detection.py と共通仕様）。

    入力 (C, H, W) に対し n_thresholds 段の閾値で二値化し、
    (n_thresholds * C, H, W) を返す。閾値外側 = threshold順にチャンネル連結。
    """

    def __init__(self, n_thresholds: int):
        self.n_thresholds = n_thresholds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [(x > (i + 1) / (self.n_thresholds + 1)).float()
             for i in range(self.n_thresholds)], dim=0
        )


class VOCDetectionCenterNet(torch.utils.data.Dataset):
    """Pascal VOC → CenterNet ターゲットのデータセット。"""

    def __init__(
        self,
        root: str = './data-voc',
        train: bool = True,
        img_size: int = 160,
        n_thresholds: int = 3,
        output_stride: int = 4,
        max_objects: int = 20,
        n_classes: int = 20,
        use_rgb: bool = False,
        download: bool = False,
        limit: int | None = None,
        kd_teacher_input: bool = False,
    ):
        """
        Args:
            root: VOCデータのルートディレクトリ（VOCdevkit がこの下に展開される）
            train: True=VOC2007+2012 trainval, False=VOC2007 test
            img_size: リサイズ後の正方形サイズ
            n_thresholds: 閾値二値化の段数（use_rgb=False のとき有効）
            output_stride: ヒートマップの出力ストライド
            max_objects: 1画像あたりの最大オブジェクト数
            n_classes: クラス数（VOC=20）
            use_rgb: True=生RGB 3ch（CNN用）, False=ThresholdBinarize（LUT用）
            download: torchvision による自動ダウンロード
            limit: データセット長の上限（スモーク用、None=全件）
            kd_teacher_input: True で __getitem__ が (student_image, targets,
                teacher_image) を返す（W3-det KD用）。student=二値化(9ch),
                teacher=生RGB(3ch)。use_rgb=True とは併用不可（student は LUT 二値化前提）。
        """
        if kd_teacher_input and use_rgb:
            raise ValueError(
                'kd_teacher_input=True は use_rgb=False（LUT student 二値化入力）専用です'
            )
        self.kd_teacher_input = kd_teacher_input
        self.img_size = img_size
        self.n_thresholds = n_thresholds
        self.output_stride = output_stride
        self.max_objects = max_objects
        self.n_classes = n_classes
        self.use_rgb = use_rgb

        self.hm_h = img_size // output_stride
        self.hm_w = img_size // output_stride

        if train:
            specs = [('2007', 'trainval'), ('2012', 'trainval')]
        else:
            specs = [('2007', 'test')]

        self.bases = []
        for year, image_set in specs:
            self.bases.append(torchvision.datasets.VOCDetection(
                root=root, year=year, image_set=image_set, download=download,
            ))

        # ベースをまたぐインデックス境界
        lengths = [len(b) for b in self.bases]
        self.offsets = [0]
        for n in lengths:
            self.offsets.append(self.offsets[-1] + n)
        self.total = self.offsets[-1]
        if limit is not None:
            self.total = min(self.total, limit)

        self.to_tensor = torchvision.transforms.ToTensor()
        self.binarize = ThresholdBinarize(n_thresholds) if n_thresholds > 0 else None

    def __len__(self):
        return self.total

    def _locate(self, idx):
        """グローバルidx → (base, local_idx)。"""
        for bi in range(len(self.bases)):
            if self.offsets[bi] <= idx < self.offsets[bi + 1]:
                return self.bases[bi], idx - self.offsets[bi]
        raise IndexError(idx)

    @staticmethod
    def _parse_objects(annotation: dict):
        """VOC XML アノテーション辞書 → (gt_boxes_xyxy, gt_labels, (W, H))。"""
        size = annotation['size']
        orig_w = float(size['width'])
        orig_h = float(size['height'])

        objs = annotation.get('object', [])
        if isinstance(objs, dict):  # 単一オブジェクトは dict で来る
            objs = [objs]

        boxes = []
        labels = []
        for obj in objs:
            name = obj['name']
            if name not in VOC_CLASS_TO_IDX:
                continue
            bb = obj['bndbox']
            # VOC は 1-indexed。0-indexed に補正
            x1 = float(bb['xmin']) - 1.0
            y1 = float(bb['ymin']) - 1.0
            x2 = float(bb['xmax']) - 1.0
            y2 = float(bb['ymax']) - 1.0
            boxes.append((x1, y1, x2, y2))
            labels.append(VOC_CLASS_TO_IDX[name])
        return boxes, labels, (orig_w, orig_h)

    def _encode(self, boxes_xyxy, labels, orig_wh):
        """xyxy@原寸 → リサイズ後 (cx,cy,w,h) へスケールし CenterNet エンコード。"""
        orig_w, orig_h = orig_wh
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h

        gt_cxcywh = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            rx1, ry1 = x1 * sx, y1 * sy
            rx2, ry2 = x2 * sx, y2 * sy
            cx = (rx1 + rx2) / 2.0
            cy = (ry1 + ry2) / 2.0
            w = max(0.0, rx2 - rx1)
            h = max(0.0, ry2 - ry1)
            gt_cxcywh.append((cx, cy, w, h))

        return encode_centernet_targets(
            gt_cxcywh, labels,
            n_classes=self.n_classes,
            hm_h=self.hm_h, hm_w=self.hm_w,
            output_stride=self.output_stride,
            max_objects=self.max_objects,
        )

    def __getitem__(self, idx):
        base, local = self._locate(idx)
        img, target = base[local]  # img: PIL RGB, target: {'annotation': {...}}
        img = img.convert('RGB')

        boxes_xyxy, labels, orig_wh = self._parse_objects(target['annotation'])

        # 画像リサイズ → テンソル
        img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        image_rgb = self.to_tensor(img_resized)  # (3, H, W) [0,1]

        image = image_rgb
        if not self.use_rgb and self.binarize is not None:
            image = self.binarize(image_rgb)  # (3*n_thresholds, H, W)

        targets = self._encode(boxes_xyxy, labels, orig_wh)

        if self.kd_teacher_input:
            # student=二値化(image)、teacher(CNN)=生RGB(image_rgb)
            return image, targets, image_rgb
        return image, targets
