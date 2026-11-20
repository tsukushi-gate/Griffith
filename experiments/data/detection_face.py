"""
WIDER FACE 顔検出データセット（CenterNet形式）。

torchvision.datasets.WIDERFace（train/val）をラップし、Pascal VOC/fewclass
データセット（detection_voc.py, detection_fewclass.py）と同一の CenterNet
ターゲット（ヒートマップ・サイズ・オフセット）を生成する。

- クラスは 'face' の1クラスのみ（labels は常に0）
- test split は使わない（WIDER FACEのtest GTは非公開のため）
- 異方性スケール（sx=img_size/W, sy=img_size/H を独立適用）でリサイズ後座標へ変換
- リサイズ後 min(w,h) < min_box_px の小さすぎる顔、invalid==1（既定で除外）、
  blur > max_blur の顔、退化box(w<=0 or h<=0)をフィルタする
- フィルタ後に1顔も残らない画像はデータセットから除外する
- フィルタ結果（画像インデックス）は root 配下の .json にキャッシュする
  （キーに img_size/min_box_px/max_blur/drop_invalid を含める。含めないと
  閾値変更が黙って無視され、間違ったデータで学習する事故になるため）

VOCDetectionCenterNet（detection_voc.py）と共通部品（ThresholdBinarize、
encode_centernet_targets）を再利用する。データ取得は scripts/fetch_widerface.py。
"""

import json
import os
import sys

import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detection_utils import encode_centernet_targets

from data.detection import make_binarize  # noqa: E402  (低レベル特徴)
from data.detection_voc import ThresholdBinarize


def filter_face_boxes(
    bbox_xywh,
    attrs: dict,
    orig_wh: tuple[float, float],
    *,
    img_size: int,
    min_box_px: float = 4.0,
    drop_invalid: bool = True,
    max_blur: int = 2,
) -> tuple[list[tuple[float, float, float, float]], list[int]]:
    """WIDER FACE原寸のxywh boxをフィルタし、リサイズ後xyxyへ変換する。

    Args:
        bbox_xywh: (N, 4) 原寸ピクセル座標の [x, y, w, h]（インデックス可能な
            シーケンス。torch.Tensor / list[tuple] いずれも可）
        attrs: 属性dict。'blur'・'invalid' キー（torchvision WIDERFaceの
            annotation形式、(N,) のインデックス可能なシーケンス）を参照する。
            それ以外のキー（expression等）は無視する。
        orig_wh: 原寸 (W, H)
        img_size: リサイズ後の正方形サイズ
        min_box_px: リサイズ後 min(w, h) がこの値未満の顔を除外する
            （境界値そのもの、つまり ==min_box_px は残す）
        drop_invalid: True で invalid==1 の顔を除外する
        max_blur: この値を超えるblurの顔を除外する（== max_blur は残す）

    Returns:
        (boxes_xyxy, labels): boxes_xyxy はリサイズ後座標の
        [(x1, y1, x2, y2), ...]（面積降順）、labels は全て0。
        max_objects による切り詰めは呼び出し側（encode_centernet_targets）が
        単純breakで行うため、切り捨てるなら小さい顔から落とせるよう
        面積降順で返す。
    """
    orig_w, orig_h = orig_wh
    sx = img_size / orig_w
    sy = img_size / orig_h

    blur = attrs.get('blur')
    invalid = attrs.get('invalid')

    n = len(bbox_xywh)
    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    for i in range(n):
        x, y, w, h = (float(v) for v in bbox_xywh[i])

        if w <= 0 or h <= 0:
            continue
        if drop_invalid and invalid is not None and int(invalid[i]) == 1:
            continue
        if blur is not None and int(blur[i]) > max_blur:
            continue

        x1, y1 = x * sx, y * sy
        x2, y2 = (x + w) * sx, (y + h) * sy
        rw, rh = x2 - x1, y2 - y1

        if min(rw, rh) < min_box_px:
            continue

        kept.append((rw * rh, (x1, y1, x2, y2)))

    kept.sort(key=lambda t: t[0], reverse=True)
    boxes = [box for _, box in kept]
    labels = [0] * len(boxes)
    return boxes, labels


class WiderFaceDetection(torch.utils.data.Dataset):
    """WIDER FACE → CenterNet ターゲット（1クラス='face'）のデータセット。"""

    def __init__(
        self,
        root: str = './data-widerface',
        train: bool = True,
        img_size: int = 128,
        n_thresholds: int = 3,
        output_stride: int = 4,
        max_objects: int = 20,
        use_rgb: bool = False,
        edge_thresholds: tuple[float, ...] = (),
        curv_thresholds: tuple[float, ...] = (),
        chroma_thresholds: tuple[float, ...] = (),
        download: bool = False,
        limit: int | None = None,
        min_box_px: float = 4.0,
        drop_invalid: bool = True,
        max_blur: int = 2,
        cache_index: bool = True,
    ):
        """
        Args:
            root: WIDER FACEデータのルートディレクトリ（torchvision.datasets.
                WIDERFace の規約通り、この下の root/widerface/ を見る）
            train: True=train split, False=val split（testは使わない。
                GTが非公開のため）
            img_size: リサイズ後の正方形サイズ
            n_thresholds: 閾値二値化の段数（use_rgb=False のとき有効）
            output_stride: ヒートマップの出力ストライド
            max_objects: 1画像あたりの最大顔数
            use_rgb: True=生RGB 3ch（CNN用）, False=ThresholdBinarize（LUT用）
            download: 自動ダウンロード非対応。True を渡すと ValueError
                （scripts/fetch_widerface.py を使うよう案内）
            limit: データセット長の上限（スモーク用、None=全件）
            min_box_px: リサイズ後 min(w, h) がこの値未満の顔を除外する
            drop_invalid: True で invalid==1 の顔を除外する
            max_blur: この値を超えるblurの顔を除外する
            cache_index: True でフィルタ結果を root 配下の .json にキャッシュ
                する。キーに img_size/min_box_px/max_blur/drop_invalid を
                含める（閾値変更が黙って無視される事故を防ぐため）。
        """
        if download:
            raise ValueError(
                'WiderFaceDetection は download=True 未対応です。'
                '事前に `uv run python scripts/fetch_widerface.py` を実行して'
                'データを取得してください。'
            )
        self.root = root
        self.train = train
        self.classes = ['face']
        self.n_classes = 1

        self.img_size = img_size
        self.n_thresholds = n_thresholds
        self.output_stride = output_stride
        self.max_objects = max_objects
        self.use_rgb = use_rgb
        self.min_box_px = min_box_px
        self.drop_invalid = drop_invalid
        self.max_blur = max_blur

        self.hm_h = img_size // output_stride
        self.hm_w = img_size // output_stride

        self.split = 'train' if train else 'val'
        self.base = torchvision.datasets.WIDERFace(
            root=root, split=self.split, download=False,
        )

        self._indices = self._build_index(cache_index)
        self.total = len(self._indices)
        if limit is not None:
            self.total = min(self.total, limit)

        self.to_tensor = torchvision.transforms.ToTensor()
        self.edge_thresholds = tuple(edge_thresholds)
        self.curv_thresholds = tuple(curv_thresholds)
        self.chroma_thresholds = tuple(chroma_thresholds)
        # Conv-DLGN §3.5 の低レベル特徴検出器。全て空なら
        # 従来の ThresholdBinarize と厳密に同一（make_binarize が保証）。
        self.binarize = make_binarize(
            n_thresholds,
            edge_thresholds=self.edge_thresholds,
            curv_thresholds=self.curv_thresholds,
            chroma_thresholds=self.chroma_thresholds,
        )

    def _cache_path(self, split: str) -> str:
        key = (
            f'{split}_s{self.img_size}_m{self.min_box_px:g}'
            f'_b{self.max_blur}_i{int(self.drop_invalid)}'
        )
        return os.path.join(self.root, f'.widerface_index_{key}.json')

    def _filter(self, target: dict, orig_wh: tuple[float, float]):
        return filter_face_boxes(
            target['bbox'], target, orig_wh,
            img_size=self.img_size,
            min_box_px=self.min_box_px,
            drop_invalid=self.drop_invalid,
            max_blur=self.max_blur,
        )

    def _build_index(self, cache_index: bool) -> list[int]:
        """フィルタ後に1顔以上残る画像のインデックス一覧を作る。"""
        cache_path = self._cache_path(self.split) if cache_index else None
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
            return [int(i) for i in data]

        indices = []
        for i in range(len(self.base)):
            img, target = self.base[i]
            orig_wh = img.size  # (W, H)。PIL遅延読み込みでヘッダのみ参照
            _, labels = self._filter(target, orig_wh)
            if labels:
                indices.append(i)

        if cache_path is not None:
            try:
                os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(indices, f)
            except OSError:
                pass  # キャッシュ書き込み失敗は致命的でない（毎回再計算されるだけ）

        return indices

    def __len__(self):
        return self.total

    def _encode(self, boxes_xyxy, labels):
        """既に img_size 座標系の xyxy を CenterNet エンコード。"""
        gt_cxcywh = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            gt_cxcywh.append((cx, cy, w, h))
        return encode_centernet_targets(
            gt_cxcywh, labels,
            n_classes=self.n_classes,
            hm_h=self.hm_h, hm_w=self.hm_w,
            output_stride=self.output_stride,
            max_objects=self.max_objects,
        )

    def __getitem__(self, idx):
        base_idx = self._indices[idx]
        img, target = self.base[base_idx]
        orig_wh = img.size  # (W, H)
        img = img.convert('RGB')

        boxes_xyxy, labels = self._filter(target, orig_wh)

        img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        image_rgb = self.to_tensor(img_resized)  # (3, H, W) [0,1]

        image = image_rgb
        if not self.use_rgb and self.binarize is not None:
            image = self.binarize(image_rgb)

        targets = self._encode(boxes_xyxy, labels)

        return image, targets
