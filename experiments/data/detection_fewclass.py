"""
少クラス実物体検出データセット（CenterNet形式、W8）。

Pascal VOC を少数クラス（既定: person, car, dog）にフィルタする。
- 該当クラスを1つ以上含む画像のみを採用し、GTボックスも該当クラスのみ残す
  （他クラスの物体は無視。画像自体は捨てない＝背景に写り込んでよい）
- クラスIDは `classes` の並び順で 0..len(classes)-1 に再割当てされる
  （VOC全20クラスの元IDとは別体系）
- フィルタ結果（画像インデックス）は data-voc 配下に .json でキャッシュし、
  teacher pseudo-label 生成スクリプトと訓練スクリプトで同一インデックス列を共有する
- pseudo_label_path 指定時、実GTの代わりに teacher 推論由来の疑似ボックスで
  CenterNetターゲットを構築する（gen_teacher_pseudolabels.py が生成するファイル形式）

VOCDetectionCenterNet（detection_voc.py）と共通部品（ThresholdBinarize、
encode_centernet_targets、VOC_CLASS_TO_IDX）を再利用する。
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
from data.detection_voc import VOC_CLASS_TO_IDX, ThresholdBinarize

DEFAULT_FEWCLASS = ('person', 'car', 'dog')


def _parse_objects_fewclass(annotation: dict, class_to_idx: dict[str, int]):
    """VOC XML アノテーション辞書 → (gt_boxes_xyxy, gt_labels, (W, H))。

    class_to_idx に含まれるクラス名の物体のみ残し、ラベルは class_to_idx の
    値（0..len(classes)-1）へ再割当てする。VOCDetectionCenterNet._parse_objects
    の少クラス版（class_to_idx を差し替えられるようにした変種）。
    """
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
        if name not in class_to_idx:
            continue
        bb = obj['bndbox']
        # VOC は 1-indexed。0-indexed に補正
        x1 = float(bb['xmin']) - 1.0
        y1 = float(bb['ymin']) - 1.0
        x2 = float(bb['xmax']) - 1.0
        y2 = float(bb['ymax']) - 1.0
        boxes.append((x1, y1, x2, y2))
        labels.append(class_to_idx[name])
    return boxes, labels, (orig_w, orig_h)


class FewClassVOCDetection(torch.utils.data.Dataset):
    """Pascal VOC を少数クラスへフィルタした CenterNet ターゲット データセット。"""

    def __init__(
        self,
        root: str = './data-voc',
        train: bool = True,
        img_size: int = 128,
        n_thresholds: int = 3,
        output_stride: int = 4,
        max_objects: int = 20,
        classes: tuple[str, ...] = DEFAULT_FEWCLASS,
        use_rgb: bool = False,
        kd_teacher_input: bool = False,
        edge_thresholds: tuple[float, ...] = (),
        curv_thresholds: tuple[float, ...] = (),
        chroma_thresholds: tuple[float, ...] = (),
        download: bool = False,
        limit: int | None = None,
        pseudo_label_path: str | None = None,
        cache_index: bool = True,
    ):
        """
        Args:
            root: VOCデータのルートディレクトリ
            train: True=VOC2007+2012 trainval, False=VOC2007 test
            img_size: リサイズ後の正方形サイズ（実物体は96pxだと小さすぎるため
                既定128。VOCと同様160も可）
            n_thresholds: 閾値二値化の段数（use_rgb=False のとき有効）
            output_stride: ヒートマップの出力ストライド
            max_objects: 1画像あたりの最大オブジェクト数
            classes: 採用するVOCクラス名のタプル（並び順がクラスID 0..N-1）
            use_rgb: True=生RGB 3ch（CNN用）, False=ThresholdBinarize（LUT用）
            download: torchvision による自動ダウンロード
            limit: データセット長の上限（スモーク用、None=全件）
            pseudo_label_path: 指定時、gen_teacher_pseudolabels.py が生成した
                .pt ファイルを読み込み、実GTの代わりに疑似ボックスでターゲットを
                構築する（teacher KD腕用）。img_size/classes がファイルの
                メタデータと一致しないと ValueError。
            cache_index: True でフィルタ結果を root 配下の .json にキャッシュし、
                2回目以降の構築を高速化する（teacher pseudo-label生成と訓練で
                同一のインデックス列を共有するために重要）。
        """
        # 2段階蒸留のため online KD を解禁する。student(LUT)は二値化入力、
        # teacher(CNN)は生RGB 3ch を消費するので、__getitem__ が3つ組を返せる
        # ようにする(multimnist_color と同一契約)。
        # 引数の矛盾は**データセット読み込みより先に**落とす(重い VOC 構築の後で
        # 落ちると、設定ミスの発見が数十秒遅れる)。
        if kd_teacher_input and use_rgb:
            raise ValueError(
                'kd_teacher_input=True は use_rgb=False(LUT student 二値化入力)専用です')
        for c in classes:
            if c not in VOC_CLASS_TO_IDX:
                raise ValueError(f'不明なVOCクラス名: {c}')
        self.root = root
        self.train = train
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.n_classes = len(self.classes)

        self.img_size = img_size
        self.n_thresholds = n_thresholds
        self.output_stride = output_stride
        self.max_objects = max_objects
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

        self._pairs = self._build_index(cache_index)
        self.total = len(self._pairs)
        if limit is not None:
            self.total = min(self.total, limit)

        self.to_tensor = torchvision.transforms.ToTensor()
        self.kd_teacher_input = kd_teacher_input
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

        self.pseudo_labels = None
        if pseudo_label_path is not None:
            self.pseudo_labels = self._load_pseudo_labels(pseudo_label_path)

    def _cache_path(self) -> str:
        split = 'train' if self.train else 'test'
        key = f'{split}_{"_".join(sorted(self.classes))}'
        return os.path.join(self.root, f'.fewclass_index_{key}.json')

    def _build_index(self, cache_index: bool) -> list[tuple[int, int]]:
        """該当クラスを1つ以上含む画像の (base_idx, local_idx) 一覧を作る。"""
        cache_path = self._cache_path() if cache_index else None
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
            return [(int(p[0]), int(p[1])) for p in data]

        pairs = []
        for bi, base in enumerate(self.bases):
            for li in range(len(base)):
                _, target = base[li]
                _, labels, _ = _parse_objects_fewclass(
                    target['annotation'], self.class_to_idx)
                if labels:
                    pairs.append((bi, li))

        if cache_path is not None:
            try:
                os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(pairs, f)
            except OSError:
                pass  # キャッシュ書き込み失敗は致命的でない（毎回再計算されるだけ）

        return pairs

    def _load_pseudo_labels(self, path: str) -> dict[int, dict]:
        data = torch.load(path, weights_only=False)
        if data.get('img_size') != self.img_size:
            raise ValueError(
                f'pseudo-labelの img_size={data.get("img_size")} が '
                f'データセットの img_size={self.img_size} と不一致: {path}'
            )
        if list(data.get('classes', [])) != self.classes:
            raise ValueError(
                f'pseudo-labelの classes={data.get("classes")} が '
                f'データセットの classes={self.classes} と不一致: {path}'
            )
        return data['items']

    def __len__(self):
        return self.total

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

    def _encode_pseudo(self, boxes_xyxy_scaled, labels):
        """既に img_size 座標系の xyxy（teacher予測）を CenterNet エンコード。"""
        gt_cxcywh = []
        for (x1, y1, x2, y2) in boxes_xyxy_scaled:
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
        bi, li = self._pairs[idx]
        base = self.bases[bi]
        img, target = base[li]
        img = img.convert('RGB')

        boxes_xyxy, labels, orig_wh = _parse_objects_fewclass(
            target['annotation'], self.class_to_idx)

        img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        image_rgb = self.to_tensor(img_resized)  # (3, H, W) [0,1]

        image = image_rgb
        if not self.use_rgb and self.binarize is not None:
            image = self.binarize(image_rgb)

        if self.pseudo_labels is not None and idx in self.pseudo_labels:
            p = self.pseudo_labels[idx]
            targets = self._encode_pseudo(
                p['boxes'].tolist(), p['labels'].tolist())
        else:
            targets = self._encode(boxes_xyxy, labels, orig_wh)

        if self.kd_teacher_input:
            # teacher(CNN)=生RGB、student(LUT)=二値化(multimnist_color と同型)
            return image, targets, image_rgb
        return image, targets
