"""Pascal VOC セグメンテーションデータセット(21クラス)。

torchvision.datasets.VOCSegmentation(year=2012)のパレットPNGマスクを直読みする。
VOCのセグメンテーションマスクはPILモード'P'(パレット画像)で、パレット index が
そのままクラスindex(0=background, 1..20=クラス, 255=void/境界)である
(`TF.pil_to_tensor`はパレット→RGB変換をせず生indexを返す。パレット表示色は
可視化用でありクラス番号とは無関係)。void(255)は griffith_logic.datasets.
segmentation.IGNORE_INDEX と同一値のため、そのまま損失側のignore_indexとして使える。

入力エンコーディングは segmentation.py(OxfordPetBinarySeg)と同一契約:
- 'float'       — [0,1] float画像(CNN/teacher用)。3チャネル
- 'thermometer' — 複数閾値二値化(ThermometerEncode3ch、閾値式 (i+1)/(n+1))。
                  3×n_thresholds チャネルの {0,1} テンソル(DLGN用)

画像はbilinear、マスクはnearestでリサイズする(ラベル値を保つ)。
データ配置はrepo直下 data-voc を自動発見(resolve_data_root、雛形=detection_voc.py)。

データ拡張(augment=True、trainのみ): 幾何拡張(RandomScale 0.5〜2.0 +
RandomCrop to img_size + RandomHorizontalFlip、画像/マスクへ同一適用、マスクは
nearest、パディング領域はIGNORE_INDEX)と測光拡張(ColorJitter軽め、画像のみ、
thermometer/float化の前)を行う。valは常に無変換(load_voc_seg側で強制)。
"""

from __future__ import annotations

import random as _random

import torch
from torch.utils.data import Dataset

from griffith_logic.datasets.segmentation import (
    ENCODINGS,
    IGNORE_INDEX,
    ThermometerEncode3ch,
)
from griffith_logic.datasets.vision import resolve_data_root

# 幾何拡張のスケール範囲(RandomScale)
AUGMENT_SCALE_RANGE = (0.5, 2.0)


def random_scale_crop_flip(img, mask, img_size: int,
                            scale_range=AUGMENT_SCALE_RANGE, rng=None):
    """画像PILとマスクPILへ同一の幾何拡張(scale+crop+flip)を適用する。

    Args:
        img: PIL 'RGB'画像
        mask: PIL 'P'(パレット index)画像。値=クラスindex(0..20/255=void)
        img_size: 出力の正方形サイズ
        scale_range: (min, max) の一様分布からスケール係数を抽出
        rng: random.Random互換オブジェクト(uniform/randint/random)。
             Noneならモジュール既定のrandomを使う(訓練時は非決定的でよい、
             テストでは差し替えて決定的に検証する)。

    Returns:
        (img_out, mask_out): 共に img_size x img_size のPIL画像。
        スケールダウンでcanvasがimg_sizeに満たない領域は画像=黒、マスク=
        IGNORE_INDEXで埋める(パディング領域を損失から除外するため)。
    """
    if rng is None:
        rng = _random
    from PIL import Image

    scale = rng.uniform(*scale_range)
    new_size = max(1, round(img_size * scale))
    img_s = img.resize((new_size, new_size), Image.BILINEAR)
    mask_s = mask.resize((new_size, new_size), Image.NEAREST)

    if new_size < img_size:
        pad = img_size - new_size
        pad_l = rng.randint(0, pad)
        pad_t = rng.randint(0, pad)
        img_canvas = Image.new('RGB', (img_size, img_size), color=(0, 0, 0))
        img_canvas.paste(img_s, (pad_l, pad_t))
        mask_canvas = Image.new(mask_s.mode, (img_size, img_size),
                                color=IGNORE_INDEX)
        mask_canvas.paste(mask_s, (pad_l, pad_t))
        img_s, mask_s = img_canvas, mask_canvas
        new_size = img_size

    max_off = new_size - img_size
    off_x = rng.randint(0, max_off) if max_off > 0 else 0
    off_y = rng.randint(0, max_off) if max_off > 0 else 0
    img_c = img_s.crop((off_x, off_y, off_x + img_size, off_y + img_size))
    mask_c = mask_s.crop((off_x, off_y, off_x + img_size, off_y + img_size))

    if rng.random() < 0.5:
        img_c = img_c.transpose(Image.FLIP_LEFT_RIGHT)
        mask_c = mask_c.transpose(Image.FLIP_LEFT_RIGHT)

    return img_c, mask_c

# background(0) + 20クラス(標準VOC順序、detection_voc.VOC_CLASSESと同一20クラス)
VOC_SEG_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
VOC_SEG_N_CLASSES = len(VOC_SEG_CLASSES)  # 21


def _torchvision():
    try:
        import torchvision
    except ImportError as e:
        raise ImportError(
            'VOCセグメンテーションデータセットにはtorchvisionが必要です'
            '(uv sync --extra cpu または --extra cu128)'
        ) from e
    return torchvision


class VOCSeg(Dataset):
    """Pascal VOC 2012 セグメンテーション(21クラス: background + 20)。

    各サンプルは (image, mask):
        image: encoding='float'       → (3, S, S) float in [0,1]
               encoding='thermometer' → (3*n_thresholds, S, S) の {0,1}
        mask:  (S, S) long ∈ {0..20, 255}(255=void、IGNORE_INDEXと同一値)

    return_raw=True のとき __getitem__ は (image, mask, raw_image) の3要素を返す
    (KD時にteacherへ渡す生[0,1]画像、segmentation.OxfordPetBinarySegと同一規約)。

    augment=True: 幾何拡張(RandomScale 0.5〜2.0+RandomCrop to img_size+
    RandomHorizontalFlip、画像/マスク同一適用)と測光拡張(ColorJitter軽め、
    画像のみ)を毎回ランダムに適用する。既定Falseで後方互換(旧来のbilinear/
    nearestリサイズのみ)。呼び出し側(load_voc_seg)がtrainのみaugment=Trueに
    することを想定(valは常に無変換)。
    """

    def __init__(
        self,
        root: str | None = None,
        image_set: str = 'train',
        year: str = '2012',
        img_size: int = 128,
        encoding: str = 'float',
        n_thresholds: int = 3,
        download: bool = False,
        return_raw: bool = False,
        augment: bool = False,
    ):
        assert encoding in ENCODINGS, f'encoding は {ENCODINGS} のいずれか: {encoding}'
        tv = _torchvision()
        data_root = resolve_data_root('data-voc', root)
        self.base = tv.datasets.VOCSegmentation(
            data_root, year=year, image_set=image_set, download=download,
        )
        self.img_size = img_size
        self.encoding = encoding
        self.n_thresholds = n_thresholds
        self.return_raw = return_raw
        self.augment = augment
        self._to_tensor = tv.transforms.ToTensor()
        self._resize_img = tv.transforms.Resize(
            (img_size, img_size),
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
        )
        self._thermo = (ThermometerEncode3ch(n_thresholds)
                        if encoding == 'thermometer' else None)
        # 測光拡張(画像のみ、軽め)。augment=Falseなら未使用
        self._color_jitter = tv.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def in_channels(self) -> int:
        """モデル入力チャネル数(encodingに依存)"""
        return 3 * self.n_thresholds if self.encoding == 'thermometer' else 3

    @property
    def n_classes(self) -> int:
        return VOC_SEG_N_CLASSES

    def __getitem__(self, idx: int):
        img, target = self.base[idx]
        img = img.convert('RGB')
        import torchvision.transforms.functional as TF

        if self.augment:
            # 幾何拡張(scale+crop+flip、画像/マスク同一適用) → 既に img_size
            img, target = random_scale_crop_flip(img, target, self.img_size)
            x_raw = self._to_tensor(img)
            x_raw = self._color_jitter(x_raw)  # 測光拡張(画像のみ)
            t = torch.as_tensor(TF.pil_to_tensor(target), dtype=torch.long)
        else:
            # 画像: PIL → float[0,1] → RGB3ch保証 → bilinearリサイズ
            x_raw = self._resize_img(self._to_tensor(img))
            # マスク: PIL(mode P, 値=クラスindex 0..20/255=void) → nearestリサイズ
            t = torch.as_tensor(TF.pil_to_tensor(target), dtype=torch.long)  # (1, H, W)
            t = TF.resize(
                t, [self.img_size, self.img_size],
                interpolation=_torchvision().transforms.InterpolationMode.NEAREST,
            )
        x = self._thermo(x_raw) if self._thermo is not None else x_raw
        mask = t.squeeze(0)
        assert IGNORE_INDEX == 255, 'VOC void(255)はIGNORE_INDEXと同一値である前提'
        if self.return_raw:
            return x, mask, x_raw
        return x, mask


def load_voc_seg(
    root: str | None = None,
    img_size: int = 128,
    encoding: str = 'float',
    n_thresholds: int = 3,
    year: str = '2012',
    download: bool = False,
    return_raw: bool = False,
    augment: bool = False,
):
    """(train_set, val_set) を返す。VOC2012 segmentation公式分割(train/val)。

    return_raw=True のとき両セットは (image, mask, raw_image) の3要素を返す
    (KDで teacher へ生float画像を渡すため)。
    augment=True のとき **train_setのみ** データ拡張を有効化する(
    valは常に無変換=augment=Falseを強制、後方互換の評価基準を保つ)。
    """
    common = dict(root=root, img_size=img_size, encoding=encoding,
                  n_thresholds=n_thresholds, year=year, download=download,
                  return_raw=return_raw)
    train_set = VOCSeg(image_set='train', augment=augment, **common)
    test_set = VOCSeg(image_set='val', augment=False, **common)
    return train_set, test_set
