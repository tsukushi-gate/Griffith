"""セグメンテーションデータセット(Track W1)。

Oxford-IIIT Pet のバイナリセグメンテーション(pet vs 背景)ローダ。
torchvision OxfordIIITPet(target_types='segmentation')の trimap
(1=前景, 2=背景, 3=境界)をバイナリマスクへ変換する。

入力エンコーディング(vision.py の流儀に合わせる):
- 'float'       — [0,1] float画像(CNN/teacher用)。3チャネル
- 'thermometer' — 複数閾値二値化(ThresholdTransform と同一式)。
                  3×n_thresholds チャネルの {0,1} テンソル(DLGN用)

torchvision は optional 依存のため関数内で遅延importする(vision.py と同じ)。
データ配置は repo直下 data-oxford-pet を自動発見(resolve_data_root)。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .vision import resolve_data_root


def _torchvision():
    try:
        import torchvision
    except ImportError as e:
        raise ImportError(
            'セグメンテーションデータセットにはtorchvisionが必要です'
            '(uv sync --extra cpu または --extra cu128)'
        ) from e
    return torchvision


# trimap 境界(=3)の扱い
BORDER_MODES = ('foreground', 'background', 'ignore')
# 'ignore' 時にマスクへ書き込む無視ラベル(損失側で除外する)
IGNORE_INDEX = 255

ENCODINGS = ('float', 'thermometer')


def trimap_to_binary_mask(trimap: torch.Tensor,
                          border: str = 'foreground') -> torch.Tensor:
    """trimap(1=前景, 2=背景, 3=境界)→バイナリマスク(1=pet, 0=背景)。

    Args:
        trimap: (H, W) または (1, H, W) の整数テンソル(値 ∈ {1,2,3})
        border: 境界画素(=3)の扱い。'foreground'(既定)/'background'/
                'ignore'(IGNORE_INDEX=255 を書き込む)

    Returns:
        (H, W) の long テンソル。値 ∈ {0, 1}(ignore時は {0,1,255})
    """
    assert border in BORDER_MODES, f'border は {BORDER_MODES} のいずれか: {border}'
    t = trimap.squeeze(0) if trimap.dim() == 3 else trimap
    t = t.long()
    mask = torch.zeros_like(t)
    mask[t == 1] = 1
    if border == 'foreground':
        mask[t == 3] = 1
    elif border == 'ignore':
        mask[t == 3] = IGNORE_INDEX
    # 'background' は 0 のまま
    return mask


class ThermometerEncode3ch:
    """3chカラー画像の複数閾値二値化(pickle対応)。

    vision.ThresholdTransform と同一式: 閾値 (i+1)/(n+1), i=0..n-1。
    出力は (3*n_thresholds, H, W) の {0,1} float テンソル。
    チャネル順は [閾値0のRGB, 閾値1のRGB, ...](torch.catのdim=0連結)。
    """

    def __init__(self, n_thresholds: int):
        assert n_thresholds >= 1
        self.n_thresholds = n_thresholds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [(x > (i + 1) / (self.n_thresholds + 1)).float()
             for i in range(self.n_thresholds)], dim=0
        )


class OxfordPetBinarySeg(Dataset):
    """Oxford-IIIT Pet バイナリセグメンテーション(pet=1 / 背景=0)。

    各サンプルは (image, mask):
        image: encoding='float'       → (3, S, S) float in [0,1]
               encoding='thermometer' → (3*n_thresholds, S, S) の {0,1}
        mask:  (S, S) long ∈ {0,1}(border='ignore' 時は {0,1,255})

    画像は bilinear、マスクは nearest でリサイズする(ラベル値を保つ)。

    return_raw=True のとき __getitem__ は (image, mask, raw_image) の3要素を返す。
    raw_image は encoding に依らず常に生の (3, S, S) float[0,1] 画像で、
    KD時に thermometer入力(student)とは別系統で teacher へ渡すための入力
    (teacher入力規約=生[0,1]→teacher内部でImageNet正規化)。
    """

    def __init__(
        self,
        root: str | None = None,
        split: str = 'trainval',
        img_size: int = 64,
        encoding: str = 'float',
        n_thresholds: int = 3,
        border: str = 'foreground',
        download: bool = True,
        return_raw: bool = False,
    ):
        assert encoding in ENCODINGS, f'encoding は {ENCODINGS} のいずれか: {encoding}'
        assert border in BORDER_MODES, f'border は {BORDER_MODES} のいずれか: {border}'
        tv = _torchvision()
        data_root = resolve_data_root('data-oxford-pet', root)
        self.base = tv.datasets.OxfordIIITPet(
            data_root, split=split, target_types='segmentation',
            download=download,
        )
        self.img_size = img_size
        self.encoding = encoding
        self.n_thresholds = n_thresholds
        self.border = border
        self.return_raw = return_raw
        self._to_tensor = tv.transforms.ToTensor()
        self._resize_img = tv.transforms.Resize(
            (img_size, img_size),
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
        )
        self._thermo = (ThermometerEncode3ch(n_thresholds)
                        if encoding == 'thermometer' else None)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def in_channels(self) -> int:
        """モデル入力チャネル数(encodingに依存)"""
        return 3 * self.n_thresholds if self.encoding == 'thermometer' else 3

    def __getitem__(self, idx: int):
        img, trimap = self.base[idx]
        # 画像: PIL → float[0,1] → RGB3ch保証 → bilinearリサイズ
        x_raw = self._resize_img(self._to_tensor(img.convert('RGB')))
        x = self._thermo(x_raw) if self._thermo is not None else x_raw
        # trimap: PIL(mode L, 値1/2/3) → nearestリサイズ → バイナリ化
        import torchvision.transforms.functional as TF
        t = torch.as_tensor(
            TF.pil_to_tensor(trimap), dtype=torch.long,
        )  # (1, H, W)
        t = TF.resize(
            t, [self.img_size, self.img_size],
            interpolation=_torchvision().transforms.InterpolationMode.NEAREST,
        )
        mask = trimap_to_binary_mask(t, border=self.border)
        if self.return_raw:
            # teacher入力用の生float画像(encodingに依らず [0,1] RGB)も返す
            return x, mask, x_raw
        return x, mask


def load_oxford_pet_seg(
    root: str | None = None,
    img_size: int = 64,
    encoding: str = 'float',
    n_thresholds: int = 3,
    border: str = 'foreground',
    download: bool = True,
    return_raw: bool = False,
):
    """(train_set, test_set) を返す。split は trainval / test(公式分割)。

    return_raw=True のとき両セットは (image, mask, raw_image) の3要素を返す
    (KDで teacher へ生float画像を渡すため)。呼び出し側でtrain側だけ有効化
    したい場合は、返り値の `.return_raw` 属性を個別に切り替えてもよい。
    """
    common = dict(root=root, img_size=img_size, encoding=encoding,
                  n_thresholds=n_thresholds, border=border, download=download,
                  return_raw=return_raw)
    train_set = OxfordPetBinarySeg(split='trainval', **common)
    test_set = OxfordPetBinarySeg(split='test', **common)
    return train_set, test_set
