"""
WIDER FACE顔検出データセット（WiderFaceDetection）のテスト。

ネットワーク不要（torchvision.datasets.WIDERFace を合成スタブに差し替え）。
filter_face_boxes（純関数）の境界条件と、Datasetのフィルタ・キャッシュ・
配管（build_datasets）を検証する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))


# ============================================================
# filter_face_boxes（純関数）
# ============================================================

class TestFilterFaceBoxes:
    def test_anisotropic_scale_independent_sx_sy(self):
        """原寸1024x512→img128: sx=0.125, sy=0.25 が独立に適用される。"""
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(100.0, 100.0, 200.0, 100.0)]  # x,y,w,h @原寸
        attrs = {'blur': [0], 'invalid': [0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(1024.0, 512.0), img_size=128,
        )
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        # sx=0.125, sy=0.25
        assert x1 == pytest.approx(100.0 * 0.125)
        assert y1 == pytest.approx(100.0 * 0.25)
        assert x2 == pytest.approx(300.0 * 0.125)
        assert y2 == pytest.approx(200.0 * 0.25)
        assert labels == [0]

    def test_min_box_px_boundary_width_axis_kept_at_exactly_4(self):
        """原寸32px幅(sx=0.125)→リサイズ後ちょうど4.0のboxは残る。"""
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(0.0, 0.0, 32.0, 20.0)]  # w=32(→4.0), h=20(sy=0.25→5.0)
        attrs = {'blur': [0], 'invalid': [0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(1024.0, 512.0), img_size=128,
            min_box_px=4.0,
        )
        assert len(boxes) == 1
        _, _, x2, y2 = boxes[0]
        assert x2 == pytest.approx(4.0)
        assert y2 == pytest.approx(5.0)

    def test_min_box_px_boundary_width_axis_dropped_below_4(self):
        """原寸31px幅(sx=0.125)→リサイズ後3.875 < 4.0は落ちる。"""
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(0.0, 0.0, 31.0, 20.0)]
        attrs = {'blur': [0], 'invalid': [0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(1024.0, 512.0), img_size=128,
            min_box_px=4.0,
        )
        assert boxes == []
        assert labels == []

    def test_min_box_px_checks_both_axes(self):
        """min(w', h') が判定基準。幅は十分でも高さが小さければ落ちる。"""
        from data.detection_face import filter_face_boxes
        # sx=0.125, sy=0.25。w=100(→12.5, 十分) だが h=8(→2.0, 不足)
        bbox_xywh = [(0.0, 0.0, 100.0, 8.0)]
        attrs = {'blur': [0], 'invalid': [0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(1024.0, 512.0), img_size=128,
            min_box_px=4.0,
        )
        assert boxes == []

    def test_invalid_dropped_by_default(self):
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(0.0, 0.0, 60.0, 60.0), (0.0, 0.0, 60.0, 60.0)]
        attrs = {'blur': [0, 0], 'invalid': [0, 1]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(128.0, 128.0), img_size=128,
        )
        assert len(boxes) == 1

    def test_invalid_kept_when_drop_invalid_false(self):
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(0.0, 0.0, 60.0, 60.0), (0.0, 0.0, 60.0, 60.0)]
        attrs = {'blur': [0, 0], 'invalid': [0, 1]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(128.0, 128.0), img_size=128,
            drop_invalid=False,
        )
        assert len(boxes) == 2

    def test_degenerate_box_always_dropped(self):
        """(0,0,0,0)のような退化boxはdrop_invalid=Falseでも常に落ちる。"""
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(10.0, 10.0, 0.0, 0.0), (0.0, 0.0, 60.0, 60.0)]
        attrs = {'blur': [0, 0], 'invalid': [0, 0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(128.0, 128.0), img_size=128,
            drop_invalid=False,
        )
        assert len(boxes) == 1

    def test_blur_threshold(self):
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(0.0, 0.0, 60.0, 60.0), (0.0, 0.0, 60.0, 60.0)]
        attrs = {'blur': [2, 3], 'invalid': [0, 0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(128.0, 128.0), img_size=128,
            max_blur=2,
        )
        assert len(boxes) == 1  # blur=2は残り、blur=3は落ちる

    def test_area_descending_sort(self):
        from data.detection_face import filter_face_boxes
        bbox_xywh = [
            (0.0, 0.0, 10.0, 10.0),   # area=100 (小)
            (0.0, 0.0, 40.0, 40.0),   # area=1600 (大)
            (0.0, 0.0, 20.0, 20.0),   # area=400 (中)
        ]
        attrs = {'blur': [0, 0, 0], 'invalid': [0, 0, 0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(128.0, 128.0), img_size=128,
        )
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
        assert areas == sorted(areas, reverse=True)
        assert areas[0] == pytest.approx(1600.0)
        assert areas[-1] == pytest.approx(100.0)

    def test_labels_all_zero(self):
        from data.detection_face import filter_face_boxes
        bbox_xywh = [(0.0, 0.0, 60.0, 60.0), (10.0, 10.0, 30.0, 30.0)]
        attrs = {'blur': [0, 0], 'invalid': [0, 0]}
        boxes, labels = filter_face_boxes(
            bbox_xywh, attrs, orig_wh=(128.0, 128.0), img_size=128,
        )
        assert labels == [0] * len(boxes)
        assert len(boxes) == 2


# ============================================================
# WiderFaceDetection（合成スタブ + monkeypatch）
# ============================================================

class _FakeWiderFaceBase:
    """torchvision.datasets.WIDERFace の合成スタブ。

    - item0: 1顔(60x60、有効) → 採用
    - item1: 1顔だが3x3(既定min_box_px=4未満) → フィルタで0件 → 不採用
    - item2: 2顔(10x10=面積100, 40x40=面積1600) → 採用、面積降順で並ぶはず
    - item3: 顔なし → 不採用
    全て 128x128 画像（img_size既定128と一致させ sx=sy=1 で計算を単純化）。
    """

    _ITEMS = [
        {'w': 128, 'h': 128, 'boxes': [(10.0, 10.0, 60.0, 60.0)],
         'blur': [0], 'invalid': [0]},
        {'w': 128, 'h': 128, 'boxes': [(10.0, 10.0, 3.0, 3.0)],
         'blur': [0], 'invalid': [0]},
        {'w': 128, 'h': 128, 'boxes': [(0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 40.0, 40.0)],
         'blur': [0, 0], 'invalid': [0, 0]},
        {'w': 128, 'h': 128, 'boxes': [], 'blur': [], 'invalid': []},
    ]

    def __init__(self, root, split, download):
        from PIL import Image
        self._imgs = [
            Image.new('RGB', (it['w'], it['h']), color=(10, 20, 30))
            for it in self._ITEMS
        ]

    def __len__(self):
        return len(self._ITEMS)

    def __getitem__(self, i):
        it = self._ITEMS[i]
        n = len(it['boxes'])
        bbox = torch.tensor(it['boxes'], dtype=torch.float32) if n else torch.zeros(0, 4)
        target = {
            'bbox': bbox,
            'blur': torch.tensor(it['blur'], dtype=torch.long),
            'invalid': torch.tensor(it['invalid'], dtype=torch.long),
        }
        return self._imgs[i], target


class _FakeWiderFaceEmptyBase:
    """全画像0顔の合成スタブ（キャッシュ再利用の検証用）。"""

    def __init__(self, root, split, download):
        pass

    def __len__(self):
        return 4

    def __getitem__(self, i):
        from PIL import Image
        img = Image.new('RGB', (128, 128), color=(0, 0, 0))
        target = {
            'bbox': torch.zeros(0, 4),
            'blur': torch.zeros(0, dtype=torch.long),
            'invalid': torch.zeros(0, dtype=torch.long),
        }
        return img, target


@pytest.fixture
def patched(monkeypatch):
    import torchvision
    monkeypatch.setattr(torchvision.datasets, 'WIDERFace', _FakeWiderFaceBase)


class TestWiderFaceDetectionFilter:
    def test_filters_to_images_with_surviving_faces(self, patched, tmp_path):
        """フィルタ後に1件以上顔が残る画像のみ採用される。"""
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=False,
        )
        assert len(ds) == 2  # item0, item2 のみ

    def test_zero_face_image_removed_from_index(self, patched, tmp_path):
        """min_box_px未満で0件に落ちた画像(item1)はインデックスから消える。"""
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=False,
        )
        # item1(3x3の1顔のみ)がインデックスに含まれないことを、採用数=2で確認
        assert len(ds) == 2

    def test_lut_channels(self, patched, tmp_path):
        """LUT入力: 3ch RGB × 3閾値 = 9ch"""
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128, n_thresholds=3,
            use_rgb=False, cache_index=False,
        )
        image, targets = ds[0]
        assert image.shape == (9, 128, 128)
        assert targets['heatmap'].shape == (1, 32, 32)  # n_classes=1

    def test_cnn_channels(self, patched, tmp_path):
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=False,
        )
        image, targets = ds[0]
        assert image.shape == (3, 128, 128)

    def test_mask_sum_equals_surviving_box_count(self, patched, tmp_path):
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=False,
        )
        _, targets0 = ds[0]  # item0: 1顔
        assert targets0['mask'].sum().item() == 1

        _, targets1 = ds[1]  # item2: 2顔
        assert targets1['mask'].sum().item() == 2

    def test_max_objects_truncation_keeps_largest_area(self, patched, tmp_path):
        """max_objects切り詰めは面積降順ソート後の先頭(=大きい顔)を残す。"""
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=False, max_objects=1,
        )
        _, targets = ds[1]  # item2: 面積100(10x10)と面積1600(40x40)
        mask = targets['mask'] > 0
        assert mask.sum().item() == 1
        box = targets['gt_boxes'][mask][0]
        # 大きい方(40x40 @ (50,50)-(90,90))が残るはず
        x1, y1, x2, y2 = box.tolist()
        area = (x2 - x1) * (y2 - y1)
        assert area == pytest.approx(1600.0)

    def test_limit(self, patched, tmp_path):
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, use_rgb=True,
            limit=1, cache_index=False,
        )
        assert len(ds) == 1

    def test_download_true_raises(self, tmp_path):
        from data.detection_face import WiderFaceDetection
        with pytest.raises(ValueError):
            WiderFaceDetection(root=str(tmp_path), download=True, cache_index=False)

    def test_classes_attribute(self, patched, tmp_path):
        from data.detection_face import WiderFaceDetection
        ds = WiderFaceDetection(
            root=str(tmp_path), train=False, use_rgb=True, cache_index=False,
        )
        assert ds.classes == ['face']
        assert ds.n_classes == 1


class TestWiderFaceDetectionCache:
    def test_cache_written_and_reused(self, patched, tmp_path, monkeypatch):
        from data.detection_face import WiderFaceDetection
        ds1 = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=True,
        )
        assert len(ds1) == 2
        cache_path = ds1._cache_path('val')
        assert os.path.exists(cache_path)

        # ベースを「全画像0顔」スタブへ差し替え、キャッシュが使われるかを検証
        import torchvision
        monkeypatch.setattr(torchvision.datasets, 'WIDERFace', _FakeWiderFaceEmptyBase)
        ds2 = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=True,
        )
        assert len(ds2) == 2  # キャッシュ由来のためベース変更の影響を受けない

    def test_cache_disabled_recomputes(self, patched, tmp_path):
        from data.detection_face import WiderFaceDetection
        ds1 = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=False,
        )
        assert not os.path.exists(ds1._cache_path('val'))

    def test_cache_key_includes_min_box_px(self, patched, tmp_path):
        """min_box_pxが違えば別キャッシュファイル・別lenになる。"""
        from data.detection_face import WiderFaceDetection
        ds_default = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=True, min_box_px=4.0,
        )
        ds_strict = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=True, min_box_px=100.0,
        )
        assert ds_default._cache_path('val') != ds_strict._cache_path('val')
        assert os.path.exists(ds_default._cache_path('val'))
        assert os.path.exists(ds_strict._cache_path('val'))
        # min_box_px=100では全ての顔(<=60x60)がフィルタされ0件になる
        assert len(ds_strict) == 0
        assert len(ds_default) == 2

    def test_cache_key_includes_img_size(self, patched, tmp_path):
        from data.detection_face import WiderFaceDetection
        ds_a = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=128,
            use_rgb=True, cache_index=True,
        )
        ds_b = WiderFaceDetection(
            root=str(tmp_path), train=False, img_size=96,
            use_rgb=True, cache_index=True,
        )
        assert ds_a._cache_path('val') != ds_b._cache_path('val')


class TestBuildDatasetsFaceDispatch:
    """train_detection.build_datasets の --dataset face 分岐。"""

    @staticmethod
    def _args(arch, root, min_box_px=None, keep_invalid=None):
        import argparse
        return argparse.Namespace(
            dataset='face', arch=arch, canvas_size=128,
            face_root=root, face_min_box_px=min_box_px, face_max_blur=None,
            face_keep_invalid=keep_invalid,
            n_thresholds=3, max_objects=20, limit_samples=1,
        )

    def test_n_classes_is_one(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from train_detection import build_datasets
        args = self._args('lut', str(tmp_path))
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        assert n_classes == 1

    def test_lut_uses_binarized_input(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from train_detection import build_datasets
        args = self._args('lut', str(tmp_path))
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        image, _ = train_ds[0]
        assert image.shape[0] == 9

    def test_cnn_uses_raw_rgb(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from train_detection import build_datasets
        args = self._args('cnn', str(tmp_path))
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        image, _ = train_ds[0]
        assert image.shape[0] == 3

    def test_face_root_is_effective(self, patched, tmp_path, monkeypatch):
        """--face-root で指定したディレクトリがそのままdataset rootに使われる。"""
        other_root = tmp_path / 'other-widerface-root'
        other_root.mkdir()
        monkeypatch.chdir(tmp_path)
        from train_detection import build_datasets
        args = self._args('lut', str(other_root))
        train_ds, _, _ = build_datasets(args, output_stride=4, train=True, test=False)
        assert train_ds.root == str(other_root)

    def test_face_min_box_px_propagates(self, patched, tmp_path, monkeypatch):
        """--face-min-box-px がデータセットに伝播し、フィルタ結果に反映される。"""
        monkeypatch.chdir(tmp_path)
        from train_detection import build_datasets
        args_default = self._args('lut', str(tmp_path))
        train_default, _, _ = build_datasets(
            args_default, output_stride=4, train=True, test=False)
        assert train_default.min_box_px == 4.0

        args_strict = self._args('lut', str(tmp_path), min_box_px=100.0)
        train_strict, _, _ = build_datasets(
            args_strict, output_stride=4, train=True, test=False)
        assert train_strict.min_box_px == 100.0
