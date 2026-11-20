"""
W8: 少クラス実物体検出（FewClassVOCDetection）のテスト。

ネットワーク不要（torchvision.datasets.VOCDetection を合成スタブに差し替え）。
フィルタ・ラベル再割当・インデックスキャッシュ・pseudo-label差し替えを検証する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))


class _FakeVOCFewclassBase:
    """torchvision.datasets.VOCDetection の合成スタブ（複数サンプル・複数クラス）。

    - item0: person のみ → 採用（対象クラスに person を含む場合）
    - item1: cat のみ → fewclass={person,dog} では不採用
    - item2: dog + person → 採用、2ボックス
    - item3: 物体なし → 不採用
    """

    _ITEMS = [
        {'w': 100, 'h': 100, 'objects': [('person', (10, 10, 50, 50))]},
        {'w': 100, 'h': 100, 'objects': [('cat', (10, 10, 50, 50))]},
        {'w': 100, 'h': 100, 'objects': [
            ('dog', (0, 0, 20, 20)), ('person', (50, 50, 90, 90))]},
        {'w': 200, 'h': 100, 'objects': []},
    ]

    def __init__(self, root, year, image_set, download):
        from PIL import Image
        self._imgs = [
            Image.new('RGB', (it['w'], it['h']), color=(10, 20, 30))
            for it in self._ITEMS
        ]

    def __len__(self):
        return len(self._ITEMS)

    def __getitem__(self, i):
        it = self._ITEMS[i]
        objs = [
            {
                'name': name,
                'bndbox': {
                    'xmin': str(b[0] + 1), 'ymin': str(b[1] + 1),
                    'xmax': str(b[2] + 1), 'ymax': str(b[3] + 1),
                },
            }
            for name, b in it['objects']
        ]
        ann = {'annotation': {
            'size': {'width': str(it['w']), 'height': str(it['h']), 'depth': '3'},
            'object': objs,
        }}
        return self._imgs[i], ann


class _FakeVOCEmptyBase:
    """物体を一切含まない合成スタブ（キャッシュ再利用の検証用）。"""

    def __init__(self, root, year, image_set, download):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)


@pytest.fixture
def patched(monkeypatch):
    import torchvision
    monkeypatch.setattr(torchvision.datasets, 'VOCDetection', _FakeVOCFewclassBase)


class TestFewClassVOCDetectionFilter:
    def test_filters_to_target_classes_only(self, patched, tmp_path):
        """対象クラスを含む画像のみ採用され、他クラス専用画像は除外される。"""
        from data.detection_fewclass import FewClassVOCDetection
        ds = FewClassVOCDetection(
            root=str(tmp_path), train=False, img_size=128,
            classes=('person', 'dog'), use_rgb=True, cache_index=False,
        )
        assert len(ds) == 2  # item0, item2 のみ（item1=catのみ, item3=物体なし は除外）

    def test_label_remapping(self, patched, tmp_path):
        """ラベルは classes の並び順で 0..N-1 に再割当てされる。"""
        from data.detection_fewclass import FewClassVOCDetection
        ds = FewClassVOCDetection(
            root=str(tmp_path), train=False, img_size=128,
            classes=('person', 'dog'), use_rgb=True, cache_index=False,
        )
        _, targets1 = ds[0]  # item0: personのみ
        mask1 = targets1['mask'] > 0
        assert targets1['gt_labels'][mask1].tolist() == [0]  # person -> idx0

        _, targets2 = ds[1]  # item2: dog + person
        mask2 = targets2['mask'] > 0
        labels2 = sorted(targets2['gt_labels'][mask2].tolist())
        assert labels2 == [0, 1]  # person=0, dog=1

    def test_other_class_objects_dropped_in_multiclass_image(self, patched, tmp_path):
        """対象クラスでない物体は同一画像内でも無視される（cat単独画像がゼロ件）。"""
        from data.detection_fewclass import FewClassVOCDetection
        ds = FewClassVOCDetection(
            root=str(tmp_path), train=False, img_size=128,
            classes=('person',), use_rgb=True, cache_index=False,
        )
        # item0(person)のみ採用。item2はdog+personだがpersonは対象なので採用。
        assert len(ds) == 2

    def test_invalid_class_name_raises(self, tmp_path):
        from data.detection_fewclass import FewClassVOCDetection
        with pytest.raises(ValueError):
            FewClassVOCDetection(
                root=str(tmp_path), classes=('spaceship',), cache_index=False,
            )

    def test_lut_channels(self, patched, tmp_path):
        """LUT入力: 3ch RGB × 3閾値 = 9ch"""
        from data.detection_fewclass import FewClassVOCDetection
        ds = FewClassVOCDetection(
            root=str(tmp_path), train=False, img_size=128, n_thresholds=3,
            classes=('person', 'dog'), use_rgb=False, cache_index=False,
        )
        image, targets = ds[0]
        assert image.shape == (9, 128, 128)
        assert targets['heatmap'].shape == (2, 32, 32)  # n_classes=2

    def test_limit(self, patched, tmp_path):
        from data.detection_fewclass import FewClassVOCDetection
        ds = FewClassVOCDetection(
            root=str(tmp_path), train=False, classes=('person', 'dog'),
            use_rgb=True, limit=1, cache_index=False,
        )
        assert len(ds) == 1


class TestFewClassVOCDetectionCache:
    def test_cache_written_and_reused(self, patched, tmp_path, monkeypatch):
        """1回目のフィルタ結果がキャッシュされ、2回目は再計算されず再利用される。"""
        from data.detection_fewclass import FewClassVOCDetection
        ds1 = FewClassVOCDetection(
            root=str(tmp_path), train=False, classes=('person', 'dog'),
            use_rgb=True, cache_index=True,
        )
        assert len(ds1) == 2
        cache_path = ds1._cache_path()
        assert os.path.exists(cache_path)

        # ベースを「物体なし」スタブへ差し替え、キャッシュが使われるかを検証
        import torchvision
        monkeypatch.setattr(torchvision.datasets, 'VOCDetection', _FakeVOCEmptyBase)
        ds2 = FewClassVOCDetection(
            root=str(tmp_path), train=False, classes=('person', 'dog'),
            use_rgb=True, cache_index=True,
        )
        assert len(ds2) == 2  # キャッシュ由来のためベース変更の影響を受けない

    def test_cache_disabled_recomputes(self, patched, tmp_path, monkeypatch):
        """cache_index=False は毎回再計算する（キャッシュファイルを作らない）。"""
        from data.detection_fewclass import FewClassVOCDetection
        ds1 = FewClassVOCDetection(
            root=str(tmp_path), train=False, classes=('person', 'dog'),
            use_rgb=True, cache_index=False,
        )
        assert not os.path.exists(ds1._cache_path())


class TestFewClassPseudoLabels:
    def _make_pseudo_file(self, tmp_path, img_size=128, classes=('person', 'dog')):
        path = os.path.join(str(tmp_path), 'pseudo.pt')
        torch.save({
            'img_size': img_size,
            'classes': list(classes),
            'model': 'fake',
            'items': {
                0: {
                    'boxes': torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
                    'labels': torch.tensor([1]),  # dog
                },
                # idx1 は未登録 → 実GTにフォールバック
            },
        }, path)
        return path

    def test_pseudo_overrides_real_gt(self, patched, tmp_path):
        from data.detection_fewclass import FewClassVOCDetection
        pseudo_path = self._make_pseudo_file(tmp_path)
        ds = FewClassVOCDetection(
            root=str(tmp_path), train=False, img_size=128,
            classes=('person', 'dog'), use_rgb=True, cache_index=False,
            pseudo_label_path=pseudo_path,
        )
        _, targets = ds[0]  # 実GTはperson(idx0)だが、pseudoはdog(idx1)
        mask = targets['mask'] > 0
        assert targets['gt_labels'][mask].tolist() == [1]

    def test_missing_idx_falls_back_to_real_gt(self, patched, tmp_path):
        from data.detection_fewclass import FewClassVOCDetection
        pseudo_path = self._make_pseudo_file(tmp_path)
        ds = FewClassVOCDetection(
            root=str(tmp_path), train=False, img_size=128,
            classes=('person', 'dog'), use_rgb=True, cache_index=False,
            pseudo_label_path=pseudo_path,
        )
        _, targets = ds[1]  # idx1はpseudoファイルに無い→item2の実GT(dog+person)
        mask = targets['mask'] > 0
        assert sorted(targets['gt_labels'][mask].tolist()) == [0, 1]

    def test_img_size_mismatch_raises(self, patched, tmp_path):
        from data.detection_fewclass import FewClassVOCDetection
        pseudo_path = self._make_pseudo_file(tmp_path, img_size=160)
        with pytest.raises(ValueError):
            FewClassVOCDetection(
                root=str(tmp_path), train=False, img_size=128,
                classes=('person', 'dog'), use_rgb=True, cache_index=False,
                pseudo_label_path=pseudo_path,
            )

    def test_classes_mismatch_raises(self, patched, tmp_path):
        from data.detection_fewclass import FewClassVOCDetection
        pseudo_path = self._make_pseudo_file(tmp_path, classes=('dog', 'person'))
        with pytest.raises(ValueError):
            FewClassVOCDetection(
                root=str(tmp_path), train=False, img_size=128,
                classes=('person', 'dog'), use_rgb=True, cache_index=False,
                pseudo_label_path=pseudo_path,
            )


class TestBuildDatasetsFewclassDispatch:
    """train_detection.build_datasets の --dataset fewclass 分岐。"""

    @staticmethod
    def _args(arch, pseudo=None):
        import argparse
        return argparse.Namespace(
            dataset='fewclass', arch=arch, canvas_size=128,
            fewclass_classes='person,dog', fewclass_pseudo_labels=pseudo,
            n_thresholds=3, max_objects=20, limit_samples=1,
        )

    def test_lut_uses_binarized_input(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs('data-voc', exist_ok=True)
        from train_detection import build_datasets
        args = self._args('lut')
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        assert n_classes == 2
        image, _ = train_ds[0]
        assert image.shape[0] == 9  # 3ch * 3thresholds

    def test_cnn_uses_raw_rgb(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        os.makedirs('data-voc', exist_ok=True)
        from train_detection import build_datasets
        args = self._args('cnn')
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        image, _ = train_ds[0]
        assert image.shape[0] == 3

    def test_pseudo_labels_wired_to_train_only(self, patched, tmp_path, monkeypatch):
        """pseudo-labelは訓練データセットのみに適用され、テストは常に実GT。"""
        monkeypatch.chdir(tmp_path)
        os.makedirs('data-voc', exist_ok=True)
        pseudo_path = os.path.join('data-voc', 'pseudo.pt')
        torch.save({
            'img_size': 128, 'classes': ['person', 'dog'], 'model': 'fake',
            'items': {},
        }, pseudo_path)
        from train_detection import build_datasets
        args = self._args('lut', pseudo=pseudo_path)
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        assert train_ds.pseudo_labels is not None
        assert test_ds.pseudo_labels is None


class TestCocoToLocalMapping:
    """teacher(COCO)クラスIDから少クラスローカルIDへの写像（純関数、ネットワーク不要）。"""

    def test_eval_pretrained_build_coco_to_classes(self):
        from eval_pretrained_detector_voc import build_coco_to_classes
        coco_categories = ['__background__', 'person', 'bicycle', 'car', 'dog']
        mapping = build_coco_to_classes(coco_categories, ['person', 'car', 'dog'])
        assert mapping == {1: 0, 3: 1, 4: 2}

    def test_gen_pseudolabels_build_coco_to_local(self):
        from gen_teacher_pseudolabels import build_coco_to_local
        coco_categories = ['__background__', 'person', 'bicycle', 'car', 'dog']
        mapping = build_coco_to_local(coco_categories, ['person', 'car', 'dog'])
        assert mapping == {1: 0, 3: 1, 4: 2}

    def test_unknown_coco_category_raises(self):
        from eval_pretrained_detector_voc import build_coco_to_classes
        with pytest.raises(KeyError):
            build_coco_to_classes(['__background__'], ['person'])
