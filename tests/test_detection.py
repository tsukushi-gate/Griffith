"""
CenterNet + LUT 物体検出モデルのテスト。

SpatialGroupSum、データセット、モデル、損失関数のユニットテスト。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from griffith_logic import LutConv2d, OrPooling2d, UpSample2d
from griffith_logic.spatial_group_sum import SpatialGroupSum

# ============================================================
# 共通CenterNetターゲットエンコード（detection_utils）
# ============================================================

class TestEncodeCenterNetTargets:
    def test_known_values(self):
        """既知の単一ボックスで各フィールドが期待値になるか（挙動固定）。"""
        from detection_utils import encode_centernet_targets
        # クラス2、中心(50,30)px、サイズ28×28px、stride4、24×24ヒートマップ
        t = encode_centernet_targets(
            [(50.0, 30.0, 28.0, 28.0)], [2],
            n_classes=10, hm_h=24, hm_w=24,
            output_stride=4, max_objects=3,
        )
        assert t['heatmap'].shape == (10, 24, 24)
        assert t['heatmap'][2].max() > 0.9  # ピーク
        # cx_hm=12.5→12, cy_hm=7.5→7 → index=7*24+12=180
        assert t['indices'][0].item() == 180
        assert abs(t['size'][0, 0].item() - (7.0 / 24)) < 1e-6
        assert abs(t['size'][0, 1].item() - (7.0 / 24)) < 1e-6
        assert abs(t['offset'][0, 0].item() - 0.5) < 1e-6
        assert abs(t['offset'][0, 1].item() - 0.5) < 1e-6
        assert t['mask'][0].item() == 1.0
        assert t['mask'][1].item() == 0.0
        assert t['gt_boxes'][0].tolist() == [36.0, 16.0, 64.0, 44.0]
        assert t['gt_labels'][0].item() == 2

    def test_multimnist_delegates_to_shared(self):
        """MultiMNIST._make_targets が共通エンコーダと一致するか（回帰）。"""
        from data.detection import MultiMNISTDetection
        from detection_utils import encode_centernet_targets
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=1,
        )
        boxes = [(40.0, 50.0, 20.0, 24.0), (10.0, 12.0, 8.0, 8.0)]
        labels = [3, 7]
        via_method = ds._make_targets(boxes, labels)
        via_shared = encode_centernet_targets(
            boxes, labels, n_classes=10, hm_h=24, hm_w=24,
            output_stride=4, max_objects=3,
        )
        for key in via_method:
            assert torch.equal(via_method[key], via_shared[key]), key


# ============================================================
# Pascal VOC データセット（ネットワーク不要・合成XML）
# ============================================================

class _FakeVOCBase:
    """torchvision.datasets.VOCDetection の合成スタブ（1サンプル）。"""

    def __init__(self, root, year, image_set, download):
        from PIL import Image
        self._img = Image.new('RGB', (100, 200), color=(120, 30, 200))
        self._ann = {'annotation': {
            'size': {'width': '100', 'height': '200', 'depth': '3'},
            'object': [{
                'name': 'cat',
                'bndbox': {'xmin': '11', 'ymin': '21', 'xmax': '51', 'ymax': '61'},
            }],
        }}

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self._img, self._ann


class TestVOCDetection:
    @pytest.fixture
    def patched(self, monkeypatch):
        import torchvision
        monkeypatch.setattr(
            torchvision.datasets, 'VOCDetection', _FakeVOCBase)

    def test_lut_channels(self, patched):
        """LUT入力: 3ch RGB × 3閾値 = 9ch"""
        from data.detection_voc import VOCDetectionCenterNet
        ds = VOCDetectionCenterNet(
            train=False, img_size=160, n_thresholds=3,
            output_stride=4, max_objects=20, use_rgb=False,
        )
        image, targets = ds[0]
        assert image.shape == (9, 160, 160)
        assert targets['heatmap'].shape == (20, 40, 40)

    def test_cnn_rgb_channels(self, patched):
        """CNN入力: 生RGB 3ch"""
        from data.detection_voc import VOCDetectionCenterNet
        ds = VOCDetectionCenterNet(
            train=False, img_size=160, use_rgb=True, output_stride=4,
        )
        image, _ = ds[0]
        assert image.shape == (3, 160, 160)
        assert image.min() >= 0 and image.max() <= 1

    def test_box_scaling(self, patched):
        """XML box(1-indexed)→160px座標へ正しくスケール・エンコードされるか。"""
        from data.detection_voc import VOC_CLASS_TO_IDX, VOCDetectionCenterNet
        ds = VOCDetectionCenterNet(
            train=False, img_size=160, use_rgb=True, output_stride=4,
        )
        _, targets = ds[0]
        # xmin11,ymin21,xmax51,ymax61 → 0-indexed(10,20,50,60)
        # sx=160/100=1.6, sy=160/200=0.8 → (16,16,80,48)
        assert targets['gt_boxes'][0].tolist() == [16.0, 16.0, 80.0, 48.0]
        assert targets['gt_labels'][0].item() == VOC_CLASS_TO_IDX['cat']
        assert targets['mask'][0].item() == 1.0

    def test_parse_objects_single_dict(self):
        """単一オブジェクトが dict で来ても list 化して処理するか。"""
        from data.detection_voc import VOCDetectionCenterNet
        ann = {
            'size': {'width': '50', 'height': '50'},
            'object': {  # dict（単一）
                'name': 'dog',
                'bndbox': {'xmin': '5', 'ymin': '5', 'xmax': '25', 'ymax': '25'},
            },
        }
        boxes, labels, (w, h) = VOCDetectionCenterNet._parse_objects(ann)
        assert len(boxes) == 1 and len(labels) == 1
        assert (w, h) == (50.0, 50.0)

    def test_limit(self, patched):
        """limit でデータセット長が制限されるか。"""
        from data.detection_voc import VOCDetectionCenterNet
        ds = VOCDetectionCenterNet(train=False, use_rgb=True, limit=1)
        assert len(ds) == 1


# ============================================================
# SpatialGroupSum
# ============================================================

class TestSpatialGroupSum:
    def test_output_shape(self):
        """出力形状が正しいか"""
        layer = SpatialGroupSum(n_outputs=2, scale=1.0)
        x = torch.rand(4, 32, 8, 8)
        y = layer(x)
        assert y.shape == (4, 2, 8, 8)

    def test_output_range(self):
        """出力値域が[0, group_size * scale]に収まるか"""
        scale = 0.5
        layer = SpatialGroupSum(n_outputs=3, scale=scale)
        x = torch.rand(2, 12, 6, 6)  # group_size = 4
        y = layer(x)
        assert y.min() >= 0
        assert y.max() <= 4 * scale + 1e-6

    def test_gradient_flow(self):
        """勾配が正しく流れるか"""
        layer = SpatialGroupSum(n_outputs=2, scale=1.0)
        x = torch.rand(2, 8, 4, 4, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert (x.grad > 0).all()  # sum + scale → 勾配は全て正

    def test_channel_divisibility_error(self):
        """チャンネル数が割り切れない場合にエラーが出るか"""
        layer = SpatialGroupSum(n_outputs=3)
        x = torch.rand(2, 7, 4, 4)
        with pytest.raises(AssertionError):
            layer(x)

    def test_scale_effect(self):
        """scaleが出力に正しく反映されるか"""
        x = torch.ones(1, 4, 2, 2)
        layer1 = SpatialGroupSum(n_outputs=2, scale=1.0)
        layer2 = SpatialGroupSum(n_outputs=2, scale=0.5)
        y1 = layer1(x)
        y2 = layer2(x)
        assert torch.allclose(y1, y2 * 2)


# ============================================================
# データセット
# ============================================================

class TestMultiMNISTDetection:
    @pytest.fixture
    def dataset(self):
        from data.detection import MultiMNISTDetection
        return MultiMNISTDetection(
            train=True, canvas_size=96,
            min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4,
            num_samples=10, seed=42,
        )

    def test_output_format(self, dataset):
        """出力形式が正しいか"""
        image, targets = dataset[0]
        assert image.shape[0] == 3  # n_thresholds=3
        assert image.shape[1] == 96
        assert image.shape[2] == 96
        assert 'heatmap' in targets
        assert 'size' in targets
        assert 'offset' in targets
        assert 'indices' in targets
        assert 'mask' in targets

    def test_heatmap_shape(self, dataset):
        """ヒートマップの形状と値域"""
        _, targets = dataset[0]
        hm = targets['heatmap']
        assert hm.shape == (10, 24, 24)
        assert hm.min() >= 0
        assert hm.max() <= 1

    def test_heatmap_has_peaks(self, dataset):
        """ヒートマップにピークが存在するか"""
        _, targets = dataset[0]
        mask = targets['mask']
        n_objects = int(mask.sum().item())
        assert n_objects >= 1
        assert targets['heatmap'].max() > 0.9  # ガウシアンの中心は≈1

    def test_reproducibility(self, dataset):
        """同じインデックスで同じ結果が得られるか"""
        img1, t1 = dataset[5]
        img2, t2 = dataset[5]
        assert torch.equal(img1, img2)
        assert torch.equal(t1['heatmap'], t2['heatmap'])

    def test_collate(self, dataset):
        """collate関数のテスト"""
        from data.detection import collate_detection
        batch = [dataset[i] for i in range(4)]
        images, targets = collate_detection(batch)
        assert images.shape == (4, 3, 96, 96)
        assert targets['heatmap'].shape == (4, 10, 24, 24)


# ============================================================
# カラー版 Multi-MNIST（ネットワーク不要）
# ============================================================

class _FakeCIFAR10:
    """torchvision.datasets.CIFAR10 の合成スタブ（ネットワーク不要、決定的）。"""

    def __init__(self, root, train, download):
        rng = np.random.RandomState(0)
        self.data = rng.randint(0, 256, size=(8, 32, 32, 3)).astype(np.uint8)
        self.targets = [0] * 8


class TestMultiMNISTDetectionColor:
    @pytest.fixture
    def patched_cifar(self, monkeypatch):
        import torchvision
        monkeypatch.setattr(torchvision.datasets, 'CIFAR10', _FakeCIFAR10)

    def test_lut_input_channels(self, patched_cifar):
        """color=True, use_rgb=False(既定): 3*n_thresholds ch（LUT用）"""
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            color=True,
        )
        image, targets = ds[0]
        assert image.shape == (9, 96, 96)
        assert image.min() >= 0 and image.max() <= 1
        assert 'heatmap' in targets

    def test_cnn_input_channels(self, patched_cifar):
        """color=True, use_rgb=True: 生RGB 3ch（CNN用）"""
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            color=True, use_rgb=True,
        )
        image, _ = ds[0]
        assert image.shape == (3, 96, 96)
        assert image.min() >= 0 and image.max() <= 1

    def test_low_level_features_extend_channels(self, patched_cifar):
        """edge/curv/chroma を有効にすると契約どおりのチャネル数になる。"""
        from data.detection import MultiMNISTDetection

        from griffith_logic.encoding.image_features import n_output_channels

        feats = dict(
            edge_thresholds=(0.15, 0.35),
            curv_thresholds=(0.2,),
            chroma_thresholds=(0.1, 0.25),
        )
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            color=True, **feats,
        )
        image, _ = ds[0]
        expected = n_output_channels(3, n_thresholds=3, **feats)
        assert image.shape == (expected, 96, 96)
        assert torch.all((image == 0) | (image == 1))

    def test_low_level_features_default_off_is_identical(self, patched_cifar):
        """既定（全て空）は従来の ThresholdBinarize と厳密に同一の出力。"""
        from data.detection import MultiMNISTDetection
        common = dict(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0, color=True,
        )
        a, _ = MultiMNISTDetection(**common)[0]
        b, _ = MultiMNISTDetection(
            **common, edge_thresholds=(), curv_thresholds=(), chroma_thresholds=(),
        )[0]
        assert torch.equal(a, b)

    def test_chroma_requires_color(self):
        """chroma_thresholds は color=True 専用（データ不要）。"""
        from data.detection import MultiMNISTDetection
        with pytest.raises(ValueError, match='chroma_thresholds'):
            MultiMNISTDetection(
                train=True, num_samples=1, color=False, chroma_thresholds=(0.1,),
            )

    def test_use_rgb_requires_color(self):
        """use_rgb=True かつ color=False は ValueError。"""
        from data.detection import MultiMNISTDetection
        with pytest.raises(ValueError):
            MultiMNISTDetection(
                train=True, num_samples=1, use_rgb=True, color=False,
            )

    def test_kd_with_use_rgb_raises(self):
        """kd_teacher_input と use_rgb の併用は ValueError（データ不要）。"""
        from data.detection import MultiMNISTDetection
        with pytest.raises(ValueError):
            MultiMNISTDetection(
                train=True, num_samples=1, color=True, use_rgb=True,
                kd_teacher_input=True,
            )

    def test_box_coords_within_canvas(self, patched_cifar):
        """color=True でも box座標は数字配置と同じロジック（境界内・妥当な形状）。

        color=True は背景/色選択で追加のrng消費があるため配置座標自体は
        グレースケール版とは一致しない（rng列がずれるため）。ここでは
        _make_targets/encode_centernet_targets が無改造であることを、
        座標が妥当な範囲・形状であることで確認する。
        """
        from data.detection import MultiMNISTDetection
        color_ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=7,
            color=True,
        )
        _, targets = color_ds[2]
        n_objects = int(targets['mask'].sum().item())
        assert n_objects >= 1
        boxes = targets['gt_boxes'][:n_objects]
        assert boxes.shape[1] == 4
        assert (boxes >= 0).all() and (boxes <= 96).all()
        # x2>x1, y2>y1（妥当な矩形）
        assert (boxes[:, 2] > boxes[:, 0]).all()
        assert (boxes[:, 3] > boxes[:, 1]).all()

    def test_composite_produces_color_pixels(self, patched_cifar):
        """数字が鮮やか色で合成され、R/G/Bが一様でないピクセルが存在するか。"""
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=2, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=3,
            color=True, use_rgb=True,
        )
        image, _ = ds[0]
        channel_spread = image.max(dim=0).values - image.min(dim=0).values
        assert (channel_spread > 0.1).any()

    def test_reproducibility(self, patched_cifar):
        """同じインデックスで同じ結果が得られるか（背景・色とも決定的）。"""
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=1,
            color=True, use_rgb=True,
        )
        img1, _ = ds[1]
        img2, _ = ds[1]
        assert torch.equal(img1, img2)

    def test_kd_teacher_input_is_raw_rgb(self, patched_cifar):
        """kd_teacher_input=True: student=二値化(9ch), teacher=生RGB(3ch)（VOCと同型）。"""
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            color=True, kd_teacher_input=True,
        )
        item = ds[0]
        assert len(item) == 3
        student_image, targets, teacher_image = item
        assert student_image.shape == (9, 96, 96)
        assert teacher_image.shape == (3, 96, 96)
        assert 'heatmap' in targets

    def test_kd_teacher_n_thresholds_decouples_mono_teacher_input(self):
        """mono(color=False)のKDで student 31閾値 × teacher 3閾値を両立する。

        旧実装は teacher にも student と同一テンソルを渡していたため、
        student だけ 5bit へ上げると teacher の conv1 が
        `expected input to have 3 channels, but got 31` で落ちた（実際に踏んだ）。
        """
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=31, output_stride=4, num_samples=4, seed=0,
            kd_teacher_input=True, kd_teacher_n_thresholds=3,
        )
        student_image, targets, teacher_image = ds[0]
        assert student_image.shape == (31, 96, 96)
        assert teacher_image.shape == (3, 96, 96)
        assert 'heatmap' in targets
        # teacher入力は student の該当閾値と一致する（同じ canvas から作るため）
        assert torch.equal(teacher_image[1], student_image[15])

    def test_kd_teacher_n_thresholds_none_shares_tensor(self):
        """閾値数が同じ/未指定なら従来どおり student と同一テンソルを共有する。"""
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            kd_teacher_input=True,
        )
        student_image, _targets, teacher_image = ds[0]
        assert teacher_image is student_image

    def test_collate_color(self, patched_cifar):
        """collate_detection がカラー版でも正しく動作するか。"""
        from data.detection import MultiMNISTDetection, collate_detection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            color=True, use_rgb=True,
        )
        batch = [ds[i] for i in range(4)]
        images, targets = collate_detection(batch)
        assert images.shape == (4, 3, 96, 96)
        assert targets['heatmap'].shape == (4, 10, 24, 24)


# ============================================================
# モデル
# ============================================================

class TestCenterNetLUT:
    @pytest.fixture
    def model(self):
        from detection_model import CenterNetLUT
        return CenterNetLUT(
            n_classes=10, k=8, in_channels=3,
            tree_depth=2, head_depth=2, n_inputs=2,
            size_group=4, offset_group=2,
            device='cpu',
        )

    def test_forward_shape(self, model):
        """forward passの出力形状"""
        x = torch.rand(2, 3, 96, 96)
        outputs = model(x)
        assert outputs['hm'].shape == (2, 10, 24, 24)
        assert outputs['size'].shape == (2, 2, 24, 24)
        assert outputs['off'].shape == (2, 2, 24, 24)

    def test_output_range(self, model):
        """出力値域の確認"""
        x = torch.rand(2, 3, 96, 96)
        model.train()
        outputs = model(x)
        # ヒートマップは[0,1]
        assert outputs['hm'].min() >= -1e-6
        assert outputs['hm'].max() <= 1 + 1e-6

    def test_gradient_flow(self, model):
        """勾配が全パラメータに流れるか"""
        x = torch.rand(1, 3, 96, 96)
        outputs = model(x)
        loss = outputs['hm'].sum() + outputs['size'].sum() + outputs['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_eval_mode(self, model):
        """evalモードでも動作するか"""
        model.eval()
        x = torch.rand(2, 3, 96, 96)
        with torch.no_grad():
            outputs = model(x)
        assert outputs['hm'].shape == (2, 10, 24, 24)

    def test_num_gates(self, model):
        """ゲート数が正の値か"""
        assert model.num_gates > 0


# ============================================================
# 損失関数
# ============================================================

class TestCenterNetLoss:
    def test_loss_computation(self):
        """損失関数の基本動作"""
        from detection_loss import CenterNetLoss
        loss_fn = CenterNetLoss()

        outputs = {
            'hm': torch.rand(2, 10, 24, 24, requires_grad=True),
            'size': torch.rand(2, 2, 24, 24, requires_grad=True),
            'off': torch.rand(2, 2, 24, 24, requires_grad=True),
        }
        targets = {
            'heatmap': torch.zeros(2, 10, 24, 24),
            'size': torch.rand(2, 3, 2),
            'offset': torch.rand(2, 3, 2),
            'indices': torch.randint(0, 24 * 24, (2, 3)),
            'mask': torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
        }
        # 正解ヒートマップにピークを設定
        targets['heatmap'][0, 3, 12, 12] = 1.0
        targets['heatmap'][1, 7, 5, 5] = 1.0

        loss, loss_dict = loss_fn(outputs, targets)
        assert loss.requires_grad
        assert loss.item() > 0
        assert 'hm_loss' in loss_dict
        assert 'size_loss' in loss_dict
        assert 'off_loss' in loss_dict


# ============================================================
# KD損失（W3-det）
# ============================================================

def _make_kd_fixtures(seed=0, B=2, C=10, H=8, W=8, n_obj=3):
    """CenterNetKDLoss 用の outputs/targets(teacher予測込み)を作る。"""
    g = torch.Generator().manual_seed(seed)
    outputs = {
        'hm': torch.rand(B, C, H, W, generator=g),
        'size': torch.rand(B, 2, H, W, generator=g),
        'off': torch.rand(B, 2, H, W, generator=g),
    }
    heatmap = torch.zeros(B, C, H, W)
    heatmap[0, 3, 4, 4] = 1.0
    heatmap[1, 7, 2, 5] = 1.0
    targets = {
        'heatmap': heatmap,
        'size': torch.rand(B, n_obj, 2, generator=g),
        'offset': torch.rand(B, n_obj, 2, generator=g),
        'indices': torch.randint(0, H * W, (B, n_obj), generator=g),
        'mask': torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
        'teacher_hm': torch.rand(B, C, H, W, generator=g),
        'teacher_size': torch.rand(B, 2, H, W, generator=g),
        'teacher_off': torch.rand(B, 2, H, W, generator=g),
    }
    return outputs, targets


class TestCenterNetHmKDLoss:
    """centernet_hm_kd_loss(二値Bernoulli KL)の数理性質。"""

    def test_self_kd_is_zero(self):
        """teacher==student のとき KD項は 0（各温度で）。"""
        from detection_loss import centernet_hm_kd_loss
        p = torch.rand(2, 10, 8, 8)
        for T in (1.0, 2.0, 4.0):
            loss = centernet_hm_kd_loss(p, p.clone(), temperature=T)
            assert abs(loss.item()) < 1e-9

    def test_nonnegative(self):
        """KL なので常に非負。"""
        from detection_loss import centernet_hm_kd_loss
        g = torch.Generator().manual_seed(1)
        s = torch.rand(2, 10, 8, 8, generator=g)
        t = torch.rand(2, 10, 8, 8, generator=g)
        assert centernet_hm_kd_loss(s, t, temperature=2.0).item() >= 0.0

    def test_known_value_T1(self):
        """T=1 の既知値: KL(Bern(0.8) || Bern(0.5)) を素朴式と照合。"""
        import math

        from detection_loss import centernet_hm_kd_loss
        s = torch.full((1, 1, 1, 1), 0.5)
        t = torch.full((1, 1, 1, 1), 0.8)
        expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
        got = centernet_hm_kd_loss(s, t, temperature=1.0).item()
        assert abs(got - expected) < 1e-5

    def test_t2_scale(self):
        """T²スケール: 素朴実装（logit→temper→二値KL×T²）と一致。"""
        import math

        from detection_loss import centernet_hm_kd_loss
        ps, pt, T = 0.3, 0.9, 2.0
        zs = math.log(ps / (1 - ps))
        zt = math.log(pt / (1 - pt))
        qs = 1 / (1 + math.exp(-zs / T))
        qt = 1 / (1 + math.exp(-zt / T))
        expected = (qt * math.log(qt / qs)
                    + (1 - qt) * math.log((1 - qt) / (1 - qs))) * T * T
        s = torch.full((1, 1, 1, 1), ps)
        t = torch.full((1, 1, 1, 1), pt)
        got = centernet_hm_kd_loss(s, t, temperature=T).item()
        assert abs(got - expected) < 1e-5

    def test_grad_flows(self):
        """student側に有限で非零の勾配が流れる。"""
        from detection_loss import centernet_hm_kd_loss
        g = torch.Generator().manual_seed(2)
        s = torch.rand(2, 10, 8, 8, generator=g).requires_grad_(True)
        t = torch.rand(2, 10, 8, 8, generator=g)
        loss = centernet_hm_kd_loss(s, t, temperature=2.0)
        (grad,) = torch.autograd.grad(loss, s)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0


class TestCenterNetKDLoss:
    def test_alpha_one_matches_plain_loss(self):
        """α=1（純hard）で通常 CenterNetLoss と厳密一致（teacher無視）。"""
        from detection_loss import CenterNetKDLoss, CenterNetLoss
        outputs, targets = _make_kd_fixtures()
        plain = CenterNetLoss(lambda_size=0.1, lambda_off=1.0)
        kd = CenterNetKDLoss(alpha=1.0, temperature=2.0,
                             lambda_size=0.1, lambda_off=1.0)
        loss_plain, _ = plain(outputs, targets)
        loss_kd, _ = kd(outputs, targets)
        assert abs(loss_plain.item() - loss_kd.item()) < 1e-6

    def test_teacher_equals_student_kd_term_zero(self):
        """α=0（純KD）で teacher予測==student出力 なら損失 0。"""
        from detection_loss import CenterNetKDLoss
        outputs, targets = _make_kd_fixtures()
        targets = {
            **targets,
            'teacher_hm': outputs['hm'].clone(),
            'teacher_size': outputs['size'].clone(),
            'teacher_off': outputs['off'].clone(),
        }
        kd = CenterNetKDLoss(alpha=0.0, temperature=2.0)
        loss, _ = kd(outputs, targets)
        assert abs(loss.item()) < 1e-6

    def test_kd_term_decreases_toward_teacher(self):
        """student が teacher に近いほど KD損失が小さい（α=0）。"""
        from detection_loss import CenterNetKDLoss
        outputs, targets = _make_kd_fixtures()
        kd = CenterNetKDLoss(alpha=0.0, temperature=2.0)
        far, _ = kd(outputs, targets)
        near_outputs = {
            'hm': 0.5 * outputs['hm'] + 0.5 * targets['teacher_hm'],
            'size': 0.5 * outputs['size'] + 0.5 * targets['teacher_size'],
            'off': 0.5 * outputs['off'] + 0.5 * targets['teacher_off'],
        }
        near, _ = kd(near_outputs, targets)
        assert near.item() < far.item()

    def test_loss_dict_keys_and_grad(self):
        """loss_dict のキー（hm_hard/hm_kd 追加）と勾配疎通。"""
        from detection_loss import CenterNetKDLoss
        outputs, targets = _make_kd_fixtures()
        outputs = {k: v.requires_grad_(True) for k, v in outputs.items()}
        kd = CenterNetKDLoss(alpha=0.25, temperature=2.0)
        loss, loss_dict = kd(outputs, targets)
        for key in ('hm_loss', 'size_loss', 'off_loss', 'total_loss',
                    'hm_hard', 'hm_kd'):
            assert key in loss_dict
        loss.backward()
        for k, v in outputs.items():
            assert v.grad is not None, f'{k} に勾配がありません'


class TestKDTeacherLoading:
    def test_load_kd_teacher_from_ckpt(self, tmp_path):
        """CNN teacher ckpt の glob 解決・config自動判別・in_ch推定。"""
        from detection_model_cnn import CenterNetCNN
        from train_detection import load_kd_teacher_detection
        model = CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8)
        ckpt = {
            'model_state_dict': model.state_dict(),
            'metadata': {'args': {'arch': 'cnn', 'cnn_width': 8,
                                  'cnn_head_ch': 8}},
            'best_metric': 0.5,
            'step': 100,
        }
        ckpt_dir = tmp_path / 'run_det_cnn_w8_obj3' / 'checkpoints'
        ckpt_dir.mkdir(parents=True)
        torch.save(ckpt, ckpt_dir / 'best.pt')

        teacher = load_kd_teacher_detection(
            str(tmp_path / '*_det_cnn_*' / 'checkpoints' / 'best.pt'), 'cpu')
        assert isinstance(teacher, CenterNetCNN)
        assert not teacher.training  # eval固定
        assert all(not p.requires_grad for p in teacher.parameters())
        out = teacher(torch.rand(1, 3, 32, 32))
        assert out['hm'].shape == (1, 10, 8, 8)

    def test_load_kd_teacher_missing_raises(self, tmp_path):
        """globが一致しなければ FileNotFoundError。"""
        from train_detection import load_kd_teacher_detection
        with pytest.raises(FileNotFoundError):
            load_kd_teacher_detection(str(tmp_path / 'nothing_*.pt'), 'cpu')

    def test_load_kd_teacher_non_cnn_raises(self, tmp_path):
        """teacher arch が cnn 以外なら ValueError。"""
        from detection_model_cnn import CenterNetCNN
        from train_detection import load_kd_teacher_detection
        model = CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8)
        ckpt = {
            'model_state_dict': model.state_dict(),
            'metadata': {'args': {'arch': 'lut'}},
            'best_metric': 0.0, 'step': 0,
        }
        torch.save(ckpt, tmp_path / 'best.pt')
        with pytest.raises(ValueError):
            load_kd_teacher_detection(str(tmp_path / 'best.pt'), 'cpu')


class TestKDDatasetPlumbing:
    def test_multimnist_kd_teacher_input(self):
        """kd_teacher_input=True で3つ組を返し、teacher入力はstudentと同一。"""
        from data.detection import MultiMNISTDetection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            kd_teacher_input=True,
        )
        item = ds[0]
        assert len(item) == 3
        image, targets, teacher_image = item
        assert torch.equal(image, teacher_image)
        assert 'heatmap' in targets

    def test_collate_kd_triples(self):
        """collate_detection が3つ組バッチを (images, targets, t_images) に。"""
        from data.detection import MultiMNISTDetection, collate_detection
        ds = MultiMNISTDetection(
            train=True, canvas_size=96, min_objects=1, max_objects=3,
            n_thresholds=3, output_stride=4, num_samples=4, seed=0,
            kd_teacher_input=True,
        )
        batch = [ds[i] for i in range(4)]
        images, targets, teacher_images = collate_detection(batch)
        assert images.shape == (4, 3, 96, 96)
        assert teacher_images.shape == (4, 3, 96, 96)
        assert targets['heatmap'].shape == (4, 10, 24, 24)

    def test_voc_kd_with_use_rgb_raises(self):
        """VOC: kd_teacher_input と use_rgb の併用は ValueError（データ不要）。"""
        from data.detection_voc import VOCDetectionCenterNet
        with pytest.raises(ValueError):
            VOCDetectionCenterNet(
                root='/nonexistent', train=True, use_rgb=True,
                kd_teacher_input=True, download=False,
            )


class TestBuildDatasetsColorDispatch:
    """train_detection.build_datasets の --dataset multimnist_color 分岐。"""

    @staticmethod
    def _args(arch, dataset='multimnist_color', distill_teacher_ckpt=None):
        import argparse
        return argparse.Namespace(
            dataset=dataset, arch=arch, canvas_size=32,
            min_objects=1, max_objects=2, n_thresholds=3,
            num_train=4, num_test=4, seed=0,
            distill_teacher_ckpt=distill_teacher_ckpt,
        )

    def test_lut_uses_binarized_input(self, monkeypatch):
        """arch=lut: use_rgb=False → 3*n_thresholds ch。"""
        import torchvision
        monkeypatch.setattr(torchvision.datasets, 'CIFAR10', _FakeCIFAR10)
        from train_detection import build_datasets
        args = self._args('lut')
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        assert n_classes == 10
        image, _ = train_ds[0]
        assert image.shape[0] == 9

    def test_cnn_uses_raw_rgb(self, monkeypatch):
        """arch=cnn: use_rgb=True → 生RGB 3ch。"""
        import torchvision
        monkeypatch.setattr(torchvision.datasets, 'CIFAR10', _FakeCIFAR10)
        from train_detection import build_datasets
        args = self._args('cnn')
        train_ds, test_ds, n_classes = build_datasets(
            args, output_stride=4, train=True, test=True,
        )
        image, _ = train_ds[0]
        assert image.shape[0] == 3


# ============================================================
# 検出ユーティリティ
# ============================================================

class TestDetectionUtils:
    def test_nms_by_maxpool(self):
        """maxpool NMSがピークのみを残すか"""
        from detection_utils import nms_by_maxpool
        hm = torch.zeros(1, 1, 8, 8)
        hm[0, 0, 3, 3] = 0.9  # ピーク
        hm[0, 0, 3, 4] = 0.5  # 隣接（抑制されるべき）
        result = nms_by_maxpool(hm)
        assert result[0, 0, 3, 3] == 0.9
        assert result[0, 0, 3, 4] == 0.0

    def test_nms_tiebreak(self):
        """同値ピクセルが複数ある場合、1つだけ残るか（softモード対策）"""
        from detection_utils import nms_by_maxpool
        hm = torch.full((1, 1, 8, 8), 0.5)  # 全ピクセル同値
        hm[0, 0, 4, 4] = 0.9  # 明確なピーク
        result = nms_by_maxpool(hm)
        # 0.5のピクセルは大半が抑制されるべき
        n_kept = (result[0, 0] > 0).sum().item()
        assert n_kept < 20  # 64ピクセル中20未満しか残らない
        assert result[0, 0, 4, 4] == 0.9  # ピークは残る

    def test_decode_detections(self):
        """bbox復号の基本動作"""
        from detection_utils import decode_detections
        hm = torch.zeros(1, 10, 24, 24)
        hm[0, 5, 12, 12] = 0.95  # クラス5のオブジェクト
        size = torch.full((1, 2, 24, 24), 0.3)  # 正規化サイズ
        off = torch.full((1, 2, 24, 24), 0.5)  # オフセット
        results = decode_detections(hm, size, off, output_stride=4, threshold=0.5)
        assert len(results) == 1
        assert results[0]['labels'][0].item() == 5
        assert results[0]['scores'][0].item() > 0.9
        assert results[0]['boxes'].shape[1] == 4

    def test_compute_iou(self):
        """IoU計算の正確性"""
        from detection_utils import compute_iou
        boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        iou = compute_iou(boxes1, boxes2)
        assert abs(iou[0, 0].item() - 1.0) < 1e-5

        # 重なりなし
        boxes3 = torch.tensor([[20, 20, 30, 30]], dtype=torch.float32)
        iou2 = compute_iou(boxes1, boxes3)
        assert abs(iou2[0, 0].item()) < 1e-5

    def test_gaussian_radius(self):
        """ガウシアン半径が正の値か"""
        from detection_utils import gaussian_radius
        r = gaussian_radius((7, 7), min_overlap=0.7)
        assert r >= 0

    def test_encode_decode_roundtrip(self):
        """GTエンコード→decode_detections往復で中心・サイズが復元されるか"""
        from detection_utils import decode_detections, draw_gaussian
        stride = 4
        hm_size = 24
        # GT: クラス2、中心(50, 30)px、サイズ28×28px
        cx, cy, w, h = 50.0, 30.0, 28.0, 28.0
        cx_hm, cy_hm = cx / stride, cy / stride
        cx_int, cy_int = int(cx_hm), int(cy_hm)

        hm = torch.zeros(1, 10, hm_size, hm_size)
        draw_gaussian(hm[0, 2], (cx_int, cy_int), radius=2)
        # サイズ/オフセットGT（データセットと同じ正規化）
        size = torch.zeros(1, 2, hm_size, hm_size)
        size[0, 0, cy_int, cx_int] = (w / stride) / hm_size
        size[0, 1, cy_int, cx_int] = (h / stride) / hm_size
        off = torch.zeros(1, 2, hm_size, hm_size)
        off[0, 0, cy_int, cx_int] = cx_hm - cx_int
        off[0, 1, cy_int, cx_int] = cy_hm - cy_int

        results = decode_detections(hm, size, off, output_stride=stride,
                                    threshold=0.5)
        assert results[0]['labels'][0].item() == 2
        x1, y1, x2, y2 = results[0]['boxes'][0].tolist()
        assert abs((x1 + x2) / 2 - cx) < 1.0
        assert abs((y1 + y2) / 2 - cy) < 1.0
        assert abs((x2 - x1) - w) < 1.0
        assert abs((y2 - y1) - h) < 1.0

    def test_compute_map_perfect(self):
        """完全一致予測で mAP=1.0"""
        from detection_utils import compute_map
        gt = [{'boxes': torch.tensor([[10., 10., 38., 38.]]),
               'labels': torch.tensor([3])}]
        pred = [{'boxes': torch.tensor([[10., 10., 38., 38.]]),
                 'scores': torch.tensor([0.9]),
                 'labels': torch.tensor([3])}]
        mAP, _ = compute_map(pred, gt, n_classes=10, iou_threshold=0.5)
        assert abs(mAP - 1.0) < 1e-6

    def test_compute_map_all_miss(self):
        """全外し予測で mAP=0"""
        from detection_utils import compute_map
        gt = [{'boxes': torch.tensor([[10., 10., 38., 38.]]),
               'labels': torch.tensor([3])}]
        pred = [{'boxes': torch.tensor([[60., 60., 90., 90.]]),
                 'scores': torch.tensor([0.9]),
                 'labels': torch.tensor([3])}]
        mAP, _ = compute_map(pred, gt, n_classes=10, iou_threshold=0.5)
        assert mAP < 1e-6

    def test_compute_map_does_not_match_across_images(self):
        """画像をまたいだ照合をしない（画像Aの予測が画像BのGTに当たらない）。

        回帰テスト: compute_map が全画像の予測・GTをプールして照合していた
        ため、「箱を多く出すだけ」のモデルほど mAP が高く出ていた。
        2枚とも同じ位置にGTがあり、予測は互いに入れ替わっている状況では
        画像内での正解は1つも無いので mAP=0 でなければならない。
        """
        from detection_utils import compute_map
        box_a = torch.tensor([[10., 10., 38., 38.]])
        box_b = torch.tensor([[60., 60., 88., 88.]])
        gt = [
            {'boxes': box_a, 'labels': torch.tensor([3])},
            {'boxes': box_b, 'labels': torch.tensor([3])},
        ]
        # 画像0にはBの箱、画像1にはAの箱を出す（=どちらも画像内では外れ）
        pred = [
            {'boxes': box_b, 'scores': torch.tensor([0.9]),
             'labels': torch.tensor([3])},
            {'boxes': box_a, 'scores': torch.tensor([0.9]),
             'labels': torch.tensor([3])},
        ]
        mAP, _ = compute_map(pred, gt, n_classes=10, iou_threshold=0.5)
        assert mAP < 1e-6

    def test_compute_map_scopes_each_image_independently(self):
        """画像ごとに正しく当たっていれば mAP=1.0（スコープ導入の副作用が無い）。"""
        from detection_utils import compute_map
        box_a = torch.tensor([[10., 10., 38., 38.]])
        box_b = torch.tensor([[60., 60., 88., 88.]])
        gt = [
            {'boxes': box_a, 'labels': torch.tensor([3])},
            {'boxes': box_b, 'labels': torch.tensor([3])},
        ]
        pred = [
            {'boxes': box_a, 'scores': torch.tensor([0.9]),
             'labels': torch.tensor([3])},
            {'boxes': box_b, 'scores': torch.tensor([0.8]),
             'labels': torch.tensor([3])},
        ]
        mAP, _ = compute_map(pred, gt, n_classes=10, iou_threshold=0.5)
        assert abs(mAP - 1.0) < 1e-6


# ============================================================
# CNNベースライン
# ============================================================

class TestCenterNetCNN:
    @pytest.fixture
    def model(self):
        from detection_model_cnn import CenterNetCNN
        return CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8)

    def test_forward_shape(self, model):
        """forward passの出力形状（LUT版と同一仕様）"""
        x = torch.rand(2, 3, 96, 96)
        outputs = model(x)
        assert outputs['hm'].shape == (2, 10, 24, 24)
        assert outputs['size'].shape == (2, 2, 24, 24)
        assert outputs['off'].shape == (2, 2, 24, 24)

    def test_output_range(self, model):
        """全ヘッドがsigmoidで[0,1]（focal loss/decode互換）"""
        x = torch.rand(2, 3, 96, 96)
        outputs = model(x)
        for key in ('hm', 'size', 'off'):
            assert outputs[key].min() >= 0
            assert outputs[key].max() <= 1

    def test_gradient_flow(self, model):
        """勾配が全パラメータに流れるか"""
        x = torch.rand(1, 3, 96, 96)
        outputs = model(x)
        loss = outputs['hm'].sum() + outputs['size'].sum() + outputs['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_num_gates_zero(self, model):
        """CNN版はゲートを持たない（API互換ダミー）"""
        assert model.num_gates == 0


# ============================================================
# W14ladder: CenterNetCNN downsample={stride,orpool} (L0-L2の土台)
# ============================================================

class TestCenterNetCNNDownsample:
    def test_default_is_stride(self):
        from detection_model_cnn import CenterNetCNN
        model = CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8)
        assert model.downsample == 'stride'
        assert not any(isinstance(m, OrPooling2d) for m in model.modules())

    def test_orpool_inserts_orpooling_and_keeps_conv_count(self):
        """'orpool'は6層のconv_bn_reluは維持したままOrPooling2d(2)を2つ挿入する
        （strideによる空間ダウンサンプルだけを差し替える、パラメータ数不変)"""
        from detection_model_cnn import CenterNetCNN
        m_stride = CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8,
                                downsample='stride')
        m_orpool = CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8,
                                downsample='orpool')
        n_conv_stride = sum(1 for m in m_stride.backbone if isinstance(m, nn.Sequential))
        n_conv_orpool = sum(1 for m in m_orpool.backbone if isinstance(m, nn.Sequential))
        assert n_conv_stride == n_conv_orpool == 6
        assert sum(1 for m in m_orpool.modules() if isinstance(m, OrPooling2d)) == 2
        n_params_stride = sum(p.numel() for p in m_stride.parameters())
        n_params_orpool = sum(p.numel() for p in m_orpool.parameters())
        assert n_params_stride == n_params_orpool

    def test_forward_shape_matches_stride(self):
        """'orpool'でも出力ストライド4は不変(96→24)"""
        from detection_model_cnn import CenterNetCNN
        model = CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8,
                             downsample='orpool')
        x = torch.rand(2, 3, 96, 96)
        out = model(x)
        assert out['hm'].shape == (2, 10, 24, 24)
        assert out['size'].shape == (2, 2, 24, 24)
        assert out['off'].shape == (2, 2, 24, 24)

    def test_gradient_flow(self):
        from detection_model_cnn import CenterNetCNN
        model = CenterNetCNN(n_classes=10, in_channels=3, width=8, head_ch=8,
                             downsample='orpool')
        x = torch.rand(1, 3, 32, 32)
        out = model(x)
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_invalid_downsample_raises(self):
        from detection_model_cnn import CenterNetCNN
        with pytest.raises(ValueError, match='downsample'):
            CenterNetCNN(n_classes=10, in_channels=3, downsample='bogus')


class TestResolveInputMode:
    """train_detection._resolve_input_mode (W14ladder --cnn-input)"""

    def _args(self, arch='cnn', cnn_input='default', n_thresholds=3):
        import argparse
        return argparse.Namespace(arch=arch, cnn_input=cnn_input,
                                  n_thresholds=n_thresholds)

    def test_default_preserves_existing_behavior(self):
        """'default'は既存挙動(use_rgb=(arch=='cnn'), n_thresholds=args値)を維持"""
        from train_detection import _resolve_input_mode
        assert _resolve_input_mode(self._args(arch='cnn')) == (True, 3)
        assert _resolve_input_mode(self._args(arch='lut')) == (False, 3)

    def test_raw_forces_use_rgb_and_zero_thresholds(self):
        from train_detection import _resolve_input_mode
        assert _resolve_input_mode(self._args(arch='cnn', cnn_input='raw')) == (True, 0)
        assert _resolve_input_mode(self._args(arch='lut', cnn_input='raw')) == (True, 0)

    def test_thermo_forces_binarized(self):
        from train_detection import _resolve_input_mode
        assert _resolve_input_mode(
            self._args(arch='cnn', cnn_input='thermo', n_thresholds=5)) == (False, 5)


class TestLadderPreset:
    """train_detection.apply_ladder_preset / LADDER_PRESETS (W14ladder)"""

    def _args(self, ladder):
        import argparse
        return argparse.Namespace(
            ladder=ladder, arch='lut', cnn_input='default',
            cnn_downsample='stride', downsample='orpool', neck='none',
            decoupled_head=False, reg_bins=0, head_type='lut',
        )

    def test_none_is_noop(self):
        from train_detection import apply_ladder_preset
        args = self._args('none')
        before = vars(args).copy()
        apply_ladder_preset(args, parser=None)
        assert vars(args) == before

    @pytest.mark.parametrize('level,expected', [
        ('L0', dict(arch='cnn', cnn_input='raw', cnn_downsample='stride')),
        ('L1', dict(arch='cnn', cnn_input='thermo', cnn_downsample='stride')),
        ('L2', dict(arch='cnn', cnn_input='thermo', cnn_downsample='orpool')),
        ('L3', dict(arch='lut', cnn_input='default', downsample='orpool',
                    head_type='float')),
        ('L4', dict(arch='lut', cnn_input='default', downsample='orpool',
                    head_type='lut')),
    ])
    def test_presets_set_expected_fields(self, level, expected):
        from train_detection import apply_ladder_preset
        args = self._args(level)
        apply_ladder_preset(args, parser=None)
        for k, v in expected.items():
            assert getattr(args, k) == v, k

    def test_l4_matches_plain_centernetlut_defaults(self):
        """L4はneck/decoupled_head/reg_binsを触らず既定のまま(=既存CenterNetLUT
        の既定head_type='lut'相当)であることを確認する(梯子の妥当性検証)"""
        from train_detection import apply_ladder_preset
        args = self._args('L4')
        apply_ladder_preset(args, parser=None)
        assert args.neck == 'none'
        assert args.decoupled_head is False
        assert args.reg_bins == 0


# ============================================================
# LUT版 hard eval（離散LUT）でのヒートマップ連続性
# ============================================================

class TestCenterNetLUTHardEval:
    def test_hard_heatmap_is_continuous(self):
        """hard eval(model.eval())でもhm_group平均により連続値[0,1]になるか。

        旧実装（単一LutConv2d直出力）はhard evalで0/1二値に崩壊し、
        NMS/topKが全ピーク同点になる欠陥があった（hm_group導入で修復）。
        """
        from detection_model import CenterNetLUT
        model = CenterNetLUT(
            n_classes=10, k=8, in_channels=3,
            tree_depth=2, head_depth=2, n_inputs=2,
            hm_group=8, size_group=4, offset_group=2,
            parameterization='softmax16g', device='cpu',
        )
        model.eval()
        x = (torch.rand(2, 3, 96, 96) > 0.5).float()
        with torch.no_grad():
            outputs = model(x)
        hm = outputs['hm']
        assert hm.min() >= -1e-6 and hm.max() <= 1 + 1e-6
        # hard evalの各LUT出力は0/1 → hm_group=8の平均は 1/8 刻みの離散値。
        # 二値(0/1のみ)より細かい値が存在することを確認
        unique_vals = torch.unique(hm)
        assert len(unique_vals) > 2, (
            f'hard evalヒートマップが二値化している: {unique_vals}'
        )

    def test_softmax16g_forward(self):
        """softmax16gパラメータ化でsoft/hard両方のforwardが通るか"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(
            n_classes=10, k=8, in_channels=3,
            tree_depth=2, head_depth=2,
            hm_group=4, size_group=4, offset_group=2,
            parameterization='softmax16g', device='cpu',
        )
        x = torch.rand(1, 3, 96, 96)
        out_soft = model(x)
        model.eval()
        with torch.no_grad():
            out_hard = model(x)
        for key in ('hm', 'size', 'off'):
            assert out_soft[key].shape == out_hard[key].shape


# ============================================================
# mAP改善アーキ改造4件(downsample/neck/decoupled_head+reg_bins/head_type)
# ============================================================

def _tiny_kw(**overrides):
    """テスト共通の小型CenterNetLUT設定(CPU高速)。"""
    kw = dict(
        n_classes=3, k=4, in_channels=3,
        tree_depth=2, head_depth=2, n_inputs=2,
        hm_group=2, size_group=2, offset_group=2,
        parameterization='softmax16g', n_layers=3, device='cpu',
    )
    kw.update(overrides)
    return kw


class TestCenterNetLUTDownsample:
    """--downsample orpool(既定)|strided"""

    def test_default_is_orpool_pooling(self):
        """既定'orpool'ではstage内のダウンサンプル層がOrPooling2dのまま"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw())
        assert isinstance(model.stage1[-1], OrPooling2d)
        assert isinstance(model.stage2[-1], OrPooling2d)

    def test_strided_uses_lutconv2d_downsample(self):
        """'strided'ではダウンサンプル層がstride=2のLutConv2dに置換される"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(downsample='strided'))
        assert isinstance(model.stage1[-1], LutConv2d)
        assert model.stage1[-1].stride == (2, 2)
        assert isinstance(model.stage2[-1], LutConv2d)
        assert model.stage2[-1].stride == (2, 2)
        # ゲート数がorpool版より増える(ダウンサンプルにもゲートが追加される)
        model_orpool = CenterNetLUT(**_tiny_kw())
        assert model.num_gates > model_orpool.num_gates

    def test_forward_shape_and_grad(self):
        """'strided'でforward形状・勾配疎通が保たれるか"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(downsample='strided'))
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        assert out['hm'].shape == (2, 3, 8, 8)
        assert out['size'].shape == (2, 2, 8, 8)
        assert out['off'].shape == (2, 2, 8, 8)
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_invalid_downsample_raises(self):
        from detection_model import CenterNetLUT
        with pytest.raises(ValueError, match='downsample'):
            CenterNetLUT(**_tiny_kw(downsample='bogus'))


class TestCenterNetLUTNeck:
    """--neck none(既定)|fpn|fpnpan"""

    def test_default_none_has_no_neck_modules(self):
        """既定'none'ではneck関連モジュールが一切構築されない(既存挙動と数値不変)"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw())
        for attr in ('neck_up', 'neck_td', 'neck_down', 'neck_fuse',
                    'neck_down2', 'neck_fuse2'):
            assert not hasattr(model, attr), attr

    @pytest.mark.parametrize('neck', ['fpn', 'fpnpan'])
    def test_neck_builds_expected_modules(self, neck):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(neck=neck))
        assert isinstance(model.neck_up, UpSample2d)
        assert isinstance(model.neck_td, LutConv2d)
        assert isinstance(model.neck_fuse, LutConv2d)
        has_pan = hasattr(model, 'neck_fuse2')
        assert has_pan == (neck == 'fpnpan')

    @pytest.mark.parametrize('neck', ['fpn', 'fpnpan'])
    def test_forward_shape_and_grad(self, neck):
        """neck有効時もforward形状(出力ストライド4は不変)・勾配疎通が保たれるか"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(neck=neck))
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        assert out['hm'].shape == (2, 3, 8, 8)
        assert out['size'].shape == (2, 2, 8, 8)
        assert out['off'].shape == (2, 2, 8, 8)
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_fpnpan_has_more_gates_than_fpn(self):
        from detection_model import CenterNetLUT
        m_fpn = CenterNetLUT(**_tiny_kw(neck='fpn'))
        m_fpnpan = CenterNetLUT(**_tiny_kw(neck='fpnpan'))
        assert m_fpnpan.num_gates > m_fpn.num_gates

    def test_invalid_neck_raises(self):
        from detection_model import CenterNetLUT
        with pytest.raises(ValueError, match='neck'):
            CenterNetLUT(**_tiny_kw(neck='bogus'))


class TestCenterNetLUTDecoupledHead:
    """--decoupled-head"""

    def test_default_false_no_branch_modules(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw())
        assert not hasattr(model, 'cls_branch')
        assert not hasattr(model, 'reg_branch')

    def test_decoupled_builds_branch_modules(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(decoupled_head=True))
        assert isinstance(model.cls_branch, LutConv2d)
        assert isinstance(model.reg_branch, LutConv2d)

    def test_forward_shape_and_grad(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(decoupled_head=True))
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        assert out['hm'].shape == (2, 3, 8, 8)
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()
        assert model.cls_branch.weights.grad is not None
        assert model.reg_branch.weights.grad is not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_composes_with_neck(self):
        """decoupled_head + neck='fpn' の併用が通るか"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(decoupled_head=True, neck='fpn'))
        x = torch.rand(1, 3, 32, 32)
        out = model(x)
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()


class TestCenterNetLUTRegBins:
    """--reg-bins N>0: DFL風ビン分布回帰"""

    def test_default_zero_matches_scalar_head(self):
        """reg_bins=0(既定)はビン機構を一切構築しない(既存挙動と数値不変)"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw())
        assert not hasattr(model, '_bin_centers')
        assert model.size_sum.n_outputs == 2

    def test_reg_bins_output_shape_and_range(self):
        """reg_bins>0でもsize/offの出力形状は不変、値域は[0,1)(ビン中心の凸結合)"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(reg_bins=5))
        assert model.size_sum.n_outputs == 10  # 2 * reg_bins
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        assert out['size'].shape == (2, 2, 8, 8)
        assert out['off'].shape == (2, 2, 8, 8)
        assert out['size'].min() >= 0.0 and out['size'].max() < 1.0
        assert out['off'].min() >= 0.0 and out['off'].max() < 1.0

    def test_expectation_matches_manual_softmax(self):
        """_regressの期待値計算を素朴実装と直接照合(1e-10級)"""
        from detection_model import CenterNetLUT
        torch.manual_seed(0)
        model = CenterNetLUT(**_tiny_kw(reg_bins=4))
        x = torch.rand(1, 3, 16, 16)
        # size_convはbackbone出力(feat)を要求するため直接featを取り出して照合する
        c1, c2, c3 = model._backbone_forward(x)
        feat = c3
        raw = model.size_sum(model.size_conv(feat))
        B, _, H, W = raw.shape
        probs_manual = torch.softmax(raw.view(B, 2, 4, H, W), dim=2)
        centers = ((torch.arange(4, dtype=torch.float64) + 0.5) / 4).to(torch.float32)
        expected = (probs_manual * centers.view(1, 1, 4, 1, 1)).sum(dim=2)
        got = model._regress(model.size_conv, model.size_sum, feat)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_forward_backward_smoke(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(reg_bins=3))
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_negative_reg_bins_raises(self):
        from detection_model import CenterNetLUT
        with pytest.raises(ValueError, match='reg_bins'):
            CenterNetLUT(**_tiny_kw(reg_bins=-1))

    def test_reg_bins_with_float_head_raises(self):
        """reg_bins>0 と head_type='float' の併用は禁止(の整合)"""
        from detection_model import CenterNetLUT
        with pytest.raises(ValueError, match='reg_bins'):
            CenterNetLUT(**_tiny_kw(reg_bins=3, head_type='float'))


class TestCenterNetLUTHeadType:
    """--head-type lut(既定)|float ハイブリッド"""

    def test_default_lut_uses_spatial_group_sum(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw())
        assert isinstance(model.hm_sum, SpatialGroupSum)
        assert isinstance(model.hm_conv, LutConv2d)

    def test_float_head_uses_plain_conv2d(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(head_type='float'))
        assert isinstance(model.hm_conv, nn.Conv2d)
        assert not isinstance(model.hm_conv, LutConv2d)
        assert isinstance(model.size_conv, nn.Conv2d)
        assert isinstance(model.off_conv, nn.Conv2d)
        assert model.hm_sum is None
        assert model.size_sum is None
        assert model.off_sum is None

    def test_float_head_backbone_still_lut(self):
        """バックボーンはfloatヘッド時もLutConv2dのまま(ハイブリッド)"""
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(head_type='float'))
        assert isinstance(model.stage1[0], LutConv2d)
        assert model.num_gates > 0  # backboneのゲートはカウントされる

    def test_forward_shape_range_and_grad(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(**_tiny_kw(head_type='float'))
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        assert out['hm'].shape == (2, 3, 8, 8)
        assert out['size'].shape == (2, 2, 8, 8)
        assert out['off'].shape == (2, 2, 8, 8)
        # hmはsigmoidで[0,1]、size/offはunbounded float回帰
        assert out['hm'].min() >= 0.0 and out['hm'].max() <= 1.0
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    def test_invalid_head_type_raises(self):
        from detection_model import CenterNetLUT
        with pytest.raises(ValueError, match='head_type'):
            CenterNetLUT(**_tiny_kw(head_type='bogus'))


class TestCenterNetLUTW14DefaultInvariance:
    """全4件の既定値組合せが従来挙動と厳密一致するか(回帰の要)。"""

    def test_default_kwargs_match_no_kwargs(self):
        """新kwargsを明示せず構築したモデルと、新kwargsを既定値で明示したモデルが
        同一seedで厳密一致(state_dict形状・出力とも)する。"""
        from detection_model import CenterNetLUT
        torch.manual_seed(42)
        m1 = CenterNetLUT(**_tiny_kw())
        torch.manual_seed(42)
        m2 = CenterNetLUT(**_tiny_kw(
            downsample='orpool', neck='none', decoupled_head=False,
            reg_bins=0, head_type='lut',
        ))
        assert list(m1.state_dict().keys()) == list(m2.state_dict().keys())
        x = torch.rand(2, 3, 32, 32)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            o1, o2 = m1(x), m2(x)
        for key in ('hm', 'size', 'off'):
            assert torch.equal(o1[key], o2[key])
        assert m1.num_gates == m2.num_gates


class TestCenterNetLUTSelection:
    """selection={soft,hard_st,gumbel_st}本線統合(W13a-1)"""

    def _build(self, selection, **kw):
        from detection_model import CenterNetLUT
        return CenterNetLUT(
            n_classes=3, k=4, in_channels=3, tree_depth=2, head_depth=2,
            hm_group=2, size_group=2, offset_group=2,
            selection=selection, device='cpu', **kw,
        )

    def test_soft_default_matches_plain_lutconv2d(self):
        """既定selection='soft'ではLutConv2d(HardSTでない)が使われる"""
        from griffith_logic.conv_hard_st import LutConv2dHardST
        model = self._build('soft')
        for m in model.modules():
            assert not isinstance(m, LutConv2dHardST)

    @pytest.mark.parametrize('selection', ['hard_st', 'gumbel_st'])
    def test_hard_st_uses_hardst_layers(self, selection):
        from griffith_logic.conv_hard_st import LutConv2dHardST
        model = self._build(selection)
        hardst_layers = [m for m in model.modules() if isinstance(m, LutConv2dHardST)]
        assert len(hardst_layers) > 0
        for layer in hardst_layers:
            assert layer.selection == selection
        # LutConv2dHardSTはLutConv2dのサブクラスなのでnum_gatesは変わらず動く
        assert model.num_gates > 0

    @pytest.mark.parametrize('selection', ['soft', 'hard_st', 'gumbel_st'])
    def test_forward_backward_smoke(self, selection):
        model = self._build(selection)
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        loss = out['hm'].sum() + out['size'].sum() + out['off'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'

    @pytest.mark.parametrize('selection', ['hard_st', 'gumbel_st'])
    def test_train_eval_gap_zero_on_binary_input(self, selection):
        """hard_st/gumbel_stは訓練forwardと推論forwardが一致(selection gap 0)"""
        model = self._build(selection)
        for m in model.modules():
            if hasattr(m, 'weights'):
                m.weights.data.normal_()
        x = (torch.rand(2, 3, 32, 32) > 0.5).float()
        model.train()
        model.hm_conv.gumbel_scale = 0.0
        model.size_conv.gumbel_scale = 0.0
        model.off_conv.gumbel_scale = 0.0
        for m in model.modules():
            if hasattr(m, 'gumbel_scale'):
                m.gumbel_scale = 0.0
        with torch.no_grad():
            y_train = model(x)
        model.eval()
        with torch.no_grad():
            y_eval = model(x)
        for key in ('hm', 'size', 'off'):
            assert torch.equal(y_train[key], y_eval[key]), key

    def test_hard_st_wiring_learnable_coexist(self):
        """selection=hard_st × wiring='learnable' の共存(W13a-1注意事項)"""
        model = self._build('hard_st', wiring='learnable')
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        loss = out['hm'].sum()
        loss.backward()
        assert model.stage1[0].mapping_weights.grad is not None

    def test_invalid_selection_raises(self):
        with pytest.raises(ValueError, match='selection'):
            self._build('bogus')

    def test_cage_scheduler_updates_tau_b(self):
        from griffith_logic.conv_hard_st import CAGEConvScheduler
        model = self._build('hard_st')
        sched = CAGEConvScheduler(model, tau_max=3.0, tau_min=0.5, beta=0.9)
        tau0 = sched.tau_b
        for _ in range(5):
            sched.step()
        assert sched.tau_b != tau0 or sched.tau_b == 3.0  # 一様重み初期はτ_max据え置きもあり得る
        assert model.hm_conv.tau_b == sched.tau_b

    def test_tau_anneal_conv_scheduler_decays_and_anneals_noise(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
        from cage_scheduler import TauAnnealConvScheduler
        model = self._build('gumbel_st')
        sched = TauAnnealConvScheduler(
            model, tau_0=1.0, tau_min=0.1, total_steps=10, anneal_noise=True,
        )
        for _ in range(10):
            sched.step()
        assert abs(sched.tau_b - 0.1) < 1e-9
        assert model.hm_conv.gumbel_scale == 0.0


class TestTrainStepGradClip:
    """train_step の勾配クリップ"""

    def _setup(self):
        from detection_model import CenterNetLUT
        model = CenterNetLUT(
            n_classes=2, k=4, in_channels=3, tree_depth=2, head_depth=2,
            hm_group=2, size_group=2, offset_group=2,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        images = torch.rand(2, 3, 16, 16)
        targets = {
            'heatmap': torch.zeros(2, 2, 4, 4),
            'size': torch.zeros(2, 2, 4, 4),
            'offset': torch.zeros(2, 2, 4, 4),
            'mask': torch.zeros(2, 1, 4, 4),
        }
        return model, opt, images, targets

    def _loss_fn(self, big: bool):
        scale = 1e6 if big else 1.0

        def fn(outputs, targets):
            loss = scale * sum(v.sum() for v in outputs.values())
            return loss, {'total_loss': float(loss)}
        return fn

    def test_grad_clip_bounds_the_norm(self):
        """巨大な損失でも clip 後の grad_norm 報告値どおりに実勾配が抑えられる。"""
        from train_detection import train_step
        model, opt, images, targets = self._setup()
        out = train_step(model, images, targets, self._loss_fn(big=True), opt,
                         grad_clip=1.0)
        assert 'grad_norm' in out
        # clip 後のパラメータ勾配ノルムは閾値以下
        after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                               if p.grad is not None))
        assert after <= 1.0 + 1e-4, after

    def test_grad_clip_off_by_default(self):
        from train_detection import train_step
        model, opt, images, targets = self._setup()
        out = train_step(model, images, targets, self._loss_fn(big=False), opt)
        assert 'grad_norm' not in out


class TestBuildLrScheduler:
    """build_lr_scheduler（-ni モードで cosine が無効だったバグの回帰テスト）"""

    def _args(self, **overrides):
        import argparse
        ns = argparse.Namespace(
            scheduler='cosine', epochs=None, num_iterations=None, eta_min=1e-4, lr=0.01,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def _opt(self, lr=0.01):
        return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)

    def test_none_when_scheduler_disabled(self):
        from train_detection import build_lr_scheduler
        args = self._args(scheduler='none', epochs=10)
        assert build_lr_scheduler(args, self._opt(), use_epochs=True) is None

    def test_tmax_follows_epochs_in_epoch_mode(self):
        from train_detection import build_lr_scheduler
        args = self._args(epochs=60)
        sched = build_lr_scheduler(args, self._opt(), use_epochs=True)
        assert sched.T_max == 60

    def test_tmax_follows_iterations_in_iteration_mode(self):
        """★旧実装はここが 100 固定だった（-ni 40000 でも T_max=100）。"""
        from train_detection import build_lr_scheduler
        args = self._args(num_iterations=40000)
        sched = build_lr_scheduler(args, self._opt(), use_epochs=False)
        assert sched.T_max == 40000

    def test_lr_actually_anneals_over_the_iteration_budget(self):
        """-ni 相当の回数 step すると lr が eta_min まで下がる。"""
        from train_detection import build_lr_scheduler
        opt = self._opt(lr=0.01)
        args = self._args(num_iterations=200, eta_min=1e-4)
        sched = build_lr_scheduler(args, opt, use_epochs=False)
        assert opt.param_groups[0]['lr'] == pytest.approx(0.01)
        for _ in range(100):
            sched.step()
        mid = opt.param_groups[0]['lr']
        assert 1e-4 < mid < 0.01, f'中間で減衰していない: {mid}'
        for _ in range(100):
            sched.step()
        assert opt.param_groups[0]['lr'] == pytest.approx(1e-4, abs=1e-6)

    def test_invalid_tmax_raises(self):
        from train_detection import build_lr_scheduler
        args = self._args(num_iterations=None)
        with pytest.raises(ValueError, match='T_max'):
            build_lr_scheduler(args, self._opt(), use_epochs=False)


class TestBuildSelectionSchedulerDetection:
    """build_selection_scheduler(W13a-1、train_detection.py)"""

    def _args(self, **overrides):
        import argparse
        ns = argparse.Namespace(
            arch='lut', selection='soft', tau_schedule=None,
            tau_b_init=1.0, tau_b_min=0.1, gumbel_scale_init=1.0,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def _model(self, selection, **kw):
        from detection_model import CenterNetLUT
        return CenterNetLUT(
            n_classes=3, k=4, in_channels=3, tree_depth=2, head_depth=2,
            hm_group=2, size_group=2, offset_group=2,
            selection=selection, **kw,
        )

    def test_soft_returns_none(self):
        from train_detection import build_selection_scheduler
        model = self._model('soft')
        args = self._args(selection='soft')
        assert build_selection_scheduler(args, model, 100) is None

    def test_cnn_arch_returns_none(self):
        from train_detection import build_selection_scheduler
        model = self._model('hard_st')
        args = self._args(selection='hard_st', arch='cnn')
        assert build_selection_scheduler(args, model, 100) is None

    def test_hard_st_default_is_none_schedule(self):
        from train_detection import build_selection_scheduler
        model = self._model('hard_st')
        args = self._args(selection='hard_st')
        assert build_selection_scheduler(args, model, 100) is None

    def test_gumbel_st_default_is_anneal(self):
        from cage_scheduler import TauAnnealConvScheduler
        from train_detection import build_selection_scheduler
        model = self._model('gumbel_st')
        args = self._args(selection='gumbel_st')
        sched = build_selection_scheduler(args, model, 50)
        assert isinstance(sched, TauAnnealConvScheduler)
        assert sched.total_steps == 50

    def test_explicit_cage_schedule(self):
        from train_detection import build_selection_scheduler

        from griffith_logic.conv_hard_st import CAGEConvScheduler
        model = self._model('hard_st')
        args = self._args(selection='hard_st', tau_schedule='cage')
        sched = build_selection_scheduler(args, model, 50)
        assert isinstance(sched, CAGEConvScheduler)


class TestLowLevelFeatureFactory:
    """低レベル特徴の二値化器ファクトリと fewclass/face への配管。

    経緯: 低レベル特徴が multimnist_color の分岐にしか配管されておらず、
    detreal(fewclass)へ当てるアームが起動即エラー(EXITCODE=2)で失われた。
    3データセットが同一規則で二値化器を作ることをテストで固定する。
    """

    def test_make_binarize_falls_back_to_threshold_binarize(self):
        from data.detection import LowLevelFeatureBinarize, ThresholdBinarize, make_binarize
        assert isinstance(make_binarize(3), ThresholdBinarize)
        assert make_binarize(0) is None
        assert isinstance(make_binarize(3, edge_thresholds=(0.1,)),
                          LowLevelFeatureBinarize)

    def test_make_binarize_empty_features_matches_threshold_binarize(self):
        """全て空なら従来の ThresholdBinarize と出力が厳密一致する。"""
        from data.detection import make_binarize
        x = torch.rand(3, 8, 8)
        assert torch.equal(make_binarize(5)(x), make_binarize(5, (), (), ())(x))

    def test_fewclass_and_face_accept_lowlevel_thresholds(self):
        """データセット側が引数を受け取り binarize に反映すること（配管漏れ防止）。"""
        import inspect

        from data.detection_face import WiderFaceDetection
        from data.detection_fewclass import FewClassVOCDetection
        for cls in (FewClassVOCDetection, WiderFaceDetection):
            params = inspect.signature(cls.__init__).parameters
            for name in ('edge_thresholds', 'curv_thresholds', 'chroma_thresholds'):
                assert name in params, f'{cls.__name__} に {name} が無い'


class TestFewClassKdTeacherInput:
    """fewclass の online KD 配管(2段階蒸留の前提)。

    fewclass は長らく online KD 禁止(dataset が (image, targets) の2つ組しか
    返さないため)だった。2段階蒸留(COCO事前学習検出器→中間CNN→DLGN)を回すため
    multimnist_color と同一契約の3つ組を返せるようにした。
    """

    def test_signature_accepts_kd_teacher_input(self):
        import inspect

        from data.detection_fewclass import FewClassVOCDetection
        params = inspect.signature(FewClassVOCDetection.__init__).parameters
        assert 'kd_teacher_input' in params

    def test_kd_teacher_input_rejects_use_rgb(self):
        """teacher入力を返すのは student が二値化入力のときだけ(色版と同契約)。"""
        from data.detection_fewclass import FewClassVOCDetection
        with pytest.raises(ValueError, match='use_rgb=False'):
            FewClassVOCDetection(
                root='/nonexistent', kd_teacher_input=True, use_rgb=True,
                download=False,
            )


class TestHardEvalInputRounding:
    """hard評価の入力 round を「入力が本来0/1」のときだけ掛ける。

    背景: evaluate() は hardモードで無条件に images.round() していた。
    LUT student は thermometer 二値化済みなので no-op だが、**生RGBを消費する
    CNN teacher では画素が0/1に潰れ、精度を破壊していた**。実画像タスク
    (fewclass/face)の CNN teacher の値がすべて過小評価されていた原因。
    """

    def _probe(self, binary_input):
        """evaluate に渡された画像を記録するだけのダミーモデル。"""
        seen = []

        class Probe(torch.nn.Module):
            def forward(self, x):
                seen.append(x.clone())
                B = x.shape[0]
                return {'hm': torch.zeros(B, 1, 4, 4),
                        'size': torch.zeros(B, 2, 4, 4),
                        'off': torch.zeros(B, 2, 4, 4)}

        images = torch.rand(2, 3, 16, 16)  # 生RGB相当(0/1でない中間値)
        targets = {'heatmap': torch.zeros(2, 1, 4, 4),
                   'size': torch.zeros(2, 8, 2),
                   'offset': torch.zeros(2, 8, 2),
                   'mask': torch.zeros(2, 8),
                   'indices': torch.zeros(2, 8, dtype=torch.long),
                   'gt_boxes': torch.zeros(2, 8, 4),
                   'gt_labels': torch.zeros(2, 8, dtype=torch.long)}
        loader = [(images, targets)]

        from detection_loss import CenterNetLoss
        from train_detection import evaluate
        evaluate(Probe(), loader, 'cpu', CenterNetLoss(), mode=False,
                 desc=None, n_vis=0, n_classes=1, binary_input=binary_input)
        return seen[0]

    def test_binary_input_true_rounds(self):
        """既定(LUT想定)では従来どおり round される。"""
        x = self._probe(binary_input=True)
        assert torch.equal(x, x.round()), '0/1 に丸められているはず'

    def test_binary_input_false_preserves_pixels(self):
        """生RGB を食う CNN では画素をそのまま渡す。"""
        x = self._probe(binary_input=False)
        assert not torch.equal(x, x.round()), '中間値が保たれているはず'
