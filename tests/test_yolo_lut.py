"""
LogicYOLO(anchor-free YOLO型物体検出の論理ゲート化)のテスト。

1. 出力shape契約(cls/obj/reg 3分岐、グリッドサイズ)
2. 勾配疎通(3分岐すべて)
3. hard離散性(バックボーン中間 {0,1})
4. デコード往復健全性(GT→ターゲット→デコードで IoU>0.8 復元)
5. 極小収束(固定位置の単一物体を数百stepで検出、実測較正)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from detection_utils import compute_iou
from yolo_lut_model import (
    YoloLutLoss,
    YoloLutModel,
    build_targets,
    decode_reg,
    decode_yolo,
)


def _make_model(n_classes=10, k=8, n_bins=8):
    return YoloLutModel(
        n_classes=n_classes, k=k, in_channels=3,
        tree_depth=2, head_depth=2, n_inputs=2,
        n_bins=n_bins, cls_group=4, obj_group=4, device='cpu',
    )


# ============================================================
# 1. 出力shape契約
# ============================================================

class TestOutputShape:
    def test_head_shapes(self):
        model = _make_model()
        x = torch.rand(2, 3, 96, 96)
        out = model(x)
        # 出力ストライド4 → グリッド 24×24
        assert out['cls'].shape == (2, 10, 24, 24)
        assert out['obj'].shape == (2, 1, 24, 24)
        assert out['reg'].shape == (2, 4, 8, 24, 24)

    def test_prob_range(self):
        """cls/obj は [0,1] の確率、reg ビン重みも [0,1]。"""
        model = _make_model()
        x = torch.rand(2, 3, 96, 96)
        out = model(x)
        for key in ('cls', 'obj', 'reg'):
            assert out[key].min() >= -1e-6
            assert out[key].max() <= 1 + 1e-6

    def test_num_gates_positive(self):
        assert _make_model().num_gates > 0


# ============================================================
# 2. 勾配疎通(3分岐すべて)
# ============================================================

class TestGradientFlow:
    def test_all_branches_grad(self):
        model = _make_model()
        x = torch.rand(1, 3, 96, 96)
        out = model(x)
        # 3分岐すべてを含む損失
        loss = out['cls'].sum() + out['obj'].sum() + out['reg'].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f'{name} に勾配がありません'
            assert torch.isfinite(param.grad).all(), f'{name} の勾配に非有限値'

    def test_head_params_receive_grad(self):
        """各ヘッド(cls_conv/obj_conv/reg_conv)に勾配が届く。"""
        model = _make_model()
        x = torch.rand(1, 3, 96, 96)
        out = model(x)
        (out['cls'].sum() + out['obj'].sum() + out['reg'].sum()).backward()
        for head in ('cls_conv', 'obj_conv', 'reg_conv'):
            grad = getattr(model, head).weights.grad
            assert grad is not None and grad.abs().sum() > 0, f'{head} に勾配なし'

    def test_loss_backward(self):
        model = _make_model()
        x = torch.rand(2, 3, 96, 96)
        out = model(x)
        targets = {
            'gt_boxes': torch.tensor([
                [[10., 10., 30., 30.]], [[40., 40., 60., 60.]],
            ]),
            'gt_labels': torch.tensor([[3], [7]]),
            'mask': torch.tensor([[1.], [1.]]),
        }
        loss_fn = YoloLutLoss(n_classes=10, n_bins=8)
        loss, loss_dict = loss_fn(out, targets)
        assert loss.requires_grad and loss.item() > 0
        loss.backward()
        for key in ('obj_loss', 'cls_loss', 'reg_loss', 'total_loss'):
            assert key in loss_dict


# ============================================================
# 3. hard離散性(バックボーン中間 {0,1})
# ============================================================

class TestHardDiscreteness:
    def test_backbone_binary_in_eval(self):
        model = _make_model()
        model.eval()
        # 二値化済み入力(thermometer相当)
        x = (torch.rand(2, 3, 96, 96) > 0.5).float()
        with torch.no_grad():
            feat = model.backbone(x)
        uniq = torch.unique(feat)
        assert torch.all((uniq == 0) | (uniq == 1)), (
            f'バックボーン中間が {{0,1}} でない: {uniq.tolist()}'
        )


# ============================================================
# 4. デコード往復健全性
# ============================================================

class TestDecodeRoundtrip:
    def test_roundtrip_single_box(self):
        """単一GTをターゲット化 → そのままデコードで元boxを IoU>0.8 復元。"""
        n_bins = 32
        stride = 4
        grid = 24
        box = torch.tensor([[[18.0, 22.0, 46.0, 50.0]]])  # (1,1,4)
        labels = torch.tensor([[5]])
        mask = torch.tensor([[1.0]])

        tgt = build_targets(
            box, labels, mask, grid, grid, stride, n_classes=10, n_bins=n_bins,
        )
        # ターゲット分布をそのままモデル出力に見立ててデコード
        outputs = {
            'obj': tgt['obj'],
            'cls': tgt['cls'],
            'reg': tgt['reg'],
        }
        dets = decode_yolo(outputs, output_stride=stride, threshold=0.5)
        assert len(dets) == 1
        assert dets[0]['boxes'].shape[0] >= 1
        # 最高スコアの検出とGTのIoU
        iou = compute_iou(dets[0]['boxes'], box[0])
        best = iou.max().item()
        assert best > 0.8, f'往復IoUが低い: {best:.3f}'
        # クラスも復元
        assert (dets[0]['labels'] == 5).any()

    def test_reg_value_roundtrip(self):
        """decode_reg の期待値がエンコード元の連続値に一致(端数含む)。"""
        n_bins = 32
        stride = 4
        grid = 24
        # 中心 (dx=0.5, dy=0.25), w_norm≈?, h_norm≈?
        box = torch.tensor([[[20.0, 20.0, 40.0, 48.0]]])
        labels = torch.tensor([[1]])
        mask = torch.tensor([[1.0]])
        tgt = build_targets(box, labels, mask, grid, grid, stride, 10, n_bins)
        reg_val = decode_reg(tgt['reg'])  # (1,4,H,W)
        pos = tgt['pos'][0, 0] > 0
        ys, xs = torch.where(pos)
        gy, gx = ys[0].item(), xs[0].item()
        cx = (gx + reg_val[0, 0, gy, gx].item()) * stride
        cy = (gy + reg_val[0, 1, gy, gx].item()) * stride
        w = reg_val[0, 2, gy, gx].item() * stride * grid
        h = reg_val[0, 3, gy, gx].item() * stride * grid
        # 元box中心・サイズと一致
        assert abs(cx - 30.0) < 0.5
        assert abs(cy - 34.0) < 0.5
        assert abs(w - 20.0) < 1.0
        assert abs(h - 28.0) < 1.0


# ============================================================
# 5. 極小収束(実測較正、緩和禁止)
# ============================================================

@pytest.mark.learning
class TestTinyConvergence:
    def test_single_fixed_object_converges(self):
        """固定位置の単一物体を数百stepで検出できる。"""
        torch.manual_seed(0)
        device = 'cpu'
        stride = 4
        canvas = 32
        grid = canvas // stride  # 8
        n_classes = 3
        n_bins = 8

        model = YoloLutModel(
            n_classes=n_classes, k=16, in_channels=1,
            tree_depth=2, head_depth=2, n_inputs=2,
            n_bins=n_bins, cls_group=4, obj_group=4, device=device,
        ).to(device)
        loss_fn = YoloLutLoss(n_classes=n_classes, n_bins=n_bins,
                              output_stride=stride)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

        # 固定入力: 中央に矩形パターン、単一物体
        img = torch.zeros(1, 1, canvas, canvas)
        img[0, 0, 12:20, 12:20] = 1.0
        gt_box = torch.tensor([[[12.0, 12.0, 20.0, 20.0]]])
        gt_label = torch.tensor([[1]])
        mask = torch.tensor([[1.0]])
        targets = {'gt_boxes': gt_box, 'gt_labels': gt_label, 'mask': mask}

        model.train()
        for _ in range(400):
            out = model(img)
            loss, _ = loss_fn(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # soft評価で検出できるか
        model.eval()
        with torch.no_grad():
            out = model(img)
        dets = decode_yolo(out, output_stride=stride, threshold=0.2)[0]
        assert dets['boxes'].shape[0] >= 1, '物体を検出できていない'
        iou = compute_iou(dets['boxes'], gt_box[0])
        best_iou, best_j = iou.max(0)
        best_pred = iou.max().item()
        # クラス一致かつ IoU≥0.5 の検出が存在
        matched = False
        for p in range(dets['boxes'].shape[0]):
            biou, _ = iou[p].max(0)
            if biou.item() >= 0.5 and dets['labels'][p].item() == 1:
                matched = True
                break
        assert matched, f'クラス一致IoU≥0.5の検出なし(bestIoU={best_pred:.3f})'
