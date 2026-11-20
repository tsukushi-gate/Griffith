"""experiments/seg_losses.py の不均衡対策損失テスト。

VOC 21クラス背景collapse対策として実装した focal/dice/weighted_ce/focal_dice
損失の数理性質(既知値・IGNORE_INDEXマスク・勾配疎通)と、median-frequency
balancingによるクラス重み計算を検証する。
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from seg_losses import (  # noqa: E402
    SEG_LOSS_NAMES,
    build_seg_loss,
    compute_class_pixel_counts,
    compute_class_weights_from_loader,
    dice_loss,
    focal_dice_loss,
    focal_loss,
    median_frequency_weights,
)

IGNORE = 255
_GEN = torch.Generator().manual_seed(20260714)


def _rand_logits(*shape):
    return torch.randn(*shape, generator=_GEN, dtype=torch.float64)


# ============================================================
# median_frequency_weights
# ============================================================

class TestMedianFrequencyWeights:
    def test_known_value_two_class_imbalance(self):
        """counts=[900,100] → freq=[0.9,0.1], median=0.5
        → weight=[0.5556, 5.0](多数派<1, 少数派>1)。"""
        counts = torch.tensor([900, 100])
        w = median_frequency_weights(counts)
        assert abs(w[0].item() - 0.5 / 0.9) < 1e-4
        assert abs(w[1].item() - 0.5 / 0.1) < 1e-4

    def test_majority_downweighted_minority_upweighted(self):
        counts = torch.tensor([7300, 100, 100, 2500])  # bg優勢を模した例
        w = median_frequency_weights(counts)
        assert w[0] < 1.0  # 最頻(背景)は抑制
        assert w[1] > 1.0 and w[2] > 1.0  # 少数派は拡大

    def test_zero_count_class_gets_weight_one(self):
        """出現数0のクラスは重み1.0(欠測、中立値)。"""
        counts = torch.tensor([500, 500, 0])
        w = median_frequency_weights(counts)
        assert abs(w[2].item() - 1.0) < 1e-9

    def test_uniform_counts_all_weight_one(self):
        counts = torch.tensor([100, 100, 100, 100])
        w = median_frequency_weights(counts)
        assert torch.allclose(w, torch.ones(4), atol=1e-5)

    def test_all_zero_counts_no_crash(self):
        counts = torch.zeros(3, dtype=torch.long)
        w = median_frequency_weights(counts)
        assert torch.allclose(w, torch.ones(3))


class TestComputeClassWeightsFromLoader:
    def test_matches_manual_counts(self):
        """(x,y)バッチのDataLoader風イテラブルから正しく集計されるか。"""
        y1 = torch.tensor([[0, 0, 1], [0, 255, 1]])
        y2 = torch.tensor([[0, 0, 0], [1, 1, 255]])
        loader = [(None, y1), (None, y2)]
        counts = compute_class_pixel_counts(loader, n_classes=2, ignore_index=255)
        # class0: y1に3個 + y2に3個 = 6、class1: y1に2個+y2に2個=4、ignore2個は除外
        assert counts.tolist() == [6, 4]
        w = compute_class_weights_from_loader(loader, n_classes=2, ignore_index=255)
        expected = median_frequency_weights(counts)
        assert torch.allclose(w, expected)

    def test_kd_style_three_tuple_batch(self):
        """(x, y, x_raw)の3要素バッチでも y=batch[1] のみ使われる。"""
        y = torch.tensor([[0, 1, 1]])
        loader = [(None, y, None)]
        counts = compute_class_pixel_counts(loader, n_classes=2, ignore_index=255)
        assert counts.tolist() == [1, 2]

    def test_max_batches_limits_scan(self):
        y1 = torch.tensor([[0, 0]])
        y2 = torch.tensor([[1, 1]])
        loader = [(None, y1), (None, y2)]
        counts = compute_class_pixel_counts(
            loader, n_classes=2, ignore_index=255, max_batches=1)
        assert counts.tolist() == [2, 0]


# ============================================================
# focal_loss
# ============================================================

class TestFocalLoss:
    def test_gamma_zero_matches_plain_ce(self):
        """gamma=0(alpha無し)は通常CEと一致((1-pt)^0=1)。"""
        s = _rand_logits(2, 3, 4, 4)
        y = torch.randint(0, 3, (2, 4, 4))
        got = focal_loss(s, y, gamma=0.0, alpha=None, ignore_index=IGNORE)
        expected = F.cross_entropy(s, y, ignore_index=IGNORE)
        assert abs(got.item() - expected.item()) < 1e-10

    def test_known_value_single_pixel(self):
        """1画素の既知値: p=softmax(logits)[label]、focal=(1-p)^gamma * -log(p)。"""
        logits = torch.tensor([[[[2.0]], [[0.0]], [[-1.0]]]], dtype=torch.float64)
        label = torch.tensor([[[0]]])
        p = F.softmax(logits[0, :, 0, 0], dim=0)[0].item()
        gamma = 2.0
        expected = (1 - p) ** gamma * (-math.log(p))
        got = focal_loss(logits, label, gamma=gamma, ignore_index=IGNORE)
        assert abs(got.item() - expected) < 1e-8

    def test_ignore_index_excluded(self):
        s = _rand_logits(1, 2, 1, 3)
        y_all_valid = torch.tensor([[[0, 1, 0]]])
        y_with_ignore = torch.tensor([[[0, IGNORE, 0]]])
        # ignore画素のロジットを大きく変えても、ignore除外後の損失は不変
        s2 = s.clone()
        s2[..., 1] = 999.0
        loss_a = focal_loss(s, y_with_ignore, ignore_index=IGNORE)
        loss_b = focal_loss(s2, y_with_ignore, ignore_index=IGNORE)
        assert abs(loss_a.item() - loss_b.item()) < 1e-9
        # y_all_valid はダミー(未使用)。ignore画素除外の検証は上記2行のみで十分。
        del y_all_valid

    def test_alpha_class_weight_changes_loss(self):
        s = _rand_logits(2, 3, 4, 4)
        y = torch.randint(0, 3, (2, 4, 4))
        unweighted = focal_loss(s, y, alpha=None, ignore_index=IGNORE)
        alpha = torch.tensor([0.1, 1.0, 1.0], dtype=torch.float64)
        weighted = focal_loss(s, y, alpha=alpha, ignore_index=IGNORE)
        assert abs(unweighted.item() - weighted.item()) > 1e-6

    def test_nonnegative_and_grad_flows(self):
        s = _rand_logits(2, 3, 4, 4).requires_grad_(True)
        y = torch.randint(0, 3, (2, 4, 4))
        loss = focal_loss(s, y, gamma=2.0, ignore_index=IGNORE)
        assert loss.item() >= 0.0
        (grad,) = torch.autograd.grad(loss, s)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0

    def test_all_ignored_no_crash(self):
        """全画素ignoreでもクラッシュせず有限値を返す(分母クランプ)。"""
        s = _rand_logits(1, 2, 2, 2)
        y = torch.full((1, 2, 2), IGNORE, dtype=torch.long)
        loss = focal_loss(s, y, ignore_index=IGNORE)
        assert torch.isfinite(loss)


# ============================================================
# dice_loss
# ============================================================

class TestDiceLoss:
    def test_perfect_prediction_near_zero(self):
        """確信度の高い正しい予測 → dice loss はほぼ0。"""
        y = torch.tensor([[[0, 1], [1, 0]]])
        logits = torch.full((1, 2, 2, 2), -10.0, dtype=torch.float64)
        for b in range(1):
            for i in range(2):
                for j in range(2):
                    logits[b, y[b, i, j], i, j] = 10.0
        loss = dice_loss(logits, y, ignore_index=IGNORE)
        assert loss.item() < 1e-3

    def test_completely_wrong_prediction_near_one(self):
        """常に逆クラスを高確信で予測 → dice loss は1に近い。"""
        y = torch.tensor([[[0, 1], [1, 0]]])
        wrong = 1 - y
        logits = torch.full((1, 2, 2, 2), -10.0, dtype=torch.float64)
        for i in range(2):
            for j in range(2):
                logits[0, wrong[0, i, j], i, j] = 10.0
        loss = dice_loss(logits, y, ignore_index=IGNORE)
        assert loss.item() > 0.9

    def test_ignore_index_excluded(self):
        s = _rand_logits(1, 2, 1, 3)
        y = torch.tensor([[[0, IGNORE, 1]]])
        s2 = s.clone()
        s2[..., 1] = 999.0  # ignore画素だけ変える
        loss_a = dice_loss(s, y, ignore_index=IGNORE)
        loss_b = dice_loss(s2, y, ignore_index=IGNORE)
        assert abs(loss_a.item() - loss_b.item()) < 1e-9

    def test_grad_flows(self):
        s = _rand_logits(2, 4, 4, 4).requires_grad_(True)
        y = torch.randint(0, 4, (2, 4, 4))
        loss = dice_loss(s, y, ignore_index=IGNORE)
        (grad,) = torch.autograd.grad(loss, s)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0

    def test_bounded_zero_one_ish(self):
        s = _rand_logits(2, 5, 3, 3)
        y = torch.randint(0, 5, (2, 3, 3))
        loss = dice_loss(s, y, ignore_index=IGNORE)
        assert -1e-6 <= loss.item() <= 1.0 + 1e-6


# ============================================================
# focal_dice_loss
# ============================================================

class TestFocalDiceLoss:
    def test_equals_weighted_sum(self):
        s = _rand_logits(2, 3, 4, 4)
        y = torch.randint(0, 3, (2, 4, 4))
        f = focal_loss(s, y, gamma=2.0, ignore_index=IGNORE)
        d = dice_loss(s, y, ignore_index=IGNORE)
        got = focal_dice_loss(
            s, y, gamma=2.0, focal_weight=0.3, dice_weight=0.7,
            ignore_index=IGNORE)
        expected = 0.3 * f + 0.7 * d
        assert abs(got.item() - expected.item()) < 1e-10

    def test_default_weights_are_half_half(self):
        s = _rand_logits(1, 3, 3, 3)
        y = torch.randint(0, 3, (1, 3, 3))
        f = focal_loss(s, y, ignore_index=IGNORE)
        d = dice_loss(s, y, ignore_index=IGNORE)
        got = focal_dice_loss(s, y, ignore_index=IGNORE)
        assert abs(got.item() - (0.5 * f + 0.5 * d).item()) < 1e-10


# ============================================================
# build_seg_loss(セレクタ)
# ============================================================

class TestBuildSegLoss:
    def test_ce_matches_cross_entropy_loss(self):
        s = _rand_logits(2, 4, 5, 5)
        y = torch.randint(0, 4, (2, 5, 5))
        y[0, 0, 0] = IGNORE
        fn = build_seg_loss('ce', n_classes=4, ignore_index=IGNORE)
        got = fn(s, y)
        expected = F.cross_entropy(s, y.long(), ignore_index=IGNORE)
        assert abs(got.item() - expected.item()) < 1e-10

    def test_weighted_ce_uses_class_weights(self):
        s = _rand_logits(2, 3, 4, 4)
        y = torch.randint(0, 3, (2, 4, 4))
        w = torch.tensor([0.2, 1.0, 5.0], dtype=torch.float64)
        fn = build_seg_loss(
            'weighted_ce', n_classes=3, ignore_index=IGNORE, class_weights=w)
        got = fn(s, y)
        expected = F.cross_entropy(s, y.long(), weight=w, ignore_index=IGNORE)
        assert abs(got.item() - expected.item()) < 1e-10

    def test_focal_dice_dispatch(self):
        s = _rand_logits(1, 3, 4, 4)
        y = torch.randint(0, 3, (1, 4, 4))
        fn = build_seg_loss(
            'focal_dice', n_classes=3, ignore_index=IGNORE,
            gamma=1.5, focal_weight=0.4, dice_weight=0.6)
        got = fn(s, y)
        expected = focal_dice_loss(
            s, y, gamma=1.5, focal_weight=0.4, dice_weight=0.6,
            ignore_index=IGNORE)
        assert abs(got.item() - expected.item()) < 1e-10

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            build_seg_loss('bogus', n_classes=3)

    @pytest.mark.parametrize('name', SEG_LOSS_NAMES)
    def test_all_names_grad_flow(self, name):
        s = _rand_logits(2, 4, 4, 4).requires_grad_(True)
        y = torch.randint(0, 4, (2, 4, 4))
        w = torch.ones(4, dtype=torch.float64)
        fn = build_seg_loss(name, n_classes=4, ignore_index=IGNORE, class_weights=w)
        loss = fn(s, y)
        assert torch.isfinite(loss)
        (grad,) = torch.autograd.grad(loss, s)
        assert torch.isfinite(grad).all()
