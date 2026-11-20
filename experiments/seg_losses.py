"""セグメンテーション不均衡対策損失。

VOC 21クラス(背景73%)× 離散LUT(softmax16g)+重みなしCE で LUT scratch が
mIoU 0.035 に固着する(背景collapse)問題への対策。focal loss / dice loss /
class-weighted CE(median-frequency balancing)を実装し、
`build_seg_loss()` で名前から損失関数を選択できるようにする。

すべて IGNORE_INDEX(既定255)対応: 無視画素は損失にもクラス重み計算にも
寄与しない。scratch訓練(experiments/train_segmentation.py)と
KD(griffith_logic/convert/distill.py の make_dense_kd_loss hard_loss_fn)の
両方で同一の損失関数を再利用できるよう、統一シグネチャ
`loss_fn(logits[B,C,H,W], labels[B,H,W]) -> スカラー` を持つ。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

SEG_LOSS_NAMES = ('ce', 'weighted_ce', 'focal', 'dice', 'focal_dice')


# ============================================================
# クラス重み(median-frequency balancing)
# ============================================================

def median_frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    """クラス別画素数からmedian-frequency balancing重みを計算する(純関数)。

    weight_c = median(全クラスの出現頻度) / freq_c
        (freq_c = count_c / sum(counts)、Eigen & Fergus 2015 のMFB定義)

    多数派クラス(freq_c > median)は重み<1に抑制され、少数派クラス
    (freq_c < median)は重み>1に拡大される。出現数0のクラスは重み1.0
    (欠測クラス、寄与なしなので中立値)とする。中央値は
    `torch.quantile(..., 0.5)`(線形補間、要素数偶数時は中央2値の平均。
    numpy.median と同じ規約)で求める(`torch.Tensor.median()`は偶数個の場合
    下側要素を返す実装のため、ここでは意図的に使わない)。

    Args:
        counts: [C] 各クラスの(ignore画素を除く)画素数。

    Returns:
        [C] float32 の重みテンソル。
    """
    counts = counts.to(torch.float64)
    total = counts.sum().clamp(min=1.0)
    freq = counts / total
    present = freq > 0
    weights = torch.ones_like(freq)
    if present.any():
        median_freq = torch.quantile(freq[present], 0.5)
        weights = torch.where(present, median_freq / freq.clamp(min=1e-12), weights)
    return weights.to(torch.float32)


def compute_class_pixel_counts(
    loader,
    n_classes: int,
    ignore_index: int = 255,
    max_batches: int | None = None,
) -> torch.Tensor:
    """DataLoaderを走査してクラス別(ignore除外)画素数を集計する。

    バッチは (x, y) または (x, y, x_raw)(KD時の生画像同梱)のいずれの形式
    でもよい(y=batch[1]のみ使用)。

    Args:
        loader: torch.utils.data.DataLoader(またはイテラブル)。
        n_classes: クラス数C。
        ignore_index: 除外するラベル値(既定255)。
        max_batches: 集計に使う先頭バッチ数の上限(Noneで全バッチ)。

    Returns:
        [C] long の画素数テンソル。
    """
    counts = torch.zeros(n_classes, dtype=torch.long)
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        y = batch[1]
        y = y.reshape(-1)
        y = y[y != ignore_index]
        if y.numel() == 0:
            continue
        counts += torch.bincount(y.long(), minlength=n_classes)[:n_classes]
    return counts


def compute_class_weights_from_loader(
    loader,
    n_classes: int,
    ignore_index: int = 255,
    max_batches: int | None = None,
) -> torch.Tensor:
    """DataLoaderからmedian-frequency balancingのクラス重みを直接計算する。

    compute_class_pixel_counts + median_frequency_weights のショートカット。
    """
    counts = compute_class_pixel_counts(
        loader, n_classes, ignore_index=ignore_index, max_batches=max_batches)
    return median_frequency_weights(counts)


# ============================================================
# 損失関数本体
# ============================================================

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """多クラスfocal loss(Lin et al. 2017)、IGNORE_INDEX対応。

    focal_c = (1 - p_t)^gamma * CE_c を有効画素で平均する。数値安定のため
    p_t = exp(-CE)(CE は F.cross_entropy の log_softmax 経由、log(0)を
    直接扱わない)で求める。alpha(クラス重み)は F.cross_entropy の
    weight引数にそのまま渡し(多数派クラスの易しい画素をさらに抑制)、
    ignore_index はF.cross_entropyの標準機構で該当画素のCEを0にしてから
    有効画素数で正規化する(重みなしの単純平均。alpha使用時もclass-weighted
    な重み付き平均ではなく画素数平均である点に注意)。

    Args:
        logits: [B, C, H, W]。
        labels: [B, H, W] long(値 ∈ [0, C) ∪ {ignore_index})。
        gamma: focusingパラメータ(既定2.0。gamma=0で通常CEと一致)。
        alpha: [C] クラス重み(Noneで重みなし)。
        ignore_index: 除外ラベル値。

    Returns:
        スカラー(有効画素平均)。
    """
    labels = labels.long()
    ce = F.cross_entropy(
        logits, labels, weight=alpha, ignore_index=ignore_index, reduction='none',
    )  # [B,H,W](ignore画素は0)
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    valid = (labels != ignore_index).to(focal.dtype)
    n_valid = valid.sum().clamp(min=1.0)
    return (focal * valid).sum() / n_valid


def dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    """多クラスsoft dice loss(クラス平均)、IGNORE_INDEX対応。

    各クラスについて soft dice係数 = (2·Σ(p·t) + eps) / (Σp + Σt + eps) を
    バッチ+空間全体で集計し(全画素を1つの集合として扱うマクロ平均)、
    `1 - クラス平均dice` を返す。ignore画素は確率・one-hotターゲット双方を
    0埋めして集計から除外する。

    領域重なりを直接最適化するため、不均衡データでも少数派クラスの
    IoU/Diceが希釈されにくい(Milletari et al. 2016)。

    Args:
        logits: [B, C, H, W]。
        labels: [B, H, W] long(値 ∈ [0, C) ∪ {ignore_index})。
        ignore_index: 除外ラベル値。
        eps: ゼロ割防止(Laplace smoothing)。

    Returns:
        スカラー(1 - クラス平均dice、s==t で0に漸近)。
    """
    labels = labels.long()
    n_classes = logits.shape[1]
    valid = labels != ignore_index  # [B,H,W]
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    probs = F.softmax(logits, dim=1)  # [B,C,H,W]
    onehot = F.one_hot(safe_labels, num_classes=n_classes)  # [B,H,W,C]
    onehot = onehot.permute(0, 3, 1, 2).to(probs.dtype)  # [B,C,H,W]

    vmask = valid.unsqueeze(1).to(probs.dtype)  # [B,1,H,W]
    probs = probs * vmask
    onehot = onehot * vmask

    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dim=dims)  # [C]
    denom = probs.sum(dim=dims) + onehot.sum(dim=dims)  # [C]
    dice_c = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice_c.mean()


def focal_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    focal_weight: float = 0.5,
    dice_weight: float = 0.5,
    ignore_index: int = 255,
) -> torch.Tensor:
    """focal_weight·focal_loss + dice_weight·dice_loss(既定0.5:0.5)。"""
    f = focal_loss(logits, labels, gamma=gamma, alpha=alpha, ignore_index=ignore_index)
    d = dice_loss(logits, labels, ignore_index=ignore_index)
    return focal_weight * f + dice_weight * d


# ============================================================
# 損失セレクタ
# ============================================================

def build_seg_loss(
    name: str,
    n_classes: int,
    ignore_index: int = 255,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
    focal_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """`--seg-loss` 名から統一シグネチャの損失関数を構築する。

    返す loss_fn(logits[B,C,H,W], labels[B,H,W]) -> スカラー は
    scratch訓練(そのまま呼ぶ)とKD(make_dense_kd_lossのhard_loss_fnに渡す)の
    両方で使える。'ce'(既定)は torch.nn.CrossEntropyLoss(ignore_index=...)と
    厳密一致(後方互換)。

    Args:
        name: 'ce' / 'weighted_ce' / 'focal' / 'dice' / 'focal_dice'。
        n_classes: クラス数C(未使用だが将来の検証用に受け取る)。
        ignore_index: 除外ラベル値。
        class_weights: [C] クラス重み(weighted_ce/focalで使用。Noneなら
            重みなし)。dice/focal_diceのdice項には適用されない
            (dice係数はクラス平均自体が不均衡耐性を持つため)。
        gamma: focal loss のgamma(focal/focal_dice)。
        focal_weight: focal_dice内のfocal項重み。
        dice_weight: focal_dice内のdice項重み。

    Returns:
        loss_fn(logits, labels) -> スカラー。

    Raises:
        ValueError: name が未対応。
    """
    if name not in SEG_LOSS_NAMES:
        raise ValueError(f'未対応 --seg-loss: {name!r}(対応: {SEG_LOSS_NAMES})')

    del n_classes  # 現状は形状をlogits.shapeから直接見るため未使用

    if name == 'ce':
        def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(logits, labels.long(), ignore_index=ignore_index)
        return _ce

    if name == 'weighted_ce':
        def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            w = (class_weights.to(device=logits.device, dtype=logits.dtype)
                 if class_weights is not None else None)
            return F.cross_entropy(
                logits, labels.long(), weight=w, ignore_index=ignore_index)
        return _weighted_ce

    if name == 'focal':
        def _focal(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            w = (class_weights.to(device=logits.device, dtype=logits.dtype)
                 if class_weights is not None else None)
            return focal_loss(logits, labels, gamma=gamma, alpha=w,
                              ignore_index=ignore_index)
        return _focal

    if name == 'dice':
        def _dice(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return dice_loss(logits, labels, ignore_index=ignore_index)
        return _dice

    # focal_dice
    def _focal_dice(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = (class_weights.to(device=logits.device, dtype=logits.dtype)
             if class_weights is not None else None)
        return focal_dice_loss(
            logits, labels, gamma=gamma, alpha=w,
            focal_weight=focal_weight, dice_weight=dice_weight,
            ignore_index=ignore_index)
    return _focal_dice
