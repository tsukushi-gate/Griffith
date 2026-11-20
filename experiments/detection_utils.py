"""
CenterNet用ユーティリティ。

ガウシアンヒートマップ生成、ピーク検出、bbox復号、mAP計算。
"""

import math

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def gaussian_radius(det_size: tuple[int, int], min_overlap: float = 0.7) -> int:
    """
    CenterNet準拠のガウシアン半径計算。

    バウンディングボックスのサイズから、IoU >= min_overlapを保証する
    最小のガウシアン半径を求める。

    Args:
        det_size: (height, width) バウンディングボックスのサイズ
        min_overlap: 最小IoU閾値

    Returns:
        ガウシアン半径（整数）
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: torch.Tensor, center: tuple[int, int], radius: int):
    """
    2Dガウシアンをヒートマップ上に描画（element-wise max）。

    Args:
        heatmap: (H, W) ヒートマップテンソル（in-place更新）
        center: (x, y) 中心座標
        radius: ガウシアン半径
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0

    # ガウシアンカーネル生成
    x_range = torch.arange(diameter, dtype=torch.float32, device=heatmap.device)
    y_range = torch.arange(diameter, dtype=torch.float32, device=heatmap.device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    gaussian = torch.exp(
        -((x_grid - radius) ** 2 + (y_grid - radius) ** 2) / (2 * sigma ** 2)
    )

    cx, cy = center
    H, W = heatmap.shape

    # クリッピング
    left = max(0, cx - radius)
    right = min(W, cx + radius + 1)
    top = max(0, cy - radius)
    bottom = min(H, cy + radius + 1)

    g_left = max(0, radius - cx)
    g_right = g_left + (right - left)
    g_top = max(0, radius - cy)
    g_bottom = g_top + (bottom - top)

    if left < right and top < bottom:
        heatmap[top:bottom, left:right] = torch.max(
            heatmap[top:bottom, left:right],
            gaussian[g_top:g_bottom, g_left:g_right],
        )


def encode_centernet_targets(
    gt_boxes: list[tuple[float, float, float, float]],
    gt_labels: list[int],
    *,
    n_classes: int,
    hm_h: int,
    hm_w: int,
    output_stride: int,
    max_objects: int,
) -> dict[str, torch.Tensor]:
    """
    CenterNet形式のGT（ヒートマップ・サイズ・オフセット等）を生成する。

    Multi-MNIST / Pascal VOC 両データセットで共有するエンコード実装。

    Args:
        gt_boxes: [(cx, cy, w, h), ...] ピクセル座標（入力画像座標系）
        gt_labels: [class_id, ...]
        n_classes: クラス数
        hm_h: ヒートマップ高さ（= 入力H // output_stride）
        hm_w: ヒートマップ幅（= 入力W // output_stride）
        output_stride: 出力ストライド
        max_objects: 1画像あたりの最大オブジェクト数

    Returns:
        'heatmap': (n_classes, hm_h, hm_w)
        'size': (max_objects, 2) 正規化サイズ [w_norm, h_norm]
        'offset': (max_objects, 2) サブピクセルオフセット [dx, dy]
        'indices': (max_objects,) ヒートマップ上のフラットインデックス
        'mask': (max_objects,) 有効マスク
        'gt_boxes': (max_objects, 4) [x1, y1, x2, y2] ピクセル座標
        'gt_labels': (max_objects,) クラスラベル
    """
    heatmap = torch.zeros(n_classes, hm_h, hm_w)
    size = torch.zeros(max_objects, 2)
    offset = torch.zeros(max_objects, 2)
    indices = torch.zeros(max_objects, dtype=torch.long)
    mask = torch.zeros(max_objects, dtype=torch.float32)
    boxes_out = torch.zeros(max_objects, 4)
    labels_out = torch.zeros(max_objects, dtype=torch.long)

    s = output_stride

    for i, ((cx, cy, w, h), label) in enumerate(zip(gt_boxes, gt_labels)):
        if i >= max_objects:
            break

        # ヒートマップ座標に変換
        cx_hm = cx / s
        cy_hm = cy / s
        w_hm = w / s
        h_hm = h / s

        # 整数座標
        cx_int = int(cx_hm)
        cy_int = int(cy_hm)
        cx_int = min(cx_int, hm_w - 1)
        cy_int = min(cy_int, hm_h - 1)

        # ガウシアン半径
        radius = max(1, gaussian_radius((int(h_hm), int(w_hm)), min_overlap=0.7))
        draw_gaussian(heatmap[label], (cx_int, cy_int), radius)

        # 正規化サイズ（ヒートマップサイズで割る）
        size[i, 0] = w_hm / hm_w
        size[i, 1] = h_hm / hm_h

        # サブピクセルオフセット
        offset[i, 0] = cx_hm - cx_int
        offset[i, 1] = cy_hm - cy_int

        # フラットインデックス
        indices[i] = cy_int * hm_w + cx_int

        # マスク
        mask[i] = 1.0

        # bbox (x1, y1, x2, y2)
        boxes_out[i] = torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        labels_out[i] = label

    return {
        'heatmap': heatmap,
        'size': size,
        'offset': offset,
        'indices': indices,
        'mask': mask,
        'gt_boxes': boxes_out,
        'gt_labels': labels_out,
    }


def nms_by_maxpool(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    3×3 max poolingによるNMS代替（CenterNet方式）。

    ピーク（周囲より値が大きい点）のみを残す。
    softモード（連続値）で同値ピクセルが複数残る問題を防ぐため、
    位置依存の微小ノイズでタイブレークする。

    Args:
        heatmap: (B, C, H, W) ヒートマップ
        kernel_size: max poolingカーネルサイズ

    Returns:
        ピークのみ残したヒートマップ（非ピークは0）
    """
    pad = kernel_size // 2
    # 同値ピクセル対策: 微小な位置依存ノイズでタイブレーク
    # 勾配に影響しないよう detach する
    B, C, H, W = heatmap.shape
    noise = torch.arange(H * W, device=heatmap.device, dtype=heatmap.dtype)
    noise = noise.reshape(1, 1, H, W) * 1e-6
    heatmap_tb = heatmap + noise
    hmax = F.max_pool2d(heatmap_tb, kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap_tb).float()
    return heatmap * keep


def decode_detections(
    heatmap: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    output_stride: int = 4,
    top_k: int = 100,
    threshold: float = 0.1,
) -> list[dict]:
    """
    モデル出力からバウンディングボックスを復号する。

    Args:
        heatmap: (B, n_classes, H, W) ヒートマップ（[0,1]）
        size: (B, 2, H, W) 正規化サイズ [w_norm, h_norm]
        offset: (B, 2, H, W) サブピクセルオフセット [dx, dy]
        output_stride: 出力ストライド（入力画像との縮小比）
        top_k: 保持する上位検出数
        threshold: 検出閾値

    Returns:
        バッチごとの検出結果リスト。各要素は:
            'boxes': (N, 4) [x1, y1, x2, y2] ピクセル座標
            'scores': (N,) 信頼度スコア
            'labels': (N,) クラスラベル
    """
    B, n_classes, H, W = heatmap.shape
    results = []

    # ピーク検出
    heatmap_nms = nms_by_maxpool(heatmap)

    for b in range(B):
        # (n_classes, H, W) → (n_classes * H * W,)
        scores_flat = heatmap_nms[b].reshape(-1)

        # 上位K個を選択
        K = min(top_k, scores_flat.shape[0])
        topk_scores, topk_inds = scores_flat.topk(K)

        # 閾値フィルタ
        mask = topk_scores > threshold
        topk_scores = topk_scores[mask]
        topk_inds = topk_inds[mask]

        if topk_inds.numel() == 0:
            results.append({
                'boxes': torch.zeros(0, 4, device=heatmap.device),
                'scores': torch.zeros(0, device=heatmap.device),
                'labels': torch.zeros(0, dtype=torch.long, device=heatmap.device),
            })
            continue

        # インデックスからクラス・座標を復元
        topk_classes = topk_inds // (H * W)
        topk_spatial = topk_inds % (H * W)
        topk_ys = topk_spatial // W
        topk_xs = topk_spatial % W

        # オフセット取得
        off_x = offset[b, 0, topk_ys, topk_xs]
        off_y = offset[b, 1, topk_ys, topk_xs]

        # サイズ取得（正規化値 → ピクセル値）
        w = size[b, 0, topk_ys, topk_xs]
        h = size[b, 1, topk_ys, topk_xs]

        # 中心座標（ピクセル座標系）
        cx = (topk_xs.float() + off_x) * output_stride
        cy = (topk_ys.float() + off_y) * output_stride

        # サイズをピクセル座標に変換
        w_px = w * output_stride * W
        h_px = h * output_stride * H

        # (x1, y1, x2, y2) 形式
        x1 = cx - w_px / 2
        y1 = cy - h_px / 2
        x2 = cx + w_px / 2
        y2 = cy + h_px / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        results.append({
            'boxes': boxes,
            'scores': topk_scores,
            'labels': topk_classes,
        })

    return results


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> list[bool]:
    """各予測がGTと正しくマッチするかを判定する。

    Returns:
        予測ごとのboolリスト（True=正解、False=不正解）
    """
    n_pred = pred_boxes.shape[0]
    if n_pred == 0 or gt_boxes.numel() == 0:
        return [False] * n_pred

    iou_matrix = compute_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    results = []

    for i in range(n_pred):
        best_iou, best_j = iou_matrix[i].max(0)
        j = best_j.item()
        if (best_iou >= iou_threshold
                and j not in matched_gt
                and pred_labels[i] == gt_labels[j]):
            results.append(True)
            matched_gt.add(j)
        else:
            results.append(False)

    return results


def visualize_detections(
    image: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    score_threshold: float = 0.1,
    iou_threshold: float = 0.5,
) -> tuple[Image.Image, bool]:
    """
    検出結果をバウンディングボックス付きで可視化する。

    色の意味:
        - 水色（cyan）破線風: GT（正解ボックス）
        - 緑（lime）: 正解の予測（GTとIoU≥0.5かつクラス一致）
        - 赤（red）: 不正解の予測（誤検出 or クラス誤り）

    Args:
        image: (C, H, W) 入力画像テンソル [0,1]
        gt_boxes: (M, 4) GT [x1, y1, x2, y2]
        gt_labels: (M,) GTラベル
        pred_boxes: (N, 4) 予測 [x1, y1, x2, y2]
        pred_scores: (N,) 信頼度スコア
        pred_labels: (N,) 予測ラベル
        score_threshold: 描画する最低スコア
        iou_threshold: 正解判定のIoU閾値

    Returns:
        (PIL画像, 全予測が正解かどうか)
    """
    import numpy as np

    # グレースケール → RGB
    gray = image[0].cpu().numpy()
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    # 画像を3倍に拡大（テキストを読みやすくする）
    SCALE = 3
    h, w = rgb.shape[:2]
    pil_img = Image.fromarray(rgb).resize((w * SCALE, h * SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(pil_img)

    def s(v):
        return v * SCALE

    # スコアでフィルタ
    if pred_boxes.numel() > 0:
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]

    # 正解/不正解の判定
    is_correct = _match_predictions(
        pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold,
    )

    # 見逃したGTを特定
    n_gt = gt_boxes.shape[0]
    matched_gt = set()
    if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
        iou_matrix = compute_iou(pred_boxes, gt_boxes)
        for i, correct in enumerate(is_correct):
            if correct:
                _, best_j = iou_matrix[i].max(0)
                matched_gt.add(best_j.item())

    # GT（水色、太線）
    for i in range(n_gt):
        x1, y1, x2, y2 = gt_boxes[i].tolist()
        label = int(gt_labels[i].item())
        color = 'cyan' if i in matched_gt else 'orange'
        draw.rectangle([s(x1), s(y1), s(x2), s(y2)], outline=color, width=2)
        tag = f'GT:{label}'
        if i not in matched_gt:
            tag += ' MISS'
        draw.text((s(x1), max(0, s(y1) - 12)), tag, fill=color)

    # 予測（正解=緑、不正解=赤）
    for i in range(pred_boxes.shape[0]):
        x1, y1, x2, y2 = pred_boxes[i].tolist()
        score = pred_scores[i].item()
        label = int(pred_labels[i].item())

        if is_correct[i]:
            color = 'lime'
            tag = f'{label}:{score:.2f} ✓'
        else:
            color = 'red'
            tag = f'{label}:{score:.2f} ✗'

        # 予測boxは座標が反転しうる(未学習/floatヘッドでw,hが負になる等)。
        # PILのrectangleはx1>=x0,y1>=y0を要求し ValueError で落ちるため、
        # 描画時のみ正規化する(mAP計算側の値は変更しない=評価に影響なし)。
        px1, px2 = sorted((s(x1), s(x2)))
        py1, py2 = sorted((s(y1), s(y2)))
        draw.rectangle([px1, py1, px2, py2], outline=color, width=2)
        # 予測ラベルはboxの下に表示（GTラベルと重ならないように）
        draw.text((px1, py2 + 2), tag, fill=color)

    all_correct = all(is_correct) and len(matched_gt) == n_gt if n_gt > 0 else True
    return pil_img, all_correct


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU（Intersection over Union）を計算する。

    Args:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]

    Returns:
        (N, M) IoU行列
    """
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return inter / (union + 1e-6)


def compute_ap_per_class(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
    pred_image_ids: torch.Tensor | None = None,
    gt_image_ids: torch.Tensor | None = None,
) -> float:
    """
    単一クラスのAverage Precision (AP) を計算する。

    Args:
        pred_boxes: (N, 4) 予測ボックス
        pred_scores: (N,) 信頼度スコア
        gt_boxes: (M, 4) 正解ボックス
        iou_threshold: IoU閾値
        pred_image_ids: (N,) 各予測がどの画像のものか。gt_image_ids と対で
            与えると、**予測は同じ画像のGTとしか対応付かない**（標準の
            VOC/COCO プロトコル）。省略すると全予測・全GTを1枚の画像として
            扱う（データセット全体をプールした照合になるため、複数画像の
            評価では使ってはならない）。
        gt_image_ids: (M,) 各GTがどの画像のものか。

    Returns:
        AP値
    """
    if gt_boxes.numel() == 0:
        return 1.0 if pred_boxes.numel() == 0 else 0.0
    if pred_boxes.numel() == 0:
        return 0.0

    # スコア降順でソート
    sorted_indices = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[sorted_indices]

    n_gt = gt_boxes.shape[0]
    matched = torch.zeros(n_gt, dtype=torch.bool, device=gt_boxes.device)

    tp = []
    fp = []

    iou_matrix = compute_iou(pred_boxes, gt_boxes)
    if pred_image_ids is not None and gt_image_ids is not None:
        # 画像スコープの強制: 別画像のGTとのIoUを閾値未満（-1）へ潰す。
        # これが無いと「画像Aの予測が画像BのGTに当たる」照合が起き、
        # 「とにかく箱を多く出す」モデルほどAPが高く出てしまう。
        same_image = (
            pred_image_ids[sorted_indices].unsqueeze(1) == gt_image_ids.unsqueeze(0)
        )
        iou_matrix = iou_matrix.masked_fill(~same_image, -1.0)

    for i in range(pred_boxes.shape[0]):
        ious = iou_matrix[i]
        best_iou, best_gt = ious.max(0)

        if best_iou >= iou_threshold and not matched[best_gt]:
            tp.append(1)
            fp.append(0)
            matched[best_gt] = True
        else:
            tp.append(0)
            fp.append(1)

    tp = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp = torch.tensor(fp, dtype=torch.float32).cumsum(0)

    precision = tp / (tp + fp)
    recall = tp / n_gt

    # 11点補間AP
    ap = 0.0
    for t in torch.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max().item()
    ap /= 11.0

    return ap


def compute_map(
    all_pred: list[dict],
    all_gt: list[dict],
    n_classes: int = 10,
    iou_threshold: float = 0.5,
) -> tuple[float, list[float]]:
    """
    全クラスのmAP (mean Average Precision) を計算する。

    Args:
        all_pred: バッチ分の予測結果（decode_detectionsの出力）
        all_gt: バッチ分の正解データ
            各要素: {'boxes': (M, 4), 'labels': (M,)}
        n_classes: クラス数
        iou_threshold: IoU閾値

    Returns:
        (mAP, クラスごとのAP リスト)
    """
    aps = []

    for c in range(n_classes):
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_images = []
        all_gt_boxes = []
        all_gt_images = []

        for image_id, (pred, gt) in enumerate(zip(all_pred, all_gt)):
            pred_mask = pred['labels'] == c
            gt_mask = gt['labels'] == c

            if pred_mask.any():
                all_pred_boxes.append(pred['boxes'][pred_mask])
                all_pred_scores.append(pred['scores'][pred_mask])
                all_pred_images.append(torch.full(
                    (int(pred_mask.sum()),), image_id, dtype=torch.long))
            if gt_mask.any():
                all_gt_boxes.append(gt['boxes'][gt_mask])
                all_gt_images.append(torch.full(
                    (int(gt_mask.sum()),), image_id, dtype=torch.long))

        if not all_gt_boxes:
            continue

        pred_boxes = torch.cat(all_pred_boxes) if all_pred_boxes else torch.zeros(0, 4)
        pred_scores = torch.cat(all_pred_scores) if all_pred_scores else torch.zeros(0)
        pred_images = (torch.cat(all_pred_images) if all_pred_images
                       else torch.zeros(0, dtype=torch.long))
        gt_boxes = torch.cat(all_gt_boxes)
        gt_images = torch.cat(all_gt_images)

        # 画像IDを必ず渡す（渡さないとデータセット全体をプールして照合され、
        # 画像をまたいだ偶然の一致がTPになる）
        ap = compute_ap_per_class(
            pred_boxes, pred_scores, gt_boxes, iou_threshold,
            pred_image_ids=pred_images, gt_image_ids=gt_images,
        )
        aps.append(ap)

    mean_ap = sum(aps) / len(aps) if aps else 0.0
    return mean_ap, aps
