"""
事前学習検出器（COCO）による Pascal VOC2007 test 参照評価。

torchvision の COCO 事前学習検出器を VOC2007 test に推論し、
COCO→VOC クラスマップで20クラスへ射影、予測boxを160px座標系へスケールして、
train_detection と同一プロトコル（detection_utils.compute_map, mAP@0.5）で評価する。

DLGN/CNN CenterNet の到達目標（上限）を示す参照値。

使用例:
    uv run python experiments/eval_pretrained_detector_voc.py
    uv run python experiments/eval_pretrained_detector_voc.py --models fasterrcnn --limit 100
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torchvision
from data.detection_voc import VOC_CLASSES, VOCDetectionCenterNet
from detection_utils import compute_map
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

load_dotenv()

# VOCクラス名 → COCOカテゴリ名（固定辞書）
VOC_TO_COCO_NAME = {
    'aeroplane': 'airplane',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'boat': 'boat',
    'bottle': 'bottle',
    'bus': 'bus',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'cow': 'cow',
    'diningtable': 'dining table',
    'dog': 'dog',
    'horse': 'horse',
    'motorbike': 'motorcycle',
    'person': 'person',
    'pottedplant': 'potted plant',
    'sheep': 'sheep',
    'sofa': 'couch',
    'train': 'train',
    'tvmonitor': 'tv',
}

IMG_SIZE = 160


def build_coco_to_voc(coco_categories: list[str]) -> dict[int, int]:
    """COCOラベルid → VOCクラスidの写像を weights.meta から構築する。"""
    coco_name_to_id = {name: i for i, name in enumerate(coco_categories)}
    coco_to_voc = {}
    for voc_idx, voc_name in enumerate(VOC_CLASSES):
        coco_name = VOC_TO_COCO_NAME[voc_name]
        if coco_name not in coco_name_to_id:
            raise KeyError(f'COCOカテゴリに {coco_name} が見つかりません')
        coco_to_voc[coco_name_to_id[coco_name]] = voc_idx
    return coco_to_voc


MODEL_REGISTRY = {
    'fasterrcnn': (
        torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        lambda: torchvision.models.detection
        .FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    ),
    'ssdlite': (
        torchvision.models.detection.ssdlite320_mobilenet_v3_large,
        lambda: torchvision.models.detection
        .SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    ),
    # W8: 少クラス実物体検出のteacher候補（COCO box_map=46.7、mobilenet系より
    # はるかに強い。resnet50+FPN、2段検出器）
    'fasterrcnn_r50': (
        torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        lambda: torchvision.models.detection
        .FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    ),
    'retinanet_r50': (
        torchvision.models.detection.retinanet_resnet50_fpn_v2,
        lambda: torchvision.models.detection
        .RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
    ),
}


def build_coco_to_classes(
    coco_categories: list[str], classes: list[str],
) -> dict[int, int]:
    """COCOラベルid → classes内ローカルidの写像を構築する（build_coco_to_vocの一般化）。

    classes は VOC_CLASSES のサブセット（順序が local id 0..len(classes)-1 を決める）。
    """
    coco_name_to_id = {name: i for i, name in enumerate(coco_categories)}
    coco_to_local = {}
    for local_idx, voc_name in enumerate(classes):
        coco_name = VOC_TO_COCO_NAME[voc_name]
        if coco_name not in coco_name_to_id:
            raise KeyError(f'COCOカテゴリに {coco_name} が見つかりません')
        coco_to_local[coco_name_to_id[coco_name]] = local_idx
    return coco_to_local


@torch.no_grad()
def evaluate_model(model_key, device, score_threshold, limit, download,
                    classes=None, img_size=IMG_SIZE):
    """1モデルを VOC2007 test に評価して mAP@0.5 を返す。

    Args:
        classes: None=VOC全20クラス。指定時はそのサブセットのみでGT/予測を
            フィルタし、mAPもそのクラス数で計算する（W8少クラス参照評価）。
        img_size: GT/予測をスケールする正方形座標系のサイズ。
    """
    build_fn, weights_fn = MODEL_REGISTRY[model_key]
    weights = weights_fn()
    model = build_fn(weights=weights, box_score_thresh=score_threshold)
    model.eval().to(device)

    coco_categories = weights.meta['categories']
    eval_classes = list(classes) if classes is not None else VOC_CLASSES
    n_eval_classes = len(eval_classes)
    coco_to_local = build_coco_to_classes(coco_categories, eval_classes)
    class_to_local = {c: i for i, c in enumerate(eval_classes)}

    # GT取得用（推論は原寸画像で行い box を img_size へスケールするので use_rgb不問）
    if classes is not None:
        from data.detection_fewclass import FewClassVOCDetection
        ds = FewClassVOCDetection(
            train=False, img_size=img_size, classes=tuple(eval_classes),
            use_rgb=True, download=download, limit=limit,
        )
    else:
        ds = VOCDetectionCenterNet(
            train=False, img_size=img_size, use_rgb=True,
            download=download, limit=limit,
        )
    to_tensor = torchvision.transforms.ToTensor()

    all_pred = []
    all_gt = []

    for idx in tqdm(range(len(ds)), desc=f'{model_key}', leave=False):
        if classes is not None:
            bi, li = ds._pairs[idx]
            base = ds.bases[bi]
            img, target = base[li]
            img = img.convert('RGB')
            from data.detection_fewclass import _parse_objects_fewclass
            boxes_xyxy, labels, (orig_w, orig_h) = _parse_objects_fewclass(
                target['annotation'], class_to_local)
        else:
            base, local = ds._locate(idx)
            img, target = base[local]
            img = img.convert('RGB')
            boxes_xyxy, labels, (orig_w, orig_h) = ds._parse_objects(
                target['annotation'])

        sx = img_size / orig_w
        sy = img_size / orig_h

        # GT（img_size座標）
        gt_boxes = torch.tensor(
            [[x1 * sx, y1 * sy, x2 * sx, y2 * sy]
             for (x1, y1, x2, y2) in boxes_xyxy],
            dtype=torch.float32,
        ).reshape(-1, 4)
        gt_labels = torch.tensor(labels, dtype=torch.long)
        all_gt.append({'boxes': gt_boxes, 'labels': gt_labels})

        # 推論（原寸画像）
        x = to_tensor(img).to(device)
        out = model([x])[0]
        p_boxes = out['boxes'].cpu()
        p_scores = out['scores'].cpu()
        p_labels = out['labels'].cpu()

        keep_boxes, keep_scores, keep_labels = [], [], []
        for b in range(p_boxes.shape[0]):
            coco_id = int(p_labels[b].item())
            if coco_id not in coco_to_local:
                continue
            x1, y1, x2, y2 = p_boxes[b].tolist()
            keep_boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
            keep_scores.append(p_scores[b].item())
            keep_labels.append(coco_to_local[coco_id])

        all_pred.append({
            'boxes': torch.tensor(keep_boxes, dtype=torch.float32).reshape(-1, 4),
            'scores': torch.tensor(keep_scores, dtype=torch.float32),
            'labels': torch.tensor(keep_labels, dtype=torch.long),
        })

    mAP, aps = compute_map(
        all_pred, all_gt, n_classes=n_eval_classes, iou_threshold=0.5)
    return mAP, aps


def main():
    parser = argparse.ArgumentParser(
        description='COCO事前学習検出器のVOC2007 test参照評価')
    parser.add_argument('--models', type=str, default='fasterrcnn,ssdlite',
                        help='評価するモデル(カンマ区切り): fasterrcnn,ssdlite')
    parser.add_argument('--score-threshold', type=float, default=0.05,
                        help='予測スコア閾値 (default: 0.05)')
    parser.add_argument('--limit', type=int, default=0,
                        help='評価画像数の上限(スモーク用、0=全件)')
    parser.add_argument('--download', action='store_true',
                        help='VOCデータを自動ダウンロード')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--classes', type=str, default=None,
                        help='少クラス参照評価用のVOCクラス名(カンマ区切り)。'
                             '未指定=VOC全20クラス (W8: 例 person,car,dog)')
    parser.add_argument('--img-size', type=int, default=IMG_SIZE,
                        help=f'GT/予測のスケール先座標系サイズ (default: {IMG_SIZE})')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbログを無効化')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'デバイス: {args.device}')

    model_keys = [m.strip() for m in args.models.split(',') if m.strip()]
    limit = args.limit if args.limit > 0 else None
    classes = ([c.strip() for c in args.classes.split(',') if c.strip()]
               if args.classes else None)
    eval_classes = classes if classes is not None else VOC_CLASSES

    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=f'pretrained_ref_voc_{"_".join(model_keys)}',
            config_dict=vars(args),
            model=None,
            device=args.device,
            tags=['w2det', 'pretrained-ref', 'voc'] + (['w8det', 'fewclass'] if classes else []),
            group='detseg-web',
        )

    results = {}
    for key in model_keys:
        t0 = time.time()
        print(f'\n=== {key} を VOC2007 test で評価 '
              f'({"全20クラス" if classes is None else ",".join(classes)}) ===')
        mAP, aps = evaluate_model(
            key, args.device, args.score_threshold, limit, args.download,
            classes=classes, img_size=args.img_size,
        )
        dt = str(timedelta(seconds=int(time.time() - t0)))
        print(f'{key}: mAP@0.5 = {mAP:.4f}  (所要 {dt})')
        for c, ap in enumerate(aps):
            print(f'  {eval_classes[c]:12s} AP={ap:.4f}')
        results[key] = mAP
        if use_wandb:
            log_metrics({f'pretrained/{key}_mAP50': mAP})

    print('\n=== 参照評価まとめ (mAP@0.5) ===')
    for key, mAP in results.items():
        print(f'  {key:12s}: {mAP:.4f}')

    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
