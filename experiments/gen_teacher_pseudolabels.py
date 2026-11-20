"""
teacher（torchvision COCO事前学習検出器）による少クラスVOC訓練画像の
pseudo-label生成（W8: LUT KD腕用）。

CenterNet-LUT(Conv-DLGN) と COCO事前学習検出器(RetinaNet/FasterRCNN)は
アーキテクチャが根本的に異なる(box regressor vs dense heatmap)ため、
ロジットレベルのKD(CenterNetKDLoss)は適用できない。代わりに teacher の
確信度が高い予測boxをそのまま pseudo GT として CenterNetターゲットへ
エンコードし、student をこの pseudo-target で通常のCenterNetLoss学習させる
（pseudo-label方式）。

出力ファイル形式（torch.save、data/detection_fewclass.py の
FewClassVOCDetection(pseudo_label_path=...) が読み込む）:
    {
        'img_size': int,
        'classes': list[str],   # FewClassVOCDetection(classes=...) と同順
        'model': str,           # teacher モデルキー
        'items': {idx: {'boxes': FloatTensor[N,4] xyxy@img_size座標,
                         'labels': LongTensor[N]}},
    }

使用例:
    uv run python experiments/gen_teacher_pseudolabels.py \
        --model fasterrcnn_r50 --classes person,car,dog --img-size 128
"""

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from data.detection_fewclass import DEFAULT_FEWCLASS, FewClassVOCDetection
from env_utils import load_dotenv
from eval_pretrained_detector_voc import MODEL_REGISTRY, VOC_TO_COCO_NAME

load_dotenv()


def build_coco_to_local(coco_categories: list[str], classes: list[str]) -> dict[int, int]:
    """COCOラベルid → classes内ローカルid(0..len(classes)-1)の写像を構築する。"""
    coco_name_to_id = {name: i for i, name in enumerate(coco_categories)}
    mapping = {}
    for local_idx, voc_name in enumerate(classes):
        coco_name = VOC_TO_COCO_NAME[voc_name]
        if coco_name not in coco_name_to_id:
            raise KeyError(f'COCOカテゴリに {coco_name} が見つかりません')
        mapping[coco_name_to_id[coco_name]] = local_idx
    return mapping


@torch.no_grad()
def generate_pseudo_labels(
    model_key: str,
    classes: list[str],
    img_size: int,
    device: str,
    score_threshold: float,
    limit: int | None,
    download: bool,
) -> dict[int, dict]:
    """teacher を fewclass 訓練画像に推論し、pseudo boxes/labels を生成する。

    推論は原寸画像で行い、box を img_size 座標系へスケールする（
    FewClassVOCDetection._encode と同一のスケーリング規約）。
    """
    build_fn, weights_fn = MODEL_REGISTRY[model_key]
    weights = weights_fn()
    model = build_fn(weights=weights, box_score_thresh=score_threshold)
    model.eval().to(device)

    coco_categories = weights.meta['categories']
    coco_to_local = build_coco_to_local(coco_categories, classes)

    # 訓練split・同一classesでインデックス列を構築（FewClassVOCDetectionの
    # .fewclass_index_*.json キャッシュを訓練スクリプトと共有する）
    ds = FewClassVOCDetection(
        train=True, img_size=img_size, classes=tuple(classes),
        use_rgb=True, download=download, limit=limit,
    )
    to_tensor = torchvision.transforms.ToTensor()

    items = {}
    n_boxes_total = 0
    n_empty = 0
    for idx in tqdm(range(len(ds)), desc=f'pseudo-label({model_key})'):
        bi, li = ds._pairs[idx]
        base = ds.bases[bi]
        img, _ = base[li]
        img = img.convert('RGB')
        orig_w, orig_h = img.size
        sx = img_size / orig_w
        sy = img_size / orig_h

        x = to_tensor(img).to(device)
        out = model([x])[0]
        p_boxes = out['boxes'].cpu()
        p_scores = out['scores'].cpu()
        p_labels = out['labels'].cpu()

        boxes_out, labels_out = [], []
        for b in range(p_boxes.shape[0]):
            if p_scores[b].item() < score_threshold:
                continue
            coco_id = int(p_labels[b].item())
            if coco_id not in coco_to_local:
                continue
            x1, y1, x2, y2 = p_boxes[b].tolist()
            boxes_out.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
            labels_out.append(coco_to_local[coco_id])

        if not boxes_out:
            n_empty += 1
        n_boxes_total += len(boxes_out)

        items[idx] = {
            'boxes': torch.tensor(boxes_out, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(labels_out, dtype=torch.long),
        }

    print(f'pseudo-label統計: {len(items)}枚, 総box数={n_boxes_total}, '
          f'box無し={n_empty}枚 ({100 * n_empty / max(1, len(items)):.1f}%)')
    return items


def main():
    parser = argparse.ArgumentParser(
        description='COCO事前学習検出器による少クラスVOC訓練画像のpseudo-label生成')
    parser.add_argument('--model', type=str, default='fasterrcnn_r50',
                        choices=list(MODEL_REGISTRY),
                        help='teacherモデル (default: fasterrcnn_r50、COCO box_map=46.7)')
    parser.add_argument('--classes', type=str, default=','.join(DEFAULT_FEWCLASS),
                        help=f'対象VOCクラス名(カンマ区切り) (default: {",".join(DEFAULT_FEWCLASS)})')
    parser.add_argument('--img-size', type=int, default=128,
                        help='pseudo box座標系のサイズ (default: 128、'
                             'FewClassVOCDetection(img_size=...)と一致必須)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='pseudo GT採用の最低確信度 (default: 0.5、'
                             '参照評価の0.05より厳しめ=確信boxのみをGT化)')
    parser.add_argument('--limit', type=int, default=0,
                        help='訓練画像数の上限(スモーク用、0=全件)')
    parser.add_argument('--download', action='store_true',
                        help='VOCデータを自動ダウンロード')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--out', type=str, default=None,
                        help='出力.ptパス (default: data-voc/pseudo_<model>_<classes>_<size>.pt)')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'デバイス: {args.device}')

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    limit = args.limit if args.limit > 0 else None

    t0 = time.time()
    print(f'=== teacher={args.model} classes={classes} img_size={args.img_size} '
          f'score_threshold={args.score_threshold} ===')
    items = generate_pseudo_labels(
        args.model, classes, args.img_size, args.device,
        args.score_threshold, limit, args.download,
    )
    dt = str(timedelta(seconds=int(time.time() - t0)))
    print(f'生成完了 (所要 {dt})')

    out_path = args.out or os.path.join(
        'data-voc', f'pseudo_{args.model}_{"_".join(classes)}_{args.img_size}.pt')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save({
        'img_size': args.img_size,
        'classes': classes,
        'model': args.model,
        'items': items,
    }, out_path)
    print(f'保存先: {out_path}')


if __name__ == '__main__':
    main()
