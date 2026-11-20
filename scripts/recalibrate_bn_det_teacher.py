"""CenterNetCNN teacher のBN再較正(後半、データセット非依存に一般化)。

対象は `train_detection.py --arch cnn` で作った任意のデータセットの teacher
(face / multimnist / multimnist_color / fewclass / voc)。判明した
「保存ckptのBN統計がval集合で汚染される既存バグ」は全CNN teacherに波及するため、
**KD teacher として使う前・ONNX化する前に必ずこれを通すこと**。
旧名 `recalibrate_bn_face_teacher.py`(改名)。

(det/seg の Web 比較レポート §3)で確立した手法を流用する:
  1. 全 BatchNorm2d の running統計をreset + momentum=None(累積平均化)
  2. model.train()状態でWIDER FACE訓練データを~300バッチ、no_gradで順伝播
     (forward前に images.round() を適用。hard評価経路がround入力のため)
  3. model.eval()に戻し、experiments/train_detection.py::evaluate(mode=False)
     でval全体のhard mAPを較正前後で比較
  4. 改善していれば best_bncal.pt として保存(元ckptは上書きしない)。
     metadataに bn_recalibrated=True と before/after hard mAPを記録

使い方:
    uv run python scripts/recalibrate_bn_det_teacher.py \\
        checkpoints/mono_cnn_teacher/best.pt \\
        checkpoints/mono_cnn_teacher/best_bncal.pt \\
        --n-batches 300
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'experiments'))

from data.detection import collate_detection  # noqa: E402
from detection_loss import CenterNetLoss  # noqa: E402
from detection_model_cnn import CenterNetCNN  # noqa: E402
from train_detection import build_datasets, evaluate  # noqa: E402


def build_model_from_args(a: dict, in_ch: int, n_classes: int) -> CenterNetCNN:
    return CenterNetCNN(
        n_classes=n_classes, in_channels=in_ch,
        width=a.get('cnn_width', 64), head_ch=a.get('cnn_head_ch', 64),
        downsample=a.get('cnn_downsample', 'stride'),
    )


class _Args:
    """build_datasets()が期待するargparse.Namespace相当のダック型。"""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k.replace('-', '_'), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='入力best.pt(BN未較正)')
    ap.add_argument('out', help='出力先(改善時のみ書き出し)')
    ap.add_argument('--n-batches', type=int, default=300)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--expect-dataset', default=None,
                    help='指定するとckptのdatasetがこれと一致するか検査する(取り違え防止)')
    args = ap.parse_args()

    print(f'ckpt読み込み: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    a = ckpt.get('metadata', {}).get('args', {}) or {}
    assert a.get('arch') == 'cnn', f'arch != cnn: {a.get("arch")}'
    ds = a.get('dataset')
    if args.expect_dataset is not None:
        assert ds == args.expect_dataset, f'dataset != {args.expect_dataset}: {ds}'
    print(f'dataset={ds}  (BN再較正はデータセット非依存: 訓練データで統計を取り直すだけ)')

    ds_args = _Args(a)
    output_stride = 4
    train_ds, test_ds, n_classes = build_datasets(
        ds_args, output_stride, train=True, test=True,
    )
    print(f'train={len(train_ds)}枚 val={len(test_ds)}枚 n_classes={n_classes}')

    in_ch = test_ds[0][0].shape[0]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=a.get('batch_size', 32), shuffle=True,
        collate_fn=collate_detection, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=a.get('batch_size', 32), shuffle=False,
        collate_fn=collate_detection, drop_last=False,
    )

    model = build_model_from_args(a, in_ch, n_classes).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])

    loss_fn = CenterNetLoss(
        lambda_size=a.get('lambda_size', 0.1), lambda_off=a.get('lambda_off', 1.0),
    )

    print('較正前 hard mAP を実測中(val全体)...')
    map_before, loss_before, _, _ = evaluate(
        model, test_loader, args.device, loss_fn, mode=False,
        output_stride=output_stride, desc='before(hard)', n_vis=0,
        n_classes=n_classes,
    )
    print(f'  before: hard mAP={map_before:.4f}  loss={loss_before:.4f}')

    # --- BN再較正 ---
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    print(f'BatchNorm2d {len(bn_modules)}層をreset_running_stats + momentum=None')
    for m in bn_modules:
        m.reset_running_stats()
        m.momentum = None  # 累積平均(cumulative moving average)

    model.train()
    n_done = 0
    with torch.no_grad():
        for images, _targets in train_loader:
            images = images.to(torch.float32).to(args.device).round()
            model(images)
            n_done += 1
            if n_done >= args.n_batches:
                break
    print(f'  順伝播 {n_done}バッチ完了(round入力、no_grad)')

    model.eval()
    print('較正後 hard mAP を実測中(val全体)...')
    map_after, loss_after, _, _ = evaluate(
        model, test_loader, args.device, loss_fn, mode=False,
        output_stride=output_stride, desc='after(hard)', n_vis=0,
        n_classes=n_classes,
    )
    print(f'  after:  hard mAP={map_after:.4f}  loss={loss_after:.4f}')
    print(f'  差分: {map_after - map_before:+.4f}')

    if map_after > map_before:
        new_ckpt = copy.deepcopy(ckpt)
        new_ckpt['model_state_dict'] = model.state_dict()
        new_ckpt.setdefault('metadata', {})
        new_ckpt['metadata']['bn_recalibrated'] = True
        new_ckpt['metadata']['bn_recal_hard_map_before'] = float(map_before)
        new_ckpt['metadata']['bn_recal_hard_map_after'] = float(map_after)
        new_ckpt['metadata']['bn_recal_n_batches'] = int(n_done)
        # best_metricは元のsoft/hard混同値を保持しつつ、hard再較正値も別キーで残す
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_ckpt, args.out)
        print(f'改善したため保存: {args.out}')
    else:
        print('改善なし。元ckptのままONNX化してください(bncal保存はスキップ)。')


if __name__ == '__main__':
    main()
