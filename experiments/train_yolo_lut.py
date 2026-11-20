"""
LogicYOLO(anchor-free YOLO型物体検出の論理ゲート化)訓練スクリプト。

合成Multi-MNISTデータセット(CenterNet-LUTと完全に同一のデータ・評価)で
YOLOX型 decoupled head の論理ゲートモデルを訓練する。
将来の比較のため、データ生成(MultiMNISTDetection)と評価(compute_map)は
detection系と共通のものを流用している。

使用例:
    # スモーク(2分以内完走)
    uv run python experiments/train_yolo_lut.py --max-steps 30 --no-wandb --device cpu

    # 通常訓練(1物体、位置推定の動作確認)
    uv run python experiments/train_yolo_lut.py -k 64 --max-objects 1 -ni 10000

    # 3物体
    uv run python experiments/train_yolo_lut.py -k 64 --max-objects 3 -ni 20000
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import save_checkpoint
from data.detection import MultiMNISTDetection, collate_detection
from detection_utils import compute_map, visualize_detections
from env_utils import load_dotenv
from run_utils import RunDir
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_images,
    log_metrics,
    log_model_checkpoint,
    normalize_tags,
)
from wandb_utils import (
    finish as wandb_finish,
)
from yolo_lut_model import YoloLutLoss, YoloLutModel, decode_yolo

load_dotenv()

OUTPUT_STRIDE = 4


def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ。"""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i >= n:
                return


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, mode=False, desc=None,
             max_batches=None, n_vis=32):
    """mAP・損失を計算する。mode=False→hard(離散化), True→soft。"""
    orig = model.training
    model.train(mode=mode)

    all_pred, all_gt = [], []
    total_loss, n_batches = 0.0, 0
    vis_correct, vis_incorrect, vis_count = [], [], 0

    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    for images, targets in it:
        images = images.to(torch.float32).to(device)
        targets_device = {k: v.to(device) for k, v in targets.items()}
        if not mode:
            images = images.round()

        outputs = model(images)
        _, loss_dict = loss_fn(outputs, targets_device)
        total_loss += loss_dict['total_loss']
        n_batches += 1

        detections = decode_yolo(outputs, output_stride=OUTPUT_STRIDE,
                                 threshold=0.1)
        for det in detections:
            all_pred.append({
                'boxes': det['boxes'].cpu(),
                'scores': det['scores'].cpu(),
                'labels': det['labels'].cpu(),
            })
        B = images.shape[0]
        for b in range(B):
            mask_b = targets['mask'][b] > 0
            all_gt.append({
                'boxes': targets['gt_boxes'][b][mask_b].cpu(),
                'labels': targets['gt_labels'][b][mask_b].cpu(),
            })

        if vis_count < n_vis:
            for b in range(B):
                if vis_count >= n_vis:
                    break
                mask_b = targets['mask'][b] > 0
                vis_img, ok = visualize_detections(
                    image=images[b].cpu(),
                    gt_boxes=targets['gt_boxes'][b][mask_b].cpu(),
                    gt_labels=targets['gt_labels'][b][mask_b].cpu(),
                    pred_boxes=detections[b]['boxes'].cpu(),
                    pred_scores=detections[b]['scores'].cpu(),
                    pred_labels=detections[b]['labels'].cpu(),
                )
                (vis_correct if ok else vis_incorrect).append(vis_img)
                vis_count += 1

        if max_batches and n_batches >= max_batches:
            break

    mAP, _ = compute_map(all_pred, all_gt, n_classes=10, iou_threshold=0.5)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    model.train(mode=orig)
    return mAP, avg_loss, vis_correct, vis_incorrect


def main():
    parser = argparse.ArgumentParser(
        description='LogicYOLO(anchor-free YOLO) 物体検出 訓練スクリプト'
    )
    # モデル
    parser.add_argument('-k', type=int, default=64, help='幅パラメータ k')
    parser.add_argument('-d', '--tree-depth', type=int, default=3,
                        help='バックボーンのツリー深さ')
    parser.add_argument('--head-depth', type=int, default=3,
                        help='ヘッドのツリー深さ')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='ゲート入力数')
    parser.add_argument('--n-bins', type=int, default=16, help='回帰ビン数 K')
    parser.add_argument('--cls-group', type=int, default=8,
                        help='clsヘッドのグループサイズ')
    parser.add_argument('--obj-group', type=int, default=8,
                        help='objヘッドのグループサイズ')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp'])
    parser.add_argument('--grad-factor', type=float, default=1.0)
    parser.add_argument('--input-channels-per-tree', type=int, default=2)
    # データセット(detection系と共通仕様)
    parser.add_argument('--canvas-size', type=int, default=96)
    parser.add_argument('--min-objects', type=int, default=1)
    parser.add_argument('--max-objects', type=int, default=3)
    parser.add_argument('--n-thresholds', type=int, default=3)
    parser.add_argument('--num-train', type=int, default=10000)
    parser.add_argument('--num-test', type=int, default=2000)
    # 訓練
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.002)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大ステップ数(スモーク用、num-iterationsを上書き)')
    parser.add_argument('-ef', '--eval-freq', type=int, default=2000)
    parser.add_argument('--eval-max-batches', type=int, default=None,
                        help='評価バッチ数の上限(スモークで軽量化、Noneで全バッチ)')
    parser.add_argument('--lambda-cls', type=float, default=1.0)
    parser.add_argument('--lambda-reg', type=float, default=0.5)
    # その他
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化')
    add_experiment_args(parser)
    args = parser.parse_args()

    if args.max_steps is not None:
        args.num_iterations = args.max_steps
        # スモーク: CPUでも2分以内に完走するよう軽量化する
        # (評価は数バッチ、バッチサイズは16以下に制限)
        if args.eval_max_batches is None:
            args.eval_max_batches = 2
        args.batch_size = min(args.batch_size, 16)

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データセット(CenterNet-LUTと完全同一)
    ds_kw = dict(
        canvas_size=args.canvas_size, min_objects=args.min_objects,
        max_objects=args.max_objects, n_thresholds=args.n_thresholds,
        output_stride=OUTPUT_STRIDE,
    )
    train_dataset = MultiMNISTDetection(
        train=True, num_samples=args.num_train, seed=args.seed, **ds_kw)
    test_dataset = MultiMNISTDetection(
        train=False, num_samples=args.num_test, seed=args.seed + 1, **ds_kw)

    use_pin = args.device == 'cuda'
    num_workers = 0 if args.max_steps is not None else 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_detection, pin_memory=use_pin,
        drop_last=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_detection, pin_memory=use_pin,
        drop_last=False, num_workers=num_workers)

    in_ch = args.n_thresholds if args.n_thresholds > 0 else 1
    print('データセット: Multi-MNIST検出(detection系と共通)')
    print(f'キャンバス: {args.canvas_size}×{args.canvas_size}, '
          f'オブジェクト数: {args.min_objects}-{args.max_objects}, 閾値数: {args.n_thresholds}')

    model = YoloLutModel(
        n_classes=10, k=args.k, in_channels=in_ch,
        tree_depth=args.tree_depth, head_depth=args.head_depth,
        n_inputs=args.n_inputs, n_bins=args.n_bins,
        cls_group=args.cls_group, obj_group=args.obj_group,
        device=args.device, grad_factor=args.grad_factor,
        input_channels_per_tree=args.input_channels_per_tree,
        parameterization=args.parameterization,
    ).to(args.device)

    total_gates = model.num_gates
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nモデル: LogicYOLO k={args.k} tree_depth={args.tree_depth} '
          f'n_inputs={args.n_inputs} n_bins={args.n_bins}')
    print(f'総ロジックゲート数: {total_gates:,} / 総パラメータ数: {total_params:,}')

    loss_fn = YoloLutLoss(
        n_classes=10, n_bins=args.n_bins, output_stride=OUTPUT_STRIDE,
        lambda_cls=args.lambda_cls, lambda_reg=args.lambda_reg,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = f'yolo_k{args.k}_d{args.tree_depth}_K{args.n_bins}_obj{args.max_objects}'
    run_dir = RunDir(run_name, base_dir=args.output_dir)
    run_dir.save_config({
        **vars(args), 'total_gates': total_gates, 'total_params': total_params,
        'model_architecture': str(model),
    })
    print(f'出力先: {run_dir.root}/')

    ckpt_metadata = {'args': vars(args)}

    # wandb
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name, config_dict=vars(args), model=model,
            device=args.device,
            tags=['detection', 'yolo', 'anchor-free', f'n{args.n_inputs}']
            + normalize_tags(model='yolo_lut', dataset='synthetic', seed=args.seed),
            group='yolo_lut',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )
        if use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    run_dir.save_wandb_run_id(wandb.run.id)
            except ImportError:
                pass

    batches_per_epoch = max(1, len(train_loader))
    num_iterations = args.num_iterations
    print(f'\n訓練開始: {num_iterations:,}イテレーション(評価={args.eval_freq:,}ごと)')
    print('=' * 70)

    train_start = time.time()
    best_map = 0.0

    pbar = tqdm(enumerate(load_n(train_loader, num_iterations)),
                desc='train', total=num_iterations)
    for i, (images, targets) in pbar:
        images = images.to(torch.float32).to(args.device)
        targets_device = {k: v.to(args.device) for k, v in targets.items()}

        model.train()
        outputs = model(images)
        loss, loss_dict = loss_fn(outputs, targets_device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_t = time.time() - train_start
        sec_per_iter = elapsed_t / (i + 1)
        remaining = sec_per_iter * (num_iterations - (i + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')
        pbar.set_postfix(loss=f'{loss_dict["total_loss"]:.4f}',
                         epoch=f'{(i + 1) / batches_per_epoch:.1f}', eta=eta_str)

        if (i + 1) % args.eval_freq == 0 or (i + 1) == num_iterations:
            map_hard, loss_hard, correct_h, incorrect_h = evaluate(
                model, test_loader, args.device, loss_fn, mode=False,
                desc=f'eval(hard) {i+1}/{num_iterations}',
                max_batches=args.eval_max_batches)
            map_soft, loss_soft, correct_s, incorrect_s = evaluate(
                model, test_loader, args.device, loss_fn, mode=True,
                desc=f'eval(soft) {i+1}/{num_iterations}',
                max_batches=args.eval_max_batches)

            is_best = map_hard > best_map
            if is_best:
                best_map = map_hard

            save_checkpoint(run_dir.latest_path, model, optimizer, None,
                            i + 1, best_map, ckpt_metadata)
            if is_best:
                save_checkpoint(run_dir.best_path, model, optimizer, None,
                                i + 1, best_map, ckpt_metadata)
                if use_wandb:
                    log_model_checkpoint(
                        run_dir.best_path, 'yolo-lut-best',
                        metadata={'iteration': i + 1, 'mAP_hard': map_hard})

            metrics = {
                'iteration': i + 1,
                'epoch': (i + 1) / batches_per_epoch,
                **loss_dict,
                'mAP_hard': map_hard, 'mAP_soft': map_soft, 'best_mAP': best_map,
                'eval_loss_hard': loss_hard, 'eval_loss_soft': loss_soft,
            }
            run_dir.log_metrics(metrics)

            # 可視化: ローカル保存 + wandb必須ログ
            if correct_h:
                run_dir.save_visualizations(correct_h, 'hard_correct', i + 1)
            if incorrect_h:
                run_dir.save_visualizations(incorrect_h, 'hard_incorrect', i + 1)
            if use_wandb:
                if incorrect_h:
                    log_images('yolo/hard_incorrect', incorrect_h)
                if correct_h:
                    log_images('yolo/hard_correct', correct_h)
                if incorrect_s:
                    log_images('yolo/soft_incorrect', incorrect_s)
                if correct_s:
                    log_images('yolo/soft_correct', correct_s)
                log_metrics(metrics)

            elapsed = time.time() - train_start
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')
            print(
                f'\n[iter {i+1:,}] loss={loss_dict["total_loss"]:.4f} '
                f'(obj={loss_dict["obj_loss"]:.4f} cls={loss_dict["cls_loss"]:.4f} '
                f'reg={loss_dict["reg_loss"]:.4f})  '
                f'mAP(hard)={map_hard:.4f}  mAP(soft)={map_soft:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
            )
            run_dir.log(
                f'ITER {i+1}/{num_iterations} loss={loss_dict["total_loss"]:.4f} '
                f'mAP_hard={map_hard:.4f} mAP_soft={map_soft:.4f} '
                f'best={str(is_best).lower()} elapsed={elapsed_str}')

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終mAP(hard): {best_map:.4f}  (総訓練時間: {total_time})')
    run_dir.log(f'DONE best_mAP={best_map:.4f} total_time={total_time}')
    run_dir.close()
    if use_wandb:
        log_checkpoint_artifact(
            run_dir.best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_map': best_map}, model=model)
        wandb_finish()


if __name__ == '__main__':
    main()
