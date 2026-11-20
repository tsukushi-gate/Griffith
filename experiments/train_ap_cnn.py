"""
AP法によるCDD-LGN訓練スクリプト

使用例:
    # MNIST Small (k=16)
    uv run python experiments/train_ap_cnn.py --dataset mnist -k 16 --lr 0.1 -e 30

    # MNIST Medium (k=64)
    uv run python experiments/train_ap_cnn.py --dataset mnist -k 64 --lr 0.1 -e 50

    # 小規模テスト
    uv run python experiments/train_ap_cnn.py --dataset mnist -k 4 --lr 0.1 -e 3 --device cpu
"""

import argparse
import math
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

from ap_trainer_cnn import CNNAPTrainer
from checkpoint_utils import setup_checkpoint_dir
from env_utils import load_dotenv
from train_cnn import build_cifar10_model, build_mnist_model, load_dataset
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

load_dotenv()


def save_ap_checkpoint(path, model, step, best_metric, metadata):
    """AP法用チェックポイント保存"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'step': step,
        'best_metric': best_metric,
        'metadata': metadata,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='AP法 CDD-LGN 訓練')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar-10'])
    parser.add_argument('-k', type=int, default=16,
                        help='チャンネルスケール (default: 16)')
    parser.add_argument('--tree-depth', type=int, default=3)
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--n-thresholds', type=int, default=0,
                        help='入力二値化の閾値数 (0=連続値)')
    parser.add_argument('--input-channels-per-tree', type=int, default=2)
    parser.add_argument('--output-gate-factor', type=int, default=1)
    parser.add_argument('--wide-last-layer', action='store_true',
                        help='論文A.1.1(*)準拠: 最終Dense層もogf倍(640k·ogf→320k·ogf)')
    parser.add_argument('--residual-init-dense', action='store_true',
                        help='Dense LutLayer3層をResidual Init（ゲートAパススルー）で初期化'
                             '(build_mnist_model/build_cifar10_modelが要求する属性)')
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--update-mode', type=int, default=1, choices=[1, 2])
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1)
    parser.add_argument('--grad-factor', type=float, default=1.0)
    parser.add_argument('--connections', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()

    # AP法はindependentパラメータ化を強制
    args.parameterization = 'independent'
    # CDD-LGNのresidual_initはAP法では無効化
    args.residual_init = False

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

    # データ
    train_loader, test_loader, _ = load_dataset(args)
    print(f'データセット: {args.dataset}')

    # モデル
    if args.dataset == 'mnist':
        model, tau = build_mnist_model(args)
    else:
        model, tau = build_cifar10_model(args)
    print(model)

    # APTrainer
    trainer = CNNAPTrainer(model, lr=args.lr, update_mode=args.update_mode)
    print(f'AP法: update_mode={args.update_mode}, lr={args.lr}')

    # ログ
    from train_cnn import setup_logger
    log, log_file = setup_logger('train_ap_cnn', args.dataset, args.log_dir)
    log(f'START dataset={args.dataset} k={args.k} tree_depth={args.tree_depth} '
        f'lr={args.lr} update_mode={args.update_mode} bs={args.batch_size} '
        f'epochs={args.epochs} seed={args.seed} device={args.device}')

    # チェックポイント
    ckpt_prefix = f'ap_cnn_{args.dataset}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args), 'trainer': 'ap_cnn'}
    print(f'チェックポイント: {run_dir}/')

    # wandb
    run_name = f'ap_cnn_{args.dataset}_k{args.k}_d{args.tree_depth}_lr{args.lr}_e{args.epochs}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={**vars(args), 'trainer': 'ap_cnn'},
        model=model,
        device=args.device,
        tags=['ap', 'cnn', args.dataset],
        group='ap_cnn',
    )

    # 訓練
    batches_per_epoch = len(train_loader)
    best_test_acc = 0.0
    train_start = time.time()

    print(f'\n訓練開始: {args.epochs}エポック ({batches_per_epoch}バッチ/エポック)')
    print('=' * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}', leave=False)
        for x, y in pbar:
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)

            metrics = trainer.train_step(x, y)
            epoch_loss += metrics['loss']
            epoch_correct += int(metrics['accuracy'] * y.shape[0])
            epoch_total += y.shape[0]

            elapsed_t = time.time() - train_start
            progress = (epoch - 1 + (pbar.n + 1) / batches_per_epoch) / args.epochs
            remaining = elapsed_t * (1 - progress) / progress if progress > 0 else 0
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')
            pbar.set_postfix(
                loss=f'{metrics["loss"]:.4f}',
                acc=f'{epoch_correct / epoch_total:.4f}',
                eta=eta_str,
            )

        avg_loss = epoch_loss / batches_per_epoch
        train_acc = epoch_correct / epoch_total

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            test_acc_soft = trainer.evaluate(test_loader, args.device, binarize=False)
            test_acc_hard = trainer.evaluate(test_loader, args.device, binarize=True)

            is_best = test_acc_hard > best_test_acc
            if is_best:
                best_test_acc = test_acc_hard

            step_now = epoch * batches_per_epoch
            save_ap_checkpoint(latest_path, model, step_now, best_test_acc, ckpt_metadata)
            if is_best:
                save_ap_checkpoint(best_path, model, step_now, best_test_acc, ckpt_metadata)

            elapsed = time.time() - train_start
            sec_per_epoch = elapsed / epoch
            remaining = sec_per_epoch * (args.epochs - epoch)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')

            print(
                f'[epoch {epoch:>3d}/{args.epochs}] '
                f'loss={avg_loss:.4f}  '
                f'train_acc={train_acc:.4f}  '
                f'test_acc(soft)={test_acc_soft:.4f}  '
                f'test_acc(hard)={test_acc_hard:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, ETA {eta_str}]'
            )

            log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                f'train_acc={train_acc:.4f} '
                f'test_acc_soft={test_acc_soft:.4f} '
                f'test_acc_hard={test_acc_hard:.4f} '
                f'best={str(is_best).lower()} '
                f'elapsed={elapsed_str} eta={remain_str}')
            if use_wandb:
                log_metrics({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'train_acc': train_acc,
                    'test_acc_soft': test_acc_soft,
                    'test_acc_hard': test_acc_hard,
                    'best_test_acc': best_test_acc,
                })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度: {best_test_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_test_acc={best_test_acc:.4f} total_time={total_time}')
    log_file.close()
    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_test_acc': best_test_acc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
