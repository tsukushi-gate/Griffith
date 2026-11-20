"""
AP法（Akiho Propagation）によるLUTネットワーク訓練スクリプト。

autogradを使わない学習アルゴリズムでMNIST等の分類タスクを解く。

使用例:
    # 基本
    uv run python experiments/train_ap.py --dataset mnist -k 16000 -l 4 --lr 0.1 -e 30

    # update-mode 2（最大basis入力のみ更新）
    uv run python experiments/train_ap.py --dataset mnist -k 16000 -l 4 --lr 0.1 -e 30 --update-mode 2

    # 小規模テスト
    uv run python experiments/train_ap.py --dataset mnist -k 1000 -l 2 --lr 0.1 -e 5
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

from ap_metrics import ConvergenceTracker, WallClock
from ap_trainer import APTrainer
from checkpoint_utils import setup_checkpoint_dir
from env_utils import load_dotenv
from train import DATASET_INFO, build_model, load_dataset, setup_logger
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
    """AP法用チェックポイント保存（optimizerなし）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'step': step,
        'best_metric': best_metric,
        'metadata': metadata,
    }
    torch.save(data, path)


def main():
    parser = argparse.ArgumentParser(description='AP法 LUTネットワーク訓練')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=list(DATASET_INFO.keys()),
                        help='データセット')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='LUTの入力数 (default: 2)')
    parser.add_argument('-k', '--num-neurons', type=int, default=16000,
                        help='層あたりニューロン数 (default: 16000)')
    parser.add_argument('-l', '--num-layers', type=int, default=4,
                        help='LUT層数 (default: 4)')
    parser.add_argument('-t', '--tau', type=float, default=10.0,
                        help='GroupSum温度 (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='学習率 (default: 0.1)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='エポック数 (default: 50)')
    parser.add_argument('-ef', '--eval-freq', type=int, default=1,
                        help='評価頻度: Nエポックごと (default: 1)')
    parser.add_argument('--update-mode', type=int, default=1, choices=[1, 2],
                        help='重み更新モード: 1=全重み, 2=最大basisのみ (default: 1)')
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help='接続方式 (default: random)')
    parser.add_argument('--wiring', type=str, default='fixed',
                        choices=['fixed', 'learnable'],
                        help='入力接続の方式 (default: fixed。'
                             'learnableでDWN方式の配線学習を有効化)')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数（AP法では未使用、モデル構築互換用）')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先ディレクトリ (default: logs)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')

    args = parser.parse_args()

    # AP法はindependentパラメータ化を強制
    args.parameterization = 'independent'

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データ読み込み
    train_loader, test_loader = load_dataset(args)
    print(f'データセット: {args.dataset}')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, テスト: {len(test_loader.dataset):,}サンプル')

    # モデル構築
    model = build_model(args)

    # APTrainer
    trainer = APTrainer(
        model,
        lr=args.lr,
        update_mode=args.update_mode,
    )
    print(f'AP法: update_mode={args.update_mode}, lr={args.lr}')

    # ログ
    log, log_file = setup_logger('train_ap', args.dataset, args.log_dir)
    log(f'START dataset={args.dataset} n_inputs={args.n_inputs} '
        f'k={args.num_neurons} l={args.num_layers} tau={args.tau} '
        f'lr={args.lr} bs={args.batch_size} epochs={args.epochs} '
        f'update_mode={args.update_mode} seed={args.seed} device={args.device}')

    # チェックポイント
    ckpt_prefix = f'train_ap_{args.dataset}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args), 'trainer': 'ap'}
    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'ap_{args.dataset}_k{args.num_neurons}_l{args.num_layers}'
                f'_n{args.n_inputs}_m{args.update_mode}_e{args.epochs}')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={**vars(args), 'trainer': 'ap'},
        model=model,
        device=args.device,
        tags=['ap', 'lut', args.dataset, f'n{args.n_inputs}'],
        group='ap',
    )

    # 訓練ループ
    batches_per_epoch = len(train_loader)
    best_test_acc = 0.0

    # AP法(BP-free)学習の収束・停滞・時間計測(hard test acc基準)。
    # ACC_HARD_THRESHOLD=1.0は論理タスク(xor/xor3/maj3/rnd4)の真理値表完全一致
    # (tests/l4_learningのhard_acc=100%収束基準)に合わせた既定値。mnist等では
    # 到達しないため steps_to_threshold は None(-1)のまま記録される。
    acc_hard_tracker = ConvergenceTracker(mode='max')
    wall_clock = WallClock()
    ACC_HARD_THRESHOLD = 1.0

    print(f'\n訓練開始: {args.epochs}エポック ({batches_per_epoch}バッチ/エポック)')
    print('=' * 60)

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(
            train_loader,
            desc=f'epoch {epoch}/{args.epochs}',
            leave=False,
        )
        for x, y in pbar:
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)

            metrics = trainer.train_step(x, y)
            epoch_loss += metrics['loss']
            epoch_correct += int(metrics['accuracy'] * y.shape[0])
            epoch_total += y.shape[0]

            elapsed_t = time.time() - train_start
            progress = (epoch - 1 + (pbar.n + 1) / batches_per_epoch) / args.epochs
            remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(
                loss=f'{metrics["loss"]:.4f}',
                acc=f'{epoch_correct / epoch_total:.4f}',
                eta=eta_str,
            )

        avg_loss = epoch_loss / batches_per_epoch
        train_acc = epoch_correct / epoch_total

        # 評価
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            # 連続値での評価
            test_acc_soft = trainer.evaluate(test_loader, args.device, binarize=False)
            # binarize後の評価
            test_acc_hard = trainer.evaluate(test_loader, args.device, binarize=True)

            is_best = test_acc_hard > best_test_acc
            if is_best:
                best_test_acc = test_acc_hard

            step_now_metrics = epoch * batches_per_epoch
            acc_hard_tracker.record(step_now_metrics, test_acc_hard)
            steps_to_acc1 = acc_hard_tracker.steps_to_threshold(ACC_HARD_THRESHOLD, mode='max')
            plateau_step = acc_hard_tracker.plateau_step()
            wall_clock_sec = wall_clock.elapsed_sec()

            # チェックポイント
            step_now = epoch * batches_per_epoch
            save_ap_checkpoint(latest_path, model,
                               step_now, best_test_acc, ckpt_metadata)
            if is_best:
                save_ap_checkpoint(best_path, model,
                                   step_now, best_test_acc, ckpt_metadata)

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
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
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
                    'metrics/steps_to_acc1.0': steps_to_acc1 if steps_to_acc1 is not None else -1,
                    'metrics/plateau_step': plateau_step if plateau_step is not None else -1,
                    'metrics/wall_clock_sec': wall_clock_sec,
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
