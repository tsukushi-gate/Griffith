"""DLGN を EGGROLL（進化戦略, Sarkar 2026）で訓練するスクリプト。

論文 "Evolution Strategies at the Hyperscale" の手法を DLGN（LutLayer + GroupSum）に
適用する。backprop を一切使わず、forward のみで LutLayer の logits を最適化する。

使用例:
    # スモークテスト（小規模で動作確認）
    uv run python experiments/train_eggroll_dlgn.py --dataset mnist20x20 \
        -k 1000 -l 2 --rank 1 --sigma 0.05 --alpha 0.05 \
        --n-workers 64 --total-steps 100

    # 本実験（backprop ベースラインも並列実行）
    uv run python experiments/train_eggroll_dlgn.py --dataset mnist20x20 \
        -k 8000 -l 4 --n-inputs 2 --tau 30 \
        --rank 1 --sigma 0.01 --alpha 0.01 --n-workers 256 \
        --total-steps 5000 --batch-size 128 --eval-interval 50 \
        --run-baseline

    # rank ablation
    uv run python experiments/train_eggroll_dlgn.py --rank 4 --n-workers 256 ...
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
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import setup_checkpoint_dir
from env_utils import load_dotenv

# train.py のデータローダ・モデル構築・評価関数を再利用
from train import DATASET_INFO, build_model, evaluate, load_dataset, setup_logger
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import LutLayer
from griffith_logic.eggroll import EGGROLLConfig, EGGROLLOptimizer

load_dotenv()


def find_lut_layers(model: torch.nn.Module) -> list[LutLayer]:
    """モデル内の全 LutLayer を順序保持で返す"""
    return [m for m in model.modules() if isinstance(m, LutLayer)]


def make_fitness_fn(model, x, y, fitness_mode):
    """現在の model.parameters() で fitness（=-loss）を返すクロージャを作る。

    fitness_mode:
        'soft' — model.train() で連続 softmax 出力（勾配信号が滑らか）
        'hard' — model.eval() で離散化（推論と同条件、論文の non-differentiable 主張に準拠）
    """
    def fitness():
        if fitness_mode == 'hard':
            model.eval()
            with torch.no_grad():
                x_in = x.round()
                output = model(x_in)
        else:
            model.train()
            with torch.no_grad():
                output = model(x)
        loss = F.cross_entropy(output, y)
        return -float(loss)
    return fitness


def run_eggroll_loop(args, model, train_loader, test_loader, device, log, run_dir):
    """EGGROLL 訓練ループ"""
    lut_layers = find_lut_layers(model)
    cfg = EGGROLLConfig(
        rank=args.rank,
        sigma=args.sigma,
        alpha=args.alpha,
        n_workers=args.n_workers,
        base_seed=args.seed,
        score_function=args.score_function,
        antithetic=args.antithetic,
    )
    optim = EGGROLLOptimizer([layer.weights for layer in lut_layers], cfg)

    print(f'EGGROLL: rank={args.rank}, σ={args.sigma}, α={args.alpha}, '
          f'N_workers={args.n_workers}, score={args.score_function}, '
          f'fitness_mode={args.fitness_mode}, antithetic={args.antithetic}')
    print(f'対象 LutLayer 数: {len(lut_layers)}')
    print(f'各層 weight shape: {[tuple(l.weights.shape) for l in lut_layers]}')

    best_test_acc = 0.0
    train_iter = iter(train_loader)
    train_start = time.time()

    for t in range(args.total_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(torch.float32).to(device)
        y = y.to(device)

        fitness_fn = make_fitness_fn(model, x, y, args.fitness_mode)
        stats = optim.step(fitness_fn, t)

        if (t + 1) % args.eval_interval == 0 or t + 1 == args.total_steps:
            test_acc_hard = evaluate(model, test_loader, device, mode=False)
            test_acc_soft = evaluate(model, test_loader, device, mode=True)
            is_best = test_acc_hard > best_test_acc
            if is_best:
                best_test_acc = test_acc_hard

            elapsed = time.time() - train_start
            metrics = {
                **stats,
                'step': t + 1,
                'test_acc_hard': test_acc_hard,
                'test_acc_soft': test_acc_soft,
                'best_test_acc': best_test_acc,
                'time/elapsed_sec': elapsed,
            }
            log_metrics(metrics)

            print(
                f'[step {t+1:>5d}/{args.total_steps}] '
                f'fitness_mean={stats["fitness/mean"]:.4f}  '
                f'fitness_max={stats["fitness/max"]:.4f}  '
                f'test_acc(hard)={test_acc_hard:.4f}  '
                f'test_acc(soft)={test_acc_soft:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
            )
            log(f'STEP {t+1} fitness_mean={stats["fitness/mean"]:.4f} '
                f'test_acc_hard={test_acc_hard:.4f} test_acc_soft={test_acc_soft:.4f}')

            ckpt_path = os.path.join(run_dir, 'latest.pt')
            torch.save({
                'step': t + 1,
                'model_state_dict': model.state_dict(),
                'best_test_acc': best_test_acc,
                'eggroll_config': vars(cfg),
            }, ckpt_path)

    return best_test_acc


def run_backprop_baseline(args, train_loader, test_loader, device, log):
    """同条件のモデルを Adam + CE で訓練する比較ベースライン"""
    print('\n' + '=' * 60)
    print('Backprop ベースライン訓練開始')
    print('=' * 60)
    model = build_model(args)
    optim = torch.optim.Adam(model.parameters(), lr=args.baseline_lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_test_acc = 0.0
    train_iter = iter(train_loader)

    for t in range(args.total_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(torch.float32).to(device)
        y = y.to(device)

        model.train()
        output = model(x)
        loss = loss_fn(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (t + 1) % args.eval_interval == 0 or t + 1 == args.total_steps:
            test_acc_hard = evaluate(model, test_loader, device, mode=False)
            test_acc_soft = evaluate(model, test_loader, device, mode=True)
            is_best = test_acc_hard > best_test_acc
            if is_best:
                best_test_acc = test_acc_hard
            log_metrics({
                'baseline/step': t + 1,
                'baseline/loss': loss.item(),
                'baseline/test_acc_hard': test_acc_hard,
                'baseline/test_acc_soft': test_acc_soft,
                'baseline/best_test_acc': best_test_acc,
            })
            print(
                f'[backprop step {t+1:>5d}] '
                f'loss={loss.item():.4f}  '
                f'test_acc(hard)={test_acc_hard:.4f}  '
                f'test_acc(soft)={test_acc_soft:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
            )
            log(f'BASELINE STEP {t+1} loss={loss.item():.4f} '
                f'test_acc_hard={test_acc_hard:.4f}')

    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description='EGGROLL で DLGN を訓練')

    # データ・モデル（train.py と同形式）
    parser.add_argument('--dataset', type=str, default='mnist20x20',
                        choices=list(DATASET_INFO.keys()))
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('-k', '--num-neurons', type=int, default=8000)
    parser.add_argument('-l', '--num-layers', type=int, default=4)
    parser.add_argument('-t', '--tau', type=float, default=30.0)
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'])
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'])
    parser.add_argument('--tau-warp', type=float, default=1.0)
    parser.add_argument('--no-gumbel', action='store_true')
    parser.add_argument('--grad-factor', type=float, default=1.0)

    # EGGROLL ハイパーパラメータ
    parser.add_argument('--rank', type=int, default=1,
                        help='low-rank 摂動の rank r（論文の主要主張は r=1 でも収束）')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='摂動スケール σ（論文: σ_d = o(d^{-1/2}) 以下が安定）')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='更新ステップサイズ α')
    parser.add_argument('--n-workers', type=int, default=256,
                        help='集団サイズ N（論文は最大 2^20）')
    parser.add_argument('--score-function', type=str, default='gaussian',
                        choices=['gaussian', 'centered_rank'],
                        help='gaussian: ŝ(E)=-E (論文 Eq.5), centered_rank: Salimans 2017')
    parser.add_argument('--fitness-mode', type=str, default='soft',
                        choices=['soft', 'hard'],
                        help='soft: 連続 softmax 出力, hard: 離散化（論文の non-diff 主張準拠）')
    parser.add_argument('--antithetic', action='store_true',
                        help='Mirror sampling (Salimans 2017): 各 worker で ±E の両評価。'
                             'softmax 非線形で信号が小さい DLGN では事実上必須')

    # ループ制御
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('-bs', '--batch-size', type=int, default=128)

    # 比較ベースライン
    parser.add_argument('--run-baseline', action='store_true',
                        help='backprop SGD ベースラインを同条件で並列実行')
    parser.add_argument('--baseline-lr', type=float, default=0.01)

    # 共通
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データ
    train_loader, test_loader = load_dataset(args)
    print(f'データセット: {args.dataset}')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル')

    # モデル
    model = build_model(args)

    # ログ
    log, log_file = setup_logger('train_eggroll_dlgn', args.dataset, args.log_dir)
    log(f'START EGGROLL rank={args.rank} sigma={args.sigma} alpha={args.alpha} '
        f'n_workers={args.n_workers} score={args.score_function} '
        f'fitness_mode={args.fitness_mode} total_steps={args.total_steps} '
        f'seed={args.seed} device={args.device}')

    # チェックポイント
    run_dir, _, _ = setup_checkpoint_dir(
        f'eggroll_dlgn_{args.dataset}', base_dir=args.checkpoint_dir,
    )
    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'eggroll_{args.dataset}_k{args.num_neurons}_l{args.num_layers}'
                f'_n{args.n_inputs}_r{args.rank}_N{args.n_workers}')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['eggroll', 'lut', args.dataset, f'n{args.n_inputs}', f'rank{args.rank}'],
        group='eggroll',
    )

    # EGGROLL 本体
    train_start = time.time()
    best_eggroll = run_eggroll_loop(
        args, model, train_loader, test_loader, args.device, log, run_dir,
    )
    eggroll_elapsed = time.time() - train_start
    print(f'\nEGGROLL 完了: best test_acc(hard)={best_eggroll:.4f}  '
          f'時間={timedelta(seconds=int(eggroll_elapsed))}')

    # backprop ベースライン
    if args.run_baseline:
        baseline_start = time.time()
        best_baseline = run_backprop_baseline(
            args, train_loader, test_loader, args.device, log,
        )
        baseline_elapsed = time.time() - baseline_start
        print('\n比較結果:')
        print(f'  EGGROLL  best={best_eggroll:.4f}  '
              f'時間={timedelta(seconds=int(eggroll_elapsed))}')
        print(f'  backprop best={best_baseline:.4f}  '
              f'時間={timedelta(seconds=int(baseline_elapsed))}')
        log(f'COMPARISON eggroll_best={best_eggroll:.4f} '
            f'baseline_best={best_baseline:.4f} '
            f'eggroll_sec={eggroll_elapsed:.1f} '
            f'baseline_sec={baseline_elapsed:.1f}')
        if use_wandb:
            log_metrics({
                'compare/eggroll_best': best_eggroll,
                'compare/baseline_best': best_baseline,
                'compare/eggroll_sec': eggroll_elapsed,
                'compare/baseline_sec': baseline_elapsed,
            })

    log(f'DONE eggroll_best_test_acc={best_eggroll:.4f}')
    log_file.close()
    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
