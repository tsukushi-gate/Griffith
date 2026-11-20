"""
CAGE実験: Hard-ST vs Soft-Mix 収束速度比較

CAGE論文 (Kim, 2026) の再現実験。
MNIST-Binaryで Soft-Mix / Hard-ST / Hard-ST+CAGE を比較する。

使用例:
    # Soft-Mix（ベースライン）
    uv run python experiments/train_cage_comparison.py --selection soft --seed 0

    # Hard-ST（固定τ_b=1.0）
    uv run python experiments/train_cage_comparison.py --selection hard_st --tau-b 1.0 --seed 0

    # Hard-ST + CAGE（適応τ_b）
    uv run python experiments/train_cage_comparison.py --selection hard_st --cage --seed 0

    # 3シードまとめて実行
    for s in 0 1 2; do
        uv run python experiments/train_cage_comparison.py --selection soft --seed $s
        uv run python experiments/train_cage_comparison.py --selection hard_st --tau-b 1.0 --seed $s
        uv run python experiments/train_cage_comparison.py --selection hard_st --cage --seed $s
    done
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
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cage_scheduler import CAGEScheduler
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import GroupSum, LutLayer

load_dotenv()


# ============================================================
# データセット
# ============================================================

def load_dataset(args):
    """MNIST（28×28, 784次元）を読み込む。"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data-mnist')
    train_ds = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


# ============================================================
# モデル構築
# ============================================================

IN_DIM = 784  # 28×28
NUM_CLASSES = 10

def build_model(args):
    k = args.num_neurons
    l = args.num_layers
    assert k % NUM_CLASSES == 0, f'ニューロン数({k})はクラス数({NUM_CLASSES})で割り切れる必要があります'

    llkw = dict(
        n_inputs=2,
        device=args.device,
        connections='random',
        parameterization='softmax',
        selection=args.selection,
    )

    layers = [torch.nn.Flatten()]
    layers.append(LutLayer(in_dim=IN_DIM, out_dim=k, **llkw))
    for _ in range(l - 1):
        layers.append(LutLayer(in_dim=k, out_dim=k, **llkw))
    layers.append(GroupSum(NUM_CLASSES, tau=args.tau))

    model = torch.nn.Sequential(*layers).to(args.device)

    total_neurons = sum(m.num_neurons for m in model.modules() if isinstance(m, LutLayer))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'モデル: {l}層 × {k}ニューロン, N=2, selection={args.selection}')
    print(f'総ゲート数: {total_neurons:,}, 総パラメータ数: {total_params:,}')

    return model


# ============================================================
# 訓練・評価
# ============================================================

def train_step(model, x, y, loss_fn, optimizer):
    output = model(x)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, loader, device, mode=False):
    """
    Args:
        mode: True=soft（softmax重み）, False=hard（離散化）
    """
    orig_mode = model.training
    model.train(mode=mode)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            if not mode:
                x = x.round()
            output = model(x)
            pred = output.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    model.train(mode=orig_mode)
    return correct / total if total > 0 else 0.0


def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ"""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                return


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='CAGE比較実験')

    # アーキテクチャ（論文デフォルト）
    parser.add_argument('-k', '--num-neurons', type=int, default=64000)
    parser.add_argument('-l', '--num-layers', type=int, default=6)
    parser.add_argument('-t', '--tau', type=float, default=10.0)

    # 訓練
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-bs', '--batch-size', type=int, default=512)
    parser.add_argument('-ni', '--num-iterations', type=int, default=50000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)

    # CAGE関連
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式: soft=Soft-Mix, hard_st=Hard-ST')
    parser.add_argument('--cage', action='store_true',
                        help='CAGE適応温度スケジューラを有効化')
    parser.add_argument('--tau-b', type=float, default=1.0,
                        help='Hard-STの逆伝播温度（CAGE無効時に使用）')
    parser.add_argument('--cage-tau-max', type=float, default=3.0)
    parser.add_argument('--cage-tau-min', type=float, default=0.5)
    parser.add_argument('--cage-beta', type=float, default=0.99)

    # ログ
    parser.add_argument('--log-dir', type=str, default='logs')

    args = parser.parse_args()

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    # CAGEはhard_stでのみ有効
    if args.cage and args.selection != 'hard_st':
        parser.error('--cage は --selection hard_st と組み合わせてください')

    # 手法名の決定
    if args.selection == 'soft':
        method_name = 'SoftMix'
    elif args.cage:
        method_name = 'HardST+CAGE'
    else:
        method_name = f'HardST(tau_b={args.tau_b})'

    print(f'=== CAGE比較実験: {method_name} ===')
    print(f'デバイス: {args.device}')

    # シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データ
    train_loader, test_loader = load_dataset(args)
    print(f'MNIST: 訓練={len(train_loader.dataset):,}, テスト={len(test_loader.dataset):,}')

    # モデル
    model = build_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Hard-STの初期τ_b設定
    if args.selection == 'hard_st' and not args.cage:
        for m in model.modules():
            if isinstance(m, LutLayer):
                m.tau_b = args.tau_b

    # CAGEスケジューラ
    cage_scheduler = None
    if args.cage:
        cage_scheduler = CAGEScheduler(
            model,
            tau_max=args.cage_tau_max,
            tau_min=args.cage_tau_min,
            beta=args.cage_beta,
        )
        print(f'CAGE: τ_max={args.cage_tau_max}, τ_min={args.cage_tau_min}, β={args.cage_beta}')

    # ログ
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    method_tag = method_name.replace('(', '_').replace(')', '').replace('=', '').replace('+', '_')
    log_path = os.path.join(args.log_dir, f'cage_{method_tag}_s{args.seed}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    def log(msg):
        log_file.write(f'{datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')
    print(f'ログ: {log_path}')

    log(f'START method={method_name} k={args.num_neurons} l={args.num_layers} '
        f'lr={args.lr} bs={args.batch_size} ni={args.num_iterations} seed={args.seed}')

    # wandb
    run_name = f'cage_{method_tag}_s{args.seed}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['cage', method_name, f'seed{args.seed}'],
        group='cage-comparison',
    )

    # 訓練ループ
    num_iterations = args.num_iterations
    eval_freq = args.eval_freq
    batches_per_epoch = len(train_loader)

    print(f'\n訓練開始: {num_iterations:,}イテレーション (評価={eval_freq}イテレーションごと)')
    print('=' * 70)

    train_start = time.time()
    best_test_acc = 0.0
    converged_iter = None  # 97%到達イテレーション

    pbar = tqdm(
        enumerate(load_n(train_loader, num_iterations)),
        desc=f'{method_name}',
        total=num_iterations,
    )
    for i, (x, y) in pbar:
        x = x.to(torch.float32).to(args.device)
        y = y.to(args.device)

        # MNIST-Binary: 訓練時も二値化入力（論文の条件）
        x_binary = x.round()

        loss = train_step(model, x_binary, y, loss_fn, optimizer)

        # CAGEスケジューラ更新
        if cage_scheduler is not None:
            cage_scheduler.step()

        tau_b_now = cage_scheduler.tau_b if cage_scheduler else args.tau_b
        pbar.set_postfix(loss=f'{loss:.4f}', tau_b=f'{tau_b_now:.2f}')

        # 評価
        if (i + 1) % eval_freq == 0:
            test_acc_hard = evaluate(model, test_loader, args.device, mode=False)
            test_acc_soft = evaluate(model, test_loader, args.device, mode=True)
            selection_gap = test_acc_soft - test_acc_hard

            is_best = test_acc_hard > best_test_acc
            if is_best:
                best_test_acc = test_acc_hard

            if converged_iter is None and test_acc_hard >= 0.97:
                converged_iter = i + 1

            elapsed = time.time() - train_start
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remaining = elapsed / (i + 1) * (num_iterations - (i + 1))
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')

            confidence_str = ''
            if cage_scheduler:
                confidence_str = f'  conf={cage_scheduler.ema_confidence:.4f}'

            print(
                f'\n[iter {i+1:>6,}] '
                f'loss={loss:.4f}  '
                f'hard={test_acc_hard:.4f}  '
                f'soft={test_acc_soft:.4f}  '
                f'gap={selection_gap:+.4f}  '
                f'τ_b={tau_b_now:.3f}'
                f'{confidence_str}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}, ETA {eta_str}]'
            )

            log(f'ITER {i+1} loss={loss:.4f} hard={test_acc_hard:.4f} '
                f'soft={test_acc_soft:.4f} gap={selection_gap:+.4f} '
                f'tau_b={tau_b_now:.3f} best={str(is_best).lower()}')

            if use_wandb:
                metrics = {
                    'iteration': i + 1,
                    'loss': loss,
                    'test_acc_hard': test_acc_hard,
                    'test_acc_soft': test_acc_soft,
                    'selection_gap': selection_gap,
                    'tau_b': tau_b_now,
                    'best_test_acc': best_test_acc,
                }
                if cage_scheduler:
                    metrics['confidence'] = cage_scheduler.ema_confidence
                log_metrics(metrics)

    total_time = time.time() - train_start
    total_str = str(timedelta(seconds=int(total_time)))

    print(f'\n{"=" * 70}')
    print(f'実験完了: {method_name}')
    print(f'  最終精度(hard): {best_test_acc:.4f}')
    print(f'  97%到達: {converged_iter if converged_iter else "未到達"}イテレーション')
    print(f'  総訓練時間: {total_str}')

    log(f'DONE method={method_name} best_acc={best_test_acc:.4f} '
        f'converged_iter={converged_iter} total_time={total_str}')
    log_file.close()
    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
