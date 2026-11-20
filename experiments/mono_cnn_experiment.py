"""AP-CNN(CDD-LGN)の単調度 V / 極性 記録実験ドライバ。

背景:
    AP法(誤差ブロードキャスト型学習則)は CDD-LGN(畳み込みLUT)で 11.65% に崩壊する
    ことが知られる。「単調性破れ → AP失敗」仮説の検証のため、この崩壊 run の
    単調度 V・極性を層(モジュール)別に記録する。

崩壊設定(既定): k=16, tree-depth=3, n-inputs=2, dataset=mnist。
記録は record_model_monotonicity(ap_mode=True) を使う。ap_mode=True は
CNNAPTrainer が重みを sigmoid ではなく生の[0,1]確率(clamp)として扱うことに対応する
(ap_trainer_cnn._conv_forward_ap / _lut_forward が weights.data.clamp(0,1) を使用)。

使用例:
    uv run python experiments/mono_cnn_experiment.py --seed 0
    uv run python experiments/mono_cnn_experiment.py --seed 0 -e 1 \
        --max-train-batches 15 --interval 5 --no-wandb   # スモーク

Modal互換: 相対パス規約(sys.path.insert)・相対 out-dir を守る。
"""

import argparse
import itertools
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ap_interventions import MonotoneCNNAPTrainer
from ap_trainer_cnn import CNNAPTrainer
from env_utils import load_dotenv
from monotonicity import record_model_monotonicity
from train_cnn import build_cifar10_model, build_mnist_model, load_dataset
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_metrics,
)

load_dotenv()

# 崩壊判定の閾値(soft acc)。CDD-LGN の AP崩壊は ~11.65% で、ランダム(10%)近傍。
COLLAPSE_THRESHOLD = 1.2 / 10.0


def _resolve_device(device: str | None) -> str:
    """デバイスを自動解決する。"""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _build_model_args(cli) -> SimpleNamespace:
    """build_mnist_model / load_dataset 用の args を構築する。

    AP法は independent パラメータ化を要求し、重みは [0,1] 生確率として扱う。
    崩壊設定(train_ap_cnn の既定)に合わせる。
    """
    return SimpleNamespace(
        dataset=cli.dataset,
        k=cli.k,
        tree_depth=cli.tree_depth,
        n_inputs=cli.n_inputs,
        n_thresholds=0,               # 連続値入力(train_ap_cnn の既定)
        input_channels_per_tree=2,
        output_gate_factor=1,
        tau=None,                     # τ 自動計算
        grad_factor=1.0,
        connections='random',
        parameterization='independent',  # AP法の要求
        residual_init=False,
        device=cli.device,
        batch_size=cli.batch_size,
    )


def _limited(loader, max_batches: int | None):
    """max_batches が指定されればバッチ数を制限する(デバッグ/スモーク高速化)。"""
    if max_batches is None:
        return loader
    return itertools.islice(loader, max_batches)


def _snapshot(model, step, history, use_wandb, train_loss=None, train_acc=None):
    """現時点の単調性を記録し history に追記、wandb にもログする。"""
    rec = record_model_monotonicity(model, ap_mode=True)
    point = {
        'step': step,
        'V_mean': rec['V_mean'],
        'mixed_mean': rec['mixed_mean'],
        'per_module': [
            {'type': m['type'], 'V': m['V'], 'frac_mixed': m['frac_mixed']}
            for m in rec['per_module']
        ],
    }
    history.append(point)
    if use_wandb:
        log = {'V_mean': rec['V_mean'], 'mixed_mean': rec['mixed_mean']}
        if train_loss is not None:
            log['train_loss'] = train_loss
        if train_acc is not None:
            log['train_acc'] = train_acc
        log_metrics(log, step=step)
    return point


def parse_args():
    parser = argparse.ArgumentParser(
        description='AP-CNN(CDD-LGN)の単調度V/極性 記録実験'
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variant', type=str, default='ap',
                        choices=['ap', 'ap_monotone'],
                        help='トレーナー変種 (ap=既定 / ap_monotone=単調制約AP)')
    parser.add_argument('-k', type=int, default=16, help='チャンネルスケール (崩壊設定=16)')
    parser.add_argument('--tree-depth', type=int, default=3)
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--interval', type=int, default=100,
                        help='単調性を記録する global step 間隔')
    parser.add_argument('--max-train-batches', type=int, default=None,
                        help='1エポックの訓練バッチ数上限(デバッグ用)。'
                             '指定時は評価もこのバッチ数に制限する')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar-10'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='results/mono')
    parser.add_argument('--no-wandb', action='store_true')
    return parser.parse_args()


def main():
    cli = parse_args()
    cli.device = _resolve_device(cli.device)
    print(f'デバイス: {cli.device}')

    torch.manual_seed(cli.seed)
    random.seed(cli.seed)
    np.random.seed(cli.seed)

    margs = _build_model_args(cli)
    train_loader, test_loader = load_dataset(margs)
    print(f'データセット: {cli.dataset}')

    if cli.dataset == 'mnist':
        model, tau = build_mnist_model(margs)
    else:
        model, tau = build_cifar10_model(margs)

    if cli.variant == 'ap_monotone':
        trainer = MonotoneCNNAPTrainer(model, lr=cli.lr)
    else:
        trainer = CNNAPTrainer(model, lr=cli.lr)
    print(f'AP-CNN[{cli.variant}]: k={cli.k}, tree_depth={cli.tree_depth}, '
          f'n_inputs={cli.n_inputs}, lr={cli.lr}, tau={tau:.2f}')

    # wandb
    use_wandb = False
    if not cli.no_wandb:
        use_wandb = init_wandb(
            run_name=f'mono_cnn_{cli.variant}_k{cli.k}_s{cli.seed}',
            config_dict={
                'model': 'cnn', 'method': cli.variant, 'variant': cli.variant,
                'dataset': cli.dataset,
                'seed': cli.seed, 'k': cli.k, 'tree_depth': cli.tree_depth,
                'n_inputs': cli.n_inputs, 'epochs': cli.epochs, 'lr': cli.lr,
                'batch_size': cli.batch_size, 'interval': cli.interval,
            },
            model=model,
            device=cli.device,
            tags=['monotonicity', 'cnn', cli.variant],
            group='monotonicity',
        )

    # デバッグ時(max_train_batches指定)は評価も同数に制限して高速化
    eval_batches = cli.max_train_batches

    history: list[dict] = []
    epoch_acc: list[dict] = []
    global_step = 0

    # 初期記録(step=0)
    p0 = _snapshot(model, 0, history, use_wandb)
    print(f'[step 0] V_mean={p0["V_mean"]:.4f} mixed_mean={p0["mixed_mean"]:.4f}')

    best_hard = 0.0
    final_soft = 0.0
    final_hard = 0.0

    for epoch in range(1, cli.epochs + 1):
        last_loss = None
        last_acc = None
        for batch_idx, (x, y) in enumerate(train_loader):
            if cli.max_train_batches is not None and batch_idx >= cli.max_train_batches:
                break
            x = x.to(torch.float32).to(cli.device)
            y = y.to(cli.device)
            metrics = trainer.train_step(x, y)
            global_step += 1
            last_loss = metrics['loss']
            last_acc = metrics['accuracy']
            if global_step % cli.interval == 0:
                _snapshot(model, global_step, history, use_wandb,
                          train_loss=last_loss, train_acc=last_acc)

        # エポック評価(soft: 連続, hard: 二値化)
        soft = trainer.evaluate(
            _limited(test_loader, eval_batches), cli.device, binarize=False,
        )
        hard = trainer.evaluate(
            _limited(test_loader, eval_batches), cli.device, binarize=True,
        )
        best_hard = max(best_hard, hard)
        final_soft, final_hard = soft, hard
        epoch_acc.append({
            'epoch': epoch,
            'train_acc': last_acc,
            'soft': soft,
            'hard': hard,
        })
        print(f'[epoch {epoch}/{cli.epochs}] loss={last_loss:.4f} '
              f'train_acc={last_acc:.4f} soft={soft:.4f} hard={hard:.4f}')
        if use_wandb:
            log_metrics({
                'epoch': epoch,
                'test_soft': soft,
                'test_hard': hard,
                'best_hard': best_hard,
            }, step=global_step)

    collapsed = final_soft < COLLAPSE_THRESHOLD

    result = {
        'model': 'cnn',
        'method': cli.variant,
        'variant': cli.variant,
        'config': {
            'seed': cli.seed, 'k': cli.k, 'tree_depth': cli.tree_depth,
            'n_inputs': cli.n_inputs, 'epochs': cli.epochs, 'lr': cli.lr,
            'batch_size': cli.batch_size, 'interval': cli.interval,
            'dataset': cli.dataset,
        },
        'final_soft': final_soft,
        'final_hard': final_hard,
        'best_hard': best_hard,
        'collapsed': collapsed,
        'epoch_acc': epoch_acc,
        'history': history,
    }

    os.makedirs(cli.out_dir, exist_ok=True)
    out_path = os.path.join(
        cli.out_dir,
        f'mono_cnn_{cli.variant}_k{cli.k}_d{cli.tree_depth}_s{cli.seed}.json',
    )
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n保存: {out_path}')
    print(f'final_soft={final_soft:.4f} final_hard={final_hard:.4f} '
          f'best_hard={best_hard:.4f} collapsed={collapsed}')

    if use_wandb:
        wandb_finish()


if __name__ == '__main__':
    main()
