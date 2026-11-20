"""Fashion-MNIST用 teacher-float MLP(784-300-100-10 ReLU)学習。

mnist-intlower-v1(int lowering ゲート通過構成)を
Fashion-MNISTへ転移する実験の第一段階。convert_transcribe_mnist.py は
--dataset 引数を持たず教師フェーズがMNIST専用のため、本スクリプトは
その学習・保存ロジック(train_teacher / _save_teacher_ckpt 等)をそのまま
import して再利用し、データソースだけを load_fashion_mnist に差し替える
(convert_transcribe_mnist.py 自体は無改造)。

teacher-ste は' パイプライン(convert_int_lowering_mnist.py --dataset
fashion_mnist)では使わないため学習しない(teacher-floatのみ)。

使用例:
    # 煙実行
    uv run python experiments/convert_fashion_teacher.py --quick --no-wandb

    # 本実行(5epoch, wandb online)
    uv run python experiments/convert_fashion_teacher.py --epochs 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import convert_transcribe_mnist as base
import numpy as np
import torch
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.datasets.vision import load_fashion_mnist

base.load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST teacher-float MLP学習")
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    device = base._auto_device(args.device)
    print(f'デバイス: {device}')
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    if args.quick:
        args.epochs = min(args.epochs, 1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Fashion-MNIST(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = load_fashion_mnist(binarize_threshold=0.5)
    tr_n = 4000 if args.quick else None
    te_n = 2000 if args.quick else None
    x_tr, y_tr = base._flatten_binary(train_set, n=tr_n)
    x_te, y_te = base._flatten_binary(test_set, n=te_n)
    print(f'train={len(x_tr)} test={len(x_te)}')

    use_wandb = init_wandb(
        run_name='teacher_mlp_fashion_mnist',
        config_dict={**vars(args), 'phase': 'teacher', 'dataset': 'fashion_mnist'},
        device=device,
        tags=['D3-6prime', 'teacher', 'fashion-mnist'],
        group='convert-fashion-intlower',
    )

    print('\n=== teacher-float(Fashion-MNIST) 学習 ===')
    model, test_acc = base.train_teacher(
        'float', x_tr, y_tr, x_te, y_te,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        device=device, seed=args.seed, use_wandb=use_wandb,
    )
    # ckpt名を teacher_mlp_float_fashion_mnist_* にしてMNIST版と衝突させない
    # (_save_teacher_ckpt 本体は無改造、prefix生成部だけこちらで担う)
    from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        'teacher_mlp_float_fashion_mnist', base_dir=args.checkpoint_dir)
    metadata = {
        'args': vars(args), 'kind': 'float',
        'model_class': base.TeacherMLP, 'model_kwargs': {},
        'num_classes': 10, 'input_size': (1, 28, 28),
        'transform': None, 'test_acc': test_acc, 'dataset': 'fashion_mnist',
    }
    dummy_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    save_checkpoint(best_path, model, dummy_opt, None, 0, test_acc, metadata)
    save_checkpoint(latest_path, model, dummy_opt, None, 0, test_acc, metadata)
    print(f'  ckpt: {best_path}')

    if use_wandb:
        log_metrics({'teacher_float_acc': test_acc})
        wandb_finish()
    print(f'\nteacher-float(fashion-mnist) test_acc={test_acc:.4f}')
    print(f'ckpt: {best_path}')


if __name__ == '__main__':
    main()
