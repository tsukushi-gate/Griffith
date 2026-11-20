"""DLGN(difflogic, Petersen 2022)公式レシピによるMNIST再現実験。

現行L3テスト(tests/l3_paper/test_dlgn_mnist.py: 実測hard acc 0.9285)と
論文97.69%(smallモデル)の残差原因を、公式レシピとの差分を個別に
切り替えて特定するためのスタンドアロンスクリプト。

公式レシピ(difflogic README / experiments/main.py):
    python experiments/main.py -bs 100 -t 10 --dataset mnist20x20 \\
        -ni 200_000 -ef 1_000 -k 8_000 -l 6
    - 入力: 28x28から黒境界を適応除去した20x20(400次元)
    - 訓練入力は連続値[0,1]のまま、評価時のみ round() で二値化
    - 接続: unique / 重み初期化: randn(out,16) / Adam lr=0.01 / CE loss

現行テストとの主な差分(本スクリプトのフラグで個別に再現可能):
    データ(mnist vs mnist20x20) / 訓練入力(--binarize-train) / 層数 /
    tau / batch size / iteration数 / 接続方式(--connections)

使用例:
    # 公式レシピ(まず5万iterで傾向確認)
    MODAL_GPU=A10G uv run modal run scripts/modal_app.py \\
        --script recipe_dlgn_repro.py --args "-ni 50000 -ef 5000"

    # 現行テスト構成の対照実験(60k×60ep≈28k iter相当)
    MODAL_GPU=A10G uv run modal run scripts/modal_app.py \\
        --script recipe_dlgn_repro.py --args "--dataset mnist -l 4 \\
        -t 28.284 -bs 128 --connections random --binarize-train -ni 28000"
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffith_logic import GroupSum, LutLayer

JST = timezone(timedelta(hours=9))

BLACK_ROW_MAX = 14   # 片側の黒行/列カウント上限(difflogic準拠)
BORDER_TOTAL = 8     # 除去する黒行/列の合計(28→20)


def now_jst() -> str:
    return datetime.now(JST).strftime('%H:%M:%S')


def _trim_counts(black: torch.Tensor) -> tuple[int, int]:
    """黒い行(列)のブール列(28,)から上下(左右)の除去量を決める。

    difflogicのMNISTRemoveBorderTransformと同一のアルゴリズム。
    """
    lead = 0
    while black[lead] and lead < BLACK_ROW_MAX:
        lead += 1
    trail = 0
    while black[-trail - 1] and trail < BLACK_ROW_MAX:
        trail += 1
    assert lead + trail >= BORDER_TOTAL, (lead, trail)
    while lead + trail >= BORDER_TOTAL + 2:
        if lead > 0:
            lead -= 1
        if trail > 0:
            trail -= 1
    if lead + trail == BORDER_TOTAL + 1:
        if lead > 0:
            lead -= 1
        else:
            trail -= 1
    assert lead + trail == BORDER_TOTAL, (lead, trail)
    return lead, trail


def remove_border(img: torch.Tensor) -> torch.Tensor:
    """(28,28)画像から黒境界を適応除去して(20,20)にする(difflogic準拠)。"""
    top, bottom = _trim_counts((img == 0.0).all(dim=1))
    img = img[top:28 - bottom]
    left, right = _trim_counts((img == 0.0).all(dim=0))
    return img[:, left:28 - right]


def load_mnist_tensors(dataset: str, device: str):
    """MNIST全件を前処理済みGPUテンソルとして返す(20x20はキャッシュ)。

    Returns:
        (train_x, train_y, test_x, test_y): xは(N, in_dim) float32 [0,1]
    """
    cache_path = f'./data-mnist/preproc_{dataset}.pt'
    if os.path.exists(cache_path):
        blobs = torch.load(cache_path)
        print(f'前処理キャッシュを使用: {cache_path}')
    else:
        blobs = {}
        for split, train_flag in (('train', True), ('test', False)):
            ds = torchvision.datasets.MNIST(
                './data-mnist', train=train_flag, download=True,
            )
            imgs = ds.data.to(torch.float32) / 255.0  # (N,28,28)
            if dataset == 'mnist20x20':
                imgs = torch.stack([remove_border(im) for im in imgs])
            blobs[f'{split}_x'] = imgs.reshape(imgs.shape[0], -1)
            blobs[f'{split}_y'] = ds.targets.clone()
        torch.save(blobs, cache_path)
        print(f'前処理キャッシュを保存: {cache_path}')
    return (
        blobs['train_x'].to(device), blobs['train_y'].to(device),
        blobs['test_x'].to(device), blobs['test_y'].to(device),
    )


def evaluate(model, x_all, y_all, hard: bool, batch_size: int = 2000) -> float:
    """テスト精度を測定する。入力は常にround()で二値化(difflogic準拠)。

    Args:
        hard: True=推論モード(離散回路) / False=訓練モード(softmax加重)
    """
    orig_mode = model.training
    model.train(mode=not hard)
    correct = 0
    with torch.no_grad():
        for s in range(0, x_all.shape[0], batch_size):
            x = x_all[s:s + batch_size].round()
            pred = model(x).argmax(dim=-1)
            correct += (pred == y_all[s:s + batch_size]).sum().item()
    model.train(mode=orig_mode)
    return correct / x_all.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description='DLGN公式レシピ MNIST再現実験(デフォルト=公式レシピ)'
    )
    parser.add_argument('--dataset', type=str, default='mnist20x20',
                        choices=['mnist', 'mnist20x20'],
                        help='mnist=784入力 / mnist20x20=境界除去400入力(公式small)')
    parser.add_argument('-k', '--num-neurons', type=int, default=8000)
    parser.add_argument('-l', '--num-layers', type=int, default=6)
    parser.add_argument('-t', '--tau', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-bs', '--batch-size', type=int, default=100)
    parser.add_argument('-ni', '--num-iterations', type=int, default=200_000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=5_000)
    parser.add_argument('--connections', type=str, default='unique',
                        choices=['random', 'unique'])
    parser.add_argument('--binarize-train', action='store_true',
                        help='訓練入力を(x>0.5)で二値化(現行テスト方式)。'
                             'デフォルトは公式準拠で連続値のまま')
    parser.add_argument('--n-train', type=int, default=0,
                        help='訓練サンプル数の制限(0=全60k)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[{now_jst()}] device={device} config={vars(args)}')

    train_x, train_y, test_x, test_y = load_mnist_tensors(args.dataset, device)
    if args.n_train > 0:
        train_x, train_y = train_x[:args.n_train], train_y[:args.n_train]
    if args.binarize_train:
        train_x = (train_x > 0.5).to(torch.float32)
    in_dim = train_x.shape[1]
    print(f'train={train_x.shape[0]} test={test_x.shape[0]} in_dim={in_dim} '
          f'訓練入力={"二値化(>0.5)" if args.binarize_train else "連続値[0,1]"}')

    k = args.num_neurons
    llkw = dict(n_inputs=2, device=device, connections=args.connections)
    layers = [LutLayer(in_dim, k, **llkw)]
    for _ in range(args.num_layers - 1):
        layers.append(LutLayer(k, k, **llkw))
    layers.append(GroupSum(k=10, tau=args.tau))
    model = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'モデル: {args.num_layers}層 × {k}ニューロン, '
          f'tau={args.tau}, connections={args.connections}')

    n_train = train_x.shape[0]
    steps_per_epoch = n_train // args.batch_size  # drop_last相当
    best_hard = 0.0
    start = time.time()
    model.train()

    perm = None
    for it in range(args.num_iterations):
        step_in_epoch = it % steps_per_epoch
        if step_in_epoch == 0:
            perm = torch.randperm(n_train, device=device)
        idx = perm[step_in_epoch * args.batch_size:
                   (step_in_epoch + 1) * args.batch_size]
        x, y = train_x[idx], train_y[idx]

        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

        if (it + 1) % args.eval_freq == 0 or (it + 1) == args.num_iterations:
            acc_hard = evaluate(model, test_x, test_y, hard=True)
            acc_soft = evaluate(model, test_x, test_y, hard=False)
            best_hard = max(best_hard, acc_hard)
            elapsed = time.time() - start
            ips = (it + 1) / elapsed
            eta_min = (args.num_iterations - it - 1) / ips / 60
            print(f'[{now_jst()}] iter={it + 1:>7,}/{args.num_iterations:,} '
                  f'loss={loss.item():.4f} '
                  f'test_acc_hard={acc_hard:.4f} test_acc_soft={acc_soft:.4f} '
                  f'best_hard={best_hard:.4f} '
                  f'({ips:.1f} it/s, 残り{eta_min:.1f}分)', flush=True)

    acc_hard = evaluate(model, test_x, test_y, hard=True)
    acc_soft = evaluate(model, test_x, test_y, hard=False)
    # 現行テストとの測定差の切り分け用: 先頭2000サンプルのhard acc
    acc_hard_2k = evaluate(model, test_x[:2000], test_y[:2000], hard=True)
    elapsed_min = (time.time() - start) / 60
    print(f'[FINAL] dataset={args.dataset} l={args.num_layers} k={k} '
          f'tau={args.tau} bs={args.batch_size} ni={args.num_iterations} '
          f'conn={args.connections} '
          f'train_input={"binary" if args.binarize_train else "continuous"}')
    print(f'[FINAL] test_acc_hard={acc_hard:.4f} test_acc_soft={acc_soft:.4f} '
          f'test_acc_hard_first2000={acc_hard_2k:.4f} '
          f'best_hard={best_hard:.4f} elapsed={elapsed_min:.1f}分')


if __name__ == '__main__':
    main()
