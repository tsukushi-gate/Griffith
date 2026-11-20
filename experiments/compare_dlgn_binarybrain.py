"""
DLGN vs BinaryBrain 比較実験スクリプト

複数の統制条件でDLGNとBinaryBrainをMNISTで比較する。

使用例:
    # 全実験を実行（50エポック）
    uv run python experiments/compare_dlgn_binarybrain.py

    # 特定の実験のみ
    uv run python experiments/compare_dlgn_binarybrain.py --experiments A1,B2

    # エポック数を変更
    uv run python experiments/compare_dlgn_binarybrain.py --epochs 10
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from flops_counter import count_flops_binarybrain, count_flops_dlgn, format_flops

from griffith_logic import (
    BinaryToReal,
    DifferentiableLutBlock,
    GroupSum,
    LutLayer,
    RealToBinary,
)

# ============================================================
# 実験定義
# ============================================================

def compute_bb_layer_sizes_for_gates(target_gates: int, n_inputs: int,
                                     num_classes: int = 10) -> list:
    """
    目標ゲート数に近いBinaryBrainの6層構成を自動計算。

    最終3層は固定（N²×C, N×C, C）、前3層で残りを按分。
    """
    N = n_inputs
    C = num_classes
    # 最終3層（固定）
    tail = [N * N * C, N * C, C]
    tail_gates = sum(tail)

    remaining = max(target_gates - tail_gates, N * N * N * 3)

    # 前3層を按分（4:2:1の比率）
    layer1 = int(remaining * 4 / 7)
    layer2 = int(remaining * 2 / 7)
    layer3 = remaining - layer1 - layer2

    # N倍制約の確認・調整
    # layer1は入力(784)から接続するため制約なし
    # layer2 <= layer1 / N → OK (layer2 < layer1)
    # layer3 <= layer2 / N
    if layer3 > layer2 // N:
        layer3 = layer2 // N

    # 最終3層の接続制約: tail[0] <= layer3 * N^? → 通常OK

    sizes = [layer1, layer2, layer3] + tail
    return sizes


EXPERIMENTS = {
    # ─── グループA: N=2 比較 ───
    'A1': {
        'method': 'dlgn',
        'n_inputs': 2,
        'k': 8000,
        'layers': 6,
        'tau': 10.0,
        'lr': 0.01,
        'batch_size': 128,
        'desc': 'DLGN標準構成 (N=2, k=8000, L=6)',
    },
    'A2': {
        'method': 'binarybrain',
        'n_inputs': 2,
        'layer_sizes': [256, 128, 64, 40, 20, 10],
        'lr': 0.001,
        'batch_size': 64,
        'desc': 'BinaryBrain標準公式 (N=2)',
    },
    'A3': {
        'method': 'binarybrain',
        'n_inputs': 2,
        'layer_sizes': 'auto',
        'target_gates': 48000,
        'lr': 0.001,
        'batch_size': 64,
        'desc': 'BinaryBrain大規模 (N=2, A1とゲート数一致)',
    },
    # ─── グループB: N=6 比較 ───
    'B1': {
        'method': 'dlgn',
        'n_inputs': 6,
        'k': 4000,
        'layers': 4,
        'tau': 10.0,
        'lr': 0.01,
        'batch_size': 128,
        'desc': 'DLGN N=6構成 (k=4000, L=4)',
    },
    'B2': {
        'method': 'binarybrain',
        'n_inputs': 6,
        'layer_sizes': [2304, 384, 64, 360, 60, 10],
        'lr': 0.001,
        'batch_size': 64,
        'desc': 'BinaryBrain標準構成 (N=6)',
    },
    'B3': {
        'method': 'dlgn',
        'n_inputs': 6,
        'k': 530,
        'layers': 6,
        'tau': 10.0,
        'lr': 0.01,
        'batch_size': 128,
        'desc': 'DLGN小規模 (N=6, B2とゲート数一致)',
    },
    'B4': {
        'method': 'binarybrain',
        'n_inputs': 6,
        'layer_sizes': 'auto',
        'target_gates': 16000,
        'lr': 0.001,
        'batch_size': 64,
        'desc': 'BinaryBrain大規模 (N=6, B1とゲート数一致)',
    },
}


# ============================================================
# モデル構築
# ============================================================

class BinaryBrainModel(nn.Module):
    """BinaryBrainモデル（train_binarybrain.pyと同一）"""

    def __init__(self, in_dim, num_classes, n_inputs, layer_sizes,
                 frame_modulation_size=15, device='cpu'):
        super().__init__()
        self.frame_modulation_size = frame_modulation_size
        self.flatten = nn.Flatten()
        self.real_to_binary = RealToBinary(frame_modulation_size)
        self.lut_blocks = nn.ModuleList()
        prev_dim = in_dim
        for out_dim in layer_sizes:
            self.lut_blocks.append(DifferentiableLutBlock(
                in_dim=prev_dim, out_dim=out_dim,
                n_inputs=n_inputs, device=device,
            ))
            prev_dim = out_dim
        self.binary_to_real = BinaryToReal(frame_modulation_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.real_to_binary(x)
        for block in self.lut_blocks:
            x = block(x)
        x = self.binary_to_real(x)
        return x

    def clamp_weights(self):
        for block in self.lut_blocks:
            block.clamp_weights()


def build_dlgn_model(in_dim, num_classes, k, layers, n_inputs, tau, device):
    """DLGNモデルを構築"""
    assert k % num_classes == 0
    llkw = dict(n_inputs=n_inputs, device=device, parameterization='softmax')
    model_layers = [nn.Flatten()]
    model_layers.append(LutLayer(in_dim=in_dim, out_dim=k, **llkw))
    for _ in range(layers - 1):
        model_layers.append(LutLayer(in_dim=k, out_dim=k, **llkw))
    model_layers.append(GroupSum(num_classes, tau=tau))
    return nn.Sequential(*model_layers).to(device)


def build_model(exp_config, device):
    """実験設定からモデルを構築"""
    in_dim = 784
    num_classes = 10

    if exp_config['method'] == 'dlgn':
        model = build_dlgn_model(
            in_dim, num_classes,
            exp_config['k'], exp_config['layers'],
            exp_config['n_inputs'], exp_config['tau'], device,
        )
        total_gates = sum(
            m.num_neurons for m in model.modules() if isinstance(m, LutLayer)
        )
        flops_info = count_flops_dlgn(
            total_gates, exp_config['n_inputs'], exp_config['k'],
        )
    else:
        layer_sizes = exp_config['layer_sizes']
        if layer_sizes == 'auto':
            layer_sizes = compute_bb_layer_sizes_for_gates(
                exp_config['target_gates'], exp_config['n_inputs'],
            )
        model = BinaryBrainModel(
            in_dim, num_classes, exp_config['n_inputs'],
            layer_sizes, device=device,
        ).to(device)
        total_gates = sum(layer_sizes)
        flops_info = count_flops_binarybrain(
            total_gates, exp_config['n_inputs'],
        )

    total_params = sum(p.numel() for p in model.parameters())

    return model, {
        'total_gates': total_gates,
        'total_params': total_params,
        'layer_sizes': layer_sizes if exp_config['method'] == 'binarybrain' else None,
        **flops_info,
    }


# ============================================================
# データセット
# ============================================================

def load_dataset(batch_size, device):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(
        './data-mnist', train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.MNIST(
        './data-mnist', train=False, transform=transform,
    )
    pin = (device == 'cuda')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        pin_memory=pin, drop_last=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        pin_memory=pin, drop_last=True, num_workers=2,
    )
    return train_loader, test_loader


# ============================================================
# 評価
# ============================================================

def evaluate_dlgn(model, loader, device, mode=False):
    """DLGN評価。mode=False→Hard(推論), mode=True→Soft(訓練)"""
    orig = model.training
    model.train(mode=mode)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            if not mode:
                x = x.round()
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    model.train(mode=orig)
    return correct / total if total > 0 else 0.0


def evaluate_bb(model, loader, device):
    """BinaryBrain評価"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / total if total > 0 else 0.0


# ============================================================
# 1実験の実行
# ============================================================

def run_experiment(exp_id, exp_config, epochs, device, csv_writer):
    """1つの実験を実行し、結果をCSVに書き込む"""
    print(f'\n{"=" * 70}')
    print(f'実験 {exp_id}: {exp_config["desc"]}')
    print(f'{"=" * 70}')

    method = exp_config['method']
    batch_size = exp_config['batch_size']

    # モデル構築
    model, model_info = build_model(exp_config, device)
    total_gates = model_info['total_gates']
    total_params = model_info['total_params']
    flops_per_sample = model_info['flops_per_sample']

    print(f'手法: {method}')
    print(f'N={exp_config["n_inputs"]}入力LUT')
    if model_info.get('layer_sizes'):
        print(f'層サイズ: {model_info["layer_sizes"]}')
    print(f'総ゲート数: {total_gates:,}')
    print(f'総パラメータ数: {total_params:,}')
    print(f'FLOPS/サンプル: {format_flops(flops_per_sample)}')

    # データ
    train_loader, test_loader = load_dataset(batch_size, device)

    # Optimizer
    lr = exp_config['lr']
    if method == 'binarybrain':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_start = time.time()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # 訓練
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'{exp_id} epoch {epoch}/{epochs}', leave=False)
        for x, y in pbar:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if method == 'binarybrain':
                model.clamp_weights()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - train_start
        cumulative_flops = flops_per_sample * batch_size * len(train_loader) * epoch

        # 評価
        if method == 'dlgn':
            test_acc = evaluate_dlgn(model, test_loader, device, mode=False)
            test_acc_soft = evaluate_dlgn(model, test_loader, device, mode=True)
        else:
            test_acc = evaluate_bb(model, test_loader, device)
            test_acc_soft = None

        best_acc = max(best_acc, test_acc)

        soft_str = f'  soft={test_acc_soft:.4f}' if test_acc_soft is not None else ''
        print(f'  [{exp_id} epoch {epoch:>3d}] loss={avg_loss:.4f}  '
              f'test_acc={test_acc:.4f}{soft_str}  '
              f'[{timedelta(seconds=int(elapsed))}]')

        # CSV書き込み
        csv_writer.writerow({
            'exp_id': exp_id,
            'method': method,
            'n_inputs': exp_config['n_inputs'],
            'total_gates': total_gates,
            'total_params': total_params,
            'flops_per_sample': flops_per_sample,
            'epoch': epoch,
            'train_loss': f'{avg_loss:.6f}',
            'test_acc': f'{test_acc:.4f}',
            'test_acc_soft': f'{test_acc_soft:.4f}' if test_acc_soft else '',
            'wall_time_sec': f'{elapsed:.1f}',
            'cumulative_flops': cumulative_flops,
            'batch_size': batch_size,
            'lr': lr,
        })

    print(f'  最終精度: {best_acc:.4f}  '
          f'(総時間: {timedelta(seconds=int(time.time() - train_start))})')

    return best_acc


# ============================================================
# サマリー出力
# ============================================================

def print_summary(results):
    """全実験のサマリーをMarkdownテーブルで表示"""
    print(f'\n{"=" * 90}')
    print('サマリー')
    print(f'{"=" * 90}')
    print()
    print('| ID | 手法 | N | ゲート数 | パラメータ数 | FLOPS/サンプル | 最終精度 |')
    print('|----|------|---|---------|------------|--------------|---------|')
    for exp_id, info in sorted(results.items()):
        print(f'| {exp_id} | {info["method"]:12s} | {info["n_inputs"]} | '
              f'{info["total_gates"]:>8,} | {info["total_params"]:>11,} | '
              f'{format_flops(info["flops_per_sample"]):>14s} | '
              f'{info["best_acc"]:.4f}  |')


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DLGN vs BinaryBrain 比較実験')
    parser.add_argument('--experiments', type=str, default=None,
                        help='実行する実験ID（カンマ区切り、例: A1,B2）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='エポック数 (default: 50)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--seed', type=int, default=1,
                        help='ランダムシード (default: 1)')
    args = parser.parse_args()

    # デバイス
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

    # 実験選択
    if args.experiments:
        exp_ids = [e.strip() for e in args.experiments.split(',')]
    else:
        exp_ids = list(EXPERIMENTS.keys())

    print(f'実験: {", ".join(exp_ids)}')
    print(f'エポック数: {args.epochs}')

    # CSV出力
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    csv_path = f'results/compare_dlgn_bb_{timestamp}.csv'
    csv_fields = [
        'exp_id', 'method', 'n_inputs', 'total_gates', 'total_params',
        'flops_per_sample', 'epoch', 'train_loss', 'test_acc',
        'test_acc_soft', 'wall_time_sec', 'cumulative_flops',
        'batch_size', 'lr',
    ]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    print(f'CSV出力: {csv_path}')

    # 実験実行
    results = {}
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f'警告: 実験 {exp_id} は定義されていません。スキップ。')
            continue

        exp_config = EXPERIMENTS[exp_id]

        # シードをリセット（各実験を再現可能に）
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        model, model_info = build_model(exp_config, args.device)
        best_acc = run_experiment(
            exp_id, exp_config, args.epochs, args.device, csv_writer,
        )

        results[exp_id] = {
            'method': exp_config['method'],
            'n_inputs': exp_config['n_inputs'],
            'total_gates': model_info['total_gates'],
            'total_params': model_info['total_params'],
            'flops_per_sample': model_info['flops_per_sample'],
            'best_acc': best_acc,
        }

    csv_file.close()
    print(f'\nCSV保存完了: {csv_path}')

    # サマリー
    if results:
        print_summary(results)


if __name__ == '__main__':
    main()
