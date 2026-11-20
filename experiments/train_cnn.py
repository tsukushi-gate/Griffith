"""
CDDLGN (Convolutional Differentiable Logic Gate Networks) 訓練スクリプト

LogicTreeNetアーキテクチャを論文（NeurIPS 2024）に準拠して実装。

使用例:
    # MNIST Small (k=16)
    uv run python experiments/train_cnn.py --dataset mnist -k 16 -e 50

    # MNIST Medium (k=64)
    uv run python experiments/train_cnn.py --dataset mnist -k 64 -e 100 -bs 512

    # CIFAR-10 Small (k=32)
    uv run python experiments/train_cnn.py --dataset cifar-10 -k 32 -e 200

    # CIFAR-10 Medium (k=256)
    uv run python experiments/train_cnn.py --dataset cifar-10 -k 256 -e 700

    # イテレーションモード
    uv run python experiments/train_cnn.py --dataset mnist -k 64 -ni 50000
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
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import GroupSum, LutConv2d, LutLayer, OrPooling2d

# .envから環境変数を読み込み
load_dotenv()


# ============================================================
# データセット
# ============================================================

class ThresholdBinarize:
    """閾値による二値化変換（pickle可能なクラス）。Section 3.5準拠。"""

    def __init__(self, n_thresholds: int):
        self.n_thresholds = n_thresholds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [(x > (i + 1) / (self.n_thresholds + 1)).float()
             for i in range(self.n_thresholds)], dim=0
        )


def load_dataset(args):
    if args.dataset == 'mnist':
        n_thresholds = args.n_thresholds
        if n_thresholds > 0:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                ThresholdBinarize(n_thresholds),
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        train_set = torchvision.datasets.MNIST(
            './data-mnist', train=True, download=True, transform=transforms,
        )
        test_set = torchvision.datasets.MNIST(
            './data-mnist', train=False, transform=transforms,
        )
    elif args.dataset == 'cifar-10':
        n_thresholds = args.n_thresholds
        if n_thresholds > 0:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                ThresholdBinarize(n_thresholds),
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        train_set = torchvision.datasets.CIFAR10(
            './data-cifar', train=True, download=True, transform=transforms,
        )
        test_set = torchvision.datasets.CIFAR10(
            './data-cifar', train=False, transform=transforms,
        )
    else:
        raise ValueError(f'未対応データセット: {args.dataset}')

    use_pin_memory = args.device == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )
    # train_set も返す(KD時に PackedDistillDataset で包むため)
    return train_loader, test_loader, train_set


def build_raw_train_set(args):
    """teacher推論用の生[0,1]画像 train dataset(ToTensorのみ、augmentationなし)。

    student側(load_dataset)と**同一の torchvision データセット・同一split・
    同一順序**を ToTensor() だけで読む。teacher は生[0,1]画像を見て、正規化は
    convert.teacher の teacher_transform 側で行う(runner._apply_distill と同規約)。
    student の transform が決定的(ThresholdBinarize)なので index 整列が保たれる。
    """
    raw_tf = torchvision.transforms.ToTensor()
    if args.dataset == 'mnist':
        return torchvision.datasets.MNIST(
            './data-mnist', train=True, download=True, transform=raw_tf,
        )
    elif args.dataset == 'cifar-10':
        return torchvision.datasets.CIFAR10(
            './data-cifar', train=True, download=True, transform=raw_tf,
        )
    raise ValueError(f'未対応データセット: {args.dataset}')


# ============================================================
# LogicTreeNetアーキテクチャ（Appendix A.1準拠）
# ============================================================

def build_mnist_model(args):
    """
    MNIST用LogicTreeNet（Appendix A.1.2）。

    Conv block 1: k kernels, rf=5×5, d=3, no padding → k×24×24
    OrPool → k×12×12
    Conv block 2: 3k kernels, rf=3×3, d=3 → 3k×12×12
    OrPool → 3k×6×6
    Conv block 3: 9k kernels, rf=3×3, d=3 → 9k×6×6
    OrPool → 9k×3×3
    Flatten → 81k
    Rand: 81k → 1280k
    Rand: 1280k → 640k
    Rand: 640k → 320k
    GroupSum: 320k → 10
    """
    k = args.k
    d = args.tree_depth
    n = args.n_inputs
    device = args.device
    gf = args.grad_factor
    n_classes = 10

    # 入力チャンネル数（閾値数に依存）
    in_ch = args.n_thresholds if args.n_thresholds > 0 else 1

    # Dense層の出力ゲートファクター
    ogf = args.output_gate_factor

    # wide-last-layer時は最終Dense層の出力も320k·ogf → GroupSumが320k·ogf個を集約
    last_dense_out = 320 * k * ogf if args.wide_last_layer else 320 * k
    n_outputs_per_class = last_dense_out // n_classes
    # τの自動計算: τ ∝ √(n_outputs_per_class)
    if args.tau is None:
        tau = max(1.0, math.sqrt(n_outputs_per_class))
    else:
        tau = args.tau

    # conv木: softmax16gはconv専用、warpはconv未対応→softmaxへフォールバック
    conv_param = ('softmax' if args.parameterization == 'warp'
                  else args.parameterization)
    conv_kw = dict(n_inputs=n, device=device, grad_factor=gf,
                   input_channels_per_tree=args.input_channels_per_tree,
                   parameterization=conv_param)
    # Dense層のパラメータ化: conv専用のsoftmax16gはdenseでは（N=2で等価な）softmaxに写像
    dense_param = 'softmax' if args.parameterization == 'softmax16g' else args.parameterization
    lut_kw = dict(n_inputs=n, device=device, grad_factor=gf,
                  connections=args.connections,
                  parameterization=dense_param,
                  residual_init=args.residual_init_dense,
                  # selectionはdense層のみ(conv木のhard/gumbel-STは非ゴール)
                  selection=args.selection)

    layers = []

    # Conv blocks
    layers.append(LutConv2d(in_ch, k, kernel_size=5, tree_depth=d,
                            padding=0, **conv_kw))
    layers.append(OrPooling2d(2))  # → k×12×12

    layers.append(LutConv2d(k, 3 * k, kernel_size=3, tree_depth=d,
                            padding=1, **conv_kw))
    layers.append(OrPooling2d(2))  # → 3k×6×6

    layers.append(LutConv2d(3 * k, 9 * k, kernel_size=3, tree_depth=d,
                            padding=1, **conv_kw))
    layers.append(OrPooling2d(2))  # → 9k×3×3

    # Flatten
    layers.append(torch.nn.Flatten())
    flat_dim = 9 * k * 3 * 3  # = 81k

    # Dense LGN layers
    layers.append(LutLayer(flat_dim, 1280 * k * ogf, **lut_kw))
    layers.append(LutLayer(1280 * k * ogf, 640 * k * ogf, **lut_kw))
    layers.append(LutLayer(640 * k * ogf, last_dense_out, **lut_kw))

    # GroupSum
    layers.append(GroupSum(n_classes, tau=tau))

    model = torch.nn.Sequential(*layers).to(device)
    return model, tau


def build_cifar10_model(args):
    """
    CIFAR-10用LogicTreeNet（Appendix A.1.1）。

    Conv: k, rf=3×3, d=3, padding=1 → k×32×32
    OrPool → k×16×16
    Conv: 4k, rf=3×3, d=3 → 4k×16×16
    OrPool → 4k×8×8
    Conv: 16k, rf=3×3, d=3 → 16k×8×8
    OrPool → 16k×4×4
    Conv: 32k, rf=3×3, d=3 → 32k×4×4
    OrPool → 32k×2×2
    Flatten → 128k
    Rand: 128k → 1280k
    Rand: 1280k → 640k
    Rand: 640k → 320k
    GroupSum: 320k → 10
    """
    k = args.k
    d = args.tree_depth
    n = args.n_inputs
    device = args.device
    gf = args.grad_factor
    n_classes = 10

    # 入力チャンネル数
    in_ch = 3 * args.n_thresholds if args.n_thresholds > 0 else 3

    ogf = args.output_gate_factor

    last_dense_out = 320 * k * ogf if args.wide_last_layer else 320 * k
    n_outputs_per_class = last_dense_out // n_classes
    if args.tau is None:
        tau = max(1.0, math.sqrt(n_outputs_per_class))
    else:
        tau = args.tau

    # conv木: softmax16gはconv専用、warpはconv未対応→softmaxへフォールバック
    conv_param = ('softmax' if args.parameterization == 'warp'
                  else args.parameterization)
    conv_kw = dict(n_inputs=n, device=device, grad_factor=gf,
                   input_channels_per_tree=args.input_channels_per_tree,
                   parameterization=conv_param)
    # Dense層のパラメータ化: conv専用のsoftmax16gはdenseでは（N=2で等価な）softmaxに写像
    dense_param = 'softmax' if args.parameterization == 'softmax16g' else args.parameterization
    lut_kw = dict(n_inputs=n, device=device, grad_factor=gf,
                  connections=args.connections,
                  parameterization=dense_param,
                  residual_init=args.residual_init_dense,
                  # selectionはdense層のみ(conv木のhard/gumbel-STは非ゴール)
                  selection=args.selection)

    layers = []

    # Conv blocks
    layers.append(LutConv2d(in_ch, k, kernel_size=3, tree_depth=d,
                            padding=1, **conv_kw))
    layers.append(OrPooling2d(2))  # → k×16×16

    layers.append(LutConv2d(k, 4 * k, kernel_size=3, tree_depth=d,
                            padding=1, **conv_kw))
    layers.append(OrPooling2d(2))  # → 4k×8×8

    layers.append(LutConv2d(4 * k, 16 * k, kernel_size=3, tree_depth=d,
                            padding=1, **conv_kw))
    layers.append(OrPooling2d(2))  # → 16k×4×4

    layers.append(LutConv2d(16 * k, 32 * k, kernel_size=3, tree_depth=d,
                            padding=1, **conv_kw))
    layers.append(OrPooling2d(2))  # → 32k×2×2

    # Flatten
    layers.append(torch.nn.Flatten())
    flat_dim = 32 * k * 2 * 2  # = 128k

    # Dense LGN layers
    layers.append(LutLayer(flat_dim, 1280 * k * ogf, **lut_kw))
    layers.append(LutLayer(1280 * k * ogf, 640 * k * ogf, **lut_kw))
    layers.append(LutLayer(640 * k * ogf, last_dense_out, **lut_kw))

    # GroupSum
    layers.append(GroupSum(n_classes, tau=tau))

    model = torch.nn.Sequential(*layers).to(device)
    return model, tau


# ============================================================
# ユーティリティ
# ============================================================

def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ"""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i >= n:
                return


# ============================================================
# ログ
# ============================================================

def setup_logger(script_name, dataset, log_dir='logs'):
    """訓練メトリクスのログファイルを作成する。行バッファリングで即時flush。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{script_name}_{dataset}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


# ============================================================
# 訓練・評価
# ============================================================

def train_step(model, x, y, loss_fn, optimizer, tau_sched=None,
               spectral_reg=0.0):
    output = model(x)
    loss = loss_fn(output, y)
    if spectral_reg > 0:
        from griffith_logic.regularization import spectral_penalty
        loss = loss + spectral_reg * spectral_penalty(model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if tau_sched is not None:
        tau_sched.step()
    return loss.item()


def evaluate(model, loader, device, mode=False, desc=None):
    """
    Args:
        mode: True=訓練モード（softmax重み）, False=推論モード（離散化）
        desc: プログレスバーの説明文（Noneで非表示）
    """
    orig_mode = model.training
    model.train(mode=mode)
    correct = 0
    total = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    with torch.no_grad():
        for x, y in it:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            if not mode:
                x = x.round()  # 推論モードではバイナリ化
            output = model(x)
            pred = output.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
            if desc and isinstance(it, tqdm):
                it.set_postfix(acc=f'{correct/total:.4f}')
    model.train(mode=orig_mode)
    return correct / total if total > 0 else 0.0


# ============================================================
# メイン
# ============================================================

def interactive_classify(model, test_loader, device, dataset_name):
    """対話的分類デモ。ランダムテストサンプルで予測を表示。"""
    model.eval()
    test_dataset = test_loader.dataset

    if 'cifar' in dataset_name:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        class_names = [str(i) for i in range(10)]

    print(f'\n=== 分類デモ ({dataset_name}) ===')
    print('"r"でランダムテストサンプル、"q"で終了:')

    while True:
        try:
            cmd = input('> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd == 'q':
            break
        if cmd != 'r':
            print('"r"でランダムサンプル、"q"で終了')
            continue

        idx = random.randint(0, len(test_dataset) - 1)
        x, y = test_dataset[idx]
        x_input = x.to(torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            x_input = x_input.round()
            output = model(x_input)
            pred = output.argmax(dim=-1).item()

        pred_name = class_names[pred] if class_names else str(pred)
        true_name = class_names[y] if class_names else str(y)
        mark = '✓' if pred == y else '✗'
        print(f'サンプル #{idx}: 予測={pred_name}, 正解={true_name} {mark}')


def main():
    parser = argparse.ArgumentParser(
        description='CDDLGN (LogicTreeNet) 訓練スクリプト'
    )

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar-10'],
                        help='データセット (default: mnist)')
    parser.add_argument('-k', type=int, default=16,
                        help='モデル幅パラメータ k (default: 16)')
    parser.add_argument('-d', '--tree-depth', type=int, default=3,
                        help='ツリーの深さ (default: 3)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='ゲートの入力数 (default: 2)')
    parser.add_argument('--tau', type=float, default=None,
                        help='GroupSum温度（Noneで自動計算）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学習率（Noneでデータセットに応じた既定値）')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay（Noneでデータセットに応じた既定値）')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数（--num-iterationsと排他）')
    parser.add_argument('-ni', '--num-iterations', type=int, default=None,
                        help='反復回数（--epochsと排他）')
    parser.add_argument('-ef', '--eval-freq', type=int, default=None,
                        help='評価頻度（エポック時=Nエポックごと[default:1], '
                             'イテレーション時=Nイテレーションごと[default:2000]）')
    parser.add_argument('--n-thresholds', type=int, default=3,
                        help='入力二値化の閾値数 (default: 3, 0=二値化なし)')
    parser.add_argument('--input-channels-per-tree', type=int, default=2,
                        help='各ツリーが参照する入力チャンネル数 (default: 2)')
    parser.add_argument('--output-gate-factor', type=int, default=2,
                        help='Dense層の出力ゲートファクター (default: 2)')
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help='Dense層の接続方式 (default: random)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'softmax16g',
                                 'warp'],
                        help='パラメータ化方式 (default: softmax。warpは'
                             'dense層のみ、conv木はsoftmaxへフォールバック)')
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式(dense層のみ。default: soft)')
    parser.add_argument('--tau-anneal', action='store_true',
                        help='TauAnnealScheduler(τ_b幾何減衰+ノイズ減衰)を有効化')
    parser.add_argument('--tau-b0', type=float, default=1.0)
    parser.add_argument('--tau-b-min', type=float, default=0.2)
    parser.add_argument('--spectral-reg', type=float, default=0.0,
                        help='Spectral正則化係数λ(DWN §3.4。0で無効)')
    parser.add_argument('--residual-init-dense', action='store_true',
                        help='Dense LutLayer3層をResidual Init（ゲートAパススルー）で初期化')
    parser.add_argument('--wide-last-layer', action='store_true',
                        help='論文A.1.1(*)準拠: 最終Dense層もogf倍(640k·ogf→320k·ogf)')
    parser.add_argument('--wandb-group', type=str, default='cnn',
                        help="wandbのgroup(既定'cnn')。他実験と同一groupで比較したい時に指定する")
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数 (default: 1.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先ディレクトリ (default: logs)')

    # チェックポイント・推論
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開（パス指定）')
    parser.add_argument('--interactive', action='store_true',
                        help='対話推論モード（--resume必須）')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')

    # 知識蒸留(KD): Conv-DLGN student + 強teacher(HF resnet18等)
    parser.add_argument('--distill-teacher', type=str, default=None,
                        help="蒸留teacher指定(例 'hf:edadaltocg/resnet18_cifar10' / "
                             "'ckpt:<path>')。未指定でKD無効 (default: None)")
    parser.add_argument('--distill-alpha', type=float, default=0.25,
                        help='hard(CE)項の重み。α=1で純CE、α=0で純KD (default: 0.25)')
    parser.add_argument('--distill-temperature', type=float, default=4.0,
                        help='KD温度 T(二重温度 τ_gs×T の T 成分) (default: 4.0)')
    parser.add_argument('--distill-standardize', action='store_true', default=True,
                        help='KD項の両ロジットをZ-score標準化する(既定ON。'
                             '二重温度崩壊を防ぐため必須) (default: True)')
    parser.add_argument('--no-distill-standardize', dest='distill_standardize',
                        action='store_false',
                        help='ロジット標準化を無効化する(非推奨)')
    parser.add_argument('--distill-teacher-transform', type=str,
                        default='cifar10-resnet18-hf',
                        help='teacher入力正規化のキー(TEACHER_TRANSFORMS参照) '
                             "(default: cifar10-resnet18-hf)")
    parser.add_argument('--distill-pseudo-labels', action='store_true',
                        help='hard項の正解ラベルをteacher argmax(疑似ラベル)にする'
                             '(完全ラベルフリー) (default: off)')
    parser.add_argument('--distill-cache', action='store_true', default=True,
                        help='teacherロジットを data-teacher-cache/ にキャッシュする '
                             '(default: True)')
    parser.add_argument('--no-distill-cache', dest='distill_cache',
                        action='store_false',
                        help='teacherロジットのキャッシュを無効化する(常時再計算)')

    args = parser.parse_args()

    if args.interactive and not args.resume:
        parser.error('--interactive には --resume が必要です')

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # 学習率・weight decayのデフォルト設定（Table 6準拠）
    if args.lr is None:
        args.lr = 0.01 if args.dataset == 'mnist' else 0.02
    if args.weight_decay is None:
        args.weight_decay = 0.0 if args.dataset == 'mnist' else 0.002

    # シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データ読み込み
    train_loader, test_loader, train_set = load_dataset(args)
    print(f'データセット: {args.dataset}')
    print(f'閾値数: {args.n_thresholds} (入力ビット数: '
          f'{args.n_thresholds if args.n_thresholds > 0 else 1})')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル')

    # モデル構築
    if args.dataset == 'mnist':
        model, tau = build_mnist_model(args)
    else:
        model, tau = build_cifar10_model(args)

    # パラメータ統計
    total_gates = sum(
        m.num_gates for m in model.modules() if isinstance(m, LutConv2d)
    )
    total_neurons = sum(
        m.num_neurons for m in model.modules() if isinstance(m, LutLayer)
    )
    total_params = sum(p.numel() for p in model.parameters())

    param_labels = {'softmax': 'softmax (DLGN方式)', 'independent': 'independent (BinaryBrain方式)', 'iwp': 'iwp (LDDLGN方式)', 'softmax16g': 'softmax16g (16ゲート真理値表, conv表現力回復)'}
    param_label = param_labels[args.parameterization]
    print(f'\nモデル構成: k={args.k}, tree_depth={args.tree_depth}, '
          f'n_inputs={args.n_inputs}')
    print(f'パラメータ化: {param_label}')
    print(f'Conv総ゲート数: {total_gates:,}')
    print(f'Dense総ゲート数: {total_neurons:,}')
    print(f'総ロジックゲート数: {total_gates + total_neurons:,} (LUT{args.n_inputs})')
    print(f'総パラメータ数: {total_params:,}')
    print(f'温度 τ = {tau:.1f}')
    print(model)

    # オプティマイザ（AdamW: Decoupled Weight Decay）
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    tau_sched = None
    if args.tau_anneal and args.selection in ('hard_st', 'gumbel_st'):
        from cage_scheduler import TauAnnealScheduler
        _total = (args.num_iterations if args.num_iterations is not None
                  else (args.epochs or 30) * 500)
        tau_sched = TauAnnealScheduler(
            model, tau_0=args.tau_b0, tau_min=args.tau_b_min,
            total_steps=_total,
            anneal_noise=(args.selection == 'gumbel_st'),
        )

    # ============================================================
    # 知識蒸留(KD)セットアップ(--distill-teacher指定時のみ)
    # ============================================================
    # teacherロジットを事前計算し、train_loaderをpacked-target([B,1+C])へ差し替える。
    # 評価(test_loader)は素の正解ラベルのまま=一切変更しない。
    distill_on = args.distill_teacher is not None
    if distill_on:
        from griffith_logic.convert.cache import (
            PackedDistillDataset,
            get_or_compute_logits,
            teacher_cache_key,
        )
        from griffith_logic.convert.distill import make_kd_loss
        from griffith_logic.convert.teacher import (
            TEACHER_TRANSFORMS,
            resolve_teacher,
        )

        if args.distill_teacher_transform not in TEACHER_TRANSFORMS:
            parser.error(
                f'未知の --distill-teacher-transform: '
                f'{args.distill_teacher_transform!r} '
                f'(対応: {sorted(TEACHER_TRANSFORMS)})'
            )

        # teacher解決 + 正規化transform上書き(teacher側で正規化する規約)
        teacher = resolve_teacher(args.distill_teacher, device=args.device)
        teacher.transform = TEACHER_TRANSFORMS[args.distill_teacher_transform]
        n_classes = 10

        # teacher推論用の生[0,1] train dataset(student と同一順序、augmentationなし)
        raw_train = build_raw_train_set(args)

        # teacherロジットを計算 or キャッシュからロード(index整列)
        key = teacher_cache_key(
            teacher.fingerprint, args.dataset,
            {'transform': args.distill_teacher_transform},
            split='train', n=len(raw_train),
        )
        print(f'\n[KD] teacher={args.distill_teacher} '
              f'transform={args.distill_teacher_transform} '
              f'α={args.distill_alpha} T={args.distill_temperature} '
              f'standardize={args.distill_standardize} '
              f'pseudo_labels={args.distill_pseudo_labels}')
        print(f'[KD] teacherロジット計算/ロード中(cache={args.distill_cache}, '
              f'key={key})...')
        if args.distill_cache:
            teacher_logits = get_or_compute_logits(
                teacher, raw_train, key=key, device=args.device,
            )
        else:
            import tempfile
            teacher_logits = get_or_compute_logits(
                teacher, raw_train, key=key, device=args.device,
                root=tempfile.mkdtemp(prefix='md-distill-nocache-'),
            )
        print(f'[KD] teacherロジット shape={tuple(teacher_logits.shape)}')

        # student(二値化済み)train setをpacked-targetで包み、loaderを再構築
        packed_train = PackedDistillDataset(
            train_set, teacher_logits,
            pseudo_labels=args.distill_pseudo_labels,
        )
        use_pin_memory = args.device == 'cuda'
        train_loader = torch.utils.data.DataLoader(
            packed_train, batch_size=args.batch_size, shuffle=True,
            pin_memory=use_pin_memory, drop_last=True, num_workers=2,
        )

        # packed-target契約に合わせたKD損失へ差し替え。
        # target=[B,1+C]: 列0=ラベル(long化はloss内)、列1..=teacherロジット。
        _kd = make_kd_loss(
            alpha=args.distill_alpha,
            temperature=args.distill_temperature,
            standardize=args.distill_standardize,
            kind='kl',
        )

        def loss_fn(output, target):
            labels = target[..., 0]
            teacher_l = target[..., 1:1 + n_classes]
            return _kd(output, labels, teacher_l)

    # エポック / イテレーション の解決
    use_epochs = args.epochs is not None
    if use_epochs and args.num_iterations is not None:
        parser.error('--epochs と --num-iterations は同時に指定できません')
    if not use_epochs and args.num_iterations is None:
        args.epochs = 50  # どちらも未指定ならエポックのデフォルト
        use_epochs = True

    # ログファイル
    log, log_file = setup_logger('train_cnn', args.dataset, args.log_dir)
    log(f'START dataset={args.dataset} k={args.k} tree_depth={args.tree_depth} '
        f'n_inputs={args.n_inputs} tau={tau:.1f} lr={args.lr} '
        f'weight_decay={args.weight_decay} bs={args.batch_size} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iterations=" + str(args.num_iterations)} '
        f'seed={args.seed} device={args.device}')

    # チェックポイント
    ckpt_prefix = f'cnn_{args.dataset}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}

    # チェックポイントから復元
    best_test_acc = 0.0
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(
            args.resume, model, optimizer, device=args.device,
        )
        start_step = ckpt_info['step']
        best_test_acc = ckpt_info['best_metric']
        print(f'  ステップ {start_step} から再開 (best_acc={best_test_acc:.4f})')

        if args.interactive:
            interactive_classify(model, test_loader, args.device, args.dataset)
            return

    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'{args.dataset}_k{args.k}_d{args.tree_depth}'
                f'_n{args.n_inputs}_e{args.epochs or args.num_iterations}')
    wandb_tags = ['cnn', args.dataset, f'n{args.n_inputs}']
    if distill_on:
        run_name = f'kd_{run_name}'
        wandb_tags.append('kd')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=wandb_tags,
        group=args.wandb_group,
    )
    batches_per_epoch = len(train_loader)

    if use_epochs:
        num_iterations = args.epochs * batches_per_epoch
        eval_freq = args.eval_freq if args.eval_freq is not None else 1
        print(f'\n訓練開始: {args.epochs}エポック '
              f'({batches_per_epoch}バッチ/エポック, '
              f'評価={eval_freq}エポックごと)')
    else:
        num_iterations = args.num_iterations
        eval_freq = args.eval_freq if args.eval_freq is not None else 2000
        print(f'\n訓練開始: {num_iterations:,}イテレーション'
              f' ({num_iterations / batches_per_epoch:.1f}エポック相当, '
              f'評価={eval_freq:,}イテレーションごと)')

    print('=' * 70)

    train_start = time.time()

    if use_epochs:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f'epoch {epoch}/{args.epochs}',
                leave=False,
            )
            for x, y in pbar:
                x = x.to(torch.float32).to(args.device)
                y = y.to(args.device)
                loss = train_step(model, x, y, loss_fn, optimizer, tau_sched,
                                  spectral_reg=args.spectral_reg)
                epoch_loss += loss
                elapsed_t = time.time() - train_start
                progress = (epoch - 1 + (pbar.n + 1) / len(train_loader)) / args.epochs
                remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
                pbar.set_postfix(loss=f'{loss:.4f}', eta=eta_str)

            avg_loss = epoch_loss / batches_per_epoch

            # 時間計算
            elapsed = time.time() - train_start
            sec_per_epoch = elapsed / epoch
            remaining = sec_per_epoch * (args.epochs - epoch)
            eta = datetime.now(JST) + timedelta(seconds=remaining)

            # 評価
            if epoch % eval_freq == 0 or epoch == args.epochs:
                test_acc = evaluate(model, test_loader, args.device, mode=False,
                                   desc=f'eval(hard) {epoch}/{args.epochs}')
                test_acc_soft = evaluate(model, test_loader, args.device, mode=True,
                                        desc=f'eval(soft) {epoch}/{args.epochs}')

                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc

                # チェックポイント保存
                step_now = epoch * batches_per_epoch
                save_checkpoint(latest_path, model, optimizer, None,
                                step_now, best_test_acc, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    step_now, best_test_acc, ckpt_metadata)

                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')

                print(
                    f'[epoch {epoch:>3d}/{args.epochs}] '
                    f'loss={avg_loss:.4f}  '
                    f'test_acc(hard)={test_acc:.4f}  '
                    f'test_acc(soft)={test_acc_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                    f'test_acc_hard={test_acc:.4f} test_acc_soft={test_acc_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'epoch': epoch,
                        'loss': avg_loss,
                        'test_acc_hard': test_acc,
                        'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })
    else:
        pbar = tqdm(
            enumerate(load_n(train_loader, num_iterations)),
            desc='train',
            total=num_iterations,
        )
        for i, (x, y) in pbar:
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)

            model.train()
            loss = train_step(model, x, y, loss_fn, optimizer, tau_sched,
                                  spectral_reg=args.spectral_reg)

            epoch_now = (i + 1) / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / (i + 1)
            remaining_sec = sec_per_iter * (num_iterations - (i + 1))
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{loss:.4f}', epoch=f'{epoch_now:.1f}', eta=eta_str)

            if (i + 1) % eval_freq == 0:
                test_acc = evaluate(model, test_loader, args.device, mode=False,
                                    desc=f'eval(hard) {i+1}/{num_iterations}')
                test_acc_soft = evaluate(model, test_loader, args.device, mode=True,
                                         desc=f'eval(soft) {i+1}/{num_iterations}')

                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc

                # チェックポイント保存
                save_checkpoint(latest_path, model, optimizer, None,
                                i + 1, best_test_acc, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    i + 1, best_test_acc, ckpt_metadata)

                elapsed = time.time() - train_start
                sec_per_iter = elapsed / (i + 1)
                remaining = sec_per_iter * (num_iterations - (i + 1))
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')

                print(
                    f'\n[iter {i+1:,} / epoch {epoch_now:.1f}] '
                    f'loss={loss:.4f}  '
                    f'test_acc(hard)={test_acc:.4f}  '
                    f'test_acc(soft)={test_acc_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'ITER {i+1}/{num_iterations} epoch={epoch_now:.1f} '
                    f'loss={loss:.4f} test_acc_hard={test_acc:.4f} '
                    f'test_acc_soft={test_acc_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'iteration': i + 1,
                        'epoch': epoch_now,
                        'loss': loss,
                        'test_acc_hard': test_acc,
                        'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度(hard): {best_test_acc:.4f}  (総訓練時間: {total_time})')
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
