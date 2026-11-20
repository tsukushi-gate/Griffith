"""
griffith-logic 訓練スクリプト

使用例:
    # エポックモード（デフォルト）: エポック単位で訓練、毎エポック評価
    uv run python experiments/train.py --dataset mnist --n-inputs 2 -k 64000 -l 6 -t 30 -e 50

    # エポックモード: 5エポックごとに評価
    uv run python experiments/train.py --dataset mnist --n-inputs 2 -k 64000 -l 6 -t 30 -e 50 -ef 5

    # イテレーションモード（difflogic互換）
    uv run python experiments/train.py --dataset mnist --n-inputs 2 -k 64000 -l 6 -t 30 -ni 100000

    # N=4
    uv run python experiments/train.py --dataset mnist --n-inputs 4 -k 16000 -l 4 -t 30 -e 30

    # N=6 (FPGA LUT6互換)
    uv run python experiments/train.py --dataset mnist --n-inputs 6 -k 4000 -l 4 -t 30 -e 30

    # torch.compile有効化
    uv run python experiments/train.py --dataset mnist -e 10 --compile
"""

import argparse
import itertools
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
    log_ir_artifact,
    log_metrics,
)

from griffith_logic import GroupSum, LutLayer
from griffith_logic.training.param_groups import build_param_groups

# .envから環境変数を読み込み
load_dotenv()


# ============================================================
# データセット
# ============================================================

class MNISTRemoveBorderTransform:
    """
    28x28 MNIST画像から黒いボーダーを適応的に除去して20x20にする変換。
    difflogicと同一のアルゴリズム: 各画像ごとに上下左右の黒行/列を検出し、
    合計8ピクセルずつ除去する。Resize(補間)と異なり元のピクセル値を保持する。
    """
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # 水平方向: 完全に黒い行を検出
        horizontal_black_lines = (image == 0.).all(dim=2)
        top_black = 0
        while horizontal_black_lines[0, top_black] and top_black < 14:
            top_black += 1
        bottom_black = 0
        while horizontal_black_lines[0, -bottom_black - 1] and bottom_black < 14:
            bottom_black += 1
        assert top_black + bottom_black >= 8, (top_black, bottom_black)
        while top_black + bottom_black >= 10:
            if top_black > 0:
                top_black -= 1
            if bottom_black > 0:
                bottom_black -= 1
        if top_black + bottom_black == 9:
            if top_black > 0:
                top_black -= 1
            else:
                bottom_black -= 1
        assert top_black + bottom_black == 8, (top_black, bottom_black)
        image = image[:, top_black:28 - bottom_black]

        # 垂直方向: 完全に黒い列を検出
        vertical_black_lines = (image == 0.).all(dim=1)
        left_black = 0
        while vertical_black_lines[0, left_black] and left_black < 14:
            left_black += 1
        right_black = 0
        while vertical_black_lines[0, -right_black - 1] and right_black < 14:
            right_black += 1
        assert left_black + right_black >= 8, (left_black, right_black)
        while left_black + right_black >= 10:
            if left_black > 0:
                left_black -= 1
            if right_black > 0:
                right_black -= 1
        if left_black + right_black == 9:
            if left_black > 0:
                left_black -= 1
            else:
                right_black -= 1
        assert left_black + right_black == 8, (left_black, right_black)
        image = image[:, :, left_black:28 - right_black]

        return image


class ThresholdTransform:
    """ピクセル値を複数の閾値で二値化し結合する変換（pickle対応）"""
    def __init__(self, threshold_count):
        self.threshold_count = threshold_count

    def __call__(self, x):
        return torch.cat(
            [(x > (i + 1) / (self.threshold_count + 1)).float()
             for i in range(self.threshold_count)], dim=0
        )


DATASET_INFO = {
    'mnist': {'in_dim': 784, 'num_classes': 10},
    'mnist20x20': {'in_dim': 400, 'num_classes': 10},
    'fashion-mnist': {'in_dim': 784, 'num_classes': 10},
    'cifar-10-3-thresholds': {'in_dim': 3 * 32 * 32 * 3, 'num_classes': 10},
    'cifar-10-31-thresholds': {'in_dim': 3 * 32 * 32 * 31, 'num_classes': 10},
    'cifar-100-3-thresholds': {'in_dim': 3 * 32 * 32 * 3, 'num_classes': 100},
    'cifar-100-31-thresholds': {'in_dim': 3 * 32 * 32 * 31, 'num_classes': 100},
    # 論理関数タスク: 全2^n真理値表を学習/評価に用いる(train==test)。
    # 2クラス分類として扱う(GroupSum(num_classes=2)が層幅を割り切れる必要がある)。
    'xor': {'in_dim': 2, 'num_classes': 2, 'logic_n_inputs': 2},
    'xor3': {'in_dim': 3, 'num_classes': 2, 'logic_n_inputs': 3},
    'maj3': {'in_dim': 3, 'num_classes': 2, 'logic_n_inputs': 3},
    'rnd4': {'in_dim': 4, 'num_classes': 2, 'logic_n_inputs': 4},
    # parity-in-noise(系統,配線学習の本命タスク): 関連3ビットのparity
    # (=xor3と同じラベル関数)を PARITY3N_N_NOISE 個の無関係ノイズビットに埋め込む。
    # 定義はtests/test_learnable_wiring.py::TestLearnableWiringConvergenceSmoke
    # (N_RELEVANT=3, N_NOISE=9)を出典としつつ、本タスクは全2^n真理値表を
    # 列挙する既存のlogic_task_dataset方式に合わせるためノイズ数を3に固定
    # (2^(3+3)=64パターンを閾値なく全列挙。ノイズ数が大きいほど固定配線が
    # 不利になる効果は同じ方向に働く)。固定配線では層1のニューロンが
    # 関連3ビット全てに到達する保証がなく停滞、学習配線で到達可能になる
    # ことが効果を最も顕著に示す。
    'parity3n': {
        'in_dim': 3 + 3, 'num_classes': 2, 'logic_n_inputs': 3 + 3,
    },
}

# 論理関数タスク名の集合(load_dataset/build_modelの分岐に使う)
LOGIC_TASKS = frozenset({'xor', 'xor3', 'maj3', 'rnd4', 'parity3n'})

# parity3n: 関連ビット数・ノイズビット数(configurable化は不要という判断のため固定値。
# tests/test_learnable_wiring.pyのN_RELEVANT/N_NOISEと同名で出典を明示)。
PARITY3N_N_RELEVANT = 3
PARITY3N_N_NOISE = 3

# xor3/maj3/rnd4 は distill_undercap_3input.TARGETS と同一定義を流用する
# (verify_capacity_ladder.py L38の前例に倣うimport)。xor(2入力)のみここで追加。
from distill_undercap_3input import TARGETS as _UNDERCAP_TARGETS  # noqa: E402

LOGIC_TARGETS = {
    'xor': (2, lambda b: b[0] ^ b[1]),
    **_UNDERCAP_TARGETS,
    # parity3n: 先頭PARITY3N_N_RELEVANTビットのparity(=xor3と同じ関数)がラベル。
    # 残りPARITY3N_N_NOISEビットはラベルに寄与しないノイズ(全2^nに対し均等に列挙)。
    'parity3n': (
        PARITY3N_N_RELEVANT + PARITY3N_N_NOISE,
        lambda b: sum(b[:PARITY3N_N_RELEVANT]) % 2,
    ),
}


def logic_task_dataset(name: str, *, repeat: int = 1) -> torch.utils.data.TensorDataset:
    """論理関数タスクの全2^nパターンをTensorDatasetにする（train==test）。

    X: (2^n・repeat, n) float32 {0,1}。Y: (2^n・repeat,) long（2クラス分類ラベル）。
    parity3n も含め全て決定論的な全数列挙(ノイズビットも固定に2^n全パターンを走査)。
    """
    n_inputs, fn = LOGIC_TARGETS[name]
    x = torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=n_inputs)), dtype=torch.float32,
    )
    y = torch.tensor(
        [fn(tuple(int(v) for v in row)) for row in x.tolist()], dtype=torch.long,
    )
    if repeat > 1:
        x = x.repeat(repeat, 1)
        y = y.repeat(repeat)
    return torch.utils.data.TensorDataset(x, y)


def load_dataset(args):
    if args.dataset in LOGIC_TASKS:
        # 論理関数タスク: 全2^nパターンをtrain/test共通で使う(データセットが真理値表そのもの)。
        # バッチサイズに満たない場合にdrop_last=Trueで空になるのを避けるため、
        # 全パターンを繰り返し複製してエポックあたり・評価あたりのバッチ数を確保する
        # (train/testとも同一の真理値表を使う。論理関数タスクに汎化ギャップは無い)。
        n_inputs = DATASET_INFO[args.dataset]['logic_n_inputs']
        n_patterns = 1 << n_inputs
        repeat = max(1, math.ceil(args.batch_size / n_patterns))
        train_set = logic_task_dataset(args.dataset, repeat=repeat)
        test_set = logic_task_dataset(args.dataset, repeat=repeat)
    elif args.dataset == 'mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.MNIST(
            './data-mnist', train=True, download=True, transform=transforms
        )
        test_set = torchvision.datasets.MNIST(
            './data-mnist', train=False, transform=transforms
        )
    elif args.dataset == 'mnist20x20':
        # difflogic互換: Resize(補間)ではなくボーダー除去(crop)で20x20にする
        # 補間はピクセル値を中間値にしてしまい、バイナリ論理ゲートに不利
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            MNISTRemoveBorderTransform(),
        ])
        train_set = torchvision.datasets.MNIST(
            './data-mnist', train=True, download=True, transform=transforms
        )
        test_set = torchvision.datasets.MNIST(
            './data-mnist', train=False, transform=transforms
        )
    elif args.dataset == 'fashion-mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.FashionMNIST(
            './data-fashion-mnist', train=True, download=True, transform=transforms
        )
        test_set = torchvision.datasets.FashionMNIST(
            './data-fashion-mnist', train=False, transform=transforms
        )
    elif 'cifar-' in args.dataset:
        # 'cifar-10-3-thresholds' -> 3, 'cifar-100-31-thresholds' -> 31
        threshold_count = int(args.dataset.split('-thresholds')[0].split('-')[-1])
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ThresholdTransform(threshold_count),
        ])
        if 'cifar-100' in args.dataset:
            dataset_cls = torchvision.datasets.CIFAR100
            data_dir = './data-cifar-100'
        else:
            dataset_cls = torchvision.datasets.CIFAR10
            data_dir = './data-cifar'
        train_set = dataset_cls(data_dir, train=True, download=True, transform=transforms)
        test_set = dataset_cls(data_dir, train=False, transform=transforms)
    else:
        raise ValueError(f'未対応データセット: {args.dataset}')

    # MPS では pin_memory 非対応
    use_pin_memory = args.device == 'cuda'

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2
    )
    return train_loader, test_loader


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
# モデル構築
# ============================================================

def build_model(args):
    info = DATASET_INFO[args.dataset]
    in_dim = info['in_dim']
    num_classes = info['num_classes']
    k = args.num_neurons
    l = args.num_layers
    n = args.n_inputs
    device = args.device

    # ニューロン数がクラス数で割り切れるか確認
    assert k % num_classes == 0, (
        f'ニューロン数({k})はクラス数({num_classes})で割り切れる必要があります'
    )

    llkw = dict(
        device=device,
        grad_factor=args.grad_factor,
        connections=args.connections,
        parameterization=args.parameterization,
        wiring=getattr(args, 'wiring', 'fixed'),
        selection=getattr(args, 'selection', 'soft'),
        wh_degree_cap=getattr(args, 'wh_degree_cap', None),
        residual_init=getattr(args, 'residual_init', False),
        estimator=getattr(args, 'estimator', 'sigmoid'),
        ri_mu=getattr(args, 'ri_mu', 1.2),
        ri_sigma=getattr(args, 'ri_sigma', 0.25),
    )

    # 層別fan-inプロファイル(W11、Tier E): --fanin-profile "6,2,2,6" で
    # 層ごとにn_inputsを指定(--n-inputsと排他)。未指定なら全層n。
    profile = getattr(args, 'fanin_profile', None)
    if profile:
        fanins = [int(v) for v in profile.split(',')]
        assert len(fanins) == l, (
            f'--fanin-profileの要素数({len(fanins)})が層数({l})と不一致'
        )
        assert all(v in (2, 4, 6) for v in fanins), fanins
    else:
        fanins = [n] * l

    layers = [torch.nn.Flatten()]
    layer0 = LutLayer(in_dim=in_dim, out_dim=k, n_inputs=fanins[0], **llkw)
    layers.append(layer0)
    # 学習配線は第1層のみ(RDDLGN LutStackと同じ定石)。隠れ層に適用すると
    # マッピング行列 k×(k·n) が層ごと数百MパラメータになりOOM(実測:
    # k=8000,n=6で384M×3層+Adam状態が24GB超)
    hidden_kw = dict(llkw)
    hidden_kw['wiring'] = 'fixed'
    hidden_layers = []
    for li in range(l - 1):
        hl = LutLayer(in_dim=k, out_dim=k, n_inputs=fanins[li + 1],
                      **hidden_kw)
        hidden_layers.append(hl)
        layers.append(hl)
    layers.append(GroupSum(num_classes, tau=args.tau))

    # WARP設定をLutLayerに適用
    if args.parameterization == 'warp':
        for lut in [layer0] + hidden_layers:
            lut.tau_warp = args.tau_warp
            lut.gumbel = not args.no_gumbel

    model = torch.nn.Sequential(*layers).to(device)

    # パラメータ数の表示
    total_neurons = sum(
        m.num_neurons for m in model.modules() if isinstance(m, LutLayer)
    )
    total_params = sum(p.numel() for p in model.parameters())
    param_labels = {
        'softmax': 'softmax (DLGN方式)',
        'independent': 'independent (BinaryBrain方式)',
        'iwp': 'iwp (LDDLGN方式)',
        'warp': 'warp (WARP方式)',
    }
    param_label = param_labels[args.parameterization]
    print(f'モデル構成: {l}層 × {k}ニューロン, N={n}入力LUT')
    print(f'パラメータ化: {param_label}')
    print(f'総ロジックゲート数: {total_neurons:,} (LUT{n})')
    print(f'総パラメータ数: {total_params:,}')
    print(model)

    return model


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
# メイン
# ============================================================

def interactive_classify(model, test_loader, device, dataset_name):
    """対話的分類デモ。ランダムテストサンプルで予測を表示。"""
    model.eval()
    test_dataset = test_loader.dataset

    # クラス名（データセットに応じて）
    if 'cifar-100' in dataset_name:
        class_names = None  # 100クラスは表示が煩雑なのでインデックスのみ
    elif 'cifar' in dataset_name:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif 'fashion' in dataset_name:
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
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
    parser = argparse.ArgumentParser(description='griffith-logic 訓練スクリプト')

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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学習率 (default: 0.01)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数（--num-iterationsと排他）')
    parser.add_argument('-ni', '--num-iterations', type=int, default=None,
                        help='反復回数（--epochsと排他）')
    parser.add_argument('-ef', '--eval-freq', type=int, default=None,
                        help='評価頻度（エポック時=Nエポックごと[default:1], '
                             'イテレーション時=Nイテレーションごと[default:2000]）')
    parser.add_argument('--wandb-group', type=str, default='lut',
                        help="wandbのgroup(既定'lut')。他実験と同一groupで比較したい時に指定する")
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help='接続方式 (default: random)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'],
                        help='パラメータ化方式 (default: softmax)')
    parser.add_argument('--estimator', type=str, default='sigmoid',
                        choices=['sigmoid', 'sin'],
                        help='エントリ確率の推定器ρ (default: sigmoid。'
                             'sinは論文arXiv:2510.03250 §C.1.1のsin01。'
                             'independentパラメータ化のみ対応)')
    parser.add_argument('--ri-mu', type=float, default=1.2,
                        help='estimator=sin時のresidual init平均μ (default: 1.2)')
    parser.add_argument('--ri-sigma', type=float, default=0.25,
                        help='estimator=sin時のresidual init標準偏差σ '
                             '(default: 0.25)')
    parser.add_argument('--wiring', type=str, default='fixed',
                        choices=['fixed', 'learnable'],
                        help='入力接続の方式 (default: fixed。'
                             'learnableでDWN方式の配線学習を有効化)')
    parser.add_argument('--selection', type=str, default='soft',
                        choices=['soft', 'hard_st', 'gumbel_st'],
                        help='ゲート選択方式 (default: soft)')
    parser.add_argument('--tau-anneal', action='store_true',
                        help='TauAnnealScheduler(τ_b幾何減衰+ノイズ減衰)を有効化')
    parser.add_argument('--tau-b0', type=float, default=1.0)
    parser.add_argument('--tau-b-min', type=float, default=0.2)
    parser.add_argument('--spectral-reg', type=float, default=0.0,
                        help='Spectral正則化係数λ(DWN §3.4。0で無効)')
    parser.add_argument('--progressive-wiring', action='store_true',
                        help='漸進配線ハード化(SparseLUT式2フェーズ。'
                             '--wiring learnable時のみ)')
    parser.add_argument('--pw-switch', type=float, default=0.8,
                        help='hard切替の訓練進捗率 (default: 0.8)')
    parser.add_argument('--pw-tau0', type=float, default=1.0)
    parser.add_argument('--pw-tau-min', type=float, default=0.05)
    parser.add_argument('--residual-init', action='store_true',
                        help='Residual Init(pass-throughバイアス初期化。'
                             'N>2はindependent厳密/softmax減衰)')
    parser.add_argument('--wh-degree-cap', type=int, default=None,
                        help='WH係数の度数キャップ(warp専用。PolyLUT式'
                             '低次junta bias。例: n=6でd=3)')
    parser.add_argument('--fanin-profile', type=str, default=None,
                        help='層別fan-in指定(カンマ区切り。例: "6,2,2,6"。'
                             '層数と同じ要素数が必要。--n-inputsより優先)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Adam weight decay(default: 0=従来挙動)')
    parser.add_argument('--tau-warp', type=float, default=1.0,
                        help='WARP温度パラメータ (default: 1.0)')
    parser.add_argument('--no-gumbel', action='store_true',
                        help='WARPのGumbel smoothingを無効化')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数 (default: 1.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compileでモデルを最適化')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先ディレクトリ (default: logs)')

    # チェックポイント・推論
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開（パス指定）')
    parser.add_argument('--interactive', action='store_true',
                        help='対話推論モード（--resume必須）')
    parser.add_argument('--export-ir', action='store_true',
                        help='訓練終了時にIR(中間表現)JSONをエクスポートしwandbにも記録')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--init-ckpt', type=str, default=None,
                        help='T4: 訓練開始前にこのckptから**重みのみ**をロードする'
                             '(optimizerは新規作成)。AP法(train_ap.py)のckptをBPで'
                             '継続学習する用途を想定。ckpt metadata["args"]の'
                             'dataset/n_inputs/num_neurons/num_layersが現行argsと'
                             '不一致ならraiseする(既定None=非回帰)。')

    args = parser.parse_args()

    if args.interactive and not args.resume:
        parser.error('--interactive には --resume が必要です')
    if args.resume and args.init_ckpt:
        parser.error('--resume と --init-ckpt は同時に指定できません'
                      '(--resumeは訓練再開、--init-ckptは他ckptからの重み初期化で用途が異なります)')

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

    # エポック / イテレーション の解決
    use_epochs = args.epochs is not None
    if use_epochs and args.num_iterations is not None:
        parser.error('--epochs と --num-iterations は同時に指定できません')
    if not use_epochs and args.num_iterations is None:
        args.epochs = 50  # どちらも未指定ならエポックのデフォルト
        use_epochs = True

    batches_per_epoch = len(train_loader)

    if use_epochs:
        num_iterations = args.epochs * batches_per_epoch
        # エポックモードでは eval_freq はエポック単位（デフォルト: 毎エポック）
        eval_freq = args.eval_freq if args.eval_freq is not None else 1
        print(f'モード: エポック ({args.epochs}エポック, {batches_per_epoch}バッチ/エポック, '
              f'評価={eval_freq}エポックごと)')
    else:
        num_iterations = args.num_iterations
        # イテレーションモードでは eval_freq はイテレーション単位
        eval_freq = args.eval_freq if args.eval_freq is not None else 2000
        print(f'モード: イテレーション ({num_iterations:,}イテレーション, '
              f'評価={eval_freq:,}イテレーションごと)')

    # モデル構築
    model = build_model(args)

    # 重みロード(T4): --init-ckpt指定時、モデル構築直後に**重みのみ**をロードする
    # (optimizerは新規作成。AP法(train_ap.py等)のckptをBPで継続学習する用途)。
    # train_kd_charlm_rddlgn.pyの--init-ckpt実装と同一スキームで、
    # ckpt metadataのdataset/n_inputs/num_neurons/num_layers/parameterization/
    # connections/wiringを先に検証してからload_checkpointでmodelへ流し込む
    # (shape不一致によるload_state_dict例外を分かりやすいエラーメッセージに変換
    # するため)。特にparameterizationはAP法(train_ap.py)がindependentを強制する
    # ため、BP側でも--parameterization independentを明示しないとLutLayer.weights
    # の形状が一致しない(softmaxのn_inputs=2は16ゲート化でshapeが異なる)。
    # optimizer_state_dict=NoneのAP法ckptも受理する(load_checkpointはoptimizer=None
    # 時にoptimizer復元をスキップする)。
    init_ckpt_step = 0
    if args.init_ckpt is not None:
        ckpt_probe = torch.load(args.init_ckpt, map_location=args.device, weights_only=False)
        ckpt_args = ckpt_probe.get('metadata', {}).get('args', {})
        mismatches = []
        for key in ('dataset', 'n_inputs', 'num_neurons', 'num_layers',
                    'parameterization', 'connections', 'wiring'):
            ckpt_val = ckpt_args.get(key)
            cur_val = getattr(args, key, None)
            if ckpt_val != cur_val:
                mismatches.append(f'{key}: ckpt={ckpt_val!r} vs current={cur_val!r}')
        if mismatches:
            raise ValueError(
                f'--init-ckpt {args.init_ckpt} は現行argsと構成が一致しません: '
                f'{"; ".join(mismatches)}。'
                ' dataset/n_inputs/num_neurons/num_layers/parameterization/'
                'connections/wiringが一致するckptを指定してください'
                '(AP法ckptはparameterization=independentを強制するため、'
                'BP側にも--parameterization independentの明示指定が必要です)。'
            )
        del ckpt_probe  # model.load_state_dictはload_checkpoint内で行うため不要
        init_info = load_checkpoint(args.init_ckpt, model, optimizer=None, scheduler=None, device=args.device)
        init_ckpt_step = init_info['step']
        print(
            f'--init-ckpt {args.init_ckpt} から重みをロードしました'
            f' (step={init_info["step"]}, best_metric={init_info["best_metric"]})'
        )

    if args.compile:
        model = torch.compile(model)
        print('torch.compile 有効')
    loss_fn = torch.nn.CrossEntropyLoss()
    # IWP系(ω∈[0,1]の直接パラメータ化)はweight decayが離散化を悪化させるため
    # wd=0グループへ分離する(arXiv:2510.03250 §C.1)
    optimizer = torch.optim.Adam(build_param_groups(model, args.weight_decay),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    tau_sched = None
    _total = (args.num_iterations if args.num_iterations
              else args.epochs * max(1, len(train_loader)))
    if args.tau_anneal and args.selection in ('hard_st', 'gumbel_st'):
        from cage_scheduler import TauAnnealScheduler
        tau_sched = TauAnnealScheduler(
            model, tau_0=args.tau_b0, tau_min=args.tau_b_min,
            total_steps=_total,
            anneal_noise=(args.selection == 'gumbel_st'),
        )

    # 漸進配線ハード化(W10。--wiring learnable時のみ有効)
    if args.progressive_wiring:
        assert args.wiring == 'learnable', (
            '--progressive-wiring は --wiring learnable と併用してください'
        )
        from cage_scheduler import ProgressiveWiringScheduler
        pw_sched = ProgressiveWiringScheduler(
            model, total_steps=_total, switch_at=args.pw_switch,
            tau_0=args.pw_tau0, tau_min=args.pw_tau_min,
        )
        if tau_sched is None:
            tau_sched = pw_sched
        else:
            # 両スケジューラを合成(train_stepは単一の.step()を呼ぶ)
            class _Composite:
                def __init__(self, scheds):
                    self.scheds = scheds

                def step(self):
                    for sc in self.scheds:
                        sc.step()
            tau_sched = _Composite([tau_sched, pw_sched])


    # ログファイル
    log, log_file = setup_logger('train', args.dataset, args.log_dir)
    log(f'START dataset={args.dataset} n_inputs={args.n_inputs} '
        f'k={args.num_neurons} l={args.num_layers} tau={args.tau} '
        f'lr={args.lr} bs={args.batch_size} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iterations=" + str(num_iterations)} '
        f'seed={args.seed} device={args.device}')

    # チェックポイント
    ckpt_prefix = f'train_{args.dataset}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}

    # チェックポイントから復元
    start_step = 0
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
    run_name = (f'{args.dataset}_k{args.num_neurons}_l{args.num_layers}'
                f'_n{args.n_inputs}_e{args.epochs or args.num_iterations}')
    # T4: --init-ckpt指定時のみrun_nameに追記する(非指定時は既存run_nameとbitwise同一)。
    if args.init_ckpt is not None:
        run_name += f'_initstep{init_ckpt_step}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={**vars(args), 'init_ckpt_step': init_ckpt_step},
        model=model,
        device=args.device,
        tags=['lut', args.dataset, f'n{args.n_inputs}'],
        group=args.wandb_group,
    )

    if use_epochs:
        print(f'\n訓練開始: {args.epochs}エポック ({batches_per_epoch}バッチ/エポック)')
        print('=' * 60)

        train_start = time.time()

        for epoch in range(1, args.epochs + 1):
            # エポック訓練
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
                train_acc = evaluate(model, train_loader, args.device, mode=False,
                                     desc=f'train(hard) {epoch}/{args.epochs}')
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
                    f'train_acc={train_acc:.4f}  '
                    f'test_acc={test_acc:.4f}  '
                    f'test_acc(soft)={test_acc_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                    f'train_acc={train_acc:.4f} test_acc={test_acc:.4f} '
                    f'test_acc_soft={test_acc_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'epoch': epoch,
                        'loss': avg_loss,
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })
    else:
        print(f'\n訓練開始: {num_iterations:,}イテレーション'
              f' ({num_iterations / batches_per_epoch:.1f}エポック相当)')
        print('=' * 60)

        train_start = time.time()

        pbar = tqdm(
            enumerate(load_n(train_loader, num_iterations)),
            desc='train',
            total=num_iterations,
        )
        for i, (x, y) in pbar:
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)

            loss = train_step(model, x, y, loss_fn, optimizer, tau_sched,
                                  spectral_reg=args.spectral_reg)

            epoch_now = (i + 1) / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / (i + 1)
            remaining_sec = sec_per_iter * (num_iterations - (i + 1))
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{loss:.4f}', epoch=f'{epoch_now:.1f}', eta=eta_str)

            if (i + 1) % eval_freq == 0:
                train_acc = evaluate(model, train_loader, args.device, mode=False,
                                     desc=f'train(hard) {i+1}/{num_iterations}')
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
                    f'train_acc={train_acc:.4f}  '
                    f'test_acc={test_acc:.4f}  '
                    f'test_acc(soft)={test_acc_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'ITER {i+1}/{num_iterations} epoch={epoch_now:.1f} '
                    f'loss={loss:.4f} train_acc={train_acc:.4f} '
                    f'test_acc={test_acc:.4f} test_acc_soft={test_acc_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'iteration': i + 1,
                        'epoch': epoch_now,
                        'loss': loss,
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度: {best_test_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_test_acc={best_test_acc:.4f} total_time={total_time}')
    log_file.close()

    # IRエクスポート(最新重み。--export-ir指定時)
    if args.export_ir:
        try:
            from griffith_logic.export import export_ir
            from griffith_logic.ir import save_ir
            first_lut = next(
                m for m in model.modules() if isinstance(m, LutLayer)
            )
            model.eval()
            ir = export_ir(model, input_width=first_lut.in_dim, name=run_name)
            ir_path = save_ir(ir, os.path.join(run_dir, f'{run_name}.ir.json'))
            print(f'IRエクスポート: {ir_path}')
            if use_wandb:
                npz_path = ir_path[:-5] + '.npz'
                log_ir_artifact(
                    ir_path,
                    [npz_path] if os.path.exists(npz_path) else [],
                    name=run_name,
                )
        except Exception as e:
            print(f'IRエクスポート失敗: {e}')

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
