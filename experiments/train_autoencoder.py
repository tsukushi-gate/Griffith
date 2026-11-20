"""
griffith-logic オートエンコーダー訓練スクリプト

LUTベースのオートエンコーダーでMNIST画像を再構成する。
LutLayerの出力が自然に[0,1]範囲のため、ピクセル再構成に直接使用可能。

使用例:
    # 基本実行
    uv run python experiments/train_autoencoder.py --dataset mnist -k 8000 --bottleneck 256 -e 100

    # 小さいモデルで素早くテスト
    uv run python experiments/train_autoencoder.py --dataset mnist -k 2000 --bottleneck 64 -e 20

    # N=4 LUT
    uv run python experiments/train_autoencoder.py --dataset mnist --n-inputs 4 -k 4000 --bottleneck 128 -e 100

    # チェックポイントから再開
    uv run python experiments/train_autoencoder.py --dataset mnist -k 8000 --bottleneck 256 --resume checkpoints/xxx/latest.pt

    # 再構成デモ
    uv run python experiments/train_autoencoder.py --dataset mnist -k 8000 --bottleneck 256 --resume checkpoints/xxx/best.pt --interactive
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
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv

# train.pyからデータ読み込み関連をインポート
from train import DATASET_INFO, load_dataset, load_n
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_images,
    log_metrics,
)

from griffith_logic import LutLayer

# .envから環境変数を読み込み
load_dotenv()


# ============================================================
# モデル構築
# ============================================================

def build_autoencoder(args):
    """LUTオートエンコーダーを構築する。

    構成: Flatten → Encoder(4層) → Decoder(4層)
    GroupSumは不要（再構成タスクのため）。
    """
    info = DATASET_INFO[args.dataset]
    in_dim = info['in_dim']
    k = args.num_neurons
    bottleneck = args.bottleneck
    n = args.n_inputs
    device = args.device

    llkw = dict(
        n_inputs=n,
        device=device,
        grad_factor=args.grad_factor,
        connections=args.connections,
        parameterization=args.parameterization,
    )

    # Encoder: in_dim → k → k/2 → k/4 → bottleneck
    # Decoder: bottleneck → k/4 → k/2 → k → in_dim
    encoder_sizes = [k, k // 2, k // 4, bottleneck]
    decoder_sizes = [k // 4, k // 2, k, in_dim]

    layers = [torch.nn.Flatten()]

    # Encoder
    prev = in_dim
    for size in encoder_sizes:
        layers.append(LutLayer(in_dim=prev, out_dim=size, **llkw))
        prev = size

    # Decoder
    for size in decoder_sizes:
        layers.append(LutLayer(in_dim=prev, out_dim=size, **llkw))
        prev = size

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
    }
    param_label = param_labels[args.parameterization]
    print('オートエンコーダー構成:')
    print(f'  Encoder: {in_dim} → {" → ".join(str(s) for s in encoder_sizes)}')
    print(f'  Decoder: {bottleneck} → {" → ".join(str(s) for s in decoder_sizes)}')
    print(f'  N={n}入力LUT, パラメータ化: {param_label}')
    print(f'  総ロジックゲート数: {total_neurons:,} (LUT{n})')
    print(f'  総パラメータ数: {total_params:,}')
    print(model)

    return model


# ============================================================
# 訓練・評価
# ============================================================

def train_step(model, x, optimizer):
    """1バッチの訓練ステップ。ターゲットは入力自身。"""
    x_flat = x.view(x.size(0), -1)
    output = model(x)
    output = output.clamp(1e-6, 1 - 1e-6)
    loss = F.binary_cross_entropy(output, x_flat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, loader, device, mode=False, desc=None):
    """再構成品質を評価する。

    Args:
        mode: True=訓練モード（softmax重み）, False=推論モード（離散化）
        desc: プログレスバーの説明文

    Returns:
        (mse, bce) のタプル
    """
    orig_mode = model.training
    model.train(mode=mode)
    total_mse = 0.0
    total_bce = 0.0
    total_pixels = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    with torch.no_grad():
        for x, _ in it:
            x = x.to(torch.float32).to(device)
            x_flat = x.view(x.size(0), -1)
            if not mode:
                x = x.round()
            output = model(x)
            n_pixels = x_flat.numel()
            total_mse += F.mse_loss(output, x_flat, reduction='sum').item()
            output_clamped = output.clamp(1e-6, 1 - 1e-6)
            total_bce += F.binary_cross_entropy(
                output_clamped, x_flat, reduction='sum'
            ).item()
            total_pixels += n_pixels
            if desc and isinstance(it, tqdm):
                it.set_postfix(mse=f'{total_mse / total_pixels:.6f}')
    model.train(mode=orig_mode)
    return total_mse / total_pixels, total_bce / total_pixels


def save_reconstruction_grid(model, test_loader, device, epoch, output_dir):
    """元画像 vs 再構成画像のグリッドを保存する。"""
    model_mode = model.training
    x, _ = next(iter(test_loader))
    x = x[:8].to(torch.float32).to(device)

    with torch.no_grad():
        # Soft再構成
        model.train(mode=True)
        recon_soft = model(x).view(-1, 1, 28, 28)

        # Hard再構成
        model.train(mode=False)
        x_hard = x.round()
        recon_hard = model(x_hard).view(-1, 1, 28, 28)

    # 3行: 元画像 / soft再構成 / hard再構成
    grid = torchvision.utils.make_grid(
        torch.cat([x[:8], recon_soft[:8], recon_hard[:8]]),
        nrow=8, padding=2,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'recon_epoch{epoch:03d}.png')
    torchvision.utils.save_image(grid, path)
    print(f'  再構成画像を保存: {path}')

    model.train(mode=model_mode)
    return grid


# ============================================================
# ログ
# ============================================================

def setup_logger(script_name, dataset, log_dir='logs'):
    """訓練メトリクスのログファイルを作成する。"""
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
# 対話的再構成デモ
# ============================================================

def interactive_reconstruct(model, test_loader, device, dataset_name):
    """対話的再構成デモ。ランダムテストサンプルの再構成MSEを表示。"""
    model.eval()
    test_dataset = test_loader.dataset

    print(f'\n=== 再構成デモ ({dataset_name}) ===')
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
        x_flat = x_input.view(1, -1)

        with torch.no_grad():
            x_hard = x_input.round()
            output = model(x_hard)
            mse = F.mse_loss(output, x_flat).item()

        print(f'サンプル #{idx} (ラベル={y}): MSE={mse:.6f}')


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='griffith-logic オートエンコーダー訓練スクリプト'
    )

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=list(DATASET_INFO.keys()),
                        help='データセット')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='LUTの入力数 (default: 2)')
    parser.add_argument('-k', '--num-neurons', type=int, default=8000,
                        help='最大層幅 (default: 8000)')
    parser.add_argument('--bottleneck', type=int, default=256,
                        help='ボトルネック次元 (default: 256)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学習率 (default: 0.01)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数（--num-iterationsと排他）')
    parser.add_argument('-ni', '--num-iterations', type=int, default=None,
                        help='反復回数（--epochsと排他）')
    parser.add_argument('-ef', '--eval-freq', type=int, default=None,
                        help='評価頻度')
    parser.add_argument('--connections', type=str, default='random',
                        choices=['random', 'unique'],
                        help='接続方式 (default: random)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp'],
                        help='パラメータ化方式 (default: softmax)')
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
    parser.add_argument('--output-dir', type=str, default='outputs/autoencoder',
                        help='再構成画像の保存先 (default: outputs/autoencoder)')

    # チェックポイント・推論
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開（パス指定）')
    parser.add_argument('--interactive', action='store_true',
                        help='対話推論モード（--resume必須）')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')

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
        args.epochs = 100
        use_epochs = True

    batches_per_epoch = len(train_loader)

    if use_epochs:
        num_iterations = args.epochs * batches_per_epoch
        eval_freq = args.eval_freq if args.eval_freq is not None else 1
        print(f'モード: エポック ({args.epochs}エポック, {batches_per_epoch}バッチ/エポック, '
              f'評価={eval_freq}エポックごと)')
    else:
        num_iterations = args.num_iterations
        eval_freq = args.eval_freq if args.eval_freq is not None else 2000
        print(f'モード: イテレーション ({num_iterations:,}イテレーション, '
              f'評価={eval_freq:,}イテレーションごと)')

    # モデル構築
    model = build_autoencoder(args)
    if args.compile:
        model = torch.compile(model)
        print('torch.compile 有効')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ログファイル
    log, log_file = setup_logger('train_ae', args.dataset, args.log_dir)
    log(f'START dataset={args.dataset} n_inputs={args.n_inputs} '
        f'k={args.num_neurons} bottleneck={args.bottleneck} '
        f'lr={args.lr} bs={args.batch_size} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iterations=" + str(num_iterations)} '
        f'seed={args.seed} device={args.device}')

    # チェックポイント
    ckpt_prefix = f'ae_{args.dataset}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}

    # チェックポイントから復元
    start_step = 0
    best_test_mse = float('inf')
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(
            args.resume, model, optimizer, device=args.device,
        )
        start_step = ckpt_info['step']
        best_test_mse = ckpt_info['best_metric']
        print(f'  ステップ {start_step} から再開 (best_mse={best_test_mse:.6f})')

        if args.interactive:
            interactive_reconstruct(model, test_loader, args.device, args.dataset)
            return

    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'ae_{args.dataset}_k{args.num_neurons}_bn{args.bottleneck}'
                f'_n{args.n_inputs}_e{args.epochs or args.num_iterations}')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['autoencoder', args.dataset, f'n{args.n_inputs}'],
        group='autoencoder',
    )

    if use_epochs:
        print(f'\n訓練開始: {args.epochs}エポック ({batches_per_epoch}バッチ/エポック)')
        print('=' * 60)

        train_start = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f'epoch {epoch}/{args.epochs}',
                leave=False,
            )
            for x, _ in pbar:
                x = x.to(torch.float32).to(args.device)
                loss = train_step(model, x, optimizer)
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
                test_mse, test_bce = evaluate(
                    model, test_loader, args.device, mode=False,
                    desc=f'eval(hard) {epoch}/{args.epochs}',
                )
                test_mse_soft, test_bce_soft = evaluate(
                    model, test_loader, args.device, mode=True,
                    desc=f'eval(soft) {epoch}/{args.epochs}',
                )

                is_best = test_mse < best_test_mse
                if is_best:
                    best_test_mse = test_mse

                # チェックポイント保存
                step_now = epoch * batches_per_epoch
                save_checkpoint(latest_path, model, optimizer, None,
                                step_now, best_test_mse, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    step_now, best_test_mse, ckpt_metadata)

                # 再構成画像を保存（ローカル + wandb）
                grid = save_reconstruction_grid(
                    model, test_loader, args.device, epoch, args.output_dir,
                )
                if use_wandb:
                    from torchvision.transforms.functional import to_pil_image
                    log_images('autoencoder/reconstruction', [to_pil_image(grid)],
                               captions=[f'epoch {epoch}'])

                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')

                print(
                    f'[epoch {epoch:>3d}/{args.epochs}] '
                    f'loss={avg_loss:.4f}  '
                    f'test_mse={test_mse:.6f}  '
                    f'test_bce={test_bce:.4f}  '
                    f'test_mse(soft)={test_mse_soft:.6f}  '
                    f'test_bce(soft)={test_bce_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                    f'test_mse={test_mse:.6f} test_bce={test_bce:.4f} '
                    f'test_mse_soft={test_mse_soft:.6f} test_bce_soft={test_bce_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'epoch': epoch,
                        'loss': avg_loss,
                        'test_mse': test_mse,
                        'test_bce': test_bce,
                        'test_mse_soft': test_mse_soft,
                        'test_bce_soft': test_bce_soft,
                        'best_test_mse': best_test_mse,
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
        for i, (x, _) in pbar:
            x = x.to(torch.float32).to(args.device)
            loss = train_step(model, x, optimizer)

            epoch_now = (i + 1) / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / (i + 1)
            remaining_sec = sec_per_iter * (num_iterations - (i + 1))
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{loss:.4f}', epoch=f'{epoch_now:.1f}', eta=eta_str)

            if (i + 1) % eval_freq == 0:
                test_mse, test_bce = evaluate(
                    model, test_loader, args.device, mode=False,
                    desc=f'eval(hard) {i+1}/{num_iterations}',
                )
                test_mse_soft, test_bce_soft = evaluate(
                    model, test_loader, args.device, mode=True,
                    desc=f'eval(soft) {i+1}/{num_iterations}',
                )

                is_best = test_mse < best_test_mse
                if is_best:
                    best_test_mse = test_mse

                # チェックポイント保存
                save_checkpoint(latest_path, model, optimizer, None,
                                i + 1, best_test_mse, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    i + 1, best_test_mse, ckpt_metadata)

                # 再構成画像を保存（ローカル + wandb）
                epoch_int = max(1, int(epoch_now))
                grid = save_reconstruction_grid(
                    model, test_loader, args.device, epoch_int, args.output_dir,
                )
                if use_wandb:
                    from torchvision.transforms.functional import to_pil_image
                    log_images('autoencoder/reconstruction', [to_pil_image(grid)],
                               captions=[f'iter {i+1} (epoch {epoch_now:.1f})'])

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
                    f'test_mse={test_mse:.6f}  '
                    f'test_bce={test_bce:.4f}  '
                    f'test_mse(soft)={test_mse_soft:.6f}  '
                    f'test_bce(soft)={test_bce_soft:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )

                log(f'ITER {i+1}/{num_iterations} epoch={epoch_now:.1f} '
                    f'loss={loss:.4f} test_mse={test_mse:.6f} test_bce={test_bce:.4f} '
                    f'test_mse_soft={test_mse_soft:.6f} test_bce_soft={test_bce_soft:.4f} '
                    f'best={str(is_best).lower()} '
                    f'elapsed={elapsed_str} eta={remain_str}')
                if use_wandb:
                    log_metrics({
                        'iteration': i + 1,
                        'epoch': epoch_now,
                        'loss': loss,
                        'test_mse': test_mse,
                        'test_bce': test_bce,
                        'test_mse_soft': test_mse_soft,
                        'test_bce_soft': test_bce_soft,
                        'best_test_mse': best_test_mse,
                    })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テストMSE: {best_test_mse:.6f}  (総訓練時間: {total_time})')
    log(f'DONE best_test_mse={best_test_mse:.6f} total_time={total_time}')
    log_file.close()
    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_test_mse': best_test_mse},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
