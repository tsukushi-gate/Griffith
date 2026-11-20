"""
BinaryBrain CNN Autoencoder 忠実再現スクリプト

BinaryBrainの MnistAeDifferentiableLutCnn.ipynb を mdlogic で忠実に再現。
Conv2d + MaxPooling + UpSampling によるEncoder-Decoderアーキテクチャ。

使用例:
    # デフォルト（BinaryBrainと同じ設定）
    uv run python experiments/train_autoencoder_cnn.py

    # エポック数変更
    uv run python experiments/train_autoencoder_cnn.py --epochs 10

    # N=4入力LUT
    uv run python experiments/train_autoencoder_cnn.py --n-inputs 4

    # チェックポイントから再開
    uv run python experiments/train_autoencoder_cnn.py --resume checkpoints/xxx/latest.pt
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
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
    log_images,
    log_metrics,
)

from griffith_logic import (
    BinaryBrainConv2d,
    BinaryToReal,
    DifferentiableLutBlock,
    RealToBinary,
    UpSample2d,
)
from griffith_logic.pooling import OrPooling2d

load_dotenv()


# ============================================================
# モデル
# ============================================================

class BinaryBrainAutoEncoder(nn.Module):
    """
    BinaryBrainのMnistAeDifferentiableLutCnn忠実再現。

    構成:
        RealToBinary → Encoder(Conv×4 + Pool×2 + Dense×3) →
        Decoder(Dense×3 + UpSample×2 + Conv×4) → BinaryToReal

    BinaryBrainのノートブック準拠:
    - Conv2d: im2col + DifferentiableLut チェーン + col2im
    - MaxPooling: 2×2 (OrPooling2d = max pooling)
    - UpSampling: 2×2 nearest neighbor
    - バイナリ変調: frame_modulation_size=7
    """

    def __init__(
        self,
        frame_modulation_size: int = 7,
        n_inputs: int = 6,
        device: str = 'cpu',
        unbinarize_bias: float = 0.25,
    ):
        super().__init__()
        self.frame_modulation_size = frame_modulation_size
        N = n_inputs

        # バイナリ変調
        self.real_to_binary = RealToBinary(frame_modulation_size)

        # --- Encoder ---
        # Conv2d: DifferentiableLut([192, 32]), 3×3, same → 28×28×32
        self.enc_conv1 = BinaryBrainConv2d(
            lut_sizes=[N * 32, 32], in_channels=1,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
        )
        # Conv2d: DifferentiableLut([192, 32]), 3×3, same → 28×28×32
        self.enc_conv2 = BinaryBrainConv2d(
            lut_sizes=[N * 32, 32], in_channels=32,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
        )
        # MaxPooling 2×2 → 14×14×32
        self.pool1 = OrPooling2d(kernel_size=2)

        # Conv2d: DifferentiableLut([384, 64]), 3×3, same → 14×14×64
        self.enc_conv3 = BinaryBrainConv2d(
            lut_sizes=[N * 64, 64], in_channels=32,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
        )
        # Conv2d: DifferentiableLut([384, 64]), 3×3, same → 14×14×64
        self.enc_conv4 = BinaryBrainConv2d(
            lut_sizes=[N * 64, 64], in_channels=64,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
        )
        # MaxPooling 2×2 → 7×7×64
        self.pool2 = OrPooling2d(kernel_size=2)

        # Dense (Flatten → DifferentiableLut)
        # 7×7×64 = 3136 → 1152 → 192 → 32
        flat_dim = 7 * 7 * 64
        self.enc_dense1 = DifferentiableLutBlock(
            in_dim=flat_dim, out_dim=1152,
            n_inputs=N, device=device, unbinarize_bias=unbinarize_bias,
        )
        self.enc_dense2 = DifferentiableLutBlock(
            in_dim=1152, out_dim=192,
            n_inputs=N, device=device, unbinarize_bias=unbinarize_bias,
        )
        self.enc_dense3 = DifferentiableLutBlock(
            in_dim=192, out_dim=32,
            n_inputs=N, device=device, unbinarize_bias=unbinarize_bias,
        )

        # --- Decoder ---
        # Dense: 32 → 125 → 1024 → 3136 (reshape to 64×7×7)
        self.dec_dense1 = DifferentiableLutBlock(
            in_dim=32, out_dim=125,
            n_inputs=N, device=device, unbinarize_bias=unbinarize_bias,
        )
        self.dec_dense2 = DifferentiableLutBlock(
            in_dim=125, out_dim=1024,
            n_inputs=N, device=device, unbinarize_bias=unbinarize_bias,
        )
        self.dec_dense3 = DifferentiableLutBlock(
            in_dim=1024, out_dim=flat_dim,
            n_inputs=N, device=device, unbinarize_bias=unbinarize_bias,
        )

        # UpSampling 2×2: 7×7 → 14×14
        self.up1 = UpSample2d(scale_factor=2)

        # Conv2d: DifferentiableLut([384, 64]), 3×3, same → 14×14×64
        self.dec_conv1 = BinaryBrainConv2d(
            lut_sizes=[N * 64, 64], in_channels=64,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
        )
        self.dec_conv2 = BinaryBrainConv2d(
            lut_sizes=[N * 64, 64], in_channels=64,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
        )

        # UpSampling 2×2: 14×14 → 28×28
        self.up2 = UpSample2d(scale_factor=2)

        # Conv2d: DifferentiableLut([192, 32]), 3×3, same → 28×28×32
        self.dec_conv3 = BinaryBrainConv2d(
            lut_sizes=[N * 32, 32], in_channels=64,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
        )
        # 最終Conv: DifferentiableLut([216, 36, 6, 1]), batch_norm=False → 28×28×1
        self.dec_conv4 = BinaryBrainConv2d(
            lut_sizes=[N * 36, 36, N, 1], in_channels=32,
            kernel_size=3, padding='same', n_inputs=N, device=device,
            unbinarize_bias=unbinarize_bias,
            batch_norm=False,
        )

        # フレーム積算
        self.binary_to_real = BinaryToReal(frame_modulation_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28) MNIST画像 [0, 1]

        Returns:
            (B, 1, 28, 28) 再構成画像 [0, 1]
        """
        # バイナリ変調: (B, 1, 28, 28) → (B*M, 1, 28, 28)
        x = self.real_to_binary(x)

        # Encoder
        x = self.enc_conv1(x)   # (B*M, 32, 28, 28)
        x = self.enc_conv2(x)   # (B*M, 32, 28, 28)
        x = self.pool1(x)       # (B*M, 32, 14, 14)
        x = self.enc_conv3(x)   # (B*M, 64, 14, 14)
        x = self.enc_conv4(x)   # (B*M, 64, 14, 14)
        x = self.pool2(x)       # (B*M, 64, 7, 7)

        # Flatten → Dense
        BM = x.shape[0]
        x = x.reshape(BM, -1)   # (B*M, 3136)
        x = self.enc_dense1(x)  # (B*M, 1152)
        x = self.enc_dense2(x)  # (B*M, 192)
        x = self.enc_dense3(x)  # (B*M, 32)

        # Decoder Dense
        x = self.dec_dense1(x)  # (B*M, 125)
        x = self.dec_dense2(x)  # (B*M, 1024)
        x = self.dec_dense3(x)  # (B*M, 3136)

        # Reshape to spatial
        x = x.reshape(BM, 64, 7, 7)

        # UpSample + Conv
        x = self.up1(x)         # (B*M, 64, 14, 14)
        x = self.dec_conv1(x)   # (B*M, 64, 14, 14)
        x = self.dec_conv2(x)   # (B*M, 64, 14, 14)
        x = self.up2(x)         # (B*M, 64, 28, 28)
        x = self.dec_conv3(x)   # (B*M, 32, 28, 28)
        x = self.dec_conv4(x)   # (B*M, 1, 28, 28)

        # フレーム積算: (B*M, 1, 28, 28) → (B, 1, 28, 28)
        x = self.binary_to_real(x)

        return x

    def clamp_weights(self):
        """全DifferentiableLutBlockの重みを[0,1]にクランプ。"""
        for module in self.modules():
            if hasattr(module, 'clamp_weights') and module is not self:
                module.clamp_weights()


# ============================================================
# データセット
# ============================================================

def load_dataset(args):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(
        './data-mnist', train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.MNIST(
        './data-mnist', train=False, transform=transform,
    )

    use_pin_memory = args.device == 'cuda'

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )
    return train_loader, test_loader


# ============================================================
# 訓練・評価
# ============================================================

def evaluate(model, loader, device, desc=None):
    """再構成品質を評価する（MSE）。"""
    model.eval()
    total_mse = 0.0
    total_pixels = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    with torch.no_grad():
        for x, _ in it:
            x = x.to(torch.float32).to(device)
            output = model(x)
            total_mse += F.mse_loss(output, x, reduction='sum').item()
            total_pixels += x.numel()
            if desc and isinstance(it, tqdm):
                it.set_postfix(mse=f'{total_mse / total_pixels:.6f}')
    return total_mse / total_pixels


def save_reconstruction_grid(model, test_loader, device, epoch, output_dir):
    """元画像 vs 再構成画像のグリッドを保存する。"""
    model_mode = model.training
    model.eval()
    x, _ = next(iter(test_loader))
    x = x[:8].to(torch.float32).to(device)

    with torch.no_grad():
        recon = model(x)

    # 2行: 元画像 / 再構成
    grid = torchvision.utils.make_grid(
        torch.cat([x[:8], recon[:8]]),
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

def setup_logger(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'train_ae_cnn_mnist_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='BinaryBrain CNN Autoencoder 忠実再現スクリプト',
    )
    parser.add_argument('--n-inputs', type=int, default=6, choices=[2, 4, 6],
                        help='LUTの入力数 (default: 6)')
    parser.add_argument('--frame-modulation-size', type=int, default=7,
                        help='バイナリ変調フレーム数 (default: 7)')
    parser.add_argument('--unbinarize-bias', type=float, default=0.25,
                        help='Unbinarize bias (default: 0.25)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学習率 (default: 0.001)')
    parser.add_argument('-bs', '--batch-size', type=int, default=16,
                        help='バッチサイズ (default: 16, MPS/CUDAメモリ制約のためBinaryBrainの32より小さい)')
    parser.add_argument('-e', '--epochs', type=int, default=4,
                        help='エポック数 (default: 4)')
    parser.add_argument('--seed', type=int, default=1,
                        help='ランダムシード (default: 1)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先ディレクトリ (default: logs)')
    parser.add_argument('--output-dir', type=str, default='outputs/autoencoder_cnn',
                        help='再構成画像の保存先 (default: outputs/autoencoder_cnn)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開（パス指定）')

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

    # シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データ読み込み
    train_loader, test_loader = load_dataset(args)
    print('データセット: MNIST')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, テスト: {len(test_loader.dataset):,}サンプル')

    # モデル構築
    N = args.n_inputs
    model = BinaryBrainAutoEncoder(
        frame_modulation_size=args.frame_modulation_size,
        n_inputs=N,
        device=args.device,
        unbinarize_bias=args.unbinarize_bias,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'モデル構成: BinaryBrain CNN AutoEncoder, N={N}入力LUT')
    print(f'バイナリ変調: frame_modulation_size={args.frame_modulation_size}')
    print(f'総パラメータ数: {total_params:,}')

    # BinaryBrainと同じ: MSE損失、Adam optimizer (eps=1e-7)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    # ログ
    log, log_file = setup_logger(args.log_dir)
    log(f'START n_inputs={N} frame_mod={args.frame_modulation_size} '
        f'lr={args.lr} bs={args.batch_size} epochs={args.epochs} '
        f'seed={args.seed} device={args.device}')

    # チェックポイント
    ckpt_prefix = 'ae_cnn_mnist'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}

    # チェックポイントから復元
    start_epoch = 0
    best_test_mse = float('inf')
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(
            args.resume, model, optimizer, device=args.device,
        )
        start_epoch = ckpt_info['step']
        best_test_mse = ckpt_info['best_metric']
        print(f'  エポック {start_epoch} から再開 (best_mse={best_test_mse:.6f})')

    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = f'ae_cnn_mnist_n{N}_e{args.epochs}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['autoencoder', 'cnn', 'binarybrain', 'mnist', f'n{N}'],
        group='autoencoder_cnn',
    )

    batches_per_epoch = len(train_loader)
    print(f'\n訓練開始: {args.epochs}エポック ({batches_per_epoch}バッチ/エポック)')
    print('=' * 60)

    train_start = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # 訓練
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f'epoch {epoch}/{args.epochs}',
            leave=False,
        )
        for x, _ in pbar:
            x = x.to(torch.float32).to(args.device)

            output = model(x)
            loss = loss_fn(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # BinaryBrain忠実再現: optimizer step後に重みを[0,1]にクランプ
            model.clamp_weights()

            epoch_loss += loss.item()

            elapsed_t = time.time() - train_start
            progress = (epoch - 1 + (pbar.n + 1) / batches_per_epoch) / args.epochs
            remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{loss.item():.4f}', eta=eta_str)

        avg_loss = epoch_loss / batches_per_epoch

        # 評価
        test_mse = evaluate(model, test_loader, args.device,
                            desc=f'eval {epoch}/{args.epochs}')

        is_best = test_mse < best_test_mse
        if is_best:
            best_test_mse = test_mse

        # チェックポイント保存
        save_checkpoint(latest_path, model, optimizer, None,
                        epoch, best_test_mse, ckpt_metadata)
        if is_best:
            save_checkpoint(best_path, model, optimizer, None,
                            epoch, best_test_mse, ckpt_metadata)

        # 再構成画像を保存（ローカル + wandb）
        grid = save_reconstruction_grid(
            model, test_loader, args.device, epoch, args.output_dir,
        )
        if use_wandb:
            from torchvision.transforms.functional import to_pil_image
            log_images('autoencoder_cnn/reconstruction', [to_pil_image(grid)],
                       captions=[f'epoch {epoch}'])

        # 時間計算
        elapsed = time.time() - train_start
        sec_per_epoch = elapsed / (epoch - start_epoch)
        remaining = sec_per_epoch * (args.epochs - epoch)
        eta = datetime.now(JST) + timedelta(seconds=remaining)
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remain_str = str(timedelta(seconds=int(remaining)))
        eta_str = eta.strftime('%H:%M:%S')

        print(
            f'[epoch {epoch:>3d}/{args.epochs}] '
            f'loss={avg_loss:.4f}  '
            f'test_mse={test_mse:.6f}'
            f'{"  ★ BEST" if is_best else ""}'
            f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
        )

        log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
            f'test_mse={test_mse:.6f} '
            f'best={str(is_best).lower()} '
            f'elapsed={elapsed_str} eta={remain_str}')
        if use_wandb:
            log_metrics({
                'epoch': epoch,
                'loss': avg_loss,
                'test_mse': test_mse,
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
