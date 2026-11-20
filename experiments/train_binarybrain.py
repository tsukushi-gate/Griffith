"""
BinaryBrain MNIST再現スクリプト

BinaryBrainのMnistDifferentiableLutSimple.ipynbを忠実に再現。
6入力LUT + BatchNorm + Binarize + バイナリ変調のパイプライン。

使用例:
    # デフォルト（BinaryBrainと同じ設定）
    uv run python experiments/train_binarybrain.py

    # エポック数変更
    uv run python experiments/train_binarybrain.py --epochs 10

    # N=4入力LUT
    uv run python experiments/train_binarybrain.py --n-inputs 4
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
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic import BinaryToReal, DifferentiableLutBlock, RealToBinary

load_dotenv()


# ============================================================
# モデル
# ============================================================


class BinaryBrainModel(nn.Module):
    """
    BinaryBrainのMnistDifferentiableLutSimpleと等価なモデル。

    構成:
        Flatten → RealToBinary → DifferentiableLutBlock×6 → BinaryToReal

    BinaryBrainのネットワーク設計ルール:
    - N入力LUTなので、前層のノード数が次層のN倍以上でないと未接続ノードが発生
    - 最終層はクラス数（10）で、GroupSumは不要
    """

    def __init__(
        self,
        in_dim: int = 784,
        num_classes: int = 10,
        n_inputs: int = 6,
        frame_modulation_size: int = 15,
        unbinarize_bias: float = 0.25,
        device: str = "cpu",
        layer_sizes: list = None,
    ):
        super().__init__()
        self.frame_modulation_size = frame_modulation_size
        N = n_inputs

        # BinaryBrainのデフォルト層サイズ（N=6の場合）
        if layer_sizes is None:
            layer_sizes = [
                N * N * 64,  # 2304 (= 6*6*64)
                N * 64,  # 384  (= 6*64)
                64,  # 64
                N * N * num_classes,  # 360 (= 6*6*10)
                N * num_classes,  # 60  (= 6*10)
                num_classes,  # 10
            ]

        self.flatten = nn.Flatten()
        self.real_to_binary = RealToBinary(frame_modulation_size)

        # DifferentiableLutBlock層を構築
        self.lut_blocks = nn.ModuleList()
        prev_dim = in_dim
        for out_dim in layer_sizes:
            self.lut_blocks.append(
                DifferentiableLutBlock(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    n_inputs=n_inputs,
                    device=device,
                    connections="random",
                    unbinarize_bias=unbinarize_bias,
                )
            )
            prev_dim = out_dim

        self.binary_to_real = BinaryToReal(frame_modulation_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.real_to_binary(x)
        for block in self.lut_blocks:
            x = block(x)
        x = self.binary_to_real(x)
        return x

    def clamp_weights(self):
        """全DifferentiableLutBlockの重みを[0,1]にクランプ"""
        for block in self.lut_blocks:
            block.clamp_weights()


# ============================================================
# データセット
# ============================================================


def load_dataset(args):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(
        "./data-mnist",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = torchvision.datasets.MNIST(
        "./data-mnist",
        train=False,
        transform=transform,
    )

    use_pin_memory = args.device == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
        drop_last=True,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
        drop_last=True,
        num_workers=2,
    )
    return train_loader, test_loader


# ============================================================
# 訓練・評価
# ============================================================


def evaluate(model, loader, device, desc=None):
    model.eval()
    correct = 0
    total = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    with torch.no_grad():
        for x, y in it:
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            output = model(x)
            pred = output.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
            if desc and isinstance(it, tqdm):
                it.set_postfix(acc=f"{correct / total:.4f}")
    return correct / total if total > 0 else 0.0


# ============================================================
# ログ
# ============================================================


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_binarybrain_mnist_{timestamp}.log")
    log_file = open(log_path, "a", buffering=1)
    print(f"ログファイル: {log_path}")

    def log(msg):
        ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{ts}] {msg}\n")

    return log, log_file


# ============================================================
# メイン
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="BinaryBrain MNIST再現スクリプト",
    )
    parser.add_argument(
        "--n-inputs",
        type=int,
        default=6,
        choices=[2, 4, 6],
        help="LUTの入力数 (default: 6)",
    )
    parser.add_argument(
        "--frame-modulation-size",
        type=int,
        default=15,
        help="バイナリ変調フレーム数 (default: 15)",
    )
    parser.add_argument(
        "--unbinarize-bias",
        type=float,
        default=0.25,
        help="Unbinarize bias (default: 0.25)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="学習率 (default: 0.001)"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=64, help="バッチサイズ (default: 64)"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=4, help="エポック数 (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="ランダムシード (default: 1)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="デバイス (default: 自動検出)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="ログ出力先ディレクトリ (default: logs)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="チェックポイント保存先 (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="チェックポイントから訓練再開（パス指定）",
    )

    args = parser.parse_args()

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"デバイス: {args.device}")

    # シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # データ読み込み
    train_loader, test_loader = load_dataset(args)
    print("データセット: MNIST")
    print(
        f"訓練: {len(train_loader.dataset):,}サンプル, テスト: {len(test_loader.dataset):,}サンプル"
    )

    # モデル構築
    N = args.n_inputs
    model = BinaryBrainModel(
        in_dim=784,
        num_classes=10,
        n_inputs=N,
        frame_modulation_size=args.frame_modulation_size,
        unbinarize_bias=args.unbinarize_bias,
        device=args.device,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデル構成: BinaryBrain DifferentiableLut×6, N={N}入力LUT")
    print(f"バイナリ変調: frame_modulation_size={args.frame_modulation_size}")
    print(f"総パラメータ数: {total_params:,}")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    # BinaryBrainのOptimizerAdamと同じeps=1e-7を使用
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    # ログ
    log, log_file = setup_logger(args.log_dir)
    log(
        f"START n_inputs={N} frame_mod={args.frame_modulation_size} "
        f"lr={args.lr} bs={args.batch_size} epochs={args.epochs} "
        f"seed={args.seed} device={args.device}"
    )

    # チェックポイント
    ckpt_prefix = "train_binarybrain_mnist"
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix,
        base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {"args": vars(args)}

    # チェックポイントから復元
    start_epoch = 0
    best_test_acc = 0.0
    if args.resume:
        print(f"チェックポイントを読み込み: {args.resume}")
        ckpt_info = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device=args.device,
        )
        start_epoch = ckpt_info["step"]
        best_test_acc = ckpt_info["best_metric"]
        print(f"  エポック {start_epoch} から再開 (best_acc={best_test_acc:.4f})")

    print(f"チェックポイント保存先: {run_dir}/")

    # wandb
    run_name = f"binarybrain_mnist_n{N}_e{args.epochs}"
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=["binarybrain", "mnist", f"n{N}"],
        group="binarybrain",
    )

    batches_per_epoch = len(train_loader)
    print(f"\n訓練開始: {args.epochs}エポック ({batches_per_epoch}バッチ/エポック)")
    print("=" * 60)

    train_start = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # 訓練
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
        )
        for x, y in pbar:
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # BinaryBrain忠実再現: optimizer step後に重みを[0,1]にクランプ
            model.clamp_weights()

            epoch_loss += loss.item()

            elapsed_t = time.time() - train_start
            progress = (epoch - 1 + (pbar.n + 1) / batches_per_epoch) / args.epochs
            remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime(
                "%H:%M:%S"
            )
            pbar.set_postfix(loss=f"{loss.item():.4f}", eta=eta_str)

        avg_loss = epoch_loss / batches_per_epoch

        # 評価
        test_acc = evaluate(
            model, test_loader, args.device, desc=f"eval {epoch}/{args.epochs}"
        )

        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc

        # チェックポイント保存
        save_checkpoint(
            latest_path, model, optimizer, None, epoch, best_test_acc, ckpt_metadata
        )
        if is_best:
            save_checkpoint(
                best_path, model, optimizer, None, epoch, best_test_acc, ckpt_metadata
            )

        # 時間計算
        elapsed = time.time() - train_start
        sec_per_epoch = elapsed / (epoch - start_epoch)
        remaining = sec_per_epoch * (args.epochs - epoch)
        eta = datetime.now(JST) + timedelta(seconds=remaining)
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remain_str = str(timedelta(seconds=int(remaining)))
        eta_str = eta.strftime("%H:%M:%S")

        print(
            f"[epoch {epoch:>3d}/{args.epochs}] "
            f"loss={avg_loss:.4f}  "
            f"test_acc={test_acc:.4f}"
            f"{'  ★ BEST' if is_best else ''}"
            f"  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]"
        )

        log(
            f"EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} "
            f"test_acc={test_acc:.4f} "
            f"best={str(is_best).lower()} "
            f"elapsed={elapsed_str} eta={remain_str}"
        )
        if use_wandb:
            log_metrics(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "test_acc": test_acc,
                    "best_test_acc": best_test_acc,
                }
            )

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f"\n最終テスト精度: {best_test_acc:.4f}  (総訓練時間: {total_time})")
    log(f"DONE best_test_acc={best_test_acc:.4f} total_time={total_time}")
    log_file.close()
    if use_wandb:
        wandb_finish()


if __name__ == "__main__":
    main()
