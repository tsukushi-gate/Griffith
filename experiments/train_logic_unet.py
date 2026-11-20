"""LogicUNet 訓練スクリプト(合成セグメンテーション)。

MNISTから合成した密予測タスクで LogicUNet(完全論理ゲート版U-Net)を学習する。
    - 入力: MNIST数字画像を二値化し、salt-and-pepperノイズを付加したもの [0,1]
    - ターゲット: 前景マスク(元画像 > 閾値 の2値)
損失は BCEWithLogits、評価指標は前景 IoU。予測マスクのサンプルを wandb にログする。

使用例:
    # デフォルト設定で学習
    uv run python experiments/train_logic_unet.py

    # スモーク(2分以内、CPU、wandbなし)
    uv run python experiments/train_logic_unet.py --max-steps 30 --no-wandb --device cpu

    # 深さ・チャンネル幅・ノイズを変更
    uv run python experiments/train_logic_unet.py --depth 2 --base-channels 32 --noise 0.1
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_images,
    log_metrics,
    normalize_tags,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic.datasets.vision import load_mnist
from griffith_logic.models.logic_unet import LogicUNet, LogicUNetConfig

load_dotenv()

# 前景判定のピクセル閾値(二値化済み画像なので 0.5)
FG_THRESHOLD = 0.5


# ============================================================
# データセット(合成セグメンテーション)
# ============================================================

class SyntheticSegDataset(torch.utils.data.Dataset):
    """MNISTから前景セグメンテーションを合成する。

    入力 = 二値化MNIST画像にノイズ(画素反転)を付加したもの、
    ターゲット = ノイズなし前景マスク(元画像の前景 = 1)。
    ノイズを入れることで、skip接続と論理回路が「デノイズしつつ前景を復元する」
    非自明な密予測を解く必要が生じる。
    """

    def __init__(self, base_set, noise: float, seed: int = 0):
        self.base_set = base_set
        self.noise = noise
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_set)

    def __getitem__(self, idx: int):
        img, _ = self.base_set[idx]  # (1, 28, 28) 二値化済み [0,1]
        mask = (img > FG_THRESHOLD).float()
        x = mask.clone()
        if self.noise > 0:
            # インデックス依存の決定的ノイズ(再現性確保)
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
            flip = (torch.rand(x.shape, generator=g) < self.noise).float()
            x = (x + flip) % 2
        return x, mask


def load_dataset(args):
    train_base, test_base = load_mnist(binarize_threshold=FG_THRESHOLD)
    train_set = SyntheticSegDataset(train_base, args.noise, seed=1)
    test_set = SyntheticSegDataset(test_base, args.noise, seed=2)

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
# 評価
# ============================================================

def compute_iou(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """バッチの (intersection, union) を返す(前景=logit>0)。"""
    pred = (logits > 0).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return inter, union


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """eval時(hard論理回路)の前景 IoU を返す。"""
    model.eval()
    total_inter = 0.0
    total_union = 0.0
    for i, (x, mask) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(torch.float32).to(device)
        mask = mask.to(torch.float32).to(device)
        logits = model(x)
        inter, union = compute_iou(logits, mask)
        total_inter += inter
        total_union += union
    return total_inter / total_union if total_union > 0 else 0.0


@torch.no_grad()
def make_prediction_grid(model, loader, device):
    """入力 / 予測マスク / 正解マスク のグリッド画像を返す(wandb用PIL)。"""
    import torchvision
    from torchvision.transforms.functional import to_pil_image

    model_mode = model.training
    model.eval()
    x, mask = next(iter(loader))
    x = x[:8].to(torch.float32).to(device)
    mask = mask[:8].to(torch.float32).to(device)
    logits = model(x)
    pred = (logits > 0).float()

    # 3行: 入力 / 予測 / 正解
    grid = torchvision.utils.make_grid(
        torch.cat([x, pred, mask]).cpu(), nrow=8, padding=2,
    )
    model.train(mode=model_mode)
    return to_pil_image(grid)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LogicUNet 訓練スクリプト(合成セグメンテーション)',
    )
    parser.add_argument('--base-channels', type=int, default=32,
                        help='最浅stageの論理チャネル幅 (default: 32)')
    parser.add_argument('--depth', type=int, default=2,
                        help='ダウン/アップサンプル段数 (default: 2)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='LUTの入力数 (default: 2)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp'],
                        help='LUTパラメータ化方式 (default: softmax)')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='入力に付加するsalt-pepperノイズ率 (default: 0.1)')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='学習率 (default: 0.02)')
    parser.add_argument('-bs', '--batch-size', type=int, default=16,
                        help='バッチサイズ (default: 16)')
    parser.add_argument('-e', '--epochs', type=int, default=4,
                        help='エポック数 (default: 4)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大訓練ステップ数(スモーク用、指定時はepochsより優先)')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--output-dir', type=str, default='outputs/logic_unet',
                        help='予測画像の保存先 (default: outputs/logic_unet)')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開(パス指定)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbログを無効化する')

    add_experiment_args(parser)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader = load_dataset(args)
    print('データセット: 合成セグメンテーション(MNIST由来)')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル, ノイズ率={args.noise}')

    config = LogicUNetConfig(
        in_channels=1,
        base_channels=args.base_channels,
        depth=args.depth,
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        out_channels=1,
    )
    model = LogicUNet(config).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'モデル構成: LogicUNet depth={args.depth} base={args.base_channels} '
          f'N={args.n_inputs} param={args.parameterization}')
    print(f'総パラメータ数: {total_params:,}')

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_prefix = 'logic_unet_seg'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}
    print(f'チェックポイント保存先: {run_dir}/')

    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(args.resume, model, optimizer, device=args.device)
        start_epoch = ckpt_info['step']
        best_iou = ckpt_info['best_metric']
        print(f'  エポック {start_epoch} から再開 (best_iou={best_iou:.4f})')

    # wandb
    run_name = (f'seg_d{args.depth}_c{args.base_channels}_'
                f'n{args.n_inputs}_e{args.epochs}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=args.device,
            tags=['logic_unet', 'segmentation', 'mnist', f'n{args.n_inputs}',
                  f'd{args.depth}']
            + normalize_tags(model='logic_unet', dataset='mnist', seed=args.seed),
            group='logic_unet',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    batches_per_epoch = len(train_loader)
    print(f'\n訓練開始: {args.epochs}エポック ({batches_per_epoch}バッチ/エポック)')
    if args.max_steps is not None:
        print(f'  max-steps={args.max_steps} (スモークモード)')
    print('=' * 60)

    train_start = time.time()
    global_step = 0
    stop = False

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, mask in train_loader:
            x = x.to(torch.float32).to(args.device)
            mask = mask.to(torch.float32).to(args.device)

            logits = model(x)
            loss = loss_fn(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                elapsed_t = time.time() - train_start
                print(f'  step {global_step} loss={loss.item():.4f} '
                      f'[{elapsed_t:.0f}s経過]')

            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        avg_loss = epoch_loss / max(1, n_batches)

        # 評価(スモーク時はバッチ数を絞る)
        eval_batches = 20 if args.max_steps is not None else None
        test_iou = evaluate(model, test_loader, args.device, max_batches=eval_batches)

        is_best = test_iou > best_iou
        if is_best:
            best_iou = test_iou

        save_checkpoint(latest_path, model, optimizer, None,
                        epoch, best_iou, ckpt_metadata)
        if is_best:
            save_checkpoint(best_path, model, optimizer, None,
                            epoch, best_iou, ckpt_metadata)

        # 予測グリッドを保存(ローカル + wandb)
        grid = make_prediction_grid(model, test_loader, args.device)
        os.makedirs(args.output_dir, exist_ok=True)
        grid_path = os.path.join(args.output_dir, f'pred_epoch{epoch:03d}.png')
        grid.save(grid_path)
        print(f'  予測画像を保存: {grid_path}')
        if use_wandb:
            log_images('logic_unet/prediction', [grid],
                       captions=[f'epoch {epoch} (input/pred/gt)'])

        elapsed = time.time() - train_start
        done_epochs = epoch - start_epoch
        sec_per_epoch = elapsed / done_epochs
        remaining = sec_per_epoch * (args.epochs - epoch)
        eta = datetime.now(JST) + timedelta(seconds=remaining)
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remain_str = str(timedelta(seconds=int(remaining)))
        eta_str = eta.strftime('%H:%M:%S')

        print(
            f'[epoch {epoch:>3d}/{args.epochs}] '
            f'loss={avg_loss:.4f}  test_iou={test_iou:.4f}'
            f'{"  ★ BEST" if is_best else ""}'
            f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
        )

        if use_wandb:
            log_metrics({
                'epoch': epoch,
                'loss': avg_loss,
                'test_iou': test_iou,
                'best_iou': best_iou,
            })

        if stop:
            print(f'max-steps={args.max_steps} 到達で終了')
            break

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テストIoU(best): {best_iou:.4f}  (総訓練時間: {total_time})')

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_iou': best_iou},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
