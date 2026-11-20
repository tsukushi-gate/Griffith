"""CIFAR-10 3段エンコーダ事前学習スクリプト(W13a-4)。

det/seg backboneと**同構造の3段エンコーダ**(LutConv2d k→2k→4k、
softmax16g、OrPooling配置も一致)をCIFAR-10分類で事前学習し、そのstate_dict
を `--init-from` (train_detection.py/train_segmentation.py、W13a-3) で
det/segモデルへ部分ロードするための重みを作る。

エンコーダ部のsubmodule名は --target に応じて以下のいずれかに一致させてある
(部分ロードが素通しになる命名。encoder.*等のprefixラップは行わない):
    --target det: CenterNetLUT(n_layers=3) の stage1/stage2/stage3_pre
    --target seg: SegConvDLGN の c1/p1/c2/p2/c3

分類ヘッド(head_pool/head_conv/head_sum)はCIFAR事前学習専用でdet/seg側に
対応キーが無いため、load_partial_state_dict経由の部分ロード時は自然に
無視される(skipped_missing側、意図した挙動)。

使用例:
    # det向けエンコーダ事前学習(k=32)
    uv run python experiments/pretrain_cifar_encoder.py --target det -k 32 -e 60

    # seg向けエンコーダ事前学習(k=64、learnable wiring)
    uv run python experiments/pretrain_cifar_encoder.py --target seg -k 64 \\
        --wiring learnable -e 60

    # CPUスモーク
    WANDB_MODE=offline uv run python experiments/pretrain_cifar_encoder.py \\
        --target det -k 4 -d 2 --head-group 2 -e 1 --limit-samples 64 \\
        -bs 8 --device cpu

    # 事前学習ckptをdet訓練へ投入
    uv run python experiments/train_detection.py \\
        --init-from checkpoints/cifar_pretrain_det_.../best.pt -k 32
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
import torch.nn as nn
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from train_cnn import ThresholdBinarize
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import GroupSum, LutConv2d, OrPooling2d

load_dotenv()


# ============================================================
# モデル(det/seg backboneと同一命名の3段エンコーダ + 分類ヘッド)
# ============================================================

def _conv_kw(n_inputs, device, grad_factor, input_channels_per_tree, wiring):
    return dict(
        n_inputs=n_inputs, device=device, grad_factor=grad_factor,
        input_channels_per_tree=input_channels_per_tree,
        parameterization='softmax16g',  # det/seg backboneと同一(構造欠陥回避)
        wiring=wiring,
    )


class CifarPretrainDet(nn.Module):
    """CenterNetLUT(n_layers=3)バックボーンと同一命名(stage1/stage2/stage3_pre)
    の3段エンコーダ + GroupSum分類ヘッド(CIFAR-10、10クラス)。

    エンコーダ出力: in_channels@32x32 → k@32x32 →(OrPool)→ k@16x16
    → 2k@16x16 →(OrPool)→ 2k@8x8 → 4k@8x8(stage3_pre、以降プールなし、
    CenterNetLUTの出力ストライド4と同一)。
    ヘッド: OrPool→4k@4x4 → LutConv2d(kernel=4,pad=0)→10*head_group@1x1
    → flatten → GroupSum(10)。
    """

    def __init__(
        self, in_channels: int, k: int, tree_depth: int = 3,
        n_classes: int = 10, head_group: int = 16,
        input_channels_per_tree: int = 2, n_inputs: int = 2,
        wiring: str = 'fixed', device: str = 'cpu', grad_factor: float = 1.0,
    ):
        super().__init__()
        conv_kw = _conv_kw(n_inputs, device, grad_factor,
                           input_channels_per_tree, wiring)

        def conv(in_ch, out_ch):
            return LutConv2d(in_ch, out_ch, kernel_size=3, tree_depth=tree_depth,
                             padding=1, **conv_kw)

        # CenterNetLUT(n_layers=3)と同一命名(W13a-4、部分ロード素通し用)
        self.stage1 = nn.ModuleList([conv(in_channels, k), OrPooling2d(2)])
        self.stage2 = nn.ModuleList([conv(k, 2 * k), OrPooling2d(2)])
        self.stage3_pre = nn.ModuleList([conv(2 * k, 4 * k)])

        feat_ch = 4 * k
        self.head_pool = OrPooling2d(2)
        self.head_conv = LutConv2d(
            feat_ch, n_classes * head_group, kernel_size=4, tree_depth=tree_depth,
            padding=0, **conv_kw,
        )
        self.head_sum = GroupSum(n_classes, tau=max(1.0, math.sqrt(head_group)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for layer in self.stage1:
            feat = layer(feat)
        for layer in self.stage2:
            feat = layer(feat)
        for layer in self.stage3_pre:
            feat = layer(feat)
        feat = self.head_pool(feat)
        logits_map = self.head_conv(feat)
        return self.head_sum(logits_map.flatten(1))

    @property
    def num_gates(self) -> int:
        return sum(m.num_gates for m in self.modules() if isinstance(m, LutConv2d))


class CifarPretrainSeg(nn.Module):
    """SegConvDLGNと同一命名(c1/p1/c2/p2/c3)の3段エンコーダ + GroupSum分類
    ヘッド(CIFAR-10、10クラス)。構造はCifarPretrainDetと同一、命名のみ相違。
    """

    def __init__(
        self, in_channels: int, k: int, tree_depth: int = 3,
        n_classes: int = 10, head_group: int = 16,
        input_channels_per_tree: int = 4, n_inputs: int = 2,
        wiring: str = 'fixed', device: str = 'cpu', grad_factor: float = 1.0,
    ):
        super().__init__()
        conv_kw = _conv_kw(n_inputs, device, grad_factor,
                           input_channels_per_tree, wiring)

        def conv(in_ch, out_ch):
            return LutConv2d(in_ch, out_ch, kernel_size=3, tree_depth=tree_depth,
                             padding=1, **conv_kw)

        # SegConvDLGNと同一命名(W13a-4、部分ロード素通し用)
        self.c1 = conv(in_channels, k)
        self.p1 = OrPooling2d(2)
        self.c2 = conv(k, 2 * k)
        self.p2 = OrPooling2d(2)
        self.c3 = conv(2 * k, 4 * k)

        feat_ch = 4 * k
        self.head_pool = OrPooling2d(2)
        self.head_conv = LutConv2d(
            feat_ch, n_classes * head_group, kernel_size=4, tree_depth=tree_depth,
            padding=0, **conv_kw,
        )
        self.head_sum = GroupSum(n_classes, tau=max(1.0, math.sqrt(head_group)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.c1(x)
        f2 = self.c2(self.p1(f1))
        f3 = self.c3(self.p2(f2))
        feat = self.head_pool(f3)
        logits_map = self.head_conv(feat)
        return self.head_sum(logits_map.flatten(1))

    @property
    def num_gates(self) -> int:
        return sum(m.num_gates for m in self.modules() if isinstance(m, LutConv2d))


TARGET_MODELS = {'det': CifarPretrainDet, 'seg': CifarPretrainSeg}
# det=CenterNetLUT既定(input_channels_per_tree=2)、seg=SegConvDLGN既定(=4)
TARGET_DEFAULT_ICPT = {'det': 2, 'seg': 4}


def build_model(args, in_channels: int) -> nn.Module:
    icpt = (args.input_channels_per_tree if args.input_channels_per_tree is not None
           else TARGET_DEFAULT_ICPT[args.target])
    cls = TARGET_MODELS[args.target]
    model = cls(
        in_channels=in_channels, k=args.k, tree_depth=args.tree_depth,
        n_classes=10, head_group=args.head_group,
        input_channels_per_tree=icpt, wiring=args.wiring,
        device=args.device, grad_factor=args.grad_factor,
    )
    return model.to(args.device)


# ============================================================
# データセット
# ============================================================

def load_dataset(args):
    n_thresholds = args.n_thresholds
    if n_thresholds > 0:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ThresholdBinarize(n_thresholds),
        ])
    else:
        transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(
        './data-cifar', train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        './data-cifar', train=False, download=True, transform=transform,
    )
    if args.limit_samples > 0:
        n = args.limit_samples
        train_set = torch.utils.data.Subset(train_set, range(min(n, len(train_set))))
        test_set = torch.utils.data.Subset(test_set, range(min(n, len(test_set))))

    use_pin_memory = args.device == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin_memory, drop_last=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin_memory, drop_last=False, num_workers=2,
    )
    return train_loader, test_loader


# ============================================================
# ユーティリティ
# ============================================================

def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ(他スクリプトと同型)"""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i >= n:
                return


def train_step(model, x, y, loss_fn, optimizer):
    output = model(x)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, loader, device, mode=False, desc=None):
    """mode: True=soft(訓練モード), False=hard(推論モード、入力も丸め)"""
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
                x = x.round()
            output = model(x)
            pred = output.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
            if desc and isinstance(it, tqdm):
                it.set_postfix(acc=f'{correct / max(total, 1):.4f}')
    model.train(mode=orig_mode)
    return correct / total if total > 0 else 0.0


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 3段エンコーダ事前学習(det/segバックボーンと'
                    '同構造、W13a-4)'
    )
    parser.add_argument('--target', type=str, required=True,
                        choices=['det', 'seg'],
                        help='エンコーダのsubmodule命名先(det=CenterNetLUT'
                             '(stage1/stage2/stage3_pre)、'
                             'seg=SegConvDLGN(c1/p1/c2/p2/c3))')
    parser.add_argument('-k', type=int, default=32,
                        help='エンコーダ幅パラメータ k (default: 32)')
    parser.add_argument('-d', '--tree-depth', type=int, default=3,
                        help='LUTツリーの深さ (default: 3、det/seg既定と同一)')
    parser.add_argument('--n-thresholds', type=int, default=3,
                        help='入力二値化の閾値数 (default: 3)')
    parser.add_argument('--input-channels-per-tree', type=int, default=None,
                        help='各ツリーが参照する入力チャンネル数 '
                             '(default: None → target=detで2、seg で4、'
                             'それぞれのモデル既定と一致)')
    parser.add_argument('--wiring', type=str, default='fixed',
                        choices=['fixed', 'learnable'],
                        help='LutConv2dの入力配線方式 (default: fixed)')
    parser.add_argument('--head-group', type=int, default=16,
                        help='分類ヘッドのGroupSumグループサイズ (default: 16)')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数 (default: 1.0)')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='学習率 (default: 0.02)')
    parser.add_argument('--weight-decay', type=float, default=0.002,
                        help='Weight decay (default: 0.002)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数(--num-iterationsと排他)')
    parser.add_argument('-ni', '--num-iterations', type=int, default=None,
                        help='反復回数(--epochsと排他)')
    parser.add_argument('-ef', '--eval-freq', type=int, default=None,
                        help='評価頻度(エポック時=Nエポックごと[既定1]、'
                             'イテレーション時=Nイテレーションごと[既定2000])')
    parser.add_argument('--limit-samples', type=int, default=0,
                        help='train/testデータセット長の上限(スモーク用、'
                             '0=全件) (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから訓練再開(パス指定)')

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

    # データ
    train_loader, test_loader = load_dataset(args)
    in_ch = 3 * args.n_thresholds if args.n_thresholds > 0 else 3
    print(f'データセット: CIFAR-10 (target={args.target})')
    print(f'閾値数: {args.n_thresholds} (入力チャンネル: {in_ch})')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル')

    # モデル
    model = build_model(args, in_ch)
    total_gates = model.num_gates
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nモデル構成: target={args.target}, k={args.k}, '
          f'tree_depth={args.tree_depth}, wiring={args.wiring}')
    print(f'総ロジックゲート数: {total_gates:,} (LUT2, softmax16g)')
    print(f'総パラメータ数: {total_params:,}')
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # エポック / イテレーション の解決
    use_epochs = args.epochs is not None
    if use_epochs and args.num_iterations is not None:
        parser.error('--epochs と --num-iterations は同時に指定できません')
    if not use_epochs and args.num_iterations is None:
        args.epochs = 60
        use_epochs = True

    # チェックポイント
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        f'cifar_pretrain_{args.target}', base_dir=args.checkpoint_dir,
    )
    print(f'チェックポイント保存先: {run_dir}/')
    log_path = os.path.join(run_dir, 'log.txt')
    log_file = open(log_path, 'a', buffering=1)

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    log(f'START target={args.target} k={args.k} tree_depth={args.tree_depth} '
        f'wiring={args.wiring} lr={args.lr} wd={args.weight_decay} '
        f'bs={args.batch_size} '
        f'{"epochs=" + str(args.epochs) if use_epochs else "iterations=" + str(args.num_iterations)} '
        f'seed={args.seed} device={args.device}')

    ckpt_metadata = {'args': vars(args)}
    best_test_acc = 0.0
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(args.resume, model, optimizer, device=args.device)
        best_test_acc = ckpt_info['best_metric']
        print(f'  ステップ {ckpt_info["step"]} から再開 '
              f'(best_test_acc={best_test_acc:.4f})')

    # wandb
    run_name = f'cifar_pretrain_{args.target}_k{args.k}_d{args.tree_depth}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['w13', 'cifar_pretrain', args.target, f'k{args.k}'],
        group='detseg-web',
    )

    batches_per_epoch = len(train_loader)
    if use_epochs:
        num_iterations = args.epochs * batches_per_epoch
        eval_freq = args.eval_freq if args.eval_freq is not None else 1
        print(f'\n訓練開始: {args.epochs}エポック '
              f'({batches_per_epoch}バッチ/エポック, 評価={eval_freq}エポックごと)')
    else:
        num_iterations = args.num_iterations
        eval_freq = args.eval_freq if args.eval_freq is not None else 2000
        print(f'\n訓練開始: {num_iterations:,}イテレーション '
              f'(評価={eval_freq:,}イテレーションごと)')
    print('=' * 70)

    train_start = time.time()

    if use_epochs:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}', leave=False)
            for x, y in pbar:
                x = x.to(torch.float32).to(args.device)
                y = y.to(args.device)
                loss = train_step(model, x, y, loss_fn, optimizer)
                epoch_loss += loss
                elapsed_t = time.time() - train_start
                progress = (epoch - 1 + (pbar.n + 1) / len(train_loader)) / args.epochs
                remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
                pbar.set_postfix(loss=f'{loss:.4f}', eta=eta_str)
            avg_loss = epoch_loss / batches_per_epoch

            if epoch % eval_freq == 0 or epoch == args.epochs:
                test_acc = evaluate(model, test_loader, args.device, mode=False,
                                    desc=f'eval(hard) {epoch}/{args.epochs}')
                test_acc_soft = evaluate(model, test_loader, args.device, mode=True,
                                         desc=f'eval(soft) {epoch}/{args.epochs}')
                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc

                step_now = epoch * batches_per_epoch
                save_checkpoint(latest_path, model, optimizer, None,
                                step_now, best_test_acc, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    step_now, best_test_acc, ckpt_metadata)

                elapsed = time.time() - train_start
                remaining = elapsed / epoch * (args.epochs - epoch)
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))

                print(f'[epoch {epoch:>3d}/{args.epochs}] loss={avg_loss:.4f}  '
                      f'test_acc(hard)={test_acc:.4f}  test_acc(soft)={test_acc_soft:.4f}'
                      f'{"  ★ BEST" if is_best else ""}'
                      f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]')
                log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
                    f'test_acc_hard={test_acc:.4f} test_acc_soft={test_acc_soft:.4f} '
                    f'best={str(is_best).lower()}')
                if use_wandb:
                    log_metrics({
                        'epoch': epoch, 'loss': avg_loss,
                        'test_acc_hard': test_acc, 'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })
    else:
        pbar = tqdm(enumerate(load_n(train_loader, num_iterations)),
                   desc='train', total=num_iterations)
        for i, (x, y) in pbar:
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)
            model.train()
            loss = train_step(model, x, y, loss_fn, optimizer)

            epoch_now = (i + 1) / batches_per_epoch
            elapsed_t = time.time() - train_start
            sec_per_iter = elapsed_t / (i + 1)
            remaining_sec = sec_per_iter * (num_iterations - (i + 1))
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{loss:.4f}', epoch=f'{epoch_now:.1f}', eta=eta_str)

            if (i + 1) % eval_freq == 0 or (i + 1) == num_iterations:
                test_acc = evaluate(model, test_loader, args.device, mode=False,
                                    desc=f'eval(hard) {i+1}/{num_iterations}')
                test_acc_soft = evaluate(model, test_loader, args.device, mode=True,
                                         desc=f'eval(soft) {i+1}/{num_iterations}')
                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc

                save_checkpoint(latest_path, model, optimizer, None,
                                i + 1, best_test_acc, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    i + 1, best_test_acc, ckpt_metadata)

                elapsed = time.time() - train_start
                remaining = elapsed / (i + 1) * (num_iterations - (i + 1))
                eta_str = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))

                print(f'\n[iter {i+1:,} / epoch {epoch_now:.1f}] loss={loss:.4f}  '
                      f'test_acc(hard)={test_acc:.4f}  test_acc(soft)={test_acc_soft:.4f}'
                      f'{"  ★ BEST" if is_best else ""}'
                      f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]')
                log(f'ITER {i+1}/{num_iterations} epoch={epoch_now:.1f} '
                    f'loss={loss:.4f} test_acc_hard={test_acc:.4f} '
                    f'test_acc_soft={test_acc_soft:.4f} best={str(is_best).lower()}')
                if use_wandb:
                    log_metrics({
                        'iteration': i + 1, 'epoch': epoch_now, 'loss': loss,
                        'test_acc_hard': test_acc, 'test_acc_soft': test_acc_soft,
                        'best_test_acc': best_test_acc,
                    })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度(hard): {best_test_acc:.4f}  (総訓練時間: {total_time})')
    print(f'BEST_CHECKPOINT={best_path}')
    log(f'DONE best_test_acc={best_test_acc:.4f} total_time={total_time} '
        f'best_ckpt={best_path}')
    log_file.close()

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_test_acc': best_test_acc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
