"""LogicResNet(残差論理ConvNet)訓練スクリプト。

CDDLGN(LogicTreeNet)型の「Conv→OrPool→...→Dense」ではなく、明示的なskip接続
(Residual: x OR/XOR f(x))を持つResNet型論理ConvNetを、MNIST/CIFAR-10で訓練する。

使用例:
    # スモーク(CPU、2分以内、wandb無し)
    uv run python experiments/train_logic_resnet.py --dataset mnist --max-steps 30 --no-wandb --device cpu

    # 本番(MNIST、ステップベース、wandb有効)
    uv run python experiments/train_logic_resnet.py --dataset mnist --max-steps 20000

    # 本番(CIFAR-10、エポックベース)
    uv run python experiments/train_logic_resnet.py --dataset cifar10 --epochs 100

    # Modal実行
    uv run modal run scripts/modal_app.py --script train_logic_resnet.py --args "--dataset cifar10 --max-steps 20000"
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import (
    add_experiment_args,
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
    normalize_tags,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic.models.logic_resnet import LogicResNet, LogicResNetConfig

# .envから環境変数を読み込み
load_dotenv()

_N_CLASSES = 10  # mnist/cifar10ともに10クラス


# ============================================================
# データセット
# ============================================================
def load_dataset(args):
    """rawピクセル値[0,1](ToTensorのみ)のDataLoaderを返す。

    モデル側(LogicResNet.encode)がthermometer符号化を担うため、
    train_cnn.pyのThresholdBinarizeのような事前二値化は行わない。
    """
    import torchvision

    from griffith_logic.datasets.vision import resolve_data_root

    transform = torchvision.transforms.ToTensor()
    if args.dataset == 'mnist':
        root = resolve_data_root('data-mnist', args.data_dir)
        train_set = torchvision.datasets.MNIST(
            root, train=True, download=True, transform=transform,
        )
        test_set = torchvision.datasets.MNIST(
            root, train=False, download=True, transform=transform,
        )
        in_channels = 1
    else:
        root = resolve_data_root('data-cifar', args.data_dir)
        train_set = torchvision.datasets.CIFAR10(
            root, train=True, download=True, transform=transform,
        )
        test_set = torchvision.datasets.CIFAR10(
            root, train=False, download=True, transform=transform,
        )
        in_channels = 3

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=2,
    )
    return train_loader, test_loader, in_channels


def load_n(loader, n):
    """DataLoaderからn回分のバッチを返すジェネレータ(train_cnn.py準拠)"""
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i >= n:
                return


# ============================================================
# 評価
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> float:
    """hard eval(離散化重み)での分類精度"""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(torch.float32).to(device)
        y = y.to(device)
        logits = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.shape[0]
    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='LogicResNet(残差論理ConvNet)訓練スクリプト'
    )
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'], help='データセット (default: mnist)')

    # モデル構成
    parser.add_argument('--stem-channels', type=int, default=32,
                        help='論理チャネル幅(残差ブロックは幅不変) (default: 32)')
    parser.add_argument('--stage-blocks', type=str, default='2,2',
                        help='カンマ区切りの各stage残差ブロック数 (default: 2,2)')
    parser.add_argument('--residual-mode', type=str, default='or', choices=['or', 'xor'],
                        help='残差マージ方式 (default: or)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='ゲートの入力数 (default: 2)')
    parser.add_argument('--tree-depth', type=int, default=2,
                        help='各LutConv2dのツリー深さ (default: 2)')
    parser.add_argument('--kernel-size', type=int, default=3,
                        help='残差ブロック内convのカーネルサイズ (default: 3)')
    parser.add_argument('--encode-bits', type=int, default=4,
                        help='thermometer符号化ビット数/チャネル (default: 4)')
    parser.add_argument('--head-group-size', type=int, default=4,
                        help='head 1x1convの出力=num_classes*head_group_size (default: 4)')
    parser.add_argument('--input-channels-per-tree', type=int, default=2,
                        help='各ツリーが参照する入力チャンネル数上限 (default: 2)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp'],
                        help='パラメータ化方式 (default: softmax)')
    parser.add_argument('--tau', type=float, default=None,
                        help='GroupSum温度(Noneで既定値1.0)')
    parser.add_argument('--grad-factor', type=float, default=1.0,
                        help='勾配スケーリング係数 (default: 1.0)')

    # 最適化
    parser.add_argument('--lr', type=float, default=0.01, help='学習率 (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大ステップ数(--epochsと排他)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='エポック数(--max-stepsと排他)')
    parser.add_argument('--eval-freq', type=int, default=200,
                        help='評価頻度(ステップベース時のみ) (default: 200)')
    parser.add_argument('--eval-batches', type=int, default=None,
                        help='評価に使うバッチ数上限(Noneで全件)')

    parser.add_argument('--seed', type=int, default=0, help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None, help='デバイス (default: 自動検出)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='データ保存先 (default: repo直下 data-mnist/data-cifar を自動発見)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化(スモーク用)')
    add_experiment_args(parser)

    args = parser.parse_args()

    if args.max_steps is not None and args.epochs is not None:
        parser.error('--max-steps と --epochs は同時に指定できません')
    if args.max_steps is None and args.epochs is None:
        args.max_steps = 3000  # どちらも未指定ならステップベースの既定値

    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    device = args.device
    print(f'デバイス: {device}')

    torch.manual_seed(args.seed)

    stage_blocks = tuple(int(v) for v in args.stage_blocks.split(','))

    # データ読み込み
    train_loader, test_loader, in_channels = load_dataset(args)
    print(f'データセット: {args.dataset}')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル')

    # モデル構築
    config = LogicResNetConfig(
        in_channels=in_channels,
        encode_bits=args.encode_bits,
        stem_channels=args.stem_channels,
        stage_blocks=stage_blocks,
        residual_mode=args.residual_mode,
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        num_classes=_N_CLASSES,
        kernel_size=args.kernel_size,
        tree_depth=args.tree_depth,
        head_group_size=args.head_group_size,
        input_channels_per_tree=args.input_channels_per_tree,
        grad_factor=args.grad_factor,
        tau=args.tau if args.tau is not None else 1.0,
        device=device,
    )
    model = LogicResNet(config).to(device)

    print(f'\nモデル構成: stem_channels={args.stem_channels}, '
          f'stage_blocks={stage_blocks}, residual_mode={args.residual_mode}, '
          f'n_inputs={args.n_inputs}, tree_depth={args.tree_depth}')
    print(f'総ゲート数: {model.num_gates:,}')
    print(f'総パラメータ数: {sum(p.numel() for p in model.parameters()):,}')
    print(model)

    # オプティマイザ(AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # チェックポイント
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        f'logic_resnet_{args.dataset}', base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}
    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    use_steps = args.max_steps is not None
    run_name = (f'{args.dataset}_sc{args.stem_channels}'
                f'_sb{"-".join(map(str, stage_blocks))}_{args.residual_mode}'
                f'_n{args.n_inputs}_'
                f'{"s" + str(args.max_steps) if use_steps else "e" + str(args.epochs)}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=device,
            tags=['logic_resnet', args.dataset, args.residual_mode, f'n{args.n_inputs}']
            + normalize_tags(model='logic_resnet', dataset=args.dataset, seed=args.seed),
            group='logic_resnet',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    print(f'\n訓練開始: {"ステップ" if use_steps else "エポック"}ベース '
          f'({args.max_steps if use_steps else args.epochs})')
    print('=' * 70)

    best_acc = 0.0
    train_start = time.time()

    if use_steps:
        step = 0
        for x, y in load_n(train_loader, args.max_steps):
            step += 1
            model.train()
            x = x.to(torch.float32).to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.eval_freq == 0 or step == args.max_steps or step == 1:
                acc = evaluate(model, test_loader, device, max_batches=args.eval_batches)
                is_best = acc > best_acc
                if is_best:
                    best_acc = acc

                save_checkpoint(latest_path, model, optimizer, None,
                                step, best_acc, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    step, best_acc, ckpt_metadata)

                elapsed = time.time() - train_start
                sec_per_step = elapsed / step
                remaining = sec_per_step * (args.max_steps - step)
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                print(
                    f'[step {step:>6d}/{args.max_steps}] '
                    f'loss={loss.item():.4f}  test_acc={acc:.4f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
                )
                if use_wandb:
                    log_metrics({
                        'step': step, 'loss': loss.item(),
                        'test_acc': acc, 'best_acc': best_acc,
                    })
    else:
        batches_per_epoch = len(train_loader)
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                x = x.to(torch.float32).to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / batches_per_epoch

            acc = evaluate(model, test_loader, device, max_batches=args.eval_batches)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc

            step_now = epoch * batches_per_epoch
            save_checkpoint(latest_path, model, optimizer, None,
                            step_now, best_acc, ckpt_metadata)
            if is_best:
                save_checkpoint(best_path, model, optimizer, None,
                                step_now, best_acc, ckpt_metadata)

            elapsed = time.time() - train_start
            sec_per_epoch = elapsed / epoch
            remaining = sec_per_epoch * (args.epochs - epoch)
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            print(
                f'[epoch {epoch:>3d}/{args.epochs}] '
                f'loss={avg_loss:.4f}  test_acc={acc:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, 終了予定 {eta.strftime("%H:%M:%S")}]'
            )
            if use_wandb:
                log_metrics({
                    'epoch': epoch, 'loss': avg_loss,
                    'test_acc': acc, 'best_acc': best_acc,
                })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print('=' * 70)
    print(f'\n最終テスト精度: {best_acc:.4f}  (総訓練時間: {total_time})')

    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_test_acc': best_acc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
