"""
教師CNN訓練スクリプト(ROADMAP-v5 Track D)

蒸留(C4)・直接転写/回復学習(C1〜C3)の変換元として使う小型CNNを訓練する。
保存したチェックポイントは `griffith_logic/convert/teacher.py` の
`resolve_teacher('ckpt:<path>')` でそのまま読み戻せる形式
(`{'model_state_dict', 'metadata': {'model_class', 'num_classes',
'input_size', 'model_kwargs', 'transform', 'fingerprint'}}`)で保存する。

目安精度(ローカル数分、既定epochs=5):
    MNIST         ~99.2%
    Fashion-MNIST ~93%

使用例:
    # MNIST
    uv run python experiments/train_teacher_cnn.py --dataset mnist --epochs 5

    # Fashion-MNIST
    uv run python experiments/train_teacher_cnn.py --dataset fashion_mnist --epochs 5

    # 煙実行(数十バッチのみ、wandb無効)
    uv run python experiments/train_teacher_cnn.py --dataset mnist --epochs 1 \\
        --limit-batches 20 --no-wandb

    # Modal(GPU)実行
    uv run modal run scripts/modal_app.py --script train_teacher_cnn.py \\
        --args "--dataset mnist --epochs 5"

読み戻し例:
    from griffith_logic.convert.teacher import resolve_teacher
    teacher = resolve_teacher('ckpt:checkpoints/teacher_cnn_mnist_.../best.pt')
    logits = teacher.predict_logits(batch)  # (B, num_classes)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic.convert.teacher_models import SmallTeacherCNN

# .envから環境変数を読み込み
load_dotenv()


# ============================================================
# データセット
# ============================================================

DATASET_CHOICES = ['mnist', 'fashion_mnist']

# datasets/vision.py が既定で二値化(binarize_threshold=0.5)するため、
# 教師CNNは実数入力で学習する(binarize_threshold=Noneで無効化)。
_VISION_LOADER_NAME = {
    'mnist': 'load_mnist',
    'fashion_mnist': 'load_fashion_mnist',
}


def load_dataset(args):
    """(train_loader, test_loader) を返す。

    griffith_logic.datasets.vision に該当ローダがあればそれを使い
    (root自動発見: repo直下 data-mnist / data-fashion-mnist)、
    無ければtorchvisionを直接使う(ToTensorのみ、実数[0,1]入力)。
    """
    try:
        from griffith_logic.datasets import vision as vision_mod
        loader_name = _VISION_LOADER_NAME[args.dataset]
        loader_fn = getattr(vision_mod, loader_name, None)
    except ImportError:
        loader_fn = None

    if loader_fn is not None:
        train_set, test_set = loader_fn(binarize_threshold=None)
    else:
        import torchvision

        transform = torchvision.transforms.ToTensor()
        if args.dataset == 'mnist':
            train_set = torchvision.datasets.MNIST(
                './data-mnist', train=True, download=True, transform=transform,
            )
            test_set = torchvision.datasets.MNIST(
                './data-mnist', train=False, transform=transform,
            )
        else:
            train_set = torchvision.datasets.FashionMNIST(
                './data-fashion-mnist', train=True, download=True, transform=transform,
            )
            test_set = torchvision.datasets.FashionMNIST(
                './data-fashion-mnist', train=False, transform=transform,
            )

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

def train_step(model, x, y, loss_fn, optimizer):
    output = model(x)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, loader, device, desc=None, limit_batches=None):
    orig_mode = model.training
    model.eval()
    correct = 0
    total = 0
    it = tqdm(loader, desc=desc, leave=False) if desc else loader
    for i, (x, y) in enumerate(it):
        if limit_batches is not None and i >= limit_batches:
            break
        x = x.to(torch.float32).to(device)
        y = y.to(device)
        output = model(x)
        pred = output.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
        if desc and isinstance(it, tqdm):
            it.set_postfix(acc=f'{correct/total:.4f}' if total else '0.0000')
    model.train(mode=orig_mode)
    return correct / total if total > 0 else 0.0


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='教師CNN訓練スクリプト(蒸留・変換元teacher用)'
    )

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=DATASET_CHOICES,
                        help='データセット (default: mnist)')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='エポック数 (default: 5)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学習率 (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='全結合層前のdropout率 (default: 0.25)')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='ログ出力先ディレクトリ (default: logs)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--limit-batches', type=int, default=None,
                        help='1エポックあたりの最大バッチ数(煙実行用。'
                             '評価バッチ数も同数に制限)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbを無効化する(動作確認・煙実行用)')

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

    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # データ読み込み
    train_loader, test_loader = load_dataset(args)
    print(f'データセット: {args.dataset}')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル')

    num_classes = 10
    in_channels = 1
    input_size = (in_channels, 28, 28)

    model_kwargs = dict(
        num_classes=num_classes, in_channels=in_channels, dropout=args.dropout,
    )
    model = SmallTeacherCNN(**model_kwargs).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'モデル: SmallTeacherCNN(num_classes={num_classes}, '
          f'in_channels={in_channels}, dropout={args.dropout})')
    print(f'総パラメータ数: {total_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # ログファイル
    log, log_file = setup_logger('train_teacher_cnn', args.dataset, args.log_dir)
    log(f'START dataset={args.dataset} epochs={args.epochs} lr={args.lr} '
        f'weight_decay={args.weight_decay} bs={args.batch_size} '
        f'seed={args.seed} device={args.device} '
        f'limit_batches={args.limit_batches}')

    # チェックポイント(相対パス規約: Modal互換)
    ckpt_prefix = f'teacher_cnn_{args.dataset}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    print(f'チェックポイント保存先: {run_dir}/')

    # resolve_teacher('ckpt:...') が要求するmetadata
    # (griffith_logic/convert/teacher.py `_resolve_ckpt_teacher` 参照)
    ckpt_metadata = {
        'args': vars(args),
        'model_class': SmallTeacherCNN,
        'model_kwargs': model_kwargs,
        'num_classes': num_classes,
        'input_size': input_size,
        # 実数[0,1]入力をそのまま使う(正規化なし)ため恒等transform
        'transform': None,
    }

    # wandb
    run_name = f'{args.dataset}_e{args.epochs}_bs{args.batch_size}_lr{args.lr}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict=vars(args),
        model=model,
        device=args.device,
        tags=['teacher', 'cnn', args.dataset],
        group='teacher',
    )

    batches_per_epoch = (
        len(train_loader) if args.limit_batches is None
        else min(args.limit_batches, len(train_loader))
    )
    print(f'\n訓練開始: {args.epochs}エポック '
          f'({batches_per_epoch}バッチ/エポック)')
    print('=' * 70)

    best_test_acc = 0.0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f'epoch {epoch}/{args.epochs}',
            leave=False,
            total=batches_per_epoch,
        )
        for i, (x, y) in enumerate(pbar):
            if args.limit_batches is not None and i >= args.limit_batches:
                break
            x = x.to(torch.float32).to(args.device)
            y = y.to(args.device)
            loss = train_step(model, x, y, loss_fn, optimizer)
            epoch_loss += loss
            n_batches += 1

            elapsed_t = time.time() - train_start
            progress = (epoch - 1 + (i + 1) / batches_per_epoch) / args.epochs
            remaining_sec = elapsed_t * (1 - progress) / progress if progress > 0 else 0
            eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(loss=f'{loss:.4f}', eta=eta_str)

        avg_loss = epoch_loss / max(n_batches, 1)

        elapsed = time.time() - train_start
        sec_per_epoch = elapsed / epoch
        remaining = sec_per_epoch * (args.epochs - epoch)
        eta = datetime.now(JST) + timedelta(seconds=remaining)

        test_acc = evaluate(
            model, test_loader, args.device,
            desc=f'eval {epoch}/{args.epochs}',
            limit_batches=args.limit_batches,
        )

        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc

        save_checkpoint(latest_path, model, optimizer, None,
                        epoch * batches_per_epoch, best_test_acc, ckpt_metadata)
        if is_best:
            save_checkpoint(best_path, model, optimizer, None,
                            epoch * batches_per_epoch, best_test_acc, ckpt_metadata)

        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remain_str = str(timedelta(seconds=int(remaining)))
        eta_str = eta.strftime('%H:%M:%S')

        print(
            f'[epoch {epoch:>3d}/{args.epochs}] '
            f'loss={avg_loss:.4f}  '
            f'test_acc={test_acc:.4f}'
            f'{"  * BEST" if is_best else ""}'
            f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
        )

        log(f'EPOCH {epoch}/{args.epochs} loss={avg_loss:.4f} '
            f'test_acc={test_acc:.4f} best={str(is_best).lower()} '
            f'elapsed={elapsed_str} eta={remain_str}')
        if use_wandb:
            log_metrics({
                'epoch': epoch,
                'loss': avg_loss,
                'test_acc': test_acc,
                'best_test_acc': best_test_acc,
            })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度: {best_test_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_test_acc={best_test_acc:.4f} total_time={total_time}')
    log_file.close()

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_test_acc': best_test_acc},
            model=model,
        )
        wandb_finish()

    print(f'\nckpt: {best_path}')
    print("resolve_teacher('ckpt:" + best_path + "') で読み込み可能")


if __name__ == '__main__':
    main()
