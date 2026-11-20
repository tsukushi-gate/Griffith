"""D6 LogicBinaryDiffusion 訓練スクリプト(二値MNIST生成)。

XORノイズ離散拡散(デノイザ=完全論理ゲート)を binary MNIST で学習する。
    - 前向き過程: x_t = x_0 ⊕ Bernoulli(flip_prob(t))(BDPM準拠、XOR)
    - デノイザ: LogicUNet(全段 LutConv2d)による x_0 予測
    - 損失: ランダム t で q_sample → デノイザ → BCEWithLogits(x_0予測)
    - 生成: 完全ノイズ Bernoulli(0.5) から逆過程で逐次デノイズ

逆過程の生成グリッド(ステップ経過含む)を outputs/logic_diffusion/ とwandbの
両方へ画像ログする。

使用例:
    # デフォルト設定で学習
    uv run python experiments/train_logic_diffusion.py

    # スモーク(2分以内、CPU、wandbなし)
    uv run python experiments/train_logic_diffusion.py --max-steps 30 --no-wandb --device cpu

    # 拡散ステップ数・デノイザ深さ・幅を変更
    uv run python experiments/train_logic_diffusion.py --timesteps 16 --depth 2 --base-channels 32

    # Modal 実行(GPU)
    uv run modal run scripts/modal_app.py --script train_logic_diffusion.py --args "--epochs 4"
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
from griffith_logic.models.logic_diffusion import (
    LogicBinaryDiffusion,
    LogicDiffusionConfig,
)

load_dotenv()

# binary MNIST の二値化閾値
BINARIZE_THRESHOLD = 0.5


# ============================================================
# データ
# ============================================================

def load_dataset(args):
    """binary MNIST(28×28 {0,1})の train/test DataLoader を返す。"""
    train_set, test_set = load_mnist(binarize_threshold=BINARIZE_THRESHOLD)
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

@torch.no_grad()
def evaluate_denoise_acc(model, loader, device, timesteps, max_batches=None):
    """eval(hard)経路での x_0 復元精度を、小ノイズ(t=1)・高ノイズ(t=T)で測る。

    - acc@t1: 低ノイズでの x_0 予測精度
    - acc@tT / copy@tT: 高ノイズでの復元精度と passthrough(複写)ベースライン
      (model が copy を上回れば実際にノイズ除去できている)
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    sums = {'acc_t1': 0.0, 'acc_tT': 0.0, 'copy_tT': 0.0, 'n': 0}
    for i, (x0, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x0 = x0.to(torch.float32).to(device)
        # 小ノイズ t=1
        x_t1 = model.q_sample(x0, 1, generator=gen)
        acc_t1 = ((model.forward(x_t1, 1) > 0).float() == x0).float().mean().item()
        # 高ノイズ t=T
        x_tT = model.q_sample(x0, timesteps, generator=gen)
        acc_tT = ((model.forward(x_tT, timesteps) > 0).float() == x0).float().mean().item()
        copy_tT = (x_tT == x0).float().mean().item()
        sums['acc_t1'] += acc_t1
        sums['acc_tT'] += acc_tT
        sums['copy_tT'] += copy_tT
        sums['n'] += 1
    n = max(1, sums['n'])
    return {
        'acc_t1': sums['acc_t1'] / n,
        'acc_tT': sums['acc_tT'] / n,
        'copy_tT': sums['copy_tT'] / n,
    }


@torch.no_grad()
def make_sample_grid(model, device, n: int = 8, generator=None):
    """逆過程の生成グリッド(行=サンプル、列=ステップ経過)を PIL 画像で返す。"""
    import torchvision
    from torchvision.transforms.functional import to_pil_image

    model_mode = model.training
    model.eval()
    _final, trajectory = model.sample(n, generator=generator, return_trajectory=True)
    # trajectory: 各ステップ (n, 1, H, W)。列=ステップ、行=サンプルに並べる。
    # 全ステップを時系列に並べたグリッド(nrow=ステップ数)。
    steps = len(trajectory)
    stacked = torch.cat(trajectory, dim=0).cpu()  # (steps*n, 1, H, W)
    # サンプルごとに行、ステップごとに列にするため並べ替える
    stacked = stacked.reshape(steps, n, *stacked.shape[1:]).permute(1, 0, 2, 3, 4)
    stacked = stacked.reshape(steps * n, *stacked.shape[2:])
    grid = torchvision.utils.make_grid(stacked, nrow=steps, padding=1)
    model.train(mode=model_mode)
    return to_pil_image(grid)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='D6 LogicBinaryDiffusion 訓練スクリプト(二値MNIST生成)',
    )
    parser.add_argument('--timesteps', type=int, default=16,
                        help='拡散ステップ数 T (default: 16)')
    parser.add_argument('--flip-prob-max', type=float, default=0.5,
                        help='t=T でのビット反転率 (default: 0.5)')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='デノイザ最浅stageの論理チャネル幅 (default: 32)')
    parser.add_argument('--depth', type=int, default=2,
                        help='デノイザ(LogicUNet)の down/up 段数 (default: 2)')
    parser.add_argument('--time-embed-bits', type=int, default=8,
                        help='時刻 t の thermometer 符号ビット数 (default: 8)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='LUTの入力数 (default: 2)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp'],
                        help='LUTパラメータ化方式 (default: softmax)')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='学習率 (default: 0.02)')
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='バッチサイズ (default: 64)')
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
    parser.add_argument('--output-dir', type=str, default='outputs/logic_diffusion',
                        help='生成画像の保存先 (default: outputs/logic_diffusion)')
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
    print('データセット: binary MNIST (28×28 {0,1})')
    print(f'訓練: {len(train_loader.dataset):,}サンプル, '
          f'テスト: {len(test_loader.dataset):,}サンプル')

    config = LogicDiffusionConfig(
        image_size=28,
        in_channels=1,
        timesteps=args.timesteps,
        flip_prob_max=args.flip_prob_max,
        base_channels=args.base_channels,
        depth=args.depth,
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        time_embed_bits=args.time_embed_bits,
    )
    model = LogicBinaryDiffusion(config).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'モデル構成: LogicBinaryDiffusion T={args.timesteps} depth={args.depth} '
          f'base={args.base_channels} N={args.n_inputs} param={args.parameterization}')
    print(f'総パラメータ数: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_prefix = 'logic_diffusion_mnist'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}
    print(f'チェックポイント保存先: {run_dir}/')

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(args.resume, model, optimizer, device=args.device)
        start_epoch = ckpt_info['step']
        best_acc = ckpt_info['best_metric']
        print(f'  エポック {start_epoch} から再開 (best_acc={best_acc:.4f})')

    # wandb
    run_name = (f'mnist_T{args.timesteps}_d{args.depth}_c{args.base_channels}_'
                f'n{args.n_inputs}_e{args.epochs}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=args.device,
            tags=['logic_diffusion', 'diffusion', 'mnist', f'n{args.n_inputs}',
                  f'T{args.timesteps}', f'd{args.depth}']
            + normalize_tags(model='logic_diffusion', dataset='mnist', seed=args.seed),
            group='logic_diffusion',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    # 生成グリッド用の固定ジェネレータ(進捗を比較しやすくする)
    sample_gen = torch.Generator(device=args.device).manual_seed(123)

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
        for x0, _ in train_loader:
            x0 = x0.to(torch.float32).to(args.device)

            out = model.loss(x0)
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                elapsed_t = time.time() - train_start
                print(f'  step {global_step} loss={loss.item():.4f} '
                      f't_mean={out["t_mean"].item():.1f} [{elapsed_t:.0f}s経過]')

            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        avg_loss = epoch_loss / max(1, n_batches)

        # 評価(スモーク時はバッチ数を絞る)
        eval_batches = 5 if args.max_steps is not None else 20
        metrics = evaluate_denoise_acc(
            model, test_loader, args.device, args.timesteps,
            max_batches=eval_batches,
        )
        # 高ノイズでの復元マージン(model が copy をどれだけ上回るか)
        denoise_margin = metrics['acc_tT'] - metrics['copy_tT']

        is_best = metrics['acc_t1'] > best_acc
        if is_best:
            best_acc = metrics['acc_t1']

        save_checkpoint(latest_path, model, optimizer, None,
                        epoch, best_acc, ckpt_metadata)
        if is_best:
            save_checkpoint(best_path, model, optimizer, None,
                            epoch, best_acc, ckpt_metadata)

        # 生成グリッドを保存(ローカル + wandb)
        grid = make_sample_grid(model, args.device, n=8, generator=sample_gen)
        os.makedirs(args.output_dir, exist_ok=True)
        grid_path = os.path.join(args.output_dir, f'sample_epoch{epoch:03d}.png')
        grid.save(grid_path)
        print(f'  生成グリッドを保存: {grid_path} (行=サンプル, 列=逆過程ステップ)')
        if use_wandb:
            log_images('logic_diffusion/samples', [grid],
                       captions=[f'epoch {epoch} (rows=samples, cols=reverse steps)'])

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
            f'loss={avg_loss:.4f}  acc@t1={metrics["acc_t1"]:.4f}  '
            f'acc@tT={metrics["acc_tT"]:.4f}(copy={metrics["copy_tT"]:.4f}, '
            f'margin={denoise_margin:+.4f})'
            f'{"  ★ BEST" if is_best else ""}'
            f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
        )

        if use_wandb:
            log_metrics({
                'epoch': epoch,
                'loss': avg_loss,
                'acc_t1': metrics['acc_t1'],
                'acc_tT': metrics['acc_tT'],
                'copy_tT': metrics['copy_tT'],
                'denoise_margin': denoise_margin,
                'best_acc_t1': best_acc,
            })

        if stop:
            print(f'max-steps={args.max_steps} 到達で終了')
            break

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終 acc@t1(best): {best_acc:.4f}  (総訓練時間: {total_time})')

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_acc_t1': best_acc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
