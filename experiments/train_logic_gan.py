"""LogicGAN(GANの論理ゲート化)訓練スクリプト。

乱数ビット列を論理回路(LUTツリー)に通して画像を生成する GAN を、binary MNIST で
敵対学習する。Generator は「乱数ビット列 → 論理回路 → 画像ビット」、Discriminator は
画素を LUT スタック + GroupSum で real/fake 判定する(詳細は
griffith_logic/models/logic_gan.py)。

敵対的勾配 × 離散出力は本質的に不安定なため、安定化オプション(D入力ビット反転ノイズ・
label smoothing・G:D 更新比)を用意する。品質基準は本走で較正する。

使用例:
    # 基本実行(binary MNIST)
    uv run python experiments/train_logic_gan.py --epochs 30

    # スモークテスト(CPU・wandbなし・2分以内)
    uv run python experiments/train_logic_gan.py --max-steps 30 --no-wandb --device cpu

    # 安定化を強めに(Gを多く更新・入力ノイズ増)
    uv run python experiments/train_logic_gan.py --epochs 30 --g-steps 2 --flip-prob 0.15
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
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

from griffith_logic.datasets.vision import load_mnist
from griffith_logic.models.logic_gan import LogicGAN, LogicGANConfig

load_dotenv()


def build_loader(args):
    """binary MNIST(画素>0.5)の train_loader を返す。"""
    train_set, _ = load_mnist(binarize_threshold=0.5, download=True)
    use_pin = args.device == 'cuda'
    return torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin, drop_last=True, num_workers=2,
    )


def bit_flip(x: torch.Tensor, p: float) -> torch.Tensor:
    """Discriminator 入力へのインスタンスノイズ(ビット反転)。

    p=0 なら恒等。real/fake の両方に適用して D の過学習を抑え、敵対バランスを保つ。
    """
    if p <= 0.0:
        return x
    mask = (torch.rand_like(x) < p).to(x.dtype)
    return x * (1.0 - mask) + (1.0 - x) * mask


@torch.no_grad()
def make_sample_grid(model, n: int, device):
    """prior 乱数ビット列からの hard 生成グリッドを返す(eval時=純ブール回路)。"""
    was_training = model.training
    model.eval()
    gen = model.generate(n)  # (n,C,H,W){0,1}
    grid = torchvision.utils.make_grid(gen.to(device), nrow=min(n, 8), padding=2)
    if was_training:
        model.train()
    return grid


def save_grid(grid, step, output_dir, use_wandb):
    """生成グリッドをローカル保存し、wandb にも必ずログする(リポジトリ規約)。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'gen_step{step:06d}.png')
    torchvision.utils.save_image(grid, path)
    print(f'  画像を保存: {path}')
    if use_wandb:
        from torchvision.transforms.functional import to_pil_image
        log_images('logic_gan/samples', [to_pil_image(grid)],
                   captions=[f'step {step} (乱数ビット列→論理回路 生成)'])


def main():
    parser = argparse.ArgumentParser(
        description='LogicGAN(GANの論理ゲート化)訓練スクリプト'
    )
    parser.add_argument('--noise-bits', type=int, default=64,
                        help='入力乱数ビット列の長さ (default: 64)')
    parser.add_argument('--hidden', type=int, default=512,
                        help='Generator の LutLayer 幅 (default: 512)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Generator の LutLayer スタック段数 (default: 3)')
    parser.add_argument('--disc-hidden', type=int, default=256,
                        help='Discriminator の LutLayer 幅 (default: 256)')
    parser.add_argument('--disc-depth', type=int, default=2,
                        help='Discriminator の LutLayer スタック段数 (default: 2)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='LUT の入力数 (default: 2)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'raw', 'warp'],
                        help='パラメータ化方式 (default: softmax)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Generator の学習率 (default: 0.01)')
    parser.add_argument('--disc-lr', type=float, default=0.003,
                        help='Discriminator の学習率(小さめでDの過学習抑制) (default: 0.003)')
    parser.add_argument('--g-steps', type=int, default=1,
                        help='Dの1更新あたりのG更新回数(G:D更新比) (default: 1)')
    parser.add_argument('--flip-prob', type=float, default=0.05,
                        help='D入力へのビット反転ノイズ確率(安定化) (default: 0.05)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='real ターゲットの label smoothing(安定化) (default: 0.1)')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='エポック数 (default: 30)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='最大ステップ数(スモーク用、エポックより優先)')
    parser.add_argument('--eval-freq', type=int, default=500,
                        help='評価・画像ログの頻度(ステップ) (default: 500)')
    parser.add_argument('--seed', type=int, default=0,
                        help='ランダムシード (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス (default: 自動検出)')
    parser.add_argument('--output-dir', type=str, default='outputs/logic_gan',
                        help='画像の保存先 (default: outputs/logic_gan)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbログを無効化')
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # データ
    train_loader = build_loader(args)
    print(f'binary MNIST: 訓練 {len(train_loader.dataset):,}')

    # モデル
    cfg = LogicGANConfig(
        image_size=28, in_channels=1,
        noise_bits=args.noise_bits, hidden=args.hidden, depth=args.depth,
        n_inputs=args.n_inputs, parameterization=args.parameterization,
        disc_hidden=args.disc_hidden, disc_depth=args.disc_depth,
    )
    model = LogicGAN(cfg).to(args.device)
    n_g = sum(p.numel() for p in model.generator.parameters())
    n_d = sum(p.numel() for p in model.discriminator.parameters())
    print(f'LogicGAN: noise_bits={args.noise_bits} hidden={args.hidden} '
          f'depth={args.depth} disc_hidden={args.disc_hidden} '
          f'disc_depth={args.disc_depth} n_inputs={args.n_inputs} '
          f'param={args.parameterization}')
    print(f'パラメータ数: Generator={n_g:,} / Discriminator={n_d:,}')
    print(f'安定化: g_steps={args.g_steps} flip_prob={args.flip_prob} '
          f'label_smoothing={args.label_smoothing} disc_lr={args.disc_lr}')

    opt_g = torch.optim.Adam(model.generator.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.disc_lr)

    # ステップ数の決定
    batches_per_epoch = len(train_loader)
    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * batches_per_epoch
    print(f'総ステップ数: {total_steps:,} ({batches_per_epoch}バッチ/エポック)')

    # チェックポイント
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        'logic_gan', base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}
    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'lgan_z{args.noise_bits}_h{args.hidden}_d{args.depth}'
                f'_dh{args.disc_hidden}_n{args.n_inputs}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=args.device,
            tags=['logic_gan', 'mnist', f'n{args.n_inputs}',
                  f'z{args.noise_bits}'],
            group='logic_gan',
        )

    print(f'\n訓練開始: {total_steps:,}ステップ')
    print('=' * 60)
    train_start = time.time()
    # 敵対バランスの目安として D精度が0.5に近いほど良い(完全収束は要求しない)
    best_dacc_gap = float('inf')  # |D精度 - 0.5| の最小値

    model.train()
    step = 0
    done = False
    while not done:
        for x, _ in train_loader:
            real = x.to(torch.float32).to(args.device)
            bsz = real.shape[0]

            # --- Discriminator 更新 ---
            z = model.generator.sample_noise(bsz)
            fake = model.generator(z)
            opt_d.zero_grad()
            d_loss = model.d_loss(
                bit_flip(real, args.flip_prob),
                bit_flip(fake, args.flip_prob),
                label_smoothing=args.label_smoothing,
            )
            d_loss.backward()
            opt_d.step()

            # --- Generator 更新(G:D = g_steps:1) ---
            g_loss_val = 0.0
            for _ in range(args.g_steps):
                z = model.generator.sample_noise(bsz)
                fake = model.generator(z)
                g_loss = model.g_loss(model.discriminator(fake))
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_loss_val = g_loss.item()
            step += 1

            if step % args.eval_freq == 0 or step == total_steps:
                d_loss_v = d_loss.item()
                # D精度(敵対バランスの監視)
                with torch.no_grad():
                    real_acc = (model.discriminator(real) > 0).float().mean().item()
                    fake_acc = (model.discriminator(fake) < 0).float().mean().item()
                    dacc = (real_acc + fake_acc) / 2.0
                gap = abs(dacc - 0.5)

                is_best = gap < best_dacc_gap
                if is_best:
                    best_dacc_gap = gap
                save_checkpoint(latest_path, model, opt_g, None,
                                step, best_dacc_gap, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, opt_g, None,
                                    step, best_dacc_gap, ckpt_metadata)

                grid = make_sample_grid(model, 16, args.device)
                save_grid(grid, step, args.output_dir, use_wandb)

                elapsed = time.time() - train_start
                progress = step / total_steps
                remaining = elapsed * (1 - progress) / progress if progress > 0 else 0
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')

                print(
                    f'[step {step:>6d}/{total_steps}] '
                    f'd_loss={d_loss_v:.3f}  g_loss={g_loss_val:.3f}  '
                    f'D_acc={dacc:.3f}(real={real_acc:.2f}/fake={fake_acc:.2f})'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )
                if use_wandb:
                    log_metrics({
                        'step': step,
                        'd_loss': d_loss_v,
                        'g_loss': g_loss_val,
                        'd_acc': dacc,
                        'd_acc_real': real_acc,
                        'd_acc_fake': fake_acc,
                        'best_dacc_gap': best_dacc_gap,
                    })

            if step >= total_steps:
                done = True
                break

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最良D精度gap: {best_dacc_gap:.4f}(0が理想)  (総訓練時間: {total_time})')

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_dacc_gap': best_dacc_gap},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
