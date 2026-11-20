"""LogicVAE(二値潜在VAEの論理ゲート化)訓練スクリプト。

二値潜在(Bernoulliビット列)VAEを微分可能論理ゲート(LUT)で構成し、binary MNIST を
再構成・生成する。潜在のサンプリングは「確率的閾値比較=比較器」で論理回路と整合する
(詳細は griffith_logic/models/logic_vae.py)。

使用例:
    # 基本実行(binary MNIST)
    uv run python experiments/train_logic_vae.py --epochs 30

    # スモークテスト(CPU・wandbなし・2分以内)
    uv run python experiments/train_logic_vae.py --max-steps 30 --no-wandb --device cpu

    # β(KL重み)を変えて posterior collapse を調整
    uv run python experiments/train_logic_vae.py --epochs 30 --kl-weight 0.3
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
from griffith_logic.models.logic_vae import LogicVAE, LogicVAEConfig

load_dotenv()


def build_loaders(args):
    """binary MNIST(画素>0.5)の (train_loader, test_loader) を返す。"""
    train_set, test_set = load_mnist(binarize_threshold=0.5, download=True)
    use_pin = args.device == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=use_pin, drop_last=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_pin, drop_last=True, num_workers=2,
    )
    return train_loader, test_loader


@torch.no_grad()
def make_sample_grids(model, x_ref, device):
    """再構成グリッドと prior 生成グリッドを (recon_grid, gen_grid) で返す。"""
    was_training = model.training
    model.eval()
    n = min(8, x_ref.size(0))
    x = x_ref[:n].to(device)
    out = model(x)
    recon = torch.sigmoid(out['recon_logits'])
    # 元画像 / 再構成 の2行
    recon_grid = torchvision.utils.make_grid(
        torch.cat([x, recon]), nrow=n, padding=2,
    )
    # prior Bernoulli(0.5) からの生成
    gen = model.generate(n)
    gen_grid = torchvision.utils.make_grid(gen, nrow=n, padding=2)
    if was_training:
        model.train()
    return recon_grid, gen_grid


def save_grids(recon_grid, gen_grid, step, output_dir, use_wandb):
    """グリッドをローカル保存し、wandb にも必ずログする(リポジトリ規約)。"""
    os.makedirs(output_dir, exist_ok=True)
    recon_path = os.path.join(output_dir, f'recon_step{step:06d}.png')
    gen_path = os.path.join(output_dir, f'gen_step{step:06d}.png')
    torchvision.utils.save_image(recon_grid, recon_path)
    torchvision.utils.save_image(gen_grid, gen_path)
    print(f'  画像を保存: {recon_path} / {gen_path}')
    if use_wandb:
        from torchvision.transforms.functional import to_pil_image
        log_images('logic_vae/reconstruction', [to_pil_image(recon_grid)],
                   captions=[f'step {step} (上=元画像/下=再構成)'])
        log_images('logic_vae/prior_samples', [to_pil_image(gen_grid)],
                   captions=[f'step {step} (prior Bernoulli(0.5)生成)'])


def main():
    parser = argparse.ArgumentParser(
        description='LogicVAE(二値潜在VAEの論理ゲート化)訓練スクリプト'
    )
    parser.add_argument('--latent-bits', type=int, default=64,
                        help='二値潜在ビット数 (default: 64)')
    parser.add_argument('--hidden', type=int, default=256,
                        help='エンコーダ/デコーダのLutLayer幅 (default: 256)')
    parser.add_argument('--depth', type=int, default=2,
                        help='LutLayerスタックの段数 (default: 2)')
    parser.add_argument('--encode-bits', type=int, default=4,
                        help='Thermometer符号化のビット数 (default: 4)')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6],
                        help='LUTの入力数 (default: 2)')
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'raw', 'warp'],
                        help='パラメータ化方式 (default: softmax)')
    parser.add_argument('--kl-weight', type=float, default=1.0,
                        help='KL重みβ(β-VAE、小さいほどposterior collapse回避) (default: 1.0)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学習率 (default: 0.01)')
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
    parser.add_argument('--output-dir', type=str, default='outputs/logic_vae',
                        help='画像の保存先 (default: outputs/logic_vae)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='チェックポイント保存先 (default: checkpoints)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandbログを無効化')
    add_experiment_args(parser)
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
    train_loader, test_loader = build_loaders(args)
    x_ref, _ = next(iter(test_loader))
    x_ref = x_ref.to(torch.float32)
    print(f'binary MNIST: 訓練 {len(train_loader.dataset):,} / '
          f'テスト {len(test_loader.dataset):,}')

    # モデル
    cfg = LogicVAEConfig(
        image_size=28, in_channels=1,
        encode_bits=args.encode_bits, latent_bits=args.latent_bits,
        hidden=args.hidden, depth=args.depth, n_inputs=args.n_inputs,
        parameterization=args.parameterization, kl_weight=args.kl_weight,
    )
    model = LogicVAE(cfg).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'LogicVAE: latent_bits={args.latent_bits} hidden={args.hidden} '
          f'depth={args.depth} n_inputs={args.n_inputs} '
          f'param={args.parameterization} β={args.kl_weight}')
    print(f'総パラメータ数: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ステップ数の決定
    batches_per_epoch = len(train_loader)
    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * batches_per_epoch
    print(f'総ステップ数: {total_steps:,} ({batches_per_epoch}バッチ/エポック)')

    # チェックポイント
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        'logic_vae', base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args)}
    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'lvae_L{args.latent_bits}_h{args.hidden}_d{args.depth}'
                f'_n{args.n_inputs}_beta{args.kl_weight}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=vars(args),
            model=model,
            device=args.device,
            tags=['logic_vae', 'mnist', f'n{args.n_inputs}',
                  f'L{args.latent_bits}']
            + normalize_tags(model='logic_vae', dataset='mnist', seed=args.seed),
            group='logic_vae',
            notes=args.notes,
            hypothesis=args.hypothesis,
        )

    print(f'\n訓練開始: {total_steps:,}ステップ')
    print('=' * 60)
    train_start = time.time()
    best_recon = float('inf')

    model.train()
    step = 0
    done = False
    while not done:
        for x, _ in train_loader:
            x = x.to(torch.float32).to(args.device)
            out = model(x)
            losses = model.elbo_loss(x, out)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
            step += 1

            if step % args.eval_freq == 0 or step == total_steps:
                recon_v = losses['recon'].item()
                kl_v = losses['kl'].item()
                loss_v = losses['loss'].item()
                # アクティブな潜在ビット数(posterior collapse監視)
                with torch.no_grad():
                    q = torch.sigmoid(out['latent_logits'])
                    active = float(((q > 0.1) & (q < 0.9)).float().mean())

                is_best = recon_v < best_recon
                if is_best:
                    best_recon = recon_v
                save_checkpoint(latest_path, model, optimizer, None,
                                step, best_recon, ckpt_metadata)
                if is_best:
                    save_checkpoint(best_path, model, optimizer, None,
                                    step, best_recon, ckpt_metadata)

                recon_grid, gen_grid = make_sample_grids(
                    model, x_ref, args.device,
                )
                save_grids(recon_grid, gen_grid, step, args.output_dir, use_wandb)

                elapsed = time.time() - train_start
                progress = step / total_steps
                remaining = elapsed * (1 - progress) / progress if progress > 0 else 0
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remain_str = str(timedelta(seconds=int(remaining)))
                eta_str = eta.strftime('%H:%M:%S')

                print(
                    f'[step {step:>6d}/{total_steps}] '
                    f'loss={loss_v:.3f}  recon={recon_v:.3f}  kl={kl_v:.3f}  '
                    f'active_bits={active:.2f}'
                    f'{"  ★ BEST" if is_best else ""}'
                    f'  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
                )
                if use_wandb:
                    log_metrics({
                        'step': step,
                        'loss': loss_v,
                        'recon': recon_v,
                        'kl': kl_v,
                        'active_bits': active,
                        'best_recon': best_recon,
                    })

            if step >= total_steps:
                done = True
                break

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最良再構成: {best_recon:.4f}  (総訓練時間: {total_time})')

    if use_wandb:
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_recon': best_recon},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
