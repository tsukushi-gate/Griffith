"""
AP法によるRDDLGN訓練スクリプト（seq2seq）

使用例:
    # copyタスク（tiny構成）
    uv run python experiments/train_ap_rddlgn.py --task copy --config tiny --lr 0.1 -ni 5000 -ef 500

    # reverseタスク
    uv run python experiments/train_ap_rddlgn.py --task reverse --config tiny --lr 0.1 -ni 10000 -ef 1000

    # AP法×KD: 先にBP（またはAP）でteacherを学習してckptを作る
    uv run python experiments/train_ap_rddlgn.py --task copy --config tiny \\
        -ni 5000 -ef 500
    # ↑ このtrain_ap_rddlgn.py自身の実行でも checkpoints/ap_rddlgn_copy_*/best.pt が
    #   保存される（save_ap_checkpointが常時ckptを保存するため）。
    #   train_rddlgn.py（BP）で学習したbest.ptも同様に使える:
    #   uv run python experiments/train_rddlgn.py --task copy --config tiny -ni 5000 -ef 500
    #
    # 得られたbest.ptをteacherとして渡し、AP法をKDで学習する:
    uv run python experiments/train_ap_rddlgn.py --task copy --config tiny --lr 0.1 \\
        -ni 5000 -ef 500 \\
        --teacher-ckpt checkpoints/rddlgn_copy_tiny_XXXXXXXX_XXXXXX/best.pt \\
        --kd-alpha 0.5 --kd-temperature 1.0
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ap_trainer_rddlgn import RDDLGNAPTrainer
from checkpoint_utils import setup_checkpoint_dir
from data import SPECIAL_TOKENS, TASK_DATASETS, TOKEN_BOS, TOKEN_EOS, TOKEN_PAD
from env_utils import load_dotenv
from train_rddlgn import (
    build_config,
    load_n,
    parse_int_list,
    prepare_batch,
    setup_logger,
)
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import RDDLGN, RDDLGNConfig

load_dotenv()


def load_teacher_model(ckpt_path: str, fallback_args, device: str) -> RDDLGN:
    """teacher-ckptからRDDLGNモデルを復元する（no_grad推論専用、eval固定）。

    metadata['config']（train_rddlgn.pyのBPチェックポイント）があればそこから直接
    RDDLGNConfigを再構築する。無ければmetadata['args']から現行のbuild_configと
    同じロジックでconfigを再構築する（train_ap_rddlgn.py自身が保存したckpt向け）。
    どちらも無い場合は現行実行時のargsから構築したconfigと同構成と仮定する。

    モデル構築時のランダム初期化（load_state_dictで直後に上書きされるが、その前に
    グローバルRNGを消費する）が呼び出し側のシード状態に影響しないよう、
    torchのRNG状態を退避・復元する（kd_alpha=1.0時の非KD実行との数値一致を保証するため）。
    """
    rng_state = torch.get_rng_state()
    try:
        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        metadata = data.get('metadata', {}) or {}

        if 'config' in metadata:
            teacher_config = RDDLGNConfig(**metadata['config'])
        elif 'args' in metadata:
            teacher_args = argparse.Namespace(**metadata['args'])
            # build_configは`parameterization`属性を要求する（train_ap_rddlgn.pyの
            # argsには元々存在しない場合があるため補完）
            if not hasattr(teacher_args, 'parameterization'):
                teacher_args.parameterization = 'independent'
            if not hasattr(teacher_args, 'tau_warp'):
                teacher_args.tau_warp = 1.0
            if not hasattr(teacher_args, 'no_gumbel'):
                teacher_args.no_gumbel = False
            teacher_config = build_config(teacher_args)
            if teacher_args.vocab_size is not None and teacher_args.config != 'custom':
                teacher_config.vocab_size = teacher_args.vocab_size
            if getattr(teacher_args, 'seq_len', None) is not None and teacher_args.config != 'custom':
                teacher_config.seq_len = teacher_args.seq_len
        else:
            print('  警告: teacher-ckptにconfig/args情報が無いため、現行実行のconfigと同構成を仮定します')
            teacher_config = build_config(fallback_args)

        teacher_model = RDDLGN(teacher_config, device=device).to(device)
        teacher_model.load_state_dict(data['model_state_dict'])
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        return teacher_model
    finally:
        torch.set_rng_state(rng_state)


def save_ap_checkpoint(path, model, step, best_metric, metadata):
    """AP法用チェックポイント保存（optimizerなし）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'step': step,
        'best_metric': best_metric,
        'metadata': metadata,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='AP法 RDDLGN 訓練')

    parser.add_argument('--task', type=str, default='copy',
                        choices=list(TASK_DATASETS.keys()),
                        help='タスク (default: copy)')
    parser.add_argument('--num-samples', type=int, default=50000)
    parser.add_argument('--num-test-samples', type=int, default=5000)

    # モデル構成
    parser.add_argument('--config', type=str, default='tiny',
                        choices=['tiny', 'paper', 'custom'])
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--vocab-size', type=int, default=None)
    parser.add_argument('--d-emb', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--n-sizes', type=str, default='512,512')
    parser.add_argument('--k-sizes', type=str, default='1024,1024')
    parser.add_argument('--l-sizes', type=str, default='512,512')
    parser.add_argument('--p-sizes', type=str, default='1024,768')
    parser.add_argument('--m-sizes', type=str, default='2000,2000,1000')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--tau-warp', type=float, default=1.0,
                        help='WARP温度パラメータ（parameterization=warp時のみ、'
                             'build_config互換のため定義。AP法はindependent固定）')
    parser.add_argument('--no-gumbel', action='store_true',
                        help='WARPのGumbel smoothingを無効化（build_config互換のため定義）')

    # AP法設定
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--update-mode', type=int, default=1, choices=[1, 2])

    # KD設定（AP法×KD）
    parser.add_argument('--teacher-ckpt', type=str, default=None,
                        help='teacherチェックポイントパス（BPまたはAPで学習した'
                             'RDDLGNのbest.pt）。指定時のみKDを有効化 (default: None)')
    parser.add_argument('--kd-alpha', type=float, default=1.0,
                        help='one-hotターゲットとteacher分布の混合率。'
                             'target_dist = kd_alpha*one_hot + (1-kd_alpha)*teacher_probs。'
                             '1.0（既定）はteacher無しと完全一致 (default: 1.0)')
    parser.add_argument('--kd-temperature', type=float, default=1.0,
                        help='teacher softmaxの温度T。teacher_probs = '
                             'softmax(teacher_logits / T) (default: 1.0)')

    # 訓練設定
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=5000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=500)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()

    # AP法はindependentパラメータ化を強制
    args.parameterization = 'independent'

    # デバイス
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'デバイス: {args.device}')

    # シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # モデル構成
    config = build_config(args)
    config.parameterization = 'independent'
    print(f'タスク: {args.task}')
    print(f'構成: {args.config}')

    # データ
    dataset_cls = TASK_DATASETS[args.task]
    train_set = dataset_cls(
        num_samples=args.num_samples,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
    )
    test_set = dataset_cls(
        num_samples=args.num_test_samples,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )
    print(f'訓練: {len(train_set):,}サンプル, テスト: {len(test_set):,}サンプル')

    # モデル
    model = RDDLGN(config, device=args.device).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'パラメータ数: {total_params:,}')
    print(model)

    # APTrainer
    trainer = RDDLGNAPTrainer(
        model, lr=args.lr, update_mode=args.update_mode, pad_token=TOKEN_PAD,
    )
    print(f'AP法: update_mode={args.update_mode}, lr={args.lr}')

    # Teacherモデル（KD）
    teacher_model = None
    if args.teacher_ckpt is not None:
        print(f'Teacherチェックポイントを読み込み: {args.teacher_ckpt}')
        teacher_model = load_teacher_model(
            args.teacher_ckpt, fallback_args=args, device=args.device,
        )
        print(f'KD有効: kd_alpha={args.kd_alpha}, kd_temperature={args.kd_temperature}')

    # ログ
    log, log_file = setup_logger(f'ap_{args.task}', args.log_dir)
    log(f'START task={args.task} config={args.config} lr={args.lr} '
        f'update_mode={args.update_mode} bs={args.batch_size} '
        f'ni={args.num_iterations} seed={args.seed} device={args.device} '
        f'teacher_ckpt={args.teacher_ckpt} kd_alpha={args.kd_alpha} '
        f'kd_temperature={args.kd_temperature}')

    # チェックポイント
    ckpt_prefix = f'ap_rddlgn_{args.task}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args), 'trainer': 'ap_rddlgn'}
    print(f'チェックポイント: {run_dir}/')

    # wandb
    run_name = (f'ap_rddlgn_{args.task}_{args.config}_'
                f'lr{args.lr}_m{args.update_mode}_ni{args.num_iterations}')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={**vars(args), 'trainer': 'ap_rddlgn'},
        model=model,
        device=args.device,
        tags=['ap', 'rddlgn', args.task],
        group='ap_rddlgn',
    )

    # 訓練
    best_acc = 0.0
    train_start = time.time()

    print(f'\n訓練開始: {args.num_iterations:,}イテレーション')
    print('=' * 60)

    pbar = tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        total=args.num_iterations, desc='train',
    )
    for i, (src, tgt) in pbar:
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        src_batch, tgt_full = prepare_batch(src, tgt)

        teacher_probs = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits, _ = teacher_model(src_batch, tgt_full[:, :-1])
                teacher_probs = F.softmax(
                    teacher_logits / args.kd_temperature, dim=-1
                )

        metrics = trainer.train_step(
            src_batch, tgt_full,
            teacher_probs=teacher_probs, kd_alpha=args.kd_alpha,
        )

        elapsed = time.time() - train_start
        remaining = elapsed / (i + 1) * (args.num_iterations - i - 1)
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining)).strftime('%H:%M:%S')
        pbar.set_postfix(
            loss=f'{metrics["loss"]:.4f}',
            acc=f'{metrics["accuracy"]:.4f}',
            eta=eta_str,
        )

        if (i + 1) % args.eval_freq == 0:
            test_acc = trainer.evaluate(
                test_loader, args.device,
                pad_token=TOKEN_PAD, bos_token=TOKEN_BOS, eos_token=TOKEN_EOS,
            )

            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc

            save_ap_checkpoint(
                latest_path, model, i + 1, best_acc, ckpt_metadata,
            )
            if is_best:
                save_ap_checkpoint(
                    best_path, model, i + 1, best_acc, ckpt_metadata,
                )

            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))

            print(
                f'\n[iter {i+1:,}] '
                f'loss={metrics["loss"]:.4f}  '
                f'train_acc={metrics["accuracy"]:.4f}  '
                f'test_acc={test_acc:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  [{elapsed_str}<{remain_str}, ETA {eta_str}]'
            )

            log(f'ITER {i+1}/{args.num_iterations} '
                f'loss={metrics["loss"]:.4f} '
                f'train_acc={metrics["accuracy"]:.4f} '
                f'test_acc={test_acc:.4f} '
                f'best={str(is_best).lower()} '
                f'elapsed={elapsed_str} eta={remain_str}')
            if use_wandb:
                log_metrics({
                    'iteration': i + 1,
                    'loss': metrics['loss'],
                    'train_acc': metrics['accuracy'],
                    'test_acc': test_acc,
                    'best_acc': best_acc,
                })

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度: {best_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_acc={best_acc:.4f} total_time={total_time}')
    log_file.close()
    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_acc': best_acc},
            model=model,
        )
        wandb_finish()


if __name__ == '__main__':
    main()
