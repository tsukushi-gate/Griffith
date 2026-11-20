"""
DLGN Boolean Attention 訓練スクリプト（連想記憶タスク）

Gate-Level Boolean Evolutionary Geometric Attention (arXiv:2511.17550) に基づく
最小構成のAttention PoC。

使用例:
    # デフォルト（4ペア、4bitキー、4bitバリュー）
    uv run python experiments/train_boolean_attention.py -ni 10000 -ef 500

    # スモークテスト
    uv run python experiments/train_boolean_attention.py -ni 500 -ef 100 --num-samples 2000

    # スケールテスト
    uv run python experiments/train_boolean_attention.py --num-pairs 8 --key-bits 6 -ni 20000
"""

import argparse
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import load_checkpoint, save_checkpoint, setup_checkpoint_dir
from data import AssociativeRecallDataset
from env_utils import load_dotenv
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import BooleanAttentionConfig, DLGNBooleanAttention

load_dotenv()


# ============================================================
# 訓練・評価
# ============================================================

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',')]


def train_step(model, x, y, optimizer, label_smoothing=0.0):
    """1ステップの訓練。標準分類（cross entropy）。"""
    model.train()
    logits = model(x, hard_attention=False)
    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, loader, device, hard_attention: bool):
    """
    精度を計算。
      hard_attention=False → 訓練時と同じ softmax 加重
      hard_attention=True  → 推論時の argmax Hard Attention

    Returns:
        value_acc: バリュー精度（正解クラス）
        attn_acc: attention 精度（argmax(attn) == 正解ペア index）
    """
    was_training = model.training
    # hard_attention を明示指定して常に同じ計算を使う
    # model.train(False) だと hard_attention=None -> not training -> True になる
    # hard_attention 引数で明示制御するので train/eval モードは単にLutLayerの振る舞い制御用
    model.train(mode=not hard_attention)

    value_correct = 0
    attn_correct = 0
    total = 0

    for x, y, q_idx in loader:
        x = x.to(device)
        y = y.to(device)
        q_idx = q_idx.to(device)

        logits = model(x, hard_attention=hard_attention)
        preds = logits.argmax(dim=-1)
        value_correct += (preds == y).sum().item()

        # attention 診断
        attn = model.last_attention  # (batch, num_pairs)
        attn_argmax = attn.argmax(dim=-1)
        attn_correct += (attn_argmax == q_idx).sum().item()

        total += y.shape[0]

    model.train(mode=was_training)
    return value_correct / total, attn_correct / total


# ============================================================
# ログ
# ============================================================

def setup_logger(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'boolean_attention_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


def load_n(loader, n):
    """DataLoader から n ステップ分のバッチを返す。"""
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i == n:
                return


# ============================================================
# メイン
# ============================================================

def build_config(args) -> BooleanAttentionConfig:
    return BooleanAttentionConfig(
        num_pairs=args.num_pairs,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        encoder_sizes=parse_int_list(args.encoder_sizes),
        value_encoder_sizes=parse_int_list(args.value_encoder_sizes),
        decoder_sizes=parse_int_list(args.decoder_sizes),
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        tau_attn=args.tau_attn,
        share_key_query=not args.no_share_key_query,
        tau_out=args.tau_out,
        tau_warp=args.tau_warp,
        gumbel=not args.no_gumbel,
    )


def main():
    parser = argparse.ArgumentParser(description='DLGN Boolean Attention 訓練')

    # タスク設定
    parser.add_argument('--num-pairs', type=int, default=4)
    parser.add_argument('--key-bits', type=int, default=4)
    parser.add_argument('--value-bits', type=int, default=4)
    parser.add_argument('--num-samples', type=int, default=50000)
    parser.add_argument('--num-test-samples', type=int, default=5000)

    # モデル構成
    parser.add_argument('--encoder-sizes', type=str, default='32,16',
                        help='Key/Query encoder LutStack サイズ (default: 32,16)')
    parser.add_argument('--value-encoder-sizes', type=str, default='32,16')
    parser.add_argument('--decoder-sizes', type=str, default='64,32')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'])
    parser.add_argument('--tau-attn', type=float, default=1.0,
                        help='Attention softmax 温度 (default: 1.0)')
    parser.add_argument('--tau-out', type=float, default=1.0,
                        help='GroupSum 温度 (default: 1.0)')
    parser.add_argument('--no-share-key-query', action='store_true',
                        help='Key/Query encoderを別々のパラメータに')
    parser.add_argument('--tau-warp', type=float, default=1.0)
    parser.add_argument('--no-gumbel', action='store_true')

    # 訓練設定
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=10000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=500)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--lr-factor', type=float, default=0.8)

    # チェックポイント
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    # 環境
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--no-wandb', action='store_true')

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

    # シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 構成
    config = build_config(args)

    print('\n=== DLGN Boolean Attention 訓練 ===')
    print('タスク: 連想記憶 (Associative Recall)')
    print(f'num_pairs={config.num_pairs} key_bits={config.key_bits} '
          f'value_bits={config.value_bits}')
    print(f'入力次元: {config.input_dim} ビット')
    print(f'出力クラス数: {config.num_classes}')
    print(f'encoder_sizes: {config.encoder_sizes}')
    print(f'value_encoder_sizes: {config.value_encoder_sizes}')
    print(f'decoder_sizes: {config.decoder_sizes}')
    print(f'tau_attn={config.tau_attn} tau_out={config.tau_out} '
          f'share_key_query={config.share_key_query}')

    # データセット
    train_set = AssociativeRecallDataset(
        num_pairs=config.num_pairs,
        key_bits=config.key_bits,
        value_bits=config.value_bits,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    test_set = AssociativeRecallDataset(
        num_pairs=config.num_pairs,
        key_bits=config.key_bits,
        value_bits=config.value_bits,
        num_samples=args.num_test_samples,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )
    print(f'訓練サンプル: {len(train_set):,}')
    print(f'テストサンプル: {len(test_set):,}')

    # モデル構築
    model = DLGNBooleanAttention(config, device=args.device).to(args.device)
    counts = model.count_parameters()
    total_gates = sum(
        m.num_neurons for m in model.modules()
        if hasattr(m, 'num_neurons')
    )
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    print(f'総ロジックゲート数: {total_gates:,} (LUT{config.n_inputs})')

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.patience,
    )

    # ログ
    log, log_file = setup_logger(args.log_dir)
    log(f'START num_pairs={config.num_pairs} key_bits={config.key_bits} '
        f'value_bits={config.value_bits} lr={args.lr} bs={args.batch_size} '
        f'ni={args.num_iterations} device={args.device} params={counts["total"]:,}')

    # チェックポイント
    ckpt_prefix = f'boolattn_np{config.num_pairs}_kb{config.key_bits}_vb{config.value_bits}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args), 'config': asdict(config)}

    start_step = 0
    best_test_acc = 0.0
    if args.resume:
        print(f'チェックポイントを読み込み: {args.resume}')
        ckpt_info = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=args.device,
        )
        start_step = ckpt_info['step']
        best_test_acc = ckpt_info['best_metric']
        ckpt_metadata.update(ckpt_info.get('metadata', {}))
        print(f'  ステップ {start_step} から再開 (best_acc={best_test_acc:.4f})')

    print(f'チェックポイント保存先: {run_dir}/')

    # wandb
    run_name = (f'boolattn_np{config.num_pairs}_kb{config.key_bits}_vb{config.value_bits}'
                f'_ni{args.num_iterations}')
    wandb_config = {**vars(args), **asdict(config)}
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=['boolean_attention', 'associative_recall',
                  f'np{config.num_pairs}', f'kb{config.key_bits}'],
            group='boolean_attention',
        )

    remaining_iters = args.num_iterations - start_step
    print(f'\n訓練開始: {args.num_iterations:,} イテレーション（残り {remaining_iters:,}）')
    print('=' * 60)

    train_start = time.time()

    pbar = tqdm(
        enumerate(load_n(train_loader, remaining_iters), start=start_step),
        desc='train',
        total=args.num_iterations,
        initial=start_step,
    )

    for step, (x, y, q_idx) in pbar:
        x = x.to(args.device)
        y = y.to(args.device)

        loss = train_step(
            model, x, y, optimizer,
            label_smoothing=args.label_smoothing,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1 - start_step + 1e-9)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')

        pbar.set_postfix(
            loss=f'{loss:.4f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.5f}',
            eta=eta_str,
        )

        # 評価
        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            val_acc_soft, attn_acc_soft = evaluate(
                model, test_loader, args.device, hard_attention=False,
            )
            val_acc_hard, attn_acc_hard = evaluate(
                model, test_loader, args.device, hard_attention=True,
            )

            scheduler.step(1.0 - val_acc_soft)

            is_best = val_acc_soft > best_test_acc
            if is_best:
                best_test_acc = val_acc_soft

            save_checkpoint(latest_path, model, optimizer, scheduler,
                            step + 1, best_test_acc, ckpt_metadata)
            if is_best:
                save_checkpoint(best_path, model, optimizer, scheduler,
                                step + 1, best_test_acc, ckpt_metadata)

            elapsed = time.time() - train_start
            sec_per_iter = elapsed / (step + 1 - start_step + 1e-9)
            remaining = sec_per_iter * (args.num_iterations - (step + 1))
            eta = datetime.now(JST) + timedelta(seconds=remaining)
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remain_str = str(timedelta(seconds=int(remaining)))
            eta_str = eta.strftime('%H:%M:%S')
            iter_per_sec = (step + 1 - start_step) / elapsed

            current_lr = optimizer.param_groups[0]['lr']

            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={loss:.4f}  '
                f'val(soft)={val_acc_soft:.4f} val(hard)={val_acc_hard:.4f}  '
                f'attn(soft)={attn_acc_soft:.4f} attn(hard)={attn_acc_hard:.4f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  lr={current_lr:.5f}'
                f'  [{elapsed_str}<{remain_str}, {iter_per_sec:.1f}it/s, 終了予定 {eta_str}]'
            )

            log(f'ITER {step+1}/{args.num_iterations} '
                f'loss={loss:.4f} val_soft={val_acc_soft:.4f} '
                f'val_hard={val_acc_hard:.4f} attn_soft={attn_acc_soft:.4f} '
                f'attn_hard={attn_acc_hard:.4f} lr={current_lr:.5f} '
                f'best={str(is_best).lower()} elapsed={elapsed_str}')

            if use_wandb:
                log_metrics({
                    'iteration': step + 1,
                    'loss': loss,
                    'val/acc_soft': val_acc_soft,
                    'val/acc_hard': val_acc_hard,
                    'attn/acc_soft': attn_acc_soft,
                    'attn/acc_hard': attn_acc_hard,
                    'best_val_acc': best_test_acc,
                    'lr': current_lr,
                }, step=step + 1)

    # 最終結果
    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終ベスト精度(soft): {best_test_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_acc={best_test_acc:.4f} total_time={total_time}')
    log_file.close()
    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_test_acc': best_test_acc},
            model=model,
        )
        wandb_finish()

    # 予測例
    print('\n=== 予測例 ===')
    model.eval()
    x_sample, y_sample, q_idx_sample = next(iter(test_loader))
    x_sample = x_sample[:4].to(args.device)
    y_sample = y_sample[:4].to(args.device)
    q_idx_sample = q_idx_sample[:4].to(args.device)

    with torch.no_grad():
        logits_sample = model(x_sample, hard_attention=True)
        preds = logits_sample.argmax(dim=-1)
        attn_sample = model.last_attention

    for i in range(min(4, x_sample.shape[0])):
        pred = preds[i].item()
        tgt = y_sample[i].item()
        qi = q_idx_sample[i].item()
        attn_i = attn_sample[i].tolist()
        match_val = '✓' if pred == tgt else '✗'
        match_attn = '✓' if attn_i.index(max(attn_i)) == qi else '✗'
        print(f'  sample {i}: target={tgt:3d} pred={pred:3d} {match_val}  '
              f'correct_key_idx={qi} attn={[f"{a:.2f}" for a in attn_i]} {match_attn}')


if __name__ == '__main__':
    main()
