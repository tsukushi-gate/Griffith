"""
DLGN Pointer Attention 訓練スクリプト（Task 5: Pointer Network 版 seq2seq）

内容ベースのキー、自己回帰デコード、マスキング付き Attention で sort を解く。
reverse / copy も扱える（サニティチェック用）。

使用例:
    # sort 本番
    uv run python experiments/train_pointer_attention.py --task sort -ni 30000 -ef 500

    # copy（サニティ）
    uv run python experiments/train_pointer_attention.py --task copy -ni 5000 -ef 500

    # reverse（サニティ）
    uv run python experiments/train_pointer_attention.py --task reverse -ni 10000 -ef 500

    # スモークテスト
    uv run python experiments/train_pointer_attention.py --task sort -ni 500 -ef 100 \\
        --num-samples 3000 --no-wandb --device cpu

    # 小さい value_bits で sort
    uv run python experiments/train_pointer_attention.py --task sort \\
        --value-bits 3 -ni 20000 -ef 500
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
from data import Seq2SeqWithPointersDataset
from env_utils import load_dotenv
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import DLGNPointerAttention, PointerAttentionConfig

load_dotenv()


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',')]


def train_step(model, x, y, ptr, optimizer, lambda_ptr, label_smoothing=0.0):
    """
    1ステップの訓練。Teacher Forcing + (value + pointer) 損失。

    Args:
        y: (B, L) 正解値（クラスID）
        ptr: (B, L) 正解ポインタ（入力位置）
    """
    model.train()
    logits, attn = model(x, target_pointers=ptr, hard_attention=False)
    B, L, C = logits.shape
    N = attn.shape[-1]

    # Value 損失
    value_loss = F.cross_entropy(
        logits.reshape(B * L, C), y.reshape(B * L),
        label_smoothing=label_smoothing,
    )
    # Pointer 損失（attention 分布が正解ポインタと一致するよう誘導）
    # attn は softmax 後なので小さな値を log するのを避けるため log_softmax 経由でロスを組む
    # attn = softmax(scores) → NLL = -log(attn[target])
    log_attn = torch.log(attn.clamp_min(1e-10))  # (B, L, N)
    ptr_loss = F.nll_loss(
        log_attn.reshape(B * L, N), ptr.reshape(B * L),
    )

    total_loss = value_loss + lambda_ptr * ptr_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), value_loss.item(), ptr_loss.item()


@torch.no_grad()
def evaluate(model, loader, device, hard_attention: bool, use_teacher_forcing: bool):
    """
    評価メトリクス。

    Args:
        hard_attention: True=推論時argmax、False=softmax加重
        use_teacher_forcing: True=正解ポインタで prev_val を更新、False=モデル自身の出力

    Returns:
        elem_acc, seq_acc, ptr_acc, mask_violation_rate
    """
    was_training = model.training
    model.train(mode=not hard_attention)

    elem_correct = 0
    seq_correct = 0
    ptr_correct = 0
    mask_violations = 0
    total_elems = 0
    total_seqs = 0
    total_steps = 0

    for x, y, ptr in loader:
        x = x.to(device)
        y = y.to(device)
        ptr = ptr.to(device)
        target_ptr = ptr if use_teacher_forcing else None

        logits, attn = model(x, target_pointers=target_ptr, hard_attention=hard_attention)
        preds = logits.argmax(dim=-1)  # (B, L)

        elem_correct += (preds == y).sum().item()
        seq_correct += ((preds == y).all(dim=-1)).sum().item()
        total_elems += y.numel()
        total_seqs += y.shape[0]

        # Pointer 精度: argmax(attn) が target_pointer と一致する率
        attn_argmax = attn.argmax(dim=-1)  # (B, L)
        ptr_correct += (attn_argmax == ptr).sum().item()
        total_steps += ptr.numel()

        # マスク違反: 同じ位置を2回以上 argmax した率
        # ptr を使わずに attn argmax から評価
        B, L = attn_argmax.shape
        for b in range(B):
            picked = attn_argmax[b].tolist()
            if len(set(picked)) < L:
                mask_violations += 1

    model.train(mode=was_training)
    return (
        elem_correct / total_elems,
        seq_correct / total_seqs,
        ptr_correct / total_steps,
        mask_violations / total_seqs,
    )


def setup_logger(task: str, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'pointer_attention_{task}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


def load_n(loader, n):
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i == n:
                return


def build_config(args) -> PointerAttentionConfig:
    return PointerAttentionConfig(
        seq_len=args.seq_len,
        value_bits=args.value_bits,
        value_encoder_sizes=parse_int_list(args.value_encoder_sizes),
        key_encoder_sizes=parse_int_list(args.key_encoder_sizes),
        decoder_cell_sizes=parse_int_list(args.decoder_cell_sizes),
        query_producer_sizes=parse_int_list(args.query_producer_sizes),
        output_decoder_sizes=parse_int_list(args.output_decoder_sizes),
        n_inputs=args.n_inputs,
        parameterization=args.parameterization,
        tau_attn=args.tau_attn,
        mask_scale=args.mask_scale,
        tau_out=args.tau_out,
        tau_warp=args.tau_warp,
        gumbel=not args.no_gumbel,
    )


def main():
    parser = argparse.ArgumentParser(description='DLGN Pointer Attention 訓練')

    # タスク設定
    parser.add_argument('--task', type=str, default='sort',
                        choices=['copy', 'reverse', 'sort'])
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--value-bits', type=int, default=4)
    parser.add_argument('--num-samples', type=int, default=50000)
    parser.add_argument('--num-test-samples', type=int, default=5000)

    # モデル構成
    parser.add_argument('--value-encoder-sizes', type=str, default='32,16')
    parser.add_argument('--key-encoder-sizes', type=str, default='32,16')
    parser.add_argument('--decoder-cell-sizes', type=str, default='64,16')
    parser.add_argument('--query-producer-sizes', type=str, default='32,16')
    parser.add_argument('--output-decoder-sizes', type=str, default='64,32')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'])
    parser.add_argument('--tau-attn', type=float, default=1.0)
    parser.add_argument('--mask-scale', type=float, default=10.0)
    parser.add_argument('--tau-out', type=float, default=1.0)
    parser.add_argument('--tau-warp', type=float, default=1.0)
    parser.add_argument('--no-gumbel', action='store_true')

    # 損失・訓練
    parser.add_argument('--lambda-ptr', type=float, default=0.5,
                        help='Pointer 損失の重み（0で無効）')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=30000)
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

    config = build_config(args)

    print('\n=== DLGN Pointer Attention 訓練 ===')
    print(f'タスク: {args.task}')
    print(f'seq_len={config.seq_len} value_bits={config.value_bits} '
          f'num_classes={config.num_classes}')
    print(f'入力次元: {config.input_dim} ビット')
    print(f'lambda_ptr={args.lambda_ptr} mask_scale={config.mask_scale} '
          f'tau_attn={config.tau_attn}')

    train_set = Seq2SeqWithPointersDataset(
        task=args.task, seq_len=config.seq_len, value_bits=config.value_bits,
        num_samples=args.num_samples, seed=args.seed,
    )
    test_set = Seq2SeqWithPointersDataset(
        task=args.task, seq_len=config.seq_len, value_bits=config.value_bits,
        num_samples=args.num_test_samples, seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )
    print(f'訓練サンプル: {len(train_set):,}')
    print(f'テストサンプル: {len(test_set):,}')

    model = DLGNPointerAttention(config, device=args.device).to(args.device)
    counts = model.count_parameters()
    total_gates = sum(
        m.num_neurons for m in model.modules() if hasattr(m, 'num_neurons')
    )
    print('\nパラメータ数:')
    for name, count in counts.items():
        print(f'  {name}: {count:,}')
    print(f'総ロジックゲート数: {total_gates:,} (LUT{config.n_inputs})')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.patience,
    )

    log, log_file = setup_logger(args.task, args.log_dir)
    log(f'START task={args.task} seq_len={config.seq_len} value_bits={config.value_bits} '
        f'lambda_ptr={args.lambda_ptr} lr={args.lr} bs={args.batch_size} '
        f'ni={args.num_iterations} device={args.device} params={counts["total"]:,}')

    ckpt_prefix = f'ptrattn_{args.task}_sl{config.seq_len}_vb{config.value_bits}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    ckpt_metadata = {'args': vars(args), 'config': asdict(config)}

    start_step = 0
    best_seq_acc = 0.0
    if args.resume:
        ckpt_info = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=args.device,
        )
        start_step = ckpt_info['step']
        best_seq_acc = ckpt_info['best_metric']
        ckpt_metadata.update(ckpt_info.get('metadata', {}))

    print(f'チェックポイント保存先: {run_dir}/')

    run_name = (f'ptrattn_{args.task}_sl{config.seq_len}_vb{config.value_bits}'
                f'_ni{args.num_iterations}')
    wandb_config = {**vars(args), **asdict(config)}
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=['pointer_attention', args.task,
                  f'sl{config.seq_len}', f'vb{config.value_bits}'],
            group='pointer_attention',
        )

    remaining_iters = args.num_iterations - start_step
    print(f'\n訓練開始: {args.num_iterations:,} イテレーション（残り {remaining_iters:,}）')
    print('=' * 60)

    train_start = time.time()

    pbar = tqdm(
        enumerate(load_n(train_loader, remaining_iters), start=start_step),
        desc='train', total=args.num_iterations, initial=start_step,
    )

    for step, (x, y, ptr) in pbar:
        x = x.to(args.device)
        y = y.to(args.device)
        ptr = ptr.to(args.device)

        total_loss, value_loss, ptr_loss = train_step(
            model, x, y, ptr, optimizer,
            lambda_ptr=args.lambda_ptr, label_smoothing=args.label_smoothing,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1 - start_step + 1e-9)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')

        pbar.set_postfix(
            loss=f'{total_loss:.4f}',
            vl=f'{value_loss:.4f}',
            pl=f'{ptr_loss:.4f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.5f}',
            eta=eta_str,
        )

        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            # Teacher-forcing 評価（デコーダーが正解ポインタで状態更新）
            elem_tf_soft, seq_tf_soft, ptr_tf_soft, mv_tf_soft = evaluate(
                model, test_loader, args.device,
                hard_attention=False, use_teacher_forcing=True,
            )
            # Free-running 評価（モデル自身の出力で連鎖、これが本来の推論）
            elem_free_soft, seq_free_soft, ptr_free_soft, mv_free_soft = evaluate(
                model, test_loader, args.device,
                hard_attention=False, use_teacher_forcing=False,
            )
            elem_free_hard, seq_free_hard, ptr_free_hard, mv_free_hard = evaluate(
                model, test_loader, args.device,
                hard_attention=True, use_teacher_forcing=False,
            )

            scheduler.step(1.0 - seq_free_soft)

            is_best = seq_free_soft > best_seq_acc
            if is_best:
                best_seq_acc = seq_free_soft

            save_checkpoint(latest_path, model, optimizer, scheduler,
                            step + 1, best_seq_acc, ckpt_metadata)
            if is_best:
                save_checkpoint(best_path, model, optimizer, scheduler,
                                step + 1, best_seq_acc, ckpt_metadata)

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
                f'loss={total_loss:.3f} vl={value_loss:.3f} pl={ptr_loss:.3f}\n'
                f'  TF/soft:   elem={elem_tf_soft:.3f} seq={seq_tf_soft:.3f} ptr={ptr_tf_soft:.3f}\n'
                f'  Free/soft: elem={elem_free_soft:.3f} seq={seq_free_soft:.3f} '
                f'ptr={ptr_free_soft:.3f} mask_viol={mv_free_soft:.3f}\n'
                f'  Free/hard: elem={elem_free_hard:.3f} seq={seq_free_hard:.3f} '
                f'ptr={ptr_free_hard:.3f} mask_viol={mv_free_hard:.3f}'
                f'{"  ★ BEST" if is_best else ""}'
                f'  lr={current_lr:.5f}  [{elapsed_str}<{remain_str}, 終了予定 {eta_str}]'
            )

            log(f'ITER {step+1}/{args.num_iterations} loss={total_loss:.4f} '
                f'vl={value_loss:.4f} pl={ptr_loss:.4f} '
                f'tf_seq={seq_tf_soft:.4f} free_seq_soft={seq_free_soft:.4f} '
                f'free_seq_hard={seq_free_hard:.4f} '
                f'ptr_free_soft={ptr_free_soft:.4f} ptr_free_hard={ptr_free_hard:.4f} '
                f'best={str(is_best).lower()}')

            if use_wandb:
                log_metrics({
                    'iteration': step + 1,
                    'loss/total': total_loss,
                    'loss/value': value_loss,
                    'loss/pointer': ptr_loss,
                    'tf/elem': elem_tf_soft,
                    'tf/seq': seq_tf_soft,
                    'tf/ptr': ptr_tf_soft,
                    'free_soft/elem': elem_free_soft,
                    'free_soft/seq': seq_free_soft,
                    'free_soft/ptr': ptr_free_soft,
                    'free_soft/mask_viol': mv_free_soft,
                    'free_hard/elem': elem_free_hard,
                    'free_hard/seq': seq_free_hard,
                    'free_hard/ptr': ptr_free_hard,
                    'free_hard/mask_viol': mv_free_hard,
                    'best_seq_acc': best_seq_acc,
                    'lr': current_lr,
                }, step=step + 1)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終ベスト系列精度 (free soft): {best_seq_acc:.4f}  (総訓練時間: {total_time})')
    log(f'DONE best_seq_acc={best_seq_acc:.4f} total_time={total_time}')
    log_file.close()
    if use_wandb:
        # 重みをwandb Artifactsに保存(WANDB_ARTIFACTS=off/best/allで制御)
        log_checkpoint_artifact(
            best_path, run_name, stage='final',
            metadata={'args': vars(args), 'best_seq_acc': best_seq_acc},
            model=model,
        )
        wandb_finish()

    # 予測例 + attention 可視化（free running, hard）
    print('\n=== 予測例（free-running / hard attention） ===')
    model.eval()
    x_sample, y_sample, ptr_sample = next(iter(test_loader))
    x_sample = x_sample[:3].to(args.device)
    y_sample = y_sample[:3].to(args.device)
    ptr_sample = ptr_sample[:3].to(args.device)

    with torch.no_grad():
        logits_sample, attn_sample = model(
            x_sample, target_pointers=None, hard_attention=True,
        )
        preds = logits_sample.argmax(dim=-1)  # (3, L)

    vb = config.value_bits
    sl = config.seq_len
    for i in range(x_sample.shape[0]):
        print(f'--- sample {i} ---')
        bits = x_sample[i].int().tolist()
        values = [int(''.join(str(b) for b in bits[t*vb:(t+1)*vb]), 2) for t in range(sl)]
        print(f'  input:           {values}')
        print(f'  target values:   {y_sample[i].tolist()}')
        print(f'  target pointers: {ptr_sample[i].tolist()}')
        print(f'  predicted values:{preds[i].tolist()}')
        print(f'  picked pointers: {attn_sample[i].argmax(dim=-1).tolist()}')
        print('  attention matrix (output_pos → input_pos):')
        for t in range(sl):
            row = attn_sample[i, t].tolist()
            mark_t = ' ✓' if preds[i, t].item() == y_sample[i, t].item() else ' ✗'
            print(f'    out{t}{mark_t}: {[f"{a:.2f}" for a in row]}')


if __name__ == '__main__':
    main()
