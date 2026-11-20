"""
DLGNAttentionTranslator 訓練スクリプト（Bahdanau-style seq2seq）

RDDLGN (Bührer et al., 2025) を超えることを目標とした Cross-Attention 付き seq2seq モデル。
`train_rddlgn.py` と同じ CLI・評価パターンを踏襲し、直接比較できるように設計。

使用例:
    # copy タスク（RDDLGN tiny と比較）
    uv run python experiments/train_attention_translator.py \\
        --task copy --config tiny -ni 50000 -ef 5000

    # reverse タスク
    uv run python experiments/train_attention_translator.py \\
        --task reverse --config tiny -ni 50000 -ef 5000

    # sort タスク
    uv run python experiments/train_attention_translator.py \\
        --task sort --config tiny -ni 100000 -ef 10000

    # 長系列 reverse (RDDLGN崩壊圏)
    uv run python experiments/train_attention_translator.py \\
        --task reverse --config custom --src-len 32 --tgt-len 32 \\
        --vocab-size 50 --d-emb 128 -ni 30000 -ef 2000

    # WMT'14 scale=0.1
    uv run python experiments/train_attention_translator.py \\
        --task wmt14 --config paper --scale 0.1 -ni 13000 -ef 1000

    # スモークテスト
    uv run python experiments/train_attention_translator.py \\
        --task copy --config tiny -ni 500 -ef 100 \\
        --num-samples 2000 --num-test-samples 500 --no-wandb --device cpu
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
from data import SPECIAL_TOKENS, TASK_DATASETS, TOKEN_BOS, TOKEN_EOS, TOKEN_PAD
from env_utils import load_dotenv
from wandb_utils import (
    finish as wandb_finish,
)
from wandb_utils import (
    init_wandb,
    log_checkpoint_artifact,
    log_metrics,
)

from griffith_logic import DLGNAttentionTranslator, TranslatorConfig

load_dotenv()


# ============================================================
# 訓練・評価
# ============================================================

def get_emb_reg_weight(step: int, ramp_start: int = 1000, ramp_end: int = 100000,
                       max_weight: float = 0.1) -> float:
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return max_weight
    return max_weight * (step - ramp_start) / (ramp_end - ramp_start)


def train_step(model, src, tgt, optimizer, step, label_smoothing=0.1):
    model.train()
    vocab_size = model.config.vocab_size

    tgt_input = tgt[:, :-1]
    tgt_label = tgt[:, 1:]

    logits, emb_loss = model(src, tgt_input)

    main_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        tgt_label.reshape(-1),
        label_smoothing=label_smoothing,
        ignore_index=TOKEN_PAD,
    )
    emb_weight = get_emb_reg_weight(step)
    total_loss = main_loss + emb_weight * emb_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), main_loss.item(), emb_loss.item()


@torch.no_grad()
def evaluate(model, loader, device, use_soft=True):
    was_training = model.training
    model.train(mode=use_soft)

    correct = 0
    total = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_batch, tgt_full = prepare_batch(src, tgt)

        tgt_input = tgt_full[:, :-1]
        tgt_label = tgt_full[:, 1:]

        logits, _ = model(src_batch, tgt_input)
        preds = logits.argmax(dim=-1)

        mask = tgt_label != TOKEN_PAD
        correct += ((preds == tgt_label) & mask).sum().item()
        total += mask.sum().item()

    model.train(mode=was_training)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def greedy_decode(model, src, max_len, bos_token=TOKEN_BOS):
    model.eval()
    batch = src.shape[0]
    device = src.device

    # Encode
    src_emb = model.src_embedding(src)
    memory, src_keys = model.encoder(src_emb)

    generated = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_emb = model.tgt_embedding(generated)
        logits = model.decoder(tgt_emb, memory, src_keys)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated[:, 1:]


@torch.no_grad()
def compute_bleu(model, loader, tokenizer, device, max_samples=1000):
    if tokenizer is None:
        return None
    try:
        import sacrebleu
    except ImportError:
        return None

    model.eval()
    hypotheses = []
    references = []
    count = 0

    for src, tgt in loader:
        if count >= max_samples:
            break
        src = src.to(device)
        tgt = tgt.to(device)
        seq_len = tgt.shape[1]

        preds = greedy_decode(model, src, max_len=seq_len)

        for i in range(src.shape[0]):
            if count >= max_samples:
                break
            hyp = tokenizer.decode(preds[i].tolist())
            ref = tokenizer.decode(tgt[i].tolist())
            if ref.strip():
                hypotheses.append(hyp)
                references.append(ref)
                count += 1

    if not hypotheses:
        return 0.0
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


# ============================================================
# ログ
# ============================================================

def setup_logger(task, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'attntrans_{task}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)
    print(f'ログファイル: {log_path}')

    def log(msg):
        ts = datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'[{ts}] {msg}\n')

    return log, log_file


# ============================================================
# メイン
# ============================================================

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',')]


def build_config(args) -> TranslatorConfig:
    if args.config == 'paper':
        config = TranslatorConfig.from_paper(scale=args.scale)
    elif args.config == 'tiny':
        config = TranslatorConfig.tiny()
    else:
        config = TranslatorConfig(
            vocab_size=args.vocab_size,
            d_emb=args.d_emb,
            src_len=args.src_len,
            tgt_len=args.tgt_len,
            n_layer_sizes=parse_int_list(args.n_sizes),
            key_encoder_sizes=parse_int_list(args.key_sizes),
            l_layer_sizes=parse_int_list(args.l_sizes),
            p_layer_sizes=parse_int_list(args.p_sizes),
            query_producer_sizes=parse_int_list(args.q_sizes),
            m_layer_sizes=parse_int_list(args.m_sizes),
            n_inputs=args.n_inputs,
            parameterization=args.parameterization,
            tau=args.tau,
            tau_attn=args.tau_attn,
        )

    if args.vocab_size is not None and args.config != 'custom':
        config.vocab_size = args.vocab_size
    if args.src_len is not None and args.config != 'custom':
        config.src_len = args.src_len
    if args.tgt_len is not None and args.config != 'custom':
        config.tgt_len = args.tgt_len
    if args.n_inputs != 2:
        config.n_inputs = args.n_inputs
    if args.parameterization != 'softmax':
        config.parameterization = args.parameterization
    config.tau_warp = args.tau_warp
    config.gumbel = not args.no_gumbel
    config.tau_attn = args.tau_attn

    return config


def prepare_batch(src, tgt, bos_token=TOKEN_BOS, eos_token=TOKEN_EOS):
    batch = src.shape[0]
    device = src.device
    bos = torch.full((batch, 1), bos_token, dtype=torch.long, device=device)
    eos = torch.full((batch, 1), eos_token, dtype=torch.long, device=device)
    tgt_full = torch.cat([bos, tgt, eos], dim=1)
    return src, tgt_full


def load_n(loader, n):
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i == n:
                return


def main():
    parser = argparse.ArgumentParser(description='DLGN Attention Translator 訓練')

    parser.add_argument('--task', type=str, default='copy',
                        choices=list(TASK_DATASETS.keys()) + ['wmt14'])
    parser.add_argument('--num-samples', type=int, default=50000)
    parser.add_argument('--num-test-samples', type=int, default=5000)

    # モデル
    parser.add_argument('--config', type=str, default='tiny',
                        choices=['paper', 'tiny', 'custom'])
    parser.add_argument('--scale', type=float, default=1.0)

    parser.add_argument('--vocab-size', type=int, default=None)
    parser.add_argument('--d-emb', type=int, default=32)
    parser.add_argument('--src-len', type=int, default=None)
    parser.add_argument('--tgt-len', type=int, default=None)
    parser.add_argument('--n-sizes', type=str, default='64,64')
    parser.add_argument('--key-sizes', type=str, default='32,16')
    parser.add_argument('--l-sizes', type=str, default='64,64')
    parser.add_argument('--p-sizes', type=str, default='96,64')
    parser.add_argument('--q-sizes', type=str, default='32,16')
    parser.add_argument('--m-sizes', type=str, default='128,128,300')
    parser.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--parameterization', type=str, default='softmax',
                        choices=['softmax', 'independent', 'iwp', 'warp'])
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--tau-attn', type=float, default=1.0)
    parser.add_argument('--tau-warp', type=float, default=1.0)
    parser.add_argument('--no-gumbel', action='store_true')

    # 訓練
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-ni', '--num-iterations', type=int, default=50000)
    parser.add_argument('-ef', '--eval-freq', type=int, default=1000)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--lr-factor', type=float, default=0.8)

    # チェックポイント
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    # 環境
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--data-dir', type=str, default='./data-wmt14')
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

    # vocab_size
    if args.vocab_size is None and args.config == 'custom':
        config.vocab_size = 100
    elif args.vocab_size is not None:
        config.vocab_size = args.vocab_size

    # seq_len
    if args.src_len is not None:
        config.src_len = args.src_len
    if args.tgt_len is not None:
        config.tgt_len = args.tgt_len

    # M最終層が vocab_size で割り切れるか
    m_last = config.m_layer_sizes[-1]
    if m_last % config.vocab_size != 0:
        group_factor = max(1, m_last // config.vocab_size)
        config.m_layer_sizes[-1] = config.vocab_size * max(group_factor, 1)
        print(f'M最終層を {m_last} → {config.m_layer_sizes[-1]} に調整 '
              f'(vocab_size={config.vocab_size} の倍数)')

    print('\n=== DLGN Attention Translator 訓練 ===')
    print(f'タスク: {args.task}')
    print(f'語彙サイズ: {config.vocab_size}')
    print(f'埋め込み次元: {config.d_emb}')
    print(f'src_len={config.src_len} tgt_len={config.tgt_len} pos_bits={config.pos_bits}')
    print(f'N-layers: {config.n_layer_sizes} (Encoder FF, parallel)')
    print(f'key-encoder: {config.key_encoder_sizes}')
    print(f'L-layers: {config.l_layer_sizes}')
    print(f'P-layers: {config.p_layer_sizes} (Decoder RNN)')
    print(f'query-producer: {config.query_producer_sizes}')
    print(f'M-layers: {config.m_layer_sizes}')
    print(f'tau_attn={config.tau_attn}')

    # データセット
    wmt_tokenizer = None
    if args.task == 'wmt14':
        from data.wmt14 import download_and_tokenize
        train_set, test_set, wmt_tokenizer = download_and_tokenize(
            vocab_size=config.vocab_size,
            seq_len=config.src_len,
            data_dir=args.data_dir,
        )
        config.vocab_size = wmt_tokenizer.vocab_size
    else:
        # 合成タスクは src_len == tgt_len
        seq_len = config.src_len
        assert seq_len == config.tgt_len, (
            '合成タスクでは src_len と tgt_len を同一にしてください'
        )
        dataset_cls = TASK_DATASETS[args.task]
        train_set = dataset_cls(config.vocab_size, seq_len, args.num_samples)
        test_set = dataset_cls(config.vocab_size, seq_len, args.num_test_samples)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    print(f'訓練サンプル: {len(train_set):,}')
    print(f'テストサンプル: {len(test_set):,}')

    # モデル
    model = DLGNAttentionTranslator(config, device=args.device).to(args.device)
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
    log(f'START task={args.task} config={args.config} vocab={config.vocab_size} '
        f'd_emb={config.d_emb} src_len={config.src_len} tgt_len={config.tgt_len} '
        f'lr={args.lr} bs={args.batch_size} ni={args.num_iterations} '
        f'device={args.device} params={counts["total"]:,}')

    ckpt_prefix = f'attntrans_{args.task}_{args.config}'
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        ckpt_prefix, base_dir=args.checkpoint_dir,
    )
    vocab_path = None
    if wmt_tokenizer is not None:
        vocab_path = os.path.join(args.data_dir, f'vocab_v{config.vocab_size}.json')
    ckpt_metadata = {
        'args': vars(args), 'config': asdict(config),
        'task': args.task, 'vocab_path': vocab_path,
    }

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

    run_name = (f'attntrans_{args.task}_{args.config}'
                f'_v{config.vocab_size}_sl{config.src_len}_ni{args.num_iterations}')
    wandb_config = {**vars(args), **asdict(config)}
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict=wandb_config,
            model=model,
            device=args.device,
            tags=['attention_translator', args.task, args.config],
            group='attention_translator',
        )

    remaining_iters = args.num_iterations - start_step
    print(f'\n訓練開始: {args.num_iterations:,} イテレーション（残り {remaining_iters:,}）')
    print('=' * 60)

    train_start = time.time()

    pbar = tqdm(
        enumerate(load_n(train_loader, remaining_iters), start=start_step),
        desc='train', total=args.num_iterations, initial=start_step,
    )

    for step, (src, tgt) in pbar:
        src = src.to(args.device)
        tgt = tgt.to(args.device)

        src_batch, tgt_full = prepare_batch(src, tgt)

        total_loss, main_loss, emb_loss = train_step(
            model, src_batch, tgt_full, optimizer, step,
            label_smoothing=args.label_smoothing,
        )

        elapsed = time.time() - train_start
        sec_per_iter = elapsed / (step + 1 - start_step + 1e-9)
        remaining_sec = sec_per_iter * (args.num_iterations - (step + 1))
        eta_str = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')

        pbar.set_postfix(
            loss=f'{total_loss:.4f}',
            main=f'{main_loss:.4f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.5f}',
            eta=eta_str,
        )

        if (step + 1) % args.eval_freq == 0 or step == args.num_iterations - 1:
            test_acc_soft = evaluate(model, test_loader, args.device, use_soft=True)
            test_acc_hard = evaluate(model, test_loader, args.device, use_soft=False)

            scheduler.step(1.0 - test_acc_soft)

            is_best = test_acc_soft > best_test_acc
            if is_best:
                best_test_acc = test_acc_soft

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
            emb_w = get_emb_reg_weight(step)
            current_lr = optimizer.param_groups[0]['lr']

            bleu_str = ''
            if args.task == 'wmt14' and (step == args.num_iterations - 1):
                bleu = compute_bleu(model, test_loader, wmt_tokenizer, args.device)
                if bleu is not None:
                    bleu_str = f'  BLEU={bleu:.2f}'

            print(
                f'\n[iter {step+1:,}/{args.num_iterations:,}] '
                f'loss={total_loss:.4f} main={main_loss:.4f} emb={emb_loss:.4f}(w={emb_w:.4f})  '
                f'acc(soft)={test_acc_soft:.4f} acc(hard)={test_acc_hard:.4f}'
                f'{bleu_str}{"  ★ BEST" if is_best else ""}'
                f'  lr={current_lr:.5f}'
                f'  [{elapsed_str}<{remain_str}, {iter_per_sec:.1f}it/s, 終了予定 {eta_str}]'
            )

            log(f'ITER {step+1}/{args.num_iterations} '
                f'loss={total_loss:.4f} main={main_loss:.4f} emb={emb_loss:.4f} '
                f'emb_w={emb_w:.4f} acc_soft={test_acc_soft:.4f} '
                f'acc_hard={test_acc_hard:.4f}{bleu_str} lr={current_lr:.5f} '
                f'best={str(is_best).lower()}')

            if use_wandb:
                wandb_metrics = {
                    'iteration': step + 1,
                    'loss/total': total_loss,
                    'loss/main': main_loss,
                    'loss/emb_reg': emb_loss,
                    'loss/emb_reg_weight': emb_w,
                    'acc/soft': test_acc_soft,
                    'acc/hard': test_acc_hard,
                    'best_test_acc': best_test_acc,
                    'lr': current_lr,
                }
                if args.task == 'wmt14' and bleu_str:
                    wandb_metrics['bleu'] = bleu
                log_metrics(wandb_metrics, step=step + 1)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n最終テスト精度(soft): {best_test_acc:.4f}  (総訓練時間: {total_time})')
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

    # デコード例 + attention
    print('\n=== デコード例 ===')
    model.eval()
    sample_src, sample_tgt = next(iter(test_loader))
    sample_src = sample_src[:4].to(args.device)
    sample_tgt = sample_tgt[:4].to(args.device)

    _, sample_tgt_full = prepare_batch(sample_src, sample_tgt)
    with torch.no_grad():
        sample_logits, _ = model(sample_src, sample_tgt_full[:, :-1])
        sample_preds = sample_logits.argmax(dim=-1)
        attn = model.decoder.last_attention  # (B, tgt_len, src_len)

    for i in range(min(4, sample_src.shape[0])):
        src_ids = sample_src[i].tolist()
        tgt_ids = sample_tgt[i].tolist()
        pred_ids = sample_preds[i, :len(tgt_ids)].tolist()

        if wmt_tokenizer is not None:
            print(f'  src:  {wmt_tokenizer.decode(src_ids)}')
            print(f'  tgt:  {wmt_tokenizer.decode(tgt_ids)}')
            print(f'  pred: {wmt_tokenizer.decode(pred_ids)}')
        else:
            match = '✓' if pred_ids == tgt_ids else '✗'
            print(f'  src:  {src_ids}')
            print(f'  tgt:  {tgt_ids}')
            print(f'  pred: {pred_ids}  {match}')
            # attention 最大位置（各 tgt step ごと）
            attn_argmax = attn[i, :len(tgt_ids)].argmax(dim=-1).tolist()
            print(f'  attn→: {attn_argmax}')
        print()


if __name__ == '__main__':
    main()
