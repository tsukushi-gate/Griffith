"""
teacherコーパス生成スクリプト（data-free文字レベルKD、ROADMAP-v5）

nanoGPT文字レベルteacher（`resolve_text_teacher('hf-nanogpt-char:...')`）から
自己回帰サンプリングで固定長シーケンスを大量生成し、`.pt` に保存する。
生成後は同じ乱数系列から disjoint な held-out eval セットも切り出す。

data-freeの意味: 実データ（tiny_shakespeare等）は一切参照せず、teacherの
自己回帰サンプルのみを student 訓練データとする（train_kd_charlm_rddlgn.py側の
BPC評価でのみ実データを読む。このスクリプトはteacherサンプルの生成のみ）。

使用例:
    # スモーク（少量、CPU）
    uv run python experiments/gen_teacher_corpus.py \\
        --n-seqs 500 --length 64 --seed 42 --device cpu

    # フルスケール（GPU推奨、Modal経由）
    uv run python experiments/gen_teacher_corpus.py \\
        --teacher hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights \\
        --n-seqs 100000 --length 64 --gen-temperature 1.0 --seed 42 \\
        --eval-frac 0.02 --device auto
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.convert.text_teacher import (
    SHAKESPEARE_CHAR_VOCAB,
    decode_shakespeare_char,
    resolve_text_teacher,
)


def parse_args():
    parser = argparse.ArgumentParser(description='teacherコーパス生成（data-free文字LM KD用）')
    parser.add_argument(
        '--teacher', type=str,
        default='hf-nanogpt-char:sosier/nanoGPT-shakespeare-char-tied-weights',
        help='resolve_text_teacher に渡すスキーム文字列',
    )
    parser.add_argument('--n-seqs', type=int, default=100000, help='総生成シーケンス数')
    parser.add_argument('--length', type=int, default=64, help='各シーケンスの長さ（文字数）')
    parser.add_argument('--gen-temperature', type=float, default=1.0, help='生成温度')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--eval-frac', type=float, default=0.02,
                        help='held-out eval に回す割合（train/evalはdisjoint）')
    parser.add_argument('--batch-size', type=int, default=256, help='生成バッチサイズ')
    parser.add_argument('--device', type=str, default='auto',
                        help="'auto'/'cpu'/'cuda'/'mps'")
    parser.add_argument('--out-dir', type=str, default='data-charlm',
                        help='出力ディレクトリ（相対パス、Modal無改造規約）')
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != 'auto':
        return device_arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f'デバイス: {device}')
    print(f'teacher: {args.teacher}')

    os.makedirs(args.out_dir, exist_ok=True)

    print('teacherをロード中...')
    teacher = resolve_text_teacher(args.teacher, device=device)
    fingerprint = teacher.fingerprint
    print(f'  vocab_size={teacher.vocab_size} fingerprint={fingerprint}')

    # eval_frac分をdisjointに切り出す（乱数系列は連続シードで分離: train=seed,
    # eval=seed+1。同一seedのtrain内バッチ分割だと将来batch_size変更時に
    # train/evalの境界がズレるおそれがあるため、独立シードで完全disjointにする）。
    n_eval = max(1, int(args.n_seqs * args.eval_frac))
    n_train = args.n_seqs - n_eval
    print(f'総シーケンス数: {args.n_seqs:,} (train={n_train:,}, eval={n_eval:,})')

    def generate_all(n_total: int, seed: int) -> torch.Tensor:
        """batch_size分割でteacher.generateを繰り返し、(n_total, length)を返す。"""
        chunks = []
        n_batches = (n_total + args.batch_size - 1) // args.batch_size
        start_time = time.time()
        pbar = tqdm(range(n_batches), desc=f'生成中(seed={seed})')
        generated = 0
        for i in pbar:
            bs = min(args.batch_size, n_total - generated)
            # バッチごとに異なるseedにして同一乱数の繰り返しを避けつつ、
            # (base_seed, batch_index)で決定的にする。
            batch_seed = seed * 1_000_003 + i
            tokens = teacher.generate(
                bs, args.length,
                temperature=args.gen_temperature,
                seed=batch_seed,
                device=device,
            )
            chunks.append(tokens.to('cpu', dtype=torch.int16))
            generated += bs

            elapsed = time.time() - start_time
            sec_per_batch = elapsed / (i + 1)
            remaining_sec = sec_per_batch * (n_batches - (i + 1))
            eta = (datetime.now(JST) + timedelta(seconds=remaining_sec)).strftime('%H:%M:%S')
            pbar.set_postfix(
                seqs=f'{generated}/{n_total}',
                eta=eta,
            )
        return torch.cat(chunks, dim=0)[:n_total]

    train_tokens = generate_all(n_train, seed=args.seed)
    eval_tokens = generate_all(n_eval, seed=args.seed + 1)

    created_jst = datetime.now(JST).isoformat()
    meta = {
        'teacher_fingerprint': fingerprint,
        'repo_id': args.teacher,
        'vocab': SHAKESPEARE_CHAR_VOCAB,
        'n_seqs': args.n_seqs,
        'length': args.length,
        'gen_temperature': args.gen_temperature,
        'seed': args.seed,
        'eval_frac': args.eval_frac,
        'created_jst': created_jst,
    }

    fp8 = fingerprint[:8]
    base_name = f'corpus_{fp8}_{args.n_seqs}x{args.length}_T{args.gen_temperature}_s{args.seed}'
    train_path = os.path.join(args.out_dir, f'{base_name}.pt')
    eval_path = os.path.join(args.out_dir, f'{base_name}_eval.pt')

    torch.save({'tokens': train_tokens, 'meta': meta}, train_path)
    eval_meta = dict(meta, n_seqs=n_eval, split='eval')
    torch.save({'tokens': eval_tokens, 'meta': eval_meta}, eval_path)

    print('\n保存完了:')
    print(f'  train: {train_path}  shape={tuple(train_tokens.shape)}')
    print(f'  eval:  {eval_path}  shape={tuple(eval_tokens.shape)}')

    # サニティ確認: 先頭サンプルを200文字分デコードして表示
    sample_ids = train_tokens[0].tolist()
    sample_text = decode_shakespeare_char(sample_ids)[:200]
    print('\n=== デコードサンプル(先頭200文字) ===')
    print(sample_text)


if __name__ == '__main__':
    main()
