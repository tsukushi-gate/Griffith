"""適格度トレースλ×タスク×系列長T×シードの実験マトリクスを回すスイープ。

各ジョブは独立プロセスで実行(CPUコア並列)。結果はJSONLに逐次追記し、
最後に集計CSV/JSONを書き出す。図は analyze_ap_eligibility.py が生成する。

メトリクスは「回答トークン精度」= デコーダstep0(=回答位置)のargmax一致率。
末尾EOS予測を含めず、遅延想起/XORの成否のみを測る(chance: recall=1/n_symbols, xor=0.5)。

使用例(tinyフルマトリクス):
  uv run python experiments/sweep_ap_eligibility.py \
    --tasks delayed_recall temporal_xor copy \
    --lambdas 0 0.5 0.9 0.99 1.0 --seq-lens 5 10 20 40 --seeds 0 1 2 \
    --iters 3000 --eval-every 300 --workers 10 --tag tiny

  # 受け入れ基準2に絞った最小セット:
  uv run python experiments/sweep_ap_eligibility.py \
    --tasks delayed_recall --lambdas 0 0.9 0.99 --seq-lens 10 --seeds 0 1 2 \
    --iters 3000 --eval-every 300 --workers 9 --tag ac2
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

# 結果はCWD相対に置く。ローカルはリポジトリ直下から実行しresults/へ、
# Modalはcwd=/workspace(Volume)なので/workspace/results/へ永続化される。
RESULTS_DIR = os.path.join(os.getcwd(), 'results', 'ap_eligibility')

# --- スイープ用の固定tiny構成(copy向けtinyより小さく、高速) ---
SWEEP_VOCAB = 16
SWEEP_GROUP_FACTOR = 4
SWEEP_BATCH = 64
TRAIN_SAMPLES = 20000
TEST_SAMPLES = 2000
CONVERGENCE_THRESHOLD = 0.9  # 収束判定(回答精度)

# 容量サイズ別の層構成(d_emb, N, K, L, P, M中間)
CONFIG_SIZES = {
    'small': dict(d_emb=32, n=[128, 128], k=[256, 256], l=[128, 128],
                  p=[256, 128], m_mid=[256, 256]),
    'mid': dict(d_emb=64, n=[256, 256], k=[512, 512], l=[256, 256],
                p=[512, 256], m_mid=[512, 512]),
    'big': dict(d_emb=128, n=[512, 512], k=[1024, 1024], l=[512, 512],
                p=[1024, 768], m_mid=[2000, 2000]),
}


def _build_config(size: str = 'small'):
    from griffith_logic import RDDLGNConfig
    s = CONFIG_SIZES[size]
    return RDDLGNConfig(
        vocab_size=SWEEP_VOCAB,
        d_emb=s['d_emb'],
        seq_len=4,  # ジョブごとに上書き
        n_layer_sizes=list(s['n']),
        k_layer_sizes=list(s['k']),
        l_layer_sizes=list(s['l']),
        p_layer_sizes=list(s['p']),
        m_layer_sizes=list(s['m_mid']) + [SWEEP_VOCAB * SWEEP_GROUP_FACTOR],
        tau=1.0,
        n_inputs=2,
        parameterization='independent',
        connections='random',
    )


def _chance_level(task: str) -> float:
    if task == 'temporal_xor':
        return 0.5
    if task == 'delayed_recall':
        from data.temporal import _DEFAULT_N_SYMBOLS
        return 1.0 / _DEFAULT_N_SYMBOLS
    return 0.0


def _answer_accuracy(trainer, loader, device):
    """回答位置(デコーダstep0)のトークン精度を計算。"""
    import torch
    from ap_trainer_rddlgn import _stack_forward
    from data import TOKEN_BOS, TOKEN_EOS

    from griffith_logic.rddlgn import _init_hidden

    model = trainer.model
    cfg = model.config
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            batch = src.shape[0]
            bos = torch.full((batch, 1), TOKEN_BOS, dtype=torch.long, device=device)
            eos = torch.full((batch, 1), TOKEN_EOS, dtype=torch.long, device=device)
            tgt_full = torch.cat([bos, tgt, eos], dim=1)
            tgt_input = tgt_full[:, :-1]
            answer = tgt_full[:, 1]  # 回答トークン(=tgt[:,0])

            src_emb = trainer._embedding_forward(model.src_embedding, src)
            n_out, _ = trainer._encode_n_layers(src_emb)
            context, _ = trainer._encode_k_layers(n_out, batch, device)
            tgt_emb = trainer._embedding_forward(model.tgt_embedding, tgt_input)
            l_out, _ = trainer._decode_l_layers(tgt_emb)

            p_prev = _init_hidden(
                batch, trainer._p_out_dim, device,
                cfg.init_mode, cfg.init_scale,
            )
            l_t = l_out[:, 0, :]
            p_input = torch.cat([p_prev, context, l_t], dim=-1)
            p_t, _ = _stack_forward(model.decoder.p_layers, p_input)
            m_input = torch.cat([p_t, context, l_t], dim=-1)
            m_t, _ = _stack_forward(model.decoder.m_layers, m_input)
            logits0 = trainer._group_sum(m_t)
            pred = logits0.argmax(dim=-1)
            correct += (pred == answer).sum().item()
            total += batch
    return correct / total if total > 0 else 0.0


def run_job(job: dict) -> dict:
    """1ジョブ(task, lambda, T, seed)を訓練・評価して結果を返す。"""
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(1)

    task = job['task']
    lam = job['lambda']
    T = job['seq_len']
    seed = job['seed']
    iters = job['iters']
    eval_every = job['eval_every']
    trace_norm = job['trace_norm']
    config_size = job.get('config_size', 'small')
    device = job.get('device', 'cpu')

    from ap_trainer_rddlgn_eligibility import EligibilityRDDLGNAPTrainer
    from data import TASK_DATASETS, TOKEN_PAD
    from train_rddlgn import load_n, prepare_batch

    from griffith_logic import RDDLGN

    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg = _build_config(config_size)
    cfg.seq_len = T

    ds_cls = TASK_DATASETS[task]
    train_set = ds_cls(vocab_size=cfg.vocab_size, seq_len=T, num_samples=TRAIN_SAMPLES)
    test_set = ds_cls(vocab_size=cfg.vocab_size, seq_len=T, num_samples=TEST_SAMPLES)
    train_loader = DataLoader(train_set, batch_size=SWEEP_BATCH, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=SWEEP_BATCH, shuffle=False, drop_last=True)

    model = RDDLGN(cfg, device=device).to(device)
    trainer = EligibilityRDDLGNAPTrainer(
        model, lr=job['lr'], update_mode=job['update_mode'],
        pad_token=TOKEN_PAD, eligibility_lambda=lam, trace_norm=trace_norm,
    )

    curve = []  # [(iter, acc)]
    best_acc = 0.0
    best_iter = 0
    converged_iter = None
    for i, (src, tgt) in enumerate(load_n(train_loader, iters)):
        src = src.to(device)
        tgt = tgt.to(device)
        src_b, tgt_full = prepare_batch(src, tgt)
        trainer.train_step(src_b, tgt_full)
        if (i + 1) % eval_every == 0 or (i + 1) == iters:
            acc = _answer_accuracy(trainer, test_loader, device)
            curve.append([i + 1, acc])
            if acc > best_acc:
                best_acc = acc
                best_iter = i + 1
            if converged_iter is None and acc >= CONVERGENCE_THRESHOLD:
                converged_iter = i + 1

    return {
        'task': task,
        'lambda': lam,
        'seq_len': T,
        'seed': seed,
        'trace_norm': trace_norm,
        'config_size': config_size,
        'iters': iters,
        'best_acc': best_acc,
        'best_iter': best_iter,
        'final_acc': curve[-1][1] if curve else 0.0,
        'converged_iter': converged_iter,
        'chance': _chance_level(task),
        'curve': curve,
        'wall_sec': round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='適格度トレース スイープ')
    parser.add_argument('--tasks', nargs='+', default=['delayed_recall', 'temporal_xor', 'copy'])
    parser.add_argument('--lambdas', nargs='+', type=float, default=[0, 0.5, 0.9, 0.99, 1.0])
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[5, 10, 20, 40])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--eval-every', type=int, default=300)
    parser.add_argument('--trace-norm', type=str, default='none',
                        choices=['none', 'one_minus_lambda', 'seq_len'])
    parser.add_argument('--config-size', type=str, default='small',
                        choices=['small', 'mid', 'big'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--update-mode', type=int, default=1, choices=[1, 2])
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--tag', type=str, default='tiny')
    args = parser.parse_args()

    # Modal等の非TTY環境でも進捗が逐次見えるよう行バッファリングを強制
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    os.makedirs(RESULTS_DIR, exist_ok=True)
    jsonl_path = os.path.join(RESULTS_DIR, f'sweep_{args.tag}.jsonl')
    summary_path = os.path.join(RESULTS_DIR, f'sweep_{args.tag}_summary.json')

    jobs = []
    for task in args.tasks:
        for lam in args.lambdas:
            for T in args.seq_lens:
                for seed in args.seeds:
                    jobs.append({
                        'task': task, 'lambda': lam, 'seq_len': T, 'seed': seed,
                        'iters': args.iters, 'eval_every': args.eval_every,
                        'trace_norm': args.trace_norm, 'lr': args.lr,
                        'update_mode': args.update_mode,
                        'config_size': args.config_size, 'device': args.device,
                    })

    print(f'ジョブ数: {len(jobs)}  workers={args.workers}  tag={args.tag}')
    print(f'マトリクス: tasks={args.tasks} lambdas={args.lambdas} '
          f'T={args.seq_lens} seeds={args.seeds} iters={args.iters}')
    print(f'出力: {jsonl_path}')

    # 既存JSONLはtag単位で上書き(再現性のため)
    open(jsonl_path, 'w').close()

    results = []
    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            elapsed = time.time() - t_start
            eta = elapsed / done * (len(jobs) - done)
            print(f'[{done}/{len(jobs)}] {r["task"]:14s} '
                  f'λ={r["lambda"]:<4} T={r["seq_len"]:<2} s={r["seed"]} '
                  f'best={r["best_acc"]:.3f} conv={r["converged_iter"]} '
                  f'({r["wall_sec"]}s)  ETA {eta/60:.1f}min', flush=True)

    with open(summary_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f,
                  ensure_ascii=False, indent=2)
    print(f'\n完了: {done}ジョブ  総時間 {(time.time()-t_start)/60:.1f}min')
    print(f'集計: {summary_path}')


if __name__ == '__main__':
    main()
