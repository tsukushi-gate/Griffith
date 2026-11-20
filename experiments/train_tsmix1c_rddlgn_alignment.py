"""
RDDLGN をアライメント可視化実験と同じ小タスクに載せた比較

(`train_tsmix1b_alignment.py`)は LogicSeqMixer の3アーム
(broadcast / gather / cross)を比較した。本スクリプトは**同じ条件・同程度の
ゲート数**で **RDDLGN**(Bührer et al., 2025 のリカレント方式)を同じ3タスク
(copy / reverse / delayed_recall)に載せ、**1b と同一手法の感度マップ**で
直接比較できる図を作る。

RDDLGN には attention 行列が存在しない(エンコーダは系列全体を K-layer の
固定幅リカレント状態1本に畳み込み、デコーダは単一 context を全出力位置で
共有する)。したがって 1b の (b) attention 重みに相当するものは無く、
**(a) 勾配感度マップのみ**を測る。算出式・行正規化・描画コード・カラースケール・
軸ラベルはすべて 1b と共通(`train_tsmix1b_alignment` から import)。

    sens[t, s] = mean_batch  Σ_d | ∂ logit[t, y_t] / ∂ src_emb[s, d] |

**収束の判定基準(1b と同じ事前宣言。実測に合わせて緩めない)**:
    `argmax_acc == 1.0` かつ `corr_target >= 0.5`

使用例:
    uv run python experiments/train_tsmix1c_rddlgn_alignment.py \\
        --task reverse -ni 4000 --align-every 250
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import matplotlib

matplotlib.use('Agg')
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import TASK_DATASETS, TOKEN_EOS
from env_utils import load_dotenv
from train_rddlgn import evaluate, prepare_batch, train_step
from train_tsmix1b_alignment import (
    CONVERGE_ARGMAX_ACC,
    CONVERGE_CORR,
    _tick_labels,
    alignment_metrics,
    save_gif,
    save_panel,
    save_single,
)
from wandb_utils import (
    LossLogPacer,
    add_experiment_args,
    count_logic_gates,
    init_wandb,
    log_artifact,
    log_images,
    log_metrics,
    normalize_tags,
    progress_metrics,
)
from wandb_utils import (
    finish as wandb_finish,
)

from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig

load_dotenv()


# ============================================================
# ゲート数
# ============================================================

def gate_breakdown(model: RDDLGN) -> dict[str, int]:
    """RDDLGN のゲート数(LUT ニューロン数)を層グループ別に数える。

    埋め込みは nn.Embedding(実数パラメータ)であり LUT ゲートではないので 0。
    N/K = encoder(前段FF / リカレント)、L/P = decoder(前段FF / 自己回帰
    リカレント)、M = 出力(GroupSum 直前)。
    """

    def _gates(root) -> int:
        return sum(int(m.num_neurons) for m in root.modules()
                   if hasattr(m, 'num_neurons'))

    counts = {
        'embedding': 0,
        'N_enc_ff': _gates(model.encoder.n_layers),
        'K_enc_recurrent': _gates(model.encoder.k_layers),
        'L_dec_ff': _gates(model.decoder.l_layers),
        'P_dec_recurrent': _gates(model.decoder.p_layers),
        'M_out': _gates(model.decoder.m_layers),
    }
    counts['total'] = sum(int(m.num_neurons) for m in model.modules()
                          if hasattr(m, 'num_neurons'))
    return counts


# ============================================================
# 測定(感度マップ) — 1b と同一の定義
# ============================================================

def sensitivity_map(model, src, tgt_input, tgt_label) -> np.ndarray:
    """感度マップ (n_out, n_src) を autograd で実測する(1b と同一定義)。

    sens[t, s] = mean_batch Σ_d |∂ logit[t, y_t] / ∂ src_emb[s, d]|

    RDDLGN では src_emb → encoder(N/K) → context(固定幅1本) → decoder という
    経路なので、**全ての src 位置の寄与が単一の context を通る**。それでも
    勾配自体は src 位置ごとに分かれて流れるため、1b と同じ式で測れる。
    soft モード(`model.train()`)で計算する。
    """
    was_training = model.training
    model.train()

    src_emb = model.src_embedding(src).detach().clone().requires_grad_(True)
    context = model.encoder(src_emb)
    tgt_emb = model.tgt_embedding(tgt_input)
    logits = model.decoder(tgt_emb, context)        # (B, T, V)

    n_out = logits.shape[1]
    n_src = src.shape[1]
    out = np.zeros((n_out, n_src), dtype=np.float64)
    for t in range(n_out):
        scalar = logits[:, t, :].gather(1, tgt_label[:, t:t + 1]).sum()
        grad, = torch.autograd.grad(
            scalar, src_emb, retain_graph=(t < n_out - 1))
        out[t] = grad.abs().sum(dim=-1).mean(dim=0).detach().float().cpu().numpy()

    model.train(mode=was_training)
    return out


# ============================================================
# 設定
# ============================================================

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',')]


def build_config(args) -> RDDLGNConfig:
    """CLI からゲート数を揃えた RDDLGN 構成を作る。

    既定値は LogicSeqMixer tiny(1,340 ゲート)に対して
    合計 1,344 ゲート(+0.30%)になるよう選んである。
    """
    config = RDDLGNConfig(
        vocab_size=args.vocab_size,
        d_emb=args.d_emb,
        seq_len=args.seq_len,
        n_layer_sizes=parse_int_list(args.n_sizes),
        k_layer_sizes=parse_int_list(args.k_sizes),
        l_layer_sizes=parse_int_list(args.l_sizes),
        p_layer_sizes=parse_int_list(args.p_sizes),
        m_layer_sizes=parse_int_list(args.m_sizes),
        tau=args.tau,
    )
    # M 最終層は vocab_size の倍数でなければならない(GroupSum の制約)
    m_last = config.m_layer_sizes[-1]
    if m_last % config.vocab_size != 0:
        group_factor = max(1, m_last // config.vocab_size)
        config.m_layer_sizes[-1] = config.vocab_size * group_factor
        print(f'M最終層を {m_last} → {config.m_layer_sizes[-1]} に調整 '
              f'(vocab_size={config.vocab_size} の倍数)')
    return config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='RDDLGN をアライメント可視化実験と同じ小タスクで比較')
    p.add_argument('--task', type=str, default='reverse',
                   choices=['copy', 'reverse', 'delayed_recall'])
    # --- 1b と完全に同一にする条件 ---
    p.add_argument('--num-samples', type=int, default=20000)
    p.add_argument('--num-test-samples', type=int, default=1024)
    p.add_argument('--vocab-size', type=int, default=32)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=0.001)
    p.add_argument('-bs', '--batch-size', type=int, default=64)
    p.add_argument('-ni', '--num-iterations', type=int, default=4000)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)

    # --- RDDLGN 側の寸法(既定でゲート数 1,344 = 1b tiny の +0.30%) ---
    p.add_argument('--d-emb', type=int, default=64)
    p.add_argument('--n-sizes', type=str, default='96,96')
    p.add_argument('--k-sizes', type=str, default='160,128')
    p.add_argument('--l-sizes', type=str, default='96,96')
    p.add_argument('--p-sizes', type=str, default='160,128')
    p.add_argument('--m-sizes', type=str, default='128,128,128')
    p.add_argument('--tau', type=float, default=1.0)

    p.add_argument('--align-every', type=int, default=250)
    p.add_argument('--align-batch', type=int, default=8)
    p.add_argument('--eval-freq', type=int, default=500)

    p.add_argument('--out-dir', type=str, default='results/tsmix1c')
    p.add_argument('--device', type=str, default=None,
                   help='未指定なら自動検出 (cuda→mps→cpu)')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--job-type', type=str, default='ablation',
                   choices=['train', 'ablation', 'sweep', 'eval', 'smoke'])
    add_experiment_args(p)
    return p


# ============================================================
# メイン
# ============================================================

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    parser = build_parser()
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f'device={args.device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)
    arm = 'rddlgn'
    tag = f'{args.task}_{arm}'
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n=== RDDLGN アライメント可視化 ({tag}) ===')
    print(f'task={args.task} vocab={config.vocab_size} seq_len={config.seq_len} '
          f'd_emb={config.d_emb}')
    print(f'N={config.n_layer_sizes} K={config.k_layer_sizes} '
          f'L={config.l_layer_sizes} P={config.p_layer_sizes} '
          f'M={config.m_layer_sizes}')
    print(f'出力先: {os.path.abspath(out_dir)}')

    dataset_cls = TASK_DATASETS[args.task]
    train_set = dataset_cls(config.vocab_size, config.seq_len, args.num_samples)
    test_set = dataset_cls(config.vocab_size, config.seq_len, args.num_test_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, drop_last=True)

    model = RDDLGN(config, device=args.device).to(args.device)
    gates = gate_breakdown(model)
    print('ゲート数内訳:')
    for k, v in gates.items():
        print(f'  {k}: {v:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)

    # 可視化に使う固定検証バッチ(shuffle=False なので毎回同一)
    fixed_src, fixed_tgt = next(iter(test_loader))
    fixed_src = fixed_src[:args.align_batch].to(args.device)
    fixed_tgt = fixed_tgt[:args.align_batch].to(args.device)
    _, fixed_tgt_full = prepare_batch(fixed_src, fixed_tgt)
    fixed_tgt_input = fixed_tgt_full[:, :-1]
    fixed_tgt_label = fixed_tgt_full[:, 1:]

    n_out = fixed_tgt_input.shape[1]
    n_src = fixed_src.shape[1]
    label0 = fixed_tgt_label[0].tolist()
    valid_rows = [t for t in range(n_out) if label0[t] != TOKEN_EOS]
    if args.task == 'delayed_recall':
        valid_rows = [0]
    print(f'アライメント行列: {n_out}(out) x {n_src}(src)  指標対象行={valid_rows}')

    col_labels = _tick_labels(fixed_src[0].tolist())
    row_labels = _tick_labels(label0)

    run_name = (f'tsmix1c_align_{args.task}_rddlgn_v{config.vocab_size}'
                f'_L{config.seq_len}_ni{args.num_iterations}')
    use_wandb = False
    if not args.no_wandb:
        use_wandb = init_wandb(
            run_name=run_name,
            config_dict={**vars(args), **asdict(config), 'gates': gates},
            model=model,
            device=args.device,
            tags=(['rddlgn', 'tsmix-1c', 'alignment', args.task, 'arm:rddlgn']
                  + normalize_tags(model='rddlgn', dataset=args.task,
                                   seed=args.seed, series='TSMIX-1')),
            group='tsmix-1',
            job_type=args.job_type,
            notes=args.notes,
            hypothesis=args.hypothesis,
        )
        if use_wandb:
            log_metrics(count_logic_gates(model), step=0)

    history: list[dict] = []
    sens_pngs: list[str] = []
    sens_frames: list[tuple[int, np.ndarray]] = []

    def snapshot(step: int) -> dict:
        sens = sensitivity_map(model, fixed_src, fixed_tgt_input, fixed_tgt_label)
        sens_frames.append((step, sens))
        m = alignment_metrics(sens, args.task, valid_rows)
        record = {'step': step, 'sens': m}
        path = os.path.join(out_dir, f'sens_step{step:06d}.png')
        save_single(
            path, sens,
            f'{args.task} / rddlgn / sensitivity (step {step:,})\n'
            f'corr_target={m["corr_target"]:.3f} argmax={m["argmax_acc"]:.3f}',
            row_labels=row_labels, col_labels=col_labels)
        sens_pngs.append(path)

        if use_wandb:
            from PIL import Image
            log_images(
                f'align/{tag}', [Image.open(path)],
                captions=[f'感度マップ(勾配|∂logit/∂src_emb|・重み行列ではない) '
                          f'step={step}'],
                step=max(step, 1))
            metrics = {f'align/sens/{k}': v for k, v in m.items()}
            metrics['align/corr_target'] = m['corr_target']
            metrics['align/argmax_acc'] = m['argmax_acc']
            log_metrics(metrics, step=max(step, 1))

        print(f'[align step {step:,}] sens: corr_t={m["corr_target"]:+.3f} '
              f'corr_diag={m["corr_diag"]:+.3f} '
              f'corr_anti={m["corr_anti_diag"]:+.3f} '
              f'argmax={m["argmax_acc"]:.3f}', flush=True)
        history.append(record)
        return record

    snapshot(0)

    print(f'\n訓練開始: {args.num_iterations:,} イテレーション')
    train_start = time.time()
    pacer = LossLogPacer()
    step = 0
    acc_soft = acc_hard = float('nan')

    while step < args.num_iterations:
        for src, tgt in train_loader:
            if step >= args.num_iterations:
                break
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            src_batch, tgt_full = prepare_batch(src, tgt)
            total_loss, main_loss, emb_loss = train_step(
                model, src_batch, tgt_full, optimizer, step,
                label_smoothing=args.label_smoothing)
            step += 1

            elapsed = time.time() - train_start
            is_last = step == args.num_iterations

            # SKILL.md §11: train loss は壁時計で最低5分に1回
            if pacer.should_log(step, elapsed, is_last=is_last):
                if pacer.resolved_now:
                    print(pacer.describe(), flush=True)
                if use_wandb:
                    log_metrics({
                        'iteration': step,
                        'loss/total': total_loss,
                        'loss/main': main_loss,
                        'loss/emb_reg': emb_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                        **progress_metrics(step, step, args.num_iterations,
                                           elapsed),
                    }, step=step)

            if step % args.eval_freq == 0 or is_last:
                acc_soft = evaluate(model, test_loader, args.device, use_soft=True)
                acc_hard = evaluate(model, test_loader, args.device, use_soft=False)
                sec_per_iter = elapsed / max(step, 1)
                remaining = sec_per_iter * (args.num_iterations - step)
                eta = datetime.now(JST) + timedelta(seconds=remaining)
                print(f'[iter {step:,}/{args.num_iterations:,}] '
                      f'loss={total_loss:.4f} acc(soft)={acc_soft:.4f} '
                      f'acc(hard)={acc_hard:.4f} '
                      f'[{str(timedelta(seconds=int(elapsed)))}<'
                      f'{str(timedelta(seconds=int(remaining)))}, '
                      f'{step / max(elapsed, 1e-9):.1f}it/s, '
                      f'終了予定 {eta.strftime("%H:%M:%S")}]', flush=True)
                if use_wandb:
                    log_metrics({'acc/soft': acc_soft, 'acc/hard': acc_hard,
                                 'iteration': step}, step=step)

            if step % args.align_every == 0 or is_last:
                snapshot(step)

    total_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f'\n訓練完了 ({total_time})  acc(soft)={acc_soft:.4f} '
          f'acc(hard)={acc_hard:.4f}')

    # ---- 3枚組パネル(step 0 → 中間 → 最終)と GIF ----
    def pick3(frames):
        if len(frames) <= 3:
            return frames
        return [frames[0], frames[len(frames) // 2], frames[-1]]

    # 図中テキストは日本語フォント未導入環境で豆腐になるため英語
    sens_caption = (
        'Sensitivity map = mean_batch sum_d |d logit[t, y_t] / d src_emb[s, d]|,'
        ' measured by autograd (identical definition to).'
        ' RDDLGN has no attention matrix: the encoder compresses the whole'
        ' source into a single fixed-width recurrent state (K-layer output),'
        ' which every decoder step shares. Rows are normalized to sum to 1.')

    artifacts: list[str] = []
    panel = os.path.join(out_dir, 'panel_sensitivity.png')
    save_panel(panel, pick3(sens_frames),
               f'{args.task} / rddlgn: gradient sensitivity over training',
               sens_caption, row_labels=row_labels, col_labels=col_labels)
    artifacts.append(panel)

    gif_path = save_gif(os.path.join(out_dir, 'sensitivity.gif'), sens_pngs)
    if gif_path:
        artifacts.append(gif_path)

    # ---- 収束判定(1b と同じ事前宣言のしきい値。緩めない) ----
    final = history[-1]
    verdicts = {
        'sens': bool(final['sens']['argmax_acc'] >= CONVERGE_ARGMAX_ACC
                     and final['sens']['corr_target'] >= CONVERGE_CORR)
    }

    # 収束 step(基準を初めて満たした step。未達なら None)
    converged_step = None
    for rec in history:
        if (rec['sens']['argmax_acc'] >= CONVERGE_ARGMAX_ACC
                and rec['sens']['corr_target'] >= CONVERGE_CORR):
            converged_step = rec['step']
            break

    summary = {
        'task': args.task,
        'arm': arm,
        'model': 'rddlgn',
        'gates': gates,
        'seq_len': config.seq_len,
        'vocab_size': config.vocab_size,
        'num_iterations': args.num_iterations,
        'device': args.device,
        'acc_soft': acc_soft,
        'acc_hard': acc_hard,
        'converge_criteria': {'argmax_acc': CONVERGE_ARGMAX_ACC,
                              'corr_target': CONVERGE_CORR},
        'converged': verdicts,
        'converged_step': converged_step,
        'history': history,
        'valid_rows': valid_rows,
        'figures': [os.path.abspath(p) for p in artifacts],
    }
    json_path = os.path.join(out_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    artifacts.append(json_path)
    print(f'\n収束判定 (argmax_acc>={CONVERGE_ARGMAX_ACC} かつ '
          f'corr_target>={CONVERGE_CORR}): {verdicts} '
          f'(初到達step={converged_step})')
    print(f'metrics: {os.path.abspath(json_path)}')

    if use_wandb:
        log_metrics({
            'align/final_corr_target': final['sens']['corr_target'],
            'align/final_argmax_acc': final['sens']['argmax_acc'],
            'acc/soft': acc_soft, 'acc/hard': acc_hard,
        }, step=args.num_iterations)
        log_artifact(artifacts, f'tsmix1c-align-{tag}', type='result',
                     aliases=['latest'],
                     metadata={k: v for k, v in summary.items()
                               if k != 'history'})
        wandb_finish()


if __name__ == '__main__':
    main()
