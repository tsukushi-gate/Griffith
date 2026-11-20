"""(N2): fan-in2転写初期化 → native DLGN 全面fine-tune + widen(MNIST)。

主張検証: 事前学習MLPをfan-in2まで枝刈り→STE fine-tune→n_inputs=2で厳密転写
して得た「teacher由来のnative DLGN IR」を softmax16g で学習可能に復元し、
少データ(6k)で全面fine-tuneすると、同一トポロジーのランダム真理値表初期化より
序盤の収束が速い/最終精度が高いか。効いていれば widen(f∈{4,16})で容量を足す。

パイプライン(タスク仕様):
  1. teacher-float ckpt → apply_fanin_pruning(fanins=(32,12,6,3,2))+STE fine-tune
     (convert_prune_transcribe_mnist.py の機構を再利用)。fan-in2 STE hard acc を記録
     (kill: <0.85 → fan-in3で続行して両方報告)。
  2. TranscribeSpec(n_inputs=2)で厳密転写→IR(validate+simulate照合)。
     estimate_resources でゲート数記録。
  3. rebuild(softmax16g)→全面fine-tune。τ上書き{2.83,10}×損失{CE,KD}の4構成。
     KD = make_kd_loss(alpha=0.5,T=4,standardize=True)+PackedDistillDataset。
  4. 対照: 同一トポロジー(同じindices・同じ幅)でランダム真理値表初期化、同一設定。
     teacher初期化が序盤5k stepで対照を上回らなければ widen中止。
  5. widen: widen_native_ir(f∈{4,16})→rebuild→ベスト設定でfine-tune。
  6. steps_to(0.90)/最終acc/ゲート数を集計。

基線(scratch quick30、6kゲート): steps_to(0.90)中央値 3744 / 最終 0.9088。
効率比 = 3744 / arm の steps_to(0.90)。

実行(GPU機D推奨、recover系はMPSハングのためcuda):
    set -a && . ./.env && set +a && \
    uv run python experiments/convert_native_init_mnist.py \
        --float-ckpt checkpoints/teacher_mlp_float_mnist_20260703_022747/best.pt \
        --device cuda

    # 煙実行(少step・少データ・grid縮約、wandb無効)
    uv run python experiments/convert_native_init_mnist.py --quick --no-wandb
"""

import argparse
import copy
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# 既存' パイプラインの機構を再利用(import、無改造)
from convert_prune_transcribe_mnist import (
    _build_ste_from_float,
    _find_latest_float_ckpt,
    _flatten_binary,
    run_arm,
    transcribe_and_eval,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.prune import PruneSchedule, max_fanin
from griffith_logic.convert.teacher import resolve_teacher
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.convert.widen import widen_native_ir
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.export.estimate import estimate_resources
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir.build import build_torch
from griffith_logic.ir.validate import IRValidationError, validate

load_dotenv()

# scratch quick30 基線(6kゲート)
SCRATCH_STEPS_TO_090 = 3744
SCRATCH_FINAL_ACC = 0.9088
GATE_ACC_TARGET = 0.90       # steps_to のしきい値(Tier-S)
KILL_FANIN2_STE = 0.85       # fan-in2 STE hard acc がこれ未満なら fan-in3 で続行


# ============================================================
# native DLGN 復元(softmax16g)+ 全面 fine-tune
# ============================================================

def rebuild_softmax16g(graph):
    """転写IRを softmax16g(N=2で全16ゲート表現)で学習可能DLGNへ復元する。

    convert/recover.py::rebuild_trainable は softmax16g を許可しない
    (RECOVER_PARAMETERIZATIONS=independent/raw/warp)ため、本タスク用に
    同等処理をローカルで行う。LutLayer は parameterization='softmax' かつ
    n_inputs=2 のとき内部で16ゲートモード(_use_16gate)になり、
    restore_lut_layer が softmax16g で真理値表を厳密復元する(ビルド直後の
    hard forward が元IRの simulate とビット一致することは検証済み)。
    """
    g = copy.deepcopy(graph)
    for node in g.nodes:
        if node.type == 'lut_layer':
            node.train_info = {
                **node.train_info,
                'parameterization': 'softmax',  # N=2 → 16ゲートモード
                'initialized': True,
            }
    return build_torch(g)


def rebuild_random_tt(graph, seed):
    """同一トポロジー(indices・幅)でランダム真理値表初期化した対照モデル。

    転写IRの各lut_layerの truth_tables をランダム値に差し替えて rebuild する。
    隠れ層は 16 ゲート(0..15)一様、最終層(クラス群)の tt も一様
    (縮退LUTだが softmax16g で表現可能)。indices と幅は teacher と同一。
    """
    from griffith_logic.ir.serialize import tt_to_hex
    g = copy.deepcopy(graph)
    rng = np.random.default_rng(seed)
    for node in g.nodes:
        if node.type == 'lut_layer':
            w = int(node.params['out_width'])
            node.params['truth_tables'] = tt_to_hex(
                [int(rng.integers(0, 16)) for _ in range(w)]
            )
            node.train_info = {
                **node.train_info,
                'parameterization': 'softmax',
                'initialized': True,
            }
    return build_torch(g)


def _set_tau(model, tau):
    for m in model.modules():
        if isinstance(m, GroupSum):
            m.tau = float(tau)


@torch.no_grad()
def _hard_acc_ir_from_model(model, x, y, in_width, batch=4000):
    """soft学習中のモデルを export せず、simulate相当の hard eval を近似する。

    ここでは学習曲線の hard acc を素早く測るため、モデルの hard forward
    (LutLayer は eval() で argmax/離散化 hard 経路)を使う。GroupSum は
    argmax がτに不変なので tau上書きは精度に影響しない。
    """
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32,
                             device=next(model.parameters()).device)
        pred = model(xb).argmax(-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    if was_training:
        model.train()
    return correct / len(x)


def finetune_native(
    model, x_tr, y_tr, x_te, y_te, *, device, tau, loss_kind,
    teacher_logits=None, lr=0.01, batch_size=128, max_steps=14000,
    eval_every=500, arm_tag='', use_wandb=False, log_prefix='',
):
    """native DLGN(softmax16g)を全面fine-tuneし、500step毎に hard acc を記録する。

    Returns:
        {'steps_to_090': int|None, 'final_hard_acc': float,
         'curve': [(step, hard_acc)...]}
    """
    _set_tau(model, tau)
    model = model.to(device).train()
    in_width = x_tr.shape[1]
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    tl = (torch.as_tensor(teacher_logits, dtype=torch.float32)
          if teacher_logits is not None else None)
    n = len(x_tr)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    kd = (make_kd_loss(alpha=0.5, temperature=4.0, standardize=True, kind='kl')
          if loss_kind == 'kd' else None)

    step = 0
    curve = []
    steps_to_090 = None
    best_acc = 0.0
    t0 = time.time()
    stop = False
    while not stop:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            logits = model(xb)
            if kd is not None:
                tb = tl[idx].to(device)
                loss = kd(logits, yb, tb)
            else:
                loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0:
                acc = _hard_acc_ir_from_model(model, x_te, y_te, in_width)
                curve.append((step, acc))
                best_acc = max(best_acc, acc)
                if steps_to_090 is None and acc >= GATE_ACC_TARGET:
                    steps_to_090 = step
                if use_wandb:
                    log_metrics({
                        f'{log_prefix}hard_acc': acc,
                        f'{log_prefix}train_loss': float(loss.item()),
                    }, step=step)
                if step >= max_steps:
                    stop = True
                    break
        if step >= max_steps:
            stop = True
    final = _hard_acc_ir_from_model(model, x_te, y_te, in_width)
    best_acc = max(best_acc, final)
    dt = time.time() - t0
    # 序盤(~5k step)hard acc: 曲線から step<=5000 の最大を採る(対照判定用)
    early = [a for (s, a) in curve if s <= 5000]
    acc_at_5k = max(early) if early else best_acc
    return {
        'steps_to_090': steps_to_090,
        'final_hard_acc': final,
        'best_hard_acc': best_acc,
        'acc_at_5k': acc_at_5k,
        'curve': curve,
        'seconds': round(dt, 1),
        'model': model,
    }


# ============================================================
# 転写(fan-in2)+ ゲート数計測
# ============================================================

def _circuit_gate_count(ir):
    est = estimate_resources(ir)
    return est['total_luts']


def prune_and_transcribe(args, x_tr, y_tr, x_te, y_te, calib_x, device,
                         prune_fanin, transcribe_ninputs=None):
    """arm A(teacher初期化)を fanins=(...,prune_fanin) で枝刈り→STE→厳密転写。

    transcribe_ninputs は転写LUTのN(2/4/6のみ、Noneならprune_faninを使う)。
    native DLGN(softmax16g)はN=2専用なので、fan-in2 kill時の fan-in3 続行は
    prune_fanin=3 かつ transcribe_ninputs=4(fan-in≤4を厳密転写)で診断報告する。

    Returns:
        (transcribe_ir, {'ste_hard_acc':.., 'transcribe_hard_acc':.., 'gates':..,
                         'n_inputs':.., 'native_ok':..})
    """
    n_inputs = transcribe_ninputs if transcribe_ninputs is not None else prune_fanin
    if n_inputs not in (2, 4, 6):
        raise ValueError(f'transcribe n_inputsは2/4/6のみ: {n_inputs}')
    fanins = tuple(f for f in args.fanins if f >= prune_fanin)
    if not fanins or fanins[-1] != prune_fanin:
        fanins = (*fanins, prune_fanin)
    sched = PruneSchedule(fanins=fanins, epochs_per_stage=args.epochs_per_stage,
                          lr=args.prune_lr)
    model = _build_ste_from_float(args.float_ckpt, device)

    class _A:
        pass
    a = _A()
    a._arm_wandb = False
    a.batch_size = args.batch_size
    hist = run_arm('A', model, sched, x_tr, y_tr, x_te, y_te, a, device)
    ste_acc = hist['final_acc']
    print(f'  fan-in{prune_fanin} STE hard acc = {ste_acc:.4f} '
          f'(max_fanin={max_fanin(model)})')

    tr = transcribe_and_eval('A', model, calib_x, x_te, y_te, n_inputs,
                             args.weight_bits, device)
    # 転写IR自体を取り出す(transcribe_and_eval は metrics のみ返すため再転写)
    model_cpu = model.to('cpu').eval()
    ir = transcribe_mlp(model_cpu, calib_x,
                        TranscribeSpec(n_inputs=n_inputs,
                                       weight_bits=args.weight_bits, isf=False))
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'転写IR検証失敗: {errs[:3]}')
    model.to(device)
    gates = _circuit_gate_count(ir)
    # native(softmax16g)復元は隠れ層LUTがN=2のときのみ有効
    native_ok = n_inputs == 2
    print(f'  転写IR: gates={gates} n_inputs={n_inputs} '
          f'transcribe_hard_acc={tr["hard_acc"]:.4f} native_ok={native_ok}')
    return ir, {
        'ste_hard_acc': ste_acc,
        'transcribe_hard_acc': tr['hard_acc'],
        'gates': gates,
        'fanins': list(fanins),
        'n_inputs': n_inputs,
        'native_ok': native_ok,
    }


# ============================================================
# teacher logits(KD用)
# ============================================================

@torch.no_grad()
def _teacher_logits(ckpt_path, x, device, batch=4000):
    teacher = resolve_teacher(f'ckpt:{ckpt_path}', device=device)
    m = teacher.model.to(device).eval()
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        outs.append(m(xb).cpu())
    return torch.cat(outs, dim=0).numpy()


# ============================================================
# メイン: grid 実行
# ============================================================

def _eff_ratio(steps_to):
    if steps_to is None or steps_to <= 0:
        return None
    return round(SCRATCH_STEPS_TO_090 / steps_to, 3)


def run_finetune_grid(ir, tag, args, x_tr, y_tr, x_te, y_te, device,
                      teacher_logits, gates, seed, *, init='teacher'):
    """1つのIRに対し τ×損失の4構成(または縮約)で fine-tune し結果表行を返す。"""
    rows = []
    taus = args.taus
    losses = args.losses
    for tau in taus:
        for loss_kind in losses:
            run_name = f'd5_n2_{tag}_{init}_{loss_kind}_tau{tau}'
            use_wandb = init_wandb(
                run_name=run_name,
                config_dict={
                    'task': 'D5-3', 'arm': 'N2', 'init': init, 'tag': tag,
                    'loss': loss_kind, 'tau': tau, 'gates': gates,
                    'lr': args.ft_lr, 'batch_size': args.batch_size,
                    'max_steps': args.max_steps, 'train_n': len(x_tr),
                    'seed': seed, 'dataset': 'mnist-binarized',
                    'scratch_steps_to_090': SCRATCH_STEPS_TO_090,
                },
                device=device, tags=['D5-3', 'N2'],
                group='convert-mnist-n2',
            )
            if init == 'teacher':
                model = rebuild_softmax16g(ir)
            else:
                model = rebuild_random_tt(ir, seed=seed)
            res = finetune_native(
                model, x_tr, y_tr, x_te, y_te, device=device, tau=tau,
                loss_kind=loss_kind, teacher_logits=teacher_logits,
                lr=args.ft_lr, batch_size=args.batch_size,
                max_steps=args.max_steps, eval_every=args.eval_every,
                use_wandb=use_wandb, log_prefix='',
            )
            eff = _eff_ratio(res['steps_to_090'])
            run_id = os.environ.get('WANDB_RUN_ID', '')
            try:
                import wandb
                run_id = wandb.run.id if wandb.run is not None else run_id
            except Exception:
                pass
            if use_wandb:
                log_metrics({
                    'summary/steps_to_090': res['steps_to_090'] or -1,
                    'summary/final_hard_acc': res['final_hard_acc'],
                    'summary/best_hard_acc': res['best_hard_acc'],
                    'summary/eff_ratio': eff or 0.0,
                    'summary/gates': gates,
                    'summary/acc_at_5k': res['acc_at_5k'],
                })
                wandb_finish()
            row = {
                'tag': tag, 'init': init, 'loss': loss_kind, 'tau': tau,
                'steps_to_090': res['steps_to_090'],
                'eff_ratio': eff, 'final_acc': res['final_hard_acc'],
                'best_acc': res['best_hard_acc'],
                'acc_at_5k': res['acc_at_5k'], 'gates': gates,
                'run_id': run_id, 'seconds': res['seconds'],
            }
            rows.append(row)
            print(f'  [{run_name}] steps_to(0.90)={res["steps_to_090"]} '
                  f'eff={eff} final={res["final_hard_acc"]:.4f} '
                  f'best={res["best_hard_acc"]:.4f} gates={gates} '
                  f'({res["seconds"]}s) run={run_id}')
    return rows


def _print_table(rows, title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)
    print('| tag | init | loss | tau | steps_to(0.90) | eff比 | 最終acc | '
          'best acc | acc@5k | gates | WANDB_RUN_ID |')
    print('|' + '---|' * 11)
    for r in rows:
        print(f"| {r['tag']} | {r['init']} | {r['loss']} | {r['tau']} | "
              f"{r['steps_to_090']} | {r['eff_ratio']} | {r['final_acc']:.4f} | "
              f"{r['best_acc']:.4f} | {r['acc_at_5k']:.4f} | {r['gates']} | "
              f"{r['run_id']} |")


def _best_row(rows):
    """steps_to(0.90) 最小(未達は最後)を勝ち構成として返す。"""
    reached = [r for r in rows if r['steps_to_090'] is not None]
    if reached:
        return min(reached, key=lambda r: r['steps_to_090'])
    return max(rows, key=lambda r: r['best_acc'])


def main():
    p = argparse.ArgumentParser(description='fan-in2転写初期化+widen(MNIST)')
    p.add_argument('--float-ckpt', type=str, default=None)
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p.add_argument('--fanins', type=int, nargs='+', default=[32, 12, 6, 3, 2])
    p.add_argument('--epochs-per-stage', type=int, default=2)
    p.add_argument('--prune-lr', type=float, default=1e-3)
    p.add_argument('--weight-bits', type=int, default=2,
                   help='最終層量子化ビット幅。native DLGNは≤24kゲート予算のため'
                        '低bit(既定2で総ゲート~1k、8だと最終層の縮退展開で~74k)')
    p.add_argument('--ft-lr', type=float, default=0.01)
    p.add_argument('-bs', '--batch-size', type=int, default=128)
    p.add_argument('--taus', type=float, nargs='+', default=[2.83, 10.0])
    p.add_argument('--losses', type=str, nargs='+', default=['ce', 'kd'])
    p.add_argument('--train-n', type=int, default=6000,
                   help='少データfine-tune件数(基線6k)')
    p.add_argument('--calib-n', type=int, default=10000)
    p.add_argument('--eval-n', type=int, default=10000)
    p.add_argument('--max-steps', type=int, default=14000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--widen-factors', type=int, nargs='+', default=[4, 16])
    p.add_argument('--jitter-frac', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--skip-widen', action='store_true')
    args = p.parse_args()

    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    device = args.device or (
        'cuda' if torch.cuda.is_available()
        else ('mps' if getattr(torch.backends, 'mps', None)
              and torch.backends.mps.is_available() else 'cpu'))

    if args.float_ckpt is None:
        args.float_ckpt = _find_latest_float_ckpt(args.checkpoint_dir)
        if args.float_ckpt is None:
            print('!! teacher-float ckpt が見つからない(--float-ckpt を指定)')
            sys.exit(1)

    if args.quick:
        args.fanins = [6, 3, 2]
        args.epochs_per_stage = 1
        args.taus = [10.0]
        args.losses = ['ce']
        args.train_n = 2000
        args.calib_n = 2000
        args.eval_n = 2000
        args.max_steps = 1500
        args.widen_factors = [4]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'デバイス: {device}  teacher-float: {args.float_ckpt}')
    print(f'grid: taus={args.taus} losses={args.losses} '
          f'train_n={args.train_n} max_steps={args.max_steps}')

    print('MNIST(binarize 0.5)読み込み中...')
    train_set, test_set = load_mnist(binarize_threshold=0.5)
    x_tr_all, y_tr_all = _flatten_binary(train_set,
                                         n=max(args.calib_n, args.train_n, 20000))
    x_te, y_te = _flatten_binary(test_set, n=args.eval_n)
    calib_x = x_tr_all[:args.calib_n]
    x_ft, y_ft = x_tr_all[:args.train_n], y_tr_all[:args.train_n]
    print(f'train_all={len(x_tr_all)} calib={len(calib_x)} '
          f'ft={len(x_ft)} test={len(x_te)}')

    started = time.time()

    # --- Step1-2: fan-in2 枝刈り→転写(kill判定つき、破損時fan-in3を診断報告)---
    # native DLGN(softmax16g)は N=2 専用。よって fine-tune/widen は常に
    # fan-in2 転写IR(native_ok)で行う。kill時は fan-in3(prune)→n_inputs=4
    # 転写の hard acc を「fan-in容量を1本増やせば回復するか」の診断として併記する。
    print('\n[Step1-2] fan-in2 枝刈り→STE→厳密転写')
    ir2, info2 = prune_and_transcribe(args, x_tr_all, y_tr_all, x_te, y_te,
                                      calib_x, device, prune_fanin=2)
    fanin_used = 2
    active_ir, active_info = ir2, info2
    ir3 = info3 = None
    if info2['ste_hard_acc'] < KILL_FANIN2_STE:
        print(f'!! kill発動: fan-in2 STE acc={info2["ste_hard_acc"]:.4f} '
              f'< {KILL_FANIN2_STE} → fan-in3(n_inputs=4転写)を診断報告。'
              f'native fine-tune/widen は fan-in2 IR で継続(N=2専用のため)')
        ir3, info3 = prune_and_transcribe(args, x_tr_all, y_tr_all, x_te, y_te,
                                          calib_x, device, prune_fanin=3,
                                          transcribe_ninputs=4)

    # --- teacher logits(KD用、fine-tune入力=x_ft と整列)---
    teacher_logits = None
    if 'kd' in args.losses:
        print('teacher logits 計算(KD用)...')
        teacher_logits = _teacher_logits(args.float_ckpt, x_ft, device)

    # --- Step3-4: base(f=1)で teacher vs random 対照 fine-tune ---
    print(f'\n[Step3-4] base fine-tune (fan-in{fanin_used}, gates='
          f'{active_info["gates"]}) teacher vs random 対照')
    tag_base = f'fin{fanin_used}_f1'
    rows_teacher = run_finetune_grid(
        active_ir, tag_base, args, x_ft, y_ft, x_te, y_te, device,
        teacher_logits, active_info['gates'], args.seed, init='teacher')
    rows_random = run_finetune_grid(
        active_ir, tag_base, args, x_ft, y_ft, x_te, y_te, device,
        teacher_logits, active_info['gates'], args.seed, init='random')

    all_rows = rows_teacher + rows_random

    # 対照判定: 序盤(5k step)hard acc で teacher vs random を比較する。
    # 各群の最良構成(steps_to最小 or best_acc最大)の acc_at_5k を代表値にする。
    t_best = _best_row(rows_teacher)
    r_best = _best_row(rows_random)
    t_early = max(r['acc_at_5k'] for r in rows_teacher)
    r_early = max(r['acc_at_5k'] for r in rows_random)
    teacher_wins = t_early >= r_early + 0.005 or (
        t_best['steps_to_090'] is not None and (
            r_best['steps_to_090'] is None
            or t_best['steps_to_090'] <= r_best['steps_to_090']))
    print(f'\n--- 対照判定(序盤5k step hard acc): '
          f'teacher={t_early:.4f} vs random={r_early:.4f} → '
          f'{"teacher優位=widen続行" if teacher_wins else "teacher非優位=widen中止"} '
          f'(teacher最良steps_to={t_best["steps_to_090"]} '
          f'random最良steps_to={r_best["steps_to_090"]})')

    # --- Step5: widen(ベスト設定 × f∈{4,16})---
    widen_rows = []
    if not args.skip_widen and teacher_wins:
        best = t_best
        print(f'\n[Step5] widen(ベスト設定: loss={best["loss"]} '
              f'tau={best["tau"]}) f∈{args.widen_factors}')
        for f in args.widen_factors:
            wir = widen_native_ir(active_ir, f, seed=args.seed,
                                  jitter_frac=args.jitter_frac,
                                  init_new_final='const0')
            errs = validate(wir)
            if errs:
                raise IRValidationError(f'widen f={f} 検証失敗: {errs[:3]}')
            wgates = _circuit_gate_count(wir)
            print(f'  widen f={f}: gates={wgates} (base={active_info["gates"]})')
            # ベスト設定1点のみ(taus/lossesをbestに固定)
            sub = argparse.Namespace(**vars(args))
            sub.taus = [best['tau']]
            sub.losses = [best['loss']]
            rows = run_finetune_grid(
                wir, f'fin{fanin_used}_f{f}', sub, x_ft, y_ft, x_te, y_te,
                device, teacher_logits, wgates, args.seed, init='teacher')
            widen_rows += rows
        all_rows += widen_rows
    elif args.skip_widen:
        print('\n[Step5] widen スキップ(--skip-widen)')
    else:
        print('\n[Step5] widen 中止(teacher非優位)')

    # --- Step6: 集計 ---
    print('\n' + '#' * 78)
    print('(N2) 最終集計')
    print('#' * 78)
    print(f'\nfan-in2 STE hard acc = {info2["ste_hard_acc"]:.4f} '
          f'(kill基準 {KILL_FANIN2_STE}: '
          f'{"PASS" if info2["ste_hard_acc"] >= KILL_FANIN2_STE else "FAIL→fan-in3続行"})')
    print(f'fan-in2 転写IR: gates={info2["gates"]} '
          f'transcribe_hard_acc={info2["transcribe_hard_acc"]:.4f}')
    if info3 is not None:
        print(f'fan-in3 STE hard acc = {info3["ste_hard_acc"]:.4f} '
              f'転写IR gates={info3["gates"]} '
              f'transcribe_hard_acc={info3["transcribe_hard_acc"]:.4f}')
    print(f'\n基線(scratch quick30, 6kゲート): '
          f'steps_to(0.90)={SCRATCH_STEPS_TO_090} 最終={SCRATCH_FINAL_ACC}')

    _print_table(all_rows, '全run 結果表(効率比 = 3744 / steps_to(0.90))')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'\n完了(総時間 {total})')


if __name__ == '__main__':
    main()
