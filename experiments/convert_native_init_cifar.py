"""CIFAR規模の初期値転写対照 — 転移initはランダムinitより少epochで高精度か。

MNISTでは同一トポロジーで teacher-init 0.595 vs random-init 0.218(@5k step)と
決定的だったが、CIFAR規模では未検証。本スクリプトで直接対照する。

CIFARの teacher は resnet18(CNN)で transcribe_mlp(Linear限定)に直接転写不能。
事前学習知識を初期値へ運ぶ連鎖:

    HF resnet18(0.9498) ──KD──▶ MLPプロキシ(binarized CIFAR入力, 9216-2000-500-10)
                                  │ 枝刈り(fanins ...,2)+STE ft
                                  ▼ TranscribeSpec(n_inputs=2)厳密転写
                               N=2 native DLGN 初期値(IR)
                                  │ widen(f=4, convert/widen.py)
                                  ▼
                  arm A: 転写init ─┐ 同一トポロジー・同一学習
                  arm B: random init ┘ (KD α=0.25/T=4/standardize, resnetがteacher)

再利用: experiments/convert_native_init_mnist.py と convert_prune_transcribe_mnist.py の
機構を import する(無改造)。CIFAR固有の差分のみここに書く:
    - 入力 9216 次元(binarized CIFAR, cifar-10-3-thresholds)
    - MLPプロキシ(TeacherMLP)を KD で本スクリプト内で学習(CIFAR teacher-float ckptが無い)
    - teacher = HF resnet18(hf:edadaltocg/resnet18_cifar10, transform=cifar10-resnet18-hf)
      ロジットは data-teacher-cache のインデックス整列キャッシュ(fingerprintで自動)
    - STE builder / random-tt はCIFAR次元用にローカル定義(MNIST版はdims固定のため)

実行(GPU マシン、cuda。recover系はMPSハングのためcuda厳守):
    set -a && . ./.env && set +a && \
    uv run python experiments/convert_native_init_cifar.py --device cuda

    # 煙実行(少step・少データ・grid縮約、wandb無効)
    uv run python experiments/convert_native_init_cifar.py --quick --no-wandb
"""

import argparse
import copy
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# 既存パイプラインの機構を再利用(import、無改造)
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
from convert_native_init_mnist import (
    finetune_native,
    rebuild_random_tt,
    rebuild_softmax16g,
)
from convert_prune_transcribe_mnist import (
    run_arm,
    transcribe_and_eval,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.cache import get_or_compute_logits, teacher_cache_key
from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.convert.prune import PruneSchedule, max_fanin
from griffith_logic.convert.teacher import (
    TEACHER_TRANSFORMS,
    resolve_teacher,
)
from griffith_logic.convert.teacher_models import TeacherMLP, TeacherMLPSte
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.convert.widen import widen_native_ir
from griffith_logic.datasets.vision import load_cifar10, load_raw
from griffith_logic.export.estimate import estimate_resources
from griffith_logic.ir.validate import IRValidationError, validate

load_dotenv()

CIFAR_IN_WIDTH = 3 * 32 * 32 * 3  # 9216 (3ch × 32×32 × 3 thresholds)
NUM_CLASSES = 10
TEACHER_SOURCE = 'hf:edadaltocg/resnet18_cifar10'
TEACHER_TRANSFORM = 'cifar10-resnet18-hf'
KILL_FANIN2_STE = 0.30  # CIFAR規模。fan-in2 STE hard acc がこれ未満なら fan-in3診断


# ============================================================
# データ(binarized CIFAR-10, index整列)
# ============================================================

def _flatten_binary_cifar(dataset, n=None):
    """二値化済み CIFAR-10 を (N,9216) int64 0/1 と (N,) ラベルへ展開する。"""
    total = len(dataset) if n is None else min(n, len(dataset))
    xs = np.empty((total, CIFAR_IN_WIDTH), dtype=np.int64)
    ys = np.empty((total,), dtype=np.int64)
    for i in range(total):
        img, label = dataset[i]
        xs[i] = img.reshape(-1).numpy().astype(np.int64)
        ys[i] = int(label)
    uniq = set(np.unique(xs[: min(total, 2000)]).tolist())
    assert uniq.issubset({0, 1}), f'二値化CIFARのはずが0/1以外: {uniq}'
    return xs, ys


# ============================================================
# teacher(resnet18)ロジット — data-teacher-cache のindex整列キャッシュ
# ============================================================

def _teacher_train_logits(device, n_raw):
    """raw CIFAR train(未二値化 [0,1])+ resnet18 transform で train全件ロジット。

    runner._apply_distill と同一のキー・整列規約(cifar-10-3-thresholds、
    transform=cifar10-resnet18-hf、split=train)。data-teacher-cache に既存なら
    fingerprint一致でロードのみ(GPU機Dにキャッシュ済みの見込み)。返り値は
    train全件 [N,10](CPU)。binarized student と index整列している。
    """
    teacher = resolve_teacher(TEACHER_SOURCE, device=device)
    teacher.transform = TEACHER_TRANSFORMS[TEACHER_TRANSFORM]
    raw_train, _ = load_raw('cifar-10-3-thresholds', download=True)
    key = teacher_cache_key(
        teacher.fingerprint, 'cifar-10-3-thresholds',
        {'transform': TEACHER_TRANSFORM}, split='train', n=len(raw_train),
    )
    logits = get_or_compute_logits(teacher, raw_train, key=key, device=device)
    return logits[:n_raw].numpy()


@torch.no_grad()
def _teacher_test_acc(device):
    """resnet18 の test acc を実測(報告用の健全性チェック)。"""
    teacher = resolve_teacher(TEACHER_SOURCE, device=device)
    teacher.transform = TEACHER_TRANSFORMS[TEACHER_TRANSFORM]
    _, raw_test = load_raw('cifar-10-3-thresholds', download=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(raw_test, batch_size=512, shuffle=False)
    correct = tot = 0
    for x, y in loader:
        pred = teacher.predict_logits(x).argmax(-1).cpu()
        correct += int((pred == y).sum())
        tot += len(y)
    return correct / tot


# ============================================================
# Step1: MLPプロキシ学習(KD、binarized CIFAR入力)
# ============================================================

def _mlp_hidden(args):
    return tuple(args.mlp_hidden)


@torch.no_grad()
def _mlp_hard_acc(model, x, y, device, batch=4000):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        pred = model(xb).argmax(-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def train_mlp_proxy(args, x_tr, y_tr, x_te, y_te, teacher_logits_tr, device):
    """binarized CIFAR上で TeacherMLP(9216-...-10)を KD で学習する。

    teacher_logits_tr は x_tr と index整列した resnet18 ロジット。
    KD = make_kd_loss(alpha, T, standardize)。返り値は (model, best_acc)。
    """
    hidden = _mlp_hidden(args)
    model = TeacherMLP(in_width=CIFAR_IN_WIDTH, hidden=hidden,
                       num_classes=NUM_CLASSES).to(device)
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    tl = torch.as_tensor(teacher_logits_tr, dtype=torch.float32)
    n = len(x_tr)
    opt = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
    kd = make_kd_loss(alpha=args.mlp_alpha, temperature=args.mlp_temp,
                      standardize=True, kind='kl')
    best = 0.0
    for ep in range(args.mlp_epochs):
        model.train()
        perm = torch.randperm(n)
        run_loss = nb = 0
        for i in range(0, n, args.mlp_bs):
            idx = perm[i:i + args.mlp_bs]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            tb = tl[idx].to(device)
            loss = kd(model(xb), yb, tb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.item())
            nb += 1
        acc = _mlp_hard_acc(model, x_te, y_te, device)
        best = max(best, acc)
        print(f'  [MLP proxy] ep{ep + 1}/{args.mlp_epochs} '
              f'loss={run_loss / max(nb, 1):.4f} test_acc={acc:.4f}')
    return model, best


def _save_proxy_ckpt(model, hidden, best_acc):
    """MLPプロキシを ckpt:形式で保存(resolve_teacher('ckpt:...')でロード可能)。

    metadata に model_class=TeacherMLP, model_kwargs(in_width/hidden/num_classes)を
    埋め、_build_ste_from_float_cifar が resolve_teacher で再構築できるようにする。
    """
    run_dir, _, best_path = setup_checkpoint_dir('teacher_mlp_proxy_cifar')
    save_checkpoint(
        best_path, model, torch.optim.Adam(model.parameters()), None,
        step=0, best_metric=best_acc,
        metadata={
            'model_class': TeacherMLP,
            'model_kwargs': {'in_width': CIFAR_IN_WIDTH,
                             'hidden': hidden, 'num_classes': NUM_CLASSES},
            'num_classes': NUM_CLASSES,
            'input_size': (CIFAR_IN_WIDTH,),
            'dataset': 'cifar-10-3-thresholds-binarized',
        },
    )
    print(f'  MLPプロキシ保存: {best_path}')
    return best_path


# ============================================================
# Step2: 枝刈り→STE→転写(CIFAR次元用STE builder)
# ============================================================

def _build_ste_from_proxy(ckpt_path, hidden, device):
    """MLPプロキシ ckpt の Linear 重みで TeacherMLPSte(CIFAR次元)を初期化する。

    convert_prune_transcribe_mnist._build_ste_from_float の CIFAR版
    (MNIST版は dims が 784-300-100 固定のため流用不可)。TeacherMLP と
    TeacherMLPSte は同形の Linear 列を持つので、順にコピーする。
    """
    teacher = resolve_teacher(f'ckpt:{ckpt_path}', device='cpu')
    src = teacher.model  # TeacherMLP
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    ste = TeacherMLPSte(in_width=CIFAR_IN_WIDTH, hidden=hidden,
                        num_classes=NUM_CLASSES)
    dst_linears = list(ste.linears)
    if len(src_linears) != len(dst_linears):
        raise ValueError(
            f'Linear層数不一致: proxy={len(src_linears)} ste={len(dst_linears)}')
    with torch.no_grad():
        for s, d in zip(src_linears, dst_linears, strict=True):
            if s.weight.shape != d.weight.shape:
                raise ValueError(
                    f'Linear形状不一致: {s.weight.shape} vs {d.weight.shape}')
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
    return ste.to(device)


def _circuit_gate_count(ir):
    return estimate_resources(ir)['total_luts']


def prune_and_transcribe(args, ckpt_path, hidden, x_tr, y_tr, x_te, y_te,
                         calib_x, device, prune_fanin, transcribe_ninputs=None):
    """MLPプロキシを fanins=(...,prune_fanin) で枝刈り→STE→n_inputs厳密転写。

    convert_prune_transcribe_mnist の run_arm / transcribe_and_eval を再利用
    (どちらも次元非依存に Linear 列を走査する)。
    """
    n_inputs = transcribe_ninputs if transcribe_ninputs is not None else prune_fanin
    if n_inputs not in (2, 4, 6):
        raise ValueError(f'transcribe n_inputsは2/4/6のみ: {n_inputs}')
    fanins = tuple(f for f in args.fanins if f >= prune_fanin)
    if not fanins or fanins[-1] != prune_fanin:
        fanins = (*fanins, prune_fanin)
    sched = PruneSchedule(fanins=fanins, epochs_per_stage=args.epochs_per_stage,
                          lr=args.prune_lr)
    model = _build_ste_from_proxy(ckpt_path, hidden, device)

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
    model_cpu = model.to('cpu').eval()
    ir = transcribe_mlp(model_cpu, calib_x,
                        TranscribeSpec(n_inputs=n_inputs,
                                       weight_bits=args.weight_bits, isf=False))
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'転写IR検証失敗: {errs[:3]}')
    model.to(device)
    gates = _circuit_gate_count(ir)
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
# Step3-4: 対照 fine-tune(核心)
# ============================================================

def run_finetune_grid(ir, tag, args, x_tr, y_tr, x_te, y_te, device,
                      teacher_logits_ft, gates, seed, *, init='teacher'):
    """1つのIRに対し {損失}×{tau} 構成で fine-tune し結果表行を返す。"""
    rows = []
    for tau in args.taus:
        for loss_kind in args.losses:
            run_name = f'd6_init_cifar_{tag}_{init}_{loss_kind}_tau{tau}'
            use_wandb = init_wandb(
                run_name=run_name,
                config_dict={
                    'task': 'D6-6', 'arm': 'N2', 'init': init, 'tag': tag,
                    'loss': loss_kind, 'tau': tau, 'gates': gates,
                    'lr': args.ft_lr, 'batch_size': args.batch_size,
                    'max_steps': args.max_steps, 'train_n': len(x_tr),
                    'seed': seed, 'dataset': 'cifar-10-3-thresholds-binarized',
                    'teacher': TEACHER_SOURCE,
                    'kd_alpha': args.kd_alpha, 'kd_temp': args.kd_temp,
                },
                device=device, tags=['D6-6', 'N2', 'cifar'],
                group='convert-cifar-n2',
            )
            if init == 'teacher':
                model = rebuild_softmax16g(ir)
            else:
                model = rebuild_random_tt(ir, seed=seed)
            res = finetune_native(
                model, x_tr, y_tr, x_te, y_te, device=device, tau=tau,
                loss_kind=loss_kind, teacher_logits=teacher_logits_ft,
                lr=args.ft_lr, batch_size=args.batch_size,
                max_steps=args.max_steps, eval_every=args.eval_every,
                use_wandb=use_wandb, log_prefix='',
            )
            run_id = os.environ.get('WANDB_RUN_ID', '')
            try:
                import wandb
                run_id = wandb.run.id if wandb.run is not None else run_id
            except Exception:
                pass
            # @15ep相当 step と 最終 step の hard acc を曲線から拾う
            steps_per_epoch = max(1, len(x_tr) // args.batch_size)
            step_15ep = 15 * steps_per_epoch
            acc_15ep = _curve_at(res['curve'], step_15ep)
            if use_wandb:
                log_metrics({
                    'summary/final_hard_acc': res['final_hard_acc'],
                    'summary/best_hard_acc': res['best_hard_acc'],
                    'summary/acc_at_5k': res['acc_at_5k'],
                    'summary/acc_at_15ep': acc_15ep,
                    'summary/steps_to_090': res['steps_to_090'] or -1,
                    'summary/gates': gates,
                })
                wandb_finish()
            row = {
                'tag': tag, 'init': init, 'loss': loss_kind, 'tau': tau,
                'acc_at_5k': res['acc_at_5k'], 'acc_at_15ep': acc_15ep,
                'final_acc': res['final_hard_acc'], 'best_acc': res['best_hard_acc'],
                'gates': gates, 'run_id': run_id, 'seconds': res['seconds'],
                'step_15ep': step_15ep,
            }
            rows.append(row)
            print(f'  [{run_name}] acc@5k={res["acc_at_5k"]:.4f} '
                  f'acc@15ep={acc_15ep:.4f} final={res["final_hard_acc"]:.4f} '
                  f'best={res["best_hard_acc"]:.4f} gates={gates} '
                  f'({res["seconds"]}s) run={run_id}')
    return rows


def _curve_at(curve, target_step):
    """曲線 [(step,acc)...] から step<=target_step の最後(なければ最初)の acc。"""
    if not curve:
        return 0.0
    le = [a for (s, a) in curve if s <= target_step]
    if le:
        return le[-1]
    return curve[0][1]


def _print_table(rows, title):
    print('\n' + '=' * 90)
    print(title)
    print('=' * 90)
    print('| tag | init | loss | tau | acc@5k | acc@15ep | 最終acc | '
          'best acc | gates | WANDB_RUN_ID |')
    print('|' + '---|' * 10)
    for r in rows:
        print(f"| {r['tag']} | {r['init']} | {r['loss']} | {r['tau']} | "
              f"{r['acc_at_5k']:.4f} | {r['acc_at_15ep']:.4f} | "
              f"{r['final_acc']:.4f} | {r['best_acc']:.4f} | {r['gates']} | "
              f"{r['run_id']} |")


def _best_row(rows):
    """acc@5k 最大を代表(序盤収束が本タスクの核心指標)。"""
    return max(rows, key=lambda r: r['acc_at_5k'])


# ============================================================
# メイン
# ============================================================

def main():
    p = argparse.ArgumentParser(description='CIFAR初期値転写対照(N2)')
    # MLPプロキシ
    p.add_argument('--mlp-hidden', type=int, nargs='+', default=[2000, 500])
    p.add_argument('--mlp-epochs', type=int, default=12)
    p.add_argument('--mlp-lr', type=float, default=1e-3)
    p.add_argument('--mlp-bs', type=int, default=256)
    p.add_argument('--mlp-alpha', type=float, default=0.25)
    p.add_argument('--mlp-temp', type=float, default=4.0)
    p.add_argument('--proxy-ckpt', type=str, default=None,
                   help='既存MLPプロキシckpt(指定時は学習をスキップ)')
    # 枝刈り→転写
    p.add_argument('--fanins', type=int, nargs='+', default=[32, 12, 6, 3, 2])
    p.add_argument('--epochs-per-stage', type=int, default=2)
    p.add_argument('--prune-lr', type=float, default=1e-3)
    p.add_argument('--weight-bits', type=int, default=2)
    # 対照 fine-tune
    p.add_argument('--ft-lr', type=float, default=0.01)
    p.add_argument('-bs', '--batch-size', type=int, default=128)
    p.add_argument('--taus', type=float, nargs='+', default=[10.0])
    p.add_argument('--losses', type=str, nargs='+', default=['kd'])
    p.add_argument('--kd-alpha', type=float, default=0.25)
    p.add_argument('--kd-temp', type=float, default=4.0)
    p.add_argument('--train-n', type=int, default=20000)
    p.add_argument('--calib-n', type=int, default=10000)
    p.add_argument('--eval-n', type=int, default=10000)
    p.add_argument('--max-steps', type=int, default=20000,
                   help='50ep相当。eval-everyで曲線記録')
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--widen-factor', type=int, default=4)
    p.add_argument('--jitter-frac', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--skip-widen', action='store_true')
    p.add_argument('--ce-control', action='store_true',
                   help='余裕枠: CE版(KDなし)の2armも実行')
    args = p.parse_args()

    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'mps':
        raise SystemExit('MPSはrecover系ハングのため禁止。cudaを使うこと。')

    if args.quick:
        args.mlp_hidden = [512, 128]
        args.mlp_epochs = 2
        args.fanins = [6, 3, 2]
        args.epochs_per_stage = 1
        args.taus = [10.0]
        args.losses = ['kd']
        args.train_n = 3000
        args.calib_n = 2000
        args.eval_n = 2000
        args.max_steps = 1500
        args.widen_factor = 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    hidden = _mlp_hidden(args)

    print(f'デバイス: {device}  MLP hidden={hidden} '
          f'grid: taus={args.taus} losses={args.losses} '
          f'train_n={args.train_n} max_steps={args.max_steps}')

    # --- teacher 健全性 ---
    print('\n[teacher] resnet18 test acc 実測...')
    teacher_acc = _teacher_test_acc(device)
    print(f'  resnet18 test acc = {teacher_acc:.4f}')

    # --- データ(binarized CIFAR, index整列) ---
    print('\nbinarized CIFAR-10(cifar-10-3-thresholds)読み込み中...')
    train_set, test_set = load_cifar10(threshold_count=3)
    take = max(args.calib_n, args.train_n, 20000)
    x_tr_all, y_tr_all = _flatten_binary_cifar(train_set, n=take)
    x_te, y_te = _flatten_binary_cifar(test_set, n=args.eval_n)
    calib_x = x_tr_all[:args.calib_n]
    x_ft, y_ft = x_tr_all[:args.train_n], y_tr_all[:args.train_n]
    print(f'train_all={len(x_tr_all)} calib={len(calib_x)} '
          f'ft={len(x_ft)} test={len(x_te)} in_width={x_tr_all.shape[1]}')

    # --- teacher train logits(index整列、cache) ---
    print('\n[teacher] train全件ロジット(cache)...')
    teacher_logits_all = _teacher_train_logits(device, len(x_tr_all))
    teacher_logits_ft = teacher_logits_all[:args.train_n]
    print(f'  teacher_logits: {teacher_logits_all.shape}')

    started = time.time()

    # --- Step1: MLPプロキシ学習(KD) ---
    if args.proxy_ckpt:
        proxy_ckpt = args.proxy_ckpt
        proxy_best = float('nan')
        print(f'\n[Step1] MLPプロキシ既存ckpt使用: {proxy_ckpt}')
    else:
        print('\n[Step1] MLPプロキシ KD学習(binarized CIFAR入力)')
        proxy_model, proxy_best = train_mlp_proxy(
            args, x_tr_all[:args.train_n], y_tr_all[:args.train_n],
            x_te, y_te, teacher_logits_all[:args.train_n], device)
        proxy_ckpt = _save_proxy_ckpt(proxy_model, hidden, proxy_best)
    print(f'  MLPプロキシ best test acc = {proxy_best}')

    # --- Step2: 枝刈り→STE→転写(fan-in2) ---
    print('\n[Step2] fan-in2 枝刈り→STE→厳密転写')
    ir2, info2 = prune_and_transcribe(
        args, proxy_ckpt, hidden, x_tr_all, y_tr_all, x_te, y_te,
        calib_x, device, prune_fanin=2)
    active_ir, active_info = ir2, info2
    info3 = None
    if info2['ste_hard_acc'] < KILL_FANIN2_STE:
        print(f'!! fan-in2 STE acc={info2["ste_hard_acc"]:.4f} < {KILL_FANIN2_STE} '
              f'→ fan-in3(n_inputs=4転写)を診断報告。native fine-tune/widenは'
              f'fan-in2 IRで継続(N=2専用のため)')
        _, info3 = prune_and_transcribe(
            args, proxy_ckpt, hidden, x_tr_all, y_tr_all, x_te, y_te,
            calib_x, device, prune_fanin=3, transcribe_ninputs=4)

    # --- Step3-4: base(f=1)で teacher vs random 対照 ---
    print(f'\n[Step3-4] base fine-tune (fan-in2, gates={active_info["gates"]}) '
          f'teacher vs random 対照(KD α={args.kd_alpha}/T={args.kd_temp})')
    tag_base = 'fin2_f1'
    rows_teacher = run_finetune_grid(
        active_ir, tag_base, args, x_ft, y_ft, x_te, y_te, device,
        teacher_logits_ft, active_info['gates'], args.seed, init='teacher')
    rows_random = run_finetune_grid(
        active_ir, tag_base, args, x_ft, y_ft, x_te, y_te, device,
        teacher_logits_ft, active_info['gates'], args.seed, init='random')
    all_rows = rows_teacher + rows_random

    t_best = _best_row(rows_teacher)
    r_best = _best_row(rows_random)
    teacher_wins = t_best['acc_at_5k'] >= r_best['acc_at_5k'] + 0.005
    print(f'\n--- 対照判定(序盤5k step hard acc): '
          f'teacher={t_best["acc_at_5k"]:.4f} vs random={r_best["acc_at_5k"]:.4f} '
          f'(差 {t_best["acc_at_5k"] - r_best["acc_at_5k"]:+.4f}) → '
          f'{"teacher優位=widen続行" if teacher_wins else "teacher非優位=widen中止"}')

    # --- Step5: widen(f=4、ベスト設定 × teacher/random 両arm) ---
    widen_rows = []
    if not args.skip_widen and teacher_wins:
        f = args.widen_factor
        print(f'\n[Step5] widen f={f}(ベスト設定 loss={t_best["loss"]} '
              f'tau={t_best["tau"]}、teacher/random両arm)')
        wir = widen_native_ir(active_ir, f, seed=args.seed,
                              jitter_frac=args.jitter_frac,
                              init_new_final='const0')
        errs = validate(wir)
        if errs:
            raise IRValidationError(f'widen f={f} 検証失敗: {errs[:3]}')
        wgates = _circuit_gate_count(wir)
        print(f'  widen f={f}: gates={wgates} (base={active_info["gates"]})')
        sub = argparse.Namespace(**vars(args))
        sub.taus = [t_best['tau']]
        sub.losses = [t_best['loss']]
        for init in ('teacher', 'random'):
            widen_rows += run_finetune_grid(
                wir, f'fin2_f{f}', sub, x_ft, y_ft, x_te, y_te, device,
                teacher_logits_ft, wgates, args.seed, init=init)
        all_rows += widen_rows
    elif args.skip_widen:
        print('\n[Step5] widen スキップ(--skip-widen)')
    else:
        print('\n[Step5] widen 中止(teacher非優位)')

    # --- 余裕枠: CE版(KDなし)2arm ---
    ce_rows = []
    if args.ce_control:
        print('\n[余裕枠] CE版(KDなし)teacher vs random')
        sub = argparse.Namespace(**vars(args))
        sub.taus = [t_best['tau']]
        sub.losses = ['ce']
        for init in ('teacher', 'random'):
            ce_rows += run_finetune_grid(
                active_ir, 'fin2_f1_ce', sub, x_ft, y_ft, x_te, y_te, device,
                teacher_logits_ft, active_info['gates'], args.seed, init=init)
        all_rows += ce_rows

    # --- 集計 ---
    print('\n' + '#' * 90)
    print('(CIFAR N2) 最終集計')
    print('#' * 90)
    print(f'\nresnet18 teacher test acc = {teacher_acc:.4f}')
    print(f'MLPプロキシ best test acc = {proxy_best}')
    print(f'fan-in2 STE hard acc = {info2["ste_hard_acc"]:.4f} '
          f'(kill基準 {KILL_FANIN2_STE}: '
          f'{"PASS" if info2["ste_hard_acc"] >= KILL_FANIN2_STE else "FAIL→fan-in3診断"})')
    print(f'fan-in2 転写IR: gates={info2["gates"]} '
          f'transcribe_hard_acc={info2["transcribe_hard_acc"]:.4f}')
    if info3 is not None:
        print(f'fan-in3 STE hard acc = {info3["ste_hard_acc"]:.4f} '
              f'転写IR gates={info3["gates"]} '
              f'transcribe_hard_acc={info3["transcribe_hard_acc"]:.4f}')

    _print_table(all_rows, '全run 結果表(CIFAR)')

    # 対照サマリ(核心)
    print('\n--- 対照サマリ(核心): 転移init vs random init ---')
    for tag in sorted({r['tag'] for r in all_rows}):
        tr = [r for r in all_rows if r['tag'] == tag and r['init'] == 'teacher']
        rr = [r for r in all_rows if r['tag'] == tag and r['init'] == 'random']
        if tr and rr:
            tb, rb = _best_row(tr), _best_row(rr)
            print(f'[{tag}] @5k: teacher={tb["acc_at_5k"]:.4f} '
                  f'random={rb["acc_at_5k"]:.4f} 差={tb["acc_at_5k"] - rb["acc_at_5k"]:+.4f} '
                  f'| @15ep: {tb["acc_at_15ep"]:.4f} vs {rb["acc_at_15ep"]:.4f} '
                  f'差={tb["acc_at_15ep"] - rb["acc_at_15ep"]:+.4f} '
                  f'| 最終: {tb["final_acc"]:.4f} vs {rb["final_acc"]:.4f} '
                  f'差={tb["final_acc"] - rb["final_acc"]:+.4f}')

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'\n完了(総時間 {total})')


if __name__ == '__main__':
    main()
