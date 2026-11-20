"""転写IRへの回復学習(=真理値表磨き、MNIST) — ROADMAP-v5。

前提:(C1'枝刈り転写)で
C1'枝刈り転写が崩壊(0.097)を arm A=0.9268 / arm B=0.9080 まで回復したが、
主ゲート(hard acc>=0.955=teacher-2%)未達。fan-in 6制約が容量の壁。

仮説(H4/H5/H6):
    C1'転写後のLUT網は「線形閾値ゲート」として初期化されており、
    6入力LUTの表現空間(任意ブール関数)を未使用。C3(convert/recover.py)で
    真理値表を fine-tune すれば残ギャップ(+2.4pt)を埋め、0.955に届くか。
    - H4: CE磨きで +1pt 以上
    - H5: KD磨き(float teacherロジット)が CE磨きを上回る(soft targetの情報量)
    - H6: 事前学習 arm A 由来の転写IRは、ランダム arm B 由来より磨き後も優位

パイプライン:
    1. C1'再実行(convert_prune_transcribe_mnist の機構を import 再利用)。
       arm A(事前学習float ckpt初期化)/ arm B(ランダム)、epochs_per_stage=4、
       seed固定 → 転写IR取得。IRを JSON 保存(ir/serialize.save_ir、outputs/配下)。
    2. C3磨き: rebuild_trainable(graph,'independent') → 微調整 → 再離散化・再抽出 →
       hard acc(ir/simulate、test 10k)。
       - CE損失: recover の既定(F.cross_entropy)。
       - KD損失: make_distill_task(alpha,T).loss_fn を recover の loss_fn に注入。
         train_loader は PackedDistillDataset(student二値化X + teacher float ロジット)。
         teacher ロジットは float MLP ckpt を student と同じ二値化Xで推論して整列。
       - RecoverConfig 小スイープ(構成数<=8、arm A中心、arm B はベスト設定のみ)。
    3. 接続不変assert: 磨き前後で各 lut_layer の indices が完全一致
       (recover は真理値表のみ学習、接続は buffer=固定)。
    4. wandb: group='convert-mnist-c3polish'、tags=['D3-4prime']、1構成=1run。

主ゲート: 磨き後 hard acc >= 0.955。

使用例:
    # 煙実行(数十秒、wandb無効、少データ・fanin(6)1段・2構成)
    uv run python experiments/convert_c3_polish_mnist.py --quick --no-wandb

    # 本実行(wandb online)
    uv run python experiments/convert_c3_polish_mnist.py \
        --float-ckpt checkpoints/teacher_mlp_float_mnist_20260703_022747/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

JST = timezone(timedelta(hours=9))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#' の機構を無改造で再利用(import のみ)
from convert_prune_transcribe_mnist import (
    GATE_ACC,
    TEACHER_FLOAT_ACC,
    _auto_device,
    _build_ste_from_float,
    _build_ste_random,
    _find_latest_float_ckpt,
    _flatten_binary,
    run_arm,
)
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.cache import PackedDistillDataset
from griffith_logic.convert.distill import make_distill_task
from griffith_logic.convert.prune import PruneSchedule, max_fanin
from griffith_logic.convert.recover import (
    RecoverConfig,
    _hard_accuracy,
    rebuild_trainable,
    recover,
)
from griffith_logic.convert.teacher import resolve_teacher
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.ir.serialize import save_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate

load_dotenv()


# ============================================================
# 転写('の transcribe_and_eval を簡略化、IRも返す)
# ============================================================

def _lut_indices_ordered(graph) -> list[np.ndarray]:
    """グラフ内 lut_layer ノードの indices をノード順で返す(接続の同一性照合用)。"""
    return [
        np.asarray(n.params['indices']).copy()
        for n in graph.nodes if n.type == 'lut_layer'
    ]


def _simulate_pred(ir, x: np.ndarray, batch: int = 2000) -> np.ndarray:
    preds = []
    for i in range(0, len(x), batch):
        preds.append(simulate(ir, x[i:i + batch]).argmax(axis=1))
    return np.concatenate(preds)


def transcribe_to_ir(arm, model, calib_x, final_fanin, weight_bits, device):
    """枝刈り済みモデルを転写して IR を返す(CPU/float64で決定的)。"""
    mf = max_fanin(model)
    if mf > final_fanin:
        raise RuntimeError(
            f'arm {arm}: 枝刈り後fan-in={mf} > 転写fan-in={final_fanin}。厳密転写不可'
        )
    model_cpu = model.to('cpu').eval()
    spec = TranscribeSpec(n_inputs=final_fanin, weight_bits=weight_bits, isf=False)
    ir = transcribe_mlp(model_cpu, calib_x, spec)
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'arm {arm}: IR検証失敗 {errs[:3]}')
    model.to(device)
    return ir


# ============================================================
# C1' 実行(arm A/B の転写IRを作る)。' の機構を再利用
# ============================================================

@dataclass
class _ArmArgs:
    """run_arm(') が参照する属性のみを持つ薄いホルダ。"""
    batch_size: int
    _arm_wandb: bool = False


def build_transcribe_ir(arm, args, x_tr, y_tr, x_te, y_te, calib_x, sched, device):
    """1 arm を C1' 枝刈り転写して転写IRと段階履歴を返す(wandbは張らない)。"""
    if arm == 'A':
        model = _build_ste_from_float(args.float_ckpt, device)
    else:
        model = _build_ste_random(args.seed, device)
    arm_args = _ArmArgs(batch_size=args.batch_size, _arm_wandb=False)
    hist = run_arm(arm, model, sched, x_tr, y_tr, x_te, y_te, arm_args, device)
    ir = transcribe_to_ir(arm, model, calib_x, sched.fanins[-1],
                          args.weight_bits, device)
    # 転写IRのhard acc(=磨き前ベースライン。simulate で厳密測定)
    sim_pred = _simulate_pred(ir, x_te)
    transcribe_acc = float((sim_pred == y_te).sum()) / len(y_te)
    return ir, hist, transcribe_acc


# ============================================================
# teacher ロジット(KD磨き用)。float MLP ckpt を二値化Xで推論
# ============================================================

def compute_teacher_logits(float_ckpt: str, x: np.ndarray, device: str) -> torch.Tensor:
    """float teacher MLP を student と同じ二値化X(N,784 0/1)で推論し [N,C] を返す。

    student(転写IR)が見るのと同一の二値化入力で teacher ロジットを取ることで、
    インデックス整列(PackedDistillDataset の不変条件)を満たす。
    """
    teacher = resolve_teacher(f'ckpt:{float_ckpt}', device=device)
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), 2000):
            xb = torch.as_tensor(x[i:i + 2000], dtype=torch.float32, device=device)
            logits.append(teacher.model(xb).detach().cpu())
    return torch.cat(logits, dim=0).float()


# ============================================================
# C3 磨き(1構成 = 1 wandb run)
# ============================================================

def polish_one(
    *, arm, ir, transcribe_acc, loss_kind, tau, lr, epochs,
    x_tr, y_tr, x_te, y_te, teacher_logits, args, device,
):
    """1構成の C3 磨きを実行し metrics を返す(wandb run を張る)。

    loss_kind: 'ce' | 'kd'
    接続不変assert: 磨き前後で lut_layer indices が完全一致すること。
    """
    n_classes = 10
    # train/eval データ(recover は eval_loader の (x, label) を hard acc に使う)
    x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64)
    x_te_t = torch.as_tensor(x_te, dtype=torch.float32)
    y_te_t = torch.as_tensor(y_te, dtype=torch.int64)
    eval_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=2000)

    if loss_kind == 'kd':
        # student二値化X + teacher float ロジットを PackedDistillDataset で整列。
        # loss_fn は make_distill_task(...).loss_fn(output, target) の2引数契約。
        student_ds = TensorDataset(x_tr_t, y_tr_t)
        packed_ds = PackedDistillDataset(student_ds, teacher_logits)
        train_loader = DataLoader(packed_ds, batch_size=args.batch_size, shuffle=True)
        task = make_distill_task(
            n_classes, alpha=args.kd_alpha, temperature=args.kd_temp,
            standardize=False, kind='kl',
        )
        loss_fn = task.loss_fn
    else:  # ce
        train_loader = DataLoader(
            TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True,
        )
        loss_fn = None  # recover 既定 = F.cross_entropy

    cfg = RecoverConfig(
        parameterization='independent', epochs=epochs, lr=lr,
        batch_size=args.batch_size, device=device, tau=tau,
    )

    # 接続の同一性: 磨き前 indices を控える
    idx_before = _lut_indices_ordered(ir)

    # wandb run(1構成=1run)
    run_name = f'c3polish_arm{arm}_{loss_kind}_tau{tau}_lr{lr}'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={
            'task': 'D3-4prime-c3polish',
            'arm': arm,
            'init': 'pretrained-float' if arm == 'A' else 'random',
            'loss': loss_kind,
            'tau': tau, 'lr': lr, 'epochs': epochs,
            'kd_alpha': args.kd_alpha if loss_kind == 'kd' else None,
            'kd_temp': args.kd_temp if loss_kind == 'kd' else None,
            'parameterization': 'independent',
            'final_fanin': args.fanins[-1], 'weight_bits': args.weight_bits,
            'transcribe_hard_acc': transcribe_acc,
            'batch_size': args.batch_size, 'seed': args.seed,
            'dataset': 'mnist-binarized', 'float_ckpt': args.float_ckpt,
        },
        device=device, tags=['D3-4prime'], group='convert-mnist-c3polish',
    )
    run_id = None
    try:
        import wandb
        run_id = wandb.run.id if (use_wandb and wandb.run is not None) else None
    except Exception:
        pass

    t0 = time.time()
    recovered, metrics = recover(ir, train_loader, eval_loader, cfg, loss_fn=loss_fn)
    dt = time.time() - t0

    # 接続不変assert(recover は真理値表のみ学習・接続は固定 buffer)
    idx_after = _lut_indices_ordered(recovered)
    if len(idx_before) != len(idx_after):
        raise RuntimeError(
            f'arm {arm} {loss_kind}: lut_layer 数が磨き前後で不一致 '
            f'{len(idx_before)} vs {len(idx_after)}'
        )
    for i, (b, a) in enumerate(zip(idx_before, idx_after, strict=True)):
        if b.shape != a.shape or not np.array_equal(b, a):
            raise RuntimeError(
                f'arm {arm} {loss_kind}: lut[{i}] の接続(indices)が磨きで変化した。'
                f'recover は真理値表のみ学習すべき(接続は固定 buffer)。'
                f'shape {b.shape} vs {a.shape}'
            )

    hard_before = metrics['hard_acc_before']
    hard_after = metrics['hard_acc_after']
    delta = hard_after - hard_before
    gap = TEACHER_FLOAT_ACC - hard_after
    print(
        f'  [arm {arm}/{loss_kind}/tau={tau}/lr={lr}] '
        f'hard {hard_before:.4f} -> {hard_after:.4f} (Δ{delta:+.4f}) '
        f'gap={gap:+.4f} soft {metrics["soft_acc_before"]:.4f}->'
        f'{metrics["soft_acc_after"]:.4f} indices固定OK ({dt:.1f}s) '
        f'run={run_id}'
    )

    result = {
        'arm': arm, 'loss': loss_kind, 'tau': tau, 'lr': lr, 'epochs': epochs,
        'hard_before': hard_before, 'hard_after': hard_after, 'delta': delta,
        'gap': gap, 'soft_before': metrics['soft_acc_before'],
        'soft_after': metrics['soft_acc_after'],
        'indices_preserved': True, 'seconds': round(dt, 2),
        'run_id': run_id, 'recovered_ir': recovered,
    }
    if use_wandb:
        log_metrics({
            f'{arm}/{loss_kind}/hard_before': hard_before,
            f'{arm}/{loss_kind}/hard_after': hard_after,
            f'{arm}/{loss_kind}/delta': delta,
            f'{arm}/{loss_kind}/gap': gap,
            f'{arm}/{loss_kind}/soft_after': metrics['soft_acc_after'],
            f'{arm}/{loss_kind}/indices_preserved': 1,
            'hard_after': hard_after, 'delta': delta, 'gap': gap,
        })
        wandb_finish()
    return result


# ============================================================
# 集計・ゲート判定
# ============================================================

def summarize(results: list[dict], base_acc: dict, gate_acc: float,
              out_dir: str, best_ir_path: str | None) -> None:
    print('\n' + '=' * 80)
    print("転写IRへのC3磨き(真理値表fine-tune) 結果")
    print('=' * 80)
    print(f'磨き前ベースライン: arm A={base_acc.get("A"):.4f} '
          f'arm B={base_acc.get("B", float("nan")):.4f} '
          f'(teacher-float={TEACHER_FLOAT_ACC})')
    header = ('| arm | 損失 | tau | lr | 磨き前 | 磨き後 | Δ | gap(vs teacher) '
              '| indices固定 | run_id |')
    print(header)
    print('|' + '---|' * 10)
    for r in results:
        print(f"| {r['arm']} | {r['loss']} | {r['tau']} | {r['lr']} | "
              f"{r['hard_before']:.4f} | {r['hard_after']:.4f} | "
              f"{r['delta']:+.4f} | {r['gap']:+.4f} | "
              f"{'✓' if r['indices_preserved'] else '✗'} | {r['run_id']} |")

    best = max(results, key=lambda r: r['hard_after'])
    print(f'\n--- 主ゲート判定 (磨き後 hard acc >= {gate_acc}) ---')
    print(f"最良構成: arm {best['arm']} / {best['loss']} / tau={best['tau']} / "
          f"lr={best['lr']} → hard_acc={best['hard_after']:.4f} "
          f"{'PASS ✅' if best['hard_after'] >= gate_acc else 'FAIL'}")
    if best['hard_after'] < gate_acc:
        print(f"  残ギャップ(対ゲート): {gate_acc - best['hard_after']:+.4f} / "
              f"対teacher: {best['gap']:+.4f}")
    if best_ir_path:
        print(f"  最良IR保存先: {best_ir_path}")

    # H4/H5/H6 検証
    print('\n--- 仮説検証 ---')
    ce_a = [r for r in results if r['arm'] == 'A' and r['loss'] == 'ce']
    kd_a = [r for r in results if r['arm'] == 'A' and r['loss'] == 'kd']
    if ce_a:
        best_ce = max(ce_a, key=lambda r: r['hard_after'])
        print(f"H4(CE磨きで+1pt以上): arm A CE best Δ={best_ce['delta']:+.4f} "
              f"→ {'支持' if best_ce['delta'] >= 0.01 else '不支持'} "
              f"(磨き後 {best_ce['hard_after']:.4f})")
    if ce_a and kd_a:
        best_ce = max(ce_a, key=lambda r: r['hard_after'])
        best_kd = max(kd_a, key=lambda r: r['hard_after'])
        diff = best_kd['hard_after'] - best_ce['hard_after']
        print(f"H5(KD>CE): arm A KD best={best_kd['hard_after']:.4f} vs "
              f"CE best={best_ce['hard_after']:.4f} (差 {diff:+.4f}) "
              f"→ {'支持' if diff > 0 else '不支持'}")
    a_res = [r for r in results if r['arm'] == 'A']
    b_res = [r for r in results if r['arm'] == 'B']
    if a_res and b_res:
        best_a = max(a_res, key=lambda r: r['hard_after'])
        best_b = max(b_res, key=lambda r: r['hard_after'])
        print(f"H6(arm A > arm B、磨き後): A best={best_a['hard_after']:.4f} vs "
              f"B best={best_b['hard_after']:.4f} "
              f"(差 {best_a['hard_after'] - best_b['hard_after']:+.4f}) "
              f"→ {'支持' if best_a['hard_after'] > best_b['hard_after'] else '不支持'}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="転写IRへのC3磨き(真理値表fine-tune、MNIST)")
    parser.add_argument('--float-ckpt', type=str, default=None,
                        help='teacher-float ckpt(arm A初期化 + KD teacher。省略で最新自動発見)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--out-dir', type=str, default='outputs/d3_4prime_c3polish',
                        help='生成IRの保存先(gitignore域 outputs/)')
    parser.add_argument('--fanins', type=int, nargs='+', default=[32, 12, 6])
    parser.add_argument('--epochs-per-stage', type=int, default=4,
                        help='C1'"'"'枝刈り fine-tune のstage毎エポック('"'"'既定4)')
    parser.add_argument('--prune-lr', type=float, default=1e-3)
    parser.add_argument('--weight-bits', type=int, default=8)
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--calib-n', type=int, default=10000)
    parser.add_argument('--eval-n', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gate-acc', type=float, default=GATE_ACC)
    # C3磨きスイープ軸(構成数<=8に抑える)
    parser.add_argument('--recover-epochs', type=int, default=10)
    parser.add_argument('--recover-train-n', type=int, default=20000,
                        help='C3磨きの微調整に使う訓練サブサンプル数(0で全train)。'
                             '磨きは既存真理値表のfine-tuneなので部分集合で十分。'
                             '最終展開層(~7万LUT出力)のsoft forwardが重いため既定20k')
    parser.add_argument('--taus', type=float, nargs='+', default=[4.0, 10.0])
    parser.add_argument('--recover-lrs', type=float, nargs='+', default=[0.01, 0.05])
    parser.add_argument('--kd-alpha', type=float, default=0.5)
    parser.add_argument('--kd-temp', type=float, default=4.0)
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(少データ・fanin(6)1段・2構成のみ)')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    device = _auto_device(args.device)
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.float_ckpt is None:
        args.float_ckpt = _find_latest_float_ckpt(args.checkpoint_dir)
        if args.float_ckpt is None:
            print('!! teacher-float ckpt が見つからない(--float-ckpt を指定)')
            sys.exit(1)

    if args.fanins[-1] not in (2, 4, 6):
        print(f'!! 最終fan-in={args.fanins[-1]} は transcribe_mlp 非対応(2/4/6のみ)')
        sys.exit(1)

    if args.quick:
        args.calib_n = min(args.calib_n, 2000)
        args.eval_n = min(args.eval_n, 2000)
        args.fanins = [6]
        args.epochs_per_stage = 1
        args.recover_epochs = 2
        args.taus = [4.0]
        args.recover_lrs = [0.05]

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')

    print(f'デバイス: {device}  float ckpt: {args.float_ckpt}')
    print(f"C1'スケジュール: fanins={args.fanins} epochs/stage="
          f"{args.epochs_per_stage} prune_lr={args.prune_lr} "
          f"weight_bits={args.weight_bits}")
    print(f'C3磨きスイープ: taus={args.taus} lrs={args.recover_lrs} '
          f'epochs={args.recover_epochs} (KD alpha={args.kd_alpha} T={args.kd_temp})')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('MNIST(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = load_mnist(binarize_threshold=0.5)
    tr_take = None if not args.quick else max(args.calib_n, 4000)
    x_tr, y_tr = _flatten_binary(train_set, n=tr_take)
    x_te, y_te = _flatten_binary(test_set, n=args.eval_n if args.quick else None)
    calib_x = x_tr[:args.calib_n]
    x_te = x_te[:args.eval_n]
    y_te = y_te[:args.eval_n]
    print(f'train={len(x_tr)} calib={len(calib_x)} test={len(x_te)}')

    started = time.time()
    sched = PruneSchedule(
        fanins=tuple(args.fanins), epochs_per_stage=args.epochs_per_stage,
        lr=args.prune_lr,
    )

    # ---- 1. C1' 転写IR(arm A / B)----
    print('\n########## 1. C1\' 枝刈り転写(arm A / B)##########')
    ir_a, hist_a, acc_a = build_transcribe_ir(
        'A', args, x_tr, y_tr, x_te, y_te, calib_x, sched, device)
    print(f"arm A 転写IR hard acc(磨き前) = {acc_a:.4f}")
    ir_b, hist_b, acc_b = build_transcribe_ir(
        'B', args, x_tr, y_tr, x_te, y_te, calib_x, sched, device)
    print(f"arm B 転写IR hard acc(磨き前) = {acc_b:.4f}")

    # 転写IRを保存(gitignore域)
    p_a = save_ir(ir_a, os.path.join(args.out_dir, f'transcribe_armA_{ts}'))
    p_b = save_ir(ir_b, os.path.join(args.out_dir, f'transcribe_armB_{ts}'))
    print(f'転写IR保存: A={p_a}  B={p_b}')

    base_acc = {'A': acc_a, 'B': acc_b}

    # ---- 2. KD用 teacher ロジット(student二値化Xで推論)----
    print('\n########## 2. teacher ロジット計算(KD磨き用)##########')
    teacher_logits = compute_teacher_logits(args.float_ckpt, x_tr, device)
    print(f'teacher logits: {tuple(teacher_logits.shape)}')

    # C3磨きの微調整サブサンプル(最終展開層のsoft forwardが重いため既定20k)。
    # 磨きは既存真理値表のfine-tuneなので部分集合で信号は十分。indexアラインを
    # 保つため x_tr / y_tr / teacher_logits を同じ範囲で先頭スライスする。
    n_pol = len(x_tr) if args.recover_train_n in (0, None) else min(
        args.recover_train_n, len(x_tr))
    x_pol, y_pol = x_tr[:n_pol], y_tr[:n_pol]
    teacher_logits_pol = teacher_logits[:n_pol]
    print(f'C3磨き訓練サブサンプル: {n_pol}/{len(x_tr)} '
          f'(eval は test 全 {len(x_te)} 件)')

    # ---- 3. C3磨きスイープ(構成数<=8)----
    print('\n########## 3. C3磨きスイープ ##########')
    results: list[dict] = []
    # arm A: CE と KD の両方を tau×lr でスイープ
    for tau in args.taus:
        for lr in args.recover_lrs:
            results.append(polish_one(
                arm='A', ir=ir_a, transcribe_acc=acc_a, loss_kind='ce',
                tau=tau, lr=lr, epochs=args.recover_epochs,
                x_tr=x_pol, y_tr=y_pol, x_te=x_te, y_te=y_te,
                teacher_logits=teacher_logits_pol, args=args, device=device,
            ))
    for tau in args.taus:
        for lr in args.recover_lrs:
            results.append(polish_one(
                arm='A', ir=ir_a, transcribe_acc=acc_a, loss_kind='kd',
                tau=tau, lr=lr, epochs=args.recover_epochs,
                x_tr=x_pol, y_tr=y_pol, x_te=x_te, y_te=y_te,
                teacher_logits=teacher_logits_pol, args=args, device=device,
            ))

    # arm B: arm A のベスト CE / ベスト KD 設定のみ(H6検証、構成節約)
    if not args.quick:
        a_ce = [r for r in results if r['arm'] == 'A' and r['loss'] == 'ce']
        a_kd = [r for r in results if r['arm'] == 'A' and r['loss'] == 'kd']
        best_ce = max(a_ce, key=lambda r: r['hard_after'])
        best_kd = max(a_kd, key=lambda r: r['hard_after'])
        for spec in (best_ce, best_kd):
            results.append(polish_one(
                arm='B', ir=ir_b, transcribe_acc=acc_b, loss_kind=spec['loss'],
                tau=spec['tau'], lr=spec['lr'], epochs=args.recover_epochs,
                x_tr=x_pol, y_tr=y_pol, x_te=x_te, y_te=y_te,
                teacher_logits=teacher_logits_pol, args=args, device=device,
            ))

    # ---- 4. ベスト構成のIRを保存 ----
    best = max(results, key=lambda r: r['hard_after'])
    best_ir_path = save_ir(
        best['recovered_ir'],
        os.path.join(
            args.out_dir,
            f"best_polished_arm{best['arm']}_{best['loss']}_"
            f"tau{best['tau']}_lr{best['lr']}_{ts}",
        ),
    )

    summarize(results, base_acc, args.gate_acc, args.out_dir, best_ir_path)

    total = str(timedelta(seconds=int(time.time() - started)))
    print('\n生成物(gitignore域 outputs/):')
    print(f'  転写IR:   A={p_a}  B={p_b}')
    print(f'  ベストIR: {best_ir_path}')
    print(f'完了(総時間 {total})')


if __name__ == '__main__':
    main()
