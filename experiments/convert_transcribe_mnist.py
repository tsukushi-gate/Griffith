"""C1直接転写PoC + C2二値化経路のMNIST変換実験(ROADMAP-v5)。

事前学習済み float MLP を training-free で LUT 回路(IR)へ転写したときの
精度ギャップを測る。

仮説:
    H1 float活性teacherの転写は隠れ活性1bit化で大きく劣化する
    H2 二値活性(STE)で学習したteacherは転写ギャップが小さい
       (SNN変換QCFS「変換前に量子化活性で学習」の類推)
    H3 ISF(較正データ列挙)はexact列挙と同等以上

意味論の整合(重要):
    transcribe_mlp / lower_bnn_mlp は隠れ層を y_i=1[Wx+b>0] として転写する。
    teacher-ste(convert/teacher_models.TeacherMLPSte)の隠れ活性 HeavisideSTE も
    同じ y=1[preact>0] なので、STE teacher の転写は fan-in疎化以外の誤差がゼロ。
    データは二値化MNIST(x∈{0,1}、binarize_threshold=0.5)を前提とする。

フェーズ(--phase):
    teacher — teacher-float と teacher-ste を学習して ckpt 保存(wandb group='teacher')
    sweep   — 各 teacher × C1(疎化×ISF方策×量子化ビット)+ C2(二値化lowering)を
              転写して simulate 精度を測る(1構成=1 wandb run)
    all     — teacher → sweep を連続実行(既定)

使用例:
    # 煙実行(1分以内、wandb無効、少データ)
    uv run python experiments/convert_transcribe_mnist.py --phase all \\
        --quick --no-wandb

    # teacher 2種を本学習(wandb online)
    uv run python experiments/convert_transcribe_mnist.py --phase teacher --epochs 5

    # 転写スイープ(既存 ckpt を使う。--float-ckpt/--ste-ckpt 省略時は
    #   checkpoints/ から最新を自動発見)
    uv run python experiments/convert_transcribe_mnist.py --phase sweep
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from checkpoint_utils import save_checkpoint, setup_checkpoint_dir
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.binarize import lower_bnn_mlp
from griffith_logic.convert.teacher import resolve_teacher
from griffith_logic.convert.teacher_models import TeacherMLP, TeacherMLPSte
from griffith_logic.convert.transcribe import TranscribeSpec, transcribe_mlp
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate

load_dotenv()


# ============================================================
# データ
# ============================================================

def _flatten_binary(dataset, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """二値化済み dataset を (N, 784) int64 の 0/1 と (N,) ラベルへ展開する。

    load_mnist(binarize_threshold=0.5) の各サンプルは (1,28,28) の {0,1} float。
    n を与えると先頭 n サンプルに切る(較正・評価の高速化)。
    """
    total = len(dataset) if n is None else min(n, len(dataset))
    xs = np.empty((total, 28 * 28), dtype=np.int64)
    ys = np.empty((total,), dtype=np.int64)
    for i in range(total):
        img, label = dataset[i]
        xs[i] = img.reshape(-1).numpy().astype(np.int64)
        ys[i] = int(label)
    assert set(np.unique(xs)).issubset({0, 1}), '二値化MNISTのはずが0/1以外を検出'
    return xs, ys


# ============================================================
# teacher 学習
# ============================================================

def _auto_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@torch.no_grad()
def _eval_acc(model, x: np.ndarray, y: np.ndarray, device: str,
              batch: int = 2000) -> float:
    """(N,784) 0/1 入力に対する teacher の hard accuracy(float forward)。"""
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        logits = model(xb)
        pred = logits.argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def train_teacher(
    kind: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    seed: int,
    use_wandb: bool,
) -> tuple[torch.nn.Module, float]:
    """teacher-float / teacher-ste を二値化MNISTで学習し (model, test_acc) を返す。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_cls = TeacherMLP if kind == 'float' else TeacherMLPSte
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    n = len(x_tr)
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        test_acc = _eval_acc(model, x_te, y_te, device)
        avg = epoch_loss / max(n_batches, 1)
        print(f'  [teacher-{kind}] epoch {epoch}/{epochs} '
              f'loss={avg:.4f} test_acc={test_acc:.4f}')
        if use_wandb:
            log_metrics({f'{kind}/epoch': epoch, f'{kind}/loss': avg,
                         f'{kind}/test_acc': test_acc})
    return model, _eval_acc(model, x_te, y_te, device)


def _save_teacher_ckpt(model, kind: str, test_acc: float, args,
                       ckpt_dir: str) -> str:
    """teacher を resolve_teacher('ckpt:') 互換形式で保存し best.pt パスを返す。"""
    model_cls = TeacherMLP if kind == 'float' else TeacherMLPSte
    run_dir, latest_path, best_path = setup_checkpoint_dir(
        f'teacher_mlp_{kind}_mnist', base_dir=ckpt_dir,
    )
    metadata = {
        'args': vars(args),
        'kind': kind,
        'model_class': model_cls,
        'model_kwargs': {},
        'num_classes': 10,
        'input_size': (1, 28, 28),
        # 入力は二値化済み 0/1 をそのまま使う(正規化なし)
        'transform': None,
        'test_acc': test_acc,
    }
    # optimizer は保存不要だが save_checkpoint 規約に合わせダミーを渡す
    dummy_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    save_checkpoint(best_path, model, dummy_opt, None, 0, test_acc, metadata)
    save_checkpoint(latest_path, model, dummy_opt, None, 0, test_acc, metadata)
    print(f'  ckpt: {best_path}')
    return best_path


def phase_teacher(args, x_tr, y_tr, x_te, y_te, device) -> dict[str, str]:
    """teacher-float と teacher-ste を学習・保存。{kind: ckpt_path} を返す。"""
    use_wandb = init_wandb(
        run_name='teacher_mlp_mnist',
        config_dict={**vars(args), 'phase': 'teacher'},
        device=device,
        tags=['D3-1', 'teacher', 'mnist'],
        group='teacher',
    )
    ckpts: dict[str, str] = {}
    accs: dict[str, float] = {}
    for kind in ('float', 'ste'):
        print(f'\n=== teacher-{kind} 学習 ===')
        model, test_acc = train_teacher(
            kind, x_tr, y_tr, x_te, y_te,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            device=device, seed=args.seed, use_wandb=use_wandb,
        )
        accs[kind] = test_acc
        ckpts[kind] = _save_teacher_ckpt(model, kind, test_acc, args,
                                         args.checkpoint_dir)
    if use_wandb:
        log_metrics({'teacher_float_acc': accs['float'],
                     'teacher_ste_acc': accs['ste']})
        wandb_finish()
    print(f'\nteacher accs: float={accs["float"]:.4f} ste={accs["ste"]:.4f}')
    return ckpts


# ============================================================
# 転写スイープ
# ============================================================

def _circuit_stats(ir) -> dict[str, int]:
    """IR の lut_layer 数・総LUT出力数を集計する。"""
    lut_layers = [n for n in ir.nodes if n.type == 'lut_layer']
    total_outputs = sum(int(n.params['out_width']) for n in lut_layers)
    return {
        'n_lut_layers': len(lut_layers),
        'total_lut_outputs': total_outputs,
    }


def _simulate_acc(ir, x: np.ndarray, y: np.ndarray, *,
                  batch: int = 2000) -> float:
    """IR を simulate してクラススコア argmax の hard accuracy を測る。

    simulate 入力は 0/1 の int64。メモリ節約のためバッチ分割する。
    """
    correct = 0
    for i in range(0, len(x), batch):
        scores = simulate(ir, x[i:i + batch])
        pred = scores.argmax(axis=1)
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def _teacher_model_from_ckpt(ckpt_path: str, device: str):
    """ckpt から teacher の nn.Module 本体を取り出す(転写器に渡す用)。"""
    teacher = resolve_teacher(f'ckpt:{ckpt_path}', device=device)
    # TorchTeacher.model が nn.Module 本体。転写は CPU/float64 で行う
    return teacher.model


def _bnn_reference_acc(model, x: np.ndarray, y: np.ndarray) -> float:
    """C2対照: sign二値化した bipolar BNN の torch forward の hard accuracy。

    tests/l1_math/test_binarize_lowering.py の _bnn_forward と同一意味論
    (隠れ y=1[Σq(2x-1)+b'>0]、最終 argmax(qx+β))。「lowering自体は無損失」を
    数値で確認するための参照(IR simulate 精度と一致するはず)。
    """
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    h = x.astype(np.int64)
    for lin in linears[:-1]:
        w = lin.weight.detach().cpu().numpy().astype(np.float64)
        b = lin.bias.detach().cpu().numpy().astype(np.float64)
        q = np.where(w >= 0, 1, -1)
        alpha = np.abs(w).mean()
        b_prime = np.round(b / alpha)
        s = (q[None, :, :] * (2 * h[:, None, :] - 1)).sum(2) + b_prime[None, :]
        h = (s > 0).astype(np.int64)
    lin = linears[-1]
    w = lin.weight.detach().cpu().numpy().astype(np.float64)
    b = lin.bias.detach().cpu().numpy().astype(np.float64)
    q = np.where(w >= 0, 1.0, -1.0)
    alpha = np.abs(w).mean()
    beta = np.round(b / alpha)
    scores = h @ q.T + beta[None, :]
    pred = scores.argmax(axis=1)
    return int((pred == y).sum()) / len(x)


def _c1_configs() -> list[TranscribeSpec]:
    """C1スイープ軸: n_inputs∈{4,6} × 方策(exact/isf-zero/isf-majority)
    × weight_bits∈{1,8} = 12構成/teacher。"""
    specs: list[TranscribeSpec] = []
    for n_inputs in (4, 6):
        for weight_bits in (1, 8):
            # exact
            specs.append(TranscribeSpec(
                n_inputs=n_inputs, weight_bits=weight_bits, isf=False))
            # isf, dont_care=zero
            specs.append(TranscribeSpec(
                n_inputs=n_inputs, weight_bits=weight_bits,
                isf=True, dont_care='zero'))
            # isf, dont_care=majority
            specs.append(TranscribeSpec(
                n_inputs=n_inputs, weight_bits=weight_bits,
                isf=True, dont_care='majority'))
    return specs


def _policy_label(spec: TranscribeSpec) -> str:
    if not spec.isf:
        return 'exact'
    return f'isf-{spec.dont_care}'


def _run_c1(kind, model, spec, calib_x, x_te, y_te, teacher_acc, args, device):
    """1つのC1構成を転写・simulate評価し結果dictを返す(1 wandb run)。"""
    label = _policy_label(spec)
    run_name = (f'c1_{kind}_n{spec.n_inputs}_{label}_w{spec.weight_bits}')
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={
            'teacher': kind, 'route': 'C1', 'n_inputs': spec.n_inputs,
            'policy': label, 'weight_bits': spec.weight_bits,
            'phase': 'sweep', **_common_cfg(args),
        },
        device=device,
        tags=['D3-1', 'transcribe'],
        group='convert-mnist-c1',
    )
    t0 = time.time()
    ir = transcribe_mlp(model, calib_x, spec)
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'{run_name}: IR検証失敗 {errs[:3]}')
    stats = _circuit_stats(ir)
    hard_acc = _simulate_acc(ir, x_te, y_te)
    gap = teacher_acc - hard_acc
    dt = time.time() - t0
    row = {
        'teacher': kind, 'route': 'C1', 'n_inputs': spec.n_inputs,
        'policy': label, 'weight_bits': spec.weight_bits,
        'hard_acc': hard_acc, 'teacher_acc': teacher_acc, 'gap': gap,
        **stats, 'seconds': round(dt, 2),
    }
    print(f'  [C1 {kind:5s} n{spec.n_inputs} {label:13s} w{spec.weight_bits}] '
          f'hard_acc={hard_acc:.4f} gap={gap:+.4f} '
          f'luts={stats["total_lut_outputs"]} ({dt:.1f}s)')
    if use_wandb:
        log_metrics(row)
        wandb_finish()
    return row


def _run_c2(kind, model, x_te, y_te, teacher_acc, args, device):
    """C2二値化lowering(1構成/teacher)。BNN参照精度も記録。"""
    run_name = f'c2_{kind}_binarize'
    use_wandb = init_wandb(
        run_name=run_name,
        config_dict={'teacher': kind, 'route': 'C2', **_common_cfg(args)},
        device=device,
        tags=['D3-2', 'binarize'],
        group='convert-mnist-c1',
    )
    t0 = time.time()
    ir = lower_bnn_mlp(model)
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'{run_name}: IR検証失敗 {errs[:3]}')
    stats = _circuit_stats(ir)
    hard_acc = _simulate_acc(ir, x_te, y_te)
    bnn_acc = _bnn_reference_acc(model, x_te, y_te)
    gap = teacher_acc - hard_acc
    dt = time.time() - t0
    row = {
        'teacher': kind, 'route': 'C2', 'n_inputs': None,
        'policy': 'sign-lowering', 'weight_bits': 1,
        'hard_acc': hard_acc, 'teacher_acc': teacher_acc, 'gap': gap,
        'bnn_ref_acc': bnn_acc, 'lowering_lossless': abs(bnn_acc - hard_acc) < 1e-9,
        **stats, 'seconds': round(dt, 2),
    }
    print(f'  [C2 {kind:5s} binarize] hard_acc={hard_acc:.4f} '
          f'bnn_ref={bnn_acc:.4f} gap={gap:+.4f} '
          f'lossless={row["lowering_lossless"]} '
          f'luts={stats["total_lut_outputs"]} ({dt:.1f}s)')
    if use_wandb:
        log_metrics(row)
        wandb_finish()
    return row


def _common_cfg(args) -> dict:
    return {
        'calib_n': args.calib_n, 'eval_n': args.eval_n,
        'dataset': 'mnist-binarized', 'seed': args.seed,
    }


def _find_latest_ckpt(kind: str, ckpt_dir: str) -> str | None:
    """checkpoints/teacher_mlp_{kind}_mnist_*/best.pt の最新を返す。"""
    pattern = os.path.join(ckpt_dir, f'teacher_mlp_{kind}_mnist_*', 'best.pt')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def phase_sweep(args, ckpts, x_tr, y_tr, x_te, y_te, device) -> list[dict]:
    """各 teacker(float/ste)× C1 12構成 + C2 1構成 を転写評価する。"""
    # 転写は CPU/float64 で行う(数値の決定性・simulate 整合のため)
    calib_x = x_tr[:args.calib_n]
    x_eval = x_te[:args.eval_n]
    y_eval = y_te[:args.eval_n]
    specs = _c1_configs()
    if args.quick:
        specs = specs[:2]  # 煙実行は2構成のみ

    rows: list[dict] = []
    for kind in ('float', 'ste'):
        ckpt_path = ckpts.get(kind) or _find_latest_ckpt(kind, args.checkpoint_dir)
        if not ckpt_path:
            print(f'!! teacher-{kind} の ckpt が見つからない(--phase teacher を先に)')
            continue
        print(f'\n=== teacher-{kind} 転写 (ckpt={ckpt_path}) ===')
        model = _teacher_model_from_ckpt(ckpt_path, device='cpu').eval()
        teacher_acc = _eval_acc(model, x_eval, y_eval, 'cpu')
        print(f'  teacher_acc(eval {len(x_eval)})={teacher_acc:.4f}')

        for spec in specs:
            rows.append(_run_c1(kind, model, spec, calib_x, x_eval, y_eval,
                                teacher_acc, args, device))
        if not args.quick:
            rows.append(_run_c2(kind, model, x_eval, y_eval, teacher_acc,
                                args, device))
    return rows


# ============================================================
# 集計
# ============================================================

def _markdown_table(rows: list[dict]) -> str:
    header = ('| teacher | 経路 | n_inputs | 方策 | wbits | hard_acc | '
              'teacher_acc | gap | n_lut_layers | total_lut_outputs |')
    sep = '|' + '---|' * 10
    lines = [header, sep]
    for r in rows:
        ni = r['n_inputs'] if r['n_inputs'] is not None else '-'
        lines.append(
            f"| {r['teacher']} | {r['route']} | {ni} | {r['policy']} | "
            f"{r['weight_bits']} | {r['hard_acc']:.4f} | "
            f"{r['teacher_acc']:.4f} | {r['gap']:+.4f} | "
            f"{r['n_lut_layers']} | {r['total_lut_outputs']} |"
        )
    return '\n'.join(lines)


def summarize(rows: list[dict]) -> None:
    """全構成の表を出力し、ゲート判定(teacher-10%以内)を行う。"""
    if not rows:
        print('\n(結果行なし)')
        return
    print('\n' + '=' * 70)
    print('結果表')
    print('=' * 70)
    print(_markdown_table(rows))

    # ゲート: いずれかの構成で teacher-10%以内(gap <= 0.10)
    within = [r for r in rows if r['gap'] <= 0.10]
    best = min(rows, key=lambda r: r['gap'])
    print('\n--- ゲート判定(teacher-10%以内) ---')
    print(f'最良構成: teacher={best["teacher"]} route={best["route"]} '
          f'policy={best["policy"]} n_inputs={best["n_inputs"]} '
          f'wbits={best["weight_bits"]} gap={best["gap"]:+.4f}')
    print(f'gate_pass(gap<=0.10): {"PASS" if within else "FAIL"} '
          f'({len(within)}/{len(rows)} 構成)')

    # 結果を JSONL でも残す(参照用)
    out_path = os.path.join(os.getcwd(), 'convert_transcribe_mnist_results.jsonl')
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n結果 JSONL: {out_path}')


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='C1転写PoC+C2二値化経路のMNIST変換実験')
    parser.add_argument('--phase', choices=['teacher', 'sweep', 'all'],
                        default='all', help='実行フェーズ (default: all)')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='teacher学習エポック数 (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3, help='学習率')
    parser.add_argument('-bs', '--batch-size', type=int, default=128,
                        help='バッチサイズ (default: 128)')
    parser.add_argument('--seed', type=int, default=0, help='シード')
    parser.add_argument('--device', type=str, default=None,
                        help='teacher学習デバイス (default: 自動)')
    parser.add_argument('--calib-n', type=int, default=10000,
                        help='C1較正サンプル数(train先頭) (default: 10000)')
    parser.add_argument('--eval-n', type=int, default=10000,
                        help='test評価サンプル数 (default: 10000)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='ckpt保存/探索先 (default: checkpoints)')
    parser.add_argument('--float-ckpt', type=str, default=None,
                        help='sweep時のteacher-float ckpt(省略で自動発見)')
    parser.add_argument('--ste-ckpt', type=str, default=None,
                        help='sweep時のteacher-ste ckpt(省略で自動発見)')
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(少データ・少エポック・2構成のみ)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandb無効化(煙実行のみ)')
    args = parser.parse_args()

    device = _auto_device(args.device)
    print(f'デバイス: {device}  phase={args.phase}')
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.calib_n = min(args.calib_n, 1000)
        args.eval_n = min(args.eval_n, 1000)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # データ(二値化MNIST。転写意味論が x∈{0,1} 前提)
    print('MNIST(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = load_mnist(binarize_threshold=0.5)
    tr_n = None if not args.quick else 4000
    x_tr, y_tr = _flatten_binary(train_set, n=max(args.calib_n, tr_n or 0)
                                 if args.quick else None)
    x_te, y_te = _flatten_binary(test_set, n=args.eval_n if args.quick else None)
    print(f'train={len(x_tr)} test={len(x_te)}')

    started = time.time()
    ckpts: dict[str, str] = {}
    if args.phase in ('teacher', 'all'):
        ckpts = phase_teacher(args, x_tr, y_tr, x_te, y_te, device)

    if args.phase == 'sweep':
        if args.float_ckpt:
            ckpts['float'] = args.float_ckpt
        if args.ste_ckpt:
            ckpts['ste'] = args.ste_ckpt

    rows: list[dict] = []
    if args.phase in ('sweep', 'all'):
        rows = phase_sweep(args, ckpts, x_tr, y_tr, x_te, y_te, device)
        summarize(rows)

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'\n完了(総時間 {total})')
    if ckpts:
        print('teacher ckpts:')
        for k, p in ckpts.items():
            print(f'  {k}: {p}')


if __name__ == '__main__':
    main()
