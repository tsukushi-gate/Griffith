"""C1' 枝刈り転写パイプライン(MNIST) — ROADMAP-v5'。

naive疎化転写が崩壊(全構成 hard_acc≒0.097)したため
pivot方針を実装する:

    事前学習重みを初期値に、fan-in制約を反復magnitude枝刈りで漸進的に課しながら
    STE活性で短期fine-tune → fan-in内なので transcribe_mlp で厳密転写。

主ゲート: 転写後 hard acc >= teacher-2% (0.9747*0.98 ≒ 0.955)。
副指標: ランダム初期化対照(arm B)との到達精度・収束速度差
        = 事前学習資産(初期値+接続選択)の価値の定量化。

arm:
    A(本命)  teacher-float ckpt の重みで TeacherMLPSte を初期化(ReLU→STE切替相当)
              → PruneSchedule 適用(fine-tune中も常にSTE・マスク維持)→ 最終fan-in 6
    B(対照)  同構造・同スケジュールをランダム初期化から

両arm完了後: transcribe_mlp(TranscribeSpec(n_inputs=6, weight_bits=8),
較正=train先頭10k)→ ir/validate → simulate(test 10k)で hard acc を測る。
厳密性assert: 疎STEモデルの torch forward argmax と simulate argmax が全一致
(不一致はバグなので停止して原因調査する)。

使用例:
    # 煙実行(数十秒、wandb無効、少データ・少枝刈り)
    uv run python experiments/convert_prune_transcribe_mnist.py --quick --no-wandb

    # 本実行(wandb online、arm A/B)
    uv run python experiments/convert_prune_transcribe_mnist.py \\
        --float-ckpt checkpoints/teacher_mlp_float_mnist_20260703_022747/best.pt
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env_utils import load_dotenv
from wandb_utils import finish as wandb_finish
from wandb_utils import init_wandb, log_metrics

from griffith_logic.convert.prune import (
    PruneSchedule,
    apply_fanin_pruning,
    max_fanin,
)
from griffith_logic.convert.teacher import resolve_teacher
from griffith_logic.convert.teacher_models import TeacherMLP, TeacherMLPSte
from griffith_logic.convert.transcribe import (
    TranscribeSpec,
    _quantize_final,
    transcribe_mlp,
)
from griffith_logic.datasets.vision import load_mnist
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate

load_dotenv()

TEACHER_FLOAT_ACC = 0.9747  # 実験レポート(内部)の teacher-float test acc(ゲート基準)
GATE_ACC = 0.955            # teacher-2%(主ゲート)


# ============================================================
# データ
# ============================================================

def _flatten_binary(dataset, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """二値化済み MNIST を (N,784) int64 0/1 と (N,) ラベルへ展開する。"""
    total = len(dataset) if n is None else min(n, len(dataset))
    xs = np.empty((total, 28 * 28), dtype=np.int64)
    ys = np.empty((total,), dtype=np.int64)
    for i in range(total):
        img, label = dataset[i]
        xs[i] = img.reshape(-1).numpy().astype(np.int64)
        ys[i] = int(label)
    assert set(np.unique(xs)).issubset({0, 1}), '二値化MNISTのはずが0/1以外を検出'
    return xs, ys


def _auto_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ============================================================
# モデル構築(arm A: 事前学習初期化 / arm B: ランダム)
# ============================================================

def _find_latest_float_ckpt(ckpt_dir: str) -> str | None:
    pattern = os.path.join(ckpt_dir, 'teacher_mlp_float_mnist_*', 'best.pt')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _build_ste_from_float(ckpt_path: str, device: str) -> TeacherMLPSte:
    """teacher-float ckpt の Linear 重みで TeacherMLPSte を初期化(ReLU→STE切替)。

    TeacherMLP は self.net(Sequential: Linear,ReLU,Linear,ReLU,Linear)、
    TeacherMLPSte は self.linears(ModuleList of Linear)。両者の Linear は
    784-300 / 300-100 / 100-10 で同形なので、順に重み・バイアスをコピーする。
    事前学習の「重み初期値+接続の相対強度(=どの接続を枝刈りで残すか)」を継承する。
    """
    teacher = resolve_teacher(f'ckpt:{ckpt_path}', device='cpu')
    src = teacher.model  # TeacherMLP
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    ste = TeacherMLPSte()
    dst_linears = list(ste.linears)
    if len(src_linears) != len(dst_linears):
        raise ValueError(
            f'Linear層数不一致: float={len(src_linears)} ste={len(dst_linears)}'
        )
    with torch.no_grad():
        for s, d in zip(src_linears, dst_linears, strict=True):
            if s.weight.shape != d.weight.shape:
                raise ValueError(f'Linear形状不一致: {s.weight.shape} vs {d.weight.shape}')
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
    return ste.to(device)


def _build_ste_random(seed: int, device: str) -> TeacherMLPSte:
    """同構造 TeacherMLPSte をランダム初期化(arm B 対照)。"""
    torch.manual_seed(seed)
    return TeacherMLPSte().to(device)


# ============================================================
# 学習・評価(枝刈りのtrain_fn / eval_fn を作る)
# ============================================================

@torch.no_grad()
def _eval_acc(model, x: np.ndarray, y: np.ndarray, device: str,
              batch: int = 4000) -> float:
    """(N,784) 0/1 入力に対する torch forward の hard accuracy。"""
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def _make_train_fn(x_tr, y_tr, device, lr, batch_size, arm, use_wandb, step_state):
    """apply_fanin_pruning に渡す train_fn を作る(段ごとに新規Adam)。

    step_state['step'] を跨いでインクリメントし、arm A/B の学習曲線を
    step単位で wandb に記録する(収束速度比較のため)。
    """
    x_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64)
    n = len(x_tr)
    loss_fn = nn.CrossEntropyLoss()

    def train_fn(model, epochs):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        logs = []
        for _ in range(epochs):
            perm = torch.randperm(n)
            ep_loss, nb = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = x_t[idx].to(device)
                yb = y_t[idx].to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                nb += 1
                step_state['step'] += 1
                if use_wandb and step_state['step'] % 50 == 0:
                    log_metrics({f'{arm}/train_loss': loss.item()},
                                step=step_state['step'])
            logs.append(ep_loss / max(nb, 1))
        return {'epoch_losses': logs}

    return train_fn


def run_arm(arm, model, sched, x_tr, y_tr, x_te, y_te, args, device):
    """1つのarm(A/B)を枝刈りfine-tuneし、段階メトリクス履歴を返す。"""
    print(f'\n=== arm {arm}: 枝刈りfine-tune (fanins={sched.fanins}, '
          f'epochs/stage={sched.epochs_per_stage}) ===')
    step_state = {'step': 0}
    use_wandb = args._arm_wandb
    train_fn = _make_train_fn(
        x_tr, y_tr, device, sched.lr, args.batch_size, arm, use_wandb, step_state,
    )

    def eval_fn(m):
        return _eval_acc(m, x_te, y_te, device)

    # 枝刈り前の初期精度(arm A の事前学習資産の可視化)
    init_acc = eval_fn(model)
    print(f'  arm {arm} 枝刈り前 test_acc={init_acc:.4f} (STE切替直後)')

    hist = apply_fanin_pruning(model, sched, train_fn, eval_fn)
    hist['init_acc'] = init_acc
    for s in hist['stages']:
        print(f'  [arm {arm}] stage fanin={s["fanin"]:2d} '
              f'max_fanin_after={s["max_fanin_after"]:2d} acc={s["acc"]:.4f}')
        if use_wandb:
            log_metrics({
                f'{arm}/stage_fanin': s['fanin'],
                f'{arm}/stage_acc': s['acc'],
                f'{arm}/stage_max_fanin': s['max_fanin_after'],
            })
    print(f'  arm {arm} 最終fan-in={max_fanin(model)} '
          f'fine-tune後acc={hist["final_acc"]:.4f}')
    return hist


# ============================================================
# 転写(厳密性assert付き)
# ============================================================

def _circuit_stats(ir) -> dict[str, int]:
    lut_layers = [n for n in ir.nodes if n.type == 'lut_layer']
    total_outputs = sum(int(n.params['out_width']) for n in lut_layers)
    return {'n_lut_layers': len(lut_layers), 'total_lut_outputs': total_outputs}


def _simulate_pred(ir, x: np.ndarray, batch: int = 2000) -> np.ndarray:
    preds = []
    for i in range(0, len(x), batch):
        preds.append(simulate(ir, x[i:i + batch]).argmax(axis=1))
    return np.concatenate(preds)


@torch.no_grad()
def _torch_pred(model, x: np.ndarray, device: str, batch: int = 4000) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        preds.append(model(xb).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def _quantized_forward_pred(model, x: np.ndarray, weight_bits: int) -> np.ndarray:
    """転写が実装する意味論(STE隠れ + 最終層を weight_bits で整数量子化)の argmax。

    transcribe_mlp は隠れ層を y=1[Wx+b>0]、最終層を共有スケール整数量子化した
    score_c=Σ q_cj x_j + β_c として厳密にIR化する。素の model.forward は最終層を
    量子化しないため、weight_bits<∞ では量子化で argmax がずれ得る(バグではない)。
    厳密性assertはこの「転写が表現している関数」= 量子化最終層の forward と比較する。
    CPU/float64 で決定的に計算する(modelはmutateしない)。
    """
    linears = [mm for mm in model.children() if isinstance(mm, nn.Linear)]
    if not linears:
        linears = [mm for mm in model.modules() if isinstance(mm, nn.Linear)]
    h = torch.as_tensor(x, dtype=torch.float64)
    for lin in linears[:-1]:
        w = lin.weight.detach().to(torch.float64)
        b = lin.bias.detach().to(torch.float64)
        h = ((h @ w.T + b) > 0).to(torch.float64)
    fin = linears[-1]
    q, beta = _quantize_final(
        fin.weight.detach().to(torch.float64),
        fin.bias.detach().to(torch.float64), weight_bits,
    )
    scores = h.numpy() @ q.numpy().T.astype(np.float64) + beta.numpy()[None, :]
    return scores.argmax(axis=1)


def transcribe_and_eval(arm, model, calib_x, x_te, y_te, final_fanin,
                        weight_bits, device):
    """枝刈り済みモデルを転写し、厳密性assert + simulate hard acc を返す。"""
    print(f'\n=== arm {arm}: 転写 (n_inputs={final_fanin}, wbits={weight_bits}) ===')
    mf = max_fanin(model)
    if mf > final_fanin:
        raise RuntimeError(
            f'arm {arm}: 枝刈り後fan-in={mf} > 転写fan-in={final_fanin}。厳密転写不可'
        )
    # 転写は CPU/float64 で決定的に行う
    model_cpu = model.to('cpu').eval()
    spec = TranscribeSpec(n_inputs=final_fanin, weight_bits=weight_bits, isf=False)
    t0 = time.time()
    ir = transcribe_mlp(model_cpu, calib_x, spec)
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'arm {arm}: IR検証失敗 {errs[:3]}')

    # 厳密性: 転写が表現している関数(STE隠れ + 量子化最終層)の argmax と
    # simulate argmax が test 全体で一致(=lowering無損失)。不一致はバグ。
    ref_pred = _quantized_forward_pred(model_cpu, x_te, weight_bits)
    sim_pred = _simulate_pred(ir, x_te)
    n_mismatch = int((ref_pred != sim_pred).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'arm {arm}: 厳密転写違反 {n_mismatch}/{len(x_te)} 不一致。'
            f'バグなので停止して原因調査せよ(fan-in={mf}, wbits={weight_bits})'
        )
    # 参考: 素のfloat logits argmax(量子化前)との差 = 量子化誤差(バグでない)
    full_pred = _torch_pred(model_cpu, x_te, 'cpu')
    n_quant_flip = int((full_pred != sim_pred).sum())
    hard_acc = float((sim_pred == y_te).sum()) / len(y_te)
    stats = _circuit_stats(ir)
    dt = time.time() - t0
    model.to(device)  # 元のデバイスへ戻す
    print(f'  arm {arm}: 厳密転写OK(量子化最終層まで全{len(x_te)}件一致) '
          f'hard_acc={hard_acc:.4f} 量子化flip={n_quant_flip} '
          f'luts={stats["total_lut_outputs"]} ({dt:.1f}s)')
    return {
        'hard_acc': hard_acc, 'exact': True, 'n_mismatch': n_mismatch,
        'n_quant_flip': n_quant_flip,
        **stats, 'seconds': round(dt, 2),
    }


# ============================================================
# 集計・ゲート判定
# ============================================================

def _stage_acc_str(hist) -> str:
    return ' -> '.join(f'{s["fanin"]}:{s["acc"]:.4f}' for s in hist['stages'])


def summarize(results: dict, gate_acc: float) -> None:
    print('\n' + '=' * 72)
    print("枝刈り転写パイプライン結果")
    print('=' * 72)
    header = ('| arm | 初期化 | 枝刈り前acc | 段階acc(fanin:acc) | '
              '最終ft acc | 転写hard acc | gap(vs teacher) | 厳密 | 総LUT出力 |')
    print(header)
    print('|' + '---|' * 9)
    for arm in ('A', 'B'):
        r = results[arm]
        hist, tr = r['hist'], r['transcribe']
        gap = TEACHER_FLOAT_ACC - tr['hard_acc']
        init = '事前学習' if arm == 'A' else 'ランダム'
        print(f"| {arm} | {init} | {hist['init_acc']:.4f} | "
              f"{_stage_acc_str(hist)} | {hist['final_acc']:.4f} | "
              f"{tr['hard_acc']:.4f} | {gap:+.4f} | {tr['exact']} | "
              f"{tr['total_lut_outputs']} |")

    a_acc = results['A']['transcribe']['hard_acc']
    b_acc = results['B']['transcribe']['hard_acc']
    print(f'\n--- 主ゲート判定 (転写後hard acc >= {gate_acc}) ---')
    print(f'arm A(本命): hard_acc={a_acc:.4f} '
          f'{"PASS" if a_acc >= gate_acc else "FAIL"}')
    print(f'arm B(対照): hard_acc={b_acc:.4f} '
          f'{"PASS" if b_acc >= gate_acc else "FAIL"}')
    print('\n--- 事前学習初期値の価値(A - B) ---')
    print(f'到達精度差(転写後): {a_acc - b_acc:+.4f}')
    print(f'枝刈り前精度差: {results["A"]["hist"]["init_acc"]:.4f} vs '
          f'{results["B"]["hist"]["init_acc"]:.4f} '
          f'(= {results["A"]["hist"]["init_acc"] - results["B"]["hist"]["init_acc"]:+.4f})')
    # 第1段(最も緩いfanin)到達精度で収束速度の初速を見る
    a_s0 = results['A']['hist']['stages'][0]['acc']
    b_s0 = results['B']['hist']['stages'][0]['acc']
    print(f'第1段(fanin={results["A"]["hist"]["stages"][0]["fanin"]})到達acc: '
          f'A={a_s0:.4f} B={b_s0:.4f} (= {a_s0 - b_s0:+.4f})')


# ============================================================
# メイン
# ============================================================

def _run_pipeline(args, x_tr, y_tr, x_te, y_te, calib_x, sched, device):
    results: dict = {}
    for arm in ('A', 'B'):
        if arm == 'A':
            model = _build_ste_from_float(args.float_ckpt, device)
        else:
            model = _build_ste_random(args.seed, device)
        # arm 個別に wandb run を開く(学習曲線をrun内step単位で残す)
        use_wandb = init_wandb(
            run_name=f'c1prime_arm{arm}',
            config_dict={
                'arm': arm,
                'init': 'pretrained-float' if arm == 'A' else 'random',
                'fanins': list(sched.fanins),
                'epochs_per_stage': sched.epochs_per_stage,
                'lr': sched.lr,
                'final_fanin': sched.fanins[-1],
                'weight_bits': args.weight_bits,
                'calib_n': args.calib_n, 'eval_n': args.eval_n,
                'dataset': 'mnist-binarized', 'seed': args.seed,
                'float_ckpt': args.float_ckpt,
            },
            device=device,
            tags=['D3-3prime'],
            group='convert-mnist-c1prime',
        )
        args._arm_wandb = use_wandb
        hist = run_arm(arm, model, sched, x_tr, y_tr, x_te, y_te, args, device)
        tr = transcribe_and_eval(
            arm, model, calib_x, x_te, y_te, sched.fanins[-1],
            args.weight_bits, device,
        )
        gap = TEACHER_FLOAT_ACC - tr['hard_acc']
        if use_wandb:
            log_metrics({
                f'{arm}/init_acc': hist['init_acc'],
                f'{arm}/final_ft_acc': hist['final_acc'],
                f'{arm}/transcribe_hard_acc': tr['hard_acc'],
                f'{arm}/transcribe_gap': gap,
                f'{arm}/transcribe_exact': int(tr['exact']),
                f'{arm}/total_lut_outputs': tr['total_lut_outputs'],
                f'{arm}/final_max_fanin': sched.fanins[-1],
            })
            wandb_finish()
        results[arm] = {'hist': hist, 'transcribe': tr}
    return results


def main():
    parser = argparse.ArgumentParser(
        description="C1'枝刈り転写パイプライン(MNIST、')")
    parser.add_argument('--float-ckpt', type=str, default=None,
                        help='arm A 初期化元の teacher-float ckpt(省略で最新自動発見)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--fanins', type=int, nargs='+', default=[32, 12, 6],
                        help='段階fan-inスケジュール(最終値=転写fan-in)')
    parser.add_argument('--epochs-per-stage', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--weight-bits', type=int, default=8,
                        help='最終層量子化ビット幅(既定8)')
    parser.add_argument('--calib-n', type=int, default=10000)
    parser.add_argument('--eval-n', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gate-acc', type=float, default=GATE_ACC)
    parser.add_argument('--retry-on-fail', action='store_true', default=True,
                        help='ゲート未達なら epochs_per_stage=4 で1回だけ再試行')
    parser.add_argument('--no-retry', dest='retry_on_fail', action='store_false')
    parser.add_argument('--quick', action='store_true',
                        help='煙実行(少データ・少枝刈り・fanin(6)1段のみ)')
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

    # 転写fan-inは transcribe_mlp 対応の2/4/6のみ有効
    if args.fanins[-1] not in (2, 4, 6):
        print(f'!! 最終fan-in={args.fanins[-1]} は transcribe_mlp 非対応(2/4/6のみ)')
        sys.exit(1)

    if args.quick:
        args.calib_n = min(args.calib_n, 2000)
        args.eval_n = min(args.eval_n, 2000)
        args.fanins = [6]
        args.epochs_per_stage = 1
        args.retry_on_fail = False

    print(f'デバイス: {device}  arm A初期化元: {args.float_ckpt}')
    print(f'スケジュール: fanins={args.fanins} epochs/stage={args.epochs_per_stage} '
          f'lr={args.lr} weight_bits={args.weight_bits}')

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
        fanins=tuple(args.fanins),
        epochs_per_stage=args.epochs_per_stage,
        lr=args.lr,
    )
    results = _run_pipeline(args, x_tr, y_tr, x_te, y_te, calib_x, sched, device)
    summarize(results, args.gate_acc)

    # ゲート未達なら epochs_per_stage=4 で1回だけ再試行
    a_acc = results['A']['transcribe']['hard_acc']
    if args.retry_on_fail and a_acc < args.gate_acc and not args.quick:
        print(f'\n!! arm A ゲート未達({a_acc:.4f} < {args.gate_acc}) '
              f'→ epochs_per_stage=4 で1回再試行')
        sched2 = PruneSchedule(
            fanins=tuple(args.fanins), epochs_per_stage=4, lr=args.lr,
        )
        results = _run_pipeline(args, x_tr, y_tr, x_te, y_te, calib_x, sched2, device)
        summarize(results, args.gate_acc)

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'\n完了(総時間 {total})')


if __name__ == '__main__':
    main()
