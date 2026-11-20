"""fan-in 32整数閾値ニューロンの加算木厳密lowering(MNIST)。

'でfan-in 32段のSTEモデルが 0.9555(ゲート相当)に到達済み。本スクリプトは
これを LUT 回路として厳密に保存する:

    arm A(事前学習ckpt初期化)を PruneSchedule(fanins=(32,), epochs_per_stage=4)で
    枝刈りfine-tune → weight_bits_hidden∈{3,4,6} でニューロン別スケール整数量子化 →
    unary展開 + popcount加算木 + per-neuron閾値で厳密lowering(convert/binarize.py の
    quantize_threshold_layer / lower_int_mlp)→ ir/validate → simulate(test 10k)。

主ゲート: IR simulate hard acc >= 0.955(test 10k)+ 回路規模記録。
厳密性assert: 量子化torch参照の argmax と simulate argmax が test 全体で全一致
(不一致はloweringバグなので停止する)。

量子化損失の分離測定: bits別に
    (量子化後torch参照acc / [必要なら1epoch QAT後acc] / lowering後hard acc / 回路規模)
を記録する。0.955 を下回る bits のみ短期QAT(量子化STE)で回復を試みる。

使用例:
    # 煙実行(数十秒、wandb無効、少データ・bits=4のみ)
    uv run python experiments/convert_int_lowering_mnist.py --quick --no-wandb

    # 本実行
    uv run python experiments/convert_int_lowering_mnist.py \\
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

from griffith_logic.convert.binarize import (
    lower_int_mlp,
    quantize_threshold_layer,
)
from griffith_logic.convert.prune import (
    PruneSchedule,
    apply_fanin_pruning,
    max_fanin,
)
from griffith_logic.convert.teacher import resolve_teacher
from griffith_logic.convert.teacher_models import TeacherMLPSte
from griffith_logic.convert.transcribe import _quantize_final
from griffith_logic.datasets.vision import load_fashion_mnist, load_mnist
from griffith_logic.ir.serialize import save_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate

load_dotenv()

TEACHER_FLOAT_ACC = 0.9747
GATE_ACC = 0.955


# ============================================================
# データ
# ============================================================

def _flatten_binary(dataset, n: int | None = None):
    total = len(dataset) if n is None else min(n, len(dataset))
    xs = np.empty((total, 28 * 28), dtype=np.int64)
    ys = np.empty((total,), dtype=np.int64)
    for i in range(total):
        img, label = dataset[i]
        xs[i] = img.reshape(-1).numpy().astype(np.int64)
        ys[i] = int(label)
    assert set(np.unique(xs)).issubset({0, 1}), '二値化MNISTのはずが0/1以外を検出'
    return xs, ys


def _auto_device(arg):
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _find_latest_float_ckpt(ckpt_dir):
    pattern = os.path.join(ckpt_dir, 'teacher_mlp_float_mnist_*', 'best.pt')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _build_ste_from_float(ckpt_path, device):
    """teacher-float ckpt の Linear 重みで TeacherMLPSte を初期化('流儀)。"""
    teacher = resolve_teacher(f'ckpt:{ckpt_path}', device='cpu')
    src = teacher.model
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    ste = TeacherMLPSte()
    dst_linears = list(ste.linears)
    if len(src_linears) != len(dst_linears):
        raise ValueError(
            f'Linear層数不一致: float={len(src_linears)} ste={len(dst_linears)}')
    with torch.no_grad():
        for s, d in zip(src_linears, dst_linears, strict=True):
            if s.weight.shape != d.weight.shape:
                raise ValueError(f'Linear形状不一致: {s.weight.shape} vs {d.weight.shape}')
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
    return ste.to(device)


# ============================================================
# 学習・評価
# ============================================================

@torch.no_grad()
def _eval_acc(model, x, y, device, batch=4000):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(x)


def _make_train_fn(x_tr, y_tr, device, lr, batch_size):
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
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                nb += 1
            logs.append(ep_loss / max(nb, 1))
        return {'epoch_losses': logs}

    return train_fn


def run_prune_finetune(model, sched, x_tr, y_tr, x_te, y_te, args, device):
    """arm A を PruneSchedule で枝刈りfine-tuneし、履歴を返す。"""
    print(f'\n=== arm A: 枝刈りfine-tune (fanins={sched.fanins}, '
          f'epochs/stage={sched.epochs_per_stage}, lr={sched.lr}) ===')
    train_fn = _make_train_fn(x_tr, y_tr, device, sched.lr, args.batch_size)

    def eval_fn(m):
        return _eval_acc(m, x_te, y_te, device)

    init_acc = eval_fn(model)
    print(f'  枝刈り前 test_acc={init_acc:.4f} (STE切替直後)')
    hist = apply_fanin_pruning(model, sched, train_fn, eval_fn)
    hist['init_acc'] = init_acc
    for s in hist['stages']:
        print(f'  stage fanin={s["fanin"]:2d} max_fanin_after='
              f'{s["max_fanin_after"]:2d} acc={s["acc"]:.4f}')
    print(f'  最終fan-in={max_fanin(model)} fine-tune後acc={hist["final_acc"]:.4f}')
    return hist


# ============================================================
# 量子化参照 forward(lowering が表現する意味論)
# ============================================================

def _hidden_linears_of(model):
    linears = [m for m in model.children() if isinstance(m, nn.Linear)]
    if not linears:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return linears


@torch.no_grad()
def _quantized_int_forward_pred(model, x, bits_hidden, bits_final):
    """lower_int_mlp が実装する意味論の argmax(CPU/float64、決定的)。

    隠れ: quantize_threshold_layer(bits_hidden) で q,β → 1[Σqx+β>0]。
    最終: _quantize_final(bits_final) の共有スケール整数スコア。
    素の model.forward は隠れ重みを量子化しないため、これが厳密性assertの
    比較基準(= lowering が表現する関数)になる。
    """
    linears = _hidden_linears_of(model)
    h = torch.as_tensor(x, dtype=torch.float64)
    for lin in linears[:-1]:
        w = lin.weight.detach().to(torch.float64)
        b = lin.bias.detach().to(torch.float64)
        q, beta = quantize_threshold_layer(w, b, bits_hidden)
        s = h @ q.to(torch.float64).T + beta.to(torch.float64)[None, :]
        h = (s > 0).to(torch.float64)
    fin = linears[-1]
    q, beta = _quantize_final(
        fin.weight.detach().to(torch.float64),
        fin.bias.detach().to(torch.float64), bits_final)
    scores = h.numpy() @ q.numpy().T.astype(np.float64) + beta.numpy()[None, :]
    return scores.argmax(axis=1)


def _quantized_int_acc(model, x, y, bits_hidden, bits_final):
    pred = _quantized_int_forward_pred(model, x, bits_hidden, bits_final)
    return float((pred == y).sum()) / len(y)


# ============================================================
# 短期QAT(量子化STE: forward=量子化重み、backward=素通し)
# ============================================================

class _QuantHiddenSTE(torch.autograd.Function):
    """隠れ重み量子化のSTE。forward=quantize_threshold_layer相当、backward=素通し。"""

    @staticmethod
    def forward(ctx, weight, bias, bits):
        q, beta = quantize_threshold_layer(weight.detach(), bias.detach(), bits)
        # 量子化重み(実数化して forward に使う)。s_i は行スケール。
        # q は既に整数、そのまま実数重みとして使う(発火 1[Σqx+β>0] を再現)。
        ctx.save_for_backward()
        return q.to(weight.dtype), beta.to(bias.dtype)

    @staticmethod
    def backward(ctx, grad_q, grad_beta):
        # 素通し(quantize は round のため勾配1と近似)
        return grad_q, grad_beta, None


def qat_finetune(model, x_tr, y_tr, device, bits_hidden, bits_final,
                 epochs, lr, batch_size, masks):
    """量子化STEで1(数)epoch QAT。隠れ層は量子化重みでforward、最終層は素の実数。

    masks: 枝刈りマスク(fan-in維持)。QAT中も枝刈り位置を0固定する。
    modelはin-placeで更新される。
    """
    linears = _hidden_linears_of(model)
    hidden = linears[:-1]
    final = linears[-1]
    x_t = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_tr, dtype=torch.int64, device=device)
    n = len(x_tr)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def forward_qat(xb):
        h = xb.to(torch.float32)
        for lin in hidden:
            qw, qb = _QuantHiddenSTE.apply(lin.weight, lin.bias, bits_hidden)
            s = h @ qw.T + qb[None, :]
            h = (s > 0).to(torch.float32) + s - s.detach()  # 発火STE(hardtanh風不要、値素通し)
        return final(h)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = forward_qat(x_t[idx])
            loss = loss_fn(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            # 枝刈りマスク維持
            with torch.no_grad():
                for lin, m in zip(hidden, masks, strict=True):
                    lin.weight.mul_(m)


# ============================================================
# lowering + 厳密性assert
# ============================================================

def _circuit_stats(ir):
    lut_layers = [n for n in ir.nodes if n.type == 'lut_layer']
    total_outputs = sum(int(n.params['out_width']) for n in lut_layers)
    return {'n_lut_layers': len(lut_layers), 'total_lut_outputs': total_outputs}


def _simulate_pred(ir, x, batch=2000):
    preds = []
    for i in range(0, len(x), batch):
        preds.append(simulate(ir, x[i:i + batch]).argmax(axis=1))
    return np.concatenate(preds)


def lower_and_eval(model, x_te, y_te, bits_hidden, bits_final):
    """lower_int_mlp → validate → simulate + 厳密性assert。"""
    model_cpu = model.to('cpu').eval()
    t0 = time.time()
    ir = lower_int_mlp(model_cpu, weight_bits_hidden=bits_hidden,
                       weight_bits_final=bits_final)
    errs = validate(ir)
    if errs:
        raise IRValidationError(f'IR検証失敗 {errs[:3]}')

    ref_pred = _quantized_int_forward_pred(model_cpu, x_te, bits_hidden, bits_final)
    sim_pred = _simulate_pred(ir, x_te)
    n_mismatch = int((ref_pred != sim_pred).sum())
    if n_mismatch != 0:
        raise RuntimeError(
            f'厳密lowering違反 {n_mismatch}/{len(x_te)} 不一致。'
            f'バグなので停止(bits_hidden={bits_hidden}, bits_final={bits_final})')
    hard_acc = float((sim_pred == y_te).sum()) / len(y_te)
    stats = _circuit_stats(ir)
    dt = time.time() - t0
    print(f'  厳密loweringOK(全{len(x_te)}件一致) hard_acc={hard_acc:.4f} '
          f'luts={stats["total_lut_outputs"]} layers={stats["n_lut_layers"]} '
          f'({dt:.1f}s)')
    return ir, {'hard_acc': hard_acc, 'exact': True, **stats,
                'seconds': round(dt, 2)}


# ============================================================
# 集計
# ============================================================

def summarize(rows, gate_acc):
    print('\n' + '=' * 78)
    print("整数閾値ニューロン加算木lowering 結果(test 10k)")
    print('=' * 78)
    print('| bits_hidden | 量子化acc | QAT後acc | lowering後hard acc | 総LUT出力 | LUT層数 | 厳密 |')
    print('|' + '---|' * 7)
    for r in rows:
        qat = f'{r["qat_acc"]:.4f}' if r['qat_acc'] is not None else '-'
        print(f'| {r["bits"]} | {r["quant_acc"]:.4f} | {qat} | '
              f'{r["hard_acc"]:.4f} | {r["total_lut_outputs"]} | '
              f'{r["n_lut_layers"]} | {r["exact"]} |')
    best = max(rows, key=lambda r: r['hard_acc'])
    print(f'\n--- 主ゲート判定 (lowering後 hard acc >= {gate_acc}) ---')
    for r in rows:
        print(f'bits_hidden={r["bits"]}: hard_acc={r["hard_acc"]:.4f} '
              f'{"PASS" if r["hard_acc"] >= gate_acc else "FAIL"}')
    print(f'\nベスト: bits_hidden={best["bits"]} hard_acc={best["hard_acc"]:.4f} '
          f'({best["total_lut_outputs"]} LUT出力)')
    return best


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="整数閾値ニューロン加算木lowering(MNIST)")
    parser.add_argument('--float-ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='データセット(既定mnist。fashion_mnistで転移実験)')
    parser.add_argument('--extra-tags', type=str, nargs='*', default=[],
                        help='wandb tagsへ追加するタグ(既定なし)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--fanins', type=int, nargs='+', default=[32])
    parser.add_argument('--epochs-per-stage', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('--bits-hidden', type=int, nargs='+', default=[3, 4, 6])
    parser.add_argument('--bits-final', type=int, default=8)
    parser.add_argument('--qat-epochs', type=int, default=1)
    parser.add_argument('--qat-lr', type=float, default=3e-4)
    parser.add_argument('--calib-n', type=int, default=10000)
    parser.add_argument('--eval-n', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gate-acc', type=float, default=GATE_ACC)
    parser.add_argument('--out-dir', type=str, default='outputs/d3_5prime_intlower')
    parser.add_argument('--quick', action='store_true')
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

    if args.quick:
        args.calib_n = min(args.calib_n, 2000)
        args.eval_n = min(args.eval_n, 2000)
        args.epochs_per_stage = 1
        args.bits_hidden = [4]

    print(f'デバイス: {device}  arm A初期化元: {args.float_ckpt}')
    print(f'fanins={args.fanins} epochs/stage={args.epochs_per_stage} '
          f'lr={args.lr} bits_hidden={args.bits_hidden} bits_final={args.bits_final}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = load_mnist if args.dataset == 'mnist' else load_fashion_mnist
    print(f'{args.dataset}(binarize_threshold=0.5)読み込み中...')
    train_set, test_set = loader(binarize_threshold=0.5)
    tr_take = None if not args.quick else max(args.calib_n, 4000)
    x_tr, y_tr = _flatten_binary(train_set, n=tr_take)
    x_te, y_te = _flatten_binary(test_set, n=args.eval_n if args.quick else None)
    x_te = x_te[:args.eval_n]
    y_te = y_te[:args.eval_n]
    print(f'train={len(x_tr)} test={len(x_te)}')

    started = time.time()

    # --- arm A 枝刈りfine-tune ---
    model = _build_ste_from_float(args.float_ckpt, device)
    sched = PruneSchedule(fanins=tuple(args.fanins),
                          epochs_per_stage=args.epochs_per_stage, lr=args.lr)
    hist = run_prune_finetune(model, sched, x_tr, y_tr, x_te, y_te, args, device)
    torch_acc = hist['final_acc']

    # 枝刈りマスク(QAT でfan-in維持用)
    hidden = _hidden_linears_of(model)[:-1]
    masks = [(lin.weight.detach() != 0).to(lin.weight.dtype) for lin in hidden]

    # --- wandb ---
    is_fashion = args.dataset == 'fashion_mnist'
    use_wandb = init_wandb(
        run_name='d3_6prime_intlower' if is_fashion else 'd3_5prime_intlower',
        config_dict={
            'arm': 'A', 'init': 'pretrained-float',
            'fanins': list(sched.fanins),
            'epochs_per_stage': sched.epochs_per_stage, 'lr': sched.lr,
            'bits_hidden': args.bits_hidden, 'bits_final': args.bits_final,
            'qat_epochs': args.qat_epochs, 'qat_lr': args.qat_lr,
            'eval_n': args.eval_n,
            'dataset': f'{args.dataset}-binarized',
            'seed': args.seed, 'float_ckpt': args.float_ckpt,
            'torch_ft_acc': torch_acc,
            'gate_acc': args.gate_acc,
        },
        device=device,
        tags=(['D3-6prime'] if is_fashion else ['D3-5prime']) + args.extra_tags,
        group='convert-fashion-intlower' if is_fashion else 'convert-mnist-intlower',
    )
    if use_wandb:
        log_metrics({'A/init_acc': hist['init_acc'],
                     'A/final_ft_acc': torch_acc,
                     'A/final_max_fanin': max_fanin(model)})

    # --- bits別 量子化→(QAT)→lowering ---
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    best_ir = None
    best_acc = -1.0
    for bits in args.bits_hidden:
        print(f'\n=== bits_hidden={bits} ===')
        # 元の(枝刈り済み)重みを退避(各bitsで同一起点から量子化するため)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # 量子化参照 forward は float64(CPU)で決定的に評価する
        model.to('cpu')
        quant_acc = _quantized_int_acc(model, x_te, y_te, bits, args.bits_final)
        print(f'  量子化後torch参照acc={quant_acc:.4f}')

        qat_acc = None
        if quant_acc < args.gate_acc and args.qat_epochs > 0:
            print(f'  量子化acc<{args.gate_acc} → {args.qat_epochs}epoch QAT(STE)で回復試行')
            # QATもCPUで実行(量子化STEは軽量、device juggling回避)
            masks_cpu = [m.to('cpu') for m in masks]
            qat_finetune(model, x_tr, y_tr, 'cpu', bits, args.bits_final,
                         args.qat_epochs, args.qat_lr, args.batch_size, masks_cpu)
            qat_acc = _quantized_int_acc(model, x_te, y_te, bits, args.bits_final)
            print(f'  QAT後 量子化torch参照acc={qat_acc:.4f}')

        ir, stats = lower_and_eval(model, x_te, y_te, bits, args.bits_final)
        model.to(device)
        row = {'bits': bits, 'quant_acc': quant_acc, 'qat_acc': qat_acc,
               **stats}
        rows.append(row)
        if use_wandb:
            log_metrics({
                f'bits{bits}/quant_acc': quant_acc,
                f'bits{bits}/qat_acc': qat_acc if qat_acc is not None else quant_acc,
                f'bits{bits}/lowering_hard_acc': stats['hard_acc'],
                f'bits{bits}/total_lut_outputs': stats['total_lut_outputs'],
                f'bits{bits}/n_lut_layers': stats['n_lut_layers'],
            })
        if stats['hard_acc'] > best_acc:
            best_acc = stats['hard_acc']
            best_ir = (bits, ir)

        # 次bitsのため重みを起点へ戻す(QATのmutateを巻き戻す)
        model.load_state_dict(state)
        model.to(device)

    best = summarize(rows, args.gate_acc)

    # ベストIRをoutputs/へ保存(gitignore域)
    if best_ir is not None:
        bits, ir = best_ir
        ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
        path = os.path.join(args.out_dir,
                            f'best_intlower_bits{bits}_{ts}.json')
        save_ir(ir, path)
        print(f'\nベストIR保存: {path}')

    if use_wandb:
        log_metrics({'best/bits_hidden': best['bits'],
                     'best/hard_acc': best['hard_acc'],
                     'best/total_lut_outputs': best['total_lut_outputs'],
                     'best/gate_pass': int(best['hard_acc'] >= args.gate_acc)})
        wandb_finish()

    total = str(timedelta(seconds=int(time.time() - started)))
    print(f'\n完了(総時間 {total})  torch_ft_acc={torch_acc:.4f} '
          f'best_hard_acc={best["hard_acc"]:.4f} '
          f'gate={"PASS" if best["hard_acc"] >= args.gate_acc else "FAIL"}')


if __name__ == '__main__':
    main()
