"""バタフライQAT→fan-in=2転写 PoC(MNIST)。

密層をバタフライ因子積(行あたり非ゼロ≤2)+ternary+二値活性でQATし、
確定後に各因子を n_inputs=2 のlut_layerとして**ロスレス**転写する。
成功基準: 「QAT精度低下 ≤ 2〜3pt(float MLP比)」∧「転写bit厳密(agreement=1.0)」。

forwardが最初から離散(HeavisideSTE)なので soft/hard ギャップは構造ゼロ。
検証は (1)float baseline比の精度低下 (2)IR simulate vs hard forwardの
全テスト点スコア厳密一致assert (3)gate_idヒストグラム。

使用例:
    uv run python experiments/butterfly_qat_mnist.py --quick --no-wandb
    for s in 0 1 2; do uv run python experiments/butterfly_qat_mnist.py --seed $s; done
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from convert_prune_transcribe_mnist import _auto_device, _flatten_binary  # noqa: E402
from env_utils import load_dotenv  # noqa: E402
from wandb_utils import finish as wandb_finish  # noqa: E402
from wandb_utils import init_wandb, log_metrics  # noqa: E402

from griffith_logic.convert.butterfly_qat import (  # noqa: E402
    ButterflyMLP,
    transcribe_butterfly,
)
from griffith_logic.convert.teacher_models import TeacherMLP  # noqa: E402
from griffith_logic.datasets.vision import load_mnist  # noqa: E402
from griffith_logic.ir.simulate import simulate  # noqa: E402
from griffith_logic.ir.validate import validate  # noqa: E402

load_dotenv()
JST = timezone(timedelta(hours=9))


def _now() -> str:
    return datetime.now(JST).strftime('%H:%M:%S')


def _train(model, x_tr, y_tr, x_te, y_te, *, epochs, lr, batch, device,
           use_wandb, tag):
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr)
    xe = torch.as_tensor(x_te, dtype=torch.float32, device=device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xe), 1024):
                preds.append(model(xe[i:i + 1024]).argmax(-1).cpu())
        acc = float((torch.cat(preds).numpy() == y_te).mean())
        best = max(best, acc)
        eta = datetime.now(JST) + timedelta(
            seconds=(time.time() - t0) / (ep + 1) * (epochs - ep - 1))
        print(f'[{_now()}] {tag} ep{ep + 1}/{epochs} test={acc:.4f} '
              f'best={best:.4f} [終了予定 {eta.strftime("%H:%M:%S")}]')
        if use_wandb:
            log_metrics({f'{tag}/test_acc': acc}, step=ep)
    return model, best


def main():
    p = argparse.ArgumentParser(description='バタフライQAT MNIST PoC')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--blocks', type=int, default=2,
                   help='バタフライブロック数(levels = blocks·log2(1024))')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--tau', type=float, default=10.0,
                   help='GroupSum温度(102本/クラスの和のスケーリング)')
    p.add_argument('--train-n', type=int, default=60000)
    p.add_argument('--test-n', type=int, default=10000)
    p.add_argument('--baseline-epochs', type=int, default=8)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()

    if args.quick:
        args.epochs = 2
        args.baseline_epochs = 1
        args.train_n, args.test_n = 4000, 1000

    device = _auto_device(None)
    torch.manual_seed(args.seed)
    run_name = f'p3_butterfly_b{args.blocks}_tau{args.tau:g}_s{args.seed}'
    use_wandb = not args.no_wandb and init_wandb(
        run_name, vars(args), tags=['p3', 'butterfly-qat'],
        group='multilut-decomp', job_type='train')

    train_set, test_set = load_mnist()
    x_tr, y_tr = _flatten_binary(train_set, n=args.train_n)
    x_te, y_te = _flatten_binary(test_set, n=args.test_n)

    # float baseline(同データ・同エポック級の素朴MLP)
    print(f'[{_now()}] float baseline(TeacherMLP 784-300-100-10)学習')
    fb, acc_float = _train(
        TeacherMLP(), x_tr, y_tr, x_te, y_te,
        epochs=args.baseline_epochs, lr=1e-3,
        batch=args.batch_size, device=device,
        use_wandb=use_wandb, tag='float')

    # バタフライQAT
    print(f'[{_now()}] ButterflyMLP QAT(blocks={args.blocks})')
    model = ButterflyMLP(in_width=784, num_classes=10,
                         n_blocks=args.blocks, tau=args.tau)
    model, acc_qat = _train(
        model, x_tr, y_tr, x_te, y_te,
        epochs=args.epochs, lr=args.lr, batch=args.batch_size,
        device=device, use_wandb=use_wandb, tag='qat')

    # 転写→bit厳密assert(全テスト点、スコア一致)
    print(f'[{_now()}] 転写とbit厳密検証')
    model = model.cpu().double().eval()
    graph = transcribe_butterfly(model)
    validate(graph)
    agree_pts = 0
    total = 0
    for i in range(0, len(x_te), 512):
        xb = x_te[i:i + 512]
        scores_ir = simulate(graph, xb)
        with torch.no_grad():
            # tau=1.0のIR group_sumと合わせるため生スコアで比較
            h = torch.as_tensor(xb, dtype=torch.float64)
            hh = h[:, torch.arange(model.width) % 784]
            for f in model.factors:
                hh = f(hh)
            scores_t = hh[:, :model.usable].reshape(
                len(xb), 10, -1).sum(-1).numpy().astype(np.int64)
        agree_pts += int((scores_ir == scores_t).all(axis=1).sum())
        total += len(xb)
    transcribe_exact = agree_pts == total
    assert transcribe_exact, (
        f'転写がbit厳密でない: {agree_pts}/{total}')
    pred_ir = []
    for i in range(0, len(x_te), 512):
        pred_ir.append(simulate(graph, x_te[i:i + 512]).argmax(-1))
    acc_ir = float((np.concatenate(pred_ir) == y_te).mean())

    drop = acc_float - acc_qat
    hist = graph.metadata['butterfly']['gate_histogram']
    print(f'[{_now()}] ==== 結果 ====')
    print(f'float baseline: {acc_float:.4f}')
    print(f'QAT hard      : {acc_qat:.4f}  (低下 {drop * 100:.2f}pt)')
    print(f'転写IR        : {acc_ir:.4f}  (bit厳密 ✓)')
    print(f'総LUT数: {graph.metadata["butterfly"]["n_luts_total"]}  '
          f'ゲート分布: {hist}')
    verdict = ('成功' if drop <= 0.03 and transcribe_exact
               else 'QAT律速' if drop > 0.03 else '転写律速')
    print(f'判定: {verdict}(基準: 低下≤3pt ∧ bit厳密)')

    out_dir = Path(__file__).resolve().parents[1] / 'results' / 'multilut'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    payload = {
        'args': vars(args), 'acc_float': acc_float, 'acc_qat': acc_qat,
        'acc_ir': acc_ir, 'transcribe_exact': bool(transcribe_exact),
        'drop_pt': drop * 100, 'verdict': verdict,
        'n_luts': graph.metadata['butterfly']['n_luts_total'],
        'gate_histogram': hist,
        'n_factors': len(model.factors),
        'width': model.width,
        'log2_depth': int(math.log2(model.width)) * args.blocks,
    }
    path = out_dir / f'butterfly_mnist_s{args.seed}_{ts}.json'
    with open(path, 'w') as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f'[{_now()}] 保存: {path}')
    if use_wandb:
        log_metrics({'final/acc_float': acc_float,
                     'final/acc_qat': acc_qat, 'final/acc_ir': acc_ir,
                     'final/drop_pt': drop * 100})
        wandb_finish()


if __name__ == '__main__':
    main()
