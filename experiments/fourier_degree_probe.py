"""タスクデータの本質次数プローブ: 次数別特徴の予測力カーブ(訓練前診断)。

「次数dまでの相互作用特徴で線形予測して、精度が飽和する次数=本質次数」。
Fourier重みW^dの直接推定は高次元で項数爆発(comb(n,d))により破綻するため、
予測力カーブで代替する。閉形式ridge回帰(反復なし)で軽量。

これは「タスクが低次(浅い網で足りる)か高次(深さ/高arityが要る)か」を
勾配DLGN訓練なしで推定する、SAT必要サイズ(E)の理論版。

Usage:
    uv run python experiments/fourier_degree_probe.py --tasks mnist,charlm,sst2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, sys.path[0] or '.')
from sat_min_size import load_task  # noqa: E402


def ridge_cv_acc(F, y, k, l2=10.0, seed=0):
    rng = np.random.default_rng(seed)
    N = len(F)
    F = np.hstack([F, np.ones((N, 1))])
    Y = np.eye(k)[y]
    perm = rng.permutation(N)
    folds = np.array_split(perm, 3)
    accs = []
    for f in range(3):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(3) if j != f])
        A = F[tr].T @ F[tr]
        A[np.diag_indices_from(A)] += l2
        W = np.linalg.solve(A, F[tr].T @ Y[tr])
        accs.append(((F[te] @ W).argmax(1) == y[te]).mean())
    return float(np.mean(accs))


def degree_curve(X, y, k, max_deg=3, n_inter=1200, seed=0):
    rng = np.random.default_rng(seed)
    Xpm = 1.0 - 2.0 * X
    keep = Xpm.std(0) > 1e-6
    Xz = ((Xpm - Xpm.mean(0)) / (Xpm.std(0) + 1e-9))[:, keep]
    n = Xz.shape[1]
    cur = Xz.copy()
    out = {}
    for d in range(1, max_deg + 1):
        if d >= 2:
            idx = rng.integers(0, n, size=(n_inter, d))
            newf = np.ones((len(Xz), n_inter))
            for j in range(d):
                newf *= Xz[:, idx[:, j]]
            cur = np.hstack([cur, newf])
        out[d] = ridge_cv_acc(cur, y, k, seed=seed)
    return out, n


def load_sst2_bow(n, seed):
    """SST-2をbag-of-words(語彙presence bit)でロード。"""
    sys.path.insert(0, 'experiments')
    from data.sst2 import download_and_tokenize
    train_set, _, tok = download_and_tokenize(
        vocab_size=2000, seq_len=64, data_dir='./data-sst2', train_limit=None)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_set), size=min(n, len(train_set)), replace=False)
    V = 2000
    X = np.zeros((len(idx), V), dtype=np.int64)
    y = np.zeros(len(idx), dtype=np.int64)
    for r, i in enumerate(idx):
        toks, label = train_set[int(i)]
        for t in np.asarray(toks).reshape(-1).tolist():
            if 0 <= t < V:
                X[r, t] = 1
        y[r] = int(label)
    return X, y, 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', default='mnist,charlm')
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--max-deg', type=int, default=3)
    ap.add_argument('--ctx-len', type=int, default=4)
    ap.add_argument('--charlm-classes', type=int, default=4)
    args = ap.parse_args()

    for task in args.tasks.split(','):
        if task == 'sst2':
            X, y, k = load_sst2_bow(args.n, 0)
        else:
            kw = ({'ctx_len': args.ctx_len, 'charlm_classes': args.charlm_classes}
                  if task == 'charlm' else {})
            X, y, k = load_task(task, args.n, 0, **kw)
        curve, n = degree_curve(X, y, k, max_deg=args.max_deg)
        base = max((y == c).mean() for c in range(k))
        print(f"\n=== {task} (有効{n}bit, k={k}, 最頻クラス率={base:.3f}) ===",
              flush=True)
        prev = base
        for d in sorted(curve):
            print(f"  degree<={d}: acc={curve[d]:.3f}  "
                  f"(前次数から{curve[d]-prev:+.3f})", flush=True)
            prev = curve[d]


if __name__ == '__main__':
    main()
