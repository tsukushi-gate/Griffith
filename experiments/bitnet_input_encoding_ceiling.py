"""入力thermometer符号化ビット数 vs agreement天井の掃引(BitNet)。

問い: 入力活性を温度計bビットで符号化した復元ニューロン
    1[Σ_i u_i·x̂_i > T_opt]
のheld-out a3天井(ceil25)は、bを上げるとどこまで実数教師に迫るか。
回路実装(分解)を伴わない純粋な符号化上限の測定で、
「0.95に必要な入力ビット数」を確定する。

使用例(GPU機B): .venv/bin/python -u experiments/bitnet_input_encoding_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from convert_bitnet_multilut import (  # noqa: E402
    _load_bitnet_layer,
    _thermo_input_expand,
)
from env_utils import load_dotenv  # noqa: E402
from neuron_lut_lower_bound import optimal_threshold_teacher  # noqa: E402

load_dotenv()
JST = timezone(timedelta(hours=9))


def _uniform_global_recon(x_sup: np.ndarray, fit_i: np.ndarray,
                          bits: int) -> np.ndarray:
    """全次元共通の均一値グリッドで量子化した復元 x̂(等重み集約用)。

    しきい値を lo+kΔ(k=1..b、Δ=(hi−lo)/(b+1)、lo/hi=fit半のpooled
    0.5%/99.5%分位)に置く。全ワイヤの復元差分がΔで等しいため、
    集約は等重みpopcount木(≈2.3 LUT/線)で厳密実装できる。
    その代償として裾のクリップと非適応binによる天井低下を測るのが目的。
    """
    pooled = x_sup[fit_i].ravel()
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    delta = (hi - lo) / (bits + 1)
    taus = lo + delta * np.arange(1, bits + 1)
    level = (x_sup[:, :, None] >= taus[None, None, :]).sum(axis=2)
    return lo + delta * (level + 0.5)  # bin中央(等間隔復元)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--n-neurons', type=int, default=128)
    p.add_argument('--calib-n', type=int, default=1024)
    p.add_argument('--bits-grid', type=int, nargs='+',
                   default=[1, 2, 3, 4, 6, 8])
    p.add_argument('--recon', choices=('mean', 'uniform'), default='mean',
                   help='mean=分位点+条件付き平均(適応) | '
                        'uniform=全次元共通の均一グリッド(等重み集約可)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    q, scale, x = _load_bitnet_layer(args)
    xb = (x > 0).astype(np.int64)
    rng = np.random.default_rng(args.seed)
    rows = rng.permutation(q.shape[0])[:args.n_neurons]
    n = x.shape[0]
    perm = rng.permutation(n)
    fit_i, val_i = perm[:n // 2], perm[n // 2:]

    ceilings: dict[str, list[float]] = {'binary': []}
    for b in args.bits_grid:
        ceilings[f'thermo{b}'] = []
    t0 = time.time()
    for i, r in enumerate(rows):
        sup = np.nonzero(q[r])[0]
        if len(sup) < 8:
            continue
        u = q[r][sup].astype(np.float64)
        y_real = (x @ (q[r] * scale) > 0).astype(np.int64)
        s_bin = xb[:, sup] @ u.astype(np.int64)
        t_opt, _, _ = optimal_threshold_teacher(s_bin[fit_i], y_real[fit_i])
        ceilings['binary'].append(float(
            ((s_bin[val_i] > t_opt).astype(np.int64)
             == y_real[val_i]).mean()))
        for b in args.bits_grid:
            if args.recon == 'mean':
                w25, _, T25 = _thermo_input_expand(u, x[:, sup], fit_i, b)
                s25 = T25 @ w25
            else:  # uniform: 復元x̂を直接作り Σu·x̂ を評価(等重み集約相当)
                xhat = _uniform_global_recon(x[:, sup], fit_i, b)
                s25 = xhat @ u
            t25, _, _ = optimal_threshold_teacher(
                np.round(s25[fit_i], 6), y_real[fit_i])
            ceilings[f'thermo{b}'].append(float(
                ((s25[val_i] > t25).astype(np.int64)
                 == y_real[val_i]).mean()))
        if (i + 1) % 16 == 0:
            print(f'[{datetime.now(JST).strftime("%H:%M:%S")}] '
                  f'{i + 1}/{len(rows)} '
                  f'({(time.time() - t0) / (i + 1):.1f}s/n)')

    summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                   'min': float(np.min(v)),
                   'q05': float(np.quantile(v, 0.05))}
               for k, v in ceilings.items()}
    print(json.dumps(summary, indent=1))
    out_dir = Path(__file__).resolve().parents[1] / 'results' / 'multilut'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    out = out_dir / \
        f'input_encoding_ceiling_{args.recon}_s{args.seed}_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary,
                   'per_neuron': ceilings}, f, indent=1)
    print(f'保存: {out}')


if __name__ == '__main__':
    main()
