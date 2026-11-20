"""SAT必要サイズ解析: 実タスクの訓練サブサンプルによるモデルサイズ下界。

原理(健全な下界): 訓練データのサブサンプルN点すら完全分離できない
(UNSAT)構成(深さd×幅W)は、全データでも不可能 ⟹ そのサイズは
タスクに対する**必要条件を満たさない**。Nを増やすと下界は単調に締まる。

S1(実行可能性ゲート): MNIST / CIFAR-10 をデフォルトdense DLGN
(LutLayer×d → GroupSum(k=10))でN=64から解析できるかを確認しつつ、
幅を昇順スキャンして min_width(N, d) を求める。
ソルバ時間・節数・メモリを毎クエリ記録する(成立性レポート込み)。

幅方向の単調性: 各グループへ定数FALSEゲートを1個ずつ足す埋め込みで
SATが保存されるため、幅はkの倍数刻みで単調 → 最初のSATで打ち切り。

Usage:
    uv run python experiments/sat_min_size.py --task mnist --smoke
    uv run python experiments/sat_min_size.py --task mnist \
        --n-samples 64,256,1024 --depths 1,2 --widths 10,20,40,80,160,320
    uv run python experiments/sat_min_size.py --task charlm ...  # S2

wandb不使用。結果はstdout+JSON保存。
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from griffith_logic.analysis.sat_capacity import (
    LutTopology,
    TargetSpec,
    can_realize,
)

JST = timezone(timedelta(hours=9))


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if ru > 1e9 else ru / 1024  # darwin=bytes, linux=KB


def load_task(task: str, n: int, seed: int,
              **kwargs) -> tuple[np.ndarray, np.ndarray, int]:
    """タスクの訓練サブサンプル(層化)を(X(N,bits) 0/1, y(N,), k)で返す。

    kwargs: charlmタスク用に ctx_len(文脈文字数)/ charlm_classes(次文字の
    クラスマージ数、Noneで全65)。
    """
    rng = np.random.default_rng(seed)

    if task in ('mnist', 'cifar10'):
        from griffith_logic.datasets.vision import load_cifar10, load_mnist
        if task == 'mnist':
            train_set, _ = load_mnist()
        else:
            train_set, _ = load_cifar10(threshold_count=3)
        n_classes = 10
        labels = np.asarray(train_set.targets)
        # 層化サンプリング
        idx: list[int] = []
        per = n // n_classes
        for c in range(n_classes):
            pool = np.where(labels == c)[0]
            idx.extend(rng.choice(pool, size=per, replace=False).tolist())
        rest = n - len(idx)
        if rest > 0:
            all_pool = np.setdiff1d(np.arange(len(train_set)), np.array(idx))
            idx.extend(rng.choice(all_pool, size=rest, replace=False).tolist())
        X, y = [], []
        for i in idx:
            xi, yi = train_set[int(i)]
            X.append(xi.reshape(-1).numpy())
            y.append(int(yi))
        X = np.stack(X).astype(np.int64)
        assert set(np.unique(X)).issubset({0, 1}), '入力が2値でない'
        return X, np.array(y, dtype=np.int64), n_classes

    if task == 'charlm':
        # 生成タスクのE: 固定文脈ctx_len文字 → 次文字(k-classマージ)。
        # 各文字を vocab_bits=7bit(65<128)でバイナリ化。入力=ctx_len*7bit。
        # 「N個の(文脈,次文字)ペアを厳密分類できる最小容量」のSAT下界。
        from collections import Counter
        from pathlib import Path

        from data.char_corpus import _download_tiny_shakespeare

        from griffith_logic.convert.text_teacher import (
            SHAKESPEARE_CHAR_VOCAB,
            encode_shakespeare_char,
        )

        ctx_len = int(kwargs.get('ctx_len', 4))
        k_cap = kwargs.get('charlm_classes', None)
        vocab_bits = 7

        cache = Path('./data-charlm/tiny_shakespeare_input.txt')
        text = _download_tiny_shakespeare(cache)
        vocab_set = set(SHAKESPEARE_CHAR_VOCAB)
        text = ''.join(ch for ch in text if ch in vocab_set)
        ids = encode_shakespeare_char(text)  # 0..64

        # 次文字クラス: 頻度top-(k-1) + その他(1クラス)にマージ
        if k_cap and k_cap < 65:
            freq = Counter(ids)
            top = [c for c, _ in freq.most_common(k_cap - 1)]
            remap = {c: i for i, c in enumerate(top)}
            other = k_cap - 1

            def cls(c: int) -> int:
                return remap.get(c, other)
            k = k_cap
        else:
            def cls(c: int) -> int:
                return c
            k = 65

        # 全(文脈,次文字)ペアを作り、次文字クラスで層化サンプル
        labels_all = np.array([cls(ids[i]) for i in range(ctx_len, len(ids))])
        ctx_start = np.arange(ctx_len, len(ids))  # 各ペアの次文字位置
        per = max(1, n // k)
        idx = []
        for c in range(k):
            pool = ctx_start[labels_all == c]
            if len(pool) == 0:
                continue
            take = min(per, len(pool))
            idx.extend(rng.choice(pool, size=take, replace=False).tolist())
        rest = n - len(idx)
        if rest > 0:
            pool = np.setdiff1d(ctx_start, np.array(idx))
            idx.extend(rng.choice(pool, size=min(rest, len(pool)),
                                  replace=False).tolist())

        X = np.zeros((len(idx), ctx_len * vocab_bits), dtype=np.int64)
        y = np.zeros(len(idx), dtype=np.int64)
        for r, pos in enumerate(idx):
            ctx = ids[pos - ctx_len:pos]
            for j, c in enumerate(ctx):
                for b in range(vocab_bits):
                    X[r, j * vocab_bits + b] = (c >> b) & 1
            y[r] = cls(ids[pos])
        return X, y, k

    raise ValueError(f'未知タスク: {task}')


def scan_min_width(X: np.ndarray, y: np.ndarray, k: int, depth: int,
                   widths: list[int], *, conf_budget: int,
                   log, lut_arity: int = 2) -> dict:
    """幅を昇順スキャンし最初のSATを探す(単調性でそれ以降はSAT確定)。"""
    target = TargetSpec(dataset=(X, y))
    n_in = X.shape[1]
    rows = {}
    min_w = None
    for w in widths:
        if w % k != 0:
            continue  # GroupSum k-classは幅がkの倍数である必要
        topo = LutTopology(n_in=n_in, layer_widths=[w] * depth,
                           readout='groupsum', groupsum_k=k,
                           lut_arity=lut_arity)
        t0 = time.time()
        r = can_realize(topo, target, wiring='fixed_random', seed=0,
                        conf_budget=conf_budget, verify=(len(X) <= 256))
        dt = time.time() - t0
        rows[w] = {
            'status': r.status, 'total_s': dt, 'rss_mb': _rss_mb(),
            **{k2: r.stats[k2] for k2 in
               ('n_vars', 'n_clauses', 'encode_time_s', 'solve_time_s')},
        }
        log(f'    W={w:<5d} {r.status.upper():7s} '
            f'vars={r.stats["n_vars"]:,} clauses={r.stats["n_clauses"]:,} '
            f'encode={r.stats["encode_time_s"]:.1f}s '
            f'solve={r.stats["solve_time_s"]:.1f}s rss={_rss_mb():.0f}MB')
        if r.status == 'sat':
            min_w = w
            break
    return {'min_width_sat': min_w, 'per_width': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=str, required=True,
                        choices=['mnist', 'cifar10', 'charlm', 'sst2'])
    parser.add_argument('--n-samples', type=str, default='64,256,1024')
    parser.add_argument('--depths', type=str, default='1,2')
    parser.add_argument('--widths', type=str,
                        default='10,20,40,80,160,320,640')
    parser.add_argument('--conf-budget', type=int, default=5_000_000)
    parser.add_argument('--lut-arity', type=int, default=2,
                        choices=[2, 4, 6],
                        help='LUTのfan-in(n=4/6の必要幅frontier測定用)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ctx-len', type=int, default=4,
                        help='charlm: 文脈文字数(入力=ctx_len*7bit)')
    parser.add_argument('--charlm-classes', type=int, default=None,
                        help='charlm: 次文字のクラスマージ数(Noneで全65)')
    parser.add_argument('--smoke', action='store_true',
                        help='N=64, d=1, W=10,40のみ(成立確認)')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    ns = [int(v) for v in args.n_samples.split(',')]
    depths = [int(v) for v in args.depths.split(',')]
    widths = [int(v) for v in args.widths.split(',')]
    if args.smoke:
        ns, depths, widths = [64], [1], [10, 40]

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  task={args.task} '
        f'arity={args.lut_arity} '
        f'N={ns} d={depths} W={widths} budget={args.conf_budget:,}')

    results = {}
    for n in ns:
        X, y, k = load_task(args.task, n, args.seed,
                            ctx_len=args.ctx_len,
                            charlm_classes=args.charlm_classes)
        # 重複行はTargetSpec側でdedup+多数決される(画像はほぼ全て一意)
        log(f'  N={n}: 入力{X.shape[1]}bit, k={k}クラス')
        for d in depths:
            log(f'  --- N={n} depth={d}')
            t0 = time.time()
            res = scan_min_width(X, y, k, d, widths,
                                 conf_budget=args.conf_budget, log=log,
                                 lut_arity=args.lut_arity)
            key = f'N{n}_d{d}'
            results[key] = {'n': n, 'depth': d, **res,
                            'scan_time_s': time.time() - t0}
            mw = res['min_width_sat']
            log(f'  => N={n} d={d}: min_width_sat='
                f'{mw if mw is not None else ">max(未達)"}  '
                f'[{time.time() - t0:.0f}s]')

    out = args.out or (
        f'sat_min_size_{args.task}_arity{args.lut_arity}_results.json'
    )
    with open(out, 'w') as f:
        json.dump({'meta': {'task': args.task, 'n_samples': ns,
                            'depths': depths, 'widths': widths,
                            'conf_budget': args.conf_budget,
                            'lut_arity': args.lut_arity,
                            'seed': args.seed,
                            'timestamp': datetime.now(JST).isoformat()},
                   'results': results}, f, ensure_ascii=False, indent=2)
    log(f'結果JSON保存: {out}')
    log(f'終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
