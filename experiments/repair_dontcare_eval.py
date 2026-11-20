"""don't-care修復パスの評価: {soft, gumbel_st}訓練 × {none, refit, flip, sat}。

停滞しがちなセル(SAT表現可能だが学習で取りこぼす領域)で、訓練後の
hard網を離散修復(analysis/dontcare_repair)するとhard精度が上がるかを測る。

腕:
    none        — 修復なし(ベースライン)
    refit       — 観測パターン再適合(soft参照多数決、未観測=don't care)
    flip        — 貪欲ゲートflip(coordinate descent)
    flip+sat    — flip後に最終層SAT再合成

Usage:
    uv run python experiments/repair_dontcare_eval.py --quick
    uv run python experiments/repair_dontcare_eval.py --seeds 8

wandb不使用。結果はstdout整形表 + JSON保存。
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as F
from sat_capacity_map import TARGETS
from sat_vs_training_map import hard_acc, make_net, target_xy

from griffith_logic.analysis.dontcare_repair import (
    repair_flip,
    repair_output_layer_sat,
    repair_refit,
)

JST = timezone(timedelta(hours=9))

TRAIN_ARMS = ('soft', 'gumbel_st')
REPAIR_ARMS = ('none', 'refit', 'flip', 'flip+sat')

DEFAULT_CELLS = [
    ('maj3', 1, 4),
    ('xor3', 2, 8),
    ('rnd4', 2, 16),
    ('parity4', 3, 16),
    ('parity3n6', 2, 8),
]


def train(name: str, depth: int, width: int, seed: int, selection: str,
          *, max_steps: int, lr: float = 0.02):
    n_in, _ = TARGETS[name]
    x, y = target_xy(name)
    net = make_net(n_in, depth, width, seed, selection=selection)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(max_steps):
        net.train()
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
    return net, x, y


def apply_repair(arm: str, net, x, y) -> dict:
    if arm == 'none':
        return {'acc': hard_acc(net, x, y)}
    if arm == 'refit':
        rep = repair_refit(net, x, y)
    elif arm == 'flip':
        rep = repair_flip(net, x, y)
    elif arm == 'flip+sat':
        rep = repair_flip(net, x, y)
        rep2 = repair_output_layer_sat(net, x, y)
        return {'acc': rep2.acc_after,
                'n_changes': len(rep.layer_changes) + len(rep2.layer_changes)}
    return {'acc': rep.acc_after, 'n_changes': len(rep.layer_changes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    cells = DEFAULT_CELLS
    max_steps = args.max_steps
    if args.quick:
        cells = [('maj3', 1, 4), ('xor3', 2, 8)]
        seeds = seeds[:4]
        max_steps = 2000

    total = len(cells) * len(TRAIN_ARMS) * len(seeds)
    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  '
          f'{len(cells)}セル × {TRAIN_ARMS} × {len(seeds)}seed '
          f'× 修復{REPAIR_ARMS}')
    print('-' * 96)

    t0, done = time.time(), 0
    results = {}
    for name, depth, width in cells:
        for tarm in TRAIN_ARMS:
            accs: dict[str, list[float]] = {a: [] for a in REPAIR_ARMS}
            for seed in seeds:
                net, x, y = train(name, depth, width, seed, tarm,
                                  max_steps=max_steps)
                for rarm in REPAIR_ARMS:
                    net_c = copy.deepcopy(net)
                    r = apply_repair(rarm, net_c, x, y)
                    accs[rarm].append(r['acc'])
                done += 1
            key = f'{name}_d{depth}_W{width}_{tarm}'
            results[key] = {
                'target': name, 'depth': depth, 'width': width,
                'train_arm': tarm, 'n': len(seeds),
                **{f'hard_{a}': float(np.mean(accs[a]))
                   for a in REPAIR_ARMS},
                **{f'lift_{a}': float(np.mean(accs[a]) - np.mean(accs['none']))
                   for a in REPAIR_ARMS if a != 'none'},
            }
            c = results[key]
            elapsed = time.time() - t0
            eta = datetime.now(JST) + timedelta(
                seconds=elapsed / done * (total - done))
            print(f'  {key:28s} none={c["hard_none"]:.3f} '
                  f'refit={c["hard_refit"]:.3f}({c["lift_refit"]:+.3f}) '
                  f'flip={c["hard_flip"]:.3f}({c["lift_flip"]:+.3f}) '
                  f'flip+sat={c["hard_flip+sat"]:.3f}'
                  f'({c["lift_flip+sat"]:+.3f})  [終了予定 {eta:%H:%M:%S}]')

    out = args.out or 'repair_dontcare_eval_results.json'
    with open(out, 'w') as f:
        json.dump({'meta': {'seeds': len(seeds), 'max_steps': max_steps,
                            'timestamp': datetime.now(JST).isoformat()},
                   'results': results}, f, ensure_ascii=False, indent=2)
    print(f'\n結果JSON保存: {out}')
    print(f'終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
