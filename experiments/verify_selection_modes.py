"""選択方式比較: soft / hard_st / hard_st+CAGE / gumbel_st / gumbel_st+anneal。

SAT容量マップ(sat_capacity_map / sat_vs_training_map)で特定した境界セル
(SAT表現可能だが学習が不安定な領域)で、ゲート選択方式ごとのhard到達性を
同一プロトコル(同一配線seed・同一step数)で比較する。

追加でensemble-of-hard-circuits診断も測る:
    学習済みロジットからGumbel-maxでK個のhard回路をサンプルし多数決。
    単一argmax回路との差 = 「gapのうち単一点推定損失の割合」。

Phase A予測の検証: UNSATセルではどの選択方式も救えないはず。

Usage:
    uv run python experiments/verify_selection_modes.py --quick
    uv run python experiments/verify_selection_modes.py --seeds 16

wandb不使用。結果はstdout整形表 + JSON保存。
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as F
from cage_scheduler import CAGEScheduler, TauAnnealScheduler
from sat_capacity_map import TARGETS
from sat_vs_training_map import hard_acc, make_net, target_xy

JST = timezone(timedelta(hours=9))

ARMS = ('soft', 'hard_st', 'hard_st_cage', 'gumbel_st', 'gumbel_st_anneal')

# (target, depth, width): SAT境界近傍+代表セル
DEFAULT_CELLS = [
    ('maj3', 1, 4),        # SAT/配線税0.75, 学習は取りこぼす
    ('maj3', 1, 8),
    ('xor3', 2, 8),        # SAT, 学習は不安定(事前の実測: d2W16で94%)
    ('xor3', 3, 8),
    ('rnd4', 2, 16),
    ('parity4', 3, 16),
    ('parity3n6', 2, 8),   # 配線税0.5領域
    ('parity3n8', 3, 16),
    ('xor3', 1, 8),        # UNSAT対照: どの方式も救えないはず
]


def make_arm_net_and_sched(arm: str, n_in: int, depth: int, width: int,
                           seed: int, max_steps: int):
    selection = ('soft' if arm == 'soft'
                 else 'hard_st' if arm.startswith('hard_st')
                 else 'gumbel_st')
    net = make_net(n_in, depth, width, seed, selection=selection)
    sched = None
    if arm == 'hard_st_cage':
        sched = CAGEScheduler(net)
    elif arm == 'gumbel_st_anneal':
        sched = TauAnnealScheduler(net, tau_0=1.0, tau_min=0.2,
                                   total_steps=max_steps, anneal_noise=True)
    return net, sched


def train_arm(arm: str, name: str, depth: int, width: int, seed: int,
              *, max_steps: int, lr: float = 0.02,
              check_every: int = 100) -> dict:
    n_in, _ = TARGETS[name]
    x, y = target_xy(name)
    net, sched = make_arm_net_and_sched(arm, n_in, depth, width, seed,
                                        max_steps)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    best_hard, step_to_100 = 0.0, None
    for step in range(1, max_steps + 1):
        net.train()
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step % check_every == 0:
            h = hard_acc(net, x, y)
            if h > best_hard:
                best_hard = h
            if h == 1.0 and step_to_100 is None:
                step_to_100 = step
    final_hard = hard_acc(net, x, y)

    # ensemble-of-hard-circuits診断(K=32、Gumbel-maxサンプル+多数決)
    ens = ensemble_hard_acc(net, x, y, k=32)
    net.train()
    with torch.no_grad():
        soft_acc = (net(x).argmax(-1) == y).float().mean().item()
    net.eval()
    return {
        'final_hard': final_hard, 'best_hard': best_hard,
        'step_to_100': step_to_100, 'soft_acc': soft_acc,
        'ensemble_hard': ens,
    }


@torch.no_grad()
def ensemble_hard_acc(net, x: torch.Tensor, y: torch.Tensor,
                      k: int = 32) -> float:
    """K個のGumbel-maxサンプル回路の多数決精度(単一点推定損失の診断)。"""
    from griffith_logic.layers import LutLayer

    layers = [m for m in net.modules() if isinstance(m, LutLayer)]
    saved = [layer.weights.detach().clone() for layer in layers]
    net.eval()
    votes = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(k):
        for layer, w in zip(layers, saved):
            u = torch.rand_like(w)
            g = -torch.log((-torch.log(u + 1e-20)).clamp_min(1e-20))
            layer.weights.copy_(w + g)
        pred = net(x.round()).argmax(-1)
        votes += pred
    for layer, w in zip(layers, saved):
        layer.weights.copy_(w)
    maj = (votes.float() / k > 0.5).long()
    return (maj == y).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    cells = DEFAULT_CELLS
    max_steps = args.max_steps
    if args.quick:
        cells = [('maj3', 1, 4), ('xor3', 3, 8), ('xor3', 1, 8)]
        seeds = seeds[:4]
        max_steps = 2000

    total = len(cells) * len(ARMS) * len(seeds)
    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  '
          f'{len(cells)}セル × {len(ARMS)}腕 × {len(seeds)}seed '
          f'(全{total} runs, max_steps={max_steps})')
    print('-' * 96)

    t0, done = time.time(), 0
    results = {}
    for name, depth, width in cells:
        for arm in ARMS:
            runs = []
            for seed in seeds:
                runs.append(train_arm(arm, name, depth, width, seed,
                                      max_steps=max_steps))
                done += 1
            key = f'{name}_d{depth}_W{width}_{arm}'
            reached = [r['step_to_100'] for r in runs
                       if r['step_to_100'] is not None]
            results[key] = {
                'target': name, 'depth': depth, 'width': width, 'arm': arm,
                'n': len(runs),
                'reach100_best': float(np.mean(
                    [r['best_hard'] == 1.0 for r in runs])),
                'reach100_final': float(np.mean(
                    [r['final_hard'] == 1.0 for r in runs])),
                'mean_final_hard': float(np.mean(
                    [r['final_hard'] for r in runs])),
                'mean_soft': float(np.mean([r['soft_acc'] for r in runs])),
                'mean_ensemble_hard': float(np.mean(
                    [r['ensemble_hard'] for r in runs])),
                'median_step_to_100': (float(np.median(reached))
                                       if reached else None),
            }
            c = results[key]
            elapsed = time.time() - t0
            eta = datetime.now(JST) + timedelta(
                seconds=elapsed / done * (total - done))
            s100 = (f'{c["median_step_to_100"]:.0f}'
                    if c['median_step_to_100'] else '---')
            print(f'  {name}_d{depth}_W{width:<3d} {arm:17s} '
                  f'reach100={c["reach100_best"]:4.0%} '
                  f'hard={c["mean_final_hard"]:.3f} '
                  f'soft={c["mean_soft"]:.3f} '
                  f'ens={c["mean_ensemble_hard"]:.3f} '
                  f'steps={s100:>5s}  [終了予定 {eta:%H:%M:%S}]')

    print('\n' + '=' * 96)
    print('セル×腕まとめ(reach100=学習中best hard 100%到達率)')
    print('=' * 96)
    for name, depth, width in cells:
        row = []
        for arm in ARMS:
            c = results[f'{name}_d{depth}_W{width}_{arm}']
            row.append(f'{arm}:{c["reach100_best"]:.0%}')
        print(f'  {name}_d{depth}_W{width:<3d} ' + '  '.join(row))

    out = args.out or 'verify_selection_modes_results.json'
    with open(out, 'w') as f:
        json.dump({'meta': {'seeds': len(seeds), 'max_steps': max_steps,
                            'arms': list(ARMS),
                            'timestamp': datetime.now(JST).isoformat()},
                   'results': results}, f, ensure_ascii=False, indent=2)
    print(f'\n結果JSON保存: {out}')
    print(f'終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
