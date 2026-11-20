"""SAT表現可能性 vs 勾配学習到達性のクロスマップ(4象限)。

各セル(標的×深さ×幅)で:
  1. SAT判定(自由配線+固定ランダム配線perセルseed)
  2. 勾配学習(soft訓練→hard eval、verify_capacity_ladderと同一プロトコル)
     を**SATと同一の固定配線**で実施
  3. SAT-witness-init腕: SATのwitnessから初期化して継続学習し、
     softランドスケープが正しいhard解を保存するか壊すかを測る

4象限:
    UNSAT                → 真の容量壁(学習法では絶対に救えない)
    SAT ∧ learned        → 健全(表現可能で学習も到達)
    SAT ∧ stuck          → ★表現可能だが学習不能(学習法・配線の問題)
    (witness-init崩壊)   → softランドスケープがhard解を破壊する証拠

Usage:
    uv run python experiments/sat_vs_training_map.py --quick
    uv run python experiments/sat_vs_training_map.py --seeds 8

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
import torch.nn as nn
import torch.nn.functional as F
from sat_capacity_map import TARGETS  # noqa: E402

from griffith_logic.analysis.sat_capacity import (
    LutTopology,
    TargetSpec,
    can_realize,
    witness_to_model,
)
from griffith_logic.group_sum import GroupSum
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))


def target_xy(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    n_in, fn = TARGETS[name]
    x = torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=n_in)), dtype=torch.float32,
    )
    y = torch.tensor([fn(tuple(int(v) for v in r)) for r in x.tolist()],
                     dtype=torch.long)
    return x, y


def make_net(n_in: int, depth: int, width: int, seed: int,
             selection: str = 'soft') -> nn.Sequential:
    """verify_capacity_ladderと同一構成(同一seed→同一配線)。"""
    torch.manual_seed(seed)
    gf = 1.0 if depth == 1 else 2.0
    layers: list[nn.Module] = []
    d = n_in
    for _ in range(depth):
        layers.append(LutLayer(d, width, n_inputs=2,
                               parameterization='softmax',
                               grad_factor=gf, selection=selection))
        d = width
    layers.append(GroupSum(k=2, tau=max(1.0, (width // 2) ** 0.5)))
    return nn.Sequential(*layers)


def extract_wiring(model: nn.Sequential) -> list[np.ndarray]:
    return [m.indices.numpy().copy() for m in model
            if isinstance(m, LutLayer)]


def hard_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(x.round()).argmax(-1) == y).float().mean().item()


def train_eval(model: nn.Sequential, x: torch.Tensor, y: torch.Tensor,
               *, max_steps: int = 6000, lr: float = 0.02,
               check_every: int = 100,
               record_at: tuple[int, ...] = ()) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_hard = 0.0
    trace = {0: hard_acc(model, x, y)} if 0 in record_at else {}
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        if step % check_every == 0 or step in record_at:
            h = hard_acc(model, x, y)
            best_hard = max(best_hard, h)
            if step in record_at:
                trace[step] = h
    return {'hard': hard_acc(model, x, y), 'best_hard': best_hard,
            'trace': trace}


def gate_flip_count(model: nn.Sequential, ref_gates: list[np.ndarray]) -> int:
    """witness初期化時のargmaxゲートからのflip数。"""
    flips = 0
    li = 0
    for m in model:
        if isinstance(m, LutLayer):
            cur = m.weights.argmax(-1).numpy()
            flips += int((cur != ref_gates[li]).sum())
            li += 1
    return flips


def run_cell(name: str, depth: int, width: int, seeds: list[int],
             max_steps: int, witness_selection: str = 'soft') -> dict:
    n_in, fn = TARGETS[name]
    target = TargetSpec(fn=fn)
    topo = LutTopology(n_in=n_in, layer_widths=[width] * depth,
                       readout='groupsum')
    x, y = target_xy(name)

    free = can_realize(topo, target, wiring='free', conf_budget=2_000_000)

    per_seed = []
    for seed in seeds:
        # 学習と同一配線をSATに渡す(make_netと同一seedで構築)
        net = make_net(n_in, depth, width, seed)
        wiring = extract_wiring(net)
        sat = can_realize(topo, target, wiring=wiring,
                          conf_budget=2_000_000)

        tr = train_eval(net, x, y, max_steps=max_steps)

        row = {
            'seed': seed,
            'fixed_sat': sat.status == 'sat',
            'trained_hard': tr['hard'],
            'trained_best': tr['best_hard'],
        }

        # SAT-witness-init腕: witnessから初期化して継続学習
        if sat.status == 'sat':
            wnet = witness_to_model(sat, topo)
            if witness_selection != 'soft':
                for m in wnet.modules():
                    if isinstance(m, LutLayer):
                        m.selection = witness_selection
            h0 = hard_acc(wnet, x, y)
            assert h0 == 1.0, f'witness初期化がhard acc 1.0でない: {h0}'
            ref_gates = [m.weights.argmax(-1).numpy().copy()
                         for m in wnet if isinstance(m, LutLayer)]
            wr = train_eval(wnet, x, y, max_steps=max_steps,
                            record_at=(0, 1000, max_steps))
            row['witness_trace'] = wr['trace']
            row['witness_final_hard'] = wr['hard']
            row['witness_gate_flips'] = gate_flip_count(wnet, ref_gates)
        per_seed.append(row)

    n = len(per_seed)
    sat_frac = float(np.mean([r['fixed_sat'] for r in per_seed]))
    learned = [r['trained_best'] == 1.0 for r in per_seed]
    # 4象限判定(セルの支配的状態)
    sat_stuck = [r['fixed_sat'] and r['trained_best'] < 1.0
                 for r in per_seed]
    wit = [r for r in per_seed if 'witness_final_hard' in r]
    return {
        'target': name, 'depth': depth, 'width': width, 'n': n,
        'free_sat': free.status == 'sat',
        'fixed_sat_frac': sat_frac,
        'learned_frac': float(np.mean(learned)),
        'sat_and_stuck_frac': float(np.mean(sat_stuck)),
        'mean_trained_hard': float(np.mean(
            [r['trained_hard'] for r in per_seed])),
        'witness_survives_frac': (
            float(np.mean([r['witness_final_hard'] == 1.0 for r in wit]))
            if wit else None),
        'witness_mean_flips': (
            float(np.mean([r['witness_gate_flips'] for r in wit]))
            if wit else None),
        'per_seed': per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--witness-selection', type=str, default='soft',
                        choices=['soft', 'hard_st'],
                        help='witness-init継続訓練のゲート選択方式(E-a: '
                             'hard_stならsoft破壊性の対照になる)')
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    if args.quick:
        cells = [('maj3', 1, 4), ('xor3', 1, 8), ('xor3', 3, 8),
                 ('parity3n6', 2, 8)]
        seeds = seeds[:4]
        max_steps = 2000
    else:
        cells = []
        for name in ('xor3', 'maj3', 'rnd4', 'parity4', 'parity3n6',
                     'parity3n8'):
            for depth in (1, 2, 3):
                for width in (4, 8, 16):
                    cells.append((name, depth, width))
        max_steps = args.max_steps

    total = len(cells)
    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  '
          f'{total}セル × {len(seeds)}seed, max_steps={max_steps}')
    print('-' * 84)

    t0 = time.time()
    results = {}
    for i, (name, depth, width) in enumerate(cells, 1):
        c = run_cell(name, depth, width, seeds, max_steps,
                     witness_selection=args.witness_selection)
        key = f'{name}_d{depth}_W{width}'
        results[key] = c
        quad = ('UNSAT' if not c['free_sat'] else
                'SAT+learned' if c['sat_and_stuck_frac'] == 0.0 else
                'SAT+STUCK')
        wit = (f" wit_survive={c['witness_survives_frac']:.2f}"
               f"(flips{c['witness_mean_flips']:.0f})"
               if c['witness_survives_frac'] is not None else '')
        elapsed = time.time() - t0
        eta = datetime.now(JST) + timedelta(
            seconds=elapsed / i * (total - i))
        print(f'  {key:18s} {quad:12s} fixed_sat={c["fixed_sat_frac"]:.2f} '
              f'learned={c["learned_frac"]:.2f} '
              f'stuck={c["sat_and_stuck_frac"]:.2f}{wit}'
              f'  [終了予定 {eta:%H:%M:%S}]')

    print('\n' + '=' * 84)
    print('4象限まとめ(★=SAT∧stuck: 表現可能なのに学習が到達しないセル)')
    print('=' * 84)
    for key, c in results.items():
        if c['free_sat'] and c['sat_and_stuck_frac'] > 0:
            print(f"  ★ {key}: fixed_sat={c['fixed_sat_frac']:.2f} "
                  f"learned={c['learned_frac']:.2f} "
                  f"stuck={c['sat_and_stuck_frac']:.2f}")

    out = args.out or 'sat_vs_training_map_results.json'
    with open(out, 'w') as f:
        json.dump({'meta': {'seeds': len(seeds), 'max_steps': max_steps,
                            'timestamp': datetime.now(JST).isoformat()},
                   'results': results}, f, ensure_ascii=False, indent=2)
    print(f'\n結果JSON保存: {out}')
    print(f'終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
