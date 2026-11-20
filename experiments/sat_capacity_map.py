"""SAT容量マップ: 標的×深さ×幅×配線のグリッドで表現可能性境界を引く。

Boolean matching→SAT(analysis/sat_capacity)により、
  - free_sat:        自由配線での表現可能性(構造的境界)
  - fixed_sat_frac:  固定ランダム配線16seedのSAT率(1 - 配線税)
を各セルで厳密判定する。勾配学習の結果(verify_capacity_ladder等)と
突き合わせることで「容量壁」と「配線/最適化問題」を切り分ける土台になる。

Usage:
    uv run python experiments/sat_capacity_map.py            # 全グリッド
    uv run python experiments/sat_capacity_map.py --quick    # 短縮

wandb不使用(ローカルのみ)。結果はstdout整形表 + JSON保存。
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np

# undercap実験と同一の標的定義を流用
from distill_undercap_3input import TARGETS as BASE_TARGETS  # noqa: E402

from griffith_logic.analysis.sat_capacity import (
    LutTopology,
    TargetSpec,
    can_realize,
)

JST = timezone(timedelta(hours=9))


def _parity_in_noise(k: int, n: int):
    """先頭kビットのparity + (n-k)ノイズビット。"""
    return (n, lambda b: int(sum(b[:k]) % 2))


# name -> (n_in, label_fn)
TARGETS: dict = {
    'xor3': BASE_TARGETS['xor3'],
    'maj3': BASE_TARGETS['maj3'],
    'rnd4': BASE_TARGETS['rnd4'],
    'parity4': (4, lambda b: int(sum(b) % 2)),
    'parity5': (5, lambda b: int(sum(b) % 2)),
    'parity3n6': _parity_in_noise(3, 6),
    'parity3n8': _parity_in_noise(3, 8),
}


def run_grid(targets: list[str], depths: list[int], widths: list[int],
             n_wiring_seeds: int, conf_budget: int = 2_000_000) -> dict:
    """グリッド判定。自由配線はW方向の単調性で刈り込む。

    単調性: 偶数幅W→W+2は「両グループへ定数FALSEゲートを1個ずつ足す」
    埋め込みで解が保存されるため、free SAT at W ⟹ free SAT at W+2。
    よって昇順に解き、最初のSAT以降は解かずにSAT確定とする。
    UNSAT証明が重いセルはconflict予算で打ち切りstatus='unknown'。
    """
    t0 = time.time()
    results = {}
    n_done = 0
    total = len(targets) * len(depths) * len(widths)
    for name in targets:
        n_in, fn = TARGETS[name]
        target = TargetSpec(fn=fn)
        for depth in depths:
            free_known_sat = False
            for w in sorted(widths):
                topo = LutTopology(n_in=n_in, layer_widths=[w] * depth,
                                   readout='groupsum')
                cell_t0 = time.time()
                if free_known_sat:
                    free_status = 'sat'  # 単調性による確定(解かない)
                else:
                    free = can_realize(topo, target, wiring='free',
                                       conf_budget=conf_budget)
                    free_status = free.status
                    if free_status == 'sat':
                        free_known_sat = True
                fixed = [
                    can_realize(topo, target, wiring='fixed_random',
                                seed=s, conf_budget=conf_budget).status
                    for s in range(n_wiring_seeds)
                ]
                key = f'{name}_d{depth}_W{w}'
                n_done += 1
                results[key] = {
                    'target': name, 'depth': depth, 'width': w,
                    'free_status': free_status,
                    'free_sat': free_status == 'sat',
                    'fixed_sat_frac': float(np.mean(
                        [s == 'sat' for s in fixed])),
                    'fixed_unknown_frac': float(np.mean(
                        [s == 'unknown' for s in fixed])),
                    'n_wiring_seeds': n_wiring_seeds,
                    'solve_time_s': time.time() - cell_t0,
                }
                c = results[key]
                print(f'  {key:18s} free={free_status.upper():7s} '
                      f'fixed_sat={c["fixed_sat_frac"]:.2f} '
                      f'({time.time() - cell_t0:.1f}s) '
                      f'[{n_done}/{total}, 累計{time.time() - t0:.0f}s]',
                      flush=True)
    return results


def print_summary(results: dict, targets: list[str], depths: list[int],
                  widths: list[int]) -> None:
    print('\n' + '=' * 78)
    print('SAT容量マップ: セル = free配線判定 / 固定ランダム配線SAT率(配線税)')
    print('=' * 78)
    for name in targets:
        n_in = TARGETS[name][0]
        print(f'\n### {name} ({n_in}入力)')
        print('       ' + ''.join(f'  W={w:<9d}' for w in widths))
        for d in depths:
            row = []
            for w in widths:
                c = results[f'{name}_d{d}_W{w}']
                mark = {'sat': 'SAT  ', 'unsat': 'UNSAT',
                        'unknown': '?????'}[c.get('free_status',
                                                  'sat' if c['free_sat']
                                                  else 'unsat')]
                row.append(f'{mark}/{c["fixed_sat_frac"]:.2f}')
            print(f'  d={d}  ' + '  '.join(f'{r:11s}' for r in row))
    print('\n読み方: UNSAT=自由配線でも構造的に不可能(真の容量壁)。')
    print('SAT/x.xx: 表現可能だが固定ランダム配線ではx.xxの確率でしか配線が揃わない。')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--wiring-seeds', type=int, default=16)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        targets = ['xor3', 'maj3', 'parity3n6']
        depths, widths = [1, 2], [4, 8]
    else:
        targets = list(TARGETS)
        depths, widths = [1, 2, 3, 4], [2, 4, 8, 16, 32]

    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  '
          f'targets={targets} depth={depths} W={widths} '
          f'wiring_seeds={args.wiring_seeds}')
    print('-' * 78)
    results = run_grid(targets, depths, widths, args.wiring_seeds)
    print_summary(results, targets, depths, widths)

    out = args.out or 'sat_capacity_map_results.json'
    payload = {
        'meta': {'targets': targets, 'depths': depths, 'widths': widths,
                 'wiring_seeds': args.wiring_seeds,
                 'timestamp': datetime.now(JST).isoformat()},
        'results': results,
    }
    with open(out, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n結果JSON保存: {out}')
    print(f'終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
