"""容量ラダー検証: 構造的に表現可能なサイズならLUT網は標的を学習しきるか。

[under-capacity実験](distill_undercap_3input.py)で単層LUT網はxor3(parity3)を
chance 0.5でしか解けなかった。これが「表現の壁(容量不足)」なのか「訓練/最適化の
失敗(バグ)」なのかを切り分けるための検証。**teacherもKDも使わず**、素のCE
(label-only)で depth×width を上げていき、各標的が hard acc 100% に到達することを
確認する。到達すれば単層0.5は真に容量壁であり、パイプラインは容量が足りれば
ちゃんと学習する、と言える。

student: LutLayer(nin,W) を depth段 → GroupSum(2)。depth∈{1,2,3}, W∈{4,16}。

Usage:
    uv run python experiments/verify_capacity_ladder.py            # 3標的×depth×W, 16 seeds
    uv run python experiments/verify_capacity_ladder.py --seeds 4  # 短縮

wandb不使用。結果はstdout整形表 + JSON保存。
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# under-capacity実験と同一の標的定義を流用
from distill_undercap_3input import TARGETS, target_data  # noqa: E402

from griffith_logic.export import export_ir
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir import simulate
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))


def make_net(nin: int, depth: int, width: int, seed: int) -> nn.Sequential:
    """depth段の2入力LUT網 → GroupSum(2)。深い段はgrad_factorを上げて安定化。"""
    torch.manual_seed(seed)
    gf = 1.0 if depth == 1 else 2.0  # 深い網の勾配減衰対策(LutLayer既定の推奨)
    layers: list[nn.Module] = []
    d = nin
    for _ in range(depth):
        layers.append(LutLayer(d, width, n_inputs=2, parameterization='softmax',
                               grad_factor=gf))
        d = width
    layers.append(GroupSum(k=2, tau=max(1.0, (width // 2) ** 0.5)))
    return nn.Sequential(*layers)


def train_eval(name: str, depth: int, width: int, seed: int,
               *, max_steps: int = 6000, lr: float = 0.02,
               check_every: int = 100) -> dict:
    """label-only CE で学習し、hard/soft acc と best hard、IR一致を返す。"""
    nin, _ = TARGETS[name]
    x, y = target_data(name)
    model = make_net(nin, depth, width, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_hard = 0.0
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        if step % check_every == 0:
            model.eval()
            with torch.no_grad():
                h = (model(x.round()).argmax(-1) == y).float().mean().item()
            best_hard = max(best_hard, h)
            model.train()

    model.eval()
    with torch.no_grad():
        hard_logits = model(x.round())
        hard = (hard_logits.argmax(-1) == y).float().mean().item()
    model.train()
    with torch.no_grad():
        soft = (model(x).argmax(-1) == y).float().mean().item()
    model.eval()

    ir_ok = True
    try:
        ir = export_ir(model, input_width=nin, name=f'ladder_{name}')
        sc = simulate(ir, x.round().numpy().astype(np.int64))
        ir_ok = bool((hard_logits.numpy().argmax(-1) == sc.argmax(-1)).all())
    except Exception:
        ir_ok = False

    return {'hard': hard, 'soft': soft, 'best_hard': best_hard, 'ir': ir_ok}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--out', type=str, default=None,
                        help='結果JSON保存先(既定=results/verify_capacity_ladder_results.json)')
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    targets = ['xor3', 'maj3', 'rnd4']
    depths = [1, 2, 3]
    widths = [4, 16]
    cells = list(itertools.product(targets, depths, widths))
    total = len(cells) * len(seeds)
    t0, done = time.time(), 0

    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  label-only CE  '
          f'seeds={len(seeds)} targets={targets} depth={depths} W={widths} '
          f'(全{total} runs)')
    print('-' * 78)

    results = {}
    for name, depth, W in cells:
        runs = [train_eval(name, depth, W, s) for s in seeds]
        done += len(seeds)
        n = len(runs)
        key = f'{name}_d{depth}_W{W}'
        results[key] = {
            'target': name, 'depth': depth, 'width': W, 'n': n,
            'reach100': sum(r['hard'] == 1.0 for r in runs) / n,
            'mean_hard': float(np.mean([r['hard'] for r in runs])),
            'max_hard': float(np.max([r['hard'] for r in runs])),
            'mean_soft': float(np.mean([r['soft'] for r in runs])),
            'best_reach100': sum(r['best_hard'] == 1.0 for r in runs) / n,
            'ir_rate': sum(r['ir'] for r in runs) / n,
        }
        c = results[key]
        elapsed = time.time() - t0
        eta = datetime.now(JST) + timedelta(seconds=elapsed / done * (total - done))
        print(f'  {key:16s} reach100={c["reach100"]:5.0%} (best {c["best_reach100"]:4.0%}) '
              f'meanHard={c["mean_hard"]:.3f} maxHard={c["max_hard"]:.3f} '
              f'IR={c["ir_rate"]:.0%}  [終了予定 {eta:%H:%M:%S}]')

    print('\n' + '=' * 78)
    print('容量ラダー: 各標的が depth×W を上げると hard acc 100% に到達するか')
    print('(reach100=最終100%到達率, best=学習中に一度でも100%到達した率)')
    print('=' * 78)
    for name in targets:
        nin = TARGETS[name][0]
        print(f'\n### {name} ({nin}入力)')
        print(f'{"":6s}' + ''.join(f'  W={w:<3d}' for w in widths))
        for depth in depths:
            cells_row = []
            for W in widths:
                c = results[f'{name}_d{depth}_W{W}']
                cells_row.append(f'{c["reach100"]:3.0%}/{c["best_reach100"]:3.0%}')
            print(f'  d={depth}   ' + '  '.join(cells_row))
        print('  (各セル = 最終100%到達率 / best100%到達率, over seeds)')

    out = args.out or 'results/verify_capacity_ladder_results.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, 'w') as f:
            json.dump({'meta': {'seeds': len(seeds), 'targets': targets,
                                'depths': depths, 'widths': widths,
                                'timestamp': datetime.now(JST).isoformat()},
                       'results': results}, f, ensure_ascii=False, indent=2)
        print(f'\n結果JSON保存: {out}')
    except OSError as e:
        print(f'\n[warn] JSON保存失敗: {e}')
    print(f'\n終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
