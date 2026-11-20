"""容量十分なサイズでラベルフリー蒸留(α=0)も学習しきるか検証。

[容量ラダー](verify_capacity_ladder.py)は「表現可能なサイズなら**label-only CE**で
hard acc 100%到達」を示した。[under-capacity実験](distill_undercap_3input.py)は
「**単層(容量不足)**ではα(label/teacher)を変えても天井が動かない」を示した。
だが「**容量十分なサイズで α=0.0(teacher soft のみ、ground truth不使用)**」は
未検証。本実験はそこを埋める: 容量十分な深さ×幅で α∈{0.0,0.5,1.0} を比較し、
ラベルフリー(α=0)も100%到達するかを見る。

予測: teacherが100%正確で student容量が十分なら、teacher softへの一致=全域で
関数一致 → ラベルフリーでも100%(XOR最小系の結論の深い版)。要検証。

student: LutLayer(nin,W)×depth → GroupSum(2)。容量十分域(depth∈{2,3}, W=16)。
teacher: 強MLP(100%到達)。make_kd_loss(std=off, T=2)。

Usage:
    uv run python experiments/verify_labelfree_at_capacity.py            # 3標的×depth×α, 16 seeds
    uv run python experiments/verify_labelfree_at_capacity.py --seeds 4  # 短縮

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

# 標的定義とteacher学習は既存実験から流用
from distill_undercap_3input import TARGETS, target_data, train_teacher  # noqa: E402

from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.export import export_ir
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir import simulate
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))


def make_net(nin: int, depth: int, width: int, seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    gf = 1.0 if depth == 1 else 2.0
    layers: list[nn.Module] = []
    d = nin
    for _ in range(depth):
        layers.append(LutLayer(d, width, n_inputs=2, parameterization='softmax',
                               grad_factor=gf))
        d = width
    layers.append(GroupSum(k=2, tau=max(1.0, (width // 2) ** 0.5)))
    return nn.Sequential(*layers)


def distill(teacher, name: str, depth: int, width: int, alpha: float, seed: int,
            *, temperature: float = 2.0, max_steps: int = 5000, lr: float = 0.02) -> dict:
    """容量十分studentを make_kd_loss(std=off) で蒸留。α=0はground truth不使用。"""
    nin, _ = TARGETS[name]
    x, y = target_data(name)
    with torch.no_grad():
        teacher_logits = teacher(x)
    model = make_net(nin, depth, width, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_kd_loss(alpha=alpha, temperature=temperature,
                           standardize=False, kind='kl')
    for _ in range(max_steps):
        model.train()
        opt.zero_grad()
        loss_fn(model(x), y, teacher_logits).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        hl = model(x.round())
        hard = (hl.argmax(-1) == y).float().mean().item()
    ir_ok = True
    try:
        ir = export_ir(model, input_width=nin, name=f'lf_{name}')
        sc = simulate(ir, x.round().numpy().astype(np.int64))
        ir_ok = bool((hl.numpy().argmax(-1) == sc.argmax(-1)).all())
    except Exception:
        ir_ok = False
    return {'hard': hard, 'ir': ir_ok}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--out', type=str, default=None,
                        help='結果JSON保存先(既定=results/verify_labelfree_at_capacity_results.json)')
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    targets = ['xor3', 'maj3', 'rnd4']
    depths = [2, 3]
    width = 16
    alphas = [0.0, 0.5, 1.0]
    cells = list(itertools.product(targets, depths, alphas))
    total = len(cells) * len(seeds)
    t0, done = time.time(), 0

    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  容量十分(W={width}) '
          f'seeds={len(seeds)} targets={targets} depth={depths} α={alphas} '
          f'(std=off, T=2)  (全{total} runs)')
    print('-' * 78)

    tcache: dict = {}

    def get_teacher(name, seed):
        if (name, seed) not in tcache:
            tcache[(name, seed)] = train_teacher(name, seed)
        return tcache[(name, seed)]

    results = {}
    for name, depth, alpha in cells:
        runs = []
        for s in seeds:
            teacher, _ = get_teacher(name, s)
            runs.append(distill(teacher, name, depth, width, alpha, s))
            done += 1
        n = len(runs)
        key = f'{name}_d{depth}_a{alpha}'
        results[key] = {
            'target': name, 'depth': depth, 'alpha': alpha, 'n': n,
            'reach100': sum(r['hard'] == 1.0 for r in runs) / n,
            'mean_hard': float(np.mean([r['hard'] for r in runs])),
            'ir_rate': sum(r['ir'] for r in runs) / n,
        }
        c = results[key]
        eta = datetime.now(JST) + timedelta(seconds=(time.time() - t0) / done * (total - done))
        print(f'  {key:16s} reach100={c["reach100"]:5.0%} meanHard={c["mean_hard"]:.3f} '
              f'IR={c["ir_rate"]:.0%}  [終了予定 {eta:%H:%M:%S}]')

    print('\n' + '=' * 78)
    print('容量十分域: ラベルフリー(α=0.0) vs ラベル(α=1.0) の hard 100%到達率')
    print('=' * 78)
    for name in targets:
        print(f'\n### {name}')
        print(f'{"":8s}' + ''.join(f'α={a:<6}' for a in alphas))
        for depth in depths:
            row = ''.join(f'{results[f"{name}_d{depth}_a{a}"]["reach100"]:4.0%}   '
                          for a in alphas)
            print(f'  d={depth}   {row}')
        print('  (各セル = hard 100%到達率, over seeds)')

    out = args.out or 'results/verify_labelfree_at_capacity_results.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, 'w') as f:
            json.dump({'meta': {'seeds': len(seeds), 'targets': targets,
                                'depths': depths, 'width': width, 'alphas': alphas,
                                'timestamp': datetime.now(JST).isoformat()},
                       'results': results}, f, ensure_ascii=False, indent=2)
        print(f'\n結果JSON保存: {out}')
    except OSError as e:
        print(f'\n[warn] JSON保存失敗: {e}')
    print(f'\n終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
