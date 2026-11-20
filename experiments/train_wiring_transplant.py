"""配線4腕比較: UNSAT配線の回避とwitness配線転写の効果(③+配線ガード)。

配線税セル(parity-in-noise)で以下の4腕を比較する:
    fixed            — 固定ランダム配線(ベースライン、配線税をそのまま被る)
    guard            — SATガード配線: find_sat_wiringでUNSAT配線を棄却して
                       固定(「UNSATになる配線をしない」の直接実装)
    learnable        — DWN式LearnableMapping(random init)+strictペアLutLayer
    transplant       — 自由配線SAT witnessの配線をLearnableMappingの
                       one-hot初期値に転写(learnableのwitness init)

さらにlearnable/transplantは**収束後のargmax配線をcan_realizeで事後判定**し、
「学習配線はSAT集合に入るか」を直接測定する(DWNが配線税を回避できて
いるかの検証)。

訓練: {soft, hard_st} × 8seed。全パターン列挙、hard acc到達で早期終了。

Usage:
    uv run python experiments/train_wiring_transplant.py --quick
    uv run python experiments/train_wiring_transplant.py --seeds 8

wandb不使用。結果はstdout+JSON保存。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat_capacity_map import TARGETS
from sat_vs_training_map import hard_acc, target_xy

from griffith_logic.analysis.sat_capacity import (
    LutTopology,
    TargetSpec,
    can_realize,
    find_sat_wiring,
)
from griffith_logic.group_sum import GroupSum
from griffith_logic.layers import LutLayer
from griffith_logic.layers.dwn import LearnableMapping

JST = timezone(timedelta(hours=9))

ARMS = ('fixed', 'guard', 'learnable', 'transplant')


def _strict_pair_indices(width: int) -> torch.Tensor:
    """LearnableMapping出力(2W)をLutLayerの(pin0,pin1)=(2i,2i+1)で読む配線。"""
    return torch.stack([2 * torch.arange(width), 2 * torch.arange(width) + 1])


class MappedLutNet(nn.Module):
    """LearnableMapping → strictペアLutLayer ×depth → GroupSum。

    層lのLearnableMappingが「前層出力→2W本のピン信号」の配線を学習し、
    LutLayerは固定strictペアで読む(= 配線学習の自由度をMapping側へ集約)。
    """

    def __init__(self, n_in: int, depth: int, width: int, *,
                 selection: str = 'soft', tau: float | None = None):
        super().__init__()
        self.mappings = nn.ModuleList()
        self.luts = nn.ModuleList()
        d = n_in
        for _ in range(depth):
            self.mappings.append(LearnableMapping(d, 2 * width))
            self.luts.append(LutLayer(
                2 * width, width, n_inputs=2, parameterization='softmax',
                connections=_strict_pair_indices(width),
                selection=selection,
            ))
            d = width
        self.group_sum = GroupSum(
            k=2, tau=tau or max(1.0, (width // 2) ** 0.5))

    def forward(self, x):
        for mp, lut in zip(self.mappings, self.luts):
            x = lut(mp(x))
        return self.group_sum(x)

    @torch.no_grad()
    def argmax_wiring(self) -> list[np.ndarray]:
        """学習配線のargmax(層ごと(2, W))を返す。"""
        out = []
        for mp, lut in zip(self.mappings, self.luts):
            # LearnableMapping.weights: (in_features, out_features=2W)
            # 列qのargmaxが「ピンqが読む前層index」
            sel = mp.weights.argmax(0).cpu().numpy()  # (2W,)
            w = lut.out_dim
            out.append(np.stack([sel[0::2], sel[1::2]])[:, :w])
        return out

    @torch.no_grad()
    def init_from_wiring(self, wiring: list[np.ndarray], logit: float = 6.0):
        """witness配線をone-hot(強logit)初期値として転写する。"""
        for mp, lut, wr in zip(self.mappings, self.luts, wiring):
            w = torch.full_like(mp.weights, 0.0)
            for o in range(wr.shape[1]):
                w[int(wr[0][o]), 2 * o] = logit
                w[int(wr[1][o]), 2 * o + 1] = logit
            mp.weights.copy_(w)


def make_fixed_net(n_in, depth, width, wiring, selection):
    layers: list[nn.Module] = []
    d = n_in
    for li in range(depth):
        layers.append(LutLayer(
            d, width, n_inputs=2, parameterization='softmax',
            connections=torch.as_tensor(wiring[li]), selection=selection,
            grad_factor=1.0 if depth == 1 else 2.0,
        ))
        d = width
    layers.append(GroupSum(k=2, tau=max(1.0, (width // 2) ** 0.5)))
    return nn.Sequential(*layers)


def train_arm(arm, name, depth, width, seed, selection, *,
              max_steps=6000, lr=0.02, check_every=100):
    n_in, fn = TARGETS[name]
    x, y = target_xy(name)
    target = TargetSpec(fn=fn)
    topo = LutTopology(n_in=n_in, layer_widths=[width] * depth,
                       readout='groupsum')

    torch.manual_seed(seed)
    info = {}
    if arm == 'fixed':
        from griffith_logic.analysis.sat_capacity import _fixed_wiring
        wiring = _fixed_wiring(topo, seed)
        net = make_fixed_net(n_in, depth, width, wiring, selection)
    elif arm == 'guard':
        g = find_sat_wiring(topo, target, max_tries=32, seed0=seed * 100)
        info['guard_tried'] = g['tried']
        if g['wiring'] is None:
            return {'hard': 0.0, 'best_hard': 0.0, 'guard_failed': True,
                    **info}
        torch.manual_seed(seed)
        net = make_fixed_net(n_in, depth, width, g['wiring'], selection)
    elif arm in ('learnable', 'transplant'):
        net = MappedLutNet(n_in, depth, width, selection=selection)
        if arm == 'transplant':
            r = can_realize(topo, target, wiring='free')
            if r.status != 'sat':
                return {'hard': 0.0, 'best_hard': 0.0,
                        'no_free_witness': True}
            net.init_from_wiring(r.wiring)
    else:
        raise ValueError(arm)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = 0.0
    for step in range(1, max_steps + 1):
        net.train()
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
        if step % check_every == 0:
            h = hard_acc(net, x, y)
            best = max(best, h)
            if h == 1.0:
                break
    res = {'hard': hard_acc(net, x, y), 'best_hard': best, **info}

    # 学習配線の事後SAT判定(learnable系のみ)
    if arm in ('learnable', 'transplant'):
        learned = net.argmax_wiring()
        chk = can_realize(topo, target, wiring=learned, verify=False)
        res['learned_wiring_sat'] = chk.status == 'sat'
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    cells = [('parity3n6', 2, 8), ('parity3n8', 2, 8), ('parity3n8', 2, 16)]
    selections = ['soft', 'hard_st']
    max_steps = args.max_steps
    if args.quick:
        cells, seeds, max_steps = [('parity3n6', 2, 8)], seeds[:4], 2000

    total = len(cells) * len(ARMS) * len(selections) * len(seeds)
    print(f'開始 {datetime.now(JST):%H:%M:%S}  {len(cells)}セル×{ARMS}×'
          f'{selections}×{len(seeds)}seed (全{total})', flush=True)

    t0, done = time.time(), 0
    results = {}
    for name, depth, width in cells:
        for sel in selections:
            for arm in ARMS:
                runs = []
                for seed in seeds:
                    runs.append(train_arm(arm, name, depth, width, seed, sel,
                                          max_steps=max_steps))
                    done += 1
                key = f'{name}_d{depth}_W{width}_{sel}_{arm}'
                reach = float(np.mean([r['best_hard'] == 1.0 for r in runs]))
                wsat = [r.get('learned_wiring_sat') for r in runs
                        if 'learned_wiring_sat' in r]
                results[key] = {
                    'cell': f'{name}_d{depth}_W{width}', 'selection': sel,
                    'arm': arm, 'n': len(runs),
                    'reach100': reach,
                    'mean_hard': float(np.mean([r['hard'] for r in runs])),
                    'learned_wiring_sat_frac': (
                        float(np.mean(wsat)) if wsat else None),
                    'runs': runs,
                }
                c = results[key]
                elapsed = time.time() - t0
                eta = datetime.now(JST) + timedelta(
                    seconds=elapsed / done * (total - done))
                ws = (f' wiringSAT={c["learned_wiring_sat_frac"]:.2f}'
                      if c['learned_wiring_sat_frac'] is not None else '')
                print(f'  {key:44s} reach100={reach:4.0%} '
                      f'hard={c["mean_hard"]:.3f}{ws}  '
                      f'[終了予定 {eta:%H:%M:%S}]', flush=True)

    out = args.out or 'train_wiring_transplant_results.json'
    with open(out, 'w') as f:
        json.dump({'meta': {'seeds': len(seeds), 'max_steps': max_steps,
                            'timestamp': datetime.now(JST).isoformat()},
                   'results': results}, f, ensure_ascii=False, indent=2)
    print(f'結果JSON保存: {out}')


if __name__ == '__main__':
    main()
