"""表現ギャップを開けた最小系: under-capacity(単層)LUT網へ蒸留。

[XOR/1-LUT実験](distill_xor_mlp_to_1lut.py)は汎化/表現/能力の3ギャップが全て閉じ
「蒸留が知識を運ぶか」を測れなかった(≒DLGNトートロジー)。
本実験は**表現ギャップを意図的に開ける**: student を**単層**の2入力LUT網
(LutLayer(nin,W,n2) → GroupSum(2)) にする。単層+線形和は表現力が構造的に乏しく、
標的によっては厳密表現不能。**soft(連続緩和)は達成できてもhard(argmax離散化)で
関数が壊れうる** → 離散化ロバスト性が初めて非自明になる。

標的3種で「壁 vs 勾配ギャップ」を対比:
  - xor3(parity3): 単層では**表現の壁**。hard精度は約0.5(chance)に張り付く。
    「KDは無い容量を捏造できるか?」の否定対照。
  - maj3(多数決): **勾配的な容量ギャップ**。Wとともにhard天井が上昇、soft-hardギャップが縮む。
  - rnd4(4入力ランダム関数): 別の勾配的ギャップ(4入力=16点、汎化も少し絡む)。

各セルで測る: (1) soft acc − hard acc の離散化ギャップ、(2) KD lift =
hard_acc(α=0 or 0.5) − hard_acc(α=1.0 ラベルのみ) = 容量制約下で teacher soft が
hard精度を上乗せするか(表現ギャップ版「蒸留は知識を運ぶか」)。

standardizeは2クラス病理(§2)を踏まえoff、T=2固定、teacherは強MLP(100%到達)。

Usage:
    uv run python experiments/distill_undercap_3input.py            # 3標的, W sweep, 16 seeds
    uv run python experiments/distill_undercap_3input.py --seeds 4 --quick

wandb不使用(ローカルのみ)。結果はstdout整形表 + JSON保存。
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

from griffith_logic.convert.distill import make_kd_loss
from griffith_logic.export import export_ir
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir import simulate
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))

# 4入力ランダム関数の固定真理値表(index = a*8+b*4+c*2+d、8/16が1でバランス)。
_RND4_TT = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]

# 標的: name -> (n_inputs, label_fn(bits_tuple)->int)
TARGETS = {
    'xor3': (3, lambda b: b[0] ^ b[1] ^ b[2]),
    'maj3': (3, lambda b: int(sum(b) >= 2)),
    'rnd4': (4, lambda b: _RND4_TT[b[0] * 8 + b[1] * 4 + b[2] * 2 + b[3]]),
}


def target_data(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """標的の全2^nパターン。X:(2^n, n) float, Y:(2^n,) long。"""
    nin, fn = TARGETS[name]
    x = torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=nin)), dtype=torch.float32,
    )
    y = torch.tensor([fn(tuple(int(v) for v in row)) for row in x.tolist()],
                     dtype=torch.long)
    return x, y


class TeacherMLP(nn.Module):
    """強teacher: nin -> H -> H -> 2。"""

    def __init__(self, nin: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_teacher(name: str, seed: int, *, max_steps: int = 5000, lr: float = 0.03,
                  retries: int = 8) -> tuple[TeacherMLP, dict]:
    """標的を100% hard accまで学習(強MLPは全標的で容易)。"""
    nin, _ = TARGETS[name]
    x, y = target_data(name)
    for r in range(retries):
        torch.manual_seed(seed * 1000 + r)
        teacher = TeacherMLP(nin)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        for _ in range(max_steps):
            teacher.train()
            opt.zero_grad()
            F.cross_entropy(teacher(x), y).backward()
            opt.step()
        teacher.eval()
        with torch.no_grad():
            acc = (teacher(x).argmax(-1) == y).float().mean().item()
        if acc == 1.0:
            return teacher, {'reached': True, 'hard_acc': acc}
    return teacher, {'reached': False, 'hard_acc': acc}


def make_student(name: str, width: int, seed: int) -> nn.Sequential:
    """**単層**の2入力LUT網: LutLayer(nin, W) -> GroupSum(2)。表現力を意図的に制限。"""
    nin, _ = TARGETS[name]
    torch.manual_seed(seed)
    tau = max(1.0, (width // 2) ** 0.5)
    return nn.Sequential(
        LutLayer(nin, width, n_inputs=2, parameterization='softmax'),
        GroupSum(k=2, tau=tau),
    )


def distill(teacher: TeacherMLP, name: str, width: int, alpha: float, seed: int,
            *, temperature: float = 2.0, max_steps: int = 4000, lr: float = 0.03,
            check_every: int = 100) -> dict:
    """under-capacity(単層)studentを make_kd_loss(std=off) で蒸留。"""
    x, y = target_data(name)
    with torch.no_grad():
        teacher_logits = teacher(x)

    model = make_student(name, width, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_kd_loss(alpha=alpha, temperature=temperature,
                           standardize=False, kind='kl')

    best_hard = 0.0
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad()
        loss_fn(model(x), y, teacher_logits).backward()
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
        hard_acc = (hard_logits.argmax(-1) == y).float().mean().item()
    model.train()
    with torch.no_grad():
        soft_acc = (model(x).argmax(-1) == y).float().mean().item()
    model.eval()

    return {
        'soft_acc': soft_acc,
        'hard_acc': hard_acc,
        'best_hard': best_hard,
        'disc_gap_acc': soft_acc - hard_acc,  # ハード化で失う精度
        'ir_match': _check_ir(model, name, x, hard_logits),
    }


def _check_ir(model: nn.Sequential, name: str, x: torch.Tensor,
              hard_logits: torch.Tensor) -> bool:
    """torch hard eval == IR simulate(argmax一致)。"""
    nin, _ = TARGETS[name]
    try:
        ir = export_ir(model, input_width=nin, name=f'undercap_{name}')
        scores = simulate(ir, x.round().numpy().astype(np.int64))
        return bool((hard_logits.numpy().argmax(-1) == scores.argmax(-1)).all())
    except Exception:
        return False


def _fmt_eta(done: int, total: int, t0: float) -> str:
    elapsed = time.time() - t0
    remain = elapsed / max(done, 1) * (total - done)
    eta = datetime.now(JST) + timedelta(seconds=remain)

    def hms(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f'{m}m{sec:02d}s'

    return f'[{hms(elapsed)}<{hms(remain)}, 終了予定 {eta:%H:%M:%S}]'


def run_grid(seeds: list[int], targets: list[str], widths: list[int],
             alphas: list[float]) -> dict:
    cells = list(itertools.product(targets, widths, alphas))
    total = len(cells) * len(seeds)
    t0, done = time.time(), 0

    teacher_cache: dict[tuple[str, int], tuple[TeacherMLP, dict]] = {}

    def get_teacher(name: str, seed: int):
        key = (name, seed)
        if key not in teacher_cache:
            teacher_cache[key] = train_teacher(name, seed)
        return teacher_cache[key]

    results = {}
    teacher_reach = {}
    for name, W, alpha in cells:
        runs = []
        treach = []
        for seed in seeds:
            teacher, tinfo = get_teacher(name, seed)
            treach.append(tinfo['reached'])
            runs.append(distill(teacher, name, W, alpha, seed))
            done += 1
        n = len(runs)
        key = f'{name}_W{W}_a{alpha}'
        results[key] = {
            'target': name, 'width': W, 'alpha': alpha, 'n': n,
            'mean_hard_acc': float(np.mean([r['hard_acc'] for r in runs])),
            'mean_soft_acc': float(np.mean([r['soft_acc'] for r in runs])),
            'mean_disc_gap': float(np.mean([r['disc_gap_acc'] for r in runs])),
            'max_hard_acc': float(np.max([r['hard_acc'] for r in runs])),
            'ir_match_rate': sum(r['ir_match'] for r in runs) / n,
        }
        teacher_reach[name] = float(np.mean(treach))
        c = results[key]
        print(f'  {key:16s} hard={c["mean_hard_acc"]:.3f} soft={c["mean_soft_acc"]:.3f} '
              f'gap={c["mean_disc_gap"]:.3f} IR={c["ir_match_rate"]:.0%}  '
              f'{_fmt_eta(done, total, t0)}')
    results['_teacher_reach'] = teacher_reach
    return results


def print_summary(results: dict) -> None:
    print('\n' + '=' * 82)
    print('集計: 標的×幅W×α — hard精度天井 / soft-hard離散化ギャップ / KD lift')
    print('=' * 82)
    treach = results.get('_teacher_reach', {})

    targets = sorted({c['target'] for k, c in results.items() if k != '_teacher_reach'})
    widths = sorted({c['width'] for k, c in results.items() if k != '_teacher_reach'})
    alphas = sorted({c['alpha'] for k, c in results.items() if k != '_teacher_reach'})

    for name in targets:
        print(f'\n### 標的={name}  (teacher 100%到達率={treach.get(name, 0):.0%})')
        print('  hard_acc(平均):     ' + ''.join(f'α={a:<5}' for a in alphas)
              + '  | soft-hard gap')
        for W in widths:
            hards = [results[f'{name}_W{W}_a{a}']['mean_hard_acc'] for a in alphas]
            gaps = [results[f'{name}_W{W}_a{a}']['mean_disc_gap'] for a in alphas]
            hrow = ''.join(f'{h:.3f}  ' for h in hards)
            grow = ' '.join(f'{g:.3f}' for g in gaps)
            print(f'    W={W:<3d} {hrow} | {grow}')
        print('  KD lift (hard[α]−hard[α=1.0], 正=蒸留が容量制約下で上乗せ):')
        for W in widths:
            base = results[f'{name}_W{W}_a1.0']['mean_hard_acc']
            lifts = [f'α={a}:{results[f"{name}_W{W}_a{a}"]["mean_hard_acc"] - base:+.3f}'
                     for a in alphas if a != 1.0]
            print(f'    W={W:<3d} (label-only={base:.3f})  ' + '  '.join(lifts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--quick', action='store_true',
                        help='短縮: targets=[maj3], W=[2,8], α=[0,1]')
    parser.add_argument('--out', type=str, default=None,
                        help='結果JSON保存先(既定=results/distill_undercap_3input_results.json)')
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    if args.quick:
        targets, widths, alphas = ['maj3'], [2, 8], [0.0, 1.0]
    else:
        targets, widths, alphas = ['xor3', 'maj3', 'rnd4'], [2, 4, 8, 16], [0.0, 0.5, 1.0]

    n = len(seeds) * len(targets) * len(widths) * len(alphas)
    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  seeds={len(seeds)} '
          f'targets={targets} W={widths} α={alphas} (単層, T=2, std=off)  (全{n} runs)')
    print('-' * 82)

    results = run_grid(seeds, targets, widths, alphas)
    print_summary(results)

    out = args.out or 'results/distill_undercap_3input_results.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'meta': {'seeds': len(seeds), 'targets': targets, 'widths': widths,
                 'alphas': alphas, 'temperature': 2.0, 'standardize': False,
                 'student': 'single-layer LutLayer->GroupSum',
                 'timestamp': datetime.now(JST).isoformat()},
        'results': results,
    }
    try:
        with open(out, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'\n結果JSON保存: {out}')
    except OSError as e:
        print(f'\n[warn] JSON保存失敗: {e}')
    print(f'\n終了 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST')


if __name__ == '__main__':
    main()
