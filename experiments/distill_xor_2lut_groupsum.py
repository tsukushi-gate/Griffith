"""XOR: MLP teacher → 2-LUT + GroupSum(多クラス) student 蒸留実験。

[distill_xor_mlp_to_1lut.py](distill_xor_mlp_to_1lut.py) の binary 1-LUT 版に対する
「多クラス/GroupSum頭」版。既存の蒸留インフラ(convert/distill.py の make_kd_loss、
GroupSum、export_ir/simulate)を**無改造で流用**し、binary版では出なかった2軸を検証する:

  1. 二重温度 τ_gs × T: student は GroupSum(tau=τ_gs) で既にモデル側軟化が掛かる。
     KD側温度 T と合わせ実効 τ_gs×T(convert/distill.py docstring)。τ_gs はレシピ固定、
     T をスイープするのが想定運用。ここでは τ_gs=1.0 固定で純 T 効果を見る。
  2. Logit Standardization(arXiv:2403.01427): teacher logit と student(GroupSum)logit は
     スケールが無関係。standardize=True でサンプルごとZ-score正規化してから比較する
     make_kd_loss のオプションの効果を on/off で対照。

student = LutLayer(2,2,n_inputs=2,softmax) + GroupSum(k=2, tau). 2つのLUTがそれぞれ
1クラスを駆動する(score[c]=LUT[c]/τ)。XOR分類の収束形は (LUT0,LUT1)=(XNOR,XOR)=
(1001, 0110)。ハード分類精度=100%到達をXOR成立の判定に使う。

Usage:
    uv run python experiments/distill_xor_2lut_groupsum.py            # 16 seeds, H×α×T×std grid
    uv run python experiments/distill_xor_2lut_groupsum.py --seeds 4 --quick

wandbは使わない(ローカルのみ)。結果はstdout整形表 + JSON保存。
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
from griffith_logic.export import export_ir, truth_tables_from_lut_layer
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir import simulate
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))

# 収束形: LUT0=XNOR(1001=9, 00と11で発火), LUT1=XOR(0110=6, 01と10で発火)。
# GroupSum(k=2)で class0=LUT0, class1=LUT1 → argmaxが正しくXORを分類。
XOR_PAIR = (0b1001, 0b0110)  # (9, 6)
TAU_GS = 1.0  # τ_gs: GroupSum温度(レシピ固定。二重温度のうちモデル側)


def xor_data() -> tuple[torch.Tensor, torch.Tensor]:
    """全4パターンのXOR。X:(4,2) float, Y:(4,) long in {0,1}。"""
    x = torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=2)), dtype=torch.float32,
    )
    y = (x[:, 0].int() ^ x[:, 1].int()).long()
    return x, y


class TeacherMLP(nn.Module):
    """XOR teacher: 2 -> H -> 2(2クラスロジット)。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, 2)


def train_teacher(
    hidden: int, seed: int, *, max_steps: int = 4000, lr: float = 0.05,
    retries: int = 8,
) -> tuple[TeacherMLP, dict]:
    """XOR teacherを100% hard accまで学習。未達なら再シードでリトライ。"""
    x, y = xor_data()
    for r in range(retries):
        s = seed * 1000 + r
        torch.manual_seed(s)
        teacher = TeacherMLP(hidden)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        for _ in range(max_steps):
            teacher.train()
            opt.zero_grad()
            loss = F.cross_entropy(teacher(x), y)
            loss.backward()
            opt.step()
        teacher.eval()
        with torch.no_grad():
            logits = teacher(x)
            hard_acc = (logits.argmax(-1) == y).float().mean().item()
            probs = F.softmax(logits, dim=-1)
            conf = (probs.max(-1).values - 0.5).abs().mean().item() * 2
        if hard_acc == 1.0:
            return teacher, {'hard_acc': hard_acc, 'reached': True,
                             'seed_used': s, 'conf': conf}
    return teacher, {'hard_acc': hard_acc, 'reached': False,
                     'seed_used': s, 'conf': conf}


def make_student(seed: int) -> nn.Sequential:
    """2-LUT + GroupSum student。両LUTが入力[0,1]を見るよう接続固定。"""
    torch.manual_seed(seed)
    lut = LutLayer(2, 2, n_inputs=2, parameterization='softmax')
    # indices: (n_inputs, out_dim) = (2,2)。slot0->入力0, slot1->入力1(両LUT共通)。
    lut.indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    return nn.Sequential(lut, GroupSum(k=2, tau=TAU_GS))


def lut_gate_margins(lut: LutLayer) -> float:
    """2つのLUTのゲート信頼マージン(top1-top2 softmax)の最小値。"""
    with torch.no_grad():
        p = F.softmax(lut.weights, dim=-1)  # (2, 16)
        top2 = torch.topk(p, 2, dim=-1).values  # (2, 2)
        margins = (top2[:, 0] - top2[:, 1])  # (2,)
        return margins.min().item()


def truth_pair(lut: LutLayer) -> tuple[int, int]:
    """2つのLUTのハード真理値表(int, int)。"""
    tt = truth_tables_from_lut_layer(lut)
    return int(tt[0]), int(tt[1])


def distill(
    teacher: TeacherMLP, temperature: float, alpha: float, standardize: bool,
    seed: int, *, max_steps: int = 3000, lr: float = 0.05, check_every: int = 50,
) -> dict:
    """2-LUT+GroupSum studentを make_kd_loss で蒸留する。

    make_kd_loss(alpha, T, standardize, kind='kl') が返す
        loss(student_logits, labels, teacher_logits)
    = alpha*CE(student, labels) + (1-alpha)*KL(softmax(t/T)||softmax(s/T))*T^2。
    alpha=0.0 が完全ラベルフリー(labelsは損失に寄与しない)。
    """
    x, y = xor_data()
    with torch.no_grad():
        teacher_logits = teacher(x)  # (4, 2)

    model = make_student(seed)
    lut = model[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_kd_loss(
        alpha=alpha, temperature=temperature,
        standardize=standardize, kind='kl',
    )

    def hard_acc() -> float:
        model.eval()
        with torch.no_grad():
            acc = (model(x.round()).argmax(-1) == y).float().mean().item()
        model.train()
        return acc

    conv_step = None
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad()
        student_logits = model(x)  # (4, 2)
        loss = loss_fn(student_logits, y, teacher_logits)
        loss.backward()
        opt.step()
        if step % check_every == 0 and conv_step is None:
            if hard_acc() == 1.0:
                conv_step = step

    # --- 検証 ---
    model.eval()
    with torch.no_grad():
        hard_logits = model(x.round())
    h_acc = (hard_logits.argmax(-1) == y).float().mean().item()
    reached = (h_acc == 1.0)
    pair = truth_pair(lut)
    is_xor_pair = (pair == XOR_PAIR)
    margin = lut_gate_margins(lut)

    # soft(train) vs hard(eval) 離散化ギャップ(logit空間)
    model.train()
    with torch.no_grad():
        soft_logits = model(x)
    disc_gap = (soft_logits - hard_logits).abs().max().item()
    model.eval()

    # 三段一致: torch hard eval == IR simulate(GroupSum終端=整数カウント)
    ir_match, ir_issues = _check_ir_equivalence(model, x, hard_logits)

    return {
        'hard_acc': h_acc,
        'reached': reached,
        'pair': f'{pair[0]:04b}/{pair[1]:04b}',
        'is_xor_pair': is_xor_pair,
        'margin': margin,
        'conv_step': conv_step,
        'disc_gap': disc_gap,
        'ir_match': ir_match,
        'ir_issues': ir_issues,
    }


def _check_ir_equivalence(
    model: nn.Sequential, x: torch.Tensor, hard_logits: torch.Tensor,
) -> tuple[bool, list[str]]:
    """torch hard eval == IR simulate を照合(GroupSum: 整数カウント + argmax)。"""
    try:
        ir = export_ir(model, input_width=2, name='xor_2lut_gs')
        x_np = x.round().numpy().astype(np.int64)
        scores = simulate(ir, x_np)  # (batch, k) 整数スコア
        torch_counts = (hard_logits * TAU_GS).numpy()  # τ=1でlogit=整数カウント
        counts_match = bool(np.abs(torch_counts - scores).max() < 1e-3)
        argmax_match = bool((torch_counts.argmax(-1) == scores.argmax(-1)).all())
        return (counts_match and argmax_match), []
    except Exception as e:
        return False, [f'{type(e).__name__}: {e}']


def _fmt_eta(done: int, total: int, t0: float) -> str:
    elapsed = time.time() - t0
    per = elapsed / max(done, 1)
    remain = per * (total - done)
    eta = datetime.now(JST) + timedelta(seconds=remain)

    def hms(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f'{m}m{s:02d}s'

    return f'[{hms(elapsed)}<{hms(remain)}, 終了予定 {eta:%H:%M:%S}]'


def run_grid(
    seeds: list[int], temps: list[float], alphas: list[float],
    stds: list[bool], hiddens: list[int],
) -> dict:
    """(hidden, T, alpha, standardize) セルごとに多シードで蒸留し集計する。"""
    cells = list(itertools.product(hiddens, temps, alphas, stds))
    total = len(cells) * len(seeds)
    t0, done = time.time(), 0

    teacher_cache: dict[tuple[int, int], tuple[TeacherMLP, dict]] = {}

    def get_teacher(hidden: int, seed: int) -> tuple[TeacherMLP, dict]:
        key = (hidden, seed)
        if key not in teacher_cache:
            teacher_cache[key] = train_teacher(hidden, seed)
        return teacher_cache[key]

    results = {}
    for hidden, T, alpha, std in cells:
        runs = []
        for seed in seeds:
            teacher, tinfo = get_teacher(hidden, seed)
            r = distill(teacher, T, alpha, std, seed)
            r['teacher_reached'] = tinfo['reached']
            r['teacher_conf'] = tinfo['conf']
            runs.append(r)
            done += 1
        n = len(runs)
        reach_rate = sum(r['reached'] for r in runs) / n
        xor_pair_rate = sum(r['is_xor_pair'] for r in runs) / n
        ir_rate = sum(r['ir_match'] for r in runs) / n
        conv = [r['conv_step'] for r in runs if r['conv_step'] is not None]
        key = f'H{hidden}_T{T}_a{alpha}_std{int(std)}'
        results[key] = {
            'hidden': hidden, 'temperature': T, 'alpha': alpha, 'standardize': std,
            'n': n,
            'reach_rate': reach_rate,
            'xor_pair_rate': xor_pair_rate,
            'ir_match_rate': ir_rate,
            'mean_margin': float(np.mean([r['margin'] for r in runs])),
            'mean_disc_gap': float(np.mean([r['disc_gap'] for r in runs])),
            'mean_conv_step': (float(np.mean(conv)) if conv else None),
            'teacher_reach_rate': sum(r['teacher_reached'] for r in runs) / n,
            'pair_hist': _pair_hist(runs),
        }
        c = results[key]
        print(
            f'  {key:24s} 分類acc={reach_rate:5.1%} XORpair={xor_pair_rate:5.1%} '
            f"IR={ir_rate:5.1%} margin={c['mean_margin']:.3f} "
            f"gap={c['mean_disc_gap']:.2e}  {_fmt_eta(done, total, t0)}"
        )
    return results


def _pair_hist(runs: list[dict]) -> dict:
    hist: dict[str, int] = {}
    for r in runs:
        hist[r['pair']] = hist.get(r['pair'], 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def print_summary(results: dict) -> None:
    print('\n' + '=' * 92)
    print('集計: (hidden, T, alpha, standardize) セルごと 分類到達率 / XORpair率 / ロバスト性')
    print('=' * 92)
    print(f'{"cell":24s} {"分類acc":>7s} {"XORpr":>6s} {"IR":>6s} '
          f'{"margin":>8s} {"gap":>9s} {"conv":>7s} {"T到達":>6s}')
    print('-' * 92)
    for key, c in results.items():
        conv = f"{c['mean_conv_step']:.0f}" if c['mean_conv_step'] else '  -'
        print(
            f'{key:24s} {c["reach_rate"]:6.1%} {c["xor_pair_rate"]:5.1%} '
            f'{c["ir_match_rate"]:5.1%} {c["mean_margin"]:8.3f} '
            f'{c["mean_disc_gap"]:9.2e} {conv:>7s} {c["teacher_reach_rate"]:5.1%}'
        )

    # 主対照: label-free(α=0) vs 純ラベル(α=1)
    print('\n--- 主対照: 蒸留は関数を運ぶか (同一H,T,std下の分類到達率) ---')
    by = {}
    for c in results.values():
        by.setdefault((c['hidden'], c['temperature'], c['standardize']), {})[c['alpha']] = c
    for (h, t, s), amap in sorted(by.items()):
        parts = [f'α={a}:{amap[a]["reach_rate"]:.0%}' for a in sorted(amap)]
        print(f'  H={h} T={t} std={int(s)}:  ' + '  '.join(parts))

    # standardize効果(同一H,T,α)
    print('\n--- Logit Standardization効果 (std=0 vs std=1) ---')
    by2 = {}
    for c in results.values():
        by2.setdefault((c['hidden'], c['temperature'], c['alpha']), {})[c['standardize']] = c
    for (h, t, a), smap in sorted(by2.items()):
        if False in smap and True in smap:
            r0, r1 = smap[False]['reach_rate'], smap[True]['reach_rate']
            m0, m1 = smap[False]['mean_margin'], smap[True]['mean_margin']
            print(f'  H={h} T={t} α={a}:  分類acc std0={r0:.0%}→std1={r1:.0%}  '
                  f'margin std0={m0:.3f}→std1={m1:.3f}')

    # 温度依存(二重温度 τ_gs×T のうちTを振った効果): margin/gap
    print('\n--- 温度T依存(τ_gs=1.0固定, α=0.5, std=0): 到達率でなくマージンを見る ---')
    for h in sorted({c['hidden'] for c in results.values()}):
        row = []
        for t in sorted({c['temperature'] for c in results.values()}):
            k = f'H{h}_T{t}_a0.5_std0'
            if k in results:
                c = results[k]
                row.append(f'T={t}:acc{c["reach_rate"]:.0%}/mgn{c["mean_margin"]:.2f}/gap{c["mean_disc_gap"]:.1e}')
        print(f'  H={h}: ' + '  '.join(row))


def demo() -> None:
    teacher, _ = train_teacher(hidden=8, seed=0)
    r = distill(teacher, temperature=2.0, alpha=0.5, standardize=False, seed=0)
    print('\n--- 代表run (H=8, T=2, α=0.5, std=0, seed=0) の2-LUT真理値表 ---')
    print(f"  (LUT0,LUT1) = {r['pair']}  (収束形 1001/0110=XNOR/XOR? {r['is_xor_pair']})")
    print(f"  分類hard acc={r['hard_acc']:.0%}  IR simulate一致={r['ir_match']}  "
          f"離散化gap={r['disc_gap']:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--quick', action='store_true',
                        help='短縮: T=[2], α=[0,0.5,1], std=[F,T], H=[2,8]')
    parser.add_argument('--out', type=str, default=None,
                        help='結果JSON保存先(既定=results/distill_xor_2lut_groupsum_results.json)')
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    if args.quick:
        temps, alphas, stds, hiddens = [2.0], [0.0, 0.5, 1.0], [False, True], [2, 8]
    else:
        temps, alphas, stds, hiddens = [1.0, 2.0, 4.0], [0.0, 0.5, 1.0], [False, True], [2, 8]

    n_runs = len(seeds) * len(temps) * len(alphas) * len(stds) * len(hiddens)
    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  seeds={len(seeds)} '
          f'T={temps} α={alphas} std={stds} H={hiddens}  (全{n_runs} runs)')
    print('-' * 92)

    results = run_grid(seeds, temps, alphas, stds, hiddens)
    print_summary(results)
    demo()

    out = args.out or 'results/distill_xor_2lut_groupsum_results.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'meta': {
            'seeds': len(seeds), 'temps': temps, 'alphas': alphas,
            'standardize': [False, True], 'hiddens': hiddens,
            'xor_pair': list(XOR_PAIR), 'tau_gs': TAU_GS,
            'timestamp': datetime.now(JST).isoformat(),
        },
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
