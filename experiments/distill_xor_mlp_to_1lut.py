"""XOR: 最小構成MLP teacher → 1-LUT student 蒸留 & 離散化ロバスト性実験。

「本当に(ラベル経由でなくteacherの関数として)蒸留できているか」と
「softで学んだLUTをargmaxハード化しても関数が保たれるか(離散化ロバスト性)」を、
最小・完全観測可能なXORで白箱検証する。

XORは単一の2入力LUTが厳密表現できる(16ゲートのうちXORゲート=真理値表0b0110=6)。
teacher = MLP(2->H->1)、student = 1個の LutLayer(2,1) にして蒸留し、ハード化後の
真理値表がXORゲートに一致するかを直接読む。torch hard eval == IR simulate の
三段一致(HDLはiverilog未導入ならスキップ)でパイプライン健全性も確認する。

Usage:
    uv run python experiments/distill_xor_mlp_to_1lut.py            # 既定: 16 seeds, T/α grid, H=2 & H=8
    uv run python experiments/distill_xor_mlp_to_1lut.py --seeds 4 --quick   # 短縮スモーク

wandbは使わない(ローカル実行のみ)。結果はstdout整形表 + JSON保存。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic.export import export_ir, truth_tables_from_lut_layer
from griffith_logic.ir import simulate, validate
from griffith_logic.layers import LutLayer

JST = timezone(timedelta(hours=9))

# XORゲートの真理値表: butterfly順 bit k = パターンk への出力
#   k=0(00)->0, k=1(01)->1, k=2(10)->1, k=3(11)->0  => 0b0110 = 6
XOR_TT = 0b0110  # = 6

# soft出力が実質one-hot(ハード化が安全)と見なす閾値。
# tests/conftest.py TOL_F32_EQ(=1e-6)を参照値とし、experimentでは
# 「離散化ギャップ」を実測値として報告する(閾値でテストを緩めない)。
_EPS_CLAMP = 1e-6  # BCEでp=0/1をinf化させないためのクランプ


def xor_data() -> tuple[torch.Tensor, torch.Tensor]:
    """全4パターンのXORデータ。X:(4,2) float, Y:(4,) float in {0,1}。

    パターン列挙はbutterfly順(bit j = スロットjの値)。
    """
    x = torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=2)), dtype=torch.float32,
    )
    # itertools.productは (0,0),(0,1),(1,0),(1,1) を返す(左=最上位)。
    # butterfly順に合わせるため列順を [slot0, slot1] = [下位, 上位] で解釈する。
    # ここでは x[:,0]=slot0, x[:,1]=slot1 とし、XOR = slot0 XOR slot1。
    y = (x[:, 0].int() ^ x[:, 1].int()).float()
    return x, y


class TeacherMLP(nn.Module):
    """最小構成XOR teacher: 2 -> H -> 1(単一ロジット)。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)


def train_teacher(
    hidden: int, seed: int, *, max_steps: int = 4000, lr: float = 0.05,
    retries: int = 8,
) -> tuple[TeacherMLP, dict]:
    """XOR teacherを100% hard accまで学習する。未達なら再シードでリトライ。

    Returns:
        (teacher, info)。info は {'hard_acc', 'reached', 'seed_used', 'conf'}。
        conf = 4点でのteacher確率|0.5境界からの平均距離(自信度)。
    """
    x, y = xor_data()
    for r in range(retries):
        s = seed * 1000 + r
        torch.manual_seed(s)
        teacher = TeacherMLP(hidden)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        for _ in range(max_steps):
            teacher.train()
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(teacher(x), y)
            loss.backward()
            opt.step()
        teacher.eval()
        with torch.no_grad():
            z = teacher(x)
            hard_acc = ((z > 0).float() == y).float().mean().item()
            probs = torch.sigmoid(z)
            conf = (probs - 0.5).abs().mean().item()  # 自信度(0.5からの距離)
        if hard_acc == 1.0:
            return teacher, {
                'hard_acc': hard_acc, 'reached': True,
                'seed_used': s, 'conf': conf,
            }
    # 全リトライで100%未達: 最後のteacherを「品質不足teacher」として返す
    return teacher, {
        'hard_acc': hard_acc, 'reached': False,
        'seed_used': s, 'conf': conf,
    }


def make_student(seed: int) -> LutLayer:
    """厳密1LUT student: LutLayer(2,1,n_inputs=2,softmax=16ゲートモード)。

    2つの入力スロットが必ず異なる入力(slot0=x[:,0], slot1=x[:,1])を見るよう
    接続を固定する(random接続だと両スロットが同一入力を見てXOR不能になり得る)。
    """
    torch.manual_seed(seed)
    lut = LutLayer(2, 1, n_inputs=2, parameterization='softmax')
    # indices: (n_inputs, out_dim) = (2,1)。slot j -> 入力 j。
    lut.indices = torch.tensor([[0], [1]], dtype=torch.int64)
    return lut


def gate_margin(lut: LutLayer) -> float:
    """勝ちゲートのsoftmax重み - 次点の差(離散化信頼マージン)。"""
    with torch.no_grad():
        p = F.softmax(lut.weights[0], dim=-1)  # (16,)
        top2 = torch.topk(p, 2).values
        return (top2[0] - top2[1]).item()


def hard_truth_table(lut: LutLayer) -> int:
    """学習後LUTのハード真理値表(python int, bit k = パターンkの出力)。"""
    return int(truth_tables_from_lut_layer(lut)[0])


def distill(
    teacher: TeacherMLP, temperature: float, alpha: float, seed: int,
    *, max_steps: int = 3000, lr: float = 0.05, check_every: int = 50,
) -> dict:
    """1-LUT studentをbinary蒸留する。

    損失: L = alpha*BCE(p, y) + (1-alpha)*BCE(p, q_T)
        p   = student soft出力(LutLayer, [0,1])
        q_T = sigmoid(z_teacher / T)  (温度Tで軟化したteacher確率)
    GroupSumが無いのでτ_gs非関与=純T効果。

    Returns:
        run結果dict(truth table, is_xor, margin, 収束step, ギャップ, 三段一致 等)。
    """
    x, y = xor_data()
    with torch.no_grad():
        q_soft = torch.sigmoid(teacher(x) / temperature)  # teacher soft target

    lut = make_student(seed)
    opt = torch.optim.Adam(lut.parameters(), lr=lr)

    conv_step = None  # 初めてハード真理値表がXORになったstep
    for step in range(1, max_steps + 1):
        lut.train()
        opt.zero_grad()
        p = lut(x).squeeze(-1).clamp(_EPS_CLAMP, 1.0 - _EPS_CLAMP)
        loss_hard = F.binary_cross_entropy(p, y)
        loss_kd = F.binary_cross_entropy(p, q_soft)
        loss = alpha * loss_hard + (1.0 - alpha) * loss_kd
        loss.backward()
        opt.step()

        if step % check_every == 0 and conv_step is None:
            if hard_truth_table(lut) == XOR_TT:
                conv_step = step

    # --- 検証 ---
    lut.eval()
    tt = hard_truth_table(lut)
    is_xor = (tt == XOR_TT)
    margin = gate_margin(lut)

    # soft(train) vs hard(eval) の離散化ギャップを実測
    lut.train()
    with torch.no_grad():
        soft_vals = lut(x).squeeze(-1)
    lut.eval()
    with torch.no_grad():
        hard_vals = lut(x).squeeze(-1)
    disc_gap = (soft_vals - hard_vals).abs().max().item()

    # 三段一致: torch hard eval == IR simulate(整数, 厳密一致)
    ir_ok, ir_issues = _check_ir_equivalence(lut, x, hard_vals)

    return {
        'tt': tt,
        'tt_bits': format(tt, '04b'),
        'is_xor': is_xor,
        'margin': margin,
        'conv_step': conv_step,
        'disc_gap': disc_gap,
        'ir_match': ir_ok,
        'ir_issues': ir_issues,
    }


def _check_ir_equivalence(
    lut: LutLayer, x: torch.Tensor, hard_vals: torch.Tensor,
) -> tuple[bool, list[str]]:
    """torch hard eval == IR simulate を厳密(整数)照合する。"""
    try:
        ir = export_ir(nn.Sequential(lut), input_width=2, name='xor_1lut')
        issues = validate(ir, target='hdl')  # list[str], 空=問題なし
        x_np = x.numpy().astype(np.int64)
        sim = simulate(ir, x_np)  # (batch, out_dim) 整数
        sim_bits = sim.reshape(-1).astype(np.int64)
        hard_bits = hard_vals.numpy().reshape(-1).round().astype(np.int64)
        match = bool((sim_bits == hard_bits).all())
        return match, list(issues)
    except Exception as e:  # export未対応等はbest-effort降格
        return False, [f'{type(e).__name__}: {e}']


def _fmt_eta(done: int, total: int, t0: float) -> str:
    """[経過<残り, 終了予定 HH:MM:SS] を返す(training-script規約)。"""
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
    hiddens: list[int],
) -> dict:
    """(hidden, T, alpha) セルごとに多シードで蒸留し集計する。"""
    x, _ = xor_data()
    cells = list(itertools.product(hiddens, temps, alphas))
    total = len(cells) * len(seeds)
    t0 = time.time()
    done = 0

    # teacherは(hidden, seed)にのみ依存するのでキャッシュ(T/alpha間で再学習しない)
    teacher_cache: dict[tuple[int, int], tuple[TeacherMLP, dict]] = {}

    def get_teacher(hidden: int, seed: int) -> tuple[TeacherMLP, dict]:
        key = (hidden, seed)
        if key not in teacher_cache:
            teacher_cache[key] = train_teacher(hidden, seed)
        return teacher_cache[key]

    results = {}
    for hidden, T, alpha in cells:
        runs = []
        for seed in seeds:
            teacher, tinfo = get_teacher(hidden, seed)
            r = distill(teacher, T, alpha, seed)
            r['teacher_reached'] = tinfo['reached']
            r['teacher_conf'] = tinfo['conf']
            runs.append(r)
            done += 1
        n = len(runs)
        xor_rate = sum(r['is_xor'] for r in runs) / n
        ir_rate = sum(r['ir_match'] for r in runs) / n
        conv = [r['conv_step'] for r in runs if r['conv_step'] is not None]
        key = f'H{hidden}_T{T}_a{alpha}'
        results[key] = {
            'hidden': hidden, 'temperature': T, 'alpha': alpha,
            'n': n,
            'xor_rate': xor_rate,
            'ir_match_rate': ir_rate,
            'mean_margin': float(np.mean([r['margin'] for r in runs])),
            'mean_disc_gap': float(np.mean([r['disc_gap'] for r in runs])),
            'mean_conv_step': (float(np.mean(conv)) if conv else None),
            'teacher_reach_rate': sum(r['teacher_reached'] for r in runs) / n,
            'mean_teacher_conf': float(np.mean([r['teacher_conf'] for r in runs])),
            'gate_hist': _gate_hist(runs),
        }
        print(
            f'  {key:20s} XOR率={xor_rate:5.1%} IR一致={ir_rate:5.1%} '
            f"margin={results[key]['mean_margin']:.3f} "
            f"gap={results[key]['mean_disc_gap']:.2e}  "
            f'{_fmt_eta(done, total, t0)}'
        )
    return results


def _gate_hist(runs: list[dict]) -> dict:
    """run群が落ちたハード真理値表の分布(bit列 -> 回数)。"""
    hist: dict[str, int] = {}
    for r in runs:
        hist[r['tt_bits']] = hist.get(r['tt_bits'], 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def print_summary(results: dict) -> None:
    """主結果を人間可読に整形表示する。"""
    print('\n' + '=' * 78)
    print('集計: (hidden, T, alpha) セルごとの XOR到達率 / 離散化ロバスト性')
    print('=' * 78)
    header = f'{"cell":20s} {"XOR率":>7s} {"IR一致":>7s} {"margin":>8s} {"gap":>9s} {"conv":>7s} {"T到達":>6s}'
    print(header)
    print('-' * 78)
    for key, c in results.items():
        conv = f"{c['mean_conv_step']:.0f}" if c['mean_conv_step'] else '  -'
        print(
            f'{key:20s} {c["xor_rate"]:6.1%} {c["ir_match_rate"]:6.1%} '
            f'{c["mean_margin"]:8.3f} {c["mean_disc_gap"]:9.2e} {conv:>7s} '
            f'{c["teacher_reach_rate"]:5.1%}'
        )

    # 主対照: alpha=0(純KD) vs alpha=1(純ラベル) の到達率
    print('\n--- 主対照: 蒸留は関数を運ぶか (同一H,T下) ---')
    by_ht: dict[tuple, dict] = {}
    for c in results.values():
        by_ht.setdefault((c['hidden'], c['temperature']), {})[c['alpha']] = c
    for (h, t), amap in sorted(by_ht.items()):
        parts = []
        for a in sorted(amap):
            parts.append(f'α={a}:{amap[a]["xor_rate"]:.0%}')
        print(f'  H={h} T={t}:  ' + '  '.join(parts))

    # teacher品質依存
    print('\n--- teacher品質依存 (H小=最小構成 vs H大) ---')
    by_h: dict[int, list] = {}
    for c in results.values():
        by_h.setdefault(c['hidden'], []).append(c)
    for h, cs in sorted(by_h.items()):
        xr = np.mean([c['xor_rate'] for c in cs])
        mg = np.mean([c['mean_margin'] for c in cs])
        tr = np.mean([c['teacher_reach_rate'] for c in cs])
        tc = np.mean([c['mean_teacher_conf'] for c in cs])
        print(
            f'  H={h}: XOR率(平均)={xr:.1%}  margin(平均)={mg:.3f}  '
            f'teacher100%到達={tr:.0%}  teacher自信度={tc:.3f}'
        )


def demo_truth_table() -> None:
    """代表runの真理値表を人間可読で表示(XORゲートであることの提示)。"""
    teacher, tinfo = train_teacher(hidden=8, seed=0)
    r = distill(teacher, temperature=2.0, alpha=0.5, seed=0)
    print('\n--- 代表run (H=8, T=2, α=0.5, seed=0) の学習済み1-LUT真理値表 ---')
    lut = make_student(0)  # 表示用に再学習は省略、直近rの結果を使う
    tt = r['tt']
    print(f"  ハード真理値表 = 0b{r['tt_bits']} (= {tt}),  XOR? {r['is_xor']}")
    labels = ['00', '01', '10', '11']  # (slot1 slot0) 表記
    for k in range(4):
        bit = (tt >> k) & 1
        s0, s1 = k & 1, (k >> 1) & 1
        print(f'  slot0={s0} slot1={s1}  ->  {bit}')
    print(f"  離散化ギャップ(soft-hard)={r['disc_gap']:.2e}  "
          f"IR simulate一致={r['ir_match']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=16, help='シード数')
    parser.add_argument('--quick', action='store_true',
                        help='短縮グリッド(T=[2], alpha=[0,0.5,1], H=[2,8])')
    parser.add_argument('--out', type=str, default=None,
                        help='結果JSON保存先(既定=results/distill_xor_mlp_to_1lut_results.json)')
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    if args.quick:
        temps, alphas, hiddens = [2.0], [0.0, 0.5, 1.0], [2, 8]
    else:
        temps, alphas, hiddens = [1.0, 2.0, 4.0], [0.0, 0.5, 1.0], [2, 8]

    print(f'開始 {datetime.now(JST):%Y-%m-%d %H:%M:%S} JST  '
          f'seeds={len(seeds)} T={temps} α={alphas} H={hiddens}  '
          f'(全{len(seeds) * len(temps) * len(alphas) * len(hiddens)} runs)')
    print('-' * 78)

    results = run_grid(seeds, temps, alphas, hiddens)
    print_summary(results)
    demo_truth_table()

    out = args.out or 'results/distill_xor_mlp_to_1lut_results.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'meta': {
            'seeds': len(seeds), 'temps': temps, 'alphas': alphas,
            'hiddens': hiddens, 'xor_tt': XOR_TT,
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
