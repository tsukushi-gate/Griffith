"""適格度トレース スイープ結果の集計・作図・受け入れ基準判定。

外部依存(matplotlib等)なしでSVG折れ線図を生成する。
  - 図1: 回答精度 vs 系列長T(λ別曲線群) — タスクごと
  - 図2: λ vs 学習可能な最大系列長 — タスクごと
  - 比較表(Markdown): (task, λ, T)ごとの mean±std
  - 受け入れ基準2判定: delayed_recall T=10 で λ∈{0.9,0.99} が λ=0 を上回るか

使用例:
  uv run python experiments/analyze_ap_eligibility.py --tags tiny
  uv run python experiments/analyze_ap_eligibility.py --tags matrix regime
"""

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.getcwd(), 'results', 'ap_eligibility')
FIG_DIR = os.path.join(os.getcwd(), 'results', 'figures')

# λ別の色(視認性重視・色覚差配慮のカテゴリ配色)
LAMBDA_COLORS = {
    0.0: '#888888', 0.5: '#4C9F70', 0.9: '#3B7DD8',
    0.99: '#D8642F', 1.0: '#9B4DCA',
}
LEARNABLE_MARGIN = 0.15  # chance + margin 以上を「学習した」とみなす


def load_rows(tags):
    rows = []
    for tag in tags:
        path = os.path.join(RESULTS_DIR, f'sweep_{tag}.jsonl')
        if not os.path.exists(path):
            print(f'警告: {path} が無い')
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def aggregate(rows):
    """(task, lambda, seq_len) -> {mean,std,n,vals,chance}。best_accを集計。"""
    groups = collections.defaultdict(list)
    chance = {}
    for r in rows:
        key = (r['task'], r['lambda'], r['seq_len'])
        groups[key].append(r['best_acc'])
        chance[r['task']] = r['chance']
    agg = {}
    for key, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        agg[key] = {
            'mean': mean, 'std': math.sqrt(var), 'n': n,
            'vals': sorted(vals), 'chance': chance[key[0]],
        }
    return agg


def welch_t(a, b):
    """Welchのt統計量(自由度は返さない。n小なので参考値)。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0:
        return math.inf if ma > mb else (-math.inf if ma < mb else 0.0)
    return (ma - mb) / denom


# --------------------------------------------------------------------------
# SVG 折れ線図
# --------------------------------------------------------------------------

def _svg_line_chart(series, x_vals, title, xlabel, ylabel,
                    y_range=(0.0, 1.0), chance=None, width=680, height=440):
    """series: list of dict(label,color,points=[(x,y)|None]). x_vals: x軸目盛。"""
    ml, mr, mt, mb = 70, 150, 46, 56
    pw, ph = width - ml - mr, height - mt - mb
    xmin, xmax = min(x_vals), max(x_vals)
    y0, y1 = y_range
    xspan = (xmax - xmin) or 1
    yspan = (y1 - y0) or 1

    def sx(x):
        return ml + (x - xmin) / xspan * pw

    def sy(y):
        return mt + (1 - (y - y0) / yspan) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" '
        f'font-size="16" font-weight="bold">{title}</text>',
    ]
    # y軸グリッド+目盛
    steps = 5
    for i in range(steps + 1):
        yv = y0 + yspan * i / steps
        yy = sy(yv)
        out.append(
            f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" '
            f'stroke="#e5e5e5"/>'
        )
        out.append(
            f'<text x="{ml-8}" y="{yy+4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#555">{yv:.2f}</text>'
        )
    # x軸目盛
    for xv in x_vals:
        xx = sx(xv)
        out.append(
            f'<line x1="{xx:.1f}" y1="{mt+ph}" x2="{xx:.1f}" y2="{mt+ph+5}" '
            f'stroke="#555"/>'
        )
        out.append(
            f'<text x="{xx:.1f}" y="{mt+ph+20:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#555">{xv}</text>'
        )
    # 軸ラベル
    out.append(
        f'<text x="{ml+pw/2:.1f}" y="{height-14}" text-anchor="middle" '
        f'font-size="13">{xlabel}</text>'
    )
    out.append(
        f'<text x="18" y="{mt+ph/2:.1f}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 18 {mt+ph/2:.1f})">{ylabel}</text>'
    )
    # chance基準線
    if chance is not None:
        cy = sy(chance)
        out.append(
            f'<line x1="{ml}" y1="{cy:.1f}" x2="{ml+pw}" y2="{cy:.1f}" '
            f'stroke="#c00" stroke-width="1" stroke-dasharray="5 4"/>'
        )
        out.append(
            f'<text x="{ml+pw+6}" y="{cy+4:.1f}" font-size="11" fill="#c00">'
            f'chance={chance:.2f}</text>'
        )
    # 系列
    for s in series:
        pts = [(sx(x), sy(y)) for (x, y) in s['points'] if y is not None]
        if len(pts) >= 2:
            d = 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in pts)
            out.append(
                f'<path d="{d}" fill="none" stroke="{s["color"]}" '
                f'stroke-width="2.2"/>'
            )
        for x, y in pts:
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{s["color"]}"/>'
            )
    # 凡例
    lx, ly = ml + pw + 12, mt + 8
    for i, s in enumerate(series):
        yy = ly + i * 20
        out.append(
            f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" '
            f'stroke="{s["color"]}" stroke-width="2.6"/>'
        )
        out.append(
            f'<text x="{lx+28}" y="{yy+4}" font-size="12">{s["label"]}</text>'
        )
    out.append('</svg>')
    return '\n'.join(out)


def fig1_acc_vs_T(agg, task, lambdas, seq_lens):
    chance = None
    series = []
    for lam in lambdas:
        pts = []
        for T in seq_lens:
            key = (task, lam, T)
            if key in agg:
                pts.append((T, agg[key]['mean']))
                chance = agg[key]['chance']
        if pts:
            series.append({
                'label': f'λ={lam}',
                'color': LAMBDA_COLORS.get(lam, '#333'),
                'points': pts,
            })
    return _svg_line_chart(
        series, seq_lens,
        f'回答精度 vs 系列長T — {task}', '系列長 T', '回答精度(seed平均)',
        chance=chance,
    )


def fig2_lambda_vs_maxT(agg, tasks, lambdas, seq_lens):
    series = []
    colors = {'delayed_recall': '#3B7DD8', 'temporal_xor': '#D8642F',
              'copy': '#4C9F70'}
    lam_idx = list(range(len(lambdas)))
    for task in tasks:
        pts = []
        for i, lam in enumerate(lambdas):
            max_t = 0
            for T in sorted(seq_lens):
                key = (task, lam, T)
                if key in agg:
                    thr = agg[key]['chance'] + LEARNABLE_MARGIN
                    if agg[key]['mean'] >= thr:
                        max_t = T
            pts.append((i, max_t))
        if pts:
            series.append({
                'label': task, 'color': colors.get(task, '#333'),
                'points': pts,
            })
    ymax = max((y for s in series for _, y in s['points']), default=1) or 1
    svg = _svg_line_chart(
        series, lam_idx,
        'λ vs 学習可能な最大系列長', 'λ (index)', '学習可能な最大T',
        y_range=(0, ymax * 1.1 + 1), chance=None,
    )
    # x目盛をλ値ラベルに差し替え(index→λ)
    for i, lam in enumerate(lambdas):
        svg = svg.replace(f'>{i}</text>', f'>λ={lam}</text>', 1)
    return svg


def build_table(agg, tasks, lambdas, seq_lens):
    lines = ['| task | λ | ' + ' | '.join(f'T={T}' for T in seq_lens)
             + ' | chance |', '|---|---|' + '---|' * (len(seq_lens) + 1)]
    for task in tasks:
        for lam in lambdas:
            cells = []
            ch = None
            for T in seq_lens:
                key = (task, lam, T)
                if key in agg:
                    a = agg[key]
                    ch = a['chance']
                    cells.append(f"{a['mean']:.3f}±{a['std']:.3f}")
                else:
                    cells.append('—')
            lines.append(
                f'| {task} | {lam} | ' + ' | '.join(cells)
                + f' | {ch if ch is not None else "—"} |'
            )
    return '\n'.join(lines)


def criterion2(agg, rows):
    """delayed_recall T=10: λ∈{0.9,0.99} が λ=0 を上回るか判定。"""
    task, T = 'delayed_recall', 10
    by = collections.defaultdict(list)
    for r in rows:
        if r['task'] == task and r['seq_len'] == T:
            by[r['lambda']].append(r['best_acc'])
    base = by.get(0.0, [])
    report = [f'### 受け入れ基準2: {task} T={T}']
    if not base:
        report.append('- λ=0 のデータが無い(未実行)')
        return '\n'.join(report), None
    chance = agg.get((task, 0.0, T), {}).get('chance', 0.25)
    mb = sum(base) / len(base)
    report.append(f'- λ=0.0: mean={mb:.3f} (n={len(base)}, vals={[round(x,3) for x in sorted(base)]}, chance={chance})')
    verdict = 'λ=0を有意に上回るλは無い(トレースはT=10で救済せず)'
    beat = False
    for lam in (0.9, 0.99):
        g = by.get(lam, [])
        if not g:
            report.append(f'- λ={lam}: データ無し')
            continue
        mg = sum(g) / len(g)
        t = welch_t(g, base)
        sep = min(g) > max(base)
        clearly = mg >= mb + 0.10 and sep
        report.append(
            f'- λ={lam}: mean={mg:.3f} (n={len(g)}, vals={[round(x,3) for x in sorted(g)]}), '
            f'Welch t={t if t is None else round(t,2)}, '
            f'群非重複={sep}, 明確に上回る={clearly}'
        )
        if clearly:
            beat = True
    if beat:
        verdict = 'λ∈{0.9,0.99}のいずれかがλ=0を明確に上回る(トレース有効)'
    report.append(f'- **判定**: {verdict}')
    return '\n'.join(report), beat


def log_to_wandb(agg, rows, tags, fig_paths, table_path, verdict_beat):
    """図・JSON・数値結果をwandbにも記録(プロジェクトの記録規約)。

    画像(SVG)・JSONL・表はArtifactに、集計値はrun summary/Tableに載せる。
    """
    from env_utils import load_dotenv
    from wandb_utils import finish as wandb_finish
    from wandb_utils import init_wandb

    load_dotenv()
    ok = init_wandb(
        run_name=f'ap_eligibility_analysis_{"_".join(tags)}',
        config_dict={'tags': tags, 'analysis': 'ap_eligibility_trace'},
        tags=['ap', 'eligibility', 'rddlgn', 'analysis'],
        group='ap_eligibility', job_type='analysis',
    )
    if not ok:
        print('wandb無効(WANDB_API_KEY未設定?)。ローカル成果物のみ。')
        return False

    import wandb

    table = wandb.Table(columns=['task', 'lambda', 'seq_len', 'mean_best', 'std', 'n', 'chance'])
    for (task, lam, T), a in sorted(agg.items()):
        table.add_data(task, lam, T, round(a['mean'], 4), round(a['std'], 4), a['n'], a['chance'])
    wandb.log({'results_table': table})
    wandb.summary['criterion2_lambda_beats_baseline'] = bool(verdict_beat)
    wandb.summary['n_results'] = len(rows)
    wandb.summary['verdict'] = 'negative: トレースは全λ・全Tでchance(RNN学習不成立)'

    art = wandb.Artifact('ap_eligibility_results', type='analysis')
    for p in fig_paths + [table_path]:
        if os.path.exists(p):
            art.add_file(p)
    for tag in tags:
        jp = os.path.join(RESULTS_DIR, f'sweep_{tag}.jsonl')
        if os.path.exists(jp):
            art.add_file(jp)
    wandb.log_artifact(art)
    wandb_finish()
    print('wandb記録完了(run summary + results_table + Artifact)')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--tasks', nargs='+',
                    default=['delayed_recall', 'temporal_xor', 'copy'])
    ap.add_argument('--lambdas', nargs='+', type=float,
                    default=[0, 0.5, 0.9, 0.99, 1.0])
    ap.add_argument('--seq-lens', nargs='+', type=int, default=[5, 10, 20, 40])
    ap.add_argument('--wandb', action='store_true',
                    help='図・JSON・集計をwandbにも記録する')
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    rows = load_rows(args.tags)
    if not rows:
        print('結果が無い。先にスイープを実行すること。')
        return
    agg = aggregate(rows)
    tasks = [t for t in args.tasks if any(r['task'] == t for r in rows)]

    fig_paths = []
    for task in tasks:
        svg = fig1_acc_vs_T(agg, task, args.lambdas, args.seq_lens)
        p = os.path.join(FIG_DIR, f'ap_elig_fig1_acc_vs_T_{task}.svg')
        with open(p, 'w') as f:
            f.write(svg)
        fig_paths.append(p)
        print(f'図1: {p}')

    svg2 = fig2_lambda_vs_maxT(agg, tasks, args.lambdas, args.seq_lens)
    p2 = os.path.join(FIG_DIR, 'ap_elig_fig2_lambda_vs_maxT.svg')
    with open(p2, 'w') as f:
        f.write(svg2)
    fig_paths.append(p2)
    print(f'図2: {p2}')

    table = build_table(agg, tasks, args.lambdas, args.seq_lens)
    c2, verdict_beat = criterion2(agg, rows)
    tbl_path = os.path.join(FIG_DIR, 'ap_elig_table.md')
    with open(tbl_path, 'w') as f:
        f.write('# 適格度トレース スイープ集計\n\n' + table + '\n\n' + c2 + '\n')
    print(f'表+基準2: {tbl_path}')
    print('\n' + table + '\n\n' + c2)

    if args.wandb:
        log_to_wandb(agg, rows, args.tags, fig_paths, tbl_path, verdict_beat)


if __name__ == '__main__':
    main()
