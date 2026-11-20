"""較正アンカー(C3 / C4)の再走結果を集計して原値と突き合わせる。

run_calibration.py が results/ に落とした run ごとの JSON を読み、腕ごとに
    seed 別の値 / 2 seed 平均 / 原値との差(pt) / 許容判定
を markdown 表と JSON にする。原値と許容幅は anchors.json(claims.json からの転記)
だけを見る。**判定の閾値をこのスクリプトに直書きしない**。

使用例:
    uv run python experiments/computed-wiring-overlay/aggregate.py
    uv run python experiments/computed-wiring-overlay/aggregate.py \
        --results outputs/computed-wiring-overlay/results --out-dir results/cwo
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))

HERE = Path(__file__).resolve().parent
ANCHORS_PATH = HERE / 'anchors.json'
DEFAULT_RESULTS = Path('outputs/computed-wiring-overlay/results')


def load_anchors(path: Path = ANCHORS_PATH) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_runs(results_dir: Path) -> list[dict]:
    """results/ の run JSON を読む(壊れている / 空のファイルは飛ばす)。"""
    runs = []
    for path in sorted(results_dir.glob('*.json')):
        try:
            runs.append(json.loads(path.read_text(encoding='utf-8')))
        except json.JSONDecodeError:
            print(f'警告: JSON として読めないので飛ばす: {path}', file=sys.stderr)
    return runs


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value, digits: int = 4) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def _fmt_pt(value) -> str:
    return '-' if value is None else f'{value:+.2f}'


def compare_pt(value: float, minimum: float, comparison: str) -> bool:
    """差の規則の比較(anchors.json の comparison。'gte' = 以上 / 'gt' = 超過)。"""
    if comparison == 'gt':
        return value > minimum
    if comparison == 'gte':
        return value >= minimum
    raise ValueError(f'未知の comparison: {comparison!r}(gte / gt)')


def collect_arm(runs: list[dict], recipe: str, metric: str) -> dict:
    """1 腕(レシピ)の seed 別の値をまとめる。"""
    rows = {
        int(r['seed']): r for r in runs
        if r.get('recipe') == recipe and r.get('error') is None
    }
    seeds = sorted(rows)
    values = [rows[s].get(metric) for s in seeds]
    values = [v for v in values if v is not None]
    return {
        'recipe': recipe,
        'seeds': seeds,
        'values': [rows[s].get(metric) for s in seeds],
        'mean': mean(values),
        'smoke': any(rows[s].get('smoke') for s in seeds),
        'wall_clock_s': [rows[s].get('wall_clock_s') for s in seeds],
        'library_commit': next(
            (rows[s].get('library_commit') for s in seeds), None
        ),
    }


def judge_anchor(anchor_id: str, anchor: dict, runs: list[dict]) -> dict:
    """1 アンカー分の判定(腕ごとの許容と、差の規則)。"""
    metric = anchor['metric']
    tol_pt = float(anchor['tolerance_pt'])
    arms = []
    by_recipe: dict[str, dict] = {}
    for recipe, spec in anchor['arms'].items():
        got = collect_arm(runs, recipe, metric)
        original = [float(v) for v in spec['original']]
        original_mean = mean(original)
        delta_pt = (
            None if got['mean'] is None
            else (got['mean'] - original_mean) * 100
        )
        row = {
            **got,
            'arm': spec['arm'],
            'label': spec['label'],
            'original': original,
            'original_mean': original_mean,
            'delta_pt': delta_pt,
            'within_tolerance': (
                None if delta_pt is None else abs(delta_pt) <= tol_pt
            ),
        }
        arms.append(row)
        by_recipe[recipe] = row

    rules = []
    for rule in anchor.get('difference_rules', []):
        pos = by_recipe.get(rule['positive'], {}).get('mean')
        neg = by_recipe.get(rule['negative'], {}).get('mean')
        diff_pt = None if pos is None or neg is None else (pos - neg) * 100
        rules.append({
            **rule,
            'rerun_pt': diff_pt,
            'ok': None if diff_pt is None else compare_pt(
                diff_pt, float(rule['min_pt']), rule.get('comparison', 'gte')
            ),
        })

    checked = [a['within_tolerance'] for a in arms if a['within_tolerance'] is not None]
    rule_ok = [r['ok'] for r in rules if r['ok'] is not None]
    return {
        'anchor': anchor_id,
        'claim': anchor['claim'],
        'metric': metric,
        'metric_note': anchor['metric_note'],
        'tolerance_pt': tol_pt,
        'arms': arms,
        'difference_rules': rules,
        'complete': len(checked) == len(arms) and len(rule_ok) == len(rules),
        'ok': bool(checked) and all(checked) and all(rule_ok),
    }


def render_markdown(verdicts: list[dict], anchors: dict) -> str:
    """再走表(腕ごとの seed 別値・平均・原値との差・判定)を markdown にする。"""
    lines = [
        '# 計算型配線 較正アンカーの再走',
        '',
        f'生成: {datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")} JST',
        '',
        f'原値と許容幅の出所: `{anchors["source"]}`(anchors.json へ転記)',
        '',
    ]
    for verdict in verdicts:
        lines += [
            f'## {verdict["anchor"]}: {verdict["claim"]}',
            '',
            f'指標: `{verdict["metric"]}`({verdict["metric_note"]})/ '
            f'許容 ±{verdict["tolerance_pt"]}pt',
            '',
            '| 腕 | レシピ | 原値(seed 別) | 再走(seed 別) | 原値平均 | 再走平均 | 差(pt) | 判定 |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for arm in verdict['arms']:
            original = ' / '.join(_fmt(v) for v in arm['original'])
            rerun = ' / '.join(_fmt(v) for v in arm['values']) or '未実行'
            mark = {True: 'OK', False: '外れ', None: '未実行'}[
                arm['within_tolerance']
            ]
            smoke = '(スモーク)' if arm['smoke'] else ''
            lines.append(
                f'| {arm["arm"]} | `{arm["recipe"]}` | {original} | '
                f'{rerun}{smoke} | {_fmt(arm["original_mean"])} | '
                f'{_fmt(arm["mean"])} | {_fmt_pt(arm["delta_pt"])} | {mark} |'
            )
        if verdict['difference_rules']:
            lines += [
                '',
                '| 差の規則 | 下限(pt) | 原値(pt) | 再走(pt) | 判定 |',
                '|---|---|---|---|---|',
            ]
            for rule in verdict['difference_rules']:
                mark = {True: 'OK', False: '未達', None: '未実行'}[rule['ok']]
                sign = '>' if rule.get('comparison') == 'gt' else '>='
                lines.append(
                    f'| {rule["name"]} | {sign} {rule["min_pt"]:+.2f} | '
                    f'{rule["original_pt"]:+.2f} | {_fmt_pt(rule["rerun_pt"])} | '
                    f'{mark} |'
                )
        state = ('全件合格' if verdict['ok']
                 else ('未実行あり' if not verdict['complete'] else '未達あり'))
        lines += ['', f'判定: **{state}**', '']
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='較正アンカーの再走結果を原値と突き合わせる')
    parser.add_argument('--results', default=str(DEFAULT_RESULTS),
                        help=f'run JSON のディレクトリ(既定: {DEFAULT_RESULTS})')
    parser.add_argument('--out-dir',
                        help='表の出力先(既定: --results の 1 つ上)')
    parser.add_argument('--anchors', default=str(ANCHORS_PATH),
                        help='原値と許容幅(既定: 同ディレクトリの anchors.json)')
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        parser.error(f'結果ディレクトリがありません: {results_dir}')
    out_dir = Path(args.out_dir) if args.out_dir else results_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors = load_anchors(Path(args.anchors))
    runs = load_runs(results_dir)
    verdicts = [
        judge_anchor(anchor_id, anchor, runs)
        for anchor_id, anchor in anchors['anchors'].items()
    ]

    markdown = render_markdown(verdicts, anchors)
    md_path = out_dir / 'calibration_table.md'
    json_path = out_dir / 'calibration_table.json'
    md_path.write_text(markdown, encoding='utf-8')
    json_path.write_text(
        json.dumps(
            {
                'generated_at': datetime.now(JST).isoformat(),
                'anchors_source': anchors['source'],
                'n_runs': len(runs),
                'verdicts': verdicts,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding='utf-8',
    )
    print(markdown)
    print(f'書き出し: {md_path}')
    print(f'書き出し: {json_path}')
    return 0 if all(v['ok'] for v in verdicts) else 1


if __name__ == '__main__':
    sys.exit(main())
