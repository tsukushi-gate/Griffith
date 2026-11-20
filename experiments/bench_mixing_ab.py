"""モデル全体(代理)の s/it を新旧交互A/Bで測るドライバ。

`experiments/bench_mixing_blocks.py` を**無改造で子プロセス実行**し、
MD_BACKWARD_W(legacy / 未設定=適応)を切り替えて交互に走らせる。
別コンテナ間の比較は 15〜30% ずれるので、必ず**同一コンテナ内**で往復させること
(bench_kernel_gate.py / bench_backward_w.py と同じ作法)。

読み取るのは子が出力した結果JSONの
  - train_step.median_s          … 1 train step の中央値(全体指標)
  - blocks.by_category[*].total_s … ブロック別の内訳(token_mix が本タスクの標的)

使用例:
    uv run modal run scripts/modal_app.py --script bench_mixing_ab.py \\
        --args "--repeats 2 --scale paper"

出力: results/kernel2/mixing_ab_<YYYYmmdd_HHMMSS>.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

VARIANTS = ('legacy', 'new')
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'bench_mixing_blocks.py')
JSON_MARKER = '結果JSON: '


def run_child(variant: str, scale: str, extra: list[str]) -> dict:
    """bench_mixing_blocks.py を1回実行し、その結果JSONを読み込んで返す。"""
    env = dict(os.environ)
    if variant == 'legacy':
        env['MD_BACKWARD_W'] = 'legacy'
    else:
        env.pop('MD_BACKWARD_W', None)
    cmd = [sys.executable, CHILD, '--scale', scale, '--no-wandb',
           '--tag', f'ab_{variant}'] + extra
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    proc.check_returncode()

    for line in proc.stdout.splitlines():
        if line.startswith(JSON_MARKER):
            path = line[len(JSON_MARKER):].strip()
            with open(path, encoding='utf-8') as f:
                return json.load(f)
    raise RuntimeError('子プロセスの出力から結果JSONのパスを取得できなかった')


def summarize(payload: dict) -> dict:
    blocks = payload.get('blocks') or {}
    by_cat = blocks.get('by_category') or {}
    return {
        'step_median_s': payload['train_step']['median_s'],
        'by_category_total_s': {k: v['total_s'] for k, v in by_cat.items()},
        'peak_memory_bytes': payload.get('peak_memory_bytes'),
    }


def _med(values):
    return statistics.median(values) if values else None


def print_report(samples: dict) -> None:
    print('\n=== モデル全体 1 train step (中央値 s/it、同一コンテナ内交互A/B) ===')
    leg = _med([s['step_median_s'] for s in samples['legacy']])
    new = _med([s['step_median_s'] for s in samples['new']])
    print(f'  legacy = {leg:.6f} s/it')
    print(f'  new    = {new:.6f} s/it')
    print(f'  全体speedup = {leg / new:.3f}x')

    print('\n=== ブロック別 total_s(中央値) ===')
    header = f'{"category":>16} {"legacy_s":>12} {"new_s":>12} {"speedup":>9}'
    print(header)
    print('-' * len(header))
    cats = sorted({c for v in VARIANTS for s in samples[v]
                   for c in s['by_category_total_s']})
    for cat in cats:
        lv = _med([s['by_category_total_s'].get(cat)
                   for s in samples['legacy'] if cat in s['by_category_total_s']])
        nv = _med([s['by_category_total_s'].get(cat)
                   for s in samples['new'] if cat in s['by_category_total_s']])
        if lv is None or nv is None:
            continue
        print(f'{cat:>16} {lv:>12.6f} {nv:>12.6f} {lv / nv:>8.3f}x')


def parse_args():
    p = argparse.ArgumentParser(description='モデル全体の新旧A/B')
    p.add_argument('--repeats', type=int, default=2, help='往復回数(既定2)')
    p.add_argument('--scale', type=str, default='paper')
    p.add_argument('--skip-profile', action='store_true',
                   help='子のprofilerを省く(s/itだけ欲しいとき)')
    p.add_argument('--out', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    extra = ['--skip-profile'] if args.skip_profile else []
    t0 = time.perf_counter()

    samples: dict = {v: [] for v in VARIANTS}
    raw: list[dict] = []
    for rep in range(args.repeats):
        for variant in VARIANTS:
            print(f'\n--- モデル全体 variant={variant} '
                  f'(往復 {rep + 1}/{args.repeats}) ---', flush=True)
            payload = run_child(variant, args.scale, extra)
            s = summarize(payload)
            samples[variant].append(s)
            raw.append({'variant': variant, 'rep': rep, **s})
            print(f'  [{variant}] step={s["step_median_s"]:.6f} s/it', flush=True)

    print_report(samples)

    out = args.out or os.path.join(
        'results', 'kernel2', f'mixing_ab_{datetime.now(JST):%Y%m%d_%H%M%S}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'rows': raw, 'scale': args.scale,
                   'elapsed_s': time.perf_counter() - t0}, f,
                  ensure_ascii=False, indent=2)
    print(f'\n結果JSON: {os.path.abspath(out)}')
    print(f'所要 {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
