"""head(実スケール out_dim=400k)の退行チェック用新旧交互A/B。

`experiments/bench_lut_kernel.py --scale large` を**無改造で子プロセス実行**し、
MD_BACKWARD_W(legacy / 未設定=適応)を切り替えて**同一コンテナ内で交互に**走らせる。
別コンテナ間の比較は 15〜30% ずれるので使わない(これを踏んで
存在しない退行を報告した事故がある)。

読み取るのは子の結果JSONの `micro_large` 行の forward / forward_backward 中央値。

使用例:
    uv run modal run scripts/modal_app.py --script bench_lut_kernel_ab.py \\
        --args "--repeats 2"

出力: results/kernel2/lut_kernel_ab_<YYYYmmdd_HHMMSS>.json
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
                     'bench_lut_kernel.py')
JSON_MARKER = '結果JSON: '


def run_child(variant: str, extra: list[str]) -> dict:
    env = dict(os.environ)
    if variant == 'legacy':
        env['MD_BACKWARD_W'] = 'legacy'
    else:
        env.pop('MD_BACKWARD_W', None)
    cmd = [sys.executable, CHILD, '--scale', 'large', '--no-wandb',
           '--skip-models', '--tag', f'ab_{variant}'] + extra
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    proc.check_returncode()
    for line in proc.stdout.splitlines():
        if line.startswith(JSON_MARKER):
            with open(line[len(JSON_MARKER):].strip(), encoding='utf-8') as f:
                return json.load(f)
    raise RuntimeError('子プロセスの出力から結果JSONのパスを取得できなかった')


def row_key(r: dict) -> str:
    size = (str(r.get('out_dim')) if r.get('variant', 'single_layer') == 'single_layer'
            else str(r.get('layer_sizes')))
    return (f"{r.get('variant', 'single_layer')}/{r['parameterization']}"
            f"/b{r['batch']}/{size}")


def collect(payload: dict) -> dict:
    out = {}
    for r in payload.get('micro_large') or []:
        if r.get('status') != 'ok':
            continue
        out[row_key(r)] = {
            'forward_s': r['forward']['median_s'],
            'fwd_bwd_s': r['forward_backward']['median_s'],
        }
    return out


def _med(values):
    return statistics.median(values) if values else None


def print_report(samples: dict) -> None:
    print('\n=== head(実スケール)退行チェック: fwd+bwd 中央値 s/it ===')
    header = f'{"case":>44} {"legacy_s":>12} {"new_s":>12} {"speedup":>9}'
    print(header)
    print('-' * len(header))
    keys = sorted({k for v in VARIANTS for s in samples[v] for k in s})
    for key in keys:
        lv = _med([s[key]['fwd_bwd_s'] for s in samples['legacy'] if key in s])
        nv = _med([s[key]['fwd_bwd_s'] for s in samples['new'] if key in s])
        if lv is None or nv is None:
            continue
        print(f'{key:>44} {lv:>12.6f} {nv:>12.6f} {lv / nv:>8.3f}x')


def parse_args():
    p = argparse.ArgumentParser(description='head退行チェックの新旧A/B')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--out', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()
    samples: dict = {v: [] for v in VARIANTS}
    raw: list[dict] = []
    for rep in range(args.repeats):
        for variant in VARIANTS:
            print(f'\n--- head A/B variant={variant} '
                  f'(往復 {rep + 1}/{args.repeats}) ---', flush=True)
            got = collect(run_child(variant, []))
            samples[variant].append(got)
            raw.append({'variant': variant, 'rep': rep, 'rows': got})

    print_report(samples)

    out = args.out or os.path.join(
        'results', 'kernel2',
        f'lut_kernel_ab_{datetime.now(JST):%Y%m%d_%H%M%S}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'rows': raw, 'elapsed_s': time.perf_counter() - t0}, f,
                  ensure_ascii=False, indent=2)
    print(f'\n結果JSON: {os.path.abspath(out)}')
    print(f'所要 {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
