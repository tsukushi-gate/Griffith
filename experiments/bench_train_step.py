"""学習1ステップの所要時間ベンチ: Python実装 / 融合CUDAカーネル / +CUDA Graphs の交互比較。

`docs/guides/lut-fast-path.md` §8 の性能表を追試するためのスクリプト。
同じプロセス・同じGPUの中で、次の3条件を**ラウンドごとに順番を入れ替えながら交互に**測る
(別プロセス・別コンテナ間の比較はGPUの状態差でずれるため使わない)。

    python  — 融合カーネルを使わない(環境変数 MD_LUT_PYTHON_PATH で iwp/warp を
              Python 実装へ戻す。softmax/independent には Python 経路の切替が無いので
              この条件は iwp/warp でのみ測る)
    fused   — 融合CUDAカーネル + 通常(eager)の学習ステップ
    graphs  — 融合CUDAカーネル + make_train_step(CUDA Graphs)

各条件はラウンドごとに同じseedでモデルを作り直し、warmup ステップを捨ててから
`--steps` ステップを測る。1ラウンドの値は「steps ステップの合計時間 / steps」
(前後で torch.cuda.synchronize)。報告値はラウンド間の**中央値**(最小・最大も併記)。
入力はあらかじめGPU上に作った乱数データで、DataLoader の読み込み時間は含まない。

使用例:
    uv run python experiments/bench_train_step.py --param softmax --n-inputs 2 --batch 512
    uv run python experiments/bench_train_step.py --param iwp --n-inputs 6 --rounds 7
    uv run python experiments/bench_train_step.py --param warp --n-inputs 6 --steps 20

結果JSONは --out(既定 results/bench_train_step/<日時>.json)へ書き、wandb が有効なら
指標とJSON Artifactも記録する(group='bench-train-step'。--no-wandb で無効化)。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from griffith_logic.group_sum import GroupSum  # noqa: E402
from griffith_logic.layers.lut import LutLayer  # noqa: E402
from griffith_logic.training import make_train_step  # noqa: E402

JST = timezone(timedelta(hours=9))
PYTHON_PATH_ENV = 'MD_LUT_PYTHON_PATH'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--param', default='softmax',
                   choices=['softmax', 'independent', 'iwp', 'warp'])
    p.add_argument('--n-inputs', type=int, default=2, choices=[2, 4, 6])
    p.add_argument('--in-dim', type=int, default=256)
    p.add_argument('--width', type=int, default=8000, help='各LutLayerのゲート数')
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--classes', type=int, default=10)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--warmup', type=int, default=10, help='各ラウンドで捨てるステップ数')
    p.add_argument('--steps', type=int, default=100, help='各ラウンドで測るステップ数')
    p.add_argument('--rounds', type=int, default=5, help='交互測定のラウンド数')
    p.add_argument('--arms', nargs='+', default=None,
                   choices=['python', 'fused', 'graphs'],
                   help='測る条件(既定: iwp/warpは3条件、それ以外は fused/graphs)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--no-wandb', action='store_true', help='wandbを無効化')
    p.add_argument('--tag', default='', help='結果ラベル')
    return p.parse_args()


def build(args: argparse.Namespace, *, capturable: bool):
    torch.manual_seed(args.seed)
    dims = [args.in_dim] + [args.width] * args.layers
    layers: list[torch.nn.Module] = [
        LutLayer(dims[i], dims[i + 1], n_inputs=args.n_inputs, device='cuda',
                 parameterization=args.param)
        for i in range(args.layers)
    ]
    layers.append(GroupSum(args.classes, tau=1.0))
    model = torch.nn.Sequential(*layers).cuda()
    model.train()
    if capturable:
        opt = torch.optim.AdamW(model.parameters(),
                                lr=torch.tensor(1e-2, device='cuda'), capturable=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, opt


def make_step(args: argparse.Namespace, arm: str):
    model, opt = build(args, capturable=(arm == 'graphs'))

    def compute_loss(x, y):
        return F.cross_entropy(model(x), y)

    if arm == 'graphs':
        step, info = make_train_step(model, opt, compute_loss, mode='force')
        return step, info

    def eager(x, y):
        opt.zero_grad()
        loss = compute_loss(x, y)
        loss.backward()
        opt.step()
        return loss.detach()

    return eager, None


def measure(args: argparse.Namespace, arm: str, data) -> tuple[float, dict | None]:
    prev = os.environ.get(PYTHON_PATH_ENV)
    if arm == 'python':
        os.environ[PYTHON_PATH_ENV] = args.param
    else:
        os.environ.pop(PYTHON_PATH_ENV, None)
    try:
        step, info = make_step(args, arm)
        n = len(data)
        for i in range(args.warmup):
            step(*data[i % n])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(args.steps):
            step(*data[i % n])
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.steps, info
    finally:
        if prev is None:
            os.environ.pop(PYTHON_PATH_ENV, None)
        else:
            os.environ[PYTHON_PATH_ENV] = prev


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDAが必要です')
    arms = args.arms or (['python', 'fused', 'graphs'] if args.param in ('iwp', 'warp')
                         else ['fused', 'graphs'])
    if 'python' in arms and args.param not in ('iwp', 'warp'):
        raise SystemExit("'python' 条件は iwp/warp のみ(MD_LUT_PYTHON_PATH の対象)")

    g = torch.Generator(device='cuda').manual_seed(args.seed + 1)
    data = [
        (torch.bernoulli(torch.full((args.batch, args.in_dim), 0.5, device='cuda'),
                         generator=g),
         torch.randint(0, args.classes, (args.batch,), device='cuda', generator=g))
        for _ in range(8)
    ]

    samples: dict[str, list[float]] = {a: [] for a in arms}
    graph_info = None
    for r in range(args.rounds):
        order = arms[r % len(arms):] + arms[:r % len(arms)]  # ラウンドごとに順番を回す
        for arm in order:
            per_step, info = measure(args, arm, data)
            samples[arm].append(per_step)
            if info is not None:
                graph_info = {k: info[k] for k in ('enabled', 'summary')}
            print(f'round {r} {arm:>6}: {per_step * 1e3:.3f} ms/step', flush=True)

    summary = {
        a: {'median_ms': statistics.median(v) * 1e3, 'min_ms': min(v) * 1e3,
            'max_ms': max(v) * 1e3, 'samples_ms': [x * 1e3 for x in v]}
        for a, v in samples.items()
    }
    print('\n=== ms/step(ラウンド間の中央値 [最小, 最大])===')
    for a, s in summary.items():
        print(f'{a:>6}: {s["median_ms"]:.3f} [{s["min_ms"]:.3f}, {s["max_ms"]:.3f}]')
    if graph_info is not None:
        print(f'make_train_step: {graph_info}')

    payload = {
        'args': vars(args),
        'arms': arms,
        'summary': summary,
        'graph_info': graph_info,
        'device': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'timestamp_jst': datetime.now(JST).isoformat(timespec='seconds'),
    }
    out = Path(args.out) if args.out else (
        _REPO_ROOT / 'results' / 'bench_train_step'
        / f'{datetime.now(JST):%Y%m%d_%H%M%S}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'結果JSON: {out}')
    if not args.no_wandb:
        log_to_wandb(args, payload, out)


def log_to_wandb(args: argparse.Namespace, payload: dict, json_path: Path) -> None:
    """wandbへ指標とJSON Artifactを記録する。失敗時は警告を出して続行。"""
    try:
        from griffith_logic.training.wandb_utils import (
            finish,
            init_wandb,
            log_artifact,
            log_metrics,
        )
    except Exception as exc:  # noqa: BLE001
        print(f'警告: wandb_utilsをimportできずwandb記録をスキップ: {exc}')
        return
    try:
        suffix = f'_{args.tag}' if args.tag else ''
        name = f'train_step_{args.param}_n{args.n_inputs}_b{args.batch}{suffix}'
        ok = init_wandb(
            run_name=name,
            config_dict={**payload['args'], 'device_name': payload['device'],
                         'torch': payload['torch']},
            device='cuda',
            tags=['bench', 'train-step', args.param],
            group='bench-train-step',
            job_type='eval',
        )
        if not ok:
            print('警告: wandbが無効のため記録をスキップ(結果はJSONに保存済み)')
            return
        log_metrics({f'{arm}/median_ms': s['median_ms']
                     for arm, s in payload['summary'].items()})
        log_artifact(str(json_path), name=f'bench-{name}', type='benchmark',
                     metadata={'device': payload['device']})
        finish()
    except Exception as exc:  # noqa: BLE001
        print(f'警告: wandb記録に失敗(結果はJSONに保存済み): {exc}')


if __name__ == '__main__':
    main()
