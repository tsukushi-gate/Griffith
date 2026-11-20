"""退行ゲート専用のA/B計測。

`experiments/bench_lut_kernel.py` は「現行コードの絶対性能」を測るベンチだが、
本スクリプトは **同一コンテナ・同一GPU上で新旧を交互に測って差分を出す**ことに
特化している。目的は「LogicMixer b=128 が baseline 比 0.86x」という報告が
実在の退行なのか実行間ノイズなのかを判定すること。

退行は2成分に分解できる:
  1. 起動設定(lut_kernels.cu): 占有率クエリのキャッシュ / cudaDeviceSynchronize 削除
     / clamp_block_y。環境変数 MD_LUT_LEGACY_LAUNCH=1 で旧挙動に戻せる(プロセス静的)
  2. 転置ディスパッチ(cuda/transpose.py の fast_transpose)。
     griffith_logic.core.functions.fast_transpose を差し替えれば
     プロセス内で `x.t().contiguous()`(=変更前と同一の1行)に戻せる

1 はプロセス静的なので子プロセスを分けて交互実行、2 はプロセス内で交互実行する。
どちらも「同じ条件を複数回・交互に」測ることで、コンテナ差やGPUクロック変動を
差分から追い出す。

使い方:
    uv run python experiments/bench_kernel_gate.py --repeats 5
    uv run modal run scripts/modal_app.py --script bench_kernel_gate.py \\
        --args "--repeats 5"
    # 損益分岐スイープ(幅×バッチ)だけを回す
    uv run modal run scripts/modal_app.py --script bench_kernel_gate.py \\
        --args "--only sweep"
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

# モデル構成・計測プリミティブは既存ベンチと厳密に同一のものを使う(比較可能性のため)
from bench_lut_kernel import (  # noqa: E402
    MODEL_SPECS,
    detect_device,
    free_memory,
    is_oom_error,
    route_counter,
    route_label,
    sync,
    timed_median,
)

import griffith_logic.core.functions as md_functions  # noqa: E402
from griffith_logic.layers.lut import LutLayer  # noqa: E402

JST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 計測パラメータ(定数として明示)
# ---------------------------------------------------------------------------

MODEL_BATCHES = (128, 512)
# 既存ベンチ(warmup=5, iters=20)より大幅に増やす。0.86x がノイズかを判定するには
# 1測定あたりの分散を落とす必要があるため。
MODEL_WARMUP = 20
MODEL_ITERS = 100
DEFAULT_REPEATS = 5

# 損益分岐スイープ(単層LutLayer、境界転置が必ず発生する条件)
SWEEP_WIDTHS = (1024, 4096, 16384, 65536, 400_000)
SWEEP_BATCHES = (64, 128, 512, 1024)
SWEEP_N_INPUTS = 2
SWEEP_PARAM = 'softmax'
SWEEP_WARMUP = 10
SWEEP_ITERS = 30
SWEEP_REPEATS = 3

LR = 0.01
LAUNCH_VARIANTS = ('legacy', 'current')
TRANSPOSE_VARIANTS = ('legacy', 'current')


# ---------------------------------------------------------------------------
# 転置経路の切替(プロセス内)
# ---------------------------------------------------------------------------

_TRANSPOSE_CURRENT = md_functions.fast_transpose


def _transpose_legacy(x):
    """最適化前の実装そのもの(functions.py に直書きされていた1行)。"""
    return x.t().contiguous()


def set_transpose(variant: str) -> None:
    md_functions.fast_transpose = (
        _TRANSPOSE_CURRENT if variant == 'current' else _transpose_legacy
    )


def launch_variant_from_env() -> str:
    return 'legacy' if os.environ.get('MD_LUT_LEGACY_LAUNCH') == '1' else 'current'


# ---------------------------------------------------------------------------
# 子プロセス: 実測本体
# ---------------------------------------------------------------------------

def measure_models(device: str, repeats: int) -> list[dict]:
    """モデル級の s/it を (転置変種 × repeats) で交互計測する。"""
    rows = []
    for name, builder, step_maker in MODEL_SPECS:
        model, cfg = builder(device)
        model.train()
        for batch in MODEL_BATCHES:
            step = step_maker(model, cfg, batch, device)

            set_transpose('current')
            with route_counter() as counts:
                step()
                sync(device)
            route = route_label(counts)

            samples = {v: [] for v in TRANSPOSE_VARIANTS}
            for _ in range(repeats):
                for variant in TRANSPOSE_VARIANTS:
                    set_transpose(variant)
                    stats = timed_median(step, device, MODEL_WARMUP, MODEL_ITERS)
                    samples[variant].append(stats['median_s'])
            set_transpose('current')

            for variant, values in samples.items():
                rows.append({
                    'kind': 'model',
                    'model': name,
                    'batch': batch,
                    'route': route,
                    'launch': launch_variant_from_env(),
                    'transpose': variant,
                    'median_s': statistics.median(values),
                    'samples_s': values,
                })
                print(f'  [{launch_variant_from_env()}/{variant}] {name} b={batch} '
                      f'median={statistics.median(values):.6f}s/it '
                      f'spread={(max(values) - min(values)) / statistics.median(values):.2%}',
                      flush=True)
        del model
        free_memory(device)
    return rows


def measure_sweep(device: str, repeats: int) -> list[dict]:
    """幅×バッチの損益分岐スイープ(単層LutLayerのfwd+bwd)。"""
    rows = []
    for width in SWEEP_WIDTHS:
        for batch in SWEEP_BATCHES:
            try:
                layer = LutLayer(
                    in_dim=width, out_dim=width, n_inputs=SWEEP_N_INPUTS,
                    parameterization=SWEEP_PARAM, connections='random',
                ).to(device)
                layer.train()
                x = torch.rand(batch, width, device=device)

                def fwd_bwd(layer=layer, x=x):
                    layer.zero_grad(set_to_none=True)
                    xin = x.detach().requires_grad_(True)
                    layer(xin).sum().backward()

                samples = {v: [] for v in TRANSPOSE_VARIANTS}
                for _ in range(repeats):
                    for variant in TRANSPOSE_VARIANTS:
                        set_transpose(variant)
                        stats = timed_median(fwd_bwd, device, SWEEP_WARMUP, SWEEP_ITERS)
                        samples[variant].append(stats['median_s'])
                set_transpose('current')

                for variant, values in samples.items():
                    rows.append({
                        'kind': 'sweep',
                        'width': width,
                        'batch': batch,
                        'numel': batch * width,
                        'launch': launch_variant_from_env(),
                        'transpose': variant,
                        'status': 'ok',
                        'median_s': statistics.median(values),
                        'samples_s': values,
                    })
                leg = statistics.median(samples['legacy'])
                cur = statistics.median(samples['current'])
                print(f'  sweep w={width} b={batch} numel={batch * width} '
                      f'legacy={leg:.6f} current={cur:.6f} ratio={leg / cur:.3f}x',
                      flush=True)
                del layer, x
            except Exception as exc:  # noqa: BLE001
                if not is_oom_error(exc):
                    raise
                for variant in TRANSPOSE_VARIANTS:
                    rows.append({
                        'kind': 'sweep', 'width': width, 'batch': batch,
                        'numel': batch * width,
                        'launch': launch_variant_from_env(),
                        'transpose': variant, 'status': 'oom',
                        'median_s': None, 'samples_s': [],
                    })
                print(f'  sweep w={width} b={batch}: OOM(記録してスキップ)', flush=True)
            finally:
                set_transpose('current')
                free_memory(device)
    return rows


def run_child(args) -> None:
    device = detect_device()
    print(f'[child] device={device} launch={launch_variant_from_env()}', flush=True)
    rows = []
    if args.only in ('all', 'models'):
        rows += measure_models(device, args.repeats)
    if args.only in ('all', 'sweep'):
        rows += measure_sweep(device, SWEEP_REPEATS)
    print('###JSON###' + json.dumps(rows), flush=True)


# ---------------------------------------------------------------------------
# 親プロセス: 起動設定変種を交互に回して集計
# ---------------------------------------------------------------------------

def spawn_child(launch: str, only: str, repeats: int) -> list[dict]:
    env = dict(os.environ)
    if launch == 'legacy':
        env['MD_LUT_LEGACY_LAUNCH'] = '1'
    else:
        env.pop('MD_LUT_LEGACY_LAUNCH', None)
    cmd = [sys.executable, os.path.abspath(__file__), '--child',
           '--repeats', str(repeats), '--only', only]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    proc.check_returncode()
    for line in proc.stdout.splitlines():
        if line.startswith('###JSON###'):
            return json.loads(line[len('###JSON###'):])
    raise RuntimeError('子プロセスから結果JSONが得られなかった')


def aggregate(rows: list[dict]) -> dict:
    """(キー) → {変種: 中央値} に畳む。"""
    table: dict = {}
    for r in rows:
        if r['median_s'] is None:
            continue
        if r['kind'] == 'model':
            key = (r['kind'], r['model'], r['batch'])
        else:
            key = (r['kind'], r['width'], r['batch'])
        cell = table.setdefault(key, {})
        cell.setdefault((r['launch'], r['transpose']), []).append(r['median_s'])
    return table


def print_model_report(table: dict) -> None:
    print('\n=== モデル級 A/B(中央値 s/it。baseline = legacy起動 + legacy転置) ===')
    header = (f'{"model":>18} {"batch":>6} {"baseline":>11} {"iso(起動のみ)":>14} '
              f'{"既定経路":>11} {"iso/base":>9} {"既定/base":>10}')
    print(header)
    print('-' * len(header))
    for key in sorted(k for k in table if k[0] == 'model'):
        cell = table[key]
        base = _med(cell.get(('legacy', 'legacy')))
        iso = _med(cell.get(('current', 'legacy')))
        cur = _med(cell.get(('current', 'current')))
        if None in (base, iso, cur):
            continue
        print(f'{key[1]:>18} {key[2]:>6} {base:>11.6f} {iso:>14.6f} {cur:>11.6f} '
              f'{base / iso:>8.3f}x {base / cur:>9.3f}x')


def print_sweep_report(table: dict) -> None:
    print('\n=== 損益分岐スイープ(単層LutLayer fwd+bwd 中央値 s/it、既定起動設定) ===')
    header = (f'{"width":>8} {"batch":>6} {"numel":>12} {"変更前":>11} '
              f'{"既定経路":>11} {"speedup":>9}')
    print(header)
    print('-' * len(header))
    for key in sorted(k for k in table if k[0] == 'sweep'):
        cell = table[key]
        legacy = _med(cell.get(('current', 'legacy')))
        current = _med(cell.get(('current', 'current')))
        if None in (legacy, current):
            continue
        print(f'{key[1]:>8} {key[2]:>6} {key[1] * key[2]:>12} {legacy:>11.6f} '
              f'{current:>11.6f} {legacy / current:>8.3f}x')


def _med(values):
    return statistics.median(values) if values else None


def parse_args():
    p = argparse.ArgumentParser(description='退行ゲートのA/B計測')
    p.add_argument('--child', action='store_true', help='内部用(子プロセス実行)')
    p.add_argument('--repeats', type=int, default=DEFAULT_REPEATS,
                   help='モデル級の反復回数(既定5)')
    p.add_argument('--launch-repeats', type=int, default=2,
                   help='起動設定変種を何往復させるか(既定2)')
    p.add_argument('--only', choices=('all', 'models', 'sweep'), default='all')
    p.add_argument('--out', type=str, default=None, help='結果JSONの保存先')
    return p.parse_args()


def main():
    args = parse_args()
    if args.child:
        run_child(args)
        return

    if not torch.cuda.is_available():
        print('CUDAが無い環境では本ベンチの意味がないため終了する')
        return

    t0 = time.perf_counter()
    rows: list[dict] = []

    # モデル級: 起動設定 legacy/current を交互に(コンテナ差・クロック変動の除去)
    if args.only in ('all', 'models'):
        for rep in range(args.launch_repeats):
            for launch in LAUNCH_VARIANTS:
                print(f'\n--- モデル級 起動設定={launch} '
                      f'(往復 {rep + 1}/{args.launch_repeats}) ---', flush=True)
                rows += spawn_child(launch, 'models', args.repeats)

    # 損益分岐スイープ: 既定の起動設定のみ(転置経路の閾値の根拠を取るのが目的)
    if args.only in ('all', 'sweep'):
        print('\n--- 損益分岐スイープ(既定の起動設定) ---', flush=True)
        rows += spawn_child('current', 'sweep', SWEEP_REPEATS)

    table = aggregate(rows)
    if args.only in ('all', 'models'):
        print_model_report(table)
    if args.only in ('all', 'sweep'):
        print_sweep_report(table)

    out = args.out or os.path.join(
        'results', f'kernel_gate_{datetime.now(JST):%Y%m%d_%H%M%S}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'rows': rows, 'elapsed_s': time.perf_counter() - t0}, f, indent=2)
    print(f'\n結果JSON: {out}')
    print(f'所要 {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
