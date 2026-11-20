"""`lut_backward_w` の行方向並列化(gridDim.x)の新旧A/B計測。

■ 何を測るか
    1 step の 76.5% を占めていた `lut_backward_w_kernel` を**単体で**、
    旧実装(gridDim.x = 1 固定 = 行方向32スレッド)と
    新実装(gridDim.x = R、Rは out_dim/行数/メモリから適応決定)で比較する。

■ なぜ子プロセスを分けるか
    切替フラグ MD_BACKWARD_W はプロセス静的(getenv を静的初期化で1回だけ読む)なので、
    プロセス内では交互に測れない。**同一コンテナ内で子プロセスを交互に起動**して、
    コンテナ差・GPUクロック変動を差分から追い出す(bench_kernel_gate.py と同じ作法)。
    別コンテナ間の比較は 15〜30% ずれるので使わない。

■ 何を出すか
    1. スイープ: out_dim × 行数 の格子で legacy/new の中央値と speedup。
       ここから損益分岐(どの形状で新経路が勝つか)と、
       **大 out_dim(head 400k)側が退行していないこと**を読む。
    2. token_mix 代表形状の詳細(の ToeplitzTokenMix と同じ潰し方の形状)。

使用例:
    uv run modal run scripts/modal_app.py --script bench_backward_w.py \\
        --args "--repeats 3"
    uv run python experiments/bench_backward_w.py --repeats 3   # ローカルGPU

出力: results/kernel2/backward_w_<YYYYmmdd_HHMMSS>.json
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from griffith_logic.cuda import get_cuda_module  # noqa: E402

JST = timezone(timedelta(hours=9))

VARIANTS = ('legacy', 'new')

WARMUP = 3
MAX_ITERS = 30
MIN_ITERS = 5
DEFAULT_REPEATS = 3
# 1ケースあたりの計測予算(秒)。旧経路は out_dim=1 で総スレッド32本しか立たず
# 1呼び出しが秒オーダーになりうるため、パイロット計測から反復数を決める。
CASE_BUDGET_S = 3.0

# 入出力のGPUメモリ暴走を避けるためのガード(要素数)。
# grad_y = out_dim × rows(fp32、4バイト/要素)。5e8要素 = 2GiB。
MAX_GRAD_Y_ELEMS = 5e8

IN_DIM = 64
N_INPUTS = 2

# 損益分岐スイープ。out_dim は token_mix(1〜16)から head(400k)まで、
# 行数は 1k(小バッチ)から 3.3M(の token_mix 実スケール)まで。
SWEEP_OUT_DIMS = (1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536, 400_000)
SWEEP_ROWS = (1024, 65_536, 1_048_576, 3_244_032)

# 損益分岐点の細分スイープ。初回スイープで rows=1024 は新経路が負け(0.71〜0.87x)、
# rows=65536 は全 out_dim で勝ち(1.24〜16.7x)だったので、その間を刻んで境界を出す。
BREAKEVEN_OUT_DIMS = (16, 64, 256, 1024, 4096)
BREAKEVEN_ROWS = (1024, 2048, 4096, 8192, 16_384, 32_768, 65_536)

# ToeplitzTokenMix 代表形状(out_dim=K が 1〜16、行数 = B*T*Cg)。
TOKEN_MIX_CASES = (
    {'out_dim': 1, 'rows': 3_244_032},
    {'out_dim': 2, 'rows': 3_244_032},
    {'out_dim': 4, 'rows': 3_244_032},
    {'out_dim': 8, 'rows': 3_244_032},
    {'out_dim': 16, 'rows': 3_244_032},
)


def variant_from_env() -> str:
    return 'legacy' if os.environ.get('MD_BACKWARD_W') == 'legacy' else 'new'


def sync():
    torch.cuda.synchronize()


def timed_median(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def is_oom(exc: BaseException) -> bool:
    return 'out of memory' in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError)


def measure_case(cuda, out_dim: int, rows: int, n_inputs: int = N_INPUTS) -> dict:
    """`cuda.backward_w` 単体の中央値を返す(形状は本番と同じ転置済みレイアウト)。"""
    if out_dim * rows > MAX_GRAD_Y_ELEMS:
        return {'status': 'skip_too_large', 'median_s': None}
    torch.manual_seed(0)
    try:
        x_t = torch.rand(IN_DIM, rows, device='cuda').contiguous()
        indices = torch.randint(0, IN_DIM, (n_inputs, out_dim),
                                device='cuda', dtype=torch.int64).contiguous()
        grad_y = torch.randn(out_dim, rows, device='cuda').contiguous()

        def call():
            cuda.backward_w(x_t, indices, grad_y, n_inputs)

        # パイロット1回で所要時間を掴み、計測予算に収まる反復数を決める
        # (旧経路は極小 out_dim で1呼び出しが秒オーダーになるため)
        sync()
        t0 = time.perf_counter()
        call()
        sync()
        pilot = max(time.perf_counter() - t0, 1e-6)
        iters = int(min(MAX_ITERS, max(MIN_ITERS, CASE_BUDGET_S / pilot)))
        warmup = min(WARMUP, iters)

        med = timed_median(call, warmup, iters)
        x_t, indices, grad_y = None, None, None  # noqa: F841 (GPUメモリ解放のため)
        torch.cuda.empty_cache()
        return {'status': 'ok', 'median_s': med, 'iters': iters}
    except Exception as exc:  # noqa: BLE001
        if not is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return {'status': 'oom', 'median_s': None}


def run_child(args) -> None:
    cuda = get_cuda_module()
    if cuda is None:
        raise RuntimeError('CUDAカーネルが読み込めない')
    variant = variant_from_env()
    print(f'[child] variant={variant} '
          f'gpu={torch.cuda.get_device_name(0)}', flush=True)

    rows_out = []
    if args.only == 'breakeven':
        for out_dim in BREAKEVEN_OUT_DIMS:
            for rows in BREAKEVEN_ROWS:
                r = measure_case(cuda, out_dim, rows)
                rows_out.append({'kind': 'breakeven', 'variant': variant,
                                 'n_inputs': N_INPUTS,
                                 'out_dim': out_dim, 'rows': rows, **r})
                print(f'  breakeven out_dim={out_dim} rows={rows} '
                      f"{variant}: {r['median_s']} ({r['status']})", flush=True)
        print('###JSON###' + json.dumps(rows_out), flush=True)
        return

    for case in TOKEN_MIX_CASES:
        r = measure_case(cuda, case['out_dim'], case['rows'])
        rows_out.append({'kind': 'token_mix', 'variant': variant,
                         'n_inputs': N_INPUTS, **case, **r})
        print(f"  token_mix out_dim={case['out_dim']} rows={case['rows']} "
              f"{variant}: {r['median_s']} ({r['status']})", flush=True)

    if args.only in ('all', 'sweep'):
        for out_dim in SWEEP_OUT_DIMS:
            for rows in SWEEP_ROWS:
                r = measure_case(cuda, out_dim, rows)
                rows_out.append({'kind': 'sweep', 'variant': variant,
                                 'n_inputs': N_INPUTS,
                                 'out_dim': out_dim, 'rows': rows, **r})
                print(f'  sweep out_dim={out_dim} rows={rows} '
                      f"{variant}: {r['median_s']} ({r['status']})", flush=True)

    print('###JSON###' + json.dumps(rows_out), flush=True)


def spawn_child(variant: str, only: str) -> list[dict]:
    env = dict(os.environ)
    if variant == 'legacy':
        env['MD_BACKWARD_W'] = 'legacy'
    else:
        env.pop('MD_BACKWARD_W', None)
    cmd = [sys.executable, os.path.abspath(__file__), '--child', '--only', only]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    proc.check_returncode()
    for line in proc.stdout.splitlines():
        if line.startswith('###JSON###'):
            return json.loads(line[len('###JSON###'):])
    raise RuntimeError('子プロセスから結果JSONが得られなかった')


def aggregate(rows: list[dict]) -> dict:
    table: dict = {}
    for r in rows:
        if r.get('median_s') is None:
            continue
        key = (r['kind'], r['out_dim'], r['rows'])
        table.setdefault(key, {}).setdefault(r['variant'], []).append(r['median_s'])
    return table


def print_report(table: dict, kind: str, title: str) -> None:
    print(f'\n=== {title} ===')
    header = (f'{"out_dim":>9} {"rows":>10} {"legacy_s":>12} {"new_s":>12} '
              f'{"speedup":>9}')
    print(header)
    print('-' * len(header))
    for key in sorted(k for k in table if k[0] == kind):
        cell = table[key]
        leg = statistics.median(cell['legacy']) if 'legacy' in cell else None
        new = statistics.median(cell['new']) if 'new' in cell else None
        if leg is None or new is None:
            continue
        print(f'{key[1]:>9} {key[2]:>10} {leg:>12.6f} {new:>12.6f} '
              f'{leg / new:>8.3f}x')


def parse_args():
    p = argparse.ArgumentParser(description='backward_w のA/B計測')
    p.add_argument('--child', action='store_true', help='内部用(子プロセス)')
    p.add_argument('--repeats', type=int, default=DEFAULT_REPEATS,
                   help='legacy/new を何往復させるか(既定3)')
    p.add_argument('--only', choices=('all', 'token_mix', 'sweep', 'breakeven'),
                   default='all')
    p.add_argument('--out', type=str, default=None)
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
    for rep in range(args.repeats):
        for variant in VARIANTS:
            print(f'\n--- variant={variant} (往復 {rep + 1}/{args.repeats}) ---',
                  flush=True)
            rows += spawn_child(variant, args.only)

    table = aggregate(rows)
    if args.only == 'breakeven':
        print_report(table, 'breakeven', '損益分岐点の細分スイープ(backward_w 単体 中央値)')
    else:
        print_report(table, 'token_mix', 'token_mix 代表形状(backward_w 単体 中央値)')
        if args.only in ('all', 'sweep'):
            print_report(table, 'sweep', '損益分岐スイープ(backward_w 単体 中央値)')

    out = args.out or os.path.join(
        'results', 'kernel2', f'backward_w_{datetime.now(JST):%Y%m%d_%H%M%S}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'rows': rows, 'elapsed_s': time.perf_counter() - t0}, f, indent=2)
    print(f'\n結果JSON: {out}')
    print(f'所要 {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
