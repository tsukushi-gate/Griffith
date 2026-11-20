"""LutLayer訓練経路の計測ハーネス。

LutLayerの訓練forward/backwardと、モデル級(LogicMixer / RDDLGNClassifier)の
1 train stepにかかる時間を実測し、torch.profilerでの内訳とあわせてJSON/wandbに残す。
**最適化は行わない**(このスクリプトは計測専用)。

使用例:
    uv run python experiments/bench_lut_kernel.py                # フル計測
    uv run python experiments/bench_lut_kernel.py --quick        # 数十秒の短縮版
    uv run python experiments/bench_lut_kernel.py --no-wandb --skip-models
    uv run python experiments/bench_lut_kernel.py --scale large    # 実スケール追加
    uv run modal run scripts/modal_app.py --script bench_lut_kernel.py --args "--scale large"

`--scale large`(= `--large`)は(40.8M params / 1,528,952 gates)の支配的コストである
出力ヘッドを2通りで測る:
  - single_layer: 1段相当(in_dim=out_dim=400_000)の単一LutLayer
  - stack: 出力ヘッド相当の連鎖(LutStack, layer_sizes=[400k,400k,480k])
単層は「LutLayer以外との境界」で必ず転置コピーが出るためコピーコストを過大評価しうる。
実モデルは連鎖構造なので両方を同条件で測り、aten::copy_ の占有率を直接対比する。
OOM時はサイズを半分ずつ下げて載る最大サイズを実測し、使った値・縮小回数・ピークメモリ・
profilerの self CPU/GPU 合計と copy_/LUTカーネル別内訳をJSONへ記録する。

出力:
    results/kernel1/bench_<device>_<YYYYmmdd_HHMMSS>.json  (+ wandb Artifact)
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import torch
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

# experiments/ 直下のユーティリティをimportできるようにする(Modal実行でも同様)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from griffith_logic.core import functions as lut_functions  # noqa: E402
from griffith_logic.cuda import cuda_available  # noqa: E402
from griffith_logic.layers.lut import LutLayer  # noqa: E402
from griffith_logic.models.mlp_mixer import LogicMixer, LogicMixerConfig  # noqa: E402
from griffith_logic.models.rddlgn import (  # noqa: E402
    LutStack,
    RDDLGNClassifier,
    RDDLGNConfig,
)

JST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 計測パラメータ(すべて定数として明示。数値の推測をしないため)
# ---------------------------------------------------------------------------

# マイクロベンチ: 単一LutLayerの入出力幅(固定)
MICRO_IN_DIM = 1024
MICRO_OUT_DIM = 1024

MICRO_N_INPUTS = (2, 4, 6)
MICRO_PARAMETERIZATIONS = ('softmax', 'independent')
MICRO_BATCHES = (128, 512)
MICRO_WARMUP = 10
MICRO_ITERS = 50

# 実スケールのマイクロベンチ(--scale large / --large でのみ実行)。
# 出典: wandb run 524l1cnd の config と
# griffith_logic/models/logic_seq_mixer.py(mixer 実験ブランチ時点の版)。
#(40.8M params / 1,528,952 gates)のコストは出力ヘッド
# m_layer_sizes=[400000, 400000, 480000] が支配的(全ゲートの約84%)なので、
# その1段に相当する単一LutLayerを in_dim=out_dim=400_000 で測る。
# batch=64 は batch_size、batch=1024 は batch_size×seq_len(64×16)の
# 実効トークン数。
LARGE_IN_DIM = 400_000
LARGE_OUT_DIM = 400_000
LARGE_N_INPUTS = (2,)
LARGE_PARAMETERIZATIONS = ('softmax', 'independent')
LARGE_BATCHES = (64, 1024)
LARGE_WARMUP = 3
LARGE_ITERS = 10
LARGE_PROFILE_STEPS = 3
# OOM時は out_dim を半分ずつ下げて載る最大サイズを実測で探す(下限まで)
LARGE_OOM_SHRINK_FACTOR = 2
LARGE_MIN_OUT_DIM = 25_000

# 連鎖(LutStack)版: 出力ヘッド m_layer_sizes=[400000,400000,480000] を模す。
# 単層マイクロベンチは「LutLayer以外との境界」で転置コピーが必ず発生するため
# コピーコストを過大評価しうる。実モデルは連鎖構造なので、同条件で連鎖版も測って対比する。
# 実モデルと同じ経路を測るため griffith_logic/models/rddlgn.py の LutStack を使う。
LARGE_STACK_SIZES = (400_000, 400_000, 480_000)

# profilerから構造化して抜き出す演算(名前の部分一致)
PROFILE_OP_PATTERNS = {
    'copy_': 'aten::copy_',
    'lut_forward_kernel': 'lut_forward_kernel',
    'lut_backward_w_kernel': 'lut_backward_w_kernel',
    'lut_backward_x_kernel': 'lut_backward_x_kernel',
}

# --quick 時の縮小格子
QUICK_N_INPUTS = (2, 4)
QUICK_PARAMETERIZATIONS = ('softmax',)
QUICK_BATCHES = (128,)
QUICK_MICRO_WARMUP = 3
QUICK_MICRO_ITERS = 5

# モデル級ベンチ
MODEL_BATCHES = (128, 512)
MODEL_WARMUP = 5
MODEL_ITERS = 20
QUICK_MODEL_BATCHES = (128,)
QUICK_MODEL_WARMUP = 2
QUICK_MODEL_ITERS = 3

# profiler
PROFILE_STEPS = 5
QUICK_PROFILE_STEPS = 3
PROFILE_TOP_K = 15

LR = 0.01

# LogicMixer tiny 構成。
# 形状(image_size=28 / patch_size=7 / in_channels=1 / num_classes=10)は
# experiments/train_mlp_mixer.py のMNIST既定 + LogicMixerConfig の既定値(mlp_mixer.py:75-84)。
# 容量パラメータは既定(encode_bits=8, d_channels=64, n_blocks=4, depth=2)を
# 計測用に縮小した tiny 値を用いる。
MIXER_TINY = dict(
    image_size=28,
    patch_size=7,
    in_channels=1,
    encode_bits=4,
    d_channels=32,
    n_blocks=2,
    depth=1,
    n_inputs=2,
    parameterization='softmax',
    num_classes=10,
)

# RDDLGNClassifier tiny 構成。
# バックボーンは RDDLGNConfig.tiny()(griffith_logic/models/rddlgn.py:83-98)をそのまま使用。
# n_classes=2 は SST-2 二値分類、head_sizes は RDDLGNClassifier の既定 [1024, 512]
# (rddlgn.py:518-519)を明示指定したもの。
RDDLGN_N_CLASSES = 2
RDDLGN_HEAD_SIZES = [1024, 512]


# ---------------------------------------------------------------------------
# デバイス / 同期
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """cuda → mps → cpu の順で利用可能なデバイスを返す。"""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def sync(device: str) -> None:
    """デバイスの非同期実行を待つ(計測の正確性に必須)。"""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def reset_peak_memory(device: str) -> None:
    """ピークメモリ統計をリセットする(cudaのみ統計を持つ)。"""
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device: str) -> int | None:
    """ピークメモリ(bytes)。取得できないデバイスではNone。"""
    if device == 'cuda':
        return int(torch.cuda.max_memory_allocated())
    if device == 'mps' and hasattr(torch.mps, 'driver_allocated_memory'):
        return int(torch.mps.driver_allocated_memory())
    return None


def free_memory(device: str) -> None:
    """割り当て済みキャッシュを解放する(OOM再試行の前に呼ぶ)。"""
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    """CUDA/MPS/CPU いずれのメモリ不足かどうかを文字列で判定する。"""
    msg = str(exc).lower()
    return any(k in msg for k in (
        'out of memory', 'invalid buffer size', 'cannot allocate',
        'insufficient memory', 'mps backend out of memory',
    ))


def env_report(device: str) -> dict:
    """torch / triton / CUDAカーネルの有無を集めて標準出力に出す。"""
    try:
        import triton
        triton_version = getattr(triton, '__version__', 'unknown')
    except Exception:
        triton_version = None

    info = {
        'device': device,
        'torch_version': torch.__version__,
        'triton_version': triton_version,
        'cuda_is_available': torch.cuda.is_available(),
        'mps_is_available': bool(
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        ),
        'griffith_cuda_kernel_available': cuda_available(),
    }
    if device == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)

    print(f'device={device}')
    print(f'  torch={info["torch_version"]}  triton={triton_version or "未インストール"}')
    print(f'  torch.cuda.is_available()={info["cuda_is_available"]}  '
          f'mps={info["mps_is_available"]}')
    print(f'  griffith cuda_available()={info["griffith_cuda_kernel_available"]}')
    if 'gpu_name' in info:
        print(f'  GPU: {info["gpu_name"]}')
    return info


# ---------------------------------------------------------------------------
# 実際に通ったautograd Functionの実測(モンキーパッチ。ライブラリ本体は不変)
# ---------------------------------------------------------------------------

ROUTE_CLASS_NAMES = (
    'LutFunction',
    'LutFunctionHardST',
    'LutFunctionWARPHardST',
    'LutFunctionIndependent',
    'LutFunctionRaw',
    'LutFunctionIWP',
    'LutFunctionWARP',
    'CudaLutFunction',
    'CudaLutFunctionIndependent',
)


@contextmanager
def route_counter():
    """autograd Functionのforward呼び出し回数を数える(with脱出時に必ず復元)。"""
    counts: dict[str, int] = {}
    originals: dict[str, object] = {}

    def make_wrapper(name, orig):
        def wrapper(ctx, *args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return orig(ctx, *args, **kwargs)
        return staticmethod(wrapper)

    for name in ROUTE_CLASS_NAMES:
        cls = getattr(lut_functions, name, None)
        if cls is None:
            continue
        orig = cls.forward
        originals[name] = orig
        cls.forward = make_wrapper(name, orig)
    try:
        yield counts
    finally:
        for name, orig in originals.items():
            getattr(lut_functions, name).forward = staticmethod(orig)


def route_label(counts: dict) -> str:
    """呼び出し回数から代表的な経路名を作る(未検出は 'unknown')。"""
    if not counts:
        return 'unknown'
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# 計測プリミティブ
# ---------------------------------------------------------------------------

def timed_median(step_fn, device: str, warmup: int, iters: int) -> dict:
    """step_fn を warmup 回空回し後 iters 回計測し、中央値等(秒)を返す。"""
    for _ in range(warmup):
        step_fn()
    sync(device)

    samples = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        step_fn()
        sync(device)
        samples.append(time.perf_counter() - t0)

    return {
        'median_s': statistics.median(samples),
        'mean_s': statistics.fmean(samples),
        'min_s': min(samples),
        'max_s': max(samples),
        'iters': iters,
        'warmup': warmup,
    }


def fmt_eta(elapsed: float, done: int, total: int) -> str:
    """[経過<残り, 終了予定 HH:MM:SS] 形式の進捗文字列を返す。"""
    remaining = (elapsed / done) * (total - done) if done else 0.0
    eta = datetime.now(JST) + timedelta(seconds=remaining)
    return (f'[{elapsed:.1f}s<{remaining:.1f}s, '
            f'終了予定 {eta.strftime("%H:%M:%S")}]')


# ---------------------------------------------------------------------------
# マイクロベンチ(単一LutLayer)
# ---------------------------------------------------------------------------

def bench_micro(device: str, warmup: int, iters: int,
                n_inputs_grid, param_grid, batch_grid) -> list[dict]:
    results = []
    combos = [(n, p, b) for n in n_inputs_grid for p in param_grid for b in batch_grid]
    t_start = time.perf_counter()

    for idx, (n_inputs, param, batch) in enumerate(combos, 1):
        layer = LutLayer(
            in_dim=MICRO_IN_DIM, out_dim=MICRO_OUT_DIM, n_inputs=n_inputs,
            parameterization=param, connections='random',
        ).to(device)
        layer.train()
        x = torch.rand(batch, MICRO_IN_DIM, device=device)

        def fwd(layer=layer, x=x):
            with torch.no_grad():
                layer(x)

        def fwd_bwd(layer=layer, x=x):
            layer.zero_grad(set_to_none=True)
            xin = x.detach().requires_grad_(True)
            out = layer(xin)
            out.sum().backward()

        # 経路判定(1回のfwd+bwdで実測)
        with route_counter() as counts:
            fwd_bwd()
            sync(device)
        route = route_label(counts)

        fwd_stats = timed_median(fwd, device, warmup, iters)
        fb_stats = timed_median(fwd_bwd, device, warmup, iters)

        results.append({
            'n_inputs': n_inputs,
            'parameterization': param,
            'batch': batch,
            'in_dim': MICRO_IN_DIM,
            'out_dim': MICRO_OUT_DIM,
            'route': route,
            'route_counts': dict(counts),
            'forward': fwd_stats,
            'forward_backward': fb_stats,
        })

        elapsed = time.perf_counter() - t_start
        print(f'  micro {idx}/{len(combos)} n={n_inputs} {param} b={batch} '
              f'route={route} fwd={fwd_stats["median_s"]:.6f}s/it '
              f'fwd+bwd={fb_stats["median_s"]:.6f}s/it '
              f'{fmt_eta(elapsed, idx, len(combos))}')

    return results


# ---------------------------------------------------------------------------
# 実スケールのマイクロベンチ(大サイズ)
# ---------------------------------------------------------------------------

def _try_large_case(device: str, n_inputs: int, param: str, batch: int,
                    in_dim: int, out_dim: int, warmup: int, iters: int,
                    profile_steps: int) -> dict:
    """指定サイズで1ケースを実測する。OOM時は例外をそのまま送出する。"""
    layer = LutLayer(
        in_dim=in_dim, out_dim=out_dim, n_inputs=n_inputs,
        parameterization=param, connections='random',
    ).to(device)
    layer.train()
    x = torch.rand(batch, in_dim, device=device)

    def fwd():
        with torch.no_grad():
            layer(x)

    def fwd_bwd():
        layer.zero_grad(set_to_none=True)
        xin = x.detach().requires_grad_(True)
        out = layer(xin)
        out.sum().backward()

    with route_counter() as counts:
        fwd_bwd()
        sync(device)
    route = route_label(counts)

    reset_peak_memory(device)
    fwd_stats = timed_median(fwd, device, warmup, iters)
    fb_stats = timed_median(fwd_bwd, device, warmup, iters)
    peak = peak_memory_bytes(device)

    prof_res = profile_step(fwd_bwd, device, profile_steps)
    print(f'--- profiler: large n={n_inputs} {param} b={batch} '
          f'out_dim={out_dim} ({prof_res["steps"]} steps, '
          f'sort_by={prof_res["sort_by"]}) ---')
    print(prof_res['table'])
    print_profile_consistency(prof_res)

    return {
        'case': 'tsmix1_scale',
        'variant': 'single_layer',
        'module': 'griffith_logic.layers.lut.LutLayer',
        'n_inputs': n_inputs,
        'parameterization': param,
        'batch': batch,
        'in_dim': in_dim,
        'out_dim': out_dim,
        'route': route,
        'route_counts': dict(counts),
        'forward': fwd_stats,
        'forward_backward': fb_stats,
        'peak_memory_bytes': peak,
        'profile': {k: v for k, v in prof_res.items() if k != 'table'},
    }


def _try_large_stack_case(device: str, n_inputs: int, param: str, batch: int,
                          in_dim: int, layer_sizes: list[int], warmup: int,
                          iters: int, profile_steps: int) -> dict:
    """LutLayerを連鎖させた版(出力ヘッド相当)を1ケース実測する。

    実モデルと同じ経路を測るため griffith_logic/models/rddlgn.py の LutStack を使う。
    """
    stack = LutStack(
        in_dim, list(layer_sizes), n_inputs=n_inputs,
        parameterization=param, connections='random',
    ).to(device)
    stack.train()
    x = torch.rand(batch, in_dim, device=device)

    def fwd():
        with torch.no_grad():
            stack(x)

    def fwd_bwd():
        stack.zero_grad(set_to_none=True)
        xin = x.detach().requires_grad_(True)
        out = stack(xin)
        out.sum().backward()

    with route_counter() as counts:
        fwd_bwd()
        sync(device)
    route = route_label(counts)

    reset_peak_memory(device)
    fwd_stats = timed_median(fwd, device, warmup, iters)
    fb_stats = timed_median(fwd_bwd, device, warmup, iters)
    peak = peak_memory_bytes(device)

    prof_res = profile_step(fwd_bwd, device, profile_steps)
    print(f'--- profiler: large-stack n={n_inputs} {param} b={batch} '
          f'sizes={list(layer_sizes)} ({prof_res["steps"]} steps, '
          f'sort_by={prof_res["sort_by"]}) ---')
    print(prof_res['table'])
    print_profile_consistency(prof_res)

    return {
        'case': 'tsmix1_scale',
        'variant': 'stack',
        'module': 'griffith_logic.models.rddlgn.LutStack',
        'n_inputs': n_inputs,
        'parameterization': param,
        'batch': batch,
        'in_dim': in_dim,
        'layer_sizes': list(layer_sizes),
        'n_layers': len(layer_sizes),
        'out_dim': layer_sizes[-1],
        'route': route,
        'route_counts': dict(counts),
        'forward': fwd_stats,
        'forward_backward': fb_stats,
        'peak_memory_bytes': peak,
        'profile': {k: v for k, v in prof_res.items() if k != 'table'},
    }


def _run_with_oom_shrink(device: str, variant: str, n_inputs: int, param: str,
                         batch: int) -> dict:
    """1ケースをOOM縮小付きで実測する。

    single_layer は out_dim を、stack は全層サイズを LARGE_OOM_SHRINK_FACTOR で
    割りながら載る最大サイズを探し、実際に使った値と縮小回数を必ず記録する。
    """
    shrinks = 0
    record: dict | None = None
    last_error = ''
    out_dim = LARGE_OUT_DIM
    sizes = list(LARGE_STACK_SIZES)

    def current_min() -> int:
        return out_dim if variant == 'single_layer' else min(sizes)

    def desc() -> str:
        return (f'out_dim={out_dim}' if variant == 'single_layer'
                else f'sizes={sizes}')

    while current_min() >= LARGE_MIN_OUT_DIM:
        try:
            if variant == 'single_layer':
                record = _try_large_case(
                    device, n_inputs, param, batch, LARGE_IN_DIM, out_dim,
                    LARGE_WARMUP, LARGE_ITERS, LARGE_PROFILE_STEPS,
                )
            else:
                record = _try_large_stack_case(
                    device, n_inputs, param, batch, LARGE_IN_DIM, sizes,
                    LARGE_WARMUP, LARGE_ITERS, LARGE_PROFILE_STEPS,
                )
            break
        except Exception as exc:  # noqa: BLE001 — OOM以外は再送出する
            if not is_oom_error(exc):
                raise
            last_error = str(exc).splitlines()[0]
            free_memory(device)
            print(f'  OOM: {variant} n={n_inputs} {param} b={batch} '
                  f'{desc()} → 縮小して再試行 ({last_error})')
            out_dim //= LARGE_OOM_SHRINK_FACTOR
            sizes = [s // LARGE_OOM_SHRINK_FACTOR for s in sizes]
            shrinks += 1

    if record is None:
        record = {
            'case': 'tsmix1_scale',
            'variant': variant,
            'n_inputs': n_inputs,
            'parameterization': param,
            'batch': batch,
            'in_dim': LARGE_IN_DIM,
            'out_dim': None,
            'layer_sizes': None,
            'status': 'oom',
        }
    else:
        record['status'] = 'ok' if shrinks == 0 else 'ok_shrunk'

    record['out_dim_requested'] = LARGE_OUT_DIM
    record['layer_sizes_requested'] = list(LARGE_STACK_SIZES)
    record['oom_shrinks'] = shrinks
    record['oom_last_error'] = last_error
    free_memory(device)
    return record


def bench_micro_large(device: str) -> list[dict]:
    """実スケールのマイクロベンチ(単層 + 連鎖LutStackの両方)。

    単層(in_dim=out_dim=400k)と連鎖(LutStack, layer_sizes=[400k,400k,480k])を
    同条件で測り、転置コピー(aten::copy_)の占有率を直接対比できるようにする。
    """
    results = []
    combos = [(v, n, p, b)
              for v in ('single_layer', 'stack')
              for n in LARGE_N_INPUTS
              for p in LARGE_PARAMETERIZATIONS
              for b in LARGE_BATCHES]
    t_start = time.perf_counter()

    for idx, (variant, n_inputs, param, batch) in enumerate(combos, 1):
        record = _run_with_oom_shrink(device, variant, n_inputs, param, batch)
        results.append(record)
        elapsed = time.perf_counter() - t_start
        head = (f'  large {idx}/{len(combos)} [{variant}] n={n_inputs} {param} '
                f'b={batch}')

        if record['status'] == 'oom':
            print(f'{head}: 下限{LARGE_MIN_OUT_DIM}まで縮小してもOOM。'
                  f'計測不能として記録 {fmt_eta(elapsed, idx, len(combos))}')
            continue

        peak = record['peak_memory_bytes']
        peak_str = f'{peak / 1024 ** 3:.2f}GiB' if peak else 'n/a'
        size_str = (f'out_dim={record["out_dim"]}(要求{LARGE_OUT_DIM})'
                    if variant == 'single_layer'
                    else f'sizes={record["layer_sizes"]}'
                         f'(要求{list(LARGE_STACK_SIZES)})')
        copy_share = record['profile']['op_breakdown']['copy_']['device_share']
        copy_str = f'{copy_share * 100:.1f}%' if copy_share is not None else 'n/a'
        print(f'{head} {size_str} 縮小{record["oom_shrinks"]}回 '
              f'route={record["route"]} '
              f'fwd={record["forward"]["median_s"]:.6f}s/it '
              f'fwd+bwd={record["forward_backward"]["median_s"]:.6f}s/it '
              f'peak={peak_str} copy_(GPU)={copy_str} '
              f'{fmt_eta(elapsed, idx, len(combos))}')

    return results


# ---------------------------------------------------------------------------
# モデル級ベンチ
# ---------------------------------------------------------------------------

def build_mixer(device: str):
    cfg = LogicMixerConfig(**MIXER_TINY)
    model = LogicMixer(cfg).to(device)
    return model, cfg


def make_mixer_step(model, cfg, batch: int, device: str):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.rand(batch, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)
    y = torch.randint(0, cfg.num_classes, (batch,), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    return step


def build_rddlgn(device: str):
    cfg = RDDLGNConfig.tiny()
    model = RDDLGNClassifier(
        cfg, n_classes=RDDLGN_N_CLASSES, head_sizes=list(RDDLGN_HEAD_SIZES),
        device=device,
    ).to(device)
    return model, cfg


def make_rddlgn_step(model, cfg, batch: int, device: str):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tokens = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len), device=device)
    y = torch.randint(0, RDDLGN_N_CLASSES, (batch,), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits, emb_reg = model(tokens)
        loss = F.cross_entropy(logits, y) + emb_reg
        loss.backward()
        optimizer.step()

    return step


MODEL_SPECS = (
    ('LogicMixer', build_mixer, make_mixer_step),
    ('RDDLGNClassifier', build_rddlgn, make_rddlgn_step),
)


def _is_device_level_event(ev) -> bool:
    """そのエントリが「カーネルレベル」(デバイス側)かどうか。

    torch/autograd/profiler_util.py の _build_table が "Self CUDA time total" を
    合計するときと同じ規則: device_type が CUDA/PrivateUse1/MTIA/XPU で
    is_user_annotation でないもの、および legacy profiler の CPU エントリ。
    ATen op レベル(aten::copy_ 等)は device_type=CPU なので False になる。
    """
    dev_type = getattr(ev, 'device_type', None)
    if dev_type is None:
        return False
    if dev_type == DeviceType.CPU:
        return bool(getattr(ev, 'is_legacy', False))
    device_levels = [DeviceType.CUDA]
    for name in ('PrivateUse1', 'MTIA', 'XPU'):
        member = getattr(DeviceType, name, None)
        if member is not None:
            device_levels.append(member)
    return dev_type in device_levels and not getattr(ev, 'is_user_annotation', False)


def _parse_table_totals(table: str) -> dict:
    """profilerが印字した "Self CPU/CUDA time total" をusに直して返す(検証用)。"""
    units = {'us': 1.0, 'ms': 1e3, 's': 1e6}
    out: dict[str, float | None] = {
        'table_self_cpu_total_us': None,
        'table_self_device_total_us': None,
    }
    pattern = re.compile(
        r'Self\s+(?P<what>[A-Za-z0-9_]+)\s+time total:\s*'
        r'(?P<val>[0-9.]+)(?P<unit>us|ms|s)\b'
    )
    for m in pattern.finditer(table or ''):
        value = float(m.group('val')) * units[m.group('unit')]
        if m.group('what').upper() == 'CPU':
            out['table_self_cpu_total_us'] = value
        else:
            out['table_self_device_total_us'] = value
    return out


def profile_step(step_fn, device: str, steps: int) -> dict:
    """torch.profilerで数stepを取り、self time上位のテーブルと配列を返す。"""
    activities = [ProfilerActivity.CPU]
    use_cuda = device == 'cuda'
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    step_fn()  # プロファイル対象外のwarmup
    sync(device)

    with profile(activities=activities) as prof:
        for _ in range(steps):
            step_fn()
        sync(device)

    sort_key = 'self_cuda_time_total' if use_cuda else 'self_cpu_time_total'
    try:
        table = prof.key_averages().table(sort_by=sort_key, row_limit=PROFILE_TOP_K)
    except Exception as exc:  # sort_byが未対応の環境向けフォールバック
        print(f'  警告: profiler table(sort_by={sort_key}) 失敗: {exc}')
        table = prof.key_averages().table(row_limit=PROFILE_TOP_K)

    rows = []
    for ev in prof.key_averages():
        rows.append({
            'name': ev.key,
            'count': int(ev.count),
            'device_type': str(ev.device_type),
            'is_kernel_level': _is_device_level_event(ev),
            'self_cpu_time_total_us': float(ev.self_cpu_time_total),
            'cpu_time_total_us': float(ev.cpu_time_total),
            'self_device_time_total_us': float(
                getattr(ev, 'self_device_time_total', 0.0) or 0.0
            ),
        })
    key = 'self_device_time_total_us' if use_cuda else 'self_cpu_time_total_us'
    rows.sort(key=lambda r: r[key], reverse=True)

    # GPU律速/CPU律速の判定材料。
    # 分母は profiler の "Self CUDA time total" と一致させる = カーネルレベルの
    # エントリのみを合計する(torch/autograd/profiler_util.py の _build_table と同じ規則)。
    # 全エントリを素朴に合計すると ATen op レベル(aten::copy_ 等、子カーネルの時間が
    # 帰属される)とカーネルレベル(void ...elementwise_kernel...)を二重計上して
    # ちょうど2倍になるため、それは naive_sum として別枠に記録するだけに留める。
    self_cpu_total = sum(r['self_cpu_time_total_us'] for r in rows)
    device_kernel_total = sum(
        r['self_device_time_total_us'] for r in rows if r['is_kernel_level']
    )
    device_op_total = sum(
        r['self_device_time_total_us'] for r in rows if not r['is_kernel_level']
    )
    device_naive_sum = device_kernel_total + device_op_total

    if device_kernel_total > 0:
        device_denominator = device_kernel_total
        denominator_source = 'kernel_level'
    elif device_op_total > 0:
        device_denominator = device_op_total
        denominator_source = 'op_level_fallback'
    else:
        device_denominator = 0.0
        denominator_source = 'none'

    # 注目演算(転置コピー / LUTカーネル)を構造化して抜き出す。
    # 分子も二重計上しないよう op レベル / カーネルレベルを分けて集計し、
    # 両側に当たった場合は同じ実時間の重複なので合計せず max を採る(フラグで可視化)。
    breakdown = {}
    for label, pattern in PROFILE_OP_PATTERNS.items():
        hits = [r for r in rows if pattern in r['name']]
        kernel_hits = [r for r in hits if r['is_kernel_level']]
        op_hits = [r for r in hits if not r['is_kernel_level']]
        dev_kernel = sum(r['self_device_time_total_us'] for r in kernel_hits)
        dev_op = sum(r['self_device_time_total_us'] for r in op_hits)
        both = bool(kernel_hits) and bool(op_hits)
        dev_us = max(dev_kernel, dev_op) if both else dev_kernel + dev_op
        cpu_us = sum(r['self_cpu_time_total_us'] for r in hits)
        count = (max(sum(r['count'] for r in kernel_hits),
                     sum(r['count'] for r in op_hits))
                 if both else sum(r['count'] for r in hits))
        breakdown[label] = {
            'count': count,
            'self_cpu_time_total_us': cpu_us,
            'self_device_time_total_us': dev_us,
            'self_device_time_kernel_level_us': dev_kernel,
            'self_device_time_op_level_us': dev_op,
            'matched_both_levels': both,
            'device_share': (
                (dev_us / device_denominator) if device_denominator else None
            ),
            'cpu_share': (cpu_us / self_cpu_total) if self_cpu_total else None,
            'matched_names': [
                {'name': r['name'], 'is_kernel_level': r['is_kernel_level'],
                 'count': r['count'],
                 'self_device_time_total_us': r['self_device_time_total_us']}
                for r in hits
            ],
        }

    result = {
        'steps': steps, 'sort_by': sort_key, 'table': table,
        'self_cpu_time_total_us': self_cpu_total,
        # 以降、device側は分母=カーネルレベル合計(profilerのSelf CUDA time totalと一致)
        'self_device_time_total_us': device_denominator,
        'self_device_time_denominator_us': device_denominator,
        'self_device_time_denominator_source': denominator_source,
        'self_device_time_kernel_level_us': device_kernel_total,
        'self_device_time_op_level_us': device_op_total,
        'self_device_time_naive_sum_us': device_naive_sum,
        'naive_sum_over_denominator': (
            (device_naive_sum / device_denominator) if device_denominator else None
        ),
        'op_breakdown': breakdown,
        'top_ops': rows[:PROFILE_TOP_K],
    }
    # profilerが自分で印字した合計を文字列から拾い、突き合わせ可能にする(検証用)
    result.update(_parse_table_totals(table))
    return result


def print_profile_consistency(prof_res: dict) -> None:
    """算出した分母がprofilerの印字合計と一致しているかを検算表示する。"""
    denom = prof_res['self_device_time_denominator_us']
    table_dev = prof_res.get('table_self_device_total_us')
    table_cpu = prof_res.get('table_self_cpu_total_us')
    naive = prof_res['self_device_time_naive_sum_us']
    ratio = prof_res['naive_sum_over_denominator']
    ratio_s = f'{ratio:.3f}' if ratio is not None else 'n/a'
    dev_match = ('n/a' if not (denom and table_dev)
                 else ('一致' if abs(denom - table_dev) <= 0.01 * table_dev
                       else f'不一致(差{denom - table_dev:.1f}us)'))
    cpu_match = ('n/a' if not table_cpu
                 else ('一致'
                       if abs(prof_res['self_cpu_time_total_us'] - table_cpu)
                       <= 0.01 * table_cpu else '不一致'))
    print(f'  検算: 分母({prof_res["self_device_time_denominator_source"]})='
          f'{denom / 1000:.3f}ms / profiler印字='
          f'{(table_dev / 1000) if table_dev else float("nan"):.3f}ms → {dev_match}'
          f' | naive合計={naive / 1000:.3f}ms (naive/分母={ratio_s})'
          f' | CPU合計={prof_res["self_cpu_time_total_us"] / 1000:.3f}ms vs 印字='
          f'{(table_cpu / 1000) if table_cpu else float("nan"):.3f}ms → {cpu_match}')


def bench_models(device: str, batches, warmup: int, iters: int,
                 profile_steps: int) -> list[dict]:
    results = []
    total = len(MODEL_SPECS) * len(batches)
    t_start = time.perf_counter()
    idx = 0

    for name, builder, step_maker in MODEL_SPECS:
        model, cfg = builder(device)
        model.train()
        n_params = sum(p.numel() for p in model.parameters())

        for batch in batches:
            idx += 1
            step = step_maker(model, cfg, batch, device)

            with route_counter() as counts:
                step()
                sync(device)
            route = route_label(counts)

            stats = timed_median(step, device, warmup, iters)
            prof_res = profile_step(step, device, profile_steps)

            results.append({
                'model': name,
                'batch': batch,
                'n_params': n_params,
                'route': route,
                'route_counts': dict(counts),
                'train_step': stats,
                'profile': {k: v for k, v in prof_res.items() if k != 'table'},
            })

            elapsed = time.perf_counter() - t_start
            print(f'  model {idx}/{total} {name} b={batch} route={route} '
                  f'step={stats["median_s"]:.6f}s/it '
                  f'{fmt_eta(elapsed, idx, total)}')
            print(f'--- profiler: {name} batch={batch} '
                  f'({prof_res["steps"]} steps, sort_by={prof_res["sort_by"]}) ---')
            print(prof_res['table'])
            print_profile_consistency(prof_res)

    return results


# ---------------------------------------------------------------------------
# 表示 / 保存
# ---------------------------------------------------------------------------

def print_micro_table(rows: list[dict]) -> None:
    print('\n=== マイクロベンチ: 単一LutLayer '
          f'(in_dim={MICRO_IN_DIM}, out_dim={MICRO_OUT_DIM}, 中央値) ===')
    header = (f'{"n_inputs":>8} {"param":>12} {"batch":>6} {"route":>26} '
              f'{"fwd s/it":>12} {"fwd+bwd s/it":>14}')
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["n_inputs"]:>8} {r["parameterization"]:>12} {r["batch"]:>6} '
              f'{r["route"]:>26} {r["forward"]["median_s"]:>12.6f} '
              f'{r["forward_backward"]["median_s"]:>14.6f}')


def print_large_table(rows: list[dict]) -> None:
    if not rows:
        return
    print('\n===実スケールマイクロベンチ '
          f'(in_dim={LARGE_IN_DIM}, 単層 vs 連鎖LutStack, 中央値) ===')
    header = (f'{"variant":>12} {"param":>12} {"batch":>6} {"size":>26} '
              f'{"status":>10} {"fwd s/it":>12} {"fwd+bwd s/it":>14} '
              f'{"peak GiB":>9} {"selfCPU ms":>11} {"selfGPU ms":>11} '
              f'{"copy_ GPU%":>11} {"copy_ calls":>12} {"LUT kern GPU%":>14}')
    print(header)
    print('-' * len(header))
    for r in rows:
        variant = r.get('variant', 'single_layer')
        if r.get('status') == 'oom':
            print(f'{variant:>12} {r["parameterization"]:>12} {r["batch"]:>6} '
                  f'{"-":>26} {"oom":>10} {"-":>12} {"-":>14} {"-":>9} '
                  f'{"-":>11} {"-":>11} {"-":>11} {"-":>12} {"-":>14}')
            continue
        peak = r['peak_memory_bytes']
        peak_s = f'{peak / 1024 ** 3:.2f}' if peak else '-'
        prof = r['profile']
        bd = prof['op_breakdown']
        size_s = (str(r['out_dim']) if variant == 'single_layer'
                  else str(r['layer_sizes']))
        copy_share = bd['copy_']['device_share']
        copy_s = f'{copy_share * 100:.1f}' if copy_share is not None else '-'
        lut_dev = sum(bd[k]['self_device_time_total_us'] for k in (
            'lut_forward_kernel', 'lut_backward_w_kernel', 'lut_backward_x_kernel'))
        dev_total = prof['self_device_time_total_us']
        lut_s = f'{lut_dev / dev_total * 100:.1f}' if dev_total else '-'
        print(f'{variant:>12} {r["parameterization"]:>12} {r["batch"]:>6} '
              f'{size_s:>26} {r["status"]:>10} '
              f'{r["forward"]["median_s"]:>12.6f} '
              f'{r["forward_backward"]["median_s"]:>14.6f} {peak_s:>9} '
              f'{prof["self_cpu_time_total_us"] / 1000:>11.3f} '
              f'{prof["self_device_time_total_us"] / 1000:>11.3f} '
              f'{copy_s:>11} {bd["copy_"]["count"]:>12} {lut_s:>14}')


def print_model_table(rows: list[dict]) -> None:
    if not rows:
        return
    print('\n=== モデル級ベンチ: 1 train step (fwd+loss+bwd+opt, 中央値) ===')
    header = (f'{"model":>18} {"batch":>6} {"params":>12} {"route":>26} '
              f'{"step s/it":>12}')
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["model"]:>18} {r["batch"]:>6} {r["n_params"]:>12,} '
              f'{r["route"]:>26} {r["train_step"]["median_s"]:>12.6f}')


def save_json(payload: dict, device: str, timestamp: str) -> str:
    out_dir = os.path.join('results', 'kernel1')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'bench_{device}_{timestamp}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def flat_metrics(payload: dict) -> dict:
    """wandbのsummary向けにフラットな指標辞書を作る。"""
    metrics = {}
    for r in payload['micro']:
        base = (f'micro/n{r["n_inputs"]}_{r["parameterization"]}_b{r["batch"]}')
        metrics[f'{base}/fwd_s'] = r['forward']['median_s']
        metrics[f'{base}/fwd_bwd_s'] = r['forward_backward']['median_s']
    for r in payload.get('micro_large', []):
        if r.get('status') == 'oom':
            continue
        variant = r.get('variant', 'single_layer')
        base = (f'large/{variant}_n{r["n_inputs"]}_'
                f'{r["parameterization"]}_b{r["batch"]}')
        metrics[f'{base}/fwd_s'] = r['forward']['median_s']
        metrics[f'{base}/fwd_bwd_s'] = r['forward_backward']['median_s']
        metrics[f'{base}/out_dim'] = r['out_dim']
        if r['peak_memory_bytes']:
            metrics[f'{base}/peak_memory_bytes'] = r['peak_memory_bytes']
        metrics[f'{base}/self_cpu_ms'] = r['profile']['self_cpu_time_total_us'] / 1000
        metrics[f'{base}/self_gpu_ms'] = (
            r['profile']['self_device_time_total_us'] / 1000
        )
        metrics[f'{base}/self_gpu_naive_sum_ms'] = (
            r['profile']['self_device_time_naive_sum_us'] / 1000
        )
        for label, stats in r['profile']['op_breakdown'].items():
            metrics[f'{base}/op/{label}_gpu_ms'] = (
                stats['self_device_time_total_us'] / 1000
            )
            metrics[f'{base}/op/{label}_calls'] = stats['count']
            if stats['device_share'] is not None:
                metrics[f'{base}/op/{label}_gpu_share'] = stats['device_share']
    for r in payload['models']:
        metrics[f'model/{r["model"]}_b{r["batch"]}/step_s'] = r['train_step']['median_s']
    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='LutLayer訓練経路のベンチマーク(計測専用)',
    )
    parser.add_argument('--quick', action='store_true',
                        help='格子と反復回数を縮小して数十秒で完走させる')
    parser.add_argument('--scale', type=str, default=None,
                        choices=('full', 'quick', 'large'),
                        help="格子スケール。'large'で実スケール"
                             '(in_dim=out_dim=400k)のケースを追加実行する')
    parser.add_argument('--large', action='store_true',
                        help="--scale large と同義(実スケールを追加)")
    parser.add_argument('--no-wandb', action='store_true', help='wandbを無効化')
    parser.add_argument('--skip-models', action='store_true',
                        help='モデル級ベンチ(+profiler)をスキップ')
    parser.add_argument('--tag', type=str, default='', help='結果ラベル')
    parser.add_argument('--device', type=str, default=None,
                        help='デバイス(既定は自動検出 cuda→mps→cpu)')
    parser.add_argument('--seed', type=int, default=0, help='乱数シード')
    try:
        from wandb_utils import add_experiment_args
        add_experiment_args(parser)
    except Exception:
        parser.add_argument('--notes', type=str, default=None)
        parser.add_argument('--hypothesis', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # --scale と --quick/--large の整合(--scale が明示されたらそれを優先)
    if args.scale == 'quick':
        args.quick = True
    elif args.scale == 'full':
        args.quick = False
    run_large = args.large or args.scale == 'large'

    started = datetime.now(JST)
    print(f'開始: {started.strftime("%Y-%m-%d %H:%M:%S")} JST')

    device = args.device or detect_device()
    env = env_report(device)

    if args.quick:
        n_grid, p_grid, b_grid = QUICK_N_INPUTS, QUICK_PARAMETERIZATIONS, QUICK_BATCHES
        micro_warmup, micro_iters = QUICK_MICRO_WARMUP, QUICK_MICRO_ITERS
        model_batches = QUICK_MODEL_BATCHES
        model_warmup, model_iters = QUICK_MODEL_WARMUP, QUICK_MODEL_ITERS
        profile_steps = QUICK_PROFILE_STEPS
    else:
        n_grid, p_grid, b_grid = (
            MICRO_N_INPUTS, MICRO_PARAMETERIZATIONS, MICRO_BATCHES,
        )
        micro_warmup, micro_iters = MICRO_WARMUP, MICRO_ITERS
        model_batches = MODEL_BATCHES
        model_warmup, model_iters = MODEL_WARMUP, MODEL_ITERS
        profile_steps = PROFILE_STEPS

    print(f'\nマイクロベンチ開始 (warmup={micro_warmup}, iters={micro_iters})')
    micro = bench_micro(device, micro_warmup, micro_iters, n_grid, p_grid, b_grid)

    micro_large: list[dict] = []
    if run_large:
        print(f'\nTSMIX-1実スケール マイクロベンチ開始 '
              f'(in_dim={LARGE_IN_DIM}, out_dim={LARGE_OUT_DIM}要求, '
              f'warmup={LARGE_WARMUP}, iters={LARGE_ITERS})')
        micro_large = bench_micro_large(device)

    models: list[dict] = []
    if args.skip_models:
        print('\nモデル級ベンチ: --skip-models によりスキップ')
    else:
        print(f'\nモデル級ベンチ開始 (warmup={model_warmup}, iters={model_iters})')
        models = bench_models(device, model_batches, model_warmup, model_iters,
                              profile_steps)

    timestamp = started.strftime('%Y%m%d_%H%M%S')
    payload = {
        'task': 'lut-kernel-bench',
        'tag': args.tag,
        'started_at_jst': started.strftime('%Y-%m-%d %H:%M:%S'),
        'finished_at_jst': datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S'),
        'quick': args.quick,
        'large': run_large,
        'env': env,
        'config': {
            'micro_in_dim': MICRO_IN_DIM,
            'micro_out_dim': MICRO_OUT_DIM,
            'micro_warmup': micro_warmup,
            'micro_iters': micro_iters,
            'model_warmup': model_warmup,
            'model_iters': model_iters,
            'profile_steps': profile_steps,
            'large_in_dim': LARGE_IN_DIM,
            'large_out_dim_requested': LARGE_OUT_DIM,
            'large_stack_sizes_requested': list(LARGE_STACK_SIZES),
            'large_stack_module': 'griffith_logic.models.rddlgn.LutStack',
            'large_batches': list(LARGE_BATCHES),
            'large_warmup': LARGE_WARMUP,
            'large_iters': LARGE_ITERS,
            'large_profile_steps': LARGE_PROFILE_STEPS,
            'mixer_tiny': MIXER_TINY,
            'rddlgn_tiny': 'RDDLGNConfig.tiny()',
            'rddlgn_n_classes': RDDLGN_N_CLASSES,
            'rddlgn_head_sizes': RDDLGN_HEAD_SIZES,
            'lr': LR,
            'seed': args.seed,
        },
        'micro': micro,
        'micro_large': micro_large,
        'models': models,
    }

    print_micro_table(micro)
    print_large_table(micro_large)
    print_model_table(models)

    json_path = save_json(payload, device, timestamp)
    print(f'\n結果JSON: {os.path.abspath(json_path)}')

    if not args.no_wandb:
        log_to_wandb(args, payload, device, json_path)

    finished = datetime.now(JST)
    print(f'終了: {finished.strftime("%Y-%m-%d %H:%M:%S")} JST '
          f'(所要 {(finished - started).total_seconds():.1f}s)')


def log_to_wandb(args, payload: dict, device: str, json_path: str) -> None:
    """wandbへ指標とJSON Artifactを記録する。失敗時は警告を出して続行。"""
    try:
        from wandb_utils import (
            finish,
            init_wandb,
            log_artifact,
            log_metrics,
            normalize_tags,
        )
    except Exception as exc:
        print(f'警告: wandb_utilsをimportできずwandb記録をスキップ: {exc}')
        return

    try:
        suffix = f'_{args.tag}' if args.tag else ''
        ok = init_wandb(
            run_name=f'lutbench_{device}{suffix}',
            config_dict={**payload['config'], 'tag': args.tag,
                         'quick': args.quick, 'large': payload['large'],
                         'device': device},
            device=device,
            tags=['bench', 'kernel-1', device] + normalize_tags(
                model='lut_layer', seed=args.seed, series='lut-kernel-bench',
            ),
            group='kernel-1',
            job_type='eval',
            notes=getattr(args, 'notes', None),
            hypothesis=getattr(args, 'hypothesis', None),
        )
        if not ok:
            print('警告: wandbが無効のため記録をスキップ(結果はJSONに保存済み)')
            return
        log_metrics(flat_metrics(payload))
        log_artifact(
            json_path, name=f'kernel1-bench-{device}{suffix}',
            type='benchmark',
            metadata={'task': 'lut-kernel-bench', 'device': device, 'quick': args.quick},
        )
        finish()
    except Exception as exc:
        print(f'警告: wandb記録に失敗(結果はJSONに保存済み): {exc}')


if __name__ == '__main__':
    main()
