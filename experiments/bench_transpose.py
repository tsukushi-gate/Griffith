"""
境界転置(`x.t().contiguous()`)のマイクロベンチ。

用途を転置単体に限定した使い捨て計測スクリプト(本体ベンチ bench_lut_kernel.py とは別)。
- 素の `.t().contiguous()` / `torch.empty().copy_()` / `out.t().copy_(x)` / Triton の比較
- LutStack 連鎖での copy_ 呼び出し回数(1step=4回のはず)の確認

実行例:
    uv run modal run scripts/modal_app.py --script bench_transpose.py
"""

import sys
import time

import torch

sys.path.insert(0, '/root')

from griffith_logic.cuda.transpose import (  # noqa: E402
    fast_transpose,
    transpose_backend,
)

SHAPES = [
    (128, 1024),
    (512, 1024),
    (1024, 1024),
    (1024, 4096),
    (1024, 16384),
    (1024, 65536),
    (512, 100000),
    (1024, 400000),
    (1024, 480000),
]
DTYPES = [torch.float32]


def _time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def _bench_one(m: int, n: int, dtype) -> None:
    x = torch.rand(m, n, device='cuda', dtype=dtype)
    nbytes = x.numel() * x.element_size()

    def v_baseline():
        return x.t().contiguous()

    def v_empty_copy():
        out = torch.empty((n, m), device='cuda', dtype=dtype)
        out.copy_(x.t())
        return out

    def v_out_t_copy():
        out = torch.empty((n, m), device='cuda', dtype=dtype)
        out.t().copy_(x)
        return out

    def v_triton():
        return fast_transpose(x)

    variants = (
        ('baseline .t().contiguous()', v_baseline),
        ('torch.empty().copy_(x.t())', v_empty_copy),
        ('out.t().copy_(x)', v_out_t_copy),
        ('fast_transpose(triton)', v_triton),
    )

    # 正しさ(厳密一致)
    ref = v_baseline()
    for name, fn in variants[1:]:
        assert torch.equal(fn(), ref), f'{name} mismatch at {(m, n)}'
    del ref

    print(f'--- shape=({m},{n}) dtype={dtype} size={nbytes / 1e9:.3f}GB')
    for name, fn in variants:
        ms = _time(fn)
        gbs = 2 * nbytes / (ms * 1e-3) / 1e9  # read+write
        print(f'    {name:32s} {ms:8.3f} ms  {gbs:7.1f} GB/s')


def bench_jit_cost():
    """Triton初回JITコンパイルのコスト(訓練の初回1回のみ発生)を分離計測する。"""
    for dtype in (torch.float32, torch.float16, torch.float64):
        x = torch.rand(129, 257, device='cuda').to(dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fast_transpose(x)
        torch.cuda.synchronize()
        first = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        fast_transpose(x)
        torch.cuda.synchronize()
        second = (time.perf_counter() - t0) * 1e3
        print(f'  JIT {str(dtype):16s} 初回={first:8.2f} ms  2回目={second:8.3f} ms')


def bench_variants():
    print(f'device={torch.cuda.get_device_name(0)} backend={transpose_backend()}')
    bench_jit_cost()
    for dtype in DTYPES:
        for (m, n) in SHAPES:
            _bench_one(m, n, dtype)
            torch.cuda.empty_cache()


def count_copies():
    """LutStack連鎖で copy_ 呼び出しが 1step あたり 4 回のままかを確認する。"""
    from torch.profiler import ProfilerActivity, profile

    from griffith_logic.models import LutStack

    in_dim, sizes, batch = 4096, [400000, 400000, 480000], 1024
    stack = LutStack(in_dim, sizes, n_inputs=2, parameterization='softmax',
                     connections='random').to('cuda')
    stack.train()
    x = torch.rand(batch, in_dim, device='cuda')

    def step():
        stack.zero_grad(set_to_none=True)
        xin = x.detach().requires_grad_(True)
        stack(xin).sum().backward()

    step()
    torch.cuda.synchronize()

    steps = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()

    for evt in prof.key_averages():
        if evt.key in ('aten::copy_', 'aten::sum') or 'transpose' in evt.key:
            cuda_ms = getattr(evt, 'self_device_time_total', 0) / 1e3
            print(f'    {evt.key[:60]:60s} count={evt.count:4d} '
                  f'self_cuda={cuda_ms:9.3f}ms')
    print(f'    steps={steps} (copy_ は steps*4={steps * 4} 回が期待値)')


def bench_small_layer_end_to_end():
    """小テンソル(in=out=1024)の単層fwd+bwdで、転置経路のA/Bを取る。

    閾値を巨大化して強制的にtorch経路にした場合と既定(閾値あり)を同一プロセスで比較し、
    小スケールの退行が無いことを確認する。
    """
    from griffith_logic.cuda import transpose as tp
    from griffith_logic.layers.lut import LutLayer

    def _ab(param: str, n_inputs: int, batch: int) -> None:
        layer = LutLayer(1024, 1024, n_inputs=n_inputs, device='cuda',
                         parameterization=param).to('cuda')
        x = torch.rand(batch, 1024, device='cuda')

        def step():
            layer.zero_grad(set_to_none=True)
            xin = x.detach().requires_grad_(True)
            layer(xin).sum().backward()

        orig = tp._MIN_NUMEL_FOR_TRITON
        tp._MIN_NUMEL_FOR_TRITON = 1 << 60  # 強制torch経路
        ms_torch = _time(step, iters=50, warmup=10)
        tp._MIN_NUMEL_FOR_TRITON = 0        # 強制triton経路
        ms_triton = _time(step, iters=50, warmup=10)
        tp._MIN_NUMEL_FOR_TRITON = orig     # 既定(閾値あり)
        ms_default = _time(step, iters=50, warmup=10)
        print(f'  {param:12s} n={n_inputs} b={batch:4d} '
              f'torch強制={ms_torch:.4f}ms triton強制={ms_triton:.4f}ms '
              f'既定={ms_default:.4f}ms')

    for param in ('softmax', 'independent'):
        for n_inputs in (2, 4, 6):
            for batch in (128, 512):
                _ab(param, n_inputs, batch)


if __name__ == '__main__':
    bench_variants()
    print('=== 小テンソル単層 fwd+bwd の転置経路A/B ===')
    bench_small_layer_end_to_end()
    print('=== LutStack chain copy_ count ===')
    count_copies()
