"""L2: backward_w の行方向並列化(gridDim.x = row_blocks)の正しさ検証(@cuda)。

`lut_backward_w` の行(バッチ)方向の並列度を 32スレッド固定から
R×32スレッドへ一般化した。中間バッファ `grad_w_partial` は
(out_dim, R*32, 2^n) に広がり、2段目の縮約は従来どおり `.sum(1)` が担う。
iwp / warp の backward_w(`iwp_backward_w` / `warp_backward_w`)も同じ規則で R を決める。

ここで守るべき不変条件は2つ:
  1. R がいくつであっても勾配は参照実装(core/reference.py)と float64 で一致する
  2. 行数が R*32 で割り切れない形状(=ストライドの端数)でも取りこぼしが無い

既存の @cuda テストは out_dim が大きめの形状が中心なので、
**R が実際に 1 より大きくなる小 out_dim × 大バッチ**の形状を重点的に追加する。
(R = min(メモリ上限, ceil(rows/32), 飽和上限) なので out_dim が小さいほど R は大きい)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.core import reference_grads
from griffith_logic.lut_layer import (
    CudaLutFunction,
    CudaLutFunctionIndependent,
    CudaLutFunctionIWP,
    CudaLutFunctionWARP,
    LutLayer,
)

pytestmark = pytest.mark.cuda

# R の分割境界(32スレッド × R ブロック)を跨ぐ・跨がない両方を含むバッチ列。
# 1〜1000 は損益分岐の閾値(行数4096)未満なので R=1(旧経路)へフォールスルーする側、
# 4097/8193/12345 は R>1 が実際に立つ側。いずれも 32 の倍数 +α の端数を含む。
BATCHES = [1, 31, 32, 33, 64, 65, 129, 1000, 4097, 8193, 12345]
# out_dim が小さいほど R が大きくなる(の token_mix は out_dim=1〜16)
OUT_DIMS = [1, 2, 3, 7, 16]


def _require_kernel():
    from griffith_logic.cuda import cuda_available
    if not cuda_available():
        pytest.skip('CUDAカーネルのコンパイル失敗')


def _apply(layer, x, w, n_inputs, param):
    if param == 'iwp':
        return CudaLutFunctionIWP.apply(
            x, w, layer.indices, n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
        )
    if param == 'warp':
        return CudaLutFunctionWARP.apply(
            x, w, layer.indices, n_inputs, 1.0, None,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
        )
    if param == 'softmax':
        return CudaLutFunction.apply(
            x, w, layer.indices, n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            getattr(layer, 'gate_table', None),
        )
    return CudaLutFunctionIndependent.apply(
        x, w, layer.indices, n_inputs,
        layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
    )


def _compare_with_reference(n_inputs, param, in_dim, out_dim, batch, seed=7):
    """CUDA経路の (y, grad_x, grad_w) を float64 で参照実装と照合する。"""
    _require_kernel()
    torch.manual_seed(seed)
    layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                     parameterization=param).double()
    gate_table = getattr(layer, 'gate_table', None)
    gate_table = gate_table.double() if gate_table is not None else None

    x = torch.rand(batch, in_dim, dtype=torch.float64, device='cuda')
    w = layer.weights.detach().clone()
    grad_output = torch.randn(batch, out_dim, dtype=torch.float64, device='cuda')

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    y = _apply(layer, xr, wr, n_inputs, param)
    gx, gw = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)
    torch.cuda.synchronize()

    y_ref, gx_ref, gw_ref = reference_grads(
        x, w, layer.indices, n_inputs, param, grad_output,
        gate_table=gate_table,
    )

    tag = f'param={param} N={n_inputs} out_dim={out_dim} batch={batch}'
    assert torch.allclose(y.detach(), y_ref, atol=TOL_F64_EQ, rtol=0), (
        f'{tag}: 出力不一致 {(y.detach() - y_ref).abs().max():.3e}'
    )
    assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), (
        f'{tag}: grad_x不一致 {(gx - gx_ref).abs().max():.3e}'
    )
    assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
        f'{tag}: grad_weights不一致 {(gw - gw_ref).abs().max():.3e}'
    )


@pytest.mark.parametrize('batch', BATCHES)
@pytest.mark.parametrize('out_dim', OUT_DIMS)
def test_small_out_dim_matches_reference(out_dim, batch):
    """小out_dim(=R大)× 端数バッチで参照実装と一致すること(softmax n=2)"""
    _compare_with_reference(2, 'softmax', in_dim=32, out_dim=out_dim, batch=batch)


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
@pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp', 'warp'])
def test_small_out_dim_all_kernels_match_reference(param, n_inputs):
    """N ∈ {2,4,6} × パラメータ化で、小out_dim×大バッチでも一致すること

    2^N がバッファの最終次元なので、Nごとに R のメモリ上限が変わる回帰ポイント。
    """
    _compare_with_reference(n_inputs, param, in_dim=32, out_dim=3, batch=5001)


@pytest.mark.parametrize('param', ['softmax', 'iwp', 'warp'])
@pytest.mark.parametrize('batch', [4096, 4097, 8193, 12345])
@pytest.mark.parametrize('out_dim', [1, 3, 16])
def test_modes_agree_within_float64_tolerance(out_dim, batch, param):
    """legacy(R=1) / 既定(適応) / new(飽和上限なし)の3経路が float64 で一致すること

    R が変わると `.sum(1)` の縮約長が変わり加算順序も変わるが、float64 では
    参照実装との許容誤差(1e-10)の中に収まっていなければならない。
    ここは「R の値そのものを変えても答えが変わらない」ことの直接確認。
    """
    _require_kernel()
    grads = {}
    for mode in ('legacy', '', 'new'):
        prev = os.environ.get('MD_BACKWARD_W')
        if mode:
            os.environ['MD_BACKWARD_W'] = mode
        else:
            os.environ.pop('MD_BACKWARD_W', None)
        try:
            torch.manual_seed(21)
            layer = LutLayer(32, out_dim, n_inputs=2, device='cuda',
                             parameterization=param).double()
            x = torch.rand(batch, 32, dtype=torch.float64, device='cuda')
            grad_output = torch.randn(batch, out_dim, dtype=torch.float64,
                                      device='cuda')
            wr = layer.weights.detach().clone().requires_grad_(True)
            y = _apply(layer, x, wr, 2, param)
            (gw,) = torch.autograd.grad(y, (wr,), grad_outputs=grad_output)
            torch.cuda.synchronize()
            grads[mode or 'adaptive'] = gw.clone()
        finally:
            if prev is None:
                os.environ.pop('MD_BACKWARD_W', None)
            else:
                os.environ['MD_BACKWARD_W'] = prev

    base = grads['legacy']
    for name, gw in grads.items():
        diff = (gw - base).abs().max().item()
        assert diff < TOL_F64_GRAD, (
            f'param={param} out_dim={out_dim} batch={batch}: '
            f'{name} が legacy と不一致 {diff:.3e}'
        )


def test_wide_out_dim_matches_reference():
    """out_dimが大きい(=メモリ上限と飽和上限で R が小さく抑えられる)側の一致"""
    _compare_with_reference(2, 'softmax', in_dim=64, out_dim=4096, batch=65)
    _compare_with_reference(2, 'softmax', in_dim=64, out_dim=4096, batch=4097)


@pytest.mark.parametrize('batch', [33, 1000, 4097])
def test_small_out_dim_gradcheck(batch):
    """小out_dim × 端数バッチでの float64 gradcheck"""
    _require_kernel()
    torch.manual_seed(3)
    n_inputs, in_dim, out_dim = 2, 8, 2
    layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                     parameterization='softmax').double()
    gate_table = getattr(layer, 'gate_table', None)
    x = (torch.rand(batch, in_dim, dtype=torch.float64, device='cuda')
         * 0.8 + 0.1).requires_grad_(True)
    w = layer.weights.detach().clone().requires_grad_(True)

    def fn(x, w):
        return CudaLutFunction.apply(
            x, w, layer.indices, n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            gate_table,
        )

    assert torch.autograd.gradcheck(fn, (x, w), atol=1e-4, rtol=1e-3), (
        f'out_dim={out_dim} batch={batch} の勾配チェックが失敗'
    )


def test_grad_w_is_deterministic():
    """同一入力に対して grad_w がビット単位で再現すること

    R>1 でも各スレッドの担当行は固定(atomicAddを使わない)ため、
    加算順序は決定的でなければならない。
    """
    _require_kernel()
    torch.manual_seed(1)
    n_inputs, in_dim, out_dim, batch = 2, 32, 4, 8192
    layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                     parameterization='softmax')
    x = torch.rand(batch, in_dim, device='cuda')
    grad_output = torch.randn(batch, out_dim, device='cuda')

    grads = []
    for _ in range(3):
        wr = layer.weights.detach().clone().requires_grad_(True)
        y = _apply(layer, x, wr, n_inputs, 'softmax')
        (gw,) = torch.autograd.grad(y, (wr,), grad_outputs=grad_output)
        torch.cuda.synchronize()
        grads.append(gw.clone())

    assert torch.equal(grads[0], grads[1])
    assert torch.equal(grads[0], grads[2])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
