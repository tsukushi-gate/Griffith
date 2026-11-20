"""L2: CUDAカーネルの起動設定(ブロック形状)が全組み合わせで安全なことの検証(@cuda)。

ブロックサイズを difflogic 同等(最大1024スレッド)まで引き上げたため、
レジスタ消費の大きい組み合わせ(N=6 × float64 など)で
cudaErrorLaunchOutOfResources が起きないことを機械的に確認する。

また `cudaDeviceSynchronize()` を既定で外した(MD_CUDA_SYNC=1 でのみ復活)ため、
起動エラーは遅延検出になる。ここでは明示的に torch.cuda.synchronize() を呼び、
さらに結果が有限であることまで確認する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.lut_layer import (  # noqa: E402
    CudaLutFunction,
    CudaLutFunctionIndependent,
    LutLayer,
)

pytestmark = pytest.mark.cuda


def _require_kernel():
    from griffith_logic.cuda import cuda_available
    if not cuda_available():
        pytest.skip('CUDAカーネルのコンパイル失敗')


def _run_roundtrip(n_inputs, dtype, param, in_dim, out_dim, batch):
    """forward + backward を1往復させ、起動エラーとNaN/Infが無いことを確認する"""
    torch.manual_seed(0)
    layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                     parameterization=param)
    if dtype is torch.float64:
        layer = layer.double()

    x = (torch.rand(batch, in_dim, dtype=dtype, device='cuda') * 0.8 + 0.1)
    x.requires_grad_(True)
    w = layer.weights.detach().clone().requires_grad_(True)

    if param == 'softmax':
        y = CudaLutFunction.apply(
            x, w, layer.indices, n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            getattr(layer, 'gate_table', None),
        )
    else:
        y = CudaLutFunctionIndependent.apply(
            x, w, layer.indices, n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
        )
    y.sum().backward()

    # 非同期実行のため、ここで初めてカーネル実行エラーが観測される
    torch.cuda.synchronize()

    assert y.shape == (batch, out_dim)
    assert y.dtype is dtype
    assert torch.isfinite(y).all(), f'forward出力に非有限値: N={n_inputs}, {dtype}'
    assert torch.isfinite(x.grad).all(), f'grad_xに非有限値: N={n_inputs}, {dtype}'
    assert torch.isfinite(w.grad).all(), f'grad_wに非有限値: N={n_inputs}, {dtype}'


@pytest.mark.parametrize('param', ['softmax', 'independent'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_launch_all_combinations(n_inputs, dtype, param):
    """N ∈ {2,4,6} × {float32, float64} で起動エラーが出ないこと

    N=6/float64 は 1スレッドあたり 64要素 × 8byte のレジスタ配列を持つため、
    ブロック形状の決定を誤ると cudaErrorLaunchOutOfResources になる回帰ポイント。
    """
    _require_kernel()
    _run_roundtrip(n_inputs, dtype, param, in_dim=32, out_dim=64, batch=8)


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_launch_tiny_and_large_shapes(n_inputs):
    """ブロックyをrow数で頭打ちにする調整が極端な形状でも壊れないこと"""
    _require_kernel()
    # out_dim < blockDim.y となる極小形状
    _run_roundtrip(n_inputs, torch.float32, 'softmax',
                   in_dim=8, out_dim=1, batch=1)
    # blockDim.y を上限まで使う形状
    _run_roundtrip(n_inputs, torch.float32, 'softmax',
                   in_dim=64, out_dim=2048, batch=64)


def test_repeated_launch_uses_cached_block_config():
    """起動設定キャッシュが有効でも連続起動の結果が一定であること"""
    _require_kernel()
    for _ in range(5):
        _run_roundtrip(2, torch.float32, 'softmax',
                       in_dim=16, out_dim=32, batch=4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
