"""L2: 「転置テンソル(x_t)をsave_for_backwardする」最適化を却下した根拠の実測(@cuda)。

`CudaLutFunction.backward` の `x.t().contiguous()` 再計算を省くため、
forward で作った `x_t` を save_for_backward する案を検討した。
しかし `x_t` は forward〜backward の間ずっと生存するためピークメモリが増える。

実測(A10 / 8層 × dim=4096 × batch=512 / 訓練1ステップ):
    x を保存(本番)   : 114,189,312 bytes (114.19 MB)
    x_t を保存(却下) : 124,675,584 bytes (124.68 MB)  → +9.2%

増加分は「入力が LUT 層以外から来る最初の層」の実体コピー(batch×dim×4B = 8.39MB)と
アロケータの丸め。多層連鎖の内側では x.t() が前段出力 y_t とエイリアスになるため
`.contiguous()` はコピーを伴わず、転置削減の速度メリット自体もほぼ無い。
よって本番実装は x を保存したままとし、このテストで再導入を検知する。
"""

import os
import sys
import warnings

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.cuda import get_cuda_module  # noqa: E402
from griffith_logic.lut_layer import CudaLutFunction, LutLayer  # noqa: E402

pytestmark = pytest.mark.cuda


def _require_kernel():
    from griffith_logic.cuda import cuda_available
    if not cuda_available():
        pytest.skip('CUDAカーネルのコンパイル失敗')


class TransposeSavingLutFunction(torch.autograd.Function):
    """却下案: forward の転置結果 x_t を save_for_backward する版。

    本番コードからは使用しない。ピークメモリ比較の対象としてのみ存在する。
    """

    @staticmethod
    def forward(ctx, x, weights, indices, n_inputs, rev_starts, rev_flat_y,
                rev_flat_slot, gate_table):
        cuda = get_cuda_module()
        s = F.softmax(weights, dim=-1)
        w = (s @ gate_table).contiguous() if gate_table is not None else s.contiguous()
        x_t = x.t().contiguous()
        indices_c = indices.contiguous()
        y_t = cuda.forward(x_t, indices_c, w, n_inputs)
        # ここが差分: x ではなく x_t を保存する
        ctx.save_for_backward(x_t, weights, indices_c, rev_starts, rev_flat_y,
                              rev_flat_slot)
        ctx.n_inputs = n_inputs
        ctx.gate_table = gate_table
        return y_t.t()

    @staticmethod
    def backward(ctx, grad_output):
        x_t, weights, indices, rev_starts, rev_flat_y, rev_flat_slot = \
            ctx.saved_tensors
        n_inputs = ctx.n_inputs
        gate_table = ctx.gate_table
        cuda = get_cuda_module()

        grad_y_t = grad_output.t().contiguous()
        grad_eff = cuda.backward_w(x_t, indices, grad_y_t, n_inputs)

        s = F.softmax(weights, dim=-1)
        grad_s = grad_eff @ gate_table.T if gate_table is not None else grad_eff
        dot = (grad_s * s).sum(dim=-1, keepdim=True)
        grad_weights = s * (grad_s - dot)

        grad_x = None
        if ctx.needs_input_grad[0]:
            w = (s @ gate_table).contiguous() if gate_table is not None \
                else s.contiguous()
            grad_x_t = cuda.backward_x(
                x_t, indices, w, grad_y_t,
                rev_starts, rev_flat_y, rev_flat_slot, n_inputs,
            )
            grad_x = grad_x_t.t()

        return grad_x, grad_weights, None, None, None, None, None, None


def _make_layers(n_layers, dim, n_inputs):
    torch.manual_seed(1234)
    return [
        LutLayer(dim, dim, n_inputs=n_inputs, device='cuda',
                 parameterization='softmax')
        for _ in range(n_layers)
    ]


def _train_step_peak(fn, layers, x):
    """1 train step(forward+backward)を回してピークメモリ(bytes)と損失を返す"""
    for layer in layers:
        layer.train()
        layer.weights.grad = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    h = x
    for layer in layers:
        h = fn(
            h, layer.weights, layer.indices, layer.n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            getattr(layer, 'gate_table', None),
        )
    loss = h.sum()
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(), loss.detach().clone()


def test_production_impl_is_memory_leaner_than_transpose_saving():
    """本番実装(x保存)が却下案(x_t保存)よりピークメモリが小さいこと

    本番側を x_t 保存に変えるとこのテストが落ちる(= 再導入の検知)。
    """
    _require_kernel()
    n_layers, dim, batch, n_inputs = 8, 4096, 512, 2
    layers = _make_layers(n_layers, dim, n_inputs)
    x = torch.rand(batch, dim, device='cuda')

    # 計測前のウォームアップ(カーネルJIT・アロケータの初回確保を除外)
    _train_step_peak(TransposeSavingLutFunction.apply, layers, x)
    _train_step_peak(CudaLutFunction.apply, layers, x)

    peak_variant, loss_variant = _train_step_peak(
        TransposeSavingLutFunction.apply, layers, x)
    grads_variant = [layer.weights.grad.detach().clone() for layer in layers]

    peak_prod, loss_prod = _train_step_peak(CudaLutFunction.apply, layers, x)
    grads_prod = [layer.weights.grad.detach().clone() for layer in layers]

    report = (
        f'[peak memory] 本番(x保存)={peak_prod} bytes ({peak_prod / 1e6:.2f} MB) / '
        f'却下案(x_t保存)={peak_variant} bytes ({peak_variant / 1e6:.2f} MB) '
        f'/ 比={peak_prod / peak_variant:.4f}'
    )
    print('\n' + report)
    # pytestは成功テストのstdoutを表示しないため、警告サマリにも実測値を残す
    warnings.warn(report, stacklevel=1)

    # 両者は数値的に等価(メモリ以外の差が無いことの確認)
    assert torch.allclose(loss_variant, loss_prod, atol=1e-4, rtol=1e-4)
    for g_var, g_prod in zip(grads_variant, grads_prod, strict=True):
        assert torch.allclose(g_var, g_prod, atol=1e-5, rtol=1e-4)

    assert peak_prod < peak_variant, (
        'x_t を save_for_backward するとピークメモリが増える(実測)。'
        f' 本番={peak_prod} 却下案={peak_variant}'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
