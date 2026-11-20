"""L2: CUDAカーネルとPython実装の一致検証(@cuda)。

旧tests/test_lut_layer.pyのCUDAテスト群を、gate_table引数追加後の
現行シグネチャに修正して移設したもの。
CUDA非対応環境ではconftestの自動skipにより実行されない。
Modal/GPU環境での実行: uv run pytest -m cuda

検証チェーン:
    CUDA Function == Python Function == 参照実装(core/reference.py)
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import (
    GRADCHECK_KW,
    TOL_F32_EQ,
    TOL_F64_EQ,
    TOL_F64_GRAD,
    assert_all_finite,
)

from griffith_logic.core import reference_grads
from griffith_logic.group_sum import GroupSum
from griffith_logic.lut_layer import (
    CudaLutFunction,
    CudaLutFunctionIndependent,
    CudaLutFunctionIWP,
    CudaLutFunctionWARP,
    LutFunction,
    LutFunctionIndependent,
    LutFunctionIWP,
    LutFunctionWARP,
    LutLayer,
)

pytestmark = pytest.mark.cuda


def _require_kernel():
    from griffith_logic.cuda import cuda_available
    if not cuda_available():
        pytest.skip('CUDAカーネルのコンパイル失敗')


class TestCudaForwardConsistency:
    """CUDA forward の出力がPython実装と一致するか"""

    def _compare_forward(self, n_inputs, in_dim=16, out_dim=8, batch=4,
                         param='softmax'):
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                         parameterization=param)
        gate_table = getattr(layer, 'gate_table', None)
        x = torch.rand(batch, in_dim, device='cuda')

        layer.train()
        if param == 'softmax':
            y_python = LutFunction.apply(
                x, layer.weights, layer.indices, n_inputs, gate_table,
            )
            y_cuda = CudaLutFunction.apply(
                x, layer.weights, layer.indices, n_inputs,
                layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
                gate_table,
            )
        elif param == 'iwp':
            y_python = LutFunctionIWP.apply(
                x, layer.weights, layer.indices, n_inputs,
            )
            y_cuda = CudaLutFunctionIWP.apply(
                x, layer.weights, layer.indices, n_inputs,
                layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            )
        else:
            y_python = LutFunctionIndependent.apply(
                x, layer.weights, layer.indices, n_inputs,
            )
            y_cuda = CudaLutFunctionIndependent.apply(
                x, layer.weights, layer.indices, n_inputs,
                layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            )

        assert torch.allclose(y_python, y_cuda, atol=TOL_F32_EQ, rtol=0), (
            f'param={param}, N={n_inputs}: '
            f'max diff = {(y_python - y_cuda).abs().max().item()}'
        )

    def test_softmax_n2(self):
        self._compare_forward(2, param='softmax')

    def test_softmax_n4(self):
        self._compare_forward(4, param='softmax')

    def test_softmax_n6(self):
        self._compare_forward(6, in_dim=24, param='softmax')

    def test_independent_n2(self):
        self._compare_forward(2, param='independent')

    def test_independent_n4(self):
        self._compare_forward(4, param='independent')

    def test_independent_n6(self):
        self._compare_forward(6, in_dim=24, param='independent')

    def test_iwp_n2(self):
        self._compare_forward(2, param='iwp')

    def test_iwp_n4(self):
        self._compare_forward(4, param='iwp')

    def test_iwp_n6(self):
        self._compare_forward(6, in_dim=24, param='iwp')


class TestCudaGradient:
    """CUDA backward の勾配が数値微分と一致するか(x・weights両方)"""

    def _gradcheck_cuda(self, n_inputs, in_dim=8, out_dim=4, batch=2,
                        param='softmax'):
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                         parameterization=param).double()
        x = torch.rand(batch, in_dim, dtype=torch.float64, device='cuda') \
            * 0.8 + 0.1
        x = x.requires_grad_(True)
        w = layer.weights.detach().clone().requires_grad_(True)
        gate_table = getattr(layer, 'gate_table', None)

        if param == 'softmax':
            fn = lambda x, w: CudaLutFunction.apply(
                x, w, layer.indices, n_inputs,
                layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
                gate_table,
            )
        elif param == 'iwp':
            fn = lambda x, w: CudaLutFunctionIWP.apply(
                x, w, layer.indices, n_inputs,
                layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            )
        else:
            fn = lambda x, w: CudaLutFunctionIndependent.apply(
                x, w, layer.indices, n_inputs,
                layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            )
        assert torch.autograd.gradcheck(fn, (x, w), **GRADCHECK_KW), (
            f'CUDA param={param}, N={n_inputs} の勾配チェックが失敗'
        )

    def test_softmax_n2(self):
        self._gradcheck_cuda(2, param='softmax')

    def test_softmax_n4(self):
        self._gradcheck_cuda(4, param='softmax')

    def test_softmax_n6(self):
        self._gradcheck_cuda(6, in_dim=12, param='softmax')

    def test_independent_n2(self):
        self._gradcheck_cuda(2, param='independent')

    def test_independent_n4(self):
        self._gradcheck_cuda(4, param='independent')

    def test_independent_n6(self):
        self._gradcheck_cuda(6, in_dim=12, param='independent')

    def test_iwp_n2(self):
        self._gradcheck_cuda(2, param='iwp')

    def test_iwp_n4(self):
        self._gradcheck_cuda(4, param='iwp')

    def test_iwp_n6(self):
        self._gradcheck_cuda(6, in_dim=12, param='iwp')


class TestCudaWarpForwardConsistency:
    """WARP CUDA forward(warp_forward_kernel)の出力がPython実装
    (LutFunctionWARP)と一致するか。noise=None(gumbel_training=False)
    の決定的条件で比較する(parityテストの前提: RNG非依存で厳密一致するはず)。
    """

    def _compare_forward(self, n_inputs, in_dim=16, out_dim=8, batch=4, tau=1.0):
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                         parameterization='warp')
        x = torch.rand(batch, in_dim, device='cuda')
        theta = layer.weights.detach().clone()

        y_python = LutFunctionWARP.apply(
            x, theta, layer.indices, n_inputs, tau, False,
        )
        y_cuda = CudaLutFunctionWARP.apply(
            x, theta, layer.indices, n_inputs, tau, None,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
        )

        assert torch.allclose(y_python, y_cuda, atol=TOL_F32_EQ, rtol=0), (
            f'warp forward N={n_inputs}: '
            f'max diff = {(y_python - y_cuda).abs().max().item()}'
        )

    def test_warp_n2(self):
        self._compare_forward(2)

    def test_warp_n4(self):
        self._compare_forward(4)

    def test_warp_n6(self):
        self._compare_forward(6, in_dim=24)

    def test_warp_n4_tau(self):
        # τ≠1でも正しくスケーリングされるか(1/τをカーネルに渡す設計の検証)
        self._compare_forward(4, tau=0.37)


class TestCudaWarpGradient:
    """WARP CUDA backward(warp_backward_w/warp_backward_x)の勾配が
    数値微分と一致するか(x・weights両方、float64 gradcheck)。既存
    _gradcheck_cuda の流儀を踏襲。
    """

    def _gradcheck_cuda_warp(self, n_inputs, in_dim=8, out_dim=4, batch=2,
                             tau=1.0):
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                         parameterization='warp').double()
        x = torch.rand(batch, in_dim, dtype=torch.float64, device='cuda') \
            * 0.8 + 0.1
        x = x.requires_grad_(True)
        w = layer.weights.detach().clone().requires_grad_(True)

        fn = lambda x, w: CudaLutFunctionWARP.apply(
            x, w, layer.indices, n_inputs, tau, None,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
        )
        assert torch.autograd.gradcheck(fn, (x, w), **GRADCHECK_KW), (
            f'CUDA warp N={n_inputs} の勾配チェックが失敗'
        )

    def test_warp_n2(self):
        self._gradcheck_cuda_warp(2)

    def test_warp_n4(self):
        self._gradcheck_cuda_warp(4)

    def test_warp_n6(self):
        self._gradcheck_cuda_warp(6, in_dim=12)


class TestCudaMemorySaving:
    """CUDAカーネルでメモリ使用量が削減されることを確認"""

    def test_memory_reduction(self):
        _require_kernel()
        in_dim, out_dim, n_inputs, batch = 64, 10000, 6, 32

        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                         parameterization='softmax')
        gate_table = getattr(layer, 'gate_table', None)
        x = torch.rand(batch, in_dim, device='cuda')

        torch.cuda.reset_peak_memory_stats()
        y_py = LutFunction.apply(
            x, layer.weights, layer.indices, n_inputs, gate_table,
        )
        y_py.sum().backward()
        mem_python = torch.cuda.max_memory_allocated()

        layer.weights.grad = None
        torch.cuda.reset_peak_memory_stats()
        y_cuda = CudaLutFunction.apply(
            x, layer.weights, layer.indices, n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
            gate_table,
        )
        y_cuda.sum().backward()
        mem_cuda = torch.cuda.max_memory_allocated()

        assert mem_cuda < mem_python, (
            f'CUDA({mem_cuda / 1e6:.1f}MB) >= Python({mem_python / 1e6:.1f}MB)'
        )


class TestWarpDispatchRegression:
    """warp×soft がCUDA環境でCudaLutFunction(softmax意味論)へ誤ディスパッチ
    されない回帰テスト(除外tupleにwarpが無く、θがsoftmax重み
    として誤解釈され数理が変わっていた)。CPU実行(常にPython WARP経路)と
    CUDA実行の出力一致で検証する。
    """

    def test_warp_soft_matches_cpu_python_path(self):
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(16, 8, n_inputs=2, device='cuda',
                         parameterization='warp')
        # Gumbelノイズを無効化して決定的に(train.pyの--no-gumbel相当)
        if hasattr(layer, 'gumbel'):
            layer.gumbel = False
        layer.gumbel_scale = 0.0
        layer.train()
        x = torch.rand(4, 16, device='cuda')
        y_cuda = layer(x).detach().cpu()

        layer_cpu = layer.to('cpu')
        y_cpu = layer_cpu(x.cpu()).detach()

        assert torch.allclose(y_cpu, y_cuda, atol=TOL_F32_EQ, rtol=0), (
            f'warp/softのCUDA出力がPython WARP経路と不一致: '
            f'max diff = {(y_cpu - y_cuda).abs().max().item()}'
            f'(CudaLutFunctionへの誤ディスパッチの疑い)'
        )


class TestCudaFallback:
    """CPU入力時はPythonフォールバックが使われることを確認"""

    def test_cpu_input_uses_python_path(self):
        layer = LutLayer(8, 4, n_inputs=2, parameterization='softmax')
        x = torch.rand(2, 8)
        y = layer(x)
        assert y.shape == (2, 4)
        y.sum().backward()
        assert layer.weights.grad is not None


class TestAutocastSmoke:
    """autocast(混合精度)のスモークテスト。

    融合CUDAカーネル版のautograd Function(Cuda*)は、autocastが有効な区間では
    **そのautocastのdtype**へ浮動小数点入力をキャストしてカーネルを動かす
    (パラメータ/optimizerはfp32のまま。core/functions.py参照)。

    bf16/fp16は仮数が短く、他のCUDA parityテストと同じ基準での数値一致は
    本質的に成立しない。tests/conftest.pyに低精度用の許容誤差定数は存在せず、
    新しい閾値を発明しない方針のため、数値一致(parity)テストは追加しない。代わりに
        1) autocast経由でもforward/backwardが有限値で完走する
        2) パラメータ本体(nn.Parameter)とその勾配はfp32のまま保たれる
        3) カーネルがautocastのdtype(bf16/fp16)で動く(bf16固定ではない)
        4) 数stepのAdam学習でlossが下がる(学習が破綻していない)
    を確認するスモークテストに留める。
    """

    @pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp', 'warp'])
    @pytest.mark.parametrize('amp_dtype', [torch.bfloat16, torch.float16])
    def test_kernel_runs_in_autocast_dtype(self, param, amp_dtype):
        """カーネルの出力dtypeがautocastのdtypeに一致する(fp16 AMPでbf16に化けない)"""
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(16, 8, n_inputs=4, device='cuda',
                         parameterization=param)
        layer.train()
        x = torch.rand(4, 16, device='cuda')
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            y = layer(x)
        assert y.dtype == amp_dtype, f'param={param}: {y.dtype} != {amp_dtype}'
        assert_all_finite(y, f'autocast({amp_dtype}) forward param={param}')

    def test_no_autocast_keeps_fp32(self):
        """autocast外では従来どおりfp32のまま(キャストしない)"""
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(16, 8, n_inputs=4, device='cuda')
        layer.train()
        y = layer(torch.rand(4, 16, device='cuda'))
        assert y.dtype == torch.float32

    @pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp', 'warp'])
    def test_forward_backward_finite_and_fp32_master_weights(self, param):
        _require_kernel()
        torch.manual_seed(42)
        layer = LutLayer(16, 8, n_inputs=4, device='cuda',
                         parameterization=param)
        layer.train()
        x = torch.rand(4, 16, device='cuda')

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            y = layer(x)

        assert y.shape == (4, 8)
        assert_all_finite(y, f'bf16 forward param={param}')

        y.float().sum().backward()
        assert layer.weights.grad is not None
        # マスターパラメータ・勾配ともfp32のまま(bf16へ格下げされない)こと
        assert layer.weights.dtype == torch.float32
        assert layer.weights.grad.dtype == torch.float32
        assert_all_finite(layer.weights.grad, f'bf16 backward grad param={param}')

    def test_training_loss_decreases_under_autocast(self):
        """bf16 autocastのミニ学習ループ(2層softmax LutLayer + GroupSum)で
        数step後にlossが初期値より下がることを確認する(収束閾値ではなく
        「学習が進む」ことのみを見るスモークテスト)。
        """
        _require_kernel()
        torch.manual_seed(0)
        in_dim, k, num_classes = 32, 64, 4
        model = torch.nn.Sequential(
            LutLayer(in_dim, k, n_inputs=4, device='cuda', parameterization='softmax'),
            LutLayer(k, k, n_inputs=4, device='cuda', parameterization='softmax'),
            GroupSum(num_classes),
        ).to('cuda')
        model.train()

        x = torch.bernoulli(torch.full((64, in_dim), 0.5, device='cuda'))
        y = torch.randint(0, num_classes, (64,), device='cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        losses = []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        assert_all_finite(torch.tensor(losses), 'bf16学習中のloss系列')
        assert losses[-1] < losses[0], (
            f'bf16 autocastの学習でlossが下がらなかった: '
            f'first={losses[0]:.4f} last={losses[-1]:.4f}'
        )


def _apply_cuda_fn(layer, x, w, n_inputs, param, tau=1.0, noise=None):
    """iwp/warp の融合カーネル版Functionを呼ぶ(テスト用の薄いラッパ)。"""
    if param == 'iwp':
        return CudaLutFunctionIWP.apply(
            x, w, layer.indices, n_inputs,
            layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
        )
    return CudaLutFunctionWARP.apply(
        x, w, layer.indices, n_inputs, tau, noise,
        layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot,
    )


class TestCudaIwpWarpMatchesReference:
    """iwp/warp の融合カーネルが参照実装(core/reference.py)と float64 で一致するか。

    検証チェーン CUDA Function == 参照実装 を、出力・grad_x・grad_weights の
    3つについて conftest の許容誤差(TOL_F64_EQ / TOL_F64_GRAD)で確認する。
    """

    @pytest.mark.parametrize('n_inputs', [2, 4, 6])
    @pytest.mark.parametrize('param,tau', [('iwp', 1.0), ('warp', 1.0), ('warp', 0.37)])
    def test_output_and_grads_match_reference(self, param, tau, n_inputs):
        _require_kernel()
        torch.manual_seed(11)
        in_dim, out_dim, batch = 24, 9, 37
        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                         parameterization=param).double()
        x = torch.rand(batch, in_dim, dtype=torch.float64, device='cuda')
        w = layer.weights.detach().clone()
        grad_output = torch.randn(batch, out_dim, dtype=torch.float64, device='cuda')

        xr = x.clone().requires_grad_(True)
        wr = w.clone().requires_grad_(True)
        y = _apply_cuda_fn(layer, xr, wr, n_inputs, param, tau=tau)
        gx, gw = torch.autograd.grad(y, (xr, wr), grad_outputs=grad_output)

        y_ref, gx_ref, gw_ref = reference_grads(
            x, w, layer.indices, n_inputs, param, grad_output, tau=tau,
        )
        tag = f'param={param} tau={tau} N={n_inputs}'
        assert torch.allclose(y.detach(), y_ref, atol=TOL_F64_EQ, rtol=0), (
            f'{tag}: 出力不一致 {(y.detach() - y_ref).abs().max():.3e}'
        )
        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'{tag}: grad_x不一致 {(gx - gx_ref).abs().max():.3e}'
        )
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'{tag}: grad_weights不一致 {(gw - gw_ref).abs().max():.3e}'
        )


class TestWarpGumbelRngParity:
    """Gumbelノイズありのwarpで、Python実装と融合カーネル経路が同じ乱数を同じ順で消費するか。

    CPUとCUDAの乱数生成器は別系列なので、CPU実行とCUDA実行の値は原理的に一致しない。
    そこで同一CUDAデバイス上で、Python実装(LutFunctionWARP、Gumbelノイズを内部生成)と
    LutLayer の融合カーネル経路(ノイズをPython側で生成してカーネルへ渡す)を同じseedで
    実行し、(1) 出力・勾配が一致すること、(2) 実行後の乱数生成器の状態が一致すること
    (=消費した乱数の量と順序が同じ)を確認する。
    """

    @pytest.mark.parametrize('n_inputs', [2, 6])
    def test_python_and_kernel_consume_same_random_numbers(self, n_inputs):
        _require_kernel()
        torch.manual_seed(5)
        layer = LutLayer(16, 8, n_inputs=n_inputs, device='cuda',
                         parameterization='warp').double()
        assert layer.gumbel, 'warp の既定はGumbelノイズありのはず'
        layer.train()
        x = torch.rand(6, 16, dtype=torch.float64, device='cuda')
        grad_output = torch.randn(6, 8, dtype=torch.float64, device='cuda')

        # Python実装(ノイズは LutFunctionWARP.forward 内で torch.rand から生成)
        torch.cuda.manual_seed(1234)
        w_py = layer.weights.detach().clone().requires_grad_(True)
        x_py = x.clone().requires_grad_(True)
        y_py = LutFunctionWARP.apply(
            x_py, w_py, layer.indices, n_inputs, layer.tau_warp, True,
        )
        rng_after_py = torch.cuda.get_rng_state()
        gx_py, gw_py = torch.autograd.grad(y_py, (x_py, w_py), grad_outputs=grad_output)

        # 融合カーネル経路(LutLayer.forward がノイズを生成してカーネルへ渡す)
        torch.cuda.manual_seed(1234)
        x_k = x.clone().requires_grad_(True)
        y_k = layer(x_k)
        rng_after_kernel = torch.cuda.get_rng_state()
        assert type(y_k.grad_fn).__name__.startswith('CudaLutFunctionWARP'), (
            f'融合カーネル経路を通っていない: {type(y_k.grad_fn).__name__}'
        )
        gx_k, gw_k = torch.autograd.grad(
            y_k, (x_k, layer.weights), grad_outputs=grad_output,
        )

        assert torch.equal(rng_after_py, rng_after_kernel), (
            '乱数生成器の状態がPython実装と融合カーネル経路で異なる(消費量・順序の不一致)'
        )
        assert torch.allclose(y_py.detach(), y_k.detach(), atol=TOL_F64_EQ, rtol=0), (
            f'Gumbelありの出力不一致 {(y_py - y_k).abs().max():.3e}'
        )
        assert torch.allclose(gx_py, gx_k, atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(gw_py, gw_k, atol=TOL_F64_GRAD, rtol=0)


class TestPythonPathEnvVar:
    """環境変数 MD_LUT_PYTHON_PATH で iwp/warp を Python 実装へ戻せるか。"""

    @pytest.mark.parametrize('param', ['iwp', 'warp'])
    def test_env_switches_to_python_function(self, param, monkeypatch):
        _require_kernel()
        torch.manual_seed(3)
        layer = LutLayer(16, 8, n_inputs=4, device='cuda',
                         parameterization=param).double()
        layer.train()
        if param == 'warp':
            layer.gumbel = False
        x = torch.rand(5, 16, dtype=torch.float64, device='cuda')

        monkeypatch.delenv('MD_LUT_PYTHON_PATH', raising=False)
        y_kernel = layer(x)
        assert type(y_kernel.grad_fn).__name__.startswith('CudaLutFunction')

        monkeypatch.setenv('MD_LUT_PYTHON_PATH', 'iwp,warp')
        y_python = layer(x)
        assert not type(y_python.grad_fn).__name__.startswith('CudaLutFunction'), (
            f'MD_LUT_PYTHON_PATH 指定でもカーネル経路: {type(y_python.grad_fn).__name__}'
        )
        assert torch.allclose(y_kernel, y_python, atol=TOL_F64_EQ, rtol=0)

    def test_env_only_affects_listed_parameterization(self, monkeypatch):
        _require_kernel()
        layer = LutLayer(16, 8, n_inputs=4, device='cuda', parameterization='iwp')
        layer.train()
        monkeypatch.setenv('MD_LUT_PYTHON_PATH', 'warp')
        y = layer(torch.rand(5, 16, device='cuda'))
        assert type(y.grad_fn).__name__.startswith('CudaLutFunctionIWP')


class TestLowPrecisionAccumulation:
    """half/bf16 入力でもカーネル内の累積は float(at::acc_type)で行われるか。

    同じ値(低精度へ丸めた入力)を fp32 で計算した結果を低精度へ丸めたものと、
    低精度入力でカーネルを動かした結果が**ビット単位で一致**することを確認する
    (累積・中間計算が float で、丸めが最後の1回だけなら一致する)。
    行数を大きく取るので、低精度のまま累積していれば一致しない。
    """

    @pytest.mark.parametrize('low', [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize('param', ['softmax', 'iwp', 'warp'])
    def test_kernels_accumulate_in_float(self, param, low):
        _require_kernel()
        from griffith_logic.cuda import get_cuda_module
        cuda = get_cuda_module()
        torch.manual_seed(0)
        n_inputs, in_dim, out_dim, batch = 4, 32, 6, 8192
        layer = LutLayer(in_dim, out_dim, n_inputs=n_inputs, device='cuda',
                         parameterization=param)
        idx = layer.indices.contiguous()
        rev = (layer.rev_starts, layer.rev_flat_y, layer.rev_flat_slot)

        def rounded(t):
            return t.to(low).to(torch.float32)

        x_t = rounded(torch.rand(in_dim, batch, device='cuda'))
        g_t = rounded(torch.rand(out_dim, batch, device='cuda'))
        if param == 'softmax':
            w = rounded(torch.softmax(layer.weights.detach(), dim=-1)).contiguous()
        elif param == 'iwp':
            w = rounded(torch.sigmoid(layer.weights.detach())).contiguous()
        else:
            w = rounded(layer.weights.detach()).contiguous()

        def run(dt):
            xs, gs, ws = x_t.to(dt), g_t.to(dt), w.to(dt)
            if param == 'softmax':
                return (cuda.forward(xs, idx, ws, n_inputs),
                        cuda.backward_w(xs, idx, gs, n_inputs),
                        cuda.backward_x(xs, idx, ws, gs, *rev, n_inputs))
            if param == 'iwp':
                return (cuda.iwp_forward(xs, idx, ws, n_inputs),
                        cuda.iwp_backward_w(xs, idx, ws, gs, n_inputs),
                        cuda.iwp_backward_x(xs, idx, ws, gs, *rev, n_inputs))
            return (cuda.warp_forward(xs, idx, ws, 1.0 / 0.37, None, n_inputs),
                    cuda.warp_backward_w(xs, idx, gs, n_inputs),
                    cuda.warp_backward_x(xs, idx, ws, gs, *rev, n_inputs))

        ref = [t.to(low) for t in run(torch.float32)]
        out = run(low)
        for name, a, b in zip(('forward', 'backward_w', 'backward_x'), out, ref):
            assert a.dtype == low
            assert torch.equal(a, b), (
                f'param={param} {low} {name}: fp32計算を丸めた値と不一致 '
                f'(max diff {(a.float() - b.float()).abs().max():.3e})'
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
