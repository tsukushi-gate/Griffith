"""convert_arith_native_faninrelax.py の再利用可能ヘルパの単体テスト。

`rebuild_as_native16g_learnable_wiring`(固定配線native16g DLGNを
wiring='learnable'(DWN LearnableMapping)版へ差し替えるヘルパ)が、
差し替え直後に固定配線版とビット厳密に一致することを保証する
(配線初期値をfixed版のindicesと同じargmax one-hotにしているため、理論上
必ず一致するはずの性質。CONTRIBUTING.mdのテスト規約に基づき、この性質を
専用テストで固定する)。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from convert_arith_native_faninrelax import (
    assert_native16g_learnable_wiring_bit_exact,
    rebuild_as_native16g_learnable_wiring,
)
from convert_arith_to_native_dlgn import rebuild_as_native16g

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir


def _all_patterns(width: int) -> np.ndarray:
    n = 1 << width
    return np.array(
        [[(k >> j) & 1 for j in range(width)] for k in range(n)], dtype=np.int64,
    )


def _small_n2_ir(param='independent', seed=11):
    """n_inputs=2のLutLayer列+GroupSumを持つ小さなIRを作る(擬似転写IRとして使う)。"""
    torch.manual_seed(seed)
    in_width, k_classes, group_width = 6, 3, 2
    model = nn.Sequential(
        LutLayer(in_width, 8, n_inputs=2, parameterization=param, connections='random'),
        LutLayer(8, k_classes * group_width, n_inputs=2, parameterization=param,
                 connections='random'),
        GroupSum(k=k_classes, tau=1.0),
    )
    model.eval()
    ir = export_ir(model, input_width=in_width, name='test.n2ir_dwn')
    return ir, in_width


class TestRebuildAsNative16gLearnableWiring:
    @pytest.mark.parametrize('param', ['independent', 'raw'])
    def test_bit_exact_vs_fixed_wiring(self, param, seed):
        """差し替え直後のargmaxが固定配線版と全パターン一致する。"""
        ir, in_width = _small_n2_ir(param=param)
        x = _all_patterns(in_width)

        fixed_model = rebuild_as_native16g(ir)
        learnable_model = rebuild_as_native16g_learnable_wiring(ir)

        # 例外が出ないこと自体が主張の検証
        assert_native16g_learnable_wiring_bit_exact(fixed_model, learnable_model, x)

        fixed_model.eval()
        learnable_model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32)
            y_fixed = fixed_model(x_t).numpy()
            y_learnable = learnable_model(x_t).numpy()
        assert np.array_equal(y_fixed.argmax(axis=-1), y_learnable.argmax(axis=-1))

    def test_layers_are_learnable_wiring(self, seed):
        """差し替え後の全LutLayerがwiring='learnable'でlearnable_mappingを持つ。"""
        ir, _ = _small_n2_ir()
        model = rebuild_as_native16g_learnable_wiring(ir)
        lut_layers = [m for m in model if isinstance(m, LutLayer)]
        assert len(lut_layers) == 2
        for lay in lut_layers:
            assert lay.wiring == 'learnable'
            assert hasattr(lay, 'learnable_mapping')

    def test_gradient_flows_through_wiring(self, seed):
        """learnable_mapping.weightsに勾配が流れる(配線が学習対象であることの疎通確認)。"""
        ir, in_width = _small_n2_ir()
        model = rebuild_as_native16g_learnable_wiring(ir)
        model.train()
        x = torch.rand(4, in_width)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for m in model:
            if isinstance(m, LutLayer):
                assert m.learnable_mapping.weights.grad is not None
                assert torch.isfinite(m.learnable_mapping.weights.grad).all()

    def test_mismatch_detected_when_wiring_perturbed(self, seed):
        """配線初期値をわざと崩すと厳密性assertがRuntimeErrorで検出する(検出力の確認)。"""
        ir, in_width = _small_n2_ir()
        fixed_model = rebuild_as_native16g(ir)
        learnable_model = rebuild_as_native16g_learnable_wiring(ir)
        first_layer = next(m for m in learnable_model if isinstance(m, LutLayer))
        with torch.no_grad():
            # 配線one-hotを列方向にロールして意図的にずらす(異なる接続を選ばせる)
            first_layer.learnable_mapping.weights.copy_(
                torch.roll(first_layer.learnable_mapping.weights, shifts=1, dims=0)
            )
        x = _all_patterns(in_width)
        with pytest.raises(RuntimeError, match='厳密性assert違反'):
            assert_native16g_learnable_wiring_bit_exact(fixed_model, learnable_model, x)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
