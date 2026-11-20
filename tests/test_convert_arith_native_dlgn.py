"""convert_arith_to_native_dlgn.py の再利用可能ヘルパの単体テスト。

`rebuild_as_native16g`(n_inputs=2の転写IRをsoftmax16g native DLGNへ復元する
ヘルパ)が、転写直後のIRとビット厳密に一致することを保証する
(2入力真理値表→16ゲートIDの写像は全単射なので理論上必ず一致するはずの
性質。CONTRIBUTING.mdのテスト規約に基づき、この性質を専用テストで固定する)。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from convert_arith_to_native_dlgn import (
    assert_native16g_bit_exact,
    native_gate_count,
    rebuild_as_native16g,
)

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.ir.simulate import simulate


def _all_patterns(width: int) -> np.ndarray:
    n = 1 << width
    return np.array(
        [[(k >> j) & 1 for j in range(width)] for k in range(n)], dtype=np.int64,
    )


def _small_n2_ir(param='independent', seed=7):
    """n_inputs=2のLutLayer列+GroupSumを持つ小さなIRを作る(擬似転写IRとして使う)。

    param='independent'/'raw'は任意の真理値表を厳密表現できるため、転写IRの
    近似(sparsify_fanin等)を経ずにランダムな2入力回路を作れる。
    """
    torch.manual_seed(seed)
    in_width, k_classes, group_width = 6, 3, 2
    model = nn.Sequential(
        LutLayer(in_width, 8, n_inputs=2, parameterization=param, connections='random'),
        LutLayer(8, k_classes * group_width, n_inputs=2, parameterization=param,
                 connections='random'),
        GroupSum(k=k_classes, tau=1.0),
    )
    model.eval()
    ir = export_ir(model, input_width=in_width, name='test.n2ir')
    return ir, in_width


class TestRebuildAsNative16g:
    @pytest.mark.parametrize('param', ['independent', 'raw'])
    def test_bit_exact_vs_simulate(self, param, seed):
        """rebuild_as_native16g直後のargmaxが元IRのsimulate argmaxと全パターン一致する。"""
        ir, in_width = _small_n2_ir(param=param)
        x = _all_patterns(in_width)
        model = rebuild_as_native16g(ir)
        # assert_native16g_bit_exactが例外を出さないこと自体が主張の検証
        assert_native16g_bit_exact(ir, model, x)

        model.eval()
        with torch.no_grad():
            y_model = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
        y_sim = simulate(ir, x)
        assert np.array_equal(y_model.argmax(axis=-1), y_sim.argmax(axis=-1))

    def test_rejects_non_n2_layers(self, seed):
        """n_inputs!=2のlut_layerを含むIRは明示的にValueErrorで拒否される。"""
        torch.manual_seed(3)
        model = nn.Sequential(
            LutLayer(8, 8, n_inputs=4, parameterization='independent'),
            GroupSum(k=2, tau=1.0),
        )
        model.eval()
        ir = export_ir(model, input_width=8, name='test.n4ir')
        with pytest.raises(ValueError, match='n_inputs=2専用'):
            rebuild_as_native16g(ir)

    def test_native_gate_count_matches_estimate(self, seed):
        """native_gate_countはlut_layer出力ノード数の総和(estimate_resourcesの委譲)。"""
        ir, _ = _small_n2_ir()
        # 8(第1層out)+6(第2層out=k_classes*group_width=3*2)
        assert native_gate_count(ir) == 8 + 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
