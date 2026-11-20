"""VHDLシミュレーション等価性テスト(@hdl_vhdl)。

生成VHDL(export/vhdl.py)⇔ IR simulate(golden model)の一致を
GHDLで検証する。Verilog版(test_hdl_equivalence.py)と同じベクトル形式。
GHDL非導入環境では自動skip。実行: uv run pytest -m hdl_vhdl
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutConv2d, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.export.verify import ghdl_available, run_ghdl_equivalence
from griffith_logic.pooling import OrPooling2d

pytestmark = [
    pytest.mark.hdl_vhdl,
    pytest.mark.skipif(not ghdl_available(), reason='ghdl未導入'),
]


def _binary(batch, width, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, width), generator=g).numpy().astype(np.int64)


class TestVhdlEquivalence:
    @pytest.mark.parametrize('param,n', [
        ('softmax', 2), ('softmax', 6), ('independent', 4),
        ('iwp', 4), ('warp', 6),
    ], ids=['softmax_n2', 'softmax_n6', 'independent_n4', 'iwp_n4', 'warp_n6'])
    def test_dense_random_vectors(self, param, n, tmp_path, seed):
        torch.manual_seed(41)
        model = nn.Sequential(
            LutLayer(16, 24, n_inputs=n, parameterization=param),
            LutLayer(24, 24, n_inputs=n, parameterization=param),
            GroupSum(k=4, tau=2.0),
        )
        model.eval()
        ir = export_ir(model, input_width=16, name=f'{param}_n{n}')
        x = _binary(256, 16, seed=9)
        result = run_ghdl_equivalence(ir, x, str(tmp_path))
        assert result['passed'], (
            f'{param} n={n}: VHDLとgolden modelが不一致\n{result["stdout"]}'
        )

    def test_cnn_pipeline(self, tmp_path, seed):
        torch.manual_seed(53)
        model = nn.Sequential(
            LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1),
            OrPooling2d(kernel_size=2),
            nn.Flatten(),
            LutLayer(4 * 4 * 4, 24, n_inputs=2),
            GroupSum(k=4, tau=2.0),
        )
        model.eval()
        ir = export_ir(model, input_shape=(1, 8, 8), name='cnn8_vhdl')
        x = _binary(64, 64, seed=19)
        result = run_ghdl_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_lut_only_output(self, tmp_path, seed):
        torch.manual_seed(47)
        model = nn.Sequential(LutLayer(8, 16, n_inputs=2))
        model.eval()
        ir = export_ir(model, input_width=8, name='raw_bits_vhdl')
        x = _binary(64, 8, seed=17)
        result = run_ghdl_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_group_sum_stride_assign(self, tmp_path, seed):
        """GroupSum(assign='stride')(恒等 lut_layer の転置 + group_sum)の VHDL 一致"""
        torch.manual_seed(61)
        model = nn.Sequential(
            LutLayer(16, 24, n_inputs=2),
            LutLayer(24, 24, n_inputs=2),
            GroupSum(k=4, tau=2.0, assign='stride'),
        )
        model.eval()
        ir = export_ir(model, input_width=16, name='gs_stride_vhdl')
        x = _binary(256, 16, seed=29)
        result = run_ghdl_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'hdl_vhdl'])
