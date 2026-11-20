"""HDLシミュレーション等価性テスト(チェーンB、@hdl)。

生成Verilog ⇔ IR simulate(golden model)の一致をiverilogで検証する。
チェーンA(tests/test_ir_export.py)と合わせて
    PyTorch hard eval ⇔ IR simulate ⇔ Verilog
の3段等価性が閉じる。

iverilog非導入環境では自動skip。CIではhdlジョブでaptインストール。
実行: uv run pytest -m hdl
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.export.verify import iverilog_available, run_iverilog_equivalence

pytestmark = [
    pytest.mark.hdl,
    pytest.mark.skipif(not iverilog_available(), reason='iverilog未導入'),
]

# (parameterization, n_inputs, estimator)。estimatorはindependentのみ意味を持ち、
# 他は既定の'sigmoid'(=従来ケース。意味は変わらない)。
# 'sin'ケースは論文IWP(arXiv:2510.03250 §C.1.1)の推定器で、
# hard eval ⇔ IR simulate ⇔ Verilog の3段等価性が保たれることを検証する。
PARAMS_NS = [
    ('softmax', 2, 'sigmoid'), ('softmax', 4, 'sigmoid'),
    ('softmax', 6, 'sigmoid'),
    ('independent', 4, 'sigmoid'), ('iwp', 4, 'sigmoid'),
    ('warp', 6, 'sigmoid'),
    ('independent', 2, 'sin'), ('independent', 4, 'sin'),
    ('independent', 6, 'sin'),
]
IDS = [f'{p}_n{n}_{e}' for p, n, e in PARAMS_NS]


def _binary(batch, width, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, width), generator=g).numpy().astype(np.int64)


class TestVerilogEquivalence:
    @pytest.mark.parametrize('param,n,estimator', PARAMS_NS, ids=IDS)
    def test_256_random_vectors(self, param, n, estimator, tmp_path, seed):
        torch.manual_seed(41)
        kw = {} if estimator == 'sigmoid' else {'estimator': estimator}
        model = nn.Sequential(
            LutLayer(16, 24, n_inputs=n, parameterization=param, **kw),
            LutLayer(24, 24, n_inputs=n, parameterization=param, **kw),
            GroupSum(k=4, tau=2.0),
        )
        model.eval()
        ir = export_ir(model, input_width=16, name=f'{param}_n{n}_{estimator}')

        x = _binary(256, 16, seed=9)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], (
            f'{param} n={n} est={estimator}: '
            f'Verilogとgolden modelが不一致\n{result["stdout"]}'
        )

    def test_larger_model(self, tmp_path, seed):
        """幅広め・3層のモデルでも一致(生成コードのスケール確認)"""
        torch.manual_seed(43)
        model = nn.Sequential(
            LutLayer(32, 128, n_inputs=2),
            LutLayer(128, 128, n_inputs=2),
            LutLayer(128, 64, n_inputs=2),
            GroupSum(k=8, tau=4.0),
        )
        model.eval()
        ir = export_ir(model, input_width=32, name='wide3')
        x = _binary(128, 32, seed=13)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_lut_only_output(self, tmp_path, seed):
        """GroupSumなし(生ビット出力)でも一致"""
        torch.manual_seed(47)
        model = nn.Sequential(LutLayer(8, 16, n_inputs=2))
        model.eval()
        ir = export_ir(model, input_width=8, name='raw_bits')
        x = _binary(64, 8, seed=17)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_cnn_pipeline(self, tmp_path, seed):
        """CNN(conv unroll+ORプーリング+dense+GroupSum)のVerilog一致"""
        from griffith_logic import LutConv2d
        from griffith_logic.pooling import OrPooling2d

        torch.manual_seed(53)
        model = nn.Sequential(
            LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1),
            OrPooling2d(kernel_size=2),
            nn.Flatten(),
            LutLayer(4 * 4 * 4, 24, n_inputs=2),
            GroupSum(k=4, tau=2.0),
        )
        model.eval()
        ir = export_ir(model, input_shape=(1, 8, 8), name='cnn8')
        x = _binary(128, 64, seed=19)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_binarybrain_stack(self, tmp_path, seed):
        """BinaryBrainブロック(BN折り畳み)のVerilog一致"""
        from griffith_logic import DifferentiableLutBlock

        torch.manual_seed(59)
        model = nn.Sequential(
            DifferentiableLutBlock(32, 16, n_inputs=4),
            DifferentiableLutBlock(16, 8, n_inputs=4),
        )
        model.train()
        for _ in range(3):
            model(torch.randint(0, 2, (64, 32)).float())
        model.eval()
        ir = export_ir(model, input_width=32, name='bb_stack')
        x = _binary(128, 32, seed=23)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_conv_dlgn_pipeline(self, tmp_path, seed):
        """conv-DLGN(dilated8 サンプラ × sin 推定器 × rot2 頭部 × stride 割当)の
        Verilog 一致(conv は lowering 展開)"""
        from griffith_logic.models.conv_dlgn import ConvDLGNConfig, build_conv_dlgn

        torch.manual_seed(67)
        cfg = ConvDLGNConfig(in_shape=(1, 8, 8), channels=(4, 8), sampler='dilated8',
                             head_dims=(32, 16), num_classes=4, tau=2.0)
        model = build_conv_dlgn(cfg, generator=torch.Generator().manual_seed(67)).eval()
        ir = export_ir(model, input_shape=(1, 8, 8), name='conv_dlgn8')
        x = _binary(128, 64, seed=31)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_thermometer_lut8(self, tmp_path, seed):
        """ThermometerEncoder(input_bits=8)を 8 入力 LUT に焼いた Verilog 一致"""
        from griffith_logic.encoding.thermometer import ThermometerEncoder

        torch.manual_seed(73)
        enc = ThermometerEncoder(3, mode='uniform', input_bits=8)
        enc.fit(torch.rand(300, 2))
        model = nn.Sequential(enc, LutLayer(6, 16, n_inputs=2), GroupSum(k=4, tau=1.0))
        model.eval()
        ir = export_ir(model, input_width=16, name='thermo8')
        x = _binary(256, 16, seed=41)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_imported_opab_netlist(self, tmp_path, seed):
        """op/a/b ネットリストから変換した IR(stride 割当)の Verilog 一致"""
        from griffith_logic.ir.importers import opab_netlist_to_ir

        rng = np.random.default_rng(79)
        layers, prev = [], 16
        for w in (24, 24, 8):
            layers.append([{'op': int(rng.integers(0, 16)), 'a': int(rng.integers(0, prev)),
                            'b': int(rng.integers(0, prev))} for _ in range(w)])
            prev = w
        doc = {'input_bits': 16, 'layers': layers,
               'output': {'classes': 4, 'group_size': 2, 'tau': 1.0, 'assign': 'stride'}}
        ir = opab_netlist_to_ir(doc, name='opab_hdl')
        x = _binary(256, 16, seed=43)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_ltt_lut6(self, tmp_path, seed):
        """LTT(線形閾値ゲート)を 6 入力 LUT に焼いた Verilog 一致"""
        from griffith_logic.layers.ltt import LTTLayer

        torch.manual_seed(71)
        model = nn.Sequential(
            LTTLayer(16, 24, n_inputs=6),
            LTTLayer(24, 24, n_inputs=6),
            GroupSum(k=4, tau=2.0),
        )
        model.eval()
        ir = export_ir(model, input_width=16, name='ltt6')
        x = _binary(256, 16, seed=37)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']

    def test_group_sum_stride_assign(self, tmp_path, seed):
        """GroupSum(assign='stride')(恒等 lut_layer の転置 + group_sum)の Verilog 一致"""
        torch.manual_seed(61)
        model = nn.Sequential(
            LutLayer(16, 24, n_inputs=2),
            LutLayer(24, 24, n_inputs=2),
            GroupSum(k=4, tau=2.0, assign='stride'),
        )
        model.eval()
        ir = export_ir(model, input_width=16, name='gs_stride')
        x = _binary(256, 16, seed=29)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], result['stdout']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'hdl'])
