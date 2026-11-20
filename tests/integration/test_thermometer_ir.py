"""統合: ThermometerEncoder の IR 化。

input_bits=B を指定した ThermometerEncoder は B 入力 lut_layer(特徴ごとに B ビット符号 c、
実数 q = c/(2^B−1))になり、torch の forward(実数 q)と simulate(符号ビット)が一致する。
uniform(未 fit / fit 済み)・distributive(fit 済み)、DWNClassifier の論理部分まで通し、
input_bits 未指定は従来どおり境界(NotImplementedError)。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from griffith_logic import GroupSum, LutLayer
from griffith_logic.encoding.thermometer import ThermometerEncoder
from griffith_logic.export.extract import export_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate
from griffith_logic.models.dwn import DWNClassifier

B = 8
D = 3


def _codes(n, seed=0):
    return np.random.default_rng(seed).integers(0, 1 << B, size=(n, D)).astype(np.int64)


def _code_bits(codes):
    """特徴 d の B ビット符号を LSB 先頭で位置 d·B + j に並べる"""
    bits = ((codes[:, :, None] >> np.arange(B)[None, None, :]) & 1)
    return bits.reshape(codes.shape[0], D * B).astype(np.int64)


def _q(codes):
    return torch.tensor(codes, dtype=torch.float32) / float((1 << B) - 1)


def _model(encoder, seed=11):
    torch.manual_seed(seed)
    return nn.Sequential(encoder, LutLayer(D * encoder.bits, 8, n_inputs=2),
                         GroupSum(k=2, tau=1.0)).eval()


def _hard(model, q):
    with torch.no_grad():
        return np.round(model(q).numpy()).astype(np.int64)


class TestThermometerIr:
    def test_unfit_uniform(self):
        enc = ThermometerEncoder(3, input_bits=B)
        model = _model(enc)
        ir = export_ir(model, input_width=D * B)
        node = ir.nodes[0]
        assert node.type == 'lut_layer'
        assert node.params['n_inputs'] == B
        assert node.params['out_width'] == D * 3
        assert np.asarray(node.params['indices']).shape == (B, D * 3)
        assert node.train_info['kind'] == 'thermometer'
        validate(ir, target='hdl')
        codes = _codes(200)
        assert np.array_equal(simulate(ir, _code_bits(codes)), _hard(model, _q(codes)))

    def test_encoder_alone_matches_bitwise(self):
        """符号器単体: 全 256 符号 × 3 特徴で出力ビットが forward と一致(境界 q == t も含む)"""
        enc = ThermometerEncoder(4, input_bits=B)
        enc.fit(torch.rand(500, D))
        ir = export_ir(nn.Sequential(enc), input_width=D * B)
        codes = np.stack([np.arange(256)] * D, axis=1)
        with torch.no_grad():
            want = enc(_q(codes)).numpy().astype(np.int64)
        assert np.array_equal(simulate(ir, _code_bits(codes)), want)

    @pytest.mark.parametrize('mode', ['uniform', 'distributive'])
    def test_fitted(self, mode):
        enc = ThermometerEncoder(3, mode=mode, input_bits=B)
        enc.fit(torch.rand(400, D))
        model = _model(enc)
        ir = export_ir(model, input_width=D * B)
        codes = _codes(300, seed=2)
        assert np.array_equal(simulate(ir, _code_bits(codes)), _hard(model, _q(codes)))

    def test_dwn_classifier_logic_part(self):
        torch.manual_seed(5)
        model = DWNClassifier(D, 2, layer_widths=(16, 8), bits=3, encoder_mode='uniform',
                              n_inputs=6, mapping='random').eval()
        model.encoder.input_bits = B
        logic = nn.Sequential(model.encoder, *model.layers, model.group_sum)
        ir = export_ir(logic, input_width=D * B)
        assert [n.type for n in ir.nodes] == ['lut_layer', 'lut_layer', 'lut_layer', 'group_sum']
        codes = _codes(128, seed=3)
        with torch.no_grad():
            want = np.round(model(_q(codes)).numpy() * model.group_sum.tau).astype(np.int64)
        assert np.array_equal(simulate(ir, _code_bits(codes)), want)

    def test_without_input_bits_is_boundary(self):
        enc = ThermometerEncoder(3)
        assert 'input_bits' not in enc.extra_repr()
        with pytest.raises(NotImplementedError):
            export_ir(nn.Sequential(enc), input_width=D * B)

    def test_width_must_be_multiple_of_input_bits(self):
        enc = ThermometerEncoder(3, input_bits=B)
        with pytest.raises(ValueError):
            export_ir(nn.Sequential(enc), input_width=D * B + 1)

    def test_distributive_requires_fit(self):
        enc = ThermometerEncoder(3, mode='distributive', input_bits=B)
        with pytest.raises(RuntimeError):
            export_ir(nn.Sequential(enc), input_width=D * B)

    def test_input_bits_validation(self):
        with pytest.raises(ValueError):
            ThermometerEncoder(3, input_bits=0)
        assert 'input_bits=8' in ThermometerEncoder(3, input_bits=8).extra_repr()
