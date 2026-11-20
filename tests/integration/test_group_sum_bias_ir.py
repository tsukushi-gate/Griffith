"""統合: GroupSum の判定バイアスの IR 化。

- 抽出: 非零の bias だけ group_sum.params.bias(整数)に出る。全 0 / 未設定の IR は従来と同一
- simulate がバイアスを足す(hard eval × τ と一致)、build_torch でバイアスが戻る
- validate(target='hdl') / to_vhdl は bias ≠ 0 を拒む(HDL 未対応を黙って落とさない)、
  generic の validate は長さ・整数性を検査する
- JSON 往復で bias が保たれる
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export.extract import export_ir
from griffith_logic.export.vhdl import to_vhdl
from griffith_logic.ir.build import build_torch
from griffith_logic.ir.serialize import ir_from_dict, ir_to_dict
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import IRValidationError, validate

IN, TAU = 16, 2.0


def _model(bias=None, seed=5):
    torch.manual_seed(seed)
    return nn.Sequential(LutLayer(IN, 24, n_inputs=2), LutLayer(24, 24, n_inputs=2),
                         GroupSum(k=4, tau=TAU, bias=bias)).eval()


def _x(n=64, seed=1):
    return np.random.default_rng(seed).integers(0, 2, size=(n, IN)).astype(np.int64)


def _scores(model, x):
    with torch.no_grad():
        return np.round(model(torch.tensor(x, dtype=torch.float32)).numpy() * TAU).astype(np.int64)


class TestBiasIr:
    def test_extract_and_simulate(self):
        model = _model(bias=[3, 0, -2, 0])
        ir = export_ir(model, input_width=IN)
        assert ir.nodes[-1].params['bias'] == [3, 0, -2, 0]
        validate(ir)
        x = _x()
        assert np.array_equal(simulate(ir, x), _scores(model, x))
        # バイアス無しの IR とはスコアが bias ぶんだけ違う
        plain = export_ir(_model(), input_width=IN)
        assert np.array_equal(simulate(ir, x) - simulate(plain, x),
                              np.tile([3, 0, -2, 0], (x.shape[0], 1)))

    def test_zero_bias_is_omitted(self):
        ir = export_ir(_model(bias=[0, 0, 0, 0]), input_width=IN)
        assert 'bias' not in ir.nodes[-1].params
        assert 'bias' not in export_ir(_model(), input_width=IN).nodes[-1].params

    def test_json_roundtrip_and_build_torch(self):
        model = _model(bias=[1, 0, 0, -1])
        ir = ir_from_dict(ir_to_dict(export_ir(model, input_width=IN)))
        assert list(ir.nodes[-1].params['bias']) == [1, 0, 0, -1]
        rebuilt = build_torch(ir).eval()
        gs = [m for m in rebuilt.modules() if isinstance(m, GroupSum)][0]
        assert gs.bias.tolist() == [1.0, 0.0, 0.0, -1.0]
        x = _x(seed=2)
        assert np.array_equal(_scores(rebuilt, x), simulate(ir, x))

    def test_hdl_rejected_until_supported(self):
        ir = export_ir(_model(bias=[2, 0, 0, 0]), input_width=IN)
        with pytest.raises(IRValidationError, match='bias'):
            validate(ir, target='hdl')
        with pytest.raises(ValueError, match='bias'):
            to_vhdl(ir)
        validate(export_ir(_model(), input_width=IN), target='hdl')      # 無しは従来どおり

    def test_generic_validate_checks_bias(self):
        ir = export_ir(_model(bias=[2, 0, 0, 0]), input_width=IN)
        ir.nodes[-1].params['bias'] = [1, 2]
        with pytest.raises(IRValidationError, match='bias'):
            validate(ir)
        ir.nodes[-1].params['bias'] = [1, 0.5, 0, 0]
        with pytest.raises(IRValidationError, match='bias'):
            validate(ir)

    def test_non_integer_bias_cannot_be_exported(self):
        with pytest.raises(ValueError, match='整数'):
            export_ir(_model(bias=[0.5, 0, 0, 0]), input_width=IN)
