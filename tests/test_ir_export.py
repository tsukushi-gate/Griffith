"""IRエクスポートの等価性テスト(チェーンA)。

PyTorch hard eval ⇔ IR simulate の完全一致を全パラメータ化×Nで検証。
シミュレータ不要なので常時CIで実行される(3段等価性チェーンの前半)。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir, truth_tables_from_lut_layer
from griffith_logic.export.discretize import truth_tables_16gate_direct
from griffith_logic.ir import (
    IRValidationError,
    load_ir,
    save_ir,
    simulate,
    validate,
)

PARAMS_NS = [
    ('softmax', 2), ('softmax', 4), ('softmax', 6),
    ('independent', 2), ('independent', 4), ('independent', 6),
    ('iwp', 2), ('iwp', 4),
    ('warp', 2), ('warp', 4), ('warp', 6),
]
IDS = [f'{p}_n{n}' for p, n in PARAMS_NS]

TAU = 2.5


def _model(param, n, seed=31):
    torch.manual_seed(seed)
    return nn.Sequential(
        LutLayer(16, 24, n_inputs=n, parameterization=param),
        LutLayer(24, 24, n_inputs=n, parameterization=param),
        GroupSum(k=4, tau=TAU),
    )


def _binary_input(batch=32, width=16, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, width), generator=g).float()


class TestTorchVsSimulate:
    @pytest.mark.parametrize('param,n', PARAMS_NS, ids=IDS)
    def test_hard_eval_matches_simulate(self, param, n):
        model = _model(param, n)
        x = _binary_input()

        model.eval()
        with torch.no_grad():
            logits = model(x)  # (batch, 4) = 整数カウント / tau

        ir = export_ir(model, input_width=16, name=f'{param}_n{n}')
        assert validate(ir, target='hdl') is not None  # blackboxなし
        scores = simulate(ir, x.numpy().astype(np.int64))

        torch_counts = (logits * TAU).numpy()
        assert np.abs(torch_counts - scores).max() < 1e-3, (
            f'カウント不一致: max diff = {np.abs(torch_counts - scores).max()}'
        )
        assert (torch_counts.argmax(-1) == scores.argmax(-1)).all()

    def test_16gate_direct_matches_universal(self, seed):
        """argmax直接読みと万能離散化(evalプローブ)が一致"""
        layer = LutLayer(16, 32, n_inputs=2, parameterization='softmax')
        t_universal = truth_tables_from_lut_layer(layer)
        t_direct = truth_tables_16gate_direct(layer)
        assert (t_universal == t_direct).all()


class TestRoundtrip:
    def test_save_load_simulate_identical(self, tmp_path, seed):
        model = _model('softmax', 2)
        ir = export_ir(model, input_width=16)
        x = _binary_input().numpy().astype(np.int64)
        before = simulate(ir, x)

        path = save_ir(ir, str(tmp_path / 'model.json'))
        ir2 = load_ir(path)
        validate(ir2)
        after = simulate(ir2, x)
        assert (before == after).all()

    def test_sidecar_npz(self, tmp_path, seed):
        """大配列がnpzサイドカーに退避されても同一結果"""
        model = _model('softmax', 2)
        ir = export_ir(model, input_width=16)
        x = _binary_input().numpy().astype(np.int64)
        before = simulate(ir, x)

        path = save_ir(ir, str(tmp_path / 'model.json'), inline_threshold=4)
        assert os.path.exists(str(tmp_path / 'model.npz'))
        ir2 = load_ir(path)
        after = simulate(ir2, x)
        assert (before == after).all()


class TestValidation:
    def _valid_ir(self):
        return export_ir(_model('softmax', 2), input_width=16)

    def test_valid_graph_passes(self, seed):
        assert validate(self._valid_ir()) == []

    def test_width_mismatch_detected(self, seed):
        ir = self._valid_ir()
        ir.nodes[0].params['out_width'] = 23  # 24 → 23に破壊
        with pytest.raises(IRValidationError):
            validate(ir)

    def test_indices_out_of_range_detected(self, seed):
        ir = self._valid_ir()
        ir.nodes[0].params['indices'][0][0] = 999
        with pytest.raises(IRValidationError):
            validate(ir)

    def test_truth_table_range_detected(self, seed):
        ir = self._valid_ir()
        ir.nodes[0].params['truth_tables'][0] = hex(1 << 20)  # N=2は16未満
        with pytest.raises(IRValidationError):
            validate(ir)

    def test_groupsum_divisibility(self, seed):
        ir = self._valid_ir()
        ir.node_by_id([n.id for n in ir.nodes if n.type == 'group_sum'][0]) \
            .params['k'] = 5
        with pytest.raises(IRValidationError):
            validate(ir)

    def test_cycle_detected(self, seed):
        from griffith_logic.ir import IREdge
        ir = self._valid_ir()
        # n1 → n0 の逆エッジを追加して循環を作る
        ir.edges.append(IREdge(src=ir.nodes[1].id, dst=ir.nodes[0].id))
        with pytest.raises(IRValidationError):
            validate(ir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
