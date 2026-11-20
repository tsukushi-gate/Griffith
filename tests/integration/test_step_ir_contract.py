"""統合: 1 ステップ IR の契約(export/step_contract.py)。

任意の step IR を受けて docs/reference/step-ir-contract.md の規則を検査する。
RDDLGN decoder の step IR(T4c)は PASS、group_sum 終端・recurrent 入り・外部入力 0 bit は FAIL。
lab 側のモデル(文字 LM / 制御器 / 収束型)の exporter はこの検査を import して通す。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export.extract import export_ir
from griffith_logic.export.rddlgn_ir import export_rddlgn_decoder_step_ir
from griffith_logic.export.step_contract import (
    StepIRContractError,
    check_step_ir,
    step_ir_manifest,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort
from griffith_logic.ir.simulate import simulate
from griffith_logic.models.rddlgn import RDDLGN, RDDLGNConfig


def _lut_step_ir(in_width=12, out_width=10, seed=0):
    """lut_layer 2 段の組み合わせ IR(状態幅は呼び出し側が決める)"""
    torch.manual_seed(seed)
    model = nn.Sequential(LutLayer(in_width, 16, n_inputs=2), LutLayer(16, out_width, n_inputs=2))
    return export_ir(model.eval(), input_width=in_width)


class TestContract:
    def test_rddlgn_decoder_step_passes(self):
        torch.manual_seed(0)
        model = RDDLGN(RDDLGNConfig.tiny()).eval()
        graph, spec = export_rddlgn_decoder_step_ir(model)
        cs = check_step_ir(graph, spec['state_width'])
        assert cs.ext_width == spec['ext_width']
        assert cs.readout_width == spec['readout_width']

    def test_plain_lut_step_passes_and_feedback_runs(self):
        graph = _lut_step_ir(in_width=12, out_width=10)
        cs = check_step_ir(graph, state_width=4)
        assert cs.state_in == (8, 12)
        assert cs.state_out == (0, 4)
        # ラッパの意味論: x = [外部 | 状態]、状態 ← y[state_out]
        state = np.zeros(4, dtype=np.int64)
        for t in range(3):
            ext = np.full(8, t % 2, dtype=np.int64)
            y = simulate(graph, np.concatenate([ext, state])[None, :])[0]
            state = y[cs.state_out[0]:cs.state_out[1]]
        assert state.shape == (4,)

    def test_group_sum_terminal_fails(self):
        torch.manual_seed(1)
        model = nn.Sequential(LutLayer(12, 16, n_inputs=2), GroupSum(k=4)).eval()
        graph = export_ir(model, input_width=12)
        with pytest.raises(StepIRContractError, match='group_sum'):
            check_step_ir(graph, state_width=2)

    def test_recurrent_node_fails(self):
        body = IRGraph(nodes=[IRNode(id='sv', type='scan_var', params={'width': 2, 'role': 'step'})],
                       edges=[])
        rec = IRNode(id='rec', type='recurrent', params={
            'steps': 2, 'body': body,
            'scan_vars': [{'id': 'sv', 'role': 'step', 'width': 2}],
            'state_update': {}, 'output_mode': 'stacked',
            'output_source_id': 'sv', 'output_width': 2})
        graph = IRGraph(inputs=[IRPort(id='x', width=4)], outputs=[IRPort(id='y', width=4)],
                        nodes=[rec], edges=[IREdge(src='x', dst='rec', dst_slot=0),
                                            IREdge(src='rec', dst='y')])
        with pytest.raises(StepIRContractError):
            check_step_ir(graph, state_width=1)

    def test_state_width_bounds(self):
        graph = _lut_step_ir(in_width=12, out_width=10)
        with pytest.raises(StepIRContractError, match='外部入力'):
            check_step_ir(graph, state_width=12)
        with pytest.raises(StepIRContractError, match='出力幅'):
            check_step_ir(graph, state_width=11)
        with pytest.raises(StepIRContractError):
            check_step_ir(graph, state_width=0)

    def test_two_output_ports_fail(self):
        graph = _lut_step_ir()
        graph.outputs.append(IRPort(id='y2', width=1))
        with pytest.raises(StepIRContractError, match='ポート'):
            check_step_ir(graph, state_width=2)

    def test_manifest(self):
        cs = check_step_ir(_lut_step_ir(in_width=12, out_width=10), state_width=4)
        m = step_ir_manifest(cs, [1, 0, 1, 1], steps=8)
        assert m == {'state_in': [8, 12], 'state_out': [0, 4], 'init': '0xd', 'steps': 8}
        assert 'steps' not in step_ir_manifest(cs, [0, 0, 0, 0])
        with pytest.raises(StepIRContractError):
            step_ir_manifest(cs, [1, 0, 1])
        with pytest.raises(StepIRContractError):
            step_ir_manifest(cs, [1, 0, 1, 1], steps=0)
