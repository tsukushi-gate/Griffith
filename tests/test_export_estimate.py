"""リソース見積り(export/estimate.py)のテスト。

既知構成(XORフィクスチャ・ダイヤモンドDAG・小CNN)の期待値と一致することを
検証する。conv2d/or_pool2dはlower_to_primitives展開後の実体で計測される。
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutConv2d, LutLayer
from griffith_logic.export import estimate_resources, export_ir
from griffith_logic.ir import ir_from_dict
from griffith_logic.pooling import OrPooling2d

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'ir_v01')


def _load_fixture_ir(name: str):
    with open(os.path.join(FIXTURE_DIR, f'{name}.json')) as f:
        return ir_from_dict(json.load(f)['ir'])


def _lut_node(node_id, in_width, out_width, n_inputs, rng):
    indices = [
        [int(rng.integers(0, in_width)) for _ in range(out_width)]
        for _ in range(n_inputs)
    ]
    tts = [hex(int(rng.integers(0, 1 << (1 << n_inputs))))
           for _ in range(out_width)]
    return {
        'id': node_id, 'type': 'lut_layer',
        'params': {'in_width': in_width, 'out_width': out_width,
                   'n_inputs': n_inputs, 'indices': indices,
                   'truth_tables': tts},
        'train_info': {'parameterization': 'independent'}, 'meta': {},
    }


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


class TestEstimateKnownConfigs:
    def test_xor_fixture(self):
        """XOR = LUT1個(N=2)・深さ1・ワイヤ3本(入力2+出力1)"""
        est = estimate_resources(_load_fixture_ir('xor'))
        assert est['total_luts'] == 1
        assert est['lut_counts'] == {2: 1}
        assert est['logic_depth'] == 1
        assert est['wire_count'] == 3
        assert est['group_sums'] == 0
        assert est['lowered'] is False

    def test_parity3_fixture(self):
        """parity3 = 2層(2+1=3 LUT)・深さ2"""
        est = estimate_resources(_load_fixture_ir('parity3'))
        assert est['total_luts'] == 3
        assert est['lut_counts'] == {2: 3}
        assert est['logic_depth'] == 2
        assert est['wire_count'] == 3 + 2 + 1

    def test_demo_mnist_fixture(self):
        """demo_mnist = 784→400→400+group_sum(k=10)"""
        est = estimate_resources(_load_fixture_ir('demo_mnist'))
        assert est['total_luts'] == 800
        assert est['logic_depth'] == 2
        assert est['group_sums'] == 1
        assert est['wire_count'] == 784 + 400 + 400 + 10

    def test_diamond_dag_longest_path(self):
        """ダイヤモンドDAG: 最長路の深さ3(concatは0段)・fanoutは加算しない"""
        rng = np.random.default_rng(11)
        doc = {
            'format': 'magicdelta-ir', 'version': '0.2.0',
            'metadata': {'name': 'diamond'},
            'inputs': [_port('x', 8)], 'outputs': [_port('y', 8)],
            'nodes': [
                _lut_node('n1', 8, 8, 2, rng),
                _lut_node('n2', 8, 6, 2, rng),
                _lut_node('n3', 8, 10, 2, rng),
                {'id': 'n4', 'type': 'concat',
                 'params': {'in_widths': [6, 10]},
                 'train_info': {}, 'meta': {}},
                _lut_node('n5', 16, 8, 2, rng),
            ],
            'edges': [
                {'src': 'x', 'dst': 'n1'},
                {'src': 'n1', 'dst': 'n2'},
                {'src': 'n1', 'dst': 'n3'},
                {'src': 'n2', 'dst': 'n4'},
                {'src': 'n3', 'dst': 'n4', 'dst_slot': 1},
                {'src': 'n4', 'dst': 'n5'},
                {'src': 'n5', 'dst': 'y'},
            ],
        }
        est = estimate_resources(ir_from_dict(doc))
        assert est['total_luts'] == 8 + 6 + 10 + 8
        assert est['logic_depth'] == 3
        assert est['wire_count'] == 8 + (8 + 6 + 10 + 16 + 8)
        assert est['node_counts'] == {'lut_layer': 4, 'concat': 1}


class TestEstimateConvLowering:
    def test_conv_native_ir_is_lowered_before_count(self, seed):
        """conv2d/or_pool2dは展開後のlut_layer実体で計測される"""
        torch.manual_seed(31)
        model = nn.Sequential(
            LutConv2d(1, 4, kernel_size=3, tree_depth=2),  # 8x8→6x6, L=36
            OrPooling2d(kernel_size=2),                    # →3x3
            nn.Flatten(),
            LutLayer(36, 24, n_inputs=2),
            GroupSum(k=4, tau=1.0),
        )
        model.eval()
        ir = export_ir(model, input_shape=(1, 8, 8), native_conv=True)
        est = estimate_resources(ir)
        assert est['lowered'] is True
        # conv lv0=4*36*2, lv1=4*36*1, dense=24 (N=2) / pool窓=36 (N=4)
        assert est['lut_counts'] == {2: 288 + 144 + 24, 4: 36}
        assert est['total_luts'] == 492
        assert est['logic_depth'] == 4  # lv0→lv1→pool→dense
        assert est['wire_count'] == 64 + 288 + 144 + 36 + 24 + 4
        assert est['group_sums'] == 1
        # 従来lowering経路のIRと見積りが一致する(lowered以外)
        est_legacy = estimate_resources(
            export_ir(model, input_shape=(1, 8, 8)))
        for key in ('total_luts', 'lut_counts', 'logic_depth',
                    'wire_count', 'group_sums', 'node_counts'):
            assert est[key] == est_legacy[key], key


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
