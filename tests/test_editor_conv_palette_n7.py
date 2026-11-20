"""エディタのconv2d/or_pool2dパレット解禁+初期化プリセットの検証。

フロント(tools/editor/frontend/src/blocks/nodeInit.ts)がパレットから生成する
conv2d/or_pool2d/lut_layerノードのparams生成規約を Python 側で同型に再現し、
    - validate() 通過
    - build_torch() でモデル構築(学習可能)
    - pass-through プリセットの lut_layer が恒等 simulate
を保証する(生成物がバックエンドの build/validate/simulate を通る=北極星)。

本テストは griffith_logic 本体を一切変更せず、読み取り(validate/build/simulate)
のみで確認する。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic.ir import (
    build_torch,
    ir_from_dict,
    simulate,
    validate,
)
from griffith_logic.ir.validate import conv2d_out_shape, or_pool2d_out_shape


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


def _pass_through_tt(n_inputs: int) -> str:
    """nodeInit.passThroughTT と同一: '0x' + 'a'×(2^n/4)(slot0素通し)。"""
    return '0x' + 'a' * ((1 << n_inputs) // 4)


def _random_tt(n_inputs: int, rng) -> str:
    nibbles = (1 << n_inputs) // 4
    return '0x' + ''.join(f'{int(rng.integers(0, 16)):x}' for _ in range(nibbles))


def _conv2d_node(node_id, in_shape, out_channels, kernel, stride, padding,
                 tree_depth, n_inputs, preset='random', seed=3):
    """nodeInit.conv2dInit と同型のparams生成。"""
    rng = np.random.default_rng(seed)
    c, _, _ = in_shape
    rf = c * kernel * kernel
    n_leaves = n_inputs ** tree_depth
    n_internal = tree_depth if n_inputs == 1 else (n_leaves - 1) // (n_inputs - 1)

    indices_flat = [
        [int(rng.integers(0, rf)) for _ in range(n_leaves)]
        for _ in range(out_channels)
    ]
    tts = []
    for _ in range(out_channels * n_internal):
        if preset == 'random':
            tts.append(_random_tt(n_inputs, rng))
        else:
            tts.append(_pass_through_tt(n_inputs))
    return {
        'id': node_id,
        'type': 'conv2d',
        'params': {
            'in_shape': list(in_shape),
            'out_channels': out_channels,
            'kernel_size': [kernel, kernel],
            'stride': [stride, stride],
            'padding': [padding, padding],
            'tree_depth': tree_depth,
            'n_inputs': n_inputs,
            'indices_flat': indices_flat,
            'truth_tables': tts,
        },
        'train_info': {'parameterization': 'independent'},
        'meta': {},
    }


def _or_pool_node(node_id, in_shape, kernel, stride, padding):
    return {
        'id': node_id,
        'type': 'or_pool2d',
        'params': {
            'in_shape': list(in_shape),
            'kernel_size': kernel,
            'stride': stride,
            'padding': padding,
        },
        'train_info': {},
        'meta': {},
    }


def _cnn_doc(preset='random'):
    """x(1*8*8) → conv2d(→4ch) → or_pool2d(2) → y のネイティブconv IR。"""
    in_shape = [1, 8, 8]
    conv = _conv2d_node('n0_conv', in_shape, out_channels=4, kernel=3,
                        stride=1, padding=0, tree_depth=2, n_inputs=2,
                        preset=preset)
    conv_out = conv2d_out_shape(conv['params'])  # (4,6,6)
    pool_in = list(conv_out)
    pool = _or_pool_node('n1_pool', pool_in, kernel=2, stride=2, padding=0)
    pool_out = or_pool2d_out_shape(pool['params'])  # (4,3,3)
    out_width = pool_out[0] * pool_out[1] * pool_out[2]
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': 'n7_cnn'},
        'inputs': [_port('x', in_shape[0] * in_shape[1] * in_shape[2])],
        'outputs': [_port('y', out_width)],
        'nodes': [conv, pool],
        'edges': [
            {'src': 'x', 'dst': 'n0_conv'},
            {'src': 'n0_conv', 'dst': 'n1_pool'},
            {'src': 'n1_pool', 'dst': 'y'},
        ],
    }


class TestConvPaletteGeneration:
    def test_random_conv_validates_and_builds(self):
        """random初期化のconv2d+or_pool2d IRがvalidate通過+build可能。"""
        graph = ir_from_dict(_cnn_doc('random'))
        assert validate(graph) == []
        core = build_torch(graph)
        assert core is not None

    def test_passthrough_conv_validates_and_builds(self):
        graph = ir_from_dict(_cnn_doc('pass-through'))
        assert validate(graph) == []
        assert build_torch(graph) is not None

    def test_conv_forward_finite(self):
        """build後のモデルが有限出力を返す(学習疎通の前提)。"""
        graph = ir_from_dict(_cnn_doc('random'))
        core = build_torch(graph)
        model = torch.nn.Sequential(core) if not isinstance(core, list) \
            else torch.nn.Sequential(*core)
        model.eval()
        x = torch.randint(0, 2, (5, 1 * 8 * 8)).float()
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all()


class TestPassThroughIdentity:
    def _identity_lut_doc(self, width=6):
        """pass-throughプリセット(恒等indices+0xa tt)のlut_layer単体グラフ。"""
        idx = list(range(width))
        node = {
            'id': 'n0_lut',
            'type': 'lut_layer',
            'params': {
                'in_width': width,
                'out_width': width,
                'n_inputs': 2,
                'indices': [idx, list(idx)],
                'truth_tables': [_pass_through_tt(2)] * width,
            },
            'train_info': {'parameterization': 'softmax', 'initialized': True},
            'meta': {},
        }
        return {
            'format': 'magicdelta-ir',
            'version': '0.2.0',
            'metadata': {'name': 'passthrough'},
            'inputs': [_port('x', width)],
            'outputs': [_port('y', width)],
            'nodes': [node],
            'edges': [
                {'src': 'x', 'dst': 'n0_lut'},
                {'src': 'n0_lut', 'dst': 'y'},
            ],
        }

    def test_passthrough_lut_is_identity(self):
        """pass-through lut_layer の simulate は入力の恒等写像。"""
        width = 6
        graph = ir_from_dict(self._identity_lut_doc(width))
        assert validate(graph) == []
        rng = np.random.default_rng(11)
        x = rng.integers(0, 2, size=(8, width)).astype(np.int64)
        y = simulate(graph, x)
        assert np.array_equal(np.asarray(y).astype(np.int64), x)

    def test_passthrough_tt_matches_nodeInit(self):
        """passThroughTT規約(0xa=slot0素通し)の確認。"""
        assert _pass_through_tt(2) == '0xa'
        assert _pass_through_tt(4) == '0xaaaa'
