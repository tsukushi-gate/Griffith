"""conv/or_poolネイティブIRノードのテスト。

小CNN(LutConv2d+OrPooling2d+Flatten+LutLayer+GroupSum)について:
    (a) native抽出(export_ir(native_conv=True))→IR→build_torch往復の
        hard eval完全一致
    (b) simulate == build hard eval(convフォールバック=lowering展開実行)
    (c) lower_to_primitives→to_verilog/to_vhdlの文字列妥当+
        未lowering時の明示エラー+新旧経路のVerilogバイト一致
        (ローカルiverilogがあれば3段等価性、@hdl)
    (d) 既定(native_conv=False)は従来lowering経路のまま(後方互換)
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import GroupSum, LutConv2d, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.export.lowering import lower_to_primitives
from griffith_logic.export.verify import (
    iverilog_available,
    run_iverilog_equivalence,
)
from griffith_logic.export.verilog import to_verilog
from griffith_logic.export.vhdl import to_vhdl
from griffith_logic.ir import (
    IRValidationError,
    build_torch,
    ir_from_dict,
    ir_to_dict,
    simulate,
    simulate_trace,
    validate,
)
from griffith_logic.pooling import OrPooling2d

C_IN, H_IN, W_IN = 1, 8, 8
POOLED = 4 * 3 * 3  # 8x8 → conv3 → 6x6 → pool2 → 3x3, out_channels=4


def _binary_images(batch=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, C_IN, H_IN, W_IN), generator=g).float()


def _small_cnn(param='softmax', padding=0):
    torch.manual_seed(31)
    conv_kw = {}
    if param == 'iwp':
        conv_kw['residual_init'] = False
    spatial = 3 if padding == 0 else 4  # padding=1: conv後8x8 → pool2 → 4x4
    model = nn.Sequential(
        LutConv2d(C_IN, 4, kernel_size=3, tree_depth=2, padding=padding,
                  parameterization=param, **conv_kw),
        OrPooling2d(kernel_size=2),
        nn.Flatten(),
        LutLayer(4 * spatial * spatial, 24, n_inputs=2),
        GroupSum(k=4, tau=1.0),
    )
    model.eval()
    return model


def _export_native(model, name='cnn_native'):
    return export_ir(model, input_shape=(C_IN, H_IN, W_IN), name=name,
                     native_conv=True)


class TestConvNativeSchema:
    def test_native_export_node_types(self, seed):
        """native抽出はconv2d/or_pool2dノードを生成しvalidateを通る"""
        ir = _export_native(_small_cnn())
        assert [n.type for n in ir.nodes] == \
            ['conv2d', 'or_pool2d', 'lut_layer', 'group_sum']
        assert validate(ir) == []
        conv = ir.nodes[0]
        assert conv.params['in_shape'] == [C_IN, H_IN, W_IN]
        assert conv.params['out_channels'] == 4
        # 幅規約: width = C*H*W flatten
        assert ir.inputs[0].width == C_IN * H_IN * W_IN

    def test_invalid_indices_flat_shape(self, seed):
        ir = _export_native(_small_cnn())
        conv = ir.nodes[0]
        conv.params['indices_flat'] = np.asarray(
            conv.params['indices_flat'])[:, :-1]
        with pytest.raises(IRValidationError, match='indices_flat形状'):
            validate(ir)

    def test_invalid_truth_tables_count(self, seed):
        ir = _export_native(_small_cnn())
        conv = ir.nodes[0]
        conv.params['truth_tables'] = conv.params['truth_tables'][:-1]
        with pytest.raises(IRValidationError, match='truth_tables数'):
            validate(ir)

    def test_shape_propagation_width_mismatch(self, seed):
        """in_shapeを変えると入力幅整合(shape伝播)で検出される"""
        ir = _export_native(_small_cnn())
        ir.nodes[0].params['in_shape'] = [C_IN, H_IN, W_IN + 1]
        with pytest.raises(IRValidationError, match='幅不整合'):
            validate(ir)

    def test_or_pool_window_too_large(self, seed):
        ir = _export_native(_small_cnn())
        ir.nodes[1].params['kernel_size'] = 3
        with pytest.raises(IRValidationError, match='窓3x3'):
            validate(ir)


class TestConvNativeRoundtrip:
    @pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp'])
    def test_build_roundtrip_hard_eval(self, param, seed):
        """(a) native抽出→IR→build往復のhard eval完全一致"""
        model = _small_cnn(param)
        x4 = _binary_images()
        ir = _export_native(model)
        built = build_torch(ir)
        built.eval()
        with torch.no_grad():
            y_orig = model(x4).numpy()
            y_built = built(x4.flatten(1)).numpy()
        assert np.array_equal(y_orig, y_built), (
            f'max diff = {np.abs(y_orig - y_built).max()}'
        )

    def test_build_roundtrip_with_padding(self, seed):
        """padding=1でもnative往復がビット厳密(真理値表特殊化なしの生params)"""
        model = _small_cnn(padding=1)
        x4 = _binary_images()
        ir = _export_native(model)
        built = build_torch(ir)
        built.eval()
        with torch.no_grad():
            y_orig = model(x4).numpy()
            y_built = built(x4.flatten(1)).numpy()
        assert np.array_equal(y_orig, y_built)

    def test_dict_roundtrip_simulate_stable(self, seed):
        """ir_to_dict→ir_from_dict往復後もsimulate結果が不変"""
        model = _small_cnn()
        x = _binary_images().flatten(1).numpy().astype(np.int64)
        ir = _export_native(model)
        ir2 = ir_from_dict(ir_to_dict(ir))
        validate(ir2)
        assert np.array_equal(simulate(ir, x), simulate(ir2, x))


class TestConvNativeSimulate:
    def test_simulate_matches_hard_eval(self, seed):
        """(b) simulate(convフォールバック) == build hard eval"""
        model = _small_cnn()
        x4 = _binary_images(32)
        ir = _export_native(model)
        with torch.no_grad():
            y_torch = model(x4).numpy()
        scores = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.array_equal(y_torch, scores)

    def test_simulate_with_padding_and_stride(self, seed):
        torch.manual_seed(17)
        model = nn.Sequential(
            LutConv2d(C_IN, 3, kernel_size=3, tree_depth=3, stride=2,
                      padding=1),
        )
        model.eval()
        x4 = _binary_images()
        ir = _export_native(model)
        with torch.no_grad():
            y_torch = model(x4).flatten(1).numpy()
        y_sim = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.array_equal(y_torch, y_sim)

    def test_simulate_trace_aggregates_to_original_ids(self, seed):
        """traceのキーは元ノードid(展開中間ノードは含まれない)"""
        model = _small_cnn()
        x4 = _binary_images()
        ir = _export_native(model)
        trace = simulate_trace(ir, x4.flatten(1).numpy().astype(np.int64))
        assert set(trace) == {'x'} | {n.id for n in ir.nodes}
        # convノードの値 = torch convの出力(NCHW flatten)
        with torch.no_grad():
            conv_out = model[0](x4).flatten(1).numpy()
            pool_out = model[1](model[0](x4)).flatten(1).numpy()
        assert np.array_equal(trace['n0_lutconv2d'], conv_out)
        assert np.array_equal(trace['n1_orpooling2d'], pool_out)


class TestConvNativeLowering:
    def test_lower_then_verilog_valid(self, seed):
        """(c) lower_to_primitives→to_verilogが文字列妥当"""
        ir = _export_native(_small_cnn())
        low = lower_to_primitives(ir)
        assert all(n.type != 'conv2d' and n.type != 'or_pool2d'
                   for n in low.nodes)
        validate(low, target='hdl')
        src = to_verilog(low)
        assert 'module logic_core' in src
        assert 'w_n0_lutconv2d_lv0' in src
        assert 'w_n1_orpooling2d' in src

    def test_verilog_without_lowering_raises(self, seed):
        ir = _export_native(_small_cnn())
        with pytest.raises(NotImplementedError, match='lower_to_primitives'):
            to_verilog(ir)
        with pytest.raises(NotImplementedError, match='lower_to_primitives'):
            to_vhdl(ir)

    def test_lowered_verilog_matches_legacy_path(self, seed):
        """新(native→lower)旧(直接lowering)経路のVerilog出力バイト一致"""
        model = _small_cnn()
        ir_native = _export_native(model, name='cnn')
        ir_legacy = export_ir(model, input_shape=(C_IN, H_IN, W_IN),
                              name='cnn')
        assert to_verilog(lower_to_primitives(ir_native)) == \
            to_verilog(ir_legacy)
        assert to_vhdl(lower_to_primitives(ir_native)) == to_vhdl(ir_legacy)

    def test_lower_is_pure(self, seed):
        """lower_to_primitivesは入力IRを変更しない純関数"""
        ir = _export_native(_small_cnn())
        types_before = [n.type for n in ir.nodes]
        n_edges = len(ir.edges)
        lower_to_primitives(ir)
        assert [n.type for n in ir.nodes] == types_before
        assert len(ir.edges) == n_edges

    def test_lower_identity_without_conv(self, seed):
        """展開対象が無いIRはそのまま返す"""
        torch.manual_seed(7)
        model = nn.Sequential(LutLayer(8, 8, n_inputs=2), GroupSum(k=4))
        ir = export_ir(model, input_width=8)
        assert lower_to_primitives(ir) is ir

    def test_default_export_unchanged(self, seed):
        """(d) 既定(native_conv=False)は従来どおりlut_layer列へ展開"""
        model = _small_cnn()
        ir = export_ir(model, input_shape=(C_IN, H_IN, W_IN))
        assert all(n.type in ('lut_layer', 'group_sum') for n in ir.nodes)
        assert any(n.train_info.get('lowered_from') == 'LutConv2d'
                   for n in ir.nodes)


@pytest.mark.hdl
@pytest.mark.skipif(not iverilog_available(), reason='iverilog未導入')
class TestConvNativeHdlEquivalence:
    def test_three_way_equivalence(self, seed, tmp_path):
        """(c) native IRの3段等価性: hard eval ⇔ simulate ⇔ iverilog"""
        model = _small_cnn()
        x4 = _binary_images(32)
        ir = _export_native(model)
        result = run_iverilog_equivalence(
            ir, x4.flatten(1).numpy().astype(np.int64), str(tmp_path),
        )
        assert result['passed'], result['stdout']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
