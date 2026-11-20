"""Conv系・BinaryBrain系のIR lowering等価性テスト。

LutConv2d(木構造unroll)/ OrPooling2d / SpatialGroupSum /
DifferentiableLutBlock(BN折り畳みプローブ)について
「PyTorch hard eval ⇔ IR simulate」の厳密一致を検証する。
padding>0 は真理値表特殊化(範囲外スロットの0固定畳み込み)の検証を兼ねる。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import (
    DifferentiableLutBlock,
    GroupSum,
    LutConv2d,
    LutLayer,
    SpatialGroupSum,
)
from griffith_logic.export import export_ir
from griffith_logic.ir import simulate, validate
from griffith_logic.pooling import OrPooling2d

C_IN, H_IN, W_IN = 1, 8, 8


def _binary_images(batch=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, C_IN, H_IN, W_IN), generator=g).float()


def _assert_equiv_4d(model, x4, atol=1e-4):
    """model.eval()の4D出力とIR simulateのflat出力の厳密一致"""
    model.eval()
    with torch.no_grad():
        y_torch = model(x4).flatten(1).numpy()
    ir = export_ir(model, input_shape=(C_IN, H_IN, W_IN))
    validate(ir, target='hdl')
    y_sim = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
    assert y_torch.shape == y_sim.shape, (y_torch.shape, y_sim.shape)
    assert np.abs(y_torch - y_sim).max() < atol, (
        f'max diff = {np.abs(y_torch - y_sim).max()}'
    )
    return ir


class TestConvLowering:
    @pytest.mark.parametrize('param', ['softmax', 'independent', 'iwp'])
    def test_conv_no_padding(self, param, seed):
        torch.manual_seed(11)
        model = nn.Sequential(
            LutConv2d(C_IN, 4, kernel_size=3, tree_depth=2, n_inputs=2,
                      parameterization=param),
        )
        _assert_equiv_4d(model, _binary_images())

    def test_conv_with_padding(self, seed):
        """padding=1: 範囲外スロットが真理値表特殊化で畳み込まれる"""
        torch.manual_seed(13)
        model = nn.Sequential(
            LutConv2d(C_IN, 4, kernel_size=3, tree_depth=2, padding=1),
        )
        ir = _assert_equiv_4d(model, _binary_images())
        # 入力幅は実入力のまま(定数線なし)
        assert ir.inputs[0].width == C_IN * H_IN * W_IN

    def test_conv_stride_and_depth3(self, seed):
        torch.manual_seed(17)
        model = nn.Sequential(
            LutConv2d(C_IN, 3, kernel_size=3, tree_depth=3, stride=2,
                      padding=1),
        )
        _assert_equiv_4d(model, _binary_images())

    def test_conv_multichannel(self, seed):
        torch.manual_seed(19)
        model = nn.Sequential(
            LutConv2d(C_IN, 4, kernel_size=3, tree_depth=2),
            LutConv2d(4, 6, kernel_size=3, tree_depth=2,
                      input_channels_per_tree=4),
        )
        _assert_equiv_4d(model, _binary_images())


class TestPoolingLowering:
    def test_or_pooling(self, seed):
        torch.manual_seed(23)
        model = nn.Sequential(
            LutConv2d(C_IN, 4, kernel_size=3, tree_depth=2),
            OrPooling2d(kernel_size=2),
        )
        _assert_equiv_4d(model, _binary_images())

    def test_or_pooling_window_too_large(self, seed):
        model = nn.Sequential(OrPooling2d(kernel_size=3))
        with pytest.raises(NotImplementedError):
            export_ir(model, input_shape=(C_IN, H_IN, W_IN))


class TestSpatialGroupSumLowering:
    def test_sgs_counts(self, seed):
        torch.manual_seed(29)
        model = nn.Sequential(
            LutConv2d(C_IN, 8, kernel_size=3, tree_depth=2),
            SpatialGroupSum(n_outputs=2),
        )
        _assert_equiv_4d(model, _binary_images())

    def test_sgs_scale_recorded(self, seed):
        model = nn.Sequential(
            LutConv2d(C_IN, 4, kernel_size=3, tree_depth=1),
            SpatialGroupSum(n_outputs=2, scale=0.5),
        )
        model.eval()
        x4 = _binary_images(4)
        ir = export_ir(model, input_shape=(C_IN, H_IN, W_IN))
        # simulateは整数カウント、torchはcount×scale
        with torch.no_grad():
            y_torch = model(x4).flatten(1).numpy()
        y_sim = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.abs(y_torch - y_sim * 0.5).max() < 1e-5


class TestFullCnnPipeline:
    def test_conv_pool_dense_groupsum(self, seed):
        """CNN分類器を丸ごとIR化: conv→pool→Flatten→LutLayer→GroupSum"""
        torch.manual_seed(31)
        conv_out = 4
        pooled = conv_out * 3 * 3  # 8x8 → conv3 → 6x6 → pool2 → 3x3
        model = nn.Sequential(
            LutConv2d(C_IN, conv_out, kernel_size=3, tree_depth=2),
            OrPooling2d(kernel_size=2),
            nn.Flatten(),
            LutLayer(pooled, 24, n_inputs=2),
            GroupSum(k=4, tau=2.0),
        )
        model.eval()
        x4 = _binary_images(32)
        with torch.no_grad():
            logits = model(x4).numpy()
        ir = export_ir(model, input_shape=(C_IN, H_IN, W_IN))
        validate(ir, target='hdl')
        scores = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.abs(logits * 2.0 - scores).max() < 1e-3
        assert (logits.argmax(-1) == scores.argmax(-1)).all()


class TestBinaryBrainLowering:
    def _trained_blocks(self):
        torch.manual_seed(37)
        model = nn.Sequential(
            DifferentiableLutBlock(32, 16, n_inputs=4),
            DifferentiableLutBlock(16, 8, n_inputs=4),
        )
        # BNのrunning統計を作る(意味のある折り畳みには訓練後統計が必要)
        model.train()
        for _ in range(3):
            model(torch.randint(0, 2, (64, 32)).float())
        return model

    def test_bb_stack_equivalence(self, seed):
        model = self._trained_blocks()
        model.eval()
        x = torch.randint(0, 2, (32, 32)).float()
        with torch.no_grad():
            y_torch = model(x).numpy()
        ir = export_ir(model, input_width=32)
        validate(ir, target='hdl')
        y_sim = simulate(ir, x.numpy().astype(np.int64))
        assert np.array_equal(y_torch, y_sim), (
            f'不一致ビット数: {(y_torch != y_sim).sum()}'
        )

    def test_bb_single_node_per_block(self, seed):
        """各ブロックが単一のlut_layerノードに畳まれている(BN折り畳み)"""
        model = self._trained_blocks()
        ir = export_ir(model, input_width=32)
        assert len(ir.nodes) == 2
        assert all(n.type == 'lut_layer' for n in ir.nodes)
        assert all(n.train_info.get('lowered_from') == 'DifferentiableLutBlock'
                   for n in ir.nodes)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
