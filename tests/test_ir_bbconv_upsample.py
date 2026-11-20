"""UpSample2d / BinaryBrainConv2d のIR対応テスト。

UpSample2d: nearest拡大は固定配線なので恒等置換lut_layer
    (n_inputs=1, tt=0x2)へ展開される。hard eval ⇔ simulate ⇔ HDL を検証。
BinaryBrainConv2d: im2col→DifferentiableLutBlock連鎖→col2im を
    ブロックごとのlut_layer列へ展開する(窓ごとの平坦化+プローブTT化)。
    単層/多層/BNなし最終層/padding=0 の各構成で hard eval ⇔ simulate を検証。
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
    BinaryBrainConv2d,
    GroupSum,
    LutConv2d,
    LutLayer,
    UpSample2d,
)
from griffith_logic.export import export_ir
from griffith_logic.ir import simulate, validate

TAU = 2.0
C_IN, H_IN, W_IN = 1, 4, 4


def _binary_images(shape, batch=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, *shape), generator=g).float()


def _bb_conv(lut_sizes, seed=43, **kwargs):
    """BN running統計を確定させたeval状態のBinaryBrainConv2dを返す"""
    torch.manual_seed(seed)
    conv = BinaryBrainConv2d(
        lut_sizes=lut_sizes, in_channels=C_IN, kernel_size=3, n_inputs=2,
        **kwargs,
    )
    conv.train()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        conv(torch.randint(0, 2, (32, C_IN, H_IN, W_IN), generator=g).float())
    conv.eval()
    return conv


def _assert_hard_eval_matches_simulate(model, in_shape, seed_x=7):
    """モデルのhard eval(flatten)とIR simulateのビット厳密一致を検証する"""
    model.eval()
    x4 = _binary_images(in_shape, seed=seed_x)
    ir = export_ir(model, input_shape=in_shape)
    assert validate(ir, target='hdl') == []
    with torch.no_grad():
        y = model(x4).flatten(1).numpy().astype(np.int64)
    y_sim = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
    assert np.array_equal(y, y_sim), (
        f'max diff = {np.abs(y - y_sim).max()}'
    )
    return ir


class TestUpSampleExtraction:
    @pytest.mark.parametrize('scale', [2, 3])
    def test_hard_eval_matches_simulate(self, scale, seed):
        """nearest拡大単体のhard evalとsimulateがビット厳密一致"""
        in_shape = (2, 3, 3)
        ir = _assert_hard_eval_matches_simulate(
            nn.Sequential(UpSample2d(scale)), in_shape,
        )
        node = ir.nodes[0]
        assert node.type == 'lut_layer'
        assert node.params['n_inputs'] == 1
        assert node.params['out_width'] == 2 * (3 * scale) ** 2
        assert node.train_info['lowered_from'] == 'UpSample2d'

    def test_pipeline_conv_upsample_dense(self, seed):
        """LutConv2d→UpSample2d→dense→GroupSumの合成でも一致(形状伝播)"""
        torch.manual_seed(41)
        model = nn.Sequential(
            LutConv2d(1, 2, kernel_size=3, tree_depth=2, padding=1),
            UpSample2d(2),
            nn.Flatten(),
            LutLayer(2 * 8 * 8, 16, n_inputs=2),
            GroupSum(k=4, tau=TAU),
        )
        model.eval()
        x4 = _binary_images((C_IN, H_IN, W_IN))
        ir = export_ir(model, input_shape=(C_IN, H_IN, W_IN))
        assert validate(ir, target='hdl') == []
        with torch.no_grad():
            logits = model(x4)
        scores = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.abs(logits.numpy() * TAU - scores).max() < 1e-3


class TestBinaryBrainConvExtraction:
    @pytest.mark.parametrize('lut_sizes', [[4], [8, 4]], ids=['single', 'multi'])
    def test_hard_eval_matches_simulate(self, lut_sizes, seed):
        """単層/多層ブロック構成のhard evalとsimulateがビット厳密一致"""
        conv = _bb_conv(lut_sizes)
        ir = _assert_hard_eval_matches_simulate(
            nn.Sequential(conv), (C_IN, H_IN, W_IN),
        )
        # ブロックごとに1つのlut_layerノードへ展開される
        assert [n.type for n in ir.nodes] == ['lut_layer'] * len(lut_sizes)
        assert all(n.train_info['lowered_from'] == 'BinaryBrainConv2d'
                   for n in ir.nodes)
        # padding='same'なので空間サイズ保持: 出力幅 = out_ch * H * W
        assert ir.nodes[-1].params['out_width'] == lut_sizes[-1] * H_IN * W_IN

    def test_no_bn_last_block(self, seed):
        """batch_norm=False(最終ブロックBNなし)でも一致"""
        conv = _bb_conv([8, 4], batch_norm=False)
        _assert_hard_eval_matches_simulate(
            nn.Sequential(conv), (C_IN, H_IN, W_IN),
        )

    def test_padding_zero(self, seed):
        """padding=0(valid conv)の窓平坦化でも一致"""
        conv = _bb_conv([4], padding=0)
        ir = _assert_hard_eval_matches_simulate(
            nn.Sequential(conv), (C_IN, H_IN, W_IN),
        )
        # H_out = W_out = 4 - 3 + 1 = 2
        assert ir.nodes[-1].params['out_width'] == 4 * 2 * 2

    def test_unbuilt_raises(self):
        """in_channels未確定(遅延構築前)は抽出できない"""
        conv = BinaryBrainConv2d(lut_sizes=[4], kernel_size=3, n_inputs=2)
        with pytest.raises(NotImplementedError, match='未構築'):
            export_ir(nn.Sequential(conv), input_shape=(C_IN, H_IN, W_IN))


@pytest.mark.hdl
class TestBbUpsampleHdl:
    """3段等価性: hard eval ⇔ simulate(上のテスト) ⇔ Verilog(ここ)"""

    def _run(self, model, in_shape, name, tmp_path, batch=128):
        from griffith_logic.export.verify import (
            iverilog_available,
            run_iverilog_equivalence,
        )
        if not iverilog_available():
            pytest.skip('iverilog未導入')
        model.eval()
        ir = export_ir(model, input_shape=in_shape, name=name)
        x = _binary_images(in_shape, batch=batch, seed=19) \
            .flatten(1).numpy().astype(np.int64)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], (
            f'{name}: Verilogとgolden modelが不一致\n{result["stdout"]}'
        )

    def test_upsample_pipeline_verilog(self, tmp_path, seed):
        torch.manual_seed(41)
        model = nn.Sequential(
            LutConv2d(1, 2, kernel_size=3, tree_depth=2, padding=1),
            UpSample2d(2),
            nn.Flatten(),
            LutLayer(2 * 8 * 8, 16, n_inputs=2),
            GroupSum(k=4, tau=TAU),
        )
        self._run(model, (C_IN, H_IN, W_IN), 'upsample_pipe', tmp_path)

    def test_bbconv_pipeline_verilog(self, tmp_path, seed):
        conv = _bb_conv([8, 4])
        torch.manual_seed(41)
        model = nn.Sequential(
            conv,
            nn.Flatten(),
            LutLayer(4 * H_IN * W_IN, 16, n_inputs=2),
            GroupSum(k=4, tau=TAU),
        )
        self._run(model, (C_IN, H_IN, W_IN), 'bbconv_pipe', tmp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
