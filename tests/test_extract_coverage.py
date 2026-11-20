"""抽出器カバレッジのテスト。

(1) Binarize抽出器: binarize入りモデルの export_ir 成功 +
    IR→build_torch 往復での hard eval 完全一致
(2) 抽出器lookupのMRO化: サブクラス(LutConv2dHardST)が親クラス
    (LutConv2d)の抽出器で抽出でき、lowering結果も一致する
(3) 既存抽出器の回帰: MRO化後もノード列・パラメータが変わらない
    (スナップショット的検証)+ 未登録型のエラー方針維持
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import Binarize, GroupSum, LutConv2d, LutLayer
from griffith_logic.conv_hard_st import LutConv2dHardST
from griffith_logic.export import export_ir, truth_tables_from_lut_layer
from griffith_logic.export.extract import (
    EXTRACTORS,
    _extract_lut_layer,
    find_extractor,
    register_extractor,
)
from griffith_logic.ir import build_torch, simulate, validate
from griffith_logic.ir.serialize import tt_to_hex

TAU = 2.0
C_IN, H_IN, W_IN = 1, 6, 6


def _binary_input(batch=32, width=16, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, width), generator=g).float()


def _binary_images(batch=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, C_IN, H_IN, W_IN), generator=g).float()


class TestBinarizeExtraction:
    def _model(self):
        torch.manual_seed(41)
        return nn.Sequential(
            nn.Flatten(),
            LutLayer(16, 24, n_inputs=2),
            Binarize(),
            LutLayer(24, 24, n_inputs=2),
            GroupSum(k=4, tau=TAU),
        )

    def test_export_ir_succeeds(self, seed):
        """Binarize入りモデルがexport_irでき、幅が前段から伝播する"""
        ir = export_ir(self._model(), input_width=16)
        assert validate(ir, target='hdl') is not None
        assert [n.type for n in ir.nodes] == [
            'lut_layer', 'binarize', 'lut_layer', 'group_sum',
        ]
        bin_node = ir.nodes[1]
        assert bin_node.params['width'] == 24  # 前段lut_layerのout_width
        assert bin_node.params['threshold'] == 0.5

    def test_roundtrip_hard_eval_identical(self, seed):
        """export_ir → build_torch 往復モデルのhard evalが完全一致"""
        model = self._model()
        ir = export_ir(model, input_width=16)
        rebuilt = build_torch(ir)

        x = _binary_input()
        model.eval()
        rebuilt.eval()
        with torch.no_grad():
            y_orig = model(x).numpy()
            y_rebuilt = rebuilt(x).numpy()
        assert np.array_equal(y_orig, y_rebuilt), (
            f'max diff = {np.abs(y_orig - y_rebuilt).max()}'
        )

    def test_simulate_matches_hard_eval(self, seed):
        """IR simulate(整数カウント)ともhard evalが一致"""
        model = self._model()
        model.eval()
        x = _binary_input()
        with torch.no_grad():
            logits = model(x)
        ir = export_ir(model, input_width=16)
        scores = simulate(ir, x.numpy().astype(np.int64))
        assert np.abs(logits.numpy() * TAU - scores).max() < 1e-3


class TestSubclassExtraction:
    def _hard_st_pair(self):
        """同一重みのLutConv2dHardSTとLutConv2dの組を返す"""
        torch.manual_seed(43)
        hard = LutConv2dHardST(C_IN, 4, kernel_size=3, tree_depth=2)
        plain = LutConv2d(C_IN, 4, kernel_size=3, tree_depth=2)
        plain.load_state_dict(hard.state_dict())
        return hard, plain

    def test_mro_lookup_finds_parent_extractor(self):
        """サブクラスに対して親クラス(LutConv2d)の抽出器が見つかる"""
        from griffith_logic.export import lowering  # noqa: F401
        hard, plain = self._hard_st_pair()
        assert type(hard) not in EXTRACTORS  # exact登録は無い
        assert find_extractor(hard) is EXTRACTORS[LutConv2d]

    def test_hard_st_lowering_matches_parent(self, seed):
        """サブクラスのexport_irが成功し、親クラスと同一のloweringになる"""
        hard, plain = self._hard_st_pair()
        ir_hard = export_ir(nn.Sequential(hard),
                            input_shape=(C_IN, H_IN, W_IN))
        ir_plain = export_ir(nn.Sequential(plain),
                             input_shape=(C_IN, H_IN, W_IN))

        assert [n.type for n in ir_hard.nodes] == \
            [n.type for n in ir_plain.nodes]
        for nh, np_node in zip(ir_hard.nodes, ir_plain.nodes):
            assert np.array_equal(nh.params['indices'],
                                  np_node.params['indices'])
            assert nh.params['truth_tables'] == np_node.params['truth_tables']

    def test_hard_st_hard_eval_matches_simulate(self, seed):
        """サブクラスモデルのhard evalとIR simulateが厳密一致"""
        hard, _ = self._hard_st_pair()
        model = nn.Sequential(hard)
        model.eval()
        x4 = _binary_images()
        with torch.no_grad():
            y_torch = model(x4).flatten(1).numpy()
        ir = export_ir(model, input_shape=(C_IN, H_IN, W_IN))
        validate(ir, target='hdl')
        y_sim = simulate(ir, x4.flatten(1).numpy().astype(np.int64))
        assert np.abs(y_torch - y_sim).max() < 1e-4, (
            f'max diff = {np.abs(y_torch - y_sim).max()}'
        )


class TestExistingExtractorsRegression:
    def test_lut_groupsum_node_snapshot(self, seed):
        """MRO化後もLutLayer/GroupSumの抽出結果ノード列が変わらない"""
        torch.manual_seed(31)
        model = nn.Sequential(
            LutLayer(16, 24, n_inputs=2),
            LutLayer(24, 24, n_inputs=2),
            GroupSum(k=4, tau=2.5),
        )
        ir = export_ir(model, input_width=16)

        # ノード列のスナップショット(id・型・幅パラメータ)
        assert [(n.id, n.type) for n in ir.nodes] == [
            ('n0_lutlayer', 'lut_layer'),
            ('n1_lutlayer', 'lut_layer'),
            ('n2_groupsum', 'group_sum'),
        ]
        n0, n1, n2 = ir.nodes
        assert (n0.params['in_width'], n0.params['out_width']) == (16, 24)
        assert (n1.params['in_width'], n1.params['out_width']) == (24, 24)
        assert (n2.params['in_width'], n2.params['k']) == (24, 4)
        assert n2.params['tau'] == 2.5

        # indices・真理値表が離散化器の直接出力と一致
        for node, layer in [(n0, model[0]), (n1, model[1])]:
            assert np.array_equal(node.params['indices'],
                                  layer.indices.cpu().numpy())
            expected = tt_to_hex(truth_tables_from_lut_layer(layer).tolist())
            assert node.params['truth_tables'] == expected

    def test_exact_match_takes_priority(self, seed):
        """exact一致の登録がMRO走査で親クラスの抽出器より優先される"""
        class MarkedLutLayer(LutLayer):
            pass

        @register_extractor(MarkedLutLayer)
        def _extract_marked(module, node_id, ctx):
            nodes, width, shape = _extract_lut_layer(module, node_id, ctx)
            nodes[0].train_info['marked'] = True
            return nodes, width, shape

        try:
            model = nn.Sequential(MarkedLutLayer(8, 8, n_inputs=2))
            ir = export_ir(model, input_width=8)
            assert ir.nodes[0].train_info.get('marked') is True
        finally:
            del EXTRACTORS[MarkedLutLayer]  # グローバルレジストリを汚さない

    def test_unregistered_module_raises(self):
        """未登録型(MROに登録型なし)は従来どおりNotImplementedError"""
        with pytest.raises(NotImplementedError, match='抽出器が未登録'):
            export_ir(nn.Sequential(nn.ReLU()), input_width=8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
