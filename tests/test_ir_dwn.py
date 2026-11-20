"""DWNLayerのIR対応テスト。

DWNのeval意味論(layers/dwn.py):
    addr = Σ_j 1[x[m(j,u)] > 0] << j、 y_u = 1[luts[u, addr] > 0]
は、二値入力{0,1}に対して lut_layer意味論
    addr = Σ_j x[indices[j,u]] << j、 y_u = (tt_u >> addr) & 1
(tt_u の bit k = 1[luts[u,k] > 0])と厳密一致する。ここでは
    (a) export_ir成功 + train_info(kind='dwn') + validate通過
        (train_info未知キーが拒否されないことの確認を含む)
    (b) IR往復(build_torch)のhard eval完全一致 + DWNLayer型での復元
    (c) IR simulate一致(入力全パターン掃引を含む)
    (d) HDL 3段等価性(@hdl、要iverilog)
を機械検証する。builderの復元は非可逆(±1符号化。eval等価のみ保証)。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import DWNLayer, GroupSum, LearnableMapping
from griffith_logic.export import export_ir
from griffith_logic.ir import build_torch, load_ir, save_ir, simulate, validate
from griffith_logic.ir.serialize import tt_from_hex

TAU = 2.0
IN_BITS = 16
N_LUTS = 12
N_CLASSES = 4

MAPPINGS = ('random', 'learnable')


def _model(mapping='random', n_inputs=4):
    torch.manual_seed(41)
    return nn.Sequential(
        DWNLayer(IN_BITS, N_LUTS, n_inputs=n_inputs, mapping=mapping),
        GroupSum(k=N_CLASSES, tau=TAU),
    )


def _binary_input(batch=64, width=IN_BITS, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, width), generator=g).float()


class TestDwnExport:
    @pytest.mark.parametrize('mapping', MAPPINGS)
    def test_export_ir_succeeds(self, mapping, seed):
        """DWN入りモデルがexport_irでき、kind='dwn'のlut_layerになる"""
        ir = export_ir(_model(mapping), input_width=IN_BITS)
        assert [n.type for n in ir.nodes] == ['lut_layer', 'group_sum']
        node = ir.nodes[0]
        assert node.train_info['kind'] == 'dwn'
        assert node.train_info['mapping'] == mapping
        assert node.params['in_width'] == IN_BITS
        assert node.params['out_width'] == N_LUTS
        # validateがtrain_info未知キー(kind/mapping/efd)を拒否しない
        # (HDLターゲット含む。validateはtrain_infoを参照しない契約)
        assert validate(ir, target='hdl') == []

    def test_truth_table_matches_lut_sign(self, seed):
        """tt_u の bit k = 1[luts[u,k] > 0](evalの閾値0二値化と同一規則)"""
        n_inputs = 3
        model = _model('random', n_inputs=n_inputs)
        ir = export_ir(model, input_width=IN_BITS)
        tts = tt_from_hex(ir.nodes[0].params['truth_tables'])
        luts = model[0].luts.detach()
        for u in range(N_LUTS):
            for k in range(2 ** n_inputs):
                assert (tts[u] >> k) & 1 == int(luts[u, k] > 0), (u, k)

    def test_random_mapping_indices_are_buffer(self, seed):
        """mapping='random'のindicesはDWNのindicesバッファそのもの"""
        model = _model('random')
        ir = export_ir(model, input_width=IN_BITS)
        assert np.array_equal(ir.nodes[0].params['indices'],
                              model[0].indices.cpu().numpy())

    def test_learnable_mapping_indices_are_argmax(self, seed):
        """mapping='learnable'のindicesはargmax(W)確定(evalのハード選択)"""
        model = _model('learnable')
        layer = model[0]
        ir = export_ir(model, input_width=IN_BITS)
        mapping = layer.learnable_mapping.weights.argmax(dim=0)
        expected = mapping.reshape(N_LUTS, layer.n_inputs).T.numpy()
        assert np.array_equal(ir.nodes[0].params['indices'], expected)

    def test_ste_false_not_supported(self):
        """ste=False(実数出力)は論理IRへ抽出できない"""
        torch.manual_seed(41)
        model = nn.Sequential(DWNLayer(8, 4, n_inputs=2, ste=False))
        with pytest.raises(NotImplementedError, match='ste=False'):
            export_ir(model, input_width=8)


class TestDwnRoundtrip:
    @pytest.mark.parametrize('mapping', MAPPINGS)
    def test_builder_restores_dwn_and_hard_eval_identical(self, mapping, seed):
        """kind='dwn'のbuilder復元がDWNLayer型で、hard evalが完全一致"""
        model = _model(mapping)
        ir = export_ir(model, input_width=IN_BITS)
        rebuilt = build_torch(ir)

        assert isinstance(rebuilt[0], DWNLayer)
        assert rebuilt[0].mapping_mode == mapping
        assert isinstance(rebuilt[1], GroupSum)

        x = _binary_input()
        model.eval()
        rebuilt.eval()
        with torch.no_grad():
            y_orig = model(x).numpy()
            y_rebuilt = rebuilt(x).numpy()
        assert np.array_equal(y_orig, y_rebuilt), (
            f'max diff = {np.abs(y_orig - y_rebuilt).max()}'
        )

    def test_roundtrip_via_json(self, tmp_path, seed):
        """save_ir/load_ir経由でもsimulateがビット厳密一致"""
        model = _model('random')
        model.eval()
        ir = export_ir(model, input_width=IN_BITS)
        path = save_ir(ir, str(tmp_path / 'dwn'))
        loaded = load_ir(path)
        x = _binary_input().numpy().astype(np.int64)
        assert np.array_equal(simulate(ir, x), simulate(loaded, x))


class TestDwnSimulate:
    @pytest.mark.parametrize('mapping', MAPPINGS)
    def test_simulate_matches_hard_eval(self, mapping, seed):
        """IR simulate(整数カウント)とhard evalが一致"""
        model = _model(mapping)
        model.eval()
        x = _binary_input()
        with torch.no_grad():
            logits = model(x)
        ir = export_ir(model, input_width=IN_BITS)
        scores = simulate(ir, x.numpy().astype(np.int64))
        assert np.abs(logits.numpy() * TAU - scores).max() < 1e-3

    def test_exhaustive_all_input_patterns(self, seed):
        """入力全2^6パターン掃引でDWN単層のhard evalとsimulateがビット厳密一致"""
        torch.manual_seed(45)
        in_bits = 6
        model = nn.Sequential(DWNLayer(in_bits, 5, n_inputs=3))
        model.eval()
        patterns = torch.tensor(
            [[float((v >> j) & 1) for j in range(in_bits)]
             for v in range(2 ** in_bits)]
        )
        with torch.no_grad():
            y = model(patterns).numpy().astype(np.int64)
        ir = export_ir(model, input_width=in_bits)
        y_sim = simulate(ir, patterns.numpy().astype(np.int64))
        assert np.array_equal(y, y_sim)


class TestLearnableMappingExport:
    def test_identity_permutation_matches_eval(self, seed):
        """LearnableMapping単体 → 恒等置換lut_layer(n_inputs=1, tt=0x2)"""
        torch.manual_seed(47)
        model = nn.Sequential(LearnableMapping(8, 12))
        model.eval()
        ir = export_ir(model, input_width=8)
        node = ir.nodes[0]
        assert node.type == 'lut_layer'
        assert node.params['n_inputs'] == 1
        assert node.train_info['kind'] == 'learnable_mapping'
        assert validate(ir, target='hdl') == []

        x = _binary_input(batch=32, width=8)
        with torch.no_grad():
            y = model(x).numpy().astype(np.int64)
        y_sim = simulate(ir, x.numpy().astype(np.int64))
        assert np.array_equal(y, y_sim)


@pytest.mark.hdl
class TestDwnHdl:
    """3段等価性: hard eval ⇔ simulate(上のテスト) ⇔ Verilog(ここ)"""

    @pytest.mark.parametrize('mapping', MAPPINGS)
    def test_verilog_equivalence(self, mapping, tmp_path, seed):
        from griffith_logic.export.verify import (
            iverilog_available,
            run_iverilog_equivalence,
        )
        if not iverilog_available():
            pytest.skip('iverilog未導入')

        # n_inputs=6でtt 64bit(hex文字列経路)もVerilog側で検証する
        model = _model(mapping, n_inputs=6)
        model.eval()
        ir = export_ir(model, input_width=IN_BITS, name=f'dwn_{mapping}')

        g = torch.Generator().manual_seed(9)
        x = torch.randint(0, 2, (256, IN_BITS), generator=g) \
            .numpy().astype(np.int64)
        result = run_iverilog_equivalence(ir, x, str(tmp_path))
        assert result['passed'], (
            f'dwn {mapping}: Verilogとgolden modelが不一致\n{result["stdout"]}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
