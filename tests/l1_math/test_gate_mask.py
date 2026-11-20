"""L1: ゲート種ロジットマスク(先行実験 S1)の数学的性質。

検証する仕様:
    1. 16ゲートID規約 op = 8·P00 + 4·P01 + 2·P10 + P11
       (bit = (op >> (3 − (2a+b))) & 1)。社内先行実験の op ニブルと同一。
       op3 = A pass-through / {0,15} = 定数 / {3,5,10,12} = pass-through と否定。
    2. gate_mask で除外したゲートは
       - softmax確率が厳密に0
       - eval の argmax が選ばない
       - export_ir → gate_id_from_tt_n2 の op 集合に現れない
    3. soft forward が softmax(masked logits) @ gate_table と厳密一致する
    4. gate_mask=None は既定挙動と bitwise 不変(乱数消費も含む)
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.core.basis import _compute_basis_butterfly
from griffith_logic.group_sum import GroupSum
from griffith_logic.ir.weights import gate_id_from_tt_n2
from griffith_logic.layers.lut import LutLayer

# 計画(先行実験 plan.md「腕」節)のマスク集合
CONST_OPS = (0, 15)
PASSTHROUGH_OPS = (3, 5, 10, 12)
HIDDEN_MASK = (0, 15, 3, 5, 10, 12)
FINAL_MASK = (3, 5, 10, 12)

ALL_PATTERNS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _onehot_layer(gate_mask=None, selection='soft', logit=10.0):
    """in=2 / out=16 で「出力gがゲートgに固定」された層(eval規約の全数掃引用)"""
    conn = torch.zeros(2, 16, dtype=torch.int64)
    conn[1] = 1
    layer = LutLayer(
        2, 16, n_inputs=2, connections=conn, parameterization='softmax',
        selection=selection, gate_mask=gate_mask,
    )
    with torch.no_grad():
        layer.weights.zero_()
        layer.weights[torch.arange(16), torch.arange(16)] = logit
    return layer


class TestGateIdConvention:
    """16ゲートID規約を eval 経路で全数(16ゲート × 4パターン)確認する"""

    def test_all_16_gates_match_op_nibble(self):
        layer = _onehot_layer()
        layer.eval()
        x = torch.tensor([[float(a), float(b)] for a, b in ALL_PATTERNS])
        with torch.no_grad():
            y = layer(x)
        for pi, (a, b) in enumerate(ALL_PATTERNS):
            for g in range(16):
                expected = float((g >> (3 - (2 * a + b))) & 1)
                assert y[pi, g].item() == expected, (
                    f'gate {g} の f({a},{b}) が規約と不一致: '
                    f'{y[pi, g].item()} != {expected}'
                )

    def test_named_gates(self):
        """op3=A / op5=B / op12=¬A / op10=¬B / op0=FALSE / op15=TRUE"""
        layer = _onehot_layer()
        layer.eval()
        x = torch.tensor([[float(a), float(b)] for a, b in ALL_PATTERNS])
        with torch.no_grad():
            y = layer(x)
        a_col = torch.tensor([float(a) for a, _ in ALL_PATTERNS])
        b_col = torch.tensor([float(b) for _, b in ALL_PATTERNS])
        assert torch.equal(y[:, 3], a_col)
        assert torch.equal(y[:, 5], b_col)
        assert torch.equal(y[:, 12], 1.0 - a_col)
        assert torch.equal(y[:, 10], 1.0 - b_col)
        assert torch.equal(y[:, 0], torch.zeros(4))
        assert torch.equal(y[:, 15], torch.ones(4))

    def test_mask_sets_are_constant_and_passthrough(self):
        """計画のマスク集合が定数2種・pass-through4種であることを確認する"""
        layer = _onehot_layer()
        layer.eval()
        x = torch.tensor([[float(a), float(b)] for a, b in ALL_PATTERNS])
        with torch.no_grad():
            y = layer(x)
        for g in CONST_OPS:
            col = y[:, g]
            assert col.min() == col.max(), f'op{g}が定数でない: {col}'
        a_col = torch.tensor([float(a) for a, _ in ALL_PATTERNS])
        b_col = torch.tensor([float(b) for _, b in ALL_PATTERNS])
        singles = [a_col, 1.0 - a_col, b_col, 1.0 - b_col]
        for g in PASSTHROUGH_OPS:
            assert any(torch.equal(y[:, g], s) for s in singles), (
                f'op{g}が1入力関数でない: {y[:, g]}'
            )
        assert set(HIDDEN_MASK) == set(CONST_OPS) | set(PASSTHROUGH_OPS)
        assert set(FINAL_MASK) == set(PASSTHROUGH_OPS)


class TestMaskedProbabilities:
    def test_softmax_probability_is_exactly_zero(self):
        torch.manual_seed(0)
        layer = LutLayer(8, 32, n_inputs=2, gate_mask=HIDDEN_MASK)
        p = F.softmax(layer._masked_weights(), dim=-1)
        assert torch.equal(p[:, list(HIDDEN_MASK)],
                           torch.zeros(32, len(HIDDEN_MASK)))
        assert torch.allclose(p.sum(-1), torch.ones(32))

    def test_allowed_gate_ids(self):
        layer = LutLayer(8, 4, n_inputs=2, gate_mask=FINAL_MASK)
        assert layer.allowed_gate_ids() == tuple(
            g for g in range(16) if g not in FINAL_MASK
        )
        assert LutLayer(8, 4, n_inputs=2).allowed_gate_ids() == tuple(range(16))


class TestMaskedEvalAndExport:
    @pytest.mark.parametrize('mask', [HIDDEN_MASK, FINAL_MASK, (1,)])
    def test_eval_argmax_never_selects_masked(self, mask):
        """除外ゲートに最大ロジットを与えても argmax は許可集合から選ぶ"""
        torch.manual_seed(1)
        layer = LutLayer(16, 64, n_inputs=2, gate_mask=mask)
        with torch.no_grad():
            layer.weights[:, list(mask)] = 100.0
        chosen = layer._masked_weights().argmax(-1)
        assert set(chosen.tolist()).isdisjoint(set(mask))

    def test_export_ir_op_set_excludes_masked(self):
        from griffith_logic.export import export_ir
        from griffith_logic.ir.serialize import tt_from_hex

        torch.manual_seed(2)
        layer = LutLayer(16, 40, n_inputs=2, selection='hard_st',
                         gate_mask=HIDDEN_MASK)
        with torch.no_grad():
            layer.weights[:, list(HIDDEN_MASK)] = 50.0
        model = torch.nn.Sequential(
            torch.nn.Flatten(), layer, GroupSum(10, tau=1.0)
        )
        graph = export_ir(model, input_width=16)
        node = next(n for n in graph.nodes if n.type == 'lut_layer')
        ops = {
            gate_id_from_tt_n2(tt)
            for tt in tt_from_hex(node.params['truth_tables'])
        }
        assert ops.isdisjoint(set(HIDDEN_MASK)), f'マスク対象がexportされた: {ops}'

    def test_export_ir_without_mask_can_contain_masked_ops(self):
        """対照: マスク無しなら同じ重みで除外対象の op が出る(テストの検出力確認)"""
        from griffith_logic.export import export_ir
        from griffith_logic.ir.serialize import tt_from_hex

        torch.manual_seed(2)
        layer = LutLayer(16, 40, n_inputs=2, selection='hard_st')
        with torch.no_grad():
            layer.weights[:, list(HIDDEN_MASK)] = 50.0
        model = torch.nn.Sequential(
            torch.nn.Flatten(), layer, GroupSum(10, tau=1.0)
        )
        graph = export_ir(model, input_width=16)
        node = next(n for n in graph.nodes if n.type == 'lut_layer')
        ops = {
            gate_id_from_tt_n2(tt)
            for tt in tt_from_hex(node.params['truth_tables'])
        }
        assert not ops.isdisjoint(set(HIDDEN_MASK))


class TestSoftForwardConsistency:
    def test_soft_forward_matches_masked_truth_table(self):
        """soft forward = basis·(softmax(masked)@gate_table) を float64 で厳密照合"""
        torch.manual_seed(3)
        layer = LutLayer(8, 24, n_inputs=2, selection='soft',
                         gate_mask=HIDDEN_MASK).double()
        layer.train()
        x = torch.rand(5, 8, dtype=torch.float64)
        y = layer(x)

        table = F.softmax(layer._masked_weights(), dim=-1) @ layer.gate_table
        gathered = x[:, layer.indices.T.reshape(-1)].reshape(5, 24, 2)
        basis = _compute_basis_butterfly(gathered, 2)
        expected = torch.einsum('boe,oe->bo', basis, table)
        assert torch.allclose(y, expected, atol=TOL_F64_EQ, rtol=0)

    def test_soft_truth_table_helper_agrees(self):
        """regularization.apply.soft_truth_table(masked) が同じ表を返す"""
        from griffith_logic.regularization.apply import soft_truth_table

        torch.manual_seed(4)
        layer = LutLayer(8, 12, n_inputs=2, gate_mask=FINAL_MASK).double()
        table = soft_truth_table(
            layer._masked_weights(), 'softmax16g', 2,
            gate_table=layer.gate_table,
        )
        direct = F.softmax(layer._masked_weights(), dim=-1) @ layer.gate_table
        assert torch.allclose(table, direct, atol=TOL_F64_EQ, rtol=0)


class TestBackwardCompatibility:
    @pytest.mark.parametrize('selection', ['soft', 'hard_st', 'gumbel_st'])
    def test_gate_mask_none_is_bitwise_identical(self, selection):
        torch.manual_seed(7)
        base = LutLayer(16, 32, n_inputs=2, selection=selection)
        torch.manual_seed(7)
        with_none = LutLayer(16, 32, n_inputs=2, selection=selection,
                             gate_mask=None)
        assert torch.equal(base.weights, with_none.weights)
        assert torch.equal(base.indices, with_none.indices)
        assert not hasattr(base, 'gate_logit_mask')
        assert not hasattr(with_none, 'gate_logit_mask')
        assert base.gate_mask == ()

        x = torch.rand(4, 16)
        base.eval()
        with_none.eval()
        with torch.no_grad():
            assert torch.equal(base(x), with_none(x))
        base.train()
        with_none.train()
        torch.manual_seed(11)
        y0 = base(x)
        torch.manual_seed(11)
        y1 = with_none(x)
        assert torch.equal(y0, y1)

    def test_empty_mask_is_allowed_and_inert(self):
        torch.manual_seed(7)
        layer = LutLayer(16, 8, n_inputs=2, gate_mask=[])
        assert layer.gate_mask == ()
        assert not hasattr(layer, 'gate_logit_mask')
        assert torch.equal(layer._masked_weights(), layer.weights)


class TestValidation:
    def test_rejects_non_softmax(self):
        with pytest.raises(ValueError, match='16ゲートモード'):
            LutLayer(8, 4, n_inputs=2, parameterization='independent',
                     gate_mask=[0])

    def test_rejects_n_inputs_4(self):
        with pytest.raises(ValueError, match='16ゲートモード'):
            LutLayer(8, 4, n_inputs=4, parameterization='softmax', gate_mask=[0])

    def test_rejects_out_of_range_id(self):
        with pytest.raises(ValueError, match='0〜15'):
            LutLayer(8, 4, n_inputs=2, gate_mask=[16])

    def test_rejects_all_masked(self):
        with pytest.raises(ValueError, match='16種すべて'):
            LutLayer(8, 4, n_inputs=2, gate_mask=list(range(16)))

    def test_duplicates_are_deduplicated(self):
        layer = LutLayer(8, 4, n_inputs=2, gate_mask=[3, 3, 5])
        assert layer.gate_mask == (3, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
