"""真理値表・接続の単一ニューロン編集テスト。

エディタの編集契約を検証する:
    (a) 編集可能パラメータ化(TT_EDITABLE)で「任意tt設定 →
        truth_tables_from_lut_layer 再抽出 → 完全一致」
        - softmax16g: 全16関数
        - independent / raw: ランダム64パターン(N∈{2,6})
    (b) 単一ニューロンin-place更新(edit.set_neuron_truth_table)が
        該当ニューロンのみを書き換える
    (c) 単一ニューロン接続更新(edit.set_neuron_indices)後の逆引きバッファが
        build_reverse_index(indices, in_dim) と一致する
"""

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import LutLayer
from griffith_logic.core.connections import build_reverse_index
from griffith_logic.export import truth_tables_from_lut_layer
from griffith_logic.ir import TT_EDITABLE, restore_lut_layer
from tools.editor.backend.edit import set_neuron_indices, set_neuron_truth_table

IN_DIM = 16
RANDOM_PATTERNS = 64  # independent/rawで検証するランダム真理値表の数
RNG_SEED = 20260702


def _random_tts(n_inputs: int, count: int) -> list[int]:
    rng = random.Random(RNG_SEED + n_inputs)
    limit = 1 << (1 << n_inputs)
    return [rng.randrange(limit) for _ in range(count)]


def _layer_for(param: str, n_inputs: int, out_dim: int) -> LutLayer:
    # softmax16g はコア上 softmax+N=2(_use_16gate)として実現される
    real = 'softmax' if param == 'softmax16g' else param
    return LutLayer(IN_DIM, out_dim, n_inputs=n_inputs, parameterization=real)


def _hex(tts: list[int]) -> list[str]:
    return [hex(t) for t in tts]


class TestEditableRoundtrip:
    """編集可能パラメータ化: 任意tt設定 → 再抽出 → 完全一致。"""

    def test_softmax16g_all_16_functions(self, seed):
        tts = list(range(16))
        layer = _layer_for('softmax16g', 2, len(tts))
        assert layer._use_16gate
        restore_lut_layer(layer, layer.indices, _hex(tts))
        extracted = truth_tables_from_lut_layer(layer)
        assert [int(t) for t in extracted] == tts

    @pytest.mark.parametrize('param', ['independent', 'raw'])
    @pytest.mark.parametrize('n', [2, 6])
    def test_random_patterns_exact(self, param, n, seed):
        tts = _random_tts(n, RANDOM_PATTERNS)
        layer = _layer_for(param, n, len(tts))
        restore_lut_layer(layer, layer.indices, _hex(tts))
        extracted = truth_tables_from_lut_layer(layer)
        assert [int(t) for t in extracted] == tts, (
            f'{param} N={n}: 設定した真理値表が復元されない'
        )

    def test_editable_contract(self):
        """フロント(TruthTableEditor)へコピーした規約と一致すること"""
        assert TT_EDITABLE == {'softmax16g', 'independent', 'raw'}


class TestSingleNeuronTTEdit:
    """set_neuron_truth_table: 該当ニューロンのみin-place更新。"""

    @pytest.mark.parametrize('param,n', [
        ('softmax16g', 2), ('independent', 2), ('independent', 6),
        ('raw', 2), ('raw', 6),
    ], ids=['softmax16g_n2', 'independent_n2', 'independent_n6',
            'raw_n2', 'raw_n6'])
    def test_updates_only_target_neuron(self, param, n, seed):
        out_dim = 12
        target = 5
        layer = _layer_for(param, n, out_dim)
        before = truth_tables_from_lut_layer(layer)

        rng = random.Random(RNG_SEED)
        new_tt = rng.randrange(1 << (1 << n))
        set_neuron_truth_table(layer, target, new_tt)

        after = truth_tables_from_lut_layer(layer)
        assert int(after[target]) == new_tt
        for o in range(out_dim):
            if o != target:
                assert int(after[o]) == int(before[o]), (
                    f'ニューロン{o}が巻き添え更新された'
                )

    def test_neuron_out_of_range_raises(self, seed):
        layer = _layer_for('independent', 2, 4)
        with pytest.raises(ValueError, match='範囲外'):
            set_neuron_truth_table(layer, 4, 0b1010)

    def test_unrepresentable_tt_raises(self, seed):
        """iwpはpopcount!=1の真理値表を表現できない(コア規約の伝播)"""
        layer = LutLayer(IN_DIM, 4, n_inputs=2, parameterization='iwp')
        with pytest.raises(ValueError, match='表現不能'):
            set_neuron_truth_table(layer, 0, 0b0110)


class TestSingleNeuronIndicesEdit:
    """set_neuron_indices: 接続差し替え+逆引きバッファ再計算。"""

    @pytest.mark.parametrize('n', [2, 6])
    def test_reverse_index_consistent(self, n, seed):
        out_dim = 10
        target = 3
        layer = LutLayer(IN_DIM, out_dim, n_inputs=n)
        rng = random.Random(RNG_SEED)
        slots = [rng.randrange(IN_DIM) for _ in range(n)]

        before = layer.indices.clone()
        set_neuron_indices(layer, target, slots)

        # 該当列のみ差し替わる
        assert layer.indices[:, target].tolist() == slots
        mask = torch.ones(out_dim, dtype=torch.bool)
        mask[target] = False
        assert torch.equal(layer.indices[:, mask], before[:, mask])

        # 逆引きバッファが build_reverse_index と厳密一致
        starts, flat_y, flat_slot = build_reverse_index(layer.indices, IN_DIM)
        assert torch.equal(layer.rev_starts, starts)
        assert torch.equal(layer.rev_flat_y, flat_y)
        assert torch.equal(layer.rev_flat_slot, flat_slot)

    def test_backward_works_after_edit(self, seed):
        """逆引き再計算漏れがあればbackwardが壊れる — 実際に勾配を流して検証"""
        layer = LutLayer(IN_DIM, 8, n_inputs=2)
        set_neuron_indices(layer, 0, [IN_DIM - 1, 0])
        x = torch.rand(4, IN_DIM, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_slot_count_mismatch_raises(self, seed):
        layer = LutLayer(IN_DIM, 4, n_inputs=2)
        with pytest.raises(ValueError, match='slots数'):
            set_neuron_indices(layer, 0, [1, 2, 3])

    def test_slot_out_of_range_raises(self, seed):
        layer = LutLayer(IN_DIM, 4, n_inputs=2)
        with pytest.raises(ValueError, match='入力幅'):
            set_neuron_indices(layer, 0, [0, IN_DIM])

    def test_neuron_out_of_range_raises(self, seed):
        layer = LutLayer(IN_DIM, 4, n_inputs=2)
        with pytest.raises(ValueError, match='範囲外'):
            set_neuron_indices(layer, 4, [0, 1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
