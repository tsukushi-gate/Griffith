"""truth_tables → weights復元の往復一致テスト(E2)。

「学習済みレイヤ → 真理値表抽出 → weights_from_truth_tables → 新レイヤ適用
→ 真理値表再抽出」が全パラメータ化でビット厳密に一致することを検証する。
これがエディタの往復同一性(IR→build→export=元IR)の土台になる。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from griffith_logic import LutConv2d, LutLayer
from griffith_logic.core.basis import _build_gate_table_n2
from griffith_logic.core.connections import build_reverse_index
from griffith_logic.export import truth_tables_from_lut_layer
from griffith_logic.export.discretize import truth_tables_from_conv
from griffith_logic.ir import (
    TT_EDITABLE,
    gate_id_from_tt_n2,
    restore_lut_conv2d,
    restore_lut_layer,
    weights_from_truth_tables,
)
from griffith_logic.ir.serialize import tt_to_hex

IN_DIM = 16
OUT_DIM = 24

PARAMS_NS = [
    ('softmax', 2), ('softmax', 4), ('softmax', 6),
    ('independent', 2), ('independent', 4), ('independent', 6),
    ('raw', 2), ('raw', 4),
    ('iwp', 2), ('iwp', 4),
    ('warp', 2), ('warp', 4), ('warp', 6),
]
IDS = [f'{p}_n{n}' for p, n in PARAMS_NS]


def _restored_pair(param, n):
    """(元レイヤ, 元tt, 復元レイヤ) を返す。復元レイヤは別のランダム初期値から

    元レイヤのindices+ttを書き戻したもの。
    """
    src = LutLayer(IN_DIM, OUT_DIM, n_inputs=n, parameterization=param)
    tts = truth_tables_from_lut_layer(src)
    dst = LutLayer(IN_DIM, OUT_DIM, n_inputs=n, parameterization=param)
    restore_lut_layer(dst, src.indices, tt_to_hex(tts.tolist()))
    return src, tts, dst


class TestGateIdInverse:
    def test_gate_id_is_inverse_of_gate_table(self):
        """gate_id_from_tt_n2 は _build_gate_table_n2 の厳密な逆写像"""
        table = _build_gate_table_n2()
        for g in range(16):
            tt = 0
            for k in range(4):
                tt |= int(table[g, k].item()) << k
            assert gate_id_from_tt_n2(tt) == g


class TestWeightsRoundtrip:
    @pytest.mark.parametrize('param,n', PARAMS_NS, ids=IDS)
    def test_tt_roundtrip_exact(self, param, n, seed):
        """tt抽出→weights合成→再抽出がビット厳密に一致"""
        src, tts, dst = _restored_pair(param, n)
        tts2 = truth_tables_from_lut_layer(dst)
        assert (tts == tts2).all(), (
            f'{param} N={n}: 真理値表が復元されない'
        )
        assert torch.equal(dst.indices, src.indices)

    @pytest.mark.parametrize('param,n', PARAMS_NS, ids=IDS)
    def test_reverse_index_rebuilt(self, param, n, seed):
        """restore後の逆引きバッファが build_reverse_index と一致"""
        _, _, dst = _restored_pair(param, n)
        starts, flat_y, flat_slot = build_reverse_index(dst.indices, IN_DIM)
        assert torch.equal(dst.rev_starts, starts)
        assert torch.equal(dst.rev_flat_y, flat_y)
        assert torch.equal(dst.rev_flat_slot, flat_slot)


class TestUnrepresentable:
    def test_softmax_n4_popcount2_raises(self):
        """softmax N>2 は popcount>1 の真理値表を表現できない"""
        with pytest.raises(ValueError, match='表現不能'):
            weights_from_truth_tables('softmax', 4, [0b11])

    def test_softmax_n4_popcount1_ok(self):
        w = weights_from_truth_tables('softmax', 4, [0b100, 0])
        assert w.shape == (2, 16)
        assert w[0].argmax().item() == 2

    def test_iwp_popcount0_raises(self):
        with pytest.raises(ValueError, match='表現不能'):
            weights_from_truth_tables('iwp', 2, [0])

    def test_iwp_popcount2_raises(self):
        with pytest.raises(ValueError, match='表現不能'):
            weights_from_truth_tables('iwp', 4, [0b101])

    def test_tt_out_of_range_raises(self):
        with pytest.raises(ValueError, match='値域外'):
            weights_from_truth_tables('independent', 2, [16])

    def test_unknown_parameterization_raises(self):
        with pytest.raises(ValueError, match='未知'):
            weights_from_truth_tables('nosuch', 2, [0])


class TestRestoreValidation:
    def test_indices_shape_mismatch_raises(self, seed):
        layer = LutLayer(IN_DIM, OUT_DIM, n_inputs=2)
        bad = torch.zeros(2, OUT_DIM + 1, dtype=torch.int64)
        with pytest.raises(ValueError, match='indices形状'):
            restore_lut_layer(layer, bad, ['0xa'] * OUT_DIM)

    def test_indices_out_of_range_raises(self, seed):
        layer = LutLayer(IN_DIM, OUT_DIM, n_inputs=2)
        bad = torch.full((2, OUT_DIM), IN_DIM, dtype=torch.int64)
        with pytest.raises(ValueError, match='入力幅'):
            restore_lut_layer(layer, bad, ['0xa'] * OUT_DIM)

    def test_tt_count_mismatch_raises(self, seed):
        layer = LutLayer(IN_DIM, OUT_DIM, n_inputs=2)
        idx = torch.zeros(2, OUT_DIM, dtype=torch.int64)
        with pytest.raises(ValueError, match='truth_tables数'):
            restore_lut_layer(layer, idx, ['0xa'] * (OUT_DIM - 1))


class TestTTEditable:
    def test_editable_set(self):
        """エディタでtt直接編集可能なパラメータ化の契約"""
        assert TT_EDITABLE == {'softmax16g', 'independent', 'raw'}


RNG_SEED_CONV = 20260708


def _conv(param: str, n_inputs: int = 2, out_channels: int = 3,
          tree_depth: int = 2, seed: int = 0) -> LutConv2d:
    torch.manual_seed(seed)
    return LutConv2d(
        in_channels=2, out_channels=out_channels, kernel_size=3,
        tree_depth=tree_depth, padding=1, n_inputs=n_inputs,
        parameterization=param, residual_init=False,
    )


class TestConvWeightsRoundtrip:
    """conv_weights_from_truth_tables / restore_lut_conv2d の往復一致。

    softmax16gはLutConv2d側に新規追加した分岐(weights_from_truth_tables
    のis_16gate経路の再利用)。任意のN=2真理値表(16通り全て)が
    ビット厳密に往復することを担保する(native Conv-DLGN構造転写の前提)。
    """

    def test_softmax16g_all_16_functions(self, seed):
        conv = _conv('softmax16g', n_inputs=2, out_channels=6, tree_depth=2)
        n_gates = conv.out_channels * conv.n_internal
        assert n_gates >= 16
        tts = [i % 16 for i in range(n_gates)]
        conv.eval()
        restore_lut_conv2d(conv, conv.indices_flat, tt_to_hex(tts))
        extracted = truth_tables_from_conv(conv).flatten().tolist()
        assert [int(t) for t in extracted] == tts

    @pytest.mark.parametrize('param,n', [
        ('independent', 2), ('independent', 4), ('iwp', 2), ('iwp', 4),
    ], ids=['independent_n2', 'independent_n4', 'iwp_n2', 'iwp_n4'])
    def test_existing_parameterizations_still_roundtrip(self, param, n, seed):
        """softmax16g追加が既存パラメータ化の挙動を壊していないことの回帰確認"""
        conv = _conv(param, n_inputs=n, out_channels=3, tree_depth=2)
        n_gates = conv.out_channels * conv.n_internal
        limit = 1 << (1 << n)
        if param == 'iwp':
            # iwpは単一minterm(popcount==1)のみ表現可能
            tts = [1 << (i % (1 << n)) for i in range(n_gates)]
        else:
            rng = torch.Generator().manual_seed(RNG_SEED_CONV + n)
            tts = [int(torch.randint(0, limit, (1,), generator=rng)) for _ in range(n_gates)]
        conv.eval()
        restore_lut_conv2d(conv, conv.indices_flat, tt_to_hex(tts))
        extracted = truth_tables_from_conv(conv).flatten().tolist()
        assert [int(t) for t in extracted] == tts

    def test_softmax16g_indices_preserved(self, seed):
        conv = _conv('softmax16g', n_inputs=2, out_channels=3, tree_depth=2)
        n_gates = conv.out_channels * conv.n_internal
        tts = [i % 16 for i in range(n_gates)]
        new_idx = torch.zeros_like(conv.indices_flat)  # 全リーフを受容野0番へ
        restore_lut_conv2d(conv, new_idx, tt_to_hex(tts))
        assert torch.equal(conv.indices_flat, new_idx)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
