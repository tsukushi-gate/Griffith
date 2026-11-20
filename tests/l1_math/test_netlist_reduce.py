"""dense DLGN 平坦ネットリスト縮約器の L1 性質テスト(先行実験実効率 U の測定装置)。

検証内容:
    - op(P0 ゲート ID) ⇔ tt(IR 真理値表)の 16 通り全数一致
    - 縮約 5 規則それぞれの最小ネット(手で作った期待内訳と厳密照合)
    - 未参照の 2 指標(structurally_unreferenced / unreachable_after_fold)の分離
    - LutLayer モデル → export_ir → アダプタ → 原ネット評価が ir.simulate と bit 一致
    - seed 固定のランダム小ネットで縮約前後のクラス別スコアが整数完全一致

移植元は社内先行実験の
`pilot/experiments/irreducible-dlgn-p0/reduce_netlist.py`(P0)。
統計のフィールド名・意味は P0 の `p0_results.json` の `nets` エントリと同一。
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.export import export_ir
from griffith_logic.export.netlist_reduce import (
    CONST_TTS,
    PASSTHROUGH_TTS,
    analyze,
    class_scores,
    flat_netlist_from_ir,
    op_from_tt,
    reduce_netlist,
    tt_from_op,
    verify_equivalence,
)
from griffith_logic.export.optimize import tt_eval
from griffith_logic.ir import simulate

# ============================================================
# ヘルパー
# ============================================================


def _net(input_bits, layers, classes, tau=1.0):
    """平坦ネットリストを組み立てる(group_size は最終層幅から決まる)。"""
    width = len(layers[-1])
    assert width % classes == 0
    return {
        'input_bits': input_bits,
        'binarization': 'test',
        'layers': layers,
        'output': {'classes': classes, 'group_size': width // classes,
                   'tau': tau},
    }


def _g(op, a, b):
    return {'op': op, 'a': a, 'b': b}


def _all_inputs(input_bits):
    """2^input_bits 通りの全入力パターン (2^n, n) uint8。"""
    n = 1 << input_bits
    idx = np.arange(n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(input_bits)[None, :]) & 1)
    return bits.astype(np.uint8)


def _assert_counts(stats, **expected):
    """内訳の指定キーだけを厳密照合する。"""
    for key, value in expected.items():
        assert stats[key] == value, (
            f'{key}: 期待 {value} != 実測 {stats[key]}')


def _random_net(rng, input_bits, width, depth, classes):
    layers = []
    prev_w = input_bits
    for _ in range(depth):
        layers.append([_g(int(rng.integers(0, 16)),
                          int(rng.integers(0, prev_w)),
                          int(rng.integers(0, prev_w)))
                       for _ in range(width)])
        prev_w = width
    return _net(input_bits, layers, classes)


# ============================================================
# op ⇔ tt(16 通り)
# ============================================================


class TestOpTruthTableMapping:
    """P0 の op 符号化と IR の真理値表整数の相互変換。"""

    def test_op_to_tt_16_patterns(self):
        """全 16 op × 全 4 入力で tt_eval が P0 の式 (op>>(3-(2A+B)))&1 と一致。"""
        for op in range(16):
            tt = tt_from_op(op)
            for a in (0, 1):
                for b in (0, 1):
                    expected = (op >> (3 - (2 * a + b))) & 1
                    assert tt_eval(tt, a, b) == expected, (
                        f'op={op} (A,B)=({a},{b}): '
                        f'tt_eval={tt_eval(tt, a, b)} != {expected}')

    def test_mapping_is_bijective_involution(self):
        """op → tt は 16 通りの全単射で、かつ対合(2 回適用で元に戻る)。"""
        tts = [tt_from_op(op) for op in range(16)]
        assert sorted(tts) == list(range(16))
        for op in range(16):
            assert op_from_tt(tt_from_op(op)) == op

    def test_special_op_sets(self):
        """pass-through(op 3/5/10/12)・定数(op 0/15)の tt 集合。"""
        assert tuple(tt_from_op(op) for op in (3, 5, 10, 12)) == PASSTHROUGH_TTS
        assert tuple(tt_from_op(op) for op in (0, 15)) == CONST_TTS


# ============================================================
# R1 pass-through 畳み込み / 否定吸収
# ============================================================


class TestR1Passthrough:
    def test_native_passthrough_folds_to_wire(self):
        """op 3(=A)は配線に置換され、ゲートとして数えられない。

        期待内訳: total 2 = pass-through 1(native)+ new 1、実効 1。
        出力の 1 本は入力ビットを直接読む(output_input_refs=1)。
        """
        net = _net(2, [[_g(3, 0, 1), _g(1, 0, 1)]], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=2, passthrough_folded=1,
            passthrough_folded_native=1, passthrough_folded_induced=0,
            native_passthrough_ops=1, new_gates=1, effective_gates=1,
            const_folded=0, same_input_folded=0, cse_merged=0,
            negation_absorbed=0, output_input_refs=1, output_wire_refs=2,
            output_inverted_refs=0, output_bias_consts=0)
        assert r.stats['effective_ratio'] == 0.5
        assert verify_equivalence(net, r, _all_inputs(2))

    def test_negation_absorbed_into_downstream_truth_table(self):
        """op 12(=¬A)は極性ビットになり、下流の真理値表へ吸収される(NOT は 0 ゲート)。

        期待内訳: pass-through 2(native)、否定吸収 2、new 2、実効 2。
        """
        net = _net(2, [
            [_g(12, 0, 1), _g(3, 1, 0)],
            [_g(1, 0, 1), _g(6, 0, 1)],
        ], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=4, passthrough_folded=2,
            passthrough_folded_native=2, negation_absorbed=2,
            new_gates=2, effective_gates=2, const_folded=0,
            same_input_folded=0, cse_merged=0)
        assert verify_equivalence(net, r, _all_inputs(2))

    def test_inverted_output_ref_becomes_bias_and_minus_one(self):
        """最終層が極性反転参照に解決したら「バイアス +1、係数 −1」で表す。"""
        net = _net(2, [[_g(12, 0, 1), _g(10, 0, 1)]], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=2, passthrough_folded=2,
            passthrough_folded_native=2, new_gates=0, effective_gates=0,
            output_inverted_refs=2, output_input_refs=2, output_wire_refs=2)
        assert r.bias == [1, 1]
        assert all(sign == -1 for (_c, _r, sign) in r.terms)
        assert verify_equivalence(net, r, _all_inputs(2))


# ============================================================
# R2 定数畳み込み
# ============================================================


class TestR2ConstFold:
    def test_native_const_ops(self):
        """op 0/15 は定数に畳まれ、クラス和のバイアスになる。"""
        net = _net(2, [[_g(0, 0, 1), _g(15, 0, 1)]], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=2, const_folded=2, const_folded_native=2,
            const_folded_induced=0, native_const_ops=2, new_gates=0,
            effective_gates=0, output_bias_consts=2, output_bias_ones=1,
            output_wire_refs=0)
        assert r.stats['effective_ratio'] == 0.0
        assert r.bias == [0, 1]
        assert verify_equivalence(net, r, _all_inputs(2))

    def test_induced_const_and_induced_passthrough(self):
        """片側定数のゲートは 1 入力関数へ退化し、定数 or 配線へ連鎖する。

        L1 の 2 ゲートはどちらも定数入力を持つ(const_input_gates=2)。
        AND は定数 0 で定数へ、OR は定数 0 で配線へ落ちる。
        """
        net = _net(2, [
            [_g(0, 0, 1), _g(3, 1, 0)],
            [_g(1, 0, 1), _g(7, 0, 1)],
        ], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=4, const_folded=2, const_folded_native=1,
            const_folded_induced=1, passthrough_folded=2,
            passthrough_folded_native=1, passthrough_folded_induced=1,
            const_input_gates=2, new_gates=0, effective_gates=0,
            output_bias_consts=1, output_bias_ones=0, output_input_refs=1)
        assert verify_equivalence(net, r, _all_inputs(2))


# ============================================================
# R3 同一入力畳み込み
# ============================================================


class TestR3SameInput:
    def test_same_input_folds_to_const_or_wire(self):
        """a==b のゲートは f(a,a) の 1 入力関数。XOR→定数 0、AND→配線。"""
        net = _net(2, [[_g(6, 0, 0), _g(1, 1, 1)]], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=2, same_input_folded=2, const_folded=1,
            const_folded_induced=1, const_folded_native=0,
            passthrough_folded=1, passthrough_folded_induced=1,
            passthrough_folded_native=0, const_input_gates=0,
            new_gates=0, effective_gates=0, output_bias_consts=1,
            output_bias_ones=0, output_input_refs=1)
        assert verify_equivalence(net, r, _all_inputs(2))


# ============================================================
# R4 CSE(可換正規化つき・ネット全体で 1 dict)
# ============================================================


class TestR4Cse:
    def test_identical_gates_merge(self):
        net = _net(2, [[_g(1, 0, 1), _g(1, 0, 1)]], classes=2)
        r = reduce_netlist(net)
        _assert_counts(r.stats, total_gates=2, cse_merged=1, new_gates=1,
                       effective_gates=1)
        assert r.stats['effective_ratio'] == 0.5
        assert verify_equivalence(net, r, _all_inputs(2))

    def test_commutative_normalization_merges_swapped_operands(self):
        """可換 op(AND)のオペランド入替は正規化されて CSE で統合される。"""
        net = _net(2, [[_g(1, 0, 1), _g(1, 1, 0)]], classes=2)
        r = reduce_netlist(net)
        _assert_counts(r.stats, total_gates=2, cse_merged=1, new_gates=1)
        assert verify_equivalence(net, r, _all_inputs(2))

    def test_non_commutative_swap_absorbed_into_truth_table(self):
        """非可換 op は真理値表の A/B 入替で吸収される(op2(a,b) == op4(b,a))。"""
        net = _net(2, [[_g(2, 0, 1), _g(4, 1, 0)]], classes=2)
        r = reduce_netlist(net)
        _assert_counts(r.stats, total_gates=2, cse_merged=1, new_gates=1)
        assert verify_equivalence(net, r, _all_inputs(2))

    def test_cse_dict_spans_layers(self):
        """CSE の dict はネット全体で 1 つ。層をまたいで統合する。

        L0 の AND(x0,x1) と L1 の AND(pass-through x0, pass-through x1) が
        同じ正規形になるので 1 件統合される。
        """
        net = _net(2, [
            [_g(1, 0, 1), _g(3, 0, 1), _g(5, 0, 1)],
            [_g(1, 1, 2), _g(6, 0, 1)],
        ], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=5, passthrough_folded=2,
            passthrough_folded_native=2, cse_merged=1, new_gates=2,
            effective_gates=2, structurally_unreferenced=0)
        # 層 1 の cse_merged が層をまたいだ統合であること
        assert r.stats['per_layer'][1]['cse_merged'] == 1
        assert r.stats['per_layer'][0]['cse_merged'] == 0
        assert verify_equivalence(net, r, _all_inputs(2))


# ============================================================
# R5 未参照除去(2 指標の分離)
# ============================================================


class TestR5Unreferenced:
    def test_structurally_unreferenced_counts_original_net(self):
        """縮約前の原ネットで出力から到達しないゲートを数える指標。

        L0 の OR(x0,x2) は L1 のどちらからも参照されない。この 1 件は
        縮約後も到達しないので unreachable_after_fold にも計上される
        (2 指標は排他ではない。分離の意味は次のテストが示す)。
        """
        net = _net(3, [
            [_g(1, 0, 1), _g(6, 1, 2), _g(7, 0, 2)],
            [_g(6, 0, 1), _g(9, 0, 1)],
        ], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=5, reachable_orig=4,
            structurally_unreferenced=1, structural_fanout0=1,
            new_gates=5, unreachable_after_fold=1, effective_gates=4)
        assert r.stats['per_layer_structurally_unreferenced'] == [1, 0]
        assert r.stats['per_layer_structural_fanout0'] == [1, 0]
        assert verify_equivalence(net, r, _all_inputs(3))

    def test_unreachable_after_fold_is_separate_metric(self):
        """畳み込みの結果として到達しなくなったゲートは別勘定。

        原ネットでは L0 の 2 ゲートとも L1 から参照される
        (structurally_unreferenced=0)が、L1 の定数畳み込みで L0 の
        AND ゲートが出力から切れる(unreachable_after_fold=1)。
        """
        net = _net(2, [
            [_g(1, 0, 1), _g(6, 0, 1)],
            [_g(0, 0, 1), _g(3, 1, 0)],
        ], classes=2)
        r = reduce_netlist(net)
        _assert_counts(
            r.stats, total_gates=4, structurally_unreferenced=0,
            reachable_orig=4, new_gates=2, unreachable_after_fold=1,
            effective_gates=1, const_folded=1, const_folded_native=1)
        assert verify_equivalence(net, r, _all_inputs(2))

    def test_random_wiring_has_structurally_unreferenced_gates(self):
        """ランダム配線では構造未参照が出る(seed 固定で決定的)。"""
        rng = np.random.default_rng(0)
        net = _random_net(rng, input_bits=16, width=64, depth=3, classes=4)
        r = reduce_netlist(net)
        assert r.stats['structurally_unreferenced'] > 0
        assert r.stats['structural_fanout0'] > 0

    def test_bijective_wiring_has_zero_structurally_unreferenced(self):
        """全単射的な配線(a=i, b=(i+1)%w)では構造未参照が厳密に 0。"""
        width = 32
        layers = []
        for li in range(3):
            prev_w = 16 if li == 0 else width
            layers.append([_g(1 + (i % 14), i % prev_w, (i + 1) % prev_w)
                           for i in range(width)])
        net = _net(16, layers, classes=4)
        r = reduce_netlist(net)
        assert r.stats['structurally_unreferenced'] == 0
        assert r.stats['structural_fanout0'] == 0
        assert r.stats['per_layer_structurally_unreferenced'] == [0, 0, 0]


# ============================================================
# 内訳の不変量とフィールド名の契約
# ============================================================


class TestBreakdownContract:
    """統計のキー名は P0 `p0_results.json` の `nets` エントリと同一にする。"""

    P0_KEYS = (
        'passthrough_folded', 'passthrough_folded_native',
        'passthrough_folded_induced', 'negation_absorbed',
        'const_folded', 'const_folded_native', 'const_folded_induced',
        'same_input_folded', 'const_input_gates', 'cse_merged', 'new_gates',
        'native_passthrough_ops', 'native_const_ops',
        'output_bias_consts', 'output_bias_ones', 'output_wire_refs',
        'output_inverted_refs', 'output_input_refs',
        'total_gates', 'effective_gates', 'unreachable_after_fold',
        'effective_ratio', 'reduction_ratio', 'per_layer', 'class_bias',
        'widths', 'input_bits', 'classes', 'group_size',
        'reachable_orig', 'structurally_unreferenced', 'structural_fanout0',
        'per_layer_structurally_unreferenced', 'per_layer_structural_fanout0',
    )
    P0_PER_LAYER_KEYS = (
        'n_gates', 'passthrough_folded', 'const_folded', 'same_input_folded',
        'negation_absorbed', 'cse_merged', 'new_gates', 'const_input_gates',
        'effective_gates', 'unreachable_after_fold',
    )

    def test_all_p0_fields_present(self):
        rng = np.random.default_rng(3)
        net = _random_net(rng, input_bits=12, width=16, depth=3, classes=4)
        stats = reduce_netlist(net).stats
        assert set(self.P0_KEYS) <= set(stats)
        for lc in stats['per_layer']:
            assert set(self.P0_PER_LAYER_KEYS) == set(lc)

    @pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
    def test_breakdown_sums_to_total(self, seed):
        """const + pass-through + CSE + new == 総ゲート数(P0 の内部 assert)。"""
        rng = np.random.default_rng(seed)
        net = _random_net(rng, input_bits=10, width=24, depth=3, classes=4)
        s = reduce_netlist(net).stats
        assert (s['const_folded'] + s['passthrough_folded'] + s['cse_merged']
                + s['new_gates']) == s['total_gates']
        assert s['new_gates'] - s['unreachable_after_fold'] == s['effective_gates']
        assert (s['passthrough_folded_native'] + s['passthrough_folded_induced']
                == s['passthrough_folded'])
        assert (s['const_folded_native'] + s['const_folded_induced']
                == s['const_folded'])
        assert sum(lc['effective_gates'] for lc in s['per_layer']) \
            == s['effective_gates']


# ============================================================
# 縮約前後の等価性(seed 固定のランダム小ネット)
# ============================================================


class TestReductionEquivalence:
    @pytest.mark.parametrize('width', [8, 16, 32, 64])
    @pytest.mark.parametrize('depth', [2, 3, 4])
    def test_random_nets_equivalent_before_after(self, width, depth):
        """幅 8〜64・深さ 2〜4 のランダムネットで縮約前後が整数完全一致。

        入力 10 bit は全数(1,024 通り)掃引するので、等価性は網羅的に確かめる。
        """
        rng = np.random.default_rng(1000 * width + depth)
        net = _random_net(rng, input_bits=10, width=width, depth=depth,
                          classes=4)
        r = reduce_netlist(net)
        assert verify_equivalence(net, r, _all_inputs(10))

    def test_analyze_reports_pass(self):
        rng = np.random.default_rng(7)
        net = _random_net(rng, input_bits=12, width=32, depth=3, classes=4)
        x = _all_inputs(12)
        res = analyze(net, X_data=x, n_random=500, seed=0)
        assert res['equivalence']['overall'] == 'pass'
        assert res['equivalence']['data'] == 'pass'
        assert res['equivalence']['data_n'] == 4096
        assert res['equivalence']['random'] == 'pass'
        assert res['equivalence']['data_mismatch_rows'] == 0
        assert res['equivalence']['random_mismatch_rows'] == 0
        assert res['reduce_time_s'] >= 0.0

    def test_analyze_without_data_skips(self):
        rng = np.random.default_rng(8)
        net = _random_net(rng, input_bits=10, width=16, depth=2, classes=2)
        res = analyze(net, n_random=256, seed=1)
        assert res['equivalence']['data'] == 'skip'
        assert res['equivalence']['overall'] == 'pass'


# ============================================================
# IR アダプタ(LutLayer モデル → export_ir → 平坦ネットリスト)
# ============================================================


def _lut_model(seed=31, n_inputs=2):
    torch.manual_seed(seed)
    return nn.Sequential(
        LutLayer(16, 24, n_inputs=n_inputs, parameterization='softmax'),
        LutLayer(24, 24, n_inputs=n_inputs, parameterization='softmax'),
        GroupSum(k=4, tau=2.5),
    )


class TestIrAdapter:
    def test_flat_netlist_shape_and_metadata(self):
        ir = export_ir(_lut_model(), input_width=16, name='adapter')
        net = flat_netlist_from_ir(ir, tau=2.5, binarization='test-bits')
        assert net['input_bits'] == 16
        assert net['binarization'] == 'test-bits'
        assert [len(layer) for layer in net['layers']] == [24, 24]
        assert net['output'] == {'classes': 4, 'group_size': 6, 'tau': 2.5}
        assert all(0 <= g['op'] < 16 for layer in net['layers'] for g in layer)

    def test_tau_defaults_to_group_sum_param(self):
        ir = export_ir(_lut_model(), input_width=16)
        net = flat_netlist_from_ir(ir)
        assert net['output']['tau'] == 2.5

    @pytest.mark.parametrize('seed', [11, 31, 57])
    def test_original_eval_matches_ir_simulate(self, seed):
        """アダプタ経由の原ネット評価が ir.simulate と bit 一致(golden 照合)。"""
        model = _lut_model(seed=seed)
        ir = export_ir(model, input_width=16, name='golden')
        net = flat_netlist_from_ir(ir, tau=2.5)

        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=(256, 16), dtype=np.uint8)
        golden = simulate(ir, x.astype(np.int64))
        got = class_scores(net, x)
        assert got.shape == golden.shape
        assert np.array_equal(got, golden), (
            f'ir.simulate と不一致: 行 {int((got != golden).any(axis=1).sum())} 本')

    @pytest.mark.parametrize('seed', [11, 31, 57])
    def test_reduced_net_matches_ir_simulate(self, seed):
        """縮約後 DAG も ir.simulate と bit 一致する(縮約の等価性)。"""
        model = _lut_model(seed=seed)
        ir = export_ir(model, input_width=16)
        net = flat_netlist_from_ir(ir, tau=2.5)
        r = reduce_netlist(net)

        rng = np.random.default_rng(seed + 100)
        x = rng.integers(0, 2, size=(256, 16), dtype=np.uint8)
        assert np.array_equal(r.class_scores(x), simulate(ir, x.astype(np.int64)))
        assert verify_equivalence(net, r, x)

    def test_n_inputs_not_2_raises(self):
        ir = export_ir(_lut_model(n_inputs=4), input_width=16)
        with pytest.raises(NotImplementedError, match='n_inputs'):
            flat_netlist_from_ir(ir, tau=2.5)

    def test_missing_group_sum_raises(self):
        torch.manual_seed(5)
        model = nn.Sequential(LutLayer(16, 24, n_inputs=2))
        ir = export_ir(model, input_width=16)
        with pytest.raises(NotImplementedError, match='group_sum'):
            flat_netlist_from_ir(ir, tau=1.0)
