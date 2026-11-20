"""L1: オーバーレイ生成器(配線判別・op 符号・.mem・資源見積り)の検証。

シミュレータもボードも要らない層。ここで見るのは
  - IR の indices と閉形式(rot / 引き伸ばし rot)の全ゲート一致
  - 壊れた配線を本当に拒否するか(検証器自体の検証)
  - op 4bit 符号が RTL の gate_eval と同一意味論か(16 種の往復)
  - opmem の全ニブルが IR のゲートと一致するか
  - サイクル式と M10K 見積りが先行実験の実測アンカーを再現するか
"""

import numpy as np
import pytest
from conftest import build_rot_ffn, build_rot_tiny, build_stretched_tiny

from griffith_logic.ir.serialize import ir_from_dict
from griffith_logic.ir.weights import gate_id_from_tt_n2
from mdl_fpga.hdlgen.overlay import (
    OverlayUnsupported,
    check_opmem,
    detect_layer,
    emit_addrinit,
    emit_opmem,
    estimate_gate_rom_bits,
    estimate_m10k,
    plan_layer,
    plan_overlay,
    rom32_blocks,
    simulate_layer,
    verify_wiring,
)

# 先行実験 P6(層 0 のみ 2D 近傍、隠れ層は幾何級数)のシフト量
P6_SA = [1, 0, 0, 0, 0, 0]
P6_SB = [28, 4, 20, 90, 400, 1800]


@pytest.fixture(scope='module')
def graph_48k():
    """48K ゲート(w8000 x d6)の MNIST 相当ネット(閉形式配線)。"""
    return ir_from_dict(build_rot_ffn(8000, 6, 784, 10, P6_SA, P6_SB))


@pytest.fixture(scope='module')
def graph_96k():
    """96K ゲート(w16000 x d6)。"""
    return ir_from_dict(build_rot_ffn(16000, 6, 784, 10, P6_SA, P6_SB))


# ---------------------------------------------------------------- 配線判別


def test_wiring_matches_indices_all_layers(graph_48k):
    """全層 rot の 48K ネットで indices と閉形式が全ゲート一致する。"""
    plan = plan_overlay(graph_48k, v=8)
    assert [ly.kind for ly in plan.layers] == ['rot'] * 6
    assert [ly.s_a for ly in plan.layers] == P6_SA
    assert [ly.s_b for ly in plan.layers] == P6_SB
    # 隠れ層は W_in == W_out なので歩幅は (V, 0) に縮退する
    assert [(ly.int_step, ly.rem_step) for ly in plan.layers[1:]] == \
        [(8, 0)] * 5
    stats = verify_wiring(plan, graph_48k)
    assert stats['addr_entries_checked'] == 2 * 48000


def test_wiring_matches_indices_96k(graph_96k):
    plan = plan_overlay(graph_96k, v=8)
    assert verify_wiring(plan, graph_96k)['addr_entries_checked'] == 2 * 96000


@pytest.mark.parametrize('layer,gate', [(0, 17), (3, 4001)])
def test_broken_indices_rejected(layer, gate):
    """1 ゲートだけ配線を壊した IR を拒否するか(検証器自体の検証)。"""
    doc = build_rot_ffn(8000, 6, 784, 10, P6_SA, P6_SB)
    idx = doc['nodes'][layer]['params']['indices']
    idx[0][gate] = (idx[0][gate] + 1) % doc['nodes'][layer]['params']['in_width']
    with pytest.raises(OverlayUnsupported, match='overlay 化できない'):
        plan_overlay(ir_from_dict(doc), v=8)


def test_random_wiring_rejected():
    """学習した任意配線(乱択)は計算型でないので拒否する。"""
    doc = build_rot_ffn(320, 3, 64, 2, [1, 0, 0], [3, 5, 7])
    rng = np.random.default_rng(0)
    doc['nodes'][1]['params']['indices'] = \
        rng.integers(0, 320, size=(2, 320)).tolist()
    with pytest.raises(OverlayUnsupported):
        plan_overlay(ir_from_dict(doc), v=8)


def test_mixed_kind_layer_rejected():
    """同一層で a が rot・b が引き伸ばし rot の混在は拒否する。"""
    w_in, w_out = 22, 32
    g = np.arange(w_out, dtype=np.int64)
    row_rot = (g % w_in + 1) % w_in
    row_str = ((2 * g * w_in + w_out) // (2 * w_out) + 3) % w_in
    with pytest.raises(OverlayUnsupported, match='配線規則が異なる'):
        detect_layer(np.stack([row_rot, row_str]), w_in, w_out)


@pytest.mark.parametrize('w_in,w_out,s_a,s_b,v', [
    (9216, 8000, 3, 96, 8),     # CIFAR SR2 型(層 0 の引き伸ばし rot)
    (9216, 8000, 3, 96, 32),
    (16, 8, 0, 1, 8),           # 出力が入力より狭い「穴」のケース
    (784, 8000, 1, 28, 8),      # 整数歩幅 0(余りだけで進む)
    (100, 64, 5, 7, 8),
])
def test_stretched_rot_exact(w_in, w_out, s_a, s_b, v):
    """引き伸ばし rot の閉形式とアキュムレータ再現値が全ゲート一致する。"""
    g = np.arange(w_out, dtype=np.int64)
    base = (2 * g * w_in + w_out) // (2 * w_out)
    idx = np.stack([(base + s_a) % w_in, (base + s_b) % w_in])
    kind, got_a, got_b = detect_layer(idx, w_in, w_out)
    assert kind == 'stretched_rot' or w_in == w_out
    assert (got_a, got_b) == (s_a % w_in, s_b % w_in)
    layer = plan_layer(idx, w_in, w_out, v)
    a_seq, b_seq = simulate_layer(layer, v)
    assert np.array_equal(a_seq, idx[0])
    assert np.array_equal(b_seq, idx[1])


def test_rot_layer_detected_as_rot():
    """W_in == W_out では両規則が一致するため rot と判定する(歩幅が単純)。"""
    w = 320
    g = np.arange(w, dtype=np.int64)
    idx = np.stack([(g + 4) % w, (g + 20) % w])
    assert detect_layer(idx, w, w)[0] == 'rot'


def _with_passthrough(doc: dict, node_type: str, **params) -> dict:
    """入力と層 0 の間に素通しノードを差し込む。"""
    in_w = doc['inputs'][0]['width']
    base = {'width': in_w}
    base.update(params)
    doc['nodes'].insert(0, {'id': 'pre', 'type': node_type, 'params': base})
    for e in doc['edges']:
        if e['src'] == 'x' and e['dst'] == 'l0':
            e['dst'] = 'pre'
    doc['edges'].insert(1, {'src': 'pre', 'dst': 'l0'})
    return doc


@pytest.mark.parametrize('node_type,params', [
    ('flatten', {'in_shape': [24]}),
    ('binarize', {'threshold': 0.5}),
])
def test_input_passthrough_accepted(node_type, params):
    """入力側の flatten / binarize は素通しとして受け入れ、警告に残す。"""
    doc = _with_passthrough(build_rot_tiny(), node_type, **params)
    plan = plan_overlay(ir_from_dict(doc), v=8)
    assert plan.depth == 3 and plan.in_bits == 24
    assert any('素通し' in w for w in plan.warnings)


def test_binarize_threshold_outside_unit_interval_rejected():
    """0/1 入力に対して恒等でない binarize は拒否する。"""
    doc = _with_passthrough(build_rot_tiny(), 'binarize', threshold=1.5)
    with pytest.raises(OverlayUnsupported, match='恒等でない'):
        plan_overlay(ir_from_dict(doc), v=8)


# ---------------------------------------------------------------- op 符号


def test_op_code_matches_gate_eval_semantics():
    """16 種すべてで IR の真理値表 ⇔ RTL gate_eval(op[3-{A,B}])が一致する。"""
    codes = []
    for tt in range(16):
        op = gate_id_from_tt_n2(tt)
        codes.append(op)
        for a in (0, 1):
            for b in (0, 1):
                rtl = (op >> (3 - (2 * a + b))) & 1   # gate_eval.v と同式
                ir = (tt >> (a + 2 * b)) & 1          # ir.simulate と同式
                assert rtl == ir, (tt, op, a, b)
    assert sorted(codes) == list(range(16))           # 全単射(往復可能)


def test_opmem_nibbles_match_gates():
    """opmem の全ニブルが IR のゲート op と一致する(語詰めの向きを含む)。"""
    plan = plan_overlay(ir_from_dict(build_rot_tiny()), v=8)
    roms = emit_opmem(plan)
    assert len(roms) == plan.nrom == 1
    assert len(roms[0]) == plan.gate_words
    assert check_opmem(plan, roms) == plan.total_gates
    # 語 0 のレーン k は bits[4k+3:4k](lane0 = LSB)
    word0 = int(roms[0][0], 16)
    for k in range(plan.v):
        assert (word0 >> (4 * k)) & 0xF == int(plan.ops[k])


def test_opmem_v32_lane_packing():
    """V=32 では 32bit ROM 4 本に レーン 8r..8r+7 が分かれる。"""
    plan = plan_overlay(ir_from_dict(
        build_rot_ffn(320, 3, 128, 10, [1, 0, 0], [3, 5, 7])), v=32)
    roms = emit_opmem(plan)
    assert plan.nrom == 4 and len(roms) == 4
    assert check_opmem(plan, roms) == plan.total_gates


def test_addrinit_tables():
    """初期表は層 x レーンの行を持ち、rot 層の余りは 0。"""
    plan = plan_overlay(ir_from_dict(build_stretched_tiny()), v=8)
    mems = emit_addrinit(plan)
    for name in ('addrinit_a.mem', 'addrinit_b.mem', 'reminit.mem'):
        lines = mems[name].strip().split('\n')
        assert len(lines) == plan.depth * plan.v
    rem = [int(x, 16) for x in mems['reminit.mem'].strip().split('\n')]
    assert rem[plan.v:] == [0] * (plan.v * (plan.depth - 1))   # 隠れ層は rot
    assert any(r != 0 for r in rem[:plan.v])                   # 層 0 は引き伸ばし
    init_a = [int(x, 16) for x in mems['addrinit_a.mem'].strip().split('\n')]
    assert init_a[:plan.v] == list(plan.layers[0].init_a)


# ---------------------------------------------------------------- 見積り


@pytest.mark.parametrize('depth_words,blocks', [
    (6000, 32),    # 先行実験 48K V=8(1 本)
    (12000, 48),   # 先行実験 96K V=8(1 本)
    (3000, 16),    # 先行実験 48K V=16(2 本 = 32 blk)
    (1500, 6),     # 先行実験 48K V=32(4 本 = 26 blk。1 本だけ 9 blk だった)
])
def test_rom32_blocks_calibration(depth_words, blocks):
    """32bit ROM の M10K 数が先行実験の fit 実測を再現する。"""
    assert rom32_blocks(depth_words) == blocks


def test_m10k_estimate_48k(graph_48k):
    """48K: ゲート ROM 32 blk(C1 の ±0 対象)、総数は実測 125 の ±10%。"""
    plan = plan_overlay(graph_48k, v=8)
    est = estimate_m10k(plan, n_images=1000)
    assert est['op_rom'] == 32
    assert abs(est['total'] - 125) <= 12.5


def test_m10k_estimate_96k(graph_96k):
    """96K: ゲート ROM 48 blk、総数は実測 147 の ±10%。"""
    plan = plan_overlay(graph_96k, v=8)
    est = estimate_m10k(plan, n_images=1000)
    assert est['op_rom'] == 48
    assert abs(est['total'] - 147) <= 14.7


def test_cycle_formula_anchors(graph_48k, graph_96k):
    """サイクル式が先行実験実機の 6,129 / 12,129 を再現する。"""
    assert plan_overlay(graph_48k, v=8).cycles_per_image == 6129
    assert plan_overlay(graph_96k, v=8).cycles_per_image == 12129


def test_gate_rom_bits_ratio(graph_48k):
    """4bit/ゲート化のデータ量比(保存型 32bit との比 = 8x)。"""
    plan = plan_overlay(graph_48k, v=8)
    bits = estimate_gate_rom_bits(plan)
    assert bits['computed_bits'] == 48000 * 4
    assert bits['stored_word_bits_minimal'] == 30  # op 4 + アドレス 13 x 2
    assert bits['stored_word_bits'] == 32          # 2 の冪へ切上 = 先行実験の形式
    assert bits['ratio'] == pytest.approx(8.0)


# ---------------------------------------------------------------- 構成の拒否


def test_v_must_be_multiple_of_8():
    doc = build_rot_tiny()
    with pytest.raises(OverlayUnsupported, match='V=4'):
        plan_overlay(ir_from_dict(doc), v=4)


def test_non_uniform_width_rejected():
    doc = build_rot_ffn(32, 3, 24, 2, [1, 0, 0], [3, 5, 7])
    node = doc['nodes'][2]['params']
    node['out_width'] = 16
    node['indices'] = [row[:16] for row in node['indices']]
    node['truth_tables'] = node['truth_tables'][:16]
    doc['nodes'][-1]['params']['in_width'] = 16
    with pytest.raises(OverlayUnsupported, match='一様'):
        plan_overlay(ir_from_dict(doc), v=8)


def test_group_not_divisible_rejected():
    """group_size が V で割り切れない構成(集約融合の前提)は拒否する。"""
    doc = build_rot_ffn(40, 2, 24, 2, [1, 0], [3, 5])
    with pytest.raises(OverlayUnsupported, match='group_size'):
        plan_overlay(ir_from_dict(doc), v=8)


def test_input_wider_than_layer_rejected():
    """入力が層幅より広いと活性バッファに画像が入らない。"""
    doc = build_rot_ffn(32, 2, 64, 2, [1, 0], [3, 5])
    with pytest.raises(OverlayUnsupported, match='活性バッファ'):
        plan_overlay(ir_from_dict(doc), v=8)


def test_act_words_not_power_of_two_warns():
    """W/V が 2 の冪から大きく外れる構成は警告する(先行実験の設計則)。

    無駄が 10% 未満(例 W/V=1000 → 1024 語 = 2.4%)なら警告しない。
    """
    plan = plan_overlay(ir_from_dict(
        build_rot_ffn(48, 2, 24, 2, [1, 0], [3, 5])), v=8)
    assert plan.act_words == 6          # 8 語ぶん確保 = 無駄 33%
    assert any('2 の冪' in w for w in plan.warnings)
    # W/V = 1000 → 1024 語(無駄 2.4%)は警告しない
    quiet = plan_overlay(ir_from_dict(
        build_rot_ffn(8000, 2, 784, 10, [1, 0], [28, 4])), v=8)
    assert quiet.act_words == 1000
    assert not any('2 の冪' in w for w in quiet.warnings)
