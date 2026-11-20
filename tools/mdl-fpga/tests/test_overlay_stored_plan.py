"""L1: 保存型(自由配線)オーバーレイ生成器の検証。

シミュレータもボードも要らない層。ここで見るのは
  - ゲート記録 {op 4bit, a, b} の詰め込み往復(op 16 種 x アドレス境界)
  - 置換配線・ランダム配線の 48K 相当 IR が保存型で受理され、計算型では
    拒否されること(方式の切り分けが実際に効くか)
  - wiring='auto' の判定(rot なら計算型、自由配線なら保存型、明示で上書き)
  - .mem のレーン割付と、復号 ⇔ IR の全数照合(検証器自体の検証を含む)
  - M10K 見積りとサイクル式が先行実験の実測アンカーを再現するか
"""

import numpy as np
import pytest
from conftest import build_free_ffn, build_random_tiny, build_rot_ffn

from griffith_logic.ir.serialize import ir_from_dict
from mdl_fpga.hdlgen.overlay import (
    OverlayUnsupported,
    check_gatemem,
    emit_gatemem,
    estimate_gate_rom_bits,
    estimate_m10k,
    generate_overlay,
    pack_record,
    plan_overlay,
    record_format,
    rom_blocks,
    unpack_record,
    wiring_label,
)

# 先行実験 P6 のシフト量(計算型の対照)
P6_SA = [1, 0, 0, 0, 0, 0]
P6_SB = [28, 4, 20, 90, 400, 1800]


@pytest.fixture(scope='module')
def random_48k():
    """48K ゲート(w8000 x d6)の無制約ランダム配線ネット(先行実験 C1 と同形)。"""
    return ir_from_dict(build_free_ffn(8000, 6, 784, 10, kind='random'))


@pytest.fixture(scope='module')
def perm_48k():
    """同じ形で、隠れ層が置換配線(全単射)のネット。"""
    return ir_from_dict(build_free_ffn(8000, 6, 784, 10, kind='perm'))


# ---------------------------------------------------------------- 記録の往復


@pytest.mark.parametrize('field_bits', [14, 18])
def test_record_roundtrip_all_ops_and_bounds(field_bits):
    """op 16 種 x アドレス境界で詰め込み ⇔ 復号が完全に往復する。"""
    hi = (1 << field_bits) - 1
    for op in range(16):
        for a in (0, 1, hi - 1, hi):
            for b in (0, hi):
                word = pack_record(op, a, b, field_bits)
                assert word < (1 << (4 + 2 * field_bits))
                assert unpack_record(word, field_bits) == (op, a, b)


def test_record_field_layout_matches_exp013():
    """32bit のとき {op[31:28], a[27:14], b[13:0]} = 先行実験のビット配置。"""
    word = pack_record(0xA, 0x1234, 0x0567, 14)
    assert word == (0xA << 28) | (0x1234 << 14) | 0x0567
    assert (word >> 28) & 0xF == 0xA
    assert (word >> 14) & 0x3FFF == 0x1234
    assert word & 0x3FFF == 0x0567


@pytest.mark.parametrize('op,a,b', [(16, 0, 0), (0, 1 << 14, 0),
                                    (0, 0, 1 << 14), (0, -1, 0)])
def test_record_rejects_out_of_range(op, a, b):
    with pytest.raises(OverlayUnsupported):
        pack_record(op, a, b, 14)


@pytest.mark.parametrize('max_index,bits,field', [
    (783, 32, 14), (7999, 32, 14), (16383, 32, 14),   # W <= 16,384 は先行実験形式
    (16384, 40, 18), (99999, 40, 18),                 # 超えたら 40bit へ
])
def test_record_format_selection(max_index, bits, field):
    assert record_format(max_index) == (bits, field)


def test_record_format_rejects_too_wide():
    with pytest.raises(OverlayUnsupported, match='記録幅に収まらない'):
        record_format(1 << 18)


# ---------------------------------------------------------------- 方式の判定


def test_stored_accepts_random_wiring(random_48k):
    """無制約ランダム配線の 48K が保存型で受理される。"""
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    assert plan.wiring == 'stored'
    assert [ly.kind for ly in plan.layers] == ['stored'] * 6
    assert plan.record_bits == 32 and plan.addr_field_bits == 14
    assert plan.total_gates == 48000
    assert wiring_label(plan) == 'stored(自由配線、記録 32bit/ゲート)'


def test_stored_accepts_permutation_wiring(perm_48k):
    plan = plan_overlay(perm_48k, v=8, wiring='stored')
    assert plan.wiring == 'stored'
    # 隠れ層の a は置換(全単射)であることを IR 側で確認しておく
    assert len(set(plan.idx_a[8000:16000].tolist())) == 8000


@pytest.mark.parametrize('fixture_name', ['random_48k', 'perm_48k'])
def test_computed_rejects_free_wiring(fixture_name, request):
    """同じ IR を計算型で作ろうとすると言い切って拒否する。"""
    graph = request.getfixturevalue(fixture_name)
    with pytest.raises(OverlayUnsupported, match='overlay 化できない'):
        plan_overlay(graph, v=8, wiring='computed')


def test_auto_picks_stored_for_free_wiring(random_48k):
    """auto: 閉形式に外れたら保存型へ落ち、理由が警告に残る。"""
    plan = plan_overlay(random_48k, v=8, wiring='auto')
    assert plan.wiring == 'stored'
    assert any('保存型' in w and '閉形式' in w for w in plan.warnings)


def test_auto_picks_computed_for_rot():
    """auto: 全層が閉形式なら計算型(記述 4bit)のまま。"""
    graph = ir_from_dict(build_rot_ffn(8000, 6, 784, 10, P6_SA, P6_SB))
    plan = plan_overlay(graph, v=8, wiring='auto')
    assert plan.wiring == 'computed'
    assert plan.gate_bits == 4
    assert not any('保存型' in w for w in plan.warnings)


def test_explicit_stored_overrides_computable_wiring():
    """明示指定は上書きできる(rot のネットを保存型で載せる)。"""
    graph = ir_from_dict(build_rot_ffn(8000, 6, 784, 10, P6_SA, P6_SB))
    plan = plan_overlay(graph, v=8, wiring='stored')
    assert plan.wiring == 'stored' and plan.gate_bits == 32


def test_unknown_wiring_rejected():
    with pytest.raises(OverlayUnsupported, match='wiring'):
        plan_overlay(ir_from_dict(build_random_tiny()), v=8, wiring='table')


def test_structural_errors_still_raised_in_auto():
    """auto でも構造(一様幅)の違反は保存型へ落とさず拒否する。"""
    doc = build_free_ffn(32, 3, 24, 2)
    node = doc['nodes'][2]['params']
    node['out_width'] = 16
    node['indices'] = [row[:16] for row in node['indices']]
    node['truth_tables'] = node['truth_tables'][:16]
    doc['nodes'][-1]['params']['in_width'] = 16
    with pytest.raises(OverlayUnsupported, match='一様'):
        plan_overlay(ir_from_dict(doc), v=8, wiring='auto')


def test_broken_index_rejected_in_stored():
    """範囲外の添字は保存型でも拒否する(記録に落とす前に止める)。"""
    doc = build_free_ffn(32, 3, 24, 2)
    doc['nodes'][1]['params']['indices'][0][5] = 32     # W_in = 32 の範囲外
    with pytest.raises(OverlayUnsupported, match='範囲外'):
        plan_overlay(ir_from_dict(doc), v=8, wiring='stored')


# ---------------------------------------------------------------- .mem


def test_gatemem_lane_layout(random_48k):
    """レーン k のファイルの語 w = ゲート w*V + k(先行実験と同じ割付)。"""
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    roms = emit_gatemem(plan)
    assert len(roms) == plan.nrom == plan.v == 8
    assert all(len(lines) == plan.gate_words == 6000 for lines in roms)
    assert all(len(lines[0]) == 8 for lines in roms)     # 32bit = hex 8 桁
    for k in range(plan.v):
        op, a, b = unpack_record(int(roms[k][3], 16), plan.addr_field_bits)
        g = 3 * plan.v + k
        assert (op, a, b) == (int(plan.ops[g]), int(plan.idx_a[g]),
                              int(plan.idx_b[g]))


def test_check_gatemem_full_match(random_48k):
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    assert check_gatemem(plan, emit_gatemem(plan)) == plan.total_gates


def test_check_gatemem_detects_single_corrupt_word(random_48k):
    """記録を 1 語だけ壊すと照合が落ちる(検証器自体の検証)。"""
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    roms = emit_gatemem(plan)
    roms[2][4321] = f'{int(roms[2][4321], 16) ^ 0x1:08x}'
    with pytest.raises(OverlayUnsupported, match='レーン 2 語 4321'):
        check_gatemem(plan, roms)


def test_generate_overlay_emits_lane_mems_and_sources():
    plan = plan_overlay(ir_from_dict(build_random_tiny()), v=8,
                        wiring='stored')
    gen = generate_overlay(plan)
    assert sorted(gen.mems) == [f'gatemem_lane{k}.mem' for k in range(8)]
    assert 'lgn_overlay_stored.v' in gen.sources
    assert 'overlay_harness_stored.v' in gen.sources
    assert 'lgn_overlay.v' not in gen.sources        # 計算型は混ぜない
    assert 'opmem0.mem' not in gen.mems


# ---------------------------------------------------------------- 見積り


@pytest.mark.parametrize('width,depth_words,blocks', [
    (30, 6000, 30),    # 先行実験 48K V=8 のレーン ROM(240 blk / 8 レーン)
    (32, 6000, 32),    # 先行実験の op ROM(32bit 幅は全 bit 非定数)
    (32, 12000, 48),
    (32, 1500, 6),
])
def test_rom_blocks_calibration(width, depth_words, blocks):
    """幅を一般化した ROM 見積りが先行実験の実測を再現する。"""
    assert rom_blocks(width, depth_words) == blocks


def test_m10k_estimate_stored_48k(random_48k):
    """48K V=8 保存型: ゲート ROM 240 blk(先行実験 C1 の実測と同値)。"""
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    assert plan.record_planes == 30      # op 4 + a 13 + b 13(W=8000)
    assert plan.gate_rom_blocks == 240
    est = estimate_m10k(plan, n_images=512)
    assert est['gate_rom'] == est['op_rom'] == 240
    assert est['activation'] == 16       # 先行実験 C1 の実測 16
    assert est['image_rom'] == 40        # 784bit x 512 枚 = 先行実験の実測 40
    # 先行実験 C1 の総 M10K 296(= 240 + 16 + 40)。期待値 ROM 1 blk 分だけ多い
    assert abs(est['total'] - 296) <= 2


def test_stored_48k_does_not_fit_with_1000_images(random_48k):
    """画像 1,000 枚では M10K 308 を超える(先行実験が N=512 を選んだ理由)。"""
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    assert estimate_m10k(plan, n_images=1000)['total'] > 308


def test_cycle_formula_same_as_computed(random_48k):
    """保存型でもサイクル式は計算型と同じ(先行実験 C1 = 先行実験 D1 = 6,129)。"""
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    assert plan.cycles_per_image == 6129
    rot = plan_overlay(ir_from_dict(
        build_rot_ffn(8000, 6, 784, 10, P6_SA, P6_SB)), v=8)
    assert plan.cycles_per_image == rot.cycles_per_image


def test_gate_rom_bits_ratio_is_method_independent(random_48k):
    """記述量の比は plan の方式によらない(8x = 32bit / 4bit)。"""
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    bits = estimate_gate_rom_bits(plan)
    assert bits['computed_bits'] == 48000 * 4
    assert bits['stored_bits'] == 48000 * 32
    assert bits['ratio'] == pytest.approx(8.0)


def test_wide_record_when_layer_exceeds_16384():
    """層幅が 16,384 を超えると記録が 40bit(a/b 各 18bit)になる。"""
    plan = plan_overlay(ir_from_dict(
        build_free_ffn(20480, 2, 784, 10, kind='random')), v=8,
        wiring='stored')
    assert (plan.record_bits, plan.addr_field_bits) == (40, 18)
    roms = emit_gatemem(plan)
    assert len(roms[0][0]) == 10                      # 40bit = hex 10 桁
    assert check_gatemem(plan, roms) == plan.total_gates
    # 実効 bit-plane = 4 + 15 + 15(W=20480 は 15bit)
    assert plan.record_planes == 34


def test_lane_count_and_depth_scale_with_v():
    """ROM 本数 = V、深さ = 総ゲート数 / V(レーン分割の契約)。"""
    doc = build_free_ffn(320, 3, 128, 10, kind='random')
    for v in (8, 16, 32):
        plan = plan_overlay(ir_from_dict(doc), v=v, wiring='stored')
        assert plan.nrom == v
        assert plan.gate_rom_depth == plan.total_gates // v
        assert check_gatemem(plan, emit_gatemem(plan)) == plan.total_gates


def test_stored_plan_dict_carries_wiring(random_48k):
    """overlay_params.json(生成物の契約)に方式と記録幅が出る。"""
    d = plan_overlay(random_48k, v=8, wiring='stored').to_dict()
    assert d['WIRING'] == 'stored'
    assert d['RECORD_BITS'] == 32 and d['ADDR_F'] == 14
    assert d['GATE_ROMS'] == 8 and d['GATE_ROM_DEPTH'] == 6000
    assert d['GATE_ROM_BLOCKS'] == 240
    assert d['CYCLES_PER_IMAGE'] == 6129


def test_indices_recovered_from_plan(random_48k):
    """plan の添字列が IR の indices と全数一致する(発行順の契約)。"""
    from griffith_logic.export.lowering import lower_to_primitives
    plan = plan_overlay(random_48k, v=8, wiring='stored')
    luts = [n for n in lower_to_primitives(random_48k).topological_order()
            if n.type == 'lut_layer']
    for i, node in enumerate(luts):
        idx = np.asarray(node.params['indices'], dtype=np.int64)
        seg = slice(i * plan.w, (i + 1) * plan.w)
        assert np.array_equal(plan.idx_a[seg], idx[0])
        assert np.array_equal(plan.idx_b[seg], idx[1])
