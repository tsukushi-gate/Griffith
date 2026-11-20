"""保存型オーバーレイの検証システム(トップ生成・プロジェクト・実機)の検証。

  - 生成トップ / qsf / sdc をゴールデンで固定(既存 golden は不変のまま)
  - 計算型との差が「テンプレート名と .mem の種類だけ」であること
  - ISSP・分周・7セグ・期待値 ROM は計算型と同じ契約を満たすこと
"""

import pytest
from conftest import build_random_tiny, build_rot_tiny, requires_hw

from griffith_logic.ir.serialize import ir_from_dict
from mdl_fpga.backends.quartus import (
    parse_fit_entities,
    summarize_entity_m10k,
    write_project_files,
)
from mdl_fpga.board_db import load_board
from mdl_fpga.hdlgen.overlay import plan_overlay
from mdl_fpga.hdlgen.overlay_system import generate_overlay_system
from mdl_fpga.hdlgen.ports import TopPorts
from mdl_fpga.issp import evaluate
from test_golden_hdl import _assert_matches


def _sysgen(doc, wiring, n_images=8):
    board = load_board('de0-cv')
    plan = plan_overlay(ir_from_dict(doc), v=8, wiring=wiring)
    return generate_overlay_system(plan, n_images=n_images, clock_div=2,
                                   ports=TopPorts.from_board(board))


@pytest.fixture(scope='module')
def stored_sys():
    return _sysgen(build_random_tiny(), 'stored')


@pytest.fixture(scope='module')
def computed_sys():
    return _sysgen(build_rot_tiny(), 'computed')


# ---------------------------------------------------------------- ゴールデン


def test_stored_top_golden(stored_sys):
    _assert_matches(stored_sys.top_verilog, 'top_overlay_stored_tiny.v')


def test_stored_project_golden(stored_sys, tmp_path):
    board = load_board('de0-cv')
    write_project_files(tmp_path, name='top', board=board,
                        sources=stored_sys.sources, clock_hz=50_000_000,
                        groups=stored_sys.groups, uart=None, clock_div=2)
    _assert_matches((tmp_path / 'top.qsf').read_text(encoding='utf-8'),
                    'top_overlay_stored.qsf')
    _assert_matches((tmp_path / 'top.sdc').read_text(encoding='utf-8'),
                    'top_overlay_stored.sdc')


def test_stored_sdc_identical_to_computed(stored_sys, computed_sys, tmp_path):
    """SDC はコアに依らない(差はソース一覧だけ)。"""
    board = load_board('de0-cv')
    outs = []
    for gen, sub in ((stored_sys, 'st'), (computed_sys, 'cp')):
        write_project_files(tmp_path / sub, name='top', board=board,
                            sources=gen.sources, clock_hz=50_000_000,
                            groups=gen.groups, uart=None, clock_div=2)
        outs.append((tmp_path / sub / 'top.sdc').read_text(encoding='utf-8'))
    assert outs[0] == outs[1]


def test_stored_qsf_lists_stored_templates(stored_sys, tmp_path):
    board = load_board('de0-cv')
    write_project_files(tmp_path, name='top', board=board,
                        sources=stored_sys.sources, clock_hz=50_000_000,
                        groups=stored_sys.groups, uart=None, clock_div=2)
    qsf = (tmp_path / 'top.qsf').read_text(encoding='utf-8')
    for name in ('lgn_overlay_stored.v', 'gate_rom.v', 'rom32.v', 'rom40.v',
                 'overlay_harness_stored.v', 'overlay_argmax.v'):
        assert name in qsf
    assert 'lgn_overlay.v' not in qsf.replace('lgn_overlay_stored.v', '')


# ---------------------------------------------------------------- 契約


def test_stored_top_instantiates_stored_core(stored_sys):
    v = stored_sys.top_verilog
    assert 'overlay_harness_stored #(' in v
    assert '.REC_BITS(32), .ADDR_F(14)' in v
    assert 'L0_INT_STEP' not in v                 # 計算型のパラメータは出ない
    assert 'altsource_probe' in v                 # ISSP は共用
    assert 'always @(posedge CLOCK_50) clk_core <= ~clk_core;' in v
    assert 'seg7 u_h0' in v
    assert 'gatemem_lane0..7.mem' in v            # ヘッダの記述(記録 ROM)


def test_stored_meta_contract(stored_sys):
    meta = stored_sys.to_meta()
    assert meta['kind'] == 'overlay'
    assert meta['overlay']['WIRING'] == 'stored'
    assert meta['overlay']['RECORD_BITS'] == 32
    assert meta['overlay']['GATE_ROMS'] == 8
    assert meta['cycles_per_image'] == 34         # 計算型 tiny と同値
    assert meta['probe']['cycle'][1] == 32
    assert meta['probe']['source'] == {'start': 0, 'rst': 1}


def test_stored_probe_layout_same_as_computed(stored_sys, computed_sys):
    """ISSP の probe 割付は方式によらない(issp.py の契約が共用できる)。"""
    assert stored_sys.probe_layout() == computed_sys.probe_layout()


def test_issp_evaluate_uses_stored_cycles(stored_sys):
    meta = stored_sys.to_meta()
    res = evaluate(meta, {'done': 1, 'proc': 8, 'match': 8, 'cycle': 8 * 34})
    assert res.ok and res.cyc_per_inference == 34
    assert res.inf_per_s == pytest.approx(25_000_000 / 34)


# ---------------------------------------------------------------- fit 解析


_FIT_STORED_SAMPLE = """
+----------------------------------------------------------------------+
; Fitter Resource Utilization by Entity                                 ;
+-------------------------+-----------------+---------+-------+--------+
; Compilation Hierarchy Node ; ALMs needed [=A-B+C] ; M10Ks ; Block Memory Bits ;
+-------------------------+-----------------+---------+-------+--------+
; |top                     ; 1286 (0)        ; 296     ; 2356000 ;
;    |overlay_harness_stored:u_sys| ; 1200 (3) ; 296   ; 2356000 ;
;       |gate_rom:gl0.r|    ; 0 (0)           ; 30      ; 192000 ;
;          |rom32:g32.r|    ; 0 (0)           ; 30      ; 192000 ;
;       |gate_rom:gl1.r|    ; 0 (0)           ; 30      ; 192000 ;
;          |rom32:g32.r|    ; 0 (0)           ; 30      ; 192000 ;
;       |tdp_ram:g_buf[0].bufA| ; 0 (0)      ; 1       ; 8000 ;
;       |altsyncram:img_rom_rtl_0| ; 0 (0)   ; 40      ; 401408 ;
;       |altsyncram:exp_rom_rtl_0| ; 0 (0)   ; 1       ; 2048 ;
+-------------------------+-----------------+---------+-------+--------+
"""


def test_fit_entity_summary_counts_lane_roms_once():
    """殻(gate_rom:)は数えず、実体(rom32:)だけをゲート ROM に足す。"""
    rows = parse_fit_entities(_FIT_STORED_SAMPLE)
    summary = summarize_entity_m10k(rows)
    assert summary['gate_rom'] == 60          # 30 blk x 2 レーン(抜粋)
    assert summary['activation'] == 1
    assert summary['image_rom'] == 40
    assert summary['label_rom'] == 1
    assert summary['total'] == 296


# ---------------------------------------------------------------- 実機


@requires_hw
def test_hw_overlay_stored_issp():
    """実機受入(MDL_FPGA_HW_TESTS=1 のときだけ走る)。

    保存型で書込済みのプロジェクトに対し ISSP で start を叩き、全ベクトル
    一致・サイクル式一致を確かめる(48K P0 なら 512/512・6,129 cyc/画像・
    4,079.0 inf/s @25MHz)。projdir は MDL_FPGA_OVERLAY_STORED_PROJDIR。
    """
    import os

    from mdl_fpga.issp import run_issp
    projdir = os.environ.get('MDL_FPGA_OVERLAY_STORED_PROJDIR')
    if not projdir:
        pytest.skip('MDL_FPGA_OVERLAY_STORED_PROJDIR を指定する')
    res = run_issp(projdir)
    assert res.ok, res.render()
    assert res.match == res.expected_images
    assert res.cyc_per_inference == res.expected_cycles_per_image
