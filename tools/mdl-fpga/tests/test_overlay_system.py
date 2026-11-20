"""オーバーレイ検証システム(トップ生成・プロジェクト・ISSP)の検証。

  - 生成トップ / qsf / sdc をゴールデンで固定(既存 golden は不変のまま)
  - clock_div の SDC 差分が「派生クロック 1 行だけ」であること
  - ISSP の probe 割付・Tcl・判定(実機なしで純関数として検証)
"""

import json

import pytest
from conftest import build_rot_tiny, requires_hw

from griffith_logic.ir.serialize import ir_from_dict
from mdl_fpga.backends.quartus import (
    parse_fit_entities,
    parse_fmax_all,
    summarize_entity_m10k,
    write_project_files,
)
from mdl_fpga.board_db import load_board
from mdl_fpga.hdlgen.overlay import plan_overlay
from mdl_fpga.hdlgen.overlay_system import generate_overlay_system
from mdl_fpga.hdlgen.ports import TopPorts
from mdl_fpga.issp import build_issp_tcl, decode_probe, evaluate
from test_golden_hdl import _assert_matches, _golden


@pytest.fixture(scope='module')
def sysgen():
    board = load_board('de0-cv')
    plan = plan_overlay(ir_from_dict(build_rot_tiny()), v=8)
    return generate_overlay_system(plan, n_images=8, clock_div=2,
                                   ports=TopPorts.from_board(board))


# ---------------------------------------------------------------- ゴールデン


def test_overlay_top_golden(sysgen):
    _assert_matches(sysgen.top_verilog, 'top_overlay_rot_tiny.v')


def test_overlay_project_golden(sysgen, tmp_path):
    board = load_board('de0-cv')
    write_project_files(tmp_path, name='top', board=board,
                        sources=sysgen.sources, clock_hz=50_000_000,
                        groups=sysgen.groups, uart=None, clock_div=2)
    _assert_matches((tmp_path / 'top.qsf').read_text(encoding='utf-8'),
                    'top_overlay.qsf')
    _assert_matches((tmp_path / 'top.sdc').read_text(encoding='utf-8'),
                    'top_overlay.sdc')


def test_clock_div_adds_only_generated_clock(sysgen, tmp_path):
    """clock_div の SDC 差分は create_generated_clock 1 行だけ。"""
    board = load_board('de0-cv')
    for div, out in ((1, tmp_path / 'd1'), (2, tmp_path / 'd2')):
        write_project_files(out, name='top', board=board,
                            sources=sysgen.sources, clock_hz=50_000_000,
                            groups=sysgen.groups, uart=None, clock_div=div)
    d1 = (tmp_path / 'd1' / 'top.sdc').read_text(encoding='utf-8').splitlines()
    d2 = (tmp_path / 'd2' / 'top.sdc').read_text(encoding='utf-8').splitlines()
    extra = [ln for ln in d2 if ln not in d1]
    assert len(extra) == 1
    assert extra[0].startswith('create_generated_clock -name clk_core')
    assert '-divide_by 2' in extra[0]
    assert [ln for ln in d1 if ln not in d2] == []


def test_existing_uart_golden_unchanged():
    """overlay 追加後も spatial(uart)の qsf/sdc ゴールデンは不変。"""
    from mdl_fpga.io_link import resolve_uart_link
    board = load_board('de0-cv')
    uart = resolve_uart_link(board, None).to_uart_dict()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        write_project_files(Path(td), name='top', board=board,
                            sources=['top.v', 'mdl_core.v', 'uart_rx.v',
                                     'uart_tx.v', 'frame_rx.v', 'frame_tx.v'],
                            uart=uart)
        assert (Path(td) / 'top.sdc').read_text(encoding='utf-8') == \
            _golden('top.sdc')


# ---------------------------------------------------------------- 契約


def test_system_meta_contract(sysgen):
    meta = sysgen.to_meta()
    assert meta['kind'] == 'overlay'
    assert meta['n_images'] == 8
    assert meta['clock_div'] == 2
    assert meta['core_clock_hz'] == 25_000_000
    assert meta['cycles_per_image'] == 34
    assert meta['expected_total_cycles'] == 8 * 34
    probe = meta['probe']
    assert probe['done'] == [0, 1]
    assert probe['cycle'][1] == 32
    assert probe['width'] == 1 + 2 * probe['proc'][1] + 32
    assert probe['source'] == {'start': 0, 'rst': 1}
    json.dumps(meta)   # そのまま overlay_system.json に書ける


def test_top_declares_expected_ports(sysgen):
    v = sysgen.top_verilog
    assert 'input  wire       CLOCK_50' in v
    assert 'output wire [9:0] LEDR' in v
    assert 'output wire [6:0] HEX5' in v
    assert 'altsource_probe' in v
    assert 'always @(posedge CLOCK_50) clk_core <= ~clk_core;' in v
    assert 'seg7 u_h0' in v
    # ISSP スタブは合成ソースに入れない(シム専用)
    assert 'altsource_probe_stub.v' not in sysgen.sources


# ---------------------------------------------------------------- ISSP


def test_issp_tcl_uses_layout(sysgen):
    tcl = build_issp_tcl(sysgen.to_meta())
    assert 'start_insystem_source_probe' in tcl
    assert 'wr 2 ; after 50 ; wr 0 ; after 50 ; wr 1' in tcl   # rst → start
    assert 'PROBE_RAW=' in tcl


def test_decode_probe_roundtrip(sysgen):
    layout = sysgen.to_meta()['probe']
    fields = {'done': 1, 'proc': 8, 'match': 8, 'cycle': 272}
    value = 0
    for key, val in fields.items():
        value |= val << layout[key][0]
    assert decode_probe(value, layout) == fields


def test_evaluate_pass(sysgen):
    meta = sysgen.to_meta()
    res = evaluate(meta, {'done': 1, 'proc': 8, 'match': 8, 'cycle': 8 * 34})
    assert res.ok
    assert res.cyc_per_inference == 34
    assert res.inf_per_s == pytest.approx(25_000_000 / 34)
    assert res.wraps == 0


@pytest.mark.parametrize('fields,needle', [
    ({'done': 1, 'proc': 8, 'match': 7, 'cycle': 8 * 34}, '一致数'),
    ({'done': 1, 'proc': 8, 'match': 8, 'cycle': 8 * 34 + 1}, 'サイクル'),
    ({'done': 0, 'proc': 0, 'match': 0, 'cycle': 0}, '一致数'),
])
def test_evaluate_detects_failures(sysgen, fields, needle):
    res = evaluate(sysgen.to_meta(), fields)
    assert not res.ok
    assert any(needle in m for m in res.messages)


def test_evaluate_handles_counter_wrap():
    """カウンタ幅を超える総サイクルは合同(mod 2^W)で厳密照合する。"""
    meta = {'n_images': 1000, 'cycles_per_image': 40145,
            'core_clock_hz': 25_000_000,
            'probe': {'done': [0, 1], 'proc': [1, 11], 'match': [12, 11],
                      'cycle': [23, 24], 'width': 47,
                      'source': {'start': 0, 'rst': 1}}}
    total = 1000 * 40145
    res = evaluate(meta, {'done': 1, 'proc': 1000, 'match': 1000,
                          'cycle': total % (1 << 24)})
    assert res.ok
    assert res.wraps == 2
    assert res.true_cycles == total
    assert res.cyc_per_inference == pytest.approx(40145)


# ---------------------------------------------------------------- fit 解析


_FIT_ENTITY_SAMPLE = """
+----------------------------------------------------------------------+
; Fitter Resource Utilization by Entity                                 ;
+-------------------------+-----------------+---------+-------+--------+
; Compilation Hierarchy Node ; ALMs needed [=A-B+C] ; M10Ks ; Block Memory Bits ;
+-------------------------+-----------------+---------+-------+--------+
; |top                     ; 1307 (0)        ; 125     ; 1066000 ;
;    |harness:u_harness|   ; 1200 (3)        ; 125     ; 1066000 ;
;       |rom32:u_opmem|    ; 0 (0)           ; 32      ; 192000 ;
;          |altsyncram:mem_rtl_0| ; 0 (0)    ; 32      ; 192000 ;
;       |tdp_ram:g_buf[0].bufA| ; 0 (0)      ; 2       ; 5000 ;
;       |tdp_ram:g_buf[0].bufB| ; 0 (0)      ; 2       ; 5000 ;
;       |altsyncram:img_rom_rtl_0| ; 0 (0)   ; 78      ; 784000 ;
;       |altsyncram:exp_rom_rtl_0| ; 0 (0)   ; 1       ; 4000 ;
+-------------------------+-----------------+---------+-------+--------+
"""


def test_parse_fit_entities_and_summary():
    rows = parse_fit_entities(_FIT_ENTITY_SAMPLE)
    assert rows and rows[0]['node'] == '|top'
    summary = summarize_entity_m10k(rows)
    assert summary['gate_rom'] == 32        # ゲート ROM(子の altsyncram は数えない)
    assert summary['activation'] == 4
    assert summary['image_rom'] == 78
    assert summary['label_rom'] == 1
    assert summary['total'] == 125


def test_parse_fit_entities_absent_section():
    assert parse_fit_entities('no such section') == []


_STA_SAMPLE_TWO_CLOCKS = """
  6. Slow 1100mV 85C Model Fmax Summary
  7. Timing Closure Recommendations

+-----------------------------------------+
; Slow 1100mV 85C Model Fmax Summary       ;
+------------+-----------------+-----------+
; Fmax       ; Restricted Fmax ; Clock Name ;
+------------+-----------------+-----------+
; 34.85 MHz  ; 34.85 MHz       ; clk_core   ;
; 210.13 MHz ; 210.13 MHz      ; clk50      ;
+------------+-----------------+-----------+

"""


def test_parse_fmax_all_two_clocks():
    """目次の見出し(表を持たない)に引っ張られず、表のある節を読む。"""
    out = parse_fmax_all(_STA_SAMPLE_TWO_CLOCKS)
    assert out['fmax_clk_core_mhz'] == 34.85
    assert out['fmax_clk50_mhz'] == 210.13


# ---------------------------------------------------------------- 実機


@requires_hw
def test_hw_overlay_issp(tmp_path):
    """実機受入(MDL_FPGA_HW_TESTS=1 のときだけ走る)。

    書込済みの overlay プロジェクトに対し ISSP で start を叩き、
    全ベクトル一致・サイクル式一致を確かめる。projdir は環境変数
    MDL_FPGA_OVERLAY_PROJDIR で与える。
    """
    import os

    from mdl_fpga.issp import run_issp
    projdir = os.environ.get('MDL_FPGA_OVERLAY_PROJDIR')
    if not projdir:
        pytest.skip('MDL_FPGA_OVERLAY_PROJDIR を指定する')
    res = run_issp(projdir)
    assert res.ok, res.render()
    assert res.match == res.expected_images
    assert res.cyc_per_inference == res.expected_cycles_per_image
