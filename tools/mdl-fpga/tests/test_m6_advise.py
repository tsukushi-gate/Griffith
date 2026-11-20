"""M6: advise 動詞(単独実行+resources.unit 一般化+未較正パス)の検証。"""

import json

from conftest import build_parity4
from mdl_fpga.advisor import advise, run_advise
from mdl_fpga.board_db import load_board
from mdl_fpga.ir_io import load_graph


def _graph(tmp_path, doc):
    p = tmp_path / 'm.ir.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    return load_graph(p)


def test_de0cv_behavior_unchanged(tmp_path):
    """de0-cv 実値での判定が従来と同一(parity4: spatial/overlay とも載る)。"""
    g = _graph(tmp_path, build_parity4())
    adv = advise(g, load_board('de0-cv'))
    assert adv.gates == 3
    assert adv.spatial_ok is True and adv.overlay_ok is True
    assert adv.unit == 'alm' and adv.calibrated is True
    assert adv.spatial_units_needed == adv.spatial_alm_needed
    assert not any('未較正' in m for m in adv.messages)


def test_unit_le_dynamic_keys(tmp_path):
    g = _graph(tmp_path, build_parity4())
    board = {'key': 'x', 'name': 'X board',
             'resources': {'unit': 'le', 'le': 1000},
             'limits': {'calibrated': True, 'gates_per_le': 1.0,
                        'spatial_le_fill': 0.5, 'spatial_gates_max': 100,
                        'overlay_gates_max': 200}}
    adv = advise(g, board)
    assert adv.unit == 'le' and adv.calibrated is True
    assert adv.spatial_alm_needed == int(3 / 1.0 / 0.5)
    assert adv.spatial_ok is True
    assert any('LE' in m for m in adv.messages)


def test_uncalibrated_conservative_defaults(tmp_path):
    g = _graph(tmp_path, build_parity4())
    board = {'key': 'x', 'name': 'X board',
             'resources': {'unit': 'le', 'le': 1000},
             'limits': {'calibrated': False}}
    adv = advise(g, board)
    assert adv.calibrated is False
    assert adv.overlay_ok is None            # overlay 上限未定義 → 判定不能
    assert any('未較正' in m for m in adv.messages)
    assert any('判定不能' in m for m in adv.messages)
    # 保守既定: gates=3 → units = int(3/2.0/0.25) = 6
    assert adv.spatial_alm_needed == 6
    assert adv.spatial_ok is True


def test_run_advise_with_ir_and_manifest(tmp_path):
    ir = tmp_path / 'p4.ir.json'
    ir.write_text(json.dumps(build_parity4()), encoding='utf-8')
    adv = run_advise(ir)
    assert adv.spatial_ok is True

    mf = tmp_path / 'p4.manifest.json'
    mf.write_text(json.dumps({'manifest_version': '0.2',
                              'ir': 'p4.ir.json',
                              'board': 'de0-cv'}), encoding='utf-8')
    adv2 = run_advise(mf)
    assert adv2.gates == adv.gates


def test_to_dict_json_serializable(tmp_path):
    g = _graph(tmp_path, build_parity4())
    adv = advise(g, load_board('de0-cv'))
    s = json.dumps(adv.to_dict(), ensure_ascii=False)
    assert '"gates": 3' in s
