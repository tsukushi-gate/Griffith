"""packed LUT proxy(analysis/lut_pack.py、SM3)のテスト。

パーサはgolden文字列で単体検証(yosys不要)。E2Eはyosys導入環境のみ
(未導入ではskip。黙って通さない=skip理由を明示)。
"""

import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.analysis.lut_pack import (
    find_yosys,
    packed_lut_stats,
    parse_ltp_depth,
    parse_stat_json,
)
from griffith_logic.ir.schema import IREdge, IRGraph, IRNode, IRPort

GOLDEN_STAT_JSON = """
{
  "creator": "Yosys 0.38",
  "modules": {
    "\\\\logic_core": {
      "num_wires": 24,
      "num_wire_bits": 40,
      "num_cells": 7,
      "num_cells_by_type": {
        "$lut": 7
      }
    }
  }
}
"""

GOLDEN_LTP_LOG = """
5.2. Executing LTP pass (longest topological path).

Longest topological path in logic_core (length=3):
    lut $abc$123$lut$aiger$456
"""


class TestParsers:
    def test_parse_stat_json(self):
        stats = parse_stat_json(GOLDEN_STAT_JSON)
        assert stats['lut_count'] == 7
        assert stats['cell_count'] == 7

    def test_parse_stat_json_no_luts(self):
        stats = parse_stat_json('{"modules": {"\\\\top": {"num_cells": 0}}}')
        assert stats['lut_count'] == 0

    def test_parse_ltp_depth(self):
        assert parse_ltp_depth(GOLDEN_LTP_LOG) == 3

    def test_parse_ltp_depth_missing(self):
        assert parse_ltp_depth('no ltp output here') is None


class TestFindYosys:
    def test_explicit_missing_falls_through_to_error(self, monkeypatch):
        monkeypatch.delenv('MDL_YOSYS', raising=False)
        monkeypatch.setattr(shutil, 'which', lambda _: None)
        with pytest.raises(RuntimeError, match='MDL_YOSYS'):
            find_yosys('Z:/does/not/exist/yosys.exe')

    def test_env_var_used(self, monkeypatch, tmp_path):
        fake = tmp_path / 'yosys.exe'
        fake.write_text('')
        monkeypatch.setenv('MDL_YOSYS', str(fake))
        assert find_yosys() == str(fake)


def _small_graph() -> IRGraph:
    """in(6) → l0(6→8) → l1(8→4) → out。ランダム2入力ゲート鎖。"""
    rng = np.random.default_rng(3)
    nodes = []
    prev = 6
    for li, w in enumerate((8, 4)):
        idx = rng.integers(0, prev, size=(2, w)).tolist()
        tts = [hex(int(t)) for t in rng.integers(1, 15, size=w)]
        nodes.append(IRNode(id=f'l{li}', type='lut_layer', params={
            'in_width': prev, 'out_width': w, 'n_inputs': 2,
            'indices': idx, 'truth_tables': tts,
        }))
        prev = w
    return IRGraph(
        inputs=[IRPort(id='in', width=6)],
        outputs=[IRPort(id='out', width=4)],
        nodes=nodes,
        edges=[
            IREdge(src='in', dst='l0'),
            IREdge(src='l0', dst='l1'),
            IREdge(src='l1', dst='out'),
        ],
    )


def _yosys_available() -> bool:
    try:
        find_yosys()
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not _yosys_available(),
                    reason='yosys未導入(PATHまたはMDL_YOSYSで指定)')
class TestEndToEnd:
    def test_packed_lut_stats_smoke(self, tmp_path):
        stats = packed_lut_stats(_small_graph(), k=6, workdir=str(tmp_path))
        assert stats['lut_count'] > 0
        assert stats['k'] == 6
        assert stats['sweep']['gates_before'] == 12
        if stats['gates_per_lut'] is not None:
            assert stats['gates_per_lut'] > 0
        # 中間生成物がworkdirに残ること(較正・デバッグ用)
        assert (tmp_path / 'logic_core.v').exists()

    def test_k4_maps_more_luts_than_k6(self, tmp_path):
        g = _small_graph()
        k6 = packed_lut_stats(g, k=6, workdir=str(tmp_path / 'k6'))
        k4 = packed_lut_stats(g, k=4, workdir=str(tmp_path / 'k4'))
        assert k4['lut_count'] >= k6['lut_count']
