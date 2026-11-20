"""DAG構造IRのHDL等価性テスト。

ダイヤモンド+concat+add のIRで3段等価性
    PyTorch hard eval ⇔ IR simulate ⇔ iverilog/GHDL
を検証する。生成コードの文字列妥当性(concatビット順・addビット演算・
fanout=同一ワイヤ参照)はシミュレータなしでも無条件に検証する。

シミュレータ非導入環境ではシミュレーション部のみ自動skip。
実行: uv run pytest -m hdl / -m hdl_vhdl
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.export.verify import (
    ghdl_available,
    iverilog_available,
    run_ghdl_equivalence,
    run_iverilog_equivalence,
)
from griffith_logic.export.verilog import to_verilog
from griffith_logic.export.vhdl import to_vhdl
from griffith_logic.ir import (
    IRValidationError,
    build_torch,
    ir_from_dict,
    simulate,
    validate,
)


def _rand_lut_params(in_width, out_width, n_inputs, rng):
    indices = [
        [int(rng.integers(0, in_width)) for _ in range(out_width)]
        for _ in range(n_inputs)
    ]
    tts = [hex(int(rng.integers(0, 1 << (1 << n_inputs))))
           for _ in range(out_width)]
    return indices, tts


def _lut_node(node_id, in_width, out_width, rng, n_inputs=2):
    indices, tts = _rand_lut_params(in_width, out_width, n_inputs, rng)
    return {
        'id': node_id,
        'type': 'lut_layer',
        'params': {
            'in_width': in_width,
            'out_width': out_width,
            'n_inputs': n_inputs,
            'indices': indices,
            'truth_tables': tts,
        },
        'train_info': {'parameterization': 'independent'},
        'meta': {},
    }


def _port(pid, width):
    return {'id': pid, 'width': width,
            'shape': None, 'encoding': None, 'semantics': None}


def _diamond_add_doc(mode='or', seed=17) -> dict:
    """ダイヤモンド+concat+add。n2はconcatとaddの両方へfanoutする。

    x(8) → n1(8→8) → fanout n2(8→8)/n3(8→8) → n4=concat([8,8])
    → n5(16→8) → n6=add(mode, slot0=n5, slot1=n2) → y(8)
    """
    rng = np.random.default_rng(seed)
    return {
        'format': 'magicdelta-ir',
        'version': '0.2.0',
        'metadata': {'name': f'diamond_add_{mode}'},
        'inputs': [_port('x', 8)],
        'outputs': [_port('y', 8)],
        'nodes': [
            _lut_node('n1_lutlayer', 8, 8, rng),
            _lut_node('n2_lutlayer', 8, 8, rng),
            _lut_node('n3_lutlayer', 8, 8, rng),
            {'id': 'n4_concat', 'type': 'concat',
             'params': {'in_widths': [8, 8]}, 'train_info': {}, 'meta': {}},
            _lut_node('n5_lutlayer', 16, 8, rng),
            {'id': 'n6_add', 'type': 'add',
             'params': {'width': 8, 'mode': mode},
             'train_info': {}, 'meta': {}},
        ],
        'edges': [
            {'src': 'x', 'dst': 'n1_lutlayer'},
            {'src': 'n1_lutlayer', 'dst': 'n2_lutlayer'},
            {'src': 'n1_lutlayer', 'dst': 'n3_lutlayer'},
            {'src': 'n2_lutlayer', 'dst': 'n4_concat', 'dst_slot': 0},
            {'src': 'n3_lutlayer', 'dst': 'n4_concat', 'dst_slot': 1},
            {'src': 'n4_concat', 'dst': 'n5_lutlayer'},
            {'src': 'n5_lutlayer', 'dst': 'n6_add', 'dst_slot': 0},
            {'src': 'n2_lutlayer', 'dst': 'n6_add', 'dst_slot': 1},
            {'src': 'n6_add', 'dst': 'y'},
        ],
    }


def _binary(batch, width, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (batch, width), generator=g).numpy().astype(np.int64)


def _hard_forward(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32))
    return np.round(y.numpy()).astype(np.int64)


class TestDagHdlStrings:
    """生成コードの文字列妥当性(シミュレータ不要、高速スイートで常時実行)"""

    def test_verilog_concat_add_fanout(self):
        graph = ir_from_dict(_diamond_add_doc('or'))
        src = to_verilog(graph)
        # concat: {}は左がMSBなのでslot降順(slot0=下位ビット)
        assert 'wire [15:0] w_n4_concat;' in src
        assert 'assign w_n4_concat = {w_n3_lutlayer, w_n2_lutlayer};' in src
        # add(or): ビットOR
        assert 'assign w_n6_add = w_n5_lutlayer | w_n2_lutlayer;' in src
        # fanout: w_n2_lutlayerがconcatとaddの両方から参照される
        assert src.count('w_n2_lutlayer') >= 3
        assert 'assign y = w_n6_add;' in src

    def test_verilog_add_xor(self):
        graph = ir_from_dict(_diamond_add_doc('xor'))
        src = to_verilog(graph)
        assert 'assign w_n6_add = w_n5_lutlayer ^ w_n2_lutlayer;' in src

    def test_vhdl_concat_add_fanout(self):
        graph = ir_from_dict(_diamond_add_doc('or'))
        src = to_vhdl(graph)
        # concat: &は左がMSBなのでslot降順(slot0=下位ビット)
        assert 'signal w_n4_concat : std_logic_vector(15 downto 0);' in src
        assert 'w_n4_concat <= w_n3_lutlayer & w_n2_lutlayer;' in src
        assert 'w_n6_add <= w_n5_lutlayer or w_n2_lutlayer;' in src
        assert 'y <= w_n6_add;' in src

    def test_sum_mode_rejected(self):
        graph = ir_from_dict(_diamond_add_doc('sum'))
        validate(graph)  # genericは通る
        with pytest.raises(IRValidationError, match='sum'):
            to_verilog(graph)
        with pytest.raises(IRValidationError, match='sum'):
            to_vhdl(graph)


@pytest.mark.hdl
@pytest.mark.skipif(not iverilog_available(), reason='iverilog未導入')
class TestVerilogDagEquivalence:
    @pytest.mark.parametrize('mode', ['or', 'xor'])
    def test_three_way_equivalence(self, mode, tmp_path, seed):
        """PyTorch hard eval ⇔ simulate ⇔ iverilog の3段等価性"""
        graph = ir_from_dict(_diamond_add_doc(mode))
        assert validate(graph, target='hdl') == []
        x = _binary(256, 8, seed=9)

        # 第1段: PyTorch hard eval ⇔ simulate
        expected = simulate(graph, x)
        actual = _hard_forward(build_torch(graph), x)
        assert (expected == actual).all(), 'hard evalとsimulateが不一致'

        # 第2段: simulate ⇔ iverilog
        result = run_iverilog_equivalence(graph, x, str(tmp_path))
        assert result['passed'], (
            f'add({mode}): Verilogとgolden modelが不一致\n{result["stdout"]}'
        )


@pytest.mark.hdl_vhdl
@pytest.mark.skipif(not ghdl_available(), reason='ghdl未導入')
class TestVhdlDagEquivalence:
    def test_three_way_equivalence(self, tmp_path, seed):
        graph = ir_from_dict(_diamond_add_doc('or'))
        x = _binary(256, 8, seed=9)

        expected = simulate(graph, x)
        actual = _hard_forward(build_torch(graph), x)
        assert (expected == actual).all(), 'hard evalとsimulateが不一致'

        result = run_ghdl_equivalence(graph, x, str(tmp_path))
        assert result['passed'], (
            f'VHDLとgolden modelが不一致\n{result["stdout"]}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
