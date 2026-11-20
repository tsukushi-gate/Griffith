"""プローブのスライス・内蔵レンダラ・評価隔離(シミュレータ不要)。

数値の正しさはライブラリ simulate の値との bit 照合で保証する(golden 主義)。
"""

import numpy as np
import pytest
from conftest import build_groupsum_model

from mdl_fpga.viewer.probes import (
    ProbeError,
    ProbeRegistry,
    ProbeSpec,
    evaluate_probes,
    slice_bits,
    target_width,
)
from mdl_fpga.viewer.renderers import BUILTINS, RendererHub, _resolve_auto

META = {
    'in_width': 4, 'out_width': 2, 'score_bits': 2, 'k': 2,
    'input_shape': None,
    'nodes': [
        {'id': 'gates', 'width': 4, 'type': 'lut_layer', 'score_bits': None},
        {'id': 'gs', 'width': 4, 'type': 'group_sum', 'score_bits': 2},
    ],
}


def _ctx(width, target='gates', args=None, meta=None, sl=None):
    meta = meta or META
    tm = next((n for n in meta['nodes'] if n['id'] == target), {})
    return {'width': width, 'full_width': width, 'cycle': 0,
            'target': target, 'slice': sl, 'meta': meta,
            'target_meta': tm, 'args': args or {}}


class TestSliceBits:
    def test_unpadded_hex_uses_meta_width(self):
        # ブリッジの hex は非パディング: 'f' でも幅8で展開する
        bits = slice_bits('f', 8, None)
        assert bits.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]  # LSB-first

    def test_slice_msb_lsb(self):
        bits = slice_bits('a5', 8, [7, 4])   # 0xa5=10100101 → 上位ニブル
        assert bits.tolist() == [0, 1, 0, 1]  # bits[4..7] LSB-first
        assert slice_bits('a5', 8, [0, 0]).tolist() == [1]
        assert slice_bits('a5', 8, [5, 5]).tolist() == [1]

    def test_slice_out_of_range(self):
        with pytest.raises(ProbeError):
            slice_bits('0', 4, [4, 0])
        with pytest.raises(ProbeError):
            slice_bits('0', 4, [1, 2])   # msb < lsb
        with pytest.raises(ProbeError):
            slice_bits('0', 4, [3])      # 要素数不正

    def test_empty_hex(self):
        assert slice_bits('', 3, None).tolist() == [0, 0, 0]


class TestTargetWidth:
    def test_ports_and_nodes(self):
        assert target_width(META, 'x') == 4
        assert target_width(META, 'y') == 2
        assert target_width(META, 'y_comb') == 2
        assert target_width(META, 'gs') == 4

    def test_unknown(self):
        with pytest.raises(ProbeError):
            target_width(META, 'ghost')


class TestBuiltinRenderers:
    def test_bits_hex_uint(self):
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)  # 0b1101 = 13
        assert BUILTINS['bits'](bits, _ctx(4)) == \
            {'kind': 'bits', 'hex': 'd', 'width': 4}
        assert BUILTINS['hex'](bits, _ctx(4))['text'] == '0xd'
        assert BUILTINS['uint'](bits, _ctx(4))['text'] == '13'

    def test_bars_matches_simulate_packing(self):
        """bars の解釈がライブラリ simulate のスコアと一致する(golden)。"""
        import json

        from griffith_logic.ir.serialize import ir_from_dict
        from griffith_logic.ir.simulate import simulate

        from mdl_fpga.check import expected_packed
        from mdl_fpga.vectors import gen_inputs

        graph = ir_from_dict(json.loads(json.dumps(build_groupsum_model())))
        x = gen_inputs(4, 16)
        golden = simulate(graph, x)               # (batch, k) の整数スコア
        packed, out_width = expected_packed(graph, x)
        for row_scores, packed_val in zip(golden, packed):
            bits = slice_bits(f'{packed_val:x}', out_width, None)
            payload = BUILTINS['bars'](bits, _ctx(out_width, target='gs'))
            assert payload['values'] == [float(v) for v in row_scores]
            assert payload['argmax'] == int(np.argmax(row_scores))

    def test_image_square(self):
        pytest.importorskip('PIL')
        bits = np.zeros(16, dtype=np.uint8)
        bits[0] = 1
        payload = BUILTINS['image'](bits, _ctx(16, args={}))
        assert payload['kind'] == 'image'
        assert payload['w'] == 4 and payload['h'] == 4
        assert payload['data_url'].startswith('data:image/png;base64,')

    def test_image_non_square_pads(self):
        """端数 bit 数でも表示できる(正方形既定+0パディング)。"""
        pytest.importorskip('PIL')
        bits = np.ones(6, dtype=np.uint8)
        payload = BUILTINS['image'](bits, _ctx(6))
        assert payload['kind'] == 'image'
        assert (payload['w'], payload['h']) == (3, 2)   # ceil(sqrt(6))=3

    def test_image_custom_cols(self):
        """ユーザ指定の横幅(cols)で任意の長方形に並ぶ。"""
        pytest.importorskip('PIL')
        bits = np.ones(5, dtype=np.uint8)
        payload = BUILTINS['image'](bits, _ctx(5, args={'cols': 4}))
        assert (payload['w'], payload['h']) == (4, 2)   # 5bit → 4列2行(3パディング)


class TestAutoResolution:
    def test_matrix(self):
        assert _resolve_auto(_ctx(4, target='gs')) is BUILTINS['bars']
        meta_img = {**META, 'input_shape': [2, 2]}
        assert _resolve_auto(
            _ctx(4, target='x', meta=meta_img)) is BUILTINS['image']
        assert _resolve_auto(_ctx(4, target='gates')) is BUILTINS['bits']
        assert _resolve_auto(_ctx(100, target='gates')) is BUILTINS['hex']


class TestEvaluateIsolation:
    def test_one_bad_probe_does_not_kill_others(self, tmp_path):
        hub = RendererHub(str(tmp_path / 'model.ir.json'))
        reg = ProbeRegistry()
        reg.replace_all([
            ProbeSpec(id='ok', target='gates', renderer='uint'),
            ProbeSpec(id='bad_target', target='ghost'),
            ProbeSpec(id='bad_slice', target='gates', slice=[9, 0]),
            ProbeSpec(id='bad_renderer', target='gates', renderer='nope'),
        ])
        values = {'gates': 'd', 'x': 'f', 'y': '0', 'y_comb': '0'}
        out = evaluate_probes(reg, values, META, 3, hub)
        assert out['ok'] == {'kind': 'text', 'text': '13'}
        assert out['bad_target']['kind'] == 'error'
        assert out['bad_slice']['kind'] == 'error'
        assert out['bad_renderer']['kind'] == 'error'
        assert 'nope' in out['bad_renderer']['message']
