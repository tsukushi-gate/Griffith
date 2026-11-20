"""RTLシミュレータ拡張(表示キュー・step_until)のテスト。"""

import importlib.util

import pytest
from conftest import requires_iverilog

from mdl_fpga.viewer.probes import ProbeRegistry, ProbeSpec, evaluate_probes
from mdl_fpga.viewer.renderers import RendererHub

requires_cocotb = pytest.mark.skipif(
    importlib.util.find_spec('cocotb') is None,
    reason='cocotb 未導入(uv sync --extra fpga)',
)

META = {
    'in_width': 4, 'out_width': 2, 'score_bits': None, 'k': None,
    'input_shape': None,
    'nodes': [{'id': 'sig', 'width': 4, 'type': 'lut_layer',
               'score_bits': None}],
}


def _hub(tmp_path):
    (tmp_path / 'm.ir.json').write_text('{}', encoding='utf-8')
    return RendererHub(str(tmp_path / 'm.ir.json'))


class TestDisplayQueue:
    def test_queue_fills_newest_first(self, tmp_path):
        """ユーザ例: 3 → (3,x,..)、2,8 と進むと (8,2,3,..)。"""
        hub = _hub(tmp_path)
        reg = ProbeRegistry()
        reg.replace_all([ProbeSpec(id='p', target='sig', renderer='uint',
                                   depth=5)])
        base = {'x': '0', 'y': '0', 'y_comb': '0'}

        out = evaluate_probes(reg, {**base, 'sig': '3'}, META, 1, hub)
        q = out['p']
        assert q['kind'] == 'queue' and q['depth'] == 5
        assert [i['payload']['text'] for i in q['items']] == ['3']

        evaluate_probes(reg, {**base, 'sig': '2'}, META, 2, hub)
        out = evaluate_probes(reg, {**base, 'sig': '8'}, META, 3, hub)
        texts = [i['payload']['text'] for i in out['p']['items']]
        assert texts == ['8', '2', '3']          # 新しい順
        assert [i['cycle'] for i in out['p']['items']] == [3, 2, 1]

    def test_read_does_not_duplicate(self, tmp_path):
        """同じ cycle での再評価(read/再表示)はキューに積まない。"""
        hub = _hub(tmp_path)
        reg = ProbeRegistry()
        reg.replace_all([ProbeSpec(id='p', target='sig', renderer='uint',
                                   depth=4)])
        vals = {'sig': '5', 'x': '0', 'y': '0', 'y_comb': '0'}
        evaluate_probes(reg, vals, META, 1, hub)
        out = evaluate_probes(reg, vals, META, 1, hub)   # read 相当
        assert len(out['p']['items']) == 1

    def test_depth_trims_and_reset_clears(self, tmp_path):
        hub = _hub(tmp_path)
        reg = ProbeRegistry()
        reg.replace_all([ProbeSpec(id='p', target='sig', renderer='uint',
                                   depth=2)])
        base = {'x': '0', 'y': '0', 'y_comb': '0'}
        for cyc, v in ((1, '1'), (2, '2'), (3, '3')):
            out = evaluate_probes(reg, {**base, 'sig': v}, META, cyc, hub)
        assert [i['payload']['text'] for i in out['p']['items']] == ['3', '2']
        # リセット(cycle が戻る)で履歴クリア
        out = evaluate_probes(reg, {**base, 'sig': '9'}, META, 0, hub)
        assert [i['payload']['text'] for i in out['p']['items']] == ['9']

    def test_renderer_switch_rerenders_history(self, tmp_path):
        """レンダラ変更が過去値にも遡って効く(生値保持の検証)。"""
        hub = _hub(tmp_path)
        reg = ProbeRegistry()
        reg.replace_all([ProbeSpec(id='p', target='sig', renderer='uint',
                                   depth=3)])
        base = {'x': '0', 'y': '0', 'y_comb': '0'}
        evaluate_probes(reg, {**base, 'sig': 'a'}, META, 1, hub)
        evaluate_probes(reg, {**base, 'sig': 'f'}, META, 2, hub)
        # renderer を hex へ変更(target/slice 不変 → 履歴維持)
        reg.replace_all([ProbeSpec(id='p', target='sig', renderer='hex',
                                   depth=3)])
        out = evaluate_probes(reg, {**base, 'sig': 'f'}, META, 2, hub)
        texts = [i['payload']['text'] for i in out['p']['items']]
        assert texts == ['0xf', '0xa']

    def test_slice_change_clears_history(self, tmp_path):
        hub = _hub(tmp_path)
        reg = ProbeRegistry()
        reg.replace_all([ProbeSpec(id='p', target='sig', renderer='uint',
                                   depth=3)])
        base = {'x': '0', 'y': '0', 'y_comb': '0'}
        evaluate_probes(reg, {**base, 'sig': 'f'}, META, 1, hub)
        reg.replace_all([ProbeSpec(id='p', target='sig', renderer='uint',
                                   depth=3, slice=[1, 0])])
        out = evaluate_probes(reg, {**base, 'sig': 'f'}, META, 1, hub)
        assert len(out['p']['items']) == 1       # クリアされて現在値のみ


@requires_iverilog
@requires_cocotb
class TestStepUntil:
    def test_step_until_change_and_timeout(self, parity4_path, tmp_path):
        from mdl_fpga.viewer.session import SimSession
        session = SimSession(model=str(parity4_path),
                             workdir=tmp_path / 'view')
        try:
            session.start()
            session.reset()
            # x_in を変えて step_until x → xin FF は 1clk で変化する
            session.set_x('f')
            r = session.step_until('x', max_steps=10)
            assert r['ok'] and r['changed'] is True and r['steps'] == 1
            assert int(r['values']['x'], 16) == 0xF

            # 入力固定のまま xor_l2 を監視 → 変化せず上限で打ち切り
            r = session.step_until('xor_l2', max_steps=5)
            assert r['changed'] is False and r['steps'] == 5

            # スライス指定でも動く(x[0] は既に 1 → 0 に変えると 1clk)
            session.set_x('e')
            r = session.step_until('x', bit_slice=[0, 0], max_steps=10)
            assert r['changed'] is True and r['steps'] == 1
        finally:
            session.close()
