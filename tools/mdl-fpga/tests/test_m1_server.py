"""M1: ビューアサーバ(FastAPI+WS)の往復テスト。

ブラウザの代わりに TestClient で WebSocket を張り、set_step(クリック=1clk)の
応答値が論理的に正しいこと(パリティ)と index.html 配信を確認する。
"""

import importlib.util

import pytest
from conftest import requires_iverilog

requires_cocotb = pytest.mark.skipif(
    importlib.util.find_spec('cocotb') is None,
    reason='cocotb 未導入(uv sync --extra fpga)',
)
requires_fastapi = pytest.mark.skipif(
    importlib.util.find_spec('fastapi') is None,
    reason='fastapi 未導入(uv sync --extra fpga)',
)


@requires_iverilog
@requires_cocotb
@requires_fastapi
def test_server_meta_ws_and_static(parity4_path, tmp_path):
    from starlette.testclient import TestClient

    from mdl_fpga.viewer.server import create_app
    from mdl_fpga.viewer.session import SimSession

    session = SimSession(model=str(parity4_path), workdir=tmp_path / 'view')
    try:
        session.start()
        session.reset()
        client = TestClient(create_app(session))

        meta = client.get('/api/meta').json()
        assert meta['in_width'] == 4 and meta['out_width'] == 1
        assert [n['id'] for n in meta['nodes']] == ['xor_l1', 'xor_l2']

        html = client.get('/')
        assert html.status_code == 200 and 'mdl-fpga viewer' in html.text

        with client.websocket_connect('/ws/sim') as ws:
            first = ws.receive_json()
            assert first['type'] == 'values' and first['ok']

            # クリック=1clk: x=1111 → パリティ0
            ws.send_json({'cmd': 'set_step', 'hex': 'f', 'n': 1})
            r = ws.receive_json()
            assert r['ok'] and 'latency_ms' in r
            assert int(r['values']['x'], 16) == 0xF
            assert int(r['values']['xor_l2'], 16) == 0
            assert int(r['values']['y_comb'], 16) == 0

            # x=0111 → パリティ1、y(出力レジスタ)は1clk遅れで前の値
            ws.send_json({'cmd': 'set_step', 'hex': '7', 'n': 1})
            r2 = ws.receive_json()
            assert int(r2['values']['xor_l2'], 16) == 1
            assert int(r2['values']['y'], 16) == 0      # 前クロックの y_comb
            ws.send_json({'cmd': 'step', 'n': 1})
            r3 = ws.receive_json()
            assert int(r3['values']['y'], 16) == 1      # 1clk 後に反映

            ws.send_json({'cmd': 'reset'})
            r4 = ws.receive_json()
            assert r4['cycle'] == 0
    finally:
        session.close()
