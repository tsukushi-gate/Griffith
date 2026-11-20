"""サーバ統合(meta edges / node_detail / layout / WS probes / dist 分岐)。"""

import importlib.util
import json

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


@pytest.fixture
def live_session(parity4_path, tmp_path):
    from mdl_fpga.viewer.session import SimSession
    session = SimSession(model=str(parity4_path), workdir=tmp_path / 'view')
    session.start()
    session.reset()
    yield session
    session.close()


@requires_iverilog
@requires_cocotb
@requires_fastapi
def test_meta_edges_node_detail_and_probes(live_session, parity4_path):
    from starlette.testclient import TestClient

    from mdl_fpga.viewer.persistence import sidecar_path
    from mdl_fpga.viewer.server import create_app

    client = TestClient(create_app(live_session))

    # --- meta: 回路図描画に必要な edges/inputs/outputs を含む ---
    meta = client.get('/api/meta').json()
    assert meta['inputs'] == [
        {'id': 'x', 'width': 4, 'shape': None, 'semantics': None}]
    assert {(e['src'], e['dst']) for e in meta['edges']} == {
        ('x', 'xor_l1'), ('xor_l1', 'xor_l2'), ('xor_l2', 'y')}

    # --- node_detail: 展開用 params(JSON 化済み) ---
    detail = client.get('/api/node/xor_l1').json()
    assert detail['type'] == 'lut_layer'
    assert detail['params']['indices'] == [[0, 2], [1, 3]]
    assert client.get('/api/node/ghost').status_code == 404

    # --- renderers 一覧 ---
    rnd = client.get('/api/renderers').json()
    assert 'bars' in rnd['builtins'] and rnd['sidecar'] == []

    # --- layout PUT → プローブが WS フレームに乗る ---
    cfg = {
        'probes': [
            {'id': 'p_parity', 'target': 'xor_l2', 'renderer': 'uint'},
            {'id': 'p_slice', 'target': 'x', 'slice': [1, 0],
             'renderer': 'bits'},
        ],
        'positions': {'xor_l1': {'x': 100, 'y': 50}},
        'expanded': [],
    }
    assert client.put('/api/layout', json=cfg).json() == {'ok': True}
    assert client.get('/api/layout').json()['probes'][0]['id'] == 'p_parity'
    # 永続化ファイルがモデル横に書かれている
    saved = json.loads(sidecar_path(live_session.model)
                       .read_text(encoding='utf-8'))
    assert saved['positions']['xor_l1'] == {'x': 100, 'y': 50}

    with client.websocket_connect('/ws/sim') as ws:
        first = ws.receive_json()
        assert 'probes' in first
        # x=1111 → パリティ0 / x下位2bit=11
        ws.send_json({'cmd': 'set_step', 'hex': 'f', 'n': 1})
        r = ws.receive_json()
        assert r['probes']['p_parity'] == {'kind': 'text', 'text': '0'}
        assert r['probes']['p_slice'] == {'kind': 'bits', 'hex': '3',
                                          'width': 2}
        # x=0111 → パリティ1
        ws.send_json({'cmd': 'set_step', 'hex': '7', 'n': 1})
        r2 = ws.receive_json()
        assert r2['probes']['p_parity']['text'] == '1'

    # 孤児プローブは error payload(フレームは死なない)
    cfg['probes'].append({'id': 'p_ghost', 'target': 'ghost'})
    client.put('/api/layout', json=cfg)
    with client.websocket_connect('/ws/sim') as ws:
        first = ws.receive_json()
        assert first['probes']['p_ghost']['kind'] == 'error'
        assert first['probes']['p_parity']['kind'] == 'text'


@requires_iverilog
@requires_cocotb
@requires_fastapi
def test_dist_serving_and_fallback(live_session, tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from mdl_fpga.viewer.server import create_app

    # dist なし → 旧 HTML フォールバック
    monkeypatch.setenv('MDL_FPGA_FRONTEND_DIST', str(tmp_path / 'no_dist'))
    client = TestClient(create_app(live_session))
    html = client.get('/')
    assert html.status_code == 200 and 'mdl-fpga viewer' in html.text

    # dist あり → React ビルドを配信、API ルートは生きている
    dist = tmp_path / 'dist'
    dist.mkdir()
    (dist / 'index.html').write_text(
        '<html><title>react-ui</title></html>', encoding='utf-8')
    (dist / 'app.js').write_text('// bundle', encoding='utf-8')
    monkeypatch.setenv('MDL_FPGA_FRONTEND_DIST', str(dist))
    client2 = TestClient(create_app(live_session))
    assert 'react-ui' in client2.get('/').text
    assert client2.get('/app.js').text == '// bundle'
    assert client2.get('/api/meta').json()['in_width'] == 4
