"""M6: ビューアのプロジェクト/ボード API(SessionHolder/epoch/切替)の検証。

サーバは session を isinstance 検査しない(ダック型)ため、iverilog 不要の
テストは StubSession で行い、実切替(SimSession 再生成)のみ live で検証する。
"""

import importlib.util
import json
import time
from pathlib import Path

import pytest
from conftest import build_groupsum_model, build_parity4, requires_iverilog

requires_cocotb = pytest.mark.skipif(
    importlib.util.find_spec('cocotb') is None,
    reason='cocotb 未導入(uv sync --extra fpga)',
)
requires_fastapi = pytest.mark.skipif(
    importlib.util.find_spec('fastapi') is None,
    reason='fastapi 未導入(uv sync --extra fpga)',
)


class StubSession:
    """SimSession のダック型(サーバ API の単体テスト用)。"""

    def __init__(self, model: str):
        self.model = model
        self.workdir = 'stub-wd'
        self.pipelined = False
        self.system_dir = None
        self.meta = {'name': 'stub', 'mode': 'core', 'in_width': 4,
                     'allow_set_x': True, 'nodes': [], 'edges': [],
                     'inputs': [], 'outputs': []}
        self.cycle = 0
        self.graph = None

    def _frame(self):
        return {'ok': True, 'cycle': self.cycle, 'values': {}, 'xz': []}

    def read(self):
        return self._frame()

    def reset(self):
        self.cycle = 0
        return self._frame()

    def set_x(self, hex_value):
        return self._frame()

    def step(self, n=1):
        self.cycle += n
        return self._frame()

    def step_until(self, key, bit_slice=None, max_steps=1000):
        return self._frame()

    def node_detail(self, node_id):
        raise KeyError(node_id)

    def start(self):
        return {}

    def close(self):
        pass


@pytest.fixture
def stub_client(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from mdl_fpga.viewer.server import create_app
    monkeypatch.setenv('MDL_FPGA_RECENT_PATH',
                       str(tmp_path / 'recent.json'))
    # モデルは実 IR(parity4)にして advise が計算できるようにする
    model = tmp_path / 'stub.ir.json'
    model.write_text(json.dumps(build_parity4()), encoding='utf-8')
    client = TestClient(create_app(StubSession(str(model))))
    client.model_path = model
    return client


@requires_fastapi
def test_project_info_and_epoch(stub_client):
    prj = stub_client.get('/api/project').json()
    assert prj['name'] == 'stub' and prj['mode'] == 'core'
    assert prj['epoch'] == 0 and prj['recent'] == []
    assert stub_client.get('/api/meta').json()['epoch'] == 0


@requires_fastapi
def test_examples_listing(stub_client, tmp_path, monkeypatch):
    exdir = tmp_path / 'examples'
    (exdir / 'sub').mkdir(parents=True)
    (exdir / 'a.ir.json').write_text('{}', encoding='utf-8')
    (exdir / 'sub' / 'b.ir.json').write_text('{}', encoding='utf-8')
    monkeypatch.setenv('MDL_FPGA_EXAMPLES_DIR', str(exdir))
    ex = stub_client.get('/api/examples').json()['examples']
    assert [e['key'] for e in ex] == ['a', 'sub/b']
    assert ex[0]['modes'] == [{'mode': 'core', 'ready': True}]


@requires_fastapi
def test_real_examples_include_mnist_system_gate(stub_client):
    """実 examples/: mnist24k は system モード付き(未ビルドなら ready:false)。"""
    ex = stub_client.get('/api/examples').json()['examples']
    keys = {e['key'] for e in ex}
    assert 'full_adder' in keys and 'mnist/mnist24k' in keys
    mnist = next(e for e in ex if e['key'] == 'mnist/mnist24k')
    modes = {m['mode']: m for m in mnist['modes']}
    assert 'system' in modes
    if not modes['system']['ready']:
        assert 'build_mnist_demo' in modes['system']['hint']


@requires_fastapi
def test_open_bad_requests_keep_old_session(stub_client, tmp_path, monkeypatch):
    r = stub_client.post('/api/project/open', json={'model': 'Z:/no/such.json'})
    assert r.status_code == 400 and '存在しない' in r.json()['detail']
    r = stub_client.post('/api/project/open', json={})
    assert r.status_code == 400
    # system 例の ready ゲートは偽レジストリで検証(実環境の mnist は
    # ビルド済みだと本当に開いてしまい環境依存になるため)
    import mdl_fpga.viewer.appstate as appstate
    exdir = tmp_path / 'ex'
    exdir.mkdir()
    (exdir / 'sys_demo.ir.json').write_text('{}', encoding='utf-8')
    monkeypatch.setenv('MDL_FPGA_EXAMPLES_DIR', str(exdir))
    monkeypatch.setattr(appstate, 'SYSTEM_EXAMPLES',
                        {'sys_demo': ('runs/__no_such__', 'build_xxx で生成')})
    r = stub_client.post('/api/project/open',
                         json={'example': 'sys_demo', 'mode': 'system'})
    assert r.status_code == 400 and '未ビルド' in r.json()['detail']
    # いずれの失敗でも旧セッション継続(epoch 不変)
    assert stub_client.get('/api/meta').json()['epoch'] == 0
    assert stub_client.get('/api/project').json()['name'] == 'stub'


@requires_fastapi
def test_layout_epoch_guard_and_board_merge(stub_client):
    # epoch 不一致の保存は 409(切替後の遅延デバウンス対策)
    r = stub_client.put('/api/layout?epoch=999', json={'probes': []})
    assert r.status_code == 409
    # board を知らない旧クライアントの PUT で board が消えない(サーバ側マージ)
    assert stub_client.put('/api/layout',
                           json={'probes': [], 'board': 'de0-cv'}).json() == \
        {'ok': True}
    assert stub_client.put('/api/layout?epoch=0',
                           json={'probes': []}).json() == {'ok': True}
    assert stub_client.get('/api/layout').json()['board'] == 'de0-cv'


@requires_fastapi
def test_boards_listing(stub_client):
    boards = stub_client.get('/api/boards').json()['boards']
    keys = {b['key'] for b in boards}
    assert {'de0-cv', 'de10-lite'} <= keys
    de0 = next(b for b in boards if b['key'] == 'de0-cv')
    assert de0['tier'] == 1 and de0['resources']['unit'] == 'alm'
    assert de0['detection']['silk_hint']          # 候補ダイアログの案内文
    de10 = next(b for b in boards if b['key'] == 'de10-lite')
    assert de10['unverified'] is True


@requires_fastapi
def test_board_select_advise_persist(stub_client):
    assert stub_client.get('/api/board').json() == {'key': None,
                                                    'advice': None}
    r = stub_client.put('/api/board', json={'key': 'de0-cv'}).json()
    assert r['ok'] is True and r['key'] == 'de0-cv'
    assert r['advice']['spatial_ok'] is True and r['advice']['gates'] == 3
    # サイドカーへ永続化(モデルごとに記憶)
    sidecar = stub_client.model_path.parent / 'stub.viewer.json'
    saved = json.loads(sidecar.read_text(encoding='utf-8'))
    assert saved['board'] == 'de0-cv'
    # GET で復元+キャッシュ経由でも同値
    assert stub_client.get('/api/board').json()['advice']['gates'] == 3
    # 未較正ボード(de10-lite)は calibrated:false の advice
    r2 = stub_client.put('/api/board', json={'key': 'de10-lite'}).json()
    assert r2['advice']['calibrated'] is False
    # 未知ボードは 400、None で選択解除
    assert stub_client.put('/api/board',
                           json={'key': 'nope'}).status_code == 400
    assert stub_client.put('/api/board',
                           json={'key': None}).json()['key'] is None


@requires_fastapi
def test_board_detect_endpoint_no_autoselect(stub_client):
    from mdl_fpga.detect import BoardCandidate, DetectResult, Observation
    holder = stub_client.app.state.holder
    holder.detect_impl = lambda: DetectResult(
        observation=Observation(probes_run=['jtag']),
        candidates=[BoardCandidate(key='de0-cv', name='Terasic DE0-CV',
                                   tier=1, confidence=0.6,
                                   reasons=['部品名一致'])])
    r = stub_client.post('/api/board/detect').json()
    assert r['best']['key'] == 'de0-cv'
    assert r['candidates'][0]['confidence'] == 0.6
    # サーバは自動選択しない(選択はフロントの PUT /api/board)
    assert stub_client.get('/api/board').json()['key'] is None


@requires_fastapi
def test_board_detect_501_when_impl_missing(stub_client, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'mdl_fpga.detect', None)
    r = stub_client.post('/api/board/detect')
    assert r.status_code == 501


def test_resolve_open_request_unit(tmp_path, monkeypatch):
    from mdl_fpga.viewer.appstate import resolve_open_request
    with pytest.raises(ValueError, match='どちらか'):
        resolve_open_request({})
    with pytest.raises(ValueError, match='存在しない'):
        resolve_open_request({'model': str(tmp_path / 'no.ir.json')})
    with pytest.raises(ValueError, match='system_meta'):
        m = tmp_path / 'x.ir.json'
        m.write_text('{}', encoding='utf-8')
        resolve_open_request({'model': str(m), 'system': str(tmp_path)})
    monkeypatch.setenv('MDL_FPGA_EXAMPLES_DIR', str(tmp_path / 'nope'))
    with pytest.raises(ValueError, match='未知の example'):
        resolve_open_request({'example': 'ghost'})


def test_recent_roundtrip(tmp_path, monkeypatch):
    from mdl_fpga.viewer.appstate import append_recent, load_recent
    monkeypatch.setenv('MDL_FPGA_RECENT_PATH', str(tmp_path / 'r.json'))
    assert load_recent() == []
    for i in range(12):
        append_recent({'model': f'm{i}.ir.json', 'system': None})
    items = load_recent()
    assert len(items) == 10 and items[0]['model'] == 'm11.ir.json'
    # 同一エントリは先頭へ移動(重複しない)
    append_recent({'model': 'm5.ir.json', 'system': None})
    items = load_recent()
    assert items[0]['model'] == 'm5.ir.json'
    assert sum(1 for e in items if e['model'] == 'm5.ir.json') == 1


@requires_iverilog
@requires_cocotb
@requires_fastapi
def test_live_project_switch(parity4_path, tmp_path, monkeypatch):
    """実切替: parity4 → gs_demo。旧 close/新サイドカー/epoch/失敗ロールバック。"""
    from starlette.testclient import TestClient

    from mdl_fpga.viewer.server import create_app
    from mdl_fpga.viewer.session import SimSession

    monkeypatch.setenv('MDL_FPGA_RECENT_PATH', str(tmp_path / 'recent.json'))
    gs_path = tmp_path / 'gs_demo.ir.json'
    gs_path.write_text(json.dumps(build_groupsum_model()), encoding='utf-8')

    old = SimSession(model=str(parity4_path), workdir=tmp_path / 'view')
    old.start()
    old.reset()
    app = create_app(old)
    client = TestClient(app)
    try:
        assert client.get('/api/meta').json()['name'] == 'parity4'

        # 失敗パス: 壊れた IR → 400 で旧継続
        bad = tmp_path / 'bad.ir.json'
        bad.write_text('{not json', encoding='utf-8')
        r = client.post('/api/project/open', json={'model': str(bad)})
        assert r.status_code == 400 and '継続' in r.json()['detail']
        assert client.get('/api/meta').json()['name'] == 'parity4'
        with client.websocket_connect('/ws/sim') as ws:
            ws.receive_json()
            ws.send_json({'cmd': 'step', 'n': 1})
            assert ws.receive_json()['ok'] is True   # 旧セッション生存

        # 成功パス: gs_demo へ切替
        r = client.post('/api/project/open', json={'model': str(gs_path)})
        assert r.json() == {'ok': True, 'name': 'gs_demo', 'mode': 'core',
                            'epoch': 1}
        meta = client.get('/api/meta').json()
        assert meta['name'] == 'gs_demo' and meta['epoch'] == 1
        # 新モデルのサイドカーへ保存される
        client.put('/api/layout?epoch=1',
                   json={'probes': [], 'positions': {'gs': {'x': 1, 'y': 2}}})
        saved = json.loads((tmp_path / 'gs_demo.viewer.json')
                           .read_text(encoding='utf-8'))
        assert saved['positions']['gs'] == {'x': 1, 'y': 2}
        # recent に記録
        prj = client.get('/api/project').json()
        assert prj['recent'][0]['name'] == 'gs_demo'
        # 旧セッションは裏で close される(vvp スレッド終了を待つ)
        deadline = time.time() + 20
        while time.time() < deadline:
            if old._thread is None or not old._thread.is_alive():
                break
            time.sleep(0.2)
        assert old._thread is None or not old._thread.is_alive()
        # 新セッションで WS が動く
        with client.websocket_connect('/ws/sim') as ws:
            first = ws.receive_json()
            assert first['ok'] is True
            ws.send_json({'cmd': 'step', 'n': 1})
            assert ws.receive_json()['ok'] is True
    finally:
        app.state.holder.session.close()
