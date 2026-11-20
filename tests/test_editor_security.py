"""エディタ/ビジュアライザ backend のセキュリティ機構のテスト。

エディタは**信頼された単一ユーザのローカル環境専用**ツールで認証を持たない。
「ローカルだから安全」を成立させる4つの守りを、外形(HTTPレスポンス)で固定する。

  B3  ワークスペース内包チェック   — `../` 脱出・外部絶対パスを400で拒否
  R6  Host / Origin 検証           — Host不一致421・クロスオリジンPOST403・WS 4403
  R10 アップロード上限             — 上限超過を413
  B4  チェックポイントのpickle無効 — モデル全体pickleは既定拒否、envオプトインで許可

参照: `tools/editor/backend/security.py`
"""

import io
import os
import sys
import zipfile

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from tools.editor.backend.demo import (  # noqa: E402
    UNSAFE_CKPT_MESSAGE,
    create_demo_model,
    demo_model_config,
    load_model_from_checkpoint,
    save_demo_checkpoint,
)
from tools.editor.backend.main import app  # noqa: E402
from tools.editor.backend.security import (  # noqa: E402
    WorkspacePathError,
    allowed_hosts,
    is_host_allowed,
    is_origin_allowed,
    max_upload_bytes,
    resolve_workspace_path,
)

LOCAL_BASE = 'http://localhost:8765'
LOCAL_ORIGIN = 'http://localhost:5173'


@pytest.fixture
def client():
    return TestClient(app, base_url=LOCAL_BASE)


# ---------------------------------------------------------------------------
# B3: ワークスペース内包チェック
# ---------------------------------------------------------------------------

def test_resolve_workspace_path_rejects_parent_escape():
    """`../` による workspace 外への脱出を拒否する。"""
    with pytest.raises(WorkspacePathError):
        resolve_workspace_path('../../etc/passwd')


def test_resolve_workspace_path_rejects_outside_absolute():
    """workspace 外の絶対パスを拒否する。"""
    with pytest.raises(WorkspacePathError):
        resolve_workspace_path('/etc/passwd')


def test_resolve_workspace_path_rejects_nul_byte():
    """NUL 文字入りパス(パス切り詰め攻撃)を拒否する。"""
    with pytest.raises(WorkspacePathError):
        resolve_workspace_path('README.md\x00.json')


def test_resolve_workspace_path_rejects_symlink_escape(tmp_path):
    """workspace 内シンボリックリンク経由の外部参照も拒否する。"""
    root = tmp_path / 'ws'
    root.mkdir()
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    link = root / 'link.json'
    try:
        link.symlink_to(outside)
    except OSError as e:
        # Windows は管理者権限か開発者モード無しに symlink を作れない(WinError 1314)
        pytest.skip(f'symlink を作れない環境: {e}')
    with pytest.raises(WorkspacePathError):
        resolve_workspace_path('link.json', workspace_root=root)


def test_resolve_workspace_path_rejects_bad_suffix():
    """許可拡張子以外を拒否する。"""
    with pytest.raises(WorkspacePathError):
        resolve_workspace_path('README.md', suffixes=('.json',))


def test_resolve_workspace_path_accepts_inside():
    """workspace 内の相対パスは解決できる。"""
    p = resolve_workspace_path('pyproject.toml', suffixes=('.toml',), must_exist=True)
    assert p.name == 'pyproject.toml'


@pytest.mark.parametrize('path', ['../../etc/passwd', '/etc/passwd'])
def test_ir_load_rejects_out_of_workspace(client, path):
    """POST /api/models/ir/load は workspace 外のパスを400で拒否する。"""
    res = client.post('/api/models/ir/load', json={'path': path},
                      headers={'Origin': LOCAL_ORIGIN})
    assert res.status_code == 400
    assert 'ワークスペース外' in res.json()['detail'] or '拡張子' in res.json()['detail']


def test_models_import_rejects_out_of_workspace(client):
    """POST /api/models/import も内包チェックで400を返す。"""
    res = client.post('/api/models/import', json={'path': '/tmp/evil.pt'},
                      headers={'Origin': LOCAL_ORIGIN})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# R6: Host / Origin 検証
# ---------------------------------------------------------------------------

def test_host_header_mismatch_is_421():
    """許可外 Host(DNSリバインディング)は 421 を返す。"""
    evil = TestClient(app, base_url='http://evil.example')
    res = evil.get('/api/datasets')
    assert res.status_code == 421


@pytest.mark.parametrize('base', [
    'http://localhost:8765', 'http://127.0.0.1:8765', 'http://localhost',
])
def test_loopback_host_is_allowed(base):
    """ループバック名義の Host はポートを問わず通る。"""
    c = TestClient(app, base_url=base)
    assert c.get('/api/datasets').status_code == 200


def test_duplicate_host_header_is_421(client):
    """Host ヘッダが重複したリクエスト(スマグリングの兆候)は 421 で拒否する。"""
    res = client.get('/api/datasets',
                     headers=[('Host', 'localhost:8765'), ('Host', 'evil.example')])
    assert res.status_code == 421


def test_duplicate_origin_header_is_403(client):
    """Origin ヘッダ重複も検証を素通りさせず 403 で拒否する。"""
    res = client.post('/api/models/demo', json={'k': 400, 'layers': 1},
                      headers=[('Origin', LOCAL_ORIGIN),
                               ('Origin', 'http://evil.example')])
    assert res.status_code == 403


def test_cross_origin_post_is_403(client):
    """状態変更メソッドでクロスオリジンの Origin は 403(CSRF対策)。"""
    res = client.post('/api/models/demo', json={'k': 400, 'layers': 1},
                      headers={'Origin': 'http://evil.example'})
    assert res.status_code == 403


def test_post_without_origin_is_allowed(client):
    """Origin 無し(curl等の非ブラウザ)は通す。"""
    res = client.post('/api/models/demo', json={'k': 400, 'layers': 1})
    assert res.status_code == 200


def test_loopback_origin_post_is_allowed(client):
    """ループバック Origin の POST は通る。"""
    res = client.post('/api/models/demo', json={'k': 400, 'layers': 1},
                      headers={'Origin': LOCAL_ORIGIN})
    assert res.status_code == 200


def test_is_origin_allowed_rules():
    """Origin 判定の単体仕様(null / 非http / 許可集合)。"""
    hosts = allowed_hosts()
    assert is_origin_allowed(None, hosts) is True       # 非ブラウザ
    assert is_origin_allowed('', hosts) is True
    assert is_origin_allowed('null', hosts) is False    # sandboxed iframe
    assert is_origin_allowed('file://', hosts) is False
    assert is_origin_allowed('http://localhost:5173', hosts) is True
    assert is_origin_allowed('http://127.0.0.1:1234', hosts) is True
    assert is_origin_allowed('http://evil.example', hosts) is False


def test_allowed_hosts_env_extends_and_ignores_wildcard(monkeypatch):
    """環境変数で許可を追加できるが、`0.0.0.0` 等は許可対象にしない。"""
    monkeypatch.setenv('GRIFFITH_EDITOR_ALLOWED_HOSTS', 'my-box.local, 0.0.0.0')
    hosts = allowed_hosts()
    assert 'my-box.local' in hosts
    assert '0.0.0.0' not in hosts
    assert is_host_allowed('my-box.local:8765', hosts) is True
    assert is_host_allowed(None, hosts) is False


def test_websocket_rejects_cross_origin():
    """WS も許可外 Origin を拒否する(CSWSH対策、R6c)。"""
    c = TestClient(app, base_url=LOCAL_BASE)
    with pytest.raises(Exception):  # noqa: B017 — close(4403) は接続例外になる
        with c.websocket_connect(
            '/ws/train/no-such-session',
            headers={'Origin': 'http://evil.example'},
        ) as ws:
            ws.receive_text()


# ---------------------------------------------------------------------------
# R10: アップロード上限
# ---------------------------------------------------------------------------

def test_max_upload_bytes_env(monkeypatch):
    """上限は環境変数で上書きでき、不正値は既定へフォールバックする。"""
    monkeypatch.setenv('GRIFFITH_EDITOR_MAX_UPLOAD_MB', '1')
    assert max_upload_bytes() == 1024 * 1024
    monkeypatch.setenv('GRIFFITH_EDITOR_MAX_UPLOAD_MB', 'abc')
    assert max_upload_bytes() == 512 * 1024 * 1024


def _import_client(tmp_path, monkeypatch):
    """取込APIだけを載せた単体アプリ(STORE_ROOTはtmp_pathへ退避)。"""
    from fastapi import FastAPI

    from tools.editor.backend import dataset_import as di

    monkeypatch.setattr(di, 'STORE_ROOT', tmp_path / 'store')
    monkeypatch.setattr(di, '_registry', {})
    sub = FastAPI()
    sub.include_router(di.router)
    return TestClient(sub, base_url=LOCAL_BASE)


def test_csv_upload_over_limit_is_413(tmp_path, monkeypatch):
    """上限超過のCSVアップロードは 413 を返す(全量をメモリに載せない)。"""
    monkeypatch.setenv('GRIFFITH_EDITOR_MAX_UPLOAD_MB', '0.001')  # 1KiB
    c = _import_client(tmp_path, monkeypatch)
    big = b'x,y\n' + b'1,1\n' * 2000
    res = c.post('/api/datasets/import/csv',
                 files={'file': ('big.csv', io.BytesIO(big), 'text/csv')})
    assert res.status_code == 413


def test_zip_upload_over_limit_is_413(tmp_path, monkeypatch):
    """上限超過のzipアップロードは 413 を返す。"""
    monkeypatch.setenv('GRIFFITH_EDITOR_MAX_UPLOAD_MB', '0.001')
    c = _import_client(tmp_path, monkeypatch)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('a/1.txt', 'x' * 100_000)
    res = c.post('/api/datasets/import/images',
                 files={'file': ('big.zip', io.BytesIO(buf.getvalue()),
                                 'application/zip')})
    assert res.status_code == 413


def test_zip_bomb_uncompressed_size_is_413(tmp_path, monkeypatch):
    """圧縮後は小さいが展開後が上限超過のzipも 413 で弾く。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('a/1.txt', 'x' * 4_000_000)  # 圧縮後は数KB
    payload = buf.getvalue()
    monkeypatch.setenv('GRIFFITH_EDITOR_MAX_UPLOAD_MB', '1')  # 1MiB
    assert len(payload) < 1024 * 1024, '圧縮後は上限内である前提'
    c = _import_client(tmp_path, monkeypatch)
    res = c.post('/api/datasets/import/images',
                 files={'file': ('bomb.zip', io.BytesIO(payload),
                                 'application/zip')})
    assert res.status_code == 413
    assert '展開後' in res.json()['detail']


# ---------------------------------------------------------------------------
# B4: チェックポイントの pickle 無効化
# ---------------------------------------------------------------------------

def test_state_dict_checkpoint_loads_without_pickle(tmp_path):
    """state_dict形式(`state_dict`+`config`)は安全モードのまま読める。"""
    path = save_demo_checkpoint(tmp_path / 'demo.pt', k=400, layers=1)
    model = load_model_from_checkpoint(str(path))
    out = model(torch.zeros(1, 1, 28, 28))
    assert out.shape == (1, 10)


def test_bundled_sample_checkpoint_is_state_dict_format():
    """同梱デモckptがstate_dict形式であること(モデル全体pickleではない)。"""
    from tools.editor.backend.demo import PROJECT_ROOT

    path = PROJECT_ROOT / 'tools' / 'editor' / 'samples' / 'demo_mnist_tiny.pt'
    assert path.exists(), f'同梱デモckptが無い: {path}'
    raw = torch.load(path, map_location='cpu', weights_only=True)  # pickle実行なしで読める
    assert set(raw) == {'state_dict', 'config'}
    assert raw['config']['arch'] == 'sequential_lut'
    model = load_model_from_checkpoint(str(path))
    assert model(torch.zeros(1, 1, 28, 28)).shape == (1, 10)


def test_whole_model_pickle_checkpoint_is_rejected_by_default(tmp_path, monkeypatch):
    """モデル全体をpickleした旧形式は既定で拒否し、対処を案内する。"""
    monkeypatch.delenv('GRIFFITH_ALLOW_UNSAFE_CKPT', raising=False)
    path = tmp_path / 'whole.pt'
    torch.save(create_demo_model(k=400, layers=1), path)
    with pytest.raises(ValueError) as ei:
        load_model_from_checkpoint(str(path))
    assert 'GRIFFITH_ALLOW_UNSAFE_CKPT=1' in str(ei.value)
    assert str(ei.value) == UNSAFE_CKPT_MESSAGE


def test_whole_model_pickle_checkpoint_opt_in_env(tmp_path, monkeypatch):
    """`GRIFFITH_ALLOW_UNSAFE_CKPT=1` の明示オプトイン時だけ旧形式を読む。"""
    path = tmp_path / 'whole.pt'
    torch.save(create_demo_model(k=400, layers=1), path)
    monkeypatch.setenv('GRIFFITH_ALLOW_UNSAFE_CKPT', '1')
    model = load_model_from_checkpoint(str(path))
    assert isinstance(model, torch.nn.Module)


def test_whole_model_pickle_checkpoint_opt_in_kwarg(tmp_path, monkeypatch):
    """`allow_unsafe=True` の引数オプトインでも読める。"""
    monkeypatch.delenv('GRIFFITH_ALLOW_UNSAFE_CKPT', raising=False)
    path = tmp_path / 'whole.pt'
    torch.save(create_demo_model(k=400, layers=1), path)
    model = load_model_from_checkpoint(str(path), allow_unsafe=True)
    assert isinstance(model, torch.nn.Module)


def test_state_dict_checkpoint_without_config_is_error(tmp_path):
    """state_dictだけで構造情報が無いckptは明確なエラーにする。"""
    path = tmp_path / 'nocfg.pt'
    torch.save({'state_dict': create_demo_model(k=400, layers=1).state_dict()}, path)
    with pytest.raises(ValueError, match='config'):
        load_model_from_checkpoint(str(path))


def test_models_import_endpoint_reads_bundled_sample(client):
    """POST /api/models/import が同梱state_dict ckptを読めること(経路の通し)。"""
    res = client.post('/api/models/import',
                      json={'path': 'tools/editor/samples/demo_mnist_tiny.pt'},
                      headers={'Origin': LOCAL_ORIGIN})
    assert res.status_code == 200, res.text
    assert res.json()['input_width'] == 784


def test_demo_model_config_matches_create_demo_model():
    """config から組み立てたモデルが `create_demo_model` と同形であること。"""
    from tools.editor.backend.demo import build_model_from_config

    cfg = demo_model_config(k=400, layers=2)
    a = build_model_from_config(cfg)
    b = create_demo_model(k=400, layers=2)
    assert [type(m).__name__ for m in a] == [type(m).__name__ for m in b]
    assert set(a.state_dict()) == set(b.state_dict())


def test_history_chunk_roundtrip_is_weights_only(tmp_path):
    """履歴チャンクが `weights_only=True` で読み戻せること(add_safe_globals)。"""
    from tools.editor.backend.history import LayerRecord, StepSnapshot

    snap = StepSnapshot(step=1, epoch=0, loss=0.5, acc=0.1, ts=0.0, method='bp',
                        layers={'l0': LayerRecord(weights=torch.zeros(4, 4))})
    path = tmp_path / 'chunk.pt'
    torch.save([snap], path)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    assert loaded[0].step == 1
    assert loaded[0].layers['l0'].weights.shape == (4, 4)
