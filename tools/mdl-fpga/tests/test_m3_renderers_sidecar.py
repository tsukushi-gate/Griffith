"""サイドカー renderers.py のホットリロードと耐障害性。"""

import os
import time

import numpy as np

from mdl_fpga.viewer.renderers import RendererHub


def _bump_mtime(path):
    """mtime 解像度に依存しないよう明示的に進める。"""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def _ctx(width):
    return {'width': width, 'full_width': width, 'cycle': 0, 'target': 't',
            'slice': None, 'meta': {'nodes': []}, 'target_meta': {},
            'args': {}}


def test_sidecar_load_and_hot_reload(tmp_path):
    model = tmp_path / 'm.ir.json'
    model.write_text('{}', encoding='utf-8')
    sidecar = tmp_path / 'renderers.py'
    sidecar.write_text(
        "def double(bits, ctx):\n"
        "    return {'kind': 'text', 'text': str(int(sum(bits)) * 2)}\n",
        encoding='utf-8')

    hub = RendererHub(str(model))
    listing = hub.listing()
    assert listing['sidecar'] == ['double']
    assert listing['sidecar_error'] is None

    bits = np.array([1, 1, 1], dtype=np.uint8)
    assert hub.resolve('double', _ctx(3))(bits, _ctx(3))['text'] == '6'

    # 書き換え → 自動リロード
    time.sleep(0.01)
    sidecar.write_text(
        "def double(bits, ctx):\n"
        "    return {'kind': 'text', 'text': 'v2:' + str(int(sum(bits)))}\n"
        "def extra(bits, ctx):\n"
        "    return {'kind': 'text', 'text': 'x'}\n",
        encoding='utf-8')
    _bump_mtime(sidecar)
    assert hub.resolve('double', _ctx(3))(bits, _ctx(3))['text'] == 'v2:3'
    assert set(hub.listing()['sidecar']) == {'double', 'extra'}


def test_sidecar_syntax_error_keeps_old_module(tmp_path):
    model = tmp_path / 'm.ir.json'
    model.write_text('{}', encoding='utf-8')
    sidecar = tmp_path / 'renderers.py'
    sidecar.write_text(
        "def ok(bits, ctx):\n    return {'kind': 'text', 'text': 'v1'}\n",
        encoding='utf-8')
    hub = RendererHub(str(model))
    bits = np.zeros(1, dtype=np.uint8)
    assert hub.resolve('ok', _ctx(1))(bits, _ctx(1))['text'] == 'v1'

    # 構文エラーへ書き換え → 旧モジュール維持+error 記録
    sidecar.write_text('def broken(:\n', encoding='utf-8')
    _bump_mtime(sidecar)
    listing = hub.listing()
    assert listing['sidecar_error'] is not None
    assert hub.resolve('ok', _ctx(1))(bits, _ctx(1))['text'] == 'v1'


def test_no_sidecar_builtins_only(tmp_path):
    model = tmp_path / 'm.ir.json'
    model.write_text('{}', encoding='utf-8')
    hub = RendererHub(str(model))
    listing = hub.listing()
    assert listing['sidecar'] == []
    assert 'bits' in listing['builtins'] and 'auto' in listing['builtins']


def test_builtin_shadowing_by_sidecar(tmp_path):
    model = tmp_path / 'm.ir.json'
    model.write_text('{}', encoding='utf-8')
    (tmp_path / 'renderers.py').write_text(
        "def uint(bits, ctx):\n"
        "    return {'kind': 'text', 'text': 'custom'}\n",
        encoding='utf-8')
    hub = RendererHub(str(model))
    bits = np.array([1], dtype=np.uint8)
    assert hub.resolve('uint', _ctx(1))(bits, _ctx(1))['text'] == 'custom'
