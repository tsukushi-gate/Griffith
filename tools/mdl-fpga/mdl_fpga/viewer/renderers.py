"""値レンダラ(内蔵+サイドカー renderers.py)。

payload 契約(フロントの PayloadView と一致させる):
    {'kind':'error','message':str}
    {'kind':'text','text':str}
    {'kind':'bits','hex':str,'width':int}
    {'kind':'bars','values':list[float],'labels':list[str]|None,'argmax':int|None}
    {'kind':'image','data_url':str,'w':int,'h':int}

レンダラ関数のシグネチャ(内蔵・ユーザー定義とも同一):
    def my_renderer(bits: np.ndarray, ctx: dict) -> dict
bits は LSB-first のスライス済み 0/1 配列。ctx には width/cycle/target/slice/
meta/target_meta/args が入る。

サイドカー: <モデルのディレクトリ>/renderers.py の公開関数(先頭 _ 以外)を
mtime 変更検知で自動リロードする。ロード方式は editor backend の
python_importer.py(spec_from_file_location+ユニーク名+sys.path ガード)を踏襲。
本ファイルはサーバプロセス内で実行される(ユーザー自身のローカルコード前提。README 参照)。

注意: viewer 配下では `from __future__ import annotations` を使わない。
"""

import base64
import hashlib
import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 内蔵レンダラ


def _pack_hex(bits: np.ndarray) -> str:
    v = 0
    for i, b in enumerate(bits):
        v |= int(b) << i
    return f'{v:x}'


def render_bits(bits: np.ndarray, ctx: dict) -> dict:
    return {'kind': 'bits', 'hex': _pack_hex(bits), 'width': int(bits.shape[0])}


def render_hex(bits: np.ndarray, ctx: dict) -> dict:
    digits = (int(bits.shape[0]) + 3) // 4
    return {'kind': 'text', 'text': '0x' + _pack_hex(bits).zfill(digits)}


def render_uint(bits: np.ndarray, ctx: dict) -> dict:
    return {'kind': 'text', 'text': str(int(_pack_hex(bits), 16))}


def render_bars(bits: np.ndarray, ctx: dict) -> dict:
    """score_bits 刻みのパック済みスコア列 → 棒グラフ(core.py の規約準拠)。"""
    tm = ctx.get('target_meta') or {}
    score_bits = int(ctx['args'].get('score_bits')
                     or tm.get('score_bits')
                     or ctx['meta'].get('score_bits') or 0)
    if score_bits <= 0:
        raise ValueError('score_bits が不明(args か group_sum ノードで指定)')
    width = int(bits.shape[0])
    k = int(ctx['args'].get('k') or width // score_bits)
    vals = []
    for c in range(k):
        seg = bits[c * score_bits:(c + 1) * score_bits]
        vals.append(float(int(_pack_hex(seg), 16)))
    labels = ctx['args'].get('labels') or [f'c{c}' for c in range(k)]
    argmax = int(np.argmax(vals)) if vals else None
    return {'kind': 'bars', 'values': vals, 'labels': labels, 'argmax': argmax}


def render_image(bits: np.ndarray, ctx: dict) -> dict:
    """bit 列 → 2D 画像(PNG data URL)。任意 bit 数を任意の横幅で並べる。

    横幅の決定順: args['cols'] > shape(args か入力ポート、bit 数一致時) >
    正方形(ceil(sqrt(n)))。端数は 0 パディング(どんな bit 数でも表示できる)。
    """
    try:
        from PIL import Image
    except ImportError:
        return {'kind': 'error',
                'message': 'pillow 未導入(uv sync --extra fpga で導入)'}
    n = int(bits.shape[0])
    cols = ctx['args'].get('cols')
    if cols:
        w = max(1, int(cols))
    else:
        shape = ctx['args'].get('shape') or ctx['meta'].get('input_shape')
        if shape and len(shape) == 2 and int(shape[0]) * int(shape[1]) == n:
            w = int(shape[1])
        else:
            w = max(1, math.isqrt(n - 1) + 1 if n > 1 else 1)  # ceil(sqrt(n))
    h = max(1, math.ceil(n / w))
    padded = np.zeros(h * w, dtype=np.uint8)
    padded[:n] = bits
    arr = (padded.reshape(h, w) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    return {'kind': 'image', 'data_url': f'data:image/png;base64,{data}',
            'w': w, 'h': h}


def _resolve_auto(ctx: dict):
    """auto: 対象のメタ情報から最適な内蔵レンダラを選ぶ。"""
    tm = ctx.get('target_meta') or {}
    if tm.get('score_bits') or (ctx['target'] in ('y', 'y_comb')
                                and ctx['meta'].get('score_bits')):
        return render_bars
    shape = ctx['meta'].get('input_shape')
    if ctx['target'] == 'x' and shape and len(shape) == 2 \
            and ctx.get('slice') is None:
        return render_image
    if ctx['width'] <= 64:
        return render_bits
    return render_hex


BUILTINS = {
    'bits': render_bits,
    'hex': render_hex,
    'uint': render_uint,
    'bars': render_bars,
    'image': render_image,
}


# ------------------------------------------------------- サイドカー(ユーザー定義)


class SidecarRenderers:
    """<モデルdir>/renderers.py の公開関数を mtime 監視付きで提供する。"""

    def __init__(self, model_path: str) -> None:
        self.path = Path(model_path).resolve().parent / 'renderers.py'
        self._module = None
        self._mtime_ns: int | None = None
        self.error: str | None = None

    def _load(self) -> None:
        mod_name = ('mdl_fpga_sidecar_'
                    + hashlib.sha1(str(self.path).encode()).hexdigest()[:12])
        spec = importlib.util.spec_from_file_location(mod_name, str(self.path))
        module = importlib.util.module_from_spec(spec)
        parent = str(self.path.parent)
        sys.path.insert(0, parent)
        try:
            spec.loader.exec_module(module)
        finally:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass
        self._module = module
        self.error = None

    def refresh(self) -> None:
        """mtime が変わっていたら再読込。失敗時は旧モジュール維持+error 記録。"""
        if not self.path.exists():
            self._module = None
            self._mtime_ns = None
            self.error = None
            return
        mtime = self.path.stat().st_mtime_ns
        if mtime == self._mtime_ns:
            return
        try:
            self._load()
            self._mtime_ns = mtime
        except Exception as e:  # noqa: BLE001 - 旧モジュールで継続
            self.error = f'{type(e).__name__}: {e}'
            self._mtime_ns = mtime   # 同じ壊れたファイルを毎フレーム再ロードしない

    def names(self) -> list[str]:
        if self._module is None:
            return []
        return sorted(
            name for name, obj in vars(self._module).items()
            if callable(obj) and not name.startswith('_')
            and getattr(obj, '__module__', None) == self._module.__name__)

    def get(self, name: str):
        if self._module is None:
            return None
        fn = getattr(self._module, name, None)
        return fn if callable(fn) else None


class RendererHub:
    """内蔵+サイドカーの名前解決(サイドカーが同名を上書き可能)。"""

    def __init__(self, model_path: str) -> None:
        self.sidecar = SidecarRenderers(model_path)

    def resolve(self, name: str, ctx: dict):
        self.sidecar.refresh()
        if name == 'auto':
            return _resolve_auto(ctx)
        fn = self.sidecar.get(name)
        if fn is not None:
            return fn
        if name in BUILTINS:
            return BUILTINS[name]
        hint = f'(サイドカー読込エラー: {self.sidecar.error})' \
            if self.sidecar.error else ''
        raise ValueError(
            f'レンダラが見つからない: {name}{hint}。'
            f'利用可能: auto, {", ".join(BUILTINS)}, '
            f'{self.sidecar.path.name} 内の公開関数')

    def listing(self) -> dict:
        self.sidecar.refresh()
        return {
            'builtins': ['auto', *BUILTINS.keys()],
            'sidecar': self.sidecar.names(),
            'sidecar_path': str(self.sidecar.path),
            'sidecar_error': self.sidecar.error,
        }
