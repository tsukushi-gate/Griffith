"""ビューアサーバの可変状態(SessionHolder)。

セッション切替(プロジェクト/ボードペイン)のために、create_app の閉包に
散っていた状態(session / cfg / registry / hub / cfg_path)を1箇所に束ねる。
切替は「新セッションを完全に起動 → 原子的に交換(_attach)→ 旧を後始末」
で行い、起動失敗時は何もしない=旧セッション継続がそのままロールバック。

注意: viewer 配下では `from __future__ import annotations` を使わない
(FastAPI の WS 型解決が壊れる実測罠、server.py 冒頭参照)。
"""

import asyncio
import json
import os
import time
from pathlib import Path

from .persistence import load_config, sidecar_path
from .probes import ProbeRegistry
from .renderers import RendererHub

TOOL_DIR = Path(__file__).resolve().parents[2]   # tools/mdl-fpga/

# システムモードを持つ example とその生成物ディレクトリ(明示レジストリ)。
# 生成物はビルドスクリプトが作る=無ければ ready:false で入口を塞ぐ
SYSTEM_EXAMPLES = {
    'mnist/mnist24k': (
        'runs/mnist_demo/quartus',   # build_mnist_demo.py の projdir=wd/'quartus'
        'uv run python tools/mdl-fpga/examples/mnist/build_mnist_demo.py '
        'で生成(system_meta.json / rom.hex が必要)'),
}


def examples_dir() -> Path:
    env = os.environ.get('MDL_FPGA_EXAMPLES_DIR')
    return Path(env) if env else TOOL_DIR / 'examples'


def recent_path() -> Path:
    env = os.environ.get('MDL_FPGA_RECENT_PATH')
    return Path(env) if env else (Path.home() / '.mdl-fpga'
                                  / 'viewer_recent.json')


def load_recent() -> list:
    p = recent_path()
    if not p.exists():
        return []
    try:
        items = json.loads(p.read_text(encoding='utf-8'))
        return items[:10] if isinstance(items, list) else []
    except Exception:  # noqa: BLE001 - 壊れていても機能を殺さない
        return []


def append_recent(entry: dict) -> None:
    p = recent_path()
    items = [e for e in load_recent()
             if not (e.get('model') == entry.get('model')
                     and e.get('system') == entry.get('system'))]
    items.insert(0, entry)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(items[:10], ensure_ascii=False, indent=1),
                   encoding='utf-8')
    os.replace(tmp, p)


def _system_ready(runs_rel: str) -> tuple:
    d = TOOL_DIR / runs_rel
    ok = (d / 'system_meta.json').exists() and (d / 'rom.hex').exists()
    return ok, str(d)


def list_examples() -> list:
    """examples/ の *.ir.json を列挙(「開く」ダイアログのイグザンプルタブ)。"""
    root = examples_dir()
    out = []
    if not root.exists():
        return out
    for p in sorted(root.glob('**/*.ir.json')):
        if '__pycache__' in p.parts:
            continue
        key = p.relative_to(root).as_posix()[:-len('.ir.json')]
        modes = [{'mode': 'core', 'ready': True}]
        if key in SYSTEM_EXAMPLES:
            runs_rel, hint = SYSTEM_EXAMPLES[key]
            ready, d = _system_ready(runs_rel)
            entry = {'mode': 'system', 'ready': ready, 'system_dir': d}
            if not ready:
                entry['hint'] = hint
            modes.append(entry)
        out.append({'key': key, 'name': p.name[:-len('.ir.json')],
                    'path': str(p), 'modes': modes})
    return out


def resolve_open_request(doc: dict) -> dict:
    """open 要求 → {'model','system_dir','pipelined'}(SimSession 生成前の検証)。

    失敗は ValueError(メッセージに次の一手)。旧セッションは呼び出し側で
    継続させる=ここで例外を出す限り何も壊れない。
    """
    pipelined = bool(doc.get('pipelined'))
    if doc.get('example'):
        key = doc['example']
        hits = [e for e in list_examples() if e['key'] == key]
        if not hits:
            known = ', '.join(e['key'] for e in list_examples())
            raise ValueError(f'未知の example: {key}(利用可能: {known})')
        ex = hits[0]
        mode = doc.get('mode') or 'core'
        m = next((m for m in ex['modes'] if m['mode'] == mode), None)
        if m is None:
            raise ValueError(f'{key} は mode={mode} を持たない')
        if not m.get('ready'):
            raise ValueError(
                f'{key} の {mode} モードは生成物が未ビルド。'
                f'{m.get("hint", "ビルドスクリプトを実行")}')
        return {'model': ex['path'],
                'system_dir': m.get('system_dir'),
                'pipelined': pipelined, 'example': key}

    model = doc.get('model')
    if not model:
        raise ValueError('model か example のどちらかが必要')
    if not Path(model).exists():
        raise ValueError(f'モデルが存在しない: {model}')
    system_dir = doc.get('system')
    if system_dir:
        d = Path(system_dir)
        if not (d / 'system_meta.json').exists():
            raise ValueError(
                f'{d} に system_meta.json が無い(システムモードの生成物'
                'ディレクトリを指定する。build_mnist_demo.py 等が生成)')
    return {'model': str(model), 'system_dir': system_dir,
            'pipelined': pipelined, 'example': None}


def make_recent_entry(spec: dict, name) -> dict:
    return {'model': spec['model'], 'name': name,
            'system': spec['system_dir'], 'pipelined': spec['pipelined'],
            'opened_at': time.strftime('%Y-%m-%dT%H:%M:%S')}


class SessionHolder:
    """現在の SimSession と付随する表示状態。epoch は切替ごとに +1。"""

    def __init__(self, session):
        self.session = session
        self.epoch = 0
        self.switch_lock = asyncio.Lock()   # プロジェクト切替の直列化
        self.detect_lock = asyncio.Lock()   # JTAG 検出の直列化(排他資源)
        self.sockets = set()                # 接続中 WS(切替通知用)
        self.detect_impl = None             # テスト差替え seam(None=lazy import)
        self.advice_cache = {}              # (model, board_key) -> advice dict
        self._attach(session)

    @property
    def mode(self) -> str:
        return 'system' if self.session.system_dir else 'core'

    def _attach(self, session) -> None:
        """session に付随する表示状態を張り替える(初期化と交換の共通処理)。"""
        self.session = session
        self.hub = RendererHub(session.model)
        self.cfg_path = sidecar_path(session.model, mode=self.mode)
        self.cfg = load_config(self.cfg_path)
        self.registry = ProbeRegistry()
        self.registry.replace_all(self.cfg.probes)
        self.advice_cache = {}

    def swap(self, new_session) -> None:
        """準備完了済みの新セッションへ原子的に交換(await を挟まないこと)。"""
        self._attach(new_session)
        self.epoch += 1
