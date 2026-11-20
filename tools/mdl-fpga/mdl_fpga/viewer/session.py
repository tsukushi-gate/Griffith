"""対話シミュレーションセッション(サーバ側)。

HDL 生成 → cocotb/iverilog ビルド → vvp 起動(別スレッドで runner.test が
ブロック実行)→ ブリッジの TCP 接続受理 → JSON 行の同期リクエスト。
リスク R2(GUI↔cocotb 橋)の実体。エディタの学習 WS と同型の
「サーバが唯一の真実を持ち、ブラウザへ中継する」構造にする。
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from griffith_logic.export.lowering import lower_to_primitives

from ..check import default_workdir
from ..hdlgen.core import generate_core
from ..ir_io import load_graph
from ..toolchain import discover
from .view_top import generate_view_top

VIEWER_DIR = Path(__file__).resolve().parent
_PATH_READY = False


def _ensure_sim_path() -> None:
    """oss-cad-suite の bin+lib を PATH 前置(先行実験の教訓。1回だけ)。"""
    global _PATH_READY
    if _PATH_READY:
        return
    tc = discover()
    if tc.oss_root is None:
        raise RuntimeError('oss-cad-suite が見つからない(doctor で確認)')
    bin_ = str(tc.oss_root / 'bin')
    lib_ = str(tc.oss_root / 'lib')
    os.environ['PATH'] = os.pathsep.join([bin_, lib_,
                                          os.environ.get('PATH', '')])
    _PATH_READY = True


def build_viewer_meta(graph, gen, system: dict | None = None) -> dict:
    """フロントエンドが必要とする構造情報(名前マップの拡張)。

    回路図描画のため edges / inputs / outputs も含める(v2)。ノードの params
    (indices/truth_tables 等の大物)は含めず、/api/node/{id} で必要時に取る。
    system(system_meta.json)を渡すとシステムモード: 各ノードに DUT 相対の
    信号 path を付与し、system 型ノード/エッジ/extra_signals を合成する。
    """
    core_prefix = 'u_sys.u_core.' if system else 'u_core.'
    nodes = []
    for node_id, entry in gen.name_map.items():
        if entry.get('type') == 'input':
            continue
        nodes.append({
            'id': node_id,
            'wire': entry['wire'],
            'width': entry['width'],
            'type': entry['type'],
            'alias': entry.get('alias', False),
            'score_bits': entry.get('score_bits'),
            'stage': entry.get('stage'),
            'path': core_prefix + entry['wire'],
        })
    meta = {
        'name': graph.metadata.get('name', 'model'),
        'in_width': gen.in_width,
        'out_width': gen.out_width,
        'score_bits': gen.score_bits,
        'latency': getattr(gen, 'latency', 0),
        'pipelined': getattr(gen, 'pipelined', False),
        'k': None,
        'input_semantics': getattr(graph.inputs[0], 'semantics', None),
        'input_shape': getattr(graph.inputs[0], 'shape', None),
        'nodes': nodes,
        'inputs': [{'id': p.id, 'width': p.width,
                    'shape': getattr(p, 'shape', None),
                    'semantics': getattr(p, 'semantics', None)}
                   for p in graph.inputs],
        'outputs': [{'id': p.id, 'width': p.width,
                     'semantics': getattr(p, 'semantics', None)}
                    for p in graph.outputs],
        'edges': [{'src': e.src, 'dst': e.dst, 'slot': e.dst_slot}
                  for e in graph.edges],
    }
    for n in graph.nodes:
        if n.type == 'group_sum':
            meta['k'] = n.params['k']

    if system is not None:
        meta['mode'] = 'system'
        meta['latency'] = system.get('latency', meta['latency'])
        meta['allow_set_x'] = bool(system.get('allow_set_x', False))
        meta['n_images'] = system.get('n_images')
        meta['extra_signals'] = system.get('extra_signals', {})
        meta['nodes'] = meta['nodes'] + list(system.get('nodes', []))
        meta['edges'] = meta['edges'] + list(system.get('edges', []))
    else:
        meta['mode'] = 'core'
        meta['allow_set_x'] = True
    return meta


def _json_safe(value):
    """numpy 混じりの params を JSON 化可能な素の Python 型へ変換する。"""
    import numpy as np
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class SimSession:
    model: str
    workdir: Path | None = None
    pipelined: bool = False   # True: 各層FFコア(値が1clkごとに層を進む)
    system_dir: Path | str | None = None   # 分類システム生成物のディレクトリ
    meta: dict = field(default_factory=dict)
    cycle: int = 0
    graph: object = None   # lowered IRGraph(/api/node/{id} 用に保持)

    _sources: list = field(default_factory=list)
    _sock: socket.socket | None = None
    _file = None
    _thread: threading.Thread | None = None
    _listener: socket.socket | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _runner_error: Exception | None = None

    def start(self, timeout_s: float = 120.0) -> dict:
        _ensure_sim_path()
        wd = Path(self.workdir) if self.workdir \
            else default_workdir(self.model, 'view')
        wd.mkdir(parents=True, exist_ok=True)
        wd = wd.resolve()   # 子プロセス(vvp)は cwd が異なる=相対パス禁止
        self.workdir = wd

        graph = lower_to_primitives(load_graph(self.model))
        self.graph = graph

        if self.system_dir is not None:
            # システムモード: 生成物(top.v 等+rom.hex+system_meta.json)を使う
            sysdir = Path(self.system_dir)
            system_meta = json.loads(
                (sysdir / 'system_meta.json').read_text(encoding='utf-8'))
            src_files = ['top_view.v', 'top.v', 'sys_rom.v',
                         'mdl_core_pipe.v', 'seg7.v']
            for f in src_files:
                shutil.copyfile(sysdir / f, wd / f)
            # $readmemh の cwd 罠: workdir と sim_build の両方へコロケート
            (wd / 'sim_build').mkdir(exist_ok=True)
            shutil.copyfile(sysdir / 'rom.hex', wd / 'rom.hex')
            shutil.copyfile(sysdir / 'rom.hex', wd / 'sim_build' / 'rom.hex')
            gen = generate_core(graph, pipelined=True)
            self.meta = build_viewer_meta(graph, gen, system=system_meta)
            self._sources = src_files
        else:
            gen = generate_core(graph, pipelined=self.pipelined)
            (wd / 'mdl_core.v').write_text(gen.verilog, encoding='utf-8')
            (wd / 'top_view.v').write_text(
                generate_view_top(gen.in_width, gen.out_width,
                                  pipelined=self.pipelined), encoding='utf-8')
            self.meta = build_viewer_meta(graph, gen)
            self._sources = ['top_view.v', 'mdl_core.v']
        meta_path = wd / 'viewer_meta.json'
        meta_path.write_text(json.dumps(self.meta, ensure_ascii=False,
                                        indent=1), encoding='utf-8')

        # ブリッジ受け口(エフェメラルポート)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(timeout_s)
        self._listener = listener
        port = listener.getsockname()[1]

        # cocotb はテストモジュールを test_dir 基準で import する(ランナーが
        # 子プロセスの PYTHONPATH を test_dir で組むため、親の PYTHONPATH は
        # 届かない=実測)。ブリッジモジュールを workdir へコピーして同居させる
        shutil.copyfile(VIEWER_DIR / 'cocotb_bridge.py', wd / 'cocotb_bridge.py')
        os.environ['MDL_FPGA_BRIDGE_PORT'] = str(port)
        os.environ['MDL_FPGA_VIEW_META'] = str(meta_path)

        def _run():
            try:
                from cocotb_tools.runner import get_runner
                runner = get_runner('icarus')
                runner.build(
                    verilog_sources=[str(wd / f) for f in self._sources],
                    hdl_toplevel='top_view',
                    build_dir=str(wd / 'sim_build'),
                    build_args=['-g2012'],
                    always=True,
                )
                runner.test(
                    hdl_toplevel='top_view',
                    test_module='cocotb_bridge',
                    build_dir=str(wd / 'sim_build'),
                    test_dir=str(wd),
                )
            except Exception as e:  # noqa: BLE001 - accept 側で報告する
                self._runner_error = e

        self._thread = threading.Thread(target=_run, daemon=True,
                                        name='mdl-fpga-sim')
        self._thread.start()

        # accept は短いスライスでポーリングし、ランナースレッドが先に死んだら
        # 即座に失敗を報告する(import 失敗等でも runner.test は例外を出さず
        # 正常 return することがある=実測。results.xml とログを見よ)
        listener.settimeout(2.0)
        deadline = time.monotonic() + timeout_s
        conn = None
        while conn is None:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                if not self._thread.is_alive():
                    detail = f'runner error: {self._runner_error}' \
                        if self._runner_error else \
                        f'シミュレータが接続前に終了(ログ/results.xml: {wd})'
                    raise RuntimeError(f'ブリッジ起動失敗: {detail}') from None
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f'ブリッジ接続タイムアウト(ログ: {wd})') from None
        conn.settimeout(60)
        self._sock = conn
        self._file = conn.makefile('rw', encoding='utf-8', newline='\n')

        hello_line = self._file.readline()
        if not hello_line.strip():
            raise RuntimeError(
                f'ブリッジが無言で切断(cocotb ログ/results.xml: {wd})')
        hello = json.loads(hello_line)
        if not hello.get('ok'):
            raise RuntimeError(f'ブリッジ初期化失敗: {hello.get("fatal")}')
        assert hello.get('hello'), f'ブリッジ初期応答が不正: {hello}'
        return self.meta

    def node_detail(self, node_id: str) -> dict:
        """ノードの params(indices/truth_tables 等)を返す(ゲート展開用)。

        未知の node_id は KeyError(サーバ側で 404 に変換)。
        """
        assert self.graph is not None, 'セッション未開始'
        for n in self.graph.nodes:
            if n.id == node_id:
                return {'id': n.id, 'type': n.type,
                        'params': _json_safe(n.params)}
        raise KeyError(node_id)

    def request(self, obj: dict) -> dict:
        assert self._file is not None, 'セッション未開始'
        with self._lock:
            self._file.write(json.dumps(obj) + '\n')
            self._file.flush()
            line = self._file.readline()
        if not line:
            raise RuntimeError('ブリッジが切断された(vvp 終了?)')
        resp = json.loads(line)
        if 'cycle' in resp:
            self.cycle = resp['cycle']
        return resp

    # --- 高水準 API(WS ハンドラから呼ぶ) ---

    def reset(self) -> dict:
        return self.request({'cmd': 'reset'})

    def set_x(self, hex_value: str) -> dict:
        return self.request({'cmd': 'set_x', 'hex': hex_value})

    def step(self, n: int = 1) -> dict:
        return self.request({'cmd': 'step', 'n': n})

    def step_until(self, key: str, bit_slice=None, max_steps: int = 1000) -> dict:
        return self.request({'cmd': 'step_until', 'key': key,
                             'slice': bit_slice, 'max': max_steps})

    def read(self) -> dict:
        return self.request({'cmd': 'read'})

    def close(self) -> None:
        try:
            if self._file is not None:
                with self._lock:
                    self._file.write(json.dumps({'cmd': 'quit'}) + '\n')
                    self._file.flush()
                    self._file.readline()
        except Exception:  # noqa: BLE001 - 終了処理は失敗しても続行
            pass
        for s in (self._file, self._sock, self._listener):
            try:
                if s is not None:
                    s.close()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=15)
