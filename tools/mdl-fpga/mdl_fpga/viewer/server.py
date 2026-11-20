"""ビューアサーバ(mdl-fpga view)。

FastAPI + WebSocket。ブラウザ ↔ 本サーバ ↔ cocotb ブリッジ(SimSession)の
中継役で、シミュレーションの真実は常に RTL 側にある(フロントは表示のみ)。
v2: プローブ(表示専用の値後処理)・レンダラ・レイアウト永続化・
React フロント(frontend/dist)の静的配信を追加。dist が無い環境では
旧単一 HTML(static/index.html)へフォールバックする。

注意: 本モジュール(と viewer 配下全部)では `from __future__ import
annotations` を使わないこと。注釈が文字列化されると、create_app 内で
ローカル import した WebSocket 型を FastAPI が解決できず、ハンドラ起動前に
接続が閉じられる(実測)。
"""

import asyncio
import os
import time
import webbrowser
from pathlib import Path
from threading import Timer as ThreadTimer

from pydantic import BaseModel

from .appstate import (
    SessionHolder, append_recent, list_examples, load_recent,
    make_recent_entry, resolve_open_request)
from .persistence import ViewerConfig, save_config
from .probes import evaluate_probes
from .session import SimSession

STATIC_DIR = Path(__file__).resolve().parent / 'static'


class OpenRequest(BaseModel):
    """POST /api/project/open の要求(example 指定かパス指定のどちらか)。"""
    example: str | None = None
    mode: str | None = None            # example の 'core' | 'system'
    model: str | None = None           # IR パス直接指定
    system: str | None = None          # システム生成物ディレクトリ
    pipelined: bool = False


class BoardSelect(BaseModel):
    """PUT /api/board の要求(key=None で選択解除)。"""
    key: str | None = None


def frontend_dist() -> Path:
    """React フロントのビルド成果物の場所(env で上書き可能=テスト用)。"""
    env = os.environ.get('MDL_FPGA_FRONTEND_DIST')
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / 'frontend' / 'dist'


def create_app(session: SimSession):
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title='mdl-fpga viewer')

    # 可変状態は SessionHolder に集約(セッション切替 API が交換する。
    # 各エンドポイントは holder.* を毎回参照し、閉包に session を捕まえない)
    holder = SessionHolder(session)
    app.state.holder = holder

    @app.get('/api/meta')
    async def meta():
        s = holder.session
        return {**s.meta, 'cycle': s.cycle,
                'workdir': str(s.workdir), 'epoch': holder.epoch}

    @app.get('/api/node/{node_id}')
    async def node_detail(node_id: str):
        try:
            return holder.session.node_detail(node_id)
        except KeyError:
            raise HTTPException(404, f'ノードが存在しない: {node_id}') from None

    @app.get('/api/renderers')
    async def renderers():
        return holder.hub.listing()

    @app.get('/api/layout')
    async def get_layout():
        return holder.cfg.model_dump(by_alias=True)

    @app.put('/api/layout')
    async def put_layout(cfg: ViewerConfig, epoch: int | None = None):
        # 切替直後に旧画面の遅延デバウンス保存が届いても新サイドカーを汚さない
        if epoch is not None and epoch != holder.epoch:
            raise HTTPException(
                409, f'epoch 不一致(要求 {epoch} / 現在 {holder.epoch})。'
                     'プロジェクト切替後の古い保存要求のため破棄')
        if cfg.board is None:
            cfg.board = holder.cfg.board   # board 未知の旧クライアント互換
        holder.cfg = cfg
        holder.registry.replace_all(cfg.probes)
        await asyncio.to_thread(save_config, holder.cfg_path, cfg)
        return {'ok': True}

    async def _with_probes(resp: dict) -> dict:
        s = holder.session
        resp['probes'] = await asyncio.to_thread(
            evaluate_probes, holder.registry, resp.get('values', {}),
            s.meta, s.cycle, holder.hub)
        return resp

    # --- プロジェクト(現在/イグザンプル/切替) ---

    @app.get('/api/project')
    async def project():
        s = holder.session
        return {'model': str(s.model), 'name': s.meta.get('name'),
                'mode': holder.mode,
                'system_dir': str(s.system_dir) if s.system_dir else None,
                'pipelined': bool(s.pipelined), 'epoch': holder.epoch,
                'recent': load_recent()}

    @app.get('/api/examples')
    async def examples():
        return {'examples': list_examples()}

    async def _notify_and_close_sockets():
        for sock in list(holder.sockets):
            try:
                await sock.send_json({'type': 'session_changed',
                                      'epoch': holder.epoch})
                await sock.close()
            except Exception:  # noqa: BLE001 - 既に切れたソケットは無視
                pass

    # --- ボード(DB 一覧/選択+advise/検出) ---

    @app.get('/api/boards')
    async def boards():
        from ..board_db import list_boards, load_board
        return {'boards': [load_board(k) for k in list_boards()]}

    async def _board_state() -> dict:
        key = holder.cfg.board
        if not key:
            return {'key': None, 'advice': None}
        out = {'key': key}
        cache_key = (str(holder.session.model), key)
        if cache_key in holder.advice_cache:
            out['advice'] = holder.advice_cache[cache_key]
            return out

        def _compute():
            from ..advisor import run_advise
            return run_advise(holder.session.model, board=key).to_dict()

        try:
            adv = await asyncio.to_thread(_compute)
            holder.advice_cache[cache_key] = adv
            out['advice'] = adv
        except Exception as e:  # noqa: BLE001 - 搭載可能性の判定失敗は表示情報に変換
            out['advice'] = None
            out['advice_error'] = str(e)
        return out

    @app.get('/api/board')
    async def get_board():
        return await _board_state()

    @app.put('/api/board')
    async def put_board(req: BoardSelect):
        if req.key is not None:
            from ..board_db import load_board
            try:
                load_board(req.key)
            except (KeyError, ValueError) as e:
                raise HTTPException(400, str(e)) from None
        holder.cfg.board = req.key
        await asyncio.to_thread(save_config, holder.cfg_path, holder.cfg)
        return {'ok': True, **await _board_state()}

    @app.post('/api/board/detect')
    async def board_detect():
        # JTAG は排他資源: ユーザー操作起点でのみ実行・直列化。
        # 候補が一意でもサーバは自動選択しない(detect は副作用フリー。
        # 自動選択はフロントが PUT /api/board で行う)
        if holder.detect_lock.locked():
            raise HTTPException(409, '検出実行中(完了を待って再実行)')
        async with holder.detect_lock:
            impl = holder.detect_impl
            if impl is None:
                try:
                    from ..detect import run_detect as impl
                except ImportError:
                    raise HTTPException(
                        501, 'detect 未対応(mdl_fpga.detect が読み込めない。'
                             'ツールの更新を確認)') from None
            res = await asyncio.to_thread(impl)
            return res.to_dict() if hasattr(res, 'to_dict') else res

    @app.post('/api/project/open')
    async def open_project(req: OpenRequest):
        if holder.switch_lock.locked():
            raise HTTPException(409, '切替処理中(完了を待って再実行)')
        async with holder.switch_lock:
            try:
                spec = resolve_open_request(req.model_dump())
            except ValueError as e:
                raise HTTPException(400, str(e)) from None
            # 準備 → 交換 → 旧後始末。起動失敗時は旧セッション継続=ロールバック
            new = SimSession(model=spec['model'], workdir=None,
                             pipelined=spec['pipelined'],
                             system_dir=spec['system_dir'])
            try:
                await asyncio.to_thread(new.start)
                await asyncio.to_thread(new.reset)
            except Exception as e:  # noqa: BLE001
                await asyncio.to_thread(new.close)
                raise HTTPException(
                    400, f'起動失敗: {e}(現在のプロジェクトは継続)') from None
            old = holder.session
            holder.swap(new)               # 原子的交換(await を挟まない)
            append_recent(make_recent_entry(spec, new.meta.get('name')))
            await _notify_and_close_sockets()
            # 旧 close(quit + join 15s)は応答を待たせず裏で
            asyncio.get_running_loop().create_task(
                asyncio.to_thread(old.close))
            return {'ok': True, 'name': new.meta.get('name'),
                    'mode': holder.mode, 'epoch': holder.epoch}

    @app.websocket('/ws/sim')
    async def ws_sim(ws: WebSocket):
        try:
            await _ws_sim_inner(ws)
        except WebSocketDisconnect:
            return
        except BaseException:
            # サーバ側の失敗を無言で握りつぶさない(R2 デバッグの教訓)
            import traceback
            traceback.print_exc()
            raise

    async def _ws_sim_inner(ws):
        await ws.accept()
        holder.sockets.add(ws)
        joined_epoch = holder.epoch
        try:
            # 接続直後に現在値を届ける(read は時間を進めない)
            first = await asyncio.to_thread(holder.session.read)
            await ws.send_json({'type': 'values', **await _with_probes(first)})
            while True:
                msg = await ws.receive_json()
                # 切替を跨いだ接続は自分から閉じる(フロントは再接続して
                # 全再取得する。session_changed はそのトリガ)
                if joined_epoch != holder.epoch:
                    await ws.send_json({'type': 'session_changed',
                                        'epoch': holder.epoch})
                    await ws.close()
                    return
                s = holder.session
                cmd = msg.get('cmd')
                t0 = time.perf_counter()
                if cmd == 'set_step':
                    # クリック→入力反映→1clk を 1 往復で(クリック体感の最短化)
                    await asyncio.to_thread(s.set_x, msg['hex'])
                    resp = await asyncio.to_thread(
                        s.step, int(msg.get('n', 1)))
                elif cmd == 'set_x':
                    resp = await asyncio.to_thread(s.set_x, msg['hex'])
                elif cmd == 'step':
                    resp = await asyncio.to_thread(
                        s.step, int(msg.get('n', 1)))
                elif cmd == 'step_until':
                    resp = await asyncio.to_thread(
                        s.step_until, msg['target'],
                        msg.get('slice'), int(msg.get('max', 1000)))
                elif cmd == 'reset':
                    resp = await asyncio.to_thread(s.reset)
                elif cmd == 'read':
                    resp = await asyncio.to_thread(s.read)
                else:
                    resp = {'ok': False, 'error': f'unknown cmd: {cmd}'}
                resp = await _with_probes(resp)
                resp['latency_ms'] = round((time.perf_counter() - t0) * 1e3, 2)
                await ws.send_json({'type': 'values', **resp})
        finally:
            holder.sockets.discard(ws)

    # --- フロント配信(API/WS ルート登録の後に。mount は最後) ---
    dist = frontend_dist()
    if (dist / 'index.html').exists():
        app.mount('/', StaticFiles(directory=str(dist), html=True),
                  name='frontend')
    else:
        @app.get('/')
        async def index():
            return FileResponse(STATIC_DIR / 'index.html')

    return app


def serve(model: str, port: int = 8788, workdir: str | None = None,
          open_browser: bool = True, pipelined: bool = False,
          system_dir: str | None = None) -> None:
    import uvicorn

    session = SimSession(model=model, workdir=workdir, pipelined=pipelined,
                         system_dir=system_dir)
    print(f'[mdl-fpga view] シミュレーション起動中... ({model})')
    session.start()
    session.reset()
    ui = 'React UI(frontend/dist)' if (frontend_dist() / 'index.html').exists() \
        else '簡易 UI(React UI を使うには tools/mdl-fpga/frontend で npm run build)'
    print(f'[mdl-fpga view] http://127.0.0.1:{port}/ '
          f'(モデル: {session.meta["name"]}, '
          f'入力 {session.meta["in_width"]}bit, {ui})')

    app = create_app(session)
    if open_browser:
        ThreadTimer(1.0, lambda: webbrowser.open(
            f'http://127.0.0.1:{port}/')).start()
    try:
        uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
    finally:
        # 切替後は holder が「現在の」セッションを持つ(引数の session ではなく)
        app.state.holder.session.close()
