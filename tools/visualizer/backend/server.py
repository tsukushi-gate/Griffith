"""DDLGNビジュアライザーのWebSocketサーバー。

チェックポイントファイルを読み込み、ネットワーク構造をバイナリで
ブラウザに配信する。ファイル変更を監視してリアルタイム更新も対応。

起動方法:
    uv run python tools/visualizer/backend/server.py --checkpoint outputs/.../latest.pt

    # またはモデル定義を指定してダミーモデルで起動
    uv run python tools/visualizer/backend/server.py --demo mnist

セキュリティ前提:
    **信頼された単一ユーザのローカル環境専用**ツールで、認証を持たない。
    - 既定バインドは `127.0.0.1`(監査B5)。`--host 0.0.0.0` は明示時のみで警告を出す。
    - ハンドシェイクで `Host`/`Origin` を検証し、ループバック以外を拒否する。
    - チェックポイントは既定 `weights_only=True` で読む(監査B4)。
    詳細は `tools/visualizer/README.md` と `SECURITY.md`。
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# プロジェクトルートとbackendディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

try:
    import websockets
    import websockets.http11
    # ブラウザのHTTPヘッダーがデフォルト上限(8192)を超えることがあるため拡張
    websockets.http11.MAX_LINE_LENGTH = 65536
except ImportError:
    print('websocketsが必要です: pip install websockets')
    sys.exit(1)

import torch

from extractor import extract_network, encode_binary

# セキュリティ機構とチェックポイントの安全ローダはエディタ backend と共有する
# (security.py はエディタ/ビジュアライザ共通機構として作られている)。
from tools.editor.backend.demo import load_model_from_checkpoint as _load_safe_checkpoint
from tools.editor.backend.security import (
    allowed_hosts,
    is_host_allowed,
    is_origin_allowed,
)

# 既定バインドはループバックのみ(監査B5)。外部公開は明示的な --host 指定を要求する。
DEFAULT_HOST = '127.0.0.1'
# 「全インタフェースにバインド」を意味する指定。警告対象。
_WILDCARD_HOSTS = frozenset({'0.0.0.0', '::', '[::]'})

# ヘッダが複数値だったことを表す番兵(Noneや文字列と区別するため)。
_AMBIGUOUS = object()


def _single_header(headers, name: str):
    """ヘッダを1個だけ取り出す。無ければ None、複数あれば `_AMBIGUOUS`。

    `websockets` の Headers は重複キーに対し `get` が MultipleValuesError を投げる。
    Host/Origin の重複は正当なブラウザでは起こらず、リクエストスマグリングの
    兆候なので「値なし」ではなく「曖昧」として呼び出し側で拒否させる。
    """
    values = headers.get_all(name)
    if not values:
        return None
    if len(values) > 1:
        return _AMBIGUOUS
    return values[0]


# --- モデルロード ---

def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """チェックポイントからモデルをロードする(安全既定、監査B4)。

    実装はエディタ backend の `demo.load_model_from_checkpoint` に委譲する。
    state_dict形式(`{'state_dict': ..., 'config': ...}`)を優先し、モデル全体を
    pickleした旧形式は `GRIFFITH_ALLOW_UNSAFE_CKPT=1` のオプトイン時のみ読む。
    """
    return _load_safe_checkpoint(checkpoint_path, device)


def create_demo_model(model_type: str = 'mnist'):
    """デモ用の小規模モデルを生成する。"""
    from griffith_logic import LutLayer, GroupSum

    if model_type == 'mnist':
        model = torch.nn.Sequential(
            LutLayer(784, 1000, n_inputs=2, connections='unique'),
            LutLayer(1000, 1000, n_inputs=2, connections='unique'),
            LutLayer(1000, 1000, n_inputs=2, connections='unique'),
            GroupSum(k=10, tau=1.0),
        )
    elif model_type == 'small':
        model = torch.nn.Sequential(
            LutLayer(64, 256, n_inputs=2, connections='unique'),
            LutLayer(256, 256, n_inputs=2, connections='unique'),
            GroupSum(k=4, tau=1.0),
        )
    elif model_type == 'large':
        model = torch.nn.Sequential(
            LutLayer(784, 4000, n_inputs=2, connections='unique'),
            LutLayer(4000, 4000, n_inputs=2, connections='unique'),
            LutLayer(4000, 4000, n_inputs=2, connections='unique'),
            LutLayer(4000, 4000, n_inputs=2, connections='unique'),
            GroupSum(k=10, tau=1.0),
        )
    else:
        raise ValueError(f'未知のデモモデル: {model_type}')

    model.eval()
    return model


# --- WebSocketサーバー ---

class VisualizerServer:
    """WebSocketサーバー。接続クライアントにネットワークデータを配信する。"""

    def __init__(self, model, host: str = DEFAULT_HOST, port: int = 8765):
        self.model = model
        self.host = host
        self.port = port
        self.clients: set = set()
        self.current_data: bytes | None = None
        # 許可ホスト集合: ループバック + バインド先 + GRIFFITH_EDITOR_ALLOWED_HOSTS。
        # `0.0.0.0` 等のワイルドカードは allowed_hosts 側で無視される。
        self.allowed_hosts = allowed_hosts((host,))
        if host in _WILDCARD_HOSTS:
            print(
                f'[警告] 全インタフェース({host})にバインドします。このツールは認証を持たない'
                'ローカル専用ツールです。公開ネットワークで使わないでください。'
                'アクセス元ホスト名は GRIFFITH_EDITOR_ALLOWED_HOSTS で明示的に許可が必要です。',
                flush=True,
            )
        self._build_data()

    def process_request(self, connection, request):
        """ハンドシェイク時に Host / Origin を検証する。

        - `Host` がループバック名義以外 → 421(DNSリバインディング対策)
        - `Origin` がループバック以外 → 403(CSRF/クロスオリジン接続の遮断)
          Origin 無し(非ブラウザクライアント)は許可する。
        """
        host = _single_header(request.headers, 'Host')
        if host is _AMBIGUOUS or not is_host_allowed(host, self.allowed_hosts):
            # ヘッダ重複自体がリクエストスマグリングの兆候なので同じく拒否する
            return connection.respond(
                421, 'このサーバはローカル(localhost/127.0.0.1)からのみ利用できます\n'
            )
        origin = _single_header(request.headers, 'Origin')
        if origin is _AMBIGUOUS or not is_origin_allowed(origin, self.allowed_hosts):
            return connection.respond(403, '許可されていないOriginからの接続です\n')
        return None

    def _build_data(self):
        """モデルから可視化データを構築してバイナリにエンコードする。"""
        print('ネットワークデータを抽出中...', flush=True)
        t0 = time.time()
        network = extract_network(self.model)
        self.current_data = encode_binary(network)
        elapsed = time.time() - t0
        print(
            f'抽出完了: {network["total_nodes"]:,}ノード, '
            f'{network["total_edges"]:,}エッジ, '
            f'{len(self.current_data) / 1024 / 1024:.1f}MB, '
            f'{elapsed:.2f}秒',
            flush=True,
        )

    async def handler(self, websocket):
        """クライアント接続ハンドラ。"""
        self.clients.add(websocket)
        addr = websocket.remote_address
        print(f'クライアント接続: {addr}', flush=True)

        try:
            # 接続時に現在のデータを送信
            if self.current_data:
                await websocket.send(self.current_data)
                print(f'データ送信完了: {len(self.current_data):,} bytes → {addr}', flush=True)

            # クライアントからのメッセージを待機（切断検知用）
            async for message in websocket:
                if isinstance(message, str):
                    cmd = json.loads(message)
                    if cmd.get('type') == 'reload':
                        self._build_data()
                        await self.broadcast()
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f'ハンドラエラー: {type(e).__name__}: {e}', flush=True)
        finally:
            self.clients.discard(websocket)
            print(f'クライアント切断: {addr}', flush=True)

    async def broadcast(self):
        """全クライアントにデータを配信する。"""
        if not self.current_data or not self.clients:
            return
        await asyncio.gather(
            *[client.send(self.current_data) for client in self.clients],
            return_exceptions=True,
        )

    def update_model(self, model):
        """モデルを更新してデータを再構築する。"""
        self.model = model
        self._build_data()

    async def run(self):
        """サーバーを起動する。"""
        # 静的ファイル配信用のHTTPサーバーも起動
        frontend_dir = Path(__file__).parent.parent / 'frontend'

        print(f'WebSocketサーバー起動: ws://{self.host}:{self.port}', flush=True)
        print(f'フロントエンド: http://localhost:5173 (Viteの開発サーバーを別途起動)', flush=True)
        print(flush=True)

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            max_size=None,         # メッセージサイズ制限を解除
            # Origin検証は process_request で行う(ポート任意のループバックを許可し、
            # それ以外を403で拒否する。websocketsのorigins=は固定値列挙しかできない)
            process_request=self.process_request,
            ping_interval=30,
            ping_timeout=10,
        ):
            await asyncio.Future()  # 永続実行


# --- ファイル監視 ---

async def watch_checkpoint(server: VisualizerServer, path: str, interval: float = 2.0):
    """チェックポイントファイルの変更を監視して自動更新する。"""
    last_mtime = 0.0
    while True:
        try:
            mtime = os.path.getmtime(path)
            if mtime > last_mtime:
                last_mtime = mtime
                print(f'チェックポイント更新検出: {path}')
                model = load_model_from_checkpoint(path)
                server.update_model(model)
                await server.broadcast()
        except (FileNotFoundError, RuntimeError) as e:
            pass  # ファイルが一時的に存在しない場合はスキップ
        await asyncio.sleep(interval)


# --- メイン ---

def main():
    parser = argparse.ArgumentParser(description='DDLGNビジュアライザーサーバー')
    parser.add_argument('--checkpoint', type=str, help='チェックポイントファイルパス')
    parser.add_argument('--demo', type=str, choices=['mnist', 'small', 'large'],
                        help='デモモデルで起動')
    parser.add_argument(
        '--host', type=str, default=DEFAULT_HOST,
        help=f'バインドするホスト(既定 {DEFAULT_HOST}。ローカル専用ツールのため'
             '0.0.0.0 等の外部公開は非推奨)',
    )
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--watch', action='store_true',
                        help='チェックポイントファイルを監視して自動更新')
    args = parser.parse_args()

    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint)
    elif args.demo:
        model = create_demo_model(args.demo)
    else:
        print('--checkpoint または --demo を指定してください')
        sys.exit(1)

    server = VisualizerServer(model, host=args.host, port=args.port)

    async def run():
        tasks = [server.run()]
        if args.watch and args.checkpoint:
            tasks.append(watch_checkpoint(server, args.checkpoint))
        await asyncio.gather(*tasks)

    asyncio.run(run())


if __name__ == '__main__':
    main()
