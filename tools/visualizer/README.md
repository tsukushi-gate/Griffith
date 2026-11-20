# DDLGN Visualizer

DDLGN/CDDLGN/RDDLGNの論理ゲートネットワークをブラウザ上でリアルタイムに可視化するツール。

## 起動方法

### 1. デモモデルで起動（動作確認用）

```bash
# プロジェクトルートから実行
cd griffith

# 依存インストール（初回のみ）
uv sync --extra visualizer
cd tools/visualizer/frontend && npm install && cd ../../..

# WebSocketサーバー起動（ターミナル1）
uv run python tools/visualizer/backend/server.py --demo mnist

# フロントエンド開発サーバー起動（ターミナル2）
cd tools/visualizer/frontend && npx vite

# ブラウザで http://localhost:5173 を開く
```

### 2. チェックポイントから起動

```bash
uv run python tools/visualizer/backend/server.py --checkpoint outputs/YYYYMMDD_HHMMSS_run/checkpoints/best.pt
```

チェックポイントは **state_dict 形式**(`{'state_dict': ..., 'config': {...}}`)を使うこと。
`config` から構造をコードで組み立てるため pickle を実行しない。モデルオブジェクト全体を
`torch.save` した旧形式は既定で拒否され、`GRIFFITH_ALLOW_UNSAFE_CKPT=1` を明示した場合だけ
読める(詳細は下の「セキュリティ前提」)。サンプルは
`tools/editor/samples/demo_mnist_tiny.pt`。

### 3. チェックポイント監視モード（訓練中のリアルタイム可視化）

```bash
uv run python tools/visualizer/backend/server.py --checkpoint outputs/.../checkpoints/latest.pt --watch
```

## デモモデル

| モデル | ノード数 | エッジ数 | 構成 |
|--------|---------|---------|------|
| `small` | ~580 | ~1,024 | 64→256→256→4 |
| `mnist` | ~3,800 | ~6,000 | 784→1000→1000→1000→10 |
| `large` | ~16,800 | ~32,000 | 784→4000→4000→4000→4000→10 |

## セキュリティ前提(ローカル専用ツール)

> **このツールは「信頼された単一ユーザのローカル環境専用」です。認証機構を持ちません。**
> **公開ネットワークにバインドしないでください。**

- チェックポイントを読み込むため、**信頼できないファイルを指定させないこと**。
  同じリポジトリのビジュアルエディタ(`tools/editor/`)はワークスペース内の Python を
  実行する機能も持つため、より強い前提が必要です。
- 既定バインドは **`127.0.0.1`(ループバックのみ)**。`--host 0.0.0.0` 等の外部公開は
  明示指定が必要で、起動時に警告を出します。リバースプロキシやトンネルでの公開、
  複数ユーザでの共有運用もしないでください。

「ローカルだから安全」を前提ではなく機構で成立させるため、以下を実装しています。

| 対策 | 挙動 |
|---|---|
| `Host` ヘッダ検証 | WebSocket ハンドシェイクでループバック名義以外の `Host` を **421** で拒否(DNSリバインディング対策。外部サイトが自ドメインを `127.0.0.1` に解決させて手元のサーバを叩く攻撃を遮断) |
| `Origin` ヘッダ検証 | ループバック以外の `Origin` からの WebSocket 接続を **403** で拒否(CSWSH)。`Origin` 無し(非ブラウザ)は許可 |
| チェックポイント | `torch.load` を `weights_only=True` で実行。モデル全体を pickle した ckpt は**既定で拒否**(下記) |

### 環境変数

| 変数 | 既定 | 用途 |
|---|---|---|
| `GRIFFITH_EDITOR_ALLOWED_HOSTS` | (なし) | `Host`/`Origin` 検証で追加許可するホスト名(カンマ区切り)。**非ループバックへ意図的にバインドする場合のみ**指定する(コンテナ内・リモート開発機をホスト名で叩く等)。`0.0.0.0` 等のワイルドカードはホスト名ではないので無視される。許可を足すほど DNS リバインディング/CSRF の面が広がるため、最小のホスト名だけを書くこと |
| `GRIFFITH_ALLOW_UNSAFE_CKPT` | (なし) | `1` を設定すると `torch.load(weights_only=False)`、すなわち ckpt 内の pickle 実行を許可する。未設定ではモデル全体を pickle した ckpt は理由付きエラーで拒否される。**推奨は state_dict 形式** `{'state_dict': ..., 'config': ...}`(`config` の記述からコードで構造を組み立てるため pickle を一切通さない)。自分で作った ckpt にのみ使うこと |

実装は `tools/editor/backend/security.py`(エディタ/ビジュアライザ共通)。

## 操作

- **パン**: ドラッグ
- **ズーム**: スクロール
- **ホバー**: ノードにカーソルを合わせるとゲート情報表示
- **LOD**: ズームインするとエッジ→テキストの順に表示

## アーキテクチャ

```
PyTorch訓練 → server.py (WebSocket) → ブラウザ (deck.gl WebGL)
```
