# ビジュアルエディタ (tools/editor)

微分可能論理ゲートNNのビジュアルエディタ。IR(`griffith_logic/ir/`)を共通契約として、
モデルのグラフ編集・ゲート分布の可視化・コード/HDL生成・学習のライブ監視までを1画面で行う。

- バックエンド: FastAPI + uvicorn(REST + WebSocket)
- フロントエンド: Vite + React + TypeScript + @xyflow/react (React Flow v12) + zustand + deck.gl + uPlot
- 参考UX: ComfyUI / Sony Neural Network Console / TensorFlow Playground

既存の `tools/visualizer/` (ゲート単位のWebGL描画)とは独立。こちらはレイヤ単位の
グラフ編集・学習監視が目的。

関連ドキュメント:

- アーキテクチャ概観: [../../docs/architecture/overview.md](../../docs/architecture/overview.md)
- エージェント向けREST/WS仕様(詳細): [../../docs/reference/agent-api.md](https://docs.griffith.magicdelta.jp/reference/agent-api/)
- ドキュメント索引: [../../docs/README.md](../../docs/README.md)

## セットアップ

```bash
# バックエンド依存(リポジトリルートで)
# torch は editor extra には含まれない。cpu/cu128 extra が必須
uv sync --extra editor --extra cpu --extra dev   # GPU機は --extra cu128

# フロントエンド依存
cd tools/editor/frontend
npm install
```

## 起動手順

1コマンドで両方を起動する場合(リポジトリルートで):

```bash
make editor    # backend(8765) + frontend(5173) を同時起動(個別は make backend / make front)
```

個別に起動する場合 — ターミナル1: バックエンド(リポジトリルートで)

```bash
uv run uvicorn tools.editor.backend.main:app --port 8765
```

ターミナル2: フロントエンド

```bash
cd tools/editor/frontend
npm run dev
```

ブラウザで http://localhost:5173 を開く。REST(`/api`)も学習WebSocket(`/ws`)も
vite dev serverが同一オリジンでバックエンド(既定 127.0.0.1:8765)へproxyする。
学習WSをポートへ直結すると、バックエンドを別ポートで起動した時や巨大Cookie環境で
退行する事故があったため、既定は必ず同一オリジン(`ws://localhost:5173/ws/train/{sid}`)
経由に統一している(vite proxyは学習WSからCookieを除去して400 line too longを免疫化する。B3回帰)。

別ホストのバックエンドへ直結したい場合のみ、フロント起動時に
`VITE_EDITOR_WS_BASE_URL=ws://host:port` を明示(opt-in)する。

## セキュリティ前提(ローカル専用ツール)

> **このツールは「信頼された単一ユーザのローカル環境専用」です。認証機構を持ちません。**
> **公開ネットワークにバインドしないでください。**

- **ワークスペース内の Python を実行する機能があります**。モデル定義のインポートは
  指定した Python モジュールを import して実行し、学習もプロセス内で走ります。
  つまり「APIを叩ける人＝任意コードを実行できる人」です。
- 既定バインドは **`127.0.0.1`(ループバックのみ)**。`--host 0.0.0.0` 等の外部公開は
  明示指定が必要で、起動時に警告を出します。リバースプロキシやトンネルでの公開、
  複数ユーザでの共有運用もしないでください。

「ローカルだから安全」を前提ではなく機構で成立させるため、以下を実装しています。

| 対策 | 挙動 |
|---|---|
| `Host` ヘッダ検証 | ループバック名義以外の `Host` を **421** で拒否(DNSリバインディング対策。外部サイトが自ドメインを `127.0.0.1` に解決させて手元のサーバを叩く攻撃を遮断) |
| `Origin` ヘッダ検証 | 状態変更メソッド(`POST`/`PUT`/`DELETE`/`PATCH`)と WebSocket ハンドシェイクで、ループバック以外の `Origin` を **403** で拒否(CSRF / CSWSH)。`Origin` 無し(`curl` 等の非ブラウザ)は許可 |
| パス内包チェック | API が受け取る全パスを解決してリポジトリ作業ディレクトリ配下に限定。`../` 脱出・外部絶対パス・外を指すシンボリックリンク・NUL 文字は **400** |
| チェックポイント | `torch.load` を `weights_only=True` で実行。モデル全体を pickle した ckpt は**既定で拒否**(下記) |
| アップロード上限 | データセット取込はチャンク読みで上限超過を **413**。zip は**展開後**の合計サイズも検査(zip bomb) |

### 環境変数

| 変数 | 既定 | 用途 |
|---|---|---|
| `GRIFFITH_EDITOR_ALLOWED_HOSTS` | (なし) | `Host`/`Origin` 検証で追加許可するホスト名(カンマ区切り)。**非ループバックへ意図的にバインドする場合のみ**指定する(コンテナ内・リモート開発機をホスト名で叩く等)。`0.0.0.0` 等のワイルドカードはホスト名ではないので無視される。許可を足すほど DNS リバインディング/CSRF の面が広がるため、最小のホスト名だけを書くこと |
| `GRIFFITH_ALLOW_UNSAFE_CKPT` | (なし) | `1` を設定すると `torch.load(weights_only=False)`、すなわち ckpt 内の pickle 実行を許可する。未設定ではモデル全体を pickle した ckpt は理由付きエラーで拒否される。**推奨は state_dict 形式** `{'state_dict': ..., 'config': ...}`(`config` の記述からコードで構造を組み立てるため pickle を一切通さない)。自分で作った ckpt にのみ使うこと |
| `GRIFFITH_EDITOR_MAX_UPLOAD_MB` | `512` | データセット取込の上限(MiB) |

実装は `tools/editor/backend/security.py`(エディタ/ビジュアライザ共通)。

## 画面構成(5トップタブ×EDIT内3レンズ)

NNCに倣い、ヘッダの**トップタブ帯**でワークフローを段階分けする(N系列)。並びは
キーボード `1`〜`5` と一致する(単一情報源: `frontend/src/store/uiStore.ts` の `TOP_TAB_ORDER`)。

| # | タブ | 役割 |
|---|---|---|
| 1 | **EDIT** | グラフ構築・編集。内部を3レンズ(Build/Probe/Ship)で切替(下記) |
| 2 | **TRAINING** | 学習の実行・ライブ監視・タイムマシン・run一覧(下部ドック: 学習/タイムライン/Run) |
| 3 | **EVALUATION** | 学習済み/現行モデルの一括評価・混同行列・誤答トレース・手描き推論デモ |
| 4 | **DATASET** | 組込カタログ+データ取込(CSV/画像zip)・split/augmentation・エンコードプレビュー |
| 5 | **CONFIG** | TrainConfigスキーマ駆動フォーム・Optimizer/Monitor/Executor定義・プロジェクト保存 |

EDITタブ内は**レンズ**(グラフは同一のまま着色・バッジ・下部ドックのタブを一括切替する
オーバーレイ文脈。キー `Q`/`W`/`E`)で切り替える:

- **Build**(`Q`): 構築・コード生成(下部ドック=コード)
- **Probe**(`W`): プレイグラウンド推論(下部ドック=実行)
- **Ship**(`E`): 出荷前チェック・HDL/コード出力(下部ドック=出荷)

> 旧「4レンズ(build/train/probe/ship)」は N1 で廃止。train はTRAININGタブへ吸収され、
> EDIT内は3レンズに縮約された(裁定1)。

## 使い方

### 1. モデル/プロジェクトを用意する(EDIT)

- **テンプレから作る**: 新規プロジェクトダイアログでテンプレ6種
  (ロジスティック回帰 / MLP / LeNet風CNN / AutoEncoder=重み転写 / LSTM系列 / Transformer=blackbox)
  を選ぶと、検証済みIR+TrainConfigが束ねられたプロジェクトが生成される。
- **デモモデル生成**: k(層幅)・レイヤ数を指定して生成
  (Flatten → LutLayer(784, k, N=2)×L → GroupSum(10))。kはクラス数10の倍数、
  unique接続の都合で十分大きな値(k×2 ≥ 784)が必要。
- **インポート**: checkpointパス(モデルオブジェクト全体を保存した .pt)、保存済みIRファイル、
  または Pythonファイル(`editor_model`/`build_model`。Attention/Transformer等の
  blackboxを含むモデルはこの経路)から読み込む。
- **サンプル**: 空状態のオンボーディングから、XOR・半加算器・全加算器・2:1 MUX・
  パリティ3/4・MNISTミニ学習デモの埋め込みIRをロードできる。

### 2. グラフを編集する(EDIT / Build)

- **部品パレット**: LutLayer/GroupSum/Binarize に加え、Conv2d・OrPool2d(indices自動生成+
  縮約ツリー、初期化プリセット)、RNN時間展開(LSTM/GRUセル→既存IRノード列へunroll)、
  PositionalEncoding・Embedding・DWN・UpSample2d、Attention/Transformer(blackbox)を配置できる。
- DnDでノードを追加し、エッジをつないで接続を編集する。形状(shape)不整合はキャンバス上で診断表示。
- **Unit(単位ネットワーク)**: 引数付きの再利用サブネットを定義し、インスタンス化して展開できる
  (保存時は常に展開済みIR。設計は [../../docs/architecture/editor-unit-design.md](../../docs/architecture/editor-unit-design.md))。
- **複数ネットワーク**: プロジェクト内に複数ネットワークをタブで持ち、同名レイヤ間で重み転写できる。
- **統計パネル**: LUT数・論理深さ・配線数・学習パラメータ数をライブ表示(`GET /api/stats`)。
- **NNCカード表示**: 小規模層は「1ゲート=1カード」(MIL記号SVG+真理値表+分布バー)へ自動展開。
  `A` キーでトポロジカル縦整列(NNC風)。
- **真理値表の直接編集**: LUTニューロンの真理値表(hex)をセル単位で書き換える。
- **手動配線(indices編集)**: SVG二部グラフ上で各ニューロンの入力スロットを付け替える。
- undo/redo、右クリックのQuickAdd、⌘Kコマンドパレット、キーボードショートカットに対応。
- **コード生成**: 現在のIRからPythonコードを生成し、コピー/ダウンロードできる。

### 3. データセットを扱う(DATASET)

- 組込カタログ(mnist / logic:<key> / truth_table 等)と取込データセットを一覧・プレビューする。
- **取込**: CSV(`POST /api/datasets/import/csv`)・画像zip(`.../images`。resize/grayscale指定)を
  アップロードして動的カタログへ登録する。
- **split/seed/augmentation/encoding** を設定し、実数値列のエンコード結果(0/1グリッド)を
  プレビューできる(`POST /api/datasets/encoding-preview`)。

### 4. 学習の設定と実行(CONFIG / TRAINING)

- **CONFIG**: `TrainConfig` のJSON Schema(`GET /api/config/schema`)からフォームを自動生成し、
  Optimizer/Monitor/Executor を定義してプロジェクトへ保存する。
- **TRAINING**: dataset・method(bp/ap/bb)・lr・batch・epochs・optimizer(adam/sgd)・loss を
  指定して開始する。各ランは編集中モデルの独立コピーを学習し(非破壊探索)、`apply` で明示採用する。
- pause / resume / stop に加え、**ステップ実行**(nステップ進めて自動pause)が可能。
- loss・acc(訓練バッチ)・レイヤ別‖ΔW‖チャートがWebSocket経由でライブ更新され、
  ゲート分布ミニバー/ヒートマップも最大5Hzで追従する。
- classificationタスクでは hard eval 後に混同行列を表示する。

### 5. 学習を遡る・観察する(TRAINING / タイムマシン)

- **タイムマシン**: タイムラインをスクラブして任意ステップのスナップショットを
  read-onlyで閲覧する(メトリクス・ゲート分布・要約をオンデマンド導出)。
- **巻き戻し(branch restore)**: pause中に任意ステップへ巻き戻して再学習の起点にする
  (復元先より後の履歴はtruncate)。
- **誤差逆伝播リプレイ**: 勾配フローを再生し、層の発光・勾配パーティクルで可視化する。
- ゲート分布(dense/aggregate)・活性のストリーミング購読に対応。
- **runs**: 学習後のrunサマリを一覧・比較する(プロセスをまたいで永続)。

### 6. 評価する(EVALUATION)

- **一括評価**(`POST /api/evaluate`): run または現行モデルを指定データセットで評価し、
  正解率・混同行列(P/R/F1)・**離散化ギャップ**(hard離散化結果 ≠ soft argmax/しきい値)列を出す。
- **反例トレース**: 誤答サンプルの入力ビット列をプレイグラウンドへ注入して回路を辿れる。
- **手描き推論デモ**: 28×28キャンバスへ手描き→2値化→probe/クラス確率バー。学習中のライブ推論にも対応。
- 結果はCSVエクスポート可能(`POST /api/evaluate/export.csv`)。

### 7. 構造探索(NAS)

- **NAS**(`POST /api/nas/start`): genome進化で構造を自動探索し、精度×回路面積(LUT数)の
  Trade-offグラフ+パレート前線を表示する(中断再開・JSONL台帳・決定性ゴールデン)。

### 大規模モデルビュー

- **deck.gl大規模ビュー**: LODドリルダウン(パンくずナビ)と、WSの layer_summary による
  ライブ再着色で大きなグラフを俯瞰する。

## キーボードショートカット

単一情報源は `frontend/src/hooks/useShortcuts.ts` 冒頭のキー割当表。
input/textarea/select/contentEditable にフォーカス中は無効。

| キー | 動作 | 有効範囲 |
|---|---|---|
| `1`〜`5` | トップタブ切替(EDIT/TRAINING/EVALUATION/DATASET/CONFIG) | 全タブ常時 |
| `Q`/`W`/`E` | EDIT内レンズ切替(Build/Probe/Ship) | EDITタブ内 |
| `F` | 全体表示(fitView) | キャンバス |
| `L` | 自動整列(autoLayout) | キャンバス |
| `A` | NNC縦整列(トポロジカル順に座標更新) | EDITタブ内 |
| `⌘K` / `Ctrl+K` | コマンドパレット開閉 | 全タブ |
| `⌘Z` / `⇧⌘Z` | undo / redo | 全タブ |
| `⌘S` | 即時同期(syncNow) | 全タブ |

URLハッシュ(`#edit`/`#training` 等)とタブは双方向同期し、ブラウザの戻る/進むに対応する。

## API

AIエージェントが REST/WS だけでモデル構築〜学習〜改変を完結するための詳細仕様
(全エンドポイント表・リクエスト/レスポンス形・curl/Python完全例・WSフレーム型)は
[../../docs/reference/agent-api.md](https://docs.griffith.magicdelta.jp/reference/agent-api/) を参照。
以下はカテゴリ別サマリ(正はコード `backend/main.py` および各Routerファイル)。

| カテゴリ | 主なエンドポイント |
|---|---|
| ヘルス | `GET /api/health`(server_epoch=再起動検出) |
| データセット | `GET /api/datasets`、`GET /api/datasets/{id}/samples` |
| データ取込 | `POST /api/datasets/import/csv` `.../import/images`、`PUT /api/datasets/imported/{slug}/config`、`GET .../imported/{slug}/samples`、`POST /api/datasets/encoding-preview` |
| モデル | `POST /api/models/demo` `import` `import/python`、`GET /api/models/current/ir`、`POST /api/models/ir/load` `.../save` |
| 編集 | `PUT /api/models/current/ir`、`POST .../nodes/{id}/truth_table` `.../indices`、`.../build`、`.../codegen` |
| 統計 | `GET /api/stats`(LUT数・論理深さ・配線数・学習パラメータ数) |
| プロジェクト | `POST/GET /api/projects`、`GET/PUT/DELETE /api/projects/{id}` |
| ネットワーク | `POST /api/networks/transfer`(同名レイヤ重み転写) |
| テンプレ | `GET /api/templates`、`POST /api/templates/{id}/instantiate` |
| 設定 | `GET /api/config/schema`(TrainConfigのJSON Schema) |
| 学習 | `POST /api/train/start` `.../pause` `.../resume` `.../stop` `.../step` `.../apply`、`GET /api/train/sessions` `.../status` `.../confusion` `.../uncertain_gates` |
| 履歴(タイムマシン) | `GET /api/train/{sid}/history` `.../history/{step}`、`POST .../restore` |
| 評価 | `POST /api/evaluate`、`POST /api/evaluate/export.csv` |
| NAS | `POST /api/nas/start` `.../stop`、`GET /api/nas/status` `.../trials` `.../trials/{id}/ir` |
| runs | `GET /api/runs` `.../{run_id}` |
| エクスポート・推論 | `POST /api/export/hdl`(Verilog/VHDL+リソース見積り)、`POST /api/probe`(全ノード中間値) |
| WS | `/ws/train/{sid}`(metrics / layer_summary / history_index / grad_flow など。subscribeで購読切替) |

レイヤ名はIRノードid(`n{i}_{クラス名}`、`export_ir` と同一規約)。

## テスト・型生成

```bash
# E2E(Playwright)。uvicorn(9765)+vite dev(6173)を自動起動して3スペック実行
# 前提: uv sync --extra dev --extra cpu --extra editor と npx playwright install chromium
make e2e                                  # = cd tools/editor/frontend && npm run e2e

# API契約のTS型自動生成。tools/editor/backend/schemas.py が真実源
make gen-types                            # → frontend/src/api/generated/{schema.json,types.ts}

# スキーマと実装のドリフト検出(実レスポンスをpydanticでvalidate)
uv run pytest tests/test_editor_schemas.py -q
```

- E2Eは開発サーバ(8765/5173)と衝突しない代替ポート(9765/6173)を使う
  (`E2E_BACKEND_PORT` / `E2E_FRONTEND_PORT` で変更可)
- CI(frontend.yml)は `gen-types → git diff --exit-code` で生成物のドリフトを検出する
- フロントの手書き型(`src/api/rest.ts` 等)の生成型への置換は段階2(未着手)

## トラブルシューティング

### `ModuleNotFoundError: No module named 'torch'`

`editor` extra には FastAPI/uvicorn のみ含まれ、PyTorch は `cpu` / `cu128` extra 経由で
別途インストールする(ルート [README.md](../../README.md) のクイックスタートと同じ方式)。

```bash
# Mac / CPU Linux
uv sync --extra editor --extra cpu --extra dev

# CUDA 12.8 環境(cpu と cu128 は同時指定不可)
uv sync --extra editor --extra cu128 --extra dev
```

### WebSocket が `400 Bad Request` / UI で「WS 未接続」

REST(`/api/train/.../status`)やWS接続が失敗する場合:

1. バックエンドが起動しているか確認: `curl http://127.0.0.1:8765/api/health`
2. フロントは **http://localhost:5173** 経由で開く
3. DevTools → Network → WS で接続先が **同一オリジン**(`ws://localhost:5173/ws/train/{sid}`)で
   あること。`ws://127.0.0.1:8765/...` へ直結している場合は古いフロントを再読み込み/再起動する
   (現行は vite の `/ws` proxy 経由でバックエンドへ張るのが既定。B3回帰で直結を廃止)
4. `400 line too long` は巨大Cookie(他プロジェクトのSupabase auth等)が原因。vite proxyが
   学習WSからCookieを除去して免疫化しているため、直結(旧実装)へ逸れていないか3を確認する
5. 別ホストのバックエンドへ直結したい場合のみ
   `VITE_EDITOR_WS_BASE_URL=ws://host:<port> npm run dev` で明示(opt-in)する

## 既知の制限

- 学習はCPU固定(スレッド実行の安定性優先)。GPU学習はエディタ外(Modal)で行う。
- **Attention/Transformer は blackbox** としてのみ扱う。エディタ内では学習不可(`trainable:false`)・
  HDL生成不可・IR単体からのbuild不可で、実行の正はPythonモデル。学習は Pythonファイルを
  `POST /api/models/import/python` で読み込むか、`experiments/train_llvit.py` 等のModalジョブで行う。
- 複数の訓練セッションを並列に実行できる(各ランは編集中モデルの独立コピーを学習する非破壊探索)。
  学習中でも編集/build/IR差し替えは可能で、実行中のランには影響しない。ランの重みを編集中モデルへ
  取り込むには `POST /api/train/{sid}/apply` で明示採用する。
- 訓練セッションはプロセス内メモリにのみ存在し、バックエンド再起動で消える
  (永続記録は `GET /api/runs` から取得。`/api/health` の server_epoch で再起動を検出)。
- ステップ実行・巻き戻し(restore)は paused 状態でのみ許可。
- checkpointインポートは `nn.Sequential` 直列 + モデルオブジェクト全体保存の .pt のみ対応
  (state_dictのみのチェックポイントは未対応)。
- warp/raw パラメータ化の確信度は定義が自明でないため中立値(0.5)で表示する。
- NASの進化変異は genome記述子レベル(IR変異は将来経路に温存)。
