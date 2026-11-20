---
title: 'griffith CLI'
description: '統一CLI `griffith` のサブコマンド(train / eval / export-hdl / deploy / editor / sweep / launch / collect / runs / report)。'
---

`griffith` は griffith-logic の統一CLI。学習・評価・HDLエクスポート・デプロイ(C/CUDA/Metal/Web)・エディタ起動・
実験オーケストレーション(sweep / launch / collect / runs / report)を、ひとつの入口から実行する。

実体は `pyproject.toml` の `[project.scripts]`(`md = "griffith_logic.cli:main"`)。
`uv sync` 後は `uv run md ...` で呼び出す。実装(`griffith_logic/cli.py`)は argparse の
サブコマンドディスパッチのみで、各サブコマンドのロジックは `griffith_logic/training/`
配下のモジュールに置かれている。

```bash
uv run griffith --help
```

## サブコマンド一覧

| サブコマンド | 説明 | 実装 |
|---|---|---|
| `train` | 学習を実行する(設定YAML+ドット記法オーバーライド) | `training/config.py`, `training/runner.py` |
| `eval` | チェックポイントを hard eval(離散化評価)する | `training/runner.py` |
| `export-hdl` | IR(JSON)から Verilog/VHDL を生成する | `export/verilog.py`, `export/vhdl.py` |
| `deploy` | 学習済みモデルを配布バンドルにし、C/CUDA/Metal/Web のコードを生成・検証する(サブコマンド9個) | `deploy/cli.py` |
| `editor` | ビジュアルエディタのバックエンドを起動する(clone 環境が必要) | uvicorn |
| `sweep` | wandb sweep を作成して agent を実行する | `training/sweep.py` |
| `launch` | グリッド実験を一括投入する(デカルト積展開) | `training/launch_cli.py` |
| `collect` | Modal Volume から `outputs/` の run ディレクトリを回収する | `training/launch_cli.py` |
| `runs list` / `runs compare` | wandb run の一覧・比較 | `training/runs_cli.py` |
| `report` | 実験レポートを生成する(wandb / markdown フォールバック) | `training/report_cli.py` |
| `deploy` | 配布バンドルの生成・C/CUDA/Metal/Web への出荷・等価性検証 | `deploy/cli.py` |

各サブコマンドの全フラグは `uv run md <サブコマンド> --help` で確認する
(本ページは主要フラグのみ抜粋)。

## griffith train

共通ランナーで学習を実行する。設定は YAML ファイルと `key=value` 形式のドット記法
オーバーライドで指定する。

| フラグ | 説明 |
|---|---|
| `config`(位置引数、省略可) | 設定YAMLのパス。省略時は既定値、`key=value` 形式ならオーバーライド扱い |
| `overrides`(位置引数、可変長) | ドット記法の上書き(例: `model.k=8000 run.epochs=5`) |
| `--from-run DIR` | 既存 run のディレクトリ内の `config.yaml` を読んで再実行する |
| `--out-dir DIR` | 出力先ディレクトリ(既定 `outputs/<timestamp>_<run名>`) |
| `--strict` | `expected_metrics` 未達のとき非0で終了する |

```bash
uv run griffith train configs/recipes/dlgn_mnist_quick.yaml
uv run griffith train configs/recipes/dlgn_mnist_quick.yaml optim.lr=0.02 model.k=4000
uv run griffith train --strict configs/recipes/dlgn_mnist_quick.yaml
```

`expected_metrics` はレシピYAMLの `<metric>_min` / `<metric>_max` キーで成功基準を
数値化したもの([`TrainConfig`](/reference/api/training-config/#trainconfig))。
runner 終了時に自動照合され結果に格納される(未達でも例外にはならない)。
`--strict` を付けたときだけ未達で非0 exit になる(CI・自動化フロー向け)。

## griffith eval

`griffith train` が保存したチェックポイントを読み込み、離散化した回路の精度(hard acc)を
出力する。モデル構成はチェックポイント内の設定から復元する。

| フラグ | 説明 |
|---|---|
| `checkpoint`(位置引数) | チェックポイント(`.pt`)のパス |
| `--dataset` | 評価データセット名(省略時はチェックポイントの学習時設定) |
| `--batch-size` | 評価バッチサイズ |
| `--device` | デバイス(省略時は自動検出) |

```bash
uv run griffith eval outputs/20260702_dlgn_mnist_quick/best.pt
```

## griffith export-hdl

IR(中間表現)JSON を検証し、Verilog/VHDL へトランスパイルする。

| フラグ | 説明 |
|---|---|
| `source`(位置引数) | IR JSON ファイルのパス |
| `--lang {verilog,vhdl}` | 生成する HDL 言語(既定: `verilog`) |
| `-o, --output` | 出力パス(省略時は入力と同じ場所に `.v` / `.vhdl`) |

```bash
uv run griffith export-hdl outputs/xor/ir.json --lang verilog
uv run griffith export-hdl outputs/xor/ir.json --lang vhdl -o build/xor.vhdl
```

シミュレータ照合(iverilog / ghdl)までは行わない。照合込みの手順は
[HDL等価性検証ガイド](/guides/hdl/)を参照。

## griffith deploy

学習済みモデル(IR)を **全ターゲット共通の配布バンドル** にまとめ、C / CUDA / Metal / Web のコードを生成して
`ir/simulate.py` と整数一致で検証する。最適化はバンドル化で済ませ、ターゲット側は実行するだけにする設計である。
通しの手順は [デプロイキット クイックスタート](/guides/deploy-quickstart/)、バンドルの形式は
[配布バンドル仕様](/reference/deploy-bundle/) を参照。

```bash
uv run griffith deploy --help
```

| サブコマンド | 説明 | 出力 |
|---|---|---|
| `bundle` | モデル JSON(webbench 形式)→ 配布バンドル v1(conv2d の命令列を焼き込む) | `.bundle.json` / `.bundle.mdlb` |
| `codegen-c` | バンドル → 自己完結した C ソース(組み込み・モバイル向け) | `model.c` / `model.h` + ランタイム |
| `codegen-cuda` | バンドル → CUDA ソース(NVIDIA GPU 向け) | `model_cuda.cu` + ランタイム |
| `codegen-metal` | バンドル → Metal ソース(Apple GPU 向け) | `model_metal.mm` + ランタイム |
| `codegen-web` | バンドル → Web(ブラウザ)配布物(ランタイムのコピーつき) | バンドル + `runtime/web/dist` |
| `verify-c` | 生成 C をビルドし、IR simulate との整数一致を検証する | 一致 / 不一致(終了コード) |
| `verify-metal` | 生成 Metal をビルドし、IR simulate との整数一致を検証する(macOS + Apple GPU) | 同上 |
| `verify-web` | Web ランタイム(Node 上の `verify.mjs`)を IR simulate と整数一致で検証する | 同上 |
| `info` | バンドルを検証して内容を要約表示する | 標準出力 |

CUDA には対応する `verify-cuda` が無い(コード生成のみ)。検証状況は [対応プラットフォーム](/reference/platform-support/) を参照。

### griffith deploy bundle

| フラグ | 説明 |
|---|---|
| `source`(位置引数) | モデル JSON(webbench 形式)のパス。最小構成は `{"ir": <IR>}` |
| `-o, --output` | 出力パス(省略時は入力と同じ場所に `.bundle.json`) |
| `--format {json,mdlb}` | 出力形式。既定は出力パスの拡張子から自動判定(`.mdlb` ならバイナリ形式) |
| `--gzip` | gzip 版も書き出して配信サイズを測る |
| `--compress {none,fast,max}` | lossless な回路圧縮の自動適用レベル(既定 `fast`)。モデルの種類によらず同じ意味。`none` = IR を書き換える最適化なし、`fast` = 秒で終わる lossless な削減をすべて(dense `lut_layer` 列は sweep、conv2d の木は再合成。探索上限 4、木ごとの真理値表の全数一致と実サンプルの前向き計算で検証)、`max` = `fast` + 探索上限 5 と ir/simulate による追加検証(実モデルで分オーダー)。conv2d の木の命令列化は全レベルで行う(IR は不変)。結果は `about.compress` に記録され `info` が表示する。個別フラグは `--compress` より優先 |
| `--resynth lossless` | conv2d 木の再合成を、simulate 検証つきで強制する(精度は不変、探索上限は既定 5)。既定の `--compress fast` でも再合成は自動で効く |
| `--resynth-max-cost N` / `--resynth-verify-samples N` | 再合成の探索ゲート数の上限(既定はレベル依存: `fast` 4、`max` / 明示指定 5)/ 検証に使うランダム入力の数(既定 8) |
| `--sweep` | dense(`lut_layer` 列)IR を sweep して定数・重複・未参照ゲートを除去する(lossless、整数一致検証込み)。既定の `fast` で自動適用されるが、`--sweep` は `--compress none` でも強制し、conv を含む IR は拒否する(conv は `--resynth`) |
| `--no-sweep` | 自動の sweep を行わない(`--sweep` と排他) |
| `--sweep-verify-samples N` | sweep の検証に使うランダム入力数(既定 8。`demo_samples` があれば実データも検証) |
| `--dense-flat` | dense の 2 入力 `lut_layer` 列を平坦ネットリスト(`lut_flat` ノード)へ置換して**バンドルの IR に焼く**互換経路。lossless。焼いた `lut_flat` は C / Web / CUDA / Metal のすべてで実行できる(Web と GPU は)。既定は焼かず、`codegen-c` / `codegen-web`(と `verify-c` / `verify-web`)が生成時にメモリ上で自動判定する(下記。CUDA / Metal は既定 `off` の opt-in)。適用順は resynth → sweep → dense-flat |
| `--dense-flat-verify-samples N` | `--dense-flat` の一致検証に使うランダム入力の本数(既定 16。`demo_samples` があれば実データでも検証する) |
| `--no-verify` | 命令列の全数等価性検証を省略する(デバッグ用。通常は使わない) |

```bash
uv run griffith deploy bundle out_examples/xor_model.json -o out_examples/xor.bundle.json
```

圧縮の適用結果は `about.compress`(`{level, applied, skipped, gates, always}`)に、再合成の検証方式は `about.resynth.verify`(`fast` / `simulate`)に、sweep の前後のゲート数は `about.sweep` に、`--dense-flat` を付けたときの縮約の内訳(ゲート数の前後・定数・パススルー・CSE)は IR の
`metadata.dense_flat` に記録され、どれも `griffith deploy info` が表示する。`lut_flat` の仕様は
[IRスキーマ](/reference/ir-schema/) の 2.6 を参照。

### griffith deploy info

| フラグ | 説明 |
|---|---|
| `source`(位置引数) | バンドル(`.json` / `.json.gz` / `.mdlb`)のパス |
| `--strict` | 命令列を IR から再構築して完全一致まで確認する |

### griffith deploy codegen-c

| フラグ | 説明 |
|---|---|
| `source`(位置引数) | バンドルのパス |
| `-o, --output DIR`(必須) | 出力ディレクトリ |
| `--name NAME` | 生成ファイル名の基底(既定: `model`) |
| `--no-runtime` | `runtime/c` のカーネルをコピーしない(既にリンク済みの場合) |
| `--fuse` | ゲート命令列を直線コードへ展開する(conv 系モデルで実測 2〜9% 速い代わりに、生成 C が 1.2〜2.3 倍になる) |
| `--dense-batch N` | dense 専用モデルにバッチ軸をパックした経路 `mdl_infer_batch_packed` を足す(最大 N 枚/回。conv を含むモデルでは無視される)。`--dense-flat` と併用でき、**平坦化が効いたときは既定で 32**(`0` で抑止)。平坦化しない dense では既定では出さない。32 枚で 1 枚ずつの `lut_flat` より 1.7〜5.9 倍速い(HAR k600〜k6000、MNIST。Apple M3)。1 枚だけ流すと `mdl_infer` の 5〜12 倍遅いのでバッチ専用 |
| `--dense-flat {auto,on,off}` | dense の回路を C 生成時に平坦ネットリスト(`lut_flat`)で実行するか(既定 `auto`)。バンドルは変えずメモリ上で平坦化する。`auto` = 試行して実効ゲート削減率が閾値以上なら採用、`on` = 常に採用、`off` = 従来の bit-slice カーネル。採否と削減率は標準出力と `model.h` の先頭コメント・`MDL_DENSE_FLAT` マクロに出る。信号表のぶん作業 RAM が増えるので RAM 優先なら `off` |
| `--dense-flat-min-reduction R` | `--dense-flat auto` の採用閾値(実効ゲート削減率、既定 0.25。ホスト実測の損益分岐は約 26%) |

```bash
uv run griffith deploy codegen-c out_examples/xor.bundle.json -o out_examples/xor_c
```

### griffith deploy codegen-cuda / codegen-metal

2 つはほぼ同じフラグを持つ(`--name` の既定だけが `model_cuda` / `model_metal` と異なる)。

| フラグ | 説明 |
|---|---|
| `source`(位置引数) | バンドルのパス |
| `-o, --output DIR`(必須) | 出力ディレクトリ |
| `--name NAME` | 生成ファイル名の基底 |
| `--no-runtime` | ランタイムをコピーしない(CUDA は `runtime/cuda` と `runtime/gpu`、Metal は `runtime/metal`) |
| `--batch N` | 同時に流せる最大枚数(既定 1)。GPU ではバッチがそのまま並列度になるので、大きいほど速い |
| `--n-streams N` | ストリームのパイプライン化(既定 1 = 従来どおり)。2 以上でバッファ集合を複製し、submit / collect の API を追加する |
| `--conv-engine {split,combined}` | conv2d の実行方式を明示する。未指定なら自動選択に任せる |
| `--conv-codegen` | conv2d ノードごとにモデル特化のカーネルを生成する(既定 OFF。コードサイズが 1.78〜6.65 倍になる) |
| `--dense-fuse` | 連続する dense 層と末尾の group_sum を 1 カーネルへ融合する |
| `--dense-packed` | dense モデルのバッチ軸を 32 サンプル / ワードのビットプレーンへ詰め替える |
| `--dense-flat {auto,on,off}` | dense を平坦ネットリスト(`lut_flat`)にして生成するか(既定 `off` = 平坦化しない。opt-in)。`auto` は削減率が閾値以上なら平坦化し、段ごとに並列実行する。`--dense-fuse` / `--dense-packed` を明示したときは平坦化しない。大きいモデル(数万ゲート級)を batch 128 以上で流すとスループットが 0.47〜0.71 倍に落ちた実測があるので、既定と比べて選ぶ |
| `--dense-flat-min-reduction R` | `--dense-flat auto` の採用閾値(実効ゲート削減率、既定 0.25) |
| `--no-auto` | 最適化の自動選択(実測に基づく判定表。既定 ON)を無効にする |

```bash
uv run griffith deploy codegen-cuda out_examples/xor.bundle.json -o out_examples/xor_cuda
uv run griffith deploy codegen-metal out_examples/xor.bundle.json -o out_examples/xor_metal
```

### griffith deploy codegen-web

| フラグ | 説明 |
|---|---|
| `source`(位置引数) | バンドルのパス |
| `-o, --output DIR`(必須) | 出力ディレクトリ |
| `--name NAME` | 生成ファイル名の基底(既定: `model`) |
| `--no-runtime` | `runtime/web/dist` をコピーしない(既に配信済みの場合) |
| `--slim` | バンドルでなく、webbench のモデル doc そのものを書き出す(簡約の保証を放棄する選択なので通常は使わない) |
| `--gzip` | gzip 版も書き出して配信サイズを測る |
| `--dense-flat {auto,on,off}` | dense を平坦ネットリスト(`lut_flat`)にして配るか(既定 `auto`)。Web は閾値 0 で、2 入力 `lut_layer` の列があれば常に平坦化する(従来経路より 1 ゲートあたりも速いため) |
| `--dense-flat-min-reduction R` | `--dense-flat auto` の採用閾値(既定 0) |

### griffith deploy verify-c / verify-metal / verify-web

| フラグ | 説明 |
|---|---|
| `source`(位置引数) | バンドルのパス |
| `-n, --samples N` | 検証サンプル数(既定 3) |
| `--seed SEED` | 乱数シード(既定 0) |
| `--engine {js,wasm}` | `verify-web` のみ。Web ランタイムのエンジン(既定 `js`。出荷前は両方を通す) |
| `--conv-codegen` / `--dense-fuse` | `verify-metal` のみ。該当する生成オプションを検証する |
| `--dense-batch N` | `verify-c` のみ。`codegen-c` と同じバッチ軸パック入口 `mdl_infer_batch_packed` を生成し、1 枚ずつの `mdl_infer` に加えてそれも照合する(既定は `codegen-c` と同じ = 平坦化を採用したときだけ 32、`0` で生成しない)。照合は 33 枚以上で行い、32 枚 / 語の境界と入口の上限での区切りを跨ぐ |
| `--dense-flat {auto,on,off}` / `--dense-flat-min-reduction R` | 3 つとも。対応する `codegen-*` と同じ dense-flat 判定(既定は `verify-c` / `verify-web` が `auto`、`verify-metal` が `off`。`auto` の閾値は C と Metal が 0.25、Web が 0)で生成して検証する。`verify-web` は全ケースを 1 バッチにした実行でも照合する。ゴールデンは**元バンドルの IR の simulate** なので、平坦化が lossless であることの実証になる |

```bash
uv run griffith deploy verify-c out_examples/xor.bundle.json -n 8 --seed 1
uv run griffith deploy verify-web out_examples/xor.bundle.json --engine wasm
```

## griffith editor

:::note[clone 環境が必要]
`griffith editor` はエディタのバックエンド実装(`tools/editor/backend/`)を import して起動する。
エディタは PyPI パッケージには入らないため、このサブコマンドは
**リポジトリを clone して `uv sync --extra editor` を実行した環境**でのみ動作する。
:::

エディタバックエンド(FastAPI)を uvicorn で起動する。フロントエンドは起動しないので、
バックエンドとフロントを同時に開発起動したい場合は `make editor` を使う。

| フラグ | 説明 |
|---|---|
| `--host` | バインドするホスト(既定: `127.0.0.1`) |
| `--port` | ポート番号(既定: `8765`) |

```bash
uv run griffith editor
make editor   # backend:8765 + frontend:5173 を同時起動
```

## griffith sweep

`sweep.yaml`(`{method, metric, parameters, base_config}`)から wandb sweep を作成し、
agent を実行する。

| フラグ | 説明 |
|---|---|
| `spec`(位置引数) | sweep YAML のパス |
| `--count COUNT` | 実行する試行数(省略時は grid 全件 / 無制限) |
| `--local` / `--modal` | 実行先(排他、既定: `--local`) |
| `--parallel N` | `--modal` 時の agent 並列数(既定: 1。試行数は `--count` を N 分割) |

```bash
uv run griffith sweep configs/sweeps/example_mnist.yaml --count 4 --local
uv run griffith sweep configs/sweeps/example_mnist.yaml --count 8 --modal --parallel 2
```

## griffith launch

`--grid` のリスト値を全組合せ(デカルト積)展開して学習を投入する。

| フラグ | 説明 |
|---|---|
| `--grid SPEC`(必須) | 掃引指定(例: `"optim.lr=[0.001,0.01] model.k=[64,256]"`。スカラー値は固定オーバーライド) |
| `config`(位置引数、省略可) | ベース設定YAML |
| `--local` / `--modal` | 実行先(排他、既定: `--local`) |
| `--dry-run` | 展開結果の一覧表示のみ(実行しない) |

```bash
uv run griffith launch --grid "optim.lr=[0.001,0.01] model.k=[64,256]" \
    configs/recipes/dlgn_mnist_quick.yaml --dry-run
uv run griffith launch --grid "optim.lr=[0.001,0.01]" \
    configs/recipes/dlgn_mnist_quick.yaml --modal
```

## griffith collect

Modal Volume の `outputs/` 配下から、ローカル未取得の run ディレクトリをダウンロードする。
対象のボリューム名は環境変数 `MD_MODAL_VOLUME`(未設定なら既定名)で指定する。

| フラグ | 説明 |
|---|---|
| `--dest DIR` | 回収先ディレクトリ(既定: `outputs/collected`) |

```bash
uv run griffith collect
```

## griffith runs list / griffith runs compare

`.wandb_runs.jsonl` と wandb 公開API で run を一覧・比較する。

`runs list`:

| フラグ | 説明 |
|---|---|
| `-n N` | 表示件数(既定: 20) |

`runs compare`:

| フラグ | 説明 |
|---|---|
| `run_ids`(位置引数、可変長) | 比較する wandb run ID |
| `--metrics` | 比較するメトリクス(カンマ区切り。省略時は全 summary キー) |
| `--format {md,csv}` | 出力形式 |

```bash
uv run griffith runs list -n 10
uv run griffith runs compare <run_id1> <run_id2> --metrics hard_acc,best_hard_acc,loss
```

## griffith report

実験レポートを生成する。`wandb-workspaces`(`cloud` extra)導入時は wandb レポート
(学習曲線+config差分+結論欄)を生成し URL を表示する。不可なら
`reports/auto/` へ markdown を生成する(出力先は `--dest` で変更できる)。

| フラグ | 説明 |
|---|---|
| `run_ids`(位置引数、可変長、必須) | 対象の wandb run ID |
| `--title` | レポートタイトル |
| `--dest DIR` | markdown フォールバックの出力先(既定: `reports/auto`) |

```bash
uv run griffith report <run_id> --title "lr×k掃引(MNIST 3エポック)"
```

## Modal連携

`--modal` 指定時は `scripts/modal_app.py` の Modal 関数を内部で spawn する。

- `griffith launch --modal` — グリッド展開した組合せごとに `train_config` を `.spawn()` し、全完了を待つ
- `griffith sweep --modal` — `sweep_agent` を `--parallel` 個並列 spawn する(試行数は `--count` を N 分割)

GPU種別は環境変数 `MODAL_GPU` で指定する(既定 A10G)。認証・イメージ構成・
成果物回収の詳細は[クラウドGPUガイド](/guides/cloud-gpu/)を参照。

## レシピ・スイープ定義との関係

- `configs/recipes/*.yaml` — `griffith train` / `griffith launch` が読むベース設定。
  [`TrainConfig`](/reference/api/training-config/#trainconfig) の YAML 表現
  (`data` / `model` / `optim` / `run` / `expected_metrics`)
- `configs/sweeps/*.yaml` — `griffith sweep` が読むグリッド/分布定義
  (`method` / `metric` / `parameters` / `base_config`)。`parameters` のキーは
  `TrainConfig` のドットパスで指定する
- `expected_metrics`(`<metric>_min` / `<metric>_max`)は `check_expected_metrics()`
  (`griffith_logic/training/config.py`)で照合され、`griffith train --strict` で
  自動化フローのゲートにできる

## 関連ページ

- [学習を回す](/guides/training/) — 実験ループ全体の中での位置づけ
- [クラウドGPU](/guides/cloud-gpu/) — Modal / RunPod での実行
- [デプロイキット クイックスタート](/guides/deploy-quickstart/) — `griffith deploy` の通し手順
- [Python API](/reference/api/) — CLI が内部で使う公開API
