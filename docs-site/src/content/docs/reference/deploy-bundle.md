---
title: '配布バンドル仕様'
description: 'デプロイキットの配布単位「バンドル」の形式。トップレベルの構造、programs / preproc_desc / readout の各節、.mdlb バイナリ形式、バージョン互換ポリシー、検証の規則、Web ターゲット。'
---

学習済みモデルを **1 ファイル = 全ターゲット共通の配布単位** にまとめた形式が **バンドル** である。
Web / C(組み込み・TinyML)/ CUDA / Metal / iOS / Android のどのランタイムも、同じバンドルを読んで **同じ整数出力** を返す。

実装は `griffith_logic/deploy/`(`bundle.py` / `programs.py`)にあり、CLI は `griffith deploy` である
(全サブコマンドは [griffith CLI](/reference/cli/#griffith-deploy))。通しの手順は
[デプロイキット クイックスタート](/guides/deploy-quickstart/) を参照。

## 設計の前提

**最適化はエクスポータ側で終わらせ、ターゲット側は実行するだけにする。**

conv2d の重み共有ゲート木は、バンドル生成時に **命令列へ縮約** して `programs` 節に焼き込む
(NOT 吸収 → 定数畳み込み → 退化した真理値表 → 同一参照 → 可換正規化 → CSE → dead 除去)。
実測でゲート数は **59.7〜95.1% 減る**。ターゲット側ランタイムはこの命令列を順に実行するだけで、縮約のパスを実装しない。
前処理も同様に、整数化済みの記述子としてデータで配る。

この分担は、ONNX Runtime のモバイル向け ORT フォーマット(グラフ最適化をオフラインで済ませ、端末側は最適化コードを持たない縮小ランタイムで実行する)と、
MediaPipe の `.task` バンドル(モデル + 前後処理 + メタデータを 1 ファイルに束ね、全プラットフォームの同一 API が読む)から採った。

`programs` と `preproc_desc` は、`ir` と `preproc` から **決定的に再構築できる冗長データ** である。
**真実源は依然として IR** で、冗長側が食い違ったら IR が正しい(`griffith deploy info --strict` がこの食い違いを検出する)。

## トップレベル構造

| キー | 必須 | 内容 |
|---|---|---|
| `bundle_format` | ✅ | `"magicdelta-bundle"` 固定 |
| `bundle_version` | ✅ | `"MAJOR.MINOR"`(現行 `1.0`) |
| `ir` | ✅ | IR(JSON)。真実源。スキーマは [IRスキーマ](/reference/ir-schema/) |
| `programs` | conv2d があれば | conv2d ノード id → 命令列(下記) |
| `preproc` | | 入力符号化の宣言(kind / mean / std / thresholds / bit_order など) |
| `preproc_desc` | 画像入力なら | 整数化済みの前処理記述子(下記) |
| `labels` | | クラス名の配列 |
| `readout` | | 出力の読み出し方(`class_scores` / `spatial_scores` / `detection` など) |
| `resources` | | ゲート数・論理深さ・配信サイズなどの見積り |
| `eval` | | 精度などの実測値 |
| `about` | | 由来(チェックポイント・レシピ・wandb run id・エクスポート時刻)。`griffith deploy bundle` が自動で適用した lossless なゲート削減の結果は `about.compress`(`{level, applied, skipped, gates, always}`)に残る。再合成を行ったときは `about.resynth` に残り、`verify` は `fast`(木ごとの全数一致 + 実サンプル)か `simulate`(ir/simulate による追加検証)のどちらか |
| `demo_samples` | | デモ兼 end-to-end 検証用の実サンプル |
| `embed_table` | RDDLGN の自己回帰生成なら | トークン ID → 埋め込みビット列(下記) |
| `folded_n_table` / `folded_l_table` | 任意 | 語彙折り畳み表(埋め込み + N / L-layers 通過後、下記) |

既存の webbench 配信 JSON は、`bundle_format` / `bundle_version` / `programs` を足しただけでバンドルになる。
追加キーは既存の TS ローダが無視するため、配信中のモデルを壊さない。

## `programs` 節

conv2d ノード 1 つにつき 1 エントリを持つ。

```jsonc
"programs": {
  "<node_id>": {
    "taps":        [int, ...],   // 受容野内フラット位置(dedup 済み、順序が tap ID)
    "n_instrs":    int,
    "instrs":      [tt, a, b, slot, ...],  // 4 要素/命令のフラット配列
    "n_slots":     int,          // 同時に生存する中間値の最大数(作業平面の本数)
    "oc_out":      [ref, ...],   // 出力チャネルごとの最終値への参照(長さ = out_channels)
    "write_start": [int, ...],   // 書き出しスケジュール(CSR、長さ = n_instrs + 2)
    "write_oc":    [int, ...],   // 同上(出力チャネルの順列、長さ = out_channels)
    "stats":       { ... }       // ゲート削減の内訳 16 項目
  }
}
```

### 参照(ref)のエンコード

`ref = (idx << 2) | kind` で、`kind` の意味は次のとおり。

| kind | 意味 | idx |
|---|---|---|
| 0 | 定数 | 0 = 偽、1 = 真 |
| 1 | タップ | `taps` の添字 |
| 2 | 命令 | `instrs` の添字。**必ず自分より前**(トポロジカル順) |

Python 側(`export/optimize.py`)と TS 側(`convProgram.ts`)で同一のエンコードで、バンドル化で変換はしない。

### 真理値表 `tt` のビット規約

2 入力関数で、`bit index = (b<<1)|a`、値 = `f(a,b)`。IR の `truth_tables`(bit k = 入力パターン k)と同じ規約なので、並べ替えずにそのまま使える。

### 書き出しスケジュール

「命令 j を実行した直後に、どの出力チャネルへ書き出すか」を CSR 形式で持つ(`write_start` / `write_oc`)。
バケット 0 は命令に依存しない出力(定数やタップ直結)で、命令の実行前に処理する。

```
バケット0        : write_oc[write_start[0] .. write_start[1])
命令 j の直後    : write_oc[write_start[j+1] .. write_start[j+2])
```

### 命令列が空になる場合

`n_instrs = 0` は正常な状態である。未学習または残差初期化のパススルー畳み込みでは、全ゲートが定数・別名へ畳み込まれて消え、
出力はすべて定数かタップ直結になる。

## `preproc_desc` 節

画像(u8)から二値チャネルへの符号化を、**浮動小数点演算を含まないデータ** へコンパイルしたもの。
実装は `deploy/preproc_desc.py` で、TS 側の真実源は `gpuPreproc.ts` と `cpuPreproc.ts` である。

```jsonc
"preproc_desc": {
  "desc":  [int, ...],  // channels × 16 ワード
  "qval":  [int; 256],  // I(k) = f32(k/255) * 2^31
  "channels": int, "image_channels": int, "height": int, "width": int,
  "stride": 16,
  "counts": { "thermo_step": n, "thermo_lut": n, "cmp": n }
}
```

1 チャネルは 16 ワードで、各ワードの内容は次のとおり。

| ワード | 内容 |
|---|---|
| 0 | op(0 = THERMO、1 = EDGE、2 = CURV、3 = CHROMA、4 = SAT、5 = THERMO_STEP) |
| 1 | 画像チャネル a |
| 2 | 画像チャネル b(CHROMA)/ 閾値 k0(THERMO_STEP) |
| 3, 4 | dy, dx(i32 の 2 の補数) |
| 5, 6 | cut の下位・上位 32bit(int64 の 2 の補数) |
| 7 | 比較の向き(0: `D >= cut`、1: `D <= cut`) |
| 8..15 | thermometer の 256 ビット LUT |

### なぜ整数化できるか

入力が u8 なので画素値は `f32(k/255)` の 256 通りしかなく、`I(k) := f32(k/255) × 2^31` は **厳密な整数**(`0 ≤ I(k) ≤ 2^31`)になる。
検出器が計算する量は、すべて `I` の整数結合で表せる。

```
edge   : D = I(a) - I(b)
curv   : D = I(a) - 2·I(m) + I(c)     (|D| ≤ 2^32 → 符号付き 34bit 必要)
chroma : D = I(a) - I(b)
彩度   : D = I(kmax) - I(kmin)
```

比較も、`t × 2^31` が 2 冪倍で厳密なので整数化できる。

```
d >  t  ⟺  D ≥ floor(t·2^31) + 1
d < -t  ⟺  D ≤ ceil(-t·2^31) - 1
```

thermometer だけは `(x - mean)·(1/std) > th` に任意の実係数が入るので整数化できないが、入力が 256 通りなので
**256 ビット LUT** にすれば、定義上ビット一致する。さらに `std > 0` なら k について単調なので、ほぼ必ず「`k ≥ k0`」の
1 閾値(THERMO_STEP)へ畳める(畳めたかどうかは LUT を総なめして、実際に階段関数かを検査してから決める)。
実モデルでは thermometer チャネルが過半数を占めるので、これが効く。境界は replicate(端の値を複製)である。

### 学習時エンコーダ(float32)との差

配信経路も記述子も **float64 相当** で計算するのに対し、学習時の `encode_image_features` は **torch の float32** で計算する。
`a - b` や `a - 2m + c` の f32 丸めが閾値をまたぐと、境界でごく稀にビットが食い違う(実測で 26,433 ビット中 3 件 = 0.01%)。
これは Web 配信経路にも元からある差で、記述子側は float64 参照と厳密一致することをテストで固定し、f32 との乖離率は別のテストで測って上限を置いている。

### 幾何の不変条件

`channels × height × width` が IR の入力幅と一致すること。ずれたバンドルは端末側で必ず壊れ、しかも「なぜか精度が出ない」という
分かりにくい症状になるので、`build_bundle` と `validate_bundle` が **生成時に落とす**。

## `readout` 節

出力をどう読み出すかを宣言する。分類は `class_scores`、セグメンテーションは `spatial_scores`(出力の空間形状つき)、検出は `detection` である。

検出(`kind: "detection"`)は、出力列に 3 つのヘッド(`hm` / `size` / `off`)を連結して持ち、`readout` が復号に必要な情報を与える。

| キー | 意味 |
|---|---|
| `heads` | `hm` / `size` / `off` の 3 ヘッド。それぞれ先頭位置(`offset`)・チャネル数(`n_out`)・高さ・幅・倍率(`scale`) |
| `output_stride` | 入力画像に対するスコアマップの縮小率 |
| `top_k` / `threshold` | 上位何個までを、どのスコア以上で採るか |
| `n_classes` | クラス数 |

**ランタイムはスコアマップまでを返し、ボックスへの復号(ピーク抽出 + NMS)は含まない**。復号はアプリ側の後処理で、
参照実装は `tools/webbench/src/detdecode.ts` にある。手順の要約は [デモモデルを用意する](/guides/deploy-models/#ボックスの復号はアプリ側で行う) を参照。

## `embed_table` / `folded_n_table` / `folded_l_table` 節

RDDLGN(`griffith_logic/models/rddlgn.py`)の seq2seq は、トークン埋め込みが **IR 化されない唯一の境界** を持つ
(IR の入力ポートは「埋め込み・Heaviside 済みのビット列」を直接受ける契約)。自己回帰生成では、前ステップの出力トークンを
次ステップの入力へ埋め込み直すので、IR の外でトークン ID からビット列への変換を実行時に行う必要がある。
この 3 節は、いずれも「トークン ID → 固定ビット列」の表を JSON へ焼き込む、同一の形をしている。

```jsonc
"embed_table":    {"vocab": V, "d_emb": D, "bits": [u32, ...]}
"folded_n_table": {"vocab": V, "width": W, "bits": [u32, ...]}
"folded_l_table": {"vocab": V, "width": W, "bits": [u32, ...]}
```

`bits` は、行(トークン)ごとに `ceil(width_or_d_emb / 32)` ワードを LSB-first でパックした配列である
(列 x はワード `x>>5` のビット `x&31`。C ランタイムのビットプレーン契約と同じ並び)。

- **`embed_table`**: `bits[token]` がそのトークンの埋め込みビット列そのもの(`sigmoid(embedding.weight) > 0.5`)。
  C 側は `mdl_embed_lookup(token, out_bits)` として焼き込まれる
- **`folded_n_table` / `folded_l_table`(任意)**: `bits[token]` が、そのトークンを埋め込んだ **あと N-layers(または L-layers)まで通した出力ビット列**。
  N / L-layers はトークン ID だけの関数(状態に依存しない)なので、有限語彙なら全トークン分を事前計算して表にできる(語彙折り畳み)。
  表サイズ(`vocab × ceil(width/32) × 4B`)が元の層の重みサイズ以上なら、**自動で `None` を返し、バンドルに節を追加しない**。
  C 側は `mdl_folded_n_lookup` / `mdl_folded_l_lookup` として焼き込まれ、`MDL_HAS_FOLDED_N_TABLE`(L 側は `_L_`)が定義される。
  N / L の両方が折り畳まれているときは、対応する重みと呼び出しを生成物から丸ごと落とせる(配信サイズの実削減。この場合、教師強制と非折り畳みの greedy 関数は使えなくなる)

## バージョン互換ポリシー

ONNX の opset 運用に倣う。

- **MAJOR 不一致 → ローダは明示的に拒否する。** 意味論が変わったとみなす
- **MINOR が読み手より新しい → 警告のみ。** 追加は後方互換として扱い、未知のキーは無視する
- IR のバージョンは別軸で、`bundle.py` の `SUPPORTED_IR_MAJOR_MINOR` に無いものは警告する(現行 `0.2`)

## 検証

`validate_bundle(bundle, strict=False)` が確認する項目は次のとおり。

- 必須キーとバージョン互換
- `programs` と conv2d ノードの 1 対 1 対応
- 配列長の不変条件(`instrs` = 4 × `n_instrs`、`write_start` = `n_instrs` + 2、`oc_out` と `write_oc` = `out_channels`)
- 参照の値域(定数は 0 / 1、タップは `taps` 内、命令は前方参照のみ)
- タップが受容野(`in_channels × kH × kW`)の内側で、重複が無いこと
- `write_oc` が出力チャネルの順列であること

`strict=True` では、**IR から命令列を再構築して完全一致まで確認する**。IR を書き換えたのに `programs` を作り直していない、
といった陳腐化を検出する(過去に「conv 抽出器が学習した配線を無視して Web デモが全部壊れていた」のを見逃した反省から、
エクスポート経路には必ずセルフチェックを置く方針である)。

なお、命令列は dead 除去を含むため、**出力に効かない枝の真理値表を書き換えても、再構築結果は変わらない**。
strict が検出するのは「IR の任意の変更」ではなく「実行される論理の変更」である。

`build_bundle(doc, verify=True)` は、書き出す前に全数の真理値表照合を通す。`verify=False` にしてよいのは故障注入のテストだけである。

### JSON 往復についての注意

`resources.lut_counts` は **整数キーの dict** で、JSON を往復するとキーが文字列化する(TS 側も文字列キーで読む)。
バンドルの正準形は **JSON デコード後の形** とする。`ir` と `programs` は往復で完全に不変である。

## `.mdlb` バイナリ形式

数十 KB の RAM 級の環境や、mmap によるゼロコピーのロードのために、JSON から決定的に変換できるフラットなバイナリ形式がある。
実装は `griffith_logic/deploy/mdlb.py`(`write_mdlb` / `read_mdlb`)と、TS 側の読み込み専用デコーダ `tools/webbench/src/deploy/mdlb.ts` で、
`.mdlb` の生成は常に Python が行う。設計の原則は TFLite の FlatBuffers 運用に倣った。

- magic `MDLB` + u32 version、リトルエンディアン
- セクションは TLV で、`IR_META` / `TT` / `INDICES` / `PROGRAM` / `PREPROC_DESC` / `LABELS` / `READOUT` / `EXTRA`
  (`EXTRA` は `resources` / `eval` / `about` / `demo_samples` / `embed_table` / `folded_*_table` など、後から増えたトップレベルキーを汎用エンコードでまとめる)
- セクションは 8 バイトアラインで、オフセットはペイロード領域の先頭からの相対値
- 各セクションに CRC32 を持つ(マジック・バージョン・セクション境界と合わせて、故障注入で検出できることをテストで固定している)
- JSON との双方向変換と往復テストがあり、契約は **`read_mdlb(write_mdlb(bundle))` が `json.loads(json.dumps(bundle))` と型まで含めて厳密一致する** こと

CLI では、`griffith deploy bundle -o m.mdlb`(拡張子で自動判定。`--format mdlb` でも指定できる)で書き出せる。
`info` / `codegen-c` / `codegen-cuda` / `codegen-metal` / `codegen-web` / `verify-c` / `verify-web` は、いずれも `.mdlb` と `.mdlb.gz` を透過的に読める。
codegen の経路(モデルを C ソースへ焼く)では `.mdlb` は不要で、動的にモデルを差し替えたい場合にだけ使う。

## CLI

バンドルの作成と確認に使うコマンドの要点は次のとおり(全フラグは [griffith CLI](/reference/cli/#griffith-deploy))。

```bash
griffith deploy bundle <model.json> -o <out.bundle.json> [--gzip] [--compress {none,fast,max}]
griffith deploy bundle <model.json> -o <out.bundle.mdlb>      # .mdlb バイナリ(拡張子で自動判定)
griffith deploy info <bundle.json|bundle.mdlb> [--strict]
```

`--compress` は lossless な回路圧縮の強さで、モデルの種類によらず同じ意味。既定の `fast` は秒で終わる削減をすべて行う(dense は sweep、conv2d は再合成)。`none` は IR を書き換える最適化なし、`max` は探索をより深く(上限 5)して追加検証も行う。
適用結果は `about.compress` に残り、`info` で確認できる。形は `{level, applied, skipped, gates: {パス: [前, 後]}, always: [{pass: 'conv-program', gates: [前, 後]}]}`。`skipped` は対象があるのに適用できなかった圧縮(dense と conv2d の混在で sweep を断った、失敗した、など)だけを載せ、対象のない圧縮(dense だけのモデルの再合成、conv だけのモデルの sweep)は載せない。conv2d の命令列化(`programs` 節)は全レベルで行われ、`always` に記録される。個別の `--sweep` / `--no-sweep` / `--resynth lossless` は `--compress` より優先される。

## Web ターゲット

C / CUDA / Metal は「バンドル → 自己完結したソース」を **生成** するが、Web は実行体(`runtime/web` = TypeScript 製の WASM / WebGPU エンジン群)を
最初から持っている。したがって Web バックエンド(`griffith_logic/deploy/webbackend.py`)がやることは非対称で、
**ソースコードは 1 行も生成せず、バンドルそのものを配布用 JSON として書き出す** だけである。
TS 側はバンドルの `programs` 節(命令列化済みの conv2d)を自前でパースしてそのまま実行し、ブラウザ内で縮約を再実行しない。

### `griffith deploy codegen-web`

```bash
uv run griffith deploy codegen-web <bundle.json> -o <out_dir> [--name model] [--slim] [--no-runtime] [--gzip]
```

既定(`--slim` 無し)の出力レイアウトは次のとおり。

```
<out_dir>/
  model.bundle.json          バンドルそのもの
  mdl_web.js                 公開 API(loadBundle)
  demo.js / index.html       単一ページのデモ
  verify.mjs                 node CLI(verify-web が使う出荷ゲート本体)
  release.wasm / release-simd.wasm   AssemblyScript 製カーネル
  lib-*.js, assets/          共有チャンク(ファイル名はビルド毎に変わるので dist は丸ごとコピーする)
```

- `--slim`: バンドルではなく、`bundle_format` / `bundle_version` / `programs` / `preproc_desc` を持たない **素の webbench モデル doc** を書き出す。
  conv2d の命令列化をブラウザ側で IR から再構築する前提になるので、**エクスポータ側で完結させていた簡約の保証を放棄する** 選択になる。
  通常の配布では使わず、IR を直接触るデバッグ・実験用途に限る
- `--no-runtime`: `runtime/web/dist` のコピーを省略する(既に配信済みの場合)。`dist` が無いときは `make web-runtime` を先に実行するよう指示して失敗する(古い、または存在しないランタイムを黙って配らない)

`resources.bundle_json_bytes` と `bundle_gzip_bytes`(UI ヘッダの配信サイズ表示が参照する)は、`griffith deploy bundle` の実行のたびに焼き直される。
`bundle_json_bytes` は実ファイルのバイト数と厳密に一致し、`bundle_gzip_bytes` は ±1 バイトの best-effort 値である。

実モデルでの実測(命令列化の効果)は次のとおり。

| モデル | バンドル | conv2d ノード | ゲート数(命令列化前 → 後) | 削減率 |
|---|---:|---:|---:|---:|
| seg_convdlgn_k128 | 310,782B | 5 | 8,512 → 2,811 | 67.0% |
| det_fewclass_lut_k64 | 521,320B | 6 | 6,272 → 1,503 | 76.0% |

`runtime/web/dist` 一式(モデル非依存)は合計 約 279KB(実測 278,704B)なので、配信量の見積りは **バンドルサイズ + 約 279KB** になる。
`griffith deploy bundle --resynth lossless` を前段に挟むと、命令列化後のゲート数をさらに削れる(精度は不変)。
seg_convdlgn_k128 では、最終的な命令列が 8,512 → 2,620(69.2% 減、再合成なしの 67.0% より 2.2pt 良い)になった。

### `griffith deploy verify-web`

```bash
uv run griffith deploy verify-web <bundle.json> [-n SAMPLES] [--seed SEED] [--engine js|wasm]
```

`ir/simulate.py`(numpy の golden model)を基準に、`runtime/web/dist/verify.mjs`(TS 製、Node 上で動く)を実際に起動して整数厳密に突き合わせる、出荷前の関門である。
検証の鎖は `PyTorch hard eval ⇔ ir/simulate ⇔ web(js) ⇔ web(wasm)` と伸びる。

- **bits 経路** は常に検証する(ランダムなビット列を simulate と web に流して突き合わせる)
- バンドルが `preproc_desc` を持つ(=画像入力)場合は、追加で **image 経路** も検証する。TS 側は生画像から **独自に** 二値化するので、
  両経路の一致は前処理実装の Python と TS の言語横断の契約そのものを確かめる。ここが一致しないのは重大なバグのシグナルなので、握りつぶさない
- `--engine js|wasm`(既定 `js`)。内部エンジンが別実装(純 TS と AssemblyScript 製)なので、出荷前に **両方** を通す
- **`strictPrograms=true` 固定**: 検証の経路は、バンドルの `programs` を読み込む **前** に全数検証し、1 つでも壊れていればフォールバックせず例外にする。
  通常の `loadBundle`(デモ UI など)は検証に失敗しても黙殺せず、理由を `notes` に残してブラウザ側の再構築へフォールバックする。
  出荷ゲートだけを非対称にしているのは、フォールバックが「古い、または改竄された `programs`」を検出できない抜け道になるのを防ぐためである
- Node.js が必要。`runtime/web/dist/verify.mjs` が無ければ、先に `make web-runtime` を実行するよう指示する

## 公開配布に関する注意

顔検出系や VOC 少クラス検出系のモデルの JSON には、`demo_samples` として **実データセット由来の画像**(WIDER FACE / Pascal VOC)が base64 で埋め込まれている場合がある。
**そのようなバンドルを、そのまま第三者へ公開配布してはならない**。`demo_samples` はデモ表示と end-to-end 検証のための添え物であり、
データセットの再配布許諾を得たものではない。公開用のバンドルは、`export_webbench.py det-ckpt ... --demo-samples 0` のように画像を埋め込まずに作ること
([デモモデルを用意する](/guides/deploy-models/#顔検出wider-face) を参照)。

## 関連ページ

- [デプロイキット クイックスタート](/guides/deploy-quickstart/) — IR からバンドル、C 生成、実行までの通し手順
- [対応プラットフォーム](/reference/platform-support/) — 各ターゲットの検証段階
- [IRスキーマ](/reference/ir-schema/) — バンドルの `ir` の仕様
- [griffith CLI](/reference/cli/#griffith-deploy) — `griffith deploy` の全サブコマンド
