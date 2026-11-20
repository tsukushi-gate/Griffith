# 31_deploy_ios — iPhone デモアプリ(デプロイキット)

学習済み DLGN の配布バンドルから生成した **C99 ソースをそのまま Swift(SwiftUI)アプリへ
組み込み**、iPhone / シミュレータ上で推論結果とレイテンシを表示する。
ランタイムは [`runtime/c/`](../../runtime/c/README.md)(依存ゼロ・ヒープ確保ゼロ・NEON 自動有効)。
モデルは「名前 + 生成 C + samples.json」の単位で何個でも同梱できる。

## 構成

| パス | 役割 |
|---|---|
| `GriffithDemo/` | アプリ本体。`ContentView.swift`(UI) / `Inference.swift`(samples デコード・実行・統計) / `GdModel.h`+`GdRun.c`(モデル記述子と時間計測) / `GriffithDemo-Bridging-Header.h` |
| `gen_model.py` | バンドル + samples.json を `Models/<name>/` へ取り込む(下記)。最後に registry と xcodeproj を再生成 |
| `gen_xcodeproj.py` | `GriffithDemo.xcodeproj/project.pbxproj` を生成(xcodegen 不要、標準ライブラリのみ) |
| `make_xor_samples.py` | XOR の samples.json(4パターン、expected_scores は IR simulate 由来) |
| `setup_xor.sh` | XOR を学習 → バンドル → samples → 取り込み、を1コマンドで |
| `Models/` , `GriffithDemo.xcodeproj/` | **生成物(gitignore。コミットしない)** |

ランタイム(`mdl_kernels.c` / ヘッダ)は **コピーせず `../../runtime/c/` を相対パスで参照**する
(HEADER_SEARCH_PATHS と pbxproj のファイル参照)。codegen-c が出す `mdl_kernels.*` の複製は使わない。

## 手順

```bash
# リポジトリルートで
bash examples/31_deploy_ios/setup_xor.sh      # = 生成(下の4コマンド相当)

xcodebuild -project examples/31_deploy_ios/GriffithDemo.xcodeproj -scheme GriffithDemo \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath /tmp/griffith_dd build
xcrun simctl boot "iPhone 17 Pro" || true
xcrun simctl install booted /tmp/griffith_dd/Build/Products/Debug-iphonesimulator/GriffithDemo.app
xcrun simctl launch --console-pty booted com.example.griffithdemo   # 画面の「実行」をタップ
```

> `-destination 'name=iPhone 15 Pro'` のように **最新 OS に存在しない機種名**を指定すると
> `OS:latest` で解決できず失敗する。`xcrun simctl list devices available` の機種名を使うか
> `'platform=iOS Simulator,name=iPhone 15 Pro,OS=17.5'` のように OS を指定する。

起動引数 `-gd_autorun 1 -gd_model <name> [-gd_expand 1]` を付けると、起動直後に自動実行する
(スクリーンショット・CI 用)。結果は **標準出力に `RESULT {json}` 1行**と `os_log`
(subsystem `com.example.griffithdemo`)へ出る。`--console-pty` で回収するか
`xcrun simctl spawn booted log stream --predicate 'subsystem == "com.example.griffithdemo"'`。
画面の「結果 JSON を共有」からも取り出せる(実機向け)。

### 生成の中身(setup_xor.sh が行うこと)

```bash
uv run python examples/06_deploy_c/export_model.py                       # out_examples/xor_model.json
uv run griffith deploy bundle out_examples/xor_model.json -o xor.bundle.json
uv run python examples/31_deploy_ios/make_xor_samples.py out_examples/xor_model.json samples.json
uv run python examples/31_deploy_ios/gen_model.py xor --bundle xor.bundle.json --samples samples.json
```

## モデルの追加(MNIST / HAR / 顔検出 …)

```bash
uv run python examples/31_deploy_ios/gen_model.py mnist --bundle mnist.bundle.json --samples mnist_samples.json
```

何度でも・任意の順で呼べる。`Models/*/` を走査して全モデルが同梱され(xcodeproj も registry も
再生成される)、アプリの Picker に並ぶ。`name` は小文字英数字と `_`(C シンボルの接頭辞になる)。
`samples.json` の `input_width` が生成 `model.h` の `MDL_INPUT_WIDTH` と違うと生成時に失敗する。

## samples.json の規約(ESP32 側 `examples/30_deploy_esp32/` と共通)

```json
{"input_width": 2,
 "samples": [{"bits": "02", "label": 1, "expected_scores": [0, 8]}]}
```

- `bits`: **hex 文字列、バイト内 LSB-first**。入力ビット `i`(`mdl_infer` の `input_bits[i]`)は
  バイト `i/8` の bit `i%8`。長さは `2 * ceil(input_width/8)` 桁。XOR の (x0,x1)=(0,1) は `"02"`。
- `label`: 正解クラス(無ければ `null`)。`expected_scores`: IR simulate の整数スコア(任意)。
  あれば画面に「スコア整数一致 n / N」を出す(C ⇔ simulate の整数一致を端末上で確認)。
- アプリは起動時にこれを 1 バイト 1 ビットへ展開して `mdl_infer_ctx` へ渡す。

### 画像モデル(顔検出など)

`scripts/make_device_samples.py` が作る画像用 samples は `bits` の代わりに
**`"image"`(u8 NCHW = 3×128×128 = 49,152 バイトの hex)** を持つ(`input_width` は前処理後の
二値入力幅 2,834,432)。`gen_model.py` は `model.h` に `mdl_infer_image_ctx` があれば画像モデルと
みなし、unit ラッパを `mdl_infer_image_ctx`(前処理込み。u8 画像をそのまま渡す)呼びにして
`gd_model.input_kind=1` / `input_bytes=C*H*W` を設定する。アプリは image をバイト列へデコードして渡す。
ラベルが無い(`label: null`)モデルは正解率を「—」とし、`expected_scores`(ホスト simulate)との
**スコア整数一致 n / N** とレイテンシを表示する。`mdl_threads.c` は同梱しない(単一スレッド実行)。

```bash
D=out_examples/demo_models
for n in har_dlgn_k6000 har_dlgn_k600 det_face_lut_k256; do
  uv run python examples/31_deploy_ios/gen_model.py $n --bundle $D/$n.bundle.json --samples $D/samples/$n.json
done
```

## 複数モデル同梱のシンボル衝突対策

生成 C は全モデルとも同じ公開名(`mdl_infer` / `mdl_infer_ctx` / `mdl_ctx_init` /
`mdl_infer_batch` / `mdl_infer_ctx_v`、画像モデルは `mdl_infer_image*` / `mdl_labels`)を
定義し、`model.h` の `mdl_ctx` 型も同名なので、そのままでは2モデル目でリンク・コンパイルが衝突する。
採用した方式は **unity ラッパ + `#define` 置換**:

```c
/* Models/<name>/<name>_unit.c (gen_model.py が生成) */
#define mdl_infer_ctx <name>_mdl_infer_ctx   /* ほか公開シンボル全部 */
#include "model.c"                           /* 生成Cを無改造で取り込む */
#undef  mdl_infer_ctx ...
const gd_model gd_model_<name> = { ... 関数ポインタとサイズ定数 ... };
```

- モデルごとに別の翻訳単位になるので `mdl_ctx` 型・`MDL_*` マクロも衝突しない。
  Swift には共通の記述子 `gd_model`(`GdModel.h`)だけを bridging header で見せ、
  モデル固有の `model.h` は見せない。
- `mdl_kernels.c`(全モデル共通)は **1回だけ**リンクする(モデルごとに複製しない)。
- 推論は `mdl_infer_ctx` + 呼び出し側確保の作業バッファ(`MDL_CTX_BYTES`、16B 整列)を
  実行ごとに 1 回だけ確保して使う。既定コンテキスト版 `mdl_infer` は使わない。
- 2モデル同梱のビルドとリンク(`nm` で `xor_mdl_*` / `xor_b_mdl_*` が別シンボル)、
  両モデルの実行を確認済み。

## 実機(iPhone)で動かす

署名はユーザー作業(Apple ID でのサインインが要るため自動化していない)。
プロジェクトは Bundle ID `com.example.griffithdemo` / 署名 Automatic / Team 空欄。

1. iPhone を接続し、信頼 + 「設定 > プライバシーとセキュリティ > デベロッパモード」を ON。
2. `open examples/31_deploy_ios/GriffithDemo.xcodeproj` → Signing & Capabilities で
   **Team を選択**し、Bundle ID を自分のものへ変更(他人と被らない値)。
3. 実機を選んで Run(⌘R)。初回は端末の「設定 > 一般 > VPNとデバイス管理」で開発者を信頼。

CLI の場合(Team ID は `security find-identity -v -p codesigning` や Xcode の設定で確認):

```bash
xcrun xctrace list devices          # 接続中の実機の UDID
xcodebuild -project examples/31_deploy_ios/GriffithDemo.xcodeproj -scheme GriffithDemo \
  -destination 'id=<udid>' -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=<TeamID> PRODUCT_BUNDLE_IDENTIFIER=com.example.<yourname>.griffithdemo \
  -derivedDataPath /tmp/griffith_dd build
xcrun devicectl device install app --device <udid> /tmp/griffith_dd/Build/Products/Debug-iphoneos/GriffithDemo.app
# アプリへの引数の前の `--` は必須(無いと devicectl が -gd_autorun を自分のオプションと解釈して失敗する)
xcrun devicectl device process launch --console --device <udid> com.example.<yourname>.griffithdemo -- -gd_autorun 1 -gd_model xor
```

署名なしのコンパイル確認だけなら(確認済み):
`xcodebuild ... -destination 'generic/platform=iOS' -configuration Release build CODE_SIGNING_ALLOWED=NO`

## 実測結果(シミュレータ)

> **注意: 下表は iPhone 17 シミュレータ(iOS 26.5)上の値で、実体は Apple Silicon ホスト Mac の
> ネイティブ実行時間。実機(iPhone)の値ではない。** 実機(iPhone 17)の値は下の「実機での計測結果」を参照。
> Release 構成(C は `-O3`)、`xcrun simctl launch --console-pty` の標準出力 `RESULT` 行から回収。
> 時間は 1 推論あたり(サンプル間の統計)。

| モデル | ゲート数 | 入力 | ctx (MDL_CTX_BYTES) | n | 正解率 | スコア整数一致 | 平均 | 中央値 | 最小 / 最大 |
|---|---:|---|---:|---:|---|---|---:|---:|---|
| xor | 32 | 2 bit | 32 B | 4 | 100% (4/4) | 4/4 | 0.042 µs | 0.042 µs | 0.042 / 0.042 µs |
| har_dlgn_k600 | 2,400 | 4608 bit | 1,200 B | 50 | 76% (38/50) | 50/50 | 2.20 µs | 2.19 µs | 2.19 / 2.27 µs |
| har_dlgn_k6000 | 36,000 | 4608 bit | 12,000 B | 50 | 80% (40/50) | 50/50 | 32.9 µs | 32.8 µs | 32.3 / 34.2 µs |
| mnist | 4,000 | 400 bit (20×20 二値) | 4,000 B | 50 | 84% (42/50) | 50/50 | 3.24 µs | 3.23 µs | 3.21 / 3.36 µs |
| det_face_lut_k256 | 2,686 | u8 画像 3×128×128 | 1,544,000 B | 10 | —(ラベル無し) | 10/10 | 1.76 ms | 1.74 ms | 1.73 / 1.82 ms |

- 正解率は W1-A が作ったデバイス用サンプル(UCI-HAR test からクラス均等 50 窓)上の値で、
  ホスト実行(`results/deploy/devices/host_har_dlgn_k600*_20261006.log`)と一致。
  サンプル 50 件はデータセット全体の精度(k6000 90.1% / k600 80.5%)の代表ではない。
- 顔検出は CenterNet のスコアマップ(5,120 要素)を出す画像モデルで、合成画像 10 枚に対する
  ホスト simulate 出力と整数一致(ラベル無し)。1 枚あたり約 1.7 ms は **シミュレータ(=ホスト CPU)の値**。
- 計測は `results/deploy/devices/ios_sim_<model>_20261006.{json,png}`(XOR の Release 版は
  `ios_sim_xor_release_*`。`ios_sim_xor_20261006.*` は前任者の Debug 実行)。集約は
  `results/deploy/devices/ios_summary_20261006.json`。実行ごとに ±数 % ぶれる(同一バイナリの再実行で
  k6000 の平均 29.6〜33.0 µs)。
- アプリサイズ(実機向け署名なし Release、5 モデル同梱、arm64): `.app` 内ファイル合計 1,951,086 B
  (主バイナリ 709,840 B + samples 1.2 MB 強)、署名なし `.ipa`(zip)433,067 B。
  モデル別の増分(バイナリ内 `__text+__const+__data`、bss 除く)+ samples JSON:
  xor 1.2 KB + 0.25 KB / har_dlgn_k600 11.5 KB + 64.9 KB / har_dlgn_k6000 162.8 KB + 65.2 KB / mnist 19.2 KB + 13.3 KB /
  det_face_lut_k256 115.5 KB + 1,096.5 KB(samples は画像 hex で、本番アプリでは不要)。
  `mdl_ctx`(顔 1.5 MB など)は実行時にだけ確保する。
- 画像モデルの対応で変えた点: `GdModel.h`(`input_kind` / `input_bytes`)、`gen_model.py`(画像検出・
  `mdl_infer_image_ctx` 呼びの unit・追加シンボル `mdl_infer_image_ctx_v` / `mdl_preproc_range_v` /
  `mdl_run_ctx` のプレフィックス化)、`Inference.swift`(`image` のデコード)、`ContentView.swift`
  (ラベル無しは正解率「—」、モデル選択を `.menu`)。

## ゲート削減の効果(シミュレータ)

> **注意: iPhone 17 シミュレータ(iOS 26.5)の値で、実体はホスト Mac の CPU 実行時間。実機の値ではない(実機の値は「実機での計測結果」)。**
> Release(C は `-O3`)、11 モデル同梱の同一バイナリ・同一セッションで、素(raw)と lossless 版(sweep / resynth)を
> **交互順に 6 回ずつ**実行し、各実行の中央値(µs/推論)の中央値を採用。全実行で `score_mismatch=0`。

```bash
S=/path/to/scratch; D=out_examples/demo_models
uv run griffith deploy bundle $D/har_dlgn_k6000.model.json --sweep -o $S/har_dlgn_k6000_sweep.bundle.json   # k2400 / k600 / mnist も同様
uv run python examples/31_deploy_ios/gen_model.py har_dlgn_k6000_sweep --bundle $S/har_dlgn_k6000_sweep.bundle.json --samples $D/samples/har_dlgn_k6000.json
uv run python examples/31_deploy_ios/gen_model.py det_face_lut_k256_resynth --bundle $D/det_face_lut_k256_resynth.bundle.json --samples $D/samples/det_face_lut_k256_resynth.json
```

sweep / resynth は lossless なので samples(期待スコア)は素のものを再利用する。

| モデル | variant | 実行ゲート数 | ctx | 正解率(50窓) | スコア不一致 | 中央値 µs(6回の範囲) | 素/変種 速度比 |
|---|---|---:|---:|---:|---:|---|---:|
| har_dlgn_k6000 | raw | 36,000 | 12,000 B | 80% | 0 | 29.91(29.50〜33.78) | |
| har_dlgn_k6000_sweep | sweep | 28,636 | 13,872 B | 80% | 0 | 23.59(23.54〜27.45) | 1.27x |
| har_dlgn_k2400 | raw | 9,600 | 4,800 B | 76% | 0 | 7.66(7.65〜9.46) | |
| har_dlgn_k2400_sweep | sweep | 8,681 | 6,284 B | 76% | 0 | 6.94(6.93〜7.70) | 1.10x |
| har_dlgn_k600 | raw | 2,400 | 1,200 B | 76% | 0 | 1.98(1.98〜2.45) | |
| har_dlgn_k600_sweep | sweep | 1,989 | 1,396 B | 76% | 0 | 1.65(1.65〜2.04) | 1.20x |
| mnist | raw | 4,000 | 4,000 B | 84% | 0 | 3.21(3.21〜3.97) | |
| mnist_sweep | sweep | 3,615 | 3,616 B | 84% | 0 | 2.91(2.91〜3.23) | 1.10x |
| det_face_lut_k256 | raw | 2,686 | 1,544,000 B | —(無ラベル) | 0 | 1,584(1,568〜1,593) | |
| det_face_lut_k256_resynth | resynth | 2,319 | 1,544,000 B | — | 0 | 1,577(1,558〜1,857) | 1.00x |

- **sweep の速度向上はゲート削減率にほぼ比例**(k6000: ゲート 0.795 倍 → 時間 0.789 倍)。mnist/k2400 は削減約 10% で
  約 1.10x。範囲(min〜max)の上端は初回実行などの外れ値で、中央値は安定(同条件の再実行で ±1% 程度)。
- **HAR k2400 は k6000 の 約 3.9x 速い**(素 29.91→7.66 µs。sweep 同士でも 3.4x)。全テスト精度は k6000 90.1% /
  k2400 89.1%(ホスト simulate、n=2947)/ k600 80.5% で、**k6000→k2400 は精度 −1.0pt で 3.9x 高速・ゲート 3.75x 減**
  (50 窓サンプルの正解率は 80% / 76% だが代表ではない)。
- **顔検出 resynth は速度差なし**(1.004x)。ゲートは 2,686→2,319(−14%)だが、時間は前処理・画像 conv 展開側
  (ctx 1.5 MB)が支配的でゲート数に依存しないとみられる(未切り分け)。
- sweep 版は `ctx`(MDL_CTX_BYTES)が増える(k6000 12,000→13,872 B、k2400 +31%、k600 +16%)。ゲート数は減るが
  作業バッファは大きくなる点に注意(メモリ制約の厳しい MCU では別途確認が必要)。
- 11 モデル同梱の署名なし実機向け Release ビルド(`generic/platform=iOS`)は成功。`.app` 内ファイル合計
  3,674,656 B(主バイナリ 1,063,128 B)、署名なし zip 819,023 B(5 モデル時 1,951,086 B)。
- 集約: `results/deploy/devices/ios_summary_20261007.json`(20261006 の行は `variant=raw` で転記)。
  各モデル `ios_sim_<name>_20261007.{json,png}`。

### `--dense-flat`(平坦ネットリスト lut_flat、表駆動 C)の追加計測

`griffith deploy bundle --sweep --dense-flat` 版(`out_examples/demo_models/<name>.flat.bundle.json`)を
`<name>_flat` として追加同梱(計 15 モデル)。samples は素のものを再利用(lossless)。生成 C が呼ぶ
`mdl_lut_flat_u16/u32` は共有の `runtime/c/src/mdl_kernels.c` にあるので、`gen_model.py` の相対参照のまま自動で入り、
`model.h` の公開シンボルも増えないためラッパのプレフィックス追加は不要。ゲート数表示のため `gen_model.py` の
`_count_gates` に lut_flat(`gates` の個数)を足した。計測は素/sweep/flat の順序をローテートして 6 回 × 2 セット = 12 回
(測定中のホスト負荷が高く 6 回単独では速/遅の二峰に揺れたため 12 回中央値を採用。素/sweep は前表と 1% 以内で一致)。
全 144 実行で `score_mismatch=0`。

| モデル | raw µs | sweep µs | flat µs | 実行ゲート数 raw/sweep/flat | ctx raw/sweep/flat | raw/flat | sweep/flat |
|---|---:|---:|---:|---|---|---:|---:|
| har_dlgn_k6000 | 29.81 | 23.46 | 15.42 | 36,000 / 28,636 / 15,124 | 12,000 / 13,872 / 25,736 B | 1.93x | 1.52x |
| har_dlgn_k2400 | 7.67 | 6.93 | 5.35 | 9,600 / 8,681 / 4,886 | 4,800 / 6,284 / 11,896 B | 1.43x | 1.30x |
| har_dlgn_k600 | 1.98 | 1.65 | 1.25 | 2,400 / 1,989 / 971 | 1,200 / 1,396 / 6,180 B | 1.58x | 1.32x |
| mnist | 3.22 | 2.92 | 2.98 | 4,000 / 3,615 / 2,210 | 4,000 / 3,616 / 4,612 B | 1.08x | 0.98x |

正解率(50 窓)は raw/sweep/flat で同一(k6000 80%、k2400 76%、k600 76%、mnist 84%)。

- flat は HAR 3 モデルで素比 1.4〜1.9x、sweep 比 1.3〜1.5x。ゲート数が約半分になる k6000 で最大。
  k6000→k2400 の速度比は raw 3.89x / sweep 3.39x / flat 2.88x(flat は小モデル側の固定コストが相対的に効く)。
- **mnist は flat が sweep と同等(0.98x、わずかに遅い)**。ゲートは 3,615→2,210 に減るが時間は減らない。
  表駆動の 1 ゲート当たりコストが sweep 後の専用ループより高いとみられる(未切り分け)。ゲート削減が約 40% 未満の
  モデルでは flat の利得は出ない可能性がある。
- **flat は ctx が増える**(k6000 12,000→25,736 B、k600 1,200→6,180 B)。時間と引き換えに作業メモリが 2〜5 倍。
- 15 モデル同梱の署名なし実機向け Release ビルドは成功。`.app` 内ファイル合計 4,034,511 B(主バイナリ 1,214,432 B)、
  署名なし zip 987,636 B(11 モデル時 3,674,656 B / 819,023 B)。
- 集約は `ios_summary_20261007.json` の `variant=flat` 行と `comparisons_flat_12runs`、各モデル
  `ios_sim_<name>_flat_20261007.{json,png}`。

## 実機での計測結果(iPhone 17)

iPhone 17(iPhone18,3、iOS 26.6.2 Build 23G90)、Release、5 モデル同梱のこのアプリで計測。1 推論あたりの中央値。
バンドルは現行既定(`bundle --compress fast`、codegen-c の dense-flat auto → XOR / HAR / MNIST は `lut_flat`)。
全モデル ir/simulate と整数一致(`score_mismatch=0`)。各モデル 1 回の測定。

| モデル | 実行ゲート | `MDL_CTX_BYTES` | 正解率 | 整数一致 | 中央値 |
|---|---:|---:|---|---|---:|
| xor | 3 | 24 B | 4/4 | 4/4 | 10.6 ns |
| har_dlgn_k600 | 971 | 6,180 B | 38/50 | 50/50 | 0.64 µs |
| mnist | 2,210 | 4,612 B | 42/50 | 50/50 | 1.73 µs |
| har_dlgn_k6000 | 15,124 | 25,736 B | 40/50 | 50/50 | 8.07 µs |
| det_face_lut_k256 | 1,279 | 1,373,440 B | —(合成画像 8 枚) | 8/8 | 0.803 ms |

- HAR は webbench 公開モデル(`tools/webbench/public/models/har_dlgn_k*_gumbel.json`)、MNIST は tutorial 29、
  顔検出は `tools/webbench/public/models/bundles/det_face_lut_k256.bundle.json`(再合成なし)から作成。
- 同じ `lut_flat` 経路のシミュレータ値(上の「ゲート削減の効果」flat、ホスト CPU)は k600 1.25 µs / MNIST 2.98 µs / k6000 15.42 µs。
  実機は 1.7〜2.0 倍速かったが、機種・回数(シミュレータは 12 回の中央値、実機は 1 回)・負荷を揃えていないので一般化しない。
- `-allowProvisioningUpdates` により、個人 Team のプロビジョニングプロファイルは自動作成された。
- 生データ: `results/deploy/devices/ios_device_<model>_20261014.json`、集約 `ios_device_summary_20261014.json`。
  wandb には記録していない(認証情報なし)。

## 実機で計測する手順

署名は Apple ID を使うので自動化していない(下記手順で計測済み)。

1. iPhone を USB(または同一ネットワーク)で接続し、信頼 + デベロッパモードを ON。
2. `open examples/31_deploy_ios/GriffithDemo.xcodeproj` → Signing & Capabilities で **Team を選択**し、
   Bundle ID を自分のものへ(例 `com.example.<yourname>.griffithdemo`)。
3. 実機を選んで **Release** で Run(Scheme を Edit > Run > Build Configuration = Release にするか、
   下の CLI を使う。Debug だと Swift だけ最適化されないが C は `-O3` なので数値への影響は小さい)。
4. CLI でインストール・起動し、`RESULT` 行を回収する:

```bash
xcrun devicectl list devices                       # 接続中の実機(Identifier / 名前)
xcodebuild -project examples/31_deploy_ios/GriffithDemo.xcodeproj -scheme GriffithDemo \
  -configuration Release -destination 'id=<udid>' -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=<TeamID> PRODUCT_BUNDLE_IDENTIFIER=com.example.<yourname>.griffithdemo \
  -derivedDataPath /tmp/griffith_dd build
xcrun devicectl device install app --device <udid> /tmp/griffith_dd/Build/Products/Release-iphoneos/GriffithDemo.app
for m in xor mnist har_dlgn_k600 har_dlgn_k6000 det_face_lut_k256; do
  xcrun devicectl device process launch --terminate-existing --console --device <udid> \
    com.example.<yourname>.griffithdemo -- -gd_autorun 1 -gd_model $m | tee /tmp/ios_dev_$m.log   # RESULT {json} 行
  # アプリは終了しないので RESULT が出たら Ctrl-C(devicectl が app へ SIGINT を転送する)
done
```

- **アプリ引数の前の `--` は必須**。無いと devicectl が `-gd_autorun` を自分のオプションと解釈し、
  `Please specify a 'timeout' value between 5 and ...` で失敗する。
- `devicectl device process launch --console` はアプリの標準出力を端末へ流す(iOS 17+ の
  CoreDevice 経由)。**iPhone 17 実機で `RESULT` 行をパイプ(`tee`)越しに回収できた**。アプリは `RESULT` を
  `print` 直後に `fflush(stdout)` する。出ない場合は次の代替を使う: (a) 画面の「結果 JSON を共有」から AirDrop/メモへ、
  (b) Console.app / `os_log`(subsystem `com.example.griffithdemo`)、(c) スクリーンショット
  (起動引数 `-gd_autorun 1 -gd_model <name>` は `--` の後ろに置けば devicectl でも渡せる)。
- 実機の値を記録するときは `results/deploy/devices/ios_device_<model>_<date>.{json,png}` と
  して保存し、`ios_summary_*.json` の `platform` を `ios-device` にして追記する。
- 機内モード・低電力モード OFF・充電ケーブル接続・画面点灯で計測する(熱で数 % 落ちる)。

## 期待される表示(XOR)

「結果: xor」カードに サンプル数 4 / 正解率 100.00% (4 / 4) / スコア整数一致 4 / 4 /
平均・中央値・最小/最大(1推論あたり。XOR は数十〜百 ns 台なので ns 表示)/
`MDL_CTX_BYTES` 32 B / 入力幅 2 bit / ゲート数 32。「サンプルごとの結果」を開くと
pred / exp / OK が並ぶ。実測は `results/deploy/devices/ios_sim_xor_20261006.{png,json}`
(iPhone 17 シミュレータ、ホスト Apple Silicon。**シミュレータの時間はホスト CPU の値で、
実機の値ではない**)。

## 計測の注意

- 時計は `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`(Apple Silicon で刻み約 41 ns)。
  1 推論が極小なので、**同じサンプルを約 200 µs 分まとめて回して割り**(回数は倍々で決定、
  3 回測って最小を採用)1 推論あたりの値にしている。統計(平均/中央値/最小/最大)は
  サンプル間の値。
- C は Debug ビルドでも `-O3`(`GCC_OPTIMIZATION_LEVEL=3`)。Swift 側は計測区間に入らない。
- ゲート数 = LUT 段の出力幅の合計 + conv2d の縮約後命令列ゲート数(`gen_model.py`)。

## ハマりどころ

- 生成する C の注釈に `Models/*/` と書くと `*/` でコメントが閉じてコンパイルエラーになる(経験済み)。
- C 構造体のメンバ名 `init` は Swift で `m.pointee.init` が struct の初期化子と衝突する → `ctx_init` に。
- `simctl launch` はシミュレータ起動直後に 30 秒でタイムアウトすることがある。再実行で通る。
- `GriffithDemo.xcodeproj` は生成物。モデルを足したら必ず `gen_model.py` を通す(通さずに
  `Models/` を手で足しても pbxproj に載らない)。
