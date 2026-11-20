# 30_deploy_esp32 — デプロイキットの ESP32 実機デモ

学習済みモデル(配布バンドル)を `griffith deploy codegen-c` で C99 にし、ESP32 実機で
推論して **精度・レイテンシ・メモリ** を UART に出す ESP-IDF プロジェクト。
モデル差し替えは `gen_model.py` を1回走らせるだけ(`main.c` は触らない)。
同じ `main.c` はホスト(`cc`)でもビルドでき、実機⇔ホストの整数一致を同一サンプルで確かめられる。

検証環境: ESP32-D0WDQ6 rev v1.0(Xtensa LX6 デュアル 240MHz、flash 4MB、DRAM 約 320KB)、
ESP-IDF v5.4-dev、xtensa-esp-elf 13.2。

## ファイル

| ファイル | 役割 |
|---|---|
| `gen_model.py` | バンドル → `main/model/`(model.c/.h + ランタイム + `samples.h` + `model_info.json`) |
| `main/main.c` | 同梱サンプルを順に推論して計測。`#ifdef ESP_PLATFORM` で実機/ホスト両対応 |
| `main/CMakeLists.txt` / `CMakeLists.txt` / `sdkconfig.defaults` | ESP-IDF プロジェクト設定(`-O2` 相当 `COMPILER_OPTIMIZATION_PERF`、240MHz、flash 4MB) |
| `../../scripts/esp32_capture.py` | UART ログを非対話で取得(DTR/RTS リセット)、`RESULT` 行を JSON 抽出 |
| `../../tests/deploy/test_esp32_project_host.py` | ESP-IDF 不要のホスト検証(自動テスト) |

`main/model/`・`build/`・`sdkconfig` は生成物なので `.gitignore` 済み。

## 手順

### 1. セットアップ(初回のみ)

```bash
# ESP-IDF 環境(例: ~/esp/esp-idf)。python3 が IDF の venv と別バージョンのときは
# IDF の venv の bin を先頭に置いてから export.sh を読み込む
export PATH=$HOME/.espressif/python_env/idf5.4_py3.12_env/bin:$PATH
source ~/esp/esp-idf/export.sh
```

`export.sh` は **パイプ(`| tail` 等)に繋がず**直接 `source` すること(パイプ先のサブシェルで
PATH が失われる)。

### 2. モデル生成(バンドル → `main/model/`)

```bash
# XOR(既定): examples/06_deploy_c/export_model.py → bundle を自動生成し、内蔵の4パターンを載せる
uv run python examples/30_deploy_esp32/gen_model.py

# 任意モデル: バンドル + サンプルJSON(scripts/make_device_samples.py の出力)
uv run python examples/30_deploy_esp32/gen_model.py \
    --bundle out_examples/har.bundle.json \
    --samples out_examples/demo_models/samples/har.json --name har [--max-samples 64]
```

`--samples` の JSON は(画像入力モデルは `bits` の代わりに `image`、下の「実測結果」参照)`{"input_width": N, "samples": [{"bits": "<hex>", "label": int|null,
"expected_scores": [int,...]|null}]}`。`bits` は**バイト列の hex**(`bytes.fromhex`)で、
入力 i はバイト `i//8` のビット `i%8`(LSB-first)。`label` が null なら分類精度の対象外、
`expected_scores`(IR simulate 由来)があれば全サンプルのスコアを実機上で整数照合する。

サンプルは **パック形式のまま** `samples.h` の `SAMPLE_BITS[N][ceil(W/8)]` に載せ、`main.c` が
0/1 バイト列(`mdl_infer_ctx` が要求する形式)へ展開してから推論する。展開済みを持つと
N×W バイト flash を食う(HAR は W=4608 なので 8倍になる)ため。

### 3. ビルド・書込・ログ取得

```bash
cd examples/30_deploy_esp32
idf.py set-target esp32            # 初回のみ(sdkconfig.defaults を取り込む)
idf.py build                       # 初回 約30秒(M系Mac)
idf.py -p /dev/cu.usbserial-XXXX flash
cd ../..
uv run python scripts/esp32_capture.py --port /dev/cu.usbserial-XXXX \
    --baud 115200 --seconds 30 --until DONE --out results/deploy/devices/esp32_xor.log
# → 生ログ + results/deploy/devices/esp32_xor.json(RESULT行をJSON化)
idf.py size ; idf.py size-components   # メモリ内訳(flash/IRAM/DRAM)
```

`idf.py monitor` は対話端末が要るので使わず、`esp32_capture.py` を使う。pyserial は `fpga` extra
(`uv sync --extra fpga`)にあるが、venv に無ければスクリプトが ESP-IDF の Python 環境へ自動で
切り替えて再実行する。

別モデルを別 build dir で並べたいときは `-B` と `-DMODEL_DIR` を使う(`main/model/` を触らない):

```bash
uv run python examples/30_deploy_esp32/gen_model.py --bundle B --samples S --out /tmp/m_har
idf.py -B /tmp/b_har -DSDKCONFIG=/tmp/b_har.sdkconfig -DMODEL_DIR=/tmp/m_har set-target esp32
idf.py -B /tmp/b_har -DSDKCONFIG=/tmp/b_har.sdkconfig -DMODEL_DIR=/tmp/m_har build
```

### 4. ホストで同じサンプルを実行(実機との一致確認)

```bash
cd examples/30_deploy_esp32
cc -std=c99 -O2 -DHOST_BUILD -DMDL_NO_DEFAULT_CTX -Imain -Imain/model \
   main/main.c main/model/model.c main/model/mdl_kernels.c -o /tmp/host_demo && /tmp/host_demo
```

`SAMPLE i ...` 行のうち `ns=`/`reps=` 以外(`pred`/`exp`/`h`/`scores`)と `RESULT` の `score_hash` が
実機と**完全一致**すること。`h` は1サンプルのスコア列、`score_hash` は全サンプル通しの FNV-1a。

## 実測結果(ESP32 実機)

ボード ESP32-D0WDQ6 rev v1.0(Xtensa LX6 デュアル 240MHz、flash 4MB、PSRAM なし)/
ESP-IDF v5.4-dev-2194-gd7ca8b94c8(xtensa-esp-elf 13.2.0、`-O2` 相当)。
モデルは W1-A の本番バンドル(`out_examples/demo_models/`、wandb アーティファクトから取得した学習済み IR。
この学習済みモデルは現状は開発チーム内の非公開 wandb にのみあり、公開配布先は未定)で、
サンプルは HAR がテスト窓 50 件、MNIST がテスト画像 50 枚(クラス均等)で、いずれも `label` と IR simulate 由来の `expected_scores` 付き。
時間は 1 推論あたり(サンプルごとに 5ms 以上になるまで反復して平均)。全数値は
`results/deploy/devices/esp32_summary_20261006.json`(個別ログ・`idf.py size` も同ディレクトリ)。

| モデル | ゲート | 入力 | 実機 acc | 参照スコア一致 | 1推論 mean / median(min–max) | ctx | app bin | .rodata(flash) | 静的DRAM空き |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| XOR | 32 | 2 bit | 4/4 | -(ラベルのみ) | 5.86 µs / 5.84 µs | 32 B | 172,752 B | 36,408 B | 169,784 B |
| MNIST(20x20 二値, 10クラス) | 4,000 | 400 bit | 42/50(84.0%) | 50/50 | 725.2 µs / 725.2 µs(725.2–726.0) | 4,000 B | 195,392 B | 58,936 B | 165,024 B |
| HAR k600(4層) | 2,400 | 4608 bit | 38/50(76.0%) | 50/50 | 410.9 µs / 410.8 µs(410.7–413.0) | 1,200 B | 213,760 B | 77,256 B | 163,632 B |
| HAR k6000(6層) | 36,000 | 4608 bit | 40/50(80.0%) | 50/50 | 26.60 ms / 26.60 ms(26.60–26.60) | 12,000 B | 365,072 B | 228,488 B | 152,832 B |
| HAR 36,000 ストレス(ランダム入力) | 36,000 | 4608 bit | 8/8 | 8/8 | 26.59 ms | 12,000 B | 341,488 B | 203,112 B | 153,168 B |
| 顔検出 CenterNetLUT k256(128px) | 2,686(重み共有 14,784) | 3×128×128 u8 | - | - | **ビルド不可(`dram0_0_seg` を 1,394,696 B 超過)** | 1,544,000 B | - | - | 180,736 B 中 |

- 実機とホスト(`cc -O2`)で、HAR 2 本と MNIST とも **全 50 サンプルの `SAMPLE` 行(`pred`/`exp`/`h`/`scores`)と
  `score_hash` が完全一致**(`host_*_20261006.log` と `esp32_*_20261006.log` を比較)。HAR の acc 40/50・38/50 は
  W1-A の IR simulate と同じ値(50 件は全テストの部分集合なので、全体 2947 件の 90.1% / 80.5% とは別の数)
- スループット換算: MNIST 約 1,380 枚/秒、k600 約 2,430 窓/秒、k6000 約 37.6 窓/秒(いずれもデュアルコアの片方のみ使用)
- IRAM は全モデル 48,287 B(IDF 自身。モデルのコードは flash 実行)で変化なし。静的 DRAM は
  `.data` 8,648 B + `.bss`(ctx + 入力展開バッファ + 統計)のみで、重みは flash に置かれる
- ホスト参考(Apple Silicon、`cc -O2`、スカラー): MNIST 約 3.4 µs(median)、k600 1.85 µs、k6000 27.1 µs。顔検出は 1.76 ms/枚
  (10 サンプル全て参照スコアと整数一致)で、ロジック自体は正しく動くが ESP32 には載らない
- **顔検出が載らない理由は実測**: `MDL_CTX_BYTES`=1,544,000 B の `static work[]` が `.dram0.bss` 1,566,784 B となり
  リンカが `region 'dram0_0_seg' overflowed by 1394696 bytes`(静的 DRAM は 180,736 B)で失敗。
  `malloc` に逃がしても起動直後の空きヒープが約 284〜300 KB なので不足し、PSRAM 無しの無印 ESP32 では不可。
  PSRAM 付きボード(ESP32-WROVER / ESP32-S3 の 8MB PSRAM 等)なら ctx を PSRAM へ置けば容量上は可能だが
  **未検証**。詳細は `results/deploy/devices/esp32_det_face_build_20261006.log`
- 実機の `ns` は `esp_timer`(µs 分解能)の平均なので XOR 級は 5.84〜5.93 µs の離散値になる

画像入力モデル(`input_kind: image_u8`)のサンプルは `gen_model.py` が自動認識し、`samples.h` に
`SAMPLE_INPUT_IS_IMAGE 1` を立てる(`main.c` は 0/1 展開せず `mdl_infer_image_ctx` へ渡す)。
参照スコアがあるときは終了コードの合否を「スコア整数一致」だけで判定する(実モデルの精度は 100% でないため)。

### ゲート削減の効果

`griffith deploy bundle --sweep`(dense IR の定数吸収・CSE・未参照ゲート除去。lossless=任意入力で出力不変、
整数一致検証込み)でバンドルを作り、同じ手順(`gen_model.py` → `idf.py -B/-DMODEL_DIR` → flash → capture)で実機計測した。
サンプルは上の実測と同じ 50 件(`expected_scores` も同一)。**全ケース `score_mismatch=0`、`score_hash` は素と sweep で同一、
ホスト `cc -O2` の `SAMPLE` 行(`ns=`/`reps=` 除く)とも 50/50 完全一致**。ボード・IDF・最適化は上と同じ。
全数値は `results/deploy/devices/esp32_summary_20261007.json`(前回 20261006 の素の行も転記済み)。

```bash
uv run griffith deploy bundle out_examples/demo_models/har_dlgn_k2400.model.json --sweep \
    -o out_examples/demo_models/har_dlgn_k2400.sweep.bundle.json
```

| モデル | variant | 実行ゲート | 1推論 mean | 実機 acc | score_mismatch | ctx | app bin | .rodata | DRAM 使用 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| HAR k600 | 素 | 2,400 | 410.9 µs | 38/50 | 0 | 1,200 B | 213,760 B | 77,256 B | 17,104 B |
| HAR k600 | sweep | 1,989(-17.1%) | 342.8 µs(-16.6%) | 38/50 | 0 | 1,396 B | 211,936 B | 75,416 B | 17,304 B |
| HAR k600 | flat | 971(-59.5%) | 290.7 µs | 38/50 | 0 | 6,180 B | 208,064 B | 72,504 B | 22,088 B |
| HAR k2400 | 素 | 9,600 | 5,785.5 µs | 38/50 | 0 | 4,800 B | 246,160 B | 109,640 B | 20,704 B |
| HAR k2400 | sweep | 8,681(-9.6%) | 4,163.7 µs(-28.0%) | 38/50 | 0 | 6,284 B | 242,080 B | 105,528 B | 22,192 B |
| HAR k2400 | flat | 4,886(-49.1%) | 782.2 µs | 38/50 | 0 | 11,896 B | 229,888 B | 95,672 B | 27,800 B |
| HAR k6000 | 素 | 36,000 | 26,598.7 µs | 40/50 | 0 | 12,000 B | 365,072 B | 228,488 B | 27,904 B |
| HAR k6000 | sweep | 28,636(-20.5%) | 21,174.9 µs(-20.4%) | 40/50 | 0 | 13,872 B | 331,968 B | 195,352 B | 29,776 B |
| HAR k6000 | flat | 15,124(-58.0%) | 12,688.0 µs | 40/50 | 0 | 25,736 B | 288,272 B | 154,056 B | 41,640 B |
| MNIST | 素 | 4,000 | 725.2 µs | 42/50 | 0 | 4,000 B | 195,392 B | 58,936 B | 15,712 B |
| MNIST | sweep | 3,615(-9.6%) | 638.1 µs(-12.0%) | 42/50 | 0 | 3,616 B | 193,664 B | 57,208 B | 15,328 B |
| MNIST | flat | 2,210(-44.8%) | 324.7 µs | 42/50 | 0 | 4,612 B | 190,176 B | 55,976 B | 16,328 B |

(素の k600/k6000/MNIST は上の実測の値を転記、k2400 素と sweep・flat 行は今回の実測。flat は `--sweep --dense-flat`
(sweep 後の dense IR をパススルー・定数を除いた平坦ネットリスト `lut_flat` にし、信号表を引く表駆動 C を生成)。
flat の実行ゲートは `lut_flat.gates` の総数で、削減率は元のゲート数に対する値。`RESULT` の `gates` は `model.h` のマクロ由来で flat は 0 と出る(`lut_flat` を数えない)。sweep のゲート内訳は
k6000 が 定数吸収4,246・CSE統合836・除去7,364、k2400 が 402・120・919、k600 が 270・43・411、MNIST が 56・280・385)

- **速度はほぼ実行ゲート数に比例する**: k6000 は ゲート -20.5% に対し 時間 -20.4%、k600 は -17.1% に対し -16.6%。
  1 ゲートあたりの時間は sweep の前後でほぼ同じ(k6000 0.739→0.740 µs、k600 0.171→0.172 µs)なので、
  削ったゲートの分だけ素直に速くなる。
- **比例を外れた例(実測の事実のみ)**: k2400 は ゲート -9.6% に対し 時間 -28.0%(0.603→0.480 µs/ゲート)、
  MNIST は -9.6% に対し -12.0%(0.181→0.177 µs/ゲート)。原因は未調査(層ごとに削られるゲートの偏りや
  flash の読み出し局所性が候補だが、検証していない)。
- モデル間では 1 ゲートあたり 0.17 µs(k600/MNIST)から 0.74 µs(k6000)まで開く(k2400 は素で 0.60 µs)。
  ゲート数だけでは時間を予測できず、層幅の大きいモデルほど 1 ゲートが遅い傾向が実測に出ている(原因は未検証)。
- **flash は小さくなる**: .rodata は k6000 で -33,136 B(-14.5%)、app bin は -33,104 B。.rodata は k2400 -4,112 B、k600 -1,840 B、MNIST -1,728 B。
  モデル表(命令列)がゲート数に比例して縮むため。
- **DRAM(静的)はむしろ僅かに増える**: ctx(作業RAM)が k600 +196 B、k2400 +1,484 B、k6000 +1,872 B(MNIST は -384 B)。
  sweep 後のゲート配置では生存区間解析の ctx が縮まないことがある。いずれも静的 DRAM 空きは約 150〜165 KB のままで影響は小さい。
- 精度は不変(acc 38/50・38/50・40/50・42/50 と `score_hash` が素と同一)。lossless なので当然だが実機でも確認できた。

#### flat(`--sweep --dense-flat`)の実機結果

flat も全ケース `score_mismatch=0`、`score_hash` は素・sweep と同一、ホストの `SAMPLE` 行と 50/50 一致(`host_*_flat_20261007.log`)。

| モデル | 素 → sweep → flat の mean | 素比 | sweep 比 |
|---|---|---:|---:|
| HAR k600 | 410.9 → 342.8 → 290.7 µs | 0.71(1.41x 速い) | 0.85 |
| HAR k2400 | 5,785.5 → 4,163.7 → 782.2 µs | 0.135(7.40x 速い) | 0.19(5.32x) |
| HAR k6000 | 26,598.7 → 21,174.9 → 12,688.0 µs | 0.48(2.10x 速い) | 0.60 |
| MNIST | 725.2 → 638.1 → 324.7 µs | 0.45(2.23x 速い) | 0.51 |

- **ESP32 では flat が 4 モデルすべてで最速**。ホストで見えていた「ゲートあたりは重い(0.74→1.0 ns/gate、損益分岐は削減率約26%)」
  という懸念は、削減率が 45〜60% ある今回の 4 モデルでは実機の絶対時間に現れなかった。ただし 1 ゲートあたり時間は
  flat のゲート定義(パススルーを数えない)が違うので variant 間では比べられない(k6000 0.84 µs、k2400 0.16 µs、k600 0.30 µs、MNIST 0.15 µs)。
- ホスト(`cc -O2`、今回の同サンプル 50 件、median)との差: k6000 は 22.0 → 15.6 µs と実機同様に速くなるが、
  MNIST はホストでは sweep 2.70 µs → flat 2.98 µs と**遅くなる**一方、実機では 638.1 → 324.7 µs と半減した。
  ホストの傾向を ESP32 にそのまま当てはめられない(命令/メモリ階層の違い。原因の切り分けはしていない)。
- k2400 の 7.4x は削減率(49.1%)だけでは説明できず、素・sweep で効いていた何かが flat で消えた可能性がある(未検証。
  k6000 は削減率 58.0% に対し 2.1x と、ゲート数比 0.42 より速度比 0.48 でほぼ比例)。
- **メモリは flash が減り、DRAM が増える**: .rodata は k6000 154,056 B(素 228,488 / sweep 195,352)、k2400 95,672 B、
  k600 72,504 B、MNIST 55,976 B と全モデルで sweep より小さい。静的 DRAM は信号表のぶん ctx が増え
  (k6000 25,736 B、k2400 11,896 B、k600 6,180 B、MNIST 4,612 B)、DRAM 使用は k6000 41,640 B(素 27,904 B)、k2400 27,800 B、
  k600 22,088 B、MNIST 16,328 B。静的 DRAM の空きは k6000 でも約 139 KB 残る。

## 出力形式と期待出力(XOR)

```
== model=xor platform=esp32 samples=4 input=2 output=2 ctx=32B gates=32 ==
SAMPLE 0 pred=0 exp=0 ns=5927 reps=55 h=36c02b5c scores=8,1
SAMPLE 1 pred=1 exp=1 ns=5841 reps=833 h=1d4369ed scores=0,8
...
RESULT {"model":"xor","platform":"esp32","n":4,"n_labeled":4,"correct":4,"acc":1.0000,
  "score_checked":0,"score_mismatch":0,"mean_us":5.862,"median_us":5.841,"min_us":5.841,
  "max_us":5.927,"ctx_bytes":32,"input_width":2,"output_width":2,"gates":32,
  "score_hash":"c116c6e5","cpu_mhz":240,"free_heap":300648,"min_free_stack_bytes":6840}
DONE
```

- 時間は1推論あたり。1サンプルごとに「合計 5ms 以上になるまで」(最大5000回)繰り返して平均する。
  `esp_timer`(µs 分解能)でも XOR 級の短い推論が測れる
- `acc` は `label` を持つサンプルの正解率。`score_mismatch` は `expected_scores` との不一致数
- `ctx_bytes` は `static uint32_t work[]`(.bss)。ヒープ確保なし・浮動小数点なし
- `min_free_stack_bytes` は main task のスタック余裕(`uxTaskGetStackHighWaterMark`)
- `DONE` を出したら停止(10秒ごとの `vTaskDelay` ループ)。再実行は capture が再リセットするとき

## トラブルシュート

| 症状 | 原因・対処 |
|---|---|
| `idf.py` が `Cannot import module "click"` | `python3` が IDF の venv と別バージョン。上記のとおり IDF venv の `bin` を PATH 先頭に置いてから `source export.sh` |
| `main/model/model.c が無い` | `gen_model.py` を先に実行する |
| `#error samples.h と model.h の入出力幅が一致しない` | 別バンドルの `model/` と `samples.h` が混在。`gen_model.py` を同じバンドルで再実行 |
| capture が `TIMEOUT` | リセットで起動ログが取れていない/ポート違い。`--seconds` を延ばす、`--no-reset` で現状を見る、USB ケーブル(給電のみ)を確認 |
| capture の先頭が文字化け | ROM ブートローダは 74880 baud で出すため(`ets Jun  8 ...`)。アプリのログは 115200 で正常 |
| flash が失敗 | 他プロセスがポートを掴んでいる(`idf.py monitor` 等)。`-b 115200` で速度を落とす |
| `.bss` が溢れる(`region dram0_0_seg overflowed`) | `MDL_CTX_BYTES`(作業RAM)+ 入力展開バッファが静的 DRAM 約 170KB を超えた。下の「大きいモデル」参照 |

## 大きいモデルを載せるときの注意

実測(XOR と、HAR の実 IR=36,000 ゲート dense・入力 4608 bit をランダム入力 8 サンプルで
載せたストレステスト。いずれも `results/deploy/devices/` にログと `idf.py size` あり)

| | XOR | 36,000ゲート dense |
|---|---:|---:|
| app bin | 172,752 B | 341,488 B |
| `.rodata`(flash、model.c のテーブル) | 36,408 B(IDF分込み) | 203,112 B(うち libmain 166,880 B) |
| `.text`(flash code / IRAM) | 79,048 / 47,259 B | 79,272 / 47,259 B |
| DRAM `.bss` | 2,304 B | 18,920 B(ctx 12,000 + 入力展開 4,608 + ...) |
| 1推論 | 5.9 µs | 26.6 ms |

- **重み・命令列は `static const` で `.rodata`(flash)へ行き、DRAM を食わない**(上表で確認)。
  model.c 400KB → flash 約 167KB。flash 使用は 4MB パーティション(app 1.5MB)が上限
- **DRAM は作業RAM(`MDL_CTX_BYTES`)+ 入力展開バッファ(`MDL_INPUT_WIDTH` B)** が支配的で、
  ESP32 の静的 DRAM は約 170KB(BT 無効時でも `dram0_0_seg` 180,736 B から IDF 分を引いた値)。
  顔検出 128px conv のように `MDL_CTX_BYTES` が数百KB になるモデルは**静的には載らない**。
  その場合 PSRAM 付きボードで `heap_caps_malloc(MALLOC_CAP_SPIRAM)` を使うか、
  ctx を縮める(生存区間解析は適用済み)かのどちらか。作業RAMは
  `mdl_ctx_init(&ctx, buf, bytes)` に渡す任意バッファでよいので `main.c` の差し替えだけで済む
- **IRAM は影響なし**(カーネルは flash 実行。IDF 自身が約 47KB 使用)。`CONFIG_COMPILER_OPTIMIZATION_PERF`
  で `-O2`。flash キャッシュミスで遅くなりうる(XOR の µs 域では見えない)
- **main task スタック**: 推論は再帰・大きなローカル配列を持たない(作業は ctx に取る)。
  `printf` で約 1.3KB 使い、8192 B 設定で 6.8KB 余った。サンプル数 N が大きいとき
  `main.c` の `sample_ns`/`sorted_ns`(各 4N B)は static なのでスタックには影響しない
- **watchdog**: 1 推論が数百 ms を超えるモデルは task WDT に注意(`sdkconfig.defaults` で無効化済み。
  サンプル間に `vTaskDelay(1)` で IDLE に譲っている)
- **flash 書込サイズ**: サンプル `SAMPLE_BITS` は N×ceil(W/8) B。入力が大きいモデルは
  `--max-samples` で絞る
- 速度の上限: スカラー経路のみ(Xtensa に SIMD 実装は無い)。36,000 ゲートで 26.6 ms(約 38 枚/秒)。
  デュアルコアは `mdl_run_pool`(pthread 版)をそのままは使えない(FreeRTOS タスク2本で
  ctx を2つ持つ形に書き換えが要る)
