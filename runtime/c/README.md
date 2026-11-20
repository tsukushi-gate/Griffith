# griffith-logic Cランタイム

> 公開版: docs-site の [デプロイキット クイックスタート](../../docs-site/src/content/docs/guides/deploy-quickstart.mdx)・[対応プラットフォーム](../../docs-site/src/content/docs/reference/platform-support.mdx)。

論理ゲートNN(DLGN)を **依存ゼロのC99** で動かすランタイム。組み込み /
TinyML / iOS / Android / デスクトップのどこでも同じソースが動く。

- 依存は `<stdint.h>` と `<stddef.h>` だけ(どちらもフリースタンディング処理系に
  C標準が要求するヘッダ)。libc関数は `memcpy` / `memset` のみで、
  `-DMDL_NO_LIBC` を付ければ自前実装に切り替わり **libc無しでも動く**
- **実行時のヒープ確保はゼロ**。バッファはすべてコンパイル時に確定した `static`
- **浮動小数点演算ゼロ**(前処理も含む。整数比較とLUT参照だけで符号化する)
- **配布物は命令セット別に分かれない**。SIMD(NEON / SSE2 / AVX2)はコンパイル時に
  選ばれ、無い環境ではスカラーへ落ちる。**どの経路でも整数出力は完全に同一**
  (`-DMDL_NO_SIMD` で明示的に無効化でき、SIMD経路とスカラー経路が実モデルで
  ビット完全一致することを `TestSIMDContract::
  test_scalar_forced_build_matches_host_default_build` が固定している)
- ビット位置で定義してあるので**バイト順(エンディアン)に依存しない**
- 出力は整数。JS / WASM / WebGPU / Python の `ir/simulate.py` と**ビット単位で一致**する

## 対応環境: 移植性契約

**結論**: コアは厳密なC99(依存は`<stdint.h>`/`<stddef.h>`のみ、libcは`memcpy`/`memset`
だけで`-DMDL_NO_LIBC`を付ければ自前実装に切り替わりlibcも不要)。アーキ固有コード
(NEON/SSE2/AVX2の組み込み関数・検出マクロ)は`mdl_simd.h`の1ファイルだけに隔離されていて、
対応の無いアーキ(または`-DMDL_NO_SIMD`)では自動的に**ビット同一のスカラー実装**へ
フォールバックする。ワード演算はビット位置(`x>>5`、`x&31`)で定義してあり、メモリ上の
バイト列を別の整数幅として再解釈するポインタキャストも無いので、エンディアン・アライメント
の仮定を持ち込まない。**したがってC99の(クロス)コンパイラが存在するターゲットであれば、
原則すべて動く。SIMD(NEON/SSE2/AVX2)は速くするだけで、正しさには影響しない。**

この契約は`tests/deploy/test_c_portability.py`の`TestSIMDContract`(アーキ検出マクロが
`mdl_simd.h`の外に無いこと・バイト再解釈キャストがコアに無いこと・SIMD経路とスカラー経路
〔`-DMDL_NO_SIMD`〕が実モデルでビット完全一致すること)と`TestFreestanding`(依存ヘッダが
2つだけ・ヒープ確保/浮動小数点/I/Oが無いこと)が機械的に強制している。

言い切れない残余条件が1つだけある: タップ平面の水平シフト(`mdl_kernels.c`の
`mdl_shared_shift_plane`)が、負の値に対する符号付き右シフトを「算術シフト(2の補数の床
除算)」として扱う箇所で、これは厳密なC99では処理系定義(implementation-defined)だが、
2の補数を使う実務上のあらゆるC99コンパイラで成立し(C23では標準の挙動として明文化済み)、
性能コードのため書き換えていない。監査の詳細・出典は
[`docs/reference/platform-support.md`](https://docs.griffith.magicdelta.jp/reference/platform-support/)。

### 検証実績(用途別)

上の契約により「原則動くはず」の範囲のうち、実際にビルド/実行まで確認したものを
用途でまとめる。真実源は `tests/deploy/test_c_portability.py` の `CROSS_TARGETS`
(ローカルの clang が持たないバックエンドは自動skip。`-target` は clang 専用の構文なので、
gcc しか無い環境ではクロスコンパイルのテスト自体を skip する)。

| 用途グループ | 検証段階 | 含まれるターゲット |
|---|---|---|
| 組込みARM(Cortex-M0/M0+/M3/M4/M4F/M7/M33、Cortex-A系) | ビルド確認済み | `thumbv6m` / `armv7m` / `armv7a` / `thumbv7em` / `armv8m.main` / `aarch64-none-elf` / `aarch64_be`(ビッグエンディアン) |
| Linux x86(i386/x86_64) | 実行して数値一致まで確認 | `i386` / `x86_64` |
| aarch64ホストネイティブ | 実行して数値一致まで確認(NEON⇔スカラー一致込み) | Apple Silicon(M3実測)。iPhone / Android / Raspberry Piも同アーキ |
| Windows(コア推論のみ) | MinGW gcc 実機で実行して数値一致まで確認(`tests/deploy` の C テスト一式)。MSVC ABI はクロスビルド確認のみ(`-ffreestanding -DMDL_NO_LIBC`でWindows SDK不要) | `x86_64-w64-windows-gnu`(実機)/ `x86_64-pc-windows-msvc`(クロスビルド)。cl.exe 実機での実行は未検証 |
| Xtensa(ESP32、ESP-IDF) | 実行して数値一致まで確認(HAR dense。convの顔検出128pxは静的DRAM超過で非搭載) | ESP32-D0WDQ6 rev1 240MHz・4MB flash・PSRAMなし、ESP-IDF v5.4-dev(xtensa-esp-elf 13.2.0)の実機で HAR k600(410.9µs)/ k2400(5,785.5µs)/ k6000(26.60ms)/ MNIST(725.2µs)の全50件が `ir/simulate.py` と整数一致(〜07)。`--sweep --dense-flat`(`lut_flat`)で k6000 12.69ms(最良値)・k2400 0.78ms・MNIST 324.7µs、全件 score_mismatch=0。PSRAM付きボードは未検証。`examples/30_deploy_esp32/README.md` |
| iOS(iPhone 17実機 / シミュレータ) | **実機(iPhone 17、iOS 26.6.2)で5モデルが整数一致**(HAR k6000 8.07µs、顔検出0.803ms)。シミュレータでも実測検証済み(時間はAppleシリコンのホストCPU値) | iPhone 17 シミュレータ(iOS 26.5)で XOR / HAR k600・k6000 / 顔検出(画像モデル)/ MNIST の5モデルが整数一致。`.app` 1,951,086B(5モデル同梱)。実機の値は `results/deploy/devices/ios_device_summary_20261014.json`。`examples/31_deploy_ios/README.md` |
| RISC-V / MIPS / WASM(C版。ブラウザ向けの`runtime/web/`とは別実装)/ MSP430 | バックエンドのある環境が未入手のため未検証(契約上は動く見込み) | 手元の clang にバックエンドが無く(リモートLinux機も確認したが、そちらは clang 自体が入っておらず gcc 9.4.0 の x86_64-linux-gnu ネイティブのみでクロスツールチェーンも無かった)、標準C99の範囲しか使っていないのでそれぞれのツールチェーンでビルドできるはずだが、通したという実績はまだない |

**Windows でホスト側ドライバを書くときの注意**: Windows の C ランタイムは stdin/stdout を既定で
テキストモードで開くため、生のバイト列をやり取りするプログラムでは 0x0A が 0x0D 0x0A に化ける。
`mdl_stdio.h` の `MDL_STDIO_BINARY()` を main の先頭で呼ぶとバイナリモードになる(POSIX では
何もしない)。`griffith deploy verify-c` のドライバと tests/deploy の main はこれを使っている。
ランタイム本体は stdio を使わないので、この注意はドライバ側だけの話。

ランタイムの機械語サイズは **約6KB**(macOS/arm64 で `__text` 6,216 B、
生成コードとリンクして未使用関数を落とすと 5,832 B)。他アーキでもオブジェクト
ファイルは 6.3〜9.4 KB(ELFメタデータ込み)で同じ桁に収まる。

**Windowsの並列ヘルパ(`mdl_threads.c`)は pthread 専用で非対応。** コア
(`mdl_kernels.c`+生成モデル)はWindowsでもクロスビルドが通るが、複数コアで
回す`mdl_run_pool`等はPOSIX専用。Windowsで並列に使いたい場合は、呼び出し側の
スレッド(`CreateThread`等)から`mdl_infer_ctx`(コンテキストが別なら再入可能)
を直接呼ぶ形になる。設計メモは`runtime/c/include/mdl_threads.h`のコメント参照。

UBSan + ASan で実モデルを通して未定義動作・メモリ違反ゼロを確認済み。

## 使い方

```bash
# 1) 学習済みモデル(md train の .pt)→ 配布バンドル
#    各ゲートの演算を1つに確定(離散化)してからバンドル化する。
#    モデル構成はチェックポイント内の設定を使う(無ければ --config train.yaml)
uv run md deploy bundle best.pt -o model.bundle.json
#    webbench形式のモデルJSON(前処理・ラベル付き)からも作れる
uv run md deploy bundle model.json -o model.bundle.json

# 2) バンドル → 自己完結したCソース(ランタイムも一緒にコピーされる)
uv run md deploy codegen-c model.bundle.json -o out/

# 3) 出荷前の関門: 生成CをビルドしてIR simulateとの整数一致を確かめる
uv run md deploy verify-c model.bundle.json
```

生成された `out/` には `model.h` / `model.c` とランタイム(`mdl_kernels.h` / `mdl_kernels.c` /
`mdl_simd.h`)、任意ヘルパ(`mdl_threads.h/.c`、ホストドライバ用 `mdl_stdio.h`)だけが入る。
あとは自分の `main.c` と一緒にビルドすればよい。

```bash
# ホスト(libcあり)
cc -std=c99 -O2 -I. model.c mdl_kernels.c your_main.c -o app

# ベアメタル(libcなし)。<string.h> が無い環境ではこちら
arm-none-eabi-gcc -std=c99 -Os -ffreestanding -DMDL_NO_LIBC \
  -I. model.c mdl_kernels.c your_main.c -o app.elf
```

## 学習から実行までの流れ

```
[1] 学習(PyTorch)          experiments/train_*.py → checkpoint (.pt)
      │  重みは連続値。soft評価で学習する
      ↓
[2] 離散化 + IR抽出          export/extract.py / dag.py
      │  パラメータ化(softmax16g等)の差はここで真理値表に確定させる
      │  ★ assert_ir_matches_torch: PyTorch hard eval ⇔ IR simulate を
      │    整数厳密に照合し、合わなければ書き出しを中止する
      ↓
    IR (JSON)  ← ここが真実源。以降の全ターゲットがこれを起点にする
      │
      ↓
[3] バンドル化               md deploy bundle
      │  ・conv木を命令列へ縮約(ゲート59.7〜95.1%削減)
      │  ・前処理を整数記述子へコンパイル(浮動小数点を消す)
      │  ・全数真理値表照合でセルフチェックしてから書き出す
      ↓
    バンドル (JSON) = モデル + 前処理 + readout + メタデータ の1ファイル
      │
      ├─→ [4a] md deploy codegen-c → model.c/.h + ランタイム → cc → 実行ファイル
      │        ★ md deploy verify-c: 生成CをビルドしてIR simulateと整数一致を確認
      │
      ├─→ [4b] md deploy codegen-cuda → model_cuda.cu → nvcc  ← NVIDIA GPU
      ├─→ [4c] md deploy codegen-metal → model_metal.mm → clang++  ← Apple GPU
      ├─→ [4d] tools/webbench(TS/WASM/WebGPU)      ← Web。バンドル未使用、IRを直接読む
      └─→ [4e] md export-hdl(Verilog/VHDL)         ← FPGA/ASIC
```

各段の等価性は常に `ir/simulate.py`(numpyの素朴実装)を基準に照合する。
**検証対象自身からゴールデンを作らない**のが鉄則で、これを破った過去
(エクスポート済みIRの自己整合性しか見ておらず、壊れたIR同士が一致して
全テスト緑のまま壊れたモデルを配信していた)があるため。

```
PyTorch hard eval ⇔ ir/simulate ⇔ C ランタイム(スカラー / NEON / SSE2)
                          ⇔ CUDA(RTX 4070 Ti で検証)
                          ⇔ Metal(Apple M3 で検証)
                          ⇔ TS / WASM / WebGPU
                          ⇔ Verilog / VHDL
```

**学習側は一切変えなくてよい**。既存の checkpoint から [2] 以降を回すだけで
全ターゲットの成果物が出る。

## API

```c
#include "model.h"

/* 符号化済みビット列から推論(MDL_INPUT_WIDTH 個の 0/1) */
int mdl_infer(const uint8_t *input_bits, int32_t *scores);

/* 画像から推論(前処理込み)。画像モデルでのみ生成される。
   image は [MDL_IMAGE_CHANNELS][MDL_INPUT_HEIGHT][MDL_INPUT_WIDTH_PX] の u8 */
int mdl_infer_image(const uint8_t *image, int32_t *scores);
```

作業領域を自分で持つ**コンテキスト版**もある。こちらはヒープ確保を
一切しない(メモリは呼び出し側が用意する)ので、MCU でもそのまま使える:

```c
static uint32_t work[MDL_CTX_BYTES / 4];
mdl_ctx ctx;
mdl_ctx_init(&ctx, work, sizeof(work));      /* 4バイト整列が要る */
mdl_infer_ctx(&ctx, input_bits, scores);
mdl_infer_image_ctx(&ctx, image, scores);    /* 画像モデルのみ */
```

`mdl_infer` / `mdl_infer_image` は既定コンテキスト(static)を使うだけの
薄いラッパ。**`-DMDL_NO_DEFAULT_CTX` を付けると既定コンテキストごと消え、
生成物の static RAM が 0 になる**。

**`MDL_DENSE_FLAT`。** `griffith deploy codegen-c` は既定(`--dense-flat auto`)で、
dense の `lut_layer` 列を C 生成時に縮約済みの平坦ネットリスト(`lut_flat`)へ置換するか
自動で判定する(実効ゲート削減率が閾値 0.25 以上なら採用)。結果は `model.h` の先頭コメントと
`#define MDL_DENSE_FLAT 1`(採用)/ `0`(従来の bit-slice カーネル)に出る。平坦化は信号表
(`1 + in_width + 実効ゲート数` バイト)ぶん `MDL_CTX_BYTES` が増えるので、作業 RAM が厳しい
ターゲットでは `--dense-flat off` で生成するか、`MDL_CTX_BYTES` を見てから採否を決める
(`#if MDL_DENSE_FLAT` で RAM 見積りを切り替えられる)。

サイズは `MDL_INPUT_WIDTH` / `MDL_OUTPUT_WIDTH` などのマクロで公開される。
クラス名があれば `mdl_labels[]` も生成される。

読み出しはカーネル側に用意してある:

```c
int32_t cls = mdl_argmax(scores, MDL_NUM_LABELS);          /* 分類 */
mdl_argmax_spatial(scores, n_out, h * w, labels);          /* セグメンテーション */
```

**検出モデルのボックス復号(CenterNetのピーク抽出+NMS)は入っていない。**
モデルの出力はスコアマップまでで、そこから先はアプリ側の後処理という切り分け。
復号仕様は `tools/webbench/src/detdecode.ts` にあり、バンドルの `readout` 節に
必要なメタデータ(スケール等)が入っている。

## スレッド安全性とマルチコア

**コンテキストが別なら並行に呼んでよい**。共有されるのは読み取り専用の
重み・命令列だけで、書き換わるものは全部コンテキストの中にある。
同じコンテキストを同時に使うのは不可(`mdl_infer` の既定コンテキスト版も同様)。

N枚をNコアで回すヘルパを**任意**で用意してある(`mdl_threads.h` /
`mdl_threads.c`)。**コア側は pthread を一切参照しない**ので、
ベアメタルではこの2本をリンクしなければ「依存ゼロ・ヒープ確保ゼロ」のまま:

```c
uint32_t work[N][MDL_CTX_BYTES / 4];
mdl_ctx  ctx[N];
void    *ctxp[N];
for (i = 0; i < N; i++) { mdl_ctx_init(&ctx[i], work[i], MDL_CTX_BYTES);
                          ctxp[i] = &ctx[i]; }
mdl_run_pool(mdl_infer_ctx_v, ctxp, N,
             inputs, MDL_INPUT_WIDTH,
             scores, MDL_OUTPUT_WIDTH * sizeof(int32_t), n_images);
```

```bash
cc -std=c99 -O2 -I. model.c mdl_kernels.c mdl_threads.c your_main.c -pthread -o app
```

並列化の単位は**画像1枚**。1推論はビットプレーンを次々に書き換える直線処理で
層の中に切れ目が無いため、1枚を分割するより独立した推論を並べるほうが
素直に伸びる。実測(Apple M3 = 4P + 4E コア、64枚、best of 7):

| ctx数 | seg 枚/秒 | 倍率 | detface 枚/秒 | 倍率 |
|---:|---:|---:|---:|---:|
| 1 | 3,117 | 1.00× | 1,460 | 1.00× |
| 2 | 6,255 | 2.01× | 2,925 | 2.00× |
| 4 | 11,810 | **3.79×** | 5,447 | **3.73×** |
| 8 | 18,089 | 5.80× | 6,659 | 4.56× |

Pコア4本までほぼ線形。8本目までは E コアぶんの伸びなので倍率は落ちる。

## 何が入っていて、何が入っていないか

対応しているのは **配信中の実モデルが実際に使う形だけ** :

| ノード | 対応範囲 |
|---|---|
| `conv2d` | 全般(stride/padding任意。stride=1かつ同幅は共有シフト平面の高速路) |
| `or_pool2d` | kernel=2, stride=2, padding=0 のみ |
| `upsample2d` | scale=2(最近傍)のみ |
| `permute` | チャネル軸のみ / SpatialGroupSum型(`[0,2,3,1]`+group_sum融合) |
| `concat` | チャネル軸(ビットプレーン)/ スコア列(検出の多ヘッド出力) |
| `group_sum` | 素の総和 / SpatialGroupSum融合 |
| `lut_layer` | `n_inputs <= 5` |
| `lut_flat` | 縮約済み dense 平坦ネットリスト(`griffith deploy codegen-c` が既定の `--dense-flat auto` で C 生成時に作る。`bundle --dense-flat` でIRに焼くことも可)。`mdl_lut_flat_u16/u32` が信号表を1ゲート1反復で評価。出力は置換した最終 `lut_layer` と同じビット列で、後段 `group_sum` は無改造。Web(`codegen-web` が既定で平坦化)と CUDA/Metal(`codegen-cuda`/`codegen-metal` も既定で平坦化)でも実行できる。作業RAMは信号表(`1+in_width+n_gates`バイト)ぶん `MDL_CTX_BYTES` に増える。バッチ軸には `mdl_lut_flat_packed_u16/u32`(下の「バッチ軸パック」)がある |
| `flatten` | 素通し |

範囲外の形は **生成時に例外**(`UnsupportedNode`)にする。黙って別経路へ
落とさないのは、検証していないコードを出荷しないため。必要になったら
カーネルとテストを足してから対応範囲を広げる。

`add` / `binarize` / `constant` / 汎用pool / 汎用upsample は未実装
(実モデルに現れないため)。

## 実装の要点

### ビットプレーン契約

値は `uint32_t` 配列で `[C][H][wpr]`(`wpr = ceil(W/32)`)。列 `x` はワード
`x>>5` のビット `x&31`(LSB-first)。行頭は必ずワード境界で、端数ビット
(`x >= W`)はノード値では常に0。この契約は `tools/webbench/src/engines/bitplane.ts`
と同一で、だからこそ JS/WASM/WebGPU/C が同じワード列を返す。

### バッチ軸パック(`mdl_lut_flat_packed_*`)

`lut_flat` は 1 ゲート 1 反復なので、1 枚ずつ流す限りビット並列にならない(ゲート軸でレベル化して 32 ゲートを詰める案は、
ゲート当たりのコストが 2 回のバイトgather と 1 回の store に支配されるため 0.85〜0.98 倍で不採用)。
代わりにバッチ軸で詰める: `mdl_lut_flat_packed_u16/u32` は 32 サンプルを 1 ワードに載せ、1 ゲートを bw ワードの butterfly で評価する。
入力の詰め込み `mdl_pack_batch` は `mdl_mask32` + 32x32 ビット転置に書き直してあり、層パック経路も速くなった
(値は同一。HAR k6000 n=32 で 6.10 → 4.17 µs/サンプル)。

- `griffith deploy codegen-c` は平坦化が効くとき `mdl_infer_batch_packed` を `MDL_DENSE_BATCH=32` で既定生成する。
  `--dense-batch 0` で抑止、`--dense-batch N` で最大枚数。作業バッファは呼び出し側が `MDL_DENSE_WORK_BYTES` だけ用意する(32 枚で k600 24,720B、k6000 102,932B)。flash 増は約 0.7KB。
- 実測(Apple M3、`-O2`、ホスト): n=32 で 1 枚ずつの `lut_flat` より k600 1.7 倍・k2400 4.1 倍・k6000 5.9 倍・MNIST 5.4 倍速い。
- n=1 でパック入口を呼ぶと `mdl_infer` の 5〜12 倍遅い。**バッチ専用**で、遅延重視は `mdl_infer` を使う。

### 最適化はここにはない

ゲート木の縮約(命令列化)も、タップ平面・作業平面のメモリ割り付けも、
前処理の整数化も、**すべてエクスポータ側(Python)で終わっている**。
このランタイムに来る命令列はオペランドが絶対ワードオフセットに解決済みで、
C側は種別分岐も生存区間解析も持たない。ONNX Runtime のモバイル向け
ORTフォーマットと同じ考え方。

実測でゲート数は 59.7〜95.1% 減る(モデル依存)。

## 実測(Apple M系、`-O2`、scalar C)

`uv run python scripts/bench_deploy_c.py` の出力より。

| モデル | 推論のみ | 前処理込み | (SIMD化前の e2e) | 参考: Web wasm |
|---|---:|---:|---:|---:|
| seg_convdlgn_k32 | 0.40 ms | 0.38 ms | 0.89 ms | 0.37 ms |
| det_multimnist_lut_k256 | 0.60 ms | 1.32 ms | 3.34 ms | 1.11 ms |
| det_face_lut_k256 | 0.71 ms | 1.97 ms | 5.51 ms | 1.05 ms |
| det_fewclass_lut_k64 | 0.41 ms | 0.40 ms | 2.44 ms | 0.72 ms |
| har_dlgn_k2400_gumbel | 0.03 ms | — | — | — |

Webの数字は別マシン・ブラウザ実測なので厳密な比較ではない。読み取れるのは:

- **推論だけならスカラー+SIMDのCでブラウザのSIMD WASMと同じ桁かそれ以上**
- **前処理のSIMD化(K8)で e2e が 2.3〜6.1倍**(detface 5.51 → 1.97ms)

### 前処理のSIMD化で効いたこと(K8の内訳)

op種別ごとの実測(128×128、detfaceの実記述子):

| 対策 | 効果 |
|---|---|
| op分岐をチャネル単位へホイスト(以前は画素ごとに判定していた) | 全op |
| thermometer を16画素=1命令のベクタ比較へ | **10.2倍**(2.17 → 0.21ms / 93ch) |
| 差分比較を「u8のしきい値比較」へ還元(I(k)の単調性を使う2ポインタ構築) | EDGE 2.9倍 |
| 行ポインタ・境界判定をループ外へ(replicate は端の語だけ) | CURV 1.5倍 |
| **内側ループの分岐除去**(比較結果を0/1のままシフトして積む) | detface で約2ms |
| pack(0/1バイト列 → ビット)もベクタ化 | 2.83MB で 0.45 → 0.24ms |

最後の分岐除去は**合成データのプロファイルでは見えなかった**。パターン画像だと
分岐予測が当たってしまい、乱数画像を使うベンチでだけ2msの差として現れた。

## 移植メモ(CUDA / iOS / Android)

- **iOS / Android**: このディレクトリをそのまま組み込む。Swift は C 相互運用で
  直接呼べる。Android は NDK + 薄い JNI ラッパのみ。NEON は自動で有効になる
- **CUDA / Metal**: 実装済み(`runtime/cuda`, `runtime/metal`)。カーネルの式は
  `runtime/gpu/mdl_gpu_kernels.h` で共有していて、C版とも整数一致する。
  ただし**このモデル規模ではGPUが圧勝しない**(詳細は `runtime/README.md`)
