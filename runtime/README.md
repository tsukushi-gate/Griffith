# griffith-logic ランタイム

> 公開版: docs-site の [デプロイキット クイックスタート](../docs-site/src/content/docs/guides/deploy-quickstart.mdx)・[対応プラットフォーム](../docs-site/src/content/docs/reference/platform-support.mdx)。

論理ゲートNN(DLGN)を各環境で動かす実行系。**配布バンドルは全ターゲット共通**で、
ここが違うのは実行の仕方だけ。

```
runtime/
  gpu/mdl_gpu_kernels.h      GPUカーネルの計算式(CUDAとMetalで共有する唯一の本体)
  c/                         C99 スカラー/SIMD(組み込み・モバイル・デスクトップ)
  cuda/                      NVIDIA GPU(共有カーネル + グリッド起動の殻)
  metal/                     Apple GPU(共有カーネル + Metal入口 + Obj-C++ホスト)
```

| バックエンド | 生成コマンド | 検証済み環境 |
|---|---|---|
| C(スカラー/NEON/SSE2/AVX2) | `md deploy codegen-c` | Apple M3(NEON)、Xeon W-1270P(AVX2)、**ESP32-D0WDQ6(Xtensa LX6、ESP-IDF v5.4-dev、実機でHAR k600/k2400/k6000・MNISTの全50件が整数一致。最良値は `--sweep --dense-flat`(C専用の `lut_flat`)の k6000 12.69ms。顔検出convはDRAM超過で非搭載)**、**iPhone 17実機(iOS 26.6.2、5モデルが整数一致。HAR k6000 8.07µs・顔検出0.803ms)とiOSシミュレータ(時間はホストCPU値)**、Cortex-M0/M3/M4/M33・aarch64(BE含む)・i386/x86_64 へクロスコンパイル |
| CUDA | `md deploy codegen-cuda` | RTX 4070 Ti(compute 8.9)、CUDA 12.8 |
| Metal | `md deploy codegen-metal` | Apple M3 |
| Web(TS/WASM/WebGPU) | `md deploy codegen-web` / `md deploy verify-web`(要node) | Chromium |
| Verilog / VHDL | `md export-hdl` | iverilog / ghdl |

**全バックエンドが `ir/simulate.py` と整数厳密に一致する**ことを検証済み。検証段階の詳細区分(実測検証済み/ビルドのみ確認/理論上可)は
[対応プラットフォーム](https://docs.griffith.magicdelta.jp/reference/platform-support/)を参照。

## なぜカーネルを1ファイルに閉じ込めているか

`gpu/mdl_gpu_kernels.h` は CUDA と Metal の**両方から include される唯一の
計算式**で、修飾子の違い(`__device__` / `device` / `constant`)だけを
マクロで吸収している。ここを分けると片方だけ直してビット不一致を作り込むため、
意図的に1本にしている。API 固有なのは「グリッド起動の殻」だけ:

- CUDA: `cuda/mdl_cuda.cu`(`__global__` ラッパ)
- Metal: `metal/mdl_metal_entry.metal`(バッファバインディング付きの `kernel`)

C 版(`c/src/mdl_kernels.c`)だけは別実装になっている。CPU では SIMD や
キャッシュを意識した形が GPU とまったく違うため、共有すると両方が中途半端になる。
その代わり、両者が `ir/simulate.py` と一致することをテストで固定している。

## 実測

以下は**この時点で凍結した記録**であり、数値は更新していない(このドキュメントの
ポリシーどおり、時点の実測結果としてそのまま残す)。現在のGPU効率化
キャンペーンと「さらに2倍」キャンペーンで大きく更新された値
(dense/convのCPU・CUDA・Metal各レーン、RDDLGN decodeのGPU化等)があるため、
**現在の最新性能は必ず[`docs/reference/performance.md`](https://docs.griffith.magicdelta.jp/reference/performance/)
を参照すること**。

### レイテンシ(ms/フレーム、推論のみ / 前処理込み)

| モデル | C(M3) | C: 前処理4コア | CUDA(4070 Ti) | Metal(M3) | 参考: Web wasm |
|---|---:|---:|---:|---:|---:|
| seg_convdlgn_k32 | 0.32 / 0.38 | 0.39 | 0.36 / 0.34 | 0.96 / 0.97 | 0.37 |
| det_face_lut_k256 | 0.67 / 1.59 | **0.77** | 0.80 / 0.59 | 1.60 / 1.64 | 1.05 |

### スループット(枚/秒)

| モデル | C 1ctx | C 4ctx | C 8ctx | CUDA batch128 | Metal batch256 |
|---|---:|---:|---:|---:|---:|
| seg_convdlgn_k32 | 3,163 | 12,476 | 20,266 | **69,672** | 28,626 |
| det_face_lut_k256 | 1,495 | 5,836 | 8,703 | **31,632** | 2,348(b=8) |
| har_dlgn_k6000 | 17,410 | 62,868 | 91,954 | — | — |

har は dense 専用なので `--dense-batch` のバッチ軸パックが使える:
単一スレッドで 17,410 → **51,057 枚/秒(2.93倍)**。

読み取れること:

- **GPUが単枚で圧勝しなかったのは並列度不足だった**。バッチを入れると
  Metal seg が 530 → 28,626 枚/秒(54倍)、CUDA seg が 2,937 → 69,672(23.7倍)。
  ここで初めて GPU が CPU のマルチコアを上回る
- **最適なバッチはモデルで違う**。Metal detface は batch 8 が最良で 32 では
  悪化する(作業集合がキャッシュに乗らなくなる)
- **画像が大きいモデルは前処理が支配項**(detface の e2e は76%)。
  チャネル並列化で1枚のレイテンシを 1.59 → 0.77ms に詰められる

## GPUが伸びない理由(実測で切り分け済み)

Metal(seg、Apple M3)の内訳:

| 項目 | 実測 |
|---|---:|
| 1推論 | 0.89 ms |
| CPU側のエンコード | **0.026 ms** |
| GPU待ち | **1.20 ms** |
| ディスパッチ回数 | 17回 |
| 空コマンドバッファの commit+wait(床) | 0.013 ms |
| nopカーネル22回ディスパッチ | 0.20 ms |

**ディスパッチのオーバヘッドではない。** GPUの実行そのものが遅い。原因は
**並列度の不足**で、`conv_exec` の起動スレッド数はこうなっている:

| モデル / 層 | conv_exec のスレッド数 | 1スレッドが直列に舐める命令数 |
|---|---:|---:|
| seg n1 | 128 | 132 |
| seg n3 | 32 | 202 |
| **seg n5** | **16** | **238** |
| detface n1 | 512 | 142 |
| detface n5/n6/n8/n10 | 32 | 29〜176 |

スレッド数 = 平面のワード数 = 画素数 / 32。**ビットパッキングが並列度を1/32に
する**ため、プーリング後の深い層では16〜32スレッドしか立たない。M3のGPUは
数千スレッドを前提にしているので、ほぼ空回りしている(1スレッドグループにも
満たない)。CPUでは「1命令で32画素」が強みになるのと表裏の関係。

**やってみて効かなかったこと**: conv の定数平面初期化を毎推論から init へ
ホイストしてディスパッチを 22 → 17 に減らしたが、同一条件の交互計測で
0.89ms → 0.98ms(誤差内)。**ディスパッチ数は律速ではない**と裏が取れた
(実装は無駄が減るので残してある)。

**この見立ては正しく、バッチで解決した**:
`conv_exec` の起動スレッド数を枚数倍にしたところ、Metal seg が
530 → 28,626 枚/秒(batch 256、**54倍**)、CUDA seg が 2,937 → 69,672
(batch 128、**23.7倍**)になった。ビットプレーンの並びが `[batch][C][H][wpr]`
なので、チャネル方向に独立なカーネル(pack/unpack/or_pool/upsample/
spatial_group_sum/group_sum)は**カーネルを1行も変えずに**バッチ透過だった。

もう1つのレバー(GPU側だけビットパッキングをやめる)は未着手。

## Metal のオフラインツールチェーンについて

`.metallib` を事前コンパイルすると何が速くなるかを実測した:

| 呼び出し | 時間 |
|---|---:|
| 1回目(最小シェーダ) | 51 ms ← Metalコンパイラの起動固定費 |
| 2回目(実モデル 22KB) | 73 ms ← このシェーダのコンパイル |
| 3回目(最小シェーダ再) | 0.02 ms ← キャッシュ済み |

**効くのは起動の約124msだけで、推論の0.9msには一切影響しない。**
短命プロセスやコールドスタートを詰めたいときだけ意味がある。

ランタイムは `mdl_metal_init_ex(metallib_path, source)` で
「.metallib があれば使い、無ければソースからコンパイル」に対応済み
(既定はソース。**.metallib を実際に読む経路はツールチェーン未導入のため未検証**)。

`.metallib` の作り方(Xcodeのmetalツールチェーンが要る):

```bash
xcrun metal -c model.metal -o model.air
xcrun metallib model.air -o model.metallib
```

## 詳細

- `../docs/reference/performance.md` — **最新の実測性能**(4指標・レーン別の3点比較・環境の使い分け)
- `c/README.md` — Cランタイムの使い方・対応範囲・移植メモ
- `web/README.md` — Webランタイムの使い方(dist レイアウト・loadBundle公開API・verify.mjs契約)
- `docs/reference/deploy-bundle.md` — 配布バンドルの仕様(Webターゲットの節を含む)
- デプロイキット設計メモ — Track K の全体計画と残件
