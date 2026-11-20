# examples — 実行可能な最小サンプル集

各サンプルは CPU で数分以内に完了し、末尾のassertで期待結果を自己検証する
(01〜06 は1分以内、07〜23 のモデル別サンプルは 7秒〜3分)。
図を出すサンプルは matplotlib を使うので `uv sync --extra cpu --extra figures` が要る。

各サンプルは [ドキュメントサイトのチュートリアル](https://tsukushi-gate.github.io/Griffith/tutorials/mnist/)
として実行結果つきで公開されている(生成は `uv run --group docs python scripts/build_tutorials.py`)。

## 基本(01〜06)

| サンプル | 内容 | 実行コマンド | 期待結果 |
|---|---|---|---|
| 01_xor.py | XOR学習(soft訓練→hard回路への崩壊)+学習前後の回路図 | `uv run python examples/01_xor.py` | soft/hard共にacc=1.00 |
| 02_hybrid_llvit.py | LL-ViT型ハイブリッド(AttentionはPyTorch、MLPだけLUT置換) | `uv run python examples/02_hybrid_llvit.py` | 置換2箇所+損失減少 |
| 03_learning_rules.py | 学習ルール切替(DFA / AP法をwith_learningで注釈) | `uv run python examples/03_learning_rules.py` | 両ルールでacc=1.00 |
| 04_export_hdl.py | 学習済みモデル→IR→Verilog+iverilog照合 | `uv run python examples/04_export_hdl.py` | IR予測=正解、照合ALLPASS |
| 05_cnn_export_hdl.py | CNN(論理ゲート畳み込み)→IR unroll→Verilog/VHDL+照合 | `uv run python examples/05_cnn_export_hdl.py` | simulate予測一致100%、照合ALLPASS |
| 06_mnist.py | MNIST 20x20を4000ゲートで学習(回路俯瞰図・学習曲線・誤分類例) | `uv run python examples/06_mnist.py` | hard acc 0.86前後(assertは>0.70) |
| [06_deploy_c/](06_deploy_c/) | 学習済みモデル→IR→配布バンドル→C99実行ファイル(デプロイキット) | `06_deploy_c/README.md`参照(export→bundle→codegen-c→cc) | 4パターン全一致 |

## 系列・言語(07〜12)

| サンプル | 内容 | 実行時間 | 期待結果 |
|---|---|---|---|
| 07_logic_gpt.py | 全部論理ゲートのdecoder-only Transformerで文字レベル言語モデル(Tiny Shakespeare) | 約150秒 | val bpc 3.13(uniform baseline 5.00)+生成テキスト |
| 08_rddlgn_seq2seq.py | RDDLGNでcopy/reverse。attentionなし固定長ボトルネック(384bit) | 約60秒 | copy hard acc 0.969 / reverse 0.941 |
| 09_logic_lstm.py | 忘却ゲート=MUX・出力ゲート=ビットAND。LSTM/GRU vs ゲートなしElman | 約20秒 | LSTM/GRU hard acc 1.000 / Elman 0.578 |
| 10_logic_wavenet.py | 因果convolutionと受容野のablation(1層 vs 3層)+因果性の実測 | 約8秒 | RF=2で0.817 / RF=8で1.000 |
| 11_logic_reservoir.py | 512bitの凍結ランダム論理回路+readoutのみ学習(parity / 遅延再生) | 約7秒 | delay recall 全て1.000 / parity k=3で0.822 |
| 12_logic_ladder_gpt.py | L0(fp)→L1(射影ゲート化)→L2(XNOR)→L3-C(MUX)の論理化の梯子 | 約90秒 | L0 1.000 / L2 1.000 / L3-C 0.990 |

## 視覚(13〜16)

| サンプル | 内容 | 実行時間 | 期待結果 |
|---|---|---|---|
| 13_logic_unet.py | 論理ゲートU-Netで密予測(ORプーリング+nearest+Concat skip) | 約46秒 | 前景IoU 0.568(恒等ベースライン0.456) |
| 14_logic_resnet.py | OR残差+pass-through初期化で厳密な恒等写像、その代償の飽和 | 約120秒 | max\|y-x\| = 0.0 / hard acc XOR 0.361 > OR 0.299 |
| 15_logic_mixer.py | token-mixing と channel-mixing の論理ゲート版MLP-Mixer | 約110秒 | hard acc 0.658 / token摂動の伝播 1→2 |
| 16_dwn.py | ThermometerEncoder+n入力LUTのDWN(離散化ギャップなし) | 約80秒 | test acc 0.958 / train-eval ロジット差 0.0 |

## 生成モデル(17〜19)

| サンプル | 内容 | 実行時間 | 期待結果 |
|---|---|---|---|
| 17_logic_diffusion.py | 前向きXOR・デノイザLUT・逆過程まで全てブール演算の拡散モデル | 約50秒 | acc@t1 0.851 > blank 0.761 |
| 18_logic_vae.py | Bernoulli二値潜在のVAE。196画素→96ビット→復元 | 約80秒 | 画素一致率 0.907 / recon BCE 43.3 |
| 19_logic_gan.py | 乱数ビット列から画像を生成。安定版 vs 未調整版で壊れ方を比較 | 約90秒 | STABLE: G loss 1.46 / NAIVE: G loss 7.22(発散) |

## 応用(20〜23)

| サンプル | 内容 | 実行時間 | 期待結果 |
|---|---|---|---|
| 20_attention_tour.py | XNOR類似度Attentionの4変種(Boolean / MultiQuery / Positional / Pointer) | 約64秒 | 3変種はhard acc 1.000 / Pointer 0.835 |
| 21_detection.py | LutConv2d+OrPooling2dのCenterNet(クラス非依存・合成Multi-MNIST) | 約72秒 | mAP@0.5 = 0.300 / 総ゲート数 679 |
| 22_binarybrain.py | BinaryBrain互換API(RealToBinary/DifferentiableLutBlock/BinaryToReal) | 約90〜165秒 | eval acc 0.803、丸めると0.211(離散化ギャップ) |
| 23_textcnn.py | 論理ゲートTextCNNでバイグラムXOR分類。フィルタがn-gram検出器に育つ | 約25秒 | soft/hard acc 1.000(ユニグラム上限 0.535) |

> 実行時間は MacBook / CPU 実行の実測値(単独実行時)。負荷状況で前後する。

## 行動認識とデプロイ(27〜33)

学習した回路を、依存ゼロの C99・マイコン・iPhone・ブラウザへ運ぶサンプル。
27〜29 は PyTorch の hard eval と C 推論の**整数一致**を `assert` で自己検証する(C コンパイラ `cc` / `clang` / `gcc` が要る)。
30・31 は実機/シミュレータが要るため、自己検証はホスト側のテストとドキュメントの手順で行う。

| サンプル | 内容 | 実行コマンド | 期待結果 |
|---|---|---|---|
| 27_har.py | UCI-HAR(スマホの加速度・ジャイロ)を温度計符号化して学習し、IR → バンドル → C99 へ書き出す。初回は UCI-HAR を約 60MB ダウンロードする | `uv run python examples/27_har.py` | hard acc 0.80 以上 / C 推論と PyTorch が 2,947 窓 × 6 クラスで完全一致 |
| 29_deploy_mnist.py | MNIST 20×20 を 4000 ゲートで学習し、IR → バンドル → C99 → 実行ファイルまでを 1 本で通す(デプロイキット) | `uv run python examples/29_deploy_mnist.py` | hard acc 0.85 前後(assertは>0.80) / C 推論と PyTorch が 1000 枚 × 10 クラスで完全一致 |
| [30_deploy_esp32/](30_deploy_esp32/) | バンドル → C99 → ESP32 実機(ESP-IDF)で推論し、精度・レイテンシ・メモリを UART に出す。同じ `main.c` はホストでもビルドでき、実機⇔ホストの整数一致を確かめられる | `30_deploy_esp32/README.md`参照(`gen_model.py` → `idf.py build flash`)。`gen_model.py --dense-flat {auto,on,off}` で回路の平坦化を切り替え | 行動認識・MNIST の全 50 件がホストと整数一致(要 ESP32 実機)。[ガイド](https://tsukushi-gate.github.io/Griffith/guides/deploy-esp32/) |
| [31_deploy_ios/](31_deploy_ios/) | 生成 C をそのまま Swift(SwiftUI)アプリに組み込み、iPhone / シミュレータで推論結果とレイテンシを表示する。複数モデルを 1 アプリに同梱できる | `31_deploy_ios/README.md`参照(`gen_model.py` → Xcode でビルド)。`gen_model.py --dense-flat {auto,on,off}` あり | 5 モデルがシミュレータで動き、出力が `ir/simulate.py` と整数一致(要 macOS + Xcode。実機の速度は未計測) |
| [32_deploy_web/](32_deploy_web/) | バンドルを `codegen-web` でブラウザ向けにし、自前の最小ページから `mdl_web.js` の `loadBundle` で推論する。engine(auto / js / wasm)を切り替えられる | `make web-runtime`(初回のみ)→ `uv run python examples/32_deploy_web/gen.py` → `python -m http.server 8080 -d out_examples/deploy_web` | `verify-web` が js / wasm の両方で一致 / ブラウザで予測クラスと µs/推論が表示される(要 node/npm) |
| 33_bundle_python.py | CLI を使わず Python API だけで XOR のバンドルを作り(`apply_compress` → `build_bundle`)、Python 推論と C 推論を突き合わせる。CLI の既定圧縮を Python で再現する書き方の実例 | `uv run python examples/33_bundle_python.py` | 4 入力すべてで Python 推論 = C 推論 = XOR の正解 |

## より大きな実験

- MNIST/CIFAR訓練: `uv run python experiments/train.py --dataset mnist -e 20`(詳細はREADME)
- クラウドGPU(Modal): `uv run modal run scripts/modal_app.py --script train.py --args "..."`(docs/guides/modal.md)
- 論文再現テスト: `uv run pytest -m paper`(分単位。Modal推奨)
