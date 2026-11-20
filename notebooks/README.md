# notebooks — ドキュメントのコードをそのまま実行できるノートブック

`docs-site` のチュートリアル各ページと**同じ内容**の Jupyter ノートブック。
ページ側は実行済みの出力・図が埋め込まれた読み物で、こちらは**未実行**(出力なし)。
読者が自分の環境で上から順に実行するためのものである。

> **自動生成物**: このディレクトリの `*.ipynb` は `scripts/build_tutorials.py` が
> `examples/*.py` から生成する。**直接編集しても次の再生成で上書きされる**ので、
> 直したいときは元の `examples/*.py` を編集して再生成すること。

| ノートブック | 対応ページ | 内容 |
|---|---|---|
| `01_xor.ipynb` | [付録: XORで仕組みを覗く](https://github.com/tsukushi-gate/Griffith/blob/main/docs-site/src/content/docs/tutorials/xor.md) | 16ゲートで XOR。soft→hard の崩壊を全ノード図で観察 |
| `02_hybrid_llvit.ipynb` | 2. ハイブリッドNNを作る | Transformer の MLP だけを LUT へ置換 |
| `03_learning_rules.ipynb` | 6. 学習則(DFA/AP) | BP / DFA / AP法 の loss 曲線比較 |
| `04_export_hdl.ipynb` | 3. HDLへエクスポート | 学習済みモデル → IR → Verilog + 照合 |
| `05_cnn_export_hdl.ipynb` | 5. CNNのHDLエクスポート | LutConv2d を unroll して Verilog / VHDL へ |
| `06_mnist.ipynb` | 1. MNISTを学習する | 4000ゲートで MNIST 20x20(図3枚) |
| `07_full_pipeline.ipynb` | 7. 通しチュートリアル | 学習 → IR → Verilog → FPGA までの通し |

### 系列・言語

| ノートブック | 対応ページ | 内容 |
|---|---|---|
| `07_logic_gpt.ipynb` | 論理ゲートGPT | 全部論理ゲートの decoder-only Transformer で文字レベル言語モデル |
| `08_rddlgn_seq2seq.ipynb` | seq2seq(RDDLGN) | 固定長ボトルネックで copy / reverse を解く |
| `09_logic_lstm.ipynb` | LSTM / GRU | 忘却ゲート=MUX、出力ゲート=ビットAND |
| `10_logic_wavenet.ipynb` | WaveNet(因果conv) | 受容野の ablation と因果性の実測 |
| `11_logic_reservoir.ipynb` | リザバー | 凍結ランダム論理回路 + readout だけ学習 |
| `12_logic_ladder_gpt.ipynb` | 論理化の梯子 | L0(fp)→L3-C(MUX)まで attention を1段ずつ論理化 |

### 視覚

| ノートブック | 対応ページ | 内容 |
|---|---|---|
| `13_logic_unet.ipynb` | U-Net(密予測) | ORプーリング + nearest + Concat skip で画素ごとに答える |
| `14_logic_resnet.ipynb` | ResNet(厳密な恒等) | OR残差 + pass-through 初期化で max\|y-x\| = 0.0 |
| `15_logic_mixer.ipynb` | Mixer | token-mixing / channel-mixing で attention なしに混ぜる |
| `16_dwn.ipynb` | DWN(温度計符号化) | 離散化ギャップがゼロになるパラメータ化 |

### 生成モデル

| ノートブック | 対応ページ | 内容 |
|---|---|---|
| `17_logic_diffusion.ipynb` | 拡散モデル | 前向きXOR・デノイザLUT・逆過程まで全てブール演算 |
| `18_logic_vae.ipynb` | VAE(二値潜在) | 196画素を96ビットへ畳んで復元する |
| `19_logic_gan.ipynb` | GAN | 乱数ビット列から生成。安定版 vs 未調整版の壊れ方 |

### 応用

| ノートブック | 対応ページ | 内容 |
|---|---|---|
| `20_attention_tour.ipynb` | Attentionツアー | XNOR類似度Attentionの4変種を小タスクで比較 |
| `21_detection.ipynb` | 物体検出 | LutConv2d + OrPooling2d の CenterNet |
| `22_binarybrain.ipynb` | BinaryBrain互換API | FPGA向けLUT網の学習と離散化ギャップの測定 |
| `23_textcnn.ipynb` | TextCNN | バイグラムXOR分類。フィルタが n-gram 検出器へ育つ |

> `21_detection.ipynb` と `23_textcnn.ipynb` は `experiments/`(パッケージ外)を import するため、
> **リポジトリをクローンした環境でのみ動く**(Colab のインストールセルだけでは足りない)。

先頭のセルは**環境判定インストールセル**で、`griffith_logic` が import できる環境
(ローカル)では何もしない。Colab のときだけ `pip install` を走らせる。

## ローカルで実行する

```bash
uv sync --extra dev --extra cpu --extra figures --group docs   # torch + matplotlib + jupyter
uv run --group docs jupyter lab notebooks/
```

`--extra figures`(matplotlib)は回路図・学習曲線の描画に必要。
`--group docs` に jupyter 一式(jupytext / nbconvert / ipykernel / jupyterlab)が入る。

ノートブックを開かずに一括実行して確認したいだけなら:

```bash
uv run --group docs jupyter nbconvert --to notebook --execute --stdout notebooks/01_xor.ipynb > /dev/null
```

## Google Colab で開く

### リポジトリが公開されたあと

各チュートリアルページの「▶ Colabで開く」バッジから1クリックで開ける
(バッジは公開後に有効化される。それまでは URL を `data-colab-url` 属性に保持している)。
インストールセルがそのまま走る。

### リポジトリが非公開の間

Colab から GitHub の認証を通す必要がある。

1. [colab.research.google.com](https://colab.research.google.com/) を開く
2. **[ファイル] → [ノートブックを開く] → [GitHub] タブ**
3. **「プライベート リポジトリを含める」にチェック**を入れ、GitHub アカウントを認可する
4. リポジトリに `tsukushi-gate/Griffith`、ブランチに `feature/docs-site` を選ぶ
5. 一覧から `notebooks/0N_*.ipynb` を選んで開く
6. 先頭のインストールセルを実行する(`git+https://...` からの取得にも認証が要るため、
   Colab 側の GitHub 連携を済ませてから実行すること)

インストールセルが参照するブランチは `scripts/build_tutorials.py` の `INSTALL_REF` で定義している。
main へマージしたら `INSTALL_REF = "main"` に変えて再生成する。

## 再生成とドリフト検査

```bash
uv run --group docs python scripts/build_tutorials.py --no-execute   # ノートブックだけ(実行なし・数秒)
uv run --group docs python scripts/build_tutorials.py                # ページも含めて全部(実行あり)
```

`--no-execute` は実行を伴わないので出力は決定的(セル id も連番に固定している)。
CI(`.github/workflows/docs.yml` の `tutorials` ジョブ)は再生成後に
`git diff --exit-code -- notebooks` でドリフトを検出する。
