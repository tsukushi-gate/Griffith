# Griffith

**論理ゲートでできたニューラルネットを PyTorch で学習し、Verilog / VHDL へコンパイルする。**

[![PyPI version](https://img.shields.io/pypi/v/griffith-logic.svg)](https://pypi.org/project/griffith-logic/)
[![PyPI downloads](https://static.pepy.tech/badge/griffith-logic/month)](https://pepy.tech/project/griffith-logic)
[![CI](https://github.com/tsukushi-gate/Griffith/actions/workflows/ci.yml/badge.svg)](https://github.com/tsukushi-gate/Griffith/actions/workflows/ci.yml)
[![Docs](https://github.com/tsukushi-gate/Griffith/actions/workflows/docs.yml/badge.svg)](https://tsukushi-gate.github.io/Griffith/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Named after Griffith Observatory, whose mission is to bring astronomy to everyone — we want
differentiable logic to be just as accessible.

(グリフィス天文台の「天文学をすべての人へ」という理念にちなむ。微分可能な論理も同じように
誰でも触れるものにしたい、という願いを名前に込めている。)

## Griffith とは

Griffith は**微分可能論理ゲートNN**のための PyTorch ライブラリである。各ニューロンは重み付き和
ではなく **N入力のルックアップテーブル(LUT)** で、学習時は表を連続的な混合へ緩和して勾配を流し、
`model.eval()` で `argmax` により純粋な Boolean 回路へ崩壊させる。崩壊しても精度は落ちない。

- **N入力LUT(N = 2, 4, 6)** — [difflogic](https://github.com/Felix-Petersen/difflogic) の2入力固定
  ゲートを一般化し、FPGA ネイティブな LUT6 までを1つの層で表現する
- **学習 → IR → ハードウェア** — 離散化した回路を JSON の **IR(真実源)** へ抽出し、合成可能な
  Verilog / VHDL を生成する
- **学習則を選べる** — BP / **AP法(Akiho Propagation)** / DFA / EGGROLL をレイヤ単位で注釈でき、
  1つのネットワークの中で混在させられる
- **論理ゲートだけで組んだモデル動物園** — 画像分類・系列/言語・生成・検出・NLP まで

## インストール

```bash
pip install griffith-logic
```

**Python 3.11 以上**と PyTorch が必要。`pip install griffith-logic` は環境が解決する PyTorch
ホイールをそのまま引く。CUDA 機では先に対応する PyTorch ビルドを入れてから
([pytorch.org](https://pytorch.org/get-started/locally/))、Griffith を入れること。
CPU と Apple Silicon(MPS)はそのまま動く。**CUDA は必須ではなく**、任意の CUDA カーネルが
使えない環境では自動的に純 PyTorch 実装へフォールバックする。

Griffith 自体を開発する場合(および `examples/` を動かす場合)は [uv](https://docs.astral.sh/uv/)
を使う。[開発](#開発) を参照。

## 30秒で試す

XOR を学習し、**離散化後**も完璧であることを確認して、Verilog を書き出す。

```python
import torch
import torch.nn.functional as F

import griffith_logic as gl
from griffith_logic.export import export_ir
from griffith_logic.export.verilog import write_verilog

torch.manual_seed(0)

# XOR: 入力2ビットなので、4パターンでデータセットが全部
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 0])

# LutLayer 2段(2入力ゲート)+ GroupSum の読み出し
model = torch.nn.Sequential(
    gl.LutLayer(2, 16, n_inputs=2),
    gl.LutLayer(16, 16, n_inputs=2),
    gl.GroupSum(k=2, tau=2.0),
)

opt = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(500):
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()

# eval() で各ゲートが1つの論理関数に確定する(argmax)
model.eval()
with torch.no_grad():
    hard_acc = (model(x).argmax(-1) == y).float().mean().item()
print(f"hard accuracy: {hard_acc:.2f}")   # -> 1.00

# 離散化した回路を IR へ固定し、合成可能な Verilog を書き出す
ir = export_ir(model, input_width=2, name="xor")
write_verilog(ir, "xor.v")
```

見るべき数字は常に2つ。**soft**(勾配が流れる連続混合)と **hard**(実際に FPGA へ焼ける
Boolean 回路)であり、**結果として意味があるのは後者だけ**である。

同じ書き方を MNIST 規模へ広げると次のようになる。

```python
import torch
import griffith_logic as gl

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    gl.LutLayer(in_dim=784, out_dim=16000, n_inputs=4),
    gl.LutLayer(in_dim=16000, out_dim=16000, n_inputs=4),
    gl.LutLayer(in_dim=16000, out_dim=16000, n_inputs=4),
    gl.GroupSum(k=10, tau=30.0),
)
y = model(torch.rand(32, 784))   # (32, 10) クラスロジット
```

## モデル動物園

同梱するアーキテクチャからの抜粋。数値は本走の **hard**(離散化後)値である。チュートリアルは
CPU で数分の縮小版なので、そちらの数値とは一致しない。条件・シード・注意点は
[モデル動物園](https://tsukushi-gate.github.io/Griffith/reference/model-zoo/) と
[ベンチマーク一覧](https://tsukushi-gate.github.io/Griffith/reference/benchmarks/) にある。

| モデル | 解くタスク | 代表精度(hard) | ドキュメント |
| --- | --- | --- | --- |
| DLGN(基準線) | 手書き数字の分類 | MNIST acc **0.9784**(48,000ゲート) | [MNIST](https://tsukushi-gate.github.io/Griffith/tutorials/mnist/) |
| Conv-DLGN | LUT木の畳み込みによるカラー画像分類 | CIFAR-10 acc **0.5808** | [ベンチマーク](https://tsukushi-gate.github.io/Griffith/reference/benchmarks/) |
| BinaryBrainModel | 6入力LUT + フレーム変調で数字分類 | MNIST acc **0.9483**(14,516ゲート) | [BinaryBrain](https://tsukushi-gate.github.io/Griffith/tutorials/binarybrain/) |
| DWNClassifier | 配線そのものを学習する(トイ論理関数) | learnable 配線で **100%** 到達 | [DWN](https://tsukushi-gate.github.io/Griffith/tutorials/dwn/) |
| LogicGPT | 次の1文字を予測する文字レベル言語モデル | Tiny Shakespeare val ppl **13.31** | [LogicGPT](https://tsukushi-gate.github.io/Griffith/tutorials/logic-gpt/) |
| LogicLadderGPT | Transformer のどこまで回路化できるかの梯子 | L1段 ppl **19.24 ± 0.94**(3シード) | [梯子](https://tsukushi-gate.github.io/Griffith/tutorials/logic-ladder/) |
| RDDLGN | attention を持たない文字列→文字列変換 | data-free 蒸留 char-LM BPC **2.71** | [RDDLGN](https://tsukushi-gate.github.io/Griffith/tutorials/rddlgn-seq2seq/) |
| LogicLSTMCell / LogicGRUCell | 遅延コピー(T=8)= 記憶が要る系列課題 | acc **1.00**(全シード 200step 以内) | [Logic LSTM](https://tsukushi-gate.github.io/Griffith/tutorials/logic-lstm/) |
| LogicBinaryDiffusion | 二値画像の離散拡散(ノイズ付加は XOR) | binary MNIST acc@t1 **0.886** | [拡散](https://tsukushi-gate.github.io/Griffith/tutorials/logic-diffusion/) |
| LogicVAE | 潜在が Bernoulli ビット列の VAE | binarized MNIST **193.6 nat/枚** | [VAE](https://tsukushi-gate.github.io/Griffith/tutorials/logic-vae/) |
| CenterNetLUT | Multi-MNIST の物体検出 | mAP **0.5425**(13,944 params) | [検出](https://tsukushi-gate.github.io/Griffith/tutorials/detection/) |
| TextCNNModel | 日本語の感情極性2値分類(WRIME) | acc **82%** | [TextCNN](https://tsukushi-gate.github.io/Griffith/tutorials/textcnn/) |

## difflogic との違い

「離散的なゲート選択を微分可能な混合へ緩和する」という difflogic の核は受け継ぎ、次の4点を
変えている。帰属とライセンスの詳細は
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照(CUDA カーネルの一部は
difflogic 由来・MIT)。

1. **2入力ゲートではなく N入力LUT。** difflogic は2入力の16種の Boolean 関数上の分布を学習する。
   Griffith は同じ構成を N入力の真理値表へ一般化した(N = 2, 4, 6)ので、「difflogic 互換」から
   「FPGA LUT6 ネイティブ」までを1つの層型でカバーできる
2. **CUDA は必須ではなく任意。** 参照実装は素の PyTorch で、CPU と Apple Silicon(MPS)で動く。
   JIT CUDA カーネルはあくまで高速化であり、使えなければ自動でフォールバックする
3. **学習とハードウェアの間に IR を置く。** 学習結果は JSON の IR へ固定され、検証・整数ビット演算
   でのシミュレーション・Verilog / VHDL への lowering を経る。テストは3段の一致
   (PyTorch hard eval ⇔ IR simulate ⇔ HDL シミュレーション)を保証する
4. **5種のパラメータ化と4種の学習則。** DLGN 方式の softmax に加え、BinaryBrain 方式の独立テーブル、
   LDDLGN 方式の入力ごとパラメータ化(パラメータ約10分の1)を持ち、backprop は選べる4つの
   信用割当のうちの1つでしかない

## 学習則

`with_learning()` でレイヤ単位に信用割当の方式を指定し、`HybridTrainer` で学習する。

| 学習則 | 誤差の流し方 | 微分は要るか | 使いどころ |
| --- | --- | --- | --- |
| `bp` | backprop。層ごとに変換しながら逆向きに伝播 | 要る | 既定。精度が最優先のとき |
| `ap` | **AP法(Akiho Propagation、本ライブラリの backprop 不要則)** — 出力誤差を到達可能な全 LUT へ同じ値のまま配る | 不要 | backprop 非依存を実証したいとき。`parameterization='independent'` と直列構成が前提 |
| `dfa` | Direct Feedback Alignment。出力誤差を固定ランダム行列で各層へ直送 | 局所的にのみ要る | 層間の逆伝播を切りたいとき |
| `eggroll` | 進化戦略。低ランク摂動を撒いて fitness で選択 | 不要 | 勾配が定義できない/信頼できない目的関数のとき |

```python
from griffith_logic import HybridTrainer, LearningSpec, LutLayer, with_learning

fc1 = with_learning(
    LutLayer(784, 8000, parameterization="independent"),
    LearningSpec(rule="ap"),
)
```

信用割当の対比図と詳細:
[学習則](https://tsukushi-gate.github.io/Griffith/explanation/learning-rules/)。

## ハイブリッドNN

ネットワーク全体を論理ゲートにする必要はない。置き換えたい部分だけを置き換えられる。

```python
from griffith_logic import LogicSpec, replace_mlp_with_lut

hybrid, replaced = replace_mlp_with_lut(vit, spec=LogicSpec(n_inputs=2, depth=2))
```

## ドキュメント

ドキュメント全体: **<https://tsukushi-gate.github.io/Griffith/>**

- [インストール](https://tsukushi-gate.github.io/Griffith/getting-started/installation/) ·
  [クイックスタート](https://tsukushi-gate.github.io/Griffith/getting-started/quickstart/) ·
  [基本概念](https://tsukushi-gate.github.io/Griffith/getting-started/concepts/)
- [チュートリアル](https://tsukushi-gate.github.io/Griffith/tutorials/) — `examples/` のスクリプトから
  生成した、手元で追試できる縮小版
- [DLGNとは](https://tsukushi-gate.github.io/Griffith/explanation/dlgn/) ·
  [LutLayer の内部](https://tsukushi-gate.github.io/Griffith/explanation/lut-layer/) ·
  [アーキテクチャ](https://tsukushi-gate.github.io/Griffith/explanation/architecture/)
- [FPGAに焼く](https://tsukushi-gate.github.io/Griffith/guides/fpga/) ·
  [HDLエクスポート](https://tsukushi-gate.github.io/Griffith/guides/hdl-export/) ·
  [IR](https://tsukushi-gate.github.io/Griffith/guides/ir/)
- [APIリファレンス](https://tsukushi-gate.github.io/Griffith/reference/api/) ·
  [CLIリファレンス](https://tsukushi-gate.github.io/Griffith/reference/cli/) ·
  [IRスキーマ](https://tsukushi-gate.github.io/Griffith/reference/ir-schema/)

学習・評価・HDLエクスポートの入口は `griffith` コマンドに統一されている
(`md` は旧名の後方互換エイリアス)。

## ツール

`tools/` 配下はすべてリポジトリに同梱している(PyPI パッケージには入らないので、使うには
リポジトリを clone する)。Python 側の依存は `uv sync` の extras、フロントは各ディレクトリの
`package.json` で入れる。

| ツール | 何ができるか | 起動 |
| --- | --- | --- |
| `tools/editor` | ビジュアルエディタ。IR を契約に、グラフ編集・学習のライブ監視・評価・HDL 出力を 1 画面で行う(FastAPI + React)。 | `uv sync --extra editor` の後 `make editor`(backend :8765 + frontend :5173) |
| `tools/webbench` | ブラウザ推論比較ツール。エクスポートした回路を WASM/WebGPU で走らせ、MediaPipe / TFJS とレイテンシ・ペイロードを比較する。 | `make webbench`(または `cd tools/webbench && npm ci && npm run dev`) |
| `tools/visualizer` | DDLGN / CDDLGN / RDDLGN をゲート単位で描く WebGL ビューア。活性をライブ配信できる。 | `uv sync --extra visualizer` の後 `cd tools/visualizer/frontend && npm ci && npm run dev` |
| `tools/nas` | 構造探索(genome 変異 + NSGA-II)。エディタの `POST /api/nas/start` から回すほか、Python パッケージとしても使える。 | エディタから利用、または `from tools.nas import search` |
| `tools/paper-tree` | 関連研究の読み筋ツリー。`data/papers/*.md` の frontmatter を論文とその後続関係のグラフとして描く。 | `cd tools/paper-tree && npm ci && npm run dev` |
| `tools/mdl-fpga` | Tang Nano 20K / DE0-CV 向け FPGA 支援コマンド(`doctor` / `convert` / `build` / `flash` / `hw-check`)。**統合作業中**。 | [FPGAガイド](https://tsukushi-gate.github.io/Griffith/guides/fpga/)を参照 |

## 開発

```bash
git clone https://github.com/tsukushi-gate/Griffith.git
cd Griffith
uv sync --extra dev --extra cpu     # CUDA 機は --extra cu128
uv run python examples/01_xor.py    # XOR → hard accuracy 1.00
make test                           # 高速スイート(常時グリーン基準: 2分以内)
make lint                           # ruff
```

Python コマンドは常に `uv run` を通す(`.venv` を直接 activate しない)。依存は
`uv pip install` ではなく extras + `uv sync` で解決する。テストは L1(数式)→ L2(勾配)→
L3(論文再現)→ L4(収束)→ L5/HDL(等価性)の層構造で、変更の種類ごとにどこまで書くかは
[`CONTRIBUTING.md`](CONTRIBUTING.md) に定めている。

コントリビューションを歓迎する。すべてのコミットに
[Developer Certificate of Origin](DCO) のサインオフ(`git commit -s`)が必要である。
[行動規範](CODE_OF_CONDUCT.md) も併せて読んでほしい。

## セキュリティ

脆弱性は公開 issue ではなく、[`SECURITY.md`](SECURITY.md) の手順で非公開に報告してほしい。

**信頼できないチェックポイント(`.pt`)や IR ファイルを読み込まないこと。** 他の PyTorch
プロジェクトと同様、信頼できないファイルの復元は任意コード実行につながりうる。

## 引用

研究で Griffith を使った場合は次のように引用してほしい([`CITATION.cff`](CITATION.cff) 参照)。

```bibtex
@software{griffith_logic,
  title  = {Griffith: Differentiable Logic Gate Networks for PyTorch},
  author = {Masato Akiho and Mizuki Nomaguchi and Musashi Hadano},
  year   = {2026},
  date   = {},
  note   = {Version 0.1.0},
  url    = {https://github.com/tsukushi-gate/Griffith}
}
```

基礎となった研究も併せて引用してほしい: Petersen, Borgelt, Kuehne, Deussen,
*Deep Differentiable Logic Gate Networks*, NeurIPS 2022
([arXiv:2210.08277](https://arxiv.org/abs/2210.08277))。

## ライセンス

Apache License 2.0([`LICENSE`](LICENSE) と [`NOTICE`](NOTICE) を参照)。第三者由来の素材に関する
著作権表示とライセンス全文は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) に集約している。

## 謝辞

Griffith は次の仕事の上に立っている。

- [difflogic](https://github.com/Felix-Petersen/difflogic)(Felix Petersen ら、NeurIPS 2022)
  — 微分可能論理ゲートNNの定式化と、本ライブラリの CUDA カーネルの派生元となった構造
- [BinaryBrain](https://github.com/ryuz/BinaryBrain)(Ryuji Fuchikami)
  — `independent` パラメータ化の背景にある LUTネットワークの学習手法
- [DWN](https://github.com/alanbacellar/DWN)(Bacellar ら)
  — 微分可能 weightless NN と学習可能な配線
- モデル動物園で再現した各論文の著者。公開されたレシピと数値があったから比較ができた

---

English: [README.md](README.md)
