# `experiments/` — 研究用スクリプト集 / Research scripts

**日本語** — このディレクトリは Griffith（`griffith_logic`）の**研究用スクリプト**を集めた場所です。
ライブラリ本体ではありません。**API 安定性は保証しません**（引数・出力形式・既定値は予告なく変わります）。
再利用したい処理があれば、ここからコピーして自分のコードに取り込んでください。

**English** — This directory holds **research scripts** for Griffith (`griffith_logic`).
It is *not* part of the library's public API: flags, defaults, and output formats change without notice.
Copy what you need rather than importing from here.

---

## まずここから / Start here

初めて触る場合は、この順に `--help` を眺めて 1 本ずつ動かすのがおすすめです。

| # | スクリプト | 何をするか |
|---|---|---|
| 1 | `recipe_dlgn_repro.py` | difflogic（Petersen 2022）公式レシピの MNIST 再現。ライブラリが論文どおり動くことの確認 |
| 2 | `train.py` | 汎用の DLGN 学習エントリ（MNIST/CIFAR など、密結合 LUT 網） |
| 3 | `train_cnn.py` | 畳み込み版（`LutConv2d` + `OrPooling2d`）の学習 |
| 4 | `distill_xor_mlp_to_1lut.py` | 最小の蒸留実験。MLP 教師 → 1 個の LUT。数十秒で終わり、KD の効き方が読める |
| 5 | `convert_transcribe_mnist.py` | 事前学習済み float MLP を training-free で LUT 回路（IR）へ転写 |
| 6 | `train_logic_gpt.py` | 論理ゲートによる char-level 言語モデル（Tiny Shakespeare） |
| 7 | `attention_ladder/` | attention を段階的に論理回路化するアブレーション梯子（`REPORT.md` が読みどころ） |

```bash
uv run python experiments/recipe_dlgn_repro.py --help
uv run python experiments/train.py --dataset mnist -e 20
```

### 4×4オセロで自己対戦学習と全探索評価を試す

手番側と相手側の石をそれぞれ16ビットで表し、Griffith の `LutLayer`（256ゲート）から方策と価値を予測します。
AlphaZero 形式の自己対戦では MCTS の訪問割合と終局の勝敗を教師にします。学習・対戦・評価はすべて CPU 上のソフトゲートで実行します。

リポジトリのルートで以下を実行すると、40局学習し、全探索で求めた勝敗とモデルの着手を照合できます。

```sh
uv sync --locked --extra cpu --extra dev
uv run --no-sync python -m experiments.train_othello --output checkpoints/othello/demo.pt
uv run --no-sync python -m experiments.evaluate_othello_exact --checkpoint checkpoints/othello/demo.pt
```

全探索は初期盤面から到達できる58,533局面を手番側の視点で数え、終局の石差を逆算します。JSON の `initial_black_margin` は −8（完全プレイで白の8石勝ち）です。
`outcome_depends_on_move` は、どの手を選ぶかで勝ち・引き分け・負けが変わる7,912局面での着手を集計します。例えば `win_to_loss` は勝てる局面から負けになる手を選んだ回数です。`outcome_preserved_fraction` はこの7,912局面のうち最善の勝敗を保った割合です。

400局版と4,000局版を比較するには、同じ学習の途中と最後でモデルを保存します。以下はシード0の例です。6シードの比較では `seed` を0〜5に変えて繰り返します。

```sh
seed=0
model="checkpoints/othello/seed_${seed}"
uv run --no-sync python -m experiments.train_othello --seed "$seed" --iterations 500 \
  --games 8 --updates 32 --simulations 32 --snapshot-iteration 50 --output "$model.pt"
uv run --no-sync python -m experiments.evaluate_othello --seed "$seed" --simulations 32 \
  --checkpoint "$model.pt" --opponent "${model}_iter_50.pt"
for weights in "${model}_iter_50.pt" "$model.pt"; do
  uv run --no-sync python -m experiments.evaluate_othello_exact --simulations 32 --checkpoint "$weights"
done
```

対戦結果の `W/D/L` は `--checkpoint` 側から見た勝ち・引き分け・負けの順です。
`standard` は初期盤面から先後交代で2局、`four_plies` は4手後の異なる120局面から先後交代で240局です。対戦・全探索評価では探索ノイズを切り、訪問回数が最多の手を選びます。

## ディレクトリ地図 / Map

| 接頭辞・場所 | 中身 |
|---|---|
| `train_*.py` | 学習スクリプト（分類・CNN・GPT・seq mixer・検出・セグメンテーション・AP 法など）。最大のグループ |
| `convert_*.py` | 事前学習モデル → 論理ゲート NN への変換（転写 transcribe / 二値化 lowering / 回復学習 recover） |
| `distill_*.py` | 知識蒸留（KD）実験。教師の知識が論理回路へ運ばれるかを測る |
| `verify_*.py` | 容量・表現力の検証（学習が失敗なのか容量不足なのかの切り分け） |
| `sat_*.py` | SAT ソルバによる最小回路サイズ・容量マップの解析 |
| `ap_*.py` | AP 法（非バックプロパゲーション学習則）のトレーナ・計測 |
| `mono_*.py`, `monotonicity.py` | 単調性まわりの解析 |
| `bench_*.py` | カーネル・演算のマイクロベンチマーク |
| `mixer1_*`, `mixer2_*`, `tsmix1*` | 系列 mixer 系キャンペーンの投入・集計・作図 |
| `data/` | 実験専用のデータセット定義（連想想起・文字コーパス・検出データ等）。汎用のものは `griffith_logic/datasets/` にあります |
| `attention_ladder/` | attention 論理回路化のアブレーション梯子（後述） |

補助モジュール: `wandb_utils.py`（wandb ラッパ）、`checkpoint_utils.py`、`run_utils.py`、`env_utils.py`。

## 実行方法 / How to run

```bash
# セットアップ（CPU 環境）
uv sync --extra dev --extra cpu

# まず --help を読む（このディレクトリの共通作法）
uv run python experiments/<script>.py --help

# レシピ経由（configs/recipes/*.yaml）
uv run griffith launch configs/recipes/dlgn_mnist_quick.yaml --local
```

クラウド GPU で同じスクリプトを無改造で動かすには `scripts/modal_app.py` を使います。

```bash
uv run modal run scripts/modal_app.py --script train.py --args "--dataset mnist -e 20"

# 長時間ジョブは必ず --detach と --background をセットで
uv run modal run --detach scripts/modal_app.py --background \
    --gpu H100 --script train_cnn.py --args "--dataset cifar10 -k 64 -e 100"
```

### 環境変数 / Environment variables

| 変数 | 用途 |
|---|---|
| `MD_MODAL_VOLUME` | Modal の永続 Volume 名（既定 `griffith-workspace`）。アカウント共有時の衝突回避に |
| `WANDB_API_KEY` | wandb を使う場合のみ必要 |
| `WANDB_ENTITY` / `WANDB_PROJECT` | wandb の送信先 |
| `WANDB_MODE=offline` | wandb をローカル記録だけに留める |

`.env` を置けば `env_utils.load_dotenv()` 経由で読まれます。

### 出力先 / Outputs

| ディレクトリ | 中身 |
|---|---|
| `results/` | JSON / CSV のスコア表・集計・図 |
| `outputs/` | 中間生成物（教師活性のキャッシュ等） |
| `checkpoints/` | モデルのチェックポイント |

いずれもリポジトリルートからの相対パスです。スクリプトは基本的にカレントディレクトリを
リポジトリルートとして動く前提なので、`uv run python experiments/xxx.py` の形でルートから実行してください。

### wandb は任意 / Weights & Biases is optional

wandb へのログは**任意**です。`configs/recipes/*.yaml` の `run.wandb` は既定で `false` にしてあり、
API キーが無くても全スクリプトが動きます。使う場合は `wandb: true`（または各スクリプトの
`--wandb` 相当のフラグ）に加えて `WANDB_API_KEY` を設定してください。
送信せずにローカル記録だけ残したいときは `WANDB_MODE=offline` を使います。

## `attention_ladder/` の再現ガイド

Transformer の attention（射影・スコア・正規化）を **1 段 = 1 変数**で段階的に論理回路化し、
どこまで精度が保たれるかを測るアブレーション梯子です。

- `REPORT.md` — 実験レポート（結論はここ）
- `RESULTS.md` — 生の結果表
- `ENV_NOTES.md` — 使った API と自作アダプタの設計メモ
- `results/<段>_<タスク>/summary.csv` — **実測値**（3 シード分の精度・perplexity・エントロピー等）
- `results/circuit_cost.md` — 回路コスト比較。**射影の LUT 数は `export_ir → estimate_resources` の実測値ですが、
  比較器数・MUX 数・加算器数・論理深さは解析式による推定値であり、実合成の結果ではありません**
- `src/` — 梯子用アダプタ（`ladder_attention.py` / `selectors.py` / `circuit_cost.py` ほか）
- `configs/` — 各段の設定

再現は `src/` のスクリプトを `--help` から辿ってください。sMNIST はローカル（MPS/CPU）で、
char-LM は GPU（`scripts/modal_app.py` 経由）で回した結果です。

## データセットとライセンス / Datasets and licenses

このディレクトリのスクリプトが自動ダウンロードするデータセットと、教師として使う公開モデルは
**それぞれの提供元のライセンス・利用規約に従います**。Griffith のライセンスはこれらを含みません。
利用前に各提供元の条件を必ず確認してください。

| データセット / モデル | 提供元・注記 |
|---|---|
| MNIST / Fashion-MNIST | torchvision 経由で取得。原典の配布条件に従う |
| CIFAR-10 / CIFAR-100 | torchvision 経由で取得。研究用途向けの配布条件に従う |
| Oxford-IIIT Pet / Pascal VOC | torchvision 経由。VOC は研究目的限定の条件を含む |
| Tiny Shakespeare | パブリックドメインのテキスト（char-level LM 用） |
| SST-2 (GLUE) | Hugging Face `datasets` 経由。GLUE の利用条件に従う |
| WMT14 En-De | Hugging Face `datasets` 経由。コーパス各部の条件に従う |
| WRIME | 日本語感情分析データセット。提供元の利用規約に従う（要確認） |
| Hugging Face 教師モデル（`hf:` 指定） | 各モデルカードのライセンスに従う（例: CIFAR-10 ResNet-18） |

## 公開版に含まれないもの / Not included in the public tree

以下は社内運用に固有のため、公開リポジトリには含めていません。スクリプトの説明文に
参照が残っていた場合は読み飛ばしてください。

- 社内 GPU マシン向けの SSH 投入スクリプト（`scripts/remote/`）
- RunPod 運用基盤（`scripts/runpod*`）
- 社内の進捗・調査ドキュメント（進捗管理、論文サーベイ、実験レポート本文）
- ビジュアルエディタ本体に依存するテスト・生成物
