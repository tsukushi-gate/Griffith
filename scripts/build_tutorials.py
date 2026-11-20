#!/usr/bin/env python
"""examples/*.py を実行済みチュートリアルページへ変換するパイプライン(DOCS-1)。

単一ソースは `examples/*.py`(実行可能な最小サンプル)。このスクリプトは

    examples/NN_*.py
      → py:percent 形式へ変換(一時ファイル)
      → jupytext で .ipynb 化
      → jupyter nbconvert --execute で実行(出力・図を埋め込み)
      → nbconvert --to markdown で Markdown 化
      → Starlight frontmatter + 概要/所要時間/Colabバッジを付与
      → docs-site/src/content/docs/tutorials/*.md
      → notebooks/*.ipynb(**未実行**のクリーンなノートブック)

という一方向の変換を行う。**生成先の .md / .ipynb を手で編集しない**(次回実行で
上書きされる)。文章を直したいときは元の `examples/*.py` のコメント/docstring か、
本スクリプトの `TUTORIALS` にある `intro` を編集する。

**章見出し・散文マーカー**(`insert_chapter_headings`): コードセルの先頭に置いた
`# === 章名 ===` → `## 章名`(右目次に出る)、`# --- 小見出し ---` → `### 小見出し`、
`# > 本文` → コードフェンス外の Markdown 段落、へ後処理で変換する。`---` と `>` は
その example が `# === … ===` を1つ以上持つときだけ有効(ファイル単位のオプトイン)。
規約は docs-site/README.md の「執筆メモ」節が正。

`notebooks/*.ipynb` は「読者が自分で実行する用」なので出力を含めず、先頭に
環境判定インストールセル(`INSTALL_CELL`)を差し込む。実行を伴わないため出力は
決定的で、CI でドリフト検査できる(`--no-execute`)。

使い方:

    uv run --with jupytext --with nbconvert --with ipykernel \
        python scripts/build_tutorials.py            # 全ページ+ノートブックを再生成
    ... python scripts/build_tutorials.py --only xor # slug を指定して1本だけ
    ... python scripts/build_tutorials.py --check    # 実行せず対象一覧を表示
    ... python scripts/build_tutorials.py --no-execute  # ノートブックだけ再生成(実行なし)
    python scripts/build_tutorials.py --gallery-only     # ギャラリー+シンボル索引のみ(依存なし)

`TUTORIALS` からは次の2つも派生する(どちらも実行不要・依存なしで再生成できる):

  - `tutorials/index.mdx` — 例ギャラリー(群ごとのサムネイル付きカード一覧)
  - `docs-site/src/data/example-symbols.json` — 「どの example がどの公開シンボルを
    使っているか」の索引。`scripts/gen_api_docs.py` が API→例の逆リンクに使う
    (gen_api_docs は `collect_symbol_usage()` を直接呼ぶのでJSONの鮮度に依存しない)

依存(jupytext / nbconvert / ipykernel)は pyproject の `[dependency-groups] docs`
に置く。未整備の環境では上記のように `uv run --with` で一時的に入れて実行できる。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "griffith_logic"
EXAMPLES_DIR = ROOT / "examples"
OUT_DIR = ROOT / "docs-site" / "src" / "content" / "docs" / "tutorials"
NOTEBOOKS_DIR = ROOT / "notebooks"
# 例ギャラリー(scikit-learn の Examples 相当)。TUTORIALS から自動生成する。
GALLERY_PATH = OUT_DIR / "index.mdx"
# 「どの example がどの公開シンボルを使っているか」の機械可読な索引。
# scripts/gen_api_docs.py は同じ関数を直接呼ぶ(このJSONは人が読む・外部が使う用)。
SYMBOLS_JSON = ROOT / "docs-site" / "src" / "data" / "example-symbols.json"
GITHUB_BLOB = "https://github.com/tsukushi-gate/Griffith/blob/main"
COLAB_BASE = "https://colab.research.google.com/github/tsukushi-gate/Griffith/blob/main"
# Colab のインストールセルが参照するブランチ(public 化・main マージ後は main へ)
INSTALL_REF = "feature/docs-site"

# 1セルあたりの実行タイムアウト(秒)。ハング検出のための上限で、実行時間の予算ではない。
# 実測(Apple M 系・CPU): 27 本の合計 1,080.8s、最長は logic-resnet 118.8s /
# logic-gpt 115.4s / logic-mixer 109.9s。GitHub Actions の 2 vCPU ランナーは数倍遅いので、
# CI では GRIFFITH_TUTORIAL_CELL_TIMEOUT で引き上げる(.github/workflows/docs.yml)。
EXECUTE_TIMEOUT = int(os.environ.get("GRIFFITH_TUTORIAL_CELL_TIMEOUT", "600"))

# notebooks/*.ipynb の先頭に差し込む環境判定インストールセル。
# ローカル(uv sync 済み)では import が通るので何もしない。Colab のときだけ
# git+https からインストールする。IPython の `%pip` は try 節の中でも変換される。
INSTALL_CELL = '''\
# --- 実行環境のセットアップ(ローカルでは何もしない) -------------------------
# このセルは「griffith_logic が import できるか」だけを見る。
#   ローカル(uv sync 済みの kernel)  → 何もしない
#   Google Colab                      → git+https から pip install
#   素の Jupyter(torch も無い)      → 導入手順を表示
try:
    import griffith_logic  # noqa: F401

    print("griffith-logic は導入済みです(セットアップ不要)")
except ImportError:
    try:
        import google.colab  # noqa: F401

        # ※ リポジトリが非公開の間は Colab 側の GitHub 認証が必要:
        #    [ファイル]→[ノートブックを開く]→[GitHub]タブで
        #    「プライベート リポジトリを含める」にチェックを入れて認証してから開く。
        #    リポジトリ公開後は、この行がそのまま通るようになる。
        %pip install "griffith-logic[cpu] @ git+https://github.com/tsukushi-gate/Griffith.git@{ref}"
    except ImportError:
        print(
            "griffith-logic が見つかりません。ローカルの Jupyter では\\n"
            "  uv sync --extra dev --extra cpu --extra figures --group docs\\n"
            "を実行し、その仮想環境の kernel を選んでください(torch も一緒に入ります)。"
        )
'''.replace("{ref}", INSTALL_REF)

# 生成ノートブックの固定メタデータ。実行環境依存(python の細かいバージョン等)を
# 持ち込むと再生成のたびに差分が出るため、kernelspec だけを最小限で固定する。
NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}


@dataclass
class Tutorial:
    """1チュートリアルページの生成仕様。"""

    slug: str  # 出力ファイル名(拡張子なし)= Starlight の slug 末尾
    title: str  # frontmatter title(サイドバーの表記と揃える)
    description: str  # frontmatter description
    minutes: int  # 読了+実行の目安(分)
    intro: str  # 本文冒頭に置く概要(Markdown)
    # ページ冒頭の統一ブロック(「このページで学ぶこと」/「前提」)。
    # learn は **3点以内**(4点以上は生成時に落とす)。読者が最初の10秒で
    # 「自分が読むべきページか」を判断できるようにするための要約なので、
    # intro の言い換えではなく「読み終えたとき何が分かるか」を書く。
    learn: list[str] = field(default_factory=list)
    # 前提: 先に読むと理解が早いページの Markdown リンク行(空なら「なし」と表示)。
    prereq: list[str] = field(default_factory=list)
    source: str | None = None  # 変換元の examples/*.py(None なら synth を使う)
    synth: str = ""  # 合成チュートリアル用の py:percent ソース
    next_links: list[str] = field(default_factory=list)  # 末尾「次に読む」の Markdown 行
    notebook_name: str | None = None  # notebooks/*.ipynb の名前(既定は source の basename)
    group: str = "basics"  # 例ギャラリーの群(GALLERY_GROUPS のキー)
    # ギャラリーのサムネイル(`{slug}_files/` 配下のファイル名)。
    # 「そのモデルらしい図」を選ぶ ── 生成グリッド・検出ボックス・回路図など。
    # 図番号は再生成でズレうるので、存在しなければ先頭の1枚へフォールバックする。
    thumb: str | None = None

    @property
    def notebook(self) -> str:
        """notebooks/ 配下の出力ファイル名(拡張子込み)。"""
        if self.notebook_name:
            return self.notebook_name
        assert self.source is not None
        return self.source.replace(".py", ".ipynb")


# 例ギャラリーの群。キー → (見出し, 群のリード文)。サイドバーのサブグループと同じ並び。
GALLERY_GROUPS: list[tuple[str, str, str]] = [
    (
        "intro",
        "入門コース",
        "ニューラルネットをやったことがない人のための3本。"
        "PyTorchの基礎 → 小さなMLP → 同じ問題をDLGNへ置き換え、の順に読む。",
    ),
    (
        "basics",
        "基本",
        "まずここから。学習 → 離散回路 → HDL → C という本ライブラリの背骨を通す7本+付録。",
    ),
    (
        "sequence",
        "系列・言語",
        "系列を論理ゲートで扱う。attention・再帰・因果convolution・リザバーを比べる。",
    ),
    (
        "vision",
        "視覚",
        "画像を論理ゲートで扱う。分類から密予測へ、残差・混合・二値化の作法を見る。",
    ),
    (
        "generative",
        "生成モデル",
        "乗算なしで画像を作る。拡散・VAE・GAN を丸ごとブール回路にする。",
    ),
    (
        "applications",
        "応用",
        "検出・FPGA向けLUT網・テキスト分類など、組み上げた回路を実タスクへ。",
    ),
]


# --- 入門コース(examples/24〜26)の導入文 --------------------------------
# NN未経験の読者を「PyTorchの基礎 → MLP → DLGNへの置き換え」の順に運ぶ3本。
# 他の群と違い、1本目にはDLGNが一切出てこない(まず普通のNNを体感してもらう)。

NN_BASICS_INTRO = """\
**機械学習をやったことがない人のための最初の1本。** このページに DLGN は出てこない。
テンソル → 自動微分 → `nn.Module` → 学習ループ、という PyTorch の全体像を
2次元のトイデータ(直線では分けられない三日月2つ)の分類で一周する。

主題は精度ではなく「**学習とは何をしていることなのか**」を目で見ること。
1パラメータの損失を勾配降下で下る図、学習前後の決定境界、そして
**動いた実数の重みそのもの**の3枚で、NN が実数の世界に住んでいることを確かめる。"""

PYTORCH_MLP_INTRO = """\
前ページの道具立てのまま、手書き数字(MNIST 20x20)へスケールさせる。
`nn.Linear` を3枚重ねただけの MLP が test accuracy 0.95 に届く。

このページの見どころは **「実数の重みはどこにあるのか」** が絵になること。
1層目の重みは入力と同じ 400 次元なので、20x20 に並べ直すと
**そのニューロンが探している形**がそのまま見える。次のページでは、
この重みが丸ごと**論理ゲートの真理値表**へ置き換わる。"""

MLP_TO_DLGN_INTRO = """\
**前ページと同じデータ・同じ学習ループのまま、モデルの中身だけを論理ゲートにする。**
`nn.Linear + ReLU` → `LutLayer`、出力の `nn.Linear`(+softmax)→ `GroupSum` という
1対1の置き換えを実際のコードで見せ、両者の学習曲線を1枚のグラフに重ねる。

実測は MLP 0.9535 に対し DLGN は soft 0.8745 / **hard 0.8660**(離散化ギャップ +0.0085)。
乗算を1つも持たない回路が、実数 NN の 9割の距離まで来ている。
「重み(掛ける実数)」と「真理値表(どの論理関数か)」の違い、そして DLGN で初めて
現れる **soft/hard の二相**を並べて確かめ、最後に置き換えチェックリストで締める。"""

XOR_INTRO = """\
**最小構成で、soft(連続緩和)から hard(離散回路)への崩壊のメカニズムを覗く**付録。
入力2ビット・16ゲートしかないので、学習前後の回路を**全ノード描画**でき、
「学習とは16の候補からゲートを1つずつ選び切る作業だ」という主張がそのまま絵になる。
[MNIST](/tutorials/mnist/) で起きていたことを、4サンプル・数十行に縮めて観察する回。"""

MNIST_INTRO = """\
**ただの論理ゲートの塊が、学習によって手書き数字を読めるようになる。**
実数の重みも乗算も一切持たない 4000 個の AND / OR / XOR が、勾配降下で配線先の
真理値表を決めていくだけで、MNIST の 86% を当てるようになる ── それをこのページで
実際に走らせて確かめる。

道具立ては `LutLayer` 2段 + `GroupSum` + 普通の PyTorch 学習ループだけ。
**回路の俯瞰図・学習曲線(soft/hard)・誤分類例**の3枚で、
「学習された回路」がどんな姿をしているかを見る。"""

DEPLOY_MNIST_INTRO = """\
**学習した回路を、PyTorch なしで、手元の C コンパイラだけで動かす。**
[MNIST](/tutorials/mnist/) の「論理ゲート4000個」を、IR → 配布バンドル → C99 ソースの順に
書き出し、`cc` でビルドして、テスト画像1000枚を C だけで推論する。

このページの核心は **C が返す整数スコアが PyTorch の hard eval と1要素も違わない**こと。
学習した回路が、書き出し・コード生成・コンパイルを経ても「同じ回路のまま」であることを
`assert` で固定してから、配信サイズ(約58KB、gzip で約19KB)・実行時メモリ(4KB)を数える。
1枚あたりの推論時間は著者環境(Apple M1 Max・1コア・`cc -O2`)で約3マイクロ秒だった
(機種で変わる値なのでページの出力には載せず、`latency.json` へ書き出す)。最後はゲート数を
変えた回路を並べて、配信サイズがゲート数にほぼ比例することを確かめる。

この回路を ESP32 や iPhone に載せる手順は [ESP32 に載せる](/guides/deploy-esp32/)・
[iPhone に載せる](/guides/deploy-ios/) で続き、デプロイキットの全体像は
[デプロイキット クイックスタート](/guides/deploy-quickstart/) にある。"""

HAR_INTRO = """\
**スマホの加速度から「歩く・座る・階段を上る」を当てる**。腰につけたスマートフォンの
9 チャネル × 128 サンプル(2.56 秒)の生信号を、4 閾値の**温度計符号化**で 4,608 ビットにして、
9,000 ゲートの論理回路へ入れる。画像ではなく**センサの時系列**を論理ゲートが受け取る最小の例である。

実測は hard accuracy 0.885(テスト 2,947 窓・訓練に居ない 9 人)。後半は
[デプロイする](/tutorials/deploy-mnist/) と同じ出口を通し、学習した回路を C99 に書き出して
**テスト窓すべてを C だけで推論 → PyTorch の hard eval と整数で完全一致**を確かめ、
配信サイズ(約140KB)と実行時メモリ(6KB)を数える。1窓あたりの推論時間は著者環境
(Apple M1 Max・1コア・`cc -O2`)で約6.7マイクロ秒だった(機種で変わる値なのでページの出力には
載せず、`latency.json` へ書き出す)。

この回路を ESP32 や iPhone に載せる手順は [ESP32 に載せる](/guides/deploy-esp32/)・
[iPhone に載せる](/guides/deploy-ios/) で続く。"""

HYBRID_INTRO = """\
既存の Transformer ブロックの **MLP 部分だけ**を LUT ベースの論理ゲートへ置換する
(LL-ViT 型ハイブリッド)。Attention と LayerNorm は PyTorch の実数層のまま残し、
論理系と実数系で学習率を分けて同時に学習する。"""

RULES_INTRO = """\
`with_learning()` でレイヤ単位に学習ルールを注釈し、`HybridTrainer` が
BP / DFA / EGGROLL / AP法 を協調実行する仕組みを確認する。ここでは
DFA(誤差の固定ランダム射影)と AP法(Akiho Propagation。backprop 不要の直接更新)で同じ XOR を解く。

**図の読み方**: AP法は損失(cross entropy)を最適化対象にしていないため、損失曲線では
値が高いまま短く途切れ「失敗して止まった」ように見える。実際には正解率100%に到達して
早期終了しているだけなので、AP法の成否は**右パネルの正解率**で判断すること。"""

HDL_INTRO = """\
学習済みモデルを **IR(中間表現・真実源)** へ抽出し、Verilog を生成して
golden model と照合する。「PyTorch で学習 → IR → HDL」という本ライブラリの
出口をひととおり通す。"""

CNN_INTRO = """\
論理ゲート畳み込み(`LutConv2d`)+ OR プーリング + LUT 全結合の CNN を学習し、
丸ごと Verilog / VHDL へ変換する。畳み込みは IR 化の際に木レベルごとの
`lut_layer` へ unroll され、padding の境界処理は真理値表の特殊化に畳み込まれる。"""

# --- モデル別チュートリアル(examples/07〜23)の導入文 ------------------------
# 「系列・言語」「視覚」「生成モデル」「応用」の4群。いずれも CPU で数分以内に
# 完走するよう縮小したレシピで、各ページの数値は実際の実行結果そのもの。

GPT_INTRO = """\
**Attention も FFN も出力ヘッドも、全部が論理ゲート**の decoder-only Transformer で
シェイクスピアを書く。実数の行列積を一切持たない `LogicGPT` を Tiny Shakespeare の
文字レベル言語モデルとして学習し、uniform baseline の 5.00 bpc から **3.13 bpc** まで
落ちていく様子と、生成されたテキスト(話者行・単語らしい綴り・改行構造まで出る)を見る。

同時に、このページは**離散化ギャップの一番わかりやすい実例**でもある。soft(学習時の
連続緩和)では読める文が出るのに、hard(argmax で確定した離散回路)にすると崩れる ──
言語モデルという難しいタスクほど、この差が正直に出る。"""

SEQ2SEQ_INTRO = """\
エンコーダ・デコーダの全結合層をまるごと LUT に置き換えた **RDDLGN** で、
copy / reverse という系列変換タスクを学習する。attention を持たない固定長
ボトルネック(384ビットの文脈ベクトル)が、どこまで系列を運べるのかを測る回。

**落とし穴もそのまま教材にしている**: `RDDLGNConfig.init_mode` の既定 `'zero'` は
初期状態を 0.5(中立値)にするため、hard eval のはずの文脈ベクトルへ非二値が流れ込む。
`init_mode='one'` にすると hard 経路が純ブールで閉じ、精度まで上がる(0.938 → 0.969)。"""

LSTM_INTRO = """\
**LSTM の忘却ゲートは 2:1 マルチプレクサそのもの、出力ゲートはビット AND そのもの**である。
$\\{0,1\\}$ 上ではゲーティングの掛け算が論理演算に化ける ── その同型性を、遅延コピー
タスクで LogicLSTM / LogicGRU / ゲートなし Elman の3者を比べて確かめる。

ゲートを持つ2つは hard acc 1.000、ゲートなし Elman は 0.578 で頭打ちになる。
隠れ状態は hard eval で厳密に $\\{0,1\\}$ に閉じ、忘却ゲートの開度は時刻ごとに
「いつ記憶を保持し、いつ捨てるか」を数値で見せてくれる。"""

WAVENET_INTRO = """\
WaveNet のゲート付き活性 $z = \\tanh(\\cdot) \\odot \\sigma(\\cdot)$ は、
$\\{0,1\\}$ 上では **ビット AND と同型**になる。この事実を使って因果 convolution を
論理ゲートで組み、周期ビットパターンの次トークン予測で**受容野の効き方**を ablation する。

1層(受容野 RF=2)では直前1ビットからは決まらず 0.817 で頭打ち、3層(RF=8)で 1.000。
さらに位置 $p$ の入力を反転して「変化した出力が $[p, p+RF)$ だけである」ことを実測し、
因果性が回路レベルで保たれていることを確かめる。"""

RESERVOIR_INTRO = """\
**回路を学習しない、という選択**。512ビットの凍結ランダム論理回路に系列を流し込み、
学習するのは readout(全体の 5.9%)だけ ── それでも parity や遅延再生は解けてしまう。

このページの主題は「リザバーが持っているのは**記憶**であって、**記憶への非線形演算**では
ない」こと。delay-d recall は d=0..5 すべて満点なのに、parity は k が増えるほど劣化する。
さらに **hard 化した純ブール回路(leak=1.0)は有界だが収縮しない**(1ビットの差がむしろ
拡散する)ことを摂動カーブで示し、fading memory とハードウェア化のトレードオフを扱う。"""

LADDER_INTRO = """\
**attention はどこまで論理回路にできるのか**を、1段ずつ登って確かめる梯子。
普通の浮動小数点 Transformer(L0)から、射影のゲート化(L1)→ XNOR-popcount スコア(L2)
→ MUX 選択器(L3-C)へ順に置き換え、コピー課題の精度と attention マップで比較する。

結論は直感に反する。**一番落ちるのは L1(射影のゲート化)であり、XNOR 化と softmax の
除去はほぼ無料**(L2 で 1.000 に戻り、L3-C でも 0.990)。L3-C の attention マップは
各行で1マスだけが点灯する ── つまり **attention とは学習された MUX** だった、という絵。"""

UNET_INTRO = """\
分類ではなく**画素ごとに答える**(密予測)を論理ゲートで行う。ノイズ入り MNIST から
数字の領域マスクを復元する U-Net を、OR プーリング + nearest アップサンプル +
Concat skip で組み、**ビットを壊さないまま**エンコーダ・デコーダを往復させる。

前景 IoU 0.568(恒等ベースライン 0.456)に到達し、eval 時の中間表現は厳密に
$\\{0,1\\}$ のまま。実数の U-Net で当たり前に使う補間や平均プーリングを、
論理回路で何に置き換えるのかがこのページの核心。"""

RESNET_INTRO = """\
**論理NNの残差接続は、恒等写像を「厳密に」作れる**。OR 残差と pass-through 初期化を
組み合わせると $\\max|y - x| = 0.0$ ── 近似ではなくビット単位の完全一致で入力が
そのまま出る。実数の ResNet が初期化で近似的に狙うものが、ここでは厳密に成立する。

ただし代償がある。OR は一度立ったビットを落とせないので**飽和する**。同一 seed で
比べると hard acc は XOR 残差 0.361 > OR 残差 0.299 で、恒等の厳密さと学習しやすさが
トレードオフになっていることが学習曲線に出る。"""

MIXER_INTRO = """\
attention を使わずに画像を混ぜる MLP-Mixer の論理ゲート版。画像を16トークンに刻み、
**token-mixing(転置して LutStack に通す)**と **channel-mixing** を soft-XOR 残差で
交互に積む。MNIST で hard acc 0.658。

見どころは「情報がどう混ざるか」の実測。1つのパッチだけを潰した入力との差分を追うと、
パッチ埋め込み直後は変化がそのトークンの行に閉じているのに、1ブロック通すと
隣のトークンへ漏れ出す(1ブロックあたり 1→2 token)。**混ぜているのは誰か**が絵で分かる。"""

DWN_INTRO = """\
**離散化ギャップがゼロ**という、論理NN界の例外。実数画素を単調なビット列へ落とす
`ThermometerEncoder` と 64 エントリの n 入力 LUT を組み合わせた DWN
(Differentiable Weightless Neural Network)は、`train()` と `eval()` のロジット最大差が
**0.0** ── つまり学習中のモデルと焼かれる回路が最初から同一である。

test acc 0.958。他の多くのページで扱う「soft では出るのに hard で崩れる」問題が、
パラメータ化の選び方だけで消えることを見せる回。"""

DIFFUSION_INTRO = """\
拡散モデルを丸ごとブール回路にする。前向き過程のノイズ付加は **XOR**、デノイザは
**LUT**、逆過程のビット組み直しも論理演算 ── 生成の全工程が乗算なしで閉じる。

**ベースラインの取り方まで含めて教材にしている**: 当初 "denoise margin" を主指標に
していたが、学習前から +0.26 出てしまう(=「全画素0」の自明解の分)ことが実測で判明した。
そこで **blank(全画素0)ベースライン**を明示し、「acc@t1 が blank を超える = 回路が
$x_t$ を実際に読んでいる」を合格条件に据え直した。完全ノイズからの再構成が
まだ blank 近傍に留まる現状も、誇張せずそのまま書いている。"""

VAE_INTRO = """\
VAE の再パラメータ化トリック(ガウス潜在の乗算と平方根)を**捨てる**。
代わりに潜在を「比較器そのもの」である Bernoulli ビット列に置き換えると、
VAE がまるごと論理回路に乗る ── 196画素の数字を **96ビット**へ畳んで復元する。

recon BCE 43.27 / 画素一致率 0.907。再構成は数字としてはっきり読める。
**潜在1ビットを反転すると画像がどう変わるか**の差分図(赤=点灯・青=消灯)で、
96本のビットそれぞれが何を担当しているのかを覗く。"""

GAN_INTRO = """\
物理乱数や LFSR からそのまま供給できる**乱数ビット列**を論理回路に通して画像を作る。
そして、壊す。安定化した設定(STABLE)と未調整の設定(NAIVE)を**同一の初期回路から**
学習させて並べ、敵対学習がどう壊れるかを実測で見せる。

差分は4つのつまみ(G:D=3:1 / D入力のビット反転ノイズ / label smoothing / D の学習率↓)
だけ。未調整版は G loss が 7.22 まで発散し D 精度が 1.000 に張り付く ── Generator が
淘汰された状態。縮小レシピでは両条件ともモード崩壊(1/4)が起きることも隠さず出す。"""

ATTENTION_TOUR_INTRO = """\
内積を **XNOR 類似度**に、softmax を **argmax** に置き換えた Attention の4変種を、
連想想起・位置ルックアップ・ソートの小タスクで一気に動かして比べるツアー。

Boolean / MultiQuery / Positional の3つは hard accuracy 1.000(6,144 params)で解け、
4要素ソートを解く Pointer だけが 0.835 に留まる。hard attention の重み行列を描くと
全行が one-hot になり、**Pointer では置換行列**が現れる ── 「どこを見るか」が
連続的な加重平均ではなく、離散的な選択として学習されていることが目で分かる。"""

DETECTION_INTRO = """\
**乗算器ゼロの物体検出**。`LutConv2d` と `OrPooling2d` だけで組んだ CenterNet が、
合成 Multi-MNIST の数字を見つけてバウンディングボックスを描くまでを最小構成で追う。
総ゲート数 784、mAP@0.5 = 0.255(500 step の最小予算での実測)。

数分の予算に収めるため、このページは**クラス非依存(1クラス)検出**に落としてある
(10クラス版は同じ予算では立ち上がらなかった)。GT heatmap をクラス方向の
max で畳む2行を戻せば10クラス版になる。本番レシピは
`experiments/train_detection.py` を参照(そちらの実測値は compute_map の
画像スコープ修正より前の計測なので再計測待ち)。"""

BINARYBRAIN_INTRO = """\
FPGA 向け OSS **BinaryBrain** と同じ部品構成(`RealToBinary` → `DifferentiableLutBlock`
→ `BinaryToReal`)で MNIST を学習する。実数画像を確率的に7枚の二値フレームへ変調し、
LUT6 を 430 個並べ、最後に多数決で実数へ戻す ── FPGA の LUT を直接学習する系譜。

**このページの主題は「まだ焼けない」ことの定量化**である。連続の真理値表(BinaryBrain
本来の推論経路)では eval acc 0.803 だが、エントリを 0/1 へ丸めると **0.211 に崩壊**する。
BatchNorm 統計を再較正しても 0.632 までしか戻らない ── 主因は統計ズレではなく、
真理値表エントリが 0.5 付近(std 0.048)に密集したまま**決めきれていない**こと。
短縮レシピが正直に見せてくれるこの離散化ギャップを、ヒストグラムまで含めて測る。"""

TEXTCNN_INTRO = """\
**単語を数えるだけでは絶対に解けない**問題を用意して、畳み込みが n-gram 検出器として
働くことを反証不能な形で示す。肯定語 × 良い語 / 否定語 × 悪い語 の XNOR でラベルを
決めるバイグラム XOR 分類では、どの単語も両クラスに等頻度で現れる ──
つまり**ユニグラムの情報量がゼロ**(単語頻度による規則の上限 0.535 = ほぼ偶然)。

論理ゲート版 TextCNN はこれを soft/hard ともに **1.000** で解く。しかも学習後の
フィルタを覗くと、filter 6 は "really good"、filter 4 は "never awful" ──
**特定の2語の検出器**へ育っている。[付録: XOR](/tutorials/xor/) の自然な続き。"""

FULL_INTRO = """\
学習 → 損失ログの読み方 → HDL エクスポート → FPGA 実装への接続、までを
1本の流れで通す**通しチュートリアル**。題材は最小の XOR に絞ってあるので、
個別ページ([MNIST](/tutorials/mnist/) / [HDLエクスポート](/tutorials/hdl-export/))を
先に読んでいなくても、このページだけで「学習してから回路になるまで」の全体像がつかめる。"""


# 通しチュートリアルは対応する examples/*.py を持たない(既存サンプルを合成する)。
# 実行可能性を保つため、ここでも 01/04 と同じ API のみを使う。
FULL_PIPELINE_SOURCE = '''\
# %% [markdown]
# ## 1. データとモデル
#
# XOR は「線形分離できない最小の問題」で、論理ゲート回路の表現力を測る定番のタスク。
# `griffith_logic.datasets.xor2()` は4パターンを butterfly 規約の順で列挙したデータセットを返す。
# モデルは `LutLayer` 2段 + `GroupSum`(クラスごとにゲート出力を集約)という最小構成。

# %%
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic import GroupSum, LutLayer
from griffith_logic.datasets import xor2
from griffith_logic.export import export_ir
from griffith_logic.export.verify import iverilog_available, run_iverilog_equivalence
from griffith_logic.export.verilog import write_verilog
from griffith_logic.ir import save_ir, simulate, validate

# 乱数種の固定。各ゲートが「どの論理関数寄りから学習を始めるか」は乱数依存なので、
# 固定しないと実行ごとに収束の速さが変わる。
torch.manual_seed(0)

# xor2() は4パターンを butterfly 規約の順で列挙したデータセットを返す。
# XORは4通りしかないので訓練=テストで構わない(暗記ではなく真理値表の正しさを見る)。
x, y_bits = xor2().dataset.tensors
y = y_bits.squeeze(-1).long()  # cross_entropy が要求するクラスラベル{0,1}へ

model = nn.Sequential(
    # 入力2ビット → 16ゲート。各ゲートは2入力の論理関数(AND/OR/XOR等16種)の
    # ソフト混合として学習される。どの入力線を見るかの配線は初期化時に固定される。
    LutLayer(2, 16, n_inputs=2),   # 16ゲートsoftmax(difflogic互換)
    # 2段目。XORのような非線形は、この「ゲートの合成」によって表現される。
    LutLayer(16, 16, n_inputs=2),
    # 最終層16ゲートの出力を2クラスへ均等に分けて集計(グループ和)。
    # tau はロジットの温度で、大きいほど softmax がなだらかになる。
    GroupSum(k=2, tau=2.0),
)
print(model)

# 学習前の回路を後で図にするためのスナップショット(学習は元の model で進める)。
model_init = copy.deepcopy(model)

# %% [markdown]
# ## 2. 学習と損失ログの読み方
#
# 学習は通常の PyTorch そのもの(`cross_entropy` + Adam)。特有なのは
# **soft(学習時)と hard(離散回路)で精度が別物になりうる**点で、
# 損失だけを見ていると「soft では下がっているのに回路にすると壊れる」状況を見逃す。
# そこで損失と一緒に **hard accuracy(`model.eval()` での精度)も同じ間隔でログする**。

# %%
# 最適化対象は「各ゲートがどの論理関数か」を表す連続パラメータ。配線は固定なので、
# 学習しているのは実質的に真理値表の中身だけ。
opt = torch.optim.Adam(model.parameters(), lr=0.05)
history = []
for step in range(501):
    opt.zero_grad()
    # loss はクラス分類の cross entropy。backward() で GroupSum → 2段目 → 1段目 の順に
    # ゲート分布へ勾配が流れる(連続緩和しているので微分可能)。
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    if step % 10 == 0:
        # eval() に切り替えると各ゲートが argmax で1つの論理関数に確定する。
        # つまりここで測っているのは「今この瞬間の回路」の正解率。
        model.eval()
        with torch.no_grad():
            hard_acc = (model(x).argmax(-1) == y).float().mean().item()
        model.train()  # 学習は連続緩和の状態に戻して続ける
        with torch.no_grad():
            soft_acc = (model(x).argmax(-1) == y).float().mean().item()
        history.append((step, loss.item(), soft_acc, hard_acc))

print(f"{'step':>6} {'loss':>10} {'soft acc':>10} {'hard acc':>10}")
for step, loss_value, soft_acc, hard_acc in history[::10]:
    print(f'{step:>6} {loss_value:>10.4f} {soft_acc:>10.2f} {hard_acc:>10.2f}')

# %% [markdown]
# ### 図1: 損失と soft/hard 精度の推移
#
# 表だけでは「いつ回路が決まったのか」が見えにくいので、同じ履歴を1枚の図にする。
# 左が損失、右が精度で、**soft(連続緩和のまま)と hard(argmax で確定した回路)を
# 重ねて描く**。この2本が離れているあいだは「学習中のモデルは解けているが、
# 焼いた回路は解けていない」状態である。

# %%
steps = [h[0] for h in history]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(steps, [h[1] for h in history], color='#ef4444')
axes[0].set_xlabel('step')
axes[0].set_ylabel('cross entropy loss')
axes[0].set_title('loss (continuous relaxation)')
axes[0].grid(alpha=0.3)
axes[1].plot(steps, [h[2] for h in history], label='soft (train mode)', color='#38bdf8')
axes[1].plot(steps, [h[3] for h in history], label='hard (eval = real circuit)',
             color='#10b981', linestyle='--')
axes[1].set_xlabel('step')
axes[1].set_ylabel('accuracy')
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_title('soft vs hard accuracy')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# 読み方の要点:
#
# - **loss が下がりきる前に hard acc が 1.00 に到達する**ことが多い。回路としての正解は
#   離散化後の argmax で決まるため、損失の絶対値そのものは目標ではない。
# - loss は下がるのに hard acc が上がらない場合、連続緩和が「中間的な確率分布」に
#   居座っている(離散化で壊れる)サイン。温度 `tau` や学習率、層幅を疑う。
# - 逆に hard acc が 1.00 で振動する場合は、既に十分学習できているので早期終了してよい。

# %%
# train(): 各ゲートは16種の論理関数の「確率的な混合」のまま(=連続緩和)。
model.train()
acc_soft = (model(x).argmax(-1) == y).float().mean()
# eval(): 各ゲートが1つの論理関数に確定する。ここから先はもう学習用の近似ではなく、
# FPGAに焼けるのと同じ純粋なBoolean回路。
model.eval()  # 離散化(argmaxで完全なBoolean回路に崩壊)
acc_hard = (model(x).argmax(-1) == y).float().mean()
print(f'XOR: soft acc={acc_soft:.2f}, hard acc={acc_hard:.2f}')
# hard が 1.00 であることが本質。soft だけ高い場合は回路として決めきれていない。
assert acc_hard == 1.0

# %% [markdown]
# ## 3. 同じコードがこの回路になる
#
# ここまでで書いたのは `nn.Sequential(LutLayer(2, 16), LutLayer(16, 16), GroupSum(k=2))`
# という3行だけだが、実体は **16個のゲートのノードと、それを結ぶ配線** である。
# `plot_circuit()` はモデルからその回路をそのまま図にする。
#
# ### 図2: 学習前と学習後の回路図
#
# - **左(`resolve=False`)** — 学習前。各ノードは16種の論理関数(AND/OR/XOR/…)の
#   **混合**なので `LUT` としか呼べない。色の濃さがゲート分布の確定度で、
#   薄いほど「まだどの関数か決まっていない」。
# - **右(`resolve=True`)** — 学習後。argmax で1つに確定した論理関数の名前
#   (`AND` / `XOR` / `NAND` / `!A` / `A>=B` …)がノードに入る。
#
# **学習とは、16の候補からゲートを1つずつ選び切る作業**である、という主張が
# そのまま絵になっている。ノードの名前は後段の `export_ir()` が Verilog を吐くときに
# 使う真理値表から導いているので、この図のゲートは**実際に焼かれる回路のゲート**と同一。

# %%
import matplotlib.pyplot as plt

from griffith_logic.analysis.circuit_viz import plot_circuit

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
# 緑=論理ゲート、橙=実数部(GroupSum のクラススコア)、灰=入力ビットと配線。
plot_circuit(model_init, resolve=False, ax=axes[0])
plot_circuit(model, resolve=True, ax=axes[1])
fig.tight_layout()
plt.show()

# %% [markdown]
# ゲート数が数百を超える実モデルでは全ノードを描いても潰れるだけなので、
# `plot_circuit()` は自動的に俯瞰図(層=箱、配線=本数付き矢印)へ切り替わる
# (`plot_abstract()` を直接呼んでもよい)。読み方は
# [配線の可視化](/guides/wiring-visualization/) を参照。

# %% [markdown]
# ## 4. IR へ抽出する
#
# `export_ir()` は学習済み `nn.Module` を走査し、どのパラメータ化で学習していても
# **真理値表へ万能離散化**した IR(JSON)にする。IR がハードウェア側の真実源であり、
# Verilog / VHDL / シミュレータはすべてここから派生する。`validate(ir, target='hdl')`
# で HDL 生成に必要な制約(ビット幅・トポロジ順序など)を先に検査しておく。

# %%
# export_ir はモデルを走査し、層の種類・配線・確定した真理値表をJSONへ落とす。
# どのパラメータ化(softmax / independent / iwp …)で学習していても、
# 出てくるIRは「真理値表の集まり」に正規化される。
ir = export_ir(model, input_width=2, name='xor_full_pipeline')
# ビット幅・トポロジ順序・未対応ノードの不在を先に検査しておく。
# ここで落とせば、合成ツールで初めて気づく事故を防げる。
validate(ir, target='hdl')
save_ir(ir, 'out_examples/xor_full_pipeline.json')
print(f'IR ノード数: {len(ir.nodes)}')
for node in ir.nodes:
    print(f'  - {node.type}: {node.params.get("out_width", "-")}')

# %% [markdown]
# ## 5. Verilog を生成して照合する
#
# `write_verilog()` で合成可能な Verilog を書き出し、`simulate()`(numpy のビット厳密な
# golden model)と突き合わせる。`iverilog` が入っていれば、生成した Verilog 自体を
# シミュレートして golden model と全パターン一致するかまで自動検証できる。

# %%
# IR から合成可能な RTL を書き出す(Vivado / Quartus にそのまま投げられる)。
write_verilog(ir, 'out_examples/xor_full_pipeline.v')
# simulate() は IR を numpy の整数ビット演算で解釈実行する参照実装。浮動小数の近似が
# 入らないので「回路としてこう動くはず」という基準(golden model)になる。
scores = simulate(ir, x.numpy().astype(np.int64))
print(f'IR simulate 予測: {scores.argmax(-1).tolist()} (正解 {y.tolist()})')

# 生成した Verilog 自体をシミュレートして golden model と全パターン照合する。
# PyTorch hard eval ⇔ IR simulate ⇔ HDLシミュレーション の3段等価性の最後の1段。
if iverilog_available():
    result = run_iverilog_equivalence(
        ir, x.numpy().astype(np.int64), 'out_examples/verify_full_pipeline',
    )
    print('Verilog照合: ' + ('ALLPASS' if result['passed'] else '不一致'))
else:
    print('iverilog未導入のためVerilog照合はスキップ(brew install icarus-verilog)')

# %% [markdown]
# ここまでで、同じ XOR が**3つの別々の実装**を通った:
# PyTorch の `model.eval()`、numpy 実装の `simulate()`、そして Verilog シミュレータ。
# この3つが全入力パターンで一致することが「学習したモデル = 焼かれる回路」の保証になる。
# 4パターンしかないので、3段の予測を並べて全部見てしまう。
#
# ### 図3: 同じ XOR を3つの実装で通す
#
# 行が「教師ラベル / PyTorch eval / IR simulate」、列が4つの入力パターン。
# セルの数字が予測クラスである。

# %%
model.eval()
with torch.no_grad():
    torch_pred = model(x).argmax(-1).numpy()
ir_pred = simulate(ir, x.numpy().astype(np.int64)).argmax(-1)
rows = np.vstack([y.numpy(), torch_pred, ir_pred])
labels = ['label', 'PyTorch eval', 'IR simulate']
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.imshow(rows, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels([f'({int(a)},{int(b)})' for a, b in x.numpy()])
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel('input pattern (a, b)')
ax.set_title('same XOR through three implementations')
for row in range(rows.shape[0]):
    for col in range(rows.shape[1]):
        ax.text(col, row, int(rows[row, col]), ha='center', va='center', fontsize=12)
fig.tight_layout()
plt.show()

# %% [markdown]
# **読み取り**: 3行が同じ並び `0 1 1 0` で埋まる ── 教師ラベルどおりの XOR を、
# PyTorch の離散化経路と numpy の整数ビット演算がまったく同じように再現している。
# 1マスでも食い違えば、それはエクスポートのバグである。
#
# ### 生成された Verilog を覗く
#
# 最後に、書き出された RTL の冒頭を表示する(そのまま論理合成に掛けられる)。

# %%
verilog = Path('out_examples/xor_full_pipeline.v').read_text().splitlines()
print('\\n'.join(verilog[:20]))

# %% [markdown]
# ## 6. ここから FPGA へ
#
# 生成した `.v` はベンダーツール(Vivado / Quartus 等)へそのまま投げられる合成可能 RTL。
# ピン割り当て・クロック・ビットストリーム生成といった実機側の手順は
# [HDL等価性検証ガイド](/guides/hdl/) を参照。
'''


TUTORIALS: list[Tutorial] = [
    # --- 入門コース ---------------------------------------------------------
    Tutorial(
        slug="nn-basics",
        group="intro",
        thumb="nn-basics_17_0.png",
        title="入門1. ニューラルネットとPyTorchの基礎",
        description="テンソル・自動微分・nn.Module・学習ループを、"
        "直線では分けられないトイデータの分類で一周する(DLGNは出てこない)。",
        minutes=8,
        intro=NN_BASICS_INTRO,
        source="24_nn_basics.py",
        next_links=[
            "- [入門2. PyTorchでMLPを組む](/tutorials/pytorch-mlp/)"
            " — 同じ書き方のまま手書き数字へ",
            "- [そもそもニューラルネットとは](/explanation/ai-basics/)"
            " — 数式で理解したい人向けの読み物",
            "- [NNからDLGNへ](/explanation/nn-to-dlgn/) — 何が同じで何が違うのかの見取り図",
        ],
    ),
    Tutorial(
        slug="pytorch-mlp",
        group="intro",
        thumb="pytorch-mlp_10_0.png",
        title="入門2. PyTorchでMLPを組む",
        description="nn.Linear + ReLU の MLP で MNIST 20x20 を解き、"
        "1層目の重みを20x20の画像として並べて「実数の重み」を目で見る。",
        minutes=8,
        intro=PYTORCH_MLP_INTRO,
        source="25_pytorch_mlp.py",
        next_links=[
            "- [入門3. MLPをDLGNへ置き換える](/tutorials/mlp-to-dlgn/)"
            " — 同じ問題を論理ゲートで解く",
            "- [入門1. ニューラルネットとPyTorchの基礎](/tutorials/nn-basics/)",
            "- [1. MNISTを学習する](/tutorials/mnist/)"
            " — 同じMNISTを最初から論理ゲートで解く本編",
        ],
    ),
    Tutorial(
        slug="mlp-to-dlgn",
        group="intro",
        thumb="mlp-to-dlgn_12_0.png",
        title="入門3. MLPをDLGNへ置き換える",
        description="同じデータ・同じ学習ループのまま nn.Linear→LutLayer / "
        "softmax出力→GroupSum と置き換え、学習曲線・重みと真理値表・soft/hardを比べる。",
        minutes=10,
        intro=MLP_TO_DLGN_INTRO,
        source="26_mlp_to_dlgn.py",
        next_links=[
            "- [1. MNISTを学習する](/tutorials/mnist/)"
            " — 論理ゲート版をもう一段掘る(回路の俯瞰図つき)",
            "- [PyTorchからの移行](/guides/from-pytorch/) — 層・学習・エクスポートの全対応表",
            "- [NNからDLGNへ](/explanation/nn-to-dlgn/) — 置き換えの背景にある考え方",
            "- [学習曲線の見方](/guides/training-curves/) — soft/hard の読み分け",
        ],
    ),
    # --- 基本 ---------------------------------------------------------------
    Tutorial(
        slug="mnist",
        group="basics",
        thumb="mnist_12_0.png",
        title="1. MNISTを学習する",
        description="論理ゲート4000個だけで MNIST 20x20 を学習し、"
        "離散回路のまま hard accuracy 0.86 に到達する。",
        minutes=10,
        intro=MNIST_INTRO,
        learn=[
            "`LutLayer` 2段 + `GroupSum` を**普通の PyTorch 学習ループ**で回すだけで、"
            "論理ゲート4000個が手書き数字を読めるようになること",
            "soft(連続緩和)と hard(離散回路)の精度を必ず並べて見る理由と、その読み分け",
            "学習された回路の姿 ── 俯瞰図・学習曲線・誤分類例の読み方",
        ],
        prereq=[
            "[基本コンセプト](/getting-started/concepts/)"
            " — soft/hard・LutLayer・GroupSum・IR・HDL の5概念",
            "[入門3. MLPをDLGNへ置き換える](/tutorials/mlp-to-dlgn/)"
            " — NN が初めてならここから",
        ],
        source="06_mnist.py",
        next_links=[
            "- [2. ハイブリッドNNを作る](/tutorials/hybrid/)"
            " — 既存モデルの一部だけを論理ゲート化する",
            "- [3. HDLへエクスポート](/tutorials/hdl-export/) — 学習した回路を Verilog にする",
            "- [学習曲線の見方](/guides/training-curves/) — soft/hard の読み分け",
            "- [付録: XORで仕組みを最小で覗く](/tutorials/xor/)"
            " — ここで起きていたことを16ゲートで観察する",
            "- [論理ゲートU-Net](/tutorials/logic-unet/)"
            " — 分類の次に、画素ごとに答える(密予測)へ進む",
            "- [DWN](/tutorials/dwn/) — 同じMNISTを離散化ギャップゼロで解く",
        ],
    ),
    Tutorial(
        slug="hybrid",
        group="basics",
        thumb="hybrid_15_1.png",
        title="2. ハイブリッドNNを作る",
        description="既存モデルの MLP だけを LUT へ置換し、Attention は実数のまま学習する。",
        minutes=5,
        intro=HYBRID_INTRO,
        learn=[
            "Transformer ブロックの **MLP 部分だけ**を LUT へ置換する組み方(LL-ViT 型)",
            "Attention・LayerNorm を実数のまま残し、論理系と実数系で学習率を分ける理由",
            "「全部を論理ゲートにしない」という設計判断がどこで効くか",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/) — 全部が論理ゲートの場合を先に見る",
            "[PyTorchからの移行](/guides/from-pytorch/) — 層の対応表",
        ],
        source="02_hybrid_llvit.py",
        next_links=[
            "- [3. HDLへエクスポート](/tutorials/hdl-export/) — 学習済みモデルを回路にする",
            "- [学習則(BP/AP/DFA/ES)](/explanation/learning-rules/) — 層ごとに学習ルールを変える",
            "- [アーキテクチャ全体像](/explanation/architecture/)",
            "- [Transformerを段階的に論理化](/tutorials/logic-ladder/)"
            " — 「どこまで論理ゲート化するか」を1段ずつ測る",
        ],
    ),
    Tutorial(
        slug="hdl-export",
        group="basics",
        thumb="hdl-export_3_0.png",
        title="3. HDLへエクスポート",
        description="学習済みモデルを IR へ抽出し、Verilog / VHDL を生成して等価性を検証する。",
        minutes=10,
        intro=HDL_INTRO,
        learn=[
            "学習済みモデル → IR(中間表現・真実源)→ Verilog という出口の3手順",
            "`validate(ir, target='hdl')` が合成前にどんな事故を落としてくれるのか",
            "PyTorch hard eval ⇔ IR simulate ⇔ HDL シミュレーションの3段等価性の確かめ方",
        ],
        prereq=[
            "[付録: XORで仕組みを最小で覗く](/tutorials/xor/) — ここで焼く回路そのもの",
            "[IRガイド](/guides/ir/) — IR が真実源である理由",
        ],
        source="04_export_hdl.py",
        next_links=[
            "- [5. CNNをHDLへエクスポート](/tutorials/cnn-hdl-export/) — 畳み込みを含むモデル",
            "- [HDL等価性検証](/guides/hdl/) — iverilog / ghdl の導入と3段照合",
        ],
    ),
    Tutorial(
        slug="deploy-mnist",
        group="basics",
        thumb="deploy-mnist_18_0.png",
        title="4. デプロイする",
        description="学習したMNISTの回路を IR → 配布バンドル → C99 ソースへ書き出して "
        "cc でビルドし、PyTorch hard eval と整数一致することを確かめ、"
        "配信サイズと推論時間を測る。",
        minutes=10,
        intro=DEPLOY_MNIST_INTRO,
        learn=[
            "学習済みモデル → IR → 配布バンドル → C ソース → 実行ファイル、という出口の全段",
            "C の整数スコアが PyTorch hard eval・IR simulate と**完全一致**することの確かめ方",
            "配信サイズ・実行時メモリの測り方と、ゲート数に対するサイズの伸び方"
            "(µs/枚は機種依存なので `latency.json` に記録する)",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/) — ここで運ぶ回路そのもの",
            "[3. HDLへエクスポート](/tutorials/hdl-export/) — IR と3段等価性の考え方",
            "[デプロイキット クイックスタート](/guides/deploy-quickstart/)"
            " — 先に XOR で出口の全体を軽く通したいとき(任意)",
        ],
        source="29_deploy_mnist.py",
        next_links=[
            "- [デプロイキット クイックスタート](/guides/deploy-quickstart/)"
            " — XOR で通しで動かす。CUDA / Metal / Web への分岐もここから",
            "- [デモモデルを用意する](/guides/deploy-models/)"
            " — MNIST / 行動認識 / 顔検出のバンドルを揃える手順",
            "- [ESP32 に載せる](/guides/deploy-esp32/)"
            " — 同じバンドルをマイコンで動かす(実機実測つき)",
            "- [iPhone に載せる](/guides/deploy-ios/)"
            " — Swift から生成 C を呼ぶ(シミュレータ実測つき)",
            "- [行動認識(UCI-HAR)](/tutorials/har/)"
            " — スマホの加速度を温度計符号化して、同じ手順でCまで運ぶ",
            "- [7. 通しチュートリアル](/tutorials/full-pipeline/) — 学習から Verilog / FPGA まで",
            "- [付録: XORで仕組みを最小で覗く](/tutorials/xor/)"
            " — 運ぶ回路をもっと小さくして中身を全部見る",
        ],
    ),
    Tutorial(
        slug="cnn-hdl-export",
        group="basics",
        thumb="cnn-hdl-export_14_0.png",
        title="5. CNNをHDLへエクスポート",
        description="LutConv2d の CNN を IR へ unroll し、"
        "Verilog / VHDL を生成して golden model と照合する。",
        minutes=10,
        intro=CNN_INTRO,
        learn=[
            "`LutConv2d` + OR プーリング + LUT 全結合という論理ゲート CNN の組み方",
            "畳み込みが IR 化で木レベルごとの `lut_layer` へ unroll される様子",
            "padding の境界処理が真理値表の特殊化へ畳み込まれること",
        ],
        prereq=[
            "[3. HDLへエクスポート](/tutorials/hdl-export/) — まず全結合の回路で出口を通す",
            "[1. MNISTを学習する](/tutorials/mnist/)",
        ],
        source="05_cnn_export_hdl.py",
        next_links=[
            "- [IRを扱う](/guides/ir/) — IR の構造と操作",
            "- [HDL等価性検証](/guides/hdl/)",
            "- [乗算器ゼロの物体検出](/tutorials/detection/)"
            " — 同じ LutConv2d を検出タスクへ広げる",
        ],
    ),
    Tutorial(
        slug="learning-rules",
        group="basics",
        thumb="learning-rules_4_0.png",
        title="6. 学習則を切り替える(DFA / AP法)",
        description="with_learning でレイヤ単位に学習ルールを注釈し、"
        "HybridTrainer で DFA と AP法(Akiho Propagation)を実行する。",
        minutes=5,
        intro=RULES_INTRO,
        learn=[
            "`with_learning()` + `HybridTrainer` でレイヤ単位に学習則を混ぜる書き方",
            "DFA(誤差の固定ランダム射影)と AP法(Akiho Propagation。backprop 不要の直接更新)の違い",
            "AP法の成否は損失曲線ではなく**正解率**で判断する、という図の読み方",
        ],
        prereq=[
            "[付録: XORで仕組みを最小で覗く](/tutorials/xor/) — 同じ問題を BP で解いた場合",
            "[学習を回す](/guides/training/) — 学習の基本操作",
        ],
        source="03_learning_rules.py",
        next_links=[
            "- [学習則(BP/AP/DFA/ES)](/explanation/learning-rules/) — 各ルールの数理",
            "- [学習を回す](/guides/training/)",
            "- [回路を学習しないという選択](/tutorials/logic-reservoir/)"
            " — readout だけを学習する極端なケース",
        ],
    ),
    Tutorial(
        slug="full-pipeline",
        group="basics",
        thumb="full-pipeline_9_0.png",
        title="7. 通しチュートリアル: 学習からFPGAまで",
        description="学習・損失ログの読み方・IR 抽出・Verilog 生成・"
        "実機への接続を1本の流れで通す。",
        minutes=15,
        intro=FULL_INTRO,
        learn=[
            "学習 → 損失ログ → IR → Verilog → FPGA 実装まで、途切れない1本の流れ",
            "各段でどの API を呼ぶか(このページだけで最後まで動く)",
            "深掘りしたくなったとき、どの個別ページへ戻ればよいかの地図",
        ],
        prereq=[],
        synth=FULL_PIPELINE_SOURCE,
        notebook_name="07_full_pipeline.ipynb",
        next_links=[
            "- [HDL等価性検証](/guides/hdl/) — FPGA 実装と等価性の担保",
            "- [IRを扱う](/guides/ir/) — 生成された IR を読む・書き換える",
            "- [クラウドGPU](/guides/cloud-gpu/) — 大きなモデルを学習する",
        ],
    ),
    Tutorial(
        slug="xor",
        group="basics",
        thumb="xor_15_0.png",
        title="付録: XORで仕組みを最小で覗く",
        description="16ゲートの最小構成で XOR を学習し、"
        "soft から hard(離散回路)への崩壊を全ノード描画で観察する。",
        minutes=5,
        intro=XOR_INTRO,
        learn=[
            "16ゲートの最小構成で、soft(連続緩和)から hard(離散回路)へ落とす瞬間を全部見る",
            "学習とは「16種の論理関数の候補からゲートを1つずつ選び切る」作業だ、という見方",
            "回路図の読み方(学習前 `resolve=False` の混合 / 学習後 `resolve=True` の確定ゲート)",
        ],
        prereq=[
            "[基本コンセプト](/getting-started/concepts/) — soft/hard の用語だけ先に押さえる",
        ],
        source="01_xor.py",
        next_links=[
            "- [1. MNISTを学習する](/tutorials/mnist/) — 同じ仕組みを実データへスケールさせる",
            "- [7. 通しチュートリアル](/tutorials/full-pipeline/)"
            " — 学習から Verilog 生成までを1本で通す",
            "- [LutLayerの数理](/explanation/lut-layer/) — 連続緩和の式そのもの",
            "- [論理ゲート版TextCNN](/tutorials/textcnn/)"
            " — 同じXORの構造を、単語列の上で解く",
        ],
    ),
    # --- 系列・言語 ---------------------------------------------------------
    Tutorial(
        slug="logic-gpt",
        group="sequence",
        thumb="logic-gpt_14_0.png",
        title="論理ゲートGPTでシェイクスピアを書く",
        description="attention も FFN も出力ヘッドも論理ゲートの decoder-only Transformer を"
        "Tiny Shakespeare で学習し、3.13 bpc と離散化ギャップを見る。",
        minutes=10,
        intro=GPT_INTRO,
        learn=[
            "attention・FFN・出力ヘッドまで全部が論理ゲートの decoder-only Transformer の構成",
            "文字レベル言語モデルの指標 bpc の読み方(uniform baseline 5.00 → 3.13)",
            "言語モデルでは離散化ギャップ(soft は読めるのに hard で崩れる)が最も正直に出ること",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/) — soft/hard の読み分けを先に体験する",
            "[学習曲線の見方](/guides/training-curves/)",
        ],
        source="07_logic_gpt.py",
        next_links=[
            "- [Transformerを段階的に論理化](/tutorials/logic-ladder/)"
            " — attention をどこまで論理回路にできるか段階的に確かめる",
            "- [RNNだけで作るseq2seq(RDDLGN)](/tutorials/rddlgn-seq2seq/)"
            " — attention を持たない系列変換と比べる",
            "- [学習曲線の見方](/guides/training-curves/) — soft/hard の読み分け",
        ],
    ),
    Tutorial(
        slug="rddlgn-seq2seq",
        group="sequence",
        thumb="rddlgn-seq2seq_12_0.png",
        title="RNNだけで作るseq2seq(RDDLGN)",
        description="全結合層をLUTに置き換えたRDDLGNでcopy/reverseを学習し、"
        "384ビットの固定長ボトルネックがどこまで系列を運べるかを測る。",
        minutes=8,
        intro=SEQ2SEQ_INTRO,
        learn=[
            "全結合層をまるごと LUT に置き換えた RNN(RDDLGN)で系列変換を学習する組み方",
            "attention を持たない固定長ボトルネックが、どこまで系列を運べるのかの実測",
            "`init_mode` の選択が hard 経路の純ブール性と精度を左右すること(0.938 → 0.969)",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/)",
            "[パラメータ化を選ぶ](/guides/parameterization/) — init_mode が効く理由",
        ],
        source="08_rddlgn_seq2seq.py",
        next_links=[
            "- [RNN: LSTM / GRU の論理化](/tutorials/logic-lstm/)"
            " — ゲーティングをブール演算として実装する",
            "- [論理ゲートGPT](/tutorials/logic-gpt/) — attention 付きの系列モデルと比べる",
            "- [パラメータ化を選ぶ](/guides/parameterization/)"
            " — init_mode / パラメータ化が hard 経路に効く理由",
        ],
    ),
    Tutorial(
        slug="logic-lstm",
        group="sequence",
        thumb="logic-lstm_14_0.png",
        title="ゲーティングはブール演算である(LSTM/GRU)— RNNの論理化",
        description="LSTMの忘却ゲートが2:1マルチプレクサ、出力ゲートがビットANDになることを"
        "遅延コピータスクでゲートなしElmanと比べて確かめる。",
        minutes=5,
        intro=LSTM_INTRO,
        learn=[
            "忘却ゲート = 2:1 マルチプレクサ、出力ゲート = ビット AND という同型性",
            "ゲートあり(LSTM/GRU)と無し(Elman)の差が、遅延コピー課題でどう出るか",
            "hard eval で隠れ状態が厳密に $\\{0,1\\}$ に閉じることの確かめ方",
        ],
        prereq=[
            "[RNNだけで作るseq2seq(RDDLGN)](/tutorials/rddlgn-seq2seq/)"
            " — ゲートを持たない再帰回路を先に見る",
            "[基本コンセプト](/getting-started/concepts/)",
        ],
        source="09_logic_lstm.py",
        next_links=[
            "- [因果convolutionを論理ゲートで](/tutorials/logic-wavenet/)"
            " — 再帰を使わずに履歴を見る",
            "- [リザバー計算(学習しないRNN)](/tutorials/logic-reservoir/)"
            " — 再帰回路を学習しないという選択",
            "- [RNNだけで作るseq2seq(RDDLGN)](/tutorials/rddlgn-seq2seq/)",
        ],
    ),
    Tutorial(
        slug="logic-wavenet",
        group="sequence",
        thumb="logic-wavenet_10_0.png",
        title="因果convolutionを論理ゲートで(WaveNet)",
        description="ゲート付き活性がビットANDと同型であることを使い、"
        "周期パターンの次トークン予測で受容野の効き方をablationする。",
        minutes=5,
        intro=WAVENET_INTRO,
        learn=[
            "ゲート付き活性 $\\tanh \\odot \\sigma$ が $\\{0,1\\}$ 上でビット AND と同型になること",
            "受容野(RF)を層数で伸ばす ablation の読み方(RF=2 で 0.817 → RF=8 で 1.000)",
            "入力を1ビット反転して、因果性が回路レベルで保たれていることを実測する方法",
        ],
        prereq=[
            "[RNN: LSTM / GRU の論理化](/tutorials/logic-lstm/) — 再帰で履歴を持つ場合",
            "[5. CNNをHDLへエクスポート](/tutorials/cnn-hdl-export/) — 論理ゲート畳み込みの基礎",
        ],
        source="10_logic_wavenet.py",
        next_links=[
            "- [論理ゲートU-Net](/tutorials/logic-unet/)"
            " — 同じ畳み込みを2次元の密予測へ広げる",
            "- [RNN: LSTM / GRU の論理化](/tutorials/logic-lstm/) — 再帰型と比べる",
            "- [5. CNNをHDLへエクスポート](/tutorials/cnn-hdl-export/)"
            " — 畳み込みが回路へ unroll される様子",
        ],
    ),
    Tutorial(
        slug="logic-reservoir",
        group="sequence",
        thumb="logic-reservoir_14_0.png",
        title="回路を学習しないという選択(リザバー計算=学習しないRNN)",
        description="512ビットの凍結ランダム論理回路にreadoutだけを学習させ、"
        "リザバーが持つのが記憶であって非線形演算ではないことを見る。",
        minutes=5,
        intro=RESERVOIR_INTRO,
        learn=[
            "凍結したランダム論理回路 + readout だけの学習(全体の 5.9%)で何が解けるか",
            "リザバーが持っているのは**記憶**であって、記憶への非線形演算ではないこと",
            "hard 化した純ブール回路が「有界だが収縮しない」という、ハードウェア化の落とし穴",
        ],
        prereq=[
            "[RNN: LSTM / GRU の論理化](/tutorials/logic-lstm/) — 学習する再帰回路と比べる",
        ],
        source="11_logic_reservoir.py",
        next_links=[
            "- [6. 学習則を切り替える(DFA / AP法)](/tutorials/learning-rules/)"
            " — 「どこを学習するか」を変える別の道",
            "- [RNN: LSTM / GRU の論理化](/tutorials/logic-lstm/) — 学習する再帰回路と比べる",
            "- [FPGAに焼く](/guides/fpga/) — 凍結回路をそのまま実装するときの前提",
        ],
    ),
    Tutorial(
        slug="logic-ladder",
        group="sequence",
        thumb="logic-ladder_10_0.png",
        title="Transformerの論理化の梯子: attentionはどこまで回路にできるか",
        description="浮動小数点Transformer(L0)から射影ゲート化・XNORスコア・MUX選択器へ"
        "1段ずつ論理化し、attention = 学習されたMUXであることを実演する。",
        minutes=8,
        intro=LADDER_INTRO,
        learn=[
            "浮動小数点 Transformer(L0)から MUX 選択器(L3-C)まで、1段ずつ論理化する道筋",
            "どの段で精度が落ちるか ── 一番効くのは射影のゲート化(L1)であること",
            "attention マップが one-hot になる = **attention とは学習された MUX** という見方",
        ],
        prereq=[
            "[Transformer(論理ゲートGPT)](/tutorials/logic-gpt/) — 登りきった先の完成形",
            "[2. ハイブリッドNNを作る](/tutorials/hybrid/) — 途中の段で止める設計",
        ],
        source="12_logic_ladder_gpt.py",
        next_links=[
            "- [論理ゲートAttentionの4変種ツアー](/tutorials/attention-tour/)"
            " — 登りきった先の Attention 実装を並べて比べる",
            "- [論理ゲートGPT](/tutorials/logic-gpt/) — 全部論理ゲートにした言語モデル",
            "- [2. ハイブリッドNNを作る](/tutorials/hybrid/) — 途中の段で止める設計",
        ],
    ),
    # --- 視覚 ---------------------------------------------------------------
    Tutorial(
        slug="logic-unet",
        group="vision",
        thumb="logic-unet_20_0.png",
        title="論理ゲートU-Netで画素ごとに答える",
        description="ノイズ入りMNISTから領域マスクを復元し、ORプーリング・nearestアップサンプル・"
        "Concat skip がビットを壊さないまま密予測を成立させる仕組みを見る。",
        minutes=6,
        intro=UNET_INTRO,
        learn=[
            "分類ではなく**画素ごとに答える**(密予測)を論理ゲートで組む方法",
            "実数 U-Net の平均プーリング・補間を、OR プーリング・nearest へ置き換える理由",
            "Concat skip を通してもビットが壊れず、中間表現が $\\{0,1\\}$ に閉じることの確認",
        ],
        prereq=[
            "[5. CNNをHDLへエクスポート](/tutorials/cnn-hdl-export/) — 論理ゲート畳み込みの基礎",
            "[1. MNISTを学習する](/tutorials/mnist/)",
        ],
        source="13_logic_unet.py",
        next_links=[
            "- [乗算器ゼロの物体検出](/tutorials/detection/)"
            " — 密予測ヘッドをそのまま検出へ使う",
            "- [論理NNの残差接続](/tutorials/logic-resnet/) — skip接続の厳密な性質",
            "- [5. CNNをHDLへエクスポート](/tutorials/cnn-hdl-export/)",
        ],
    ),
    Tutorial(
        slug="logic-resnet",
        group="vision",
        thumb="logic-resnet_15_0.png",
        title="論理NNの残差接続は恒等を「厳密に」作れる",
        description="OR残差とpass-through初期化で max|y-x| = 0.0 の完全な恒等写像を作り、"
        "その代償である飽和をOR対XORの学習曲線で比較する。",
        minutes=8,
        intro=RESNET_INTRO,
        learn=[
            "OR 残差 + pass-through 初期化で、恒等写像が**厳密に**($\\max|y-x| = 0$)作れること",
            "その代償 ── OR は立ったビットを落とせないので飽和する、というトレードオフ",
            "OR 残差と XOR 残差を同一 seed で比べた学習曲線の読み方",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/)",
            "[LutLayerの数理](/explanation/lut-layer/) — 残差が効く仕組みの背景",
        ],
        source="14_logic_resnet.py",
        next_links=[
            "- [LogicMixer(MLP-Mixer)](/tutorials/logic-mixer/) — 別の混ぜ方(token/channel mixing)",
            "- [論理ゲートU-Net](/tutorials/logic-unet/) — skip接続を密予測で使う",
            "- [LutLayerの数理](/explanation/lut-layer/)",
        ],
    ),
    Tutorial(
        slug="logic-mixer",
        group="vision",
        thumb="logic-mixer_9_0.png",
        title="LogicMixer(MLP-Mixer)— attentionなしで画像を混ぜる",
        description="画像を16トークンに刻み、token-mixingとchannel-mixingをsoft-XOR残差で"
        "交互に積むMLP-Mixerの論理ゲート版をMNISTで動かす。",
        minutes=8,
        intro=MIXER_INTRO,
        learn=[
            "attention を使わずに画像を混ぜる MLP-Mixer を論理ゲートで組む方法",
            "token-mixing(転置して LutStack に通す)と channel-mixing の役割分担",
            "1パッチだけを潰した差分で「情報がどこまで漏れ出すか」を実測する読み方",
        ],
        prereq=[
            "[ResNet(残差接続)の論理化](/tutorials/logic-resnet/) — soft-XOR 残差の素性",
            "[1. MNISTを学習する](/tutorials/mnist/)",
        ],
        source="15_logic_mixer.py",
        next_links=[
            "- [論理ゲートAttentionの4変種ツアー](/tutorials/attention-tour/)"
            " — 混ぜ方を attention でやる場合",
            "- [論理NNの残差接続](/tutorials/logic-resnet/) — soft-XOR残差の素性",
            "- [DWN](/tutorials/dwn/) — 入力の二値化(温度計符号化)を掘る",
        ],
    ),
    Tutorial(
        slug="dwn",
        group="vision",
        thumb="dwn_18_0.png",
        title="DWN — 温度計符号化とn入力LUTで解くMNIST",
        description="ThermometerEncoderと64エントリLUTのDWNを組み、"
        "soft/hardの離散化ギャップがゼロという論理NN界の例外を確かめる。",
        minutes=7,
        intro=DWN_INTRO,
        learn=[
            "`ThermometerEncoder` で実数画素を単調なビット列へ落とす二値化の作法",
            "64 エントリの n 入力 LUT を使う DWN の組み方(test acc 0.958)",
            "**離散化ギャップが 0.0** になる仕組み ── 学習中のモデルと焼かれる回路が同一であること",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/) — 素の LutLayer と比較するための基準",
            "[パラメータ化を選ぶ](/guides/parameterization/)",
        ],
        source="16_dwn.py",
        next_links=[
            "- [パラメータ化を選ぶ](/guides/parameterization/)"
            " — 離散化ギャップがパラメータ化で決まる理由",
            "- [BinaryBrain互換API](/tutorials/binarybrain/)"
            " — 逆に、ギャップが大きく開く実例",
            "- [1. MNISTを学習する](/tutorials/mnist/) — 素の LutLayer との比較",
        ],
    ),
    # --- 生成モデル ---------------------------------------------------------
    Tutorial(
        slug="logic-diffusion",
        group="generative",
        thumb="logic-diffusion_10_0.png",
        title="拡散モデルを丸ごとブール回路にする",
        description="前向き過程のXOR・デノイザのLUT・逆過程のビット組み直しまで、"
        "生成の全工程が乗算なしのブール演算だけで閉じた拡散モデルを動かす。",
        minutes=6,
        intro=DIFFUSION_INTRO,
        learn=[
            "前向き過程 = XOR、デノイザ = LUT、逆過程 = ビット組み直し、という拡散の論理化",
            "自明解(全画素0)に勝てているかを判定する blank ベースラインの置き方",
            "生成モデルの指標を「誇張せずに」報告するとはどういうことか",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/)",
            "[学習曲線の見方](/guides/training-curves/) — ベースラインの取り方",
        ],
        source="17_logic_diffusion.py",
        next_links=[
            "- [二値潜在VAE](/tutorials/logic-vae/) — 別の生成モデルを論理ゲート化する",
            "- [LogicGANの安定化](/tutorials/logic-gan/) — 敵対学習の壊れ方",
            "- [学習曲線の見方](/guides/training-curves/) — ベースラインの取り方",
        ],
    ),
    Tutorial(
        slug="logic-vae",
        group="generative",
        thumb="logic-vae_9_0.png",
        title="画像を96ビットに畳む — 二値潜在VAE",
        description="ガウス潜在の再パラメータ化を捨て、潜在をBernoulliビット列に置き換えたVAEで"
        "196画素の数字を96ビットへ圧縮して復元する。",
        minutes=7,
        intro=VAE_INTRO,
        learn=[
            "ガウス潜在の再パラメータ化を捨て、潜在を Bernoulli ビット列に置き換える方法",
            "196画素の数字を 96 ビットへ畳んで復元する、二値潜在オートエンコーダの構成",
            "潜在1ビットを反転した差分図から「各ビットが何を担当しているか」を読む方法",
        ],
        prereq=[
            "[1. MNISTを学習する](/tutorials/mnist/)",
            "[U-Net(セグメンテーション)](/tutorials/logic-unet/) — デコーダ側の構造",
        ],
        source="18_logic_vae.py",
        next_links=[
            "- [LogicGANの安定化](/tutorials/logic-gan/) — 乱数ビット列から画像を作る",
            "- [拡散モデルをブール回路に](/tutorials/logic-diffusion/)",
            "- [論理ゲートU-Net](/tutorials/logic-unet/) — デコーダ側の構造を掘る",
        ],
    ),
    Tutorial(
        slug="logic-gan",
        group="generative",
        thumb="logic-gan_13_0.png",
        title="乱数ビット列から画像を作る、そして壊す(LogicGAN)",
        description="乱数ビット列を論理回路に通して画像を生成し、"
        "安定版と未調整版を並べて敵対学習の壊れ方を実測で見せる。",
        minutes=8,
        intro=GAN_INTRO,
        learn=[
            "物理乱数や LFSR から供給できる**乱数ビット列**を入力に、論理回路で画像を作る方法",
            "敵対学習を安定させる4つのつまみ(更新比・入力ノイズ・label smoothing・D の学習率)",
            "壊れた GAN の見分け方 ── G loss の発散と D 精度 1.000 の張り付き、モード崩壊",
        ],
        prereq=[
            "[VAE(二値潜在オートエンコーダ)](/tutorials/logic-vae/) — 敵対学習を使わない生成",
            "[学習を回す](/guides/training/)",
        ],
        source="19_logic_gan.py",
        next_links=[
            "- [二値潜在VAE](/tutorials/logic-vae/) — 敵対学習を使わない生成",
            "- [拡散モデルをブール回路に](/tutorials/logic-diffusion/)",
            "- [学習を回す](/guides/training/) — 学習が壊れたときの見立て",
        ],
    ),
    # --- 応用 ---------------------------------------------------------------
    Tutorial(
        slug="attention-tour",
        group="applications",
        thumb="attention-tour_20_0.png",
        title="論理ゲートAttentionの4変種ツアー",
        description="内積とsoftmaxをXNOR類似度とargmaxに置き換えた4つのAttention変種を、"
        "連想想起・位置ルックアップ・ソートの小タスクで比べる。",
        minutes=7,
        intro=ATTENTION_TOUR_INTRO,
        learn=[
            "内積 → XNOR 類似度、softmax → argmax という Attention の置き換え方",
            "4変種(Boolean / MultiQuery / Positional / Pointer)の使い分けと得手不得手",
            "hard attention の重み行列の読み方 ── one-hot 化と、Pointer に現れる置換行列",
        ],
        prereq=[
            "[Transformerを段階的に論理化](/tutorials/logic-ladder/)"
            " — なぜ softmax を捨てても解けるのか",
            "[Transformer(論理ゲートGPT)](/tutorials/logic-gpt/)",
        ],
        source="20_attention_tour.py",
        next_links=[
            "- [Transformerを段階的に論理化](/tutorials/logic-ladder/)"
            " — なぜ softmax を捨てても解けるのかを段階的に確かめる",
            "- [論理ゲートGPT](/tutorials/logic-gpt/) — 変種を言語モデルに載せる",
            "- [LogicMixer(MLP-Mixer)](/tutorials/logic-mixer/) — attention を使わない混ぜ方",
        ],
    ),
    Tutorial(
        slug="detection",
        group="applications",
        thumb="detection_17_0.png",
        title="乗算器ゼロの物体検出(CenterNet)",
        description="LutConv2dとOrPooling2dだけで組んだCenterNetが合成Multi-MNISTの数字を見つけ、"
        "バウンディングボックスを描くまでを最小構成で追う。",
        minutes=7,
        intro=DETECTION_INTRO,
        learn=[
            "`LutConv2d` と `OrPooling2d` だけで CenterNet 型の検出器を組む方法(総ゲート数 784)",
            "heatmap + サイズ回帰から、バウンディングボックスを取り出すまでの後処理",
            "数分の予算に収めるための縮小の作法と、10クラス版・本番レシピへの戻し方",
        ],
        prereq=[
            "[U-Net(セグメンテーション)](/tutorials/logic-unet/) — 密予測ヘッドの基礎",
            "[5. CNNをHDLへエクスポート](/tutorials/cnn-hdl-export/)",
        ],
        source="21_detection.py",
        next_links=[
            "- [論理ゲートU-Net](/tutorials/logic-unet/) — 密予測ヘッドの基礎",
            "- [5. CNNをHDLへエクスポート](/tutorials/cnn-hdl-export/)"
            " — 検出回路をそのままHDLへ",
            "- [クラウドGPU](/guides/cloud-gpu/) — 10クラス本番レシピを回す",
        ],
    ),
    Tutorial(
        slug="binarybrain",
        group="applications",
        thumb="binarybrain_17_0.png",
        title="BinaryBrain互換APIでFPGA向けLUT網を学習する",
        description="RealToBinary / DifferentiableLutBlock / BinaryToReal でMNISTを学習し、"
        "真理値表がまだ0/1に決めきれていない離散化ギャップまで測る。",
        minutes=10,
        intro=BINARYBRAIN_INTRO,
        learn=[
            "`RealToBinary` → `DifferentiableLutBlock` → `BinaryToReal` という "
            "BinaryBrain 流の構成",
            "実数画像を確率的に二値フレームへ変調し、最後に多数決で実数へ戻す仕組み",
            "**まだ焼けない**ことの定量化 ── 真理値表を 0/1 へ丸めたときの崩壊とその主因",
        ],
        prereq=[
            "[DWN(二値NN・温度計符号化)](/tutorials/dwn/) — 逆にギャップが 0 になる例",
            "[パラメータ化を選ぶ](/guides/parameterization/)",
        ],
        source="22_binarybrain.py",
        next_links=[
            "- [DWN](/tutorials/dwn/) — 逆に離散化ギャップがゼロになるパラメータ化",
            "- [パラメータ化を選ぶ](/guides/parameterization/)"
            " — independent / raw と BinaryBrain の関係",
            "- [FPGAに焼く](/guides/fpga/) — 焼く前に何を確かめるべきか",
        ],
    ),
    Tutorial(
        slug="textcnn",
        group="applications",
        thumb="textcnn_22_0.png",
        title="論理ゲート版TextCNNで「n-gram検出器」を育てる",
        description="単語を数えるだけでは解けないバイグラムXOR分類を論理ゲートTextCNNで解き、"
        "各フィルタが特定の2語の検出器へ育つ様子を可視化する。",
        minutes=5,
        intro=TEXTCNN_INTRO,
        learn=[
            "単語頻度では原理的に解けない課題(バイグラム XOR 分類)の作り方と、その意味",
            "論理ゲート版 TextCNN の構成と、soft/hard ともに 1.000 で解けること",
            "学習後のフィルタを覗いて「特定の2語の検出器」へ育っていることを確かめる方法",
        ],
        prereq=[
            "[付録: XORで仕組みを最小で覗く](/tutorials/xor/) — 同じ XOR 構造の最小版",
            "[1. MNISTを学習する](/tutorials/mnist/)",
        ],
        source="23_textcnn.py",
        next_links=[
            "- [付録: XORで仕組みを最小で覗く](/tutorials/xor/)"
            " — このページのXOR構造を4サンプルまで縮めて見る",
            "- [論理ゲートGPT](/tutorials/logic-gpt/) — 分類ではなく生成をやる",
            "- [因果convolutionを論理ゲートで](/tutorials/logic-wavenet/)"
            " — 同じ1次元畳み込みを系列予測へ",
        ],
    ),
    Tutorial(
        slug="har",
        group="applications",
        thumb="har_12_0.png",
        title="行動認識(UCI-HAR、スマホ加速度)",
        description="スマホの加速度・ジャイロの生信号を温度計符号化して論理ゲート網で6行動を分類し、"
        "学習した回路をC99へ書き出してPyTorchと整数一致を確かめる。",
        minutes=10,
        intro=HAR_INTRO,
        learn=[
            "実数のセンサ信号を温度計符号化(閾値は訓練データの分位点)で 0/1 の入力線にする方法",
            "9ch × 128 点 × 4 閾値 = 4,608 ビットを受ける論理回路で、"
            "未知の被験者の行動を分類する流れ",
            "同じ回路を C99 へ運び、テスト窓すべてで PyTorch hard eval と整数一致させる確かめ方",
        ],
        prereq=[
            "[4. デプロイする](/tutorials/deploy-mnist/) — 後半の出口と同じ手順",
            "[1. MNISTを学習する](/tutorials/mnist/) — LutLayer + GroupSum の基本",
        ],
        source="27_har.py",
        next_links=[
            "- [4. デプロイする](/tutorials/deploy-mnist/)"
            " — 同じ出口を画像分類で通す",
            "- [DWN(二値NN・温度計符号化)](/tutorials/dwn/) — 温度計符号化を層として組み込む流儀",
            "- [3. HDLへエクスポート](/tutorials/hdl-export/) — 同じ回路を Verilog へ",
            "- [論理ゲート版TextCNN](/tutorials/textcnn/) — 系列データを論理ゲートで解く別の例",
            "- [デモモデルを用意する](/guides/deploy-models/)"
            " — この行動認識モデルを含むデモバンドルの作り方",
            "- [ESP32 に載せる](/guides/deploy-esp32/) — 行動認識をマイコン(ESP32)で動かす",
            "- [iPhone に載せる](/guides/deploy-ios/) — 行動認識を iPhone アプリへ同梱する",
        ],
    ),
]

# --- py:percent への変換 -------------------------------------------------


def _leading_comment_lines(lines: list[str], lineno: int) -> int:
    """`lineno`(1始まり)の直上に連なるトップレベルコメント行の開始行を返す。"""
    start = lineno
    i = lineno - 2  # 0始まりで1つ上の行
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#") and not lines[i].startswith((" ", "\t")):
            start = i + 1
            i -= 1
            continue
        break
    return start


def _docstring_to_markdown(doc: str) -> str:
    """モジュール docstring を py:percent の markdown セルへ整形する。"""
    body = "\n".join(f"# {line}".rstrip() for line in doc.strip().splitlines())
    return f"# %% [markdown]\n{body}\n"


def script_to_percent(path: Path) -> str:
    """examples/*.py を py:percent 形式へ変換する(ファイルは書き換えない)。

    セル分割の規則:
      - モジュール docstring → markdown セル
      - 先頭の import 群 → 1つのコードセル
      - 直上にトップレベルコメントを持つ文 / 関数・クラス定義 → 新しいコードセル
      - それ以外は直前のセルに追記
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    chunks: list[str] = []
    doc = ast.get_docstring(tree)
    body = list(tree.body)
    if doc is not None:
        chunks.append(_docstring_to_markdown(doc))
        body = body[1:]

    cells: list[tuple[int, int]] = []  # (開始行, 終了行) 1始まり・両端含む
    prev_was_import = False
    prev_was_def = False
    for node in body:
        end = node.end_lineno or node.lineno
        # デコレータ付きの def/class は `node.lineno` が `def` 行を指す(Python 3.8+)。
        # そのまま切ると `@torch.no_grad()` などが**セルから落ちる**ので、
        # デコレータ行まで含めて1セルにする。
        first = node.lineno
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            first = min(d.lineno for d in decorators)
        start = _leading_comment_lines(lines, first)
        is_import = isinstance(node, ast.Import | ast.ImportFrom)
        has_comment = start < node.lineno
        is_def = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        new_cell = (
            not cells
            or has_comment
            or is_def
            or prev_was_def
            or (prev_was_import and not is_import)
        )
        if new_cell:
            cells.append((start, end))
        else:
            cells[-1] = (cells[-1][0], end)
        prev_was_import = is_import
        prev_was_def = is_def

    for start, end in cells:
        code = "\n".join(lines[start - 1 : end]).strip("\n")
        chunks.append(f"# %%\n{code}\n")

    return "\n".join(chunks)


# --- ノートブック実行 ----------------------------------------------------


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"コマンド失敗({proc.returncode}): {' '.join(cmd)}")


def is_stderr_stream(output: dict) -> bool:
    """stderr の stream 出力か。

    tqdm の進捗バー(\\r 入り)や警告は、nbconvert が ```text フェンスの外へ生の
    インデント行として書き出し、ドリフト検査を壊す(新環境でのデータセット取得で発生)。
    教材は結果を print()(stdout)で見せる方針なので stderr はページに載せない。
    """
    return output.get("output_type") == "stream" and output.get("name") == "stderr"


def strip_execute_results(nb_path: Path) -> None:
    """実行済みノートブックから execute_result 出力を取り除く(その場で書き戻す)。"""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        cell["outputs"] = [
            o
            for o in outputs
            if o.get("output_type") != "execute_result" and not is_stderr_stream(o)
        ]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")


def to_ipynb(percent_src: str, name: str, work: Path) -> Path:
    """py:percent ソースを **未実行** の .ipynb へ変換し、そのパスを返す。"""
    py_path = work / f"{name}.py"
    py_path.write_text(percent_src, encoding="utf-8")

    # jupytext: py:percent → .ipynb
    # カーネルは ipykernel が提供する python3(= 実行中の環境)を明示する。
    # `--set-kernel -` は「実行中の python に一致する登録済みカーネル」を要求するため、
    # uv の一時環境では解決できず使わない。
    _run(
        [
            "jupytext",
            "--to",
            "ipynb",
            "--set-kernel",
            "python3",
            str(py_path),
        ],
        cwd=work,
    )
    return work / f"{name}.ipynb"


def write_clean_notebook(spec: Tutorial, nb_path: Path) -> Path:
    """未実行 .ipynb を `notebooks/` へ配置用に整形して書き出す。

    - 先頭に環境判定インストールセル(`INSTALL_CELL`)を差し込む
    - 出力・実行回数を空にする(読者が自分で実行する前提)
    - 環境依存メタデータ(jupytext のバージョン・python のマイナーバージョン等)を
      落として `NOTEBOOK_METADATA` に固定する ⇒ 再生成しても差分が出ない

    Returns:
        書き出した notebooks/*.ipynb のパス
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = [
        {
            "cell_type": "code",
            # nbformat 4.5 以降はセル id が必須。固定値にして再生成の差分を防ぐ。
            "id": "setup-install",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": INSTALL_CELL.splitlines(keepends=True),
        }
    ]
    for i, cell in enumerate(nb.get("cells", [])):
        cell["metadata"] = {}
        # jupytext はセル id をランダム生成するので、連番へ振り直して決定的にする。
        cell["id"] = f"cell-{i:03d}"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    out = {
        "cells": cells,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 5),
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    dest = NOTEBOOKS_DIR / spec.notebook
    dest.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return dest


# インライン図の出力解像度(dpi)。既定の 100 だと拡大表示で文字がつぶれるため引き上げる。
# figsize(インチ)は各サンプルのままなので**ページ上の表示サイズは変わらず**、
# 画素数だけが増える(ライトボックスで拡大したときに読める)。
FIGURE_DPI = 200

# 実行直前だけ差し込む設定セル(nbconvert に渡す前に取り除くのでページには出ない)。
FIGURE_DPI_CELL = f"""\
import matplotlib

matplotlib.rcParams["figure.dpi"] = {FIGURE_DPI}
matplotlib.rcParams["savefig.dpi"] = {FIGURE_DPI}
get_ipython().run_line_magic(
    "config",
    "InlineBackend.print_figure_kwargs = {{'dpi': {FIGURE_DPI}, 'bbox_inches': 'tight'}}",
)
"""


def execute_notebook(nb_path: Path, out_path: Path) -> None:
    """ノートブックを実行して `out_path` へ書き出す(カーネルの作業ディレクトリ=ROOT)。

    `jupyter nbconvert --execute` の CLI は **ノートブックが置かれているディレクトリ**を
    カーネルの cwd にする(プロセスの cwd ではない)。ここではノートブックを一時ディレクトリ
    に作るため、CLI に任せると examples 側の相対パス解決が全部壊れる:

      - `data-shakespeare/` のコーパスが見つからずフォールバックへ落ちる(07)
      - `sys.path` に足す `experiments/` が見つからず ImportError(21 / 23)
      - ノートブックでは `__file__` が無いので `Path.cwd()` にフォールバックするが、
        それが一時ディレクトリを指してしまう

    そこで nbclient を直接使い、`resources['metadata']['path']` で **カーネルの cwd を
    ROOT に固定**する。examples をリポジトリルートから `uv run python examples/XX.py` で
    実行したときと同じ条件になる。
    """
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(nb_path, as_version=4)
    # 新環境でのデータセット取得時に tqdm の進捗バーがページへ混入しないよう、
    # カーネル起動前に無効化する(tqdm は import 時に TQDM_DISABLE を読む)。
    os.environ.setdefault("TQDM_DISABLE", "1")
    # 図の解像度を上げる設定セルを先頭に差し込んで実行し、実行後に取り除く
    # (ページには出さない・notebooks/ のクリーン版にも影響しない)。
    nb.cells.insert(0, nbformat.v4.new_code_cell(FIGURE_DPI_CELL))
    NotebookClient(
        nb,
        timeout=EXECUTE_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    del nb.cells[0]
    nbformat.write(nb, out_path)


def execute_to_markdown(name: str, work: Path) -> tuple[str, Path, float]:
    """`work/{name}.ipynb` を実行し、(Markdown本文, 添付ファイルdir, 実行秒数) を返す。"""
    # 1段目: 実行して .ipynb を得る(カーネルの cwd は ROOT に固定される)。
    started = time.perf_counter()
    execute_notebook(work / f"{name}.ipynb", work / f"{name}.executed.ipynb")
    elapsed = time.perf_counter() - started

    # セル末尾の式の値(execute_result)を落とす。教材コメントを増やすとセルの切れ目が
    # 変わり、`model.eval()` や `save_ir(...)` の戻り値がノイズ出力として混ざるため。
    # サンプルは結果を必ず print() で見せる方針なので、stream と display_data(図)だけ残す。
    strip_execute_results(work / f"{name}.executed.ipynb")

    # nbconvert(2段目): Markdown 化(出力・図を埋め込む)。
    _run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            "--output",
            name,
            "--output-dir",
            str(work),
            str(work / f"{name}.executed.ipynb"),
        ],
        cwd=ROOT,
    )
    return (work / f"{name}.md").read_text(encoding="utf-8"), work / f"{name}_files", elapsed


# --- Markdown 後処理 -----------------------------------------------------

# `torch.manual_seed(0)` のようにセル末尾の値がそのまま execute_result になる場合の
# ノイズ(メモリアドレス入り repr)。再生成のたびに差分が出るのでページから落とす。
_REPR_NOISE = re.compile(r"^<[^>\n]+ at 0x[0-9a-f]+>$")


# nbconvert が display_data(図)を書き出す行。同じセルの stream 出力が図をまたいで
# 続くことがあるので、出力ゾーンの判定ではこの行を「地の文」とみなさない。
_IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$")


def normalize_markdown(md: str) -> str:
    """nbconvert の Markdown を Starlight 向けに整える。

    - 4スペースインデントの stream 出力を ```text フェンスへ変換(MDX/整形の揺れを避ける)
    - 3連続以上の空行を2行に畳む

    重要: 変換対象は **python コードセル直後の出力ゾーンに限る**。
    nbconvert は Markdown セル(= examples の module docstring やコメント由来の地の文)を
    そのまま出力するため、docstring 内の字下げ行(構成図・箇条書きの継続行など)まで
    「4スペース = stream 出力」と解釈すると導入部が出力ブロックに化ける(DOCS-4で発生)。
    出力ゾーンは ```python フェンスの閉じで開始し、空行・図の行・変換済み出力ブロックの
    間だけ継続する。地の文が1行でも現れたら閉じる。
    """
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_fence = False
    opened_lang = ""
    # 直前が python コードセルの出力ゾーンか(= 4スペースブロックを出力とみなしてよいか)
    in_output_zone = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_fence:
                # フェンスを閉じた。python セルなら以降が出力ゾーン。
                in_output_zone = opened_lang == "python"
            else:
                info = line[3:].strip()
                opened_lang = info.split(maxsplit=1)[0] if info else ""
                in_output_zone = False
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        # nbconvert の stream/execute_result 出力は4スペースインデントで書かれる。
        # (先頭がさらに空白の表形式もあるため、`4スペース + 何か非空白` で判定する)
        if not in_fence and in_output_zone and line.startswith("    ") and line.strip():
            block: list[str] = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                # 出力ブロックの末尾の空行は取り込まない
                if not lines[i].strip():
                    j = i
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j >= len(lines) or not lines[j].startswith("    "):
                        break
                block.append(lines[i][4:] if lines[i].startswith("    ") else "")
                i += 1
            while block and not block[-1].strip():
                block.pop()
            if len(block) == 1 and _REPR_NOISE.match(block[0].strip()):
                continue
            out.append("```text title=\"出力\"")
            out.extend(block)
            out.append("```")
            continue
        # 地の文(Markdownセル)が1行でも現れたら出力ゾーンは終わり。
        # 空行と図の行(display_data)は同一セルの出力の途中にも挟まるので継続扱い。
        if line.strip() and not _IMAGE_LINE.match(line):
            in_output_zone = False
        out.append(line)
        i += 1

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# --- 章見出し・散文マーカー(TOC-1) --------------------------------------
#
# 生成ページに h2/h3 が1つも無いと Starlight の右目次(h2/h3 から自動生成)が
# 「概要 / 次に読む」しか出ない。そこで **コードセルの先頭に置いたマーカーコメント**を
# Markdown の見出し・段落へ変換する。変換は nbconvert 済み Markdown への**後処理だけ**で
# 行うため、セル分割・図番号には一切影響しない(= 図の改番が起きない)。
#
#   # === 章名 ===        → `## 章名`(右目次に出る大章)
#   # --- 小見出し ---    → `### 小見出し`
#   # > 散文              → コードフェンス**外**の Markdown 段落(解説文)
#
# `---` と `>` は **その example が `# === … ===` を1つ以上持つときだけ**有効にする
# (ファイル単位のオプトイン)。既存 example は `# --- … ---` をセル先頭コメントとして
# 多用しているので、無条件に有効化すると全ページに差分が出るため。
# マーカーが1つも無い example では出力が**完全に一致**する(ドリフトゼロ)。
_H2_MARKER = re.compile(r"^#\s*={3,}\s*(?P<title>.+?)\s*={3,}\s*$")
_H3_MARKER = re.compile(r"^#\s*-{3,}\s*(?P<title>.+?)\s*-{3,}\s*$")
# 散文行。`# > 本文` / 空段落区切りの `# >` の両方を受ける。
_PROSE_MARKER = re.compile(r"^#\s*>(?P<text>\s.*|)$")


def _marker_title(match: re.Match[str]) -> str | None:
    """マーカーの見出しテキストを取り出す(罫線コメントなら None)。"""
    title = match.group("title").strip()
    # `# ----------------` のような純粋な罫線は見出しではない。
    if not title or not title.strip("-=~ "):
        return None
    return title


def _prose_block(prose: list[str]) -> list[str]:
    """`# >` 行の集まりを Markdown 段落(行のリスト)へ整える。"""
    while prose and not prose[0].strip():
        prose.pop(0)
    while prose and not prose[-1].strip():
        prose.pop()
    return prose


def _split_leading_markers(body: list[str]) -> tuple[list[list[str]], list[str]]:
    """コードセル本体の**先頭から連続する**マーカー行を切り出す。

    Returns:
        (フェンス前に出す Markdown ブロックの列, フェンスへ残すコード行)
        ブロックは**書かれた順**に並ぶので、`# > …` → `# === … ===` の順に書けば
        「前の章の締めの散文 → 次の章の見出し」がその順で出る。
    """
    blocks: list[list[str]] = []
    prose: list[str] = []
    i = 0
    while i < len(body):
        line = body[i]
        prose_match = _PROSE_MARKER.match(line)
        if prose_match is not None:
            text = prose_match.group("text")
            prose.append((text[1:] if text.startswith(" ") else text).rstrip())
            i += 1
            continue
        if prose:
            blocks.append(_prose_block(prose))
            prose = []
        h2 = _H2_MARKER.match(line)
        if h2 is not None:
            title = _marker_title(h2)
            if title is None:
                break
            blocks.append([f"## {title}"])
            i += 1
            continue
        h3 = _H3_MARKER.match(line)
        if h3 is not None:
            title = _marker_title(h3)
            if title is None:
                break
            blocks.append([f"### {title}"])
            i += 1
            continue
        break
    if prose:
        blocks.append(_prose_block(prose))
    return [b for b in blocks if b], body[i:]


def _iter_python_fences(lines: list[str]) -> list[tuple[int, int]]:
    """```python フェンスの (開始行index, 閉じ行index) を列挙する。"""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "```python":
            end = i + 1
            while end < len(lines) and lines[end].strip() != "```":
                end += 1
            spans.append((i, end))
            i = end + 1
            continue
        i += 1
    return spans


def insert_chapter_headings(md: str) -> str:
    """コードセル先頭のマーカーコメントを見出し・段落へ変換する。

    - `# === 章名 ===` → `## 章名` / `# --- 小見出し ---` → `### 小見出し`
    - `# > 本文` → コードフェンス外の Markdown 段落
    - マーカー行は**表示コードから除去**する(`examples/*.py` と
      `notebooks/*.ipynb` には残るので、読者が実行するノートブックは現状のまま)
    - `# === … ===` を1つも持たない example は**素通し**(出力が完全一致)
    """
    lines = md.splitlines()
    spans = _iter_python_fences(lines)
    parsed = [(start, end, *_split_leading_markers(lines[start + 1 : end])) for start, end in spans]
    # オプトイン判定: h2 マーカーを持つページだけを変換対象にする。
    if not any(block[0].startswith("## ") for _, _, blocks, _ in parsed for block in blocks):
        return md

    # 見出しは HTML へ落ちるので、生タグと解釈されうる `<` だけは早期に落とす
    # (黙って崩れたページを出すより、生成を止めて書き手に直させる)。
    for _, _, blocks, _ in parsed:
        for block in blocks:
            if block[0].startswith("#") and "<" in block[0]:
                raise SystemExit(f"見出しマーカーに < は使えません: {block[0]!r}")

    out: list[str] = []
    cursor = 0
    for start, end, blocks, rest in parsed:
        out.extend(lines[cursor:start])
        for block in blocks:
            if out and out[-1].strip():
                out.append("")
            out.extend(block)
        while rest and not rest[0].strip():
            rest.pop(0)
        if rest:
            if out and out[-1].strip():
                out.append("")
            out.append(lines[start])
            out.extend(rest)
            out.append("```")
        cursor = end + 1
    out.extend(lines[cursor:])

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    return text.strip() + "\n"


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _learn_block(spec: Tutorial) -> str:
    """ページ冒頭の「このページで学ぶこと / 前提」ブロック(未設定なら空文字)。

    全チュートリアルで**同じ見た目**にすることが目的なので、書式はここだけで決める。
    ページ側(examples の docstring)に同種の箇条書きを重ねて書かないこと。
    """
    if not spec.learn:
        return ""
    if len(spec.learn) > 3:
        raise SystemExit(f"{spec.slug}: learn は3点以内にすること(現在 {len(spec.learn)} 点)")
    items = "\n".join(f"{i}. {text}" for i, text in enumerate(spec.learn, 1))
    prereq = (
        "\n".join(f"- {line}" for line in spec.prereq)
        if spec.prereq
        else "- なし ── このページから読み始められる"
    )
    return f":::tip[このページで学ぶこと]\n{items}\n\n**前提**\n\n{prereq}\n:::\n\n"


def render_page(spec: Tutorial, body: str, source_rel: str | None) -> str:
    """frontmatter・概要・Colabバッジ・「次に読む」を付けた最終ページを組む。"""
    source_line = (
        f"元スクリプト: [`{source_rel}`]({GITHUB_BLOB}/{source_rel})"
        if source_rel
        else "元スクリプト: なし(`examples/01_xor.py` と `examples/04_export_hdl.py` を合成)"
    )
    # Colab バッジ: リポジトリが public になるまでリンクは無効なので、URL は data 属性に
    # 保持して <a> にはしない(リンク切れ検査に掛からず、公開後に href 化するだけで有効になる)。
    # 指す先は notebooks/*.ipynb(実行できる形。examples/*.py は Colab では開けない)。
    notebook_rel = f"notebooks/{spec.notebook}"
    badge = (
        "<!-- Colabバッジ: リポジトリ public 化後、この span を data-colab-url を href に\n"
        "     持つ <a> へ差し替えると「Colabで開く」が有効になる。 -->\n"
        '<span class="colab-badge" data-colab-url="'
        f'{COLAB_BASE}/{notebook_rel}">▶ Colabで開く(リポジトリ公開後に有効化)</span>\n'
    )
    notebook_line = (
        f"手元で試す: [`{notebook_rel}`]({GITHUB_BLOB}/{notebook_rel})"
        "(このページと同じ内容の未実行ノートブック。"
        "`uv run --group docs jupyter lab notebooks/` で開ける)"
    )
    generated_note = (
        "<!-- このファイルは scripts/build_tutorials.py が自動生成しています。\n"
        f"     直接編集せず、{source_rel or 'scripts/build_tutorials.py の FULL_PIPELINE_SOURCE'}"
        " を編集して再生成してください。 -->"
    )
    header = f"""---
title: {_yaml_quote(spec.title)}
description: {_yaml_quote(spec.description)}
---

{generated_note}

{spec.intro}

{_learn_block(spec)}:::note[このページについて]
所要時間の目安: **約{spec.minutes}分** / 実行環境: CPU のみ(GPU 不要)
{source_line}
{notebook_line}
コード出力はすべて **実際に実行した結果**をそのまま埋め込んでいます。
:::

{badge}"""
    footer = ""
    if spec.next_links:
        footer = "## 次に読む\n\n" + "\n".join(spec.next_links) + "\n"
    return header + "\n" + body.rstrip() + "\n\n" + footer


# --- 例ギャラリー(tutorials/index.mdx) ---------------------------------


def _thumb_for(spec: Tutorial) -> str | None:
    """`{slug}_files/` から使うサムネイルのファイル名を返す(無ければ None)。

    レジストリの `thumb` を第一候補にし、図番号のズレで消えていた場合は同ディレクトリの
    先頭の png へフォールバックする(ギャラリーが図の再生成で壊れないようにする)。
    """
    files_dir = OUT_DIR / f"{spec.slug}_files"
    if not files_dir.is_dir():
        return None
    if spec.thumb and (files_dir / spec.thumb).exists():
        return spec.thumb
    candidates = sorted(p.name for p in files_dir.glob("*.png"))
    if spec.thumb and candidates:
        print(
            f"[warn ] {spec.slug}: thumb={spec.thumb} が無いので "
            f"{candidates[0]} を使います(thumb を更新してください)",
            flush=True,
        )
    return candidates[0] if candidates else None


def _ident(slug: str) -> str:
    """slug を JS の識別子へ(`logic-gpt` → `thumbLogicGpt`)。"""
    return "thumb" + "".join(part.capitalize() for part in re.split(r"[^0-9a-zA-Z]+", slug))


# ギャラリーの見た目は**ここ1箇所**で定義する(生成 mdx に <style> として埋め込む)。
# Tailwind のユーティリティに頼らないのは、`aspect-video` が未定義変数に解決されて
# サムネイルが原寸比のまま伸び、カード高さが揃わなかったため(実測で確認)。
GALLERY_STYLE = """\
.md-gal-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: 1fr;
}
@media (min-width: 640px) {
  .md-gal-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (min-width: 1024px) {
  .md-gal-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
.md-gal-card {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  border: 1px solid var(--sl-color-gray-5);
  border-radius: 0.75rem;
  background: var(--sl-color-black);
  text-decoration: none;
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.md-gal-card:hover {
  transform: translateY(-3px);
  border-color: var(--sl-color-accent);
  box-shadow: 0 0.75rem 1.5rem rgb(0 0 0 / 25%);
}
/* サムネイルは 16:10 の固定枠。図は切らずに contain で全体を見せる。
   matplotlib 図は背景が白なので、枠自体を白い「プレビュー板」として意図的に見せる。 */
.md-gal-thumb {
  display: block;
  aspect-ratio: 16 / 10;
  margin: 0.625rem 0.625rem 0;
  padding: 0.375rem;
  border-radius: 0.5rem;
  background: #fff;
}
.md-gal-thumb img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.md-gal-body {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 0.35rem;
  padding: 0.875rem 1rem 1rem;
}
.md-gal-title {
  font-size: 0.95rem;
  font-weight: 600;
  line-height: 1.4;
  color: var(--sl-color-white);
}
.md-gal-desc {
  flex: 1;
  font-size: 0.8rem;
  line-height: 1.65;
  color: var(--sl-color-gray-3);
}
.md-gal-meta {
  font-size: 0.72rem;
  color: var(--sl-color-gray-4);
}
.md-gal-meta code { font-family: var(--__sl-font-mono, monospace); }
"""


def render_gallery() -> str:
    """TUTORIALS から例ギャラリー(`tutorials/index.mdx`)を組み立てる。

    scikit-learn の Examples ギャラリーと同じ発想で、**各ページの実際の生成図**を
    サムネイルにしたカードを群ごとに並べる。手書きの一覧を持たないので、
    レジストリに1本足せばギャラリーも自動で追従する。
    """
    thumbs = {spec.slug: _thumb_for(spec) for spec in TUTORIALS}
    # サムネイルは astro:assets の <Image> で出す。素の `<img src={imported.src}>` だと
    # 最適化前の元ファイルが emit される前提になり、**同一内容の PNG が複数チュートリアルで
    # 使われている場合に Astro が資産を1本へ重複排除する**結果、webp 版だけが残って
    # PNG の URL が 404 になる(xor と full-pipeline で実際に発生した)。
    imports = ["import { Image } from 'astro:assets';"] + [
        f"import {_ident(slug)} from './{slug}_files/{name}';"
        for slug, name in thumbs.items()
        if name
    ]

    lines = [
        "---",
        "title: 'チュートリアル'",
        "description: '実行済みチュートリアル"
        f"{len(TUTORIALS)}本の例ギャラリー。図をクリックしてページへ。'",
        "sidebar:",
        "  order: 0",
        "---",
        "",
        "{/* このファイルは scripts/build_tutorials.py が TUTORIALS から自動生成しています。",
        "    直接編集せず、レジストリ(slug / description / group / thumb)を編集して",
        "    再生成してください。 */}",
        "",
        *imports,
        "",
        # ギャラリーの見た目(GALLERY_STYLE)はこのページ専用。MDX なので JSX 式で埋める。
        "<style>{`",
        GALLERY_STYLE.rstrip(),
        "`}</style>",
        "",
        f"すべてのページは `examples/*.py` を**実際に実行**して生成しています"
        f"(全{len(TUTORIALS)}本・いずれも CPU のみで完走)。"
        "図・数値・生成テキストはその実行結果そのものです。",
        "",
        "各ページの末尾には「次に読む」があるので、群をまたいで辿れます。",
        "手元で動かすなら [`notebooks/`](https://github.com/tsukushi-gate/Griffith/"
        "blob/main/notebooks) の未実行ノートブックが同じ内容です。",
        "",
    ]

    for key, title, lead in GALLERY_GROUPS:
        members = [spec for spec in TUTORIALS if spec.group == key]
        if not members:
            continue
        lines += [f"## {title}", "", lead, "", '<div class="not-content">', ""]
        lines.append('  <div class="md-gal-grid">')
        for spec in members:
            thumb = thumbs[spec.slug]
            lines.append("")
            lines.append(f'    <a href="/tutorials/{spec.slug}/" class="md-gal-card">')
            if thumb:
                lines += [
                    # 図はアスペクト比がまちまち(横長の学習曲線〜縦長の生成グリッド)なので、
                    # 16:10 の固定枠に contain で収める(切らない・カード高さが揃う)。
                    '      <span class="md-gal-thumb">',
                    f"        <Image src={{{_ident(spec.slug)}}} "
                    f'alt="{_escape_attr(spec.title)} の生成図" loading="lazy" />',
                    "      </span>",
                ]
            lines += [
                '      <span class="md-gal-body">',
                f'        <span class="md-gal-title">{_escape_text(spec.title)}</span>',
                '        <span class="md-gal-desc">'
                f"{_escape_text(spec.description)}</span>",
                # JSX の子要素なので Markdown の記法(バッククォート等)は効かない。
                # 等幅にしたい部分はクラスで表現する。
                '        <span class="md-gal-meta">'
                f"約{spec.minutes}分 · "
                + (
                    f"<code>examples/{spec.source}</code>"
                    if spec.source
                    else "合成サンプル"
                )
                + "</span>",
                "      </span>",
                "    </a>",
            ]
        lines += ["", "  </div>", "", "</div>", ""]

    return "\n".join(lines).rstrip() + "\n"


def _escape_text(text: str) -> str:
    """MDX のテキストノードとして安全にする(JSX の式展開・タグ誤認を防ぐ)。"""
    return text.replace("{", "&#123;").replace("}", "&#125;").replace("<", "&lt;")


def _escape_attr(text: str) -> str:
    return _escape_text(text).replace('"', "&quot;")


def write_gallery() -> Path:
    """例ギャラリーを書き出してパスを返す。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_PATH.write_text(render_gallery(), encoding="utf-8")
    return GALLERY_PATH


# --- example → 公開シンボルの索引 ---------------------------------------


def _public_symbols() -> list[str]:
    """`griffith_logic/__init__.py` の `_SYMBOL_MODULES` のキー(公開シンボル名)。"""
    source = (ROOT / "griffith_logic" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SYMBOL_MODULES" for t in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise SystemExit("griffith_logic/__init__.py: _SYMBOL_MODULES が見つからない")


def _symbols_in_source(source: str, public: set[str]) -> set[str]:
    """1つの example ソースが使っている公開シンボルを AST で拾う。

    拾うのは2経路だけ(どちらも「そのAPIを実際に呼んでいる」ことの証拠になる):
      - `from griffith_logic[...] import X`(サブモジュール直輸入も含む)
      - `import griffith_logic as ml` + `ml.X` の属性アクセス
    """
    tree = ast.parse(source)
    used: set[str] = set()
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == PACKAGE or module.startswith(f"{PACKAGE}."):
                used |= {a.name for a in node.names if a.name in public}
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}."):
                    aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name) and value.id in aliases and node.attr in public:
                used.add(node.attr)
    return used


def collect_symbol_usage() -> dict[str, list[dict[str, str]]]:
    """公開シンボル名 → それを使っているチュートリアル(slug/title/source)の一覧。

    `scripts/gen_api_docs.py` はこの関数を直接呼び、APIページの各シンボル末尾へ
    「このAPIを使っている例」を生成する。example / シンボルを足せば自動で追従する。
    """
    public = set(_public_symbols())
    usage: dict[str, list[dict[str, str]]] = {}
    for spec in TUTORIALS:
        if spec.source is not None:
            source = (EXAMPLES_DIR / spec.source).read_text(encoding="utf-8")
        else:
            source = spec.synth
        entry = {
            "slug": spec.slug,
            "title": spec.title,
            "url": f"/tutorials/{spec.slug}/",
            "source": f"examples/{spec.source}" if spec.source else "",
        }
        for name in sorted(_symbols_in_source(source, public)):
            usage.setdefault(name, []).append(entry)
    return usage


def write_symbols_json() -> Path:
    """example → 公開シンボルの索引を JSON で書き出す(人・外部ツール向け)。"""
    usage = collect_symbol_usage()
    SYMBOLS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SYMBOLS_JSON.write_text(
        json.dumps(usage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SYMBOLS_JSON


# --- パイプライン --------------------------------------------------------


def build(spec: Tutorial, work_root: Path, execute: bool = True) -> float:
    """1ページ分を生成する(notebooks/*.ipynb は execute に関わらず必ず出力)。"""
    if spec.source is not None:
        src_path = EXAMPLES_DIR / spec.source
        if not src_path.exists():
            raise SystemExit(f"元スクリプトが見つかりません: {src_path}")
        percent_src = script_to_percent(src_path)
        source_rel = f"examples/{spec.source}"
    else:
        percent_src = spec.synth
        source_rel = None

    work = work_root / spec.slug
    work.mkdir(parents=True, exist_ok=True)
    nb_path = to_ipynb(percent_src, spec.slug, work)
    # まず「未実行のクリーンなノートブック」を notebooks/ へ出す(実行不要・決定的)。
    write_clean_notebook(spec, nb_path)
    if not execute:
        return 0.0

    md, attach_dir, elapsed = execute_to_markdown(spec.slug, work)
    # 章見出し・散文マーカーの展開は normalize_markdown の**後**に行う
    # (normalize の「出力ゾーン」判定に一切触れないため。docs-authoring スキル §5)。
    page = render_page(spec, insert_chapter_headings(normalize_markdown(md)), source_rel)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{spec.slug}.md").write_text(page, encoding="utf-8")

    # 図がある場合のみ添付ディレクトリを同梱する(現サンプルは図なし)
    if attach_dir.is_dir() and any(attach_dir.iterdir()):
        dest = OUT_DIR / attach_dir.name
        shutil.copytree(attach_dir, dest, dirs_exist_ok=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="生成する slug(複数指定可)")
    parser.add_argument("--check", action="store_true", help="実行せず対象一覧のみ表示")
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="ノートブック(notebooks/*.ipynb)だけを再生成する(実行しないので高速・決定的)",
    )
    parser.add_argument(
        "--gallery-only",
        action="store_true",
        help="例ギャラリー(tutorials/index.mdx)とシンボル索引だけを再生成する",
    )
    args = parser.parse_args()

    if args.gallery_only:
        print(f"ギャラリー: {write_gallery()}")
        print(f"シンボル索引: {write_symbols_json()}")
        return

    targets = TUTORIALS
    if args.only:
        wanted = set(args.only)
        targets = [t for t in TUTORIALS if t.slug in wanted]
        unknown = wanted - {t.slug for t in TUTORIALS}
        if unknown:
            raise SystemExit(f"未知の slug: {sorted(unknown)}")

    if args.check:
        for spec in targets:
            print(
                f"{spec.slug:<16} ← {spec.source or '(合成)'}"
                f"  → {OUT_DIR / (spec.slug + '.md')}"
                f" / {NOTEBOOKS_DIR / spec.notebook}"
            )
        return

    results: list[tuple[str, float]] = []
    with tempfile.TemporaryDirectory(prefix="build_tutorials_") as tmp:
        work_root = Path(tmp)
        for spec in targets:
            print(f"[build] {spec.slug} ← {spec.source or '(合成ソース)'}", flush=True)
            elapsed = build(spec, work_root, execute=not args.no_execute)
            results.append((spec.slug, elapsed))
            print(f"[done ] {spec.slug}: 実行 {elapsed:.1f}s", flush=True)

    # ギャラリーとシンボル索引はレジストリ由来なので、--only でも常に全体を作り直す。
    print(f"[gallery] {write_gallery()}", flush=True)
    print(f"[symbols] {write_symbols_json()}", flush=True)

    if args.no_execute:
        print(f"\nnotebooks/ のみ再生成しました({len(results)}件・実行なし)")
        return

    print("\n=== 実行時間サマリ ===")
    for slug, elapsed in results:
        print(f"{slug:<16} {elapsed:6.1f}s")
    print(f"{'合計':<16} {sum(e for _, e in results):6.1f}s")


if __name__ == "__main__":
    main()
