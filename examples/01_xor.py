"""XOR学習の最小例。

微分可能論理ゲート層(LutLayer)でXORをsoft学習し、
離散回路(hard eval)としても100%になることを確認する。

このライブラリの中心的な主張——「連続緩和で勾配学習した論理ゲート網は、
argmaxで離散化して純粋なBoolean回路に落としても精度を失わない」——を、
4パターン・40行という最小構成で体験するためのスクリプト。XORは1層の線形分離では
解けない最小の問題なので、「本当に非線形な論理を学習できたか」の試金石になる。

**回路の全体像(全32ゲート・入力2本・出力2クラス)**

```mermaid
graph LR
  subgraph FLOW["構造: 各段のゲート数・出力ビット数"]
    direction LR
    X["入力<br/>2ビット<br/>x₀, x₁ ∈ {0,1}<br/>全4パターン列挙"]
    L1["LutLayer #1<br/>16ゲート / 各2入力<br/>16×16 = 256 パラメータ<br/>出力: 16ビット"]
    L2["LutLayer #2<br/>16ゲート / 各2入力<br/>16×16 = 256 パラメータ<br/>出力: 16ビット"]
    GS["GroupSum k=2, τ=2.0<br/>16本を 8本 + 8本 に分割<br/>各組で1の本数を数える<br/>出力: 2実数"]
    ARG["argmax<br/>予測ラベル 0 / 1"]
    X -->|"2本ずつ<br/>ランダム固定配線"| L1
    L1 -->|"16ビット"| L2
    L2 -->|"16ビット"| GS
    GS -->|"(4, 2) 実数"| ARG
  end

  subgraph EX["具体例: x = (1, 0) を hard eval で流す"]
    direction LR
    E0["x₀=1<br/>x₁=0"] --> E1["32ゲートが<br/>順に 0/1 を確定<br/>(argmaxで表が1枚に決まっている)"]
    E1 --> E2["class0 の8本: 立った本数 少<br/>class1 の8本: 立った本数 多"]
    E2 --> E3["予測 = 1<br/>XOR(1,0) = 1 ✓"]
  end

  subgraph LEG["凡例"]
    direction LR
    LGL["緑 = 論理(0/1 のビットだけが流れる)"]
    LGR["橙 = 実数(float。ここから先はハードに焼けない)"]
  end

  style X fill:#10b98126,stroke:#10b981
  style L1 fill:#10b98126,stroke:#10b981
  style L2 fill:#10b98126,stroke:#10b981
  style GS fill:#f9731626,stroke:#f97316
  style ARG fill:#f9731626,stroke:#f97316
  style E0 fill:#10b98126,stroke:#10b981
  style E1 fill:#10b98126,stroke:#10b981
  style E2 fill:#f9731626,stroke:#f97316
  style E3 fill:#f9731626,stroke:#f97316
  style LGL fill:#10b98126,stroke:#10b981
  style LGR fill:#f9731626,stroke:#f97316
```

緑の区間はビットしか流れないので、そのまま FPGA の LUT に焼ける。
橙になるのは最後の「数えて比べる」ところだけである。
実際に学習された配線とゲート種別は、このページ2枚目の回路図(学習前 vs 学習後)で確認できる。

**どう実現しているか**

- **1ニューロン = 1枚の真理値表**。普通のNNのニューロンは「重み付き和 → 活性化関数」だが、
  LutLayerのニューロンは「入力2本を見て 0/1 を返す表」に置き換わっている
- **学習しているのは表の中身だけ**。どのニューロンが前の層のどの2本を見るか(配線)は
  最初にランダムで決めて固定する。動くのは「その表が AND なのか XOR なのか」の部分だけ
- **表は16通りの候補の混ぜ合わせとして持つ**。2入力の論理関数は全部で16種類しかないので、
  「16種類への重み付け」(= 1ゲートあたり16パラメータ)を学習すれば微分できる。
  使うときは一番重い1種類に確定させる ── これが soft → hard の離散化
- **深さが表現力を作る**。1段目の16ビットを2段目が読むことで、ゲートの合成として
  より複雑な論理を組み立てる。XOR は2入力なので1枚の表でも足りるが、
  ここでは MNIST(例06)と同じ2層構成を最小サイズで体験しておく

実行: uv run python examples/01_xor.py
期待結果: soft/hard 両方で accuracy 1.00
"""

# === 準備: import と乱数種 ===
# > 必要なのは PyTorch と matplotlib、そして本ライブラリから `LutLayer` / `GroupSum` /
# > データセット / 回路描画の4つだけ。特別な学習フレームワークは要らない。
# >
# > ライブラリは `import griffith_logic as gl` と読み込むのが公式の慣用形で、
# > 公開シンボルは `gl.LutLayer` のように参照する(`from griffith_logic import LutLayer`
# > と書いても同じものが得られる)。
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import griffith_logic as gl
from griffith_logic.analysis.circuit_viz import plot_circuit
from griffith_logic.datasets import xor2

# 乱数種を固定する。LutLayerの初期パラメータ(=各ゲートがどの論理関数寄りから
# 学習を始めるか)は乱数依存なので、固定しないと実行ごとに収束の速さが変わる。
torch.manual_seed(0)

# === データセット: XORの4パターン ===
# > XOR は入力2ビットなので、取りうる入力は `(0,0) / (0,1) / (1,0) / (1,1)` の**4通りで全部**。
# > そのため訓練データとテストデータを分ける必要がない ── 見たいのは「暗記できたか」ではなく
# > **回路として正しい真理値表に落ちたか**だからである。
x, y_bits = xor2().dataset.tensors
y = y_bits.squeeze(-1).long()  # cross_entropy用のクラスラベル{0,1}に変換

# === モデル定義: LutLayer 2段 + GroupSum ===
# > `LutLayer` は「出力ニューロン1個 = N入力の LUT(ルックアップテーブル=真理値表)1枚」に
# > 対応する層で、普通の NN の `nn.Linear + 活性化` の位置に入る。`GroupSum` は論理ゲートの
# > 世界から分類スコアへの**出口**で、最終層のビットをクラスごとに数え上げる。
# >
# > 下の3行が、このページで扱う回路のすべてである(全32ゲート)。
model = nn.Sequential(
    # 入力2ビット → 16ゲート。各ゲートは2入力の論理関数(AND/OR/XOR等16種)の
    # ソフト混合として学習される。どの入力2本を見るかの配線もランダムに固定される。
    gl.LutLayer(2, 16, n_inputs=2),   # 16ゲートsoftmax(difflogic互換)
    # 2段目。1段目の16本の出力ビットを入力として、さらに16ゲート。
    # XORのような非線形は、この深さ(ゲートの合成)によって表現される。
    gl.LutLayer(16, 16, n_inputs=2),
    # 最終層16ゲートの出力を2クラスに均等分割して集計(グループ和)。
    # tauはロジットの温度で、大きいほどsoftmaxがなだらかになる(学習の安定性に効く)。
    gl.GroupSum(k=2, tau=2.0),
)

# 学習前の状態を後で図にするためのスナップショット(学習は元のmodelで進める)。
model_init = copy.deepcopy(model)

# 最適化対象は「各ゲートがどの論理関数であるか」を表す連続パラメータ。
# 配線は固定なので、学習しているのは実質的に真理値表の中身だけ。
opt = torch.optim.Adam(model.parameters(), lr=0.05)

# === 学習の準備: soft/hard の測り方と確定度 ===
# > このライブラリでは精度を**必ず2通り**で測る。
# >
# > - **soft** = `train()` モード。各ゲートは16種の論理関数の「確率的な混合」のまま
# >   (= 連続緩和)。勾配が流れる代わりに、まだ回路ではない
# > - **hard** = `eval()` モード。各ゲートを argmax で1つの論理関数に確定させた状態。
# >   ここから先は学習用の近似ではなく、FPGA に焼けるのと同じ純粋な Boolean 回路
# >
# > **soft だけを見て喜ぶのがこの分野で最も多い失敗**なので、最初から両方を測る癖をつける。
def accuracy(m: nn.Module, mode: str) -> float:
    """mode='soft'(連続混合) / 'hard'(argmax確定) で4パターンの正解率を返す。"""
    m.eval() if mode == 'hard' else m.train()
    with torch.no_grad():
        acc = (m(x).argmax(-1) == y).float().mean().item()
    m.train()  # 呼び出し側の学習を邪魔しないよう train() に戻す
    return acc


# 「各ゲートがどれだけ迷わず1つの論理関数を選べているか」を測る。
# 学習しているのは16種の論理関数上の分布(softmax)なので、その最大確率が
# そのまま確定度になる。1/16 = 0.0625 は完全に迷っている状態(一様分布)、
# 1.0 は「このゲートはXORだ」と言い切れている状態。
def gate_confidence(m: nn.Module) -> torch.Tensor:
    """全LutLayerの全ゲートについて、論理関数分布の最大確率を並べて返す。"""
    probs = [F.softmax(layer.weights.detach(), dim=-1).max(dim=-1).values
             for layer in m if isinstance(layer, gl.LutLayer)]
    return torch.cat(probs)


# 学習前の確定度。初期化直後なので、ほぼ全ゲートが 1/16 付近に固まっているはず。
conf_before = gate_confidence(model).numpy()

# === 学習ループ ===
# > 中身は**完全に普通の PyTorch**。特別な学習則は要らない(そこがこの手法の利点)。
# > 学習曲線を描くために、毎ステップ loss と soft/hard の両方を記録する。hard は毎回
# > `eval()` へ切り替えて測り直すので、「**いつ回路として正解に到達したか**」がステップ単位で
# > 分かる(soft より遅れて立ち上がるのが典型的な挙動)。データが4パターンしかないので、
# > 毎ステップ測っても実行時間はほぼ増えない。
STEPS = 500
ZOOM_STEPS = 60  # accuracyが動くのは最初の数十ステップだけなので、図ではそこを拡大する
hist_step: list[int] = []
hist_loss: list[float] = []
hist_soft: list[float] = []
hist_hard: list[float] = []

for step in range(STEPS):
    opt.zero_grad()
    # lossはクラス分類のcross entropy。backward()で、GroupSum → 2段目のゲート分布 →
    # 1段目のゲート分布、という順に勾配が流れる(連続緩和しているので微分可能)。
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    hist_step.append(step)
    hist_loss.append(loss.item())
    hist_soft.append(accuracy(model, 'soft'))
    hist_hard.append(accuracy(model, 'hard'))

# === 評価: soft と hard を必ず両方見る ===
# > 学習が終わったら、まず2つの数字を並べる。**hard が 1.00 であることが本質**で、
# > soft だけ高くて hard が落ちる場合は、混合が中間状態に居座っており
# > 「回路として決めきれていない」サインである。
acc_soft = accuracy(model, 'soft')
acc_hard = accuracy(model, 'hard')
print(f'XOR: soft acc={acc_soft:.2f}, hard acc={acc_hard:.2f}')
# hardが1.00であることが本質。softだけ高くてhardが落ちる場合は、
# 混合が中間状態に居座っており「回路として決めきれていない」サイン。
assert acc_hard == 1.0

# hard accuracy が「以後ずっと1.00」になったステップを求める。
# 単に最初に1.00を踏んだ点ではなく最後に落ちた点の次を採るのは、学習初期には
# argmaxがたまたま正解の回路を指すことがあり(すぐ別の回路へ移る)、そこを
# 「学習できた」と読むと過大評価になるため。
last_fail = max((i for i, a in enumerate(hist_hard) if a < 1.0), default=-1)
stable_hard = hist_step[last_fail + 1] if last_fail + 1 < len(hist_step) else None
print(f'hard acc が以後ずっと 1.00 になったステップ: {stable_hard} / {STEPS}')

# === 可視化: 学習曲線・回路図・ゲート確定度 ===
# --- 図1: 学習曲線(loss と soft/hard accuracy) ---
# > 左が loss(全500ステップ)、右が2種類の accuracy。accuracy が動くのは最初の数十ステップ
# > だけなので、右は冒頭を拡大している(以降はどちらも 1.00 で平坦)。
# >
# > 見どころは右で、**hard(実線・濃い緑)が soft(破線・灰)を追いかけ、最後は両方 1.0 で
# > 重なる**こと。これが「連続緩和で学習 → argmax で離散化しても壊れない」という、
# > このライブラリの中心的な主張そのものである。
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist_step, hist_loss, color='#2e6fdb')
axes[0].set_xlabel('step')
axes[0].set_ylabel('cross entropy loss')
axes[0].set_title('training loss (all 500 steps)')
axes[0].grid(alpha=0.3)
axes[1].plot(hist_step, hist_soft, '--', color='#9aa0a6', lw=2, label='soft acc (relaxed)')
axes[1].plot(hist_step, hist_hard, '-', color='#2e9e4f', lw=2, label='hard acc (discrete circuit)')
if stable_hard is not None:
    axes[1].axvline(stable_hard, color='#e8850c', ls=':', lw=1.5,
                    label=f'hard acc = 1.00 from here (step {stable_hard})')
axes[1].set_xlim(0, ZOOM_STEPS)
axes[1].set_ylim(-0.05, 1.15)  # 0/1に張り付く線を潰さないための余白
axes[1].set_xlabel('step')
axes[1].set_ylabel('accuracy')
axes[1].set_title(f'soft vs hard accuracy (first {ZOOM_STEPS} steps)')
axes[1].legend(loc='lower right', fontsize=9)
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()

# > 右の hard が**階段状に跳ぶ**のは、argmax の結果が切り替わった瞬間にだけ出力が変わる
# > ためで、連続量がなめらかに動いているのではなく**回路そのものが離散的に書き換わっている**。
# > 縦の点線は hard が「以後ずっと」1.00 になったステップで、その手前で一瞬 1.00 に触れて
# > また落ちているのは、たまたま正解の回路を通り過ぎただけの状態である。
# --- 図2: 回路図(同じコードがこの回路になる) ---
# > 上で書いた3行の `nn.Sequential` は、実際には「ゲートのノードと、それを結ぶ配線」という
# > 回路そのもの。学習前後を左右に並べて、学習で何が起きたのかを1枚で見る。
# >
# > - **左(`resolve=False`)** — 学習前。各ノードは16種の論理関数の**混合**なので `LUT` 表示。
# >   色の濃さが確定度(分布エントロピー)で、薄いほど迷っている
# > - **右(`resolve=True`)** — 学習後。argmax で確定した論理関数の名前
# >   (`AND` / `XOR` / `NAND` / `!A` / `A>=B` …)がノードに入る
# >
# > 名前の元になる真理値表は `export_ir` が Verilog を吐くときに使うものと同一なので、
# > **この図のゲート = 実際に焼かれる回路のゲート**である。
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
plot_circuit(model_init, resolve=False, ax=axes[0])
plot_circuit(model, resolve=True, ax=axes[1])
fig.tight_layout()
plt.show()

# > 左右を見比べると、学習前は全ノードが灰色に近い `LUT` のままなのに、学習後は
# > `XOR` や `NAND` といった**具体的な論理関数の名前**が並ぶ。学習とは「16の候補から
# > ゲートを1つずつ選び切る」作業だ、という主張がそのまま絵になっている。
# --- 図3: 「16の候補から1つを選び切れたか」を数字で見る ---
# > 図2の色の濃さ(確定度)を、全32ゲート分のヒストグラムにして定量化する。横軸は各ゲートの
# > 論理関数分布の最大確率、縦軸はそこに入ったゲートの数。
# >
# > - **学習前(灰)** — 左寄りの山。初期化は乱数なので完全な一様(破線の 1/16 = 0.0625)より
# >   少しは偏っているが、どのゲートも「何者でもない」曖昧な混合のまま
# > - **学習後(緑)** — 分布ごと右へ移動し、多くのゲートが 0.9 超に到達する。各ゲートが
# >   「自分は XOR だ / AND だ」と言い切れる状態になったということ
# >
# > ここが右端に寄っていないのに hard acc が 1.00 なら、それは運が良かっただけで、わずかな
# > 摂動で回路が別物に化ける危険がある。逆に右端に寄っていれば、argmax による離散化は
# > 「ほぼ何も捨てていない」ので hard eval が壊れない。
conf_after = gate_confidence(model).numpy()
print(f'ゲート確定度(最大確率)の平均: 学習前 {conf_before.mean():.3f} '
      f'→ 学習後 {conf_after.mean():.3f} (一様分布なら 1/16 = 0.0625)')
print(f'確定度 0.9 超のゲート数: 学習前 {(conf_before > 0.9).sum()} / {conf_before.size}'
      f' → 学習後 {(conf_after > 0.9).sum()} / {conf_after.size}')

bins = [i / 20 for i in range(21)]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(conf_before, bins=bins, color='#9aa0a6', alpha=0.75, label='before training')
ax.hist(conf_after, bins=bins, color='#2e9e4f', alpha=0.75, label='after training')
ax.axvline(1 / 16, color='#e8850c', ls='--', lw=1.5, label='uniform over 16 functions (1/16)')
ax.set_xlabel('max probability over the 16 logic functions (per gate)')
ax.set_ylabel('number of gates')
ax.set_title('gate decisiveness: picking one function out of 16')
ax.legend()
fig.tight_layout()
plt.show()
