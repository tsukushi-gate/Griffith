"""MNIST(20x20)を微分可能論理ゲート網で学習する例。

XOR(examples/01_xor.py)は4パターンの玩具問題だった。ここでは同じ道具立て
(LutLayer + GroupSum + 普通のPyTorch学習ループ)のまま、入力400ビット・10クラスの
実データへスケールさせ、**4000ゲートの回路が手書き数字を読む**ところまでを見る。

difflogic公式レシピ(6層×8000ゲート・200k iter)は論文水準97.7%に届くが、
GPUで数時間かかる。このサンプルはCPUで1〜2分に収まるよう
「2層×2000ゲート・訓練20000枚・4000 iter」へ縮めた縮小版で、
到達するhard accuracyは0.86前後になる(精度を狙うレシピは
tests/l3_paper/test_dlgn_mnist.py の DLGN_FULL=1 を参照)。

このページで一番大事な概念は **soft と hard の2つの姿**である。

- **soft**(学習時の姿) … 各ゲートは16種の論理関数の連続的な混合。微分できるので学習できるが、
  この状態のままではハードウェアに焼けない
- **hard**(回路としての姿) … 混合をargmaxで1種類に確定させた本物の論理回路。
  推論はビット演算だけで済むが、微分はできない

学習はsoftで進め、実際に使うのはhard。したがって**両者の正解率の差(離散化ギャップ)**が
小さいことを常に確認する必要がある。以下の図はそのための道具立てになっている。

**回路の全体像(XORと同じ形のまま、数だけ増やす)**

```mermaid
graph LR
  subgraph FLOW["構造: 各段の解像度・ゲート数・出力ビット数"]
    direction LR
    IMG["入力画像 MNIST 20×20<br/>= 400画素 = 400本の入力線<br/>(hard eval では round)"]
    L1["LutLayer #1<br/>2,000ゲート/各2入力<br/>2000×16=32,000 param<br/>出力: 2,000ビット"]
    L2["LutLayer #2<br/>2,000ゲート/各2入力<br/>2000×16=32,000 param<br/>出力: 2,000ビット"]
    GS["GroupSum k=10, τ=10.0<br/>2,000本を200本×10組へ<br/>各組で1の本数を数える<br/>出力: 10実数"]
    ARG["argmax<br/>予測ラベル 0〜9"]
    IMG -->|"400本から<br/>2本ずつランダム固定配線"| L1
    L1 -->|"2,000ビット"| L2
    L2 -->|"2,000ビット"| GS
    GS -->|"(B, 10) 実数"| ARG
  end

  subgraph EX["具体例: 手書きの「7」を1枚流す"]
    direction LR
    E0["20×20 の画素が<br/>400ビットに並ぶ"]
    E1["合計 4,000 ゲートが 0/1 を伝播<br/>(乗算ゼロ・ビット演算のみ)"]
    E0 --> E1
    E1 --> E2["クラス7の200本: 1が多い<br/>他クラスの200本: 1が少ない"]
    E2 --> E3["予測 = 7"]
  end

  subgraph LEG["凡例"]
    direction LR
    LGL["緑 = 論理(0/1 のビットだけが流れる = FPGAに焼ける)"]
    LGR["橙 = 実数(float)"]
  end

  style IMG fill:#10b98126,stroke:#10b981
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

畳み込みも行列積も無く、**全体で4,000個の2入力ゲート**しか無いことに注目してほしい。
そのうち実数が出てくるのは最後の GroupSum 以降だけである。

**どう実現しているか**

- **1画素 = 1本の入力線**。20×20の画像をそのまま400本のビット線として回路へ差し込む
  (28×28 の外周は difflogic 公式レシピと同じく黒枠として落としてある)
- **1ゲートは前層の2本だけを見る**。全結合ではなく、各ゲートの入力2本はランダムに選んで
  固定する(疎結合)。だから2,000ゲートでもパラメータは 2000×16 = 32,000 で済む
- **クラス分けは「数える」だけ**。GroupSum は2,000本の出力を200本ずつ10グループに分け、
  各グループで1が立っている本数を数える。多いクラスがその画像の答えになる。
  τ=10 はその本数を割る温度で、softmax をなだらかにして学習を安定させる
- **soft と hard の2つの姿**。学習中(soft)は各ゲートが16種の論理関数の混合、
  推論時(hard)は argmax で1種類に確定した本物の回路。
  この例の主題は両者の差(離散化ギャップ)を測ること

実行: uv run python examples/06_mnist.py
期待結果: hard accuracy 0.86前後(assertは0.70で緩めに置いてある)
"""

# === 準備: import と乱数種 ===
# > 必要なのは PyTorch 一式と、本ライブラリの3つだけ。
# >
# > - `LutLayer` … 1ニューロン = 1枚の **LUT(ルックアップテーブル=真理値表)** になる層
# > - `GroupSum` … 論理の世界からクラススコアへの出口(ビットをクラスごとに数える)
# > - `plot_circuit` … 学習後のモデルを回路図として描く
# >
# > 専用の学習フレームワークは登場しない。以降の学習ループは普通の PyTorch そのものである。
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from griffith_logic import GroupSum, LutLayer
from griffith_logic.analysis.circuit_viz import plot_circuit
from griffith_logic.datasets import load_mnist20x20

# ゲートの初期分布は乱数依存なので、再現性のために固定する。
torch.manual_seed(0)

# --- ハイパーパラメータ ---
# > **K が最重要のつまみ**である。K は1層あたりのゲート数で、そのまま
# > 「FPGA に焼いたときのゲート数」= ハードウェアの面積になる。実数 NN のチューニングでは
# > 幅は精度と計算時間のトレードオフだが、論理ゲート網では**幅 = 実装コストそのもの**であり、
# > 精度と K のバランスを取ることがチューニングの中心になる。
# CPUで1〜2分に収まる縮小レシピ。K(幅)・層数・STEPSを増やすほど精度は上がる。
# 論理ゲート網では「幅K」が実装コストにほぼ直結する(=焼いたときのゲート数)ので、
# 精度とKのトレードオフを見るのがチューニングの中心になる。
K = 2000          # 1層あたりのゲート数
STEPS = 4000      # 学習イテレーション数
BATCH = 128       # ミニバッチサイズ
N_TRAIN = 20000   # 訓練に使う枚数(全60000のサブセット)
N_TEST = 2000     # 評価に使う枚数

# === データセット: 画像を400本のビット線として読む ===
# > MNIST の 28x28 から黒い縁を落とし、**20x20 = 400 画素**にしたものを使う
# > (difflogic 論文の公式レシピと同じ前処理)。この 400 画素が、そのまま
# > 回路の**入力線 400 本**になる ── 1画素 = 1本のワイヤ、という対応である。
# >
# > `binarize_threshold=None` なので値は $[0,1]$ の連続値のまま読み込む。
# > 学習は連続値で行い、評価だけ `round()` でビットに落とす、というのが公式レシピの作法。
# load_mnist20x20 は difflogic 互換の「黒境界を落として28x28→20x20」にした
# MNIST を返す。400ビット=1ピクセル1ビットが、そのまま回路の入力線になる。
# binarize_threshold=None なので値は[0,1]の連続値のまま(公式レシピと同じく
# 訓練は連続値、評価は round() で二値化する)。
train_set, test_set = load_mnist20x20(binarize_threshold=None)


def materialize(dataset, n):
    """先頭n件を (N, 400) のテンソルとラベルにまとめる(毎stepの前処理を省く)。"""
    subset = Subset(dataset, range(n))
    x, y = next(iter(DataLoader(subset, batch_size=n)))
    return x.reshape(n, -1), y


train_x, train_y = materialize(train_set, N_TRAIN)
test_x, test_y = materialize(test_set, N_TEST)
# --- 評価用の入力をビットに落とす ---
# > **なぜ評価だけ二値化するのか** ── hard eval は「実際に FPGA へ焼いた回路」を
# > 模した測定である。回路の入力端子に入るのは信号線の High/Low だけなので、
# > 0.37 のような実数を食わせた測定は現実の性能を過大評価してしまう。
# 評価用の入力は round() で {0,1} のビットに落とす。hard eval は
# 「実際にFPGAへ焼いた回路」を模しているので、入力も本物のビットでなければならない。
test_bits = test_x.round()
print(f'train={tuple(train_x.shape)} test={tuple(test_x.shape)} '
      f'(20x20=400ビット入力 / 10クラス)')

# === モデル定義: LutLayer 2段 + GroupSum ===
# > このページで扱う回路は下の3行がすべてである(合計 4,000 ゲート)。
# >
# > - `LutLayer(400, K, n_inputs=2)` … 400本の入力線のうち**2本だけ**を見るゲートを
# >   K 個並べる。どの2本を見るか(配線)は初期化時にランダムに決めて**固定**し、
# >   学習で動くのは「その2入力に対する真理値表」だけ
# > - 2段目の `LutLayer(K, K)` … XOR のような複雑な論理は、この**ゲートの合成**で作られる
# > - `GroupSum(k=10, tau=10.0)` … 最終2000ビットを10クラス×200本に分け、
# >   各組で1が立っている本数を数える。`tau` はその本数を割る温度で、
# >   大きいほど softmax がなだらかになり学習が安定する
# >
# > **なぜ全結合ではなく2入力なのか** ── 2入力の論理関数は全部で16種類しかないので、
# > 「16種への重み付け」を学習すれば微分できる(1ゲート16パラメータ)。
# > 全結合にすると真理値表が指数的に爆発して、この筋が使えなくなる。
# XORのときと構造は完全に同じ。幅とクラス数が変わっただけ。
# 特筆すべきは「畳み込みも正規化も活性化関数も無い」こと。あるのは2入力ゲートの層と、
# 最後にゲートの出力を数え上げるGroupSumだけ。それでも手書き数字が読める。
model = nn.Sequential(
    # 400ビット → 2000ゲート。各ゲートは入力400本のうち2本だけを見る
    # (疎なランダム配線)。学習されるのは「その2入力に対する真理値表」。
    LutLayer(400, K, n_inputs=2),
    # 2段目。1段目の2000ビットを入力に、さらに2000ゲート。
    LutLayer(K, K, n_inputs=2),
    # 最終2000ゲートを10クラス×200ゲートに分けて集計する。
    # tau はロジットの温度(difflogic公式レシピと同じ10.0)。
    GroupSum(k=10, tau=10.0),
)
print(model)

# === 学習: 普通のPyTorchループでsoft/hardを両方測る ===
# > 学習の中身は完全に普通の PyTorch(`cross_entropy` + `Adam`)で、論理ゲート網で
# > あることを意識する箇所はどこにもない。**そこがこの手法の実用上の利点**である。
# >
# > 唯一の作法が `model.train()` / `model.eval()` の使い分けで、これが soft / hard の
# > 切り替えスイッチになっている。BatchNorm のような統計の話ではなく、
# > **eval() でモデルの数学そのものが「16種の混合」から「argmax で確定した1種」へ変わる**。
# > だから両方を毎回測り、その差(離散化ギャップ)を監視する。
# 中身は普通のPyTorch。論理ゲート網であることを意識する箇所はどこにもない。
# 唯一の作法が model.train() / model.eval() の使い分けで、これが soft / hard の
# 切り替えスイッチになっている(BatchNormのような統計の話ではない点に注意)。
opt = torch.optim.Adam(model.parameters(), lr=0.01)
history = []  # (step, soft acc, hard acc)

for step in range(STEPS + 1):
    if step % 500 == 0:
        # soft = 連続緩和のまま(学習時の姿) / hard = argmaxで確定した回路の姿。
        # この2本を必ず並べて見る。両者の差が「離散化ギャップ」で、
        # softだけ高くhardが低いなら回路として決めきれていないサイン。
        with torch.no_grad():
            model.train()
            soft_acc = (model(test_x).argmax(-1) == test_y).float().mean().item()
            model.eval()
            hard_acc = (model(test_bits).argmax(-1) == test_y).float().mean().item()
        model.train()
        history.append((step, soft_acc, hard_acc))
        print(f'step {step:>5}  soft acc={soft_acc:.4f}  hard acc={hard_acc:.4f}')

    # ミニバッチは復元抽出でランダムに引く(エポックの概念を持たない difflogic 流の回し方)。
    idx = torch.randint(0, N_TRAIN, (BATCH,))
    opt.zero_grad()
    loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
    loss.backward()
    opt.step()

# 所要時間はCPUで1分前後(値は環境で変わるので出力には出さない)。
steps, soft_hist, hard_hist = zip(*history, strict=True)
final_soft, final_hard = soft_hist[-1], hard_hist[-1]
print(f'\n最終 soft acc={final_soft:.4f} / hard acc={final_hard:.4f}')
# hardが本命。縮小レシピでも0.86前後に届く(閾値は環境差を見て緩めに0.70)。
assert final_hard > 0.70, f'hard acc={final_hard:.4f} が低すぎる'

# === 可視化: 学習された回路の姿を見る ===
# > **読み取り**: 上のログで soft と hard がほぼ並んで上がり、最終的に hard accuracy が
# > 0.86 前後に達している。実数の重みも乗算も一切持たない 4,000 個のゲートが、
# > 手書き数字の 86% を当てられるということ。以下、この回路の姿を3枚の図で見ていく。
# --- 図1: 学習した回路の俯瞰 ---
# > まず全体像。ゲートが4,000個あるので全ノードは描けず、`plot_circuit` は自動的に
# > **俯瞰図**(層=箱、間の矢印=配線本数)へ切り替わる。
# > [付録: XOR](/tutorials/xor/) の全ノード描画と見比べると、同じ API が規模に応じて
# > 表示を変えていることが分かる。
# ゲートが4000個あるので、plot_circuit は全ノード描画をあきらめて
# 自動的に俯瞰図(層=箱、間の矢印=配線本数)へ切り替わる。
# XORの図(16ゲート)と見比べると、同じAPIが規模でどう表示を変えるかが分かる。
fig, ax = plt.subplots(figsize=(9, 3.4))
plot_circuit(model, resolve=True, ax=ax,
             title=f'MNIST DLGN — {2 * K:,} gates (400 bits in → 10 classes)')
fig.tight_layout()
plt.show()

# > **読み取り**: 400本の入力線が2つの LutLayer(各2,000ゲート)を通り、
# > GroupSum で10クラスへ落ちる、という一本道であることが確認できる。
# > 分岐も畳み込みも正規化も無い ── この単純さのまま 86% が出ている。
# --- 図2: 学習曲線(soft と hard) ---
# > soft(橙)と hard(緑)の2本を重ねる。**hard が本命**で、これが焼ける回路の成績である。
# > 2本の隙間が離散化ギャップ。
# 見るべきは hard(緑)。soft(橙)だけが上がって hard が付いてこない場合、
# ゲートの混合が中間状態に居座っている(=どの論理関数にするか決めきれていない)。
# 逆に2本がほぼ重なって上がっていれば、学習中の姿と焼いたあとの姿が一致しており、
# そのまま回路として取り出せる状態だと分かる。この図では序盤に少し開いた差が
# 学習が進むにつれて閉じていく様子が見える。
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, soft_hist, 'o-', color='#e8850c',
        label='soft acc (relaxed / train mode)')
ax.plot(steps, hard_hist, 's-', color='#2e9e4f',
        label='hard acc (discrete circuit / eval mode)')
ax.set_xlabel('step')
ax.set_ylabel('test accuracy')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc='lower right')
ax.set_title('MNIST 20x20 — soft vs hard test accuracy')
fig.tight_layout()
plt.show()

# > **読み取り**: 序盤で少し開いた2本の差が、学習が進むにつれて閉じていく。
# > つまり「学習中のモデル」と「焼いたあとの回路」が同じものへ収束しており、
# > そのまま回路として取り出してよい状態にある。もし橙だけが上がって緑が付いてこないなら、
# > ゲートの混合が中間状態に居座っている(どの論理関数か決めきれていない)サインで、
# > 学習量・`tau`・パラメータ化を疑うことになる。
# --- 図3: 回路が間違えた数字 ---
# > hard eval で外した10枚を、**回路が実際に受け取ったビット列**(round 後)のまま表示する。
# > 前処理後の姿で見ないと「なぜ間違えたか」を判断できないためである。
# hard eval(=焼いた回路そのもの)の誤分類を並べる。1と7、4と9のような
# 「人間でも紛らわしい」ペアに偏っていれば、回路はまともに形を捉えている。
# 逆に、まったく無関係なペア(0を7と読む等)ばかりなら学習不足を疑う。
# 表示しているのは round() 後のビット列、つまり回路が実際に受け取った入力そのもの。
# 縮小レシピでは全体の1割強を間違えるので、10枚並べればすぐ見つかる。
model.eval()
with torch.no_grad():
    pred = model(test_bits).argmax(-1)
wrong = (pred != test_y).nonzero(as_tuple=True)[0][:10]
print(f'誤分類 {int((pred != test_y).sum())} / {N_TEST} 枚(先頭10枚を表示)')

fig, axes = plt.subplots(2, 5, figsize=(9, 4.2))
for ax, i in zip(axes.ravel(), wrong.tolist(), strict=False):
    # 表示するのは回路が実際に見たビット列(round後)そのもの。
    ax.imshow(test_bits[i].reshape(20, 20), cmap='gray_r')
    ax.set_title(f'pred {int(pred[i])} / true {int(test_y[i])}', fontsize=9)
    ax.set_axis_off()
for ax in axes.ravel()[len(wrong):]:
    ax.set_axis_off()
fig.suptitle('misclassified by hard eval (discrete circuit)', fontsize=11)
fig.tight_layout()
plt.show()

# > **読み取り**: 並ぶのは 4 と 9、7 と 1、5 と 3 のように人間でも紛らわしいペアが中心で、
# > 回路が形をまともに捉えていることが分かる。無関係なペア(0 を 7 と読む等)ばかりなら、
# > 学習不足を疑うところである。
# === 評価を掘り下げる: 弱点と判断のキレ ===
# > 全体の正解率という1つの数字では、「どのクラスが弱いか」も「どれくらい自信を持って
# > 答えているか」も分からない。そこで2つの量を追加で測る。
# >
# > - **クラス別 hard 正解率** … 苦手な数字を特定する
# > - **マージン** … 1位のロジット $-$ 2位のロジット。その1枚をどれだけ引き離して
# >   選べたかを表す。GroupSum のロジットは「そのクラスの200ゲートのうち何本が1を出したか」を
# >   `tau` で割った値なので、マージンは素朴に**票差**として読める
# >
# > マージンを soft と hard で並べると、**離散化で判断のキレがどれだけ失われたか**を
# > 精度の数字とは別の角度から測れる。
# --- 図4の下ごしらえ: クラス別正解率とマージンを計算する ---
# 全体の正解率1つでは「どこが弱いか」も「どれくらい自信を持って答えているか」も分からない。
# そこで2つ測る。
#   (1) クラスごとのhard正解率 … 苦手な数字がどれかが分かる
#   (2) マージン = 1位ロジット - 2位ロジット … その1枚をどれだけ引き離して選べたか
# (2)をsoftとhardで並べると、離散化で判断のキレがどれだけ失われたかが実測できる。
# GroupSumのロジットは「そのクラスに割り当てた200ゲートのうち何個が1を出したか」を
# tauで割った値なので、マージンは素朴に「票差」と読める。
per_class = []
for c in range(10):
    mask = test_y == c
    per_class.append((pred[mask] == c).float().mean().item())

with torch.no_grad():
    model.train()
    soft_logits = model(test_x)
    model.eval()
    hard_logits = model(test_bits)


def top2_margin(logits):
    """1位と2位のロジット差(大きいほど自信を持って1クラスを選べている)。"""
    top2 = logits.topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).numpy()


soft_margin, hard_margin = top2_margin(soft_logits), top2_margin(hard_logits)
worst = min(range(10), key=lambda c: per_class[c])
print(f'クラス別hard正解率で最も苦手なのは "{worst}"(acc={per_class[worst]:.3f})')
print(f'マージン中央値: soft={float(torch.tensor(soft_margin).median()):.3f} / '
      f'hard={float(torch.tensor(hard_margin).median()):.3f}')

# --- 図4: どのクラスが苦手か(左)と、判断のキレ(右) ---
# > 左はクラス別の hard 正解率(破線=全体平均、橙=最も苦手なクラス)、
# > 右はマージンのヒストグラム(橙の塗り=soft、緑の輪郭=hard)。
# >
# > **読み取り(この図が最後なので先に書く)**: 左では 0 や 1 が易しく、
# > 8・5・2 のように「形の似た仲間が多い」クラスが沈む ── 全体の正解率1つでは
# > 見えない偏りがここで初めて見える。右では soft と hard の分布が**ほぼ重なる**。
# > これは離散化しても判断のキレが失われていない、つまりそのまま回路にしてよい、という意味である。
# > もし hard 側だけが 0 付近へ寄っていたら、argmax で情報が落ちているサインになる。
# 左: クラスごとのhard正解率。破線は全体平均で、最も苦手なクラスだけ橙で塗り分けている。
#     0 や 1 は易しく、8・5・2 のように「形の似た仲間が多い」クラスが沈む。
#     全体の正解率1つでは見えない偏りが、ここで初めて見える。
# 右: マージン(1位-2位のロジット差)のヒストグラム。左端(0付近)に積まれた枚数が
#     「僅差でかろうじて選んだ=ひっくり返りやすい」判断の量にあたる。
#     soft(橙の塗り)と hard(緑の輪郭)の分布がほぼ重なっていることが重要で、
#     これは離散化しても判断のキレが失われていない=そのまま回路にしてよい、という意味。
#     もし hard 側だけが 0 付近に寄っていたら、argmax で情報が落ちているサインになる。
hi = float(max(soft_margin.max(), hard_margin.max()))

fig, (ax_cls, ax_margin) = plt.subplots(1, 2, figsize=(12, 4.0))

bar_colors = ['#e8850c' if c == worst else '#2e9e4f' for c in range(10)]
ax_cls.bar(range(10), per_class, color=bar_colors)
ax_cls.axhline(final_hard, color='#444', linestyle='--', lw=1.2,
               label=f'overall hard acc = {final_hard:.3f}')
ax_cls.set_xticks(range(10))
ax_cls.set_xlabel('digit class')
ax_cls.set_ylabel('hard accuracy')
ax_cls.set_ylim(0, 1.25)
ax_cls.set_title(f'per-class accuracy (weakest class: {worst})')
ax_cls.grid(alpha=0.3, axis='y')
ax_cls.legend(fontsize=8, loc='upper right')

ax_margin.hist(soft_margin, bins=30, range=(0, hi), color='#e8850c', alpha=0.45,
               label='soft (relaxed)')
ax_margin.hist(hard_margin, bins=30, range=(0, hi), histtype='step', lw=2.0,
               color='#2e9e4f', label='hard (discrete circuit)')
ax_margin.set_xlabel('top1 - top2 logit margin')
ax_margin.set_ylabel('number of test images')
ax_margin.set_title('decision margin: soft vs hard')
ax_margin.grid(alpha=0.3)
ax_margin.legend(fontsize=8)

fig.suptitle('Where the circuit is weak, and how confidently it decides', fontsize=11)
fig.tight_layout()
plt.show()
