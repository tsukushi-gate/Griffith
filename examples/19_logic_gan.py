"""LogicGAN — 乱数ビット列から論理回路だけで画像を作る例(と、その壊れ方)。

18_logic_vae.py の生成は「潜在ビット → デコーダ回路 → 画像」だった。GAN では
そこに **Discriminator(判別器)** が加わり、Generator と敵対的に競わせる。
LogicGAN はその両方を LutLayer スタックで組んだもので、Generator の入力は
連続ガウスノイズではなく **乱数ビット列 z ~ Bernoulli(0.5)** ——
物理乱数源やLFSRでそのまま供給できる、論理回路にとって最も自然なノイズ源である。

    z (64ビットの乱数) → 論理回路 → 画像ビット {0,1}

## 全体データフロー(実クラス名つき)

```mermaid
flowchart LR
    Z["z 〜 Bernoulli(0.5)<br/>64ビットの乱数(コイン投げ)"] --> GB
    GB["generator.body<br/>LutLayer×2(256ゲート)"]
    GB --> GR["generator.readout<br/>LutLayer(64ゲート = 画素数)"]
    GR --> FAKE["fake 画像<br/>学習時 [0,1] / 推論時 0/1"]
    REAL["real 画像<br/>4種の二値パターン"] --> FLIP
    FAKE --> FLIP["bit_flip(p=0.1)<br/>D入力のビット反転ノイズ"]
    FLIP --> DB["discriminator.body<br/>LutLayer(32ゲート)"]
    DB --> DR["discriminator.readout<br/>LutLayer(64ゲート)"]
    DR --> DGS["GroupSum(k=1)<br/>64本のビットを足す"]
    DGS --> LOGIT["real/fake ロジット(実数)"]
    LOGIT --> DLOSS["d_loss<br/>本物を1・偽物を0に寄せる"]
    LOGIT --> GLOSS["g_loss<br/>偽物を1と言わせにいく"]
    GLOSS -.->|"勾配は soft 経路を逆流して G を更新"| GB

    style GB fill:#10b98126,stroke:#10b981
    style GR fill:#10b98126,stroke:#10b981
    style FLIP fill:#10b98126,stroke:#10b981
    style DB fill:#10b98126,stroke:#10b981
    style DR fill:#10b98126,stroke:#10b981
    style Z fill:#38bdf826,stroke:#38bdf8
    style DGS fill:#f9731626,stroke:#f97316
    style LOGIT fill:#f9731626,stroke:#f97316
```

緑が **ビットのまま進む論理回路**、橙が **実数が現れる場所**(判別器の最後の和だけ)、
青が **乱数ビット源**。生成の入口から出口まで、掛け算は1回も出てこない。

## どう実現しているか(4行で)

1. **ノイズ源 = コイン投げ**。z は連続ガウスではなく Bernoulli(0.5) の64ビット。
   u &lt; 0.5 の比較器で作れるので、物理乱数源や LFSR(疑似乱数を作る小さな回路)を
   そのまま繋げる。
2. **Generator = LutLayer スタック**。z は既にビットなので前処理(Thermometer 等)は不要。
   最後の LutLayer の出力を**そのまま画素**にする。学習時は soft(連続値=画素確率)、
   推論時は hard(純ブール回路)で厳密に 0/1 を吐く。
3. **Discriminator も論理回路**。画素を LutLayer に通し、最後に GroupSum で
   「1になったゲートの本数」を数えて real/fake のロジット1本にする。
4. **勾配は soft 相を逆流する**。学習中だけは各ゲートが16種の論理関数の重み付き和
   (連続)なので、D → G へ勾配が流れる。焼き上がった回路は離散でも、
   **学習しているあいだは滑らかな影**が並走している、という二重構造。

## 普通のGANとの違い

| 見るところ | 普通のGAN | LogicGAN |
|---|---|---|
| ノイズ z | 連続ガウス N(0, I) | 乱数ビット列 Bernoulli(0.5) |
| Generator | 転置畳み込み+BatchNorm(積和) | LutLayer スタック(真理値表引き) |
| 出力 | 連続画素 | 画素ビット 0/1(推論時は厳密に離散) |
| Discriminator | CNN | LutLayer + GroupSum(実数は最後の和だけ) |
| 損失 | non-saturating GAN loss | 同じ(BCEWithLogits) |
| 不安定さ | 敵対的 min-max | 敵対的 min-max **×離散出力**の二重 |

安定化の4つのつまみと、その効き目の実測は下の STABLE / NAIVE の比較そのものである。
ページ末尾に、実データが流れる**アーキテクチャ図**と、**学習後の Generator の実回路**
(1画素=ゲート1個。argmax で確定した論理関数の名前まで書いた図)を置いた。

この例のもう一つの主題は「壊れ方」:
    GAN は「敵対的な min-max」×「離散出力」という二重の不安定性を抱える。
    未調整の設定では Discriminator が Generator を圧倒し、G loss が発散して
    Generator が淘汰される —— これは本リポジトリで実測済みの失敗モードで、
    tests/test_logic_gan.py::test_training_dynamics_smoke がその健全域を守っている。
    このスクリプトは **安定版と未調整版を両方回して並べる** ことで、
    どの設定がどう効くのかを目で見えるようにする。

    ⚠️ 失敗しやすいポイント(下の STABLE / NAIVE の差分がそのまま対処法):
      1. Discriminator が強すぎる → D の学習率を下げ、G を複数回更新する(G:D=3:1)
      2. D が real/fake を完全分離して勾配が枯れる → D入力にビット反転ノイズを混ぜる
      3. D が自信過剰になる → label smoothing で real ターゲットを 1.0 から緩める
      4. 崩壊の検知 → D精度が 1.0 に張り付き、G loss が単調増加したら赤信号

このサンプルの縮小レシピ:
    本走(experiments/train_logic_gan.py)は 28x28 binary MNIST を30エポック回すが、
    ここはCPUで3分以内に収まるよう「8x8 の二値パターン4種」という極小タスクにした。
    到達点は「パターンの形が見え始める」レベルで、しかも縮小レシピでは
    **モード崩壊(全サンプルが似た形になる)が起きる**。それも含めて教材である。

実行: uv run python examples/19_logic_gan.py
期待結果(seed=0、CPU約90秒): 安定版 late G loss 0.9〜1.0 / late D精度 0.8〜0.9 /
          未調整版 late G loss 7 台 / late D精度 1.00(崩壊)。どちらもモード被覆は 1/4。
          GAN は浮動小数点の微小な差で軌道が分かれるので、数値は CPU(Apple Silicon と x86)で
          少し変わる。安定版のレシピは seed 0〜2 × 両 CPU で D 精度 < 0.99 を確かめた設定にしてある
          (以前の D 学習率 0.003 / smoothing 0.1 は x86 で D が完勝に張り付いた)
"""

# === 準備: import と乱数種 ===
# > 使うのは PyTorch と matplotlib、そして本ライブラリの `LogicGAN`(モデル本体)と
# > 回路描画ユーティリティだけ。GAN 用の特別なフレームワークは要らない。
# >
# > **GAN(敵対的生成ネットワーク / 敵対学習)** — 絵を描く側と、その絵が本物か偽物かを
# > 当てる側の**2つのネットワークを競わせて**前者を鍛える学習の枠組み。描く側が
# > **Generator(生成器)**、当てる側が **Discriminator(判別器)** で、以下それぞれ G / D と書く。
# > D が見抜けないほど本物らしい絵を G が出せるようになることを目指す。
# >
# > LogicGAN はその G と D を**両方とも論理ゲート(LutLayer)のスタック**で組んだもので、
# > 入口から出口まで掛け算が1回も出てこない。
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyBboxPatch

from griffith_logic.analysis.circuit_viz import gate_labels, plot_circuit
from griffith_logic.models import LogicGAN, LogicGANConfig

# LUTの初期真理値・ランダム配線・乱数ビット列すべてが乱数依存なので固定する。
torch.manual_seed(0)

# --- ハイパーパラメータ ---
# > 画像を 8x8=64画素の極小サイズに絞ってあるので、CPU でも数分で最後まで走る。
# > `NOISE_BITS` は G に入れる**乱数ビット列 z の本数**(次の章で定義する)。
# > `DISC_HIDDEN` を Generator の `HIDDEN` よりずっと小さくしてあるのは、
# > **D を勝たせすぎない**ための最初の一手である ── 理由は「学習」の章で見る。
SIZE = 8          # 画像サイズ(8x8=64画素の極小タスク)
NOISE_BITS = 64   # Generator に入れる乱数ビット列の長さ
HIDDEN = 128      # Generator の LutLayer 幅
DEPTH = 2         # Generator の段数
DISC_HIDDEN = 32  # Discriminator の LutLayer 幅(Gより弱くしておくのが安定化の要)
DISC_DEPTH = 1
ITERS = 4000      # 敵対学習のイテレーション数
BATCH = 64


# === データ: 崩壊を数えられる4パターンのトイ分布 ===
# > 訓練分布は「上半分・下半分・左半分・右半分だけが黒い 8x8 の二値パターン」の
# > **4種類だけ**にしてある。手書き数字ではなく、こんなに単純な分布を選ぶのには理由がある。
# >
# > **モード崩壊(mode collapse)** — 生成器が分布の一部しか出さなくなり、何度サンプリング
# > しても似た絵ばかりが返る現象。GAN で最も有名な失敗モードである。
# >
# > 分布が**4点しかない**なら、生成した画像を最も近い訓練パターンへ割り当てて数えるだけで
# > 「4種のうち何種を出せているか」が**機械的に測れる**。MNIST では「崩壊した」を目で
# > 判定するしかないが、このトイ分布なら**崩壊が数字になる** ── 図3がまさにそれである。
# MNIST ではなく「上半分・下半分・左半分・右半分」の4パターンにする。
# 分布が4点しかないので、モード崩壊(4種のうち1種しか出さなくなる)が
# 起きたかどうかを機械的に数えられるのがこのトイ分布の利点。
def make_patterns(size: int) -> torch.Tensor:
    pats = torch.zeros(4, 1, size, size)
    half = size // 2
    pats[0, 0, :half, :] = 1.0   # 上半分
    pats[1, 0, half:, :] = 1.0   # 下半分
    pats[2, 0, :, :half] = 1.0   # 左半分
    pats[3, 0, :, half:] = 1.0   # 右半分
    return pats


PATTERNS = make_patterns(SIZE)
PATTERN_NAMES = ['top', 'bottom', 'left', 'right']
print(f'訓練分布: {len(PATTERNS)}種の二値パターン({SIZE}x{SIZE}={SIZE * SIZE}画素)')


# === モデル: Generator と Discriminator ===
# > コードに入る前に、このページで繰り返し出てくる記号を定義しておく。
# >
# > - **z** — Generator への入力である**乱数ビット列**(0/1 が `NOISE_BITS`=64 本並んだもの)。
# >   普通の GAN が使う**潜在(latent)ベクトル** ── 「絵の種」にあたる、外から与える乱数 ──
# >   の役割をそのまま担うが、こちらは実数のガウス乱数ではなく**ビット**である。
# >   つまり物理乱数源や **LFSR(線形帰還シフトレジスタ。数個のフリップフロップと XOR だけで
# >   疑似乱数ビットを吐く小さな回路)** から**そのまま供給できる**。
# >   結果として LogicGAN は「回路の外から 0/1 を64本与えるだけで画像が出てくる」構造になる。
# > - **G(Generator)** — z を受け取って**画像のビット**(8x8=64ビット)を返す論理回路。
# > - **D(Discriminator)** — 画像のビットを受け取って、**ロジット**を1本だけ返す論理回路。
# > - **ロジット** — 「大きいほど本物らしい」という実数のスコア。sigmoid を通すと確率になり、
# >   符号がそのまま「本物と判定 / 偽物と判定」の境目(0)になる。
# >   LogicGAN で実数が現れるのは、D の最後にビットを数える `GroupSum` から先だけである。
# > - **D 精度** — D が本物/偽物を当てられた割合。**0.5 付近なら拮抗していて健全**、
# >   **1.0 に張り付いたら D が勝ちすぎ**で、G に返る勾配が枯れる(=崩壊のサイン)。
# >
# > 下のヘルパーは安定化のつまみの1つ、**入力ノイズ**の実装である。
# --- つまみ: D 入力へのビット反転ノイズ ---
def bit_flip(x: torch.Tensor, p: float) -> torch.Tensor:
    """Discriminator 入力へのインスタンスノイズ(ビット反転)。

    ⚠️ 対処法2: real も fake も同じ確率でビットを反転して D に見せる。
    D が「本物と偽物を完全に見分けられる」状態になると勾配が枯れて G が学べなくなる
    ため、あえて両方を汚して判別を難しくし、game を拮抗させる。p=0 で無効。
    """
    if p <= 0.0:
        return x
    mask = (torch.rand_like(x) < p).to(x.dtype)
    return x * (1.0 - mask) + (1.0 - x) * mask


# --- 学習ループ本体: 4つのつまみだけを引数にする ---
# > GAN は **G と D のバランスが取れているときだけ**学習が進む。どちらかが勝ってしまうと
# > 勾配が枯れて、もう一方は何も学べなくなる。だから GAN には「バランスを保つためのつまみ」が
# > 要る。この関数はそのつまみを**ちょうど4つだけ**引数にしてあり、あとで比べる
# > 安定版(STABLE)と未調整版(NAIVE)の差はこの4つに閉じている。
# >
# > - **更新比 `g_steps`** — D を1回更新するあいだに G を何回更新するかの比(D:G の更新回数比)。
# >   D のほうが学習が速いので、G を多めに回して追いつかせる。
# > - **入力ノイズ `flip_prob`** — 本物にも偽物にも同じ確率でビット反転をかけてから D に見せる。
# >   D が両者を完全に見分けられる状態になるのを防ぐ。
# > - **label smoothing `label_smoothing`** — 「**本物のラベルを 1.0 ではなく 0.8 にする**」
# >   つまみ(STABLE は 0.2)。
# >   D に「絶対に本物だ」と言い切らせないための、GAN では定番の手当て。
# > - **D の学習率 `disc_lr`** — 単純に D の歩幅を小さくして、勝ちすぎを抑える。
# >
# > 学習が終わったら `gan.generate(...)` で 128 枚サンプリングする。これは **hard eval
# > (各ゲートを argmax で1つの論理関数に確定させた、純ブール回路としての実行)** 経路なので、
# > 返ってくる画素は連続緩和の近似値ではなく**厳密に 0/1** である。
def train_gan(*, g_steps: int, flip_prob: float, label_smoothing: float,
              disc_lr: float, gen_lr: float, tag: str) -> dict:
    """LogicGAN を1本学習し、履歴と生成サンプルを返す。

    安定化に効く4つのつまみ(g_steps / flip_prob / label_smoothing / disc_lr)
    だけを引数にしてあり、STABLE と NAIVE の違いはこの4つだけになっている。
    """
    torch.manual_seed(0)  # 2条件を同じ初期回路から出発させる(比較の公平性)
    gan = LogicGAN(LogicGANConfig(
        image_size=SIZE, in_channels=1,
        noise_bits=NOISE_BITS, hidden=HIDDEN, depth=DEPTH,
        disc_hidden=DISC_HIDDEN, disc_depth=DISC_DEPTH,
        n_inputs=2,
    ))
    opt_g = torch.optim.Adam(gan.generator.parameters(), lr=gen_lr)
    opt_d = torch.optim.Adam(gan.discriminator.parameters(), lr=disc_lr)

    g_hist, d_hist, acc_hist = [], [], []
    gan.train()
    for _ in range(ITERS):
        real = PATTERNS[torch.randint(0, len(PATTERNS), (BATCH,))]

        # --- Discriminator 更新 ---
        # fake は d_loss の内部で detach されるので、この勾配は G へ流れない。
        z = gan.generator.sample_noise(BATCH)
        fake = gan.generator(z)
        opt_d.zero_grad()
        d_loss = gan.d_loss(
            bit_flip(real, flip_prob), bit_flip(fake, flip_prob),
            label_smoothing=label_smoothing,   # ⚠️ 対処法3
        )
        d_loss.backward()
        opt_d.step()

        # --- Generator 更新(G:D = g_steps:1) ---
        # ⚠️ 対処法1: D の学習率を下げ、G を複数回まわして敵対バランスを保つ。
        # 損失は non-saturating(max log D)なので初期の勾配消失も避けられる。
        for _ in range(g_steps):
            z = gan.generator.sample_noise(BATCH)
            fake = gan.generator(z)
            g_loss = gan.g_loss(gan.discriminator(fake))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        g_hist.append(g_loss.item())
        d_hist.append(d_loss.item())
        # ⚠️ 対処法4: D精度は「敵対バランス」の体温計。0.5 が理想、1.0 は崩壊。
        with torch.no_grad():
            real_acc = (gan.discriminator(real) > 0).float().mean().item()
            fake_acc = (gan.discriminator(fake) < 0).float().mean().item()
        acc_hist.append((real_acc + fake_acc) / 2.0)

    # eval(= 真理値表を確定した純ブール回路)経路で生成する。出力は厳密に {0,1}。
    samples = gan.generate(128, generator=torch.Generator().manual_seed(7))
    # 各サンプルを最も近い訓練パターンへ割り当て、ハミング距離とモード被覆を測る。
    dist = (samples.flatten(1)[:, None, :]
            - PATTERNS.flatten(1)[None, :, :]).abs().sum(-1)
    nearest = dist.argmin(1)
    coverage = torch.bincount(nearest, minlength=len(PATTERNS)).float() / len(samples)

    late = slice(-200, None)
    result = {
        'tag': tag,
        'gan': gan,
        'g_hist': g_hist, 'd_hist': d_hist, 'acc_hist': acc_hist,
        'samples': samples,
        'late_g': sum(g_hist[late]) / len(g_hist[late]),
        'late_acc': sum(acc_hist[late]) / len(acc_hist[late]),
        'hamming': dist.min(1).values.mean().item(),
        'coverage': coverage,
        'n_modes': int((coverage > 0.05).sum()),
    }
    print(f'[{tag:>6}] late G loss={result["late_g"]:.3f}  '
          f'late D精度={result["late_acc"]:.3f}  '
          f'最近傍ハミング距離={result["hamming"]:.1f}/{SIZE * SIZE}  '
          f'被覆モード数={result["n_modes"]}/{len(PATTERNS)}')
    return result


# === 学習: 安定版と未調整版を並べて回す ===
# > このページの主役はここである。**同じ初期回路から、つまみだけを変えた2本**を回す。
# >
# > - **STABLE** — 更新比 G:D=3:1 / 入力ノイズ p=0.1 / label smoothing 0.2 /
# >   D の学習率 0.001(G の 1/10)
# > - **NAIVE** — 4つの安定化を**全部外した**、素直に書いただけの GAN
# >
# > 出力の数字で見るのは 2つ。**late G loss**(終盤200イテレーションの平均)は
# > G が生きているかどうか、**late D 精度**は敵対バランスの体温計で、**1.0 は崩壊**を意味する。
# > 「GAN は不安定だ」と言葉で言う代わりに、**崩れ方を実測で並べる**のがこの例の狙いである。
# 差分は4つのつまみだけ。NAIVE は「素直に書いたGAN」で、実測で発散する。
print('\n=== 安定版(G:D=3:1 / D入力ノイズ / label smoothing / D学習率↓)===')
stable = train_gan(g_steps=3, flip_prob=0.10, label_smoothing=0.2,
                   disc_lr=0.001, gen_lr=0.01, tag='STABLE')
print('\n=== 未調整版(4つの安定化を全て外した素直な実装)===')
naive = train_gan(g_steps=1, flip_prob=0.0, label_smoothing=0.0,
                  disc_lr=0.01, gen_lr=0.01, tag='NAIVE')

print(f'\n最終メトリクス: '
      f'STABLE late_g={stable["late_g"]:.3f} / late_dacc={stable["late_acc"]:.3f} / '
      f'hamming={stable["hamming"]:.1f}  ||  '
      f'NAIVE late_g={naive["late_g"]:.3f} / late_dacc={naive["late_acc"]:.3f} / '
      f'hamming={naive["hamming"]:.1f}')
# === 健全性チェック: 教材としての主張を assert で固定する ===
# > 上の数字は実行するたびに再現してほしい**主張**なので、文章ではなく `assert` で書いておく。
# > 3本とも、この例が壊れていないことの最低条件である
# > (同じ健全域を `tests/test_logic_gan.py::test_training_dynamics_smoke` も守っている)。
# 健全性1: 安定版の G loss は有界(未調整版の発散に対して拮抗している)。
assert stable['late_g'] < 2.0, f'安定版でも G loss が発散: {stable["late_g"]:.3f}'
# 健全性2: 安定版の D は完全分離に張り付いていない。
assert stable['late_acc'] < 0.99, f'D が G を圧倒: {stable["late_acc"]:.3f}'
# 教材としての主張: 未調整版のほうが敵対バランスは崩れている。
assert naive['late_acc'] >= stable['late_acc'] - 1e-9, '未調整版が崩れていない'

# === 可視化: 生成サンプル・敵対バランス・モード被覆 ===
# --- 描画ヘルパー: 移動平均 ---
# > 敵対学習の損失と判別精度はステップごとの振れが大きく、生の系列のままでは傾向が読めない。
# > 以下の推移図はすべて、この移動平均で均してから描く。
def smooth(values, window=50):
    """移動平均(敵対学習の損失はステップ毎の振れが大きく生では読めない)。"""
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


# --- 図1: 生成サンプル(安定版 / 未調整版) ---
# > 上段が STABLE、下段が NAIVE の生成画像を8枚ずつ。見るのは
# > **「訓練分布の4パターン(半分だけ黒い正方形)の形になっているか」**の一点である。
# >
# > 描いているのは **hard eval**(argmax で確定した純ブール回路)の出力なので、白黒の各マスは
# > 文字通り**回路が出力したビットそのもの**であり、連続緩和した近似値ではない。
# eval経路なので、ここに出ている画素は文字通り回路が出力したビットそのもの。
fig, axes = plt.subplots(2, 8, figsize=(10, 3.4))
for row, res in enumerate((stable, naive)):
    for col in range(8):
        axes[row, col].imshow(res['samples'][col, 0], cmap='gray_r',
                              vmin=0, vmax=1)
        axes[row, col].set_axis_off()
    axes[row, 0].set_title(res['tag'], loc='left', fontsize=9)
fig.suptitle('LogicGAN samples — 64 random bits → logic circuit → 8x8 bits',
             fontsize=11)
fig.tight_layout()
plt.show()

# > **読み取り**: どちらの段も、8x8 の極小タスクではまだ「パターンの形が見え始めた」程度で、
# > 最近傍パターンとのハミング距離は 64 画素中 20 前後ある(実行ごとの数値は上の出力)。
# > ここで見るべき差は絵の綺麗さではなく**学習が生きているかどうか**で、それは次の図2に出る。
# > **同じ回路・同じ初期値で、違うのは4つのつまみだけ**である。どちらも8枚がよく似た形を
# > している ── これがモード崩壊で、数として見えるのは図3。
# --- 図2: 敵対バランスの推移(崩壊の見分け方) ---
# > G と D の力関係を時系列で見る。上段が G loss、下段が D 精度。見どころは**下段**で、
# > 0.5 付近をうろついていれば拮抗して健全、**1.0 に張り付いたら D が勝ちすぎ**である。
# > その結果として上段の G loss が単調に上がっていくのが、崩壊の典型的な足跡になる。
# 見るべきは D精度(下段)。0.5 付近で拮抗していれば健全、1.0 に張り付いたら
# 「D が勝ちすぎて G の勾配が枯れた」= 崩壊。上段の G loss はその結果として
# 未調整版で単調増加していく。
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for res, color in ((stable, '#2e9e4f'), (naive, '#d1495b')):
    axes[0].plot(smooth(res['g_hist']), color=color, label=res['tag'])
    axes[1].plot(smooth(res['acc_hist']), color=color, label=res['tag'])
axes[0].axhline(2.0, color='#888', ls='--', lw=1, label='divergence guide (2.0)')
axes[0].set_ylabel('G loss (moving avg)')
axes[0].set_title('is the Generator still alive? (rising G loss = red flag)')
axes[0].grid(alpha=0.3)
axes[0].legend(loc='upper left')
axes[1].axhline(0.5, color='#888', ls='--', lw=1, label='ideal (0.5 = balanced)')
axes[1].set_ylim(0.4, 1.02)
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('D accuracy (moving avg)')
axes[1].set_title('adversarial balance thermometer (pinned at 1.0 = collapse)')
axes[1].grid(alpha=0.3)
axes[1].legend(loc='lower right')
fig.tight_layout()
plt.show()

# > **読み取り**: NAIVE(赤)は D 精度が早々に 1.0 へ張り付き、G loss が 7 台まで
# > 上がり続ける ── **D が完全に勝ち、G に返る勾配が枯れた**状態である。STABLE(緑)は
# > D 精度が 0.8〜0.9 で止まり(1.0 に張り付かない)、G loss も破線(2.0)の下に留まる。
# > **4つのつまみは「G を強くする」ためではなく「D を勝たせすぎない」ために効いている**。
# --- 図3: モード被覆(縮小レシピでは崩壊するのが正常) ---
# > 生成した128枚を最も近い訓練パターンへ割り当てて、4本の棒に数える。破線が訓練分布
# > (4種が均等なので各 0.25)で、**棒が破線あたりに4本そろえば分布を覆えている**、
# > 1本だけ突出していたら**モード崩壊**である。
# 訓練分布は4パターン均等(=各0.25)。生成サンプルを最近傍パターンで数えて
# 棒グラフにする。1本だけ突出していたらモード崩壊。8x8・4000イテレーションの
# 縮小レシピではこれが起きる —— GAN の難しさが数字で出る箇所。
fig, ax = plt.subplots(figsize=(7, 4))
xs = torch.arange(len(PATTERNS))
ax.bar(xs - 0.2, stable['coverage'], width=0.4, color='#2e9e4f', label='STABLE')
ax.bar(xs + 0.2, naive['coverage'], width=0.4, color='#d1495b', label='NAIVE')
ax.axhline(1 / len(PATTERNS), color='#888', ls='--', lw=1,
           label='training distribution (uniform)')
ax.set_xticks(xs.tolist())
ax.set_xticklabels(PATTERN_NAMES)
ax.set_ylabel('fraction of generated samples')
ax.set_title('mode coverage — are all 4 patterns generated evenly?')
ax.legend()
ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
plt.show()


# > **読み取り**: どちらの条件も棒が1本だけ立ち、被覆モード数は **1/4**。つまり
# > **安定版でもモード崩壊している**。8x8・4000イテレーションという縮小レシピでは
# > ここまでしか行かない、というのがこの例の正直な到達点であり、
# > 「敵対バランスを保つこと(図2)」と「分布を覆うこと(図3)」が**別の難しさ**だと
# > いうことでもある ── 本走(`experiments/train_logic_gan.py`)は 28x28 の二値 MNIST を
# > 30エポック回してこの先へ進む。
# === アーキテクチャ図解と実回路 ===
# > 最後に、いま学習し終えたものが**本当に論理回路なのか**を2枚の図で確かめる。
# > 下の3つのヘルパーは図4を組み立てる部品(角丸の箱・矢印・図の中に貼る小さな画像)で、
# > 読み飛ばしてよい。
# --- 図の部品(箱・矢印・画像貼り付け) ---
def draw_block(ax, x, y, w, h, name, sub, color):
    """アーキテクチャ図の部品: 角丸の箱にクラス名と規模(ゲート数)を書く。"""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.004,rounding_size=0.012',
        facecolor=color, alpha=0.25, edgecolor=color, lw=1.4, zorder=2))
    ax.text(x, y + h * 0.17, name, ha='center', va='center', fontsize=8.4,
            fontweight='bold', zorder=3)
    ax.text(x, y - h * 0.22, sub, ha='center', va='center', fontsize=7.4,
            color='#333333', zorder=3)


def draw_arrow(ax, x0, y0, x1, y1, text='', rad=0.0):
    """ブロック間の矢印(必要なら上にラベルを添える)。"""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(
        arrowstyle='-|>', color='#444444', lw=1.3,
        connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.015, text, ha='center',
                va='bottom', fontsize=7.5, color='#444444')


def put_image(fig, img, rect, caption, cmap='gray_r', vmin=0.0, vmax=1.0):
    """アーキテクチャ図の中に実データの小さな画像を貼る。"""
    ax = fig.add_axes(rect)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
    ax.set_title(caption, fontsize=7.8)
    return ax


# --- 図4: アーキテクチャ図(安定版の実データを1枚流す) ---
# > 学習済みの安定版に**実データを1枚だけ流し**、その途中経過を図の中に貼り付ける。
# > 上段が G(64本の乱数ビット z → 論理回路 → 画像ビット)、下段が D(画像ビット →
# > 論理回路 → 実数ロジット1本)。箱の中の数字は**実際のゲート数**で、右端の点は
# > 本物と偽物に対して D が出したロジットの位置(破線が判定の境目)である。
# >
# > 注目してほしいのは **`readout`** の箱 ── 出力段のことで、ここは
# > **1画素 = LUT 1個**の対応になっている。色の凡例はプロジェクト共通で、
# > 緑=論理ゲート / 橙=実数が現れる場所 / 青=乱数ビット源 / 灰=データ。
# 上段=Generator(乱数ビット → 論理回路 → 画像)、下段=Discriminator
# (画像 → 論理回路 → 実数ロジット1本)。箱の中の数字は実際のゲート数。
# 配色はプロジェクト共通: 緑=論理ゲート / 橙=実数部 / 青=乱数ビット源 / 灰=データ。
C_LOGIC, C_REAL, C_SAMPLE, C_DATA = '#2e9e4f', '#e8850c', '#38bdf8', '#8a8a8a'
gan_s = stable['gan']
gan_s.eval()
with torch.no_grad():
    z_arch = gan_s.generator.sample_noise(
        1, generator=torch.Generator().manual_seed(4))
    fake_arch = gan_s.generator(z_arch)
    real_arch = PATTERNS[1:2]
    d_fake = float(gan_s.discriminator(fake_arch)[0, 0])
    d_real = float(gan_s.discriminator(real_arch)[0, 0])
g_body_gates = sum(int(m.out_dim) for m in gan_s.generator.body)
g_gates = g_body_gates + int(gan_s.generator.readout.out_dim)
d_body_gates = sum(int(m.out_dim) for m in gan_s.discriminator.body)
print(f'Generator={g_gates}ゲート / Discriminator='
      f'{d_body_gates + int(gan_s.discriminator.readout.out_dim)}ゲート  '
      f'D(real)={d_real:+.1f} / D(fake)={d_fake:+.1f}')

fig = plt.figure(figsize=(13.5, 6.4))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.set_axis_off()
bg.text(0.5, 0.965, 'LogicGAN — 64 random bits in, one real number out',
        ha='center', fontsize=12, fontweight='bold')
for i, (label, color) in enumerate((('bits / logic', C_LOGIC),
                                    ('real numbers', C_REAL),
                                    ('random bit source', C_SAMPLE),
                                    ('data', C_DATA))):
    bg.add_patch(FancyBboxPatch((0.02 + 0.135 * i, 0.915), 0.018, 0.022,
                                boxstyle='round,pad=0.002',
                                facecolor=color, alpha=0.3, edgecolor=color))
    bg.text(0.043 + 0.135 * i, 0.926, label, va='center', fontsize=8)

put_image(fig, z_arch[0].reshape(8, 8), [0.020, 0.590, 0.104, 0.220],
          'z: 64 random bits')
draw_arrow(bg, 0.130, 0.70, 0.160, 0.70)
draw_block(bg, 0.265, 0.695, 0.190, 0.175, 'generator.body',
           f'LutLayer x{DEPTH}, {g_body_gates} gates', C_LOGIC)
draw_arrow(bg, 0.362, 0.70, 0.388, 0.70)
draw_block(bg, 0.500, 0.695, 0.200, 0.175, 'generator.readout',
           f'{int(gan_s.generator.readout.out_dim)} gates = 1 per pixel',
           C_LOGIC)
draw_arrow(bg, 0.602, 0.70, 0.632, 0.70)
put_image(fig, fake_arch[0, 0], [0.640, 0.590, 0.104, 0.220],
          'fake image (8x8 bits)')
bg.text(0.885, 0.70, f'the whole generator:\n{g_gates} gates,\n'
        'zero multiplications', ha='center', va='center', fontsize=9,
        color='#333333')

put_image(fig, real_arch[0, 0], [0.020, 0.110, 0.104, 0.220],
          'real pattern')
draw_arrow(bg, 0.692, 0.575, 0.200, 0.320, rad=0.12)
bg.text(0.430, 0.430, 'the generated bits go straight into the discriminator',
        ha='center', fontsize=8.5, color='#444444')
draw_arrow(bg, 0.130, 0.220, 0.170, 0.220)
draw_block(bg, 0.270, 0.220, 0.180, 0.175, 'bit_flip(p=0.1)',
           'instance noise on D inputs', C_LOGIC)
draw_arrow(bg, 0.362, 0.220, 0.388, 0.220)
draw_block(bg, 0.480, 0.220, 0.170, 0.175, 'discriminator.body',
           f'LutLayer, {d_body_gates} gates', C_LOGIC)
draw_arrow(bg, 0.567, 0.220, 0.593, 0.220)
draw_block(bg, 0.680, 0.220, 0.170, 0.175, 'discriminator.readout',
           f'{int(gan_s.discriminator.readout.out_dim)} gates', C_LOGIC)
draw_arrow(bg, 0.767, 0.220, 0.793, 0.220)
draw_block(bg, 0.858, 0.220, 0.130, 0.175, 'GroupSum(k=1)',
           'count bits -> logit', C_REAL)
lax = fig.add_axes([0.952, 0.115, 0.036, 0.215])
lax.axhline(0.0, color='#888888', ls='--', lw=1.0)
lax.scatter([0.5], [d_real], color=C_LOGIC, s=40, zorder=3)
lax.scatter([0.5], [d_fake], color='#d1495b', s=40, zorder=3)
lax.text(0.95, d_real, 'real', fontsize=7, va='center', color=C_LOGIC)
lax.text(0.95, d_fake, 'fake', fontsize=7, va='center', color='#d1495b')
lax.set_xlim(0.3, 2.8)
lax.set_xticks([])
lax.tick_params(labelsize=7)
lax.set_title('D logit', fontsize=7.8)
plt.show()

# > **読み取り**: 橙(実数)は D の末尾、ビットを数える `GroupSum` から先にしか無い。
# > z から画像まで、そして画像から判定の直前まで、**流れているのは 0/1 のビットだけ**で
# > 掛け算は一度も現れない。ロジットは本物側のほうが偽物側より高く出ており(差は上の出力)、
# > 図2で見た「D 精度が 1.0 に張り付かない」様子の内訳がこの1枚に見えている。
# --- 図5: 実回路(学習後の Generator そのもの) ---
# > 締めくくりは、学習の結果そのものである回路の絵。**この図はコードの模式図ではなく、
# > 学習後のパラメータから起こした実物**である。
# >
# > - **左** — Generator 全体(320ゲート)。大きいので `plot_circuit` が自動的に俯瞰図
# >   (層=箱、矢印=配線の本数)へ切り替わる。
# > - **右** — 出力段 `generator.readout` を画像の並びのまま広げたもの。**1マス = 1画素 =
# >   LUT 1個**で、マスに書かれているのは argmax で確定した論理関数の名前
# >   (`XOR` / `NAND` / `A>=B` など)。マスの濃さは、生成サンプルでその画素が
# >   1になった頻度である。
# >
# > **読み取り**: 生成画像のどのマスをどのゲートが吐いているかが、そのまま名前で読める。
# > この名前は **IR(このライブラリが回路の真実源として持つ JSON 表現)** と、そこから
# > 吐かれる **Verilog(FPGA・ASIC に流し込むハードウェア記述)** に焼かれるゲート名と
# > **同一**である。最後の `print` が示すとおり出力段64ゲートは数種類の論理関数に
# > 集中しており、「乱数ビットを入れると絵が出てくる小さな回路」が確かに手元にできている。
# 左: 320ゲートあるので plot_circuit は俯瞰図(層=箱・矢印=配線本数)へ自動で切り替わる。
# 右: 出力段 generator.readout は「1画素 = LUT 1個」。argmax で確定した論理関数の名前を
# 画素の位置に並べると、生成画像のどのマスをどのゲートが吐いているかがそのまま読める
# (この名前は IR / Verilog に焼かれるゲート名と同一)。
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2),
                         gridspec_kw={'width_ratios': [1.1, 1.0]})
plot_circuit(gan_s.generator, ax=axes[0],
             title=f'the trained generator ({g_gates} gates) — auto overview')
names = gate_labels(gan_s.generator.readout)
counts: dict[str, int] = {}
for name in names:
    counts[name] = counts.get(name, 0) + 1
for pix, name in enumerate(names):
    row, col = divmod(pix, SIZE)
    axes[1].add_patch(FancyBboxPatch(
        (col + 0.06, SIZE - 1 - row + 0.06), 0.88, 0.88,
        boxstyle='round,pad=0.01,rounding_size=0.08',
        facecolor=C_LOGIC, alpha=0.18 + 0.5 * float(stable['samples'][:, 0]
                                                    .mean(0)[row, col]),
        edgecolor='#555555', lw=0.6))
    axes[1].text(col + 0.5, SIZE - 1 - row + 0.5, name, ha='center',
                 va='center', fontsize=7.5)
axes[1].set_xlim(0, SIZE)
axes[1].set_ylim(0, SIZE)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title('generator.readout — one gate per pixel, resolved by argmax\n'
                  '(darker = that pixel is on more often in the samples)',
                  fontsize=9.5)
fig.tight_layout()
plt.show()
top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
print('出力段64ゲートの内訳(上位4種): '
      + ' / '.join(f'{n}×{c}' for n, c in top))
