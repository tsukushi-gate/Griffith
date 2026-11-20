"""LogicVAE — 二値潜在VAEを論理ゲートだけで組む例。

06_mnist.py の識別器は「画像 → クラス」の一方向だった。ここでは逆向き、
**回路から画像を作る**側へ踏み込む。VAE(変分オートエンコーダ)を
LutLayer スタックだけで構成し、画像を **96ビットの潜在ビット列** に圧縮してから
復元する。

なぜ「二値潜在」なのか(この例の主題):
    普通のVAEの潜在は連続ガウスで、再パラメータ化 z = μ + σ·ε に **乗算と平方根**
    が要る。論理回路にとってこれは最悪の相手で、そのままでは回路に落ちない。
    LogicVAE は潜在を Bernoulli ビット列に取り替えることでこの壁を回避する。
    Bernoulli サンプリングは u ~ U(0,1) に対する z = 1{q > u} —— つまり
    **確率的なしきい値比較 = 比較器** であり、論理回路そのものである。
    勾配は straight-through(forward=比較器 / backward=σの勾配)で
    エンコーダまで流す(griffith_logic/models/logic_vae.py の _STBernoulli 参照)。

## 全体データフロー(実クラス名つき)

```mermaid
flowchart LR
    X["入力画像 [0,1]<br/>14x14 = 196画素"] --> TH
    TH["ThermometerEncoder<br/>1画素 → 4ビット(計784ビット)"]
    TH --> ENC["encoder_body<br/>LutLayer×2(1,024ゲート)"]
    ENC --> ERO["encoder_readout<br/>LutLayer(768ゲート = 96×8)"]
    ERO --> EGS["encoder_group = GroupSum<br/>8本のビットを足して実数化"]
    EGS --> QL["q = sigmoid(ロジット)<br/>96本の確率(実数)"]
    U["u 〜 一様乱数 U(0,1)"] --> ST
    QL --> ST["_STBernoulli<br/>z = 1 if q &gt; u(= 比較器)"]
    ST --> Z["潜在ビット z<br/>96ビット(0/1)"]
    Z --> DEC["decoder_body<br/>LutLayer×2(1,024ゲート)"]
    DEC --> DRO["decoder_readout<br/>LutLayer(1,568ゲート = 196×8)"]
    DRO --> DGS["decoder_group = GroupSum"]
    DGS --> REC["画素ロジット → 再構成画像"]
    REC -.->|"BCEWithLogits"| L["ELBO = recon + β·KL"]
    QL -.->|"KL は閉じた式で計算"| L

    style TH fill:#10b98126,stroke:#10b981
    style ENC fill:#10b98126,stroke:#10b981
    style ERO fill:#10b98126,stroke:#10b981
    style Z fill:#10b98126,stroke:#10b981
    style DEC fill:#10b98126,stroke:#10b981
    style DRO fill:#10b98126,stroke:#10b981
    style EGS fill:#f9731626,stroke:#f97316
    style QL fill:#f9731626,stroke:#f97316
    style DGS fill:#f9731626,stroke:#f97316
    style REC fill:#f9731626,stroke:#f97316
    style ST fill:#38bdf826,stroke:#38bdf8
```

緑が **ビットのまま進む論理回路**、橙が **実数が現れる場所**、青が
**実数をビットへ落とす比較器**。実数はロジット(=足し算の結果)としてしか現れず、
比較器を通した瞬間にビットへ戻る。

## どう実現しているか(4行で)

1. **エンコーダ**。画素を Thermometer 符号(明るさを「棒温度計」の 0/1 列にする)で
   ビット化し、LutLayer のスタックへ通す。最後に GroupSum が「1になったゲートの本数」を
   数え、96本ぶんの実数ロジットにする。
2. **連続値を確率とみなす**。ロジットに sigmoid をかけた q = σ(ロジット) を、
   そのまま「そのビットが1になる確率」と読む。ガウス潜在のように平均と分散の
   2本を持たず、**確率1本**だけを持つのがビット潜在の身軽さ。
3. **比較器でビットにする**。一様乱数 u を1つ引き、z = 1 if q > u とする。これは
   Bernoulli(q) からの厳密なサンプルであると同時に、**しきい値比較=比較器**そのもの。
   推論時は乱数生成器と比較器1個で済む。勾配は straight-through で、前向きは
   ハードなビット・後ろ向きは ∂q/∂ロジット = q(1−q) を流す。
4. **デコーダと損失**。96ビットを LutLayer スタックへ通し、GroupSum で画素ロジットへ。
   損失は ELBO = recon + β·KL で、recon は画素の BCE、KL は
   KL(Bern(q) ‖ Bern(0.5)) = q·log(q/0.5) + (1−q)·log((1−q)/0.5) という**閉じた式**
   (積分もサンプリングも要らない)。

初めて見る言葉:

- **潜在(latent)** — 画像を圧縮した「中身の表現」。ここでは96ビットの列。
- **ロジット** — 「大きいほど1らしい」という実数のスコア。sigmoid で確率になる。
- **straight-through(ST)** — 前向きは離散化した値を使い、後ろ向きは滑らかな近似の
   勾配を流すテクニック。離散サンプリングで勾配が切れるのを防ぐ。
- **KL** — 2つの確率分布のズレの量。ここでは「潜在ビットが表裏五分五分(Bern(0.5))から
   どれだけ偏ったか」の値。

## 普通のVAEとの違い

| 見るところ | 普通のVAE | LogicVAE |
|---|---|---|
| 潜在の形 | 連続ガウス z ∈ R^d | ビット列 z ∈ {0,1}^96 |
| エンコーダの出力 | 平均 μ と分散 σ² の2本 | 1になる確率 q の1本 |
| サンプリング | 再パラメータ化 z = μ + σ·ε(乗算と平方根) | 比較器 z = 1 if q > u |
| 勾配の通し方 | 再パラメータ化でそのまま微分可能 | straight-through(比較器 + σの勾配) |
| KL項 | ガウス同士の閉形式 | Bern(q) ‖ Bern(0.5) の閉形式 |
| 潜在の動かし方 | ベクトルを少しずらす(連続) | ビットを反転する(離散) |
| 必要な演算 | 浮動小数の乗算 | 真理値表引きと比較(実数は和だけ) |

ページ末尾に、実データが流れる**アーキテクチャ図**、**「連続値がビットになる瞬間」を
数直線で見せる図**、そして**実回路図**を置いた。

このサンプルの縮小レシピ:
    本走(experiments/train_logic_vae.py)は 28x28 binary MNIST を30エポック回すが、
    ここはCPUで2〜3分に収まるよう「14x14 に縮小・訓練8000枚・3000 step」にした。
    到達点は「再構成が数字として読める」レベルで、prior 生成は数字の"種"が
    見え始める程度に留まる(生成品質を狙うレシピは本走スクリプトを参照)。

実行: uv run python examples/18_logic_vae.py
期待結果(seed=0 実測・CPU約80秒): recon BCE=43.3 / KL=13.2 /
          再構成の画素一致率=0.907 / active bits=1.00(潜在96ビットが全て働いている)
"""

# === 準備: import と乱数種 ===
# > 使うのは PyTorch と matplotlib、そして本ライブラリの `LogicVAE`(モデル本体)・
# > `load_mnist`(データ)・`plot_circuit`(回路描画)だけ。VAE のための特別な
# > フレームワークは要らない。
# >
# > 先に言葉をそろえておく。
# >
# > - **オートエンコーダ** — 画像をいったん小さな表現へ押し込め、そこから元の画像を
# >   復元するように学習するモデル。「押し込める側」をエンコーダ、「戻す側」をデコーダと呼ぶ。
# > - **VAE(変分オートエンコーダ)** — オートエンコーダに「押し込めた表現を、あらかじめ
# >   決めた分布(prior)へ近づける」という縛りを足したもの。この縛りがあるおかげで、学習後は
# >   **分布からランダムに引くだけで新しい画像を作れる**ようになる。
# > - **潜在(latent)** — 押し込めた先の表現そのもの。この例では **96 ビットの 0/1 の列**。
# > - **事前分布(prior)** — 潜在が従ってほしい分布。ここでは全ビット表裏五分五分の
# >   Bernoulli(0.5)、つまり「96 回のコイン投げ」。
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch
from torch.utils.data import DataLoader, Subset

from griffith_logic.analysis.circuit_viz import plot_circuit
from griffith_logic.datasets import load_mnist
from griffith_logic.models import LogicVAE, LogicVAEConfig

# LUTの初期真理値・ランダム配線・Bernoulliノイズすべてが乱数依存なので固定する。
torch.manual_seed(0)

# === ハイパーパラメータとデータ ===
# > CPU で2〜3分に収まる縮小レシピ。`IMG` / `LATENT` / `STEPS` を増やすほど再構成は
# > 綺麗になるが、ページの生成時間もそのまま延びる。
# >
# > いちばん効くのは **`LATENT = 96`** で、これが**回路の情報の通り道の幅**にあたる。
# > 196 画素ぶんの絵は、必ずこの 96 本の線を通ってから復元される。
# > `KL_WEIGHT` は後の「学習」章で出てくる **β**(潜在を prior へ寄せる罰の重み)である。
# --- 縮小レシピの設定値 ---
# CPUで2〜3分に収まる縮小レシピ。IMG/LATENT/STEPS を増やすほど再構成は綺麗になる。
IMG = 14          # 画像サイズ(28x28を2x2平均プーリングで縮小)
LATENT = 96       # 二値潜在のビット数(= この回路の「情報の通り道」の幅)
HIDDEN = 512      # エンコーダ/デコーダのLutLayer幅
DEPTH = 2         # LutLayerスタックの段数
STEPS = 3000      # 学習イテレーション数
BATCH = 128
N_TRAIN = 8000    # 訓練に使う枚数
KL_WEIGHT = 0.1   # β-VAE の β。大きいほど潜在が prior に寄る(=情報を捨てる)

# --- データ: 14x14 の二値 MNIST ---
# > 手書き数字を **2値画像**にして使う。VAE の再構成損失は画素ごとの BCE(後述)なので、
# > 目標となる画素は厳密に 0 か 1 でなければならない。
# >
# > 28x28 のままだと 784 画素を 96 ビットへ落とすことになり、3000 step では形が潰れる。
# > 2x2 平均プーリングで **14x14 = 196 画素**に縮めてからしきい値 0.3 で二値化し、
# > 「**196 画素 → 96 ビット → 196 画素**」という素直な圧縮問題にしている。
# binary MNIST。VAEの再構成損失は BCE なので、目標画素は厳密に {0,1} でなければ
# ならない。28x28 のまま扱うと画素数784に対し潜在96ビットの圧縮が厳しく、
# 3000 step では形が潰れる。2x2 平均プーリングで 14x14(196画素)に落としてから
# しきい値 0.3 で二値化し、「196画素 → 96ビット → 196画素」の素直な圧縮にする。
train_set, _test_set = load_mnist(binarize_threshold=None)
raw, _labels = next(iter(DataLoader(Subset(train_set, range(N_TRAIN)),
                                    batch_size=N_TRAIN)))
data = (F.avg_pool2d(raw, 2) > 0.3).float()   # (N, 1, 14, 14) {0,1}
print(f'data={tuple(data.shape)} 前景画素率={data.mean():.3f} '
      f'({IMG}x{IMG}={IMG * IMG}画素 → {LATENT}ビット潜在)')

# === モデル: エンコーダ・比較器サンプラ・デコーダ ===
# > このページで使う記号は **q / u / z** の3つだけ。**ここで一度に定義しておく**。
# >
# > - **q** — エンコーダが出す、**潜在の各ビットが 1 になる確率**(0〜1 の実数)。
# >   潜在は 96 ビットあるので、q も 96 本ぶんある。「この画像なら 3 番目のビットは
# >   0.8 の確率で 1」という読み方をする値である。
# > - **u** — サンプリングのために引く**一様乱数**(0〜1 のあいだから等確率で引く数)。
# >   **ビットごとに 1 個ずつ**引くので、これも 96 本ぶん。
# > - **z** — 実際にサンプリングされた**潜在ビット列**(96 個の 0/1)。作り方は
# >   `z = 1 if u < q else 0` の 1 行だけ。q が大きいビットほど 1 になりやすい。
# >
# > この `u < q` という**たった 1 回の比較**が、そのまま **Bernoulli(q) からのサンプリング**に
# > なっている(**Bernoulli** = 表の出る確率が q のコイン投げ)。つまり
# > **確率 q を持つ回路の実体は「乱数源 1 個と比較器 1 個」**であり、掛け算も指数関数も要らない。
# > 後の図5の数直線は、まさにこの比較のようすを絵にしたものである。
# >
# > **なぜガウス潜在ではなく Bernoulli(二値潜在)なのか。** 普通の VAE の潜在は実数ベクトルで、
# > サンプリングは **再パラメータ化**(乱数を外から入れて微分可能にする常套手段)
# > `z = μ + σ·ε` ── **乗算と平方根**を含む式で行う。実数の潜在を選んだ時点で
# > **回路の途中に浮動小数が居座る**ことになり、そのままではハードウェアに焼けない。
# > 潜在をビット列に取り替えると、**エンコーダの出口からデコーダの入口までがちょうど
# > 96 本の配線**になり、回路として綺麗に閉じる。
# >
# > **straight-through** — 前向き(推論)では 0/1 に丸め、後ろ向き(勾配)だけは
# > 丸めなかったことにして通す近似。離散化すると勾配は本来ゼロになって学習が止まるが、
# > この手口で「比較器の向こう側」へ勾配を届かせる。離散サンプリングを微分可能に扱う定番。
# >
# > 部品の名前も見ておく。
# >
# > - **ThermometerEncoder** — 明るさを「棒温度計」のように 0/1 の列へ変換する符号化。
# >   1 画素を 4 ビットにするので 196 画素 → 784 ビットになる。
# > - **GroupSum** — 論理ゲートの出口。出力ビットを k 組に分け、**組ごとに 1 の本数を数えて**
# >   実数に変える。ここが「ビットの世界」と「実数の世界」の境目である。
# エンコーダ: 画素 → Thermometer符号化 → LutLayerスタック → GroupSum で
#             潜在ビットのロジット(実数)を作る。
# サンプラー: straight-through Bernoulli(= 比較器)。ここだけが確率的。
# デコーダ:   潜在ビット {0,1} → LutLayerスタック → GroupSum で画素ロジット。
# eval時は全 LutLayer が真理値表を確定した純ブール回路として動き、
# 中間表現は厳密に {0,1} を保つ(実数が現れるのは最終の和だけ = 周辺FP方針)。
model = LogicVAE(LogicVAEConfig(
    image_size=IMG,
    in_channels=1,
    encode_bits=4,       # 画素あたりのThermometerビット数
    latent_bits=LATENT,
    hidden=HIDDEN,
    depth=DEPTH,
    n_inputs=2,          # 1LUTが見る入力本数
    kl_weight=KL_WEIGHT,
))
n_params = sum(p.numel() for p in model.parameters())
print(f'LogicVAE: hidden={HIDDEN} depth={DEPTH} latent_bits={LATENT} '
      f'params={n_params:,}')


# --- 評価関数: 純ブール回路として測る ---
# > 品質は必ず **hard eval** の経路で測る。**hard eval** とは、各 LUT を argmax で 1 枚の
# > 真理値表に確定させ、学習用の連続緩和を一切使わない状態のこと ── FPGA に焼くのと
# > 同じ純粋な Boolean 回路として動かす、ということである。
# >
# > 見る数字は2つ。**pix_acc** は再構成画像を 0.5 でしきい値化したときの画素一致率。
# > **active** は潜在 96 ビットのうち「常時 0/常時 1 に張り付いていない」割合で、
# > **posterior collapse**(潜在が使われなくなり、デコーダが入力を無視して平均的な絵ばかり
# > 出すようになる現象)が起きていないかの見張り役である。1.0 に近いほど健全。
@torch.no_grad()
def evaluate(n: int = 512) -> dict:
    """eval(純ブール回路)経路での再構成品質を測る。

    - pix_acc: 再構成を0.5でしきい値化した画素一致率
    - active : 潜在ビットのうち「常時0/常時1に張り付いていない」割合。
               1.0に近いほど posterior collapse していない(潜在が使われている)。
    """
    model.eval()
    x = data[:n]
    out = model(x)
    recon_bits = (torch.sigmoid(out['recon_logits']) > 0.5).float()
    pix_acc = (recon_bits == x).float().mean().item()
    rate = out['z'].mean(dim=0)                      # 各ビットの発火率
    active = ((rate > 0.01) & (rate < 0.99)).float().mean().item()
    model.train()
    return {'pix_acc': pix_acc, 'active': active}


# === 学習: ELBO = recon + β·KL ===
# > 損失は **ELBO**(evidence lower bound。VAE で最小化する標準の目的関数)で、中身は
# > **`recon + β·KL`** という 2 項の足し算しかない。
# >
# > - **recon(再構成項)** — 元画像をどれだけ復元できたかの罰。画素ごとの
# >   **BCE**(binary cross entropy = 2値の食い違いを測る損失)の総和で測る。
# >   実装は **BCEWithLogits**、つまり sigmoid を内側に畳み込んだ数値的に安定な版を使う。
# > - **KL(KL ダイバージェンス)** — 2つの確率分布のズレの量。ここでは、エンコーダが出した
# >   潜在分布 Bern(q) を prior の Bern(0.5) からどれだけ離したかの罰であり、平たく言えば
# >   **「潜在に情報を詰め込むためのコスト」**。ビットが表裏五分五分から偏るほど高くつく。
# > - **β** — その KL にかける重み(このコードでは `KL_WEIGHT = 0.1`)。大きくするほど潜在は
# >   prior に寄り、情報を捨てて生成寄りになる。小さくすると復元は良くなるが潜在が prior から
# >   離れ、コイン投げからの生成が効かなくなる。
# >
# > Bernoulli 同士の KL は
# > `q·log(q/0.5) + (1−q)·log((1−q)/0.5)` という**閉じた式**で書けるので、積分も
# > サンプリングも要らない。ガウス VAE と同じく、KL 項は 1 行で済む。
# 損失は ELBO = recon(BCEWithLogits の画素総和) + β · KL(Bern(q) ‖ Bern(0.5))。
# 中身は普通のPyTorch。論理ゲート網であることを意識する箇所はどこにもない。
opt = torch.optim.Adam(model.parameters(), lr=0.01)
history = []  # (step, recon, kl, pix_acc)

for step in range(STEPS + 1):
    if step % 250 == 0:
        m = evaluate()
        with torch.no_grad():
            probe = model(data[:256])
            parts = model.elbo_loss(data[:256], probe)
        history.append((step, parts['recon'].item(), parts['kl'].item(),
                        m['pix_acc']))
        if step % 500 == 0:
            print(f'step {step:>5}  recon={parts["recon"].item():7.2f}  '
                  f'kl={parts["kl"].item():6.2f}  '
                  f'pix acc={m["pix_acc"]:.4f}  active bits={m["active"]:.2f}')

    idx = torch.randint(0, N_TRAIN, (BATCH,))
    x = data[idx]
    out = model(x)
    loss = model.elbo_loss(x, out)['loss']
    opt.zero_grad()
    loss.backward()
    opt.step()

final = evaluate()
steps, recon_hist, kl_hist, acc_hist = zip(*history, strict=True)
print(f'\n最終 recon BCE={recon_hist[-1]:.2f} / KL={kl_hist[-1]:.2f} / '
      f'画素一致率={final["pix_acc"]:.4f} / active bits={final["active"]:.2f}')
# > 上のログで、**recon が下がりながら KL が上がっていく**のが見どころ。KL の上昇は
# > 「潜在ビットを五分五分から偏らせて情報を詰め込んだ」ことの表れで、その対価として
# > 再構成が良くなっている。2 項のせめぎ合いがそのまま学習の実体である。
# >
# > `active bits` が 1.00 のまま推移していれば posterior collapse は起きておらず、
# > 96 ビットが全部働いていることになる。
# 全画素0を出すだけの自明解でも一致率は前景率ぶんだけ稼げる。その水準を明確に
# 上回っていること、かつ潜在が collapse していないことを最低条件とする。
assert final['pix_acc'] > 0.85, f'再構成が弱すぎる: {final["pix_acc"]:.4f}'
assert final['active'] > 0.5, f'潜在が collapse している: {final["active"]:.2f}'

# === 可視化: 再構成・ビット反転・生成 ===
# --- 図1: 元画像 / 96ビット潜在 / 再構成 ---
# > 8 枚の数字について、上段に**元画像**、中段に**潜在ビット列 z**(96 ビットを 8x12 に
# > 折り返して帯にしたもの)、下段に**再構成**を並べる。
# >
# > 見るべきは中段である。ここに写っているモノクロの点々が、**回路が持ち歩いている情報の
# > すべて**であり、下段の絵はこの 96 個の 0/1 だけから作られている。
model.eval()
with torch.no_grad():
    x_ref = data[:8]
    out = model(x_ref)
    recon = torch.sigmoid(out['recon_logits'])
    z_ref = out['z']

fig, axes = plt.subplots(3, 8, figsize=(10, 4.4))
for i in range(8):
    axes[0, i].imshow(x_ref[i, 0], cmap='gray_r', vmin=0, vmax=1)
    # 潜在96ビットを 8x12 に折り返して「ビットの帯」として見せる。
    axes[1, i].imshow(z_ref[i].reshape(8, 12), cmap='gray_r', vmin=0, vmax=1)
    axes[2, i].imshow(recon[i, 0], cmap='gray_r', vmin=0, vmax=1)
for ax in axes.ravel():
    ax.set_axis_off()
axes[0, 0].set_title('input', loc='left', fontsize=9)
axes[1, 0].set_title(f'latent {LATENT} bits', loc='left', fontsize=9)
axes[2, 0].set_title('reconstruction', loc='left', fontsize=9)
fig.suptitle('LogicVAE — 196 pixels → 96 latent bits → 196 pixels', fontsize=11)
fig.tight_layout()
plt.show()

# > 下段の再構成は数字として読める一方、線は元より太く、細かい跳ねは丸められている。
# > 196 画素を 96 ビットへ落としているので、**捨てられた情報が「細部」として消えている**
# > わけである。中段の帯を見比べると、同じ数字どうしでも帯の模様が違う ── 潜在は
# > 「クラス」ではなく「この 1 枚の絵」を符号化している。
# --- 図2: 潜在ビットを1本ずつ反転する ---
# > 学習済みの潜在から 1 枚選び、**先頭 8 ビットを 1 本ずつ反転**して復号し直す。
# > 上段が反転後の画像、下段がもとの画像との差分(赤=新たに点いた画素 / 青=消えた画素)。
# >
# > **なぜ「潜在を少し動かす」ができないのか。** 連続潜在の VAE なら潜在ベクトルを
# > わずかにずらして中間の絵を作れるが、二値潜在で許される操作は **ビット反転しかない**。
# > 連続潜在の補間に相当するのがこの 1 ビット反転で、**生成の操作が完全に離散**に
# > なっている。1 ビット反転とは、要するに**回路への入力線 1 本の 0/1 を差し替える**ことである。
with torch.no_grad():
    base_z = z_ref[:1]                                   # 1枚ぶんの潜在
    base_img = torch.sigmoid(model.decode(base_z))[0, 0]
    flip_bits = [0, 1, 2, 3, 4, 5, 6, 7]                 # 先頭8ビットを反転
    flipped = []
    for b in flip_bits:
        z = base_z.clone()
        z[0, b] = 1.0 - z[0, b]
        flipped.append(torch.sigmoid(model.decode(z))[0, 0])

fig, axes = plt.subplots(2, 8, figsize=(10, 3.2))
for i, (b, img) in enumerate(zip(flip_bits, flipped, strict=True)):
    axes[0, i].imshow(img, cmap='gray_r', vmin=0, vmax=1)
    axes[0, i].set_title(f'flip bit {b}', fontsize=8)
    # 差分: +1(新たに点灯)を赤、-1(消灯)を青で表示する
    axes[1, i].imshow(img - base_img, cmap='bwr', vmin=-1, vmax=1)
for ax in axes.ravel():
    ax.set_axis_off()
fig.suptitle('flipping one latent bit (top) and its pixel-wise effect (bottom)',
             fontsize=11)
fig.tight_layout()
plt.show()

# > 下段の差分を見ると、**ビットごとに担当する場所がはっきり違う**ことが分かる。あるビットは
# > 数字の上半分を、別のビットは左下の払いを塗り替える。1 本 1 本が画像のどこかの特徴を
# > 受け持っており、96 ビットはただの圧縮結果ではなく**部品として意味を持った表現**になっている。
# > また 1 ビット動かしただけで絵が大きく飛ぶことがあるのも二値潜在の特徴で、
# > 連続潜在のようなじわじわした変化は原理的に作れない。
# --- 図3: prior からの生成と、学習曲線 ---
# > 1 枚目は **prior Bernoulli(0.5) からの生成**。つまり潜在 z を**96 回のコイン投げ**で作り、
# > デコーダに通しただけのもの ── 訓練画像のどれとも違う新しい絵である。
# > 2 枚目は学習曲線で、左軸に recon(橙)、右軸に KL(緑)を重ねている。
with torch.no_grad():
    gen = model.generate(8, generator=torch.Generator().manual_seed(5))

fig, axes = plt.subplots(2, 4, figsize=(10, 4.6))
for ax, img in zip(axes.ravel()[:4], gen[:4], strict=False):
    ax.imshow(img[0], cmap='gray_r', vmin=0, vmax=1)
    ax.set_axis_off()
axes[0, 0].set_title('prior samples (z ~ Bernoulli(0.5))', loc='left', fontsize=9)
for ax, img in zip(axes.ravel()[4:], gen[4:], strict=False):
    ax.imshow(img[0], cmap='gray_r', vmin=0, vmax=1)
    ax.set_axis_off()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, recon_hist, 'o-', color='#e8850c', label='recon BCE (nats/image)')
ax.set_xlabel('step')
ax.set_ylabel('recon BCE')
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(steps, kl_hist, 's-', color='#2e9e4f', label='KL (nats)')
ax2.set_ylabel('KL to Bernoulli(0.5)')
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [ln.get_label() for ln in lines], loc='upper right')
ax.set_title('LogicVAE — recon vs KL trade-off')
fig.tight_layout()
plt.show()


# > 生成のほうは、3000 step の縮小レシピでは**「数字の種」が見え始める程度**で、はっきりした
# > 数字にはならない。潜在が prior にまだ十分寄っていないためで、生成品質を狙うレシピは
# > 本走スクリプト(`experiments/train_logic_vae.py`)を参照してほしい。
# >
# > 学習曲線のほうははっきりしている。**KL が上がりながら recon が下がる** ── 潜在ビットを
# > 五分五分から偏らせる罰を払って、そのぶん復元を買っている。これが β のつまみで
# > 調整している綱引きそのものである。
# === アーキテクチャ図解と実回路 ===
# --- 図の部品(ヘルパー関数) ---
# > ここから先はアーキテクチャ図を描く準備。角丸の箱・矢印・図中に埋め込む小さな実データ画像の
# > 3つを描く関数を用意する(図の見た目の都合だけなので、読み飛ばしてよい)。
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


# --- 図4: 端から端まで、実データを1枚流す ---
# > **実際の画像 1 枚がモデルを通り抜けるようす**を、そのまま図にする。上段がエンコーダ
# > (画像 → ビット → ゲート網 → 確率 q が 96 本)、下段がサンプラとデコーダ
# > (q → 比較器 → 潜在ビット z → ゲート網 → 復元画像)。
# >
# > 図中の小さな画像は説明用のイラストではなく、**このコードが計算した本物の値**である。
# > 箱の中の数字も実際のゲート数(= LutLayer の出力ニューロン数)。
# >
# > 配色はプロジェクト共通で、**緑=論理ゲート(ビットのまま進む) / 橙=実数が現れる場所 /
# > 青=比較器(実数をビットへ落とす) / 灰=データ**。どこまでが回路として焼ける区間かを
# > 色で追えるようにしてある。
# 上段=エンコーダ(画像 → ビット → ゲート網 → 確率96本)、
# 下段=サンプラとデコーダ(確率 → 比較器 → 96ビット → ゲート網 → 画像)。
# 箱の中の数字は実際のゲート数(= LutLayer の出力ニューロン数)。
# 配色はプロジェクト共通: 緑=論理ゲート / 橙=実数部 / 青=比較器 / 灰=データ。
C_LOGIC, C_REAL, C_SAMPLE, C_DATA = '#2e9e4f', '#e8850c', '#38bdf8', '#8a8a8a'
model.eval()
with torch.no_grad():
    x_arch = data[:1]
    lat_logits = model.encode(x_arch)
    q_arch = torch.sigmoid(lat_logits)[0]
    u_arch = torch.rand(LATENT, generator=torch.Generator().manual_seed(9))
    z_arch = (q_arch > u_arch).float()
    recon_arch = torch.sigmoid(model.decode(z_arch[None, :]))[0, 0]
enc_gates = sum(int(m.out_dim) for m in model.encoder_body)
dec_gates = sum(int(m.out_dim) for m in model.decoder_body)
print(f'ゲート数: encoder_body={enc_gates} / encoder_readout='
      f'{int(model.encoder_readout.out_dim)} / decoder_body={dec_gates} / '
      f'decoder_readout={int(model.decoder_readout.out_dim)}')

fig = plt.figure(figsize=(13.5, 6.6))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.set_axis_off()
bg.text(0.5, 0.965, 'LogicVAE — 196 pixels to 96 bits and back, end to end',
        ha='center', fontsize=12, fontweight='bold')
for i, (label, color) in enumerate((('bits / logic', C_LOGIC),
                                    ('real numbers', C_REAL),
                                    ('comparator (sampling)', C_SAMPLE),
                                    ('data', C_DATA))):
    bg.add_patch(FancyBboxPatch((0.02 + 0.135 * i, 0.915), 0.018, 0.022,
                                boxstyle='round,pad=0.002',
                                facecolor=color, alpha=0.3, edgecolor=color))
    bg.text(0.043 + 0.135 * i, 0.926, label, va='center', fontsize=8)

put_image(fig, x_arch[0, 0], [0.020, 0.575, 0.114, 0.235], 'input  14x14')
draw_arrow(bg, 0.140, 0.69, 0.172, 0.69)
draw_block(bg, 0.250, 0.685, 0.150, 0.175, 'ThermometerEncoder',
           '196 px -> 784 bits', C_LOGIC)
draw_arrow(bg, 0.328, 0.69, 0.352, 0.69)
draw_block(bg, 0.430, 0.685, 0.150, 0.175, 'encoder_body',
           f'LutLayer x{DEPTH}, {enc_gates} gates', C_LOGIC)
draw_arrow(bg, 0.508, 0.69, 0.532, 0.69)
draw_block(bg, 0.610, 0.685, 0.150, 0.175, 'encoder_readout',
           f'{int(model.encoder_readout.out_dim)} gates (96x8)', C_LOGIC)
draw_arrow(bg, 0.688, 0.69, 0.712, 0.69)
draw_block(bg, 0.790, 0.685, 0.150, 0.175, 'GroupSum(k=96)',
           'count bits -> 96 logits', C_REAL)
draw_arrow(bg, 0.868, 0.69, 0.892, 0.69)
put_image(fig, q_arch.reshape(8, 12), [0.898, 0.575, 0.090, 0.235],
          'q = sigmoid(logit)', cmap='viridis')

draw_arrow(bg, 0.943, 0.556, 0.943, 0.470)
draw_arrow(bg, 0.943, 0.462, 0.135, 0.462)
draw_arrow(bg, 0.135, 0.454, 0.135, 0.392)
bg.text(0.540, 0.478, 'the only place where a real number becomes a bit',
        ha='center', fontsize=8.5, color='#444444')
draw_block(bg, 0.135, 0.300, 0.190, 0.175, '_STBernoulli',
           'z = 1 if q > u   (comparator)', C_SAMPLE)
draw_arrow(bg, 0.232, 0.305, 0.256, 0.305)
put_image(fig, z_arch.reshape(8, 12), [0.262, 0.190, 0.090, 0.235],
          'z: 96 latent bits')
draw_arrow(bg, 0.358, 0.305, 0.386, 0.305)
draw_block(bg, 0.470, 0.300, 0.150, 0.175, 'decoder_body',
           f'LutLayer x{DEPTH}, {dec_gates} gates', C_LOGIC)
draw_arrow(bg, 0.548, 0.305, 0.560, 0.305)
draw_block(bg, 0.640, 0.300, 0.150, 0.175, 'decoder_readout',
           f'{int(model.decoder_readout.out_dim)} gates (196x8)', C_LOGIC)
draw_arrow(bg, 0.718, 0.305, 0.734, 0.305)
draw_block(bg, 0.810, 0.300, 0.140, 0.175, 'GroupSum(k=196)',
           'count bits -> pixel logits', C_REAL)
draw_arrow(bg, 0.883, 0.305, 0.897, 0.305)
put_image(fig, recon_arch, [0.900, 0.202, 0.088, 0.185], '')
bg.text(0.944, 0.178, 'reconstruction', ha='center', fontsize=7.8)
bg.text(0.06, 0.075, 'loss = BCE(reconstruction, input)  +  beta * '
        'KL(Bern(q) || Bern(0.5))', fontsize=9, color='#444444')
plt.show()

# > 橙の箱が **2 つしかない**ことがこの図の要点である。GroupSum が「1 になったゲートの本数」を
# > 数えるところだけが実数で、あとは全部ビットのまま流れている。**実数はロジット(=足し算の
# > 結果)としてしか現れず、青い比較器を通った瞬間にビットへ戻る**(= 周辺 FP 方針。実数を
# > 回路の周辺部だけに追いやり、中心はビットで閉じる設計方針)。
# >
# > 上段の出口の `q` の絵と下段の `z` の絵を見比べると、なめらかな濃淡だった q が、比較器を
# > 通った直後に白黒 96 個の点に変わっているのが分かる。ここが**この回路で唯一、実数が
# > ビットになる場所**である。
# --- 図5: 連続値がビットになる瞬間 ---
# > 前の図の青い箱 ── **比較器の中身**を数直線で拡大する。
# >
# > 左は潜在の先頭 10 ビットぶん。緑の帯の長さが **q**(そのビットが 1 になる確率)、
# > 青い三角が引いた**一様乱数 u** の位置で、u が帯の中に落ちれば **z = 1**、外なら **z = 0**。
# > **なぜ比較器 1 個で確率が作れるのか** ── `u < q` というこの 1 回の比較が、そのまま
# > Bernoulli(q) からのサンプリングだからである(乱数源 + 比較器 = 確率的ビット源)。
# >
# > 右は同じ 1 枚の画像についての 96 ビットぶんの q のヒストグラム。0 と 1 の両端に寄った
# > ビットほど「乱数によらずほぼ決め打ち」で、真ん中付近のビットほど揺らぐ。
# 左: 潜在の先頭10ビットについて、確率 q(緑の帯)と引いた乱数 u(縦線)を数直線に置く。
# 右: 96ビットぶんの q の分布。0/1 の両端に寄っているほど「決め打ち」のビット。
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                         gridspec_kw={'width_ratios': [1.7, 1.0]})
n_line = 10
for b in range(n_line):
    y = n_line - 1 - b
    q_b, u_b = float(q_arch[b]), float(u_arch[b])
    axes[0].plot([0, 1], [y, y], color='#cccccc', lw=1.0, zorder=1)
    axes[0].plot([0, q_b], [y, y], color=C_LOGIC, lw=6, alpha=0.35, zorder=2,
                 solid_capstyle='butt')
    axes[0].plot([q_b, q_b], [y - 0.28, y + 0.28], color=C_LOGIC, lw=2,
                 zorder=3)
    axes[0].plot([u_b], [y], marker='v', color=C_SAMPLE, ms=8, zorder=4)
    bit = int(q_b > u_b)
    axes[0].text(1.03, y, f'q={q_b:.2f}  u={u_b:.2f}  ->  z={bit}',
                 va='center', fontsize=8.5,
                 color=C_LOGIC if bit else '#777777')
axes[0].set_yticks(range(n_line))
axes[0].set_yticklabels([f'bit {n_line - 1 - i}' for i in range(n_line)],
                        fontsize=8)
axes[0].set_xlim(-0.02, 1.55)
axes[0].set_ylim(-0.7, n_line - 0.3)
axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axes[0].set_xlabel('probability axis  [0, 1]')
axes[0].set_title('green bar = q (chance of 1),  blue marker = random u\n'
                  'the comparator outputs 1 when u falls inside the bar',
                  fontsize=9.5)
axes[1].hist(q_arch.numpy(), bins=20, range=(0, 1), color=C_REAL, alpha=0.75)
axes[1].axvline(0.5, color='#888888', ls='--', lw=1.2)
axes[1].set_xlabel('q = sigmoid(latent logit)')
axes[1].set_ylabel('number of latent bits')
axes[1].set_title(f'all {LATENT} latent bits of this one image', fontsize=9.5)
axes[1].grid(alpha=0.3, axis='y')
fig.suptitle('how a real number turns into a bit — Bernoulli sampling is a '
             'comparator', fontsize=11)
fig.tight_layout()
plt.show()

# > 数直線の右側、q が 0 か 1 に振り切れたビットは、乱数 u をどこに引いても答えが変わらない。
# > 学習が進むほどこうした「決め打ちビット」が増える一方、KL 項がそれを押し戻すので、
# > **全ビットが振り切れることはない** ── その均衡が β で決まっている。
# --- 図6: 学習済みデコーダの実回路 ---
# > 最後に、ここまで「デコーダ」と呼んできたものの**中身そのもの**を描く。
# >
# > 左は学習済みデコーダ。ゲートが 2,592 個あるので `plot_circuit` は俯瞰図
# > (層=箱・矢印=配線本数)へ自動で切り替わる。右は同じクラス・同じ配線規則で組んだ
# > **縮小版**(潜在 8 ビット → 2x2 画素、44 ゲート)で、こちらは 1 ノード = 1 個の LUT、
# > 線 = 実際の配線をすべて描いている。
# >
# > 読み取りを先に書いておく。右の図で、**左端に並ぶ 8 個の入力が潜在ビット z** であり、
# > そこから右へ向かう配線の束がそのまま「デコーダ = 論理回路」の姿である。VAE のデコーダが
# > 行列積の連なりではなく**ゲートと配線の網**として存在していることが、この 1 枚で見て取れる。
# > 縮小版は未学習なのでノードは `LUT`(16 種の論理関数の混合)のまま表示されるが、
# > 学習が進むとこの 1 つ 1 つが AND / XOR などの具体的な論理関数に確定する。
# > 左の俯瞰図の矢印に書かれた本数は、**エンコーダの出口からここまで 96 本で閉じている**
# > という、この例の主題そのものを数字で示している。
decoder_circuit = nn.Sequential(*model.decoder_body, model.decoder_readout,
                                model.decoder_group)
mini = LogicVAE(LogicVAEConfig(
    image_size=2, in_channels=1, encode_bits=2, latent_bits=8,
    hidden=12, depth=1, n_inputs=2,
))
mini_decoder = nn.Sequential(*mini.decoder_body, mini.decoder_readout,
                             mini.decoder_group)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.0),
                         gridspec_kw={'width_ratios': [1.0, 1.5]})
plot_circuit(decoder_circuit, ax=axes[0],
             title='the trained decoder (2,592 gates) — auto overview')
plot_circuit(mini_decoder, resolve=False, ax=axes[1],
             title='the same decoder shrunk to 8 latent bits -> 2x2 pixels '
                   '(44 gates, every gate and wire drawn)')
fig.tight_layout()
plt.show()
