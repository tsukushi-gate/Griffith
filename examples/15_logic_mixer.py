"""LogicMixer(MLP-Mixerの論理ゲート版)でMNISTを解く例。

MLP-Mixer(Tolstikhin et al., 2021)は「attentionを使わずMLPだけで画像を解く」
アーキテクチャで、2種類の混合を交互に積むだけの構成をしている:

    token-mixing   … パッチ(token)をまたいで混ぜる  = 「どこを見るか」
    channel-mixing … 1パッチ内の特徴軸で混ぜる      = 「何を見るか」

LogicMixer はこの2つのMLPを **LutLayerのスタック** に置き換えた論理ゲート版
(griffith_logic/models/mlp_mixer.py)。

**用語(NNが初めてでも読めるように)**

- **token(トークン)** … 画像を7x7のタイルに切ったときの1枚。28x28を7x7で刻むと
  4x4=16枚になるので、このモデルにとって画像は「16個のtoken」でしかない
- **channel(チャネル)** … 1枚のtokenを表すビットの並び。ここでは128ビット。
  「そのタイルが持つ128個の○×質問への答え」だと思えばよい
- **LutLayer** … 2入力の論理ゲートを out_dim 個並べた層。1ゲート=1出力ビット。
  学習中は16種の論理関数(AND/OR/XOR…)の連続的な混合、確定させると本物の回路になる
- **LutStack** … 同じ幅のLutLayerを depth 段だけ積んだもの。MLPの「隠れ層」に当たる
- **転置(transpose)** … 行列の行と列を入れ替える操作。行列の中身は1ビットも変えず、
  「どの軸に沿ってゲートをかけるか」だけを切り替えられるのがMixerの肝

**全体像(このサンプルの実設定の数字を入れたもの)**

```mermaid
flowchart LR
  subgraph IN["入力(実数の世界)"]
    direction LR
    IMG["MNIST 28x28<br/>画素は[0,1]の実数"]
    PT["7x7パッチに刻む<br/>4x4 = 16 tokens<br/>1パッチ = 49画素"]
  end

  subgraph BITS["ビットの世界(ここは全部 0/1 の論理ゲート)"]
    direction LR
    TH["ThermometerEncoder<br/>1画素を4ビットの温度計符号へ<br/>1パッチ = 49x4 = 196ビット"]
    PE["patch_embed LutLayer<br/>196ビット → 128ビット<br/>128ゲート・全16tokenで共有"]
    BLK["LogicMixerBlock x3<br/>各ブロック = token-mix(32ゲート)<br/>+ channel-mix(256ゲート)"]
    HD["head LutLayer<br/>16x128 = 2,048ビット → 160ビット<br/>160ゲート(10クラス x 16本)"]
  end

  subgraph OUT["実数へ戻す"]
    direction LR
    GS["GroupSum k=10<br/>160本を16本ずつ10組に分け<br/>各組の1の本数を数える"]
    LG["logits (10) → argmax<br/>予測ラベル 0〜9"]
  end

  IMG --> PT
  PT -->|"(B, 16, 49) 実数"| TH
  TH -->|"(B, 16, 196) ビット"| PE
  PE -->|"(B, 16, 128) ビット"| BLK
  BLK -->|"(B, 16, 128) ビット"| HD
  HD -->|"(B, 160) ビット"| GS
  GS --> LG

  style IMG fill:#38bdf826,stroke:#38bdf8
  style PT fill:#38bdf826,stroke:#38bdf8
  style TH fill:#10b98126,stroke:#10b981
  style PE fill:#10b98126,stroke:#10b981
  style BLK fill:#10b98126,stroke:#10b981
  style HD fill:#10b98126,stroke:#10b981
  style GS fill:#f9731626,stroke:#f97316
  style LG fill:#f9731626,stroke:#f97316
```

**1ブロックの中身(転置で何が起きるか)**

同じ (16 tokens x 128 channels) のビット表を、**行と列を入れ替えて2回**
LutStackに通すだけ。行列の向きが変わると「1回のLutStack適用で混ざる相手」が
token同士 ⇄ チャネル同士 に切り替わる、というのがMixerの全アイデアである。

```mermaid
flowchart TB
  subgraph BLOCK["LogicMixerBlock(これを3段積む)"]
    direction TB
    X["x = (B, 16 tokens, 128 channels)<br/>行 = token / 列 = チャネル"]
    T1["transpose → (B x 128, 16)<br/>1行が「あるチャネルの16token分」になる"]
    TM["token-mix LutStack<br/>幅16 x depth2 = 32ゲート<br/>128チャネルすべてで重み共有"]
    T2["transpose back → (B, 16, 128)"]
    X1["soft-XOR 残差<br/>a + b - 2ab(0/1なら厳密にXOR)"]
    R1["reshape → (B x 16, 128)<br/>1行が「あるtokenの128チャネル分」になる"]
    CM["channel-mix LutStack<br/>幅128 x depth2 = 256ゲート<br/>16tokenすべてで重み共有"]
    R2["reshape back → (B, 16, 128)"]
    X2["soft-XOR 残差"]
    Y["ブロック出力 (B, 16, 128)<br/>形も中身も 0/1 のまま"]
  end

  X --> T1
  T1 --> TM
  TM --> T2
  T2 --> X1
  X1 --> R1
  R1 --> CM
  CM --> R2
  R2 --> X2
  X2 --> Y
  X -.->|"skip(混ぜる前のx)"| X1
  X1 -.->|"skip"| X2

  style X fill:#38bdf826,stroke:#38bdf8
  style Y fill:#38bdf826,stroke:#38bdf8
  style T1 fill:#eab30826,stroke:#eab308
  style T2 fill:#eab30826,stroke:#eab308
  style R1 fill:#eab30826,stroke:#eab308
  style R2 fill:#eab30826,stroke:#eab308
  style X1 fill:#eab30826,stroke:#eab308
  style X2 fill:#eab30826,stroke:#eab308
  style TM fill:#10b98126,stroke:#10b981
  style CM fill:#10b98126,stroke:#10b981
```

**どう実現しているか**

1. **パッチ埋め込みも論理ゲート**: 7x7パッチを温度計符号化(値の大きさを
   「棒グラフの長さ」としてビット列にする符号化)でビットに直し、全パッチ共有の
   1枚のLutLayerで128ビットへ潰す。実数NNの Conv2d(stride=patch_size) に当たる。
2. **token-mixing は転置で実現**: (B, tokens, channels) を転置してチャネル軸を
   バッチ側へ畳み込み、token軸に沿ったLutStackを1回かける。こうすると
   「全チャネルで重みを共有した token 方向の混合」になり、たった32ゲートで
   16枚のタイル全部を混ぜられる。
3. **残差は加算でなく soft-XOR** ``a + b - 2ab``。{0,1}入力に対して厳密に
   bitwise XOR と一致するので、eval時も中間表現が {0,1} のまま保たれる。
   soft-OR だと段を重ねるたび1へ飽和して情報が潰れるため採用しない
   (XORは ``a XOR b XOR b = a`` で情報を壊さない)。
4. **最後だけ実数**: 160本のビットを10組に分けて1の本数を数える(GroupSum)。
   ここで初めて float が現れ、その値がそのまま logits になる。

このサンプルでは、まず実データを流したアーキテクチャ図で上の流れを目で追い、
次に「1パッチの変化が全パッチへ伝播している」ことを確かめてから、MNISTを学習する。

実行: uv run python examples/15_logic_mixer.py
期待結果: hard accuracy 0.65前後(CPUで2分弱の縮小レシピ。
          d_channels / n_blocks / STEPS を増やせばさらに伸びる)
"""


# === 準備: import と乱数種 ===
# > 使うのは PyTorch と matplotlib、そして本ライブラリの `load_mnist` と
# > `LogicMixer` / `LogicMixerConfig` だけ。**MLP-Mixer**(attention を使わず
# > 2種類の混合MLPを交互に積むだけの画像アーキテクチャ)の論理ゲート版が LogicMixer で、
# > 実装は `griffith_logic/models/mlp_mixer.py` にある。
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from torch.utils.data import DataLoader, Subset

from griffith_logic.datasets import load_mnist
from griffith_logic.models import LogicMixer, LogicMixerConfig

torch.manual_seed(0)

# === ハイパーパラメータ: モデルの形を決める数字 ===
# > このモデルの形はほぼ下の8行で決まる。先に語を定義しておく。
# >
# > - **token(トークン)** … 画像を刻んだパッチ1個。`PATCH=7` なら 28x28 が
# >   4x4 = **16 token** になる
# > - **channel(チャネル)** … 1トークンを表す論理チャネルのビット。`D_CHANNELS=128` なので
# >   1トークン = 128ビット
# > - **(tokens, d_channels) のビット表** … 以降ずっと混ぜ続ける主役。
# >   **行 = token・列 = channel** の 16 x 128 の 0/1 の表である
# >
# > `ENCODE_BITS=4` は **ThermometerEncoder(温度計符号化 = 実数の大きさを「棒グラフの
# > 長さ」としてビット列に直す符号化)** で1画素を何ビットに開くかの指定。
# CPUで2〜3分に収まる縮小レシピ。n_blocks / depth / d_channels を増やすほど強くなる。
PATCH = 7          # 28x28 を 7x7 のパッチに刻む → 4x4 = 16 tokens
ENCODE_BITS = 4    # 1画素あたりの温度計ビット数
D_CHANNELS = 128   # 1パッチを表すビット幅
N_BLOCKS = 3       # (token-mix → channel-mix) ブロックの段数
DEPTH = 2          # 各mixing LutStackの層数
STEPS = 3000
BATCH = 64
N_TRAIN = 8000
N_TEST = 1000

# === データセット: MNISTを28x28のまま読む ===
# > 他の例と違い、画像を縮小も二値化もしない。パッチサイズ 7 で割り切れる必要があるので
# > 28x28 のまま使い(20x20 だと 7 で割り切れない)、値も `[0,1]` の実数のまま渡す。
# > ビットへの変換はモデル内部の ThermometerEncoder が引き受けるため、データ側は何もしない。
train_set, test_set = load_mnist(binarize_threshold=None)


def materialize(dataset, n):
    """先頭n件を (N, 1, 28, 28) のテンソルとラベルにまとめる。"""
    subset = Subset(dataset, range(n))
    return next(iter(DataLoader(subset, batch_size=n)))


train_x, train_y = materialize(train_set, N_TRAIN)
test_x, test_y = materialize(test_set, N_TEST)
print(f'train={tuple(train_x.shape)} test={tuple(test_x.shape)}')

# === モデル定義: LutLayerだけでMixerを組む ===
# > 上の数字を `LogicMixerConfig` に渡すだけで、patch_embed → LogicMixerBlock x3 →
# > head → GroupSum までが組み上がる。部品の名前を先に押さえておく。
# >
# > - **LutStack** … 同じ幅の LutLayer(2入力の論理ゲートを並べた層)を depth 段だけ
# >   積んだもの。MLP-Mixer の「MLP」の位置に入る部品
# > - **token-mixing / channel-mixing** … 前者は token をまたいでビットを混ぜる
# >   (= 「どこを見るか」)、後者は1トークン内の128チャネルを混ぜる(= 「何を見るか」)
# > - **残差接続** … 混ぜる前の値を混ぜた後に足し戻す配線。ここでは加算ではなく soft-XOR
# > - **GroupSum** … 最終ビットをクラスごとに数え上げて logits にする出口。
# >   このモデルで実数が現れるのはここだけ
# >
# > 表示される `model` の構造と、次の節で print されるゲート数の内訳が、この構成の全コストである。
config = LogicMixerConfig(
    image_size=28,
    patch_size=PATCH,
    in_channels=1,
    encode_bits=ENCODE_BITS,
    d_channels=D_CHANNELS,
    n_blocks=N_BLOCKS,
    depth=DEPTH,
    n_inputs=2,
    num_classes=10,
)
model = LogicMixer(config)
print(f'\ntokens={config.d_tokens}(= {config.side}x{config.side} patches), '
      f'd_channels={config.d_channels}, '
      f'1パッチの生ビット数={config.patch_pixels * config.encode_bits}')
print(model)

# === アーキテクチャ: 転置で同じ道具を2回使う ===
# > **なぜ転置するのか。** LutStack は常に「行列の**各行**」にかかる道具である。
# > (tokens, d_channels) のビット表をそのまま渡せば1行 = 1トークンの128チャネルなので
# > **channel-mixing** になり、転置して渡せば1行 = 1チャネルの16トークン分になるので
# > **token-mixing** になる。転置はビットを1つも書き換えない純粋な配線替えで、
# > **同じ部品を向きだけ変えて2回使う**ためだけに入っている。
# > 「行列の向きを変えて同じ道具を2回使う」── これが Mixer のアイデアのすべてである。
# >
# > 以下では実データを1枚だけ流し、その断面をそのまま図にして確かめる。
# --- ブロック断面の採取 ---
# > 学習前のモデルに1枚流して、(1)パッチ埋め込み直後 (2)それを転置した行列
# > (3)token-mixing後 (4)channel-mixing後 の4つのビット表を取り出す。
# ブロック内部と同じ計算をここで手で並べているだけで、model.blocks[0](x) と等価。
model.eval()
with torch.no_grad():
    arch_sample = train_x[0]
    arch_patches = model._patchify(arch_sample.unsqueeze(0))
    arch_coded = model.thermometer(arch_patches)
    arch_x0 = model.patch_embed(arch_coded.reshape(-1, arch_coded.shape[-1]))
    arch_x0 = arch_x0.reshape(1, config.d_tokens, config.d_channels)
    arch_block = model.blocks[0]
    arch_t = arch_x0.transpose(1, 2).reshape(config.d_channels, config.d_tokens)
    arch_mixed = arch_block.token_mix(arch_t)
    arch_mixed = arch_mixed.reshape(1, config.d_channels, config.d_tokens).transpose(1, 2)
    arch_x1 = arch_x0 + arch_mixed - 2.0 * arch_x0 * arch_mixed
    arch_ch = arch_block.channel_mix(arch_x1.reshape(config.d_tokens, config.d_channels))
    arch_ch = arch_ch.reshape(1, config.d_tokens, config.d_channels)
    arch_x2 = arch_x1 + arch_ch - 2.0 * arch_x1 * arch_ch

GATES_EMBED = sum(layer.out_dim for layer in model.patch_embed)  # 1段 or 2段のLutLayer
GATES_TOKEN = config.d_tokens * config.depth
GATES_CHANNEL = config.d_channels * config.depth
GATES_HEAD = model.head_lut.out_dim
print(f'\nゲート数: patch_embed={GATES_EMBED} / token-mix={GATES_TOKEN}(1ブロック) / '
      f'channel-mix={GATES_CHANNEL}(1ブロック) / head={GATES_HEAD} → '
      f'合計 {GATES_EMBED + N_BLOCKS * (GATES_TOKEN + GATES_CHANNEL) + GATES_HEAD} ゲート')

# --- 図1: 1ブロックの断面(実データが流れている) ---
# > 左から順に、入力画像 → パッチ埋め込み後のビット表 → **それを転置した行列** →
# > token-mixing後 → channel-mixing後、の5枚。見るべきは**2枚目と3枚目の関係**で、
# > 同じ 2,048 ビットを縦横入れ替えて眺めているだけ(ゲートは1個も通っていない)。
# > 各パネルの下にはその時点の shape と、そこまでに使ったゲート数を書いてある。
# > 破線は soft-XOR 残差の迂回路で、混ぜる前のビットを丸ごと持ち越す配線を表す。
# > 配色はプロジェクト共通で、**緑=論理(ビット)/ 橙=実数 / 灰=入力・配線**。
# 図中の文字はmatplotlib既定フォントで出せる英字に揃える(日本語は豆腐になる)。
C_GATE = '#2e9e4f'
C_REAL = '#e8850c'
C_WIRE = '#8a8a8a'
ARROW_Y = 0.475
SKIP_STUB_Y = 0.60
SKIP_LEVEL = 0.835
fig = plt.figure(figsize=(19.0, 6.8))
gs = fig.add_gridspec(
    1, 5, width_ratios=[1.15, 1.25, 0.5, 1.25, 1.25],
    left=0.042, right=0.988, bottom=0.27, top=0.68, wspace=1.25,
)
arch_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
arch_axes[0].imshow(arch_sample[0], cmap='gray_r', vmin=0, vmax=1)
for k in range(1, config.side):
    arch_axes[0].axhline(k * PATCH - 0.5, color=C_WIRE, lw=1.4)
    arch_axes[0].axvline(k * PATCH - 0.5, color=C_WIRE, lw=1.4)
for t in range(config.d_tokens):
    r, c = divmod(t, config.side)
    arch_axes[0].text(c * PATCH + PATCH / 2 - 0.5, r * PATCH + PATCH / 2 - 0.5, str(t),
                      color='#3d3d3d', ha='center', va='center', fontsize=10, alpha=0.75)
arch_axes[0].set_xticks([])
arch_axes[0].set_yticks([])
for spec in (
    (1, arch_x0[0], 'channel (128)', 'token (16)'),
    (2, arch_x0[0].T, 'token (16)', 'channel (128)'),
    (3, arch_x1[0], 'channel (128)', 'token (16)'),
    (4, arch_x2[0], 'channel (128)', 'token (16)'),
):
    idx, mat, xlab, ylab = spec
    arch_axes[idx].imshow(mat, cmap='gray_r', aspect='auto', vmin=0, vmax=1)
    arch_axes[idx].set_xlabel(xlab, fontsize=8)
    arch_axes[idx].set_ylabel(ylab, fontsize=8)
    arch_axes[idx].tick_params(labelsize=7)
for ax_, title, note, edge in zip(
    arch_axes,
    ('1. input image\n28x28, real values in [0,1]',
     '2. after patch_embed\nbit table (16 tokens x 128 channels)',
     '3. transposed\nsame bits, axes swapped',
     '4. after token-mixing\n(+ XOR residual, transposed back)',
     '5. after channel-mixing\n= output of one block'),
    (f'shape (1, 1, 28, 28)\n{config.d_tokens} patches of {PATCH}x{PATCH}',
     f'shape (16, 128) bits\n196 in -> {GATES_EMBED} gates (shared)',
     'shape (128, 16) bits\nsame 2048 bits, no gate',
     f'shape (16, 128) bits\n{GATES_TOKEN} gates mix the 16 tokens',
     f'shape (16, 128) bits\n{GATES_CHANNEL} gates mix the 128 channels'),
    (C_WIRE, C_GATE, C_WIRE, C_GATE, C_GATE),
    strict=True,
):
    ax_.set_title(title, fontsize=9.5)
    ax_.text(0.5, -0.17, note, transform=ax_.transAxes, ha='center', va='top',
             fontsize=7.8, color='#3d3d3d')
    for side_ in ax_.spines.values():
        side_.set_edgecolor(edge)
        side_.set_linewidth(1.8)
for i, j, col, label, has_xor in (
    (0, 1, C_WIRE, 'ThermometerEncoder (4 bit/px)\n+ patch_embed LutLayer\n'
                   '196 bits -> 128 bits', False),
    (1, 2, C_WIRE, 'transpose(1, 2)\nrows become channels\n'
                   '(no gate, pure rewiring)', False),
    (2, 3, C_GATE, 'token-mix LutStack\n16 -> 16, depth 2 = 32 gates\n'
                   'shared by all 128 channels', True),
    (3, 4, C_GATE, 'channel-mix LutStack\n128 -> 128, depth 2 = 256 gates\n'
                   'shared by all 16 tokens', True),
):
    pa, pb = arch_axes[i].get_position(), arch_axes[j].get_position()
    xa, xb = pa.x1 + 0.006, pb.x0 - 0.006
    xm = 0.5 * (xa + xb)
    segments = [(xa, xm - 0.012), (xm + 0.012, xb)] if has_xor else [(xa, xb)]
    for x0_, x1_ in segments:
        fig.add_artist(FancyArrowPatch(
            (x0_, ARROW_Y), (x1_, ARROW_Y), transform=fig.transFigure,
            arrowstyle='-|>', mutation_scale=15, color=col, lw=1.7))
    if has_xor:
        fig.text(xm, ARROW_Y, 'XOR', ha='center', va='center', fontsize=7,
                 color=col, bbox=dict(boxstyle='circle,pad=0.30', fc='white',
                                      ec=col, lw=1.5))
    fig.text(xm - 0.008, ARROW_Y - 0.045, label, ha='center', va='top',
             fontsize=7.2, color=col)
for src, i, j in ((1, 2, 3), (3, 3, 4)):
    ps = arch_axes[src].get_position()
    pa, pb = arch_axes[i].get_position(), arch_axes[j].get_position()
    xm = 0.5 * (pa.x1 + pb.x0)
    xup = 0.5 * (ps.x1 + arch_axes[src + 1].get_position().x0)
    for xs_, ys_, xe_, ye_ in (
        (ps.x1, SKIP_STUB_Y, xup, SKIP_STUB_Y),
        (xup, SKIP_STUB_Y, xup, SKIP_LEVEL),
        (xup, SKIP_LEVEL, xm, SKIP_LEVEL),
    ):
        fig.add_artist(Line2D([xs_, xe_], [ys_, ye_], transform=fig.transFigure,
                              color=C_GATE, lw=1.4, linestyle=(0, (4, 3))))
    fig.add_artist(FancyArrowPatch(
        (xm, SKIP_LEVEL), (xm, ARROW_Y + 0.032), transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=12, color=C_GATE, lw=1.4,
        linestyle=(0, (4, 3))))
fig.text(0.042, SKIP_LEVEL + 0.012,
         'dashed = skip connection: soft-XOR residual  a + b - 2ab\n'
         '(the bits before mixing are carried over untouched, so hard eval stays 0/1)',
         ha='left', va='bottom', fontsize=8.4, color=C_GATE)
fig.text(0.515, 0.085,
         f'then: block x{N_BLOCKS}  ->  flatten 16 x 128 = 2048 bits  ->  '
         f'head LutLayer ({GATES_HEAD} gates)  ->  GroupSum k=10  ->  10 logits',
         ha='center', va='bottom', fontsize=8.6, color=C_REAL)
fig.legend(handles=[Patch(facecolor=C_GATE, label='logic (bits, 0/1 only)'),
                    Patch(facecolor=C_REAL, label='real numbers (GroupSum, logits)'),
                    Patch(facecolor=C_WIRE, label='input image / wiring (no gate)')],
           loc='lower center', bbox_to_anchor=(0.515, 0.005), ncol=3, fontsize=8.4,
           frameon=False)
fig.suptitle('LogicMixer, inside one block: transpose -> mix tokens -> transpose back '
             '-> mix channels (real MNIST data flowing through)', fontsize=12.5, y=0.965)
plt.show()

# > 3枚目だけ縦横が入れ替わっていること、そして4枚目・5枚目が2枚目と**同じ向き・同じ形**
# > (16 x 128)に戻っていることを確かめてほしい。ブロックを何段積んでも表の大きさは変わらず、
# > 中身も 0/1 のままである。矢印に添えたゲート数(token-mix 32・channel-mix 256)が
# > 混合1回分のコストで、**16トークン全部を混ぜるのにたった32ゲート**しか要らない。
# --- 図2: 画像をパッチ(token)に刻む ---
# > 同じことを、今度は入力側だけ拡大して見る。左は刻み方(数字が token 番号)、
# > 右がパッチ埋め込み直後の **(tokens, d_channels) ビット表 = 行が token・列が channel**。
# > Mixerにとって画像は「16個のtoken」でしかなく、以降はこの表だけが混ぜられていく。
# 図中の文字はmatplotlib既定フォントで出せる英字に揃える(日本語は豆腐になる)。
sample = train_x[0]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
axes[0].imshow(sample[0], cmap='gray_r', vmin=0, vmax=1)
for k in range(1, config.side):
    axes[0].axhline(k * PATCH - 0.5, color='#e8850c', lw=1.5)
    axes[0].axvline(k * PATCH - 0.5, color='#e8850c', lw=1.5)
for t in range(config.d_tokens):
    r, c = divmod(t, config.side)
    axes[0].text(c * PATCH + PATCH / 2 - 0.5, r * PATCH + PATCH / 2 - 0.5, str(t),
                 color='#2e6fd6', ha='center', va='center', fontsize=13, alpha=0.8)
axes[0].set_title(f'input 28x28 → {config.d_tokens} tokens '
                  f'({PATCH}x{PATCH} patches), label={int(train_y[0])}', fontsize=10)
axes[0].set_axis_off()

model.eval()
with torch.no_grad():
    patches = model._patchify(sample.unsqueeze(0))
    coded = model.thermometer(patches)
    embedded = model.patch_embed(coded.reshape(-1, coded.shape[-1]))
    embedded = embedded.reshape(1, config.d_tokens, config.d_channels)
axes[1].imshow(embedded[0], cmap='gray_r', aspect='auto', vmin=0, vmax=1)
axes[1].set_xlabel('logic channel')
axes[1].set_ylabel('token (patch index)')
axes[1].set_title('bits right after patch embedding\n(tokens x d_channels)',
                  fontsize=10)
fig.tight_layout()
plt.show()

# > 右の表は学習前なので模様に意味は無い。読み取るべきは、**行ごとに立っているビットの
# > 並びが違う**こと ── つまり token ごとに別々の128ビットが割り当てられており、
# > この時点ではまだ token 同士が一切混ざっていないことである。
# --- 図3: token-mixing が情報を全パッチへ運ぶ ---
# > **なぜ attention 無しで大域が見えるのか**を実測で確かめる。1パッチだけを黒く潰した
# > 入力を作り、元の入力との**差分**(明るい = 変化したビット)を2枚並べる。
# > 左はパッチ埋め込み直後で、パッチ埋め込みは token ごとに独立なので変化は潰した token の
# > 1行だけに閉じるはず。右は1ブロック(token-mix → channel-mix)通過後で、token-mixing が
# > token 軸を跨いで配線されているぶん、変化が他の token へ**漏れ出す**はずである。
# > LutLayerの配線は疎(1ゲート2入力)なので1ブロックで届くのは数token分だけだが、
# > **ブロックを重ねるほど画像全体へ広がる**。これが attention を持たないMixerが
# > 大域を見る仕組みで、その1段目の実測がこの図になる。
# 潰すのは「一番インクが多いパッチ」。MNISTは四隅がほぼ空白なので、
# token 0 を消しても何も変わらず実験にならない。
patch_ink = patches[0].sum(dim=1)
target_token = int(patch_ink.argmax())
tr_, tc_ = divmod(target_token, config.side)
perturbed = sample.clone()
perturbed[:, tr_ * PATCH:(tr_ + 1) * PATCH, tc_ * PATCH:(tc_ + 1) * PATCH] = 0.0
with torch.no_grad():
    p_patches = model._patchify(perturbed.unsqueeze(0))
    p_coded = model.thermometer(p_patches)
    p_embed = model.patch_embed(p_coded.reshape(-1, p_coded.shape[-1]))
    p_embed = p_embed.reshape(1, config.d_tokens, config.d_channels)
    after = model.blocks[0](embedded)
    p_after = model.blocks[0](p_embed)

diff_embed = (embedded - p_embed).abs()[0]
diff_after = (after - p_after).abs()[0]
rows_embed = int((diff_embed.sum(dim=1) > 0).sum())
rows_after = int((diff_after.sum(dim=1) > 0).sum())
print(f'\ntoken {target_token}(最もインクの多いパッチ)を潰したときに変化したtoken数: '
      f'埋め込み直後={rows_embed} / {config.d_tokens} → '
      f'1ブロック後={rows_after} / {config.d_tokens}')
assert rows_embed == 1, 'パッチ埋め込みはパッチ間で独立なはず'
assert rows_after > rows_embed, 'token-mixingが情報を他tokenへ運べていない'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, d, title in (
    (axes[0], diff_embed, f'after patch embedding (per-token)\n'
                          f'tokens changed: {rows_embed}/{config.d_tokens}'),
    (axes[1], diff_after, f'after 1 mixer block (leaks to others)\n'
                          f'tokens changed: {rows_after}/{config.d_tokens}'),
):
    ax.imshow(d, cmap='magma', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('logic channel')
    ax.set_ylabel('token')
    ax.set_title(title, fontsize=10)
fig.suptitle(f'bit changes caused by erasing token {target_token} '
             f'(bright = changed)', fontsize=12)
fig.tight_layout()
plt.show()

# > 左は1行だけが光り、右は複数行が光る。潰した1パッチの情報が、たった1ブロックで
# > 他のtokenへ届いたということで、上の print と assert が変化した token 数を数値でも示している。
# === 学習: 普通のPyTorchループ ===
# > 学習則は特別なものを使わない。Adam + cross entropy の**完全に普通の PyTorch** である。
# > 500ステップごとに精度を2通りで測る。
# >
# > - **soft eval**(`train()`)… 各ゲートが16種の論理関数の連続混合のままの状態。勾配は流れるが
# >   まだ回路ではない
# > - **hard eval**(`eval()`)… 各ゲートを argmax で1つの論理関数に確定させた、
# >   そのままハードウェアに焼ける純粋なBoolean回路
# >
# > **hard が本番の数字**で、soft だけ高い状態は「回路として決めきれていない」サインである。
model.train()
opt = torch.optim.Adam(model.parameters(), lr=0.02)
history = []

for step in range(STEPS + 1):
    if step % 500 == 0:
        with torch.no_grad():
            model.train()
            soft_acc = (model(test_x).argmax(-1) == test_y).float().mean().item()
            model.eval()
            hard_acc = (model(test_x).argmax(-1) == test_y).float().mean().item()
        model.train()
        history.append((step, soft_acc, hard_acc))
        print(f'step {step:>5}  soft acc={soft_acc:.4f}  hard acc={hard_acc:.4f}')

    idx = torch.randint(0, N_TRAIN, (BATCH,))
    opt.zero_grad()
    loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
    loss.backward()
    opt.step()

steps, soft_hist, hard_hist = zip(*history, strict=True)
final_soft, final_hard = soft_hist[-1], hard_hist[-1]
print(f'\n最終 soft acc={final_soft:.4f} / hard acc={final_hard:.4f}')
assert final_hard > 0.50, f'hard acc={final_hard:.4f} が低すぎる'

# === 評価: 学習曲線と誤分類 ===
# --- 図4: soft と hard の学習曲線 ---
# > 橙が soft、緑が hard。見どころは値そのものより**2本が離れないこと**である。
# >
# > **なぜ soft-XOR 残差なのか。** 残差を `a + b - 2ab`(0/1 なら厳密に XOR)で取っているのは、
# > XOR が**自己逆元**(`a XOR b XOR b = a`)で、片方の入力を反転すれば出力も必ず反転するから。
# > つまり情報を潰さない。soft-OR のように段を重ねるたび1へ飽和する演算だと深いところで
# > 勾配が消えるが、XOR ならそれが起きず、中間表現も {0,1} のまま保たれる。
# > おかげで **hard が soft から離れにくい**。差が大きく開くようなら、
# > ゲートの混合が中間状態に居座っていることを疑う。
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, soft_hist, 'o-', color='#e8850c', label='soft acc (relaxed)')
ax.plot(steps, hard_hist, 's-', color='#2e9e4f', label='hard acc (discrete circuit)')
ax.set_xlabel('step')
ax.set_ylabel('test accuracy')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc='lower right')
ax.set_title(f'LogicMixer on MNIST — {config.d_tokens} tokens × '
             f'{config.d_channels} ch, {N_BLOCKS} blocks')
fig.tight_layout()
plt.show()

# > 実際、緑(hard)は橙(soft)にわずかに遅れて追随し、最後はほぼ重なる。
# > **離散回路のまま同じ精度が出ている**ということで、これが LogicMixer の主張である。
# > 値が 0.65 前後で頭打ちなのは CPU 2分の縮小レシピだからで、
# > `D_CHANNELS` / `N_BLOCKS` / `STEPS` を増やせばそのまま伸びる。
# --- 図5: 誤分類例 ---
# > 締めくくりに、確定した回路(hard eval)が間違えた10枚を並べる。似た字形どうしの
# > 取り違えが多いはずで、7x7 のタイルに刻んでしまうと局所の情報だけでは判別しづらい
# > ケースが残る。16トークン x 3ブロックという小さな混合では拾いきれなかった
# > 大域的な差であり、ブロックを増やすことが効く余地がここにある。
model.eval()
with torch.no_grad():
    pred = model(test_x).argmax(-1)
wrong = (pred != test_y).nonzero(as_tuple=True)[0][:10]
print(f'誤分類 {int((pred != test_y).sum())} / {N_TEST} 枚(先頭10枚を表示)')

fig, axes = plt.subplots(2, 5, figsize=(9, 4.2))
for ax, i in zip(axes.ravel(), wrong.tolist(), strict=False):
    ax.imshow(test_x[i, 0], cmap='gray_r', vmin=0, vmax=1)
    ax.set_title(f'pred {int(pred[i])} / true {int(test_y[i])}', fontsize=9)
    ax.set_axis_off()
for ax in axes.ravel()[len(wrong):]:
    ax.set_axis_off()
fig.suptitle('misclassified by LogicMixer (hard eval)', fontsize=11)
fig.tight_layout()
plt.show()
