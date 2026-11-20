"""BinaryBrain互換API(Binarize / RealToBinary / BinaryToReal / DifferentiableLutBlock)で
MNISTを学習する例。

BinaryBrain(https://github.com/ryuz/BinaryBrain)は、FPGAへLUT網を直接焼くことを
前提に設計された日本発のOSSで、「微分可能LUT(Differentiable LUT)」という考え方を
早くから実装していた。本ライブラリはその中核コンポーネントを PyTorch で再実装し、
`griffith_logic` から素の nn.Module として使えるようにしてある。

**なぜ互換性が要るのか**

1. 検証済みレシピをそのまま持ち込める。BinaryBrainのMNISTノートブックの
   層構成・初期値(BatchNorm gamma=0.3 / beta=0.5)・Unbinarize bias=0.25 まで
   一致させてあるので、精度が出ないときに「実装が違うのか設定が違うのか」を切り分けられる。
2. FPGA側の資産と接続できる。BinaryBrainのLUTはFPGAのLUT6に1対1で対応する。
   同じ構成で学習した重みは、そのままFPGA向けの真理値表として解釈できる。
3. パラメータ化の比較対象になる。本ライブラリのLutLayerは5通りのパラメータ化を持ち、
   BinaryBrain方式は `parameterization='raw'` に相当する(下記参照)。

**LutLayer との関係**

    DifferentiableLutBlock = Unbinarize → LutLayer(parameterization='raw')
                             → BatchNorm1d → Binarize

つまり BinaryBrain のブロックは「LutLayer を挟んだサンドイッチ」であって、別物ではない。
違いは真理値表のパラメータ化にある:

  - difflogic系(既定 `parameterization='softmax'`): 16通りの論理ゲートに対する
    softmax重みを学習し、推論時は argmax で1つのゲートに確定する。
  - BinaryBrain系(`parameterization='raw'`): 2^N 個の真理値表エントリを [0,1] の
    実数として**直接**保持し、毎ステップ clamp して範囲内に押し戻す。
    sigmoid を挟まないので勾配に sigmoid のヤコビアンが乗らない。
    FPGAへ焼くときは各エントリを 0/1 に丸めるだけでよい。

**回路の全体像(バイナリ変調で実数を挟み込む)**

```mermaid
graph LR
  subgraph MOD["入口: 実数 → ビット(バイナリ変調)"]
    direction LR
    X["MNIST 20×20<br/>= 400次元 [0,1] の実数<br/>(B, 400)"]
    R2B["RealToBinary(M=7)<br/>閾値 (f+0.5)/7 を7段<br/>濃さ→1のフレーム数<br/>出力 (B×7, 400) bit"]
    X --> R2B
  end

  subgraph NET["DifferentiableLutBlock × 3(400 → 360 → 60 → 10)"]
    direction LR
    U["Unbinarize<br/>{0,1} → {0.25, 0.75}<br/>(bias=0.25、本家の既定)"]
    LL["LutLayer(raw) 6入力LUT<br/>1ゲート = 2⁶ = 64エントリ<br/>合計 430ゲート/27,520 param"]
    BN["BatchNorm1d<br/>gamma=0.3, beta=0.5<br/>(実数の統計量)"]
    BIN["Binarize(STE)<br/>閾値0.5で 0/1 へ<br/>逆伝播は HardTanh で素通し"]
    U --> LL --> BN --> BIN
  end

  subgraph OUT["出口: ビット → 実数"]
    direction LR
    B2R["BinaryToReal(M=7)<br/>7フレームを平均<br/>出力 (B, 10) 実数"]
    ARG["argmax<br/>予測ラベル 0〜9"]
    B2R --> ARG
  end

  R2B --> U
  BIN -->|"3ブロック繰り返し<br/>(400→360, 360→60, 60→10)"| B2R

  subgraph LEG["凡例"]
    direction LR
    LGL["緑 = 論理(0/1 のビット。LUT本体)"]
    LGR["橙 = 実数(BatchNorm・平均。ここは FPGA に焼けない)"]
    LGB["青 = 学習パラメータを持たない変換"]
  end

  style X fill:#f9731626,stroke:#f97316
  style R2B fill:#38bdf826,stroke:#38bdf8
  style U fill:#38bdf826,stroke:#38bdf8
  style LL fill:#10b98126,stroke:#10b981
  style BN fill:#f9731626,stroke:#f97316
  style BIN fill:#10b98126,stroke:#10b981
  style B2R fill:#f9731626,stroke:#f97316
  style ARG fill:#f9731626,stroke:#f97316
  style LGL fill:#10b98126,stroke:#10b981
  style LGR fill:#f9731626,stroke:#f97316
  style LGB fill:#38bdf826,stroke:#38bdf8
```

**具体例が図としてどこに出るか**: このページ1枚目の図
「RealToBinary: one real image -> 7 binary frames -> mean」が、上図の R2B と B2R の実物である。
濃い画素ほど多くのフレームで1になり、7枚を平均すると元の濃淡がほぼ復元される
── これが「実数を、本数という形でビットの世界へ持ち込む」仕掛けの正体である。

**どう実現しているか**

- **濃さを「1の本数」に変える**。RealToBinary は1枚の実数画像を7枚の二値画像へ複製する。
  閾値が等間隔なので、値 0.7 の画素は7枚中およそ5枚で1になる。
  これで確率的な情報が、回路が扱えるビット列に化ける
- **層構成には設計ルールがある**。6入力LUTなので「次層のノード数×6 ≧ 前層のノード数」を
  満たさないと、誰にも読まれない未接続ノードが出る。400 → 360 → 60 → 10 はその最小構成
- **BatchNorm がビットの偏りを直す**。LUT の出力をそのまま次段へ入れると 0 か 1 に偏るので、
  実数として正規化してから Binarize で切り直す。**この BatchNorm は実数演算なので、
  焼くときは閾値としてたたみ込む**必要がある(橙のブロック)
- **Binarize は前向きが階段関数、逆向きが素通し**(STE)。だから微分できないはずの
  二値化を挟んだまま勾配が最後まで届く
- **真理値表を直接持つ(raw)**。difflogic 系が「16種の論理関数への重み」を学ぶのに対し、
  BinaryBrain 方式は 2⁶=64 個のエントリを [0,1] の実数のまま持ち、毎ステップ clamp する。
  焼くときは各エントリを丸めるだけでよい

実行: uv run python examples/22_binarybrain.py
期待結果: eval accuracy 0.80前後(CPUで約2分)

なお本サンプルは短時間レシピなので、真理値表はまだ 0/1 に決めきれていない。
「学習は動くが、そのまま焼くには足りない」ところまでを、
離散化ギャップとして最後に定量的に見る(図2・図3)。
"""

# === 準備: import と乱数種 ===
# > 使うのは BinaryBrain 互換の3部品 ── `RealToBinary`(実数をビット列へ)、
# > `DifferentiableLutBlock`(LUT のサンドイッチ1段)、`BinaryToReal`(ビット列を実数へ)──
# > と、20×20 に縮めた MNIST だけ。LUT の接続(どの入力線を見るか)はランダムに決まるので、
# > 再現性のために乱数種を固定する。
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from griffith_logic import BinaryToReal, DifferentiableLutBlock, RealToBinary
from griffith_logic.datasets import load_mnist20x20

# LUTの接続(どの入力線を見るか)はランダムに決まるので、再現性のため固定する。
torch.manual_seed(0)

# --- ハイパーパラメータ ---
# > 記号の意味を先に決めておく。**`N_INPUTS`(= LUT の入力数)** は1つの真理値表が見る線の本数で、
# > 6 は FPGA の LUT6 に1対1で対応する本命の値。エントリ数は 2⁶ = 64 になる。
# > **`FRAME_M`(= バイナリ変調のフレーム数、本文では M)** は、1枚の実数画像を何枚の
# > 二値画像に分解するかで、大きいほど濃淡を細かく表せるが、そのぶん計算量も M 倍になる。
# BinaryBrainのMnistDifferentiableLutSimple.ipynbの設定を、CPUで2分に収まるよう縮めた版。
N_INPUTS = 6      # LUTの入力数。FPGAのLUT6に対応する本命の値
FRAME_M = 7       # バイナリ変調のフレーム数(本家は15。減らすと速いが量子化が粗くなる)
STEPS = 2500      # 学習イテレーション数
BATCH = 64        # ミニバッチサイズ(実効バッチは BATCH * FRAME_M = 448 行になる)
LR = 0.001        # 本家と同じ Adam(lr=0.001)
N_TRAIN = 20000   # 訓練に使う枚数
N_TEST = 2000     # 最終評価に使う枚数
N_CURVE = 1000    # 学習中の曲線用(毎回2000枚を2通り評価すると時間を食うため間引く)

# === データ: 20×20 の MNIST を実数のまま受け取る ===
# > ここが他の例と違うところ。多くの論理ゲート NN の例では入口で画像を二値化してしまうが、
# > BinaryBrain 流では**実数の濃淡のまま**受け取り、モデルの中の `RealToBinary` が
# > 「濃さ → 1 になるフレームの本数」へ変換する。だから `binarize_threshold=None` で読む。
# 20x20=400ビットのMNIST。値は[0,1]の連続値のまま受け取る
# (連続値であることが RealToBinary のバイナリ変調に意味を与える)。
train_set, test_set = load_mnist20x20(binarize_threshold=None)


def materialize(dataset, n):
    """先頭n件を (N, 400) のテンソルとラベルにまとめる。"""
    subset = Subset(dataset, range(n))
    x, y = next(iter(DataLoader(subset, batch_size=n)))
    return x.reshape(n, -1), y


train_x, train_y = materialize(train_set, N_TRAIN)
test_x, test_y = materialize(test_set, N_TEST)
print(f'train={tuple(train_x.shape)} test={tuple(test_x.shape)} '
      f'(20x20=400ビット入力 / 10クラス)')


# === モデル: DifferentiableLutBlock を3段重ねる ===
# > 1段の中身は `Unbinarize` → `LutLayer(raw)` → `BatchNorm1d` → `Binarize` の
# > サンドイッチである。**なぜ BatchNorm を挟むのか**: LUT の出力をそのまま次段へ入れると
# > 0 か 1 に偏ってしまい、次の LUT がどの入力を見ても同じ答えを返す状態に陥る。
# > いったん実数として正規化し直してから `Binarize` で切り直すことで、
# > 各段のビットの偏りをならしている(この BatchNorm は実数演算なので、FPGA へ焼くときは
# > 閾値としてたたみ込む必要がある)。
# >
# > **`Binarize` は前向きが階段関数、逆向きが素通し**(**STE** = straight-through estimator、
# > 微分できない関数の勾配をそのまま通す近似)。だから二値化を挟んだまま勾配が最後まで届く。
# BinaryBrainのネットワーク設計ルール:
#   「N入力LUTなので、次層のノード数 x N が前層のノード数以上でないと、
#     前層に誰からも読まれない未接続ノードが生じる」
# 400 → 360(=6*6*10) → 60(=6*10) → 10 は、このルールを満たす最小構成
# (360*6=2160 >= 400、60*6=360、10*6=60)。本家の6層の後半3層に相当する。
LAYER_SIZES = [6 * 6 * 10, 6 * 10, 10]


class BinaryBrainNet(nn.Module):
    """BinaryBrainのMnistDifferentiableLutSimpleと同じ骨格の分類器。

    Flatten → RealToBinary → DifferentiableLutBlock×3 → BinaryToReal

    最終層のノード数がそのままクラス数(10)なので、difflogic系で使う GroupSum は要らない。
    その代わり BinaryToReal が「10クラス分のビットをフレーム方向に平均して
    0〜1のスコアにする」役目を担う。
    """

    def __init__(self):
        super().__init__()
        # 実数[0,1] → M枚の二値フレーム。閾値は (f+0.5)/M で等間隔に置かれる。
        # 濃いピクセルほど多くのフレームで1になる = 濃さがビットの「本数」に化ける。
        # パラメータを持たない前処理なので、勾配は流れない。
        self.real_to_binary = RealToBinary(FRAME_M)

        self.blocks = nn.ModuleList()
        prev = 400
        for out_dim in LAYER_SIZES:
            # DifferentiableLutBlock の中身(前掲):
            #   Unbinarize({0,1}→{0.25,0.75}) → LutLayer(raw) → BatchNorm → Binarize
            # Binarize の backward は HardTanh STE。前向きは階段関数でも
            # 逆向きは [0,1] の範囲内で勾配を素通しするので学習が成立する。
            self.blocks.append(DifferentiableLutBlock(
                in_dim=prev, out_dim=out_dim, n_inputs=N_INPUTS,
                unbinarize_bias=0.25,   # 本家 DifferentiableLutN.h の既定値
            ))
            prev = out_dim

        # M枚のフレームを平均して実数スコアに戻す。ここは普通の mean なので
        # PyTorchのautogradがそのまま勾配(1/M)を流してくれる。
        self.binary_to_real = BinaryToReal(FRAME_M)

    def forward(self, x):
        x = self.real_to_binary(x)      # (B, 400) → (B*M, 400)
        for block in self.blocks:
            x = block(x)                # 各ブロックの出力は {0,1} のビット
        return self.binary_to_real(x)   # (B*M, 10) → (B, 10)

    def clamp_weights(self):
        """全ブロックの真理値表を [0,1] に押し戻す(本家の m_flagClamp と等価)。

        raw パラメータ化は重みを確率としてそのまま持つので、
        optimizer.step() のあとに毎回これを呼ばないと範囲外へ逃げていく。
        """
        for block in self.blocks:
            block.clamp_weights()


model = BinaryBrainNet()
print(model)
n_luts = sum(LAYER_SIZES)
print(f'LUT{N_INPUTS} を {n_luts} 個 / 真理値表エントリ総数 '
      f'{n_luts * 2 ** N_INPUTS:,}(= LUT数 x 2^{N_INPUTS})')

# 学習前の真理値表を控えておく(あとでヒストグラムを比べる)。
init_weights = torch.cat([b.lut.weights.detach().flatten() for b in model.blocks])


# === 学習: clamp を1行足すだけの普通の PyTorch ===
# > 中身は完全に普通の PyTorch で、BinaryBrain 固有なのは
# > 「`optimizer.step()` のあとに `clamp_weights()` を呼ぶ」の1行だけ。
# > **なぜ clamp が要るのか**: `raw` パラメータ化では真理値表のエントリを
# > 「そのエントリが1を出す確率」としてそのまま持つので、勾配で [0,1] の外へ逃げると
# > 意味を失う。sigmoid で押し込める代わりに毎ステップ切り戻す、というのが本家の流儀で、
# > 勾配に sigmoid のヤコビアン(傾きの掛け算)が乗らないぶん学習が素直になる。
# >
# > 学習中は毎回2つの精度を測る。**eval**(連続の真理値表のまま = BinaryBrain 本来の推論経路)と、
# > **rounded**(エントリを 0/1 へ丸めた複製 = 素朴に FPGA へ焼いた場合)。
# > この2本の差が**離散化ギャップ**で、このページの主題そのものである。
opt = torch.optim.Adam(model.parameters(), lr=LR)
history = []  # (step, eval acc, 真理値表を丸めたあとの acc)


@torch.no_grad()
def accuracy(net, x, y):
    """推論モードで精度を測る。

    注意: BatchNormを train モードで走らせると running統計が更新されてしまうので、
    評価では必ず eval() にする(テストデータで統計を汚さないため)。
    raw パラメータ化は推論時も真理値表を**連続値のまま**使う。これは本家と同じ挙動で、
    difflogic系が eval 時に argmax で1ゲートへ確定するのとは対照的。
    """
    was_training = net.training
    net.eval()
    acc = (net(x).argmax(-1) == y).float().mean().item()
    net.train(mode=was_training)
    return acc


@torch.no_grad()
def rounded_copy(net):
    """真理値表を 0/1 に丸めた複製を返す(FPGAへ焼いた回路の素朴な模擬)。"""
    burned = copy.deepcopy(net)
    for block in burned.blocks:
        block.lut.weights.round_()
    return burned


for step in range(STEPS + 1):
    if step % 500 == 0:
        eval_acc = accuracy(model, test_x[:N_CURVE], test_y[:N_CURVE])
        # 「いま焼いたらどうなるか」を毎回一緒に測る。この2本の差が離散化ギャップ。
        hard_acc = accuracy(rounded_copy(model), test_x[:N_CURVE], test_y[:N_CURVE])
        history.append((step, eval_acc, hard_acc))
        print(f'step {step:>5}  eval acc={eval_acc:.4f}  '
              f'rounded-LUT acc={hard_acc:.4f}')

    idx = torch.randint(0, N_TRAIN, (BATCH,))
    opt.zero_grad()
    loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
    loss.backward()
    opt.step()
    # ここが BinaryBrain 流。真理値表エントリを毎ステップ [0,1] に閉じ込める。
    model.clamp_weights()

steps, eval_hist, hard_hist = zip(*history, strict=True)
# 最終値だけはテスト全件(N_TEST枚)で測り直す。
final_eval = accuracy(model, test_x, test_y)
final_hard = accuracy(rounded_copy(model), test_x, test_y)

# === 離散化ギャップ: 丸めるとなぜ壊れるのか ===
# > 丸めると精度が大きく落ちる。原因の候補は2つある。**(a)** 真理値表がまだ 0/1 に
# > 決めきれておらず、丸めが情報を壊している。**(b)** LUT 出力の分布が激変して、
# > BatchNorm が持っている移動平均の統計が合わなくなっている。
# > (b) なら統計を測り直せば戻るはずなので、訓練データを流して再較正し、
# > **どちらが主因かを実験で切り分ける**。原因の切り分けまでやるのがこの節の主題である。
# 丸めると精度が大きく落ちる。原因は2つ考えられる:
#   (a) 真理値表がまだ 0/1 に決めきれておらず、丸めが情報を壊している
#   (b) LUT出力の分布が激変し、BatchNormの running統計が合わなくなっている
# (b) なら統計を測り直せば戻るはずなので、訓練データを流して再較正してみる。
recal = rounded_copy(model)
recal.train()
with torch.no_grad():
    for _ in range(20):
        idx = torch.randint(0, N_TRAIN, (512,))
        recal(train_x[idx])   # forwardするだけでBatchNormの統計が更新される
recal_acc = accuracy(recal, test_x, test_y)

print(f'\n最終 eval accuracy={final_eval:.4f}(連続の真理値表 / BinaryBrainの推論経路)')
print(f'真理値表を0/1へ丸めた後      ={final_hard:.4f}')
print(f'  + BatchNorm統計を再較正    ={recal_acc:.4f}')
assert final_eval > 0.65, f'eval acc={final_eval:.4f} が低すぎる'

# === 可視化: 変調・学習曲線・真理値表の分布 ===
# --- 図1: バイナリ変調(RealToBinary)が何をしているか ---
# > **この図で何を見るか**: 左端が元の実数画像、続く7枚が閾値 `(f+0.5)/M` をずらして
# > 作った二値フレーム、右端がそれを平均して戻したもの。**濃い画素ほど多くのフレームで1になる**
# > という対応を、1枚ずつ目で追う図である。
# >
# > **バイナリ変調**とは、値の大きさを「1になるフレームの本数」で表す符号化のこと
# > (パルス密度変調と同じ発想)。閾値が等間隔なので、値 0.7 の画素は7枚中およそ5枚で1になる。
# > これで連続値の情報が、回路が扱えるビット列に化ける。
sample = test_x[:1]                       # (1, 400)
frames = RealToBinary(FRAME_M)(sample)    # (M, 400)
recon = BinaryToReal(FRAME_M)(frames)     # (1, 400) 平均で戻す

fig, axes = plt.subplots(1, FRAME_M + 2, figsize=(1.35 * (FRAME_M + 2), 2.1))
axes[0].imshow(sample.reshape(20, 20), cmap='gray_r', vmin=0, vmax=1)
axes[0].set_title('input\n(real)', fontsize=8)
for f in range(FRAME_M):
    axes[f + 1].imshow(frames[f].reshape(20, 20), cmap='gray_r', vmin=0, vmax=1)
    axes[f + 1].set_title(f'frame {f}\nthr={(f + 0.5) / FRAME_M:.2f}', fontsize=8)
axes[-1].imshow(recon.reshape(20, 20), cmap='gray_r', vmin=0, vmax=1)
axes[-1].set_title('mean\n(BinaryToReal)', fontsize=8)
for ax in axes:
    ax.set_axis_off()
fig.suptitle(f'RealToBinary: one real image -> {FRAME_M} binary frames -> mean',
             fontsize=11)
fig.tight_layout()
plt.show()

# > **読み取り**: 右端(7枚の平均)が左端(元画像)とほぼ同じに見えることが要点で、
# > **ビットの本数だけで濃淡が往復できている**ということ。逆に言えば、M を小さくするほど
# > この復元は粗くなる(M=7 なら濃さは 8 段階しか区別できない)。
# --- 図2: 学習曲線と離散化ギャップ ---
# > **この図で何を見るか**: 緑が BinaryBrain 本来の推論経路(連続の真理値表)、
# > 橙が「いま丸めて焼いたら」の精度、赤い破線が丸めたうえで BatchNorm を再較正した場合。
# > **緑と橙の隙間の広さ**と、**赤い破線がその隙間をどれだけ埋めるか**の2点を見る。
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, eval_hist, 's-', color='#2e9e4f',
        label='eval (continuous LUT — BinaryBrain inference)')
ax.plot(steps, hard_hist, 'o-', color='#e8850c',
        label='LUT rounded to 0/1 (naive FPGA burn)')
ax.axhline(recal_acc, ls='--', color='#c0392b',
           label=f'rounded + BatchNorm recalibration = {recal_acc:.3f}')
ax.set_xlabel('step')
ax.set_ylabel('test accuracy')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc='lower right')
ax.set_title(f'BinaryBrain-style LUT{N_INPUTS} net on MNIST 20x20')
fig.tight_layout()
plt.show()

# > **読み取り**: 緑と橙の隙間はこの縮小レシピでは大きく開いたままで、赤い破線
# > (BatchNorm 再較正)はその一部しか埋めない。つまり離散化ギャップの主因は
# > **統計のズレ(b)ではなく、真理値表がまだ 0/1 に決めきれていないこと(a)**だと結論できる。
# > その裏付けを次の図で直接見る。
# --- 図3: 真理値表エントリの分布 ---
# > **この図で何を見るか**(最後の図なので読み取り方も先に置く): 横軸は LUT エントリの値
# > (= そのエントリが1を出す確率)、縦軸は個数(対数)。灰が学習前、青が学習後。
# > **焼ける回路とは、この分布が 0 と 1 の両端へ張り付き、`round()` しても情報が失われない状態**
# > を指す。ところがここでは、分布は初期値 Normal(0.5, 0.01) から少し広がっただけで、
# > 依然として 0.5 付近に密集したままになる ── だから丸めると壊れる(図2の橙)。
# > BinaryBrain の推論が連続重みのままで成立するのもこのためで、
# > 「**学習は動くが、まだ焼ける状態ではない**」というのがこのページの正しい読み方である。
# > 両端へ寄せるには本家並みの長い学習(あるいは二値化を促す正則化)が要る。
final_weights = torch.cat([b.lut.weights.detach().flatten() for b in model.blocks])
fig, ax = plt.subplots(figsize=(7, 4))
bins = torch.linspace(0, 1, 41)
ax.hist(init_weights.numpy(), bins=bins.numpy(), alpha=0.55,
        color='#8899aa', label='init: Normal(0.5, 0.01)')
ax.hist(final_weights.numpy(), bins=bins.numpy(), alpha=0.55,
        color='#2b6cb0', label=f'after training ({STEPS} steps)')
ax.set_xlabel('LUT entry value (= P[output = 1])')
ax.set_ylabel('count')
ax.set_yscale('log')
ax.grid(alpha=0.3)
ax.legend()
ax.set_title('LUT truth-table entries — still clustered at 0.5 (not yet burnable)')
fig.tight_layout()
plt.show()

edge = ((final_weights < 0.05) | (final_weights > 0.95)).float().mean().item()
print(f'エントリ分布: std={final_weights.std():.3f} '
      f'range=[{final_weights.min():.3f}, {final_weights.max():.3f}]')
print(f'0.05未満 or 0.95超のエントリ割合={edge:.3f}(1.0に近いほど焼ける状態)')
