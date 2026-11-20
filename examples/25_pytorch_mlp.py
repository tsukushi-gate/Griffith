"""PyTorchで小さなMLPを組んで手書き数字を読む(入門コース 2/3)。

[前のページ](/tutorials/nn-basics/)で見た「テンソル・自動微分・nn.Module・学習ループ」を、
そのまま**実データ**へ持っていく。題材は手書き数字 MNIST を 20x20 に切り詰めたもので、
入力は 400 個の数(ピクセルの明るさ)、出力は 0〜9 の 10 クラス。

書くものは3つだけで、すべて前ページと同じ道具である。

- `nn.Linear` を3枚重ねた **MLP(多層パーセプトロン)**
- `F.cross_entropy` による損失
- `opt.zero_grad() → backward() → step()` の学習ループ

このページの主題は精度そのものではなく、**「実数の重みはどこにあるのか」**を
目で見て掴むことにある。1層目の重みは入力と同じ 400 次元なので、
20x20 の画像として並べ直すと **「そのニューロンが探している形」**がそのまま絵になる。
次のページ examples/26_mlp_to_dlgn.py では、ここで見た重みが丸ごと
**論理ゲートの真理値表**へ置き換わる。

実行: uv run python examples/25_pytorch_mlp.py
期待結果: test accuracy 0.95 前後(2隠れ層の MLP・CPUで10秒ほど)
"""

# === 準備: import と乱数種 ===
# > 使う道具は[入門1](/tutorials/nn-basics/)とまったく同じで、増えたのはデータの読み込みだけ。
# > `load_mnist20x20` は本ライブラリが同梱する MNIST ローダで、
# > 28x28 の画像から黒い縁を落として **20x20 = 400 画素**にしたものを返す。
# >
# > 400 画素にするのは difflogic 論文の公式レシピに合わせるためで、次ページの
# > 論理ゲート版と**同じ入力**を使うことで比較が公平になる。
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from griffith_logic.datasets import load_mnist20x20

# 重みの初期値もミニバッチの選び方も乱数なので、再現性のために固定する。
torch.manual_seed(0)

# --- ハイパーパラメータ(学習の前に人間が決める数) ---
# > **ハイパーパラメータ**とは、学習では動かず人間が事前に決める設定値のこと
# > (学習で動くのが「パラメータ」= 重み)。ここでは隠れ層の幅・イテレーション数・
# > ミニバッチサイズ・使用枚数の4つを、CPU で10秒に収まる値に絞ってある。
# CPU で10秒ほどに収まる縮小レシピ。HIDDEN や STEPS を増やせば精度は上がる。
HIDDEN = 128      # 隠れ層のニューロン数
STEPS = 3000      # 学習イテレーション数
BATCH = 128       # ミニバッチサイズ
N_TRAIN = 20000   # 訓練に使う枚数(全60000のサブセット)
N_TEST = 2000     # 評価に使う枚数

# === データセット: 手書き数字 MNIST 20x20 ===
# > 1枚の画像は 20x20 の画素からなり、それを1列に伸ばして **400次元のベクトル**として
# > モデルへ入れる。各画素の値は 0(白)〜1(黒)の**実数**である。
# >
# > `binarize_threshold=None` を渡して二値化せず、実数のまま扱っている点に注意。
# > 実数 NN は実数入力をそのまま食べられる ── ここが次ページの論理ゲート版で
# > 「ビットに落とす」必要が出てくる箇所になる。
# load_mnist20x20 は 28x28 の黒い縁を落として 20x20 にした MNIST を返す。
# 400 個のピクセル値(0〜1の実数)が、そのままモデルの入力になる。
# 次ページの論理ゲート版とまったく同じデータを使うので、比較が公平になる。
train_set, test_set = load_mnist20x20(binarize_threshold=None)


def materialize(dataset, n):
    """先頭n件を (N, 400) のテンソルとラベルにまとめる(毎stepの前処理を省く)。"""
    subset = Subset(dataset, range(n))
    x, y = next(iter(DataLoader(subset, batch_size=n)))
    return x.reshape(n, -1), y


train_x, train_y = materialize(train_set, N_TRAIN)
test_x, test_y = materialize(test_set, N_TEST)
print(f'train={tuple(train_x.shape)} test={tuple(test_x.shape)} '
      f'(20x20=400次元入力 / 10クラス)')
print(f'ピクセル値の範囲: {train_x.min():.2f} 〜 {train_x.max():.2f}(実数)')

# === モデル定義: nn.Linear 3枚のMLP ===
# > **MLP(多層パーセプトロン)** とは、`nn.Linear` と活性化関数を交互に積んだだけの
# > 最も素朴な NN のこと。ここでは 400 → 128 → 64 → 10 と幅を絞りながら3枚重ねる。
# >
# > **なぜ幅を絞るのか** ── 前半の層で「画素の集まり」から「線・曲がり」といった
# > 特徴を作り、後半で「その特徴の組み合わせが数字のどれか」を判断する、という
# > 役割分担を素直に書くとこの形になる。最後の10出力はクラスごとのロジット
# > (生のスコア)で、`softmax` を通せば「その数字である確率」になる。
# 400 → 128 → 64 → 10。各 nn.Linear が持っているのは重み行列とバイアスで、
# たとえば1層目は 400x128 の実数 51200 個。これが「学習される量」のすべて。
# 最後の出力10個はクラスごとのスコア(ロジット)で、
# softmax を通すと「その数字である確率」になる。
model = nn.Sequential(
    nn.Linear(400, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f'学習される実数パラメータ: {n_params:,} 個(すべて掛け算に使われる)')

# === 学習: ミニバッチで回す ===
# > ループの中身は入門1と同一で、違いは1点だけ ── 毎回すべてのデータを使わず、
# > `BATCH` 枚だけランダムに取り出して更新する **ミニバッチ学習**にしている。
# >
# > **なぜミニバッチか** ── 20000枚すべてで勾配を計算すると1ステップが重すぎる。
# > 一部だけで計算した勾配は本来の勾配のノイズ入り近似だが、その分ステップ数を
# > 稼げるので、同じ計算時間ならこちらが速く進む。ノイズ自体も局所解から抜ける助けになる。
# 前ページとまったく同じ5行の型。違うのは、データが多いので
# 毎回全部ではなく BATCH 枚だけランダムに取り出して使う(ミニバッチ学習)点だけ。
# F.cross_entropy は内部で softmax を掛けるので、モデル側は素のロジットを返せばよい。
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
history = []  # (step, train loss, test acc)

for step in range(STEPS + 1):
    if step % 250 == 0:
        with torch.no_grad():
            test_acc = (model(test_x).argmax(-1) == test_y).float().mean().item()
            train_loss = F.cross_entropy(model(train_x[:2000]), train_y[:2000]).item()
        history.append((step, train_loss, test_acc))
        print(f'step {step:>5}  loss={train_loss:.4f}  test acc={test_acc:.4f}')

    idx = torch.randint(0, N_TRAIN, (BATCH,))
    opt.zero_grad()
    loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
    loss.backward()
    opt.step()

steps, losses, accs = zip(*history, strict=True)
final_acc = accs[-1]
print(f'\n最終 test accuracy = {final_acc:.4f}')
assert final_acc > 0.88, f'test acc={final_acc:.4f} が低すぎる'

# === 評価: 学習曲線を読む ===
# --- 図1: 損失とテスト正解率 ---
# > 橙が訓練データの損失(左軸)、緑がテスト正解率(右軸)。2つの軸を重ねているので、
# > 「損失が下がるとき正解率が本当に上がっているか」を1枚で確かめられる。
# 損失(橙)が下がり、テスト正解率(緑)が上がる。これが学習が進んでいる証拠。
# 実数の NN では、この線が1本しかない ── 次ページの論理ゲート版では
# 「soft と hard」という2本になり、そこが最大の違いになる。
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, losses, 'o-', color='#e8850c', label='train loss')
ax.set_xlabel('step')
ax.set_ylabel('cross entropy loss')
ax.grid(alpha=0.3)
twin = ax.twinx()
twin.plot(steps, accs, 's-', color='#2e9e4f', label='test accuracy')
twin.set_ylabel('test accuracy')
twin.set_ylim(0, 1.02)
ax.set_title('PyTorch MLP on MNIST 20x20')
ax.legend(loc='center right', fontsize=9)
twin.legend(loc='lower right', fontsize=9)
fig.tight_layout()
plt.show()

# > **読み取り**: 序盤の数百ステップで損失が急落し、正解率が 0.9 を越える。以降は
# > ゆるやかに 0.95 前後へ張り付く。曲線が**1本しかない**ことに注目してほしい ──
# > 実数 NN では「学習中のモデル」と「使うモデル」が同じものなので、測る精度も1種類でよい。
# > 次ページの DLGN では、ここが soft と hard の**2本**に増える。
# === 学習された「実数の重み」を見る ===
# > このページの主題。NN が学習した中身を、数字ではなく**絵**として直接見る。
# --- 図2: 1層目の重みを20x20に並べ直す ---
# > 1層目の重みは1ニューロンにつき 400 個で、入力画像と同じ次元数を持つ。
# > だから 400 個をそのまま 20x20 に並べ直せば、**そのニューロンが画像のどこを見て
# > 反応するか**の地図になる。赤が正の重み(そこが黒いと反応が強まる)、
# > 青が負の重み(そこが黒いと抑制される)。12個のニューロンぶんを並べる。
# 1層目の重みは 1ニューロンにつき 400 個。入力が 20x20 の画像なので、
# 400 個をそのまま 20x20 に並べ直すと「そのニューロンが好む形」が絵になる。
# 赤=正の重み(そこが明るいと反応する)、青=負の重み(そこが明るいと抑制される)。
# **NN が学習しているのはこの色の濃淡**であって、構造(どこと繋がるか)ではない。
w1 = model[0].weight.detach()
vmax = float(w1.abs().max())
fig, axes = plt.subplots(2, 6, figsize=(11, 4))
for j, ax in enumerate(axes.ravel()):
    ax.imshow(w1[j].reshape(20, 20), cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title(f'neuron {j}', fontsize=8)
    ax.set_axis_off()
fig.suptitle('Learned real-valued weights of layer 1 (each 20x20 = 400 numbers)',
             fontsize=11)
fig.tight_layout()
plt.show()

print(f'1層目の重み: 最小 {w1.min():.3f} / 最大 {w1.max():.3f} / '
      f'平均絶対値 {w1.abs().mean():.3f}')
print('どれも中途半端な実数であることに注目(0/1 では表せない)。')

# > **読み取り**: どのニューロンにも、画像の中心付近に赤と青がまだらに散った
# > 「テンプレート」らしき模様が浮かぶ。学習したのは**構造**(どこと繋がるか)ではなく
# > この**色の濃淡=実数値**であり、しかもその値は平均絶対値 0.07 という
# > 中途半端な小数である(直後の出力を参照)。0/1 では表現できないこの実数こそが、推論のたびに
# > 掛け算を必要とする ── つまり乗算器が要る理由そのものである。
# === 誤分類を見る ===
# --- 図3: 間違えた数字 ---
# > 最後にモデルが外した10枚を並べる。「どう間違えるか」は精度の数字より
# > モデルの素性を語る ── 人間でも迷う形で間違えているなら、モデルは形を捉えている。
# >
# > **読み取り(先に書いておく)**: 並ぶのは 4 と 9、7 と 1、3 と 5 のように
# > 人間でも一瞬迷うペアばかりで、0 を 7 と読むような無関係な誤りは出ていない。
# > 2000枚中およそ100枚を外す ── これが「実数の重みを持つ普通の MLP」の到達点であり、
# > 次ページで論理ゲートに置き換えたときの**比較基準**になる。
# 誤分類を並べる。1と7、4と9のような「人間でも紛らわしい」ペアに偏っていれば、
# モデルは形をまともに捉えている。
with torch.no_grad():
    pred = model(test_x).argmax(-1)
wrong = (pred != test_y).nonzero(as_tuple=True)[0][:10]
print(f'誤分類 {int((pred != test_y).sum())} / {N_TEST} 枚(先頭10枚を表示)')

fig, axes = plt.subplots(2, 5, figsize=(9, 4.2))
for ax, i in zip(axes.ravel(), wrong.tolist(), strict=False):
    ax.imshow(test_x[i].reshape(20, 20), cmap='gray_r')
    ax.set_title(f'pred {int(pred[i])} / true {int(test_y[i])}', fontsize=9)
    ax.set_axis_off()
for ax in axes.ravel()[len(wrong):]:
    ax.set_axis_off()
fig.suptitle('misclassified by the real-valued MLP', fontsize=11)
fig.tight_layout()
plt.show()

print('\nここまでが「普通の NN」。次のページで、この同じ問題を')
print('nn.Linear → LutLayer / softmax出力 → GroupSum と置き換えて論理ゲートで解く。')
