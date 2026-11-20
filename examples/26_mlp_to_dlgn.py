"""MLPをDLGNへ置き換える ── 同じ問題を論理ゲートで解く(入門コース 3/3)。

[前のページ](/tutorials/pytorch-mlp/)で作った MNIST 20x20 の MLP を、
**同じデータ・同じ学習ループのまま、モデルの中身だけ**論理ゲートへ置き換える。
書き換えるのは実質2行しかない。

```python
# PyTorch(実数のMLP)              # DLGN(論理ゲート)
nn.Linear(400, 128), nn.ReLU()  →  LutLayer(400, 2000, n_inputs=2)
nn.Linear(64, 10)               →  GroupSum(k=10, tau=10.0)
```

置き換えると何が変わるのか、を実測で並べるのがこのページの目的である。

- **学習の書き方は変わらない** ── `cross_entropy` も `backward()` も `Adam` もそのまま
- **学習される中身が変わる** ── 実数の重み(掛け算に使う数)ではなく、
  各ゲートの**真理値表**(AND なのか XOR なのか、という論理関数の選択)
- **精度が2本になる** ── 学習中の連続緩和(soft)と、
  argmax で1つの論理関数に確定した回路(hard)の2つを必ず併記する。
  この差が **離散化ギャップ**で、DLGN 特有の唯一にして最大の注意点

最後に「MLP を DLGN に移すときの置き換えチェックリスト」で締める。

実行: uv run python examples/26_mlp_to_dlgn.py
期待結果: MLP test acc 0.95 前後 / DLGN hard acc 0.87 前後(CPUで1分ほど)
"""

# === 準備: 実数側と論理側、両方の道具を持ってくる ===
# > 前ページまでの PyTorch 一式に、本ライブラリから3つを足す。
# >
# > - `LutLayer` … **LUT(ルックアップテーブル=真理値表)を学習する層**。
# >   `nn.Linear + 活性化` の位置にそのまま入る
# > - `GroupSum` … 論理の世界から分類スコアへの**出口**。出力ビットをクラスごとに数える
# > - `gate_labels` … 学習後の各ゲートが16種の論理関数のどれに確定したかを名前で返す
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from griffith_logic import GroupSum, LutLayer
from griffith_logic.analysis.circuit_viz import gate_labels
from griffith_logic.datasets import load_mnist20x20

# 2つのモデルを同じ条件で比べたいので、乱数は必ず固定する。
torch.manual_seed(0)

# --- ハイパーパラメータ ---
# CPU で1分ほどに収まる縮小レシピ。K(ゲート数)や STEPS を増やすと DLGN 側が伸びる。
HIDDEN = 128      # MLP の隠れ層の幅
K = 2000          # DLGN の1層あたりのゲート数
STEPS = 2500      # 両モデル共通の学習イテレーション数
BATCH = 128
N_TRAIN = 20000
N_TEST = 2000

# === データセット: 前ページとまったく同じ入力 ===
# > **比較の公平さがこのページの生命線**なので、データは前ページと1バイトも変えない。
# > 同じ MNIST 20x20、同じ枚数、同じ乱数種。違うのはモデルの中身だけである。
# --- データ(前ページとまったく同じ) ---
train_set, test_set = load_mnist20x20(binarize_threshold=None)


def materialize(dataset, n):
    """先頭n件を (N, 400) のテンソルとラベルにまとめる。"""
    subset = Subset(dataset, range(n))
    x, y = next(iter(DataLoader(subset, batch_size=n)))
    return x.reshape(n, -1), y


train_x, train_y = materialize(train_set, N_TRAIN)
test_x, test_y = materialize(test_set, N_TEST)
# --- hard 評価用に入力をビットへ落とす ---
# > DLGN だけに必要な前処理が1つある。**hard 評価**(= 焼いた回路そのものとしての評価)では、
# > 入力も 0/1 のビットでなければならない。実際の FPGA に入るのは信号線の High/Low であって
# > 「0.37」のような実数ではないからである。`round()` で二値化した `test_bits` を作る。
# DLGN の hard 評価では、入力も本物のビットでなければならない。
# 「実際に FPGA へ焼いた回路」に入るのは 0/1 の信号線であって実数ではないため。
test_bits = test_x.round()
print(f'train={tuple(train_x.shape)} test={tuple(test_x.shape)}')

# === モデル定義: 同じ問題を解く2つのモデル ===
# > 比較対象を2つ並べる。**モデルA** は前ページと同一の実数 MLP(基準線)。
# --- モデルA: 前ページと同じ実数MLP ---
mlp = nn.Sequential(
    nn.Linear(400, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# --- モデルB: 同じ問題を解くDLGN ---
# > **モデルB** が本題。置き換えの規則は2つだけである。
# >
# > - `nn.Linear + 活性化` → `LutLayer(in, out, n_inputs=2)`。
# >   **活性化関数は付けない** ── 非線形性はゲート(AND / XOR …)そのものが持っているため
# > - 出力の `nn.Linear(h, 10)` → `GroupSum(k=10, tau=10.0)`。
# >   最終層のビットを10クラス×200本に分けて「1が立った本数」を数えるだけの層で、
# >   **重みを1つも持たない**(ハードウェア側では加算器1つ)
# >
# > **なぜ幅が 128 から 2000 へ増えるのか** ── `n_inputs=2` の1ゲートは前層の
# > **2本しか見ない**。400入力すべてを見る `nn.Linear` のニューロンとは情報量が桁違いなので、
# > 本数で補う必要がある。「幅を1〜2桁大きく取る」がDLGNの基本作法になる。
# nn.Linear + ReLU の対が LutLayer 1枚に、最後の nn.Linear(→softmax) が GroupSum になる。
# LutLayer(400, 2000, n_inputs=2) は「入力400本のうち2本だけを見るゲートを2000個並べる」層。
# 活性化関数は要らない ── 非線形性はゲート(AND/OR/XOR…)そのものが持っている。
# GroupSum(k=10, tau=10.0) は最終2000ゲートを10クラス×200ゲートに分けて数え上げるだけで、
# 重みを1つも持たない(回路側では加算器1つ)。
dlgn = nn.Sequential(
    LutLayer(400, K, n_inputs=2),
    LutLayer(K, K, n_inputs=2),
    GroupSum(k=10, tau=10.0),
)
print(dlgn)

mlp_params = sum(p.numel() for p in mlp.parameters())
dlgn_params = sum(p.numel() for p in dlgn.parameters())
print(f'\nMLP  の学習パラメータ: {mlp_params:,} 個 = 掛け算に使う実数の重み')
print(f'DLGN の学習パラメータ: {dlgn_params:,} 個 = 各ゲートが16種の論理関数の'
      f'どれになるかの点数(2層 x {K}ゲート x 16)')
print('同じ「パラメータ」でも意味が違う: 前者は掛ける数、後者は真理値表の選択。')


def evaluate(model, x, y):
    """テスト正解率を返す(モデルのモードは呼び出し側の設定に従う)。"""
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


# === 学習: 2つのモデルを同じループで回す ===
# > 同じ `for` の中で A と B を並べて更新する。**学習ループのコードは1文字も変わらない** ──
# > `cross_entropy` も `backward()` も `Adam` もそのまま使える。これが
# > 「PyTorch の資産をそのまま持ち込める」という本ライブラリの主張の実演である。
# >
# > 違いは2点だけ。(1) 学習率を DLGN 側は10倍にする(理由は下のコメント)。
# > (2) DLGN は **soft と hard の2つの精度**を測る。soft は `train()` モードでの
# > 連続緩和のまま、hard は `eval()` モードで各ゲートを1つの論理関数に確定させた
# > 「焼ける回路そのもの」の精度である。
# 違うのは lr だけ。論理側の重みは実数重みより1〜2桁大きい学習率を好むので、
# MLP が 1e-3、DLGN は 0.01 にしてある(これは経験則で、詳しくは学習ガイドを参照)。
# DLGN は各stepで soft(train mode)と hard(eval mode)の両方を測る。
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
opt_dlgn = torch.optim.Adam(dlgn.parameters(), lr=0.01)
history = []  # (step, mlp acc, dlgn soft acc, dlgn hard acc)

for step in range(STEPS + 1):
    if step % 250 == 0:
        mlp.eval()
        mlp_acc = evaluate(mlp, test_x, test_y)
        mlp.train()
        dlgn.train()
        soft_acc = evaluate(dlgn, test_x, test_y)
        dlgn.eval()
        hard_acc = evaluate(dlgn, test_bits, test_y)
        dlgn.train()
        history.append((step, mlp_acc, soft_acc, hard_acc))
        print(f'step {step:>5}  MLP={mlp_acc:.4f}  '
              f'DLGN soft={soft_acc:.4f}  DLGN hard={hard_acc:.4f}')

    idx = torch.randint(0, N_TRAIN, (BATCH,))
    for model, opt in ((mlp, opt_mlp), (dlgn, opt_dlgn)):
        opt.zero_grad()
        F.cross_entropy(model(train_x[idx]), train_y[idx]).backward()
        opt.step()

steps, mlp_hist, soft_hist, hard_hist = zip(*history, strict=True)
final_mlp, final_soft, final_hard = mlp_hist[-1], soft_hist[-1], hard_hist[-1]
gap = final_soft - final_hard
print(f'\n最終: MLP={final_mlp:.4f} / DLGN soft={final_soft:.4f} / '
      f'DLGN hard={final_hard:.4f}')
print(f'離散化ギャップ(soft - hard)= {gap:+.4f}')
assert final_hard > 0.70, f'DLGN hard acc={final_hard:.4f} が低すぎる'

# === 可視化: MLPとDLGNを並べて読む ===
# --- 図1: 同じ問題・同じ学習ループでの学習曲線 ---
# > 3本の線を1枚に重ねる。青=実数 MLP、橙=DLGN の soft、緑=DLGN の hard。
# > **見るべきは緑**で、これが「乗算器を1つも持たない回路」としての本当の成績である。
# 青が実数MLP、橙がDLGNの soft(学習中の連続緩和)、緑がDLGNの hard(焼ける回路そのもの)。
# 見るべきは緑 ── これが「乗算器を1つも使わない回路」としての本当の成績である。
# 橙と緑の差が離散化ギャップで、開きすぎているときは回路として決めきれていないサイン。
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(steps, mlp_hist, 'o-', color='#1f6fd0', label='PyTorch MLP (real-valued)')
ax.plot(steps, soft_hist, 'o-', color='#e8850c', label='DLGN soft (relaxed / train)')
ax.plot(steps, hard_hist, 's-', color='#2e9e4f', label='DLGN hard (discrete circuit)')
ax.set_xlabel('step')
ax.set_ylabel('test accuracy')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
ax.set_title('Same task, same training loop: MLP vs DLGN')
fig.tight_layout()
plt.show()

# > **読み取り**: MLP(青)が 0.95 前後で頭打ちになるのに対し、DLGN の hard(緑)は
# > 0.87 前後まで来る。差は約 0.08 で、**乗算を1つも持たない回路が実数 NN の9割の距離**まで
# > 到達している。橙(soft)と緑(hard)の隙間が**離散化ギャップ**で、この実行では
# > +0.01 程度しかない ── 学習中のモデルと焼いた回路がほぼ同じものだ、という意味である。
# > このギャップが大きく開くときは、ゲートが「どの論理関数か」を決めきれていない。
# --- 図2: 学習された「実数の重み」と「選ばれたゲート」 ---
# > 「学習された中身」の姿を左右に並べる。左は MLP の1層目の重みのヒストグラム、
# > 右は DLGN の1層目 2000 ゲートが**どの論理関数に確定したか**の内訳である。
# > 同じ「パラメータを学習した」でも、結果として得られるものがまるで違うことを見る。
# 左は MLP の1層目の重みの分布。0を中心に連続的に散らばった実数で、
# 推論のたびにこの数を掛ける(だから乗算器が要る)。
# 右は DLGN の1層目 2000ゲートが最終的にどの論理関数に確定したかの数。
# 学習の結果は「実数」ではなく **AND / XOR / NAND … という離散的な選択**として現れる。
# (並べ替えは「件数の降順 → 名前の昇順」。同数のゲートの順序を固定するため)
mlp.eval()
w1 = mlp[0].weight.detach().flatten()
dlgn.eval()
labels = gate_labels(dlgn[0])
names = sorted(set(labels), key=lambda n: (-labels.count(n), n))
counts = [labels.count(n) for n in names]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].hist(w1.numpy(), bins=60, color='#1f6fd0')
axes[0].set_xlabel('weight value')
axes[0].set_ylabel('count')
axes[0].set_title('MLP: real-valued weights of layer 1')
axes[1].bar(range(len(names)), counts, color='#2e9e4f')
axes[1].set_xticks(range(len(names)))
axes[1].set_xticklabels(names, rotation=60, ha='right', fontsize=8)
axes[1].set_ylabel('gates')
axes[1].set_title(f'DLGN: which logic function each of {K} gates became')
fig.tight_layout()
plt.show()

print('DLGN 1層目で選ばれたゲート(上位5種): '
      + ', '.join(f'{n}={c}' for n, c in zip(names[:5], counts[:5], strict=True)))

# > **読み取り**: 左は 0 を中心になめらかに広がる連続分布 ── どの値も「中途半端な実数」で、
# > 推論のたびに掛け算が要る。右は棒が16本(以下)しかない**離散的な内訳**で、
# > 各ゲートは OR / NOR / パススルー(A や B をそのまま通す)など
# > **名前の付いた論理関数**に落ちている(直後の出力に上位5種が出る)。
# > 学習の成果物が「実数のベクトル」から「回路の部品表」に変わった、というのがこの1枚の要点である。
# --- 図3: soft と hard は「別のモデル」である ---
# > 同じ1枚のテスト画像に対する10クラスのスコアを、soft(左)と hard(右)で並べる。
# > 普通の PyTorch では `eval()` は BatchNorm や Dropout の挙動を変えるだけだが、
# > **DLGN では `eval()` がモデルの数学そのものを差し替える**。この違いを実感するための図。
# 同じ1枚の画像に対する10クラスのスコアを、soft(連続緩和)と hard(確定した回路)で
# 並べる。DLGN では eval() がモデルの数学そのものを変えるので、
# 2つのスコアは似ているが同じではない。PyTorch の常識が通じない一番の勘所。
sample = 0
dlgn.train()
with torch.no_grad():
    soft_logits = dlgn(test_x[sample:sample + 1])[0]
dlgn.eval()
with torch.no_grad():
    hard_logits = dlgn(test_bits[sample:sample + 1])[0]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].imshow(test_bits[sample].reshape(20, 20), cmap='gray_r')
axes[0].set_title(f'input bits (true = {int(test_y[sample])})')
axes[0].set_axis_off()
axes[1].bar(range(10), soft_logits.numpy(), color='#e8850c')
axes[1].set_title('soft logits (train mode)')
axes[1].set_xlabel('class')
axes[2].bar(range(10), hard_logits.numpy(), color='#2e9e4f')
axes[2].set_title('hard logits (eval mode = the circuit)')
axes[2].set_xlabel('class')
fig.tight_layout()
plt.show()

print(f'soft の予測={int(soft_logits.argmax())} / '
      f'hard の予測={int(hard_logits.argmax())} / 正解={int(test_y[sample])}')

# > **読み取り**: 2つの棒グラフは形が似ているが**値は一致していない**。soft は16種の
# > 論理関数の混合が生む連続値、hard は「200本のうち何本が1を立てたか」という
# > 本数を tau で割った値であり、そもそも別の計算だからである。それでも最も高い
# > クラスは一致する ── 直後の出力で soft・hard・正解の3つが揃うことを確認できる。
# === まとめ: 置き換えチェックリスト ===
# > 最後に、MLP を DLGN へ移すときに毎回確認する8項目を出力する。
# > 各項目の背景は本ページで見てきたとおりで、詳細な層の対応表は
# > [PyTorchからの移行](/guides/from-pytorch/)にある。
# MLP を DLGN へ移すときに毎回確認する項目。詳細はガイド「PyTorchからの移行」に対応表がある。
print('''
=== MLP → DLGN 置き換えチェックリスト ===
1. nn.Linear + 活性化 → LutLayer(in, out, n_inputs=2)。活性化関数は付けない
2. 出力の nn.Linear(h, C) → GroupSum(k=C, tau)。out_dim は C で割り切れること
3. 幅を1〜2桁大きく取る(1ゲートは入力を2本しか見ないため)
4. 学習率を上げる(1e-3 ではなく 0.01 前後から)
5. 入力は [0,1] の実数か 0/1 のビット。hard 評価では round() したビットを入れる
6. 精度は soft と hard の2本を必ずログする(hard が本当の成績)
7. 離散化ギャップが大きいなら学習量・tau・パラメータ化を疑う
8. 焼く前に export_ir → validate → simulate で回路として検証する''')
