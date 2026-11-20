"""学習アルゴリズム切り替えの例。

with_learning() でレイヤ単位に学習ルールを注釈し、
HybridTrainer が BP / DFA / EGGROLL / AP法 を協調実行する。

論理ゲート網は「backpropが使えるなら使う」が、ハードウェア上での局所学習や
勾配が流せない状況も想定される。そこで本ライブラリは**層ごとに学習則を選べる**
設計になっている。ここでは同じXORを、誤差逆伝播を使わない2つの方式
(DFA / AP法)で解いて、モデル定義は同じまま学習則だけ差し替えられることを確認する。

3つの学習則の違いは一言でいうと「**誤差がどの経路で各層に届くか**」だけである。

- BP … 出力の誤差を、後ろの層の重みを使って1層ずつ逆向きに運ぶ(層が直列に依存)
- DFA … 出力の誤差を、**固定のランダム行列**で各層へ直接ばらまく(層が互いに独立)
- AP法 … 誤差を運ばない。各ゲートが自分の真理値表を局所的に書き換える(微分なし)

実行: uv run python examples/03_learning_rules.py
期待結果: DFAとAP法の両方でXORが学習される

読み方の注意: **AP法は損失(cross entropy)を最適化対象にしていない**ので、
損失曲線だけを見ると「高いまま途中で止まった=失敗」に見えてしまう。AP法の成否は
正解率で判断すること(正解率100%に達した時点で打ち切っている)。そのため下の図は
左=損失・右=正解率の2パネルにしてある。
"""

# === 準備: 学習則を切り替えるためのAPI ===
# > 本ライブラリから使うのは3つで、これだけで**層ごとに学習則を混ぜられる**。
# >
# > - `with_learning(layer, spec)` … 層をラップして「この層はこの規則で更新せよ」と
# >   注釈する。層そのものの定義は変えない
# > - `LearningSpec(rule=...)` … 学習則の指定(`'bp'` / `'dfa'` / `'ap'` / `'eggroll'`)
# > - `HybridTrainer` … 注釈を読み取り、層ごとに適切な更新を呼び分ける学習器
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from griffith_logic import (
    GroupSum,
    HybridTrainer,
    LearningSpec,
    LutLayer,
    with_learning,
)

BP_COLOR, DFA_COLOR, AP_COLOR = '#3b6fd4', '#e8850c', '#2e9e4f'


# === 3つの学習則は「誤差の届き方」だけが違う ===
# > 実験に入る前に、3つの学習則の違いを一言で押さえておく。違いは
# > **誤差(= どちらへパラメータを動かすかの信号)がどの経路で各層に届くか**だけである。
# >
# > - **BP**(誤差逆伝播) … 出力の誤差を、後ろの層の重みを使って1層ずつ逆向きに運ぶ。
# >   層が直列に依存するので、前の層の更新は後ろの層を待たなければならない
# > - **DFA**(Direct Feedback Alignment) … 出力の誤差を**固定のランダム行列**で
# >   各層へ直接ばらまく。層同士が依存しないので並列・局所的に更新できる
# > - **AP法**(Akiho Propagation) … 誤差を層間で運ばない。各ゲートが自分の真理値表を
# >   局所的に書き換える。微分を一切使わないので optimizer すら要らない
# >
# > **なぜ BP 以外が要るのか** ── ハードウェア上での学習や、勾配を流せない状況を
# > 想定しているため。局所更新で済む規則ほど回路として実装しやすい。
# --- 図1の下ごしらえ: 「誤差の経路」を描く関数 ---
# 実験を始める前に、まず3つの学習則が何をしているのかを絵で押さえておく。
# 縦に積んだ箱が層(下が入力・上が出力)で、灰色の上向き矢印がforward。
# 色付きの矢印が「誤差(=どちらへパラメータを動かすかの信号)がどう届くか」を表す。
# ここが3つの唯一の違いであり、モデルの定義は3つとも完全に同じで良い。
def draw_rule_diagram(ax, rule, color, caption):
    """1つの学習則について、層の積みと誤差の経路を模式図として描く。"""
    names = ['input x', 'LutLayer 1', 'LutLayer 2', 'GroupSum + loss']
    for i, name in enumerate(names):
        # 層の箱。y=i が下から数えた段数になる。
        ax.add_patch(plt.Rectangle((0.10, i - 0.22), 0.52, 0.44,
                                   facecolor='#f0f0f0', edgecolor='#666', lw=1.0))
        ax.text(0.36, i, name, ha='center', va='center', fontsize=8.5)
        if i < len(names) - 1:
            # forward(灰色): どの学習則でも共通で、入力から出力へ値が流れる。
            ax.annotate('', xy=(0.20, i + 0.78), xytext=(0.20, i + 0.22),
                        arrowprops=dict(arrowstyle='->', color='#999', lw=1.4))

    if rule == 'bp':
        # BP: 出力から1層ずつ、後ろの層の重みを経由して誤差が降りてくる(直列)。
        for i in (3, 2):
            ax.annotate('', xy=(0.55, i - 0.78), xytext=(0.55, i - 0.22),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
        ax.text(0.68, 1.5, 'chain rule\nlayer by layer', fontsize=8, color=color,
                va='center')
    elif rule == 'dfa':
        # DFA: 出力の誤差を固定ランダム行列 B で各層へ「直接」配る(層が独立)。
        for i, rad in ((2, -0.30), (1, -0.55)):
            ax.annotate('', xy=(0.66, i), xytext=(0.66, 3),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                        connectionstyle=f'arc3,rad={rad}'))
        ax.text(0.80, 2.0, 'fixed random B\n(direct, parallel)', fontsize=8,
                color=color, va='center')
    else:
        # AP法: 誤差は層をまたがない。各ゲートが自分の真理値表を局所的に加算更新する。
        for i in (1, 2):
            ax.annotate('', xy=(0.62, i + 0.16), xytext=(0.62, i - 0.16),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                        connectionstyle='arc3,rad=-1.6'))
        ax.text(0.80, 1.5, 'local table update\n(no gradient at all)', fontsize=8,
                color=color, va='center')

    ax.set_xlim(0.0, 1.35)
    ax.set_ylim(-0.7, 3.7)
    ax.set_title(caption, fontsize=10, color=color)
    ax.set_axis_off()


# --- 図1: 誤差はどこから来るのか ---
# > 上の説明を3枚の模式図にする。縦に積んだ箱が層(下が入力・上が出力)で、
# > 灰色の上向き矢印が forward(値の流れ)。**色付きの矢印が誤差の届き方**で、
# > ここが3つの唯一の違いである。
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
draw_rule_diagram(axes[0], 'bp', BP_COLOR, 'BP: error walks back through weights')
draw_rule_diagram(axes[1], 'dfa', DFA_COLOR, 'DFA: error is broadcast directly')
draw_rule_diagram(axes[2], 'ap', AP_COLOR, 'AP: no error transport at all')
fig.suptitle('Where does the error signal come from? (gray = forward, color = update signal)',
             fontsize=11)
fig.tight_layout()
plt.show()

# > **読み取り**: 左(BP)は矢印が層を1段ずつ降りてくる。中央(DFA)は出力から
# > 各層へ矢印が直接伸びており、層が互いを待たない。右(AP)は矢印が層をまたがず、
# > 各層の中で完結している ── 誤差の輸送そのものが無い。
# >
# > **重要なのは、モデルの定義は3つとも同じでよい**ということ。変わるのは
# > `with_learning` による注釈だけである。
# === データ: 同じXORを3つの規則で解く ===
# > 問題を [付録: XOR](/tutorials/xor/) と同じ4パターンに固定する。
# > **問題を固定して学習則だけを差し替える**ので、以降に見える差はすべて
# > 「更新のしかた」の差として読める。
# XORの4パターン(01_xor.py と同じ問題を、学習則だけ変えて解く)。
# 問題を固定して学習則だけを差し替えるので、以降の差はすべて「更新のしかた」の差になる。
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 0])

# === DFA: 誤差を固定ランダム行列でばらまく ===
# > **「でたらめな行列で本当に学習できるのか」** ── これが DFA を初めて見たときの
# > 当然の疑問である。答えは「できる」で、理由は**前向きの重みの側が、その固定行列に
# > 合わせて向きを揃えていく**から(feedback alignment と呼ばれる現象)。
# >
# > 層同士が依存しないので並列・局所的に更新でき、ハードウェア実装と相性が良い。
# > 代償として精度は BP に劣ることが多い。
# >
# > コード上は `with_learning(..., LearningSpec(rule='dfa'))` で層を包むだけで、
# > `LutLayer` の定義は XOR のページとまったく同じである。
# 通常のbackpropは「1つ前の層の重み」を使って勾配を後ろへ運ぶため、層が直列に依存する。
# DFAは出力側の誤差を**固定のランダム行列**で各層へ直接ばらまく。層同士が依存しないので
# 並列・局所的に更新でき、ハードウェア実装と相性が良い(精度はBPに劣ることが多い)。
# 「でたらめな行列で本当に学習できるのか」と思うところだが、前向きの重みの側が
# その固定行列に合わせて向きを揃えていく(feedback alignment)ため学習が進む。
torch.manual_seed(7)
model_dfa = nn.Sequential(
    # with_learning() は層をラップして「この層はこの規則で更新せよ」と注釈するだけ。
    # 層そのもの(LutLayer)の定義は 01_xor.py と全く同じで良い。
    with_learning(LutLayer(2, 32, n_inputs=2), LearningSpec(rule='dfa')),
    with_learning(LutLayer(32, 16, n_inputs=2), LearningSpec(rule='dfa')),
    GroupSum(k=2, tau=2.0),
)
# HybridTrainer が注釈を読み取り、層ごとに適切な更新を呼び分ける。
# DFAはoptimizerを介してパラメータを更新するので、ここではAdamを渡す。
trainer = HybridTrainer(model_dfa, torch.optim.Adam(model_dfa.parameters(), lr=0.05))
loss_dfa, acc_dfa = [], []
for _ in range(600):
    # train_step が forward・誤差配分・更新までを1回分やって指標を返す。
    # 返る dict の 'accuracy' は hard accuracy(離散化した回路としての正解率)。
    m = trainer.train_step(x, y)
    loss_dfa.append(m['loss'])
    acc_dfa.append(m['accuracy'])
trainer.close()  # フック等の後始末(学習則によっては内部状態を持つため必須)
print(f"DFA:  acc={m['accuracy']:.2f}")

# === AP法: 誤差を運ばない直接更新 ===
# > AP法は各ゲートについて「その真理値表エントリを選んだとき正解に近づくか」を集計し、
# > 良かった方向へスコアを直接足し込む。**微分を一切使わない**ので optimizer も要らず、
# > `HybridTrainer` に optimizer を渡していない点がそのまま現れている。
# >
# > 1つだけ制約がある。`parameterization='independent'` が必須で、真理値表の各エントリを
# > 独立な変数として持たせる必要がある ── AP法がエントリ単位で加算更新するため、
# > 16種の論理関数上の softmax 混合では書き換えられないからである
# > (**学習則がパラメータ化を要求する**、という具体例)。
# 各ゲートについて「その真理値表エントリを選んだとき正解に近づくか」を集計し、
# 良かった方向へスコアを直接足し込む。微分を一切使わないので optimizer も要らない。
# 誤差を層間で運ばないぶん、1ステップで動かせる情報量はBPより粗いが、
# XOR程度の小さい問題では数ステップで解に飛び込むことが多い。
torch.manual_seed(5)
model_ap = nn.Sequential(
    with_learning(
        # parameterization='independent': 真理値表の各エントリを独立な変数として持つ。
        # AP法はエントリ単位で加算更新するため、16種の論理関数上のsoftmax混合ではなく
        # この独立表現である必要がある(学習則がパラメータ化を要求する例)。
        LutLayer(2, 32, n_inputs=2, parameterization='independent'),
        LearningSpec(rule='ap', lr=0.5),
    ),
    with_learning(
        LutLayer(32, 16, n_inputs=2, parameterization='independent'),
        LearningSpec(rule='ap', lr=0.5),
    ),
    GroupSum(k=2, tau=2.0),
)
# optimizerを渡していない点に注目。AP法は勾配を作らず自前で値を書き換える。
trainer = HybridTrainer(model_ap)
loss_ap, acc_ap = [], []
for _ in range(300):
    m = trainer.train_step(x, y)
    loss_ap.append(m['loss'])
    acc_ap.append(m['accuracy'])
    # AP法は数十ステップで解けることが多いので、正解率100%になったら打ち切る。
    # 注意: AP法は損失を最適化対象にしていない(勾配を作らない)ため、損失は高いまま
    # 打ち切られる。成否は損失ではなく正解率で見ること。
    if m['accuracy'] == 1.0:
        break
ap_steps = len(acc_ap)  # 早期終了までのステップ数(乱数次第で変わるので実測値を使う)
print(f"AP法: acc={m['accuracy']:.2f} ({ap_steps} steps で acc=1.0 に到達し早期終了)")

# === BP: 比較のためのベースライン ===
# > 学習則を注釈しなければ、ただの PyTorch モデルとして `backward()` で学習する。
# > `HybridTrainer` すら要らず、`zero_grad → forward → backward → step` の定型どおり。
# > 3本を同じ図に重ねると「どこまで速く落ちるか」の差がそのまま見える。
# 学習則を注釈しなければ、ただのPyTorchモデルとして backward() で学習する。
# HybridTrainer すら要らず、zero_grad → forward → backward → step の定型どおり。
# 3本を同じ図に重ねると、「どこまで速く落ちるか」の差がそのまま見える。
torch.manual_seed(7)
model_bp = nn.Sequential(
    LutLayer(2, 32, n_inputs=2),
    LutLayer(32, 16, n_inputs=2),
    GroupSum(k=2, tau=2.0),
)
opt = torch.optim.Adam(model_bp.parameters(), lr=0.05)
loss_bp, acc_bp = [], []
for _ in range(600):
    opt.zero_grad()
    logits = model_bp(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    loss_bp.append(loss.item())
    acc_bp.append((logits.argmax(-1) == y).float().mean().item())
print(f"BP:   acc={acc_bp[-1]:.2f}")


# === 評価: 何ステップで解けたかを数える ===
# > 最終正解率だけを見ると3つとも 1.00 で並んでしまい、差が見えない。実際に効くのは
# > **速さ**なので、収束したステップ数を拾う。
# >
# > ただし XOR は4サンプルしかなく「たまたま1回だけ全問正解した」がすぐ起きる。
# > そこで初到達ではなく、**そこから最後まで 1.0 を維持した位置**(= 腰を据えて解けた点)を
# > 収束と定義する。過大評価を避けるための定義であり、指標の設計そのものが教材である。
# 最終正解率だけを見ると3つとも 1.00 で並んでしまい、差が見えない。
# 実際に効くのは速さなので、収束したステップ数を拾う。ただしXORは4サンプルしかなく
# 「たまたま1回だけ全問正解した」がすぐ起きるため、初到達ではなく
# **そこから最後まで1.0を維持した位置**(=腰を据えて解けた点)を収束と定義する。
def solved_step(acc_curve):
    """最後まで acc=1.0 が続く区間の先頭ステップ番号(1始まり)。未収束なら None。"""
    if acc_curve[-1] != 1.0:
        return None
    i = len(acc_curve) - 1
    while i > 0 and acc_curve[i - 1] == 1.0:
        i -= 1
    return i + 1


solved = {
    'BP': solved_step(acc_bp),
    'DFA': solved_step(acc_dfa),
    'AP': solved_step(acc_ap),
}
for name, s in solved.items():
    print(f'{name:>3}: acc=1.0 に落ち着いたステップ = {s if s else "未収束"}')

# --- 図2: loss 曲線(左)と正解率曲線(右) ---
# > 3本を左右2パネルに重ねる。横軸のスケールが大きく違う(AP法は早期打ち切り)ので、
# > 対数軸(symlog)にして形を比べる。
# >
# > **左パネルだけで AP法を評価しないこと**。AP法は損失を最適化対象にしていないため、
# > 損失は高いまま短く途切れ、一見「失敗して止まった」ように見える。
# > **AP法の成否は右パネル(正解率)で判断する**。右では AP法の終点(=早期終了した瞬間)に
# > ★と注釈を付けてあり、そこが「成功して止まった」ことの証拠になる。
# 横軸のスケールが違う(APは早期打ち切り)ので、対数軸で重ねて形を比べる。
#   BP  … 勾配を層をまたいで正確に運ぶ。最も安定して落ちる基準線
#   DFA … 誤差を固定ランダム行列で各層へ直接ばらまく。BPより遅いが層が独立
#   AP  … 微分を使わず真理値表エントリを直接加算更新する。数十ステップで解ける
#
# **左パネル(損失)だけでAP法を評価しないこと**。AP法は損失を最適化対象にしないので
# 損失は高いまま短く途切れ、一見「失敗して止まった」ように見える。右パネル(正解率)を
# 見ると、AP法が数ステップで acc=1.0 に跳ね上がって早期終了しているのが分かる。
# 右パネルではAP法の終点(=早期終了した瞬間)に★と注釈を付けている。ここが
# 「失敗して途切れた」のではなく「成功して止まった」ことの証拠。
bp_label = f'BP (backprop, {len(loss_bp)} steps)'
dfa_label = f'DFA (feedback alignment, {len(loss_dfa)} steps)'
ap_label = f'AP (no backprop, solved in {ap_steps} steps)'

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.2))

ax_loss.plot(loss_bp, label=bp_label, color=BP_COLOR)
ax_loss.plot(loss_dfa, label=dfa_label, color=DFA_COLOR)
ax_loss.plot(loss_ap, label=ap_label, color=AP_COLOR)
ax_loss.set_xscale('symlog')
ax_loss.set_xlabel('step (symlog)')
ax_loss.set_ylabel('cross entropy loss')
ax_loss.set_title('loss (AP does NOT optimize this)')
ax_loss.grid(alpha=0.3)
ax_loss.legend(fontsize=8)

ax_acc.plot(acc_bp, label=bp_label, color=BP_COLOR)
ax_acc.plot(acc_dfa, label=dfa_label, color=DFA_COLOR)
ax_acc.plot(acc_ap, label=ap_label, color=AP_COLOR)
ax_acc.plot(ap_steps - 1, acc_ap[-1], marker='*', markersize=16, color=AP_COLOR,
            zorder=5, linestyle='none')
ax_acc.annotate(f'✓ acc=1.0 → early stop (step {ap_steps})',
                xy=(ap_steps - 1, acc_ap[-1]), xytext=(46, -46),
                textcoords='offset points', fontsize=9, color=AP_COLOR,
                arrowprops=dict(arrowstyle='->', color=AP_COLOR, lw=1.2))
ax_acc.set_xscale('symlog')
ax_acc.set_xlabel('step (symlog)')
ax_acc.set_ylabel('hard accuracy')
ax_acc.set_ylim(-0.05, 1.15)
ax_acc.set_title('accuracy (this is how to judge AP)')
ax_acc.grid(alpha=0.3)
ax_acc.legend(fontsize=8, loc='lower right')

fig.suptitle('XOR: same model, three learning rules')
fig.tight_layout()
plt.show()

# > **読み取り**: 左では BP(青)と DFA(橙)の損失が下がり、AP(緑)だけが高い位置で
# > 途切れる。右を見ると3本とも hard accuracy 1.0 に到達しており、AP は最も早く到達して
# > そこで打ち切られている ── 「損失を見て失敗と判断する」誤読がいかに起きやすいかが分かる。
# --- 図3: 何ステップで解けたか(棒グラフ) ---
# > 曲線図は形を見るためのもので、**速さの比較は棒のほうが一目で分かる**。
# > 横軸は hard accuracy が 1.0 に落ち着くまでのステップ数で、短いほど速い。
# >
# > **読み取り(この図が最後なので先に書く)**: 3つとも数ステップ〜十数ステップで解けており、
# > 「backprop を捨てても解ける」ことが XOR の規模では実際に確認できる。
# > ただしこれは4サンプルの極小問題での話で、AP法が常に BP より速いという主張ではない。
# > [1. MNISTを学習する](/tutorials/mnist/) の規模では、BP が基準線として最も強いことが多い。
# 上の曲線図は形を見るためのもので、「速さ」の比較は棒にしたほうが一目で分かる。
# 棒が短いほど速い。3つとも数ステップ〜十数ステップの世界で、
# 「backpropを捨てても解ける」ことがXORの規模では実際に確認できる。
# ただしこれは4サンプルの極小問題での話で、AP法が常にBPより速いという主張ではない。
# 実データ(examples/06_mnist.py の規模)ではBPが基準線として最も強いことが多い。
names = [n for n, s in solved.items() if s is not None]
values = [solved[n] for n in names]
colors = {'BP': BP_COLOR, 'DFA': DFA_COLOR, 'AP': AP_COLOR}

fig, ax = plt.subplots(figsize=(8, 3.2))
bars = ax.barh(names, values, color=[colors[n] for n in names], height=0.55)
ax.bar_label(bars, labels=[f'{v} steps' for v in values], padding=6, fontsize=9)
ax.set_xlim(0, max(values) * 1.35)
ax.set_xlabel('steps until hard accuracy settles at 1.0 (shorter is faster)')
ax.set_title('How fast does each rule solve XOR?')
ax.grid(alpha=0.3, axis='x')
fig.tight_layout()
plt.show()
