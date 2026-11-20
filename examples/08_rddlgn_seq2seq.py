"""RDDLGN(リカレント微分可能論理ゲート網)でseq2seqを学習する例。

ここまでのサンプル(XOR・MNIST)は「入力1個 → 出力1個」の分類器だった。
本サンプルは **系列 → 系列**、つまり翻訳器と同じ形をした
エンコーダ・デコーダを、全結合層ではなく論理ゲート(LutLayer)だけで組む。

モデルは RDDLGN(Bührer et al., 2025 / arXiv:2508.06097)。5つの論理ゲート群からなる:

    エンコーダ  N層 … 各トークン埋め込みを時刻ごとに変換(フィードフォワード)
                K層 … 時間方向のリカレント。k_t = K([n_t ; k_{t-1}])
                      → 最終 k_T が固定長の「文脈ベクトル」
    デコーダ    L層 … 出力側トークン埋め込みの変換(フィードフォワード)
                P層 … 自己回帰リカレント。p_t = P([p_{t-1} ; context ; l_t])
                M層 … [p_t ; context ; l_t] から語彙ロジットを作る(GroupSumで集計)

Attentionは無い。エンコーダが読んだ系列は **文脈ベクトル1本に圧縮** され、
デコーダはそれだけを頼りに出力を組み立てる。ここが本サンプルの見どころで、
「論理ゲートだけで固定長ボトルネックを通した記憶ができるか」を確かめる。

タスクは copy(入力をそのまま出す)と reverse(入力を逆順に出す)。
どちらも語彙8・長さ4の合成データで、experiments/data/synthetic.py の
CopyDataset / ReverseDataset と同じ作り方をexample内で自己完結させている。

## アーキテクチャ図解

まずブロック単位の全体像。緑が論理ゲート(LutLayer)、橙が実数のまま残る部分、
青がビット列である。

```mermaid
flowchart LR
    SRC["src トークン列<br/>3 5 4 4"] --> ENC["エンコーダ<br/>LogicEmbedding → N層 → K層"]
    ENC --> CTX["文脈ビット列<br/>384 ビットの 0/1"]
    CTX --> DEC["デコーダ<br/>LogicEmbedding → L層 → P層 → M層"]
    TGT["ひとつ前の出力<br/>(自己回帰)"] --> DEC
    DEC --> GS["GroupSum"] --> OUT["出力トークン列<br/>3 5 4 4"]
    OUT -.->|"次時刻の入力へ"| TGT

    classDef logic fill:#10b98126,stroke:#10b981
    classDef fp fill:#f9731626,stroke:#f97316
    classDef bits fill:#38bdf826,stroke:#38bdf8
    class ENC,DEC logic
    class GS fp
    class SRC,TGT,CTX,OUT bits
```

Attention が無いので、エンコーダが読んだ4トークンぶんの情報は
**この 384 ビットを通るしかない**。ここが唯一の通り道であることが、
このモデルのいちばん大事な性質である。

時間方向に **展開** した詳しい図は、下の「図0」で描く
(横軸が時刻で、同じ箱=同じ回路が毎時刻くり返し使われる様子と、
各ブロックのゲート数が入る)。流れているのは copy タスクの実データ1件目
`3 5 4 4` → `3 5 4 4` である。

`⊕` は **連結(concat)** であって加算ではない。P層・M層は
`[p(t-1) ⊕ context ⊕ l(t)]` という長いビット列をそのまま入力に取る。

**どう実現しているか**(初めて読む人向けの要点)

- **トークン → ビット列**: 単語IDを埋め込み表で実数ベクトルにし、sigmoid で
  0〜1に押し込む。これが「論理値」。eval のときは 0.5 を境に 0/1 へ切り捨てる
  (LogicEmbedding)。ここだけは実数の表を引く普通のNN(橙)。
- **層 = 論理ゲートの束**: N/K/L/P/M はどれも LutLayer を積んだだけ。
  1個の LutLayer は「入力ビットを2本ずつランダムに拾って、16通りの2入力論理関数から
  どれを使うかを学ぶゲート」を out_dim 個並べたもの。行列積は1つも無い。
- **時間方向の記憶**: K層は前の時刻の出力 k(t-1) を **入力に連結** して受け取る
  (図の横向き矢印)。これだけで「前に見たビットを次の時刻へ運ぶ」経路になる。
- **ボトルネック**: 系列を最後まで読み終えた k(T) が文脈ビット列。デコーダはこの
  384ビットしか見ない。長い系列や順序の入れ替え(reverse)は、この384ビットに
  入りきるかどうかが勝負になる。
- **出力**: M層の出したビットを GroupSum が語彙ごとにグループ分けして「1の個数」を数え、
  それをスコアにする。数えるところだけが実数(橙)。

実行: uv run python examples/08_rddlgn_seq2seq.py
期待結果: copy / reverse ともに hard token accuracy 0.95 前後(CPUで約1分半)
"""


# === 準備と設定 ===
# > 使うのは `RDDLGN` と `RDDLGNConfig` の2つだけ。図0(アーキテクチャの時間展開図)を
# > 自前で描くので、matplotlib の図形パーツも読み込んでおく。
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch, Patch

from griffith_logic import RDDLGN, RDDLGNConfig

# --- 再現性 ---
# > 論理ゲートの初期真理値表も、ゲート間の**ランダム配線**(どのゲートが前段のどの2本を
# > 見るか)も乱数で決まる。配線は学習中も固定なので、種を変えると回路そのものが別物になる。
SEED = 0

# --- タスクとハイパーパラメータ ---
# > 語彙8・長さ4の合成データで、系列を丸ごと運べるかだけを見る極小設定。特殊トークンの
# > `PAD` / `BOS`(系列の開始印) / `EOS`(終了印)は、通常の seq2seq と同じ約束事である。
# 語彙は「特殊トークン3個 + 実データ用トークン」。特殊トークンの割り当ては
# experiments/data/synthetic.py と揃えてある(PAD=0, BOS=1, EOS=2)。
TOKEN_PAD, TOKEN_BOS, TOKEN_EOS = 0, 1, 2
N_SPECIAL = 3
VOCAB = 8       # 語彙サイズ(特殊3個 + 実トークン5種)
SEQ_LEN = 4     # 入力・出力の長さ
N_DATA = 256    # 合成データ件数
STEPS = 600     # 学習イテレーション数
N_EVAL = 128    # 途中評価に使う件数(全件だと評価のほうが学習より重くなる)
BATCH = 64
LR = 0.03
# > **なぜ二値化正則化が要るのか**: LogicEmbedding は `sigmoid(埋め込み)` を論理値として
# > 使うので、値が 0.5 付近に居座ったまま学習が終わると、eval 時の Heaviside 二値化で
# > 情報がまるごと飛ぶ。$x(1-x)$ を薄く罰して 0 か 1 の端へ寄せておくのがこの重みである。
# 埋め込みの二値化正則化の重み。RDDLGNのLogicEmbeddingは
# sigmoid(埋め込み) を論理値として使うので、0.5付近に居座ると
# eval時のHeaviside二値化で情報が飛ぶ。x*(1-x) を薄く罰して0/1へ寄せる。
EMB_REG_WEIGHT = 0.01


# === データ: copy と reverse ===
# > タスクは2つ。**copy** は入力をそのまま出す恒等写像、**reverse** は入力を逆順に出す。
# > どちらも「入力を全部覚えていないと解けない」ので、文脈ベクトルの容量を直接試せる。
# >
# > 両者の差が大事である。copy は先に読んだものを先に出すので、状態が上書きされながらでも
# > 何とかなる。reverse は**最後に読んだトークンを最初に出す**ため、系列の順序情報まで
# > 384ビットに残っていなければ解けない ── ボトルネックへの要求が一段厳しい。
def make_task(task: str) -> tuple[torch.Tensor, torch.Tensor]:
    """copy / reverse の合成データを作る。

    Returns:
        src: (N, SEQ_LEN) 入力トークン列
        tgt: (N, SEQ_LEN) 正解の出力トークン列
    """
    # データ生成用のGeneratorはモデル初期化のRNGと分離する。
    # こうしておくと「モデルのseedだけ変えて同じデータで比較」ができる。
    gen = torch.Generator().manual_seed(7)
    src = torch.randint(N_SPECIAL, VOCAB, (N_DATA, SEQ_LEN), generator=gen)
    # copy = 恒等写像。reverse = 時間反転。どちらも「入力を全部覚えていないと
    # 解けない」タスクで、文脈ベクトルの容量を直接test する。
    tgt = src.clone() if task == 'copy' else src.flip(1)
    return src, tgt


# === モデル定義: 5つの層群と init_mode ===
# > RDDLGN は5つの層群でできている。**N層**が入力側トークンの変換、**K層**が時間方向の
# > リカレント(前時刻の出力を入力に連結して受け取る)、**L層**が出力側トークンの変換、
# > **P層**がデコーダ側のリカレント、**M層**が語彙ロジットを作る出力層である。
# > どれも LutLayer を積んだだけで、行列積は1つも登場しない。
# >
# > このセルでもう1つ効いてくるのが `init_mode` ── リカレント隠れ状態の**初期値**をどう
# > 置くかの選択で、hard 経路が純粋なブール回路として閉じるかどうかを左右する。
# > 理由はコード中のコメントに書いた通りで、結果は図3で実測する。
def make_config() -> RDDLGNConfig:
    """CPUで数十秒に収まる極小RDDLGN構成。

    RDDLGNConfig.tiny() よりさらに小さい。論文Table 1(from_paper)は
    合計100万ゲート超なのでCPUでは回らない。ここでは各層群を
    「役割が分かる最小の幅」まで削っている。
    """
    return RDDLGNConfig(
        vocab_size=VOCAB,
        d_emb=48,               # 埋め込み=48ビットの論理ベクトル
        seq_len=SEQ_LEN,
        n_layer_sizes=[192],    # エンコーダ前段(フィードフォワード)
        k_layer_sizes=[384],    # エンコーダのリカレント。最終幅=文脈ベクトル幅
        l_layer_sizes=[192],    # デコーダ前段
        p_layer_sizes=[384],    # デコーダのリカレント状態
        m_layer_sizes=[512, VOCAB * 24],  # 出力層。最終幅は語彙の倍数が必須
        tau=1.0,                # GroupSumの温度
        # リカレント隠れ状態の初期値。既定の 'zero' は 0.5(論理的に「不定」)で、
        # 学習には中立でよいが **eval時に非二値が回路へ流れ込む** 。0.5を入力に
        # 受けたLUTは 0.25 / 0.75 … と中間値を出し、それが時間方向へ伝播して
        # 「離散回路のはずの文脈ベクトルが連続値」という状態になる。
        # 'one'(1.0固定)なら初期状態も正真正銘のビットなので、hard評価が
        # 最初から最後まで純粋なブール回路として閉じる(図3で確認する)。
        init_mode='one',
    )


# --- teacher forcing(学習時だけ正解を見せる) ---
# > 自己回帰モデルは本来「自分が出した1つ前のトークン」を入力に取るが、学習中それをやると
# > 序盤の誤りが雪だるま式に増えて学習が進まない。そこで学習時だけ**正解列を1つ右へずらして**
# > 入力に使う。これが teacher forcing で、seq2seq の定石である。
def shifted_decoder_input(tgt: torch.Tensor) -> torch.Tensor:
    """teacher forcing 用に、正解列を1つ右へずらしてBOSを先頭に置く。

    デコーダは「t番目を出すとき t-1 番目までの正解を見ている」状態で学習する
    (自己回帰モデルの定石)。ずらさないと答えをそのまま見てしまう。
    """
    bos = torch.full((tgt.shape[0], 1), TOKEN_BOS, dtype=tgt.dtype)
    return torch.cat([bos, tgt[:, :-1]], dim=1)


# === 学習ループ ===
# > 中身は普通の PyTorch。損失は各時刻のトークン分類 cross entropy に、埋め込みの
# > 二値化正則化を足したものである。150 step ごとに **soft**(学習時の連続緩和のままの姿)と
# > **hard**(埋め込みもゲートも二値化した「焼いた回路」の姿)の精度を両方記録しておき、
# > 学習が進むにつれて2本がどう近づくかを図1で見る。
def train(task: str) -> tuple[list, torch.Tensor, RDDLGN]:
    """1タスクを学習し、(履歴, 最終予測, モデル) を返す。"""
    torch.manual_seed(SEED)
    model = RDDLGN(make_config())
    src, tgt = make_task(task)
    dec_in = shifted_decoder_input(tgt)

    counts = model.count_parameters()
    print(f'[{task}] ゲートパラメータ {counts["total"]:,} '
          f'(N={counts["n_layers"]:,} K={counts["k_layers"]:,} '
          f'L={counts["l_layers"]:,} P={counts["p_layers"]:,} '
          f'M={counts["m_layers"]:,})')

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    history = []  # (step, soft acc, hard acc)

    for step in range(STEPS + 1):
        if step % 150 == 0:
            # soft = 学習時の連続緩和のまま / hard = 埋め込みもゲートも二値化した
            # 「焼いた回路」の姿。両方を並べるのはMNISTの例と同じ作法。
            with torch.no_grad():
                model.train()
                soft = accuracy(model, src[:N_EVAL], dec_in[:N_EVAL], tgt[:N_EVAL])
                model.eval()
                hard = accuracy(model, src[:N_EVAL], dec_in[:N_EVAL], tgt[:N_EVAL])
            history.append((step, soft, hard))
            print(f'[{task}] step {step:>4}  soft={soft:.4f}  hard={hard:.4f}')

        model.train()
        idx = torch.randint(0, N_DATA, (BATCH,))
        logits, emb_reg = model(src[idx], dec_in[idx])
        # 主損失: 各時刻のトークン分類。系列をまとめて (B*T, V) に潰して渡す。
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), tgt[idx].reshape(-1)
        )
        # 埋め込みを0/1へ寄せる正則化を足す(hard評価との差を詰めるため)。
        loss = loss + EMB_REG_WEIGHT * emb_reg
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(src, dec_in)[0].argmax(-1)
    return history, pred, model


# > 指標は**トークン単位の accuracy**(位置ごとに1文字ずつ当たったかを平均する)。
# > 系列まるごと一致で数える指標より甘いが、「どの位置で崩れたか」を図2で追える。
def accuracy(model: RDDLGN, src, dec_in, tgt) -> float:
    """teacher forcing 下のトークン単位 accuracy。"""
    logits, _ = model(src, dec_in)
    return (logits.argmax(-1) == tgt).float().mean().item()


# === 図解: アーキテクチャを時間方向に展開する ===
# > 次の2つは作図用のヘルパー(ブロックと矢印)。色は冒頭の Mermaid 図と同じ凡例
# > ── 緑=論理ゲート / 橙=実数演算 / 青=ビット列 / 灰=トークン ── に揃えてある。
def arch_box(ax, x, y, text, kind, w=1.55, h=0.66):
    """アーキテクチャ図の1ブロックを描く(色は種別で決まる)。

    kind: 'logic'(論理ゲート=緑) / 'fp'(実数=橙) / 'bits'(ビット列=青) /
          'token'(トークン=灰)
    """
    palette = {
        'logic': ('#10b981', '#10b98140'),
        'fp': ('#f97316', '#f9731640'),
        'bits': ('#38bdf8', '#38bdf840'),
        'token': ('#94a3b8', '#94a3b840'),
    }
    edge, face = palette[kind]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.05',
        linewidth=1.4, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5, zorder=4)


def arch_arrow(ax, p0, p1, label=None, dashed=False):
    """ブロック間の矢印(必要なら中間にラベルを置く)。"""
    ax.annotate('', xy=p1, xytext=p0, zorder=2, arrowprops=dict(
        arrowstyle='-|>', lw=1.1, color='#64748b', shrinkA=1, shrinkB=1,
        linestyle='--' if dashed else '-'))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.42, label,
                ha='center', va='bottom', fontsize=7, color='#475569')


# --- 図0: 時間展開したアーキテクチャ ---
# > **この図で何を見るか**: 横軸が時刻、縦軸が層。左半分がエンコーダ(4トークンを読む)、
# > 右半分がデコーダ(1トークンずつ書く)で、真ん中の青い箱が両者をつなぐ**唯一の経路**である
# > 文脈ビット列。流れているのは copy タスクの実データ1件目 `3 5 4 4` そのもの。
# >
# > 見どころは3つ。**(1) 縦に並ぶ列はどれも同じ回路**で、時刻ごとに使い回されているだけ
# > (だから時間展開しても回路は増えない)。**(2) 横向きの矢印は K層と P層にしかない** ──
# > 情報が時間方向へ渡る道はそこだけ。**(3) 青い箱の幅 384 ビット**が、系列全体の情報が
# > 通り抜けなければならない穴の大きさである。
cfg = make_config()
src_demo, _tgt_demo = make_task('copy')
demo = [int(v) for v in src_demo[0]]
n_gates, k_gates = cfg.n_layer_sizes[-1], cfg.k_layer_sizes[-1]
l_gates, p_gates = cfg.l_layer_sizes[-1], cfg.p_layer_sizes[-1]
m_gates = sum(cfg.m_layer_sizes)
enc_x = [1.1, 2.9, 4.7, 6.5]
dec_x = [10.1, 11.9, 13.7]
ctx_x = 8.3
y_tok, y_emb, y_first, y_rec, y_m, y_gs, y_out = 0.6, 1.9, 3.2, 4.5, 5.8, 7.1, 8.4

fig, ax = plt.subplots(figsize=(14.5, 7.4))
for i, x in enumerate(enc_x):
    arch_box(ax, x, y_tok, f'src({i + 1}) = {demo[i]}', 'token', w=1.25, h=0.5)
    arch_box(ax, x, y_emb, f'LogicEmbedding\n{cfg.d_emb} bits (FP table)', 'fp')
    arch_box(ax, x, y_first, f'N layer\n{n_gates} gates', 'logic')
    arch_box(ax, x, y_rec, f'K layer\n{k_gates} gates', 'logic')
    arch_arrow(ax, (x, y_tok + 0.25), (x, y_emb - 0.33))
    arch_arrow(ax, (x, y_emb + 0.33), (x, y_first - 0.33))
    arch_arrow(ax, (x, y_first + 0.33), (x, y_rec - 0.33))
    if i:
        arch_arrow(ax, (enc_x[i - 1] + 0.78, y_rec), (x - 0.78, y_rec),
                   label=f'k({i})')
arch_arrow(ax, (enc_x[-1] + 0.78, y_rec), (ctx_x - 0.9, y_rec))
arch_box(ax, ctx_x, y_rec, f'context bits\nk(4): {k_gates} bits', 'bits',
         w=1.8, h=0.8)

for i, x in enumerate(dec_x):
    token = 'BOS' if i == 0 else f'{demo[i - 1]} (prev out)'
    arch_box(ax, x, y_tok, f'dec in: {token}', 'token', w=1.5, h=0.5)
    arch_box(ax, x, y_emb, f'LogicEmbedding\n{cfg.d_emb} bits (FP table)', 'fp')
    arch_box(ax, x, y_first, f'L layer\n{l_gates} gates', 'logic')
    arch_box(ax, x, y_rec, f'P layer\n{p_gates} gates', 'logic')
    arch_box(ax, x, y_m, f'M layers\n{m_gates} gates', 'logic')
    arch_box(ax, x, y_gs, f'GroupSum\ntau={cfg.tau}', 'fp')
    arch_box(ax, x, y_out, f'out = {demo[i]}', 'token', w=1.25, h=0.5)
    arch_arrow(ax, (x, y_tok + 0.25), (x, y_emb - 0.33))
    arch_arrow(ax, (x, y_emb + 0.33), (x, y_first - 0.33))
    arch_arrow(ax, (x, y_first + 0.33), (x, y_rec - 0.33))
    arch_arrow(ax, (x, y_rec + 0.33), (x, y_m - 0.33))
    arch_arrow(ax, (x, y_m + 0.33), (x, y_gs - 0.33))
    arch_arrow(ax, (x, y_gs + 0.33), (x, y_out - 0.25))
    if i:
        arch_arrow(ax, (dec_x[i - 1] + 0.78, y_rec), (x - 0.78, y_rec),
                   label=f'p({i})')
    ax.plot([ctx_x, x], [y_rec - 0.62, y_rec - 0.62], color='#38bdf8',
            lw=1.0, ls='--', zorder=1)
    arch_arrow(ax, (x, y_rec - 0.62), (x, y_rec - 0.33), dashed=True)
ax.plot([ctx_x, ctx_x], [y_rec - 0.62, y_rec - 0.4], color='#38bdf8', lw=1.0,
        ls='--', zorder=1)

ax.text((enc_x[0] + enc_x[-1]) / 2, y_rec + 0.62,
        'ENCODER  (reads the sequence)',
        ha='center', fontsize=11, weight='bold', color='#334155')
ax.text((dec_x[0] + dec_x[-1]) / 2, y_out + 0.55,
        'DECODER  (writes it back, autoregressive)',
        ha='center', fontsize=11, weight='bold', color='#334155')
ax.text(ctx_x, y_rec - 1.15, 'the only path between them',
        ha='center', fontsize=7.5, color='#0284c7')
ax.text((enc_x[0] + enc_x[-1]) / 2, y_gs + 0.35,
        'Every column is the SAME circuit, reused at each time step.\n'
        'Only the K layer carries information sideways (k(t)),\n'
        'so the whole source sequence must fit into k(4).',
        ha='center', va='center', fontsize=9, color='#475569')
handles = [
    Patch(facecolor='#10b98140', edgecolor='#10b981', label='logic gates (LutLayer)'),
    Patch(facecolor='#f9731640', edgecolor='#f97316', label='real-valued (FP)'),
    Patch(facecolor='#38bdf840', edgecolor='#38bdf8', label='bit vector (0/1)'),
    Patch(facecolor='#94a3b840', edgecolor='#94a3b8', label='token'),
]
ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.9,
          bbox_to_anchor=(0.0, 1.0))
ax.set_xlim(0, 15)
ax.set_ylim(-0.4, 9.6)
ax.axis('off')
fig.suptitle('RDDLGN unrolled in time — copy task, src = '
             + ' '.join(str(v) for v in demo), fontsize=13)
fig.tight_layout()
plt.show()

# > **読み取り**: エンコーダ側は K層だけが横に伸びており、`k(1) → k(2) → k(3) → k(4)` と
# > 上書きされながら進む。デコーダ側の3列はすべて同じ青い箱(点線)から給電されている ──
# > つまりデコーダは元の系列を一度も見ず、**畳み込まれた 384 ビットだけ**を頼りに書いている。
# > Attention 付きのモデル(論理ゲートGPT)がこの点線の代わりに全時刻への直通路を持つ、
# > という対比で見ると両者の違いがはっきりする。
# === 学習の実行と結果 ===
# > copy と reverse を同じ設定・同じ種で順に学習する。両方が同じ回路容量で解けるのか、
# > それとも順序の反転がボトルネックに効くのかを、次の出力と図1で比べる。
results = {}
for task_name in ('copy', 'reverse'):
    results[task_name] = train(task_name)

for task_name, (hist, _pred, _m) in results.items():
    print(f'最終 [{task_name}] soft={hist[-1][1]:.4f} / hard={hist[-1][2]:.4f}')

# > hard accuracy が 0.95 前後まで来ていれば、**論理ゲートだけで固定長ボトルネックを
# > 通した系列変換ができた**ことになる。下の assert の閾値 0.80 は環境差を見込んだ緩い線で、
# > 「回路が読めている」ことの最低ラインとして置いている。
# 0.95前後まで来る。閾値は環境差を見て緩めに置く(回路が「読めている」ことの確認)。
for task_name, (hist, _pred, _m) in results.items():
    assert hist[-1][2] > 0.80, f'{task_name} の hard acc={hist[-1][2]:.4f} が低すぎる'

# === 可視化: ボトルネックはどこまで運べたか ===
# --- 図1: 学習曲線(copy vs reverse) ---
# > **この図で何を見るか**: 左が copy、右が reverse で、縦軸はトークン accuracy。
# > 橙が soft(学習時の姿)、緑が hard(argmax で確定した実回路)。2本が重なって伸びれば
# > 「学習した性能がそのまま回路に残っている」ことを意味する。
# (行頭が copy/cat/ls 等のシェルコマンド名だと jupytext がコメントを外してしまうので避ける)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, task_name in zip(axes, ('copy', 'reverse'), strict=True):
    hist = results[task_name][0]
    steps, soft_hist, hard_hist = zip(*hist, strict=True)
    ax.plot(steps, soft_hist, 'o-', color='#e8850c', label='soft (train mode)')
    ax.plot(steps, hard_hist, 's-', color='#2e9e4f', label='hard (discrete)')
    ax.set_title(f'{task_name} task')
    ax.set_xlabel('step')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('token accuracy')
axes[0].legend(loc='lower right')
fig.suptitle('RDDLGN seq2seq — copy vs reverse', fontsize=12)
fig.tight_layout()
plt.show()

# > **読み取り**: 2枚ともほぼ同じ高さまで上がる ── この長さ(4トークン)なら、順序を
# > 反転する reverse でも 384 ビットに十分収まる。ここで reverse だけが伸び悩むようなら、
# > 文脈ベクトルの幅(`k_layer_sizes`)か系列長がボトルネックに当たっているサインである。
# > soft と hard が近いのは、埋め込みの二値化正則化と `init_mode='one'` が効いているため。
# --- 図2: 予測列の答え合わせ ---
# > **この図で何を見るか**: 先頭16サンプルについて「正解/予測」を位置ごとにマス目で並べる。
# > 緑が一致、赤が外れ。全体の平均だけでなく**どの位置で崩れるか**を見るための図である。
# > Attention を持たないモデルは系列の中ほどを落としやすい、というのが典型的な予想。
n_show = 16
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, task_name in zip(axes, ('copy', 'reverse'), strict=True):
    _hist, pred, _m = results[task_name]
    _src, tgt = make_task(task_name)
    ok = (pred[:n_show] == tgt[:n_show]).numpy()
    ax.imshow(ok, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for i in range(n_show):
        for t in range(SEQ_LEN):
            # 上段=正解、下段=予測。数字が一致していれば緑マス。
            ax.text(t, i, f'{int(tgt[i, t])}/{int(pred[i, t])}',
                    ha='center', va='center', fontsize=7)
    ax.set_xticks(range(SEQ_LEN))
    ax.set_xlabel('output position t')
    ax.set_title(f'{task_name}: true/pred (green = correct)')
axes[0].set_ylabel('sample')
fig.suptitle('RDDLGN — teacher-forced predictions (hard eval)', fontsize=12)
fig.tight_layout()
plt.show()

# > **読み取り**: 赤マスが散らばっていれば「特定の位置が苦手」ではなく、単に容量が足りて
# > いない一様な誤りである。逆に特定の列(位置 t)だけ赤が集中していたら、その時刻の情報が
# > ボトルネックで押し出されている ── 系列を長くすると、まずそういう形で崩れ始める。
# --- 図3: 文脈ベクトルは本当に「離散ビット列」か ---
# > **この図で何を見るか**: エンコーダが出した文脈ベクトルを、そのまま白黒の画像として描く。
# > 黒が 1、白が 0。ここが**灰色を1つも含まない**ことが、このモデルが「そのまま HDL に
# > 落ちる」ことの担保である(灰色があれば、それは回路に存在しない中間値)。
# >
# > 直前の `assert` が実際にその判定をしている。`init_mode='zero'` にすると初期状態 0.5 が
# > 時間方向へ伝播してここが崩れる ── `init_mode` を config で選ぶ理由がこれである。
model = results['copy'][2]
src, tgt = make_task('copy')
model.eval()
with torch.no_grad():
    context = model.encoder(model.src_embedding(src[:24]))
uniq = torch.unique(context)
print(f'文脈ベクトル: shape={tuple(context.shape)} 出現値={uniq.tolist()[:5]}')
assert bool(((context == 0.0) | (context == 1.0)).all()), \
    'hard eval なのに文脈ベクトルが二値でない'

# > **読み取り**(このページ最後の図): 白と黒しか無く、中間の灰色が1つも無いはずである。
# > さらに、行ごと(=サンプルごと)に模様が違うのに縦縞のように揃った列もある ── 揃った列は
# > どの入力でも同じ値になる「使われていないビット」で、そこがボトルネックの余裕にあたる。
# > 系列を長くしていくと、まずこの余裕が食い潰され、次に図2の赤マスが増え始める。
# 幅384のうち先頭128ビットだけを表示(全部だと潰れる)。
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.imshow(context[:, :128], cmap='binary', aspect='auto', interpolation='nearest')
ax.set_xlabel('context bit (first 128 of 384)')
ax.set_ylabel('sample')
ax.set_title('encoder context vector — exactly {0,1} in hard eval')
fig.tight_layout()
plt.show()
