"""論理ゲート回路で物体検出(CenterNet-LUT)を解く例。

examples/06_mnist.py は「画像1枚 → クラス1個」の分類だった。物体検出はそこから
2段階むずかしくなる: **どこに**(座標)を**個数不定で**答えなければならない。

ここでは CenterNet(Zhou et al. 2019)の考え方をそのまま論理ゲートに載せる。
CenterNet はアンカーもNMSネットワークも使わず、3枚の特徴マップだけで検出する:

    heatmap: 物体中心がそのセルにある確率。局所ピーク=1個の検出
    size (2ch):   そのセルの物体の幅・高さ(画像サイズで正規化)
    offset (2ch): セル格子からのサブピクセルずれ

この「出力が全部Conv特徴マップ」という構造が論理ゲートNNと相性が良い。
バックボーンは LutConv2d(3x3窓の真理値表を学習する微分可能LUTツリー)と
OrPooling2d(OR/maxプーリング)だけで組め、size/offsetのような**数値**は
SpatialGroupSum(発火しているゲートの本数を数えて定数倍する)で作れる。
つまり乗算器を1つも使わずに座標回帰まで完結する検出器になる。

**回路の全体像(このサンプルの縮小レシピ k=8・3層の実寸)**

```mermaid
graph LR
  subgraph BB["バックボーン: 解像度を 64×64 → 16×16 へ、全部 LutConv2d"]
    direction LR
    IMG["入力画像<br/>64×64 グレースケール<br/>(実数)"]
    TH["閾値二値化<br/>3枚のビット平面<br/>(3, 64, 64)<br/>学習パラメータなし"]
    C1["LutConv2d 3×3<br/>3ch → 8ch<br/>深さ3のLUT木<br/>64×64"]
    P1["OrPooling2d /2<br/>8ch<br/>32×32"]
    C2["LutConv2d 3×3<br/>8ch → 16ch<br/>32×32"]
    P2["OrPooling2d /2<br/>16ch<br/>16×16"]
    C3["LutConv2d 3×3<br/>16ch → 32ch<br/>16×16 = 検出格子"]
    IMG --> TH --> C1 --> P1 --> C2 --> P2 --> C3
  end

  subgraph HEAD["3つのヘッド(同じ 32ch/16×16 特徴マップから生やす)"]
    direction LR
    HM["hm_conv: LutConv2d 3×3<br/>32ch → 16ch (= 1 × 16)"]
    HMS["SpatialGroupSum(1, 1/16)<br/>16本ずつ数えて 1/16 倍<br/>出力 (1, 16, 16)<br/>= 中心らしさ"]
    SZC["size_conv: LutConv2d 3×3<br/>32ch → 32ch (= 2 × 16)"]
    SZS["SpatialGroupSum(2, 1/16)<br/>16本ずつ数えて 1/16 倍<br/>出力 (2, 16, 16) = 幅・高さ"]
    OFC["off_conv: LutConv2d 3×3<br/>32ch → 8ch (= 2 × 4)"]
    OFS["SpatialGroupSum(2, 1/4)<br/>4本ずつ数えて 1/4 倍<br/>出力 (2, 16, 16) = dx, dy"]
    HM --> HMS
    SZC --> SZS
    OFC --> OFS
  end

  C3 --> HM
  C3 --> SZC
  C3 --> OFC

  subgraph DEC["デコード(学習パラメータなし)"]
    direction LR
    PK["ヒートマップの局所ピークを拾う<br/>score ≥ 0.15 のセルだけ採用"]
    BOX["ボックス = (セル座標 + dx,dy) × ストライド4<br/>と (幅, 高さ) から復元"]
    PK --> BOX
  end
  HMS --> PK
  SZS --> BOX
  OFS --> BOX

  subgraph LEG["凡例"]
    direction LR
    LGL["緑 = 論理(0/1 のビット。LutConv2d / OrPooling)"]
    LGR["橙 = 実数(SpatialGroupSum の出力・座標値)"]
    LGB["青 = 学習パラメータを持たない前後処理"]
  end

  style IMG fill:#f9731626,stroke:#f97316
  style TH fill:#38bdf826,stroke:#38bdf8
  style C1 fill:#10b98126,stroke:#10b981
  style P1 fill:#10b98126,stroke:#10b981
  style C2 fill:#10b98126,stroke:#10b981
  style P2 fill:#10b98126,stroke:#10b981
  style C3 fill:#10b98126,stroke:#10b981
  style HM fill:#10b98126,stroke:#10b981
  style HMS fill:#f9731626,stroke:#f97316
  style SZC fill:#10b98126,stroke:#10b981
  style OFC fill:#10b98126,stroke:#10b981
  style SZS fill:#f9731626,stroke:#f97316
  style OFS fill:#f9731626,stroke:#f97316
  style PK fill:#38bdf826,stroke:#38bdf8
  style BOX fill:#38bdf826,stroke:#38bdf8
  style LGL fill:#10b98126,stroke:#10b981
  style LGR fill:#f9731626,stroke:#f97316
  style LGB fill:#38bdf826,stroke:#38bdf8
```

**具体例が図としてどこに出るか**: 上の流れは、このページの図で実際に見られる。

1. 「input (binarized)」= 上図の TH の出力(64×64のビット平面)
2. 「GT center heatmap」vs「predicted heatmap (16x16)」= 上図の HMS の出力。
   数字の中心の位置に**山が立つ**ようになれば学習できている
3. 「predicted boxes」= 上図の BOX。山のピーク位置に size/offset を当てはめて箱を描いた結果

**どう実現しているか**

- **入口で画像をビットにする**。グレースケールを3つの閾値で切り、
  「濃さ」を3枚のビット平面(3ch)に変換する。論理回路は0/1しか受け取れないので、これが作法
- **畳み込み = 3×3窓の真理値表**。LutConv2d は 3×3=9画素の窓を見て 0/1 を返す
  LUT の木(深さ3)。CNNのフィルタが「重みの内積」なのに対し、こちらは「表引き」で、
  学習するのは表の中身だけ
- **プーリングは OR**。max プーリングは、ビットの世界では「どれか1つでも1なら1」= OR に一致する。
  比較器も加算器も要らない
- **数値(確率・幅・高さ・座標)は本数を数えて作る**。乗算器を持たない回路が実数を出す方法が
  SpatialGroupSum で、「32本のうち何本が1か」を16本ずつ数え、1/16 倍して 0〜1 の値にする。
  分解能は 1/16 刻み ── 精度が要るなら size_group を増やす。
  **ヒートマップも同じ作りにしてある**: 論理ゲートの出力は評価時に 0/1 へ離散化されるので、
  1chのままだとヒートマップが二値になりピークが全部同点になってしまう。
  16本の平均(hm_group=16)にすることで 0, 1/16, ..., 1 の17段階の確からしさが出せる
- **アンカーもNMSも要らない**。CenterNet は「中心セルだけを1にする」問題に落としてあるので、
  出力は全部特徴マップのまま。デコードは局所ピーク探索だけで済む

【importについて】
モデル定義 experiments/detection_model.py はまだパッケージ外(experiments/配下)に
あるため、本サンプルは sys.path を通して読み込む。将来 griffith_logic 側の
公開APIへ移す予定で、そのときは import 行だけが変わる。

【スケールについて】
CPUで数分に収まるよう、本サンプルは大幅に絞った縮小版になっている:
  - 64x64キャンバスに1〜2個のMNIST数字(本番レシピは96x96)
  - バックボーン幅 k=8(本番は64)・3層(本番は8層)
  - 500イテレーション
  - **クラス非依存(1クラス)検出**: 「数字がある/ない」だけを当てる
狙いは精度ではなく「ボックスがそれらしい位置・大きさに出る」ことを目で見ること。
10クラス(数字の識別まで行う)本気のレシピは experiments/train_detection.py にある
(→[モデル一覧](/reference/model-zoo/))。なおモデル一覧に載っている検出の実測値は
compute_map の画像スコープ修正より前の計測なので、再計測待ちである。

実行: uv run python examples/21_detection.py
所要: CPUで2〜4分(マシン負荷による)
期待結果: クラス非依存 mAP@0.5 は 0.20〜0.30 程度。最後の図に予測ボックスが描かれる
"""

# === 準備: import と縮小レシピの設定 ===
# > 必要なのは PyTorch と matplotlib、そして `experiments/` 側にある検出まわりの部品
# > (データセット・損失・デコード・mAP)だけ。回路本体の部品(`LutConv2d` / `OrPooling2d` /
# > `SpatialGroupSum`)は、それらのモデル定義の中から使われている。
import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# --- experiments/ を import パスに追加する ---
# > モデル定義 `detection_model.py` はまだパッケージ外(`experiments/` 配下)にあるため、
# > `sys.path` を通してから読み込む。将来 `griffith_logic` の公開 API へ移す予定で、
# > そのときは import 行だけが変わる。
# ノートブックでは __file__ が未定義なので、その場合はカレントディレクトリを
# リポジトリルートとみなす(examples はリポジトリルートから実行する規約)。
REPO_ROOT = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if '__file__' in globals()
    else os.getcwd()
)
sys.path.insert(0, os.path.join(REPO_ROOT, 'experiments'))

from data.detection import MultiMNISTDetection, collate_detection  # noqa: E402
from detection_loss import CenterNetLoss  # noqa: E402
from detection_model import CenterNetLUT  # noqa: E402
from detection_utils import compute_map, decode_detections  # noqa: E402

torch.manual_seed(0)

# --- ハイパーパラメータ(CPUで数分に収まる縮小レシピ) ---
# > **出力ストライド**(`STRIDE`)は「入力画像の何画素が検出格子の1マスに対応するか」。
# > 64×64 の画像をストライド4で見るので、検出格子は 16×16 になる。座標はこの格子の上で
# > 予測され、最後に `× STRIDE` で画像座標へ戻す。`SCORE_TH` はヒートマップのピークを
# > 「検出1個」として採用する下限スコアで、下げるほど拾い過ぎ・上げるほど取りこぼす。
CANVAS = 64        # 合成画像の一辺。heatmapは 64/4 = 16x16 になる
STRIDE = 4         # 出力ストライド
K = 8              # バックボーンの幅パラメータ(本番は64)
N_THRESH = 3       # 入力の閾値二値化チャンネル数(グレースケール→ビット平面3枚)
MAX_OBJ = 2        # 1枚あたり最大2物体
STEPS = 500        # 学習イテレーション数
BATCH = 32
LR = 0.01
N_TRAIN = 3000
N_TEST = 200
EVAL_EVERY = 125
SCORE_TH = 0.15    # heatmapピークを検出として採用する閾値
HM_GROUP = 16      # heatmapヘッドの本数(この本数の平均が確からしさになる)

# === データ: 合成 Multi-MNIST と CenterNet 形式の教師 ===
# > MNIST の数字を 64×64 キャンバスへランダムに置き(重なりなし)、CenterNet 形式の教師を作る。
# > 教師は3種類の特徴マップで、これがそのままモデルの3つの出力に対応する。
# >
# > - **heatmap** ── 物体の中心がそのセルにある確率。中心セルを1、その周りを
# >   なだらかなガウシアン(山なりの重み)にして、少しのズレを強く罰しないようにする
# > - **size**(2ch) ── そのセルの物体の幅・高さ(画像サイズで正規化した 0〜1 の値)
# > - **offset**(2ch) ── 格子は粗いので、セルの中心から真の中心までの**サブピクセルのずれ**
# >
# > 画像のほうは閾値二値化されて `(3, 64, 64)` の**ビット平面3枚**になる。
# > 論理回路の入力線は 0/1 しか受け取れないので、この二値化が入口の作法である。
def materialize(n, train, seed):
    """データセットをn件だけ一気にテンソル化する(毎stepの合成コストを省く)。"""
    ds = MultiMNISTDetection(
        mnist_root=os.path.join(REPO_ROOT, 'data-mnist'), train=train,
        canvas_size=CANVAS, min_objects=1, max_objects=MAX_OBJ,
        n_thresholds=N_THRESH, output_stride=STRIDE, n_classes=10,
        num_samples=n, seed=seed,
    )
    loader = DataLoader(ds, batch_size=n, collate_fn=collate_detection)
    images, targets = next(iter(loader))
    # --- クラス非依存化 ---
    # 10クラス分のheatmapをクラス方向のmaxで1枚に畳む。「どの数字か」を捨てて
    # 「そこに物体があるか」だけを学ぶので、小さな回路・短い学習でも山が立つ。
    # (10クラス版に戻すには、この2行を消して n_classes=10 でモデルを作るだけ)
    targets['heatmap'] = targets['heatmap'].amax(dim=1, keepdim=True)
    targets['gt_labels'] = torch.zeros_like(targets['gt_labels'])
    return images, targets


# === モデル: 全部 LutConv2d の CenterNet ===
# > 先頭3行でデータを実体化してから、検出器を組む(1つのセルにまとめてある)。
# > バックボーンは Conv→Pool→Conv→Pool→Conv の3ステージで 64×64 を 16×16 に落とし、
# > その**同じ特徴マップから3つのヘッド**(heatmap / size / offset)を生やす。
# >
# > **なぜ中心点だけで物体を検出できるのか**: 「物体を囲む枠」を直接当てにいくのではなく、
# > 「中心がどのセルか」という**セル単位の分類**と「そのセルでの幅・高さ・ずれ」という
# > 数値予測に分解しているから。アンカー(あらかじめ用意した候補枠)も
# > NMS(重複枠を消す後処理ネットワーク)も要らず、出力は全部 Conv の特徴マップのままになる。
# > この「出力が全部特徴マップ」という性質が、論理ゲート回路と相性が良い理由でもある。
# >
# > **なぜ幅・高さのような数値を出せるのか**: 乗算器を持たない回路が実数を作る手段が
# > `SpatialGroupSum` で、「32本のうち何本が1か」を16本ずつ数えて 1/16 倍する。
# > 分解能は 1/16 刻み ── 精度が要るなら `size_group` を増やす。
train_x, train_t = materialize(N_TRAIN, train=True, seed=0)
test_x, test_t = materialize(N_TEST, train=False, seed=1)
print(f'train={tuple(train_x.shape)}  heatmap={tuple(train_t["heatmap"].shape)}')

# ===========================================================================
# モデル
# ---------------------------------------------------------------------------
# 3ステージのバックボーン(Conv→Pool→Conv→Pool→Conv)で 64x64 → 16x16 に落とし、
# 同じ特徴マップから3つのヘッド(heatmap / size / offset)を生やす。
# 全てのConvが LutConv2d、すなわち学習対象は「3x3窓に対する真理値表」だけ。
# hm_group / size_group / offset_group は全て明示的に渡す ── experiments/ 側の
# 既定値は本番レシピの都合で変わりうるので、上図と数値が一致し続けるよう固定する。
# ===========================================================================

model = CenterNetLUT(
    n_classes=1, k=K, in_channels=N_THRESH, tree_depth=3,
    head_depth=3, n_inputs=2, hm_group=HM_GROUP, size_group=16,
    offset_group=4, n_layers=3,
)
print(f'CenterNet-LUT: {model.num_gates:,} gates '
      f'(backbone {K}→{2 * K}→{4 * K} ch + heads hm/size/off)')

# === 損失と評価: focal loss と mAP@0.5 ===
# > **focal loss** は「正解が極端に少ない」分類のための損失で、簡単に当たる背景セルの寄与を
# > 弱めて、中心セルの学習に重みを寄せる。16×16=256 マスのうち中心はたかだか2マスなので、
# > これが無いと「全部背景」と答えるだけの回路に落ち着いてしまう。
# >
# > 評価指標の **mAP@0.5** は物体検出の標準的な指標。予測枠と正解枠の重なり具合を
# > **IoU**(Intersection over Union = 重なりの面積 ÷ 合併の面積、0〜1)で測り、
# > IoU が 0.5 以上なら「当たり」とみなして、当たりの取りこぼしと拾い過ぎを平均で1つの数にする。
# CenterNetの複合損失:
#   heatmap = focal loss(中心セルだけを1へ、周辺のガウシアンには弱いペナルティ)
#   size/offset = 中心セルだけを mask+index で抜き出したL1回帰
loss_fn = CenterNetLoss(lambda_size=0.1, lambda_off=1.0)
opt = torch.optim.Adam(model.parameters(), lr=LR)


# --- 評価: decode → mAP ---
# > 検出のデコードは学習パラメータを持たない後処理で、ヒートマップの**局所ピーク**
# > (3×3 の窓の中で自分が最大のセル)だけを残し、そこへ size と offset を当てはめて枠を復元する。
# > 「局所ピークを残す」操作が NMS の役割をそのまま果たしているので、専用の後処理器は要らない。
def evaluate():
    """テストセットでmAP@0.5を測る。decode → mAP がCenterNetの評価経路。"""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for i in range(0, N_TEST, 50):
            out = model(test_x[i:i + 50])
            # 3x3 max poolでピークだけ残し(NMS相当)、size/offsetでボックスを復元
            preds.extend(decode_detections(
                out['hm'], out['size'], out['off'],
                output_stride=STRIDE, top_k=MAX_OBJ * 2, threshold=SCORE_TH,
            ))
    for i in range(N_TEST):
        m = test_t['mask'][i] > 0
        gts.append({'boxes': test_t['gt_boxes'][i][m],
                    'labels': test_t['gt_labels'][i][m]})
    model.train()
    return compute_map(preds, gts, n_classes=1, iou_threshold=0.5)[0]


# === 学習ループ ===
# > 中身は普通の PyTorch で、論理ゲート回路であることを意識する箇所はどこにもない。
# > `EVAL_EVERY` ステップごとに mAP を測り、損失の内訳(hm / size / off)も一緒に出す。
# > 内訳を見れば「どのヘッドが伸び悩んでいるか」がそのまま分かる。
history = []  # (step, total loss, mAP)
for step in range(STEPS + 1):
    idx = torch.randint(0, N_TRAIN, (BATCH,))
    out = model(train_x[idx])
    target = {k: v[idx] for k, v in train_t.items()}
    loss, parts = loss_fn(out, target)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % EVAL_EVERY == 0:
        m_ap = evaluate()
        history.append((step, parts['total_loss'], m_ap))
        print(f'step {step:>4}  loss={parts["total_loss"]:.4f} '
              f'(hm={parts["hm_loss"]:.3f} size={parts["size_loss"]:.3f} '
              f'off={parts["off_loss"]:.3f})  mAP@0.5={m_ap:.4f}')

final_map = history[-1][2]
print(f'\n最終 mAP@0.5(クラス非依存) = {final_map:.4f} / 総ゲート数 {model.num_gates:,}')
# 縮小レシピなので閾値は緩め(「まったく検出できていない」ことだけを弾く)。
assert final_map > 0.10, f'mAP={final_map:.4f} が低すぎる'

# === 可視化: 学習の推移と検出結果 ===
# --- 図1: 損失と mAP の推移 ---
# > **この図で何を見るか**: 左軸(橙)が CenterNet の合計損失、右軸(緑)が mAP@0.5。
# > 損失が下がるだけでなく、**検出指標のほうも一緒に伸びているか**を確かめる図である。
steps, losses, maps = zip(*history, strict=True)
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(steps, losses, 'o-', color='#e8850c', label='total loss')
ax1.set_xlabel('step')
ax1.set_ylabel('CenterNet loss', color='#e8850c')
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(steps, maps, 's-', color='#2e9e4f', label='mAP@0.5')
ax2.set_ylabel('mAP@0.5 (class-agnostic)', color='#2e9e4f')
ax2.set_ylim(0, max(0.4, max(maps) * 1.2))
ax1.set_title('CenterNet-LUT on synthetic Multi-MNIST')
fig.tight_layout()
plt.show()

# > **読み取り**: size の損失は早々に 0 付近へ落ちる ── MNIST の数字は常に 28×28 で、
# > 幅・高さが実質的に定数だからである。難しいのは heatmap と offset のほうで、
# > mAP はほぼ heatmap のピーク品質だけで決まっていることが2本の線の形から読める。
# --- 図2: 学習した heatmap ---
# > **この図で何を見るか**: 左が入力(二値化後)、中央が教師の中心ヒートマップ、
# > 右が回路の出力。**数字の中心の位置に山が立っているか**が唯一の見どころで、
# > ここが合っていれば残りは座標の細かい補正でしかない。
model.eval()
with torch.no_grad():
    sample = model(test_x[:1])

# 入力は3枚のビット平面。可視化ではチャンネルmax=「どれかの閾値を超えた画素」
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
axes[0].imshow(test_x[0].amax(0), cmap='gray_r')
axes[0].set_title('input (binarized, max over 3 planes)', fontsize=9)
axes[1].imshow(test_t['heatmap'][0, 0], cmap='hot')
axes[1].set_title('GT center heatmap', fontsize=9)
axes[2].imshow(sample['hm'][0, 0], cmap='hot')
axes[2].set_title('predicted heatmap (16x16)', fontsize=9)
for ax in axes:
    ax.set_axis_off()
fig.suptitle('center heatmap: ground truth vs learned circuit', fontsize=11)
fig.tight_layout()
plt.show()

# > **読み取り**: 教師のヒートマップが**なだらかな山**なのに対し、予測は「点が数個光る」
# > 見た目になる。評価時の LUT 出力が {0,1} へ離散化され、確からしさが 16本の多数決
# > (1/16 刻みの17段階)でしか表せないためで、山の裾野は再現されない。
# > それでも**ピークの位置さえ合っていれば**デコードは正しい枠を返す ── だからこの縮小
# > レシピでも mAP が立ち上がる。裾野の形まで欲しければ、幅 `k` と層数を増やす必要がある。
# --- 図3: 予測ボックスを画像に描く(本サンプルの締め) ---
# > **この図で何を見るか**: 水色の破線が正解の枠、実線が予測の枠(緑 = 正解と IoU 0.5 以上、
# > 赤 = 外した予測)。位置と大きさがそれらしく重なっていれば、**乗算器を1つも持たない
# > 論理ゲート回路だけで座標まで回帰できている**ことが目で確認できる、という締めの図である。
with torch.no_grad():
    out = model(test_x[:8])
# heatmapの分解能は1/16刻みなのでスコアが同点になりやすく、閾値だけでは誤検出を
# 切りきれない。図では見やすさのため上位MAX_OBJ個だけ描く(mAPの計算では上のtop_k=4のまま)。
dets = decode_detections(out['hm'], out['size'], out['off'],
                         output_stride=STRIDE, top_k=MAX_OBJ,
                         threshold=SCORE_TH)


def iou(a, b):
    """1対1のIoU([x1, y1, x2, y2] 同士)。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# > 最後の図なので、**読み取り方も先に**書いておく。8枚のうち緑の枠が付いた数字は
# > 位置も大きさも当たっている一方、赤い枠は「山が立った場所がずれた」ケースで、
# > 数字どうしが近いときに起きやすい。枠の**大きさ**がどれもよく似ているのは、
# > MNIST の数字が常に 28×28 だからで、size ヘッドが定数を覚えれば済むためである。
# > つまりこの縮小レシピで難しいのは大きさではなく**中心の当て方**だ、と図から言える。
fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
n_hit, n_pred = 0, 0
for i, ax in enumerate(axes.ravel()):
    ax.imshow(test_x[i].amax(0), cmap='gray_r')
    ax.set_axis_off()
    m = test_t['mask'][i] > 0
    gt_boxes = test_t['gt_boxes'][i][m].tolist()
    for x1, y1, x2, y2 in gt_boxes:
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False,
            edgecolor='#00b3d6', lw=1.2, ls='--'))
    for box, score in zip(dets[i]['boxes'].tolist(),
                          dets[i]['scores'].tolist(), strict=True):
        ok = any(iou(box, g) >= 0.5 for g in gt_boxes)
        n_hit += int(ok)
        n_pred += 1
        color = '#2e9e4f' if ok else '#c0392b'
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor=color, lw=1.6))
        ax.text(box[0], box[1] - 2, f'{score:.2f}', color=color, fontsize=7)
fig.suptitle('predicted boxes (cyan dashed = GT, green = IoU>=0.5, red = miss)',
             fontsize=11)
fig.tight_layout()
plt.show()
print(f'表示した8枚の予測ボックス {n_pred} 個のうち、GTとIoU>=0.5 だったもの: {n_hit} 個')
