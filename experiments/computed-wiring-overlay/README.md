# computed-wiring-overlay(計算型配線の較正アンカー)

配線を「層ごとのずらし量」で計算する **rot 型の固定配線**が、固定ランダム配線・
置換配線と比べて精度を落とさないことを確かめる実験フォルダ。overlay(ゲートをメモリに
置いて 1 個ずつ評価する方式)ではゲート記述の 87.5% が前層アドレスなので、配線を規則で
計算できると記述が 32 bit → 4 bit になる。その前提が**精度側で成り立つか**を測るのが
ここの役目で、HW 側(DE0-CV の fit と実機)は mdl-fpga が持つ。

- 学習の本体は `griffith train` の共通ランナー。腕の定義は `configs/recipes/cwo_*.yaml` にある
- 原値・許容幅の真実源は `anchors.json`(非公開の主張台帳からの転記)。**判定の閾値をコードに
  直書きしない**

## 目的(何を確かめるか)

| アンカー | 主張 | 対照 | 判定 |
|---|---|---|---|
| C3 | MNIST 48K / 96K で rot 配線は固定ランダム配線と同等以上 | P0 固定ランダム配線 | 各腕 ±0.5pt、かつ P1 − P0 >= +0.4pt |
| C4 | CIFAR-10 48K で「クラス受容野が入力を覆う」よう設計した rot は置換配線を下回らない | P 置換配線 / R 旧 rot(受容野 30.7%) | 各腕 ±0.3pt、かつ SR2 − P > 0 |

C4 の条件が本体で、rot が負けるのは**規則配線そのものではなくシフト量の設計**の問題だ、
というのが主張。受容野は学習せずに配線だけから決まるので、走らせる前に確認できる:

```python
from griffith_logic.analysis.wiring_stats import (
    class_receptive_field, layer_stats, receptive_field_upper_bound)
```

- `class_receptive_field` — GroupSum のクラス割当から後ろ向きに辿って、1 クラスが
  読める入力ビットの割合。SR2 は 1.0、旧 rot は 0.307
- `receptive_field_upper_bound` — 幅 W の 2 入力層は高々 2W 本しか入力を読めないので、
  受容野は `min(1, 2W / 入力次元)` を超えられない(CIFAR-10 9,216 入力なら
  24K = 幅 4,000 で 86.8%、12K = 幅 2,000 で 43.4%)。この上限より広い受容野を
  要求する設計は、シフト量をどう選んでも実現しない

## ファイル

| ファイル | 役割 |
|---|---|
| `anchors.json` | 原値・許容幅・差の規則(claims.json からの転記)。集計と L3 テストが読む |
| `run_calibration.py` | 7 レシピ × seed を順に回して run ごとの結果 JSON を落とす |
| `aggregate.py` | 結果を `anchors.json` と突き合わせ、判定表を md / JSON で出す |
| `export_ir.py` | 学習済みネットを IR へ書き出し、各層の indices が閉形式どおりか全数検証する |
| 引き継ぎメモ | 引き継ぎ(現状・未完了・再現手順) |

関連:

- レシピ: `configs/recipes/cwo_*.yaml`(7 本)
- L1: `tests/l1_math/test_wiring_stats.py`(受容野の解析値)
- L3: `tests/l3_paper/test_computed_wiring_anchors.py`(`@paper`。主張そのものを assert)
- 設定: `tests/test_training_wandb_config.py`(レシピから wandb の taxonomy を渡す)

## 再走手順

```bash
# 0) 準備(初回のみ)。CIFAR-10 は repo 直下 data-cifar へ自動ダウンロードされる
uv sync --extra dev --extra cu128     # GPU 環境。CPU だけなら --extra cpu

# 1) 配管の確認(1 エポック。判定には使えない)
uv run python experiments/computed-wiring-overlay/run_calibration.py \
    --smoke --seeds 0 --out-root outputs/cwo-smoke

# 2) 本走(7 レシピ × seed 0/1 = 14 run)
WANDB_PROJECT=griffith-logic \
uv run python experiments/computed-wiring-overlay/run_calibration.py --device cuda

# 3) 集計(原値との差と判定)
uv run python experiments/computed-wiring-overlay/aggregate.py
#   -> outputs/computed-wiring-overlay/calibration_table.md / .json

# 4) IR 書き出しと配線の検証(C1 / C2 の入力になる P6 のネット)
uv run python experiments/computed-wiring-overlay/export_ir.py \
    outputs/computed-wiring-overlay/cwo_mnist_48k_p6_s1
uv run python experiments/computed-wiring-overlay/export_ir.py \
    outputs/computed-wiring-overlay/cwo_mnist_96k_p6_s0
```

1 本ずつ CLI で回すなら `griffith launch` でよい(同じ `load_config` → `run_training` を通る):

```bash
uv run griffith launch --grid "run.seed=[0,1]" configs/recipes/cwo_mnist_48k_p1.yaml --local
```

データを別の場所から読むときは `--mnist-root` / `--cifar-root`
(`md` なら `data.params.root=...`)を渡す。

## 実装の tips

- **P0 の配線種を間違えない**。先行実験の P0 は `torch.randint` の**無制約乱択**なので
  レシピは `kind: random_uniform`。library 既定の `random` は使用回数が均衡する乱択で、
  先行実験では P7(ランダム置換)に相当する別の腕になる(+0.3pt ほど底上げされる)
- **指標がアンカーごとに違う**。C3 は最終エポックの `hard_acc`(先行実験は final のみを
  報告している)、C4 は 100 ステップごとの eval の最大 `best_hard_acc`(先行実験の A_best)。
  `anchors.json` の `metric` が正で、集計も L3 テストもそこから読む
- **seed をレシピに書かない**。`griffith launch --grid "run.seed=[0,1]"` で展開する。
  レシピに書くと grid と二重指定になって事故のもとになる
- **配線用と初期化用の RNG は別ストリーム**(runner の `_WIRING_SEED_OFFSET`)。
  init_mode やゲートマスクを変えても配線は変わらないので、腕どうしが同じ配線を共有できる
- **受容野は走らせる前に確認する**。新しいシフト量を試すときは
  `class_receptive_field` で 1.0 に届くかを見てから学習する(数秒で終わる)
- **λ=0 の正則化ブロックは落としてよい**。先行実験の model.yaml には
  `regularization: {kind: irreducibility, lambda: 0.0}` があるが、λ=0 では損失グラフに
  一切触れないので学習は bitwise 不変(記録用の `reg_penalty` が出なくなるだけ)

## 先行実験(原実験)との差

再走は「同じ結論が出るか」を見るもので、1 run 対 1 run の一致は**原理的に取れない**。
違いは次の 3 点:

1. **RNG ストリーム**: 先行実験は 1 つの Generator で「配線 → ゲート初期化」を逐次消費する。
   共通ランナーは配線用(seed + 1,000,003)と初期化用(seed)を分けている。バッチ順も
   DataLoader の実装が違う。したがって判定は **2 seed 平均**に対して行う
2. **P0 の配線生成器**: 先行実験は `torch.randint` を idx_a / idx_b で 2 回引く。
   ランナーは `get_uniform_random_connections` が `(2, out_dim)` を 1 回で引く。
   分布は同じ(各スロット独立一様)だが値は違う
3. **評価点**: 先行実験は最終エポックだけを測る。ランナーは `eval_interval` ごとに測って
   `hard_acc`(最終)と `best_hard_acc`(最大)を両方返す。C3 は前者、C4 は後者を使う

## 検証済みのこと

- 7 レシピが `load_config` → `_build_model` を通り、層 0 の被覆率・`(b − a)` のずらし量・
  クラス受容野が先行実験の設計値と一致する(CIFAR P 0.9900 / SR2 1.0000 /
  R 0.3074、MNIST P1 の差 1 / P6 の差 27)
- `analysis/wiring_stats` が先行実験の 6 腕のクラス受容野を**丸め誤差なしで**再現する
  (`tests/l1_math/test_wiring_stats.py`)
- 7 レシピの 1 エポックスモーク(CPU torch 2.12.1、3 プロセス並列 ×
  8 スレッド。**判定には使えない配管確認**):

  | レシピ | hard_acc | best_hard_acc | 所要 |
  |---|---|---|---|
  | `cwo_mnist_48k_p0` | 0.8791 | 0.8791 | 479s |
  | `cwo_mnist_48k_p1` | 0.8929 | 0.8929 | 468s |
  | `cwo_mnist_48k_p6` | 0.8970 | 0.8970 | 392s |
  | `cwo_mnist_96k_p6` | 0.8956 | 0.8956 | 918s |
  | `cwo_cifar10_48k_p` | 0.3917 | 0.3917 | 537s |
  | `cwo_cifar10_48k_sr2` | 0.3831 | 0.3908 | 515s |
  | `cwo_cifar10_48k_r` | 0.3290 | 0.3290 | 433s |

  1 エポックでも MNIST は P6 > P1 > P0(P1 − P0 = +1.38pt)、CIFAR-10 は
  受容野 30.7% の R だけが 6pt 以上低い。SR2 − P は 1 エポックでは −0.09pt で、
  原値の +1.2pt は 8 エポックまで回さないと出ない(スモークで判定しないこと)
- `export_netlist` と `export_ir.py`: P6(48K / 96K)と SR2 のネットで
  `validate(target='hdl')` 警告 0、各層の indices が閉形式と**全数一致**
  (48K 96,000 添字 / 96K 192,000 添字 / SR2 の層 0 は引き伸ばし rot 9,216→8,000)。
  置換配線のネットでは「閉形式を持つ層が無い」と報告して非 0 で終わる
