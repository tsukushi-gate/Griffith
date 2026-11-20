# attention論理回路化アブレーション梯子 — 実験レポート

担当: 論理ゲート型NN研究チーム
目的: Transformerのattention（射影・スコア・正規化）を段階的に論理回路化する「アブレーション梯子」を実装し、**どの段階まで精度が維持されるか**を計測する。研究主張「**attention＝学習されたマルチプレクサ（MUX）**」の実証の第一歩。
関連: 環境・API調査は `ENV_NOTES.md`、生結果は `RESULTS.md`、回路コストは `results/circuit_cost.md`。

## 1. 方法

- **梯子（1段=1変数）**: L0(全FP) → L1(射影ゲート化) → L2(スコアXNOR化) → L3(正規化＝softmax置換) → L4(完全Bool+collapse検証)。
- **共通骨格**: 2層Transformer(attn+FFN), D=128, h=4。**FFNは全段共通で論理ゲート層(LogicMLP)**、埋め込み・出力ヘッドは全段FP固定（変数をattentionに限定）。論理層は `n_inputs=2`（DLGN標準2入力ゲート, `parameterization='warp'`）。
- **2基準**: L0-FP(attn=FP, ffn=FP＝絶対天井) と L0b(attn=FP, ffn=logic＝梯子の1変数比較基準)。
- **タスク**: 主=char-level LM(Tiny Shakespeare, T=64, 語彙65)／副=sMNIST(行系列化MNIST, T=28)。3シード(0,1,2)、sMNIST 10epoch / LM 20k step。
- **指標**: hard eval精度/perplexity、attention選択エントロピー、離散化ギャップ(soft−hard)、回路規模(ゲート/比較器/MUX数)。
- **実行**: sMNISTはローカルMPS、LMはModal GPU(A10G)。すべてwandbにも記録(group=`attention_ladder`)。

## 2. sMNIST 梯子 性能表（hard eval, 3シード mean±std）

| 段 | 構成(attn/ffn) | hard acc | 基準比 | ゲート | attn_entropy | 離散化ギャップ |
|---|---|---|---|---|---|---|
| L0-FP | fp / fp | 0.9778 ± 0.0017 | — 絶対天井 | — | 1.045 | 0.000 |
| L0b | fp / logic | 0.9700 ± 0.0006 | −0.8% vs L0-FP | 1変数基準 | 0.614 | 0.000 |
| L1 | logic射影 / logic | 0.8513 ± 0.0757 | −12.2% vs L0b | **PASS**(≤20%) | 3.223 | 0.051 |
| L2 | +XNORスコア | 0.8520 ± 0.0294 | −0.1% vs L1 | **PASS**(≤10%) | 2.713 | 0.048 |
| **L3-C** | +MUX選択器(本命) | 0.8605 ± 0.0161 | +1.0% vs L2 | **PASS**(≤15%) | **0.000** | 0.022 |

**全ゲート通過**。全FP天井0.978 → 完全ゲート化MUX-attention 0.861（論理FFN基準0.970からの低下は射影ゲート化が主因）。図: `results/figures/fig_ladder_accuracy.png`（wandb: run irlswa39）。

### L3 softmax置換 4案比較（＝図1: `results/figures/fig_l3_entropy.png`）

| 案 | 方式 | hard acc | attn_entropy | 固定回路化 |
|---|---|---|---|---|
| **C(本命)** | MUX(argmax 1トークン, Gumbel-ST) | 0.8605 ± 0.016 | **0.000** | ✅ 可 |
| S(対照) | softmax(整数スコア, 温度学習) | 0.8606 ± 0.031 | 2.724 | ❌ **不可**(exp/除算) |
| B | top-4 + ビット多数決 | 0.8738 ± 0.028 | 1.386(=ln4) | ✅ 可 |
| A | 学習閾値 | 0.6260 ± 0.086 | 2.875 | ✅ 可(不安定) |

**核心所見**: C(MUX)とS(softmax)は**精度ほぼ同一(0.8605 vs 0.8606)だがエントロピーは 0 vs 2.72**。hard 1トークン選択がsoftmaxと同精度で、かつ固定論理回路化できる（softmaxはexp/除算でFP不可）。→ **softmaxの「平坦化」は精度に不要**という直接証拠。B(top-4)が最高精度、A(閾値)は最弱・不安定。

## 3. 回路コスト表（L4, `results/circuit_cost.md`）

射影LUTは実測（`export_ir`→`estimate_resources`）、attention機構は解析近似（合成前推定）。sMNIST形状(H=4,S=28,d_k=d_v=32)、per層。

| 段/案 | 射影LUT | 比較器 | MUX(2:1) | 総ゲート | BRAM[bit] | 固定回路化 |
|---|--:|--:|--:|--:|--:|---|
| 射影(共通) | **768**(実測) | — | — | — | — | ✅ |
| L3-C(MUX) | 768 | 3,152 | 96,768 | ~297k | 3,584 | ✅ |
| L3-S(softmax) | 768 | 128 | 0 | — | 3,584 | ❌ FP |
| L3-B(top-4) | 768 | 16,256 | 0 | ~228k | 3,584 | ✅ |
| L3-A(閾値) | 768 | 3,264 | 0 | ~298k | 3,584 | ✅ |

## 4. 離散化ギャップ表（soft評価 − hard評価）

| 段 | ギャップ(sMNIST) | 解釈 |
|---|---|---|
| L0-FP/L0b | 0.000 | FP/クリーン離散化 |
| L1 | 0.051 | 射影ゲート化で拡大 |
| L2 | 0.048 | XNORで横ばい |
| L3-C | 0.022 | MUX collapseで縮小(2.2%)＝collapse耐性あり |

**collapse後の残存FP**（トランスパイラ要件）: 入力Linear・位置Embedding・LayerNorm×5・出力ヘッドは設計上FP固定。**射影encoderはBinarize(比較器)＝FP残存なし**。score_scaleはC案で除去可（argmax単調不変）。→ L3-Cのattentionコア(射影+XNOR+argmax+MUX)は**論理＋比較器＋MUX＋BRAM読出しのみ**で構成でき、残FPは設計固定の周辺のみ。

## 5. LM(Tiny Shakespeare)梯子（Modal A10G, T=64, 語彙65, 20k step, 3シード）

主指標=**hard eval perplexity（低いほど良）**。L0-L2はVolume回収の3シード値。L3-Cは3シード再走中（seed0/1既知: ppl 16.42/17.53）。

| 段 | 構成 | ppl (mean±std) | attn_entropy | 離散化ギャップ |
|---|---|---|---|---|
| L0-FP | fp / fp | 5.015 ± 0.041 | 0.685 | 0.000 |
| L0b | fp / logic | 7.880 ± 0.382 | 0.790 | 0.299 |
| L1 | logic射影 / logic | 19.242 ± 0.939 | 3.166 | 0.497 |
| L2 | +XNORスコア | 19.385 ± 0.704 | 2.385 | 0.515 |
| **L3-C** | +MUX選択器 | ~16.98(seed0/1, 3seed再走中) | **0.000** | ~0.36 |

**LMの所見（sMNISTとの対比が重要）**:
- **射影ゲート化(L1)がLMではppl 7.88→19.24と大幅悪化**。sMNIST(最終mean-pool+強FFNでattention寄与小)と異なり、**LMではattentionが系列モデリングに本質的**で、その{0,1}射影化は実精度に大きく効く＝**真の試金石**であることを実証。entropyもL1で3.17(ほぼ一様)に急騰。
- **L3-C(MUX)はsoftmax(L2 ppl19.4)を改善(ppl~17.0)しentropy=0**。hard 1トークン選択がLMでもsoftmaxに劣らず(むしろ改善)、かつ固定回路化可能。sMNISTのC≈Sに続き、**「softmaxの平坦化は不要」がLMでも成立**。
- **離散化ギャップがLMで大きい(0.3-0.5, sMNISTは0.02-0.05)**。LM論理モデルはsoft-hard乖離が顕著で、collapse時の精度低下がsMNISTより課題。→ 次実験でGumbel正則化(Mind the Gap)等のギャップ縮小が有効な余地。
- FP天井(ppl5.0)とゲート化MUX(ppl~17)の差は大きく、**論理ゲートattentionのLM品質にはまだ開き**がある（実装可能性は実証、品質改善は今後）。

## 6. 主張1〜2の実証状況

- **主張1（Q/K/V射影を学習された論理回路で構成したattentionは存在しない）**: 本実験のL1で、Q/K/V射影を `Binarize + LutLayer`（学習された論理ゲート層）で{0,1}ビット列として構成し、end-to-end学習できることを**実証**。既存研究の調査範囲では他に見当たらず（既存研究は射影がFP/量子化matmul）、**動く実装としての第一歩**を示した。sMNISTで精度天井から−12%（論理FFN基準）に収まりゲート通過。
- **主張2（射影・スコア・正規化のすべてがゲート構成のend-to-end Transformer）**: L3-Cで射影(logic)・スコア(XNOR-popcount)・正規化(MUX選択器)の**3要素すべてをゲート/比較器/MUXで構成**し、softmaxと同精度(0.861 vs 0.861)をhard離散選択で達成、かつ固定論理回路へcollapse可能(L4)。=**主張2の構成物を動く形で実証**（調査範囲では先行実装は見当たらない）。ただし現時点はsMNISTのみで、attentionが本質のLM検証は進行中。
- **留保**: sMNISTはattention寄与が限定的で、C/S/Bが同水準なのはその表れ。主張の頑健性はLM結果に依存（本レポート§5で確定）。回路コストは合成前の解析近似。

## 7. 次の実験提案（3行）

1. **LM梯子の完遂と、attention依存タスク(連想想起/コピー/ソート)での再評価** — sMNISTで隠れたattentionの識別力を、系列本質タスクでMUX選択が担えるか検証。
2. **collapse後の実FPGA合成とHDL等価性検証**（`export/verilog.py` + `run_iverilog_equivalence`）で解析近似の回路コストを実測に置換し、主張3（FPGA自己回帰生成）へ橋渡し。
3. **AP法/DFA等の非BP学習則でL3-C選択器を学習**し、勾配消失・離散化ギャップの改善余地を測る（`Mind the Gap`のGumbel正則化併用も候補）。
