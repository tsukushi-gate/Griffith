# attention梯子 結果ログ（sMNIST）

実行環境: ローカルMPS（クラウドGPU側のCUDAカーネルJITが間欠ハングしたため）。
論理層 `n_inputs=2`（DLGN標準2入力ゲート、MPSで高速）、`parameterization='warp'`。3シード(0,1,2)×10エポック。
主指標=**hard eval**（`model.eval()`でLutLayer離散化）。gate=前段基準に対する相対劣化。

## sMNIST 性能表（hard acc）

| 段 | 構成(attn/ffn) | hard acc (mean±std) | 基準比 | ゲート判定 | attn_entropy | 離散化ギャップ |
|---|---|---|---|---|---|---|
| L0-FP | fp / fp | 0.9778 ± 0.0017 | — (絶対天井) | — | 1.045 | 0.000 |
| L0b | fp / logic | 0.9700 ± 0.0006 | −0.8% vs L0-FP | 梯子の1変数基準 | 0.614 | 0.000 |
| L1 | logic_proj / logic | 0.8513 ± 0.0757 | −12.2% vs L0b | **PASS** (≤20%) | 3.223 | 0.051 |
| L2 | +スコアXNOR-popcount | 0.8520 ± 0.0294 | −0.1% vs L1 | **PASS** (≤10%) | 2.713 | 0.048 |

（注: L0b/L1/L2 は再走で再生成した3シードsummary.csvの値。MPSの実行間非決定性で初回本走とは±1%以内の差があるが、梯子の関係・全ゲート判定は不変。L3-Cのゲート基準は再生成L2=0.852。）

L1 per-seed(hard acc): seed0=0.7907(劣化18.7%), seed1=0.9009(7.4%), seed2=0.8753(10.0%)。全シード<20%＝結論(PASS)はシード非依存。
L2 per-seed(hard acc): seed0=0.8402(+1.8%), seed1=0.8729(−2.0%), seed2=0.8755(−2.3%)。全シード<10%。

## 所見（順次追記）

- **L0b（FFNのFP→論理ゲート化）は天井から−0.5%のみ** = 論理ゲートFFNは非常に有効な共通部品。gap≈0でクリーン離散化。
- **L1で attn_entropy が 3.22 ≈ ln(28)=3.33（ほぼ一様）に急騰**。射影をゲート化({0,1}射影+FP内積)するとattentionがほぼ識別しなくなる。精度0.856は主に論理FFN＋V集約が担い、attention自体は弱い。→ 研究主張「attention=学習されたMUX」の観点では、L1の段階では"MUXが選んでいない"状態。L2(XNORスコア)/L3(離散選択器)で識別が回復するかが焦点。
- L1は高シード分散(std0.058)。seed0が境界(18.7%)。L2以降で不安定が悪化する場合は5シードに増やす。
- **L2（スコアXNOR化）はL1比で精度微増(0.856→0.863)かつstd激減(0.058→0.020)** = XNOR-popcountスコアが学習を安定化。entropyも3.22→2.73でattentionが**識別性を回復**（一様から離脱）。→ 研究文脈で「スコアのBoolean化はattentionの識別を損なわず、むしろ改善」。L1で失われたattentionの選択性がL2で戻り始めた。

## L3（正規化のBoolean化）4案比較（sMNIST, softmax→離散選択器）

L2から正規化のみ変更（射影・XNORスコアはL2のまま）。ゲートは本命C案のみ（vs L2=0.8629, 劣化15%以内）。S/B/Aは対照・比較。

| 案 | norm | hard acc (mean±std) | attn_entropy | 離散化ギャップ | 固定回路化 | 判定 |
|---|---|---|---|---|---|---|
| **C(本命)** | mux_gumbel (argmax 1トークン=MUX) | 0.8605 ± 0.0161 | **0.000** | 0.022 | ✅ 可 | **PASS** vs L2(劣化0.3%<15%) |
| S(対照) | softmax_int (整数スコアにsoftmax) | 0.8606 ± 0.0309 | 2.724 | 0.042 | ❌ **不可**(FP exp/除算) | 対照 |
| B | topk_ste (k=4, ビット多数決) | **0.8738** ± 0.0277 | 1.386(=ln4) | 0.018 | ✅ 可 | 比較(最高acc) |
| A | threshold (学習θ) | 0.6260 ± 0.0862 | 2.875 | 0.068 | ✅ 可 | 比較(最弱・高分散) |

L3-C per-seed(hard acc): seed0=0.842(劣化2.4%), seed1=0.8682(−0.6%), seed2=0.8713(−1.0%)。全シード<15%＝結論(PASS)はシード非依存。

- **C(MUX)とS(softmax)は精度ほぼ同一(0.8605 vs 0.8606)だがentropyは 0 vs 2.72**。hard 1トークン選択がsoftmaxと同精度、かつ固定論理回路化できる(softmaxはexp/除算でFP不可)。**softmaxの「平坦化」は精度に不要**という直接証拠（→ 図1 `fig_l3_entropy.png`）。
- B(top-4)が最高acc(0.874)。少数選択+多数決がsMNISTで有効。A(閾値)は不安定(0.626±0.086)で最弱。
- **sMNIST留保**: 最終mean-pool＋強力な論理FFNのためattentionの寄与が限定的。C/S/Bが同水準なのはその表れ。**attentionが本質のLM(Shakespeare)が真の識別テスト**。

## L4（完全Boolean + collapse検証）— L3-C(MUX)を対象

- **離散化ギャップ(soft評価−hard評価)**: L3-C = 0.022 ± 0.019（per-seed 0.043/0.015/0.008）。soft(train mode)とhard(eval collapse)の精度差は平均2.2%と小さく、collapse耐性あり。
- **回路規模**（`results/circuit_cost.md`。射影は実測・attentionは解析近似）:
  - 射影LUT: **total_luts=768**（実測, `export_ir`→`estimate_resources`）, logic_depth=1, wire=1536
  - C案attention(全S=28位置, per層): 比較器 3,152 / MUX(2:1) 96,768 / 総ゲート ~297k / BRAM 3,584 bit/token / 論理深さ ~16
- **collapse後の残存FP箇所**（トランスパイラ要件定義）: 入力Linear(28→128)・位置Embedding・LayerNorm×5・分類ヘッド(128→10)は**設計上FP固定**（変数をattentionに限定するため）。score_scaleはC案で除去可(argmaxは単調変換不変)。**射影encoderはBinarize(比較器)＝sigmoidでなくFP残存なし**。
- **結論**: L3-C(MUX-attention)の1トークン生成経路は「LUT論理＋比較器＋MUX＋BRAM読出し」のみで構成でき（設計固定の埋め込み/正規化/ヘッドを除く）、離散化ギャップも2.2%と小さい。=**collapse可能な固定組合せ回路として表現可能**。残FPは設計上固定の周辺のみ。
