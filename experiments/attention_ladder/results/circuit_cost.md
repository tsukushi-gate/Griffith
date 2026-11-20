# attention梯子 L4: 回路コスト見積り比較(C/S/B/A)

自動生成: `experiments/attention_ladder/src/circuit_cost.py`

- **射影LUT数** = 既存API `export_ir → estimate_resources`（実測値）。
- **attention機構（比較器/MUX/加算器/BRAM/深さ）= 解析式に基づく推定（推測/近似）。実合成値ではない。**
- attention系はIR抽出対象外（BLACKBOX, ENV_NOTES §4）のため解析アダプタで別途カウント。

## 形状パラメータ（sMNIST, 全案共通）

n_layers=2, H(n_heads)=4, S=Sq=Sk=28, d_model=128, qk_bits=128→d_k=32/head, v_bits=128→d_v=32/head

## 表1: 射影LUTコスト（実測: export_ir→estimate_resources）

q/k/v射影の `LutLayer` 列のみをIR化。Binarizeは含まず（表2で比較器計上）。モデル総計 = 全 LadderAttention の q+k+v 合算。

| 案 | stage | 射影LUT総数 | 論理深さ | 配線数 | LUT内訳 | 抽出 |
|---|---|--:|--:|--:|---|---|
| C | L3-C | 768 | 1 | 1,536 | N2:768 | OK |
| S | L3-S | 768 | 1 | 1,536 | N2:768 | OK |
| B | L3-B | 768 | 1 | 1,536 | N2:768 | OK |
| A | L3-A | 768 | 1 | 1,536 | N2:768 | OK |

## 表2: attention機構の解析見積り（推測/近似・1 attention層・全S位置）

比較器 = Binarize入力二値化(共有 d_model) + 選択器比較器。加算器 = popcount + 選択器加算器。総ゲート = XNOR+比較器+MUX+加算器。

| 案 | 選択器 | XNOR | 比較器 | MUX(2:1) | 加算器 | 総ゲート | BRAM[bit] | 論理深さ |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| C | argmax(1トークン選択, Gumbel-ST) | 100,352 | 3,152 | 96,768 | 97,216 | 297,488 | 3,584 | 16 |
| S | softmax(FP, 固定論理化不可) | 100,352 | 128 | 0 | 97,216 | 197,696 | 3,584 | FP(不可) |
| B | top-k + ビット多数決 | 100,352 | 16,256 | 0 | 111,552 | 228,160 | 3,584 | 29 |
| A | 閾値ゲート s_i≥θ | 100,352 | 3,264 | 0 | 193,984 | 297,600 | 3,584 | 12 |

## 表3: attention機構（推測/近似・1 attention層・1トークン=1クエリ位置）

| 案 | XNOR | 比較器 | MUX(2:1) | 加算器 | 総ゲート |
|---|--:|--:|--:|--:|--:|
| C | 3,584 | 236 | 3,456 | 3,472 | 10,748 |
| S | 3,584 | 128 | 0 | 3,472 | 7,184 |
| B | 3,584 | 704 | 0 | 3,984 | 8,272 |
| A | 3,584 | 240 | 0 | 6,928 | 10,752 |

## 表4: モデル総計（射影LUT実測 + attention解析×n_layers）

attention機構は 1層値 × n_layers(=2) で全モデル換算。BRAMも×n_layers。射影LUTは既に全層合算済み。

| 案 | 射影LUT(実測) | XNOR | 比較器 | MUX | 加算器 | attention総ゲート | BRAM[bit] |
|---|--:|--:|--:|--:|--:|--:|--:|
| C | 768 | 200,704 | 6,304 | 193,536 | 194,432 | 594,976 | 7,168 |
| S | 768 | 200,704 | 256 | 0 | 194,432 | 395,392 | 7,168 |
| B | 768 | 200,704 | 32,512 | 0 | 223,104 | 456,320 | 7,168 |
| A | 768 | 200,704 | 6,528 | 0 | 387,968 | 595,200 | 7,168 |

### C案 MUXの別表現

- 2:1 MUX表現（表2/3で採用）: 全S 96,768 個 = H·d_v·(Sk-1)。
- log段選択表現: 全S 17,920 選択線 = H·d_v·ceil(log2 Sk)。

## 仮定・近似（すべて推測/近似）

- XNORスコア: per(head,query,key)に d_k XNOR + popcount加算木（d_k-1加算器, 深さ ceil(log2 d_k)）。総XNOR=H·Sq·Sk·d_k。
- C(argmax): 比較器=(Sk-1)/(head,query), MUX=d_v·(Sk-1)/(head,query)。深さ=argmax木+MUX木=2·ceil(log2 Sk)。
- B(top-k): 比較器≈k·Sk+d_v(多数決), 加算器=d_v·k。MUXなし（マスクAND選択）。深さは逐次top-kの近似。
- A(threshold): 比較器=Sk, 加算器=d_v·(Sk-1)。正規化の除算が純論理でなく残る（近似）。
- S(softmax): 固定論理回路化不可（FP exp/除算/温度）。比較器/MUXなし。
- Binarize比較器: 射影入力 d_model を共有入力とみなし d_model/層（q/k/v非共有なら 3·d_model/層）。
- BRAM: value格納 H·Sk·d_v bit/token（KVキャッシュV部）。
- 論理深さは二分木深さ ceil(log2 n) の近似。配線・面積・FF段は未計上。

## 残存FP箇所（トランスパイラ要件定義）

### C案（stage L3-C, norm=mux_gumbel）

| 箇所 | 個数 | なぜFPか | 整数化/比較器化で除去可能か |
|---|--:|---|---|
| 入力射影 nn.Linear(28→128) | 1 | 実数MNIST行→d_modelの線形写像。乗加算がFP | 設計上FP固定(変数をattentionに限定)。整数量子化/二値射影に置換は可 |
| 位置埋め込み nn.Embedding(seq_len→d_model) | 1 | 学習FPベクトルのテーブル参照 | 設計上FP固定。Gray-code等の固定{0,1}符号なら比較器/定数化で除去可 |
| LayerNorm ×5(pre-norm×4 + final×1) | 5 | 特徴横断の平均/分散正規化(√・除算)。特徴間依存でLUT折り畳み不能 | 設計上FP固定。固定小数点/近似正規化への置換は要検討(現状FP残存) |
| 分類ヘッド nn.Linear(128→10) | 1 | d_model→クラスロジットの線形写像。乗加算がFP | 設計上FP固定。GroupSum等の整数集約に置換は可 |
| 射影encoder(q/k/v入力の実数→{0,1}) | 6 | Binarize(閾値0の比較器)であり sigmoid ではない | FP残存なし: 比較器で二値化済み(閾値0のsign比較)。ただし入力自体はpre-norm(LayerNorm)出力のためLayerNormはFP |
| score_scale(学習FPスカラ, xnor_popcountスコア) | 2 | popcount整数スコアに乗じる学習FP係数 | 除去可: argmax/top-kは正の単調変換に不変。比較前にscaleを落とし整数popcountで比較できる |
| FFN LogicMLP encoder=sigmoid / decoder=groupsum_centered(参考) | 2 | sigmoid(実数→[0,1]確率)とgroupsum(整数和/τのFPスケール)が実数⇔論理の境界に残る。attention段の変数ではないが全体では残存 | encoderをBinarize/thermometer(比較器)へ、decoderを整数GroupSumへ置換で除去可 |
| (参考) 実在モジュール数: Linear=2, LayerNorm=5, Embedding=1 | - | モデル走査による実測(内訳確認用) | - |

### S案（stage L3-S, norm=softmax_int）

| 箇所 | 個数 | なぜFPか | 整数化/比較器化で除去可能か |
|---|--:|---|---|
| 入力射影 nn.Linear(28→128) | 1 | 実数MNIST行→d_modelの線形写像。乗加算がFP | 設計上FP固定(変数をattentionに限定)。整数量子化/二値射影に置換は可 |
| 位置埋め込み nn.Embedding(seq_len→d_model) | 1 | 学習FPベクトルのテーブル参照 | 設計上FP固定。Gray-code等の固定{0,1}符号なら比較器/定数化で除去可 |
| LayerNorm ×5(pre-norm×4 + final×1) | 5 | 特徴横断の平均/分散正規化(√・除算)。特徴間依存でLUT折り畳み不能 | 設計上FP固定。固定小数点/近似正規化への置換は要検討(現状FP残存) |
| 分類ヘッド nn.Linear(128→10) | 1 | d_model→クラスロジットの線形写像。乗加算がFP | 設計上FP固定。GroupSum等の整数集約に置換は可 |
| 射影encoder(q/k/v入力の実数→{0,1}) | 6 | Binarize(閾値0の比較器)であり sigmoid ではない | FP残存なし: 比較器で二値化済み(閾値0のsign比較)。ただし入力自体はpre-norm(LayerNorm)出力のためLayerNormはFP |
| score_scale(学習FPスカラ, xnor_popcountスコア) | 2 | popcount整数スコアに乗じる学習FP係数 | 除去不可: softmax(scale·s/temp)はscaleが分布形状に影響。FP温度として残る |
| S案 softmax(softmax_temp学習FP温度) | 2 | exp・除算・FP温度による正規化。固定論理回路化不可 | 除去不可(FP softmaxが本質)。C/B/A案なら選択器で置換され消える |
| FFN LogicMLP encoder=sigmoid / decoder=groupsum_centered(参考) | 2 | sigmoid(実数→[0,1]確率)とgroupsum(整数和/τのFPスケール)が実数⇔論理の境界に残る。attention段の変数ではないが全体では残存 | encoderをBinarize/thermometer(比較器)へ、decoderを整数GroupSumへ置換で除去可 |
| (参考) 実在モジュール数: Linear=2, LayerNorm=5, Embedding=1 | - | モデル走査による実測(内訳確認用) | - |

### B案（stage L3-B, norm=topk_ste）

| 箇所 | 個数 | なぜFPか | 整数化/比較器化で除去可能か |
|---|--:|---|---|
| 入力射影 nn.Linear(28→128) | 1 | 実数MNIST行→d_modelの線形写像。乗加算がFP | 設計上FP固定(変数をattentionに限定)。整数量子化/二値射影に置換は可 |
| 位置埋め込み nn.Embedding(seq_len→d_model) | 1 | 学習FPベクトルのテーブル参照 | 設計上FP固定。Gray-code等の固定{0,1}符号なら比較器/定数化で除去可 |
| LayerNorm ×5(pre-norm×4 + final×1) | 5 | 特徴横断の平均/分散正規化(√・除算)。特徴間依存でLUT折り畳み不能 | 設計上FP固定。固定小数点/近似正規化への置換は要検討(現状FP残存) |
| 分類ヘッド nn.Linear(128→10) | 1 | d_model→クラスロジットの線形写像。乗加算がFP | 設計上FP固定。GroupSum等の整数集約に置換は可 |
| 射影encoder(q/k/v入力の実数→{0,1}) | 6 | Binarize(閾値0の比較器)であり sigmoid ではない | FP残存なし: 比較器で二値化済み(閾値0のsign比較)。ただし入力自体はpre-norm(LayerNorm)出力のためLayerNormはFP |
| score_scale(学習FPスカラ, xnor_popcountスコア) | 2 | popcount整数スコアに乗じる学習FP係数 | 除去可: argmax/top-kは正の単調変換に不変。比較前にscaleを落とし整数popcountで比較できる |
| FFN LogicMLP encoder=sigmoid / decoder=groupsum_centered(参考) | 2 | sigmoid(実数→[0,1]確率)とgroupsum(整数和/τのFPスケール)が実数⇔論理の境界に残る。attention段の変数ではないが全体では残存 | encoderをBinarize/thermometer(比較器)へ、decoderを整数GroupSumへ置換で除去可 |
| (参考) 実在モジュール数: Linear=2, LayerNorm=5, Embedding=1 | - | モデル走査による実測(内訳確認用) | - |

### A案（stage L3-A, norm=threshold）

| 箇所 | 個数 | なぜFPか | 整数化/比較器化で除去可能か |
|---|--:|---|---|
| 入力射影 nn.Linear(28→128) | 1 | 実数MNIST行→d_modelの線形写像。乗加算がFP | 設計上FP固定(変数をattentionに限定)。整数量子化/二値射影に置換は可 |
| 位置埋め込み nn.Embedding(seq_len→d_model) | 1 | 学習FPベクトルのテーブル参照 | 設計上FP固定。Gray-code等の固定{0,1}符号なら比較器/定数化で除去可 |
| LayerNorm ×5(pre-norm×4 + final×1) | 5 | 特徴横断の平均/分散正規化(√・除算)。特徴間依存でLUT折り畳み不能 | 設計上FP固定。固定小数点/近似正規化への置換は要検討(現状FP残存) |
| 分類ヘッド nn.Linear(128→10) | 1 | d_model→クラスロジットの線形写像。乗加算がFP | 設計上FP固定。GroupSum等の整数集約に置換は可 |
| 射影encoder(q/k/v入力の実数→{0,1}) | 6 | Binarize(閾値0の比較器)であり sigmoid ではない | FP残存なし: 比較器で二値化済み(閾値0のsign比較)。ただし入力自体はpre-norm(LayerNorm)出力のためLayerNormはFP |
| score_scale(学習FPスカラ, xnor_popcountスコア) | 2 | popcount整数スコアに乗じる学習FP係数 | 除去可: s·scale≥θ ⇔ s≥θ/scale。scaleを閾値θへ吸収し整数化できる |
| A案 正規化の除算(加重和 / 選択数) | 2 | 選択トークン数での割り算。純論理でない | 固定小数点除算/逆数LUT/非正規化(和のまま)への変更で除去可(近似) |
| FFN LogicMLP encoder=sigmoid / decoder=groupsum_centered(参考) | 2 | sigmoid(実数→[0,1]確率)とgroupsum(整数和/τのFPスケール)が実数⇔論理の境界に残る。attention段の変数ではないが全体では残存 | encoderをBinarize/thermometer(比較器)へ、decoderを整数GroupSumへ置換で除去可 |
| (参考) 実在モジュール数: Linear=2, LayerNorm=5, Embedding=1 | - | モデル走査による実測(内訳確認用) | - |

