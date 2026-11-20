# ENV_NOTES.md — attention梯子実装のための環境・API調査（フェーズ0）

対象リポジトリ: griffith-logic（自作PyTorchライブラリ）
方針: **コア実装（`griffith_logic/`）は一切変更しない。不足APIは `experiments/attention_ladder/` 配下にアダプタとして実装する。**
フォールバック判定: **pip `difflogic`（Petersen版）は不要**。論理ゲート層・XNOR-popcount attention・Gray-code位置符号・collapse・回路見積りのすべてが自作ライブラリに存在する（下記）。difflogicは使用しない。

---

## 0. 一言サマリ（梯子への含意）

- **L1（射影のゲート化）とL2（スコアのXNOR化）は既存 `blocks/attention.py::LogicAttention` がほぼ実装済み**。ただし「1段=1変数」を厳守するため、射影・スコア・正規化を独立フラグで切り替えられる薄いアダプタ `LadderAttention` を experiments 配下に自作し、内部で `LutLayer`/`DWNLayer`/`GroupSum` を流用する。
- **L0（全FP基準）に流用できる既製FPブロックは無い**（`LogicTransformerBlock` は全経路が論理ゲート版）。`torch.nn.MultiheadAttention`+`nn.LayerNorm`+FP FFN で自作する。
- **Gray-code位置符号は既存**（`blocks/positional.py::position_codes(..., code='gray')`）。自作不要。
- **sMNISTは薄いreshapeアダプタ**、**Tiny Shakespeare（char-LM）はデータ・語彙とも完全自作**が必要。
- **回路見積り**: `estimate_resources` は LUT数・論理深さ・配線数は出すが**比較器数・MUX数は出さない**。かつ **attention系ブロックはIR抽出対象外（BLACKBOX）**。→ L4は「射影のLutLayer列のみをIR化してestimate」＋「比較器・MUX数は解析式でアダプタ算出」の合わせ技。

---

## 1. 論理ゲート/LUT層（射影・FFNのゲート化に使用）

すべて `griffith_logic/` 配下。値域が最重要（{0,1}か実数か）。

### `layers/lut.py::LutLayer`（中核。5パラメータ化の本体）
```python
LutLayer(in_dim, out_dim, n_inputs=2, device='cpu', grad_factor=1.0,
         connections='random',            # 'random' | 'unique'
         parameterization='softmax',      # 'softmax'|'independent'|'iwp'|'raw'|'warp'  ← 「5パラメータ化」
         selection='soft')                # 'soft' | 'hard_st'（argmax forward, selection gap=0, backward=softmax(w/tau_b)）
```
- 入力 `(B, in_dim)`、**値域 [0,1] の float**。`n_inputs ∈ {2,4,6}`。
- 出力 `(B, out_dim)`：**soft（`training=True`）=実数[0,1]付近 / hard（`eval()`）={0,1} 離散**（`raw` のみ hard でも連続値なのでビット列用途に不可）。
- collapse専用メソッドは無い → **`model.eval()`/`.train()` で soft↔hard が切り替わる**。`selection='hard_st'` は訓練中もargmax forward（CAGE方式、`tau_b` は外部スケジューラ更新）。

### `layers/dwn.py::DWNLayer`（{0,1}ビット列を直接生成。L1射影の第一候補）
```python
DWNLayer(in_bits, n_luts, n_inputs=6, mapping='random',  # 'random'|'learnable'
         efd='paper', alpha=None, beta=None, ste=True, clamp_luts=True, lm_tau=0.001, device='cpu')
```
- 入力 `(B, in_bits)` **{0,1}**（内部で `>0` アドレス化）。出力 `(B, n_luts)` = **正確に `n_luts` ビットの {0,1}**（`ste=True`）。
- → **`DWNLayer(in_bits, n_luts=128, mapping='learnable', ste=True)` で d_k=128 ビットの {0,1} 射影が一発で得られる**。深さは縦積み（2段目以降 `mapping='random'`）。
- 併記: `LearnableMapping(in_features, out_features, tau=0.001)`（hard選択, backward=softmax(W/τ)）。

### `layers/logic_linear.py::LogicLinear / LogicMLP`（nn.Linear drop-in。全段共通FFNに使用）
```python
LogicLinear(in_features, out_features, *, spec: LogicSpec | None = None)
LogicMLP(in_features, out_features=None, *, spec: LogicSpec | None = None)  # out省略時 in と同じ
```
- 入力・出力とも**実数** `(..., F)`。`(B,S,D)` を内部で `reshape(-1,D)` 処理。内部で encode→LUT×depth→decode。
- 制御は `LogicSpec`（`hybrid/policy.py`）: `parameterization`（5方式+`'dwn'`）, `selection`, `depth=2`, `neurons_per_output=8`, `encoder∈{sigmoid,binarize_ste,identity,thermometer}`, `decoder∈{groupsum,groupsum_centered,layernorm,identity}`, `encoder_bits=8`, `grad_factor` 等。`resolve_width(out)=width or out*neurons_per_output`。

### 出力集約
```python
group_sum.py::GroupSum(k, tau=1.0)          # (...,D)→(...,k)。D%k==0。reshape(k, D/k).sum(-1)/tau（多数決でなく和）
spatial_group_sum.py::SpatialGroupSum(n_outputs, scale=1.0)  # (B,C,H,W)→(B,n_outputs,H,W)
```

### STE / 二値化 / その他
```python
binarize.py::Binarize(threshold=0.5, hardtanh_min=0.0, hardtanh_max=1.0)  # forward=(x>th), backward=HardTanh STE（訓練時のみ）
core/ste.py::GradFactor.apply(x, f)          # 恒等forward, backward=grad*f（勾配スケールSTE）
differentiable_lut_block.py::DifferentiableLutBlock(in_dim, out_dim, n_inputs=6, ...)  # BinaryBrain LUT6, 出力{0,1}
```
- `core/functions.py`: `LutFunction`(softmax), `LutFunctionHardST`(tau_b), `LutFunctionIndependent`(sigmoid), `LutFunctionRaw`, `LutFunctionIWP`, `LutFunctionWARP`(tau, gumbel)。L2/L3のSTE設計の参考（特に `LutFunctionHardST` = argmax forward + softmax backward）。
- 後方互換シム: `lut_layer.py`・`connections.py` は `core/`・`layers/` を再export。**新規コードは `layers/`・`core/` を直接import**。

---

## 2. 学習則切替（AP法フック用）

### `training/spec.py`
```python
@dataclass
class LearningSpec:
    rule='bp'            # 'bp'|'ap'|'dfa'|'eggroll'
    lr=None; update_mode=1; feedback_init='gaussian'; block_upstream=True; eggroll=<EGGROLLConfig>
def with_learning(module, spec) -> module   # module._learning_spec=spec を付与（forward不変）
def get_learning_spec(module) -> LearningSpec | None
```
最小例: `with_learning(LutLayer(...), LearningSpec(rule='ap', lr=0.1, update_mode=1))`

### `training/trainer.py::HybridTrainer`
```python
HybridTrainer(model, optimizer=None, loss_fn=F.cross_entropy, clamp_raw=True)
  .train_step(x, target) -> dict{'loss','accuracy',...}   # 1step=1バッチ
  .close()                                                # DFA後のフック解除
```
- **AP注釈が1つでもあると HybridTrainer は全モデルを APTrainer に委譲**（他ルール混在は `NotImplementedError`）。
- `training/runner.py::run_training(cfg)` は設定駆動だが **bpのみ**（非bpは即 `NotImplementedError`）。→ AP法は HybridTrainer 直叩き必須。

### ★ L3-C の AP法フック（ストレッチ）の現実解
公開APIには「サブモジュールだけAP」は**無い**。唯一の手：**選択器を独立 `nn.Sequential`（`LutLayer` 列+`GroupSum`, `parameterization='independent'`）として切り出し、`APTrainer(selector, lr, update_mode)` を自前instance化して選択器だけ `train_step`、本体は別optimizerでbp更新する二経路構成**。動かなければ「何がAPI的に不足か」を記録するに留める。

---

## 3. collapse（soft→hard離散化）

- **soft/hard切替 = `model.train()` / `model.eval()`**（`LutLayer.forward` が `self.training` で分岐）。規範実装は `runner._hard_eval`：`model.eval()` + **入力 `x.round()`** で hard 評価、後で `model.train(mode=was)` 復帰。
- **恒久collapse（真理値表化）= `export/discretize.py`**：`truth_tables_from_lut_layer(layer)` / `_from_dwn` / `_from_conv` / `_from_composition(layer, post)`。方式は「レイヤをdeepcopy→接続恒等化→eval()で全2^Nパターン流す」＝eval離散化コードパス再利用。
- **モデル全体のcollapse = `export/extract.py::export_ir(...)`**（下記4）。

---

## 4. IR化・回路規模見積り（L4用）

### `export/extract.py::export_ir`
```python
export_ir(model, input_width=None, *, input_shape=None, name='model', metadata=None, native_conv=False) -> IRGraph
```
- 認識対象: `LutLayer`,`GroupSum`,`Binarize`,`DWNLayer`,`LearnableMapping`, conv系(`LutConv2d`,`OrPooling2d`,`SpatialGroupSum`,`DifferentiableLutBlock`,`UpSample2d`,`BinaryBrainConv2d`)。
- **★attention系はIR化不可**：`LogicAttention`/`LogicTransformerBlock`/`LogicLayerNorm`/`ConditionalSum` は `BLACKBOX_TYPES`、`LLViT`/`DLGN*Attention*`/`LogicMLP` は `MODEL_TYPES`（対象外）。softmax・特徴横断正規化を含み真理値表化不能。未登録型を渡すと `NotImplementedError`。
- → **L4では射影の `LutLayer`/`DWNLayer` 列だけを `nn.Sequential` に取り出して `export_ir`→`estimate_resources`**。attention本体（比較器・MUX・softmax相当）は下の解析アダプタで別途カウント。

### `export/estimate.py::estimate_resources`
```python
estimate_resources(graph: IRGraph) -> {
  'total_luts', 'lut_counts'{n_inputs:count}, 'logic_depth', 'wire_count',
  'group_sums', 'node_counts'{type:count}, 'lowered'
}
```
- **入力はIRGraphのみ（torch直渡し不可）**。**比較器数・MUX数・面積は返さない** → アダプタで解析算出（§8）。
- 検証: `export/verify.py::run_iverilog_equivalence(graph, x, workdir)`（要iverilog）/ `ir/simulate.py::simulate(graph)`。使用例は `tests/test_export_estimate.py`。

### `hybrid/replace.py`（FP層→論理層置換。FFN差込に流用可）
```python
replace_mlp_with_lut(model, *, spec=None, mlp_class=None, mlp_name_pattern='*mlp*', inplace=False)
convert_linear_to_logic(model, *, spec=None, include=None, exclude=None, inplace=False)
split_param_groups(model, *, logic_lr=None, ann_lr=None)  # LUT系/ANN系でoptimizer param group分割
```

---

## 5. 既存 attention / Transformer 骨格

### ★ `blocks/attention.py::LogicAttention`（L1+L2を既に実装。数式流用の主対象）
```python
LogicAttention(d_model, n_heads=1, mode='soft', similarity='xnor',
               *, qk_sizes=None, v_sizes=None, share_qk=True, tau=1.0,
               n_inputs=2, parameterization='softmax', connections='random', grad_factor=1.0)
  .forward(q_bits, k_bits, v_bits, hard=None) -> (B, Sq, d_v)
  .last_attention -> (B, H, Sq, Sk)   # ← エントロピー計測にそのまま使える
```
- 射影 = LutStack（論理ゲート層）、スコア = 行列積化XNOR-popcount `qh@kh^T + (1-qh)@(1-kh)^T`。
- `hard=True` = argmax one-hot + softmax勾配STE（`attn = onehot + (attn - attn.detach())`）。**訓練時の重みは依然 `F.softmax(scores/tau)`（softmax置換は未実装）**。
- `SIMILARITIES=('xnor',)` のみ → **L1で必要な「FP内積スコア」は非対応 → アダプタで追加**。

### `models/attention/boolean_attention.py::DLGNBooleanAttention`＋`BooleanAttentionConfig`
- 単一ヘッドの連想記憶用。射影=LutStack、スコア=XNOR-popcount、hard時 argmax one-hot。`LogicAttention` の元ネタ。梯子では `LogicAttention` を優先。

### `blocks/transformer.py::LogicTransformerBlock`（全論理版。L0基準には不可）
```python
LogicTransformerBlock(d_model, n_heads=1, mlp_ratio=2, norm='layernorm', merge='xor',
                      *, attn_mode='soft', n_inputs=2, parameterization='softmax', connections='random')
  .forward(x) -> (B,S,D)   # h=norm1(merge(x,attn)); y=norm2(merge(h,mlp))
```
- FFN = `LogicMLP(d_model, spec=LogicSpec(encoder='identity', decoder='groupsum', width=d_model*mlp_ratio, ...))`。**全経路が[0,1]論理域なのでFP基準L0にはならない**。構成順序（attn残差→FFN残差）のみ設計参考。
- `blocks/norm.py`: `LogicLayerNorm(width)`（2x-1→LayerNorm→sigmoid）, `LogicBatchNorm(width, momentum, bn_gamma=0.3, bn_beta=0.5)`。

---

## 6. 位置エンコーディング（Gray-code 既存）

### `blocks/positional.py`
```python
POSITIONAL_CODES = ('binary', 'gray', 'thermometer')
position_codes(seq_len, width, code) -> (seq_len, width) float32 {0,1}
  # 'gray': g = p ^ (p>>1); bit j = (g>>j)&1  ← 隣接位置が1bit差（まさに要件のGray-code）
LogicPositionalEncoding(seq_len, width, code='binary')
  .forward(x) -> (B, in + seq_len*width)   # 末尾連結。学習パラメータなし（buffer）。IRはconstant+concatで展開可
```
- **L2以降の「Gray-Code位置をQ/Kビット列に連結」は `position_codes(T, width, 'gray')` を Q/K ビットに `torch.cat` するだけ**。自作不要。
- L0/L1 の「通常の学習位置埋め込み」は `nn.Embedding(T, D)` を自作。
- 参考: `models/attention/positional_attention.py`・`pointer_attention.py` は位置キーが**素の2進（MSBファースト）**でGrayではない。

---

## 7. データセット

### sMNIST（副タスク）— `datasets/vision.py::load_mnist` + 薄いアダプタ
```python
load_mnist(root=None, binarize_threshold=0.5, download=True) -> (train_set, test_set)  # 784入力/10クラス, 二値化済み
```
- **行単位系列化（T=28×28bit）ラッパは無し** → `(B,784)→(B,28,28)` reshape の薄いアダプタを `src/data_smnist.py` に自作。各行28bitがステップ入力、系列長T=28。

### Tiny Shakespeare（主タスク, char-level LM）— **完全自作**
- 全ツリーに `shakespeare`/`char-level`/`nanogpt` ヒットゼロ。既存 `tokenizer.py` は WordTokenizer（単語レベル、流用不可）。
- → `src/data_shakespeare.py` を自作：input.txt をDL（無ければ同梱ミラー/生成）、char語彙（〜65）、T=64のスライディング窓、train/val分割。`uv run python -m ...` または `uv run python src/data_shakespeare.py` で一発再現。

---

## 8. 判明した「不足API」と experiments 配下に書くアダプタ設計

| # | 不足している機能 | 実装先（experiments/attention_ladder/src/） | 内容 |
|---|---|---|---|
| A1 | FP内積スコア（L1用。LogicAttentionは'xnor'のみ） | `ladder_attention.py` | `similarity='fp_dot'` を追加した梯子用attention。射影/スコア/正規化を独立フラグ化（1段=1変数） |
| A2 | softmax置換の選択器4案（L3: C/B/A/S） | `selectors.py` | C=MUX(Gumbel-softmax straight-through, τ:2.0→0.5指数減衰) / B=STE top-k+多数決 / A=学習閾値+温度 / S=整数スコアsoftmax（温度学習） |
| A3 | ヘッド多様性正則化 | `selectors.py` or `regularizers.py` | 同一クエリ位置で複数ヘッドが同一トークン選択する確率へのペナルティ（係数λスイープ） |
| A4 | 決定的タイブレーク＋因果マスク | `ladder_attention.py` | 同点は「クエリに近い位置優先」。因果マスクは選択前にANDで適用 |
| A5 | sMNIST行系列化 | `data_smnist.py` | `load_mnist`→`(B,28,28)` reshape |
| A6 | Tiny Shakespeare char-LM | `data_shakespeare.py` | DL・char語彙・T=64窓・train/val |
| A7 | L0 FP基準Transformer | `models_fp.py` | `nn.MultiheadAttention`+`nn.LayerNorm`+FP FFN の2層 |
| A8 | 回路コスト解析見積り（比較器・MUX数） | `circuit_cost.py` | 射影は `export_ir`→`estimate_resources`（LUT数）。attentionのargmax比較器数＝ヘッド×クエリ×(Sk-1)（log深さ）、MUX＝ヘッド×クエリ×d_v幅。L3各案の回路コスト比較表を生成 |
| A9 | 離散化ギャップ計測 | 学習ハーネス | soft評価(`train()`) と collapse後hard評価(`eval()`+round) の性能差 |
| A10 | attention選択エントロピー | metrics | `last_attention` (B,H,Sq,Sk) から行方向エントロピーを平均（S案の平坦化 vs C/B案の対比図＝図1候補） |

**流用する既存骨格**: `experiments/train_boolean_attention.py`（argparse・device検出・seed固定・`load_n`・`train_step`・soft/hard `evaluate`・checkpoint・tqdm/ETA）＋ `run_utils.py::RunDir`（config.json/metrics.jsonl/visualizations）＋ `env_utils.py::load_dotenv`。
**wandb**: `wandb_utils.init_wandb(run_name, config_dict, model=None, device='cpu', tags=None, group=None, job_type='train')`, `log_metrics(metrics, step=None)`, `log_images(key, images, captions=None, step=None)`, `finish()`。**画像出力は必ず log_images でwandbにもログ**（本リポジトリの規約）。

---

## 9. 梯子×API 対応表（実装の地図）

| 段 | 変数（1段1変数） | 射影 | スコア | 正規化 | 使用API / アダプタ |
|---|---|---|---|---|---|
| L0 | 基準 | FP linear | FP内積 | FP softmax | `nn.MultiheadAttention`（A7自作） |
| L1 | 射影をゲート化 | **DWNLayer/LutLayer→{0,1}128bit** | FP内積(bit列) | FP softmax | `DWNLayer(mapping='learnable',n_luts=128)` + A1 |
| L2 | スコアをBool化 | ゲート射影 | **popcount(XNOR), 学習スカラ1個, STE** | FP softmax | `LogicAttention`のXNOR式流用 + `LutFunctionHardST`型STE + A1 |
| L3 | 正規化をBool化 | ゲート射影 | XNORスコア | **C/B/A/S 選択器** | A2/A3/A4。Gray位置=`position_codes(...,‘gray’)` |
| L4 | 完全Bool+collapse | ↑最良案 | ↑ | ↑ | `model.eval()`/`export_ir`→`estimate_resources`（射影） + A8（比較器/MUX） + A9ギャップ |

**設計上の注意**:
- D=128, h=4 → 1ヘッド d_k=32bit。タスク記載「Q/Kは d_k=128ビット」は**射影出力の総ビット幅=d_model=128**（＝32bit×4head）と解釈。configで `qk_bits`（既定128）を可変にし、この解釈をREPORT/configに明記する。
- 埋め込み・LMヘッドは全段FP固定（変数をattentionに限定）。FFNは全段共通で `LogicMLP`（既にCNN等で実績のある論理ゲート層）を流用。
- L4で「浮動小数点が残る箇所」を列挙する要件 → 現設計では soft-softmax（L2まで）・学習スカラスケール・埋め込み/LMヘッドがFP。L3-C collapse後は選択器がargmax+MUXになりFP残存は「スケールスカラ（比較の単調変換なので整数化可）」「埋め込み/LMヘッド（設計上FP固定）」に限定される見込み。実測で確認しREPORTに列挙。

---

## 10. フォールバック判定（記録）

- **`difflogic`（Petersen版 pip）は使用しない**。理由：論理ゲート層（`LutLayer` の5パラメータ化）、XNOR-popcount attention（`LogicAttention`）、Gray-code位置符号（`position_codes`）、collapse（`model.eval()`＋`export_ir`）、回路見積り（`estimate_resources`）がすべて自作ライブラリに存在し、機能的に十分。
- 自作ライブラリのコア（`griffith_logic/`）は**読み取り専用**として扱い、梯子固有のロジックはすべて `experiments/attention_ladder/` 配下のアダプタに閉じる。
