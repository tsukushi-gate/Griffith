"""学習済みモデル → IR → Verilog エクスポートの例。

XORを学習し、IR(中間表現)JSONとVerilogを生成する。
iverilogがあればgolden modelとの一致検証も行う。

このライブラリの出口——「PyTorchで学習した重みを、GPUの要らない論理回路に変換する」
——を最短で通すスクリプト。要は3段階で、(1) 学習、(2) IRへ万能離散化、
(3) IRからVerilog生成。IRが真実源で、シミュレータもHDLもすべてIRから派生するため、
「PyTorchの出力 = IRの出力 = 実際の回路の出力」を機械的に照合できる。

実行: uv run python examples/04_export_hdl.py
期待結果: out_examples/xor.json と xor.v が生成され、照合ALLPASS
"""

# === 準備: 出口までに使う4つのAPI ===
# > このページで通す道は「学習 → IR → Verilog → 照合」の4段で、それぞれに対応する
# > API を最初に import しておく。用語は初出でここに定義しておく。
# >
# > - `export_ir` … 学習済みモデルを **IR(中間表現)** へ変換する。IR は
# >   「真理値表の集まり」として正規化された JSON で、**ハードウェア側の真実源**
# > - `validate` … IR が HDL 生成の制約(ビット幅・トポロジ順序など)を満たすか検査する
# > - `write_verilog` … IR から合成可能な Verilog(RTL)を書き出す
# > - `simulate` … IR を numpy の整数ビット演算で解釈実行する参照実装(**golden model**)
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

from griffith_logic import GroupSum, LutLayer
from griffith_logic.analysis.circuit_viz import GATE_NAMES, plot_circuit
from griffith_logic.export import export_ir
from griffith_logic.export.discretize import truth_tables_from_lut_layer
from griffith_logic.export.verify import iverilog_available, run_iverilog_equivalence
from griffith_logic.export.verilog import write_verilog
from griffith_logic.ir import save_ir, simulate, validate

# === 学習: 焼く対象のXOR回路を作る ===
# > エクスポートの手順を見せるのが主題なので、対象は最小の XOR に絞る
# > ([付録: XOR](/tutorials/xor/) とまったく同じ構成の 32 ゲート)。
# > 学習ループも普通の PyTorch なので、ここは黙って 500 ステップ回すだけでよい。
torch.manual_seed(0)
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 0])

model = nn.Sequential(
    LutLayer(2, 16, n_inputs=2),   # 入力2ビット→16ゲート(各ゲートは2入力論理関数の混合)
    LutLayer(16, 16, n_inputs=2),  # 2段目。ゲートの合成でXORの非線形性を作る
    GroupSum(k=2, tau=2.0),        # 16ゲートの出力を2クラスへ集計。tauはロジットの温度
)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(500):
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()

# === エクスポート前に、これから焼く回路を見る ===
# --- 図1: 確定したゲートの網 ---
# > `model.eval()` を呼んだ時点で、各ゲートは16種の論理関数の混合ではなく
# > **1つに確定したゲート**になっている。その状態を回路図として描く。
# > 緑のノードが確定した論理ゲート、橙が GroupSum(ここから先は実数)。
# eval() 後のモデルは、もう「16種の論理関数の混合」ではなく1つに確定したゲートの網。
# plot_circuit(resolve=True) が読むゲート名は export_ir が Verilog を吐くときに使う
# 真理値表と同じ経路(export/discretize)から来るので、**この図のゲート = 生成される回路のゲート**。
model.eval()
fig, ax = plt.subplots(figsize=(9, 5))
plot_circuit(model, resolve=True, ax=ax,
             title='what will be exported: resolved gates (green) + GroupSum (orange)')
fig.tight_layout()
plt.show()

# > **読み取り**: 各ノードに `XOR` / `NAND` / `A>=B` のような**具体的な論理関数の名前**が
# > 入っている。この名前は `export_ir` が Verilog を吐くときに使う真理値表と
# > 同じ経路(`export/discretize`)から来ているので、
# > **この図のゲート = 生成される回路のゲート**である。
# --- 回路の「部品表」を数える ---
# > 同じ回路を別の角度から見る。各ゲートの真理値表(2入力なので4ビット=16通り)を
# > ゲートIDとみなし、16種の論理関数それぞれが何個使われたかを数える。
# > `truth_tables_from_lut_layer` は `export_ir` が内部で使うのと同じ離散化経路なので、
# > ここで数えた内訳は**生成される Verilog の中身の内訳と一致する**。
# 各ゲートの真理値表(4ビット=16通り)をそのままゲートIDとみなし、種類ごとに数える。
# truth_tables_from_lut_layer は export_ir が内部で使うのと同じ離散化経路なので、
# ここで数えた内訳は生成されるVerilogの中身の内訳と一致する。
# 見どころは2つ。
#   (1) 16種のうち一部しか使われず、XOR / XNOR に大きく偏る。誰も「XORを使え」とは
#       教えていないのに、勾配だけでタスクに必要な論理関数が選ばれている。
#   (2) 一方で定数(FALSE)や入力の素通し(A / !B)も残る。これらは実質的に配線か
#       無効ノードで、人間なら書かない冗長さ。学習は「32ゲート使い切れ」とは
#       言われていないので、余ったゲートが何もしない形に落ち着いた結果である
#       (合成ツールが後段で潰せる余地でもある)。
gate_ids = [int(t) for layer in model if isinstance(layer, LutLayer)
            for t in truth_tables_from_lut_layer(layer)]
counts = Counter(gate_ids)
print('ゲート種別の内訳: ' + ', '.join(
    f'{GATE_NAMES[g]}×{n}' for g, n in counts.most_common()))
print(f'ゲート総数 {len(gate_ids)} / 使われた種類 {len(counts)} (2入力論理関数は全16種)')

# --- 図2: どの論理関数が何個使われたか ---
# > 横軸に16種すべてを並べる(0個のものも空欄として残し、何が**使われなかったか**も見せる)。
# 16種すべてを横軸に並べる(0個のものも空欄として残し、「何が使われなかったか」を見せる)。
order = sorted(GATE_NAMES)
labels = [GATE_NAMES[g] for g in order]
heights = [counts.get(g, 0) for g in order]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(labels, heights, color='#2e9e4f')
ax.set_xlabel('2-input logic function (all 16)')
ax.set_ylabel('number of gates in the exported circuit')
ax.set_title('gate mix of the learned circuit')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
plt.show()

# > **読み取り**: 16種のうち一部しか使われず、XOR / XNOR に大きく偏っている ──
# > 誰も「XOR を使え」とは教えていないのに、勾配だけでタスクに必要な論理関数が選ばれた。
# > 一方で定数(`FALSE`)や入力の素通し(`A` / `!B`)も残る。これらは実質的に配線か
# > 無効ノードで、人間なら書かない冗長さである。学習は「32ゲートを使い切れ」とは
# > 言われていないので、余ったゲートが何もしない形に落ち着いた結果であり、
# > 後段の論理合成ツールが削れる余地でもある。
# === IRへ抽出する ===
# > **IR が真実源である**というのが本ライブラリの設計の要である。Verilog も VHDL も
# > シミュレータも、すべて IR から派生する ── だから「PyTorch の出力 = IR の出力 =
# > 実際の回路の出力」を機械的に照合できる。
# >
# > 手順で外せないのは2点。
# >
# > - **`eval()` にしてから抽出する**。`train()` のままだと連続混合の状態を見ることになり、
# >   回路として確定していない
# > - **`validate(ir, target='hdl')` を通す**。ビット幅・トポロジ順序・未対応ノードの不在を
# >   ここで検査しておけば、Verilog を書き出してから合成ツールで初めて気づく事故を防げる
# 学習済みモデル → IR(全パラメータ化を真理値表へ万能離散化)
# eval()にしてから抽出するのが必須。train()のままだと「連続混合」の状態を見ることになり、
# 回路として確定していない。eval()で各ゲートが1つの論理関数に決まる。
model.eval()
# export_ir はモデルを走査して、層の種類・配線・確定した真理値表をJSON化する。
# softmax / independent / iwp などどのパラメータ化で学習していても、
# 出てくるIRは「真理値表の集まり」に統一される(下流を疎結合にするための正規化)。
ir = export_ir(model, input_width=2, name='xor')
# HDL生成に必要な制約(ビット幅・トポロジ順序・未対応ノードの不在)を先に検査する。
# ここで落とせば、Verilogを書き出してから合成ツールで初めて気づく事故を防げる。
validate(ir, target='hdl')
save_ir(ir, 'out_examples/xor.json')
# === Verilogを生成する ===
# > `write_verilog` は IR から**合成可能な RTL**(レジスタ転送レベル記述)を書き出す。
# > できあがった `.v` は Vivado / Quartus といったベンダーツールへそのまま投げられる。
# > VHDL が要るなら `write_vhdl` に差し替えるだけでよい ── 下流を差し替えられるのは、
# > 上流の IR が正規化された真実源だからである。
# IRから合成可能なRTLを書き出す。ベンダーツール(Vivado/Quartus等)へそのまま投げられる。
write_verilog(ir, 'out_examples/xor.v')
print('生成: out_examples/xor.json, out_examples/xor.v')

# === 3段等価性で照合する ===
# > 出口の正しさは「3つの独立な実装が同じ答えを出すこと」で担保する。
# >
# > 1. **PyTorch hard eval** … 同じ真理値表を浮動小数のテンソル演算で通した結果
# > 2. **IR simulate** … 浮動小数を一切使わない整数ビット演算の参照実装(golden model)
# > 3. **HDL シミュレーション** … 生成した Verilog を `iverilog` で実際に走らせた結果
# >
# > 1と2が一致すれば「離散化に取りこぼしが無い」、さらに3が一致すれば
# > 「学習した重みが実際のハードウェアの挙動と一致する」と言い切れる。
# simulate() はIRをnumpyの整数ビット演算で解釈実行する参照実装。浮動小数の近似が
# 一切入らないので、「回路としてこう動くはず」という基準(golden model)になる。
# PyTorch側(model.eval())は同じ真理値表を浮動小数のテンソル演算で通しているだけなので、
# 両者が一致することは「離散化に取りこぼしが無い」ことの証明になる。
scores = simulate(ir, x.numpy().astype(np.int64))
pred_ir = scores.argmax(-1).tolist()
with torch.no_grad():
    pred_torch = model(x).argmax(-1).tolist()  # model は既に eval()
print(f'PyTorch hard 予測: {pred_torch}')
print(f'IR simulate 予測: {pred_ir} (正解 {y.tolist()})')
assert pred_torch == pred_ir == y.tolist()

# iverilogがあればVerilogとgolden modelを照合する。
# 生成したVerilogを実際にシミュレートし、全入力パターンでgolden modelと一致するか確かめる。
# PyTorch hard eval ⇔ IR simulate ⇔ HDLシミュレーション の3段等価性のうち、最後の1段。
# ここまで通れば、学習した重みが実際のハードウェアの挙動と一致していると言い切れる。
if iverilog_available():
    result = run_iverilog_equivalence(ir, x.numpy().astype(np.int64),
                                      'out_examples/verify')
    hdl_status = 'ALLPASS' if result['passed'] else '不一致'
    print(f'Verilog照合: {hdl_status}')
else:
    # iverilogは照合専用。未導入でもVerilogの生成自体は完了している。
    hdl_status = 'skipped'
    print('iverilog未導入のためVerilog照合はスキップ(brew install icarus-verilog)')

# --- 図3: 3段等価性を1枚の真理値表にする ---
# > 最後に、上の3経路の答えを1枚の表にまとめる。行が4つの入力パターン、
# > 列が答え合わせの経路(教師ラベル / PyTorch hard / IR simulate)。
# > セルの数字が出力クラスで、緑=`expected` と一致、赤=不一致である。
# >
# > **読み取り(この図が最後なので先に書く)**: 3列が同じ値で埋まり**緑一色**になる。
# > これが「学習の世界(PyTorch)」と「回路の世界(IR / HDL)」が同じ関数を
# > 表していることの確認であり、赤が1つでも出たらエクスポートのバグである。
# > 図のタイトルには `iverilog` による HDL 照合の結果も併記してある。
# 行が4つの入力パターン、列が「答え合わせの経路」。
#   expected (XOR)  — 教師ラベル。人間が決めた正解
#   PyTorch hard    — 学習済みモデルを eval() で離散化して通した結果(浮動小数のテンソル演算)
#   IR simulate     — IRを整数ビット演算で解釈実行した結果(浮動小数を一切使わない)
# 3列が全て同じ値で埋まり、緑一色になることがこのスクリプトの結論。
# これは「学習の世界(PyTorch)」と「回路の世界(IR/HDL)」が同じ関数を表している、
# つまり重みを回路に落としても意味が保存されている、ということを目で確認する図。
# セルの数字が出力クラス、緑=expectedと一致、赤=不一致(1つでも出たらエクスポートのバグ)。
rows = [f'A={int(a)}, B={int(b)}' for a, b in x.tolist()]
cols = ['expected (XOR)', 'PyTorch hard', 'IR simulate']
table = np.array([y.tolist(), pred_torch, pred_ir]).T  # (4パターン, 3経路)
match = (table == table[:, :1]).astype(float)          # expected列と一致しているか

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.imshow(match, cmap=ListedColormap(['#e8552e', '#8fd3a3']), vmin=0, vmax=1, aspect='auto')
for i in range(table.shape[0]):
    for j in range(table.shape[1]):
        ax.text(j, i, str(table[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold', color='#1a1a1a')
ax.set_xticks(range(len(cols)), cols)
ax.set_yticks(range(len(rows)), rows)
ax.set_xticks([j + 0.5 for j in range(len(cols) - 1)], minor=True)
ax.set_yticks([i + 0.5 for i in range(len(rows) - 1)], minor=True)
ax.grid(which='minor', color='white', lw=2)
ax.tick_params(which='minor', length=0)
ax.set_title(f'three-way equivalence (HDL check: {hdl_status})')
fig.tight_layout()
plt.show()
