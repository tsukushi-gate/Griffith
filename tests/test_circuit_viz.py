"""analysis/circuit_viz の回路図可視化テスト(VIZ-1)。

- PNG が実際に書き出せること(2モードとも)
- resolve=True のゲート名が、IR(=Verilog の真実源)の真理値表から
  導かれる名前と一致すること(独自命名になっていないことの担保)
"""

import pytest
import torch
import torch.nn as nn

# matplotlib は figures extra(任意)。未導入の環境ではスキップする。
matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')

from griffith_logic import GroupSum, LutLayer  # noqa: E402
from griffith_logic.analysis.circuit_viz import (  # noqa: E402
    GATE_NAMES,
    gate_labels,
    plot_abstract,
    plot_circuit,
    tt_to_gate_id_n2,
)
from griffith_logic.export import export_ir  # noqa: E402
from griffith_logic.ir.serialize import tt_from_hex  # noqa: E402


@pytest.fixture
def xor_model():
    """XOR規模の最小モデル(2ビット入力→16ゲート×2段→2クラス)。"""
    torch.manual_seed(0)
    return nn.Sequential(
        LutLayer(2, 16, n_inputs=2),
        LutLayer(16, 16, n_inputs=2),
        GroupSum(k=2, tau=2.0),
    )


def test_plot_circuit_writes_png(xor_model, tmp_path):
    """学習前(混合)・学習後(確定)の両モードで PNG が出る。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_circuit(xor_model, resolve=False, ax=axes[0])
    plot_circuit(xor_model, resolve=True, ax=axes[1])
    out = tmp_path / 'circuit.png'
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


def test_plot_abstract_writes_png(tmp_path):
    """大きな網は俯瞰図へフォールバックできる。"""
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    model = nn.Sequential(
        LutLayer(8, 512, n_inputs=2),
        LutLayer(512, 512, n_inputs=2),
        GroupSum(k=2, tau=2.0),
    )
    # ゲート総数 > MAX_FULL_GATES なので plot_circuit も自動で俯瞰図になる
    ax = plot_circuit(model, resolve=True)
    fig = ax.figure
    out = tmp_path / 'abstract.png'
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0

    ax2 = plot_abstract(model)
    plt.close(ax2.figure)


def test_gate_names_match_ir_truth_tables(xor_model):
    """resolve時のラベルが IR の真理値表から導く名前と一致する。

    IR は Verilog / VHDL / simulate の共通の真実源なので、これが一致していれば
    図のゲート名は生成される回路のゲートと同じものを指している。
    """
    xor_model.eval()
    ir = export_ir(xor_model, input_width=2, name='xor_viz')
    lut_nodes = [n for n in ir.nodes if n.type == 'lut_layer']
    luts = [m for m in xor_model if isinstance(m, LutLayer)]
    assert len(lut_nodes) == len(luts)

    for node, layer in zip(lut_nodes, luts, strict=True):
        expected = [
            GATE_NAMES[tt_to_gate_id_n2(int(t))]
            for t in tt_from_hex(node.params['truth_tables'])
        ]
        assert gate_labels(layer) == expected


def test_gate_names_are_the_editor_definition():
    """16ゲート名テーブルがビジュアルエディタのバックエンド定義と同一。"""
    assert GATE_NAMES == {
        0: 'FALSE', 1: 'AND', 2: 'A>B', 3: 'A',
        4: 'B>A', 5: 'B', 6: 'XOR', 7: 'OR',
        8: 'NOR', 9: 'XNOR', 10: '!B', 11: 'A>=B',
        12: '!A', 13: 'B>=A', 14: 'NAND', 15: 'TRUE',
    }
