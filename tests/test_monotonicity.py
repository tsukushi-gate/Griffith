"""experiments/monotonicity.py の指標(V / C / 記録フック)のテスト。

高速・CPU・slowマーカーなし。コア実装は変更せず weights を読むだけのアダプタを検証する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from monotonicity import (
    POLARITY_DECREASING,
    POLARITY_INCREASING,
    POLARITY_MIXED,
    MonotonicityRecorder,
    collect_lut_conv_layers,
    collect_lut_layers,
    conv_layer_polarity_fractions,
    conv_monotonicity_violation_V,
    conv_soft_tables,
    layer_polarity,
    lut_soft_table,
    mean_monotonicity_violation_V,
    monotonicity_violation_V,
    polarity_consistency_C,
    record_model_monotonicity,
)

from griffith_logic import GroupSum, LutConv2d, LutLayer

# e=Σx_j2^j 順の代表的な真理値表(n_inputs=2)
TABLE_AND = torch.tensor([0.0, 0.0, 0.0, 1.0])       # x0 AND x1
TABLE_OR = torch.tensor([0.0, 1.0, 1.0, 1.0])        # x0 OR x1
TABLE_ID0 = torch.tensor([0.0, 1.0, 0.0, 1.0])       # 恒等(x0のみ)
TABLE_NOT0 = torch.tensor([1.0, 0.0, 1.0, 0.0])      # NOT x0
TABLE_XOR = torch.tensor([0.0, 1.0, 1.0, 0.0])       # x0 XOR x1


def _identity_layer(n_inputs, parameterization='independent', out_dim=1):
    """neuron が入力0..n-1 を slot 0..n-1 にそのまま取る層を作る(接続を固定)。"""
    layer = LutLayer(
        in_dim=n_inputs, out_dim=out_dim, n_inputs=n_inputs,
        parameterization=parameterization,
    )
    # indices 形状 (n_inputs, out_dim): 全ニューロンが slot s に入力 s を取る
    idx = torch.arange(n_inputs).unsqueeze(1).expand(n_inputs, out_dim).contiguous()
    layer.indices.copy_(idx)
    return layer


def _corners(n_inputs):
    """全 2^n binary コーナー入力 (2^n, n_inputs)。行 e のビット j = (e>>j)&1。"""
    n_entries = 2 ** n_inputs
    corners = torch.zeros(n_entries, n_inputs)
    for e in range(n_entries):
        for j in range(n_inputs):
            corners[e, j] = (e >> j) & 1
    return corners


def _set_table(layer, table):
    """独立/AP用に truth table を全ニューロンへ設定する。"""
    layer.weights.data.copy_(table.unsqueeze(0).expand_as(layer.weights.data))


# ============================================================
# 1. 順序健全性: T のエントリ順序がコア推論forwardと整合する
# ============================================================

@pytest.mark.parametrize('table', [TABLE_AND, TABLE_OR, TABLE_ID0])
def test_soft_table_order_independent(table):
    """independent(ap_mode)の T が、コア推論forwardのコーナー出力と一致する。"""
    layer = _identity_layer(2, parameterization='independent')
    _set_table(layer, table)

    t = lut_soft_table(layer, ap_mode=True)  # (1, 4)
    assert torch.allclose(t[0], table)

    # コア推論forward(training=False)をコーナー入力で回す
    layer.eval()
    with torch.no_grad():
        out = layer(_corners(2))  # (4, 1)
    assert torch.allclose(out[:, 0], table)


def test_soft_table_order_16gate():
    """16ゲート softmax の T が、argmax離散化したコア推論forwardと整合する。"""
    layer = _identity_layer(2, parameterization='softmax')
    # AND ゲート(g=1)を強く選好させる
    layer.weights.data.zero_()
    layer.weights.data[0, 1] = 20.0

    t = lut_soft_table(layer, ap_mode=False)  # (1, 4) ≈ gate_table[1]=[0,0,0,1]
    assert torch.allclose(t[0], TABLE_AND, atol=1e-4)

    layer.eval()
    with torch.no_grad():
        out = layer(_corners(2))  # argmax → gate_table[1]
    assert torch.allclose(out[:, 0], TABLE_AND)


# ============================================================
# 2. 完全単調LUT → V=0、極性 +1
# ============================================================

@pytest.mark.parametrize('table', [TABLE_AND, TABLE_OR, TABLE_ID0])
def test_monotone_lut_zero_violation(table):
    layer = _identity_layer(2, parameterization='independent')
    _set_table(layer, table)

    v = monotonicity_violation_V(layer, ap_mode=True)
    assert v == pytest.approx(0.0, abs=1e-7)


def test_and_polarity_increasing():
    layer = _identity_layer(2, parameterization='independent')
    _set_table(layer, TABLE_AND)
    pol = layer_polarity(layer, ap_mode=True)  # (1, 2)
    assert pol[0, 0].item() == POLARITY_INCREASING
    assert pol[0, 1].item() == POLARITY_INCREASING


# ============================================================
# 3. NOT/反転LUT → V>0 かつ 極性 −1
# ============================================================

def test_not_lut_violation_and_negative_polarity():
    layer = _identity_layer(2, parameterization='independent')
    _set_table(layer, TABLE_NOT0)  # 入力0を反転(単調減少)

    v = monotonicity_violation_V(layer, ap_mode=True)
    assert v > 0.0
    # NOT x0: total_violation=2, count=1*2*2=4 → V=0.5
    assert v == pytest.approx(0.5, abs=1e-6)

    pol = layer_polarity(layer, ap_mode=True)
    assert pol[0, 0].item() == POLARITY_DECREASING  # 入力0は減少
    assert pol[0, 1].item() == POLARITY_INCREASING  # 入力1は定数 → 増加扱い


# ============================================================
# 4. XOR LUT → mixed、V>0
# ============================================================

def test_xor_lut_mixed_polarity():
    layer = _identity_layer(2, parameterization='independent')
    _set_table(layer, TABLE_XOR)

    v = monotonicity_violation_V(layer, ap_mode=True)
    assert v > 0.0

    pol = layer_polarity(layer, ap_mode=True)
    assert pol[0, 0].item() == POLARITY_MIXED
    assert pol[0, 1].item() == POLARITY_MIXED


# ============================================================
# 5. C の経路集計
# ============================================================

def _make_model(in_dim, out_dim, num_classes, n_layers):
    layers = [torch.nn.Flatten()]
    layers.append(LutLayer(in_dim, out_dim, n_inputs=2, parameterization='independent'))
    for _ in range(n_layers - 1):
        layers.append(LutLayer(out_dim, out_dim, n_inputs=2, parameterization='independent'))
    layers.append(GroupSum(num_classes, tau=1.0))
    return torch.nn.Sequential(*layers)


def test_polarity_consistency_all_monotone():
    """全LUTが単調増加(AND)なら +1経路割合≈1.0、mixed割合≈0。"""
    model = _make_model(in_dim=6, out_dim=4, num_classes=2, n_layers=2)
    luts = collect_lut_layers(model)
    for layer in luts:
        _set_table(layer, TABLE_AND)

    c = polarity_consistency_C(model, luts, n_paths=2000, ap_mode=True, seed=0)
    last = c['per_layer'][len(luts) - 1]
    assert last['frac_positive'] > 0.99
    assert last['frac_mixed'] < 0.01
    assert c['mean_frac_mixed'] < 0.01


def test_polarity_consistency_mixed_present():
    """非単調(XOR)を混ぜると mixed経路割合>0 になる。"""
    model = _make_model(in_dim=6, out_dim=4, num_classes=2, n_layers=2)
    luts = collect_lut_layers(model)
    for layer in luts:
        _set_table(layer, TABLE_AND)
    # 最終層をXOR(mixed)に差し替え
    _set_table(luts[-1], TABLE_XOR)

    c = polarity_consistency_C(model, luts, n_paths=2000, ap_mode=True, seed=0)
    assert c['mean_frac_mixed'] > 0.0
    # 最終層開始の経路は最初の辺がXOR → 高確率でmixed
    assert c['per_layer'][len(luts) - 1]['frac_mixed'] > 0.5


# ============================================================
# 6. 記録フック
# ============================================================

def test_recorder_interval():
    model = _make_model(in_dim=6, out_dim=4, num_classes=2, n_layers=2)
    luts = collect_lut_layers(model)
    for layer in luts:
        _set_table(layer, TABLE_AND)

    recorder = MonotonicityRecorder(
        model, luts, ap_mode=True, interval=10, n_paths=100
    )

    rec0 = recorder.maybe_record(0)
    assert rec0 is not None
    assert set(rec0.keys()) == {'step', 'V_per_layer', 'V_mean', 'C'}

    for step in range(1, 10):
        assert recorder.maybe_record(step) is None

    rec10 = recorder.maybe_record(10)
    rec20 = recorder.maybe_record(20)
    assert rec10 is not None and rec20 is not None
    assert [r['step'] for r in recorder.history] == [0, 10, 20]


def test_mean_violation_helper():
    """全体平均ヘルパーが層平均を返す。"""
    model = _make_model(in_dim=6, out_dim=4, num_classes=2, n_layers=2)
    luts = collect_lut_layers(model)
    _set_table(luts[0], TABLE_AND)   # V=0
    _set_table(luts[1], TABLE_NOT0)  # V=0.5

    mean_v = mean_monotonicity_violation_V(luts, ap_mode=True)
    assert mean_v == pytest.approx(0.25, abs=1e-6)


# ============================================================
# 7. CNN(LutConv2d)対応
# ============================================================

def _small_conv(out_channels=2, tree_depth=2, n_inputs=2):
    """forward不要の小さな LutConv2d(independent)。weights を直接操作して検証する。"""
    return LutConv2d(
        in_channels=2, out_channels=out_channels, kernel_size=1,
        tree_depth=tree_depth, n_inputs=n_inputs,
        parameterization='independent', residual_init=False,
    )


def _set_conv_table(conv, table):
    """全ゲートに同じ truth table を設定する(weights 形状 (oc, n_internal, 2^n))。"""
    conv.weights.data.copy_(
        table.view(1, 1, -1).expand_as(conv.weights.data)
    )


def test_conv_monotone_zero_violation():
    """(a) 単調増加(AND)ゲートの LutConv2d は V≈0。"""
    conv = _small_conv(out_channels=2, tree_depth=2, n_inputs=2)
    _set_conv_table(conv, TABLE_AND)

    # G = out_channels * n_internal、テーブルは (G, 4) で全て AND
    t = conv_soft_tables(conv, ap_mode=True)
    assert t.shape == (conv.out_channels * conv.n_internal, 4)
    assert torch.allclose(t[0], TABLE_AND)

    v = conv_monotonicity_violation_V(conv, ap_mode=True)
    assert v == pytest.approx(0.0, abs=1e-7)

    fr = conv_layer_polarity_fractions(conv, ap_mode=True)
    assert fr['frac_positive'] == pytest.approx(1.0, abs=1e-6)


def test_conv_inverted_gate_positive_violation():
    """(b) 反転(NOT)ゲートの LutConv2d は V>0 かつ frac_negative>0。"""
    conv = _small_conv(out_channels=2, tree_depth=2, n_inputs=2)
    _set_conv_table(conv, TABLE_NOT0)  # 入力0を反転(単調減少)

    v = conv_monotonicity_violation_V(conv, ap_mode=True)
    assert v > 0.0
    # NOT x0: 入力0で total_violation=2/gate, count=1*2*2=4/gate → V=0.5
    assert v == pytest.approx(0.5, abs=1e-6)

    fr = conv_layer_polarity_fractions(conv, ap_mode=True)
    assert fr['frac_negative'] > 0.0


def test_record_model_monotonicity_conv_dense():
    """(c) Conv+Dense 混在モデルで V_mean/mixed_mean を含む dict を例外なく返す。"""
    conv = _small_conv(out_channels=2, tree_depth=1, n_inputs=2)
    dense = LutLayer(in_dim=8, out_dim=4, n_inputs=2, parameterization='independent')
    model = torch.nn.Sequential(
        conv, torch.nn.Flatten(), dense, GroupSum(2, tau=1.0),
    )

    assert len(collect_lut_conv_layers(model)) == 1

    rec = record_model_monotonicity(model, ap_mode=True)
    assert 'V_mean' in rec and 'mixed_mean' in rec
    assert [m['type'] for m in rec['per_module']] == ['conv', 'dense']
    for m in rec['per_module']:
        assert {'type', 'V', 'frac_positive', 'frac_negative', 'frac_mixed'} <= set(m)
    assert isinstance(rec['V_mean'], float)
    assert isinstance(rec['mixed_mean'], float)
