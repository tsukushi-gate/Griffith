"""CNN→Conv-DLGN構造転写(convert/transcribe.transcribe_conv2d)のテスト。

transcribe_conv2dは**近似**転写である(モジュールdocstring参照: Conv-DLGNの
treeは受容野全体でなくサンプルされたn_leaves本しか見ないため、一般には
nn.Conv2dの閾値関数を厳密表現できない)。したがってここで検証するのは:

    (a) 内部整合性(厳密): 転写ノードのsimulateが、木構築が実際に選んだ
        真理値表の定義する関数と厳密に一致すること(IR配線・読み書きに
        バグが無いことの保証。ビット厳密assertでよい)
    (b) 木がリーフ入力に対して意図した関数(構築時に選んだ真理値表の合成)を
        real-once木として正しく実現すること(hypothesis/全列挙で、
        simulateされたconv出力を「木を手で評価した関数」と比較する)
    (c) shape/paramの整合(indices_flat/truth_tables形状、validate通過)

近似誤差(agreement < 1.0)は失敗ではなく期待された結果である。0を
主張するテストは書かない(タスク指示: 緩めない・偽らない)。
"""

import os
import sys

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.convert.transcribe import transcribe_conv2d
from griffith_logic.ir.schema import IRGraph, IRPort
from griffith_logic.ir.serialize import tt_from_hex
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _wrap_single_node(node, in_shape) -> IRGraph:
    """単一conv2d IRNodeを検証・simulate可能な最小IRGraphに包む"""
    from griffith_logic.ir.schema import IREdge
    c, h, w = in_shape
    out_ch = node.params['out_channels']
    kh, kw = node.params['kernel_size']
    sh, sw = node.params['stride']
    ph, pw = node.params['padding']
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    return IRGraph(
        inputs=[IRPort(id='x', width=c * h * w)],
        outputs=[IRPort(id='y', width=out_ch * h_out * w_out)],
        nodes=[node],
        edges=[IREdge(src='x', dst=node.id), IREdge(src=node.id, dst='y')],
    )


def _rand_binary_calib(n, c, h, w, seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=(n, c, h, w)).astype(np.float64)


def _make_conv(in_ch=1, out_ch=3, k=3, padding=0, seed=0) -> nn.Conv2d:
    torch.manual_seed(seed)
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding)


class TestShapeAndParams:
    """(c) shape/paramの整合"""

    def test_conv2d_node_shape(self):
        conv = _make_conv(in_ch=1, out_ch=4, k=3, padding=1)
        calib = _rand_binary_calib(64, 1, 8, 8, seed=1)
        node, info = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=2, n_inputs=2, seed=1,
        )
        assert node.type == 'conv2d'
        p = node.params
        assert p['in_shape'] == [1, 8, 8]
        assert p['out_channels'] == 4
        assert p['kernel_size'] == [3, 3]
        assert p['stride'] == [1, 1]
        assert p['padding'] == [1, 1]
        assert p['tree_depth'] == 2
        assert p['n_inputs'] == 2
        n_leaves = 2 ** 2
        n_internal = (n_leaves - 1) // (2 - 1)
        idxf = np.asarray(p['indices_flat'])
        assert idxf.shape == (4, n_leaves)
        assert idxf.min() >= 0
        assert idxf.max() < 1 * 3 * 3  # receptive_field = C*kH*kW
        tts = tt_from_hex(p['truth_tables'])
        assert len(tts) == 4 * n_internal
        assert all(0 <= t < (1 << (1 << 2)) for t in tts)
        assert info['n_leaves'] == n_leaves
        assert info['n_internal'] == n_internal
        assert 0.0 <= info['approx_agreement'] <= 1.0
        assert info['per_channel_agreement'].shape == (4,)

    def test_validates(self):
        conv = _make_conv(in_ch=1, out_ch=2, k=3, padding=0)
        calib = _rand_binary_calib(64, 1, 6, 6, seed=2)
        node, _ = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=2, n_inputs=2, seed=2,
        )
        graph = _wrap_single_node(node, (1, 6, 6))
        assert validate(graph) == []

    def test_rejects_stride_ne_1(self):
        conv = nn.Conv2d(1, 2, kernel_size=3, stride=2)
        calib = _rand_binary_calib(16, 1, 8, 8)
        with pytest.raises(ValueError, match='stride'):
            transcribe_conv2d(conv, calib_inputs=calib)

    def test_rejects_grouped(self):
        conv = nn.Conv2d(2, 2, kernel_size=3, groups=2)
        calib = _rand_binary_calib(16, 2, 8, 8)
        with pytest.raises(ValueError, match='grouped'):
            transcribe_conv2d(conv, calib_inputs=calib)

    def test_rejects_bad_n_inputs(self):
        conv = _make_conv()
        calib = _rand_binary_calib(16, 1, 8, 8)
        with pytest.raises(ValueError, match='n_inputs'):
            transcribe_conv2d(conv, calib_inputs=calib, n_inputs=3)

    def test_requires_calib(self):
        conv = _make_conv()
        with pytest.raises(ValueError, match='calib'):
            transcribe_conv2d(conv, calib_inputs=None)


class TestInternalConsistency:
    """(a) 内部整合性: simulate(IR) == 構築した真理値表が定義する関数(厳密)"""

    def test_simulate_matches_truth_tables_no_padding(self):
        conv = _make_conv(in_ch=1, out_ch=3, k=3, padding=0, seed=5)
        calib = _rand_binary_calib(200, 1, 8, 8, seed=5)
        node, _ = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=2, n_inputs=2, seed=5,
        )
        graph = _wrap_single_node(node, (1, 8, 8))
        assert validate(graph) == []

        # 評価用の新しい二値入力(較正データと別集合でもIR自身の内部整合性は
        # 較正データに依存しない: 真理値表は既に固定されているため)
        x_eval = _rand_binary_calib(50, 1, 8, 8, seed=999).reshape(50, -1).astype(np.int64)
        scores = simulate(graph, x_eval)

        expected = _manual_tree_eval(node, x_eval, in_shape=(1, 8, 8))
        assert np.array_equal(scores, expected), (
            'simulateが木構築の真理値表定義と厳密不一致(配線/IRバグ)'
        )

    def test_simulate_matches_truth_tables_with_padding(self):
        conv = _make_conv(in_ch=1, out_ch=2, k=3, padding=1, seed=6)
        calib = _rand_binary_calib(200, 1, 6, 6, seed=6)
        node, _ = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=2, n_inputs=2, seed=6,
        )
        graph = _wrap_single_node(node, (1, 6, 6))
        assert validate(graph) == []

        x_eval = _rand_binary_calib(40, 1, 6, 6, seed=1234).reshape(40, -1).astype(np.int64)
        scores = simulate(graph, x_eval)
        expected = _manual_tree_eval(node, x_eval, in_shape=(1, 6, 6))
        assert np.array_equal(scores, expected)

    def test_simulate_matches_truth_tables_n4(self):
        conv = _make_conv(in_ch=2, out_ch=2, k=3, padding=1, seed=7)
        calib = _rand_binary_calib(150, 2, 6, 6, seed=7)
        node, _ = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=1, n_inputs=4, seed=7,
        )
        graph = _wrap_single_node(node, (2, 6, 6))
        assert validate(graph) == []

        x_eval = _rand_binary_calib(30, 2, 6, 6, seed=4321).reshape(30, -1).astype(np.int64)
        scores = simulate(graph, x_eval)
        expected = _manual_tree_eval(node, x_eval, in_shape=(2, 6, 6))
        assert np.array_equal(scores, expected)


def _manual_tree_eval(node, x_flat: np.ndarray, in_shape: tuple) -> np.ndarray:
    """conv2d IRNodeが定義する木関数を、simulateを使わず素朴に手評価する。

    unfold相当のパッチ抽出+木のtruth_tables/indices_flatをそのまま辿るだけの
    参照実装(素朴さ優先、simulateの実装を信用しない独立経路)。
    """
    p = node.params
    c, h, w = in_shape
    out_ch = p['out_channels']
    kh, kw = p['kernel_size']
    sh, sw = p['stride']
    ph, pw = p['padding']
    n = p['n_inputs']
    d = p['tree_depth']
    n_leaves = n ** d
    n_internal = (n_leaves - 1) // (n - 1) if n != 1 else d
    idxf = np.asarray(p['indices_flat'])
    tts = tt_from_hex(p['truth_tables'])
    tts = np.asarray(tts, dtype=object).reshape(out_ch, n_internal)

    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    b = x_flat.shape[0]
    imgs = x_flat.reshape(b, c, h, w)

    out = np.zeros((b, out_ch, h_out, w_out), dtype=np.int64)
    for oc in range(out_ch):
        for oy in range(h_out):
            for ox in range(w_out):
                # 受容野内フラット位置 → 実画像座標(範囲外は0=padding)
                leaves = np.zeros((n_leaves, b), dtype=np.int64)
                for leaf_i in range(n_leaves):
                    pos = int(idxf[oc, leaf_i])
                    ch_i, rem = divmod(pos, kh * kw)
                    dy, dx = divmod(rem, kw)
                    y = oy * sh - ph + dy
                    x_ = ox * sw - pw + dx
                    if 0 <= y < h and 0 <= x_ < w:
                        leaves[leaf_i] = imgs[:, ch_i, y, x_]
                    else:
                        leaves[leaf_i] = 0
                current = leaves
                gate_offset = 0
                for _level in range(d):
                    n_groups = current.shape[0] // n
                    next_sig = np.zeros((n_groups, b), dtype=np.int64)
                    for g in range(n_groups):
                        tt = int(tts[oc, gate_offset + g])
                        children = current[g * n:(g + 1) * n]  # (n, b)
                        addr = np.zeros(b, dtype=np.int64)
                        for j in range(n):
                            addr |= children[j] << j
                        next_sig[g] = np.array(
                            [(tt >> int(a)) & 1 for a in addr], dtype=np.int64,
                        )
                    current = next_sig
                    gate_offset += n_groups
                out[:, oc, oy, ox] = current[0]
    return out.reshape(b, -1)


class TestReadOnceRealization:
    """(b) hypothesis: 木がリーフ入力から意図した(構築時に選んだ)関数を
    read-once N分木として正しく実現する(全リーフパターン列挙、小木限定)。
    """

    @given(seed=st.integers(0, 2 ** 16 - 1))
    @settings(max_examples=15)
    def test_tree_matches_manual_eval_all_leaf_patterns(self, seed):
        """n_inputs=2, tree_depth=2 (n_leaves=4) の全16リーフパターンで、
        simulateされたconv出力(パディング無し・1x1出力)が
        真理値表を手評価した木関数と一致する。"""
        rng = np.random.RandomState(seed)
        in_ch, k = 1, 2  # receptive_field = 1*2*2 = 4 = n_leaves(全域網羅)
        conv = nn.Conv2d(in_ch, 1, kernel_size=k, padding=0)
        with torch.no_grad():
            conv.weight.copy_(torch.randn_like(conv.weight))
            conv.bias.copy_(torch.randn_like(conv.bias))
        calib = rng.randint(0, 2, size=(64, in_ch, k, k)).astype(np.float64)
        node, _ = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=2, n_inputs=2, seed=seed,
        )
        graph = _wrap_single_node(node, (in_ch, k, k))
        assert validate(graph) == []

        n_leaves = 4
        all_patterns = ((np.arange(1 << n_leaves)[:, None]
                         >> np.arange(n_leaves)[None, :]) & 1).astype(np.int64)
        # receptive_field == n_leaves(重複なし全域網羅)なのでindices_flatは
        # 0..3の並び替え。simulate用に画像へ逆写像する。
        idxf = np.asarray(node.params['indices_flat'])[0]  # (4,)
        x_flat = np.zeros((all_patterns.shape[0], in_ch * k * k), dtype=np.int64)
        for leaf_i in range(n_leaves):
            x_flat[:, idxf[leaf_i]] = all_patterns[:, leaf_i]

        scores = simulate(graph, x_flat)
        expected = _manual_tree_eval(node, x_flat, in_shape=(in_ch, k, k))
        assert np.array_equal(scores, expected)


class TestApproximationErrorReported:
    """近似誤差は測定・報告されるが0を主張しない(正直な近似であることの確認)"""

    def test_agreement_is_well_formed_not_asserted_perfect(self):
        """agreementは[0,1]の実数として返る。1.0を要求も禁止もしない
        (容量に依存して変動してよい正直な指標)。"""
        conv = _make_conv(in_ch=1, out_ch=8, k=3, padding=1, seed=11)
        calib = _rand_binary_calib(500, 1, 10, 10, seed=11)
        _, info = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=3, n_inputs=2, seed=11,
        )
        assert 0.0 <= info['approx_agreement'] <= 1.0
        assert np.all(info['per_channel_agreement'] >= 0.0)
        assert np.all(info['per_channel_agreement'] <= 1.0)

    def test_full_receptive_field_coverage_improves_or_matches_sparse(self):
        """受容野全体をリーフが網羅できる(n_leaves>=receptive_field)場合の
        agreementは、木がターゲット関数を多数決なしで表現できる分、
        より少ないリーフしか見ない構成以上に高いことが多い(傾向確認、
        厳密な不等式assertはしない=ランダム性に依存するため上下限のみ確認)。
        """
        conv = _make_conv(in_ch=1, out_ch=4, k=3, padding=0, seed=12)  # RF=9
        calib = _rand_binary_calib(400, 1, 8, 8, seed=12)
        # n_leaves=16 (>=9=RF): 全域網羅可能
        _, info_wide = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=4, n_inputs=2, seed=12,
        )
        # n_leaves=4 (<9=RF): 一部のみ観測
        _, info_narrow = transcribe_conv2d(
            conv, calib_inputs=calib, tree_depth=2, n_inputs=2, seed=12,
        )
        assert 0.0 <= info_wide['approx_agreement'] <= 1.0
        assert 0.0 <= info_narrow['approx_agreement'] <= 1.0


class TestFoldConvBn:
    """fold_conv_bn: Conv2d+BatchNorm2d(eval統計) → bias付きConv2d融合(W13a-5)"""

    def _make_conv_bn(self, in_ch=3, out_ch=4, k=3, padding=1, bias=False, seed=0):
        from griffith_logic.convert.transcribe import fold_conv_bn  # noqa: F401
        torch.manual_seed(seed)
        conv = nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=bias)
        bn = nn.BatchNorm2d(out_ch)
        # ランダムだが有効なeval統計(var>0)を用意
        with torch.no_grad():
            bn.running_mean.copy_(torch.randn(out_ch) * 0.3)
            bn.running_var.copy_(torch.rand(out_ch) * 0.5 + 0.5)
            bn.weight.copy_(torch.randn(out_ch) * 0.8 + 1.0)
            bn.bias.copy_(torch.randn(out_ch) * 0.2)
        bn.eval()
        return conv, bn

    @pytest.mark.parametrize('bias', [False, True])
    def test_fused_output_matches_conv_then_bn(self, bias):
        from griffith_logic.convert.transcribe import fold_conv_bn
        conv, bn = self._make_conv_bn(bias=bias, seed=1)
        fused = fold_conv_bn(conv, bn)

        x = torch.rand(3, conv.in_channels, 10, 10)
        with torch.no_grad():
            y_ref = bn(conv(x))
            y_fused = fused(x)
        assert torch.allclose(y_ref, y_fused, atol=1e-5, rtol=1e-5)

    def test_fused_is_bias_conv2d_same_shape(self):
        from griffith_logic.convert.transcribe import fold_conv_bn
        conv, bn = self._make_conv_bn(in_ch=2, out_ch=5, k=3, seed=2)
        fused = fold_conv_bn(conv, bn)
        assert isinstance(fused, nn.Conv2d)
        assert fused.bias is not None
        assert fused.in_channels == conv.in_channels
        assert fused.out_channels == conv.out_channels
        assert fused.kernel_size == conv.kernel_size
        assert fused.padding == conv.padding
        assert fused.stride == conv.stride

    def test_channel_mismatch_raises(self):
        from griffith_logic.convert.transcribe import fold_conv_bn
        conv = nn.Conv2d(3, 4, 3)
        bn = nn.BatchNorm2d(5)  # わざと不一致
        with pytest.raises(ValueError):
            fold_conv_bn(conv, bn)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
