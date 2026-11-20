import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.conv import LutConv2d
from griffith_logic.group_sum import GroupSum
from griffith_logic.lut_layer import LutLayer
from griffith_logic.pooling import OrPooling2d

# ============================================================
# OrPooling2d
# ============================================================

class TestOrPooling2d:
    def test_output_shape(self):
        """2×2 poolingで空間サイズが半減するか"""
        pool = OrPooling2d(kernel_size=2, stride=2)
        x = torch.rand(2, 3, 8, 8)
        y = pool(x)
        assert y.shape == (2, 3, 4, 4)

    def test_max_behavior(self):
        """max値が正しく選ばれるか"""
        pool = OrPooling2d(kernel_size=2, stride=2)
        x = torch.tensor([[[[0.1, 0.9], [0.3, 0.5]]]]).float()
        y = pool(x)
        assert y.shape == (1, 1, 1, 1)
        assert torch.isclose(y[0, 0, 0, 0], torch.tensor(0.9))

    def test_stride_padding(self):
        """stride=1, padding=1での動作"""
        pool = OrPooling2d(kernel_size=3, stride=1, padding=1)
        x = torch.rand(2, 4, 6, 6)
        y = pool(x)
        assert y.shape == (2, 4, 6, 6)

    def test_default_stride(self):
        """strideを省略した場合、kernel_sizeと同じになるか"""
        pool = OrPooling2d(kernel_size=3)
        assert pool.stride == 3


# ============================================================
# LutConv2d: 基本テスト
# ============================================================

class TestLutConv2d:
    def test_output_shape_no_padding(self):
        """パディングなしでの出力形状"""
        conv = LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=0)
        x = torch.rand(2, 1, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 6, 6)

    def test_output_shape_with_padding(self):
        """パディングありでの出力形状（サイズ維持）"""
        conv = LutConv2d(3, 16, kernel_size=3, tree_depth=3, padding=1)
        x = torch.rand(2, 3, 16, 16)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16)

    def test_output_shape_stride2(self):
        """stride=2での出力形状"""
        conv = LutConv2d(3, 8, kernel_size=3, tree_depth=2, padding=1, stride=2)
        x = torch.rand(2, 3, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 4, 4)

    def test_output_shape_5x5_kernel(self):
        """5×5カーネルでの出力形状（MNIST第1層に相当）"""
        conv = LutConv2d(1, 16, kernel_size=5, tree_depth=3, padding=0)
        x = torch.rand(2, 1, 28, 28)
        y = conv(x)
        assert y.shape == (2, 16, 24, 24)

    def test_tree_structure(self):
        """ツリーのゲート数・リーフ数が正しいか"""
        # N=2, d=3 → 8 leaves, 7 internal nodes
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=3, n_inputs=2)
        assert conv.n_leaves == 8
        assert conv.n_internal == 7
        assert conv.weights.shape == (4, 7, 4)  # (out_ch, n_internal, 2^N)

        # N=2, d=2 → 4 leaves, 3 internal nodes
        conv2 = LutConv2d(1, 4, kernel_size=3, tree_depth=2, n_inputs=2)
        assert conv2.n_leaves == 4
        assert conv2.n_internal == 3

    def test_tree_structure_n4(self):
        """N=4のツリー構造が正しいか"""
        # N=4, d=2 → 16 leaves, 5 internal nodes = (16-1)/(4-1)
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, n_inputs=4)
        assert conv.n_leaves == 16
        assert conv.n_internal == 5
        assert conv.weights.shape == (4, 5, 16)  # (out_ch, n_internal, 2^4)

        # N=4, d=1 → 4 leaves, 1 internal node
        conv2 = LutConv2d(1, 4, kernel_size=3, tree_depth=1, n_inputs=4)
        assert conv2.n_leaves == 4
        assert conv2.n_internal == 1

    def test_tree_structure_n6(self):
        """N=6のツリー構造が正しいか"""
        # N=6, d=1 → 6 leaves, 1 internal node
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=1, n_inputs=6)
        assert conv.n_leaves == 6
        assert conv.n_internal == 1
        assert conv.weights.shape == (4, 1, 64)  # (out_ch, n_internal, 2^6)

    def test_output_shape_n4(self):
        """N=4での出力形状"""
        conv = LutConv2d(3, 8, kernel_size=3, tree_depth=2, padding=1, n_inputs=4)
        x = torch.rand(2, 3, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 8, 8)

    def test_output_shape_n6(self):
        """N=6での出力形状"""
        conv = LutConv2d(3, 8, kernel_size=3, tree_depth=1, padding=1, n_inputs=6)
        x = torch.rand(2, 3, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 8, 8)

    def test_eval_discrete_n4(self):
        """N=4のeval時にargmax離散化されるか"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1, n_inputs=4)
        conv.eval()
        x = torch.rand(2, 1, 8, 8).round()
        y = conv(x)
        assert y.shape == (2, 4, 8, 8)
        assert torch.all((y == 0.0) | (y == 1.0)), \
            f'N=4 eval時のバイナリ入力に対して出力が0/1でない: unique={y.unique()}'

    def test_invalid_n_inputs(self):
        """不正なn_inputs値でAssertionError"""
        with pytest.raises(AssertionError):
            LutConv2d(1, 4, kernel_size=3, tree_depth=2, n_inputs=3)

    def test_num_gates(self):
        """総ゲート数の計算"""
        conv = LutConv2d(3, 32, kernel_size=3, tree_depth=3, n_inputs=2)
        assert conv.num_gates == 32 * 7  # 32 trees × 7 gates

    def test_eval_discrete(self):
        """eval時にargmax離散化されるか"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1)
        conv.eval()
        x = torch.rand(2, 1, 8, 8).round()  # バイナリ入力
        y = conv(x)
        assert y.shape == (2, 4, 8, 8)
        # バイナリ入力に対してバイナリ出力であるべき
        assert torch.all((y == 0.0) | (y == 1.0)), \
            f'eval時のバイナリ入力に対して出力が0/1でない: unique={y.unique()}'

    def test_residual_init_weights(self):
        """Residual Initializationの重みが正しく設定されるか"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, n_inputs=2, residual_init=True)
        w = conv.weights.data
        # bit 0 が立つエントリ（k=1, k=3）は z3=5
        assert torch.allclose(w[:, :, 0], torch.zeros_like(w[:, :, 0]))
        assert torch.allclose(w[:, :, 1], torch.full_like(w[:, :, 1], 5.0))
        assert torch.allclose(w[:, :, 2], torch.zeros_like(w[:, :, 2]))
        assert torch.allclose(w[:, :, 3], torch.full_like(w[:, :, 3], 5.0))

    def test_no_residual_init(self):
        """residual_init=Falseの場合、ランダム初期化"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, residual_init=False)
        w = conv.weights.data
        # ランダム値なので0や5ではないはず
        assert not torch.allclose(w[:, :, 0], torch.zeros_like(w[:, :, 0]))

    def test_channel_restriction(self):
        """input_channels_per_treeの制限が効いているか"""
        conv = LutConv2d(16, 8, kernel_size=3, tree_depth=2,
                         input_channels_per_tree=2)
        # indices_flat: (out_channels, n_leaves)
        # unfoldインデックスからチャンネルを逆算: ch = idx // (kH * kW)
        kH, kW = conv.kernel_size
        channels_used = conv.indices_flat // (kH * kW)
        for k in range(conv.out_channels):
            unique_ch = channels_used[k].unique()
            assert len(unique_ch) <= 2, \
                f'ツリー{k}が{len(unique_ch)}チャンネルを参照: {unique_ch}'


# ============================================================
# LutConv2d: 勾配テスト
# ============================================================

class TestConvGradient:
    def _gradcheck(self, in_ch=1, out_ch=2, kernel_size=3, tree_depth=2,
                   n_inputs=2, H=6, W=6, batch=1):
        conv = LutConv2d(
            in_ch, out_ch, kernel_size=kernel_size,
            tree_depth=tree_depth, n_inputs=n_inputs, padding=1,
            residual_init=False,
        ).double()
        x = torch.randn(batch, in_ch, H, W, dtype=torch.float64, requires_grad=True)
        fn = lambda x: conv(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3), \
            '勾配チェックが失敗'

    def test_gradcheck_n2_d2(self):
        """N=2, d=2の勾配数値検証"""
        self._gradcheck(n_inputs=2, tree_depth=2)

    def test_gradcheck_n2_d3(self):
        """N=2, d=3の勾配数値検証"""
        self._gradcheck(n_inputs=2, tree_depth=3)

    def test_gradcheck_multichannel(self):
        """多チャンネル入力の勾配検証"""
        self._gradcheck(in_ch=3, out_ch=4, n_inputs=2, tree_depth=2)

    def test_gradcheck_n4_d1(self):
        """N=4, d=1の勾配数値検証"""
        self._gradcheck(n_inputs=4, tree_depth=1)

    def test_gradcheck_n4_d2(self):
        """N=4, d=2の勾配数値検証"""
        self._gradcheck(n_inputs=4, tree_depth=2)


# ============================================================
# CNN統合テスト
# ============================================================

class TestCNNIntegration:
    def test_conv_pool_stack(self):
        """Conv → Pool → Conv → Pool のスタックが動作するか"""
        model = torch.nn.Sequential(
            LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1),
            OrPooling2d(2),
            LutConv2d(8, 16, kernel_size=3, tree_depth=2, padding=1),
            OrPooling2d(2),
        )
        x = torch.rand(2, 1, 8, 8)
        y = model(x)
        assert y.shape == (2, 16, 2, 2)

    def test_full_cnn_model(self):
        """Conv → Pool → Flatten → LutLayer → GroupSum の完全モデル"""
        model = torch.nn.Sequential(
            LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1),
            OrPooling2d(2),
            LutConv2d(8, 16, kernel_size=3, tree_depth=2, padding=1),
            OrPooling2d(2),
            torch.nn.Flatten(),
            LutLayer(16 * 2 * 2, 20, n_inputs=2),
            GroupSum(k=2, tau=5.0),
        )
        x = torch.rand(4, 1, 8, 8)
        y = model(x)
        assert y.shape == (4, 2)

    def test_training_step(self):
        """1ステップの訓練が実行できるか"""
        model = torch.nn.Sequential(
            LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1),
            OrPooling2d(2),
            torch.nn.Flatten(),
            LutLayer(8 * 4 * 4, 20, n_inputs=2),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.rand(4, 1, 8, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_backward_flow(self):
        """勾配がConv→Pool→LutLayer→GroupSumを通じて流れるか"""
        conv = LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1)
        pool = OrPooling2d(2)
        lut = LutLayer(8 * 4 * 4, 20, n_inputs=2)
        gs = GroupSum(k=2, tau=5.0)

        x = torch.rand(2, 1, 8, 8)
        h = conv(x)
        h = pool(h)
        h = h.flatten(1)
        h = lut(h)
        y = gs(h)
        loss = y.sum()
        loss.backward()

        # Conv層の重みに勾配が流れているか
        assert conv.weights.grad is not None
        assert conv.weights.grad.shape == conv.weights.shape
        # LutLayer層の重みにも勾配が流れているか
        assert lut.weights.grad is not None

    def test_conv_pool_stack_n4(self):
        """N=4でのConv → Pool → Conv → Pool スタック"""
        model = torch.nn.Sequential(
            LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1, n_inputs=4),
            OrPooling2d(2),
            LutConv2d(8, 16, kernel_size=3, tree_depth=1, padding=1, n_inputs=4),
            OrPooling2d(2),
        )
        x = torch.rand(2, 1, 8, 8)
        y = model(x)
        assert y.shape == (2, 16, 2, 2)

    def test_training_step_n4(self):
        """N=4で1ステップの訓練が実行できるか"""
        model = torch.nn.Sequential(
            LutConv2d(1, 8, kernel_size=3, tree_depth=1, padding=1, n_inputs=4),
            OrPooling2d(2),
            torch.nn.Flatten(),
            LutLayer(8 * 4 * 4, 20, n_inputs=4),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.rand(4, 1, 8, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_mnist_like_architecture(self):
        """MNIST用アーキテクチャの構築・forward"""
        k = 4  # 小さいkでテスト
        model = torch.nn.Sequential(
            # Conv block 1: 1×28×28 → k×24×24
            LutConv2d(1, k, kernel_size=5, tree_depth=3, padding=0),
            OrPooling2d(2),  # → k×12×12
            # Conv block 2: k → 3k
            LutConv2d(k, 3 * k, kernel_size=3, tree_depth=3, padding=1),
            OrPooling2d(2),  # → 3k×6×6
            # Conv block 3: 3k → 9k
            LutConv2d(3 * k, 9 * k, kernel_size=3, tree_depth=3, padding=1),
            OrPooling2d(2),  # → 9k×3×3
            # Flatten + Dense
            torch.nn.Flatten(),
            LutLayer(9 * k * 3 * 3, 100, n_inputs=2),
            GroupSum(k=10, tau=5.0),
        )
        x = torch.rand(2, 1, 28, 28)
        y = model(x)
        assert y.shape == (2, 10)


# ============================================================
# LutConv2d: Independentパラメータ化
# ============================================================

class TestConvIndependent:
    def test_output_shape(self):
        """independent方式でも出力形状が正しいか"""
        conv = LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1,
                         parameterization='independent', residual_init=False)
        x = torch.rand(2, 1, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 8, 8)

    def test_eval_discrete(self):
        """independent eval時にsigmoid>0.5で離散化"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1,
                         parameterization='independent', residual_init=False)
        conv.eval()
        x = torch.rand(2, 1, 8, 8).round()
        y = conv(x)
        assert y.shape == (2, 4, 8, 8)

    def test_gradcheck(self):
        """independent方式の勾配数値検証"""
        conv = LutConv2d(
            1, 2, kernel_size=3, tree_depth=2, n_inputs=2, padding=1,
            residual_init=False, parameterization='independent',
        ).double()
        x = torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True)
        fn = lambda x: conv(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_backward_flow(self):
        """independent方式でConv→Pool→LutLayerの勾配が流れるか"""
        conv = LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1,
                         parameterization='independent', residual_init=False)
        pool = OrPooling2d(2)
        lut = LutLayer(8 * 4 * 4, 20, n_inputs=2, parameterization='independent')
        gs = GroupSum(k=2, tau=5.0)

        x = torch.rand(2, 1, 8, 8)
        h = conv(x)
        h = pool(h)
        h = h.flatten(1)
        h = lut(h)
        y = gs(h)
        loss = y.sum()
        loss.backward()

        assert conv.weights.grad is not None
        assert lut.weights.grad is not None

    def test_training_step(self):
        """independent方式で1ステップの訓練が実行できるか"""
        model = torch.nn.Sequential(
            LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1,
                      parameterization='independent', residual_init=False),
            OrPooling2d(2),
            torch.nn.Flatten(),
            LutLayer(8 * 4 * 4, 20, n_inputs=2, parameterization='independent'),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.rand(4, 1, 8, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_output_shape_n4(self):
        """N=4 independent方式での出力形状"""
        conv = LutConv2d(1, 8, kernel_size=3, tree_depth=1, padding=1,
                         n_inputs=4, parameterization='independent', residual_init=False)
        x = torch.rand(2, 1, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 8, 8)

    def test_invalid_parameterization(self):
        """不正なparameterization値でAssertionError"""
        with pytest.raises(AssertionError):
            LutConv2d(1, 4, kernel_size=3, tree_depth=2,
                      parameterization='invalid')


# ============================================================
# LutConv2d: IWPパラメータ化（LDDLGN方式）
# ============================================================

class TestConvIWP:
    def test_output_shape(self):
        """IWP方式でも出力形状が正しいか"""
        conv = LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1,
                         parameterization='iwp')
        x = torch.rand(2, 1, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 8, 8)

    def test_weights_shape(self):
        """IWPの重みが(out_ch, n_internal, n_inputs)であることを確認"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, n_inputs=2,
                         parameterization='iwp')
        # N=2, d=2 → 3 internal nodes
        assert conv.weights.shape == (4, 3, 2), \
            f'weights shape {conv.weights.shape} != (4, 3, 2)'

    def test_eval_discrete(self):
        """IWP eval時に離散化"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1,
                         parameterization='iwp')
        conv.eval()
        x = torch.rand(2, 1, 8, 8).round()
        y = conv(x)
        assert y.shape == (2, 4, 8, 8)

    def test_gradcheck(self):
        """IWP方式の勾配数値検証"""
        conv = LutConv2d(
            1, 2, kernel_size=3, tree_depth=2, n_inputs=2, padding=1,
            parameterization='iwp',
        ).double()
        x = torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True)
        fn = lambda x: conv(torch.sigmoid(x))
        assert torch.autograd.gradcheck(fn, x, atol=1e-4, rtol=1e-3)

    def test_backward_flow(self):
        """IWP方式でConv→Pool→LutLayerの勾配が流れるか"""
        conv = LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1,
                         parameterization='iwp')
        pool = OrPooling2d(2)
        lut = LutLayer(8 * 4 * 4, 20, n_inputs=2, parameterization='iwp')
        gs = GroupSum(k=2, tau=5.0)

        x = torch.rand(2, 1, 8, 8)
        h = conv(x)
        h = pool(h)
        h = h.flatten(1)
        h = lut(h)
        y = gs(h)
        loss = y.sum()
        loss.backward()

        assert conv.weights.grad is not None
        assert lut.weights.grad is not None

    def test_training_step(self):
        """IWP方式で1ステップの訓練が実行できるか"""
        model = torch.nn.Sequential(
            LutConv2d(1, 8, kernel_size=3, tree_depth=2, padding=1,
                      parameterization='iwp'),
            OrPooling2d(2),
            torch.nn.Flatten(),
            LutLayer(8 * 4 * 4, 20, n_inputs=2, parameterization='iwp'),
            GroupSum(k=2, tau=5.0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.rand(4, 1, 8, 8)
        target = torch.tensor([0, 1, 0, 1])

        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_parameter_count_reduction(self):
        """IWPのパラメータ数がsoftmaxより少ないことを確認"""
        conv_s = LutConv2d(1, 8, kernel_size=3, tree_depth=2, n_inputs=2,
                           parameterization='softmax', residual_init=False)
        conv_i = LutConv2d(1, 8, kernel_size=3, tree_depth=2, n_inputs=2,
                           parameterization='iwp')
        params_s = sum(p.numel() for p in conv_s.parameters())
        params_i = sum(p.numel() for p in conv_i.parameters())
        assert params_i < params_s, f'IWP({params_i}) >= softmax({params_s})'


# ============================================================
# LutConv2d: softmax16g パラメータ化（16ゲート表現力回復）
# ============================================================

from griffith_logic.core.basis import _build_gate_table_n2  # noqa: E402


def _gate_index(truth_table):
    """butterfly基底順の4エントリ真理値表 → gate_table中の行インデックス"""
    gt = _build_gate_table_n2()
    target = torch.tensor(truth_table, dtype=gt.dtype)
    idx = (gt == target).all(dim=1).nonzero().flatten()
    assert idx.numel() == 1
    return int(idx.item())


def _depth1_conv_eval_on_all_inputs(gate_truth_table):
    """depth-1のsoftmax16g convを、argmaxが指定ゲートになるよう設定し、
    (a,b)∈{0,1}² の全4通りをhard evalして出力を返す。

    butterfly基底順 k: 0→(a=0,b=0), 1→(a=1,b=0), 2→(a=0,b=1), 3→(a=1,b=1)。
    truth_table[k] = そのパターンでのゲート出力。
    """
    conv = LutConv2d(
        in_channels=1, out_channels=1, kernel_size=1,
        tree_depth=1, padding=0, n_inputs=2,
        parameterization='softmax16g', residual_init=False,
    )
    # depth1: n_leaves=2。1x1カーネルだと受容野が1画素で2リーフを別位置から取れない
    # ため、1x2カーネルにしてleaf0=左画素(a), leaf1=右画素(b)に固定する。
    conv = LutConv2d(
        in_channels=1, out_channels=1, kernel_size=(1, 2),
        tree_depth=1, padding=0, n_inputs=2,
        parameterization='softmax16g', residual_init=False,
    )
    # indices_flat を明示: unfold順 (c*kH*kW + h*kW + w) → 位置0(=a), 位置1(=b)
    conv.indices_flat.copy_(torch.tensor([[0, 1]], dtype=torch.int64))
    # argmaxが指定ゲートになるよう重み設定
    g_idx = _gate_index(gate_truth_table)
    with torch.no_grad():
        conv.weights.zero_()
        conv.weights[0, 0, g_idx] = 10.0
    conv.eval()

    outputs = {}
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            x = torch.tensor([[[[a, b]]]])  # (1,1,1,2)
            y = conv(x)  # (1,1,1,1)
            outputs[(int(a), int(b))] = round(float(y.item()))
    return outputs


class TestLutConv2dSoftmax16gExpressivity:
    """(a) 全16ゲートを表現できること（旧softmaxはmintermしか表現できない）"""

    def test_gate_A_passthrough(self):
        """ゲートA=[0,1,0,1]: 第1入力aをそのまま通す"""
        out = _depth1_conv_eval_on_all_inputs([0, 1, 0, 1])
        assert out == {(0, 0): 0, (1, 0): 1, (0, 1): 0, (1, 1): 1}

    def test_gate_OR(self):
        """OR=[0,1,1,1]"""
        out = _depth1_conv_eval_on_all_inputs([0, 1, 1, 1])
        assert out == {(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 1}

    def test_gate_XOR(self):
        """XOR=[0,1,1,0]"""
        out = _depth1_conv_eval_on_all_inputs([0, 1, 1, 0])
        assert out == {(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 0}

    def test_gate_NOT_A(self):
        """NOT-A=[1,0,1,0]: 第1入力の否定"""
        out = _depth1_conv_eval_on_all_inputs([1, 0, 1, 0])
        assert out == {(0, 0): 1, (1, 0): 0, (0, 1): 1, (1, 1): 0}


class TestLutConv2dSoftmax16gConsistency:
    """(b) 飽和ロジットでsoft forward ≈ hard forward"""

    def test_soft_hard_match_saturated(self):
        torch.manual_seed(0)
        conv = LutConv2d(
            in_channels=1, out_channels=3, kernel_size=3,
            tree_depth=2, padding=0, n_inputs=2,
            parameterization='softmax16g', residual_init=False,
        )
        # ロジットを±20相当に飽和させる（argmaxを鋭く）
        with torch.no_grad():
            conv.weights.mul_(0.0)
            idx = torch.randint(0, 16, conv.weights.shape[:2])
            for oc in range(conv.weights.shape[0]):
                for g in range(conv.weights.shape[1]):
                    conv.weights[oc, g, int(idx[oc, g])] = 20.0
        x = torch.rand(2, 1, 5, 5)

        conv.train()
        y_soft = conv(x)
        conv.eval()
        y_hard = conv(x)
        assert torch.allclose(y_soft, y_hard, atol=1e-6), \
            (y_soft - y_hard).abs().max().item()


class TestLutConv2dSoftmax16gResidualInit:
    """(d) Residual Initが活性を保存する（旧softmaxは減衰する）"""

    def test_residual_init_preserves_activation(self):
        """depth-3 softmax16g residual init は入力活性を~10%以内で保存する。
        比較: 旧softmax residual init は各レベルで≈0.5倍に減衰する。

        conv paddingや接続ランダム性の交絡を避けるため、木演算コア
        (_tree_forward_train)に一様leavesを直接与えて純粋な減衰特性を測る。
        実測(seed非依存、値は決定的): depth3で16g dev=7.6%、旧softmax dev=87.2%。
        """
        a = 0.7
        conv16 = LutConv2d(
            in_channels=1, out_channels=1, kernel_size=3,
            tree_depth=3, padding=0, n_inputs=2,
            parameterization='softmax16g', residual_init=True,
        )
        conv16.train()
        conv_old = LutConv2d(
            in_channels=1, out_channels=1, kernel_size=3,
            tree_depth=3, padding=0, n_inputs=2,
            parameterization='softmax', residual_init=True,
        )
        conv_old.train()

        leaves = torch.full((5, 1, 2 ** 3), a)  # (B*L, out_ch, n_leaves)
        y16 = conv16._tree_forward_train(leaves).mean().item()
        yold = conv_old._tree_forward_train(leaves).mean().item()

        # softmax16g: 入力活性を~10%以内で保存（実測7.6%に10%余裕）
        assert abs(y16 - a) / a < 0.10, \
            f'softmax16g residual init が活性を保存しない: {y16:.4f} (target {a})'
        # 旧softmax: 3レベルで≈0.5^3累積の著しい減衰（保存されない）
        assert yold < 0.5 * a, \
            f'旧softmax residual initが期待どおり減衰していない: {yold:.4f}'
        assert yold < y16, f'16gが旧softmaxより保存的でない: old={yold}, 16g={y16}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
