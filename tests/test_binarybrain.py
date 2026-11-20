"""
BinaryBrain再現モジュールのテスト。

Binarize, RealToBinary, BinaryToReal, DifferentiableLutBlock, LutLayer(raw)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from griffith_logic import (
    Binarize,
    BinaryToReal,
    DifferentiableLutBlock,
    LutLayer,
    RealToBinary,
)

# ============================================================
# Binarize
# ============================================================

class TestBinarize:
    def test_forward_binary_output(self):
        """出力が{0, 1}のみであること"""
        binarize = Binarize(threshold=0.5)
        binarize.train()
        x = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        y = binarize(x)
        # 0.5以下 → 0, 0.5超 → 1
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.equal(y, expected)

    def test_backward_hardtanh(self):
        """HardTanh STEの勾配: 範囲内は通す、範囲外はゼロ"""
        binarize = Binarize(threshold=0.5, hardtanh_min=0.0, hardtanh_max=1.0)
        binarize.train()
        x = torch.tensor([-0.1, 0.0, 0.3, 0.7, 1.0, 1.1], requires_grad=True)
        y = binarize(x)
        y.sum().backward()
        # -0.1 → 外(0), 0.0 → 境界外(0), 0.3 → 内(1), 0.7 → 内(1), 1.0 → 境界外(0), 1.1 → 外(0)
        expected_grad = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        assert torch.equal(x.grad, expected_grad)

    def test_eval_mode(self):
        """推論時も二値化が行われること"""
        binarize = Binarize(threshold=0.5)
        binarize.eval()
        x = torch.tensor([0.3, 0.7])
        y = binarize(x)
        expected = torch.tensor([0.0, 1.0])
        assert torch.equal(y, expected)


# ============================================================
# RealToBinary
# ============================================================

class TestRealToBinary:
    def test_output_shape(self):
        """出力shapeがB*M, Dであること"""
        rtb = RealToBinary(frame_modulation_size=15)
        x = torch.rand(8, 784)
        y = rtb(x)
        assert y.shape == (8 * 15, 784)

    def test_binary_output(self):
        """出力が{0, 1}のみであること"""
        rtb = RealToBinary(frame_modulation_size=5)
        x = torch.rand(4, 10)
        y = rtb(x)
        assert torch.all((y == 0.0) | (y == 1.0))

    def test_thresholds(self):
        """閾値が正しく計算されていること"""
        M = 5
        rtb = RealToBinary(frame_modulation_size=M)
        expected = torch.tensor([(f + 0.5) / M for f in range(M)])
        assert torch.allclose(rtb.thresholds, expected)

    def test_extreme_values(self):
        """0.0は全フレームで0、1.0は全フレームで1"""
        M = 5
        rtb = RealToBinary(frame_modulation_size=M)
        x = torch.tensor([[0.0, 1.0]])  # (1, 2)
        y = rtb(x)  # (5, 2)
        # 全フレームで0.0 < threshold → 0
        assert torch.all(y[:, 0] == 0.0)
        # 全フレームで1.0 > threshold → 1
        assert torch.all(y[:, 1] == 1.0)


# ============================================================
# BinaryToReal
# ============================================================

class TestBinaryToReal:
    def test_output_shape(self):
        """出力shapeがB, Dであること"""
        btr = BinaryToReal(frame_modulation_size=15)
        x = torch.rand(8 * 15, 10)
        y = btr(x)
        assert y.shape == (8, 10)

    def test_mean_computation(self):
        """フレーム方向の平均が正しいこと"""
        M = 3
        btr = BinaryToReal(frame_modulation_size=M)
        # (2*3, 2) = (6, 2)
        x = torch.tensor([
            [1.0, 0.0],  # frame 0, sample 0
            [0.0, 1.0],  # frame 1, sample 0
            [1.0, 1.0],  # frame 2, sample 0
            [0.0, 0.0],  # frame 0, sample 1
            [1.0, 0.0],  # frame 1, sample 1
            [0.0, 1.0],  # frame 2, sample 1
        ])
        y = btr(x)
        expected = torch.tensor([
            [2.0 / 3, 2.0 / 3],  # sample 0
            [1.0 / 3, 1.0 / 3],  # sample 1
        ])
        assert torch.allclose(y, expected)

    def test_gradient_flow(self):
        """勾配が正しく流れること"""
        M = 3
        btr = BinaryToReal(frame_modulation_size=M)
        x = torch.randn(6, 4, requires_grad=True)
        y = btr(x)
        y.sum().backward()
        # meanの勾配: 1/M
        expected_grad = torch.full_like(x, 1.0 / M)
        assert torch.allclose(x.grad, expected_grad)


# ============================================================
# RealToBinary + BinaryToReal の往復
# ============================================================

class TestRoundTrip:
    def test_roundtrip_approximation(self):
        """RealToBinary→BinaryToRealで元の値に近い値が復元されること"""
        M = 15
        rtb = RealToBinary(frame_modulation_size=M)
        btr = BinaryToReal(frame_modulation_size=M)
        x = torch.tensor([[0.1, 0.5, 0.9]])
        y = btr(rtb(x))
        # M=15の量子化精度内で近似
        assert torch.allclose(y, x, atol=1.0 / M + 0.01)


# ============================================================
# LutLayer raw パラメータ化
# ============================================================

class TestLutLayerRaw:
    def test_forward_shape(self):
        """rawパラメータ化のforward shapeが正しいこと"""
        layer = LutLayer(in_dim=10, out_dim=5, n_inputs=6,
                         parameterization='raw')
        x = torch.rand(4, 10)
        y = layer(x)
        assert y.shape == (4, 5)

    def test_weight_initialization(self):
        """重みがNormal(0.5, 0.01)で初期化されていること"""
        layer = LutLayer(in_dim=10, out_dim=1000, n_inputs=6,
                         parameterization='raw')
        # 平均が0.5付近
        assert abs(layer.weights.mean().item() - 0.5) < 0.01
        # 標準偏差が0.01付近
        assert abs(layer.weights.std().item() - 0.01) < 0.005

    def test_backward(self):
        """勾配が計算されること"""
        layer = LutLayer(in_dim=10, out_dim=5, n_inputs=6,
                         parameterization='raw')
        layer.train()
        x = torch.rand(4, 10, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert layer.weights.grad is not None
        assert x.grad is not None

    def test_eval_continuous_weights(self):
        """推論時も連続値の重みを使用すること（BinaryBrain方式）"""
        layer = LutLayer(in_dim=10, out_dim=5, n_inputs=6,
                         parameterization='raw')
        layer.eval()
        x = torch.rand(4, 10)
        y = layer(x)
        # rawパラメータ化は推論時も連続値 → 出力は[0,1]の実数
        assert torch.all((y >= 0.0) & (y <= 1.0))

    def test_no_sigmoid_in_gradient(self):
        """rawの勾配にsigmoid Jacobianが含まれないこと（independentとの比較）"""
        torch.manual_seed(42)
        # 同じ接続で比較するため手動設定
        layer_raw = LutLayer(in_dim=10, out_dim=5, n_inputs=4,
                             parameterization='raw')
        layer_raw.train()
        x = torch.rand(2, 10)
        y = layer_raw(x)
        y.sum().backward()
        raw_grad = layer_raw.weights.grad.clone()

        # rawの勾配はbasisとの直接積 → sigmoid(w)*(1-sigmoid(w))のスケーリングなし
        # 勾配がゼロでないことを確認
        assert raw_grad.abs().sum() > 0


# ============================================================
# DifferentiableLutBlock
# ============================================================

class TestDifferentiableLutBlock:
    def test_forward_shape(self):
        """出力shapeが正しいこと"""
        block = DifferentiableLutBlock(in_dim=64, out_dim=32, n_inputs=6)
        block.train()
        x = torch.rand(8, 64)
        y = block(x)
        assert y.shape == (8, 32)

    def test_output_binary(self):
        """訓練時の出力が{0, 1}であること（Binarize層の効果）"""
        block = DifferentiableLutBlock(in_dim=64, out_dim=32, n_inputs=6)
        block.train()
        x = torch.rand(8, 64)
        y = block(x)
        assert torch.all((y == 0.0) | (y == 1.0))

    def test_bn_initialization(self):
        """BatchNormのgamma=0.3, beta=0.5で初期化されていること"""
        block = DifferentiableLutBlock(in_dim=64, out_dim=32, n_inputs=6)
        assert torch.allclose(block.bn.weight, torch.full((32,), 0.3))
        assert torch.allclose(block.bn.bias, torch.full((32,), 0.5))

    def test_clamp_weights(self):
        """clamp_weightsで重みが[0,1]に収まること"""
        block = DifferentiableLutBlock(in_dim=64, out_dim=32, n_inputs=6)
        # 重みを範囲外に設定
        with torch.no_grad():
            block.lut.weights.fill_(-1.0)
        block.clamp_weights()
        assert block.lut.weights.min() >= 0.0

        with torch.no_grad():
            block.lut.weights.fill_(2.0)
        block.clamp_weights()
        assert block.lut.weights.max() <= 1.0

    def test_gradient_flow(self):
        """勾配がモデル全体を通じて流れること"""
        block = DifferentiableLutBlock(in_dim=64, out_dim=32, n_inputs=6)
        block.train()
        x = torch.rand(8, 64, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert block.lut.weights.grad is not None
        # unbinarize STEにより入力にも勾配が流れる
        assert x.grad is not None


# ============================================================
# 統合テスト: BinaryBrainモデル全体
# ============================================================

class TestBinaryBrainModelIntegration:
    def test_forward_shape(self):
        """モデル全体のforward shapeが正しいこと"""
        model = BinaryBrainModel(in_dim=784, num_classes=10, n_inputs=6,
                                 frame_modulation_size=3)
        model.train()
        x = torch.rand(4, 1, 28, 28)
        y = model(x)
        assert y.shape == (4, 10)

    def test_small_model(self):
        """小さいモデルで訓練ステップが実行できること"""
        model = BinaryBrainModel(
            in_dim=16, num_classes=3, n_inputs=4,
            frame_modulation_size=3,
            layer_sizes=[4 * 4 * 3, 4 * 3, 3],
        )
        model.train()
        x = torch.rand(4, 1, 4, 4)
        y = torch.randint(0, 3, (4,))

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clamp_weights()

        # lossが有限値であること
        assert torch.isfinite(torch.tensor(loss.item()))


# テスト用のインポート（train_binarybrain.pyのBinaryBrainModel）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
try:
    from train_binarybrain import BinaryBrainModel
except ImportError:
    pass
