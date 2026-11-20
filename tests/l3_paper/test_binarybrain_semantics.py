"""L3: BinaryBrain(Fuchikami)意味論の再現検証。

DifferentiableLutBlock が BinaryBrain の DifferentiableLutN と同じ
構成規約を守っていることを固定する:

    Unbinarize: {0,1} → {0.5±bias}(デフォルト {0.25, 0.75})、STE
    LUT(raw):   重みを[0,1]で直接保持(sigmoidなし)
    BatchNorm:  γ=0.3, β=0.5 初期化、momentum=1.0(EMAなし=バッチ統計)
    Binarize:   閾値0.5、HardTanh STE [0,1]
    clamp:      重みは[0,1]にクランプ(m_flagClamp)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import DifferentiableLutBlock
from griffith_logic.differentiable_lut_block import UnbinarizeFunction


class TestUnbinarize:
    def test_binary_remap(self):
        x = torch.tensor([0.0, 1.0, 0.3, 0.7])
        y = UnbinarizeFunction.apply(x, 0.25)
        assert torch.equal(y, torch.tensor([0.25, 0.75, 0.25, 0.75]))

    def test_ste_backward_identity(self):
        x = torch.tensor([0.0, 1.0, 0.4], requires_grad=True)
        y = UnbinarizeFunction.apply(x, 0.25)
        g = torch.tensor([1.0, 2.0, 3.0])
        (gx,) = torch.autograd.grad(y, (x,), grad_outputs=g)
        assert torch.equal(gx, g)  # STE: 勾配素通し


class TestBlockSemantics:
    def test_default_construction(self, seed):
        block = DifferentiableLutBlock(64, 32)
        assert block.lut.parameterization == 'raw'
        assert block.lut.n_inputs == 6
        assert block.unbinarize_bias == 0.25
        # BN初期値: γ=0.3, β=0.5, momentum=1.0
        assert torch.all(block.bn.weight == 0.3)
        assert torch.all(block.bn.bias == 0.5)
        assert block.bn.momentum == 1.0
        # raw重み初期化: Normal(0.5, 0.01) → ほぼ全て[0,1]内
        assert block.lut.weights.mean().item() == pytest.approx(0.5, abs=0.01)

    def test_bn_momentum_1_equals_batch_stats(self, seed):
        """momentum=1.0: running統計が直近バッチ統計と一致(EMAなし)"""
        block = DifferentiableLutBlock(32, 16)
        block.train()
        x = torch.randint(0, 2, (128, 32)).float()
        # LUT出力のバッチ統計を直接計算して照合
        with torch.no_grad():
            h = UnbinarizeFunction.apply(x, 0.25)
            lut_out = block.lut(h)
        block(x)
        assert torch.allclose(
            block.bn.running_mean, lut_out.mean(dim=0), atol=1e-5,
        )

    def test_clamp_weights(self, seed):
        block = DifferentiableLutBlock(32, 16)
        with torch.no_grad():
            block.lut.weights += 3.0  # 範囲外に飛ばす
        block.clamp_weights()
        w = block.lut.weights
        assert w.min() >= 0.0 and w.max() <= 1.0

    def test_output_is_binary(self, seed):
        block = DifferentiableLutBlock(32, 16)
        x = torch.randint(0, 2, (8, 32)).float()
        for mode in (True, False):
            block.train(mode)
            y = block(x)
            assert torch.all((y == 0.0) | (y == 1.0)), f'training={mode}'

    def test_gradient_flows_through_block(self, seed):
        """STEチェーン(Unbinarize→LUT→BN→Binarize)で勾配が生きている"""
        block = DifferentiableLutBlock(32, 16)
        block.train()
        x = torch.randint(0, 2, (16, 32)).float().requires_grad_(True)
        y = block(x)
        y.sum().backward()
        assert block.lut.weights.grad is not None
        assert block.lut.weights.grad.abs().sum() > 0
        assert x.grad is not None


@pytest.mark.paper
class TestBinaryBrainMnist:
    def test_mnist_subset_accuracy(self, seed):
        """MNIST 20kサブセット・10エポックで hard acc ≥ 85%

        (実測較正: 10k×3epoch=0.817、20k×5epoch=0.848 → 10epochで閾値0.85)
        """
        import torchvision
        from torch.utils.data import DataLoader, Subset

        from griffith_logic import GroupSum

        tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: (t > 0.5).float().flatten()),
        ])
        root = os.path.join(os.path.dirname(__file__), '..', '..', 'data-mnist')
        train = Subset(
            torchvision.datasets.MNIST(root, train=True, download=True,
                                       transform=tfm),
            range(20000),
        )
        test = Subset(
            torchvision.datasets.MNIST(root, train=False, transform=tfm),
            range(2000),
        )
        train_loader = DataLoader(train, batch_size=256, shuffle=True)
        test_loader = DataLoader(test, batch_size=512)

        torch.manual_seed(1)
        model = nn.Sequential(
            DifferentiableLutBlock(784, 2048),
            DifferentiableLutBlock(2048, 640),
            DifferentiableLutBlock(640, 100),
            GroupSum(k=10, tau=3.0),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(10):
            model.train()
            for x, y in train_loader:
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                for m in model:
                    if isinstance(m, DifferentiableLutBlock):
                        m.clamp_weights()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        acc = correct / total
        assert acc >= 0.85, f'BinaryBrain MNIST hard acc = {acc:.4f} < 0.85'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
