"""LogicResNet(残差論理ConvNet)のテスト。

1. shape契約  2. 勾配疎通  3. hard離散性
4. 残差の恒等性スモーク(pass-through寄り初期化+手動配線でOR残差が恒等になることを確認)
5. 収束(極小構成の合成2クラス画像でhard acc=100%)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.models.logic_resnet import (
    LogicResBlock,
    LogicResNet,
    LogicResNetConfig,
)


def _tiny_config(**overrides) -> LogicResNetConfig:
    """テスト用の極小構成(高速)。"""
    base = dict(
        in_channels=3,
        encode_bits=2,
        stem_channels=4,
        stage_blocks=(1, 1),
        residual_mode='or',
        n_inputs=2,
        parameterization='softmax',
        num_classes=4,
        kernel_size=3,
        tree_depth=1,
        head_group_size=2,
        input_channels_per_tree=2,
    )
    base.update(overrides)
    return LogicResNetConfig(**base)


# ============================================================
# 1. shape契約
# ============================================================
class TestShapeContract:
    @pytest.mark.parametrize('stage_blocks', [(1,), (2, 2), (1, 2, 1)])
    @pytest.mark.parametrize('residual_mode', ['or', 'xor'])
    def test_output_shape(self, stage_blocks, residual_mode):
        config = _tiny_config(stage_blocks=stage_blocks, residual_mode=residual_mode)
        model = LogicResNet(config)
        x = torch.rand(2, 3, 8, 8)
        y = model(x)
        assert y.shape == (2, config.num_classes), (
            f'stage_blocks={stage_blocks}, residual_mode={residual_mode}: '
            f'shape={y.shape}'
        )

    def test_invalid_residual_mode(self):
        with pytest.raises(ValueError):
            LogicResNet(_tiny_config(residual_mode='and'))

    def test_empty_stage_blocks(self):
        with pytest.raises(ValueError):
            LogicResNet(_tiny_config(stage_blocks=()))

    def test_downsample_count(self):
        """OrPooling2dダウンサンプルはstage間(stage数-1回)だけ挿入されるか"""
        config = _tiny_config(stage_blocks=(1, 1, 1))
        model = LogicResNet(config)
        assert len(model.downsamples) == 2
        assert len(model.stages) == 3


# ============================================================
# 2. 勾配疎通
# ============================================================
class TestGradientFlow:
    @pytest.mark.parametrize('residual_mode', ['or', 'xor'])
    def test_backward_reaches_stem_and_blocks(self, residual_mode):
        config = _tiny_config(stage_blocks=(2, 2), residual_mode=residual_mode)
        model = LogicResNet(config)
        x = torch.rand(2, 3, 8, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert model.stem.weights.grad is not None
        assert torch.any(model.stem.weights.grad != 0), 'stemに勾配が届いていない'

        for stage in model.stages:
            for block in stage:
                assert isinstance(block, LogicResBlock)
                for conv in block.residual.inner:
                    assert conv.weights.grad is not None

        assert model.head.weights.grad is not None


# ============================================================
# 3. hard離散性
# ============================================================
class TestHardDiscreteness:
    def test_forward_features_binary_in_eval(self):
        config = _tiny_config(stage_blocks=(2, 2), residual_mode='or')
        model = LogicResNet(config)
        model.eval()
        x = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            h = model.forward_features(x)
        assert torch.all((h == 0.0) | (h == 1.0)), (
            f'eval時のhead出力が0/1でない: unique={h.unique()}'
        )

    def test_thermometer_encode_always_binary(self):
        """thermometer符号は比較ベースのため train/eval 問わず常に0/1"""
        config = _tiny_config()
        model = LogicResNet(config)
        model.train()
        x = torch.rand(2, 3, 8, 8)
        code = model.encode(x)
        assert torch.all((code == 0.0) | (code == 1.0))

    def test_group_sum_output_is_integer_valued(self):
        """SpatialGroupSum→GroupSumの最終出力は0/1個数の合計なので整数値になる"""
        config = _tiny_config(stage_blocks=(1,), residual_mode='or')
        model = LogicResNet(config)
        model.eval()
        x = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            y = model(x)
        assert torch.all(y == y.round()), f'最終出力が整数値でない: {y}'


# ============================================================
# 4. 残差の恒等性スモーク
# ============================================================
class TestResidualIdentity:
    def test_or_residual_is_identity_when_inner_passthrough(self):
        """innerの2層を手動配線でchannel-wise恒等にすると、OR残差は入力を保存する。

        n_inputs=2, tree_depth=1 は 2leaf→1internal-nodeのLUTで、
        residual_init=Trueは gate'A'(leaf0をそのまま通す)にバイアスする。
        indices_flat[:,0]をチャネルkそのものに固定すれば、evalモードで
        各LutConv2dはchannel-wise恒等写像になる(kernel=1で空間ズレも無い)。
        2層恒等を合成してもinner(x)==xなので、OR残差 merge(x, x)=x が厳密に成り立つ
        ({0,1}入力に対する1-(1-a)(1-b)はa=bで冪等)。
        """
        channels = 4
        config = _tiny_config(
            stem_channels=channels,
            stage_blocks=(1,),
            residual_mode='or',
            n_inputs=2,
            tree_depth=1,
            kernel_size=1,
            residual_init=True,
        )
        block = LogicResBlock(config, channels)
        block.eval()

        for conv in block.residual.inner:
            assert conv.indices_flat.shape == (channels, 2)
            # leaf0(gate'A'が選ぶ側)をチャネル自身に固定 → channel-wise恒等
            with torch.no_grad():
                conv.indices_flat[:, 0] = torch.arange(channels)

        x = (torch.rand(3, channels, 5, 5) > 0.5).float()
        with torch.no_grad():
            y = block(x)
        assert torch.equal(y, x), 'pass-through配線のOR残差ブロックが恒等写像でない'

    def test_gradient_reaches_stem_through_many_blocks(self):
        """残差接続を持つ深いスタックでも、勾配がstemまで消えずに届くか"""
        config = _tiny_config(stage_blocks=(4,), residual_mode='or')
        model = LogicResNet(config)
        x = torch.rand(2, 3, 8, 8, requires_grad=False)
        y = model(x)
        y.sum().backward()
        grad_norm = model.stem.weights.grad.norm().item()
        assert grad_norm > 0.0, 'stemまで勾配が届いていない(深いスタックで勾配消失)'


# ============================================================
# 5. 収束(極小構成・合成2クラス画像・hard acc=100%)
# ============================================================
class TestConvergence:
    def _make_dataset(self, n_per_class: int = 8, size: int = 4):
        """左右反転パターンの合成2クラス画像。

        class0: 左半分=1, 右半分=0 / class1: 右半分=1, 左半分=0
        """
        half = size // 2
        xs, ys = [], []
        g = torch.Generator().manual_seed(0)
        for _ in range(n_per_class):
            img0 = torch.zeros(1, size, size)
            img0[:, :, :half] = 1.0
            img0 += 0.05 * torch.randn(1, size, size, generator=g)
            xs.append(img0.clamp(0, 1))
            ys.append(0)

            img1 = torch.zeros(1, size, size)
            img1[:, :, half:] = 1.0
            img1 += 0.05 * torch.randn(1, size, size, generator=g)
            xs.append(img1.clamp(0, 1))
            ys.append(1)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def test_hard_acc_reaches_100(self):
        torch.manual_seed(0)
        config = _tiny_config(
            in_channels=1,
            encode_bits=2,
            stem_channels=4,
            stage_blocks=(1,),
            residual_mode='or',
            n_inputs=2,
            tree_depth=1,
            kernel_size=3,
            head_group_size=2,
            num_classes=2,
            input_channels_per_tree=1,
        )
        model = LogicResNet(config)
        x, y = self._make_dataset(n_per_class=8, size=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        hard_acc = 0.0
        for _step in range(300):
            model.train()
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(x).argmax(dim=-1)
                hard_acc = (pred == y).float().mean().item()
            if hard_acc >= 1.0:
                break

        assert hard_acc == 1.0, f'収束せず: hard_acc={hard_acc}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
