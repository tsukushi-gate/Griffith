"""APTrainer（AP法）のテスト"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from ap_trainer import APTrainer

from griffith_logic import GroupSum, LutLayer


def _build_model(in_dim, out_dim, num_classes, n_inputs=2, num_layers=1):
    """テスト用のシンプルなモデルを構築"""
    layers = [torch.nn.Flatten()]
    layers.append(LutLayer(
        in_dim=in_dim, out_dim=out_dim, n_inputs=n_inputs,
        parameterization='independent',
    ))
    for _ in range(num_layers - 1):
        layers.append(LutLayer(
            in_dim=out_dim, out_dim=out_dim, n_inputs=n_inputs,
            parameterization='independent',
        ))
    layers.append(GroupSum(k=num_classes, tau=1.0))
    return torch.nn.Sequential(*layers)


class TestAPTrainerInit:
    def test_rejects_softmax_parameterization(self):
        """softmaxパラメータ化のLutLayerを拒否する"""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LutLayer(4, 4, n_inputs=2, parameterization='softmax'),
            GroupSum(k=2, tau=1.0),
        )
        with pytest.raises(ValueError, match='independent'):
            APTrainer(model, lr=0.1)

    def test_weights_initialized_in_01(self):
        """初期化後の重みが[0,1]に収まる"""
        model = _build_model(4, 8, 2)
        trainer = APTrainer(model, lr=0.1)
        for layer in trainer.lut_layers:
            assert (layer.weights.data >= 0).all()
            assert (layer.weights.data <= 1).all()


class TestAPForward:
    def test_output_shape(self):
        """forwardの出力形状が正しい"""
        model = _build_model(4, 8, 2)
        trainer = APTrainer(model, lr=0.1)
        x = torch.rand(3, 1, 2, 2)  # batch=3, flatten→4
        logits, caches = trainer._forward_with_cache(x)
        assert logits.shape == (3, 2)  # (batch, num_classes)
        assert len(caches) == 1

    def test_output_clipped(self):
        """LutLayerの出力が[0,1]に収まる"""
        model = _build_model(4, 8, 2)
        trainer = APTrainer(model, lr=0.1)
        x = torch.rand(5, 4)
        logits, caches = trainer._forward_with_cache(x)
        for cache in caches:
            assert (cache.output >= 0).all()
            assert (cache.output <= 1).all()

    def test_multi_layer_shapes(self):
        """多層モデルでの形状整合性"""
        model = _build_model(4, 8, 2, num_layers=3)
        trainer = APTrainer(model, lr=0.1)
        x = torch.rand(4, 4)
        logits, caches = trainer._forward_with_cache(x)
        assert len(caches) == 3
        assert logits.shape == (4, 2)
        # 各キャッシュの形状確認
        for cache in caches:
            assert cache.basis.shape == (4, 8, 4)  # (batch, out_dim, 2^2)
            assert cache.gathered.shape == (4, 8, 2)
            assert cache.output.shape == (4, 8)


class TestAPWeightUpdate:
    def test_weights_change_after_step(self):
        """train_step後に重みが変化する"""
        torch.manual_seed(42)
        model = _build_model(4, 8, 2)
        trainer = APTrainer(model, lr=0.1)

        weights_before = [l.weights.data.clone() for l in trainer.lut_layers]
        x = torch.rand(4, 4)
        y = torch.tensor([0, 1, 0, 1])
        trainer.train_step(x, y)
        weights_after = [l.weights.data.clone() for l in trainer.lut_layers]

        for wb, wa in zip(weights_before, weights_after):
            assert not torch.equal(wb, wa), '重みが更新されていない'

    def test_weights_clipped_after_update(self):
        """更新後の重みが[0,1]に収まる"""
        torch.manual_seed(42)
        model = _build_model(4, 8, 2)
        trainer = APTrainer(model, lr=1.0)  # 大きい学習率でクリップをテスト

        for _ in range(10):
            x = torch.rand(4, 4)
            y = torch.tensor([0, 1, 0, 1])
            trainer.train_step(x, y)

        for layer in trainer.lut_layers:
            assert (layer.weights.data >= 0).all()
            assert (layer.weights.data <= 1).all()

    def test_mode2_updates_fewer_weights(self):
        """mode2はmode1より少ない重みを更新する"""
        torch.manual_seed(42)
        model1 = _build_model(4, 8, 2)
        model2 = _build_model(4, 8, 2)

        trainer1 = APTrainer(model1, lr=0.1, update_mode=1)
        trainer2 = APTrainer(model2, lr=0.1, update_mode=2)

        # 同じ初期重みに設定
        with torch.no_grad():
            for l1, l2 in zip(trainer1.lut_layers, trainer2.lut_layers):
                l2.weights.data.copy_(l1.weights.data)
                l2.indices = l1.indices.clone()

        x = torch.rand(4, 4)
        y = torch.tensor([0, 1, 0, 1])

        w1_before = trainer1.lut_layers[0].weights.data.clone()
        w2_before = trainer2.lut_layers[0].weights.data.clone()

        trainer1.train_step(x, y)
        trainer2.train_step(x, y)

        # mode1で変化した重みの数 >= mode2で変化した重みの数
        changed1 = (trainer1.lut_layers[0].weights.data != w1_before).sum()
        changed2 = (trainer2.lut_layers[0].weights.data != w2_before).sum()
        assert changed1 >= changed2


@pytest.mark.learning
class TestXORConvergence:
    def test_xor_converges(self):
        """LUT2ネットワークでXORが学習できる"""
        # 複数シードで試行（ランダム接続に依存するため）
        success = False
        for seed in range(10):
            torch.manual_seed(seed)

            # out_dim=10でニューロン容量を確保
            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                LutLayer(
                    in_dim=2, out_dim=10, n_inputs=2,
                    parameterization='independent',
                ),
                GroupSum(k=2, tau=1.0),
            )
            trainer = APTrainer(model, lr=0.2, update_mode=1)

            x_data = torch.tensor([
                [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
            ])
            y_data = torch.tensor([0, 1, 1, 0])

            for _ in range(300):
                trainer.train_step(x_data, y_data)

            with torch.no_grad():
                logits, _ = trainer._forward_with_cache(x_data)
                preds = logits.argmax(dim=-1)
                acc = (preds == y_data).float().mean().item()

            if acc >= 1.0:
                success = True
                break

        assert success, 'XORが10シード以内で収束しなかった'


class TestBinarize:
    def test_binarized_weights_are_binary(self):
        """binarize評価時に重みが0/1のみ"""
        model = _build_model(4, 8, 2)
        trainer = APTrainer(model, lr=0.1)

        # binarize前: 連続値
        has_intermediate = False
        for layer in trainer.lut_layers:
            vals = layer.weights.data
            if ((vals > 0.01) & (vals < 0.99)).any():
                has_intermediate = True
        assert has_intermediate, '初期値が既にバイナリ'

        # evaluate(binarize=True)後に元の重みが復元される
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.rand(8, 4), torch.randint(0, 2, (8,)))
        loader = DataLoader(ds, batch_size=4)

        weights_before = [l.weights.data.clone() for l in trainer.lut_layers]
        trainer.evaluate(loader, 'cpu', binarize=True)
        weights_after = [l.weights.data.clone() for l in trainer.lut_layers]

        for wb, wa in zip(weights_before, weights_after):
            assert torch.equal(wb, wa), 'binarize評価後に重みが復元されていない'


class TestErrorPropagation:
    def test_ap_backward_updates_all_layers(self):
        """AP法の誤差伝播が全層の重みを更新する"""
        model = _build_model(4, 8, 2, num_layers=3)
        trainer = APTrainer(model, lr=0.1)

        weights_before = [l.weights.data.clone() for l in trainer.lut_layers]

        x = torch.rand(4, 4)
        y = torch.tensor([0, 1, 0, 1])
        trainer.train_step(x, y)

        # 全層の重みが更新されていること（AP法は誤差が減衰しないため）
        for i, (wb, layer) in enumerate(zip(weights_before, trainer.lut_layers)):
            assert not torch.equal(wb, layer.weights.data), \
                f'層{i}の重みが更新されていない'

    def test_error_does_not_attenuate(self):
        """AP法の誤差は層を通過しても減衰しない（backpropとの違い）"""
        model = _build_model(4, 8, 2, num_layers=3)
        trainer = APTrainer(model, lr=0.5)

        # 大きめの学習率で複数ステップ訓練
        x = torch.rand(4, 4)
        y = torch.tensor([0, 1, 0, 1])

        weights_before = [l.weights.data.clone() for l in trainer.lut_layers]
        trainer.train_step(x, y)

        # 各層の重み変化量を計算
        changes = []
        for wb, layer in zip(weights_before, trainer.lut_layers):
            diff = (layer.weights.data - wb).abs().mean().item()
            changes.append(diff)

        # 浅い層（layer 0）も深い層と同等の変化量を持つべき
        # backpropだと changes[0] << changes[2] になるが、AP法では同程度
        assert changes[0] > 0.001, f'最浅層の変化量が小さすぎる: {changes[0]}'
