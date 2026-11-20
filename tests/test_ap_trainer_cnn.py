"""CNNAPTrainer（CDD-LGN AP法）のテスト"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from ap_trainer_cnn import (
    CNNAPTrainer,
    _conv_backward_to_input,
    _conv_forward_ap,
    _conv_update_ap,
    _orpool_backward_ap,
    _orpool_forward_ap,
)

from griffith_logic import GroupSum, LutConv2d, LutLayer, OrPooling2d


def _build_small_cnn():
    """テスト用の小規模CDD-LGNモデル"""
    k = 4
    return torch.nn.Sequential(
        LutConv2d(1, k, kernel_size=3, tree_depth=2, padding=1,
                  n_inputs=2, parameterization='independent', residual_init=False),
        OrPooling2d(2),  # 14→7
        torch.nn.Flatten(),
        LutLayer(k * 7 * 7, 100, n_inputs=2, parameterization='independent'),
        LutLayer(100, 50, n_inputs=2, parameterization='independent'),
        GroupSum(k=10, tau=1.0),
    )


class TestConvForward:
    def test_output_shape(self):
        """Conv forwardの出力形状"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1,
                         n_inputs=2, parameterization='independent',
                         residual_init=False)
        conv.weights.data.uniform_(0, 1)
        x = torch.rand(2, 1, 14, 14)
        out, cache = _conv_forward_ap(conv, x)
        assert out.shape == (2, 4, 14, 14)
        assert len(cache.level_caches) == 2  # tree_depth=2

    def test_output_in_01(self):
        """Conv出力が[0,1]範囲"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1,
                         n_inputs=2, parameterization='independent',
                         residual_init=False)
        conv.weights.data.uniform_(0, 1)
        x = torch.rand(2, 1, 14, 14)
        out, _ = _conv_forward_ap(conv, x)
        assert (out >= 0).all() and (out <= 1).all()


class TestOrPool:
    def test_forward_backward_shape(self):
        """OrPool forward/backwardの形状"""
        pool = OrPooling2d(2)
        x = torch.rand(2, 4, 14, 14)
        out, cache = _orpool_forward_ap(pool, x)
        assert out.shape == (2, 4, 7, 7)

        error = torch.rand_like(out)
        input_error = _orpool_backward_ap(error, cache)
        assert input_error.shape == x.shape


class TestConvUpdate:
    def test_weights_change(self):
        """Conv AP更新で重みが変化する"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1,
                         n_inputs=2, parameterization='independent',
                         residual_init=False)
        conv.weights.data.uniform_(0, 1)
        x = torch.rand(2, 1, 14, 14)
        out, cache = _conv_forward_ap(conv, x)

        w_before = conv.weights.data.clone()
        error = torch.randn_like(out)
        _conv_update_ap(cache, error, lr=0.1)

        assert not torch.equal(w_before, conv.weights.data)

    def test_weights_clipped(self):
        """更新後の重みが[0,1]"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1,
                         n_inputs=2, parameterization='independent',
                         residual_init=False)
        conv.weights.data.uniform_(0, 1)
        x = torch.rand(2, 1, 14, 14)
        out, cache = _conv_forward_ap(conv, x)

        error = torch.randn_like(out) * 10
        _conv_update_ap(cache, error, lr=1.0)

        assert (conv.weights.data >= 0).all()
        assert (conv.weights.data <= 1).all()


class TestConvBackward:
    def test_input_error_shape(self):
        """Conv逆伝播の入力誤差形状"""
        conv = LutConv2d(1, 4, kernel_size=3, tree_depth=2, padding=1,
                         n_inputs=2, parameterization='independent',
                         residual_init=False)
        conv.weights.data.uniform_(0, 1)
        x = torch.rand(2, 1, 14, 14)
        out, cache = _conv_forward_ap(conv, x)

        error = torch.randn_like(out)
        input_error = _conv_backward_to_input(cache, error)
        assert input_error.shape == x.shape


class TestCNNAPTrainer:
    def test_init(self):
        """初期化が成功する"""
        model = _build_small_cnn()
        trainer = CNNAPTrainer(model, lr=0.1)
        assert len(trainer.conv_layers) == 1
        assert len(trainer.lut_layers) == 2
        assert trainer.group_sum is not None

    def test_train_step_runs(self):
        """train_stepがエラーなく実行される"""
        model = _build_small_cnn()
        trainer = CNNAPTrainer(model, lr=0.1)

        x = torch.rand(4, 1, 14, 14)
        y = torch.randint(0, 10, (4,))
        metrics = trainer.train_step(x, y)
        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_weights_change_after_step(self):
        """train_step後に全層の重みが変化"""
        torch.manual_seed(42)
        model = _build_small_cnn()
        trainer = CNNAPTrainer(model, lr=0.1)

        conv_w_before = model[0].weights.data.clone()
        lut_w_before = model[3].weights.data.clone()

        x = torch.rand(4, 1, 14, 14)
        y = torch.randint(0, 10, (4,))
        trainer.train_step(x, y)

        assert not torch.equal(conv_w_before, model[0].weights.data), \
            'Conv重みが更新されていない'
        assert not torch.equal(lut_w_before, model[3].weights.data), \
            'LutLayer重みが更新されていない'

    def test_evaluate(self):
        """evaluateが動作する"""
        model = _build_small_cnn()
        trainer = CNNAPTrainer(model, lr=0.1)

        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(
            torch.rand(8, 1, 14, 14),
            torch.randint(0, 10, (8,)),
        )
        loader = DataLoader(ds, batch_size=4)
        acc = trainer.evaluate(loader, 'cpu')
        assert 0.0 <= acc <= 1.0
