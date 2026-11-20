"""L2: DWNLayerの層仕様とLearnable Mappingの勾配式照合。

検証する数式(論文§3.2、公式実装 torch_dwn/mapping.py と照合済み):
    Learnable Mapping:
        forward:  I(W, x)_i = x_{argmax(W[:,i])}(ハードargmax、softmax不使用)
        backward: ∂L/∂W = (2x−1)^T G
                  ∂L/∂x = G · softmax_{dim=0}(W/τ)^T
    DWNLayer:
        出力 = 1[U_{δ(a)} > 0] ∈ {0,1}(訓練・推論とも同一の決定的forward)
        テーブルは訓練時に[−1,1]へclamp
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_GRAD

from griffith_logic.layers.dwn import DWNLayer, LearnableMapping


class TestLearnableMappingForward:
    def test_argmax_selection(self):
        """forward: 列iは x[:, argmax(W[:,i])] のハード選択"""
        lm = LearnableMapping(4, 3)
        with torch.no_grad():
            lm.weights.copy_(torch.tensor([
                [0.9, 0.1, 0.0],
                [0.0, 0.8, 0.0],
                [0.1, 0.2, 0.0],
                [0.0, 0.0, 0.7],
            ]))
        x = torch.tensor([[10.0, 20.0, 30.0, 40.0],
                          [1.0, 2.0, 3.0, 4.0]])
        out = lm(x)
        expected = torch.tensor([[10.0, 20.0, 40.0], [1.0, 2.0, 4.0]])
        assert torch.equal(out, expected)

    def test_weight_init_uniform01(self, seed):
        """重み初期化は一様[0,1)(公式実装準拠)"""
        lm = LearnableMapping(100, 50)
        assert lm.weights.shape == (100, 50)
        assert (lm.weights >= 0).all() and (lm.weights < 1).all()


class TestLearnableMappingBackward:
    def _case(self):
        torch.manual_seed(7)
        x = (torch.rand(6, 5, dtype=torch.float64) > 0.5).double()
        w = torch.rand(5, 8, dtype=torch.float64)
        grad_out = torch.randn(6, 8, dtype=torch.float64)
        return x, w, grad_out

    def test_grad_weights_formula(self):
        """∂L/∂W = (2x−1)^T G の直接照合(float64)"""
        x, w, grad_out = self._case()
        lm = LearnableMapping(5, 8, tau=0.001).double()
        with torch.no_grad():
            lm.weights.copy_(w)
        xr = x.clone().requires_grad_(True)
        out = lm(xr)
        out.backward(grad_out)
        expected = (2.0 * x - 1.0).T @ grad_out
        assert torch.allclose(lm.weights.grad, expected, atol=TOL_F64_GRAD, rtol=0)

    @pytest.mark.parametrize('tau', [0.001, 1.0])
    def test_grad_x_formula(self, tau):
        """∂L/∂x = G·softmax(W/τ, dim=0)^T の直接照合(float64)"""
        x, w, grad_out = self._case()
        lm = LearnableMapping(5, 8, tau=tau).double()
        with torch.no_grad():
            lm.weights.copy_(w)
        xr = x.clone().requires_grad_(True)
        out = lm(xr)
        out.backward(grad_out)
        expected = grad_out @ torch.softmax(w / tau, dim=0).T
        assert torch.allclose(xr.grad, expected, atol=TOL_F64_GRAD, rtol=0)


class TestDWNLayerShapes:
    @pytest.mark.parametrize('mapping', ['random', 'learnable'])
    @pytest.mark.parametrize('n', [2, 6])
    def test_output_shape(self, n, mapping, seed):
        layer = DWNLayer(16, 10, n_inputs=n, mapping=mapping)
        x = (torch.rand(4, 16) > 0.5).float()
        assert layer(x).shape == (4, 10)

    def test_lut_table_shape_and_range(self, seed):
        """テーブルは(n_luts, 2^n)、初期化はU(−1,1)"""
        layer = DWNLayer(16, 100, n_inputs=4)
        assert layer.luts.shape == (100, 16)
        assert (layer.luts > -1).all() and (layer.luts < 1).all()

    def test_invalid_mapping_rejected(self):
        with pytest.raises(AssertionError):
            DWNLayer(8, 4, n_inputs=2, mapping='arange')


class TestDWNLayerBinaryOutput:
    @pytest.mark.parametrize('training', [True, False], ids=['train', 'eval'])
    def test_output_is_binary(self, training, seed):
        layer = DWNLayer(12, 20, n_inputs=3)
        layer.train(training)
        x = (torch.rand(8, 12) > 0.5).float()
        y = layer(x)
        assert ((y == 0) | (y == 1)).all()

    def test_train_eval_forward_identical(self, seed):
        """DWNのforwardは決定的で訓練/推論のギャップがない"""
        layer = DWNLayer(12, 20, n_inputs=3, clamp_luts=False)
        x = (torch.rand(8, 12) > 0.5).float()
        layer.train()
        y_train = layer(x)
        layer.eval()
        y_eval = layer(x)
        assert torch.equal(y_train, y_eval)

    def test_ste_false_returns_table_values(self, seed):
        """ste=False: テーブル実数値をそのまま出力(clamp後は[−1,1])"""
        layer = DWNLayer(12, 20, n_inputs=3, ste=False)
        x = (torch.rand(8, 12) > 0.5).float()
        y = layer(x)
        assert (y >= -1).all() and (y <= 1).all()
        assert not ((y == 0) | (y == 1)).all()

    def test_clamp_luts_on_train_forward(self, seed):
        """訓練forwardごとにテーブルを[−1,1]へclamp(公式実装準拠)"""
        layer = DWNLayer(8, 4, n_inputs=2)
        with torch.no_grad():
            layer.luts.mul_(10.0)
        layer.train()
        layer((torch.rand(2, 8) > 0.5).float())
        assert (layer.luts >= -1).all() and (layer.luts <= 1).all()


class TestDWNLayerChaining:
    def test_multilayer_forward_and_grad_flow(self, seed):
        """多層連結: 二値出力が次層のアドレスになり、勾配が全層へ流れる"""
        l1 = DWNLayer(16, 32, n_inputs=6, mapping='learnable')
        l2 = DWNLayer(32, 8, n_inputs=6)
        x = (torch.rand(4, 16) > 0.5).float()
        y = l2(l1(x))
        assert y.shape == (4, 8)
        assert ((y == 0) | (y == 1)).all()

        y.sum().backward()
        assert l1.luts.grad is not None and torch.isfinite(l1.luts.grad).all()
        assert l2.luts.grad is not None and torch.isfinite(l2.luts.grad).all()
        assert l1.learnable_mapping.weights.grad is not None
        # EFDにより前段テーブルへ非ゼロ勾配が届く
        assert l1.luts.grad.abs().sum() > 0

    def test_gradient_reaches_input(self, seed):
        """入力へもEFD勾配が届く(サーモメータSTEとの連結用)"""
        layer = DWNLayer(10, 6, n_inputs=4)
        x = (torch.rand(3, 10) > 0.5).float().requires_grad_(True)
        layer(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
