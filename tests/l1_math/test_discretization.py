"""L1: eval時の離散化仕様(全パラメータ化の契約を固定)。

離散化仕様(決定・統一。IR/HDLエクスポートが依存する契約):

    | parameterization | eval時の重み離散化                     | 二値入力→出力 |
    |------------------|----------------------------------------|---------------|
    | softmax (N=2)    | argmaxで16ゲートから1つ選択            | 二値          |
    | softmax (N>2)    | softmax確率 > 0.5 のエントリのみ1      | 二値          |
    | independent      | sigmoid(w) > 0.5                       | 二値          |
    | iwp              | sigmoid(w) > 0.5 のリテラル積          | 二値          |
    | warp             | WHロジット > 0                         | 二値          |
    | raw              | 連続のまま(BinaryBrain意味論)          | 連続          |

    raw はDifferentiableLutBlock内でBatchNorm+Binarizeと組むため
    eval でも連続値を保つ。HW展開時のみ >0.5 で離散化する(M7)。

soft→hard一致性: 重みが飽和(±スケールs大)していれば、
train(soft)とeval(hard)の出力は二値入力上で一致する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.lut_layer import LutLayer

BINARY_PARAMS = ['softmax', 'independent', 'iwp', 'warp']


def _binary_inputs(in_dim, batch=8, seed=23):
    torch.manual_seed(seed)
    return torch.randint(0, 2, (batch, in_dim)).float()


class TestEvalOutputsAreBinary:
    @pytest.mark.parametrize('n', [2, 4, 6])
    @pytest.mark.parametrize('param', BINARY_PARAMS)
    def test_binary_in_binary_out(self, param, n, seed):
        """二値入力に対しeval出力は厳密に{0,1}(rawを除く全方式)"""
        layer = LutLayer(max(8, 2 * n), 6, n_inputs=n, parameterization=param)
        layer.eval()
        x = _binary_inputs(layer.in_dim)
        y = layer(x)
        assert torch.all((y == 0.0) | (y == 1.0)), (
            f'{param} n={n}: eval出力が二値でない: {y.unique()}'
        )

    def test_raw_stays_continuous(self, seed):
        """rawはeval でも連続(BinaryBrain意味論、仕様として固定)"""
        layer = LutLayer(8, 6, n_inputs=2, parameterization='raw')
        layer.eval()
        x = _binary_inputs(8)
        y = layer(x)
        # Normal(0.5, 0.01)初期化の重みがそのまま出力に混合される
        assert not torch.all((y == 0.0) | (y == 1.0)), (
            'rawのeval出力が二値化されている(仕様変更の疑い)'
        )


class TestSoftHardConsistency:
    """飽和重みでのsoft(train)→hard(eval)一致性"""

    SCALE = 20.0

    def _saturate(self, layer, param):
        with torch.no_grad():
            if param == 'iwp':
                layer.weights.copy_(
                    torch.sign(torch.randn_like(layer.weights)) * self.SCALE
                )
            elif param == 'warp':
                # 1エントリのみ±SCALE(複数係数の相殺でロジットが0近傍に
                # なるのを避け、二値入力上で|logit|=SCALEを保証する)
                w = torch.zeros_like(layer.weights)
                hot = torch.randint(0, layer.weights.shape[1],
                                    (layer.weights.shape[0],))
                sign = torch.sign(torch.randn(layer.weights.shape[0]))
                w[torch.arange(layer.weights.shape[0]), hot] = sign * self.SCALE
                layer.weights.copy_(w)
            else:
                # softmax/independent: ランダムに1エントリを大、他を小に
                w = torch.full_like(layer.weights, -self.SCALE)
                hot = torch.randint(0, layer.weights.shape[1],
                                    (layer.weights.shape[0],))
                w[torch.arange(layer.weights.shape[0]), hot] = self.SCALE
                layer.weights.copy_(w)

    @pytest.mark.parametrize('n', [2, 4])
    @pytest.mark.parametrize('param', BINARY_PARAMS)
    def test_saturated_soft_matches_hard(self, param, n, seed):
        layer = LutLayer(max(8, 2 * n), 6, n_inputs=n, parameterization=param)
        if param == 'warp':
            layer.gumbel = False  # 決定的に比較
        self._saturate(layer, param)
        x = _binary_inputs(layer.in_dim)

        layer.train()
        y_soft = layer(x)
        layer.eval()
        y_hard = layer(x)

        assert torch.allclose(y_soft, y_hard, atol=1e-6), (
            f'{param} n={n}: soft-hardギャップ '
            f'{(y_soft - y_hard).abs().max():.3e}'
        )


class TestDiscretizationRules:
    """離散化規則そのものの照合(仕様の実装確認)"""

    def test_softmax_n4_rounding_rule(self):
        """N>2 softmax: softmax確率>0.5のエントリのみ1(丸め規則)"""
        layer = LutLayer(4, 3, n_inputs=4, parameterization='softmax')
        with torch.no_grad():
            layer.weights.fill_(0.0)
            layer.weights[0, 5] = 10.0   # ニューロン0: エントリ5が支配的
            # ニューロン1: 全エントリ均等(どれも>0.5にならない→定数0)
            layer.weights[2, 3] = 10.0
        layer.eval()
        # エントリ5 = 入力パターン(x0=1,x1=0,x2=1,x3=0)
        layer.indices = torch.tensor(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.int64,
        )
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # パターンk=5
        y = layer(x)
        assert y[0, 0].item() == 1.0  # 支配エントリと入力が一致→1
        assert y[0, 1].item() == 0.0  # 均等重み→どのエントリも0.5未満→0

    def test_independent_threshold_rule(self):
        """independent: sigmoid(w)>0.5 ⇔ w>0 で真理値表が決まる"""
        layer = LutLayer(2, 1, n_inputs=2, parameterization='independent')
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)
        with torch.no_grad():
            layer.weights.copy_(torch.tensor([[0.1, -0.1, -0.1, 0.1]]))
        layer.eval()
        # 真理値表 = [1,0,0,1] (butterfly順) = XNOR
        for (a, b), expected in [((0, 0), 1), ((1, 0), 0), ((0, 1), 0), ((1, 1), 1)]:
            y = layer(torch.tensor([[float(a), float(b)]]))
            assert y.item() == expected, f'({a},{b})'

    def test_warp_sign_rule(self):
        """warp: WHロジットの符号で離散化(>0 → 1)"""
        layer = LutLayer(2, 1, n_inputs=2, parameterization='warp')
        layer.indices = torch.tensor([[0], [1]], dtype=torch.int64)
        with torch.no_grad():
            # θ = [0, 0, 0, 1] → logit = s0·s1 → x0 XNOR x1 で正
            layer.weights.copy_(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        layer.eval()
        for (a, b), expected in [((0, 0), 1), ((1, 0), 0), ((0, 1), 0), ((1, 1), 1)]:
            y = layer(torch.tensor([[float(a), float(b)]]))
            assert y.item() == expected, f'({a},{b})'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
