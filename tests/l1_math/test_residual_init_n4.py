"""L1: residual_initのN>2対応。

- independent: ±5ロジットでhard表が厳密にx_{n-1} pass-through
- softmax N>2: soft出力がx_{n-1}方向へバイアス(減衰あり、文書化済み)
- iwp: 警告して無視
"""

import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.layers import LutLayer


class TestResidualInitN4:
    @pytest.mark.parametrize('n', [4, 6])
    def test_independent_exact_passthrough(self, n):
        """independent+residual_init: hard evalが最後の入力の恒等関数"""
        torch.manual_seed(3)
        layer = LutLayer(n, 3, n_inputs=n, parameterization='independent',
                         residual_init=True,
                         connections=torch.arange(n).unsqueeze(1)
                         .expand(n, 3).contiguous())
        layer.eval()
        pats = torch.tensor(
            [[(e >> j) & 1 for j in range(n)] for e in range(2 ** n)],
            dtype=torch.float32,
        )
        with torch.no_grad():
            y = layer(pats)
        expected = pats[:, n - 1:n].expand(-1, 3)
        assert torch.equal(y, expected), 'pass-throughになっていない'

    def test_softmax_n4_biased_toward_passthrough(self):
        """softmax N=4+residual_init: soft出力が厳密に想定の減衰式に従う。

        合計1制約により厳密pass-throughは不可能(docstring文書化済み)。
        hotエントリ(bit n-1=1、2^(n-1)個)の確率 p_hot = e^5/(h·e^5+h)、
        coldは p_cold = 1/(h·e^5+h)(h=2^(n-1))。頂点入力ではbasisが
        one-hotなので y(x_{n-1}=1) = p_hot、y(0) = p_cold。
        """
        import math
        n = 4
        h = 2 ** (n - 1)
        p_hot = math.exp(5.0) / (h * math.exp(5.0) + h)
        p_cold = 1.0 / (h * math.exp(5.0) + h)
        layer = LutLayer(n, 2, n_inputs=n, parameterization='softmax',
                         residual_init=True,
                         connections=torch.arange(n).unsqueeze(1)
                         .expand(n, 2).contiguous())
        layer.train()
        x1 = torch.zeros(1, n)
        x1[0, n - 1] = 1.0
        x0 = torch.zeros(1, n)
        with torch.no_grad():
            y1 = layer(x1)
            y0 = layer(x0)
        assert torch.allclose(y1, torch.full_like(y1, p_hot), atol=1e-5), (
            f'y1={y1} != p_hot={p_hot:.6f}'
        )
        assert torch.allclose(y0, torch.full_like(y0, p_cold), atol=1e-5), (
            f'y0={y0} != p_cold={p_cold:.6f}'
        )

    def test_iwp_warns_and_ignores(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            LutLayer(4, 2, n_inputs=4, parameterization='iwp',
                     residual_init=True)
        assert any('residual_init' in str(w.message) for w in rec)

    def test_deep_stack_preserves_signal(self):
        """independent res-init 4層のsoft forwardが入力情報を保持
        (ランダム初期化は深さで情報が減衰する対照)"""
        torch.manual_seed(5)
        n, width, depth = 4, 16, 4
        x = (torch.rand(64, width) > 0.5).float()

        def build(res):
            torch.manual_seed(7)
            return torch.nn.Sequential(*[
                LutLayer(width, width, n_inputs=n,
                         parameterization='independent', residual_init=res)
                for _ in range(depth)
            ])

        with torch.no_grad():
            y_res = build(True)(x)
            y_rnd = build(False)(x)
        # residual initは分散(信号)を保持、ランダムは0.5付近へ潰れがち
        assert y_res.var() > y_rnd.var(), (
            f'res={y_res.var():.4f} rnd={y_rnd.var():.4f}'
        )
