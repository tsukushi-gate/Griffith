"""L1: ゲート・LUTの意味論(真理値表レベルの正しさ)。

検証内容:
    16ゲート全数: N=2の16種Boolean関数 × 4入力パターン = 64ケース厳密一致
    任意真理値表: N=4/6でindependentパラメータ化が任意のBoolean関数を実現
    IWP積公式:   y = Π_j [p_j x_j + (1-p_j)(1-x_j)] の直接照合
"""

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.lut_layer import LutFunctionIWP, LutLayer, _build_gate_table_n2


def _gate_truth(g: int, a: int, b: int) -> int:
    """ゲートIDの定義(独立実装): gの真理値表を(0,0),(0,1),(1,0),(1,1)順で
    読んだ2進数。g = 8·f(0,0) + 4·f(0,1) + 2·f(1,0) + 1·f(1,1)
    """
    pos = {(0, 0): 3, (0, 1): 2, (1, 0): 1, (1, 1): 0}[(a, b)]
    return (g >> pos) & 1


class TestSixteenGates:
    def test_gate_table_matches_definition(self):
        """_build_gate_table_n2 がゲートID定義と全64ケースで一致"""
        table = _build_gate_table_n2()
        # butterfly基底順: k=0→(0,0), k=1→(1,0), k=2→(0,1), k=3→(1,1)
        k_to_ab = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
        for g in range(16):
            for k, (a, b) in k_to_ab.items():
                assert table[g, k].item() == _gate_truth(g, a, b), (
                    f'gate={g}, 入力=({a},{b})'
                )

    def test_known_gates(self):
        """代表ゲートの意味確認: 1=AND, 6=XOR, 7=OR, 9=XNOR, 14=NAND"""
        named = {
            1: lambda a, b: a & b,        # AND
            6: lambda a, b: a ^ b,        # XOR
            7: lambda a, b: a | b,        # OR
            9: lambda a, b: 1 - (a ^ b),  # XNOR
            14: lambda a, b: 1 - (a & b), # NAND
            0: lambda a, b: 0,            # FALSE
            15: lambda a, b: 1,           # TRUE
        }
        for g, fn in named.items():
            for a, b in itertools.product([0, 1], repeat=2):
                assert _gate_truth(g, a, b) == fn(a, b), f'gate={g}({a},{b})'

    def test_eval_forward_all_64_cases(self):
        """LutLayer(softmax/16ゲート) evalが全ゲート×全入力で真理値表と厳密一致"""
        layer = LutLayer(2, 16, n_inputs=2, parameterization='softmax')
        layer.indices = torch.tensor([[0] * 16, [1] * 16], dtype=torch.int64)
        with torch.no_grad():
            layer.weights.fill_(-10.0)
            for g in range(16):
                layer.weights[g, g] = 10.0  # ニューロンgにゲートgを割当
        layer.eval()
        for a, b in itertools.product([0, 1], repeat=2):
            y = layer(torch.tensor([[float(a), float(b)]]))
            for g in range(16):
                assert y[0, g].item() == _gate_truth(g, a, b), (
                    f'gate={g}, 入力=({a},{b}): got {y[0, g].item()}'
                )


class TestArbitraryTruthTable:
    @pytest.mark.parametrize('n', [4, 6])
    def test_independent_realizes_any_function(self, n, seed):
        """independentパラメータ化は任意の真理値表T∈{0,1}^(2^n)を実現できる"""
        n_entries = 2 ** n
        truth = torch.randint(0, 2, (n_entries,))
        layer = LutLayer(n, 1, n_inputs=n, parameterization='independent')
        layer.indices = torch.arange(n, dtype=torch.int64).unsqueeze(-1)
        with torch.no_grad():
            layer.weights.copy_(
                torch.where(truth.bool(), 10.0, -10.0).unsqueeze(0)
            )
        layer.eval()
        for k in range(n_entries):
            x = torch.tensor(
                [[float((k >> j) & 1) for j in range(n)]]
            )
            assert layer(x).item() == truth[k].item(), f'エントリk={k}'

    @pytest.mark.parametrize('n', [4])
    def test_soft_mode_approximates(self, n, seed):
        """飽和重み(±10)ならsoft評価(train)でも真理値表を1e-4で近似"""
        n_entries = 2 ** n
        truth = torch.randint(0, 2, (n_entries,))
        layer = LutLayer(n, 1, n_inputs=n, parameterization='independent')
        layer.indices = torch.arange(n, dtype=torch.int64).unsqueeze(-1)
        with torch.no_grad():
            layer.weights.copy_(
                torch.where(truth.bool(), 10.0, -10.0).unsqueeze(0)
            )
        layer.train()
        for k in range(n_entries):
            x = torch.tensor([[float((k >> j) & 1) for j in range(n)]])
            assert abs(layer(x).item() - truth[k].item()) < 1e-4, f'k={k}'


class TestIWPProductFormula:
    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_formula_direct(self, n, seed):
        """y = Π_j [p_j x_j + (1-p_j)(1-x_j)] を素朴なループ計算と照合"""
        out_dim, batch = 3, 4
        weights = torch.randn(out_dim, n, dtype=torch.float64)
        indices = torch.stack(
            [torch.randint(0, n * 2, (out_dim,)) for _ in range(n)]
        )
        x = torch.rand(batch, n * 2, dtype=torch.float64)

        y = LutFunctionIWP.apply(x, weights, indices, n)

        p = torch.sigmoid(weights)
        for b in range(batch):
            for o in range(out_dim):
                prod = 1.0
                for j in range(n):
                    xj = x[b, indices[j, o]].item()
                    pj = p[o, j].item()
                    prod *= pj * xj + (1 - pj) * (1 - xj)
                assert abs(y[b, o].item() - prod) < TOL_F64_EQ, f'(b={b},o={o})'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
