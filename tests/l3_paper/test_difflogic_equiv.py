"""L3: difflogic(Petersen 2022)との式レベル等価性。

difflogicの16種2入力Boolean関数の実数値緩和(論文Table)を
独立に書き下し、LutLayerの16ゲートモードが
    y = Σ_g softmax(w)_g × f_g(a, b)
と厳密一致することを検証する。ゲートIDの対応は
difflogicのop順序(0=zero, 1=and, ..., 15=one)と当リポジトリの
ゲートID規約(g = 8f(0,0)+4f(0,1)+2f(1,0)+1f(1,1))が同一であることも
このテストが証明する。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_GRAD

from griffith_logic import LutLayer
from griffith_logic.lut_layer import LutFunction

# difflogicのop定義(実数値緩和、論文の多項式形を独立に書き下す)
DIFFLOGIC_OPS = [
    lambda a, b: torch.zeros_like(a),          # 0: FALSE
    lambda a, b: a * b,                        # 1: AND
    lambda a, b: a - a * b,                    # 2: A AND NOT B
    lambda a, b: a,                            # 3: A
    lambda a, b: b - a * b,                    # 4: NOT A AND B
    lambda a, b: b,                            # 5: B
    lambda a, b: a + b - 2 * a * b,            # 6: XOR
    lambda a, b: a + b - a * b,                # 7: OR
    lambda a, b: 1 - (a + b - a * b),          # 8: NOR
    lambda a, b: 1 - (a + b - 2 * a * b),      # 9: XNOR
    lambda a, b: 1 - b,                        # 10: NOT B
    lambda a, b: 1 - b + a * b,                # 11: B IMPLIES A
    lambda a, b: 1 - a,                        # 12: NOT A
    lambda a, b: 1 - a + a * b,                # 13: A IMPLIES B
    lambda a, b: 1 - a * b,                    # 14: NAND
    lambda a, b: torch.ones_like(a),           # 15: TRUE
]


class TestDifflogicEquivalence:
    def test_soft_forward_matches_difflogic_mixture(self, seed):
        """train時のsoft出力 == difflogic式のsoftmax混合(1e-10)"""
        torch.manual_seed(29)
        layer = LutLayer(8, 6, n_inputs=2, parameterization='softmax')
        w = torch.randn(6, 16, dtype=torch.float64)
        x = torch.rand(5, 8, dtype=torch.float64)

        y = LutFunction.apply(
            x, w, layer.indices, 2, layer.gate_table.double(),
        )

        # difflogic式: 各ニューロンの2入力 (a, b) = (slot0, slot1)
        a = x[:, layer.indices[0]]  # (batch, out)
        b = x[:, layer.indices[1]]
        s = F.softmax(w, dim=-1)    # (out, 16)
        y_dl = sum(
            s[None, :, g] * DIFFLOGIC_OPS[g](a, b) for g in range(16)
        )

        assert torch.allclose(y, y_dl, atol=TOL_F64_GRAD, rtol=0), (
            f'difflogic式と不一致: {(y - y_dl).abs().max():.3e}'
        )

    def test_hard_eval_matches_difflogic_argmax(self, seed):
        """eval時のhard出力 == argmaxゲートのdifflogic式(二値入力で厳密)"""
        torch.manual_seed(37)
        layer = LutLayer(8, 16, n_inputs=2, parameterization='softmax')
        layer.eval()
        x = torch.randint(0, 2, (8, 8)).float()
        y = layer(x)

        a = x[:, layer.indices[0]]
        b = x[:, layer.indices[1]]
        gate_idx = layer.weights.argmax(dim=-1)  # (out,)
        for o in range(16):
            g = int(gate_idx[o])
            expected = DIFFLOGIC_OPS[g](a[:, o], b[:, o])
            assert torch.equal(y[:, o], expected), f'neuron={o}, gate={g}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
