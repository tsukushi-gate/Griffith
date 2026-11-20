"""L1: WARP Hard-ST/Gumbel-STの数学的性質。

- binary頂点でのeval離散化 1[θ·φ(s)>0] との厳密一致(全2^n掃引)
- 正スカラー倍不変性(hard関数はθのスケールに不変)
- LutLayer dispatch: warp×{hard_st,gumbel_st}が単一minterm崩壊しない
  (旧バグ: LutFunctionHardSTのargmaxはWH係数の最大1個を選んでいた)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.core import (
    reference_warp_entry_logits,
    reference_wh_basis,
)
from griffith_logic.core.functions import LutFunctionWARPHardST
from griffith_logic.layers import LutLayer


class TestWARPHardSemantics:
    @pytest.mark.parametrize('n', [2, 4])
    def test_binary_vertices_match_eval_discretization(self, n):
        """全2^n binary頂点でhard forward == 1[θ·φ(s)>0](eval式)"""
        torch.manual_seed(7)
        out_dim = 5
        w = torch.randn(out_dim, 2 ** n, dtype=torch.float64)
        # 恒等配線: 入力=頂点そのもの
        indices = torch.arange(n).unsqueeze(1).expand(n, out_dim).contiguous()

        pats = torch.tensor(
            [[(e >> j) & 1 for j in range(n)] for e in range(2 ** n)],
            dtype=torch.float64,
        )
        y = LutFunctionWARPHardST.apply(pats, w, indices, n, 1.0, None)

        gathered = pats[:, indices.T.reshape(-1)].reshape(2 ** n, out_dim, n)
        wh = reference_wh_basis(gathered, n)
        logit = (wh * w.unsqueeze(0)).sum(-1)
        y_eval = (logit > 0).to(pats.dtype)

        assert torch.allclose(y, y_eval, atol=TOL_F64_EQ, rtol=0)

    def test_entry_logits_match_vertex_soft_logits(self):
        """参照エントリロジット == WARPソフト式のbinary頂点評価(τ=1)"""
        torch.manual_seed(11)
        n, out_dim = 4, 3
        w = torch.randn(out_dim, 2 ** n, dtype=torch.float64)
        logits = reference_warp_entry_logits(w, n)

        pats = torch.tensor(
            [[(e >> j) & 1 for j in range(n)] for e in range(2 ** n)],
            dtype=torch.float64,
        )
        wh = reference_wh_basis(pats.unsqueeze(1).expand(-1, out_dim, -1), n)
        vertex_logits = (wh * w.unsqueeze(0)).sum(-1)  # (2^n, out_dim)

        assert torch.allclose(logits.T, vertex_logits,
                              atol=TOL_F64_EQ, rtol=0)

    def test_positive_scale_invariance(self):
        """hard forwardはθの正スカラー倍に不変(符号のみ依存)"""
        torch.manual_seed(13)
        n = 4
        w = torch.randn(6, 2 ** n, dtype=torch.float64)
        indices = torch.randint(0, 8, (n, 6))
        x = torch.rand(4, 8, dtype=torch.float64)
        y1 = LutFunctionWARPHardST.apply(x, w, indices, n, 1.0, None)
        y2 = LutFunctionWARPHardST.apply(x, 3.7 * w, indices, n, 1.0, None)
        assert torch.allclose(y1, y2, atol=TOL_F64_EQ, rtol=0)


class TestWARPHardSTDispatchRegression:
    """回帰: warp×hard_stが単一mintermへ崩壊しない(発見バグ)。"""

    @pytest.mark.parametrize('selection', ['hard_st', 'gumbel_st'])
    def test_warp_hard_table_not_single_minterm(self, selection):
        """複数エントリが1のhard表(例: OR)を表現できる。

        旧バグ(LutFunctionHardSTへの誤dispatch)ではone_hot(argmax θ)
        =常に単一mintermとなり、ORのような表は絶対に出せなかった。
        """
        n = 2
        layer = LutLayer(4, 1, n_inputs=n, parameterization='warp',
                         selection=selection)
        layer.gumbel_scale = 0.0  # gumbel_stをhard_stへ退化させ決定化
        # θ設定: 定数項のみ正 → 全エントリlogit>0のTRUE関数(全エントリ1)
        with torch.no_grad():
            layer.weights.zero_()
            layer.weights[:, 0] = 1.0
        x = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        layer.train()
        y = layer(x)
        # TRUE関数: どの入力でも1(単一mintermなら特定パターンのみ1)
        assert torch.allclose(y, torch.ones_like(y), atol=1e-6), (
            f'warp×{selection}が単一minterm崩壊(旧バグ再発): y={y}'
        )

    def test_train_forward_matches_eval_for_binary(self):
        """LutLayer経由(dispatch込み)でもbinary入力のtrain/eval一致"""
        torch.manual_seed(19)
        layer = LutLayer(10, 8, n_inputs=6, parameterization='warp',
                         selection='hard_st')
        with torch.no_grad():
            layer.weights.normal_()
        x = (torch.rand(32, 10) > 0.5).float()
        layer.train()
        y_train = layer(x)
        layer.eval()
        y_eval = layer(x)
        assert torch.allclose(y_train, y_eval, atol=1e-6)


class TestWHDegreeCap:
    """W7: 度数キャップwarp(--wh-degree-cap)の性質。"""

    def test_capped_table_matches_masked_logit_threshold(self):
        """cap=dの層のhard表 == 1[マスク済み次数≤dロジット>0](PTF意味論)。

        注: キャップは「ロジット多項式の次数≤d」(=次数d PTF)であって、
        Boolean関数のフーリエ次数≤dではない(sign()は次数を上げる)。
        PolyLUT(arXiv 2309.02334)と同じ意味論。
        """
        torch.manual_seed(23)
        n, d_cap, out_dim = 4, 1, 8
        layer = LutLayer(n, out_dim, n_inputs=n, parameterization='warp',
                         wh_degree_cap=d_cap,
                         connections=torch.arange(n).unsqueeze(1)
                         .expand(n, out_dim).contiguous())
        with torch.no_grad():
            layer.weights.normal_()
        layer.eval()
        pats = torch.tensor(
            [[(e >> j) & 1 for j in range(n)] for e in range(2 ** n)],
            dtype=torch.float32,
        )
        with torch.no_grad():
            table = layer(pats).T  # (out_dim, 2^n) ∈ {0,1}
        # 期待値: マスク済み係数でのエントリロジット閾値化
        idx = torch.arange(2 ** n)
        deg = sum(((idx >> j) & 1) for j in range(n))
        masked = layer.weights.detach() * (deg <= d_cap).float()
        logits = reference_warp_entry_logits(masked.double(), n)
        expected = (logits > 0).float()
        assert torch.equal(table, expected)

    def test_cap1_cannot_express_xor(self):
        """cap=1(線形閾値関数)ではXOR(PTF次数2)が表現不能。

        θをXORのWH係数(index 3)に設定してもマスクで消え、
        出力はXORにならない(表現力制限が実際に効いている証明)。
        """
        n = 2
        layer = LutLayer(n, 1, n_inputs=n, parameterization='warp',
                         wh_degree_cap=1,
                         connections=torch.arange(n).unsqueeze(1)
                         .expand(n, 1).contiguous())
        with torch.no_grad():
            layer.weights.zero_()
            layer.weights[0, 3] = -10.0  # s1·s2係数(XOR方向)
        layer.eval()
        pats = torch.tensor(
            [[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        with torch.no_grad():
            y = layer(pats)[:, 0]
        xor = torch.tensor([0., 1., 1., 0.])
        assert not torch.equal(y, xor), 'cap=1でXORが出せてしまっている'

    def test_masked_coeffs_receive_no_gradient(self):
        """マスクされた高次係数へ勾配が流れない"""
        torch.manual_seed(29)
        n, d_cap = 4, 2
        layer = LutLayer(8, 4, n_inputs=n, parameterization='warp',
                         wh_degree_cap=d_cap)
        layer.train()
        x = torch.rand(6, 8)
        layer(x).sum().backward()
        g = layer.weights.grad
        idx = torch.arange(2 ** n)
        deg = sum(((idx >> j) & 1) for j in range(n))
        assert (g[:, deg > d_cap] == 0).all(), '高次係数に勾配が流れている'
        assert g[:, deg <= d_cap].abs().sum() > 0

    def test_cap_requires_warp(self):
        with pytest.raises(AssertionError):
            LutLayer(8, 4, n_inputs=4, parameterization='softmax',
                     wh_degree_cap=2)

    def test_cap_with_gumbel_st_respected(self):
        """warp×gumbel_st訓練経路でも高次係数がマスクされる(勾配ゼロ)"""
        torch.manual_seed(31)
        n, d_cap = 4, 1
        layer = LutLayer(8, 4, n_inputs=n, parameterization='warp',
                         selection='gumbel_st', wh_degree_cap=d_cap)
        layer.train()
        x = (torch.rand(16, 8) > 0.5).float()
        layer(x).sum().backward()
        g = layer.weights.grad
        idx = torch.arange(2 ** n)
        deg = sum(((idx >> j) & 1) for j in range(n))
        assert (g[:, deg > d_cap] == 0).all()


class TestContinuousInputGapZero:
    """回帰(改訂): 連続入力でもtrain(hard_st)とevalが厳密一致する。

    旧・多重線形拡張版は連続入力(RDDLGN等の埋め込みスタック)で連続出力
    となりevalの閾値化と乖離、再帰で増幅されhard崩壊を起こした
    (charlm n=4: soft BPC 2.87 vs hard BPC 11.06)。
    """

    @pytest.mark.parametrize('n', [2, 4, 6])
    def test_gap_zero_on_continuous_inputs(self, n):
        torch.manual_seed(41)
        layer = LutLayer(10, 8, n_inputs=n, parameterization='warp',
                         selection='hard_st')
        with torch.no_grad():
            layer.weights.normal_()
        x = torch.rand(32, 10)  # 連続入力(埋め込み想定)
        layer.train()
        y_train = layer(x)
        layer.eval()
        y_eval = layer(x)
        assert torch.equal(y_train, y_eval), (
            f'連続入力でselection gap≠0: '
            f'max diff={(y_train - y_eval).abs().max():.3e}'
        )

    def test_output_is_binary_on_continuous_inputs(self):
        """訓練forwardの出力が連続入力でもbinary(閾値意味論)"""
        torch.manual_seed(43)
        layer = LutLayer(10, 8, n_inputs=4, parameterization='warp',
                         selection='hard_st')
        with torch.no_grad():
            layer.weights.normal_()
        layer.train()
        y = layer(torch.rand(16, 10))
        assert ((y == 0) | (y == 1)).all(), '出力がbinaryでない'
