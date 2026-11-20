"""DFARuleの系列(3D→B*S)拡張の検証。

DFA層の出力が (B, S, dim) の3Dテンソルの場合、内部で (B*S, dim) に
flattenして扱う。このテストでは:
    1. 3D出力のDFA層に対し、inject()で計算される勾配が、手計算で
       (B*S,dim)にreshapeしてから同じ数式を適用した場合と厳密一致すること
    2. 2D出力(既存パス)が引き続き従来通り動作すること(非回帰)
    3. error(出力誤差)が2D (B,vocab)しかない場合、系列方向に複製して
       3D DFA層へ配れること
を検証する。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import LearningSpec, LutLayer, with_learning
from griffith_logic.training.rules.dfa import DFARule


class SeqLutLayer(nn.Module):
    """(B, S, in_dim) -> (B, S, out_dim) を返すLutLayerラッパ(テスト専用)。

    LogicGPTのLogicMLP/headと同様に「内部でLutLayerを(B*S,in_dim)へreshapeして
    呼び、出力を(B,S,out_dim)に戻す」形の3Dモジュールを模す。
    """

    def __init__(self, in_dim, out_dim, n_inputs=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lut = LutLayer(in_dim, out_dim, n_inputs=n_inputs)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        b, s, d = x.shape
        h = self.lut(x.reshape(b * s, d))
        return h.reshape(b, s, self.out_dim)


def _make_seq_model(seed=7):
    torch.manual_seed(seed)
    layer = with_learning(
        SeqLutLayer(4, 8, seed=seed), LearningSpec(rule='dfa'),
    )
    return layer


class TestDFASeqFlatten:
    def test_3d_output_stashed_and_flattened(self, seed):
        """3D出力のDFA層でpost_hookが(B*S,dim)にflattenしてstashすること"""
        model = _make_seq_model()
        rule = DFARule([('0', model)])
        rule.begin_step()

        b, s = 3, 5
        x = torch.rand(b, s, 4)
        out = model(x)
        assert out.shape == (b, s, model.out_dim)
        # 大域グラフからは切断されている(detachされて返る)
        assert not out.requires_grad

        stashed = rule._stash['0']
        assert len(stashed) == 1
        local_out, seq_shape = stashed[0]
        assert seq_shape == (b, s)
        assert local_out.shape == (b * s, model.out_dim)
        rule.remove()

    def test_inject_matches_manual_flatten_grad(self, seed):
        """inject()の勾配が、(B*S,dim)手計算のautograd.gradと厳密一致すること"""
        model = _make_seq_model(seed=11)
        rule = DFARule([('0', model)])
        rule.begin_step()

        b, s = 2, 3
        vocab = 6
        x = torch.rand(b, s, 4)
        out = model(x)  # detached (b,s,out_dim)
        error = torch.randn(b, s, vocab)  # 系列出力の誤差

        rule.inject(error)
        grads_via_rule = {
            name: p.grad.clone() for name, p in model.named_parameters()
            if p.grad is not None
        }
        rule.remove()

        # --- 手計算: 同じ固定フィードバックB、同じ局所グラフで勾配を検証 ---
        # DFARuleが生成したBを再利用して、reshapeした誤差を使い直接
        # autograd.gradを計算し、rule.inject()の結果と比較する。
        model2 = _make_seq_model(seed=11)
        rule2 = DFARule([('0', model2)])
        rule2.begin_step()
        x2 = x.clone()
        # pre_hookが入力をdetach+requires_grad_するため、forwardを直接呼ぶ
        out2 = model2(x2)
        local_out2, seq_shape2 = rule2._stash['0'][0]
        assert seq_shape2 == (b, s)

        B = rule2._get_feedback(
            '0', local_out2.shape[1], vocab, local_out2.device, local_out2.dtype,
        )
        e_flat = error.reshape(b * s, vocab)
        g = e_flat @ B
        params = [p for p in model2.parameters() if p.requires_grad]
        grads_manual = torch.autograd.grad(
            local_out2, params, grad_outputs=g, allow_unused=True,
        )
        rule2.remove()

        for (name, p), gr in zip(model2.named_parameters(), grads_manual, strict=False):
            assert torch.allclose(grads_via_rule[name], gr, atol=1e-10), (
                f'{name}: rule勾配とmanual勾配が不一致'
            )

    def test_2d_output_unaffected_by_seq_support(self, seed):
        """既存の2D出力パス(非系列)が従来通り動作すること(非回帰)"""
        torch.manual_seed(3)
        layer = with_learning(LutLayer(4, 8, n_inputs=2), LearningSpec(rule='dfa'))
        rule = DFARule([('0', layer)])
        rule.begin_step()

        x = torch.rand(5, 4)
        out = layer(x)
        assert out.shape == (5, 8)
        stashed = rule._stash['0']
        local_out, seq_shape = stashed[0]
        assert seq_shape is None
        assert local_out.shape == (5, 8)

        error = torch.randn(5, 3)
        rule.inject(error)
        assert layer.weights.grad is not None
        rule.remove()

    def test_inject_with_2d_error_broadcasts_over_seq(self, seed):
        """error(出力誤差)が2D(batch,C)のみでも3D DFA層へ系列複製して配れること"""
        model = _make_seq_model(seed=5)
        rule = DFARule([('0', model)])
        rule.begin_step()

        b, s = 2, 4
        vocab = 3
        x = torch.rand(b, s, 4)
        model(x)
        error_2d = torch.randn(b, vocab)  # 系列軸なし

        rule.inject(error_2d)
        assert all(p.grad is not None for p in model.parameters())
        rule.remove()

    def test_3d_error_shape_mismatch_raises(self, seed):
        """error(3D)のbatch/seqがstash済みと不一致ならassertで検出すること"""
        model = _make_seq_model(seed=9)
        rule = DFARule([('0', model)])
        rule.begin_step()

        x = torch.rand(2, 3, 4)
        model(x)
        wrong_error = torch.randn(2, 99, 5)  # seq不一致

        with pytest.raises(AssertionError):
            rule.inject(wrong_error)
        rule.remove()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
