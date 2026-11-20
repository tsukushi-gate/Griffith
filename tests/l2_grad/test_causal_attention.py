"""L2: causal LogicAttentionの勾配契約と参照実装照合。

    - float64 gradcheck(causal soft attention、入力q/k/vと全LUT重み)
    - reference_xnor_attention_causal(core/reference.py)との値・勾配照合(1e-10)
    - hard causalのSTE契約: forward=未来位置を除いた厳密one-hot、
      backward=causal softmaxの勾配と一致
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.blocks import LogicAttention
from griffith_logic.core.reference import reference_xnor_attention_causal


def _interior(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * 0.8 + 0.1


def _binary(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, shape, generator=g).float()


class TestCausalGradcheck:
    def test_gradcheck_soft_causal(self, seed):
        """causal=Trueのsoft attentionでfloat64 gradcheck(小型構成、S_q==S_k)"""
        attn = LogicAttention(
            2, n_heads=1, mode='soft', qk_sizes=[2], v_sizes=[2], causal=True,
        ).double()
        attn.train()
        s = 3
        q = _interior((1, s, 2), seed=4).requires_grad_(True)
        k = _interior((1, s, 2), seed=5).requires_grad_(True)
        v = _interior((1, s, 2), seed=6).requires_grad_(True)
        params = tuple(attn.parameters())

        def fn(q_, k_, v_, *ps):
            return attn(q_, k_, v_)

        assert torch.autograd.gradcheck(fn, (q, k, v, *params), **GRADCHECK_KW)


class TestCausalReferenceEquivalence:
    def test_attend_matches_causal_reference(self, seed):
        """causal attendコア(射影後)がreference_xnor_attention_causalと一致(値と勾配)"""
        attn = LogicAttention(4, n_heads=1, mode='soft', tau=1.3, causal=True)
        s = 5
        q = _interior((2, s, 4), seed=7)
        k = _interior((2, s, 4), seed=8)
        v = _interior((2, s, 6), seed=9)
        grad_output = torch.randn(2, s, 6, dtype=torch.float64)

        ins = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        y = attn.attend(*ins, hard=False)
        grads = torch.autograd.grad(y, ins, grad_outputs=grad_output)

        ins_ref = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        y_ref = reference_xnor_attention_causal(*ins_ref, tau=1.3)
        grads_ref = torch.autograd.grad(y_ref, ins_ref, grad_outputs=grad_output)

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)
        for g, g_ref in zip(grads, grads_ref):
            assert torch.allclose(g, g_ref, atol=TOL_F64_GRAD, rtol=0)


class TestCausalHardSteContract:
    def test_forward_onehot_respects_causality(self, seed):
        """forward契約: hard causal時のattention重みは未来位置に厳密0、各行one-hot"""
        attn = LogicAttention(4, n_heads=2, mode='hard', v_sizes=[4], causal=True)
        attn.eval()
        s = 5
        q = _binary((3, s, 4), seed=10)
        k = _binary((3, s, 4), seed=11)
        v = _binary((3, s, 4), seed=12)
        with torch.no_grad():
            attn(q, k, v)
        w = attn.last_attention  # (B, H, S, S)
        assert torch.all((w == 0.0) | (w == 1.0)), 'one-hotが厳密でない'
        assert torch.equal(w.sum(dim=-1), torch.ones(3, 2, s))
        # 未来位置(j>i)の重みは厳密に0
        for i in range(s):
            if i + 1 < s:
                assert torch.equal(
                    w[:, :, i, i + 1:], torch.zeros(3, 2, s - i - 1),
                ), f'未来位置への重みが非0(i={i})'

    def test_backward_contract_causal(self, seed):
        """backward契約: causal hardのq/k勾配がcausal soft勾配と一致(STE)"""
        attn = LogicAttention(4, n_heads=1, mode='hard', causal=True)
        s = 4
        q0 = _interior((2, s, 4), seed=16)
        k0 = _interior((2, s, 4), seed=17)
        v0 = _interior((2, s, 4), seed=18)
        grad_output = torch.randn(2, s, 4, dtype=torch.float64)

        ins_hard = [t.detach().clone().requires_grad_(True) for t in (q0, k0, v0)]
        y_hard = attn.attend(*ins_hard, hard=True)
        g_hard = torch.autograd.grad(y_hard, ins_hard, grad_outputs=grad_output)

        ins_soft = [t.detach().clone().requires_grad_(True) for t in (q0, k0, v0)]
        y_soft = attn.attend(*ins_soft, hard=False)
        g_soft = torch.autograd.grad(y_soft, ins_soft, grad_outputs=grad_output)

        assert torch.allclose(g_hard[0], g_soft[0], atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(g_hard[1], g_soft[1], atol=TOL_F64_GRAD, rtol=0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
