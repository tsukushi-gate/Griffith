"""L2: LogicAttention部品の勾配契約と既存モデルとの数値等価。

    - 形状パラメトリック(ヘッド数・系列長・射影次元)
    - float64 gradcheck(soft、入力q/k/vと全LUT重み)
    - attendコアのcore/reference.py照合(reference_xnor_attention)
    - hardモードのSTE契約: forward=厳密one-hot、backward=soft勾配と一致
    - boolean_attention.pyとの数値等価(同じ重みを移植して同一出力。
      既存6モデル無改造の非回帰アンカー)
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F32_EQ, TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.blocks import LogicAttention
from griffith_logic.core.reference import reference_xnor_attention
from griffith_logic.models.attention import (
    BooleanAttentionConfig,
    DLGNBooleanAttention,
)


def _interior(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * 0.8 + 0.1


def _binary(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, shape, generator=g).float()


class TestShapes:
    @pytest.mark.parametrize('n_heads,s_q,s_k,d_model,d_v', [
        (1, 3, 3, 4, 4),
        (2, 2, 5, 4, 8),
        (4, 1, 4, 8, 8),
    ])
    def test_forward_shape(self, n_heads, s_q, s_k, d_model, d_v, seed):
        attn = LogicAttention(
            d_model, n_heads=n_heads, mode='soft', v_sizes=[d_v],
        )
        attn.train()
        q = _binary((2, s_q, d_model), seed=1)
        k = _binary((2, s_k, d_model), seed=2)
        v = _binary((2, s_k, d_model), seed=3)
        out = attn(q, k, v)
        assert out.shape == (2, s_q, d_v)
        assert attn.last_attention.shape == (2, n_heads, s_q, s_k)
        assert torch.isfinite(out).all()


class TestGradcheck:
    def test_gradcheck_soft(self, seed):
        """入力q/k/vと全LUT重みについてfloat64 gradcheck(小型構成)"""
        attn = LogicAttention(2, n_heads=1, mode='soft',
                              qk_sizes=[2], v_sizes=[2]).double()
        attn.train()
        q = _interior((1, 2, 2), seed=4).requires_grad_(True)
        k = _interior((1, 2, 2), seed=5).requires_grad_(True)
        v = _interior((1, 2, 2), seed=6).requires_grad_(True)
        params = tuple(attn.parameters())

        def fn(q_, k_, v_, *ps):
            return attn(q_, k_, v_)

        assert torch.autograd.gradcheck(fn, (q, k, v, *params), **GRADCHECK_KW)


class TestReferenceEquivalence:
    def test_attend_matches_reference(self, seed):
        """attendコア(射影後)がreference_xnor_attentionと一致(値と勾配)"""
        attn = LogicAttention(4, n_heads=1, mode='soft', tau=1.3)
        q = _interior((2, 3, 4), seed=7)
        k = _interior((2, 5, 4), seed=8)
        v = _interior((2, 5, 6), seed=9)
        grad_output = torch.randn(2, 3, 6, dtype=torch.float64)

        ins = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        y = attn.attend(*ins, hard=False)
        grads = torch.autograd.grad(y, ins, grad_outputs=grad_output)

        ins_ref = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
        y_ref = reference_xnor_attention(*ins_ref, tau=1.3)
        grads_ref = torch.autograd.grad(y_ref, ins_ref,
                                        grad_outputs=grad_output)

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)
        for g, g_ref in zip(grads, grads_ref):
            assert torch.allclose(g, g_ref, atol=TOL_F64_GRAD, rtol=0)


class TestHardSteContract:
    def test_forward_onehot_exact(self, seed):
        """forward契約: hard時のattention重みは厳密なone-hot"""
        attn = LogicAttention(4, n_heads=2, mode='hard', v_sizes=[4])
        attn.eval()
        q = _binary((3, 2, 4), seed=10)
        k = _binary((3, 5, 4), seed=11)
        v = _binary((3, 5, 4), seed=12)
        with torch.no_grad():
            attn(q, k, v)
        w = attn.last_attention  # (B, H, Sq, Sk)
        assert torch.all((w == 0.0) | (w == 1.0)), 'one-hotが厳密でない'
        assert torch.equal(w.sum(dim=-1), torch.ones(3, 2, 2))

    def test_hard_selection_matches_argmax(self, seed):
        """hard選択のvalue集約がargmaxインデックスの直接選択と厳密一致"""
        attn = LogicAttention(4, n_heads=1, mode='hard').double()
        q = _interior((2, 3, 4), seed=13)
        k = _interior((2, 4, 4), seed=14)
        v = _interior((2, 4, 4), seed=15)
        out = attn.attend(q, k, v, hard=True)
        # スコアを独立に再計算してargmax選択
        scores = q @ k.transpose(-1, -2) + (1 - q) @ (1 - k).transpose(-1, -2)
        idx = scores.argmax(dim=-1)  # (B, Sq)
        for b in range(2):
            for i in range(3):
                assert torch.equal(out[b, i], v[b, idx[b, i]]), \
                    'hard選択がargmaxのvalueと不一致'

    def test_backward_contract(self, seed):
        """backward契約: q/k勾配=soft(softmax)経路、v勾配=one-hot選択の解析勾配。

        STEはattention重みの経路にのみ適用される(forward値=one-hot、
        勾配=softmaxのJacobian)。value集約の∂out/∂vはforward値(one-hot)
        を使うため、v勾配は「選択された行にgrad_outputが流れる」解析式
        Σ_i onehot[b,i,j]·grad_out[b,i] と一致する。
        """
        attn = LogicAttention(4, n_heads=1, mode='hard')
        q0 = _interior((2, 3, 4), seed=16)
        k0 = _interior((2, 4, 4), seed=17)
        v0 = _interior((2, 4, 4), seed=18)
        grad_output = torch.randn(2, 3, 4, dtype=torch.float64)

        ins_hard = [t.detach().clone().requires_grad_(True)
                    for t in (q0, k0, v0)]
        y_hard = attn.attend(*ins_hard, hard=True)
        g_hard = torch.autograd.grad(y_hard, ins_hard,
                                     grad_outputs=grad_output)
        onehot = attn.last_attention.squeeze(1)  # (B, Sq, Sk)

        ins_soft = [t.detach().clone().requires_grad_(True)
                    for t in (q0, k0, v0)]
        y_soft = attn.attend(*ins_soft, hard=False)
        g_soft = torch.autograd.grad(y_soft, ins_soft,
                                     grad_outputs=grad_output)

        # q/k: attention重み経路はsoftmaxの勾配(STE)
        assert torch.allclose(g_hard[0], g_soft[0],
                              atol=TOL_F64_GRAD, rtol=0)
        assert torch.allclose(g_hard[1], g_soft[1],
                              atol=TOL_F64_GRAD, rtol=0)
        # v: one-hot選択の解析勾配
        expected_gv = torch.einsum('bij,bid->bjd', onehot, grad_output)
        assert torch.allclose(g_hard[2], expected_gv,
                              atol=TOL_F64_GRAD, rtol=0)


class TestBooleanAttentionEquivalence:
    """boolean_attention.pyのXNOR+softmax実装との数値等価(非回帰アンカー)。

    同じLutStack重みをLogicAttentionへ移植し、同一入力で
    value集約(decoder前の中間表現)が一致することを照合する。
    """

    def _make_pair(self, seed=41):
        torch.manual_seed(seed)
        cfg = BooleanAttentionConfig(
            num_pairs=3, key_bits=3, value_bits=3,
            encoder_sizes=[8, 4], value_encoder_sizes=[8, 4],
            decoder_sizes=[16, 8],
        )
        model = DLGNBooleanAttention(cfg)
        attn = LogicAttention(
            d_model=3, n_heads=1, mode='soft',
            qk_sizes=[8, 4], v_sizes=[8, 4],
            share_qk=True, tau=cfg.tau_attn,
        )
        # 重み移植(indicesはbufferなのでstate_dictに含まれる)
        attn.k_proj.load_state_dict(model.key_encoder.state_dict())
        attn.v_proj.load_state_dict(model.value_encoder.state_dict())
        return model, attn, cfg

    def _boolean_selected(self, model, cfg, x, hard):
        """boolean_attentionの構成部品からvalue集約を直接計算する"""
        batch = x.shape[0]
        query, keys, values = model.parse_input(x)
        q = model.query_encoder(query)
        k = model.key_encoder(
            keys.reshape(batch * cfg.num_pairs, cfg.key_bits),
        ).reshape(batch, cfg.num_pairs, -1)
        v = model.value_encoder(
            values.reshape(batch * cfg.num_pairs, cfg.value_bits),
        ).reshape(batch, cfg.num_pairs, -1)
        q_exp = q.unsqueeze(1)
        xnor = q_exp * k + (1.0 - q_exp) * (1.0 - k)
        scores = xnor.sum(dim=-1)
        if hard:
            idx = scores.argmax(dim=-1)
            return values, keys, query, v[torch.arange(batch), idx]
        w = F.softmax(scores / cfg.tau_attn, dim=-1)
        return values, keys, query, (w.unsqueeze(-1) * v).sum(dim=1)

    def test_soft_equivalence(self, seed):
        model, attn, cfg = self._make_pair()
        model.train()
        attn.train()
        x = _binary((4, cfg.input_dim), seed=19)
        values, keys, query, expected = self._boolean_selected(
            model, cfg, x, hard=False,
        )
        out = attn(query.unsqueeze(1), keys, values, hard=False)
        assert torch.allclose(out.squeeze(1), expected,
                              atol=TOL_F32_EQ, rtol=0)

    def test_hard_equivalence_exact(self, seed):
        """eval(離散LUT+argmax)ではビット厳密に一致する"""
        model, attn, cfg = self._make_pair()
        model.eval()
        attn.eval()
        x = _binary((8, cfg.input_dim), seed=20)
        with torch.no_grad():
            values, keys, query, expected = self._boolean_selected(
                model, cfg, x, hard=True,
            )
            out = attn(query.unsqueeze(1), keys, values, hard=True)
        assert torch.equal(out.squeeze(1), expected), \
            'hard選択がboolean_attentionと不一致'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
