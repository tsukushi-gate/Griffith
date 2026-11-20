"""L2: CAGE conv(TreeHardST)の契約テスト。

TreeHardSTFunction(conv_hard_st.py)はforward不連続(argmax選択)のため
gradcheck(数値微分)を直接適用できない。dense版(test_ste_contracts.py)と
同じ方式で「forward契約」「backward契約」を解析式との厳密照合で検証する。

TreeHardST(CAGE論文のツリー畳み込み版):
    forward契約:  各レベルでargmax選択したhard論理木の出力
                  (推論と同一 = selection gap 0)
    backward契約: 各レベルでsoftmax(w/τ_b)のsoft勾配を持ち、値はhardで
                  伝播するstraight-through関数のautograd勾配と一致
    数値検証:     tree_depth=1ではbackward = soft関数の真の勾配
                  → 中心差分による数値gradcheckが可能
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW, TOL_F64_EQ, TOL_F64_GRAD

from griffith_logic.conv_hard_st import (
    CAGEConvScheduler,
    LutConv2dHardST,
    TreeHardSTFunction,
)
from griffith_logic.core import reference_basis

N_INPUTS = 2
N_ENTRIES = 2 ** N_INPUTS


def _make_tree_case(tree_depth, batch_l=5, out_ch=2, seed=13):
    """(leaves, weights) をfloat64で生成"""
    torch.manual_seed(seed)
    n_leaves = N_INPUTS ** tree_depth
    n_internal = (n_leaves - 1) // (N_INPUTS - 1)
    leaves = torch.rand(batch_l, out_ch, n_leaves, dtype=torch.float64) * 0.8 + 0.1
    weights = torch.randn(out_ch, n_internal, N_ENTRIES, dtype=torch.float64)
    return leaves, weights


def _tree_levels(current, weights, n_inputs):
    """各レベルの (grouped, level_weights) を順に返すジェネレータ用ヘルパ"""
    n_groups = current.shape[-1] // n_inputs
    grouped = current[..., :n_groups * n_inputs].reshape(
        *current.shape[:-1], n_groups, n_inputs,
    )
    return grouped, n_groups


def _hard_tree_reference(leaves, weights, n_inputs, tree_depth, n_entries):
    """forward契約の参照: 各レベルでargmax選択したhard論理木(素朴実装)"""
    current = leaves
    offset = 0
    for _ in range(tree_depth):
        grouped, n_groups = _tree_levels(current, weights, n_inputs)
        w = weights[:, offset:offset + n_groups, :]
        basis = reference_basis(grouped, n_inputs)
        w_hard = F.one_hot(w.argmax(dim=-1), n_entries).to(basis.dtype)
        current = (basis * w_hard.unsqueeze(0)).sum(dim=-1)
        offset += n_groups
    return current.squeeze(-1)


def _ste_tree_reference(leaves, weights, n_inputs, tree_depth, n_entries, tau_b):
    """backward契約の参照(純autograd)。

    各レベルの出力を soft + (hard - soft).detach() とすることで、
    値はhard(argmax)で伝播し、勾配はhard活性で評価したsoftmax(w/τ_b)の
    soft関数のヤコビアンになる。TreeHardSTFunctionのbackward仕様と同一。
    """
    current = leaves
    offset = 0
    for _ in range(tree_depth):
        grouped, n_groups = _tree_levels(current, weights, n_inputs)
        w = weights[:, offset:offset + n_groups, :]
        basis = reference_basis(grouped, n_inputs)
        s = F.softmax(w / tau_b, dim=-1)
        soft = (basis * s.unsqueeze(0)).sum(dim=-1)
        w_hard = F.one_hot(w.argmax(dim=-1), n_entries).to(basis.dtype)
        hard = (basis * w_hard.unsqueeze(0)).sum(dim=-1)
        current = soft + (hard - soft).detach()
        offset += n_groups
    return current.squeeze(-1)


def _soft_tree_reference(leaves, weights, n_inputs, tree_depth, n_entries, tau_b):
    """純soft木(softmax(w/τ_b)混合、soft値伝播)。tree_depth=1ではSTE版と勾配一致"""
    current = leaves
    offset = 0
    for _ in range(tree_depth):
        grouped, n_groups = _tree_levels(current, weights, n_inputs)
        w = weights[:, offset:offset + n_groups, :]
        basis = reference_basis(grouped, n_inputs)
        s = F.softmax(w / tau_b, dim=-1)
        current = (basis * s.unsqueeze(0)).sum(dim=-1)
        offset += n_groups
    return current.squeeze(-1)


def _conv_leaves(layer, x):
    """LutConv2dのunfold→リーフ収集パスを素朴に再現"""
    B, _, H, W = x.shape
    kH, kW = layer.kernel_size
    sH, sW = layer.stride
    pH, pW = layer.padding
    patches = F.unfold(x, (kH, kW), padding=(pH, pW), stride=(sH, sW))
    L = patches.shape[2]
    leaves = patches[:, layer.indices_flat]
    leaves = leaves.permute(0, 3, 1, 2).reshape(B * L, layer.out_channels, layer.n_leaves)
    h_out = (H + 2 * pH - kH) // sH + 1
    w_out = (W + 2 * pW - kW) // sW + 1
    return leaves, L, h_out, w_out


class TestTreeHardSTForwardContract:
    @pytest.mark.parametrize('tree_depth', [1, 2])
    def test_forward_equals_argmax_tree(self, tree_depth):
        """forward = 各レベルargmax選択のhard論理木出力(素朴実装と厳密一致)"""
        leaves, weights = _make_tree_case(tree_depth)
        y = TreeHardSTFunction.apply(
            leaves, weights, N_INPUTS, tree_depth, N_ENTRIES, 1.0,
        )
        y_expected = _hard_tree_reference(
            leaves, weights, N_INPUTS, tree_depth, N_ENTRIES,
        )
        assert torch.allclose(y, y_expected, atol=TOL_F64_EQ, rtol=0)

    def test_selection_gap_zero_on_binary_input(self):
        """二値入力・訓練モードのhard出力 == 同じ重みでのeval(離散)出力"""
        torch.manual_seed(17)
        layer = LutConv2dHardST(1, 2, kernel_size=2, tree_depth=2)
        layer.weights.data.normal_()  # residual_initの同値タイを排除
        x = torch.randint(0, 2, (3, 1, 4, 4)).float()

        layer.train()
        y_train = layer(x)
        layer.eval()
        y_eval = layer(x)

        assert torch.equal(y_train, y_eval), (
            f'selection gap = {(y_train - y_eval).abs().max():.3e}'
        )
        # 二値入力×hardゲートの出力は厳密に{0,1}
        assert torch.all((y_train == 0.0) | (y_train == 1.0))


class TestTreeHardSTBackwardContract:
    @pytest.mark.parametrize('tau_b', [1.0, 0.5, 2.0])
    @pytest.mark.parametrize('tree_depth', [1, 2])
    def test_backward_equals_tempered_soft_st(self, tree_depth, tau_b):
        """backward = softmax(w/τ_b)版straight-through関数のautograd勾配と一致"""
        leaves, weights = _make_tree_case(tree_depth)
        torch.manual_seed(19)
        grad_output = torch.randn(
            leaves.shape[0], weights.shape[0], dtype=torch.float64,
        )

        lr = leaves.clone().requires_grad_(True)
        wr = weights.clone().requires_grad_(True)
        y = TreeHardSTFunction.apply(lr, wr, N_INPUTS, tree_depth, N_ENTRIES, tau_b)
        gl, gw = torch.autograd.grad(y, (lr, wr), grad_outputs=grad_output)

        ls = leaves.clone().requires_grad_(True)
        ws = weights.clone().requires_grad_(True)
        y_ref = _ste_tree_reference(ls, ws, N_INPUTS, tree_depth, N_ENTRIES, tau_b)
        gl_ref, gw_ref = torch.autograd.grad(
            y_ref, (ls, ws), grad_outputs=grad_output,
        )

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)
        assert torch.allclose(gl, gl_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_leaves不一致: {(gl - gl_ref).abs().max():.3e}'
        )
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_weights不一致: {(gw - gw_ref).abs().max():.3e}'
        )


def _numerical_grad(f, tensor, grad_output, eps=1e-6):
    """中心差分による数値勾配(fはtensorを参照するクロージャ)"""
    grad = torch.zeros_like(tensor)
    flat = tensor.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        y_plus = f()
        flat[i] = orig - eps
        y_minus = f()
        flat[i] = orig
        gflat[i] = ((y_plus - y_minus) * grad_output).sum() / (2 * eps)
    return grad


class TestTreeHardSTNumericalGradcheck:
    """tree_depth=1では backward = soft関数の真の勾配。

    soft関数(連続)の中心差分と解析backwardを照合する数値gradcheck。
    (hard forward自体は不連続で直接gradcheck不可: test_gradcheck_matrix.py参照)
    """

    @pytest.mark.parametrize('tau_b', [1.0, 0.5])
    def test_gradcheck_depth1_weights_and_leaves(self, tau_b):
        tree_depth = 1
        leaves, weights = _make_tree_case(tree_depth, batch_l=4)
        torch.manual_seed(23)
        grad_output = torch.randn(
            leaves.shape[0], weights.shape[0], dtype=torch.float64,
        )

        lr = leaves.clone().requires_grad_(True)
        wr = weights.clone().requires_grad_(True)
        y = TreeHardSTFunction.apply(lr, wr, N_INPUTS, tree_depth, N_ENTRIES, tau_b)
        gl, gw = torch.autograd.grad(y, (lr, wr), grad_outputs=grad_output)

        l_num = leaves.clone()
        w_num = weights.clone()

        def f_soft():
            return _soft_tree_reference(
                l_num, w_num, N_INPUTS, tree_depth, N_ENTRIES, tau_b,
            )

        gw_num = _numerical_grad(f_soft, w_num, grad_output)
        gl_num = _numerical_grad(f_soft, l_num, grad_output)

        assert torch.allclose(gw, gw_num, **GRADCHECK_KW), (
            f'weights数値勾配不一致: {(gw - gw_num).abs().max():.3e}'
        )
        assert torch.allclose(gl, gl_num, **GRADCHECK_KW), (
            f'leaves数値勾配不一致: {(gl - gl_num).abs().max():.3e}'
        )


class TestLutConv2dHardSTLayer:
    """conv層全体(unfold→ツリー→整形)のfloat64契約検証。

    in_ch=1, out_ch=2, kernel=2, 4x4入力の最小構成。
    """

    @pytest.mark.parametrize('tree_depth', [1, 2])
    def test_layer_grads_match_ste_reference(self, tree_depth):
        torch.manual_seed(7)
        layer = LutConv2dHardST(1, 2, kernel_size=2, tree_depth=tree_depth).double()
        layer.weights.data.normal_()
        layer.train()
        tau_b = 1.5
        layer.tau_b = tau_b

        x = torch.rand(2, 1, 4, 4, dtype=torch.float64) * 0.8 + 0.1
        torch.manual_seed(29)
        grad_output = torch.randn(2, 2, 3, 3, dtype=torch.float64)

        xr = x.clone().requires_grad_(True)
        y = layer(xr)
        gx, gw = torch.autograd.grad(
            y, (xr, layer.weights), grad_outputs=grad_output,
        )

        # 参照: unfoldパス + straight-through木(純autograd)
        xs = x.clone().requires_grad_(True)
        ws = layer.weights.detach().clone().requires_grad_(True)
        leaves, L, h_out, w_out = _conv_leaves(layer, xs)
        out = _ste_tree_reference(
            leaves, ws, N_INPUTS, tree_depth, N_ENTRIES, tau_b,
        )
        y_ref = out.reshape(2, L, 2).permute(0, 2, 1).reshape(2, 2, h_out, w_out)
        gx_ref, gw_ref = torch.autograd.grad(
            y_ref, (xs, ws), grad_outputs=grad_output,
        )

        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)
        assert torch.allclose(gx, gx_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_x不一致: {(gx - gx_ref).abs().max():.3e}'
        )
        assert torch.allclose(gw, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_weights不一致: {(gw - gw_ref).abs().max():.3e}'
        )

    def test_layer_weights_numerical_gradcheck_depth1(self):
        """conv層のweights勾配をsoft-conv関数の中心差分と照合(depth=1)"""
        torch.manual_seed(31)
        layer = LutConv2dHardST(1, 2, kernel_size=2, tree_depth=1).double()
        layer.weights.data.normal_()
        layer.train()

        x = torch.rand(2, 1, 4, 4, dtype=torch.float64) * 0.8 + 0.1
        torch.manual_seed(37)
        grad_output = torch.randn(2, 2, 3, 3, dtype=torch.float64)

        y = layer(x)
        (gw,) = torch.autograd.grad(
            y, (layer.weights,), grad_outputs=grad_output,
        )

        w_num = layer.weights.detach().clone()

        def f_soft_conv():
            leaves, L, h_out, w_out = _conv_leaves(layer, x)
            out = _soft_tree_reference(leaves, w_num, N_INPUTS, 1, N_ENTRIES, 1.0)
            return out.reshape(2, L, 2).permute(0, 2, 1).reshape(2, 2, h_out, w_out)

        gw_num = _numerical_grad(f_soft_conv, w_num, grad_output)
        assert torch.allclose(gw, gw_num, **GRADCHECK_KW), (
            f'weights数値勾配不一致: {(gw - gw_num).abs().max():.3e}'
        )

    def test_rejects_non_softmax_parameterization(self):
        with pytest.raises(ValueError, match='softmax'):
            LutConv2dHardST(1, 2, kernel_size=2, parameterization='independent')


class TestCAGEConvScheduler:
    """τ_b更新式(確信度EMA、CAGE論文 Eq. 11)の数値検証"""

    TAU_MAX = 3.0
    TAU_MIN = 0.5
    BETA = 0.99

    def _make(self, weights):
        layer = LutConv2dHardST(1, 2, kernel_size=2, tree_depth=1)
        layer.weights.data.copy_(weights)
        model = nn.Sequential(layer)
        scheduler = CAGEConvScheduler(
            model, tau_max=self.TAU_MAX, tau_min=self.TAU_MIN, beta=self.BETA,
        )
        return layer, scheduler

    def test_tau_update_matches_formula(self):
        """既知の重みでc→EMA→c_norm→τ_bの2ステップを手計算照合"""
        w = torch.tensor([[[4.0, 0.0, 0.0, 0.0]], [[0.0, 2.0, 0.0, 0.0]]])
        layer, scheduler = self._make(w)
        k = float(N_ENTRIES)

        # 確信度c: 各ゲートのsoftmax最大確率の平均
        probs = torch.softmax(w, dim=-1)
        c = probs.max(dim=-1).values.mean().item()

        ema = 1.0 / k  # 初期値
        for step in range(2):
            scheduler.step()
            ema = self.BETA * ema + (1.0 - self.BETA) * c
            c_norm = (ema - 1.0 / k) / (1.0 - 1.0 / k)
            c_norm = max(0.0, min(1.0, c_norm))
            tau_expected = self.TAU_MAX - (self.TAU_MAX - self.TAU_MIN) * c_norm

            assert abs(scheduler.ema_confidence - ema) < 1e-9, f'step {step}'
            assert abs(scheduler.tau_b - tau_expected) < 1e-9, f'step {step}'
            # 層のτ_bにも反映される
            assert layer.tau_b == scheduler.tau_b

    def test_uniform_weights_keep_tau_max(self):
        """一様重み(確信度=1/K)ではc_norm=0でτ_b=τ_maxのまま"""
        w = torch.zeros(2, 1, 4)
        _, scheduler = self._make(w)
        for _ in range(5):
            scheduler.step()
        assert scheduler.tau_b == self.TAU_MAX

    def test_requires_hard_st_layer(self):
        with pytest.raises(ValueError):
            CAGEConvScheduler(nn.Sequential(nn.Linear(4, 4)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
