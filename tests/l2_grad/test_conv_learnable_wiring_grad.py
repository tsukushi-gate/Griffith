"""L2: LutConv2d(wiring='learnable')のConvLearnableMappingFunction勾配式照合。

検証する数式(DWN論文§3.2 Learnable Mappingを、ツリー(出力チャンネル)次元o
でバッチ化したもの。dense版layers/dwn.py::LearnableMappingFunctionと同じ式を
o=1に潰せば一致する):
    forward:  leaves[b,o,i] = candidate[b,o, argmax_j weights[o,j,i]]
              (ハードargmax、softmax不使用)
    backward: ∂L/∂weights[o,j,i]  = Σ_b (2·candidate[b,o,j]−1)·G[b,o,i]
              ∂L/∂candidate[b,o,j] = Σ_i G[b,o,i]·softmax_{dim=1}(weights[o,:,i]/τ)[j]

forwardはargmax選択で不連続なため数値gradcheck(torch.autograd.gradcheck)は
適用できない(new-component-tddスキル: forward不連続系の契約テスト方式)。
代わりに「forward契約」(素朴ループ参照との厳密一致)と「backward契約」
(解析式との1e-10照合)で検証する。

ground truth: griffith_logic/core/reference.py::
    reference_conv_learnable_mapping_forward
    reference_conv_learnable_mapping_grads
    reference_conv_learnable_wiring_forward (layer統合パイプライン)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ, TOL_F64_GRAD  # noqa: E402

from griffith_logic.conv import ConvLearnableMappingFunction, LutConv2d  # noqa: E402
from griffith_logic.core.reference import (  # noqa: E402
    reference_conv_learnable_mapping_forward,
    reference_conv_learnable_mapping_grads,
    reference_conv_learnable_wiring_forward,
)


def _case(batch=6, out_channels=3, n_candidates=5, n_leaves=4, seed=7, dtype=torch.float64):
    torch.manual_seed(seed)
    candidate = torch.rand(batch, out_channels, n_candidates, dtype=dtype)
    weights = torch.rand(out_channels, n_candidates, n_leaves, dtype=dtype)
    return candidate, weights


class TestConvLearnableMappingForwardContract:
    def test_argmax_selection_matches_reference(self):
        """forward: leaves[b,o,i] = candidate[b,o,argmax_j weights[o,j,i]]"""
        candidate, weights = _case()
        y = ConvLearnableMappingFunction.apply(candidate, weights, 0.001)
        y_ref = reference_conv_learnable_mapping_forward(candidate, weights)
        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)

    def test_forward_shape(self):
        candidate, weights = _case(batch=4, out_channels=2, n_candidates=9, n_leaves=8)
        y = ConvLearnableMappingFunction.apply(candidate, weights, 0.001)
        assert y.shape == (4, 2, 8)

    def test_explicit_selection_values(self):
        """既知の重みで各(o,i)がargmax位置のcandidate値を選ぶことを直接確認"""
        # out_channels=2, n_candidates=3, n_leaves=2
        weights = torch.tensor([
            [[0.9, 0.1], [0.0, 0.8], [0.1, 0.0]],   # o=0: leaf0→cand0, leaf1→cand1
            [[0.0, 0.0], [0.2, 0.9], [0.7, 0.1]],   # o=1: leaf0→cand2, leaf1→cand1
        ])
        candidate = torch.tensor([
            [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]],
        ])  # (1, 2, 3)
        y = ConvLearnableMappingFunction.apply(candidate, weights, 0.001)
        expected = torch.tensor([[[10.0, 20.0], [3.0, 2.0]]])
        assert torch.equal(y, expected)


class TestConvLearnableMappingBackwardContract:
    def _grad_case(self, batch=5, out_channels=3, n_candidates=6, n_leaves=4, seed=11):
        candidate, weights = _case(batch, out_channels, n_candidates, n_leaves, seed=seed)
        torch.manual_seed(seed + 1000)
        grad_output = torch.randn(batch, out_channels, n_leaves, dtype=torch.float64)
        return candidate, weights, grad_output

    @pytest.mark.parametrize('tau', [0.001, 1.0, 2.5])
    def test_grad_weights_and_candidate_formula(self, tau):
        """∂L/∂weights, ∂L/∂candidate の解析式を1e-10で照合"""
        candidate, weights, grad_output = self._grad_case()

        c = candidate.clone().requires_grad_(True)
        w = weights.clone().requires_grad_(True)
        y = ConvLearnableMappingFunction.apply(c, w, tau)
        y.backward(grad_output)

        y_ref, gc_ref, gw_ref = reference_conv_learnable_mapping_grads(
            candidate, weights, tau, grad_output,
        )

        assert torch.allclose(y.detach(), y_ref, atol=TOL_F64_EQ, rtol=0)
        assert torch.allclose(c.grad, gc_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_candidate不一致: {(c.grad - gc_ref).abs().max():.3e}'
        )
        assert torch.allclose(w.grad, gw_ref, atol=TOL_F64_GRAD, rtol=0), (
            f'grad_weights不一致: {(w.grad - gw_ref).abs().max():.3e}'
        )

    def test_needs_input_grad_respected(self):
        """一部の入力にrequires_grad=Falseでもbackwardが疎通する"""
        candidate, weights, grad_output = self._grad_case()
        c = candidate.clone().requires_grad_(False)
        w = weights.clone().requires_grad_(True)
        y = ConvLearnableMappingFunction.apply(c, w, 0.001)
        y.backward(grad_output)
        assert w.grad is not None
        assert c.grad is None

    def test_batches_over_out_channel_independently(self):
        """out_channels=1のスライスを取り出した結果がdense相当の式に一致する
        (トリー次元oでのバッチ化が正しいことの構造確認)"""
        candidate, weights, grad_output = self._grad_case(out_channels=3)
        c = candidate.clone().requires_grad_(True)
        w = weights.clone().requires_grad_(True)
        y = ConvLearnableMappingFunction.apply(c, w, 0.001)
        y.backward(grad_output)

        for o in range(3):
            c1 = candidate[:, o, :].clone().requires_grad_(True)
            w1 = weights[o, :, :].unsqueeze(0).clone().requires_grad_(True)
            g1 = grad_output[:, o, :].unsqueeze(1)
            y1 = ConvLearnableMappingFunction.apply(c1.unsqueeze(1), w1, 0.001)
            y1.backward(g1)
            assert torch.allclose(c.grad[:, o, :], c1.grad, atol=TOL_F64_GRAD, rtol=0)
            assert torch.allclose(w.grad[o], w1.grad[0], atol=TOL_F64_GRAD, rtol=0)


class TestLutConv2dLearnableWiringLayerContract:
    """conv層全体(unfold→候補受容野→Learnable Mapping→softmax16g木)の
    forward契約: core/reference.py::reference_conv_learnable_wiring_forwardと
    厳密一致することを確認する(層への配線が正しいことの統合検証)。"""

    def test_layer_forward_matches_reference_pipeline(self):
        torch.manual_seed(5)
        conv = LutConv2d(
            in_channels=3, out_channels=4, kernel_size=3, tree_depth=2,
            padding=1, n_inputs=2, parameterization='softmax16g',
            input_channels_per_tree=2, wiring='learnable', residual_init=False,
        ).double()
        conv.train()
        x = torch.rand(2, 3, 5, 5, dtype=torch.float64)

        y = conv(x)
        y_ref = reference_conv_learnable_wiring_forward(
            x, conv.mapping_weights, conv.wiring_candidate_indices,
            conv.weights, conv.gate_table.double(),
            conv.kernel_size, conv.stride, conv.padding,
            conv.n_inputs, conv.tree_depth,
        )
        assert torch.allclose(y, y_ref, atol=TOL_F64_EQ, rtol=0)

    def test_layer_backward_finite_and_nonzero(self):
        """layer全体の逆伝播が有限かつmapping_weights/gate weights/xへ流れる"""
        torch.manual_seed(9)
        conv = LutConv2d(
            in_channels=3, out_channels=4, kernel_size=3, tree_depth=2,
            padding=1, n_inputs=2, parameterization='softmax16g',
            input_channels_per_tree=2, wiring='learnable',
        )
        x = torch.rand(2, 3, 5, 5, requires_grad=True)
        y = conv(x)
        y.sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert conv.mapping_weights.grad is not None
        assert torch.isfinite(conv.mapping_weights.grad).all()
        assert conv.mapping_weights.grad.abs().sum() > 0
        assert conv.weights.grad is not None
        assert torch.isfinite(conv.weights.grad).all()

    def test_eval_mode_deterministic(self):
        torch.manual_seed(13)
        conv = LutConv2d(
            in_channels=2, out_channels=3, kernel_size=3, tree_depth=1,
            padding=1, n_inputs=2, parameterization='softmax16g',
            input_channels_per_tree=2, wiring='learnable',
        )
        conv.eval()
        x = torch.rand(2, 2, 4, 4)
        with torch.no_grad():
            y1 = conv(x)
            y2 = conv(x)
        assert torch.equal(y1, y2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
