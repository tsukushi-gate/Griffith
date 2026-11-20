"""勾配フロー捕捉(gradcap.py)の数値等価テスト。

エッジ勾配再計算(compute_edge_grads)が core/functions.py のbackwardと
同一の勾配を与えることを「scatter_add(edge_grads) == autogradのgrad_x」で固定する。
"""

import pytest
import torch

from griffith_logic import GroupSum, LutLayer
from tools.editor.backend.gradcap import (
    GradCapture,
    build_grad_flow,
    compute_edge_grads,
    scatter_edge_grads,
)
from tools.editor.backend.monitor import decode_u8

IN_DIM = 32
OUT_DIM = 40
BATCH = 16


def _run_capture(parameterization: str):
    """単層LutLayerでforward/backwardし、捕捉値とlayerを返す。"""
    torch.manual_seed(0)
    layer = LutLayer(
        IN_DIM, OUT_DIM, n_inputs=2,
        connections='random', parameterization=parameterization,
    )
    capture = GradCapture()
    capture.attach([('n1_lutlayer', layer)])
    try:
        x = torch.rand(BATCH, IN_DIM, requires_grad=True)
        y = layer(x)
        # 出力勾配が非自明になるようランダム係数で損失を作る
        loss = (y * torch.randn_like(y)).sum()
        loss.backward()
    finally:
        capture.detach()
    return layer, capture, x


class TestEdgeGradEquivalence:
    @pytest.mark.parametrize('param', ['softmax', 'independent', 'raw'])
    def test_scatter_add_matches_autograd_grad_x(self, param):
        layer, capture, x = _run_capture(param)
        g_out = capture.grad_out['n1_lutlayer']
        x_in = capture.x_in['n1_lutlayer']

        edge = compute_edge_grads(layer, x_in, g_out)
        assert edge is not None
        assert edge.shape == (BATCH, OUT_DIM, 2)

        # 再計算エッジ勾配のscatter_addが、autogradの入力勾配と一致する
        grad_x_re = scatter_edge_grads(edge, layer.indices, IN_DIM)
        assert x.grad is not None
        assert torch.allclose(grad_x_re, x.grad, atol=1e-5), (
            f'{param}: scatter(edge_grads)とautograd grad_xが不一致 '
            f'(max diff={float((grad_x_re - x.grad).abs().max())})'
        )

    def test_hard_st_ste_gradient(self):
        """hard_st(CAGE)はSTE勾配(softmax(w/τ_b))で等価になる。"""
        torch.manual_seed(0)
        layer = LutLayer(
            IN_DIM, OUT_DIM, n_inputs=2,
            connections='random', parameterization='softmax', selection='hard_st',
        )
        capture = GradCapture()
        capture.attach([('n1_lutlayer', layer)])
        try:
            x = torch.rand(BATCH, IN_DIM, requires_grad=True)
            y = layer(x)
            (y * torch.randn_like(y)).sum().backward()
        finally:
            capture.detach()
        edge = compute_edge_grads(
            layer, capture.x_in['n1_lutlayer'], capture.grad_out['n1_lutlayer']
        )
        assert edge is not None
        grad_x_re = scatter_edge_grads(edge, layer.indices, IN_DIM)
        assert torch.allclose(grad_x_re, x.grad, atol=1e-5)


class TestGradFlowFrame:
    def test_trace_order_is_loss_to_input(self):
        """traceのorder昇順が損失側(後段)→入力側(前段)になっている。"""
        torch.manual_seed(0)
        l1 = LutLayer(IN_DIM, OUT_DIM, n_inputs=2, connections='random')
        l2 = LutLayer(OUT_DIM, 20, n_inputs=2, connections='random')
        model = torch.nn.Sequential(l1, l2, GroupSum(k=10, tau=1.0))
        lut_layers = [('n0_lutlayer', l1), ('n1_lutlayer', l2)]

        capture = GradCapture()
        capture.attach(lut_layers)
        try:
            x = torch.rand(BATCH, IN_DIM)
            out = model(x)
            loss = torch.nn.functional.cross_entropy(
                out, torch.randint(0, 10, (BATCH,))
            )
            loss.backward()
        finally:
            capture.detach()

        frame = build_grad_flow(7, float(loss.item()), lut_layers, capture)
        assert frame['type'] == 'grad_flow'
        assert frame['step'] == 7
        # order=0 が損失側(後段=n1)、order=1 が入力側(前段=n0)
        assert frame['layers'][0]['layer'] == 'n1_lutlayer'
        assert frame['layers'][0]['order'] == 0
        assert frame['layers'][1]['layer'] == 'n0_lutlayer'
        assert frame['layers'][1]['order'] == 1
        # 各層にgrad_out/grad_w/edge_gradが揃っている
        for entry, out_dim, n in [
            (frame['layers'][0], 20, 2), (frame['layers'][1], OUT_DIM, 2),
        ]:
            assert decode_u8(entry['grad_out']).shape == (out_dim,)
            assert decode_u8(entry['grad_w']).shape == (out_dim,)
            assert decode_u8(entry['edge_grad']).shape == (out_dim, n)
        assert set(frame['layer_grad_norm']) == {'n0_lutlayer', 'n1_lutlayer'}

    def test_grad_w_matches_weights_grad(self):
        """grad_wフレーム値がweights.gradのニューロン別ノルムと一致(量子化誤差内)。"""
        layer, capture, _x = _run_capture('softmax')
        frame = build_grad_flow(1, 0.5, [('n1_lutlayer', layer)], capture)
        entry = frame['layers'][0]
        decoded = decode_u8(entry['grad_w'])
        expected = layer.weights.grad.detach().norm(dim=-1)
        scale = entry['grad_w']['scale']
        assert torch.allclose(decoded, expected, atol=scale / 255 + 1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
