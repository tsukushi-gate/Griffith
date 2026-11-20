"""DWNConv2d の勾配テスト（レバー4）。

DWN の forward は不連続（アドレスで表を引く）ので gradcheck は使えない。
代わりに **有限差分の定義そのもの**を素朴実装で再現して突き合わせる:

    FD (表の勾配) : grad_luts[o, i] = Σ_画素 1[i = addr] · G
    EFD(入力勾配) : grad_gathered[.., j] = G · Σ_k C[addr, j, k] · U_o[k]

EFD係数 C の正しさは dwn.py 側のテストで担保済みなので、ここでは
**畳み込み経路（unfold→タップ集め→画素をバッチへ畳む）を通しても
勾配が定義どおりに流れるか**を固定する。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.dwn_conv import DWNConv2d  # noqa: E402


def _addresses(layer, x):
    """素朴に各(画素, 出力ch)のアドレスを求める（テスト内 ground truth）。"""
    B, C, H, W = x.shape
    kH, kW = layer.kernel_size
    pH, pW = layer.padding
    xp = torch.nn.functional.pad(x, (pW, pW, pH, pH))
    H_out = H + 2 * pH - kH + 1
    W_out = W + 2 * pW - kW + 1
    addrs = torch.zeros(B, H_out, W_out, layer.out_channels, dtype=torch.long)
    for b in range(B):
        for oy in range(H_out):
            for ox in range(W_out):
                rf = [xp[b, c, oy + h, ox + w]
                      for c in range(C) for h in range(kH) for w in range(kW)]
                for o in range(layer.out_channels):
                    a = 0
                    for j in range(layer.n_inputs):
                        a |= (1 if rf[int(layer.taps[o, j])] > 0 else 0) << j
                    addrs[b, oy, ox, o] = a
    return addrs


def test_fd_table_gradient_matches_definition():
    """表の勾配は「アクセスしたエントリへの出力勾配の総和」に一致する。"""
    torch.manual_seed(0)
    layer = DWNConv2d(3, 4, kernel_size=3, padding=1, n_inputs=4,
                      input_channels_per_tree=3, ste=False, seed=0).eval()
    x = (torch.rand(2, 3, 5, 5) > 0.5).float()
    y = layer(x)
    g = torch.randn_like(y)
    y.backward(g)

    addrs = _addresses(layer, x)                      # (B, H, W, out_ch)
    want = torch.zeros_like(layer.luts)
    B, _, H, W = y.shape
    for b in range(B):
        for oy in range(H):
            for ox in range(W):
                for o in range(layer.out_channels):
                    want[o, addrs[b, oy, ox, o]] += g[b, o, oy, ox]
    assert torch.allclose(layer.luts.grad, want, atol=1e-5)


def test_efd_input_gradient_matches_definition():
    """入力勾配は EFD の定義 G · Σ_k C[addr, j, k] · U[k] に一致する。"""
    torch.manual_seed(1)
    layer = DWNConv2d(2, 3, kernel_size=3, padding=1, n_inputs=4,
                      input_channels_per_tree=2, ste=False, seed=1).eval()
    x = (torch.rand(1, 2, 4, 4) > 0.5).float().requires_grad_(True)
    y = layer(x)
    g = torch.randn_like(y)
    y.backward(g)

    # 素朴に、各(画素, 出力ch, スロット)へ流れる勾配を作り、タップ先へ散布する
    addrs = _addresses(layer, x.detach())
    kH, kW = layer.kernel_size
    pH, pW = layer.padding
    C = layer.in_channels
    want = torch.zeros(1, C, 4 + 2 * pH, 4 + 2 * pW)
    B, _, H, W = y.shape
    for b in range(B):
        for oy in range(H):
            for ox in range(W):
                for o in range(layer.out_channels):
                    a = int(addrs[b, oy, ox, o])
                    for j in range(layer.n_inputs):
                        coef = (layer.c_efd[a, j] * layer.luts[o]).sum()
                        contrib = g[b, o, oy, ox] * coef
                        flat = int(layer.taps[o, j])
                        c, rem = divmod(flat, kH * kW)
                        h, w = divmod(rem, kW)
                        want[b, c, oy + h, ox + w] += contrib
    want = want[:, :, pH:pH + 4, pW:pW + 4]
    assert torch.allclose(x.grad, want, atol=1e-5)


def test_gradients_flow_to_both_inputs_and_table():
    """学習に必要な2経路（表・入力）の勾配が両方とも非ゼロで流れる。"""
    torch.manual_seed(2)
    layer = DWNConv2d(3, 5, n_inputs=6, seed=2)
    x = (torch.rand(2, 3, 6, 6) > 0.5).float().requires_grad_(True)
    layer(x).sum().backward()
    assert layer.luts.grad is not None and layer.luts.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_ste_passes_gradient_through_binarization():
    """出力二値化(STE)を挟んでも勾配が素通しされる。"""
    torch.manual_seed(3)
    layer = DWNConv2d(2, 4, n_inputs=4, ste=True, seed=3)
    x = (torch.rand(1, 2, 5, 5) > 0.5).float().requires_grad_(True)
    layer(x).sum().backward()
    assert layer.luts.grad.abs().sum() > 0
