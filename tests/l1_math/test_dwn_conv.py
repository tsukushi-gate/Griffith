"""DWNConv2d（畳み込み版RAMノード、レバー4）のL1数式テスト。

素朴照合の方針:
    畳み込みは「unfold してタップを集め、アドレスで表を引く」だけなので、
    素朴な二重ループ実装をground truthに置いて突き合わせる。
    LUT評価そのもの（FD/EFD）は dwn.py 側のテストで担保済みなので、
    ここでは **畳み込み特有の部分**（受容野の切り出し、タップ規約、
    出力形状、空間方向の重み共有）を固定する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from griffith_logic.dwn_conv import DWNConv2d  # noqa: E402


def _naive_forward(layer: DWNConv2d, x: torch.Tensor) -> torch.Tensor:
    """素朴なループ実装（ground truth）。

    出力画素ごとに受容野を切り出し、タップ位置の値を並べてアドレスを作り、
    その出力チャネルの表を引く。
    """
    B, C, H, W = x.shape
    kH, kW = layer.kernel_size
    pH, pW = layer.padding
    sH, sW = layer.stride
    xp = torch.nn.functional.pad(x, (pW, pW, pH, pH))
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    out = torch.zeros(B, layer.out_channels, H_out, W_out)
    for b in range(B):
        for oy in range(H_out):
            for ox in range(W_out):
                # 受容野を unfold と同じ順（c, h, w）で平坦化
                rf = []
                for c in range(C):
                    for h in range(kH):
                        for w in range(kW):
                            rf.append(xp[b, c, oy * sH + h, ox * sW + w])
                for o in range(layer.out_channels):
                    addr = 0
                    for j in range(layer.n_inputs):
                        bit = 1 if rf[int(layer.taps[o, j])] > 0 else 0
                        addr |= bit << j          # LSB=スロット0（dwn.py と同規約）
                    val = layer.luts[o, addr]
                    out[b, o, oy, ox] = 1.0 if val > 0 else 0.0
    return out


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_matches_naive_loop(n_inputs):
    """素朴ループと完全一致（畳み込みの切り出しとタップ規約の固定）。"""
    torch.manual_seed(0)
    layer = DWNConv2d(4, 5, kernel_size=3, n_inputs=n_inputs,
                      input_channels_per_tree=4, seed=0).eval()
    x = (torch.rand(2, 4, 6, 7) > 0.5).float()
    with torch.no_grad():
        got = layer(x)
        want = _naive_forward(layer, x)
    assert torch.equal(got, want)


def test_output_shape_and_binary():
    """出力は {0,1} で、形状は通常の畳み込みと同じ規則に従う。"""
    torch.manual_seed(1)
    layer = DWNConv2d(3, 8, kernel_size=3, stride=2, padding=1,
                      n_inputs=4, seed=1).eval()
    x = (torch.rand(2, 3, 8, 8) > 0.5).float()
    with torch.no_grad():
        y = layer(x)
    assert y.shape == (2, 8, 4, 4)
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_weights_shared_across_space():
    """空間方向は重み共有。同じ受容野パターンなら同じ出力になる。"""
    torch.manual_seed(2)
    layer = DWNConv2d(2, 4, kernel_size=3, padding=1, n_inputs=4, seed=2).eval()
    x = torch.zeros(1, 2, 9, 9)
    # 左上と右下に同一のパターンを置く（padding の影響を受けない内側）
    patch = (torch.rand(2, 3, 3) > 0.5).float()
    x[0, :, 1:4, 1:4] = patch
    x[0, :, 5:8, 5:8] = patch
    with torch.no_grad():
        y = layer(x)
    assert torch.equal(y[0, :, 2, 2], y[0, :, 6, 6])


def test_gate_count_is_resolution_independent():
    """ゲート数（配信サイズ）は解像度に依存しない＝レバー3と両立する。"""
    layer = DWNConv2d(3, 16, n_inputs=6, seed=3).eval()
    assert layer.num_gates == 16
    for size in (8, 32):
        x = (torch.rand(1, 3, size, size) > 0.5).float()
        with torch.no_grad():
            layer(x)
        assert layer.num_gates == 16, 'ゲート数が解像度で変わってはいけない'


def test_train_eval_forward_identical():
    """★DWNの肝: 学習時と評価時で forward が完全に一致する。

    LutConv2d(softmax16g) は学習時=連続緩和 / 評価時=argmax で別物であり、
    この不一致が離散化ギャップの源だった。DWN は両方とも「表を引く」だけ。
    """
    torch.manual_seed(4)
    layer = DWNConv2d(3, 6, n_inputs=4, seed=4)
    x = (torch.rand(2, 3, 5, 5) > 0.5).float()
    layer.train()
    with torch.no_grad():
        y_train = layer(x)
    layer.eval()
    with torch.no_grad():
        y_eval = layer(x)
    assert torch.equal(y_train, y_eval)
