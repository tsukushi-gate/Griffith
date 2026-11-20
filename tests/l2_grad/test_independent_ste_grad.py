"""independent_ste の勾配が STE 規約どおりであること。

forward は 1[sigmoid(w) > 0.5] で微分がほぼ全域ゼロなので、素の autograd では
学習できない。backward を素通しにして、上流の勾配がそのまま表パラメータへ届く
ことをここで確認する。
"""

import pytest
import torch

from griffith_logic.conv import LutConv2d, _BinarizeTableSTE


def test_ste_forward_is_threshold():
    p = torch.tensor([0.0, 0.4, 0.5, 0.6, 1.0])
    got = _BinarizeTableSTE.apply(p)
    torch.testing.assert_close(got, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_ste_backward_is_identity():
    """backward は上流の勾配をそのまま返す(値の大きさも変えない)。"""
    p = torch.tensor([0.1, 0.9, 0.49, 0.51], requires_grad=True)
    out = _BinarizeTableSTE.apply(p)
    upstream = torch.tensor([1.0, -2.0, 3.5, 0.25])
    out.backward(upstream)
    torch.testing.assert_close(p.grad, upstream)


def test_dwn_backward_has_no_sigmoid_attenuation():
    """independent_dwn は表が飽和しても勾配が減衰しない。

    independent_ste は sigmoid を挟むので勾配が sigmoid'(x) 倍される。
    学習が進んで |x| が育つと勾配が消える(これが independent_dwn を
    用意した動機)。両者の差をここで固定する。
    """
    def grad_at(parameterization: str, value: float) -> float:
        torch.manual_seed(0)
        layer = LutConv2d(
            in_channels=8, out_channels=4, kernel_size=3, padding=1,
            tree_depth=1, n_inputs=2, parameterization=parameterization,
            input_channels_per_tree=8,
        )
        with torch.no_grad():
            layer.weights.fill_(value)
        layer.train()
        layer(torch.randint(0, 2, (2, 8, 5, 5)).float()).sum().backward()
        return float(layer.weights.grad.abs().sum())

    # 飽和領域: sigmoid'(4) ≈ 0.018 なので ste は 2桁近く小さくなる
    ste_sat = grad_at('independent_ste', 4.0)
    dwn_sat = grad_at('independent_dwn', 0.9)  # clamp 範囲内の飽和端
    assert dwn_sat > ste_sat * 10, (
        f'independent_dwn の勾配が減衰していない前提が崩れている: '
        f'dwn={dwn_sat:.4g} ste={ste_sat:.4g}'
    )


@pytest.mark.parametrize('param', ['independent_ste', 'independent_dwn'])
@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_table_params_receive_gradient(n_inputs, param):
    """閾値の両側どちらの表エントリにも勾配が届く(死んだパラメータが出ない)。"""
    torch.manual_seed(0)
    layer = LutConv2d(
        in_channels=8, out_channels=4, kernel_size=3, padding=1,
        tree_depth=2, n_inputs=n_inputs,
        parameterization=param, input_channels_per_tree=8,
    )
    layer.train()
    x = torch.randint(0, 2, (2, 8, 5, 5)).float()
    layer(x).sum().backward()

    assert layer.weights.grad is not None
    assert layer.weights.grad.abs().sum() > 0, '表パラメータに勾配が届いていない'


@pytest.mark.parametrize('n_inputs', [2, 4])
def test_gradient_flows_to_input(n_inputs):
    """入力側にも勾配が流れる(層を積めることの確認)。"""
    torch.manual_seed(0)
    layer = LutConv2d(
        in_channels=8, out_channels=4, kernel_size=3, padding=1,
        tree_depth=2, n_inputs=n_inputs,
        parameterization='independent_ste', input_channels_per_tree=8,
    )
    layer.train()
    x = torch.rand(2, 8, 5, 5, requires_grad=True)
    layer(x).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
