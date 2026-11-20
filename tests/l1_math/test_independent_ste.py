"""parameterization='independent_ste' の離散化ギャップがゼロであること。

背景
----
'independent' は訓練時に表を連続値(sigmoid)のまま使い、評価時だけ 0/1 に丸める。
表を二値へ寄せる圧力が一切ないため、丸めた瞬間に大量のゲートが反転する。
実測(multimnist_color, n_inputs=4): mAP(soft)=0.6819 に対し mAP(hard)=0.0000。

'independent_ste' は forward で表を二値化する(勾配は素通し)。訓練時の forward が
評価時と完全に一致するので、この崩壊が構造的に起きない。ここではそれを固定する。
"""

import pytest
import torch

from griffith_logic.conv import LutConv2d


def _make(parameterization: str, n_inputs: int, seed: int = 0) -> LutConv2d:
    torch.manual_seed(seed)
    return LutConv2d(
        in_channels=8, out_channels=4, kernel_size=3, padding=1,
        tree_depth=2, n_inputs=n_inputs, parameterization=parameterization,
        input_channels_per_tree=8,
    )


@pytest.mark.parametrize('param', ['independent_ste', 'independent_dwn'])
@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_train_eval_forward_identical(n_inputs, param):
    """二値入力なら train() と eval() の forward がビット単位で一致する。"""
    layer = _make(param, n_inputs)
    torch.manual_seed(1)
    x = torch.randint(0, 2, (2, 8, 5, 5)).float()

    layer.train()
    with torch.no_grad():
        got_train = layer(x)
    layer.eval()
    with torch.no_grad():
        got_eval = layer(x)

    torch.testing.assert_close(got_train, got_eval, rtol=0, atol=0)


@pytest.mark.parametrize('param', ['independent_ste', 'independent_dwn'])
@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_output_is_binary_in_training(n_inputs, param):
    """二値入力なら訓練時の出力も 0/1 になる(連続値が残らない)。"""
    layer = _make(param, n_inputs)
    layer.train()
    torch.manual_seed(2)
    x = torch.randint(0, 2, (2, 8, 5, 5)).float()
    with torch.no_grad():
        y = layer(x)
    assert torch.all((y == 0) | (y == 1)), f'二値でない値がある: {y.unique()[:8]}'


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_independent_does_have_a_gap(n_inputs):
    """対照: 'independent' は train と eval が一致しない(=直した問題の再現)。"""
    layer = _make('independent', n_inputs)
    torch.manual_seed(1)
    x = torch.randint(0, 2, (2, 8, 5, 5)).float()

    layer.train()
    with torch.no_grad():
        got_train = layer(x)
    layer.eval()
    with torch.no_grad():
        got_eval = layer(x)

    assert not torch.equal(got_train, got_eval), (
        'independent で train/eval が一致した。テストの前提が崩れている'
    )


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_matches_discretized_truth_tables(n_inputs):
    """export が読む真理値表が、eval forward と同じ規約になっている。"""
    from griffith_logic.export.discretize import truth_tables_from_conv

    ste = _make('independent_ste', n_inputs, seed=3)
    plain = _make('independent', n_inputs, seed=3)
    tt_ste = truth_tables_from_conv(ste)
    tt_plain = truth_tables_from_conv(plain)
    # 同じ seed なら重みも同じ。二値化規約が同一であることの確認。
    assert tt_ste.tolist() == tt_plain.tolist()


@pytest.mark.parametrize('n_inputs', [2, 4, 6])
def test_dwn_truth_tables_match_eval_forward(n_inputs):
    """independent_dwn の真理値表が eval forward の規約(w>0)と一致する。"""
    import numpy as np

    from griffith_logic.export.discretize import truth_tables_from_conv

    layer = _make('independent_dwn', n_inputs, seed=4)
    got = truth_tables_from_conv(layer)
    hard = (layer.weights > 0.0).to(torch.int64).numpy()
    want = np.array(
        [[int(sum(int(b) << k for k, b in enumerate(hard[oc, g])))
          for g in range(hard.shape[1])]
         for oc in range(hard.shape[0])], dtype=object)
    assert got.tolist() == want.tolist()


def test_seg_model_accepts_n6_and_has_no_gap():
    """SegConvDLGN も n_inputs=6 + independent_dwn を受け、ギャップが無い。

    seg は長らく n_inputs=2 / softmax16g がハードコードされていて、検出で勝った
    レシピを適用できなかった。引数化したうえで、検出側と同じ「train と eval の
    forward がビット一致」をここでも固定する。
    """
    from experiments.segmentation_models import SegConvDLGN

    torch.manual_seed(0)
    model = SegConvDLGN(
        in_channels=9, n_classes=2, k=8, tree_depth=1, readout_group=8,
        input_channels_per_tree=8, n_inputs=6,
        parameterization='independent_dwn',
    )
    torch.manual_seed(1)
    x = torch.randint(0, 2, (2, 9, 16, 16)).float()

    model.train()
    with torch.no_grad():
        got_train = model(x)
    model.eval()
    with torch.no_grad():
        got_eval = model(x)
    torch.testing.assert_close(got_train, got_eval, rtol=0, atol=0)

    model.train()
    model(x).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters()), '勾配が届いていない'


def test_seg_rejects_softmax16g_with_n_gt_2():
    """n>2 で softmax16g を指定したら黙って壊れず明示的に断る。"""
    from experiments.segmentation_models import SegConvDLGN

    with pytest.raises(ValueError, match='softmax16g'):
        SegConvDLGN(in_channels=9, k=8, tree_depth=1, n_inputs=6,
                    parameterization='softmax16g')


def test_dwn_weights_stay_clamped():
    """訓練 forward を通すと表パラメータが [-1,1] に収まる。"""
    layer = _make('independent_dwn', 4)
    with torch.no_grad():
        layer.weights.add_(5.0)          # 範囲外へ押し出す
    layer.train()
    layer(torch.randint(0, 2, (2, 8, 5, 5)).float())
    assert float(layer.weights.detach().max()) <= 1.0
    assert float(layer.weights.detach().min()) >= -1.0
