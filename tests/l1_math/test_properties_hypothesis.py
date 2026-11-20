"""L1: 数学的性質のproperty化(hypothesis)。

既存のL1テスト(固定点・全列挙)と重複してよい。propertyにより
入力域をランダム探索へ拡大し、境界値も自動生成することが目的。

対応するL1性質(各テストのdocstringにも対応を明記):
    分割単位性        — test_basis_properties.TestPartitionOfUnity
    二値頂点one-hot   — test_basis_properties.TestBinaryVertices
    真理値表往復同一性 — ir/weights.weights_from_truth_tables ⇔
                        export/discretize.truth_tables_from_lut_layer
    soft⇔hard一致     — test_discretization.TestSoftHardConsistency

戦略は tests/strategies.py、プロファイルは tests/conftest.py で管理。
"""

import os
import sys

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest/strategies用

from conftest import TOL_F32_EQ, TOL_F64_EQ
from strategies import binary_inputs, n_inputs_st, tensor_st, unit_floats

from griffith_logic.export.discretize import truth_tables_from_lut_layer
from griffith_logic.ir.weights import weights_from_truth_tables
from griffith_logic.lut_layer import LutLayer, _compute_basis_butterfly

# 真理値表↔重みが可逆なパラメータ化(ir/weights.pyの表現可能性より)
REVERSIBLE_PARAMS = ['softmax16g', 'independent', 'raw', 'warp']

# soft⇔hard一致に使う飽和ロジット振幅: sigmoid(±20)≈1∓2e-9 で
# soft-hardギャップがTOL_F32_EQ(1e-6)未満に収まる
SATURATION_LOGIT_M = 20.0


def _unit_with_boundaries() -> st.SearchStrategy[float]:
    """[0,1]域+境界値0/1を明示的に混合した要素戦略"""
    return st.one_of(st.sampled_from([0.0, 1.0]), unit_floats())


def _draw_torch_seed(data) -> None:
    """torch側RNGをhypothesisの描画で固定(失敗ケースをblobから完全再現可能に)"""
    torch.manual_seed(data.draw(st.integers(0, 2 ** 31 - 1), label='torch_seed'))


def _tts_st(n_inputs: int) -> st.SearchStrategy[list[int]]:
    """N入力LUTの真理値表(butterfly基底順2^Nビット整数)のリスト戦略"""
    limit = 1 << (1 << n_inputs)
    return st.lists(st.integers(0, limit - 1), min_size=1, max_size=4)


def _build_layer_from_tts(param: str, n: int, tts: list[int],
                          logit_m: float | None = None) -> LutLayer:
    """真理値表からweightsを確定したLutLayerを構築する"""
    layer_param = 'softmax' if param == 'softmax16g' else param
    layer = LutLayer(max(8, n), len(tts), n_inputs=n, parameterization=layer_param)
    kwargs = {} if logit_m is None else {'logit_m': logit_m}
    with torch.no_grad():
        layer.weights.copy_(weights_from_truth_tables(param, n, tts, **kwargs))
    return layer


class TestPartitionOfUnityProperty:
    @given(n=n_inputs_st(), data=st.data())
    def test_sum_to_one_random_domain(self, n, data):
        """分割単位性 Σ_k basis_k = 1 のproperty版(ランダム[0,1]域+境界値)。

        対応: test_basis_properties.TestPartitionOfUnity.test_sum_to_one_float64
        """
        gathered = data.draw(tensor_st(
            st.tuples(st.integers(1, 8), st.integers(1, 4), st.just(n)),
            _unit_with_boundaries(),
        ), label='gathered')
        basis = _compute_basis_butterfly(gathered, n)
        sums = basis.sum(dim=-1)
        assert torch.allclose(
            sums, torch.ones_like(sums), atol=TOL_F64_EQ, rtol=0,
        ), f'max|Σ-1|={((sums - 1).abs().max()):.3e}'

    @given(n=n_inputs_st(), data=st.data())
    def test_nonnegative_random_domain(self, n, data):
        """基底の非負性のproperty版(確率解釈の前提)。

        対応: test_basis_properties.TestPartitionOfUnity.test_nonnegative
        """
        gathered = data.draw(tensor_st(
            st.tuples(st.integers(1, 8), st.integers(1, 4), st.just(n)),
            _unit_with_boundaries(),
        ), label='gathered')
        basis = _compute_basis_butterfly(gathered, n)
        assert (basis >= 0).all()


class TestBinaryVertexOneHotProperty:
    @given(n=n_inputs_st(), data=st.data())
    def test_one_hot_random_binary_pattern(self, n, data):
        """二値頂点でのone-hot性のproperty版(ランダム二値パターン)。

        hot位置はビット規約 k = Σ_j x_j·2^j(bit(k,j)=x_j)。
        対応: test_basis_properties.TestBinaryVertices.test_one_hot_exact
        """
        bits = data.draw(binary_inputs(n), label='bits')
        gathered = torch.tensor([[bits]], dtype=torch.float64)  # (1, 1, n)
        basis = _compute_basis_butterfly(gathered, n)
        k = sum(int(b) << j for j, b in enumerate(bits))
        expected = torch.zeros(1 << n, dtype=torch.float64)
        expected[k] = 1.0
        assert torch.equal(basis[0, 0], expected)


class TestTruthTableRoundTripProperty:
    @pytest.mark.parametrize('param', REVERSIBLE_PARAMS)
    @given(data=st.data())
    def test_weights_roundtrip_identity(self, param, data):
        """真理値表→weights→eval離散化→真理値表の往復同一性のproperty版。

        可逆なパラメータ化(softmax16g/independent/raw/warp)でランダムな
        真理値表がビット厳密に復元されること。
        対応: ir/weights.py の表現可能性契約(tests/test_ir_build.pyで固定点検証済み)
        """
        _draw_torch_seed(data)
        n = 2 if param == 'softmax16g' else data.draw(n_inputs_st(), label='n')
        tts = data.draw(_tts_st(n), label='tts')
        layer = _build_layer_from_tts(param, n, tts)
        recovered = [int(t) for t in truth_tables_from_lut_layer(layer)]
        assert recovered == tts, (
            f'{param} n={n}: {[hex(t) for t in tts]} → {[hex(t) for t in recovered]}'
        )


class TestSoftHardConsistencyProperty:
    @pytest.mark.parametrize('param', REVERSIBLE_PARAMS)
    @given(data=st.data())
    def test_saturated_soft_matches_hard(self, param, data):
        """飽和重みでの二値入力上のsoft(train)⇔hard(eval)一致のproperty版。

        真理値表からweightsを確定(飽和ロジット±20)し、ランダム二値入力で
        train forwardとeval forwardの一致を検証する。
        対応: test_discretization.TestSoftHardConsistency
        """
        _draw_torch_seed(data)
        n = 2 if param == 'softmax16g' else data.draw(n_inputs_st(), label='n')
        tts = data.draw(_tts_st(n), label='tts')
        layer = _build_layer_from_tts(param, n, tts, logit_m=SATURATION_LOGIT_M)
        if layer.parameterization == 'warp':
            layer.gumbel = False  # 決定的に比較
        rows = data.draw(st.lists(
            binary_inputs(layer.in_dim), min_size=1, max_size=4,
        ), label='rows')
        x = torch.tensor(rows)

        layer.train()
        y_soft = layer(x)
        layer.eval()
        y_hard = layer(x)

        assert torch.allclose(y_soft, y_hard, atol=TOL_F32_EQ), (
            f'{param} n={n}: soft-hardギャップ {(y_soft - y_hard).abs().max():.3e}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
