"""hypothesis共通戦略ライブラリ。

property-basedテストで共有する入力生成戦略を一元管理する。
プロファイル(ci/dev)の登録は tests/conftest.py で行う。

使い方:
    sys.path.insert(0, <testsディレクトリ>)
    from strategies import unit_floats, n_inputs_st, tensor_st
"""

import numpy as np
import torch
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst  # hypothesis本体同梱(追加extra不要)

# 温度パラメータの対数域(GroupSum tau等の契約範囲)
TEMPERATURE_LOG10_MIN = -3.0  # 10^-3
TEMPERATURE_LOG10_MAX = 3.0   # 10^+3

# 極端スケールの上限(有限のまま桁あふれ挙動を叩く)
EXTREME_MAGNITUDE = 1e30


def finite_floats(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    """NaN/infを除外した[min_value, max_value]のfloat戦略"""
    return st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=False,
    )


def extreme_floats() -> st.SearchStrategy[float]:
    """±1e30級の極端スケールを含む有限float戦略(NaN/infは含まない)"""
    return finite_floats(-EXTREME_MAGNITUDE, EXTREME_MAGNITUDE)


def n_inputs_st() -> st.SearchStrategy[int]:
    """LutLayerがサポートするLUT入力数N(2, 4, 6)"""
    return st.sampled_from([2, 4, 6])


def binary_inputs(n: int) -> st.SearchStrategy[list[float]]:
    """{0.0, 1.0}のみからなる長さnのリスト(二値テンソル生成用)"""
    return st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n)


def unit_floats() -> st.SearchStrategy[float]:
    """[0, 1]の有限float(LUT入力の契約域)"""
    return finite_floats(0.0, 1.0)


def temperature_st() -> st.SearchStrategy[float]:
    """温度パラメータ: [1e-3, 1e3]の対数一様分布"""
    return finite_floats(TEMPERATURE_LOG10_MIN, TEMPERATURE_LOG10_MAX).map(
        lambda e: 10.0 ** e
    )


def tensor_st(
    shape_st: st.SearchStrategy[tuple[int, ...]],
    elements: st.SearchStrategy[float],
    dtype=np.float64,
) -> st.SearchStrategy[torch.Tensor]:
    """torchテンソル戦略(hypothesis.extra.numpy経由でnp→torch変換)。

    Args:
        shape_st: 形状タプルの戦略(例: st.tuples(st.integers(1, 8), st.just(4)))
        elements: 要素値のfloat戦略
        dtype: numpy dtype(既定float64。torch側のdtypeもこれに追従)
    """
    return shape_st.flatmap(
        lambda shape: npst.arrays(dtype=dtype, shape=shape, elements=elements)
    ).map(lambda arr: torch.from_numpy(arr.copy()))
