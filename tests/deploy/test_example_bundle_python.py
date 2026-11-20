"""examples/33_bundle_python.py(Python API だけでバンドル → Python/C 整数一致)の実行テスト。

example は `main()` の中で Python 推論 ⇔ C 推論の完全一致と XOR の正解を assert する。
ここでは import して実行し、既定(sweep と dense-flat)が Python API でも
効いていることを確かめる。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from .conftest import CC

EXAMPLE = Path(__file__).resolve().parents[2] / 'examples' / '33_bundle_python.py'


def _load_example():
    spec = importlib.util.spec_from_file_location('example_33_bundle_python', EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(CC is None, reason='Cコンパイラ(cc/clang/gcc)が無い環境')
def test_example_bundle_python(tmp_path):
    result = _load_example().main(tmp_path)
    assert np.array_equal(result['py_scores'], result['c_scores'])
    assert result['py_scores'].argmax(axis=1).tolist() == [0, 1, 1, 0]
    # apply_compress / resolve_dense_flat を呼んだので、CLI 既定と同じ圧縮が効く
    assert 'sweep' in result['compress']['applied']
    assert result['dense_flat']['applied'] is True
