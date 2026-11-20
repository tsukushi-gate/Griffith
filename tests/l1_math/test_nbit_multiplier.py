"""nbit_multiplier(n)(griffith_logic/datasets/logic.py)のL1性質テスト。

n-bit×n-bit符号なし整数乗算のビットベクタ真理値表(LSB-first)が、Pythonの
素朴な整数乗算(a*b)と全パターンで一致することを確認する(nbit_adderと同じ
規約: 入力=2n bit LSB-first(a→b連結)、出力=2n bit LSB-first(a*bの2n bit表現、
最大値(2^n-1)^2 < 2^(2n)なので桁あふれしない)。
"""

import pytest

from griffith_logic.datasets import nbit_multiplier
from griffith_logic.datasets.logic import LOGIC_DATASETS


class TestNbitMultiplier:
    """入力=2n bit(a,bをLSB-firstで連結: a_0..a_{n-1}, b_0..b_{n-1})、
    出力=2n bit(積a*b、LSB-first、2n bit幅で桁あふれしない)。
    """

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_matches_python_reference_exhaustive(self, n):
        """全2^(2n)パターンでPython組み込みa*bと厳密一致(全出力ビット)"""
        spec = nbit_multiplier(n)
        assert spec.n_inputs == 2 * n
        assert spec.n_outputs == 2 * n
        assert len(spec.dataset) == 1 << (2 * n)

        mask = (1 << (2 * n)) - 1
        for k in range(1 << (2 * n)):
            bits = tuple((k >> j) & 1 for j in range(2 * n))
            a = sum(bits[j] << j for j in range(n))
            b = sum(bits[n + j] << j for j in range(n))
            expected_product = (a * b) & mask
            expected_bits = [(expected_product >> j) & 1 for j in range(2 * n)]

            _, y = spec.dataset[k]
            assert [int(v) for v in y.tolist()] == expected_bits, (
                f'n={n}, a={a}, b={b}: 期待={expected_bits}, 実際={y.tolist()}'
            )

    def test_no_overflow(self):
        """(2^n-1)^2は常に2n bitに収まる(桁あふれなし)ことを性質として確認"""
        for n in (2, 3, 4, 5):
            max_val = (1 << n) - 1
            assert max_val * max_val < (1 << (2 * n))

    def test_widths(self):
        for n in (1, 2, 5):
            spec = nbit_multiplier(n)
            assert spec.n_inputs == 2 * n
            assert spec.n_outputs == 2 * n
            x, y = spec.dataset.tensors
            assert x.shape == (1 << (2 * n), 2 * n)
            assert y.shape == (1 << (2 * n), 2 * n)

    def test_name_and_repeat(self):
        spec = nbit_multiplier(3, repeat=2)
        assert spec.name == 'mul3'
        assert len(spec.dataset) == (1 << 6) * 2

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            nbit_multiplier(0)

    def test_registry_keys(self):
        for key, n in (('mul2', 2), ('mul3', 3), ('mul4', 4)):
            assert key in LOGIC_DATASETS
            spec = LOGIC_DATASETS[key]()
            assert spec.n_inputs == 2 * n and spec.n_outputs == 2 * n
            assert len(spec.dataset) == 1 << (2 * n)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
