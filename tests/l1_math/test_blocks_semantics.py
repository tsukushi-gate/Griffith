"""L1: blocks部品の数式性質。

    - Add: 二値頂点{0,1}でビット演算と厳密一致(全パターン掃引)
    - Concat/Residual: ビット順・マージ意味論の厳密検証
    - position_codes: binary=2進表現、gray=隣接1bit差、
      thermometer=popcount単調(コード表の正当性)
    - LogicDropout: ドロップ率の統計(±3σ)とeval時identity
"""

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ

from griffith_logic.blocks import (
    Add,
    Concat,
    LogicDropout,
    LogicPositionalEncoding,
    Residual,
    position_codes,
)


def _all_bit_pairs(n_inputs):
    """{0,1}^n_inputs の全組み合わせ(各入力は幅1)"""
    return list(itertools.product([0, 1], repeat=n_inputs))


class TestAddBinaryVertices:
    @pytest.mark.parametrize('mode,op', [
        ('or', lambda a, b: a | b),
        ('xor', lambda a, b: a ^ b),
        ('sum', lambda a, b: a + b),
    ])
    @pytest.mark.parametrize('n_in', [2, 3])
    def test_matches_bitwise_exact(self, mode, op, n_in):
        """二値頂点でsoft式が離散意味論と厳密一致(全2^n通り掃引)"""
        add = Add(mode)
        for bits in _all_bit_pairs(n_in):
            xs = [torch.tensor([[float(b)]], dtype=torch.float64)
                  for b in bits]
            y = add(xs)
            expected = bits[0]
            for b in bits[1:]:
                expected = op(expected, b)
            assert y.item() == float(expected), \
                f'add({mode}) {bits} → {y.item()} != {expected}'


class TestConcatSemantics:
    def test_column_order(self):
        """先頭要素が出力ビット列の先頭(IR concat slot 0規約と一致)"""
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0]])
        c = torch.tensor([[4.0, 5.0]])
        y = Concat()([a, b, c])
        assert torch.equal(y, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))


class TestResidualSemantics:
    def test_concat_is_identity_then_inner(self):
        """merge='concat': 前半=恒等枝、後半=inner出力"""

        class _Not(torch.nn.Module):
            def forward(self, x):
                return 1.0 - x

        x = torch.tensor([[0.0, 1.0, 1.0]])
        y = Residual(_Not(), merge='concat')(x)
        assert torch.equal(y[:, :3], x)
        assert torch.equal(y[:, 3:], 1.0 - x)

    @pytest.mark.parametrize('merge,expected', [
        ('or', [0.0, 1.0, 1.0, 1.0]),
        ('xor', [0.0, 1.0, 1.0, 0.0]),
    ])
    def test_merge_binary_semantics(self, merge, expected):
        """{0,1}二値頂点でmerge(x, inner(x))がビット演算と厳密一致"""

        class _Const(torch.nn.Module):
            """inner(x) = [0, 0, 1, 1] 固定"""

            def forward(self, x):
                return torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        x = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        y = Residual(_Const(), merge=merge)(x)
        assert torch.equal(y, torch.tensor([expected]))


class TestPositionCodes:
    def test_binary_is_base2(self):
        table = position_codes(8, 3, 'binary')
        for p in range(8):
            for j in range(3):
                assert table[p, j].item() == float((p >> j) & 1)

    def test_gray_adjacent_one_bit(self):
        """gray: 隣接位置のコードがちょうど1bit差"""
        table = position_codes(16, 4, 'gray')
        for p in range(15):
            diff = (table[p] != table[p + 1]).sum().item()
            assert diff == 1, f'位置{p}→{p + 1}の差が{diff}bit'

    @pytest.mark.parametrize('code', ['binary', 'gray'])
    def test_codes_unique(self, code):
        table = position_codes(8, 3, code)
        rows = {tuple(r.tolist()) for r in table}
        assert len(rows) == 8, f'{code}コードに重複がある'

    def test_thermometer_popcount_and_monotone(self):
        """thermometer: popcount==位置、かつ前位置のコードを包含(単調)"""
        table = position_codes(5, 4, 'thermometer')
        for p in range(5):
            assert table[p].sum().item() == float(p)
        for p in range(4):
            assert torch.all(table[p] <= table[p + 1]), '単調包含が破れている'

    def test_capacity_errors(self):
        with pytest.raises(AssertionError, match='2\\^width'):
            position_codes(9, 3, 'binary')
        with pytest.raises(AssertionError, match='seq_len-1'):
            position_codes(6, 4, 'thermometer')

    def test_encoding_module_appends_table(self):
        """forward: 出力=[x, テーブルflatten](位置優先ビット順)"""
        enc = LogicPositionalEncoding(4, 3, code='gray')
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = enc(x)
        assert y.shape == (2, 2 + 12)
        assert torch.equal(y[:, :2], x)
        flat = enc.table.reshape(-1)
        assert torch.equal(y[0, 2:], flat)
        assert torch.equal(y[1, 2:], flat)
        # bits_flat(IR constantノードのbits)とも一致
        assert enc.bits_flat == [int(b) for b in flat.tolist()]


class TestDropoutStatistics:
    N = 200_000  # 大標本(±3σ判定のため)

    @pytest.mark.parametrize('mode', ['neutral', 'flip', 'zero'])
    def test_drop_rate_within_3sigma(self, mode, seed):
        """マスク率が二項分布の±3σ内(p=0.3)"""
        p = 0.3
        drop = LogicDropout(p=p, mode=mode)
        drop.train()
        x = torch.full((1, self.N), 0.25)  # どの方式でも変化が検出できる値
        y = drop(x)
        rate = (~torch.isclose(y, x)).float().mean().item()
        sigma = (p * (1 - p) / self.N) ** 0.5
        assert abs(rate - p) < 3 * sigma, (
            f'{mode}: rate={rate:.4f}, p={p}, 3σ={3 * sigma:.4f}'
        )

    @pytest.mark.parametrize('mode,value', [
        ('neutral', 0.5), ('flip', 0.75), ('zero', 0.0),
    ])
    def test_masked_value_contract(self, mode, value, seed):
        """マスク位置の値が方式の契約どおり(x=0.25)"""
        drop = LogicDropout(p=0.5, mode=mode)
        drop.train()
        x = torch.full((1, 1000), 0.25)
        y = drop(x)
        changed = ~torch.isclose(y, x)
        assert changed.any()
        assert torch.allclose(
            y[changed], torch.full_like(y[changed], value),
            atol=TOL_F64_EQ, rtol=0,
        )

    @pytest.mark.parametrize('mode', ['neutral', 'flip', 'zero'])
    def test_eval_identity(self, mode, seed):
        """eval時は厳密に恒等"""
        drop = LogicDropout(p=0.9, mode=mode)
        drop.eval()
        x = torch.rand(8, 32)
        assert torch.equal(drop(x), x)

    def test_p_zero_identity_in_train(self, seed):
        drop = LogicDropout(p=0.0, mode='neutral')
        drop.train()
        x = torch.rand(8, 32)
        assert torch.equal(drop(x), x)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
