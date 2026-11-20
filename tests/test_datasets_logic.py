"""論理関数データセット生成器(datasets/logic.py)のテスト。

- 形状・dtype・{0,1}値域・2^n全列挙の網羅性(repeat込み)
- parity/adder/muxの真理値スポットチェック(手計算値との一致)
- random_logicのseed再現性
- L4スモーク: XORをBCE学習し、hard eval(離散化)で4パターン全問正解
"""

import os
import sys
from collections import Counter

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import LOGIC_DATASETS, LutLayer, truth_table_dataset
from griffith_logic.datasets import (
    LogicDatasetSpec,
    full_adder,
    half_adder,
    kv_attention2,
    majority3,
    mux,
    nbit_adder,
    parity,
    random_logic,
    seq101,
    xor2,
)


def _row_of(dataset, bits):
    """butterfly規約でパターンk(=Σ bit_j·2^j)の(入力,出力)行を返す"""
    k = sum(b << j for j, b in enumerate(bits))
    return dataset[k]


# ============================================================
# truth_table_dataset: 形状・dtype・値域・網羅性
# ============================================================

class TestTruthTableDataset:
    def test_shapes_and_dtype(self):
        ds = truth_table_dataset(lambda b: (b[0] & b[1], b[0] | b[1]), 2, 2)
        x, y = ds.tensors
        assert x.shape == (4, 2) and y.shape == (4, 2)
        assert x.dtype == torch.float32 and y.dtype == torch.float32

    def test_values_are_binary(self):
        spec = random_logic(4, 3, seed=7)
        x, y = spec.dataset.tensors
        assert set(x.unique().tolist()) <= {0.0, 1.0}
        assert set(y.unique().tolist()) <= {0.0, 1.0}

    @pytest.mark.parametrize('n_inputs,repeat', [(2, 1), (3, 1), (3, 4)])
    def test_exhaustive_patterns_with_repeat(self, n_inputs, repeat):
        """全2^n入力パターンがちょうどrepeat回ずつ現れる"""
        ds = truth_table_dataset(lambda b: sum(b) % 2, n_inputs, 1, repeat=repeat)
        x, _ = ds.tensors
        assert len(ds) == (1 << n_inputs) * repeat
        patterns = Counter(
            sum(int(b) << j for j, b in enumerate(row)) for row in x.tolist()
        )
        assert patterns == {k: repeat for k in range(1 << n_inputs)}

    def test_butterfly_input_order(self):
        """パターンkのビットj = (k>>j)&1(butterfly規約)"""
        ds = truth_table_dataset(lambda b: b[0], 3, 1)
        x, _ = ds.tensors
        expected = [
            [(k >> j) & 1 for j in range(3)] for k in range(8)
        ]
        assert x.tolist() == [[float(v) for v in row] for row in expected]

    def test_list_and_callable_agree(self):
        """関数指定とリスト指定(butterflyパターン順)が同一データセットになる"""
        fn_ds = truth_table_dataset(lambda b: b[0] ^ b[1], 2, 1)
        # k=0:(0,0)→0, k=1:(1,0)→1, k=2:(0,1)→1, k=3:(1,1)→0
        list_ds = truth_table_dataset([0, 1, 1, 0], 2, 1)
        assert torch.equal(fn_ds.tensors[1], list_ds.tensors[1])

    def test_mapping_table(self):
        """マッピング指定(入力ビットのtupleがキー)も同一データセットになる"""
        table = {(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 0}
        map_ds = truth_table_dataset(table, 2, 1)
        fn_ds = truth_table_dataset(lambda b: b[0] ^ b[1], 2, 1)
        assert torch.equal(map_ds.tensors[1], fn_ds.tensors[1])

    def test_invalid_table_raises(self):
        with pytest.raises(ValueError):  # 長さ不一致
            truth_table_dataset([0, 1, 1], 2, 1)
        with pytest.raises(ValueError):  # 0/1以外の値
            truth_table_dataset([0, 1, 2, 0], 2, 1)
        with pytest.raises(ValueError):  # 出力ビット数不一致
            truth_table_dataset(lambda b: (b[0], b[1]), 2, 1)


# ============================================================
# 定番ファクトリの真理値スポットチェック
# ============================================================

class TestFactories:
    def test_xor2(self):
        spec = xor2()
        assert (spec.name, spec.n_inputs, spec.n_outputs) == ('xor', 2, 1)
        _, y = _row_of(spec.dataset, (1, 0))
        assert y.tolist() == [1.0]
        _, y = _row_of(spec.dataset, (1, 1))
        assert y.tolist() == [0.0]

    def test_parity3(self):
        spec = parity(3)
        assert spec.n_inputs == 3 and len(spec.dataset) == 8
        _, y = _row_of(spec.dataset, (1, 1, 0))
        assert y.tolist() == [0.0]
        _, y = _row_of(spec.dataset, (1, 1, 1))
        assert y.tolist() == [1.0]

    def test_half_adder(self):
        spec = half_adder()
        assert spec.n_outputs == 2
        _, y = _row_of(spec.dataset, (1, 0))
        assert y.tolist() == [1.0, 0.0]  # 1+0 → sum=1, carry=0
        _, y = _row_of(spec.dataset, (1, 1))
        assert y.tolist() == [0.0, 1.0]  # 1+1 → sum=0, carry=1

    def test_full_adder(self):
        spec = full_adder()
        _, y = _row_of(spec.dataset, (1, 1, 1))
        assert y.tolist() == [1.0, 1.0]  # 1+1+1 → sum=1, carry=1
        _, y = _row_of(spec.dataset, (1, 0, 1))
        assert y.tolist() == [0.0, 1.0]  # 1+0+1 → sum=0, carry=1
        _, y = _row_of(spec.dataset, (0, 0, 1))
        assert y.tolist() == [1.0, 0.0]  # 0+0+1 → sum=1, carry=0

    def test_mux21(self):
        """2:1 MUX: 入力(a,b,sel) → sel ? b : a"""
        spec = mux(1)
        assert (spec.name, spec.n_inputs) == ('mux2', 3)
        _, y = _row_of(spec.dataset, (0, 1, 1))  # sel=1 → b=1
        assert y.tolist() == [1.0]
        _, y = _row_of(spec.dataset, (1, 0, 1))  # sel=1 → b=0
        assert y.tolist() == [0.0]
        _, y = _row_of(spec.dataset, (1, 0, 0))  # sel=0 → a=1
        assert y.tolist() == [1.0]

    def test_majority3(self):
        spec = majority3()
        assert (spec.name, spec.n_inputs, spec.n_outputs) == ('majority3', 3, 1)
        _, y = _row_of(spec.dataset, (1, 1, 0))
        assert y.tolist() == [1.0]
        _, y = _row_of(spec.dataset, (1, 0, 0))
        assert y.tolist() == [0.0]

    def test_kv_attention2(self):
        spec = kv_attention2()
        assert (spec.name, spec.n_inputs, spec.n_outputs) == ('kv_attention2', 5, 1)
        _, y = _row_of(spec.dataset, (1, 1, 1, 0, 0))  # q==k0 → v0
        assert y.tolist() == [1.0]
        _, y = _row_of(spec.dataset, (0, 1, 1, 0, 0))  # q==k1 → v1=0
        assert y.tolist() == [0.0]
        _, y = _row_of(spec.dataset, (0, 0, 0, 1, 1))  # q==k0だがv0=0
        assert y.tolist() == [0.0]

    def test_seq101(self):
        spec = seq101()
        assert (spec.name, spec.n_inputs, spec.n_outputs) == ('seq101', 3, 1)
        _, y = _row_of(spec.dataset, (1, 0, 1))
        assert y.tolist() == [1.0]
        _, y = _row_of(spec.dataset, (1, 1, 1))
        assert y.tolist() == [0.0]

    def test_mux41_all_patterns(self):
        """4:1 MUX: 全パターンで出力=データビット[セレクト値]"""
        spec = mux(2)
        assert spec.n_inputs == 6 and len(spec.dataset) == 64
        for x, y in spec.dataset:
            bits = [int(v) for v in x.tolist()]
            sel = bits[4] + 2 * bits[5]
            assert y.tolist() == [float(bits[sel])]

    def test_random_logic_seed_reproducibility(self):
        a = random_logic(4, 3, seed=123)
        b = random_logic(4, 3, seed=123)
        c = random_logic(4, 3, seed=124)
        assert torch.equal(a.dataset.tensors[1], b.dataset.tensors[1])
        assert not torch.equal(a.dataset.tensors[1], c.dataset.tensors[1])

    def test_repeat_passthrough(self):
        spec = parity(3, repeat=5)
        assert len(spec.dataset) == 8 * 5


# ============================================================
# nbit_adder: n-bit ripple-carry加算器(LSB-first規約)
# ============================================================

class TestNbitAdder:
    """入力=2n bit(a,bをLSB-firstで連結: a_0..a_{n-1}, b_0..b_{n-1})、
    出力=n+1 bit(sum、LSB-first、最終桁は繰り上がりcarry)。
    """

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_matches_python_reference_exhaustive(self, n):
        """全2^(2n)パターンでPython組み込みa+bと厳密一致(全出力ビット)"""
        spec = nbit_adder(n)
        assert spec.n_inputs == 2 * n
        assert spec.n_outputs == n + 1
        assert len(spec.dataset) == 1 << (2 * n)

        mask = (1 << (n + 1)) - 1
        for k in range(1 << (2 * n)):
            bits = tuple((k >> j) & 1 for j in range(2 * n))
            a = sum(bits[j] << j for j in range(n))
            b = sum(bits[n + j] << j for j in range(n))
            expected_sum = (a + b) & mask
            expected_bits = [(expected_sum >> j) & 1 for j in range(n + 1)]

            _, y = spec.dataset[k]
            assert [int(v) for v in y.tolist()] == expected_bits, (
                f'n={n}, a={a}, b={b}: 期待={expected_bits}, 実際={y.tolist()}'
            )

    def test_widths(self):
        for n in (1, 2, 5):
            spec = nbit_adder(n)
            assert spec.n_inputs == 2 * n
            assert spec.n_outputs == n + 1
            x, y = spec.dataset.tensors
            assert x.shape == (1 << (2 * n), 2 * n)
            assert y.shape == (1 << (2 * n), n + 1)

    def test_name_and_repeat(self):
        spec = nbit_adder(3, repeat=2)
        assert spec.name == 'adder3'
        assert len(spec.dataset) == (1 << 6) * 2


# ============================================================
# レジストリとDataLoader
# ============================================================

class TestRegistry:
    def test_registry_keys_and_specs(self):
        expected = {
            'xor', 'parity3', 'parity4', 'majority3',
            'half_adder', 'full_adder', 'mux2',
            'kv_attention2', 'seq101',
            'adder2', 'adder3', 'adder4', 'mul2', 'mul3', 'mul4',
        }
        assert set(LOGIC_DATASETS) == expected
        for key, factory in LOGIC_DATASETS.items():
            spec = factory()
            assert isinstance(spec, LogicDatasetSpec)
            assert len(spec.dataset) == 1 << spec.n_inputs
            x, y = spec.dataset.tensors
            assert x.shape[1] == spec.n_inputs and y.shape[1] == spec.n_outputs

    def test_loader_full_batch_default(self):
        spec = LOGIC_DATASETS['parity4']()
        batches = list(spec.loader())
        assert len(batches) == 1
        assert batches[0][0].shape == (16, 4)

    def test_loader_batch_size(self):
        spec = LOGIC_DATASETS['full_adder']()
        batches = list(spec.loader(batch_size=4))
        assert len(batches) == 2
        assert all(bx.shape == (4, 3) for bx, _ in batches)


# ============================================================
# L4スモーク: truth_table_dataset(xor) + BCEで学習→hard evalで全問正解
# ============================================================

class TestL4Smoke:
    def test_xor_bce_hard_exact_match(self):
        """LutLayer(2,8)→LutLayer(8,1)をBCEで学習し、離散化後に4/4正解"""
        torch.manual_seed(0)
        x, y = LOGIC_DATASETS['xor']().dataset.tensors

        model = nn.Sequential(
            LutLayer(2, 8, n_inputs=2),
            LutLayer(8, 1, n_inputs=2),
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        eps = 1e-6  # soft出力の飽和(exact 0/1)によるlog(0)を防ぐ
        for _ in range(300):
            model.train()
            opt.zero_grad()
            loss = F.binary_cross_entropy(model(x).clamp(eps, 1 - eps), y)
            loss.backward()
            opt.step()

        model.eval()  # 離散化(argmaxで完全なBoolean回路に崩壊)
        with torch.no_grad():
            pred = (model(x) > 0.5).float()
        assert torch.equal(pred, y), f'hard eval不一致: pred={pred.tolist()}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
