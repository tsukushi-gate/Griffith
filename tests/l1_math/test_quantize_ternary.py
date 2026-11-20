"""L1: BitNet b1.58式 absmean ternary量子化(convert/binarize.py)の数理性質。

参照意味論(core/reference.py::reference_quantize_ternary):
    scale = mean(|W|)                         (absmean, per='channel'は行ごと)
    q     = clip(round(W / scale), -1, +1)    (∈ {-1, 0, +1})
    mask  = (q != 0)

検証する性質:
    - 値域: q ∈ {-1, 0, +1} のみ(全数)
    - scale: 正値(>0、eps下限)
    - mask整合: mask == (q != 0)
    - 冪等性: 量子化済みモデル(実効重み q·scale)を再量子化しても q 不変
    - 実装⇔参照一致(binarize.quantize_ternary ⇔ reference)を TOL_F64_EQ で照合
    - extract_ternary_mask: ternary重みモデルから 0マスクを復元

厳密比較(値域・mask)は許容誤差なし、float一致のみ TOL_F64_EQ を使う。
"""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest定数用

from conftest import TOL_F64_EQ

from griffith_logic.convert.binarize import (
    extract_ternary_mask,
    quantize_ternary,
)
from griffith_logic.convert.transcribe import sparsify_from_mask
from griffith_logic.core.reference import reference_quantize_ternary


def _rand_weight(out: int, in_: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out, in_, generator=g, dtype=torch.float64)


class TestTernaryValueRange:
    @pytest.mark.parametrize('per', ['channel', 'tensor'])
    @pytest.mark.parametrize('seed', range(5))
    def test_q_in_ternary_set(self, per, seed):
        w = _rand_weight(8, 16, seed)
        q, scale, mask = quantize_ternary(w, per=per)
        uniq = torch.unique(q.to(torch.float64))
        assert set(uniq.tolist()).issubset({-1.0, 0.0, 1.0}), uniq

    @pytest.mark.parametrize('per', ['channel', 'tensor'])
    def test_scale_positive(self, per):
        w = _rand_weight(8, 16, seed=1)
        _, scale, _ = quantize_ternary(w, per=per)
        assert torch.all(scale > 0)

    def test_zero_row_does_not_nan(self):
        w = _rand_weight(4, 8, seed=2)
        w[1] = 0.0  # 全ゼロ行(scale=0 を eps でクランプ)
        q, scale, mask = quantize_ternary(w, per='channel')
        assert torch.all(torch.isfinite(scale))
        assert torch.all(q[1] == 0)  # ゼロ行はすべて0へ
        assert not mask[1].any()


class TestMaskConsistency:
    @pytest.mark.parametrize('per', ['channel', 'tensor'])
    @pytest.mark.parametrize('seed', range(5))
    def test_mask_equals_nonzero(self, per, seed):
        w = _rand_weight(8, 16, seed)
        q, _, mask = quantize_ternary(w, per=per)
        assert torch.equal(mask, q != 0)


class TestIdempotence:
    @pytest.mark.parametrize('per', ['channel', 'tensor'])
    @pytest.mark.parametrize('seed', range(5))
    def test_requantize_stable(self, per, seed):
        """実効重み W_q = q·scale を再量子化しても q は不変(量子化の冪等性)。"""
        w = _rand_weight(8, 16, seed)
        q1, scale1, _ = quantize_ternary(w, per=per)
        w_eff = q1.to(torch.float64) * scale1  # dequant
        q2, _, _ = quantize_ternary(w_eff, per=per)
        assert torch.equal(q1, q2)


class TestReferenceEquivalence:
    @pytest.mark.parametrize('per', ['channel', 'tensor'])
    @pytest.mark.parametrize('seed', range(8))
    def test_impl_matches_reference(self, per, seed):
        w = _rand_weight(8, 16, seed)
        q, scale, mask = quantize_ternary(w, per=per)
        q_ref, scale_ref, mask_ref = reference_quantize_ternary(w, per=per)
        assert torch.allclose(
            q.to(torch.float64), q_ref.to(torch.float64),
            atol=TOL_F64_EQ, rtol=0,
        )
        # scale形状は実装(out,)と参照(out,1)で違い得るので値でbroadcast照合
        assert torch.allclose(
            scale.reshape(-1) if scale.ndim else scale,
            scale_ref.reshape(-1) if scale_ref.ndim else scale_ref,
            atol=TOL_F64_EQ, rtol=0,
        )
        assert torch.equal(mask, mask_ref)


class TestKnownValues:
    def test_absmean_threshold(self):
        """|W|/scale が 0.5 未満は0、以上は±1(round-half-to-even境界)。"""
        # scale=mean|W|=1 になるよう構成: |w| の平均が 1
        w = torch.tensor([[0.4, 1.6, -0.4, -1.6]], dtype=torch.float64)  # mean|W|=1.0
        q, scale, _ = quantize_ternary(w, per='channel')
        assert scale.reshape(-1).item() == pytest.approx(1.0)
        # round(0.4)=0, round(1.6)=2->clip1, round(-0.4)=0, round(-1.6)=-2->clip-1
        assert q.reshape(-1).tolist() == [0.0, 1.0, 0.0, -1.0]


class TestExtractTernaryMask:
    def test_extract_from_ternary_model(self):
        """{-1,0,+1}近傍で保存されたLinear重みから0マスクを復元する。"""
        model = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 3))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([
                [1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 1.0],
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]))
        masks = extract_ternary_mask(model)
        # 少なくとも1層分のマスクが返る
        assert len(masks) >= 1
        first = next(iter(masks.values()))
        assert first.dtype == torch.bool
        assert first.shape == (4, 6)
        # 明示した0/非0構造が保存されている
        assert not first[3].any()  # 全ゼロ行
        assert first[0].tolist() == [True, False, True, False, True, False]


class TestSparsifyFromMask:
    def test_selects_only_masked_in(self):
        """マスク内非ゼロ数>=kのニューロンは masked-out を選ばない。"""
        w = torch.tensor([
            [3.0, 0.0, 2.0, 0.0, 1.0],   # masked-in {0,2,4}(3本)
            [0.0, 5.0, 0.0, 4.0, 0.0],   # masked-in {1,3}(2本)
        ], dtype=torch.float64)
        mask = w != 0
        indices, w_sel, bias = sparsify_from_mask(w, mask, k=2)
        assert indices.shape == (2, 2)   # (k, out)
        # neuron0: |W|上位2 = col0(3),col2(2) を選ぶ。col4は切り捨て
        sel0 = set(indices[:, 0].tolist())
        assert sel0 == {0, 2}
        # neuron1: masked-in {1,3} をちょうど選ぶ
        sel1 = set(indices[:, 1].tolist())
        assert sel1 == {1, 3}

    def test_pad_when_fewer_than_k(self):
        """マスク内非ゼロ数<kのニューロンは埋めスロットの重みが0。"""
        w = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        mask = w != 0                     # 1本だけ
        indices, w_sel, bias = sparsify_from_mask(w, mask, k=2)
        assert w_sel.shape == (1, 2)
        # 有効1本(3.0)+埋め1本(0.0)
        vals = sorted(w_sel[0].tolist())
        assert vals == [0.0, 3.0]

    def test_bias_recalibrates_truncated(self):
        """切り捨てた masked-in 接続が calib 平均活性でバイアスへ移る。"""
        w = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float64)  # 全masked-in
        mask = w != 0
        calib = torch.ones(4, 3, dtype=torch.float64)  # mean(x_j)=1
        # k=2: col2(1.0)が切り捨て → bias_shift = 1.0*mean = 1.0
        _, _, bias = sparsify_from_mask(w, mask, k=2, calib_inputs=calib)
        assert bias.item() == pytest.approx(1.0)
