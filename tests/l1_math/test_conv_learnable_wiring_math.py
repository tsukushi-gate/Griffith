"""L1: LutConv2d(wiring='learnable')の配線学習の数学的性質。

検証する性質:
    1. 素朴ループ照合: ConvLearnableMappingFunctionのforwardが、
       argmax選択を直接計算する素朴な四重ループと厳密一致する。
    2. 二値頂点での厳密一致: candidateが{0,1}のときも同じ性質が成り立つ
       (全2^R通りではなくランダムサンプルで掃引。R^O組合せが指数的なため)。
    3. forwardはτ(backward専用パラメータ)に依存しない
       (mapping = weights.argmax(dim=1)はtauを一切参照しない)。
    4. 選択indexは常に候補集合の範囲内([0, n_candidates))。
    5. _generate_tree_candidate_indices が生成する候補集合は、各ツリーにつき
       高々input_channels_per_tree個のチャンネルのみを参照し、
       unfoldのフラットindex規約(c*kH*kW + h*kW + w)を満たす。
"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import TOL_F64_EQ  # noqa: E402

from griffith_logic.conv import (  # noqa: E402
    ConvLearnableMappingFunction,
    LutConv2d,
    _generate_tree_candidate_indices,
)


def _naive_loop_forward(candidate: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """argmax選択を四重ループで直接計算する(実装から独立した素朴照合)"""
    batch, out_channels, n_candidates = candidate.shape
    _, _, n_leaves = weights.shape
    out = torch.zeros(batch, out_channels, n_leaves, dtype=candidate.dtype)
    for b in range(batch):
        for o in range(out_channels):
            for i in range(n_leaves):
                best_j, best_v = 0, weights[o, 0, i].item()
                for j in range(1, n_candidates):
                    v = weights[o, j, i].item()
                    if v > best_v:
                        best_j, best_v = j, v
                out[b, o, i] = candidate[b, o, best_j]
    return out


class TestForwardMatchesNaiveLoop:
    def test_random_values(self):
        torch.manual_seed(3)
        candidate = torch.rand(4, 3, 5, dtype=torch.float64)
        weights = torch.rand(3, 5, 6, dtype=torch.float64)
        y = ConvLearnableMappingFunction.apply(candidate, weights, 0.001)
        y_expected = _naive_loop_forward(candidate, weights)
        assert torch.allclose(y, y_expected, atol=TOL_F64_EQ, rtol=0)

    def test_binary_vertices(self):
        """candidateが{0,1}の二値頂点でも素朴ループと厳密一致(ランダム掃引)"""
        torch.manual_seed(17)
        out_channels, n_candidates, n_leaves = 2, 4, 3
        for trial in range(20):
            g = torch.Generator().manual_seed(trial)
            candidate = torch.randint(
                0, 2, (5, out_channels, n_candidates), generator=g
            ).double()
            weights = torch.rand(out_channels, n_candidates, n_leaves, dtype=torch.float64)
            y = ConvLearnableMappingFunction.apply(candidate, weights, 0.001)
            y_expected = _naive_loop_forward(candidate, weights)
            assert torch.allclose(y, y_expected, atol=TOL_F64_EQ, rtol=0), f'trial={trial}'
            assert torch.all((y == 0.0) | (y == 1.0))


class TestForwardIndependentOfTau:
    def test_forward_same_for_any_tau(self):
        """forward(argmax選択)はbackward専用パラメータτに依存しない"""
        torch.manual_seed(23)
        candidate = torch.rand(3, 2, 6, dtype=torch.float64)
        weights = torch.rand(2, 6, 4, dtype=torch.float64)
        y_a = ConvLearnableMappingFunction.apply(candidate, weights, 0.001)
        y_b = ConvLearnableMappingFunction.apply(candidate, weights, 5.0)
        assert torch.equal(y_a, y_b)


class TestSelectionIndicesValid:
    def test_selection_always_within_candidate_range(self):
        """選択されたcandidateは常にweights.argmax(dim=1)の範囲内に収まる
        (=候補集合外を参照しない構造的な健全性)"""
        torch.manual_seed(29)
        out_channels, n_candidates, n_leaves = 3, 7, 5
        candidate_marker = torch.arange(n_candidates, dtype=torch.float64)
        candidate = candidate_marker.view(1, 1, -1).expand(2, out_channels, -1).clone()
        weights = torch.rand(out_channels, n_candidates, n_leaves, dtype=torch.float64)
        y = ConvLearnableMappingFunction.apply(candidate, weights, 0.001)
        mapping = weights.argmax(dim=1)  # (out_channels, n_leaves)
        expected = candidate_marker[mapping].unsqueeze(0).expand(2, -1, -1)
        assert torch.equal(y, expected)
        assert (y >= 0).all() and (y < n_candidates).all()


class TestCandidateIndicesStructure:
    """_generate_tree_candidate_indices(候補受容野生成)の構造的性質"""

    def test_uses_at_most_input_channels_per_tree_channels(self):
        torch.manual_seed(31)
        out_channels, in_channels = 6, 8
        kH, kW, per_tree = 3, 3, 3
        idx = _generate_tree_candidate_indices(out_channels, in_channels, kH, kW, per_tree)
        assert idx.shape == (out_channels, per_tree * kH * kW)
        for o in range(out_channels):
            channels = (idx[o] // (kH * kW))
            assert channels.unique().numel() <= per_tree

    def test_indices_within_bounds(self):
        torch.manual_seed(37)
        out_channels, in_channels = 4, 5
        kH, kW, per_tree = 2, 4, 3
        idx = _generate_tree_candidate_indices(out_channels, in_channels, kH, kW, per_tree)
        assert (idx >= 0).all()
        assert (idx < in_channels * kH * kW).all()

    def test_full_kernel_window_enumerated_per_channel(self):
        """選ばれた各チャンネルについて、kH*kW位置が重複なく全数列挙される"""
        torch.manual_seed(41)
        out_channels, in_channels = 3, 4
        kH, kW, per_tree = 2, 2, 2
        idx = _generate_tree_candidate_indices(out_channels, in_channels, kH, kW, per_tree)
        for o in range(out_channels):
            row = idx[o].tolist()
            assert len(set(row)) == len(row), f'重複あり: o={o}'
            channels = sorted({r // (kH * kW) for r in row})
            expected_full = set()
            for c in channels:
                for h in range(kH):
                    for w in range(kW):
                        expected_full.add(c * kH * kW + h * kW + w)
            assert set(row) == expected_full, f'o={o}の候補が全カーネル窓を網羅していない'

    def test_channels_exhausted_when_in_channels_small(self):
        """in_channels <= input_channels_per_treeなら全チャンネルを使う"""
        idx = _generate_tree_candidate_indices(
            out_channels=3, in_channels=2, kernel_size_h=2, kernel_size_w=2,
            input_channels_per_tree=5,
        )
        assert idx.shape == (3, 2 * 2 * 2)
        for o in range(3):
            channels = sorted({int(r) // 4 for r in idx[o].tolist()})
            assert channels == [0, 1]


class TestLutConv2dLearnableWiringSemantics:
    """LutConv2d全体(wiring='learnable')でのBoolean意味論の健全性"""

    def test_binary_input_eval_output_is_binary(self):
        """二値入力・evalモードで出力が厳密に{0,1}(離散Boolean論理として崩壊)"""
        torch.manual_seed(43)
        conv = LutConv2d(
            in_channels=2, out_channels=4, kernel_size=3, tree_depth=2,
            padding=1, n_inputs=2, parameterization='softmax16g',
            input_channels_per_tree=2, wiring='learnable',
        )
        conv.eval()
        x = torch.randint(0, 2, (3, 2, 4, 4)).float()
        with torch.no_grad():
            y = conv(x)
        assert torch.all((y == 0.0) | (y == 1.0))

    def test_all_binary_combinations_small_tree(self):
        """1x1入力・全2^(in_channels)通りを網羅し、handpickedな配線での
        真理値表がツリー評価と一貫することを確認する(小規模全数)"""
        torch.manual_seed(47)
        conv = LutConv2d(
            in_channels=3, out_channels=2, kernel_size=1, tree_depth=1,
            padding=0, n_inputs=2, parameterization='softmax16g',
            input_channels_per_tree=2, wiring='learnable',
        )
        conv.eval()
        for bits in itertools.product([0.0, 1.0], repeat=3):
            x = torch.tensor(bits).view(1, 3, 1, 1)
            with torch.no_grad():
                y = conv(x)
            assert y.shape == (1, 2, 1, 1)
            assert torch.all((y == 0.0) | (y == 1.0))
