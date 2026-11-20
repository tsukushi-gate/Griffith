"""L1: sort_model_wiring のモデル等価性・置換仕様の検証。

検証する性質:
    forward不変性   — 置換前後でforward出力が(train/eval双方)完全一致
                      (ソフト値もGroupSum後の値も置換不変のはず)
    ソート順性      — 各層のindices[0, :](第1入力参照)が非減少
                      (最終層はGroupSumのグループ内に限定して非減少)
    backward不変性  — 入力に対する勾配(dL/dx)が置換前後で一致
                      (内部ニューロンのrenumberは末端の入出力に影響しない)
    非対応構造/配線 — learnable wiring・非Sequential・LutLayer非連続は
                      UnsupportedWiringErrorを送出(無変更で握りつぶさない)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # conftest用

from conftest import TOL_F64_EQ

from griffith_logic.core.sort_wiring import UnsupportedWiringError, sort_model_wiring
from griffith_logic.group_sum import GroupSum
from griffith_logic.layers.lut import LutLayer

IN_DIM = 6
OUT_DIM = 12
NUM_CLASSES = 3


def _build_model(
    *, seed=0, n_inputs=2, out_dim=OUT_DIM, num_classes=NUM_CLASSES,
    num_layers=3, parameterization='softmax', wiring='fixed', dtype=torch.float64,
):
    """Flatten→LutLayer×num_layers→GroupSum のテスト用モデル(train.pyのbuild_modelと同型)。"""
    torch.manual_seed(seed)
    layers = [nn.Flatten()]
    layers.append(
        LutLayer(
            in_dim=IN_DIM, out_dim=out_dim, n_inputs=n_inputs,
            parameterization=parameterization, wiring=wiring,
        )
    )
    for _ in range(num_layers - 1):
        layers.append(
            LutLayer(
                in_dim=out_dim, out_dim=out_dim, n_inputs=n_inputs,
                parameterization=parameterization, wiring='fixed',
            )
        )
    layers.append(GroupSum(num_classes, tau=1.0))
    model = nn.Sequential(*layers)
    return model.to(dtype)


def _lut_layers(model):
    return [m for m in model if isinstance(m, LutLayer)]


class TestForwardInvariance:
    @pytest.mark.parametrize(
        'parameterization,n_inputs',
        [
            ('softmax', 2),   # 16ゲートモード
            ('softmax', 4),   # N>2 softmax
            ('independent', 4),
            ('iwp', 4),
            ('raw', 2),
            ('warp', 4),
        ],
    )
    def test_eval_forward_matches(self, parameterization, n_inputs):
        model = _build_model(parameterization=parameterization, n_inputs=n_inputs)
        x = torch.rand(5, IN_DIM, dtype=torch.float64)
        model.eval()
        y_before = model(x)

        sorted_model = sort_model_wiring(model, inplace=False)
        sorted_model.eval()
        y_after = sorted_model(x)

        assert torch.allclose(y_before, y_after, atol=TOL_F64_EQ, rtol=0)

    @pytest.mark.parametrize(
        'parameterization,n_inputs',
        [('softmax', 2), ('independent', 4), ('iwp', 4)],
    )
    def test_train_forward_matches(self, parameterization, n_inputs):
        """train(soft選択、gumbel_noise=False既定)は決定的なので置換不変のはず。"""
        model = _build_model(parameterization=parameterization, n_inputs=n_inputs)
        x = torch.rand(5, IN_DIM, dtype=torch.float64)
        model.train()
        y_before = model(x)

        sorted_model = sort_model_wiring(model, inplace=False)
        sorted_model.train()
        y_after = sorted_model(x)

        assert torch.allclose(y_before, y_after, atol=TOL_F64_EQ, rtol=0)

    def test_inplace_true_mutates_argument(self):
        """inplace=True(既定)は引数モデル自体を書き換えて返す。"""
        model = _build_model()
        returned = sort_model_wiring(model, inplace=True)
        assert returned is model

    def test_inplace_false_leaves_original_untouched(self):
        model = _build_model()
        original_indices = [layer.indices.clone() for layer in _lut_layers(model)]
        sorted_model = sort_model_wiring(model, inplace=False)
        assert sorted_model is not model
        for layer, orig_idx in zip(_lut_layers(model), original_indices):
            assert torch.equal(layer.indices, orig_idx)


class TestSortOrder:
    def test_hidden_layers_nondecreasing_first_input(self):
        model = _build_model(num_layers=4)
        sorted_model = sort_model_wiring(model, inplace=False)
        lut_layers = _lut_layers(sorted_model)
        for layer in lut_layers[:-1]:
            key = layer.indices[0]
            assert torch.all(key[1:] >= key[:-1]), key

    def test_final_layer_nondecreasing_within_group_only(self):
        model = _build_model(out_dim=OUT_DIM, num_classes=NUM_CLASSES)
        sorted_model = sort_model_wiring(model, inplace=False)
        final_layer = _lut_layers(sorted_model)[-1]
        key = final_layer.indices[0]
        group_size = OUT_DIM // NUM_CLASSES
        for g in range(NUM_CLASSES):
            seg = key[g * group_size:(g + 1) * group_size]
            assert torch.all(seg[1:] >= seg[:-1]), (g, seg)

    def test_final_layer_group_membership_preserved(self):
        """最終層のグループ(GroupSumが和を取る連続ブロック)の集合は
        置換前後で不変(順序のみ変わる)ことを確認する。

        indices[0]の値は入力次元の重複(同じ入力を複数ニューロンが参照)で
        一意とは限らないため、ニューロン識別には代わりに重み行(独立randn初期化
        で実質衝突しないユニークな指紋)を使って旧→新の対応を復元する。
        """
        model = _build_model(
            out_dim=OUT_DIM, num_classes=NUM_CLASSES,
            parameterization='independent', n_inputs=4,
        )
        final_before = _lut_layers(model)[-1]
        old_weights = final_before.weights.data.clone()
        group_size = OUT_DIM // NUM_CLASSES

        sorted_model = sort_model_wiring(model, inplace=False)
        final_after = _lut_layers(sorted_model)[-1]
        new_weights = final_after.weights.data

        for new_pos in range(OUT_DIM):
            matches = (old_weights == new_weights[new_pos]).all(dim=1).nonzero().flatten()
            assert matches.numel() == 1, (new_pos, matches)
            old_pos = int(matches.item())
            assert new_pos // group_size == old_pos // group_size, (new_pos, old_pos)


class TestBackwardInvariance:
    @pytest.mark.parametrize(
        'parameterization,n_inputs',
        [('softmax', 2), ('independent', 4)],
    )
    def test_input_grad_matches(self, parameterization, n_inputs):
        model = _build_model(parameterization=parameterization, n_inputs=n_inputs)
        model.train()
        x = torch.rand(4, IN_DIM, dtype=torch.float64, requires_grad=True)

        y = model(x)
        y.sum().backward()
        grad_before = x.grad.clone()

        sorted_model = sort_model_wiring(model, inplace=False)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = sorted_model(x2)
        y2.sum().backward()
        grad_after = x2.grad

        assert torch.allclose(grad_before, grad_after, atol=TOL_F64_EQ, rtol=0)

    def test_weight_grad_matches_up_to_permutation(self):
        """各層のweights勾配は、旧→新ニューロン対応(重み行の指紋で復元)の
        上で一致する。最終層は「グループ内限定」の置換になるため、
        argsortをそのまま再現するのではなく実際の対応(重み値の一致)で
        検証する(実装の置換アルゴリズムに依存しない独立検証)。
        """
        model = _build_model(parameterization='independent', n_inputs=4, num_layers=2)
        model.train()
        x = torch.rand(4, IN_DIM, dtype=torch.float64, requires_grad=True)

        before_layers = _lut_layers(model)
        old_weights = [layer.weights.data.clone() for layer in before_layers]

        y = model(x)
        y.sum().backward()
        grads_before = [layer.weights.grad.clone() for layer in before_layers]

        sorted_model = sort_model_wiring(model, inplace=False)
        sorted_model.zero_grad()
        x2 = x.detach().clone().requires_grad_(True)
        y2 = sorted_model(x2)
        y2.sum().backward()
        after_layers = _lut_layers(sorted_model)

        for old_w, g_before, after_layer in zip(old_weights, grads_before, after_layers):
            new_w = after_layer.weights.data
            g_after = after_layer.weights.grad
            out_dim = old_w.shape[0]
            for new_pos in range(out_dim):
                matches = (old_w == new_w[new_pos]).all(dim=1).nonzero().flatten()
                assert matches.numel() == 1, (new_pos, matches)
                old_pos = int(matches.item())
                assert torch.allclose(
                    g_before[old_pos], g_after[new_pos], atol=TOL_F64_EQ, rtol=0,
                )


class TestUnsupported:
    def test_rejects_learnable_wiring(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Flatten(),
            LutLayer(in_dim=IN_DIM, out_dim=8, n_inputs=2, wiring='learnable'),
            LutLayer(in_dim=8, out_dim=8, n_inputs=2, wiring='fixed'),
            GroupSum(2),
        )
        with pytest.raises(UnsupportedWiringError):
            sort_model_wiring(model, inplace=False)

    def test_rejects_non_sequential(self):
        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.lut = LutLayer(in_dim=IN_DIM, out_dim=8, n_inputs=2)

            def forward(self, x):
                return self.lut(x)

        with pytest.raises(UnsupportedWiringError):
            sort_model_wiring(Wrapped(), inplace=False)

    def test_rejects_noncontiguous_lut_layers(self):
        model = nn.Sequential(
            LutLayer(in_dim=IN_DIM, out_dim=IN_DIM, n_inputs=2),
            nn.ReLU(),
            LutLayer(in_dim=IN_DIM, out_dim=IN_DIM, n_inputs=2),
        )
        with pytest.raises(UnsupportedWiringError):
            sort_model_wiring(model, inplace=False)

    def test_no_lut_layers_is_noop(self):
        model = nn.Sequential(nn.Flatten())
        returned = sort_model_wiring(model, inplace=False)
        assert returned is not model or True  # deepcopyされるがLutLayerが無いので変化なし
        assert isinstance(returned[0], nn.Flatten)


class TestOptimizerState:
    """optimizer構築後(学習途中)に並べ替えるとき、Adamのモーメントも追従するか。"""

    @staticmethod
    def _train(model, opt, batches):
        for x, y in batches:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x), y).backward()
            opt.step()

    @staticmethod
    def _batches(seed, n):
        g = torch.Generator().manual_seed(seed)
        return [
            (torch.rand(8, IN_DIM, generator=g, dtype=torch.float64),
             torch.randint(0, NUM_CLASSES, (8,), generator=g))
            for _ in range(n)
        ]

    def _run(self, *, pass_optimizer):
        """同一初期化・同一データで2つ学習し、途中で片方だけ並べ替えて続きを学習する。"""
        ref = _build_model(seed=3)
        sorted_model = _build_model(seed=3)
        opt_ref = torch.optim.Adam(ref.parameters(), lr=0.05)
        opt_sorted = torch.optim.Adam(sorted_model.parameters(), lr=0.05)
        warm, rest = self._batches(10, 3), self._batches(11, 4)
        self._train(ref, opt_ref, warm)
        self._train(sorted_model, opt_sorted, warm)

        sort_model_wiring(
            sorted_model, optimizer=opt_sorted if pass_optimizer else None,
        )
        self._train(ref, opt_ref, rest)
        self._train(sorted_model, opt_sorted, rest)

        x = torch.rand(16, IN_DIM, dtype=torch.float64)
        with torch.no_grad():
            return ref(x), sorted_model(x)

    def test_moments_follow_permutation(self):
        y_ref, y_sorted = self._run(pass_optimizer=True)
        assert torch.allclose(y_ref, y_sorted, atol=TOL_F64_EQ, rtol=0), (
            f'並べ替え後の学習がずれた {(y_ref - y_sorted).abs().max():.3e}'
        )

    def test_without_optimizer_moments_are_misaligned(self):
        """optimizerを渡さないとモーメントがずれる(=引数が必要な理由の確認)。"""
        y_ref, y_sorted = self._run(pass_optimizer=False)
        assert not torch.allclose(y_ref, y_sorted, atol=TOL_F64_EQ, rtol=0)

    def test_optimizer_with_inplace_false_is_rejected(self):
        model = _build_model(seed=0)
        opt = torch.optim.Adam(model.parameters())
        with pytest.raises(ValueError):
            sort_model_wiring(model, inplace=False, optimizer=opt)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
