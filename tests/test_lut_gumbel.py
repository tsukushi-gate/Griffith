"""LutLayer の Gumbelノイズオプション(Mind the Gap, arXiv:2506.07500)の契約テスト。

Mind the Gap は softmaxゲート選択ロジットに Gumbel(0,1) ノイズを注入して
学習し、離散化ギャップを縮小する。本テストは以下を担保する:

1. 後方互換: gumbel_noise=False(既定)で train/eval とも既存LutLayerと bitwise一致
2. eval不変: gumbel_noise=True でも eval 出力は False時と bitwise一致
3. train変化+微分可能: True の train forward は確率的に変化し backward が有限勾配
4. 再現性: torch.manual_seed 固定で train forward が再現
5. 収束スモーク: XOR/parity で gumbel あり学習が hard acc=100% に到達
"""

import itertools
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer  # noqa: E402

_IN_DIM = 8
_OUT_DIM = 6
_BATCH = 5


def _make_layer(seed: int = 0, **kwargs) -> LutLayer:
    """seed固定で決定的にsoftmax LutLayerを構築する(weights/indices一致用)。"""
    torch.manual_seed(seed)
    return LutLayer(
        in_dim=_IN_DIM, out_dim=_OUT_DIM, n_inputs=2,
        parameterization='softmax', **kwargs,
    )


def _make_x(seed: int = 7) -> torch.Tensor:
    """[0,1] 内部点の入力(batch, in_dim)。"""
    torch.manual_seed(seed)
    return torch.rand(_BATCH, _IN_DIM)


class TestBackwardCompat:
    """契約1: gumbel_noise=False(既定)は既存LutLayerと bitwise 一致。"""

    def test_train_forward_bitwise_identical(self):
        ref = _make_layer()                      # gumbelキーなし = 既存
        off = _make_layer(gumbel_noise=False)    # 明示False
        x = _make_x()
        ref.train()
        off.train()
        y_ref = ref(x)
        y_off = off(x)
        assert torch.equal(y_ref, y_off), 'train: gumbel_noise=False が既存と不一致'

    def test_eval_forward_bitwise_identical(self):
        ref = _make_layer()
        off = _make_layer(gumbel_noise=False)
        x = _make_x()
        ref.eval()
        off.eval()
        with torch.no_grad():
            assert torch.equal(ref(x), off(x)), 'eval: gumbel_noise=False が既存と不一致'

    def test_no_rng_consumption_when_off(self):
        """False時はGumbel乱数を消費しない(RNG消費順の後方互換を保証)。"""
        off = _make_layer(gumbel_noise=False)
        x = _make_x()
        off.train()
        torch.manual_seed(999)
        state_before = torch.get_rng_state()
        _ = off(x)
        state_after = torch.get_rng_state()
        assert torch.equal(state_before, state_after), 'False時に乱数を消費している'


class TestEvalInvariant:
    """契約2: gumbel_noise=True でも eval 出力は False時と bitwise 一致。"""

    def test_eval_equal_regardless_of_gumbel(self):
        on = _make_layer(gumbel_noise=True)
        off = _make_layer(gumbel_noise=False)
        # 同一seedで構築しているため weights/indices は一致
        assert torch.equal(on.weights, off.weights)
        x = _make_x()
        on.eval()
        off.eval()
        with torch.no_grad():
            assert torch.equal(on(x), off(x)), 'eval: gumbel有無で出力が変わっている'


class TestTrainStochasticAndDifferentiable:
    """契約3: True の train forward は確率的に変化し、backward が有限勾配。"""

    def test_train_forward_is_stochastic(self):
        layer = _make_layer(gumbel_noise=True)
        layer.train()
        x = _make_x()
        y1 = layer(x)
        y2 = layer(x)
        assert not torch.allclose(y1, y2), 'Gumbelノイズありでも出力が変化しない'

    def test_backward_gives_finite_grad_to_all_params(self):
        layer = _make_layer(gumbel_noise=True)
        layer.train()
        x = _make_x()
        loss = layer(x).sum()
        loss.backward()
        params = list(layer.parameters())
        assert params, 'パラメータが無い'
        for p in params:
            assert p.grad is not None, 'gradがNone'
            assert torch.isfinite(p.grad).all(), 'gradに非有限値'
        # weights勾配は非ゼロ(学習が進む)
        assert layer.weights.grad.abs().sum() > 0, 'weights勾配が全ゼロ'


class TestReproducibility:
    """契約4: torch.manual_seed 固定で train forward が再現。"""

    def test_seed_fixes_forward(self):
        layer = _make_layer(gumbel_noise=True)
        layer.train()
        x = _make_x()
        torch.manual_seed(2024)
        y1 = layer(x)
        torch.manual_seed(2024)
        y2 = layer(x)
        assert torch.equal(y1, y2), 'seed固定でも forward が再現しない'


# ============================================================
# 契約5: 収束スモーク(XOR/parity で hard acc=100%)
# ============================================================

def _bits_dataset(n_bits, label_fn):
    xs = torch.tensor(list(itertools.product([0.0, 1.0], repeat=n_bits)))
    ys = torch.tensor([label_fn(row) for row in xs.long().tolist()])
    return xs, ys


def _gumbel_mlp(in_dim, k_classes, widths, *, gumbel_tau=1.0):
    torch.manual_seed(0)
    layers = []
    d = in_dim
    for w in widths:
        layers.append(LutLayer(
            d, w, n_inputs=2, parameterization='softmax',
            gumbel_noise=True, gumbel_tau=gumbel_tau,
        ))
        d = w
    layers.append(GroupSum(k=k_classes, tau=max(1.0, (d // k_classes) ** 0.5)))
    return nn.Sequential(*layers)


def _train_until(model, x, y, *, lr=0.05, max_steps=2000, check_every=50):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if (step + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                return 1.0
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


class TestGumbelLearning:
    def test_xor_hard_acc_100(self, seed):
        """XOR: Gumbelノイズあり学習で hard(離散化) acc=100%。"""
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _gumbel_mlp(2, 2, [16, 16])
        acc = _train_until(model, x, y, max_steps=2000)
        assert acc == 1.0, f'gumbel XOR: hard acc={acc}'

    def test_parity3_hard_acc_100(self, seed):
        """3bit parity: Gumbelノイズあり学習で hard acc=100%。"""
        x, y = _bits_dataset(3, lambda b: (b[0] + b[1] + b[2]) % 2)
        model = _gumbel_mlp(3, 2, [32, 32])
        acc = _train_until(model, x, y, max_steps=2500)
        assert acc == 1.0, f'gumbel parity3: hard acc={acc}'


class TestUnsupportedParameterizations:
    """warp/hard_st + gumbel_noise は NotImplementedError(発明しない)。"""

    def test_warp_raises(self):
        with pytest.raises(NotImplementedError, match='softmax'):
            LutLayer(_IN_DIM, _OUT_DIM, n_inputs=2,
                     parameterization='warp', gumbel_noise=True)

    def test_hard_st_raises(self):
        with pytest.raises(NotImplementedError, match='soft'):
            LutLayer(_IN_DIM, _OUT_DIM, n_inputs=2,
                     parameterization='softmax', selection='hard_st',
                     gumbel_noise=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
