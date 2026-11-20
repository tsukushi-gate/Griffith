"""L2: AP法(Akiho Propagation)更新式の解析照合。

APTrainer.train_step(training/rules/ap.py)の重み更新量Δwを、
AP法の定義式をnumpy(float64)で独立に再実装した参照と1e-10で照合する。

AP法の定義:
    forward:  sigmoidなし。weights.clamp(0,1)を確率値として直接使い、
              y = clamp(Σ_e basis_e × w_e, 0, 1)
    誤差:     e = one_hot(target) - softmax(logits) をGroupSumの逆
              (repeat_interleave / τ)でニューロン誤差化
    伝播:     出力ニューロンjの誤差を、接続グラフで到達可能な全LUTへ
              同一値でscatter(到達集合、重複スロットは1回のみ)
    更新:     mode1: Δw = mean_batch(error × basis) × lr → clamp[0,1]
              mode2: 最大basis入力に対応する重みのみ更新
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import GroupSum, LutLayer
from griffith_logic.training.rules.ap import APTrainer

TOL_AP = 1e-10


@pytest.fixture
def float64_default():
    """モデル・トレーナ内部の一時テンソルまで含めてfloat64で実行する"""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ============================================================
# 純numpy参照実装(定義式の独立再実装)
# ============================================================

def _np_basis(gathered: np.ndarray, n_inputs: int) -> np.ndarray:
    """確率的基底(素朴な積公式): basis[...,k] = Π_j (x_j if bit(k,j) else 1-x_j)"""
    batch, out_dim, _ = gathered.shape
    basis = np.zeros((batch, out_dim, 2 ** n_inputs))
    for k in range(2 ** n_inputs):
        t = np.ones((batch, out_dim))
        for j in range(n_inputs):
            xj = gathered[:, :, j]
            t = t * (xj if (k >> j) & 1 else 1.0 - xj)
        basis[:, :, k] = t
    return basis


def _np_reach_sets(indices_list: list[np.ndarray]) -> list[dict[int, set[int]]]:
    """各層のニューロン → 到達元出力ニューロン集合(最終層から逆順に構築)"""
    last_out_dim = indices_list[-1].shape[1]
    reach = {j: {j} for j in range(last_out_dim)}
    per_layer = [reach]
    for indices in reversed(indices_list[1:]):
        n_inputs, out_dim = indices.shape
        prev: dict[int, set[int]] = {}
        for j in range(out_dim):
            if j not in reach:
                continue
            for slot in range(n_inputs):
                prev.setdefault(int(indices[slot, j]), set()).update(reach[j])
        per_layer.insert(0, prev)
        reach = prev
    return per_layer


def _reference_ap_step(
    x: np.ndarray,
    weights_list: list[np.ndarray],
    indices_list: list[np.ndarray],
    k: int,
    tau: float,
    target: np.ndarray,
    lr: float,
    update_mode: int,
) -> list[np.ndarray]:
    """AP法1ステップ後の各層重みを定義式から計算して返す"""
    batch = x.shape[0]

    # (1) sigmoidなしforward(clamp(0,1)した重みの基底加重和)
    h = x
    basis_list = []
    for w, indices in zip(weights_list, indices_list):
        n_inputs, out_dim = indices.shape
        gathered = h[:, indices.T.reshape(-1)].reshape(batch, out_dim, n_inputs)
        basis = _np_basis(gathered, n_inputs)
        out = np.clip(
            np.einsum('boe,oe->bo', basis, np.clip(w, 0.0, 1.0)), 0.0, 1.0,
        )
        basis_list.append(basis)
        h = out

    # (2) GroupSum → 出力誤差 e = one_hot - softmax(logits) → ニューロン誤差化
    last_out_dim = h.shape[1]
    n_per = last_out_dim // k
    logits = h.reshape(batch, k, n_per).sum(-1) / tau
    z = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=-1, keepdims=True)
    class_error = np.eye(k)[target] - probs
    neuron_error = np.repeat(class_error, n_per, axis=-1) / tau

    # (3) 到達マップで各層へ誤差をscatter
    reach_per_layer = _np_reach_sets(indices_list)

    # (4) Δw = mean_batch(error × basis) × lr → clamp[0,1]
    new_weights = []
    for w, indices, basis, reach in zip(
        weights_list, indices_list, basis_list, reach_per_layer,
    ):
        out_dim = indices.shape[1]
        error_l = np.zeros((batch, out_dim))
        for i, outputs in reach.items():
            for j in outputs:
                error_l[:, i] += neuron_error[:, j]

        if update_mode == 1:
            delta = np.einsum('bo,boe->oe', error_l, basis) / batch
        else:
            max_idx = basis.argmax(axis=-1)  # (batch, out_dim)
            max_basis = np.take_along_axis(
                basis, max_idx[..., None], axis=-1,
            ).squeeze(-1)
            delta_scalar = error_l * max_basis
            delta = np.zeros_like(w)
            for b in range(batch):
                for o in range(out_dim):
                    delta[o, max_idx[b, o]] += delta_scalar[b, o]
            delta /= batch

        new_weights.append(np.clip(w + delta * lr, 0.0, 1.0))
    return new_weights


def _run_and_compare(model, lr, update_mode, batch=8, seed=101):
    """APTrainerを1ステップ実行し、numpy参照のΔwと照合する"""
    trainer = APTrainer(model, lr=lr, update_mode=update_mode)

    weights_before = [
        layer.weights.data.clone().numpy().copy() for layer in trainer.lut_layers
    ]
    indices_list = [layer.indices.numpy().copy() for layer in trainer.lut_layers]
    in_dim = trainer.lut_layers[0].in_dim

    torch.manual_seed(seed)
    x = torch.rand(batch, in_dim)
    target = torch.randint(0, trainer.group_sum.k, (batch,))

    trainer.train_step(x, target)

    weights_ref = _reference_ap_step(
        x.numpy(), weights_before, indices_list,
        trainer.group_sum.k, trainer.group_sum.tau,
        target.numpy(), lr, update_mode,
    )
    for layer, w_ref in zip(trainer.lut_layers, weights_ref):
        w_after = layer.weights.data.numpy()
        diff = np.abs(w_after - w_ref).max()
        assert np.allclose(w_after, w_ref, atol=TOL_AP, rtol=0), (
            f'Δw不一致(update_mode={update_mode}): max diff={diff:.3e}'
        )


class TestAPUpdateAnalytic:
    @pytest.mark.parametrize('update_mode', [1, 2])
    def test_single_layer_update_matches_definition(
        self, float64_default, update_mode,
    ):
        """Flatten+LutLayer(4,4,independent)+GroupSum(k=2)の1ステップ照合"""
        torch.manual_seed(31)
        model = nn.Sequential(
            nn.Flatten(),
            LutLayer(4, 4, n_inputs=2, parameterization='independent'),
            GroupSum(k=2, tau=2.0),
        )
        _run_and_compare(model, lr=0.3, update_mode=update_mode)

    @pytest.mark.parametrize('update_mode', [1, 2])
    def test_two_layer_update_matches_definition(
        self, float64_default, update_mode,
    ):
        """2層(到達マップscatter経路込み)の1ステップ照合"""
        torch.manual_seed(37)
        model = nn.Sequential(
            nn.Flatten(),
            LutLayer(4, 6, n_inputs=2, parameterization='independent'),
            LutLayer(6, 4, n_inputs=2, parameterization='independent'),
            GroupSum(k=2, tau=1.5),
        )
        _run_and_compare(model, lr=0.2, update_mode=update_mode, seed=103)


class TestReachMaps:
    def test_two_layer_hand_computed(self, float64_default):
        """到達マップの手計算照合(2層、各2ニューロン)。

        層1(最終層)の接続:
            b0 ← (a0, a0)  同一入力の重複スロットは到達集合では1回
            b1 ← (a0, a1)
        期待される到達(出力j → 層0ニューロンi):
            a0 ← {b0, b1},  a1 ← {b1}
        """
        torch.manual_seed(41)
        layer0 = LutLayer(2, 2, n_inputs=2, parameterization='independent')
        layer1 = LutLayer(2, 2, n_inputs=2, parameterization='independent')
        # indices: (n_inputs, out_dim)。列jがニューロンjの入力スロット
        layer1.indices.copy_(torch.tensor([[0, 0], [0, 1]]))
        model = nn.Sequential(
            nn.Flatten(), layer0, layer1, GroupSum(k=2, tau=1.0),
        )
        trainer = APTrainer(model, lr=0.1)

        assert len(trainer._reach_per_layer) == 2

        # 最終層: 恒等マップ
        src_last, dst_last = trainer._reach_per_layer[1]
        assert set(zip(src_last.tolist(), dst_last.tolist())) == {(0, 0), (1, 1)}

        # 層0: (出力j → 到達ニューロンi) のペア集合
        src0, dst0 = trainer._reach_per_layer[0]
        assert set(zip(src0.tolist(), dst0.tolist())) == {(0, 0), (1, 0), (1, 1)}

    def test_error_scatter_hand_computed(self, float64_default):
        """到達マップ経由の誤差集約: a0にe0+e1、a1にe1が届く"""
        torch.manual_seed(43)
        layer0 = LutLayer(2, 2, n_inputs=2, parameterization='independent')
        layer1 = LutLayer(2, 2, n_inputs=2, parameterization='independent')
        layer1.indices.copy_(torch.tensor([[0, 0], [0, 1]]))
        model = nn.Sequential(
            nn.Flatten(), layer0, layer1, GroupSum(k=2, tau=1.0),
        )
        trainer = APTrainer(model, lr=0.1)

        src0, dst0 = trainer._reach_per_layer[0]
        neuron_error = torch.tensor([[10.0, 1.0]])  # e0=10, e1=1
        error_l0 = torch.zeros(1, 2)
        error_l0.scatter_add_(
            1, dst0.unsqueeze(0), neuron_error[:, src0],
        )
        assert torch.equal(error_l0, torch.tensor([[11.0, 1.0]]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
