"""L2: LutConv2d parameterization='softmax16g' の勾配契約テスト。

softmax16gのsoft tree forward(_tree_forward_train)は連続微分可能
(softmax混合 → gate_table線形写像 → butterfly基底加重和)のため、
float64での数値gradcheckを直接適用できる。

softmax16gの意味論:
    soft:  tt = softmax(w) @ gate_table(16→4)、出力 = Σ_e basis_e · tt_e
    hard:  tt = gate_table[argmax(w)]  → 真の離散Booleanゲート(全16種)

gradcheckは、モジュールから取り出した gate_table を使い、softmax16gの木演算を
純関数として書き下したものに対して行う(nn.Parameterへの再代入はautograd graphを
切るため使わない)。この純関数が実装(_tree_forward_train)と一致することは
soft/hard整合テスト(tests/test_conv.py)とend-to-end学習で担保される。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conftest import GRADCHECK_KW  # noqa: E402

from griffith_logic.conv import LutConv2d, _compute_basis  # noqa: E402
from griffith_logic.core.basis import _build_gate_table_n2  # noqa: E402


def _soft_tree_forward(leaves, weights, gate_table, tree_depth, n_inputs=2):
    """softmax16gのsoft木演算を純関数として書き下す(実装_tree_forward_trainと同式)。

    leaves:  (BL, out_ch, n_leaves)
    weights: (out_ch, n_internal, 16)
    """
    current = leaves
    gate_offset = 0
    for _level in range(tree_depth):
        n_groups = current.shape[-1] // n_inputs
        grouped = current[..., :n_groups * n_inputs].reshape(
            *current.shape[:-1], n_groups, n_inputs
        )
        level_weights = weights[:, gate_offset:gate_offset + n_groups, :]
        basis = _compute_basis(grouped, n_inputs)
        s = torch.softmax(level_weights, dim=-1)
        w = s @ gate_table.to(s.dtype)
        current = (basis * w.unsqueeze(0)).sum(dim=-1)
        gate_offset += n_groups
    return current.squeeze(-1)


class TestSoftmax16gGradcheck:
    def _case(self, tree_depth, boundary=False, seed=7):
        torch.manual_seed(seed)
        n_leaves = 2 ** tree_depth
        n_internal = (n_leaves - 1) // (2 - 1)
        gt = _build_gate_table_n2().double()
        weights = torch.randn(1, n_internal, 16, dtype=torch.float64)
        if boundary:
            leaves = torch.randint(0, 2, (5, 1, n_leaves)).double()
        else:
            leaves = torch.rand(5, 1, n_leaves, dtype=torch.float64) * 0.8 + 0.1
        return leaves, weights, gt

    def test_gradcheck_depth2_interior(self):
        """depth=2、内部点 leaves∈(0.1,0.9) でのfloat64 gradcheck"""
        leaves, w, gt = self._case(2)
        leaves = leaves.requires_grad_(True)
        w = w.requires_grad_(True)

        def fn(le, wt):
            return _soft_tree_forward(le, wt, gt, tree_depth=2)

        assert torch.autograd.gradcheck(fn, (leaves, w), **GRADCHECK_KW)

    def test_gradcheck_depth2_boundary(self):
        """境界点 leaves∈{0,1} を含むケースでのgradcheck"""
        leaves, w, gt = self._case(2, boundary=True, seed=3)
        leaves = leaves.requires_grad_(True)
        w = w.requires_grad_(True)

        def fn(le, wt):
            return _soft_tree_forward(le, wt, gt, tree_depth=2)

        assert torch.autograd.gradcheck(fn, (leaves, w), **GRADCHECK_KW)

    def test_pure_fn_matches_implementation(self):
        """純関数が実装_tree_forward_trainと一致(参照としての妥当性を担保)"""
        conv = LutConv2d(
            in_channels=1, out_channels=2, kernel_size=3, tree_depth=2,
            padding=0, n_inputs=2, parameterization='softmax16g',
            residual_init=False,
        ).double()
        conv.train()
        n_leaves = 2 ** 2
        leaves = torch.rand(5, 2, n_leaves, dtype=torch.float64) * 0.8 + 0.1
        y_impl = conv._tree_forward_train(leaves)
        y_ref = _soft_tree_forward(
            leaves, conv.weights.detach(), conv.gate_table.double(), tree_depth=2
        )
        assert torch.allclose(y_impl, y_ref, atol=1e-12, rtol=0)
