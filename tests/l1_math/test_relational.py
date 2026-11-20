"""RKD(Relational Knowledge Distillation, Park+ CVPR2019)損失の数理性質テスト。

convert/relational.py の rkd_distance_loss / rkd_angle_loss / rkd_loss を検証する。
RKDはサンプル間の**関係**(距離・角度)だけを転移するため、教師・生徒の表現次元が
異なっても比較できる(次元非依存)のが核心的性質。本テストでは:

    (a) RKD(x, x) ≈ 0        — 自己比較で損失ゼロ(退化ケース)
    (b) スケール不変          — 表現を定数倍しても損失不変(距離・角度とも
                                 正規化/角度の定義上、正のスケールに不変)
    (c) 手計算値との一致       — 3点程度の小規模ケースで距離・角度を素朴計算し照合
    (d) 次元非依存             — student/teacher表現の次元が異なっても動作する

許容誤差は tests/conftest.py の共通定数を使う(数値リテラル直書き禁止)。
"""

import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from conftest import TOL_F32_EQ, TOL_F64_EQ

from griffith_logic.convert.relational import (
    rkd_angle_loss,
    rkd_distance_loss,
    rkd_loss,
)


def _naive_pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    """(N, D) -> (N, N) の素朴二重ループL2距離(参照実装)。"""
    n = x.shape[0]
    d = torch.zeros(n, n, dtype=x.dtype)
    for i in range(n):
        for j in range(n):
            d[i, j] = torch.sqrt(((x[i] - x[j]) ** 2).sum() + 1e-12)
    return d


def _naive_angle(x: torch.Tensor, i: int, j: int, k: int) -> float:
    """三つ組(i,j,k)の角度 cos(e_ij, e_kj)(RKD論文Eq.4、頂点jの角度)を素朴計算。"""
    e_ij = x[i] - x[j]
    e_kj = x[k] - x[j]
    e_ij = e_ij / (e_ij.norm() + 1e-12)
    e_kj = e_kj / (e_kj.norm() + 1e-12)
    return float((e_ij * e_kj).sum())


# ============================================================
# (a) RKD(x, x) ≈ 0
# ============================================================

class TestRkdSelfComparisonZero:
    def test_distance_loss_self_zero(self):
        torch.manual_seed(0)
        x = torch.randn(8, 5, dtype=torch.float64)
        loss = rkd_distance_loss(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=TOL_F64_EQ, rtol=0)

    def test_angle_loss_self_zero(self):
        torch.manual_seed(0)
        x = torch.randn(8, 5, dtype=torch.float64)
        loss = rkd_angle_loss(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=TOL_F64_EQ, rtol=0)

    def test_rkd_loss_self_zero(self):
        torch.manual_seed(0)
        x = torch.randn(10, 4, dtype=torch.float64)
        loss = rkd_loss(x, x)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=TOL_F64_EQ, rtol=0)


# ============================================================
# (b) 正のスケールに不変
# ============================================================

class TestRkdScaleInvariance:
    @pytest.mark.parametrize('scale', [0.1, 2.0, 37.5])
    def test_distance_loss_scale_invariant(self, scale):
        torch.manual_seed(1)
        student = torch.randn(8, 6, dtype=torch.float64)
        teacher = torch.randn(8, 6, dtype=torch.float64)
        base = rkd_distance_loss(student, teacher)
        scaled = rkd_distance_loss(student * scale, teacher * scale)
        assert torch.allclose(base, scaled, atol=TOL_F64_EQ, rtol=1e-9)

    @pytest.mark.parametrize('scale', [0.1, 2.0, 37.5])
    def test_angle_loss_scale_invariant(self, scale):
        torch.manual_seed(1)
        student = torch.randn(8, 6, dtype=torch.float64)
        teacher = torch.randn(8, 6, dtype=torch.float64)
        base = rkd_angle_loss(student, teacher)
        scaled = rkd_angle_loss(student * scale, teacher * scale)
        assert torch.allclose(base, scaled, atol=TOL_F64_EQ, rtol=1e-9)

    def test_distance_loss_independent_scale_of_student_and_teacher(self):
        """studentとteacherを別々の正スケールで拡大しても損失不変
        (各側で自己正規化するため、双方が同じ形なら一致し続ける)。
        """
        torch.manual_seed(2)
        student = torch.randn(8, 6, dtype=torch.float64)
        base = rkd_distance_loss(student, student)
        scaled = rkd_distance_loss(student * 3.0, student * 9.0)
        assert torch.allclose(base, scaled, atol=TOL_F64_EQ, rtol=1e-9)
        assert torch.allclose(base, torch.zeros_like(base), atol=TOL_F64_EQ, rtol=0)


# ============================================================
# (c) 手計算値との一致(小規模、素朴二重/三重ループ)
# ============================================================

class TestRkdHandChecked:
    def test_distance_loss_matches_naive_pairwise(self):
        torch.manual_seed(3)
        student = torch.randn(6, 4, dtype=torch.float64)
        teacher = torch.randn(6, 3, dtype=torch.float64)  # 次元が異なっていてもよい

        d_s = _naive_pairwise_dist(student)
        d_t = _naive_pairwise_dist(teacher)
        # 対角は sqrt(eps) 由来の微小非ゼロ値を持つため(rkd_distance_lossの
        # _pairwise_l2実装と同じ規約)、d>0ではなく明示的な非対角マスクで
        # 平均を取る(実装(off_diagマスク)と同じ集合に揃える)。
        off_diag = ~torch.eye(student.shape[0], dtype=torch.bool)
        mu_s = d_s[off_diag].mean()
        mu_t = d_t[off_diag].mean()
        expected = torch.nn.functional.smooth_l1_loss(d_s / mu_s, d_t / mu_t)

        actual = rkd_distance_loss(student, teacher)
        assert torch.allclose(actual, expected, atol=TOL_F64_EQ, rtol=1e-9)

    def test_angle_loss_matches_naive_triplets(self):
        torch.manual_seed(4)
        n = 5
        student = torch.randn(n, 4, dtype=torch.float64)
        teacher = torch.randn(n, 3, dtype=torch.float64)

        triplets = list(itertools.permutations(range(n), 3))
        angles_s = torch.tensor(
            [_naive_angle(student, i, j, k) for i, j, k in triplets],
            dtype=torch.float64,
        )
        angles_t = torch.tensor(
            [_naive_angle(teacher, i, j, k) for i, j, k in triplets],
            dtype=torch.float64,
        )
        expected = torch.nn.functional.smooth_l1_loss(angles_s, angles_t)

        actual = rkd_angle_loss(student, teacher)
        assert torch.allclose(actual, expected, atol=TOL_F64_EQ, rtol=1e-9)

    def test_hand_checked_tiny_case(self):
        """3点の正三角形的配置で角度損失が理論値(全角度cos=-0.5)近傍になることを
        手計算で確認する(退化しない具体的な非ゼロケース)。
        """
        # 教師: 正三角形(全内角60度 → 隣接ベクトルのcos=-0.5)
        teacher = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254]], dtype=torch.float64,
        )
        # 生徒: 直角二等辺三角形(異なる角度分布)
        student = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64,
        )
        loss = rkd_angle_loss(student, teacher)
        assert loss.item() > 0.0  # 形状が違うので損失は非ゼロ
        # 教師どうしなら0
        loss_self = rkd_angle_loss(teacher, teacher)
        assert torch.allclose(loss_self, torch.zeros_like(loss_self),
                              atol=TOL_F64_EQ, rtol=0)


# ============================================================
# (d) 次元非依存(student/teacher表現次元が異なっても動く)
# ============================================================

class TestRkdDimensionAgnostic:
    def test_different_dims_no_error(self):
        torch.manual_seed(5)
        student = torch.randn(12, 7, dtype=torch.float32)
        teacher = torch.randn(12, 31, dtype=torch.float32)  # クラス数31=n=4算術タスク相当
        loss_d = rkd_distance_loss(student, teacher)
        loss_a = rkd_angle_loss(student, teacher)
        loss_total = rkd_loss(student, teacher)
        assert loss_d.dim() == 0
        assert loss_a.dim() == 0
        assert loss_total.dim() == 0
        assert torch.isfinite(loss_total)

    def test_rkd_loss_is_weighted_sum(self):
        torch.manual_seed(6)
        student = torch.randn(10, 5, dtype=torch.float64)
        teacher = torch.randn(10, 8, dtype=torch.float64)
        dist_w, angle_w = 1.0, 2.0
        total = rkd_loss(student, teacher, distance_weight=dist_w, angle_weight=angle_w)
        expected = (dist_w * rkd_distance_loss(student, teacher)
                    + angle_w * rkd_angle_loss(student, teacher))
        assert torch.allclose(total, expected, atol=TOL_F64_EQ, rtol=0)


# ============================================================
# 単一サンプル/縮退ケースの防御
# ============================================================

class TestRkdEdgeCases:
    def test_single_sample_batch_no_crash(self):
        x = torch.randn(1, 4, dtype=torch.float32)
        y = torch.randn(1, 4, dtype=torch.float32)
        # 距離損失: ペアが存在しない(N=1)ので0扱いになるべき
        loss = rkd_distance_loss(x, y)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=TOL_F32_EQ, rtol=0)

    def test_gradients_flow_to_student(self):
        """studentの表現に勾配が流れる(recoverの補助損失として使うための最低条件)。"""
        torch.manual_seed(7)
        student = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(8, 5, dtype=torch.float64)
        loss = rkd_loss(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()
