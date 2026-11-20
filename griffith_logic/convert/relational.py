"""RKD(Relational Knowledge Distillation, Park et al. CVPR 2019)損失。

層別BCEヒント蒸留(生の中間二値活性を直接模倣)は崩壊した。密なMLP教師と
hard離散DLGNは活性の絶対値スケールも次元も違うため、絶対活性を合わせる
FitNets系ヒントは原理的に不利。

RKDは個々の活性値でなく**サンプル間の関係**(ペア距離・三つ組の角度)だけを
転移する。関係量は表現の次元に依存せず、正のアフィン/スケール変換にも
頑健なので、教師(密MLP)と生徒(疎DLGN)の表現次元・スケールが全く違っても
そのまま比較できる。これがFitNets系と違い次元不一致に耐えられる理由であり、
本プロジェクトでは`convert/recover.py`の微調整(soft磨き)へ補助損失として
注入する(RECOVER_PARAMETERIZATIONS/selectionの制約とは独立)。

数式(Park et al. 2019, arXiv:1904.05068 Eq.3-4):
    distance-wise: ψ_D(x_i, x_j) = ||x_i - x_j||_2 / μ
        μ = ペア距離のバッチ平均(正規化: 平均距離を1に揃える)
        L_RKD-D = Huber( ψ_D(student) , ψ_D(teacher) )   (Smooth L1)
    angle-wise: ψ_A(x_i, x_j, x_k) = cos∠(x_i - x_j, x_k - x_j)
        L_RKD-A = Huber( ψ_A(student) , ψ_A(teacher) )

Huber(Smooth L1)を使うのは原論文と同じ(外れ値に頑健、L2より安定)。

次元非依存性(dimension-agnostic)は本モジュールの核心的性質であり、
`tests/l1_math/test_relational.py` の TestRkdDimensionAgnostic で検証する。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ['rkd_angle_loss', 'rkd_distance_loss', 'rkd_loss']

_EPS = 1e-12


def _pairwise_l2(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """(N, D) -> (N, N) のペアL2距離行列(対角0、数値安定のためeps付き平方根)。

    torch.cdist相当だが、勾配のNaN対策(距離0近傍での平方根微分発散)として
    ||a-b||^2 に eps を加えてから sqrt する素朴な実装にする
    (docstring通り明瞭さ優先、性能最適化はしない)。
    """
    diff = x.unsqueeze(1) - x.unsqueeze(0)          # (N, N, D)
    sq = (diff * diff).sum(dim=-1)                   # (N, N)
    return torch.sqrt(sq.clamp_min(0.0) + eps)


def rkd_distance_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """RKD distance-wise損失(Park+ 2019 Eq.3)。

    student/teacherは同一バッチのサンプル順で対応していれば、表現次元
    (最終軸)が異なっていてもよい(次元非依存)。teacher側は勾配不要なので
    呼び出し側で`.detach()`しておくことを推奨する(本関数は強制しない)。

    Args:
        student: [N, D_s] 生徒表現(バッチ次元Nが同一サンプル順であること)。
        teacher: [N, D_t] 教師表現。D_t は D_s と異なってよい。

    Returns:
        スカラー損失。N<=1(ペアが存在しない)なら0を返す。
    """
    n = student.shape[0]
    if n != teacher.shape[0]:
        raise ValueError(
            f'student/teacherのバッチサイズが不一致: {n} vs {teacher.shape[0]}'
        )
    if n <= 1:
        return student.new_zeros(())

    d_s = _pairwise_l2(student)
    d_t = _pairwise_l2(teacher.detach() if teacher.requires_grad else teacher)

    # 正規化: 非対角ペアの平均距離で割り、絶対スケールを揃える
    # (distance-wise potentialの定義通り、mu=バッチ内平均ペア距離)
    off_diag = ~torch.eye(n, dtype=torch.bool, device=student.device)
    mu_s = d_s[off_diag].mean().clamp_min(_EPS)
    mu_t = d_t[off_diag].mean().clamp_min(_EPS)

    psi_s = d_s / mu_s
    psi_t = d_t / mu_t
    return F.smooth_l1_loss(psi_s, psi_t)


def rkd_angle_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """RKD angle-wise損失(Park+ 2019 Eq.4)。

    3点の組(i, j, k)ごとに頂点jでの角度 cos∠(x_i - x_j, x_k - x_j) を計算し、
    student側とteacher側で角度分布をHuber損失で揃える。距離と異なり角度は
    そもそも単位を持たないため正規化不要(スケール不変性は自動的に成立)。

    Args:
        student: [N, D_s] 生徒表現。
        teacher: [N, D_t] 教師表現(D_t != D_s可)。

    Returns:
        スカラー損失。N<=2(三つ組が組めない)なら0を返す。
    """
    n = student.shape[0]
    if n != teacher.shape[0]:
        raise ValueError(
            f'student/teacherのバッチサイズが不一致: {n} vs {teacher.shape[0]}'
        )
    if n <= 2:
        return student.new_zeros(())

    teacher = teacher.detach() if teacher.requires_grad else teacher

    def _angles(x: torch.Tensor) -> torch.Tensor:
        # e[i, j] = x_i - x_j を単位ベクトル化 (N, N, D)
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        norm = diff.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        e = diff / norm
        # cos∠(e_ij, e_kj) を全 (j, i, k) について一括計算:
        # e[j] は行jが頂点jからの単位ベクトル群(shape (N, D)、e[j, i] = e_ij)
        # cos[j, i, k] = e[i, j] . e[k, j] であり、頂点jを固定してi,kを走らせる
        # e_perm[j, i, :] = e[i, j, :] (転置: j 軸を先頭へ)
        e_perm = e.transpose(0, 1)  # (j, i, D): e_perm[j, i] = e[i, j] = x_i - x_j (単位化済み)
        cos = torch.einsum('jid,jkd->jik', e_perm, e_perm)  # (j, i, k)
        return cos

    cos_s = _angles(student)
    cos_t = _angles(teacher)

    # i==j, k==j, i==k は退化(ゼロ除算をepsで回避しているが意味を持たない)
    # ので、i!=j かつ k!=j かつ i!=k のみを対象にする。
    idx = torch.arange(n, device=student.device)
    valid = (
        (idx.view(1, -1, 1) != idx.view(-1, 1, 1))
        & (idx.view(1, 1, -1) != idx.view(-1, 1, 1))
        & (idx.view(1, -1, 1) != idx.view(1, 1, -1))
    )
    if not bool(valid.any()):
        return student.new_zeros(())

    return F.smooth_l1_loss(cos_s[valid], cos_t[valid])


def rkd_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    distance_weight: float = 1.0,
    angle_weight: float = 2.0,
) -> torch.Tensor:
    """distance-wise + angle-wise RKD損失の重み付き和(Park+ 2019 既定比率)。

    原論文は angle-wise の重みを distance-wise の2倍に置く運用が多い
    (Table中の既定λ_D=1, λ_A=2相当)ため、既定値もそれに合わせる。

    Args:
        student: [N, D_s] 生徒表現。
        teacher: [N, D_t] 教師表現(D_t != D_s可、次元非依存)。
        distance_weight: distance-wise損失の重み(既定1.0)。
        angle_weight: angle-wise損失の重み(既定2.0)。

    Returns:
        スカラー損失 = distance_weight * rkd_distance_loss + angle_weight * rkd_angle_loss。
    """
    d = rkd_distance_loss(student, teacher)
    a = rkd_angle_loss(student, teacher)
    return distance_weight * d + angle_weight * a
