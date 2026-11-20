"""C4 系列KD損失(convert/distill.py の kd_kl_loss_seq / make_seq_kd_loss)の数理性質テスト。

kd_kl_loss_seq は kd_kl_loss([B,C])のシーケンス版([B,S,V]、時刻方向を独立
サンプルとして扱いつつmaskで平均対象を絞る)。二重温度設計・T^2補正の根拠は
tests/l1_math/test_kd_loss.py / convert/distill.py モジュールdocstring参照。

許容誤差は tests/conftest.py の定数(TOL_F64_EQ)を参照。すべて float64。
"""

import pytest
import torch
import torch.nn.functional as F

from griffith_logic.convert.distill import (
    kd_kl_loss,
    kd_kl_loss_seq,
    make_seq_kd_loss,
)
from tests.conftest import TOL_F64_EQ

_GEN = torch.Generator().manual_seed(5678)


def _rand(*shape: int) -> torch.Tensor:
    # 単一generatorを共有して呼ぶたびに前進させる(test_kd_loss.pyと同じ流儀)。
    return torch.randn(*shape, generator=_GEN, dtype=torch.float64)


# ----------------------------------------------------------------------
# 1. self-KL(student==teacher)は0(mask無し/mask全T両方)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('temperature', [1.0, 2.0, 4.0])
def test_self_kl_is_zero_no_mask(temperature):
    s = _rand(3, 5, 6)
    loss = kd_kl_loss_seq(s, s.clone(), temperature)
    assert abs(loss.item()) < TOL_F64_EQ


@pytest.mark.parametrize('temperature', [1.0, 2.0, 4.0])
def test_self_kl_is_zero_all_true_mask(temperature):
    s = _rand(3, 5, 6)
    mask = torch.ones(3, 5, dtype=torch.bool)
    loss = kd_kl_loss_seq(s, s.clone(), temperature, mask=mask)
    assert abs(loss.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 2. 非負性(乱数ロジット)
# ----------------------------------------------------------------------
def test_nonnegativity():
    s = _rand(4, 6, 5)
    t = _rand(4, 6, 5)
    loss = kd_kl_loss_seq(s, t, temperature=2.0)
    assert loss.item() >= -TOL_F64_EQ


# ----------------------------------------------------------------------
# 3. mask=None / 全Trueは既存kd_kl_loss(flatten [B*S, V])と一致
# ----------------------------------------------------------------------
@pytest.mark.parametrize('temperature', [1.0, 3.0])
def test_mask_none_matches_flattened_kd_kl_loss(temperature):
    b, seq, v = 4, 5, 7
    s = _rand(b, seq, v)
    t = _rand(b, seq, v)

    got_none = kd_kl_loss_seq(s, t, temperature)
    mask_all = torch.ones(b, seq, dtype=torch.bool)
    got_mask = kd_kl_loss_seq(s, t, temperature, mask=mask_all)

    expected = kd_kl_loss(s.reshape(b * seq, v), t.reshape(b * seq, v), temperature)

    assert abs(got_none.item() - expected.item()) < TOL_F64_EQ
    assert abs(got_mask.item() - expected.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 4. マスク位置のロジットを変えても損失は不変(bit-exact)
# ----------------------------------------------------------------------
def test_masked_positions_do_not_affect_loss():
    b, seq, v = 2, 6, 4
    s = _rand(b, seq, v)
    t = _rand(b, seq, v)
    mask = torch.zeros(b, seq, dtype=torch.bool)
    mask[:, :3] = True  # 前半のみ有効

    base = kd_kl_loss_seq(s, t, temperature=2.0, mask=mask)

    # マスク対象外(後半)のstudent/teacherロジットを別乱数に差し替える
    s_perturbed = s.clone()
    t_perturbed = t.clone()
    s_perturbed[:, 3:] = _rand(b, seq - 3, v)
    t_perturbed[:, 3:] = _rand(b, seq - 3, v)

    perturbed = kd_kl_loss_seq(s_perturbed, t_perturbed, temperature=2.0, mask=mask)

    assert base.item() == perturbed.item()  # bit-exact


# ----------------------------------------------------------------------
# 5. 全マスクFalseはValueError
# ----------------------------------------------------------------------
def test_all_masked_raises_value_error():
    s = _rand(2, 3, 4)
    t = _rand(2, 3, 4)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        kd_kl_loss_seq(s, t, temperature=1.0, mask=mask)


# ----------------------------------------------------------------------
# 6. T^2スケール関係(高温極限勾配のcos類似度がtest_kd_loss.pyと同じ形の検証)
# ----------------------------------------------------------------------
def test_high_temperature_gradient_limit_consistent_with_flat_version():
    """T→∞で kd_kl_loss_seq の勾配方向が中心化ロジット差の方向へ収束する。

    test_kd_loss.py の test_kl_high_temperature_gradient_limit と同じ論理を
    [B,S,V]版で確認する(T^2補正の妥当性検証)。
    """
    s0 = _rand(3, 4, 5)
    t = _rand(3, 4, 5)

    limit = (s0 - s0.mean(dim=-1, keepdim=True)) - (t - t.mean(dim=-1, keepdim=True))
    limit_flat = limit.reshape(-1)

    cosines = []
    for temp in (10.0, 100.0, 1000.0):
        s = s0.clone().requires_grad_(True)
        loss = kd_kl_loss_seq(s, t, temperature=temp)
        (grad,) = torch.autograd.grad(loss, s)
        grad_flat = grad.reshape(-1)
        cos = F.cosine_similarity(grad_flat, limit_flat, dim=0).item()
        cosines.append(cos)

    assert cosines[0] < cosines[1] < cosines[2]
    assert cosines[-1] > 1.0 - 1e-4


# ----------------------------------------------------------------------
# 7. make_seq_kd_loss(alpha=1.0) はマスク付きCEに帰着する
# ----------------------------------------------------------------------
def test_make_seq_kd_loss_alpha_one_is_masked_ce():
    b, seq, v = 3, 5, 6
    pad_id = 0
    student = _rand(b, seq, v)
    teacher = _rand(b, seq, v)
    # 一部をpad_idにする(末尾側をpadにするのが一般的だが、性質検証には無関係)
    target_ids = torch.randint(1, v, (b, seq))
    target_ids[:, -2:] = pad_id

    loss_fn = make_seq_kd_loss(alpha=1.0, temperature=4.0, standardize=True, pad_id=pad_id)
    got = loss_fn(student, target_ids, teacher)

    expected = F.cross_entropy(
        student.reshape(-1, v), target_ids.reshape(-1).long(), ignore_index=pad_id,
    )
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


def test_make_seq_kd_loss_kind_validation():
    with pytest.raises(ValueError):
        make_seq_kd_loss(kind='bogus')
