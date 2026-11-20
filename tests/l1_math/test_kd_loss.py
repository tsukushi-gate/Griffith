"""C4 KD損失(convert/distill.py)の数理性質テスト。

二重温度設計(distill.py docstring):
    studentロジット=GroupSum出力(グループ和/τ_gs)なので実効軟化は τ_gs×T。
    本テストは KD側 T のみを扱い(τ_gs はモデル側)、T に関する性質を検証する。

許容誤差は tests/conftest.py の定数(TOL_F64_EQ 等)を参照。すべて float64。
"""

import math

import pytest
import torch
import torch.nn.functional as F

from griffith_logic.convert.distill import (
    kd_kl_loss,
    kd_mse_loss,
    make_distill_task,
    make_kd_loss,
    pack_targets,
    standardize_logits,
)
from griffith_logic.training.tasks import TaskSpec, _classification_acc
from tests.conftest import TOL_F64_EQ

_GEN = torch.Generator().manual_seed(1234)


def _rand(*shape: int) -> torch.Tensor:
    # 単一generatorを共有して呼ぶたびに前進させる(同一seed再設定で
    # 別々のはずのテンソルが一致してしまう罠を避ける)。
    return torch.randn(*shape, generator=_GEN, dtype=torch.float64)


# ----------------------------------------------------------------------
# 1. kd_kl_loss(s, s, T) == 0(任意T)、非負性
# ----------------------------------------------------------------------
@pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0, 4.0, 10.0])
def test_kl_self_is_zero(temperature):
    s = _rand(8, 5)
    loss = kd_kl_loss(s, s.clone(), temperature)
    assert abs(loss.item()) < TOL_F64_EQ


@pytest.mark.parametrize('temperature', [1.0, 4.0])
def test_kl_nonnegative(temperature):
    s = _rand(16, 7)
    t = _rand(16, 7)
    loss = kd_kl_loss(s, t, temperature)
    # KLダイバージェンスは常に >= 0(数値誤差の余地のみ許す)
    assert loss.item() >= -TOL_F64_EQ


# ----------------------------------------------------------------------
# 2. T=1 で 2クラス手計算例と一致
# ----------------------------------------------------------------------
def test_kl_two_class_analytic():
    """T=1、B=1、2クラスの解析値と照合。

    導出:
        student s = [0, 0]  -> softmax = [1/2, 1/2]
        teacher t = [ln 3, 0] -> softmax = [3/4, 1/4]
            (e^{ln3}=3, e^0=1, 和=4 なので [3/4, 1/4])
        KL(t || s) = Σ p_t log(p_t / p_s)
                   = 3/4 * log((3/4)/(1/2)) + 1/4 * log((1/4)/(1/2))
                   = 3/4 * log(3/2) + 1/4 * log(1/2)
        reduction='batchmean' は B=1 なのでそのまま。T^2 = 1。
    """
    s = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    t = torch.tensor([[math.log(3.0), 0.0]], dtype=torch.float64)
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    loss = kd_kl_loss(s, t, temperature=1.0)
    assert abs(loss.item() - expected) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 3. make_kd_loss(alpha=1) が F.cross_entropy と厳密一致
# ----------------------------------------------------------------------
@pytest.mark.parametrize('kind', ['kl', 'mse'])
def test_alpha_one_is_pure_ce(kind):
    s = _rand(12, 6)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    t = _rand(12, 6)
    loss_fn = make_kd_loss(alpha=1.0, temperature=4.0, kind=kind)
    got = loss_fn(s, labels, t)
    expected = F.cross_entropy(s, labels.long())
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 4. teacherロジット定数加算で kd_kl_loss 不変(softmaxシフト不変性)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('shift', [-3.0, 0.7, 5.0])
def test_kl_teacher_shift_invariance(shift):
    s = _rand(10, 4)
    t = _rand(10, 4)
    base = kd_kl_loss(s, t, temperature=2.0)
    # 各サンプルに定数を足す(全列に同じ値)
    shifted = kd_kl_loss(s, t + shift, temperature=2.0)
    assert abs(base.item() - shifted.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 5. T→∞極限: studentロジット勾配が中心化ロジット差の方向へ収束
# ----------------------------------------------------------------------
def test_kl_high_temperature_gradient_limit():
    """T→∞で ∂KL·T^2/∂s の方向が (中心化 s - 中心化 t) の方向へ収束。

    高温極限では softmax(z/T) ≈ 一様 + (1/T)(中心化z)。soft KD の勾配は
    (中心化 s - 中心化 t)/C に比例することが知られる。T を増やすと
    実際の勾配とこの極限方向の cos類似度が単調に 1 へ近づくことを検証。
    """
    s0 = _rand(6, 5)
    t = _rand(6, 5)

    # 極限方向: 中心化ロジット差(サンプルごと平均を引く)
    limit = (s0 - s0.mean(dim=-1, keepdim=True)) - (t - t.mean(dim=-1, keepdim=True))
    limit_flat = limit.reshape(-1)

    cosines = []
    for temp in (10.0, 100.0, 1000.0):
        s = s0.clone().requires_grad_(True)
        loss = kd_kl_loss(s, t, temperature=temp)
        (grad,) = torch.autograd.grad(loss, s)
        grad_flat = grad.reshape(-1)
        cos = F.cosine_similarity(grad_flat, limit_flat, dim=0).item()
        cosines.append(cos)

    # 単調に1へ近づく
    assert cosines[0] < cosines[1] < cosines[2]
    assert cosines[-1] > 1.0 - 1e-4


# ----------------------------------------------------------------------
# 6. standardize_logits: 平均0/分散1、アフィン不変
# ----------------------------------------------------------------------
def test_standardize_zero_mean_unit_var():
    x = _rand(9, 8)
    z = standardize_logits(x)
    means = z.mean(dim=-1)
    # 母分散(unbiased=False)で1に近い(eps分だけわずかに小さい)
    variances = z.var(dim=-1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-10)
    assert torch.allclose(variances, torch.ones_like(variances), atol=1e-4)


@pytest.mark.parametrize('a,b', [(2.0, 3.0), (0.5, -1.0), (10.0, 0.0)])
def test_standardize_affine_invariance(a, b):
    x = _rand(7, 6)
    z1 = standardize_logits(x)
    z2 = standardize_logits(a * x + b)
    # a>0 のときアフィン不変(eps項による微小差のみ)
    assert torch.allclose(z1, z2, atol=1e-5)


# ----------------------------------------------------------------------
# 7. kd_mse_loss: シフト不変、s==t で0
# ----------------------------------------------------------------------
def test_mse_self_is_zero():
    s = _rand(5, 4)
    assert abs(kd_mse_loss(s, s.clone()).item()) < TOL_F64_EQ


@pytest.mark.parametrize('shift', [-2.0, 1.5])
def test_mse_shift_invariance(shift):
    s = _rand(6, 5)
    t = _rand(6, 5)
    base = kd_mse_loss(s, t)
    # 各サンプルに定数加算しても中心化で消える
    got = kd_mse_loss(s + shift, t - shift)
    assert abs(base.item() - got.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 8. pack_targets / make_distill_task 契約
# ----------------------------------------------------------------------
def test_pack_targets_shape_dtype():
    labels = torch.tensor([0, 2, 1])
    teacher = _rand(3, 4)
    packed = pack_targets(labels, teacher)
    assert packed.shape == (3, 5)
    assert packed.dtype == teacher.dtype
    # 列0=ラベル(float格納)、列1..=teacherロジット
    assert torch.equal(packed[:, 0].long(), labels)
    assert torch.allclose(packed[:, 1:], teacher)


def test_distill_task_metric_matches_plain_acc():
    num_classes = 4
    task = make_distill_task(num_classes)
    labels = torch.tensor([0, 1, 2, 3, 0])
    teacher = _rand(5, num_classes)
    packed = pack_targets(labels, teacher)
    output = _rand(5, num_classes)

    got = task.metric_fn(output, packed)
    expected = _classification_acc(output, labels.long())
    assert got == expected


@pytest.mark.parametrize('standardize', [False, True])
@pytest.mark.parametrize('kind', ['kl', 'mse'])
def test_distill_task_loss_matches_make_kd_loss(standardize, kind):
    num_classes = 6
    task = make_distill_task(
        num_classes, alpha=0.5, temperature=3.0,
        standardize=standardize, kind=kind,
    )
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    teacher = _rand(8, num_classes)
    packed = pack_targets(labels, teacher)
    output = _rand(8, num_classes)

    got = task.loss_fn(output, packed)
    # 手動スライスで make_kd_loss を直接呼んだ結果と一致
    ref = make_kd_loss(
        alpha=0.5, temperature=3.0, standardize=standardize, kind=kind,
    )
    expected = ref(output, labels, teacher)
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 9. TaskSpec 契約(kind/loss_name/loss_fn/metric_fn/extra_metrics)
# ----------------------------------------------------------------------
def test_distill_task_taskspec_contract():
    task = make_distill_task(3)
    assert isinstance(task, TaskSpec)
    assert task.kind == 'classification'
    assert task.loss_name == 'kd'
    assert callable(task.loss_fn)
    assert callable(task.metric_fn)
    # extra_metrics は既定(空dict)で契約を満たす
    assert task.extra_metrics == {}

    # metrics() が主メトリクスを 'acc' キーで返すこと(tasks.py契約)
    labels = torch.tensor([0, 1, 2])
    teacher = _rand(3, 3)
    packed = pack_targets(labels, teacher)
    output = _rand(3, 3)
    m = task.metrics(output, packed)
    assert set(m.keys()) == {'acc'}
