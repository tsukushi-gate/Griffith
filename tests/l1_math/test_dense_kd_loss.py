"""W3-seg 密予測(per-pixel)KD損失 make_dense_kd_loss の数理性質テスト。

分類版 make_kd_loss([B,C]) の [B,C,H,W] 拡張。式・T^2補正・standardizeの意味は
画像分類版と同一で、各ピクセルを独立サンプルとして扱う。ignore_index / pseudo_labels /
standardizeアフィン不変 / T^2スケール(分類版kd_kl_lossとの一致)を検証する。

許容誤差は tests/conftest.py の定数(TOL_F64_EQ)を参照。すべて float64。
"""

import pytest
import torch
import torch.nn.functional as F

from griffith_logic.convert.distill import (
    kd_kl_loss,
    make_dense_kd_loss,
)
from tests.conftest import TOL_F64_EQ

_GEN = torch.Generator().manual_seed(20260713)


def _rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=_GEN, dtype=torch.float64)


# ----------------------------------------------------------------------
# 1. teacher==student のとき KD 項が 0(alpha=0 で純KD → 損失0)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('standardize', [False, True])
@pytest.mark.parametrize('temperature', [1.0, 2.0, 4.0])
def test_kd_self_is_zero(standardize, temperature):
    s = _rand(2, 2, 5, 5)
    labels = torch.randint(0, 2, (2, 5, 5))
    loss_fn = make_dense_kd_loss(
        alpha=0.0, temperature=temperature, standardize=standardize)
    loss = loss_fn(s, labels, s.clone())
    assert abs(loss.item()) < TOL_F64_EQ


@pytest.mark.parametrize('kind', ['kl', 'mse'])
def test_kd_self_is_zero_both_kinds(kind):
    s = _rand(2, 2, 4, 4)
    labels = torch.randint(0, 2, (2, 4, 4))
    loss_fn = make_dense_kd_loss(alpha=0.0, kind=kind, standardize=True)
    assert abs(loss_fn(s, labels, s.clone()).item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 2. alpha=1(純CE)が dense CE(ignore_index対応)と厳密一致
# ----------------------------------------------------------------------
def test_alpha_one_is_pure_ce():
    s = _rand(2, 2, 6, 6)
    labels = torch.randint(0, 2, (2, 6, 6))
    labels[0, 0, 0] = 255  # ignore画素
    t = _rand(2, 2, 6, 6)
    loss_fn = make_dense_kd_loss(alpha=1.0, ignore_index=255)
    got = loss_fn(s, labels, t)
    expected = F.cross_entropy(s, labels.long(), ignore_index=255)
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 3. ignore_index: CE/KD双方で除外される
#    → ignore画素上でのみ teacher と student が食い違っても損失に寄与しない
# ----------------------------------------------------------------------
def test_ignore_index_excluded_from_kd():
    """ignore画素上でteacherがどれだけstudentと食い違ってもKDに寄与しない。"""
    s = _rand(1, 2, 2, 3)
    labels = torch.tensor([[[0, 255, 1], [1, 0, 255]]])  # 2画素をignore
    t = s.clone()  # まず teacher==student(KD=0の基準)
    loss_fn = make_dense_kd_loss(alpha=0.0, ignore_index=255, standardize=False)
    base = loss_fn(s, labels, t)
    assert abs(base.item()) < TOL_F64_EQ  # 完全一致 → KD=0
    # ignore画素(位置[0,1],[1,2])上でだけ teacher を大きくずらす
    t2 = s.clone()
    t2[0, :, 0, 1] = torch.tensor([9.0, -9.0], dtype=torch.float64)
    t2[0, :, 1, 2] = torch.tensor([-7.0, 7.0], dtype=torch.float64)
    perturbed = loss_fn(s, labels, t2)
    assert abs(perturbed.item()) < TOL_F64_EQ  # ignore画素の食い違いは無視


def test_ignore_index_partial_matches_valid_only():
    """一部ignore時、KD項が有効画素だけの平均になる(ignore画素を無視)。"""
    s = _rand(1, 2, 1, 3)
    labels = torch.tensor([[[0, 255, 1]]])  # 中央をignore
    t = _rand(1, 2, 1, 3)
    loss_fn = make_dense_kd_loss(alpha=0.0, standardize=False, temperature=2.0)
    got = loss_fn(s, labels, t)
    # 有効画素(idx0, idx2)だけを [N=2, C=2] にして分類版KLと一致するはず
    s_valid = s[..., 0, [0, 2]].permute(0, 2, 1).reshape(-1, 2)  # [2,2]
    t_valid = t[..., 0, [0, 2]].permute(0, 2, 1).reshape(-1, 2)
    expected = kd_kl_loss(s_valid, t_valid, temperature=2.0)
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 4. T^2スケール: 全画素有効・standardize offなら分類版 kd_kl_loss と一致
#    (per-pixel KLをflattenした [N,C] の batchmean × T^2)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('temperature', [1.0, 2.0, 4.0])
def test_dense_kd_matches_flat_kd_kl(temperature):
    b, c, h, w = 2, 2, 4, 5
    s = _rand(b, c, h, w)
    t = _rand(b, c, h, w)
    labels = torch.zeros(b, h, w, dtype=torch.long)  # 全有効
    loss_fn = make_dense_kd_loss(
        alpha=0.0, temperature=temperature, standardize=False)
    got = loss_fn(s, labels, t)
    # 参照: [N=b*h*w, C] にflattenして分類版KL(batchmean × T^2)
    s_flat = s.permute(0, 2, 3, 1).reshape(-1, c)
    t_flat = t.permute(0, 2, 3, 1).reshape(-1, c)
    expected = kd_kl_loss(s_flat, t_flat, temperature=temperature)
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 5. standardize: teacherロジットのアフィン変換(a>0)に KD 不変
# ----------------------------------------------------------------------
@pytest.mark.parametrize('a,b', [(2.0, 3.0), (0.5, -1.0)])
def test_standardize_affine_invariance(a, b):
    s = _rand(2, 2, 4, 4)
    t = _rand(2, 2, 4, 4)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    loss_fn = make_dense_kd_loss(
        alpha=0.0, temperature=2.0, standardize=True)
    base = loss_fn(s, labels, t)
    scaled = loss_fn(s, labels, a * t + b)
    assert abs(base.item() - scaled.item()) < 1e-5


def test_no_standardize_not_affine_invariant():
    """standardize off ではスケール変換で値が変わる(標準化の効果の裏取り)。"""
    s = _rand(2, 2, 4, 4)
    t = _rand(2, 2, 4, 4)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    loss_fn = make_dense_kd_loss(
        alpha=0.0, temperature=2.0, standardize=False)
    base = loss_fn(s, labels, t)
    scaled = loss_fn(s, labels, 3.0 * t)
    assert abs(base.item() - scaled.item()) > 1e-3


# ----------------------------------------------------------------------
# 6. pseudo_labels: hard項GTが teacher argmax に置換される
#    alpha=1(純CE)で CE(student, teacher.argmax) と一致・GTは無視
# ----------------------------------------------------------------------
def test_pseudo_labels_replaces_gt():
    s = _rand(2, 2, 5, 5)
    t = _rand(2, 2, 5, 5)
    # GT は意図的にデタラメ(全ignore)にしても pseudo は teacher argmax を使う
    bogus_labels = torch.full((2, 5, 5), 255, dtype=torch.long)
    loss_fn = make_dense_kd_loss(alpha=1.0, pseudo_labels=True)
    got = loss_fn(s, bogus_labels, t)
    expected = F.cross_entropy(s, t.argmax(dim=1).long())
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


def test_pseudo_labels_ignores_ignore_index():
    """pseudo_labels時は全画素有効(teacher argmaxは全画素で定義される)。"""
    s = _rand(1, 2, 3, 3)
    t = _rand(1, 2, 3, 3)
    labels = torch.full((1, 3, 3), 255, dtype=torch.long)
    loss_fn = make_dense_kd_loss(
        alpha=0.0, pseudo_labels=True, standardize=False, temperature=2.0)
    got = loss_fn(s, labels, t)
    # 全画素をflattenした分類版KLと一致(ignoreされない)
    s_flat = s.permute(0, 2, 3, 1).reshape(-1, 2)
    t_flat = t.permute(0, 2, 3, 1).reshape(-1, 2)
    expected = kd_kl_loss(s_flat, t_flat, temperature=2.0)
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


# ----------------------------------------------------------------------
# 7. 非負性・勾配疎通(studentへ勾配が流れる)
# ----------------------------------------------------------------------
def test_nonnegative_and_grad_flows():
    s = _rand(2, 2, 4, 4).requires_grad_(True)
    t = _rand(2, 2, 4, 4)
    labels = torch.randint(0, 2, (2, 4, 4))
    loss_fn = make_dense_kd_loss(alpha=0.25, temperature=2.0, standardize=True)
    loss = loss_fn(s, labels, t)
    assert loss.item() >= -TOL_F64_EQ
    (grad,) = torch.autograd.grad(loss, s)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        make_dense_kd_loss(kind='bogus')


# ----------------------------------------------------------------------
# 8. ★C=2でのstandardize退化(w3r悪化仮説の実証、W3-seg知見)
#    2要素のZ-scoreは (x1,x2) → (±1,∓1)(符号のみ)に潰れ、teacherロジット差の
#    大きさ(確信度=dark knowledge)が完全に消える。KD項は温度に依らず
#    「teacher argmaxの±1固定ターゲット」=ノイズ入りハードラベルと化す。
#    → バイナリセグではstandardize=offを推奨(--no-distill-standardize)。
# ----------------------------------------------------------------------
@pytest.mark.parametrize('a,b', [(0.1, 0.0), (5.0, 0.0), (100.0, -3.0)])
def test_standardize_binary_degenerates_to_sign(a, b):
    """C=2のZ-scoreは a>b なら常に (+1,-1)。差の大きさは消える。"""
    from griffith_logic.convert.distill import standardize_logits
    x = torch.tensor([[a, b]], dtype=torch.float64)
    z = standardize_logits(x)
    expected = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    # eps(1e-6)によるわずかな縮みのみ許容
    assert torch.allclose(z, expected, atol=1e-4)


def test_standardize_binary_erases_teacher_confidence():
    """C=2ではargmaxが同じteacher同士(確信度が全く違っても)KD損失が同一になる。

    t_weak: 差0.2(境界画素のような低確信) / t_strong: 差20(高確信)。
    standardize=Trueだと両者のKD損失が厳密一致(=確信度情報の消失)、
    standardize=Falseなら異なる(確信度が保存される)ことを両面で確認。
    """
    s = _rand(2, 2, 4, 4)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    base = _rand(2, 4, 4)
    # クラス0側にmargin/2、クラス1側に-margin/2を足す(全画素argmax=0で共通)
    def _teacher(margin: float) -> torch.Tensor:
        t = torch.stack([base + margin / 2, base - margin / 2], dim=1)
        return t
    t_weak = _teacher(0.2)
    t_strong = _teacher(20.0)

    loss_std = make_dense_kd_loss(alpha=0.0, temperature=2.0, standardize=True)
    got_weak = loss_std(s, labels, t_weak)
    got_strong = loss_std(s, labels, t_strong)
    # 退化: 確信度が全く違うのにKD損失が同一(標準化のeps=1e-6による
    # ±1のわずかな縮み差のみ許容: margin=0.2はstd=0.1なので縮み~1e-5)
    assert abs(got_weak.item() - got_strong.item()) < 1e-4

    loss_nostd = make_dense_kd_loss(alpha=0.0, temperature=2.0, standardize=False)
    raw_weak = loss_nostd(s, labels, t_weak)
    raw_strong = loss_nostd(s, labels, t_strong)
    # 非標準化では確信度差がKD損失に残る
    assert abs(raw_weak.item() - raw_strong.item()) > 1e-2


# ----------------------------------------------------------------------
# 9. hard_loss_fn差し替え(不均衡対策損失をKD hard項に注入する経路)
# ----------------------------------------------------------------------
def test_hard_loss_fn_replaces_ce_when_alpha_one():
    """alpha=1(純hard)でhard_loss_fnが指定されると、その出力がそのまま損失になる。"""
    s = _rand(2, 3, 4, 4)
    labels = torch.randint(0, 3, (2, 4, 4))
    t = _rand(2, 3, 4, 4)
    calls = []

    def custom_hard_loss(student_logits, target):
        calls.append((student_logits, target))
        return (student_logits ** 2).mean()  # CEと無関係な適当な損失

    loss_fn = make_dense_kd_loss(alpha=1.0, hard_loss_fn=custom_hard_loss)
    got = loss_fn(s, labels, t)
    expected = (s ** 2).mean()
    assert abs(got.item() - expected.item()) < TOL_F64_EQ
    assert len(calls) == 1
    assert torch.equal(calls[0][1], labels.long())


def test_hard_loss_fn_none_keeps_default_ce_behavior():
    """hard_loss_fn=None(既定)は従来のF.cross_entropy経路と一致(後方互換)。"""
    s = _rand(2, 3, 4, 4)
    labels = torch.randint(0, 3, (2, 4, 4))
    labels[0, 0, 0] = 255
    t = _rand(2, 3, 4, 4)
    with_none = make_dense_kd_loss(alpha=1.0, hard_loss_fn=None, ignore_index=255)
    without_arg = make_dense_kd_loss(alpha=1.0, ignore_index=255)
    got_a = with_none(s, labels, t)
    got_b = without_arg(s, labels, t)
    assert abs(got_a.item() - got_b.item()) < TOL_F64_EQ


def test_hard_loss_fn_with_pseudo_labels_uses_teacher_argmax():
    """pseudo_labels=Trueだとhard_loss_fnにはteacher argmaxが渡される(GT無視)。"""
    s = _rand(1, 2, 3, 3)
    t = _rand(1, 2, 3, 3)
    bogus_labels = torch.full((1, 3, 3), 255, dtype=torch.long)
    received = {}

    def custom_hard_loss(student_logits, target):
        received['target'] = target
        return F.cross_entropy(student_logits, target)

    loss_fn = make_dense_kd_loss(
        alpha=1.0, pseudo_labels=True, hard_loss_fn=custom_hard_loss)
    got = loss_fn(s, bogus_labels, t)
    assert torch.equal(received['target'], t.argmax(dim=1).long())
    expected = F.cross_entropy(s, t.argmax(dim=1).long())
    assert abs(got.item() - expected.item()) < TOL_F64_EQ


def test_hard_loss_fn_combines_with_kd_term_at_partial_alpha():
    """alpha<1でもKD(soft)項は従来通り、hard項だけがhard_loss_fnに置き換わる。"""
    s = _rand(2, 2, 4, 4)
    labels = torch.randint(0, 2, (2, 4, 4))
    t = _rand(2, 2, 4, 4)

    def custom_hard_loss(student_logits, target):
        return F.cross_entropy(student_logits, target) * 2.0  # 明確に異なる値

    alpha = 0.3
    loss_custom = make_dense_kd_loss(
        alpha=alpha, temperature=2.0, standardize=True,
        hard_loss_fn=custom_hard_loss)(s, labels, t)
    loss_plain = make_dense_kd_loss(
        alpha=alpha, temperature=2.0, standardize=True)(s, labels, t)
    ce_plain = F.cross_entropy(s, labels.long())
    # 差分は alpha * (custom_ce - plain_ce) = alpha * ce_plain のはず
    expected_diff = alpha * ce_plain
    assert abs((loss_custom - loss_plain).item() - expected_diff.item()) < 1e-9


def test_standardize_binary_equals_pm_one_target():
    """C=2のstandardize済teacherは、生ロジットを(+1,-1)·argmax向きに置いた
    固定ターゲットと同じKD損失を与える(=ハードラベル化の直接実証)。"""
    s = _rand(1, 2, 3, 3)
    labels = torch.zeros(1, 3, 3, dtype=torch.long)
    t = _rand(1, 2, 3, 3) * 7.3  # 任意スケールのteacher
    loss_std = make_dense_kd_loss(alpha=0.0, temperature=2.0, standardize=True)
    got = loss_std(s, labels, t)
    # 参照: teacherをargmax方向の±1に置換し、studentのみstandardizeを再現
    sign = torch.where(t[:, 0:1] >= t[:, 1:2],
                       torch.tensor(1.0, dtype=torch.float64),
                       torch.tensor(-1.0, dtype=torch.float64))
    t_pm1 = torch.cat([sign, -sign], dim=1)  # (±1, ∓1)
    # studentのstandardizeは維持する必要があるため、standardize=Trueのまま
    # t_pm1(すでに±1)を渡す — standardize(±1)=±1(eps縮みのみ)なので等価
    expected = loss_std(s, labels, t_pm1)
    assert abs(got.item() - expected.item()) < 1e-6
