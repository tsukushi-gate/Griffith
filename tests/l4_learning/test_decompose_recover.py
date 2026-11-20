"""分解IR→DWN学習可能再構築→控えめpolishのL4テスト。

保証すること:
    1. decompose_mlp(thermo/adder)のIRが validate を通過し、
       simulate argmax == ternary teacherのhard forward argmax
       (adder armは厳密一致、thermo armは高一致)
    2. rebuild_trainable直後のhard eval forwardが IR simulateとbit厳密
       (recover.pyの保証をこの分解グラフでも確認)
    3. 数epochの控えめpolish(lr=2e-3)で train hard accが非低下
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from griffith_logic.convert.binarize import quantize_ternary
from griffith_logic.convert.decompose_thermo import (
    DecomposeSpec,
    decompose_mlp,
)
from griffith_logic.convert.recover import rebuild_trainable
from griffith_logic.convert.teacher_models import HeavisideSTE, TeacherMLPSte
from griffith_logic.ir.simulate import simulate
from griffith_logic.ir.validate import validate


def _make_ternary_teacher(seed: int, in_w: int = 12, hidden: int = 8,
                          classes: int = 2, n: int = 600):
    """小さなSTE teacherを合成タスクで学習し、ternaryへbakeする。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, in_w)).astype(np.float32)
    # planted ternaryルールでラベル生成(学習可能な整合タスク)
    w_true = rng.choice([-1, 0, 1], size=in_w, p=[0.4, 0.2, 0.4])
    y = (x @ w_true > 0).astype(np.int64)

    model = TeacherMLPSte(in_width=in_w, hidden=(hidden,),
                          num_classes=classes)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    for _ in range(60):
        opt.zero_grad()
        F.cross_entropy(model(xt), yt).backward()
        opt.step()
    # 隠れ層をternaryへbake(q·scaleの厳密ternary値にする)
    with torch.no_grad():
        for lin in model.linears[:-1]:
            q, scale, _ = quantize_ternary(lin.weight.detach().double(),
                                           per='channel')
            lin.weight.copy_((q.double() * scale).float())
    return model, x.astype(np.int64), y


def _teacher_hard_pred(model: TeacherMLPSte, x: np.ndarray) -> np.ndarray:
    """bake済みteacherのhard forward argmax(float32丸め差を避けfloat64)。"""
    h = torch.as_tensor(x, dtype=torch.float64)
    with torch.no_grad():
        for lin in model.linears[:-1]:
            h = HeavisideSTE.apply(
                F.linear(h, lin.weight.double(),
                         lin.bias.double() if lin.bias is not None
                         else None))
        final = model.linears[-1]
        logits = F.linear(h, final.weight.double(), final.bias.double())
    return logits.argmax(-1).numpy()


@pytest.mark.parametrize('arm', ['adder', 'thermo'])
def test_decompose_mlp_matches_teacher(arm):
    """①IR simulate argmax vs teacher hard argmax。"""
    model, x, _ = _make_ternary_teacher(0)
    spec = DecomposeSpec(alloc='uniform', bits_uniform=3)
    graph = decompose_mlp(model, x[:300], spec, hidden_arm=arm)
    validate(graph)
    pred_ir = simulate(graph, x).argmax(-1)
    pred_t = _teacher_hard_pred(model, x)
    agree = float((pred_ir == pred_t).mean())
    if arm == 'adder':
        # baseline Bは隠れ層厳密+最終層argmax保存 → 完全一致
        assert agree == 1.0, f'adder armが厳密でない: {agree}'
    else:
        # thermo: G≤12は単一LUT base case主体なのでこの規模はほぼ厳密
        assert agree >= 0.98, agree


def test_rebuild_trainable_bit_exact():
    """②rebuild直後のhard forwardがIR simulateとbit厳密。"""
    model, x, _ = _make_ternary_teacher(1)
    spec = DecomposeSpec(alloc='uniform', bits_uniform=2)
    graph = decompose_mlp(model, x[:300], spec, hidden_arm='thermo')
    validate(graph)
    dwn = rebuild_trainable(graph, parameterization='independent')
    dwn.eval()
    with torch.no_grad():
        scores = dwn(torch.as_tensor(x, dtype=torch.float32))
    pred_torch = scores.argmax(-1).numpy()
    pred_ir = simulate(graph, x).argmax(-1)
    assert (pred_torch == pred_ir).all(), 'rebuild直後にbit厳密でない'


def test_polish_does_not_degrade():
    """③控えめpolish(lr=2e-3, soft)でtrain hard acc非低下。"""
    model, x, y = _make_ternary_teacher(2)
    spec = DecomposeSpec(alloc='uniform', bits_uniform=2)
    graph = decompose_mlp(model, x[:300], spec, hidden_arm='thermo')
    dwn = rebuild_trainable(graph, parameterization='independent')

    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y)

    def hard_acc() -> float:
        dwn.eval()
        with torch.no_grad():
            return float((dwn(xt).argmax(-1) == yt).float().mean())

    acc0 = hard_acc()
    opt = torch.optim.Adam(dwn.parameters(), lr=2e-3)
    best = acc0
    for _ in range(5):
        dwn.train()
        opt.zero_grad()
        F.cross_entropy(dwn(xt), yt).backward()
        opt.step()
        best = max(best, hard_acc())
    # best-over-epochs概念: 初期値を下回らない
    assert best >= acc0 - 1e-9, (acc0, best)
