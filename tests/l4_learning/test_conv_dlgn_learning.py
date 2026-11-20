"""L4: conv-DLGNの学習で hard acc が上がり、学習後も IR が hard eval と一致する。

タスク: 1×8×8 の二値画像に 2×2 の全 1 ブロックが**どこかに**あるか(平行移動不変の検出。
conv 木が 2×2 の AND を覚え、OR プール + 頭部が位置を潰す)。正解の回路は存在する
(頭部 rot2 の縮小層は fan-out 1 で 64 bit を 16 ゲートに OR できる)が、本テストが固定するのは
plan の L4 基準「hard acc が上がる」:

    seed 0・random8・lr 0.05・τ 2.0・800 ステップで train hard acc ≥ 0.75
    (較正: 同条件 3 seed の到達値 0.896 / 1.000 / 0.990。plan §5 の記録)
    + 初期 hard acc(≈ 0.5)より上がる + 学習後の native IR が hard eval と bit 一致

合格基準は緩めない(閾値 0.75 は較正値の最小 0.896 に余裕を持たせた固定値)。
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from griffith_logic.export.extract import export_ir
from griffith_logic.ir.simulate import simulate
from griffith_logic.models.conv_dlgn import ConvDLGNConfig, build_conv_dlgn

pytestmark = pytest.mark.learning

IN_SHAPE = (1, 8, 8)


def _block_dataset(n, seed, p=0.35):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, *IN_SHAPE, generator=g) < p).float()
    blocks = -F.max_pool2d(-x, 2, 1)                 # 2×2 窓の min
    y = (blocks.flatten(1).max(1).values > 0.5).long()
    return x, y


def _hard_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def test_block_detection_hard_acc_rises_and_ir_matches():
    torch.manual_seed(0)
    cfg = ConvDLGNConfig(in_shape=IN_SHAPE, channels=(8, 16), sampler='random8',
                         head_dims=(64, 32, 16), num_classes=2, tau=2.0)
    model = build_conv_dlgn(cfg, generator=torch.Generator().manual_seed(0))
    x, y = _block_dataset(512, seed=0)
    assert 0.3 < y.float().mean().item() < 0.7           # クラスが偏りすぎていない
    acc0 = _hard_acc(model, x, y)

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    best = 0.0
    for step in range(800):
        model.train()
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            best = max(best, _hard_acc(model, x, y))
            if best == 1.0:
                break
    acc = _hard_acc(model, x, y)
    assert best >= 0.75, f'hard acc が上がらない: best={best:.3f} (init {acc0:.3f})'
    assert best > acc0 + 0.1

    # 学習後の IR(native conv2d / or_pool2d)が hard eval と bit 一致
    model.eval()
    ir = export_ir(model, input_shape=IN_SHAPE, native_conv=True)
    flat = x.reshape(x.shape[0], -1).numpy().astype(np.int64)
    with torch.no_grad():
        hard = np.round(model(x).numpy() * cfg.tau).astype(np.int64)
    assert np.array_equal(simulate(ir, flat), hard)
    assert acc == _hard_acc(model, x, y)
