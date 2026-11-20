"""L4: LutConv2d(wiring='learnable')のXOR-in-noise収束テスト。

dense版(tests/test_learnable_wiring.py::TestLearnableWiringConvergenceSmoke)
と同じ設計思想: XORの2つの意味あるピクセルをノイズピクセルに埋め込んだ
1x1x8の「画像」に対し、kernel_sizeで全幅を覆う1トリー木(n_inputs=2、
tree_depth=1)を64個並べたLutConv2dで学習する。

wiring='fixed'では各ツリーが独立ランダムに2ピクセルを選ぶため、両方とも
意味あるXORピクセルに当たる保証が弱く信号を再構成しにくい。
wiring='learnable'ならforward argmax選択が勾配で有効な入力へ寄っていく
ため、同条件で明確に有利になることを期待する（このテストのhard acc=100%
はskill: new-component-tdd ④の必須要件）。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.conv import LutConv2d  # noqa: E402
from griffith_logic.datasets.logic import parity  # noqa: E402
from griffith_logic.group_sum import GroupSum  # noqa: E402

N_RELEVANT = 2   # XOR(a, b)
N_NOISE = 6
IN_DIM = N_RELEVANT + N_NOISE
HIDDEN = 64
STEPS = 300
LR = 0.02
BATCH = 64


class _ConvXORModel(nn.Module):
    """1x1x8「画像」全幅を1本の木(n_inputs=2, tree_depth=1)64本で覆う
    最小Conv-DLGN。conv出力(B,HIDDEN,1,1)をsqueezeしてGroupSumへ渡す。"""

    def __init__(self, wiring: str):
        super().__init__()
        self.conv = LutConv2d(
            in_channels=1, out_channels=HIDDEN,
            kernel_size=(1, IN_DIM), tree_depth=1, padding=0,
            n_inputs=2, parameterization='softmax16g',
            input_channels_per_tree=1, wiring=wiring,
        )
        self.readout = GroupSum(k=2, tau=5.0)

    def forward(self, x):
        # x: (B, IN_DIM) → (B, 1, 1, IN_DIM)
        y = self.conv(x.view(x.shape[0], 1, 1, IN_DIM))  # (B, HIDDEN, 1, 1)
        y = y.reshape(x.shape[0], HIDDEN)
        return self.readout(y)


def _make_batch(xs, ys, batch_size, gen):
    idx = torch.randint(0, xs.shape[0], (batch_size,), generator=gen)
    x_rel = xs[idx]
    noise = torch.randint(0, 2, (batch_size, N_NOISE), generator=gen).float()
    x = torch.cat([x_rel, noise], dim=-1)
    y = ys[idx].squeeze(-1).long()
    return x, y


def _run(wiring, seed, xs, ys):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    model = _ConvXORModel(wiring)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(STEPS):
        x, y = _make_batch(xs, ys, BATCH, gen)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert torch.isfinite(loss).all(), 'lossが発散(NaN/Inf)'

    model.eval()
    x_full = torch.cat([xs, torch.zeros(xs.shape[0], N_NOISE)], dim=-1)
    with torch.no_grad():
        logits = model(x_full)
        pred = logits.argmax(-1)
        acc = (pred == ys.squeeze(-1).long()).float().mean().item()
    return acc


class TestConvLearnableWiringConvergence:
    def test_learnable_wiring_reaches_hard_acc_100(self):
        """XOR-in-noiseタスクでwiring='learnable'がhard acc=100%へ到達する
        (new-component-tdd ④の必須要件。seed=0で決定化)"""
        spec = parity(N_RELEVANT)
        xs, ys = spec.dataset.tensors
        acc = _run('learnable', seed=0, xs=xs, ys=ys)
        assert acc == 1.0, f'wiring=learnableでhard acc=100%未到達: acc={acc}'

    def test_learnable_wiring_does_not_diverge(self):
        """訓練がNaN/Infに発散しないことの最低限スモーク(複数seed)"""
        spec = parity(N_RELEVANT)
        xs, ys = spec.dataset.tensors
        for seed in (1, 2):
            acc = _run('learnable', seed=seed, xs=xs, ys=ys)
            assert 0.0 <= acc <= 1.0

    def test_learnable_wiring_matches_or_beats_fixed(self):
        """学習配線の平均精度が固定ランダム配線と同等以上であることのスモーク
        (dense版TestLearnableWiringConvergenceSmokeと同じ比較設計)"""
        spec = parity(N_RELEVANT)
        xs, ys = spec.dataset.tensors
        seeds = (0, 1, 2)

        fixed_accs = [_run('fixed', s, xs, ys) for s in seeds]
        learnable_accs = [_run('learnable', s, xs, ys) for s in seeds]

        fixed_mean = sum(fixed_accs) / len(fixed_accs)
        learnable_mean = sum(learnable_accs) / len(learnable_accs)

        print(f'\n[conv wiring smoke] fixed_accs={fixed_accs} mean={fixed_mean:.3f}')
        print(f'[conv wiring smoke] learnable_accs={learnable_accs} mean={learnable_mean:.3f}')

        assert learnable_mean > 0.0
        assert learnable_mean >= fixed_mean, (
            f'learnable({learnable_mean:.3f}) < fixed({fixed_mean:.3f})'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
