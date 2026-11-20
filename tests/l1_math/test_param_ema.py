"""L1: ParamEMA（hard_st の argmax 安定化狙い）の数式性質。"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'experiments'))
from ema import ParamEMA  # noqa: E402


def _model(v: float = 0.0) -> torch.nn.Module:
    m = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(v)
    return m


def test_shadow_starts_as_a_copy():
    m = _model(1.0)
    ema = ParamEMA(m, decay=0.9)
    assert torch.equal(ema.shadow['weight'], m.weight)
    # 独立したコピーであること（モデル側を書き換えても shadow は動かない）
    with torch.no_grad():
        m.weight.fill_(5.0)
    assert torch.allclose(ema.shadow['weight'], torch.ones(2, 3))


def test_update_follows_the_ema_recurrence():
    """s ← d·s + (1−d)·w を素朴計算と照合。"""
    m = _model(0.0)
    ema = ParamEMA(m, decay=0.8)
    expected = torch.zeros(2, 3)
    for w in (1.0, 2.0, -1.0):
        with torch.no_grad():
            m.weight.fill_(w)
        ema.update(m)
        expected = 0.8 * expected + 0.2 * torch.full((2, 3), w)
    assert torch.allclose(ema.shadow['weight'], expected, atol=1e-6)


def test_applied_swaps_then_restores_exactly():
    m = _model(3.0)
    ema = ParamEMA(m, decay=0.5)
    with torch.no_grad():
        ema.shadow['weight'].fill_(7.0)
    before = m.weight.detach().clone()
    with ema.applied(m):
        assert torch.allclose(m.weight, torch.full((2, 3), 7.0))
    assert torch.equal(m.weight, before)


def test_applied_restores_even_on_exception():
    m = _model(3.0)
    ema = ParamEMA(m, decay=0.5)
    with torch.no_grad():
        ema.shadow['weight'].fill_(7.0)
    before = m.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with ema.applied(m):
            raise RuntimeError('boom')
    assert torch.equal(m.weight, before)


def test_ema_reduces_argmax_flips_on_a_noisy_two_way_tie():
    """★狙いの実証: 拮抗2ゲートに雑音が乗るとき、EMA は argmax の反転を減らす。

    生 logit は毎ステップ反転しうるが、EMA を通すと反転回数が大きく減ることを示す。
    """
    torch.manual_seed(0)
    m = _model(0.0)
    ema = ParamEMA(m, decay=0.95)
    raw_flips = ema_flips = 0
    prev_raw = prev_ema = None
    for _ in range(300):
        with torch.no_grad():
            # 2択がほぼ拮抗（差 0.01）した状態に雑音 0.3 が乗る
            base = torch.tensor([[0.0, 0.01, 0.0], [0.0, 0.0, 0.0]])
            m.weight.copy_(base + 0.3 * torch.randn(2, 3))
        ema.update(m)
        raw = int(m.weight[0, :2].argmax())
        av = int(ema.shadow['weight'][0, :2].argmax())
        if prev_raw is not None and raw != prev_raw:
            raw_flips += 1
        if prev_ema is not None and av != prev_ema:
            ema_flips += 1
        prev_raw, prev_ema = raw, av
    assert ema_flips < raw_flips / 3, f'EMA={ema_flips} raw={raw_flips}'


def test_invalid_decay_raises():
    with pytest.raises(ValueError, match='decay'):
        ParamEMA(_model(), decay=1.0)
