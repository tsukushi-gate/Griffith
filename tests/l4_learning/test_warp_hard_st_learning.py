"""L4: warp×{hard_st, gumbel_st}の学習収束スモーク。

合格基準はhard eval 100%(xor2 / parity4をn_inputs=4で)。
n>2のfan-inでhard経路を通す訓練が成立することの回帰固定
(softmax×hard系はn>2で単一minterm崩壊するためwarpが唯一のhard腕)。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_toy_problems import _bits_dataset, _mlp  # noqa: E402

# L4=学習収束テストは全てlearning(CI除外、ローカルmake testでは実行)
pytestmark = pytest.mark.learning


def _train_until_hard(model, x, y, *, lr=0.05, max_steps=3000,
                      check_every=50, scheduler=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(max_steps):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if (step + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x).argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                return 1.0
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


class TestWARPHardSTLearning:
    def test_xor_warp_hard_st(self, seed):
        x, y = _bits_dataset(2, lambda b: b[0] ^ b[1])
        model = _mlp(2, 2, [16, 16], param='warp', selection='hard_st')
        acc = _train_until_hard(model, x, y, max_steps=2000)
        assert acc == 1.0, f'warp×hard_st: XOR hard acc={acc}'

    def test_parity4_n4_warp_gumbel_st(self, seed):
        """parity4をn_inputs=4のwarp×gumbel_stで学習(掃引の核心腕)"""
        x, y = _bits_dataset(4, lambda b: (b[0] + b[1] + b[2] + b[3]) % 2)
        model = _mlp(4, 2, [16, 16], n_inputs=4, param='warp',
                     selection='gumbel_st')
        acc = _train_until_hard(model, x, y, max_steps=3000)
        assert acc == 1.0, f'warp×gumbel_st n=4: parity4 hard acc={acc}'

    def test_parity4_n4_warp_gumbel_st_with_anneal(self, seed):
        """TauAnnealScheduler(τ_b幾何減衰+ノイズ減衰)込みでも収束する"""
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), '..', '..', 'experiments'))
        from cage_scheduler import TauAnnealScheduler

        x, y = _bits_dataset(4, lambda b: (b[0] + b[1] + b[2] + b[3]) % 2)
        model = _mlp(4, 2, [16, 16], n_inputs=4, param='warp',
                     selection='gumbel_st')
        sched = TauAnnealScheduler(model, tau_0=1.0, tau_min=0.2,
                                   total_steps=2000, anneal_noise=True)
        acc = _train_until_hard(model, x, y, max_steps=3000, scheduler=sched)
        assert acc == 1.0, f'warp×gumbel_st+anneal n=4: parity4 hard acc={acc}'
