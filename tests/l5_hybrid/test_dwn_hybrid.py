"""L5: DWNのハイブリッドAPI統合。

LogicSpec(parameterization='dwn') で LogicLinear の内部が
DWNLayer(FD/EFD勾配)になり、thermometerエンコーダと組み合わせて
実数in/実数outのdrop-inとして機能することを検証する。
"""

import itertools
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import LogicLinear, LogicSpec, split_param_groups
from griffith_logic.layers.dwn import DWNLayer


def _dwn_spec(**kw):
    base = dict(
        parameterization='dwn', n_inputs=2, depth=2,
        neurons_per_output=8, encoder='thermometer', encoder_bits=4,
    )
    base.update(kw)
    return LogicSpec(**base)


class TestDWNLogicLinear:
    def test_internal_layers_are_dwn(self, seed):
        layer = LogicLinear(8, 4, spec=_dwn_spec())
        assert all(isinstance(m, DWNLayer) for m in layer.luts)
        assert layer.luts[0].mapping_mode == 'learnable'
        assert layer.luts[1].mapping_mode == 'random'
        # thermometer: 入力幅8×4bit=32が第1層のin_bits
        assert layer.luts[0].in_bits == 32

    def test_sandwich_gradient_flow(self, seed):
        """Linear→LogicLinear(dwn)→Linear で全パラメータに勾配"""
        torch.manual_seed(3)
        model = nn.Sequential(
            nn.Linear(12, 8),
            LogicLinear(8, 4, spec=_dwn_spec()),
            nn.Linear(4, 2),
        )
        x = torch.randn(16, 12)
        model.train()
        loss = model(x).square().mean()
        loss.backward()
        dead = [
            name for name, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0
        ]
        assert not dead, f'勾配が届いていない: {dead}'

    def test_output_shape_3d(self, seed):
        layer = LogicLinear(8, 4, spec=_dwn_spec())
        layer.train()
        y = layer(torch.randn(2, 5, 8))
        assert y.shape == (2, 5, 4)

    def test_split_param_groups_counts_dwn_as_logic(self, seed):
        model = nn.Sequential(
            nn.Linear(8, 8),
            LogicLinear(8, 4, spec=_dwn_spec()),
        )
        groups = {g['name']: g for g in split_param_groups(model)}
        dwn_params = {
            id(p) for m in model.modules() if isinstance(m, DWNLayer)
            for p in m.parameters()
        }
        logic_ids = {id(p) for p in groups['logic']['params']}
        assert dwn_params <= logic_ids

    def test_learns_xor(self, seed):
        """DWNベースのLogicLinearでXORが学習できる"""
        torch.manual_seed(11)
        x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
        y = torch.tensor([0, 1, 1, 0])
        model = LogicLinear(
            2, 2,
            spec=_dwn_spec(encoder_bits=2, neurons_per_output=16, depth=2),
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        model.train()
        acc = 0.0
        for _ in range(800):
            opt.zero_grad()
            logits = model(x)
            F.cross_entropy(logits, y).backward()
            opt.step()
            acc = (logits.argmax(-1) == y).float().mean().item()
            if acc == 1.0:
                break
        assert acc == 1.0, f'DWN LogicLinearでXORが学習できない: acc={acc}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
