"""L5: 部分置換API(replace_modules系)の構造検証。"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic import (
    LogicLinear,
    LogicMLP,
    LogicSpec,
    convert_linear_to_logic,
    replace_mlp_with_lut,
    replace_modules,
    split_param_groups,
)
from griffith_logic.layers.lut import LutLayer


def _mini_mlp():
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


class _Block(nn.Module):
    """Transformerブロック風のモジュール(attn + mlp)"""

    def __init__(self, dim=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        return x + self.mlp(self.norm2(x))


class TestConvertLinearToLogic:
    def test_replaces_all_linears(self, seed):
        model = _mini_mlp()
        new, replaced = convert_linear_to_logic(model)
        assert len(replaced) == 3
        assert all(isinstance(new.get_submodule(n), LogicLinear) for n in replaced)
        # 元モデルは無傷(inplace=False)
        assert isinstance(model[0], nn.Linear)
        # 形状契約
        y = new(torch.randn(2, 8))
        assert y.shape == (2, 4)

    def test_exclude_pattern(self, seed):
        model = nn.Sequential()
        model.add_module('body', nn.Linear(8, 8))
        model.add_module('head', nn.Linear(8, 4))
        new, replaced = convert_linear_to_logic(model, exclude='*head*')
        assert replaced == ['body']
        assert isinstance(new.head, nn.Linear)

    def test_include_pattern(self, seed):
        model = _Block()
        new, replaced = convert_linear_to_logic(model, include='mlp*')
        assert set(replaced) == {'mlp.0', 'mlp.2'}
        # attn内部のLinear(out_proj)は非置換
        assert isinstance(new.attn.out_proj, nn.Linear)

    def test_unreplaced_state_dict_preserved(self, seed):
        """未置換部分のパラメータ値が元と同一"""
        model = _Block()
        new, _ = convert_linear_to_logic(model, include='mlp*')
        for key, value in model.state_dict().items():
            if key.startswith(('attn', 'norm')):
                assert torch.equal(new.state_dict()[key], value), key


class TestReplaceMlpWithLut:
    def test_replaces_mlp_block(self, seed):
        model = nn.Sequential(_Block(), _Block())
        new, replaced = replace_mlp_with_lut(
            model, spec=LogicSpec(neurons_per_output=4),
        )
        assert replaced == ['0.mlp', '1.mlp']
        for n in replaced:
            mlp = new.get_submodule(n)
            assert isinstance(mlp, LogicMLP)
            # in/out自動推定: dim=16 → 16
            assert mlp.in_features == 16 and mlp.out_features == 16
        # forwardが通る(Attention実数 + MLP論理のハイブリッド)
        y = new(torch.randn(2, 5, 16))
        assert y.shape == (2, 5, 16)

    def test_mlp_class_match(self, seed):
        class MyFFN(nn.Sequential):
            def __init__(self):
                super().__init__(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

        model = nn.Sequential(MyFFN(), nn.Linear(8, 2))
        new, replaced = replace_mlp_with_lut(model, mlp_class=MyFFN)
        assert replaced == ['0']
        assert isinstance(new[1], nn.Linear)  # 通常Linearは残る


class TestReplaceModulesGeneric:
    def test_parent_priority(self, seed):
        """親が一致したら内部の子は走査しない"""
        model = nn.Sequential(_mini_mlp())
        new, replaced = replace_modules(
            model,
            match=lambda name, m: isinstance(m, nn.Sequential | nn.Linear)
            and name != '',
            factory=lambda name, m: nn.Identity(),
        )
        assert replaced == ['0']  # 親Sequentialのみ

    def test_inplace(self, seed):
        model = _mini_mlp()
        new, replaced = convert_linear_to_logic(model, inplace=True)
        assert new is model
        assert isinstance(model[0], LogicLinear)


class TestSplitParamGroups:
    def test_split(self, seed):
        model = nn.Sequential(
            nn.Linear(8, 16),
            LogicLinear(16, 8, spec=LogicSpec(neurons_per_output=4)),
        )
        groups = split_param_groups(model, logic_lr=0.01, ann_lr=1e-3)
        by_name = {g['name']: g for g in groups}
        assert by_name['logic']['lr'] == 0.01
        assert by_name['ann']['lr'] == 1e-3
        # LutLayerのweightsはlogic側
        lut_params = {id(p) for m in model.modules() if isinstance(m, LutLayer)
                      for p in m.parameters(recurse=False)}
        assert {id(p) for p in by_name['logic']['params']} == lut_params
        # 全パラメータが重複なくどちらかに属する
        n_total = sum(1 for _ in model.parameters())
        assert len(by_name['logic']['params']) + len(by_name['ann']['params']) == n_total
        # optimizerにそのまま渡せる
        torch.optim.AdamW(groups)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
