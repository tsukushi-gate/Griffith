"""checkpoint_utils.load_partial_state_dict のテスト(W13a-3)。

合成state_dictで部分一致/形状不一致スキップ/キー名リマップの既知値を検証する。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from checkpoint_utils import load_partial_state_dict


class _Sub(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=False)


class _Model(nn.Module):
    """named parameters: a.lin.weight, b.lin.weight"""

    def __init__(self, a_shape=(4, 3), b_shape=(5, 4)):
        super().__init__()
        self.a = _Sub(*a_shape[::-1])
        self.b = _Sub(*b_shape[::-1])


class TestLoadPartialStateDictExactMatch:
    def test_full_load_ratio_1(self):
        src = _Model()
        dst = _Model()
        result = load_partial_state_dict(dst, src.state_dict())
        assert result['load_ratio'] == 1.0
        assert result['total_keys'] == 2
        assert set(result['loaded_keys']) == {'a.lin.weight', 'b.lin.weight'}
        assert result['skipped_missing'] == []
        assert result['skipped_shape_mismatch'] == []
        assert torch.equal(dst.a.lin.weight, src.a.lin.weight)
        assert torch.equal(dst.b.lin.weight, src.b.lin.weight)


class TestLoadPartialStateDictMissingKeys:
    def test_missing_key_skipped(self):
        src = _Model()
        src_state = src.state_dict()
        del src_state['b.lin.weight']  # ckpt側に無いキー

        dst = _Model()
        before_b = dst.b.lin.weight.detach().clone()
        result = load_partial_state_dict(dst, src_state)

        assert result['loaded_keys'] == ['a.lin.weight']
        assert result['skipped_missing'] == ['b.lin.weight']
        assert result['skipped_shape_mismatch'] == []
        assert result['total_keys'] == 2
        assert result['load_ratio'] == pytest.approx(0.5)
        assert torch.equal(dst.a.lin.weight, src.a.lin.weight)
        # ロードされなかったキーは初期値のまま
        assert torch.equal(dst.b.lin.weight, before_b)


class TestLoadPartialStateDictShapeMismatch:
    def test_shape_mismatch_skipped(self):
        src = _Model(a_shape=(4, 3), b_shape=(9, 4))  # bの形状が不一致
        dst = _Model(a_shape=(4, 3), b_shape=(5, 4))
        before_b = dst.b.lin.weight.detach().clone()

        result = load_partial_state_dict(dst, src.state_dict())

        assert result['loaded_keys'] == ['a.lin.weight']
        assert result['skipped_missing'] == []
        assert len(result['skipped_shape_mismatch']) == 1
        key, model_shape, ckpt_shape = result['skipped_shape_mismatch'][0]
        assert key == 'b.lin.weight'
        assert model_shape == (5, 4)
        assert ckpt_shape == (9, 4)
        assert result['load_ratio'] == pytest.approx(0.5)
        assert torch.equal(dst.b.lin.weight, before_b)


class TestLoadPartialStateDictKeyMap:
    def test_prefix_remap_dict(self):
        """dict形式のkey_map: {model側prefix: ckpt側prefix}"""
        src = _Model()
        src_state = {
            f'encoder.{k}': v for k, v in src.state_dict().items()
        }
        dst = _Model()
        result = load_partial_state_dict(
            dst, src_state, key_map={'a': 'encoder.a', 'b': 'encoder.b'},
        )
        assert result['load_ratio'] == 1.0
        assert torch.equal(dst.a.lin.weight, src.a.lin.weight)
        assert torch.equal(dst.b.lin.weight, src.b.lin.weight)

    def test_prefix_remap_list_first_match_wins(self):
        """list形式のkey_map: 最初にマッチした規則を使う"""
        src = _Model()
        src_state = {f'enc.{k}': v for k, v in src.state_dict().items()}
        dst = _Model()
        # 'a.lin' はより具体的な規則(先頭)が優先される
        result = load_partial_state_dict(
            dst, src_state,
            key_map=[('a.lin', 'enc.a.lin'), ('', 'enc.')],
        )
        assert result['load_ratio'] == 1.0

    def test_no_match_falls_through_unremapped(self):
        """key_mapのどの規則にもマッチしないキーはそのまま探索(素通し)"""
        src = _Model()
        src_state = src.state_dict()  # リマップ不要な命名
        dst = _Model()
        result = load_partial_state_dict(
            dst, src_state, key_map={'nonexistent_prefix': 'ckpt_prefix'},
        )
        assert result['load_ratio'] == 1.0

    def test_partial_key_map_only_remaps_matching_prefix(self):
        """key_mapが一部のキーにしか一致しない場合、残りは素通しで探索される"""
        src = _Model()
        src_state = {'renamed_a.lin.weight': src.a.lin.weight,
                     'b.lin.weight': src.b.lin.weight}
        dst = _Model()
        result = load_partial_state_dict(
            dst, src_state, key_map={'a': 'renamed_a'},
        )
        assert result['load_ratio'] == 1.0
        assert torch.equal(dst.a.lin.weight, src.a.lin.weight)
        assert torch.equal(dst.b.lin.weight, src.b.lin.weight)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
