"""抽出器カバレッジの機械検査。

griffith_logicの公開nn.Module型(遅延インポートテーブル
_SYMBOL_MODULES から列挙)それぞれが、
    (1) EXTRACTORS/NATIVE_EXTRACTORSに登録あり(MRO走査、find_extractorと同一規則)
    (2) export/extract.pyの方針表に明示
        BLACKBOX_TYPES   — softmax等を含みHDL化不能(blocks_to_irがblackbox化)
        MODEL_TYPES      — 完成モデル/ハイブリッド複合(抽出対象は論理部分のみ)
        BOUNDARY_TYPES   — 実数⇔論理の境界(encoding系。ポートencodingの領分)
        STRUCTURAL_TYPES — グラフ構造部品(blocks_to_irが直接処理)
のどちらかを満たすことを検査する。**未分類の新型を公開すると
test_all_public_modules_classified が落ちる**(分類漏れの防止装置)。
方針表の名前が実在しない(リネーム・削除済み)場合も検出する。
"""

import inspect
import os
import sys

import pytest
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import griffith_logic
from griffith_logic.export import lowering  # noqa: F401 抽出器登録の副作用
from griffith_logic.export.extract import (
    BLACKBOX_TYPES,
    BOUNDARY_TYPES,
    EXTRACTORS,
    MODEL_TYPES,
    NATIVE_EXTRACTORS,
    STRUCTURAL_TYPES,
)

POLICY_TABLES = {
    'BLACKBOX_TYPES': BLACKBOX_TYPES,
    'MODEL_TYPES': MODEL_TYPES,
    'BOUNDARY_TYPES': BOUNDARY_TYPES,
    'STRUCTURAL_TYPES': STRUCTURAL_TYPES,
}
ALL_POLICY_NAMES = tuple(
    name for table in POLICY_TABLES.values() for name in table
)


def _public_module_types() -> dict[str, type]:
    """遅延インポートテーブルから公開nn.Module型を列挙する"""
    out = {}
    for name in griffith_logic._SYMBOL_MODULES:
        obj = getattr(griffith_logic, name)
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            out[name] = obj
    return out


def _has_extractor(cls: type) -> bool:
    """find_extractorと同じMRO規則で登録の有無を判定する"""
    return any(c in EXTRACTORS or c in NATIVE_EXTRACTORS
               for c in cls.__mro__)


def _in_policy(cls: type) -> bool:
    """方針表への記載をMRO込みで判定する(サブクラスは親の分類を継承)"""
    return any(c.__name__ in ALL_POLICY_NAMES for c in cls.__mro__)


class TestExtractorCoverageMatrix:
    def test_all_public_modules_classified(self):
        """全公開nn.Module型が「抽出器あり」or「方針表に明示」のいずれか"""
        unclassified = sorted(
            name for name, cls in _public_module_types().items()
            if not _has_extractor(cls) and not _in_policy(cls)
        )
        assert not unclassified, (
            f'抽出方針が未分類の公開nn.Module型: {unclassified}\n'
            'export/extract.py にregister_extractorで抽出器を追加するか、'
            '方針表(BLACKBOX/MODEL/BOUNDARY/STRUCTURAL_TYPES)へ'
            '理由コメント付きで追記すること'
        )

    def test_policy_tables_have_no_stale_names(self):
        """方針表の名前が実在の公開nn.Module型であること(リネーム検出)"""
        known = set(_public_module_types())
        stale = {
            table_name: [n for n in table if n not in known]
            for table_name, table in POLICY_TABLES.items()
        }
        stale = {k: v for k, v in stale.items() if v}
        assert not stale, (
            f'方針表に実在しない型名が残っている: {stale}\n'
            '(リネーム・削除に追従して export/extract.py の方針表を更新すること)'
        )

    def test_no_duplicate_classification(self):
        """方針表間の重複なし+抽出器登録済み型が方針表に重複記載されていない"""
        assert len(ALL_POLICY_NAMES) == len(set(ALL_POLICY_NAMES)), (
            '方針表間で型名が重複している'
        )
        doubled = sorted(
            name for name, cls in _public_module_types().items()
            if _has_extractor(cls) and cls.__name__ in ALL_POLICY_NAMES
        )
        assert not doubled, (
            f'抽出器登録済みなのに方針表にも記載されている型: {doubled}'
        )

    def test_registered_extractors_cover_core_types(self):
        """回帰: 中核型の抽出器登録が失われていない(導入時のスナップショット)"""
        core = ('LutLayer', 'GroupSum', 'Binarize', 'LutConv2d', 'OrPooling2d',
                'SpatialGroupSum', 'DifferentiableLutBlock',
                'DWNLayer', 'LearnableMapping',       #
                'UpSample2d', 'BinaryBrainConv2d')    #
        types = _public_module_types()
        missing = sorted(n for n in core if not _has_extractor(types[n]))
        assert not missing, f'抽出器登録が失われた中核型: {missing}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
