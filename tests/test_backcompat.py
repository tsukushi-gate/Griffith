"""後方互換テスト(M3パッケージ再編のセーフティネット)。

再編(フラット構造→サブパッケージ+シム)の前後で、
既存の全公開インポートパスと保存済みチェックポイントが
無変更で動作し続けることを保証する。
"""

import glob
import importlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')

# ============================================================
# 公開インポートパス一覧
# ============================================================

# トップレベル(griffith_logic/__init__.py の再エクスポート)
TOP_LEVEL_SYMBOLS = [
    'LutLayer', 'GroupSum',
    'get_random_connections', 'get_unique_connections',
    'LutConv2d', 'OrPooling2d', 'SpatialGroupSum',
    'RDDLGN', 'RDDLGNConfig', 'RDDLGNEncoder', 'RDDLGNDecoder',
    'LogicEmbedding', 'LutStack',
    'Binarize', 'RealToBinary', 'BinaryToReal',
    'DifferentiableLutBlock', 'UpSample2d', 'BinaryBrainConv2d',
    'DLGNBooleanAttention', 'BooleanAttentionConfig',
    'DLGNMultiQueryAttention', 'MultiQueryAttentionConfig',
    'DLGNPositionalAttention', 'PositionalAttentionConfig',
    'DLGNAttentionSeq2Seq', 'AttentionSeq2SeqConfig',
    'DLGNPointerAttention', 'PointerAttentionConfig',
    'DLGNAttentionTranslator', 'TranslatorConfig',
]

# 深いインポートパス(experiments/tests/visualizerが使用する形式)
DEEP_IMPORTS = {
    'griffith_logic.lut_layer': [
        'LutLayer', 'GradFactor',
        'LutFunction', 'LutFunctionHardST', 'LutFunctionIndependent',
        'LutFunctionRaw', 'LutFunctionIWP', 'LutFunctionWARP',
        'CudaLutFunction', 'CudaLutFunctionIndependent',
        '_compute_basis_butterfly', '_compute_wh_basis_butterfly',
        '_build_gate_table_n2',
    ],
    'griffith_logic.group_sum': ['GroupSum'],
    'griffith_logic.connections': [
        'get_random_connections', 'get_unique_connections', 'build_reverse_index',
    ],
    'griffith_logic.conv': ['LutConv2d'],
    'griffith_logic.conv_hard_st': [
        'LutConv2dHardST', 'TreeHardSTFunction', 'CAGEConvScheduler',
    ],
    'griffith_logic.pooling': ['OrPooling2d'],
    'griffith_logic.spatial_group_sum': ['SpatialGroupSum'],
    'griffith_logic.upsample': ['UpSample2d'],
    'griffith_logic.rddlgn': [
        'RDDLGN', 'RDDLGNConfig', 'RDDLGNEncoder', 'RDDLGNDecoder',
        'LogicEmbedding', 'LutStack', '_init_hidden',
    ],
    # 現在の正規パス(こちらも契約として固定)
    'griffith_logic.models.rddlgn': ['RDDLGN', 'LutStack', '_init_hidden'],
    'griffith_logic.models.attention': [
        'DLGNBooleanAttention', 'DLGNAttentionTranslator',
    ],
    'griffith_logic.training.rules.ap': ['APTrainer', 'LayerCache'],
    'griffith_logic.core.reference': ['reference_lut_forward'],
    'griffith_logic.layers.lut': ['LutLayer'],
    'griffith_logic.binarize': ['Binarize', 'BinarizeFunction'],
    'griffith_logic.binary_modulation': ['RealToBinary', 'BinaryToReal'],
    'griffith_logic.differentiable_lut_block': [
        'DifferentiableLutBlock', 'UnbinarizeFunction',
    ],
    'griffith_logic.bb_conv2d': ['BinaryBrainConv2d'],
    'griffith_logic.boolean_attention': [
        'DLGNBooleanAttention', 'BooleanAttentionConfig',
    ],
    'griffith_logic.multi_query_attention': [
        'DLGNMultiQueryAttention', 'MultiQueryAttentionConfig',
    ],
    'griffith_logic.positional_attention': [
        'DLGNPositionalAttention', 'PositionalAttentionConfig',
    ],
    'griffith_logic.pointer_attention': [
        'DLGNPointerAttention', 'PointerAttentionConfig',
    ],
    'griffith_logic.attention_seq2seq': [
        'DLGNAttentionSeq2Seq', 'AttentionSeq2SeqConfig',
    ],
    'griffith_logic.attention_translator': [
        'DLGNAttentionTranslator', 'TranslatorConfig',
    ],
    'griffith_logic.cuda': ['cuda_available', 'get_cuda_module'],
}

# WIPモジュール(ファイルが存在する場合のみ検証)
OPTIONAL_IMPORTS = {
    'griffith_logic.eggroll': ['EGGROLLConfig', 'EGGROLLOptimizer'],
    'griffith_logic.eggroll_lut': [
        'hw_efficient_linear_batched', 'naive_linear_batched',
    ],
}


class TestPublicImports:
    def test_top_level_symbols(self):
        """`from griffith_logic import X` 形式の全シンボル"""
        pkg = importlib.import_module('griffith_logic')
        missing = [s for s in TOP_LEVEL_SYMBOLS if not hasattr(pkg, s)]
        assert not missing, f'トップレベルに存在しないシンボル: {missing}'

    @pytest.mark.parametrize('module_name', sorted(DEEP_IMPORTS))
    def test_deep_imports(self, module_name):
        """`from griffith_logic.xxx import Y` 形式の全パス"""
        mod = importlib.import_module(module_name)
        missing = [s for s in DEEP_IMPORTS[module_name] if not hasattr(mod, s)]
        assert not missing, f'{module_name} に存在しないシンボル: {missing}'

    @pytest.mark.parametrize('module_name', sorted(OPTIONAL_IMPORTS))
    def test_optional_imports(self, module_name):
        """WIPモジュール: ファイルが存在するならimport可能であること"""
        rel = module_name.replace('.', '/') + '.py'
        if not os.path.exists(os.path.join(REPO_ROOT, rel)):
            pytest.skip(f'{rel} が存在しない')
        mod = importlib.import_module(module_name)
        missing = [s for s in OPTIONAL_IMPORTS[module_name] if not hasattr(mod, s)]
        assert not missing, f'{module_name} に存在しないシンボル: {missing}'

    def test_experiments_ap_trainer(self):
        """experiments/ap_trainer.py のインポート(sys.path挿入方式)"""
        exp_dir = os.path.join(REPO_ROOT, 'experiments')
        sys.path.insert(0, exp_dir)
        try:
            import ap_trainer
            assert hasattr(ap_trainer, 'APTrainer')
            assert hasattr(ap_trainer, 'LayerCache')
        finally:
            sys.path.remove(exp_dir)


class TestCheckpointCompat:
    def _find_checkpoint(self):
        pattern = os.path.join(REPO_ROOT, 'checkpoints', '*', 'best.pt')
        paths = sorted(glob.glob(pattern))
        if not paths:
            pytest.skip('ローカルにチェックポイントが存在しない(CI等)')
        return paths[0]

    def test_existing_checkpoint_loads(self):
        """既存チェックポイントのpickleが読み込めること(再編後も維持)"""
        path = self._find_checkpoint()
        data = torch.load(path, map_location='cpu', weights_only=False)
        assert 'model_state_dict' in data
        assert isinstance(data['model_state_dict'], dict)

    def test_state_dict_roundtrip(self):
        """LutLayerのstate_dictがクラスパス非依存でround-tripすること"""
        layer = LutLayerFactory()
        sd = layer.state_dict()
        layer2 = LutLayerFactory()
        layer2.load_state_dict(sd)
        x = torch.rand(2, 8)
        layer.eval(), layer2.eval()
        assert torch.equal(layer(x), layer2(x))


def LutLayerFactory():
    from griffith_logic import LutLayer
    torch.manual_seed(7)
    return LutLayer(8, 4, n_inputs=2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
