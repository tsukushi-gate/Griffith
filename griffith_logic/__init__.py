"""Griffith: a differentiable logic-gate neural network framework for hardware-bound AI.

Griffith(名はグリフィス天文台に由来)は、N入力LUTに対応した微分可能論理ゲートNNの
学習フレームワーク。PyTorchで学習した回路をIR(中間表現)経由でVerilog/VHDLへ
書き出し、GPUに依存しない推論ハードウェアへ落とすことを目的とする。
配布名は `griffith-logic`、import 名は `griffith_logic`(慣用別名 `gl`)。
バージョンは `griffith_logic.__version__`(インストール済みメタデータ由来)。

公開シンボルは遅延インポート(PEP 562)。`import griffith_logic as gl` 自体は
軽量で、属性アクセス時に初めて対応モジュールを読み込む。
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version('griffith-logic')
except PackageNotFoundError:  # 未インストール(ソースツリー直参照)の場合
    __version__ = '0.0.0+unknown'

# 公開シンボル → 定義モジュール(再編後の正規パス)
_SYMBOL_MODULES = {
    # コア層
    'LutLayer': '.layers.lut',
    'LTTLayer': '.layers.ltt',
    'FixedLutLayer': '.layers.fixed_lut',
    'Permute': '.layers.permute',
    'GroupSum': '.group_sum',
    'get_random_connections': '.core.connections',
    'get_unique_connections': '.core.connections',
    'get_rot_connections': '.core.connections',
    'get_permutation_connections': '.core.connections',
    'get_stretched_rot_connections': '.core.connections',
    'get_uniform_random_connections': '.core.connections',
    'connection_stats': '.core.connections',
    'sort_model_wiring': '.core.sort_wiring',
    'UnsupportedWiringError': '.core.sort_wiring',
    'get_rot_n_connections': '.core.connections',
    'get_butterfly_connections': '.core.connections',
    'get_local1d_connections': '.core.connections',
    'get_local2d_connections': '.core.connections',
    'butterfly_strides': '.core.connections',
    'local1d_strides': '.core.connections',
    'ring_offsets': '.core.connections',
    'local2d_offset_schedule': '.core.connections',
    'dedupe_connections': '.core.connections',
    'prop_map': '.core.connections',
    # 配線の受容野・被覆率(学習前に配線だけから決まる量)
    'ImageLayout': '.analysis.wiring_stats',
    'layer_stats': '.analysis.wiring_stats',
    'class_receptive_field': '.analysis.wiring_stats',
    'receptive_field_upper_bound': '.analysis.wiring_stats',
    'wiring_summary': '.analysis.wiring_stats',
    # 畳み込み・空間処理
    'LutConv2d': '.conv',
    'ConvDLGNConfig': '.models.conv_dlgn',
    'build_conv_dlgn': '.models.conv_dlgn',
    'OrPooling2d': '.pooling',
    'SpatialGroupSum': '.spatial_group_sum',
    'UpSample2d': '.upsample',
    # seq2seq
    'RDDLGN': '.models.rddlgn',
    'RDDLGNConfig': '.models.rddlgn',
    'RDDLGNEncoder': '.models.rddlgn',
    'RDDLGNDecoder': '.models.rddlgn',
    'RDDLGNClassifier': '.models.rddlgn',
    'LogicEmbedding': '.models.rddlgn',
    'LutStack': '.models.rddlgn',
    # decoder-only 因果Transformer
    'LogicGPT': '.models.logic_gpt',
    'LogicGPTConfig': '.models.logic_gpt',
    # 論理関数データセット
    'truth_table_dataset': '.datasets.logic',
    'LogicDatasetSpec': '.datasets.logic',
    'LOGIC_DATASETS': '.datasets.logic',
    # 視覚データセット中央ローダ
    'VISION_DATASETS': '.datasets.vision',
    'VisionDatasetSpec': '.datasets.vision',
    # UCI-HAR(スマホ加速度による行動認識)ローダ+温度計符号化
    'load_uci_har': '.datasets.har',
    'fit_thermometer_thresholds': '.datasets.har',
    'thermometer_encode': '.datasets.har',
    # 設定管理+共通ランナー
    'TrainConfig': '.training.config',
    'load_config': '.training.config',
    'dump_config': '.training.config',
    'save_config': '.training.config',
    'run_training': '.training.runner',
    'evaluate_checkpoint': '.training.runner',
    # 学習アルゴリズム(M5)
    'LearningSpec': '.training.spec',
    'with_learning': '.training.spec',
    'get_learning_spec': '.training.spec',
    'HybridTrainer': '.training.trainer',
    'make_train_step': '.training.gpu_accel',
    'wants_capturable': '.training.gpu_accel',
    # ハイブリッドAPI(M4)
    'LogicLinear': '.layers.logic_linear',
    'LogicMLP': '.layers.logic_linear',
    'LogicSpec': '.hybrid.policy',
    'ToLogic': '.encoding.bridges',
    'FromLogic': '.encoding.bridges',
    'replace_modules': '.hybrid.replace',
    'convert_linear_to_logic': '.hybrid.replace',
    'replace_mlp_with_lut': '.hybrid.replace',
    'split_param_groups': '.hybrid.replace',
    # エディタ連携(Python由来モデル登録)
    'EditorModelSpec': '.editor_models',
    'editor_model': '.editor_models',
    # DWN (Bacellar et al., ICML 2024)
    'ThermometerEncoder': '.encoding.thermometer',
    'DWNLayer': '.layers.dwn',
    'LearnableMapping': '.layers.dwn',
    'DWNClassifier': '.models.dwn',
    'LLViTConfig': '.models.llvit',
    'LLViT': '.models.llvit',
    'LLViTChannelMixer': '.models.llvit',
    'ConditionalSum': '.models.llvit',
    'build_llvit_cifar10': '.models.llvit',
    'build_deit_tiny_cifar10': '.models.llvit',
    'spectral_norm': '.regularization.spectral',
    # BinaryBrain互換
    'Binarize': '.binarize',
    'RealToBinary': '.encoding.modulation',
    'BinaryToReal': '.encoding.modulation',
    'DifferentiableLutBlock': '.differentiable_lut_block',
    'BinaryBrainConv2d': '.bb_conv2d',
    # 一般NN部品(A3)
    'Concat': '.blocks.structural',
    'Add': '.blocks.structural',
    'Residual': '.blocks.structural',
    'LogicLayerNorm': '.blocks.norm',
    'LogicBatchNorm': '.blocks.norm',
    'LogicDropout': '.blocks.dropout',
    'LogicPositionalEncoding': '.blocks.positional',
    'position_codes': '.blocks.positional',
    'LogicAttention': '.blocks.attention',
    'LogicTransformerBlock': '.blocks.transformer',
    'blocks_to_ir': '.blocks.to_ir',
    # Attention系
    'DLGNBooleanAttention': '.models.attention.boolean_attention',
    'BooleanAttentionConfig': '.models.attention.boolean_attention',
    'DLGNMultiQueryAttention': '.models.attention.multi_query_attention',
    'MultiQueryAttentionConfig': '.models.attention.multi_query_attention',
    'DLGNPositionalAttention': '.models.attention.positional_attention',
    'PositionalAttentionConfig': '.models.attention.positional_attention',
    'DLGNAttentionSeq2Seq': '.models.attention.attention_seq2seq',
    'AttentionSeq2SeqConfig': '.models.attention.attention_seq2seq',
    'DLGNPointerAttention': '.models.attention.pointer_attention',
    'PointerAttentionConfig': '.models.attention.pointer_attention',
    'DLGNAttentionTranslator': '.models.attention.attention_translator',
    'TranslatorConfig': '.models.attention.attention_translator',
}

__all__ = [*sorted(_SYMBOL_MODULES), '__version__']


def __getattr__(name: str):
    try:
        module_path = _SYMBOL_MODULES[name]
    except KeyError:
        raise AttributeError(
            f"module 'griffith_logic' has no attribute {name!r}"
        ) from None
    import importlib
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value  # 2回目以降は__getattr__を経由しない
    return value


def __dir__():
    return sorted(set(list(globals()) + list(_SYMBOL_MODULES)))
