"""事前学習モデル→論理ゲートNN変換(ROADMAP-v5 Track D)。

パラダイム:
    C1 直接転写(transcribe)   — 疎化+真理値表列挙(training-free)
    C2 二値化lowering(binarize) — PTB→XNOR-popcountの厳密ゲート化
    C3 回復学習(recover)       — 転写IR→DLGN再構築→微調整
    C4 蒸留(distill)           — teacherロジット/スコアからの学習
    C5 ハイブリッド置換(progressive) — hybrid/replace.py の薄いオーケストレーション
"""

from .binarize import (
    binarize_weights,
    lower_bnn_mlp,
    popcount_threshold_nodes,
)
from .cache import (
    PackedDistillDataset,
    get_or_compute_logits,
    teacher_cache_key,
)
from .distill import (
    kd_kl_loss,
    kd_mse_loss,
    make_distill_task,
    make_kd_loss,
    pack_targets,
    standardize_logits,
)
from .pipeline import (
    PIPELINE_PHASES,
    ConvertPipelineSpec,
    convert_pretrained_mlp,
)
from .prune import (
    PruneSchedule,
    apply_fanin_pruning,
    fanin_mask,
    max_fanin,
)
from .recover import (
    RECOVER_PARAMETERIZATIONS,
    RecoverConfig,
    rebuild_trainable,
    recover,
)
from .teacher import (
    TEACHER_TRANSFORMS,
    OnnxTeacher,
    Teacher,
    TorchTeacher,
    resolve_teacher,
)
from .transcribe import (
    TranscribeSpec,
    enumerate_truth_table,
    isf_truth_tables,
    sparsify_fanin,
    threshold_truth_tables,
    transcribe_conv2d,
    transcribe_mlp,
)
from .transcribe_rnn import transcribe_rnn_cell
from .transcribe_seq import (
    FfnTranscribeSpec,
    extract_ffn_linears,
    init_logic_mlp_from_ffn,
    transcribe_ffn_block,
)
from .widen import (
    INIT_NEW_FINAL_POLICIES,
    widen_native_ir,
)

__all__ = [
    'PIPELINE_PHASES',
    'ConvertPipelineSpec',
    'FfnTranscribeSpec',
    'INIT_NEW_FINAL_POLICIES',
    'PackedDistillDataset',
    'PruneSchedule',
    'RECOVER_PARAMETERIZATIONS',
    'RecoverConfig',
    'TEACHER_TRANSFORMS',
    'OnnxTeacher',
    'Teacher',
    'TorchTeacher',
    'TranscribeSpec',
    'apply_fanin_pruning',
    'binarize_weights',
    'convert_pretrained_mlp',
    'enumerate_truth_table',
    'extract_ffn_linears',
    'fanin_mask',
    'get_or_compute_logits',
    'init_logic_mlp_from_ffn',
    'isf_truth_tables',
    'kd_kl_loss',
    'kd_mse_loss',
    'lower_bnn_mlp',
    'make_distill_task',
    'make_kd_loss',
    'max_fanin',
    'pack_targets',
    'popcount_threshold_nodes',
    'rebuild_trainable',
    'recover',
    'resolve_teacher',
    'sparsify_fanin',
    'standardize_logits',
    'teacher_cache_key',
    'threshold_truth_tables',
    'transcribe_conv2d',
    'transcribe_ffn_block',
    'transcribe_mlp',
    'transcribe_rnn_cell',
    'widen_native_ir',
]
