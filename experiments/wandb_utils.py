"""後方互換シム: 正本は `griffith_logic.training.wandb_utils`。

パッケージ自立化(pip install した環境でリポジトリ相対パスに依存しない)のため、
実装は `griffith_logic/training/wandb_utils.py` へ移設した。既存の実験スクリプト
(`from wandb_utils import init_wandb` / `from experiments.wandb_utils import ...`)を
そのまま動かすため、このモジュールは全公開シンボルを再exportするだけにする。

新規コードでは `from griffith_logic.training.wandb_utils import ...` を使うこと。
"""

# 再exportシムなのでstar importとその由来不明警告は無効化する
# ruff: noqa: F403, F405

from griffith_logic.training.wandb_utils import *  # noqa: F401,F403

# 実験スクリプト・テスト・scripts/ から参照される内部シンボルも明示的に再exportする
# (star importはアンダースコア始まりを持ち込まないため)
from griffith_logic.training.wandb_utils import (  # noqa: F401
    _FINISHED_RUN_IDS,
    _LEDGER_METRIC_PRIORITY,
    HAS_WANDB,
    _git_short_hash,
    _ledger_enabled,
    _ledger_escape,
    _numeric_summary,
    _pick_ledger_metric,
    _sanitize_artifact_name,
)

__all__ = [
    'HAS_WANDB',
    'JST',
    'LEDGER_HEADER',
    'LEDGER_PATH',
    'LOSS_LOG_INTERVAL_SEC',
    'LossLogPacer',
    'add_experiment_args',
    'append_ledger_row',
    'artifact_policy',
    'count_logic_gates',
    'finish',
    'finish_wandb',
    'format_ledger_row',
    'get_device_info',
    'get_model_param_info',
    'init_wandb',
    'log_artifact',
    'log_checkpoint_artifact',
    'log_images',
    'log_ir_artifact',
    'log_metrics',
    'log_model_checkpoint',
    'normalize_tags',
    'progress_metrics',
    'should_use_wandb',
    'use_checkpoint',
]
