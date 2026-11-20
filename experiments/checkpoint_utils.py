"""後方互換シム: 正本は `griffith_logic.training.checkpoint_utils`。

パッケージ自立化(pip install した環境でリポジトリ相対パスに依存しない)のため、
実装は `griffith_logic/training/checkpoint_utils.py` へ移設した。既存の実験
スクリプト(`from checkpoint_utils import save_checkpoint` 等)をそのまま動かすため、
このモジュールは全公開シンボルを再exportするだけにする。

新規コードでは `from griffith_logic.training.checkpoint_utils import ...` を使うこと。
"""

# 再exportシムなのでstar importとその由来不明警告は無効化する
# ruff: noqa: F403, F405

from griffith_logic.training.checkpoint_utils import *  # noqa: F401,F403
from griffith_logic.training.checkpoint_utils import (  # noqa: F401
    _RUN_DIR_TIMESTAMP_RE,
    _metadata_mismatch,
    _normalize,
    _remap_key,
    _restore_dynamic_buffers,
    _weights_only,
)

__all__ = [
    'JST',
    'UNSAFE_CKPT_ENV',
    'checkpoint_paths',
    'find_resumable_checkpoint',
    'load_checkpoint',
    'load_partial_state_dict',
    'save_checkpoint',
    'setup_checkpoint_dir',
    'torch_load',
]
