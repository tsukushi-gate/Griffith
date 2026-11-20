"""学習アルゴリズムサブパッケージ(M5)。

    LearningSpec / with_learning — レイヤ単位の学習ルール注釈
    HybridTrainer                — bp/ap/dfa/eggroll混在モデルのトレーナ
"""

from .gpu_accel import make_train_step, wants_capturable
from .spec import LearningSpec, get_learning_spec, with_learning
from .trainer import HybridTrainer

__all__ = [
    'HybridTrainer',
    'LearningSpec',
    'get_learning_spec',
    'make_train_step',
    'wants_capturable',
    'with_learning',
]
