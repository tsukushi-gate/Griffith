"""学習アルゴリズム指定API。

スライドの記法をそのまま採用する:

    self.conv1 = with_learning(nn.Conv2d(1, 16, 3), LearningSpec(rule='bp'))
    self.fc1 = with_learning(LutLayer(...), LearningSpec(rule='ap'))

with_learning はモジュールを**ラップせず**、属性 `_learning_spec` を
注釈するだけ。forward・repr・state_dict は一切変わらない。
注釈のないモジュールは 'bp'(通常のautograd)として扱われる。
"""

from dataclasses import dataclass, field

from ..eggroll import EGGROLLConfig  # noqa: F401  (re-export用)


def _default_eggroll():
    return None


@dataclass
class LearningSpec:
    """レイヤ/モジュール単位の学習ルール指定。

    Attributes:
        rule: 'bp' | 'ap' | 'dfa' | 'eggroll'
        lr: ルール固有の学習率(Noneならトレーナ/optimizerの既定)
        update_mode: AP法(Akiho Propagation)固有(1=全重み更新, 2=最大basis入力のみ)
        feedback_init: DFA固有。固定ランダムフィードバック行列の初期化
        block_upstream: DFA固有。現実装はTrue(上流へ勾配を流さない)のみ対応
        eggroll: EGGROLL固有のハイパーパラメータ(NoneならEGGROLLConfig())
    """

    rule: str = 'bp'
    lr: float | None = None
    # AP固有
    update_mode: int = 1
    # DFA固有
    feedback_init: str = 'gaussian'
    block_upstream: bool = True
    # EGGROLL固有
    eggroll: 'EGGROLLConfig | None' = field(default_factory=_default_eggroll)

    RULES = ('bp', 'ap', 'dfa', 'eggroll')

    def __post_init__(self):
        assert self.rule in self.RULES, f'rule は {self.RULES} のいずれか: {self.rule}'


def with_learning(module, spec: LearningSpec):
    """モジュールに学習ルールを注釈して返す(ラップしない)。"""
    module._learning_spec = spec
    return module


def get_learning_spec(module) -> LearningSpec | None:
    """モジュールの学習ルール注釈を取得(未注釈ならNone)。"""
    return getattr(module, '_learning_spec', None)
