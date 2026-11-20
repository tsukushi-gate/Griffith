"""学習1ステップの抽象(Stepper、TaskSpec注入)。

3手法(BP/AP法/BinaryBrain風)を同一インターフェースで扱う:
  - BPStepper: TaskSpecの損失/メトリクスで学習(既定=classification+CE+Adam、従来と同一)
  - BBStepper: BP共通+raw層の[0,1]クランプ(BinaryBrain風)
  - APStepper: APTrainerサブクラスで更新量delta(.gradが存在しない)を捕捉。
    AP法は出力誤差の定義がクラス分類前提のため classification タスク専用
"""

from __future__ import annotations

import torch

from griffith_logic import LutLayer, split_param_groups
from griffith_logic.training.rules.ap import APTrainer
from griffith_logic.training.tasks import TaskSpec, make_task

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}


def _make_optimizer(name: str, params, lr: float):
    if name not in OPTIMIZERS:
        raise ValueError(f'未対応のoptimizer: {name}(対応: {sorted(OPTIMIZERS)})')
    return OPTIMIZERS[name](params, lr=lr)


class BPStepper:
    """通常の誤差逆伝播。TaskSpec(損失/メトリクス)は注入可能(既定=CE分類)。"""

    method = 'bp'

    def __init__(self, model: torch.nn.Module, lr: float,
                 task: TaskSpec | None = None, optimizer_name: str = 'adam',
                 logic_lr: float | None = None, ann_lr: float | None = None):
        self.model = model
        self.task = task or make_task('classification')
        params = (
            split_param_groups(
                model,
                logic_lr=logic_lr if logic_lr is not None else lr,
                ann_lr=ann_lr if ann_lr is not None else lr,
            )
            if logic_lr is not None or ann_lr is not None
            else model.parameters()
        )
        self.optimizer = _make_optimizer(optimizer_name, params, lr)
        self.last_metrics: dict[str, float] = {}

    def step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        output = self.model(x)
        loss = self.task.loss_fn(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_metrics = self.task.metrics(output.detach(), y)
        return loss.item(), self.last_metrics['acc']


class BBStepper(BPStepper):
    """BinaryBrain風: BP共通+raw層のweightsを毎step[0,1]へクランプ。"""

    method = 'bb'

    def __init__(self, model: torch.nn.Module, lr: float,
                 task: TaskSpec | None = None, optimizer_name: str = 'adam',
                 logic_lr: float | None = None, ann_lr: float | None = None):
        super().__init__(model, lr, task, optimizer_name, logic_lr, ann_lr)
        self._raw_layers = [
            m for m in model.modules()
            if isinstance(m, LutLayer) and m.parameterization == 'raw'
        ]

    def step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        loss, acc = super().step(x, y)
        with torch.no_grad():
            for layer in self._raw_layers:
                layer.weights.clamp_(0.0, 1.0)
        return loss, acc


class _CapturingAPTrainer(APTrainer):
    """_update_weightsをオーバーライドして更新量deltaを捕捉するAPTrainer。

    AP法は.gradを使わない(torch.no_grad()内で直接更新)ため、
    更新前後の差分としてdeltaを取り出す(clamp[0,1]の効果込み)。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_deltas: dict[int, torch.Tensor] = {}  # id(layer) → delta

    def _update_weights(self, cache, error):
        before = cache.layer.weights.data.clone()
        super()._update_weights(cache, error)
        self.last_deltas[id(cache.layer)] = (
            cache.layer.weights.data - before
        ).detach().cpu()


class APStepper:
    """AP法(Akiho Propagation)。optimizer/gradは存在しない。"""

    method = 'ap'
    optimizer = None  # keyframeのopt_state保存はスキップされる

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        lut_layers: list[tuple[str, object]],
        task: TaskSpec | None = None,
    ):
        # AP法は出力誤差=one_hot-softmax(クラス分類)前提。他タスクは未対応
        if task is not None and task.kind != 'classification':
            raise ValueError(
                f'AP法はclassificationタスク専用({task.kind}は未対応)'
            )
        self.trainer = _CapturingAPTrainer(model, lr=lr)
        self._name_by_layer = {id(layer): name for name, layer in lut_layers}
        self.last_metrics: dict[str, float] = {}

    def step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        metrics = self.trainer.train_step(x, y)
        self.last_metrics = {'acc': float(metrics['accuracy'])}
        return float(metrics['loss']), float(metrics['accuracy'])

    @property
    def last_deltas_by_name(self) -> dict[str, torch.Tensor]:
        """直近stepの更新量(IRノードid → deltaテンソル)。"""
        return {
            self._name_by_layer[lid]: delta
            for lid, delta in self.trainer.last_deltas.items()
            if lid in self._name_by_layer
        }


def make_stepper(
    method: str,
    model: torch.nn.Module,
    lr: float,
    lut_layers: list[tuple[str, object]],
    task: TaskSpec | None = None,
    optimizer_name: str = 'adam',
    logic_lr: float | None = None,
    ann_lr: float | None = None,
) -> BPStepper | BBStepper | APStepper:
    if method == 'bp':
        return BPStepper(model, lr, task, optimizer_name, logic_lr, ann_lr)
    if method == 'bb':
        return BBStepper(model, lr, task, optimizer_name, logic_lr, ann_lr)
    if method == 'ap':
        return APStepper(model, lr, lut_layers, task)
    raise ValueError(f'未対応の学習手法: {method}')
