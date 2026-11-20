"""学習タスク抽象と損失レジストリ。

エディタ/ランナーの「MNIST+CrossEntropy固定」を解消し、
タスク種別(分類 / 二値ベクトル)ごとに損失とメトリクスを束ねる。

タスク種別と終端ノードの整合:
  - classification: GroupSum終端(クラススコア)を前提。損失は ce
  - binary_vector: lut_layer / binarize 終端([0,1]域の生ビット出力)。損失は bce/mse。
    メトリクスは bit accuracy(主表示)と exact match(全ビット一致率)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

TASK_KINDS = ('classification', 'binary_vector')
LOSS_NAMES = ('ce', 'bce', 'mse')

# BCEのlog(0)防止クランプ(LUTのsoft出力は[0,1]丁度に張り付くことがある)
_BCE_EPS = 1e-6


def _ce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(output, target)


def _bce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(
        output.clamp(_BCE_EPS, 1.0 - _BCE_EPS), target.to(output.dtype)
    )


def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, target.to(output.dtype))


LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    'ce': _ce_loss,
    'bce': _bce_loss,
    'mse': _mse_loss,
}


def _classification_acc(output: torch.Tensor, target: torch.Tensor) -> float:
    return (output.argmax(dim=-1) == target).float().mean().item()


def _bit_acc(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = (output >= 0.5).to(target.dtype)
    return (pred == target).float().mean().item()


def _exact_match(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = (output >= 0.5).to(target.dtype)
    return (pred == target).all(dim=-1).float().mean().item()


@dataclass
class TaskSpec:
    """タスク種別+損失+メトリクスの束。Stepper/ランナーへ注入する。"""

    kind: str  # TASK_KINDS のいずれか
    loss_name: str  # LOSS_NAMES のいずれか
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = field(repr=False)
    # 主メトリクス(エディタの acc 表示に対応)
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float] = field(repr=False)
    # 追加メトリクス(名前→関数)。binary_vectorのexact_match等
    extra_metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = field(
        default_factory=dict, repr=False,
    )

    def metrics(self, output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """主+追加メトリクスをまとめて計算する。主メトリクスのキーは 'acc'。"""
        out = {'acc': self.metric_fn(output, target)}
        for name, fn in self.extra_metrics.items():
            out[name] = fn(output, target)
        return out


def make_task(kind: str, loss: str | None = None) -> TaskSpec:
    """タスク種別と損失名からTaskSpecを構築する(不整合はValueError)。

    lossを省略した場合の既定: classification→ce / binary_vector→bce
    """
    if kind not in TASK_KINDS:
        raise ValueError(f'未対応のタスク種別: {kind}(対応: {TASK_KINDS})')

    if kind == 'classification':
        loss = loss or 'ce'
        if loss != 'ce':
            raise ValueError(
                f'classificationタスクの損失はceのみ対応: {loss}'
            )
        return TaskSpec(
            kind=kind, loss_name=loss,
            loss_fn=LOSSES['ce'], metric_fn=_classification_acc,
        )

    # binary_vector
    loss = loss or 'bce'
    if loss not in ('bce', 'mse'):
        raise ValueError(
            f'binary_vectorタスクの損失はbce/mseのみ対応: {loss}'
        )
    return TaskSpec(
        kind=kind, loss_name=loss,
        loss_fn=LOSSES[loss], metric_fn=_bit_acc,
        extra_metrics={'exact_match': _exact_match},
    )
