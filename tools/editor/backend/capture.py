"""capture level判定とスナップショット構築。

capture level(モデル規模から自動決定、REST/起動パラメータで上書き可):
  - full:    weights毎step + grad/trace/act_mean(小規模モデル既定)
  - weights: weightsのみ、stride間引き
  - metrics: スナップショットなし(keyframeのみ)
"""

from __future__ import annotations

import time

import torch

from .history import LayerRecord, StepSnapshot

CAPTURE_FULL_MAX_BYTES = 8 * 2**20      # 総weights ≤8MB → full
CAPTURE_WEIGHTS_MAX_BYTES = 64 * 2**20  # 総weights ≤64MB → weights
WEIGHTS_LEVEL_STRIDE = 4                # weightsレベル時の既定stride

CAPTURE_LEVELS = ('full', 'weights', 'metrics')


def total_weights_bytes(lut_layers: list[tuple[str, object]]) -> int:
    """全LutLayerの学習パラメータ総バイト数。"""
    total = 0
    for _, layer in lut_layers:
        w = layer.weights
        total += w.numel() * w.element_size()
    return total


def decide_capture_level(lut_layers: list[tuple[str, object]]) -> tuple[str, int]:
    """モデル規模から (capture level, stride) を自動決定する。"""
    nbytes = total_weights_bytes(lut_layers)
    if nbytes <= CAPTURE_FULL_MAX_BYTES:
        return 'full', 1
    if nbytes <= CAPTURE_WEIGHTS_MAX_BYTES:
        return 'weights', WEIGHTS_LEVEL_STRIDE
    return 'metrics', 0


def snapshot_bytes_estimate(lut_layers: list[tuple[str, object]], level: str) -> int:
    """1スナップショットの概算バイト数(capture_configフレーム用)。"""
    w = total_weights_bytes(lut_layers)
    if level == 'full':
        return w + w // 2  # weights f32 + grad f16
    if level == 'weights':
        return w
    return 0


def build_snapshot(
    *,
    step: int,
    epoch: int,
    loss: float,
    acc: float,
    method: str,
    lut_layers: list[tuple[str, object]],
    level: str,
    grads: dict[str, torch.Tensor] | None = None,
    deltas: dict[str, torch.Tensor] | None = None,
    act_means: dict[str, torch.Tensor] | None = None,
    trace: dict | None = None,
    keyframe: bool = False,
    opt_state: dict | None = None,
) -> StepSnapshot | None:
    """現在のモデル状態からスナップショットを構築する(metricsレベルはkeyframeのみ)。"""
    if level == 'metrics' and not keyframe:
        return None

    layers: dict[str, LayerRecord] = {}
    with torch.no_grad():
        for name, layer in lut_layers:
            grad = grads.get(name) if grads else None
            delta = deltas.get(name) if deltas else None
            act = act_means.get(name) if act_means else None
            layers[name] = LayerRecord(
                weights=layer.weights.detach().to('cpu', torch.float32).clone(),
                grad=grad.detach().to('cpu', torch.float16).clone() if grad is not None else None,
                delta=delta.detach().to('cpu', torch.float16).clone() if delta is not None else None,
                act_mean=act.detach().to('cpu', torch.float16).clone() if act is not None else None,
                indices=layer.indices.detach().cpu().clone() if keyframe else None,
            )

    return StepSnapshot(
        step=step,
        epoch=epoch,
        loss=loss,
        acc=acc,
        ts=time.time(),
        method=method,
        layers=layers,
        trace=trace if level == 'full' else None,
        opt_state=opt_state if keyframe else None,
        rng=torch.get_rng_state().clone() if keyframe else None,
    )
