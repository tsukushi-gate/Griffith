"""BP-free学習(AP/DFA)の収束・停滞・時間を測る共通メトリクスヘルパ。

想定用途:
    eval毎に ConvergenceTracker.record(step, metric) を呼び、以下を測る。
        - steps_to_threshold: metricが閾値に到達した最初のstep
          (BPCのように小さいほど良い場合はmode='min'、accのように大きいほど
          良い場合はmode='max')。
        - plateau_step: patience回連続で改善がmin_delta未満だった最初のstep
          (停滞検出=BP切替トリガの目安)。
        - best / best_step: 記録済みの最良metricとそのstep。

    wall_clock計測はWallClockのelapsed_sec()で行う(time.time()ベース)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ConvergenceTracker:
    """eval毎の (step, metric) 系列から収束・停滞を判定するトラッカー。

    mode: 'min'ならmetricが小さいほど良い(BPCなど)、'max'なら大きいほど良い
        (accuracyなど)。best()/best_step()/plateau_stepの判定に使う。
    """

    mode: str = 'min'
    steps: list[int] = field(default_factory=list)
    metrics: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

    def record(self, step: int, metric: float) -> None:
        """1回のeval結果を記録する。"""
        self.steps.append(step)
        self.metrics.append(float(metric))

    def steps_to_threshold(self, threshold: float, mode: str | None = None) -> int | None:
        """metricが閾値に初めて到達したstepを返す。未到達ならNone。

        Args:
            threshold: 閾値。
            mode: 'min'なら metric <= threshold で到達判定(BPCなど、
                小さいほど良い指標向け)。'max'なら metric >= threshold
                (accuracyなど、大きいほど良い指標向け)。省略時はself.modeを使う。
        """
        use_mode = mode if mode is not None else self.mode
        for step, metric in zip(self.steps, self.metrics):
            if use_mode == 'min':
                if metric <= threshold:
                    return step
            elif use_mode == 'max':
                if metric >= threshold:
                    return step
            else:
                raise ValueError(f"mode must be 'min' or 'max', got {use_mode!r}")
        return None

    def plateau_step(self, patience: int = 5, min_delta: float = 1e-3) -> int | None:
        """patience回連続で改善がmin_delta未満だった最初のstepを返す。

        「改善」はself.modeに応じて、直前までの最良値からの差分で判定する
        (min: best - metric、max: metric - best)。連続patience回、この改善幅が
        min_delta未満であれば「停滞」とみなし、その停滞区間の最初のstepを返す。
        記録数がpatience未満、または停滞が一度も起きなければNone。
        """
        if len(self.metrics) < patience:
            return None

        best_so_far = self.metrics[0]
        stagnant_run = 0
        stagnant_start_step = None

        for i in range(1, len(self.metrics)):
            metric = self.metrics[i]
            if self.mode == 'min':
                improvement = best_so_far - metric
            else:
                improvement = metric - best_so_far

            if improvement < min_delta:
                if stagnant_run == 0:
                    stagnant_start_step = self.steps[i]
                stagnant_run += 1
                if stagnant_run >= patience:
                    return stagnant_start_step
            else:
                stagnant_run = 0
                stagnant_start_step = None

            if self.mode == 'min':
                best_so_far = min(best_so_far, metric)
            else:
                best_so_far = max(best_so_far, metric)

        return None

    def best(self) -> float | None:
        """記録済みの最良metric値。未記録ならNone。"""
        if not self.metrics:
            return None
        return min(self.metrics) if self.mode == 'min' else max(self.metrics)

    def best_step(self) -> int | None:
        """最良metric値を達成した最初のstep。未記録ならNone。"""
        if not self.metrics:
            return None
        best_value = self.best()
        for step, metric in zip(self.steps, self.metrics):
            if metric == best_value:
                return step
        return None


class WallClock:
    """time.time()ベースの経過時間計測ヘルパ。"""

    def __init__(self) -> None:
        self._start = time.time()

    def reset(self) -> None:
        """計測開始時刻を現在時刻にリセットする。"""
        self._start = time.time()

    def elapsed_sec(self) -> float:
        """計測開始からの経過秒数。"""
        return time.time() - self._start
