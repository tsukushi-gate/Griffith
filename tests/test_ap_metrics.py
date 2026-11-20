"""ap_metrics.pyのConvergenceTracker/WallClock検証。

BP-free学習(AP/DFA)の収束・停滞・時間計測ヘルパが満たすべき不変条件:
    1. steps_to_threshold: mode='min'(BPCのように小さいほど良い)/'max'
       (accのように大きいほど良い)の両方向で、閾値に初めて到達したstepを返す。
    2. plateau_step: patience回連続で改善<min_deltaになった最初のstepを返す
       (BP切替トリガの検出)。
    3. 境界条件: 閾値未到達・記録不足・改善が続く場合はNoneを返す。
    4. best()/best_step()がmode別に正しい値を返す。
    5. WallClock.elapsed_sec()が非負かつ単調増加する。

CPU専用、数秒で完結。高速スイート対象。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from ap_metrics import ConvergenceTracker, WallClock


# ----------------------------------------------------------------------
# steps_to_threshold: mode='min'(BPCのように小さいほど良い)
# ----------------------------------------------------------------------
def test_steps_to_threshold_min_reaches():
    tracker = ConvergenceTracker(mode='min')
    for step, bpc in [(10, 5.0), (20, 4.0), (30, 2.9), (40, 2.5)]:
        tracker.record(step, bpc)
    # 閾値3.0に初めて到達するのはstep=30(bpc=2.9 <= 3.0)
    assert tracker.steps_to_threshold(3.0) == 30


def test_steps_to_threshold_min_not_reached():
    tracker = ConvergenceTracker(mode='min')
    for step, bpc in [(10, 5.0), (20, 4.0), (30, 3.5)]:
        tracker.record(step, bpc)
    assert tracker.steps_to_threshold(3.0) is None


# ----------------------------------------------------------------------
# steps_to_threshold: mode='max'(accのように大きいほど良い)
# ----------------------------------------------------------------------
def test_steps_to_threshold_max_reaches():
    tracker = ConvergenceTracker(mode='max')
    for step, acc in [(10, 0.5), (20, 0.7), (30, 0.91), (40, 0.95)]:
        tracker.record(step, acc)
    # 閾値0.9に初めて到達するのはstep=30(acc=0.91 >= 0.9)
    assert tracker.steps_to_threshold(0.9) == 30


def test_steps_to_threshold_max_not_reached():
    tracker = ConvergenceTracker(mode='max')
    for step, acc in [(10, 0.5), (20, 0.7), (30, 0.8)]:
        tracker.record(step, acc)
    assert tracker.steps_to_threshold(0.9) is None


def test_steps_to_threshold_mode_override():
    # self.modeとは別に、呼び出し時にmode引数で上書きできる。
    tracker = ConvergenceTracker(mode='min')
    for step, acc in [(10, 0.5), (20, 0.95)]:
        tracker.record(step, acc)
    assert tracker.steps_to_threshold(0.9, mode='max') == 20


# ----------------------------------------------------------------------
# plateau_step: 停滞検出
# ----------------------------------------------------------------------
def test_plateau_step_detects_stagnation():
    tracker = ConvergenceTracker(mode='min')
    # 大きく改善した後、3回連続でほぼ改善なし(min_delta=1e-3未満)。
    values = [
        (0, 5.0),
        (10, 3.0),   # improvement=2.0 (改善)
        (20, 2.9999),  # improvement<1e-3 (停滞1回目)
        (30, 2.9998),  # 停滞2回目
        (40, 2.9997),  # 停滞3回目 -> patience=3ならここで検出
    ]
    for step, metric in values:
        tracker.record(step, metric)
    plateau = tracker.plateau_step(patience=3, min_delta=1e-3)
    assert plateau == 20  # 停滞区間の最初のstep


def test_plateau_step_no_stagnation_when_improving():
    tracker = ConvergenceTracker(mode='min')
    for step, metric in [(0, 5.0), (10, 4.0), (20, 3.0), (30, 2.0), (40, 1.0)]:
        tracker.record(step, metric)
    assert tracker.plateau_step(patience=3, min_delta=1e-3) is None


def test_plateau_step_insufficient_records_returns_none():
    tracker = ConvergenceTracker(mode='min')
    tracker.record(0, 5.0)
    tracker.record(10, 4.0)
    # patience=5だが記録は2件のみ -> 判定不可
    assert tracker.plateau_step(patience=5) is None


def test_plateau_step_max_mode():
    tracker = ConvergenceTracker(mode='max')
    values = [
        (0, 0.1),
        (10, 0.5),     # improvement=0.4
        (20, 0.5001),  # 停滞1
        (30, 0.5002),  # 停滞2
    ]
    for step, metric in values:
        tracker.record(step, metric)
    assert tracker.plateau_step(patience=2, min_delta=1e-3) == 20


# ----------------------------------------------------------------------
# best() / best_step()
# ----------------------------------------------------------------------
def test_best_and_best_step_min_mode():
    tracker = ConvergenceTracker(mode='min')
    for step, metric in [(10, 5.0), (20, 2.0), (30, 3.0)]:
        tracker.record(step, metric)
    assert tracker.best() == 2.0
    assert tracker.best_step() == 20


def test_best_and_best_step_max_mode():
    tracker = ConvergenceTracker(mode='max')
    for step, metric in [(10, 0.5), (20, 0.9), (30, 0.7)]:
        tracker.record(step, metric)
    assert tracker.best() == 0.9
    assert tracker.best_step() == 20


def test_best_empty_returns_none():
    tracker = ConvergenceTracker(mode='min')
    assert tracker.best() is None
    assert tracker.best_step() is None


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        ConvergenceTracker(mode='bogus')


# ----------------------------------------------------------------------
# WallClock
# ----------------------------------------------------------------------
def test_wall_clock_elapsed_nonnegative_and_monotonic():
    clock = WallClock()
    first = clock.elapsed_sec()
    second = clock.elapsed_sec()
    assert first >= 0.0
    assert second >= first


def test_wall_clock_reset():
    clock = WallClock()
    clock.elapsed_sec()
    clock.reset()
    # reset直後はほぼ0に近い(負にはならない)
    assert clock.elapsed_sec() >= 0.0
