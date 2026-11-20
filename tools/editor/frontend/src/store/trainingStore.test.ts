// trainingStore(訓練metrics履歴・再生状態のzustandストア)の単体テスト。
// コンポーネント不要: useTrainingStore.getState()でアクションを直接呼び出す。

import { beforeEach, describe, expect, it } from 'vitest';
import type { HistoryIndex, HistoryStepView } from '../api/rest';
import type {
  ActivationsFrame,
  ApDeltaFrame,
  GateDistributionFrame,
  GradFlowFrame,
  MetricsFrame,
} from './trainingStore';
import {
  ACTIVE_STATES,
  DEFAULT_TRAINING_CONFIG,
  alignValSeries,
  useTrainingStore,
} from './trainingStore';

// 実装のMAX_POINTS(未export)と同値。変更時はこのテストも更新する
const MAX_POINTS = 3000;

function frame(step: number, weightDelta: Record<string, number> = {}): MetricsFrame {
  return {
    type: 'metrics',
    step,
    epoch: 0,
    loss: 1 / (step + 1),
    acc: step / 100,
    weight_delta: weightDelta,
  };
}

const historyMeta: HistoryIndex = {
  min_step: 10,
  max_step: 100,
  stride: 10,
  keyframes: [10, 50, 100],
  count: 10,
};

beforeEach(() => {
  useTrainingStore.getState().resetAll();
  useTrainingStore.getState().setTrainConfig(DEFAULT_TRAINING_CONFIG);
});

describe('appendMetrics', () => {
  it('steps/loss/accを追記しversionを進める', () => {
    const v0 = useTrainingStore.getState().version;
    useTrainingStore.getState().appendMetrics([frame(0), frame(1)]);
    const s = useTrainingStore.getState();
    expect(s.steps).toEqual([0, 1]);
    expect(s.loss).toEqual([1, 0.5]);
    expect(s.acc).toEqual([0, 0.01]);
    expect(s.version).toBe(v0 + 1);
  });

  it('MAX_POINTS超過分を先頭から捨てる', () => {
    const frames = Array.from({ length: MAX_POINTS + 5 }, (_, i) =>
      frame(i, { l1: i }),
    );
    useTrainingStore.getState().appendMetrics(frames);
    const s = useTrainingStore.getState();
    expect(s.steps).toHaveLength(MAX_POINTS);
    expect(s.steps[0]).toBe(5); // 先頭5点が捨てられる
    expect(s.steps[MAX_POINTS - 1]).toBe(MAX_POINTS + 4);
    expect(s.loss).toHaveLength(MAX_POINTS);
    expect(s.acc).toHaveLength(MAX_POINTS);
    expect(s.deltaSteps).toHaveLength(MAX_POINTS);
    expect(s.deltaSeries.l1).toHaveLength(MAX_POINTS);
    expect(s.deltaSeries.l1[0]).toBe(5);
  });

  it('途中から現れたレイヤは過去分をnullでパディングする', () => {
    useTrainingStore.getState().appendMetrics([frame(0, { l1: 0.1 })]);
    useTrainingStore.getState().appendMetrics([frame(1, { l1: 0.2, l2: 0.9 })]);
    const s = useTrainingStore.getState();
    expect(s.deltaSeries.l1).toEqual([0.1, 0.2]);
    expect(s.deltaSeries.l2).toEqual([null, 0.9]);
  });

  it('フレームに現れなくなったレイヤはnullで系列長を揃える', () => {
    useTrainingStore.getState().appendMetrics([
      frame(0, { l1: 0.1 }),
      frame(1, {}),
    ]);
    const s = useTrainingStore.getState();
    expect(s.deltaSeries.l1).toEqual([0.1, null]);
    expect(s.deltaSteps).toEqual([0, 1]);
  });

  it('total_stepsを最新フレームから保持する', () => {
    useTrainingStore.getState().appendMetrics([{ ...frame(1), total_steps: 120 }]);
    expect(useTrainingStore.getState().totalSteps).toBe(120);
  });

  it('選択ギャップ系列(val_acc_soft)をスパースに保持する(§4-1)', () => {
    useTrainingStore.getState().appendMetrics([
      frame(0),
      { ...frame(1), val_acc: 0.7, val_acc_soft: 0.9, selection_gap: 0.2 },
      frame(2),
    ]);
    const s = useTrainingStore.getState();
    expect(s.valAcc).toEqual([[1, 0.7]]);
    expect(s.valAccSoft).toEqual([[1, 0.9]]);
  });
});

describe('training config', () => {
  it('applyTrainingPresetで学習設定を更新する', () => {
    useTrainingStore.getState().applyTrainingPreset(
      {
        dataset: 'logic:full_adder',
        loss: 'bce',
        lr: 0.08,
        batchSize: 16,
        epochs: 60,
        method: 'bp',
        optimizer: 'adam',
      },
      '全加算器 学習',
    );
    const c = useTrainingStore.getState().trainConfig;
    expect(c.datasetChoice).toBe('logic:full_adder');
    expect(c.lossChoice).toBe('bce');
    expect(c.lr).toBe('0.08');
    expect(c.batchSize).toBe('16');
    expect(c.epochs).toBe('60');
    expect(c.presetLabel).toBe('全加算器 学習');
  });

  it('resetAllはサンプル推奨の学習設定を保持する', () => {
    useTrainingStore.getState().applyTrainingPreset(
      { dataset: 'logic:half_adder', lr: 0.08, batchSize: 16, epochs: 60 },
      '半加算器 学習',
    );
    useTrainingStore.getState().appendMetrics([{ ...frame(1), total_steps: 60 }]);
    useTrainingStore.getState().resetAll();
    const s = useTrainingStore.getState();
    expect(s.trainConfig.datasetChoice).toBe('logic:half_adder');
    expect(s.trainConfig.lr).toBe('0.08');
    expect(s.totalSteps).toBeNull();
  });
});

describe('applyDetailFrame', () => {
  const u8 = { encoding: 'u8' as const, shape: [1], scale: 1, data: 'AA==' };

  it('gate_distributionはレイヤ別に保持される', () => {
    const f1: GateDistributionFrame = {
      type: 'gate_distribution',
      layer: 'l1',
      step: 1,
      n_inputs: 2,
      parameterization: 'softmax',
      mode: 'dense',
      probs: u8,
      entropy_mean: 0.5,
    };
    const f2: GateDistributionFrame = { ...f1, layer: 'l2', step: 2 };
    useTrainingStore.getState().applyDetailFrame(f1);
    useTrainingStore.getState().applyDetailFrame(f2);
    const { gateDist } = useTrainingStore.getState().liveDetail;
    expect(gateDist.l1).toBe(f1);
    expect(gateDist.l2).toBe(f2);
  });

  it('同一レイヤのgate_distributionは最新で上書きされる', () => {
    const f1: GateDistributionFrame = {
      type: 'gate_distribution',
      layer: 'l1',
      step: 1,
      n_inputs: 2,
      parameterization: 'softmax',
      mode: 'aggregate',
      hist: [1, 2],
      entropy_mean: null,
    };
    useTrainingStore.getState().applyDetailFrame(f1);
    useTrainingStore.getState().applyDetailFrame({ ...f1, step: 2 });
    expect(useTrainingStore.getState().liveDetail.gateDist.l1.step).toBe(2);
  });

  it('activations/ap_deltaはレイヤ別、grad_flowは単一で保持される', () => {
    const act: ActivationsFrame = {
      type: 'activations',
      layer: 'l1',
      step: 1,
      mean: u8,
      saturation: 0.2,
    };
    const grad: GradFlowFrame = {
      type: 'grad_flow',
      step: 1,
      loss: 0.3,
      layers: [{ layer: 'l1', order: 0 }],
      layer_grad_norm: { l1: 0.1 },
    };
    const ap: ApDeltaFrame = {
      type: 'ap_delta',
      layer: 'l1',
      step: 1,
      delta_norm: 0.05,
    };
    useTrainingStore.getState().applyDetailFrame(act);
    useTrainingStore.getState().applyDetailFrame(grad);
    useTrainingStore.getState().applyDetailFrame(ap);
    const detail = useTrainingStore.getState().liveDetail;
    expect(detail.activations.l1).toBe(act);
    expect(detail.gradFlow).toBe(grad);
    expect(detail.apDelta.l1).toBe(ap);
  });
});

describe('enterReplay / exitReplay', () => {
  it('enterReplayでreplayモードへ入りplaybackが初期化される', () => {
    useTrainingStore.getState().setPlayback({ playing: true, speed: 20 });
    useTrainingStore.getState().enterReplay(42);
    const s = useTrainingStore.getState();
    expect(s.displayMode).toBe('replay');
    expect(s.cursorStep).toBe(42);
    expect(s.playback).toEqual({ playing: false, direction: 1, speed: 5 });
  });

  it('exitReplayでliveへ戻りカーソルとフレームを破棄する', () => {
    const view: HistoryStepView = {
      resolved_step: 42,
      method: 'keyframe',
      metrics: { step: 42, epoch: 0, loss: 0.1, acc: 0.9 },
      layers: {},
    };
    useTrainingStore.getState().enterReplay(42);
    useTrainingStore.getState().applyReplayFrame(view);
    expect(useTrainingStore.getState().replayFrame).toBe(view);

    useTrainingStore.getState().exitReplay();
    const s = useTrainingStore.getState();
    expect(s.displayMode).toBe('live');
    expect(s.cursorStep).toBeNull();
    expect(s.replayFrame).toBeNull();
  });
});

describe('setCursor', () => {
  it('historyMetaの範囲内へクランプされる', () => {
    useTrainingStore.getState().setHistoryMeta(historyMeta);
    useTrainingStore.getState().setCursor(5);
    expect(useTrainingStore.getState().cursorStep).toBe(10);
    useTrainingStore.getState().setCursor(500);
    expect(useTrainingStore.getState().cursorStep).toBe(100);
    useTrainingStore.getState().setCursor(50);
    expect(useTrainingStore.getState().cursorStep).toBe(50);
  });

  it('setCursorはreplayモードへ遷移する', () => {
    useTrainingStore.getState().setCursor(50);
    expect(useTrainingStore.getState().displayMode).toBe('replay');
  });

  it('historyMeta未設定ならクランプしない', () => {
    useTrainingStore.getState().setCursor(-99);
    expect(useTrainingStore.getState().cursorStep).toBe(-99);
  });
});

describe('resetAll', () => {
  it('系列・再生状態を全て初期化しversionだけ進める', () => {
    useTrainingStore.getState().appendMetrics([frame(0, { l1: 1 })]);
    useTrainingStore.getState().setHistoryMeta(historyMeta);
    useTrainingStore.getState().enterReplay(42);
    const vBefore = useTrainingStore.getState().version;

    useTrainingStore.getState().resetAll();
    const s = useTrainingStore.getState();
    expect(s.steps).toEqual([]);
    expect(s.deltaSeries).toEqual({});
    expect(s.displayMode).toBe('live');
    expect(s.cursorStep).toBeNull();
    expect(s.historyMeta).toBeNull();
    expect(s.liveDetail).toEqual({
      gateDist: {},
      gradFlow: null,
      activations: {},
      apDelta: {},
    });
    expect(s.version).toBe(vBefore + 1);
  });

  it('val系列・エポック境界・読込状態も初期化する', () => {
    useTrainingStore.getState().appendMetrics([
      { ...frame(0), val_loss: 0.5, val_acc: 0.5 },
      { ...frame(1), epoch: 1 },
    ]);
    useTrainingStore.getState().setReplayLoading(true);

    useTrainingStore.getState().resetAll();
    const s = useTrainingStore.getState();
    expect(s.valLoss).toEqual([]);
    expect(s.valAcc).toEqual([]);
    expect(s.epochMarks).toEqual([]);
    expect(s.lastEpoch).toBeNull();
    expect(s.replayLoading).toBe(false);
  });

  it('serverEpochはリセットでnullへ戻す(次のWS接続で現行epochへ再ベースライン)', () => {
    useTrainingStore.getState().setServerEpoch(123.45);
    useTrainingStore.getState().resetAll();
    expect(useTrainingStore.getState().serverEpoch).toBeNull();
  });
});

describe('markLost', () => {
  it("statusを'lost'にしwsConnectedを落とす", () => {
    useTrainingStore.getState().setStatus('running');
    useTrainingStore.getState().setWsConnected(true);
    useTrainingStore.getState().markLost();
    const s = useTrainingStore.getState();
    expect(s.status).toBe('lost');
    expect(s.wsConnected).toBe(false);
  });

  it("'lost'はACTIVE_STATESに含まれない", () => {
    expect(ACTIVE_STATES).not.toContain('lost');
  });

  it('系列・履歴メタは保持される(消失後も表示は残す)', () => {
    useTrainingStore.getState().appendMetrics([frame(0), frame(1)]);
    useTrainingStore.getState().setHistoryMeta(historyMeta);
    useTrainingStore.getState().markLost();
    const s = useTrainingStore.getState();
    expect(s.steps).toEqual([0, 1]);
    expect(s.historyMeta).toEqual(historyMeta);
  });

  it('setServerEpochが既知エポックを更新する', () => {
    useTrainingStore.getState().setServerEpoch(1.5);
    expect(useTrainingStore.getState().serverEpoch).toBe(1.5);
    useTrainingStore.getState().setServerEpoch(2.5);
    expect(useTrainingStore.getState().serverEpoch).toBe(2.5);
  });
});

describe('val系列', () => {
  it('val_loss/val_acc付きフレームだけが[step, val]ペアに積まれる', () => {
    useTrainingStore.getState().appendMetrics([
      frame(1),
      { ...frame(2), val_loss: 0.4, val_acc: 0.75 },
      frame(3),
      { ...frame(4), val_loss: 0.2, val_acc: 0.9 },
    ]);
    const s = useTrainingStore.getState();
    expect(s.valLoss).toEqual([
      [2, 0.4],
      [4, 0.2],
    ]);
    expect(s.valAcc).toEqual([
      [2, 0.75],
      [4, 0.9],
    ]);
  });

  it('同一stepのvalは最新値で置換される(セッション終了時eval等)', () => {
    useTrainingStore.getState().appendMetrics([
      { ...frame(5), val_acc: 0.5 },
      { ...frame(5), val_acc: 0.8 },
    ]);
    expect(useTrainingStore.getState().valAcc).toEqual([[5, 0.8]]);
  });

  it('同一stepの重複metricsフレームはsteps系列を重複させない', () => {
    useTrainingStore.getState().appendMetrics([frame(1), frame(2)]);
    useTrainingStore.getState().appendMetrics([
      { ...frame(2), loss: 0.123, val_acc: 1.0 },
    ]);
    const s = useTrainingStore.getState();
    expect(s.steps).toEqual([1, 2]);
    expect(s.loss[1]).toBe(0.123); // in-placeマージ
    expect(s.valAcc).toEqual([[2, 1.0]]);
  });
});

describe('エポック境界検出', () => {
  it('epochが進んだフレームのstepをepochMarksへ記録する', () => {
    useTrainingStore.getState().appendMetrics([
      { ...frame(1), epoch: 0 },
      { ...frame(2), epoch: 0 },
      { ...frame(3), epoch: 1 },
      { ...frame(4), epoch: 1 },
      { ...frame(5), epoch: 2 },
    ]);
    const s = useTrainingStore.getState();
    expect(s.epochMarks).toEqual([3, 5]);
    expect(s.lastEpoch).toBe(2);
  });

  it('最初のフレームでは境界を記録しない(途中参加時の誤検出防止)', () => {
    useTrainingStore.getState().appendMetrics([{ ...frame(100), epoch: 5 }]);
    expect(useTrainingStore.getState().epochMarks).toEqual([]);
    expect(useTrainingStore.getState().lastEpoch).toBe(5);
  });
});

describe('alignValSeries', () => {
  it('ペアをstep軸に整列しnullパディングする', () => {
    expect(
      alignValSeries(
        [1, 2, 3, 4],
        [
          [2, 0.5],
          [4, 0.9],
        ],
      ),
    ).toEqual([null, 0.5, null, 0.9]);
  });

  it('空ペアは全nullになる', () => {
    expect(alignValSeries([1, 2], [])).toEqual([null, null]);
  });

  it('軸に存在しないstepのペアは無視される', () => {
    expect(alignValSeries([1, 3], [[2, 0.5]])).toEqual([null, null]);
  });
});

describe('replayLoading', () => {
  it('setReplayLoadingで読込中フラグを切り替える', () => {
    useTrainingStore.getState().setReplayLoading(true);
    expect(useTrainingStore.getState().replayLoading).toBe(true);
    useTrainingStore.getState().setReplayLoading(false);
    expect(useTrainingStore.getState().replayLoading).toBe(false);
  });

  it('exitReplayで読込中フラグも解除される', () => {
    useTrainingStore.getState().enterReplay(10);
    useTrainingStore.getState().setReplayLoading(true);
    useTrainingStore.getState().exitReplay();
    expect(useTrainingStore.getState().replayLoading).toBe(false);
  });
});
