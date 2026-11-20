// 訓練metrics履歴・WS接続状態のzustandストア。
// チャート系列は配列をin-placeで伸ばし、versionカウンタで再描画をトリガする
// (毎ステップの配列コピーを避けるため)。
// live/replayモード(タイムラインスクラブ)とdetailフレームの保持。

import { create } from 'zustand';
import type { HistoryIndex, HistoryStepView, U8Envelope } from '../api/rest';

export interface MetricsFrame {
  type: 'metrics';
  step: number;
  total_steps?: number | null;
  epoch: number;
  loss: number;
  acc: number;
  weight_delta: Record<string, number>;
  // hard eval直後のフレームのみ載るスパースなキー
  val_loss?: number;
  val_acc?: number;
  // §4-1: hard evalと同stepにのみ載る選択ギャップ関連キー
  val_acc_soft?: number;
  selection_gap?: number;
  // binary_vectorタスクの追加メトリクス
  exact_match?: number;
}

/** 履歴インデックスのWSプッシュフレーム。*/
export interface HistoryIndexFrame extends HistoryIndex {
  type: 'history_index';
}

export interface LayerSummaryFrame {
  type: 'layer_summary';
  layer: string;
  step: number;
  n_inputs: number;
  gate_ids: number[];
  confidence: number[];
  // §4-1: レイヤ別の迷い統計(warp等・静的プレビューでは未載でundefined許容)
  entropy_mean?: number | null;
  entropy_max?: number | null;
}

export interface GateDistributionFrame {
  type: 'gate_distribution';
  layer: string;
  step: number;
  n_inputs: number;
  parameterization: string;
  mode: 'dense' | 'aggregate';
  probs?: U8Envelope;
  hist?: number[];
  entropy_mean: number | null;
}

export interface ActivationsFrame {
  type: 'activations';
  layer: string;
  step: number;
  mean: U8Envelope;
  saturation: number;
}

export interface GradFlowLayerTrace {
  layer: string;
  order: number;
  grad_out?: U8Envelope;
  grad_w?: U8Envelope;
  edge_grad?: U8Envelope;
}

export interface GradFlowFrame {
  type: 'grad_flow';
  step: number;
  loss: number;
  layers: GradFlowLayerTrace[];
  layer_grad_norm: Record<string, number>;
}

export interface ApDeltaFrame {
  type: 'ap_delta';
  layer: string;
  step: number;
  delta_norm: number;
  delta?: U8Envelope;
}

export interface CaptureConfigFrame {
  type: 'capture_config';
  level: string;
  stride: number;
  method: string;
  snapshot_bytes: number;
  detail_max_elems: number;
}

export interface PlaybackState {
  playing: boolean;
  direction: 1 | -1;
  speed: number; // steps/sec
}

export type TrainingMethod = 'bp' | 'ap' | 'bb';
export type TrainingOptimizer = 'adam' | 'sgd';

export interface TrainingPreset {
  dataset?: string | null;
  loss?: string | null;
  lr?: number | string;
  logicLr?: number | string | null;
  annLr?: number | string | null;
  batchSize?: number | string;
  epochs?: number | string;
  method?: TrainingMethod;
  optimizer?: TrainingOptimizer;
}

export interface TrainingConfig {
  datasetChoice: string | null;
  lossChoice: string | null;
  optimizer: TrainingOptimizer;
  methodChoice: TrainingMethod;
  lr: string;
  logicLr: string;
  annLr: string;
  batchSize: string;
  epochs: string;
  presetLabel: string | null;
  // wandb連携のopt-inトグル。既定offでエディタの既定動作は変えない
  wandbEnabled: boolean;
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  datasetChoice: null,
  lossChoice: null,
  optimizer: 'adam',
  methodChoice: 'bp',
  lr: '0.01',
  logicLr: '',
  annLr: '',
  batchSize: '128',
  epochs: '3',
  presetLabel: null,
  wandbEnabled: false,
};

const MAX_POINTS = 3000; // チャート保持点数の上限

// 'lost'(バックエンド再起動でセッション消失)はアクティブに含めない
export const ACTIVE_STATES = ['starting', 'running', 'paused'];

/**
 * スパースな[step, val]ペアをxSteps軸に整列したnullパディング系列へ変換する。
 * uPlotのAlignedDataはx軸共有のため、val系列は主系列のstepに合わせて疎に載せる。
 */
export function alignValSeries(
  xSteps: number[],
  pairs: ReadonlyArray<readonly [number, number]>,
): (number | null)[] {
  const byStep = new Map<number, number>();
  for (const [s, v] of pairs) byStep.set(s, v);
  return xSteps.map((s) => byStep.get(s) ?? null);
}

interface TrainingState {
  sessionId: string | null;
  // idle / starting / running / paused / stopped / finished / error / lost
  status: string;
  wsConnected: boolean;
  trainConfig: TrainingConfig;
  steps: number[];
  totalSteps: number | null;
  loss: number[];
  acc: number[];
  deltaSteps: number[];
  deltaSeries: Record<string, (number | null)[]>;
  summaries: Record<string, LayerSummaryFrame>;
  version: number;
  // --- 検証カーブ: [step, val]ペアのスパース系列(トリム対象外) ---
  valLoss: [number, number][];
  valAcc: [number, number][];
  // §4-1: soft(連続緩和)評価精度のスパース系列(hardとの乖離=選択ギャップ)
  valAccSoft: [number, number][];
  // --- エポック境界: 新エポック最初のstep ---
  epochMarks: number[];
  lastEpoch: number | null;
  // --- 再起動検出: 既知の/api/health server_epoch ---
  serverEpoch: number | null;
  // --- 再生(タイムマシン)状態 ---
  displayMode: 'live' | 'replay';
  cursorStep: number | null;
  playback: PlaybackState;
  historyMeta: HistoryIndex | null;
  captureConfig: CaptureConfigFrame | null;
  liveDetail: {
    gateDist: Record<string, GateDistributionFrame>;
    gradFlow: GradFlowFrame | null;
    activations: Record<string, ActivationsFrame>;
    apDelta: Record<string, ApDeltaFrame>;
  };
  replayFrame: HistoryStepView | null;
  // スクラブ中の読込状態。trueの間「読込中…」を表示する
  replayLoading: boolean;
  setSession: (sid: string | null) => void;
  setStatus: (status: string) => void;
  /** 監視対象のrunを切り替える(系列はリセットし以降のWS/pollで追従)。 */
  focusSession: (sid: string) => void;
  setWsConnected: (connected: boolean) => void;
  setTotalSteps: (totalSteps: number | null) => void;
  setTrainConfig: (patch: Partial<TrainingConfig>) => void;
  applyTrainingPreset: (preset: TrainingPreset, presetLabel?: string | null) => void;
  /** バックエンド再起動によるセッション消失。系列は保持したまま操作を止める */
  markLost: () => void;
  setServerEpoch: (epoch: number) => void;
  appendMetrics: (frames: MetricsFrame[]) => void;
  setSummary: (frame: LayerSummaryFrame) => void;
  // --- 再生アクション ---
  enterReplay: (step: number) => void;
  exitReplay: () => void;
  setCursor: (step: number) => void;
  setPlayback: (patch: Partial<PlaybackState>) => void;
  setHistoryMeta: (meta: HistoryIndex | null) => void;
  setCaptureConfig: (cfg: CaptureConfigFrame) => void;
  setReplayLoading: (loading: boolean) => void;
  applyReplayFrame: (view: HistoryStepView) => void;
  applyDetailFrame: (
    frame: GateDistributionFrame | ActivationsFrame | GradFlowFrame | ApDeltaFrame,
  ) => void;
  resetAll: () => void;
}

/** [step, val]ペア配列へ追記(同一stepは置換)する。stepは単調増加前提。 */
function upsertPair(pairs: [number, number][], step: number, value: number): void {
  const last = pairs[pairs.length - 1];
  if (last && last[0] === step) {
    last[1] = value;
  } else {
    pairs.push([step, value]);
  }
}

const INITIAL_PLAYBACK: PlaybackState = { playing: false, direction: 1, speed: 5 };

const INITIAL_LIVE_DETAIL = {
  gateDist: {},
  gradFlow: null,
  activations: {},
  apDelta: {},
} as TrainingState['liveDetail'];

export const useTrainingStore = create<TrainingState>((set, get) => ({
  sessionId: null,
  status: 'idle',
  wsConnected: false,
  trainConfig: DEFAULT_TRAINING_CONFIG,
  steps: [],
  totalSteps: null,
  loss: [],
  acc: [],
  deltaSteps: [],
  deltaSeries: {},
  summaries: {},
  version: 0,
  valLoss: [],
  valAcc: [],
  valAccSoft: [],
  epochMarks: [],
  lastEpoch: null,
  serverEpoch: null,
  displayMode: 'live',
  cursorStep: null,
  playback: INITIAL_PLAYBACK,
  historyMeta: null,
  captureConfig: null,
  liveDetail: INITIAL_LIVE_DETAIL,
  replayFrame: null,
  replayLoading: false,

  setSession: (sid) => set({ sessionId: sid }),
  setStatus: (status) => set({ status }),

  // フォーカス切替は系列をリセットしてから対象runへ差し替える。過去系列の
  // バックフィルは行わない(切替後は以降の更新のみ表示。仕様上許容)。
  // statusは中立の'running'にしてWS/pollに実状態の上書きを委ねる。
  focusSession: (sid) => {
    const s = get();
    s.resetAll();
    s.setSession(sid);
    s.setStatus('running');
  },
  setWsConnected: (wsConnected) => set({ wsConnected }),
  setTotalSteps: (totalSteps) => set({ totalSteps }),
  setTrainConfig: (patch) =>
    set((state) => ({ trainConfig: { ...state.trainConfig, ...patch } })),
  applyTrainingPreset: (preset, presetLabel = null) =>
    set((state) => ({
      trainConfig: {
        ...state.trainConfig,
        datasetChoice: preset.dataset ?? state.trainConfig.datasetChoice,
        lossChoice: preset.loss ?? state.trainConfig.lossChoice,
        optimizer: preset.optimizer ?? state.trainConfig.optimizer,
        methodChoice: preset.method ?? state.trainConfig.methodChoice,
        lr: preset.lr != null ? String(preset.lr) : state.trainConfig.lr,
        logicLr:
          preset.logicLr != null ? String(preset.logicLr) : state.trainConfig.logicLr,
        annLr:
          preset.annLr != null ? String(preset.annLr) : state.trainConfig.annLr,
        batchSize:
          preset.batchSize != null ? String(preset.batchSize) : state.trainConfig.batchSize,
        epochs: preset.epochs != null ? String(preset.epochs) : state.trainConfig.epochs,
        presetLabel,
      },
    })),

  markLost: () => set({ status: 'lost', wsConnected: false }),
  setServerEpoch: (epoch) => set({ serverEpoch: epoch }),

  appendMetrics: (frames) => {
    const s = get();
    for (const f of frames) {
      if (f.total_steps != null) s.totalSteps = f.total_steps;

      // val系列: [step, val]ペアで保持。同一stepは最新で置換する
      if (f.val_loss != null) upsertPair(s.valLoss, f.step, f.val_loss);
      if (f.val_acc != null) upsertPair(s.valAcc, f.step, f.val_acc);
      if (f.val_acc_soft != null) upsertPair(s.valAccSoft, f.step, f.val_acc_soft);

      // 同一stepの重複フレーム(セッション終了時eval等)はin-placeマージのみ
      if (s.steps.length > 0 && s.steps[s.steps.length - 1] === f.step) {
        s.loss[s.loss.length - 1] = f.loss;
        s.acc[s.acc.length - 1] = f.acc;
        continue;
      }

      // エポック境界検出: 新エポック最初のstepを記録する
      if (s.lastEpoch !== null && f.epoch > s.lastEpoch) {
        s.epochMarks.push(f.step);
      }
      s.lastEpoch = f.epoch;

      s.steps.push(f.step);
      s.loss.push(f.loss);
      s.acc.push(f.acc);
      s.deltaSteps.push(f.step);
      // 新出レイヤはnullで過去分をパディングして系列を揃える
      for (const key of Object.keys(f.weight_delta)) {
        if (!(key in s.deltaSeries)) {
          s.deltaSeries[key] = new Array(s.deltaSteps.length - 1).fill(null);
        }
      }
      for (const key of Object.keys(s.deltaSeries)) {
        s.deltaSeries[key].push(f.weight_delta[key] ?? null);
      }
    }
    // 上限超過分を先頭から捨てる
    const overflow = s.steps.length - MAX_POINTS;
    if (overflow > 0) {
      s.steps.splice(0, overflow);
      s.loss.splice(0, overflow);
      s.acc.splice(0, overflow);
      s.deltaSteps.splice(0, overflow);
      for (const key of Object.keys(s.deltaSeries)) {
        s.deltaSeries[key].splice(0, overflow);
      }
    }
    set({ version: s.version + 1 });
  },

  setSummary: (frame) =>
    set((state) => ({ summaries: { ...state.summaries, [frame.layer]: frame } })),

  // --- live/replay状態遷移の仕様 ---
  // replayへの進入: タイムラインUI操作(スライダー/コマ送り/再生ボタン)経由の
  //   enterReplay/setCursor のみ。WSフレームや自動処理からは遷移しない。
  // liveへの復帰: 「LIVEへ戻る」ボタン/resume/学習step実行時の exitReplay のみ。
  enterReplay: (step) =>
    set({ displayMode: 'replay', cursorStep: step, playback: INITIAL_PLAYBACK }),

  exitReplay: () =>
    set({
      displayMode: 'live',
      cursorStep: null,
      replayFrame: null,
      replayLoading: false,
      playback: INITIAL_PLAYBACK,
    }),

  setCursor: (step) => {
    const meta = get().historyMeta;
    let clamped = step;
    if (meta?.min_step != null) clamped = Math.max(clamped, meta.min_step);
    if (meta?.max_step != null) clamped = Math.min(clamped, meta.max_step);
    set({ displayMode: 'replay', cursorStep: clamped });
  },

  setPlayback: (patch) => set((s) => ({ playback: { ...s.playback, ...patch } })),
  setHistoryMeta: (meta) => set({ historyMeta: meta }),
  setCaptureConfig: (cfg) => set({ captureConfig: cfg }),
  setReplayLoading: (loading) => set({ replayLoading: loading }),

  applyReplayFrame: (view) => {
    // カーソルが既に別stepへ移っていても、resolved_stepが近ければ表示する
    // (丸め込みがあるためcursorStepとの厳密一致は要求しない)。
    // フレーム未着時は直前のreplayFrameを保持する=スクラブ中でも
    // キャンバスが空にならない(クリアするのはexitReplayのみ)
    set({ replayFrame: view });
  },

  applyDetailFrame: (frame) => {
    const s = get();
    if (frame.type === 'gate_distribution') {
      set({
        liveDetail: {
          ...s.liveDetail,
          gateDist: { ...s.liveDetail.gateDist, [frame.layer]: frame },
        },
      });
    } else if (frame.type === 'activations') {
      set({
        liveDetail: {
          ...s.liveDetail,
          activations: { ...s.liveDetail.activations, [frame.layer]: frame },
        },
      });
    } else if (frame.type === 'grad_flow') {
      set({ liveDetail: { ...s.liveDetail, gradFlow: frame } });
    } else if (frame.type === 'ap_delta') {
      set({
        liveDetail: {
          ...s.liveDetail,
          apDelta: { ...s.liveDetail.apDelta, [frame.layer]: frame },
        },
      });
    }
  },

  // 新ラン開始時はserverEpochもnullへ戻す。次のWS接続でcheckServerEpochChangedが
  // 現行epochへ再ベースラインし、dev --reload(ラン合間の再起動)での誤lostを防ぐ。
  // ラン中の再起動検出は接続時の再ベースライン+30s健全性チェックで維持される。
  resetAll: () =>
    set({
      sessionId: null,
      status: 'idle',
      serverEpoch: null,
      wsConnected: false,
      steps: [],
      totalSteps: null,
      loss: [],
      acc: [],
      deltaSteps: [],
      deltaSeries: {},
      summaries: {},
      valLoss: [],
      valAcc: [],
      valAccSoft: [],
      epochMarks: [],
      lastEpoch: null,
      displayMode: 'live',
      cursorStep: null,
      playback: INITIAL_PLAYBACK,
      historyMeta: null,
      captureConfig: null,
      liveDetail: INITIAL_LIVE_DETAIL,
      replayFrame: null,
      replayLoading: false,
      version: get().version + 1,
    }),
}));
