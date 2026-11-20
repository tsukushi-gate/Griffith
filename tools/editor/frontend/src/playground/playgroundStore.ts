// フォワード実行プレイグラウンドのzustandストア。
// 入力トグル/mode変更は即時recompute。学習中のライブ更新はPlaygroundPanelが
// useEffectでstep変化を購読してrecomputeを呼ぶ。
// IR編集(真理値表クリック等)への追随はgraphStore購読で行う(タブ非表示でも
// GateNodeバッジ/エッジ着色が古くならないように)。

import { create } from 'zustand';
import type { IRGraphJson } from '../ir/schema';
import { useGraphStore } from '../store/graphStore';
import {
  computeReachabilityCone,
  mismatchedOutputTargets,
  rankCauseCandidates,
  type CauseCandidate,
  type TraceCone,
} from './counterexampleTrace';
import { evalCircuit, type EvalMode } from './evalCircuit';
import { readTrainingProbs } from './trainingProbs';
import {
  enumerateTruthTable,
  expectedMatchesShape,
  matchInfo,
  patternToInputBits,
  rowsToExpected,
  totalInputWidth,
  TRUTH_TABLE_MAX_BITS,
  type TruthRow,
} from './truthTable';

/** 学習中ライブ更新の再評価スロットリング間隔(ms)。高頻度なgate_dist更新を間引く */
const LIVE_RECOMPUTE_INTERVAL_MS = 200;
/** 一致率推移で保持するサンプル数の上限 */
const MATCH_HISTORY_MAX = 120;

let liveRecomputeTimer: ReturnType<typeof setTimeout> | null = null;

interface PlaygroundState {
  /** 実行モードon/off(offなら計算・描画とも従来と完全同一) */
  enabled: boolean;
  mode: EvalMode;
  /** 入力ポートid→ビット値(0/1) */
  inputBits: Record<string, number[]>;
  /** 直近評価の全ノード出力値(未評価・エラー時はnull) */
  values: Record<string, Float32Array> | null;
  /** 直近評価のエラーメッセージ(未対応ノード型等) */
  error: string | null;
  /** hard全列挙の検証テーブル(入力幅>10bitはnull) */
  tableRows: TruthRow[] | null;
  /** 検証テーブルの期待値: パターンk→出力bits(k昇順)。
      将来「この真理値表を学習」へ渡すデータ形式(butterfly規約:
      パターンkの入力bit j = (k >> j) & 1)。 */
  expectedTable: Record<number, number[]> | null;
  /** 一致率(matched/total)の推移。学習で「正解セルが増えていく」様子の表示用 */
  matchHistory: number[];
  /** 反例トレース中のパターンk(nullなら非トレース) */
  traceK: number | null;
  /** 反例の到達コーン(nullなら非トレース) */
  traceCone: TraceCone | null;
  /** 反例の原因候補ゲート(スコア降順) */
  traceCandidates: CauseCandidate[];
  setEnabled: (enabled: boolean) => void;
  setMode: (mode: EvalMode) => void;
  toggleBit: (portId: string, index: number) => void;
  /** 検証テーブルの行クリック: パターンkを入力トグルへピン留め */
  pinPattern: (k: number) => void;
  toggleExpectedBit: (k: number, j: number) => void;
  /** 期待値を現在の回路出力コピーでリセット */
  resetExpected: () => void;
  /** graphStoreのIR+学習確率(live/replay)で再評価する */
  recompute: () => void;
  /** 学習ライブ/リプレイのstep変化用のスロットリング付き再評価 */
  scheduleRecompute: () => void;
  /** 反例セルのクリック: 入力ピン留め+到達コーン計算+原因候補順位付け */
  startTrace: (k: number) => void;
  /** 反例トレースを解除する */
  clearTrace: () => void;
}

/** 入力ビットをIRのポート構成(幅)へ揃える(既存値は幅の範囲で保持) */
function normalizeInputBits(
  ir: IRGraphJson,
  prev: Record<string, number[]>,
): Record<string, number[]> {
  let changed = Object.keys(prev).length !== ir.inputs.length;
  const next: Record<string, number[]> = {};
  for (const port of ir.inputs) {
    const cur = prev[port.id];
    if (cur && cur.length === port.width) {
      next[port.id] = cur;
      continue;
    }
    changed = true;
    const bits = new Array<number>(port.width).fill(0);
    if (cur) {
      for (let i = 0; i < Math.min(cur.length, port.width); i++) bits[i] = cur[i];
    }
    next[port.id] = bits;
  }
  return changed ? next : prev;
}

export const usePlaygroundStore = create<PlaygroundState>((set, get) => ({
  enabled: false,
  mode: 'hard',
  inputBits: {},
  values: null,
  error: null,
  tableRows: null,
  expectedTable: null,
  matchHistory: [],
  traceK: null,
  traceCone: null,
  traceCandidates: [],

  setEnabled: (enabled) => {
    set({ enabled });
    get().recompute();
  },

  setMode: (mode) => {
    set({ mode });
    get().recompute();
  },

  toggleBit: (portId, index) => {
    const bits = [...(get().inputBits[portId] ?? [])];
    while (bits.length <= index) bits.push(0);
    bits[index] = bits[index] ? 0 : 1;
    set({ inputBits: { ...get().inputBits, [portId]: bits } });
    get().recompute();
  },

  pinPattern: (k) => {
    const ir = useGraphStore.getState().ir;
    if (!ir) return;
    set({ inputBits: patternToInputBits(ir, k) });
    get().recompute();
  },

  toggleExpectedBit: (k, j) => {
    const cur = get().expectedTable;
    const row = cur?.[k];
    if (!cur || !row) return;
    const bits = [...row];
    bits[j] = bits[j] ? 0 : 1;
    set({ expectedTable: { ...cur, [k]: bits } });
  },

  resetExpected: () => {
    const rows = get().tableRows;
    set({ expectedTable: rows ? rowsToExpected(rows) : null });
  },

  recompute: () => {
    const { enabled, mode, inputBits, expectedTable, matchHistory, traceK } = get();
    const ir = useGraphStore.getState().ir;
    if (!enabled || !ir) {
      set({
        values: null,
        tableRows: null,
        error: null,
        matchHistory: [],
        traceK: null,
        traceCone: null,
        traceCandidates: [],
      });
      return;
    }
    const bits = normalizeInputBits(ir, inputBits);
    try {
      const probs = readTrainingProbs(ir);
      const result = evalCircuit(ir, bits, mode, probs);
      const rows =
        totalInputWidth(ir) <= TRUTH_TABLE_MAX_BITS
          ? enumerateTruthTable(ir, probs)
          : null;
      // 期待値は初回(または次元変更時)のみ現在出力のコピーで初期化する
      const keepExpected = expectedMatchesShape(expectedTable, rows);
      const nextExpected = keepExpected
        ? expectedTable
        : rows
          ? rowsToExpected(rows)
          : null;
      // 一致率推移: 次元維持のときだけ連続サンプルとして積む(構造変更でリセット)
      let nextHistory = keepExpected ? matchHistory : [];
      if (rows && nextExpected) {
        const info = matchInfo(rows, nextExpected);
        const ratio = info.total > 0 ? info.matched / info.total : 0;
        if (nextHistory[nextHistory.length - 1] !== ratio) {
          nextHistory = [...nextHistory, ratio].slice(-MATCH_HISTORY_MAX);
        }
      }
      // トレース中は重みが動いても同じ反例パターンでコーン/候補を再計算する。
      // 構造変更で期待値がリセットされた場合はトレースも解除(コーンが古くなる)。
      const trace =
        keepExpected && traceK !== null && rows && nextExpected
          ? buildTrace(ir, traceK, rows, nextExpected, probs)
          : null;
      set({
        inputBits: bits,
        values: result.values,
        tableRows: rows,
        expectedTable: nextExpected,
        matchHistory: nextHistory,
        traceK: trace ? traceK : null,
        traceCone: trace?.cone ?? null,
        traceCandidates: trace?.candidates ?? [],
        error: null,
      });
    } catch (e) {
      set({
        inputBits: bits,
        values: null,
        tableRows: null,
        error: e instanceof Error ? e.message : String(e),
        traceK: null,
        traceCone: null,
        traceCandidates: [],
      });
    }
  },

  scheduleRecompute: () => {
    if (liveRecomputeTimer) return;
    liveRecomputeTimer = setTimeout(() => {
      liveRecomputeTimer = null;
      if (get().enabled) get().recompute();
    }, LIVE_RECOMPUTE_INTERVAL_MS);
  },

  startTrace: (k) => {
    // 反例パターンを入力へピン留め+traceKを立て、cone/候補の計算はrecomputeに委ねる
    const ir = useGraphStore.getState().ir;
    if (!ir) return;
    set({ traceK: k, inputBits: patternToInputBits(ir, k) });
    get().recompute();
  },

  clearTrace: () => set({ traceK: null, traceCone: null, traceCandidates: [] }),
}));

/** 反例トレースの到達コーンと原因候補を組み立てる(recompute内から使用) */
function buildTrace(
  ir: IRGraphJson,
  k: number,
  rows: TruthRow[],
  expected: Record<number, number[]>,
  probs: ReturnType<typeof readTrainingProbs>,
): { cone: TraceCone; candidates: CauseCandidate[] } | null {
  const row = rows.find((r) => r.k === k);
  const exp = expected[k];
  if (!row || !exp) return null;
  const bits = patternToInputBits(ir, k);
  // 原因順位付けにはmodeに依らずhard/softの両方が要る
  const hard = evalCircuit(ir, bits, 'hard', probs).values;
  const soft = evalCircuit(ir, bits, 'soft', probs).values;
  const targets = mismatchedOutputTargets(ir, row.outputBits, exp);
  const cone = computeReachabilityCone(ir, targets);
  const candidates = rankCauseCandidates(ir, cone, hard, soft);
  return { cone, candidates };
}

// IR編集(真理値表クリック・構造編集)に追随して再評価する。
// 学習フレーム(step変化)による再評価はPlaygroundPanel側のuseEffectが担う。
useGraphStore.subscribe((state, prev) => {
  if (state.ir !== prev.ir && usePlaygroundStore.getState().enabled) {
    usePlaygroundStore.getState().recompute();
  }
});
