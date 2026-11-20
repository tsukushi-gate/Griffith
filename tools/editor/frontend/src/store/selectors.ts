// 表示step解決の単一点。
// 全可視化コンポーネント(GateHistogram/LutHeatmap/DeckCanvas/GateNode等)は
// useLayerView() だけを読む。これによりタイムラインスクラブ(replay)時に
// すべての可視化が自動で当該stepへ同期する。

import { useMemo } from 'react';
import { decodeU8 } from '../api/u8';
import { useTrainingStore } from './trainingStore';

export interface LayerView {
  source: 'live' | 'replay';
  /** このビューが表すstep(データ未着はnull) */
  step: number | null;
  gateIds: number[] | null;
  confidence: number[] | null;
  nInputs: number | null;
  /** ゲート確率分布(out_dim×bins、row-major)。denseフレーム受信時のみ */
  gateProbs: Float32Array | null;
  probBins: number;
  /** ニューロン別勾配ノルム(replayのみ、fullレベル時) */
  gradNorm: Float32Array | null;
  /** ニューロン別平均活性 */
  actMean: Float32Array | null;
}

const EMPTY_VIEW: LayerView = {
  source: 'live',
  step: null,
  gateIds: null,
  confidence: null,
  nInputs: null,
  gateProbs: null,
  probBins: 0,
  gradNorm: null,
  actMean: null,
};

/** 現在表示すべきstep(replay時はカーソル、live時は最新)。 */
export function useDisplayStep(): number | null {
  const displayMode = useTrainingStore((s) => s.displayMode);
  const cursorStep = useTrainingStore((s) => s.cursorStep);
  const version = useTrainingStore((s) => s.version);
  return useMemo(() => {
    if (displayMode === 'replay') return cursorStep;
    const steps = useTrainingStore.getState().steps;
    return steps.length > 0 ? steps[steps.length - 1] : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayMode, cursorStep, version]);
}

/** レイヤ単位の表示データ(live/replayを自動解決)。 */
export function useLayerView(layerId: string): LayerView {
  const displayMode = useTrainingStore((s) => s.displayMode);
  const summary = useTrainingStore((s) => s.summaries[layerId]);
  const gateDist = useTrainingStore((s) => s.liveDetail.gateDist[layerId]);
  const activations = useTrainingStore((s) => s.liveDetail.activations[layerId]);
  const replayFrame = useTrainingStore((s) => s.replayFrame);

  return useMemo<LayerView>(() => {
    if (displayMode === 'replay') {
      const layer = replayFrame?.layers[layerId];
      if (!replayFrame || !layer) return { ...EMPTY_VIEW, source: 'replay' };
      const dist = layer.gate_distribution ?? null;
      return {
        source: 'replay',
        step: replayFrame.resolved_step,
        gateIds: layer.gate_ids ?? null,
        confidence: layer.confidence ?? null,
        nInputs: layer.n_inputs ?? null,
        gateProbs: dist ? decodeU8(dist) : null,
        probBins: dist ? (dist.shape[1] ?? 0) : 0,
        gradNorm: layer.grad_norm ? decodeU8(layer.grad_norm) : null,
        actMean: layer.act_mean ? decodeU8(layer.act_mean) : null,
      };
    }
    // live: layer_summary + detailフレーム
    const dense = gateDist?.mode === 'dense' ? gateDist : null;
    return {
      source: 'live',
      step: summary?.step ?? null,
      gateIds: summary?.gate_ids ?? null,
      confidence: summary?.confidence ?? null,
      nInputs: summary?.n_inputs ?? null,
      gateProbs: dense?.probs ? decodeU8(dense.probs) : null,
      probBins: dense?.probs ? (dense.probs.shape[1] ?? 0) : 0,
      gradNorm: null,
      actMean: activations ? decodeU8(activations.mean) : null,
    };
  }, [displayMode, summary, gateDist, activations, replayFrame, layerId]);
}

/** チャートのカーソル同期用: replay時のcursorStep(live時はnull)。 */
export function useChartCursorStep(): number | null {
  const displayMode = useTrainingStore((s) => s.displayMode);
  const cursorStep = useTrainingStore((s) => s.cursorStep);
  return displayMode === 'replay' ? cursorStep : null;
}
