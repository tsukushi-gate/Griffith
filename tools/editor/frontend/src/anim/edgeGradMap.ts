// 誤差逆伝播アニメーションのデータ変換。
// GradFlowFrameのedge_grad/grad_w(u8エンベロープ)を、描画側が直接引ける
// 「詳細エッジid/ゲートid → 正規化勾配強度[0,1]」のMapへ変換する純関数群。
// エッジid規約は irToDetailGraph.ts の `d:${layerId}:${neuron}:${slot}`、
// ゲートid規約は `${layerId}::g${neuron}` に一致させる(vitestで固定)。

import { decodeU8 } from '../api/u8';
import type { GradFlowFrame, GradFlowLayerTrace } from '../store/trainingStore';

export interface BpIntensityMaps {
  /** 詳細エッジid `d:${layerId}:${neuron}:${slot}` → 強度[0,1](層内max正規化) */
  edgeIntensity: Map<string, number>;
  /** ゲートid `${layerId}::g${neuron}` → 強度[0,1](層内max正規化) */
  gateIntensity: Map<string, number>;
}

/** 詳細エッジidを組み立てる(irToDetailGraph.tsのエッジid規約と同一)。 */
export function detailEdgeId(layerId: string, neuron: number, slot: number): string {
  return `d:${layerId}:${neuron}:${slot}`;
}

/** ゲートノードidを組み立てる(irToDetailGraph.tsの子ノードid規約と同一)。 */
export function gateNodeId(layerId: string, neuron: number): string {
  return `${layerId}::g${neuron}`;
}

/** 配列を最大値で[0,1]へ正規化する(全て0なら全て0のまま)。 */
function normalizeByMax(values: Float32Array): Float32Array {
  let max = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }
  const out = new Float32Array(values.length);
  if (max <= 0) return out;
  for (let i = 0; i < values.length; i++) {
    out[i] = values[i] / max;
  }
  return out;
}

/**
 * 層1件のedge_gradを層内max正規化した平坦配列(長さ out_dim×n_inputs、
 * row-major: index = neuron×n_inputs + slot)へ復元する。
 * L0 deck.gl着色はL0Edge.indexがこの平坦indexと同規約なのでそのまま使える。
 */
export function layerEdgeGradIntensity(entry: GradFlowLayerTrace): Float32Array | null {
  if (!entry.edge_grad || entry.edge_grad.shape.length !== 2) return null;
  return normalizeByMax(decodeU8(entry.edge_grad));
}

/**
 * GradFlowFrame全層から強度Mapを構築する(再生開始時に1回だけ呼ぶ)。
 * - edge_grad(out_dim, n_inputs) → 詳細エッジid → 強度
 * - grad_w(out_dim)              → ゲートid     → 強度
 * 正規化はいずれも層内maxで行う(層間の絶対値差はlayer_grad_norm側で表現済み)。
 */
export function buildBpIntensityMaps(frame: GradFlowFrame): BpIntensityMaps {
  const edgeIntensity = new Map<string, number>();
  const gateIntensity = new Map<string, number>();

  for (const entry of frame.layers) {
    const edgeNorm = layerEdgeGradIntensity(entry);
    if (edgeNorm && entry.edge_grad) {
      const [outDim, nInputs] = entry.edge_grad.shape;
      for (let o = 0; o < outDim; o++) {
        for (let j = 0; j < nInputs; j++) {
          edgeIntensity.set(detailEdgeId(entry.layer, o, j), edgeNorm[o * nInputs + j]);
        }
      }
    }
    if (entry.grad_w) {
      const gateNorm = normalizeByMax(decodeU8(entry.grad_w));
      for (let o = 0; o < gateNorm.length; o++) {
        gateIntensity.set(gateNodeId(entry.layer, o), gateNorm[o]);
      }
    }
  }

  return { edgeIntensity, gateIntensity };
}
