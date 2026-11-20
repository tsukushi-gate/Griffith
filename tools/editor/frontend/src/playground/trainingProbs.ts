// 学習中/リプレイ中のゲート確率をtrainingStoreから読むヘルパ。
// useLayerViewはReact hookのためstore内で使えない — getState()で同じ解決を行う。
//   replay時: replayFrameのgate_distribution(u8) / live時: liveDetailのdenseフレーム
// 確率が届いていない層はエントリを作らず、評価エンジンがIRのtruth_tablesへ
// フォールバックする。

import { decodeU8 } from '../api/u8';
import type { IRGraphJson } from '../ir/schema';
import { useTrainingStore } from '../store/trainingStore';
import type { LayerProbs } from './evalCircuit';

export function readTrainingProbs(ir: IRGraphJson): Record<string, LayerProbs> {
  const s = useTrainingStore.getState();
  const result: Record<string, LayerProbs> = {};
  for (const node of ir.nodes) {
    if (node.type !== 'lut_layer') continue;
    if (s.displayMode === 'replay') {
      const dist = s.replayFrame?.layers[node.id]?.gate_distribution;
      if (dist) {
        result[node.id] = { data: decodeU8(dist), bins: dist.shape[1] ?? 0 };
      }
      continue;
    }
    const frame = s.liveDetail.gateDist[node.id];
    if (frame?.mode === 'dense' && frame.probs) {
      result[node.id] = {
        data: decodeU8(frame.probs),
        bins: frame.probs.shape[1] ?? 0,
      };
    }
  }
  return result;
}
